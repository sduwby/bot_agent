#!/usr/bin/env python
"""SO-101 负载量程探针（第 38 周 P0 / 子任务 0.2）。

读各关节 ``Present_Load``（STS3215 只读寄存器），分「空载」「夹持」两次采集，
判定负载信号能否分辨目标物体——这是 P0 消融 B2（负载/接触特征）能否成立的
前置条件（见 docs/week38/design.md §3.3 阻塞点 C）。

**为什么探针自己下闭合指令**：负载读数只在"舵机主动保持被指令的闭合位姿"时才有意义。
而 lerobot 的 ``disable_torque_on_disconnect`` 默认为 True——任何控制进程退出都会松力矩，
夹爪随即回弹。因此"先用别的工具把夹爪摆好、再跑探针"在时序上不可靠；探针必须自己
完成「指令闭合 → 稳定 → 采样 → 退回安全位」这一整段。

物理准备（唯一需要人工的部分）：把目标物体放进**张开**的钳口，再运行 ``--phase loaded``。

用法::

    # 从仓库根目录运行；真机分两次采集
    python src/run_load_probe.py --phase empty  --repeat 5 --json runs/week38/load_probe.json
    python src/run_load_probe.py --phase loaded --repeat 5 --json runs/week38/load_probe.json

    # 已自行保证状态（遥操作等其他进程占用端口时勿用），只读不驱动
    python src/run_load_probe.py --phase empty --no-command --json runs/week38/load_probe.json

    # 离线验证链路（MockEngine 合成负载，不需要机械臂）
    python src/run_load_probe.py --dry-run --phase empty  --repeat 3 --json <tmp>/probe.json
    python src/run_load_probe.py --dry-run --phase loaded --repeat 3 --json <tmp>/probe.json

退出码：0 = 结论可分辨或采集未完成；1 = 已判定为**不可分辨**（可作门禁使用）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from real_observation import LOAD_MARGIN, decode_feetech_load, summarize_load_probe  # noqa: E402

# 闭合/松开目标位置与稳定时间（依据 week24 标定：闭≈0–10、开≈13–23）
DEFAULT_CLOSE_POS = 3.0
DEFAULT_RELEASE_POS = 20.0
DEFAULT_SETTLE_S = 1.0


class RealLoadReader:
    """真机负载读写：只连接电机总线，不加载策略/数据集（区别于 `RealEngine`）。"""

    def __init__(self, port: str, robot_id: str) -> None:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        self._robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, cameras={}))
        # calibrate=False：避免校准文件缺失时 connect() 进入交互式标定并阻塞
        self._robot.connect(calibrate=False)
        if not self._robot.is_calibrated:
            self.close()
            raise SystemExit(
                f"未找到 {robot_id} 的标定文件，拒绝继续（connect 会阻塞在交互式标定）。\n"
                "请先执行：lerobot-calibrate --robot.type=so101_follower "
                f"--robot.port={port} --robot.id={robot_id}"
            )

    def read_load(self) -> dict[str, float]:
        raw = self._robot.bus.sync_read("Present_Load", normalize=False)
        return {f"{name}.load": decode_feetech_load(value) for name, value in raw.items()}

    def read_state(self) -> dict[str, float]:
        raw = self._robot.bus.sync_read("Present_Position")
        return {f"{name}.pos": float(value) for name, value in raw.items()}

    def command_gripper(self, target_pos: float, settle_s: float) -> None:
        """把夹爪指到目标位置并等待稳定（其余关节不动）。"""
        self._robot.send_action({"gripper.pos": float(target_pos)})
        time.sleep(max(0.0, settle_s))

    def close(self) -> None:
        try:
            self._robot.disconnect()
        except Exception:  # 断开失败不影响结论输出
            pass


def _load_store(path: Path) -> dict:
    """读取既有探针记录；损坏时视为空，避免中断采集。"""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _collect(engine, key: str, pos_key: str, repeat: int) -> tuple[list[float], list[float | None]]:
    """连续采样指定关节负载与位置；缺键直接失败（探针需暴露真实读取问题）。

    位置与负载同时记录：CP0 的降级判据是"位置差 + 负载"联合判据，
    一次采集拿齐两项证据，避免二次出机。
    """
    loads: list[float] = []
    positions: list[float | None] = []
    for index in range(repeat):
        values = engine.read_load()
        if key not in values:
            raise SystemExit(f"读不到负载键 {key}；当前可用键：{sorted(values)}")
        state = engine.read_state()
        position = state.get(pos_key)
        loads.append(float(values[key]))
        positions.append(None if position is None else float(position))
        shown = "n/a" if position is None else f"{float(position):.1f}"
        print(f"  采样 {index + 1}/{repeat}: {key} = {loads[-1]:.2f} | {pos_key} = {shown}")
    return loads, positions


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 负载量程探针（P0）")
    parser.add_argument("--phase", choices=("empty", "loaded"), required=True,
                        help="empty=空载（钳口无物），loaded=钳口内有目标物体")
    parser.add_argument("--repeat", type=int, default=5, help="该阶段的采样次数")
    parser.add_argument("--joint", default="gripper", help="被测关节名（默认夹爪）")
    parser.add_argument("--margin", type=float, default=LOAD_MARGIN,
                        help="接触判定阈值（负载读数单位）")
    parser.add_argument("--close-pos", type=float, default=DEFAULT_CLOSE_POS,
                        help=f"闭合目标位置（默认 {DEFAULT_CLOSE_POS}，勿用 0 以免顶死限位）")
    parser.add_argument("--release-pos", type=float, default=DEFAULT_RELEASE_POS,
                        help=f"采样后退回的安全位置（默认 {DEFAULT_RELEASE_POS}）")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S,
                        help="指令后等待稳定的秒数")
    parser.add_argument("--no-command", action="store_true",
                        help="只读模式：不驱动夹爪（需你已用其他方式保持闭合，且不要与遥操作同时占用串口）")
    parser.add_argument("--port", default="COM6", help="follower 串口")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--json", default="runs/week38/load_probe.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="用 MockEngine 合成负载验证链路，不连接机械臂")
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat 至少为 1")
    if not args.no_command and args.close_pos <= 0:
        raise SystemExit("--close-pos 必须大于 0（0 会顶住机械限位，有堵转/过载风险）")

    key = f"{args.joint}.load"
    pos_key = f"{args.joint}.pos"
    store_path = Path(args.json)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = _load_store(store_path)

    for phase in ("empty", "loaded"):
        record = store.get(phase)
        if isinstance(record, dict) and record.get("key") not in (None, key):
            print(f"[warn] 既有 {phase} 记录使用键 {record['key']}，与本次 {key} 不一致，将覆盖")
            store.pop(phase, None)

    print(f"[{'dry-run' if args.dry_run else '真机'}] 阶段={args.phase} 关节={args.joint} "
          f"采样={args.repeat}")
    if not args.no_command:
        print(f"  夹爪时序: 张开到 {args.release_pos} → 闭合到 {args.close_pos} → 稳定 {args.settle}s "
              f"→ 采样 → 退回 {args.release_pos}")
        if args.phase == "loaded":
            print("  注意：目标物体必须此刻已在张开的钳口内，探针会闭合到物体上")

    reader = None
    try:
        if args.dry_run:
            from real_executor import MockEngine

            engine = MockEngine(gripper_load=55.0 if args.phase == "empty" else 180.0)
            loads, positions = _collect(engine, key, pos_key, args.repeat)
        else:
            reader = RealLoadReader(args.port, args.robot_id)
            if not args.no_command:
                # 显式先张开：消除"上一次连接残留的 Goal_Position"带来的不确定性
                # （例如中途跑过推理/遥操作脚本，残留目标可能不是张开位）。
                print(f"  张开到 {args.release_pos} …")
                reader.command_gripper(args.release_pos, args.settle)
                reader.command_gripper(args.close_pos, args.settle)
            loads, positions = _collect(reader, key, pos_key, args.repeat)
            if not args.no_command:
                print(f"  退回安全位置 {args.release_pos} …")
                reader.command_gripper(args.release_pos, args.settle)
    finally:
        if reader is not None:
            reader.close()

    store[args.phase] = {"key": key, "samples": loads, "positions": positions,
                         "joint": args.joint, "close_pos": None if args.no_command else args.close_pos}

    exit_code = 0
    if "empty" in store and "loaded" in store:
        result = summarize_load_probe(store["empty"]["samples"], store["loaded"]["samples"],
                                      margin=args.margin)
        store["result"] = asdict(result)
        empty_pos = _mean(store["empty"].get("positions", []))
        loaded_pos = _mean(store["loaded"].get("positions", []))
        print("\n=== 负载量程结论 ===")
        print(result.note)
        if empty_pos is not None and loaded_pos is not None:
            print(f"夹爪位置: 空载 {empty_pos:.1f} / 夹持 {loaded_pos:.1f} / "
                  f"Δ={loaded_pos - empty_pos:+.1f}")
            close_pos = store["loaded"].get("close_pos") or store["empty"].get("close_pos")
            if close_pos is not None:
                if empty_pos > close_pos + 0.5:
                    print(f"  说明: 空载停在 {empty_pos:.1f} > 指令 {close_pos}，说明已被机械限位"
                          "挡住（close-pos 已在极限之外，属正常，不是没执行到）")
                if loaded_pos > empty_pos + 0.5:
                    print(f"  说明: 夹持比空载多留 {loaded_pos - empty_pos:.1f}，物体确实挡住了钳口")
                else:
                    print("  说明: 夹持与空载几乎停在同一位置——物体没挡住钳口（太薄或没放正）")
            store["result"]["empty_pos"] = empty_pos
            store["result"]["loaded_pos"] = loaded_pos
        print(f"可分辨(resolvable) = {result.resolvable}")
        print("→ B2（负载/接触特征）可执行" if result.resolvable
              else "→ B2 需降级为「位置差 + 负载」联合判据")
        if not result.resolvable:
            exit_code = 1
    else:
        print(f"\n已记录 {args.phase}；还有另一阶段未采集，暂不出结论。")

    store_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"记录已写入 {store_path}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
