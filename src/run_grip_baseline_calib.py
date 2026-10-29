#!/usr/bin/env python
"""SO-101 空闭基准标定（第 38 周 P0 / 子任务 0.6 —— 门禁项）。

用真机管线**自身的闭合行为**，在**空载**下测 `empty_grip_baseline` 的
跨运行 / 跨姿态波动，并与 `margin` 联合标定，给出「位置 OR 负载」判据建议。

**只测量、只给建议；不擅自改动真机判定行为**（`real_verifier` / `real_executor`
里的默认值保持不动，是否采纳由后续单独决策）。

方法：对 R 次**独立连接** × P 个**姿态**，每次执行
    张开 → 闭合到 close_pos → 稳定 → 采样 N 次 gripper.pos/load → 退回张开位
并记录。之所以自己下闭合指令：舵机只在"主动保持被指令的闭合位姿"时读数才有意义，
而 lerobot 断开即松力矩（同 `run_load_probe.py` 的说明）。

判据背景（`real_observation.is_holding`）：``gripper.pos > empty_grip_baseline + margin``。
CP0 实测空载闭合停在 **3.8**，而代码默认 baseline=**1.5**、margin=**3.0** →
阈值 4.5 看似宽松，但按实测基准 3.8 计算阈值应为 6.8，恰好等于薄件的夹持位置 → **临界漏判**。

用法::

    # 离线验证链路（合成读数，不连机械臂）
    python src/run_grip_baseline_calib.py --dry-run

    # 真机（默认 3 次连接 × 3 姿态 × 每姿态 5 次采样）
    python src/run_grip_baseline_calib.py --runs 3 --poses 60,75,90 ^
        --json runs/week38/grip_baseline.json

    # 只想测当前姿态、不动机械臂
    python src/run_grip_baseline_calib.py --runs 3 --poses current
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from real_observation import LOAD_MARGIN  # noqa: E402

# 夹爪位置与稳定时间（依据 week24 标定：闭≈0–10、开≈13–23）
DEFAULT_CLOSE_POS = 3.0
DEFAULT_RELEASE_POS = 20.0
DEFAULT_SETTLE_S = 1.0
# wrist_flex 的标定范围是 49–94（week24），只在其内取姿态
DEFAULT_POSES = "60,75,90"
# 单步关节增量限幅（安全，同 RealEngine 的 max_relative_target）
DEFAULT_MAX_RELATIVE_TARGET = 8.0


# ---------------------------------------------------------------------------
# 纯逻辑：统计与建议（可离线单测，不依赖机械臂）
# ---------------------------------------------------------------------------

def stats(values: list[float]) -> dict:
    """均值/标准差/极值。标准差用**总体**口径（这是全量采样，不是抽样推断）。"""
    present = [float(v) for v in values]
    if not present:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "range": None}
    mean = sum(present) / len(present)
    variance = sum((v - mean) ** 2 for v in present) / len(present)
    return {"n": len(present), "mean": round(mean, 3), "std": round(math.sqrt(variance), 3),
            "min": round(min(present), 3), "max": round(max(present), 3),
            "range": round(max(present) - min(present), 3)}


def group_means(trials: list[dict], field: str) -> dict[str, list[float]]:
    """按 ``field``（如 ``run`` / ``pose``）分组，收集各组 trial 的均值。"""
    groups: dict[str, list[float]] = {}
    for trial in trials:
        key = str(trial.get(field))
        value = trial.get("pos_mean")
        if value is None:
            continue
        groups.setdefault(key, []).append(float(value))
    return groups


def spread_by(trials: list[dict], field: str) -> dict:
    """某维度的波动：各组均值的极差（跨运行 / 跨姿态波动的直接度量）。"""
    groups = group_means(trials, field)
    means = [sum(v) / len(v) for v in groups.values() if v]
    return {"groups": {k: round(sum(v) / len(v), 3) for k, v in groups.items() if v},
            "range": round(max(means) - min(means), 3) if len(means) > 1 else 0.0}


def suggest_threshold(empty_positions: list[float],
                      hold_pos: float | None = None,
                      load_margin: float = LOAD_MARGIN,
                      headroom: float = 0.5) -> dict:
    """由空载位置样本给 ``(empty_grip_baseline, margin)`` 建议。

    - ``baseline`` 取**实测最大值向上取整到 0.1**：判据是严格大于，取上限才不会误判为空夹持。
    - ``margin`` 取 ``(hold_pos - baseline)`` 的 ``headroom``（默认一半），
      兼顾"薄件也要能检出"与"留够余量"。若余量为负说明该薄件位置已低于新基准 → 判据必然漏判。
    """
    empty = stats(empty_positions)
    if empty["n"] == 0:
        return {"ok": False, "note": "没有空载位置样本"}
    baseline = math.ceil(empty["max"] * 10) / 10.0
    out: dict = {
        "ok": True,
        "empty_stats": empty,
        "baseline_suggested": round(baseline, 2),
        "baseline_in_use": 1.5,
        "load_margin_in_use": load_margin,
    }
    if hold_pos is None:
        out["note"] = "未提供 --hold-pos（实测夹持位置），只给 baseline；margin 需结合夹持样本定"
        return out
    allowed = hold_pos - baseline
    out["hold_pos"] = hold_pos
    out["margin_max_allowed"] = round(allowed, 2)
    if allowed <= 0:
        out["margin_suggested"] = None
        out["note"] = f"夹持位置 {hold_pos} 未高于新基准 {baseline}，位置判据对该件**必然漏判**，须依赖负载判据"
        return out
    margin = round(max(0.5, allowed * headroom), 1)
    out["margin_suggested"] = margin
    out["threshold_suggested"] = round(baseline + margin, 2)
    out["note"] = (f"阈值 = {baseline} + {margin} = {out['threshold_suggested']}，"
                   f"低于夹持位置 {hold_pos}（余量 {round(allowed - margin, 2)}）")
    return out


# ---------------------------------------------------------------------------
# 真机读写 / 离线假读出
# ---------------------------------------------------------------------------

class RealGripReader:
    """真机夹爪读写：只连接电机总线，不加载策略/数据集。"""

    def __init__(self, port: str, robot_id: str,
                 max_relative_target: float = DEFAULT_MAX_RELATIVE_TARGET) -> None:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        self._robot = SO101Follower(SO101FollowerConfig(
            port=port, id=robot_id, cameras={},
            max_relative_target=max_relative_target))
        # calibrate=False：避免校准文件缺失时 connect() 进入交互式标定并阻塞
        self._robot.connect(calibrate=False)
        if not self._robot.is_calibrated:
            self.close()
            raise SystemExit(
                f"未找到 {robot_id} 的标定文件，拒绝继续。请先执行：\n"
                f"  lerobot-calibrate --robot.type=so101_follower --robot.port={port} "
                f"--robot.id={robot_id}")

    def read_state(self) -> dict[str, float]:
        raw = self._robot.bus.sync_read("Present_Position")
        return {f"{name}.pos": float(value) for name, value in raw.items()}

    def read_load(self) -> dict[str, float]:
        from real_observation import decode_feetech_load

        raw = self._robot.bus.sync_read("Present_Load", normalize=False)
        return {f"{name}.load": decode_feetech_load(value) for name, value in raw.items()}

    def _drive(self, target: dict[str, float], timeout_s: float = 10.0, tol: float = 0.5,
               stall_eps: float = 0.02, stall_patience: int = 12) -> str:
        """反复下发目标直到「到位」/「卡住」/超时。

        **必须反复下发**：``max_relative_target`` 会把单次指令的目标裁到
        "当前位置 ± 限幅"（实测单发一帧只走 8 个单位，日志形如
        ``goal_pos 3.0 → safe 11.18``）。只发一次的话夹爪根本到不了闭合位，
        测出来的"空闭位置"就是限幅残留，完全失真。

        Returns: ``"reached"`` / ``"stalled"``（顶到限位或物体）/ ``"timeout"``。
        """
        deadline = time.perf_counter() + timeout_s
        last_state: dict[str, float] | None = None
        stalled = 0
        while time.perf_counter() < deadline:
            self._robot.send_action({k: float(v) for k, v in target.items()})
            time.sleep(0.05)
            state = self.read_state()
            if all(abs(state.get(k, 0.0) - v) <= tol for k, v in target.items()):
                return "reached"
            if last_state is not None and all(
                    abs(state.get(k, 0.0) - last_state.get(k, 0.0)) <= stall_eps for k in target):
                stalled += 1
                if stalled >= stall_patience:
                    return "stalled"
            else:
                stalled = 0
            last_state = state
        return "timeout"

    def command(self, target: dict[str, float], settle_s: float) -> str:
        """驱动到目标后稳定；顶住限位会提前返回（夹爪闭合的正常情形）。"""
        status = self._drive(target)
        time.sleep(max(0.0, settle_s))
        return status

    def move_to(self, target: dict[str, float], settle_s: float) -> str:
        status = self._drive(target)
        time.sleep(max(0.0, settle_s))
        return status

    def close(self) -> None:
        try:
            self._robot.disconnect()
        except Exception:  # 断开失败不影响结论输出
            pass


class FakeGripReader:
    """离线链路验证：合成空载闭合位置（含跨运行零位漂移、姿态偏移与采样噪声）。

    数值仅为**链路自检**用，不代表任何真机结论。
    """

    def __init__(self, run_index: int = 0, seed: int = 0, close_pos: float = DEFAULT_CLOSE_POS,
                 release_pos: float = DEFAULT_RELEASE_POS) -> None:
        self._rng = random.Random(seed + 1000 * run_index)
        self._base = 3.4 + 0.25 * run_index       # 跨运行零位漂移
        self._pose_bias = 0.0                     # 跨姿态偏移
        self._pos = float(release_pos)
        self._close_pos = float(close_pos)

    def read_state(self) -> dict[str, float]:
        return {"gripper.pos": self._base + self._pose_bias + self._rng.gauss(0.0, 0.05),
                "wrist_flex.pos": 75.0 + self._pose_bias * 100.0}

    def read_load(self) -> dict[str, float]:
        return {"gripper.load": -55.0 + self._rng.gauss(0.0, 1.0)}

    def command(self, target: dict[str, float], settle_s: float) -> str:
        """闭合到 close_pos 时模拟"被限位挡住"（stalled），与真机行为一致。"""
        if "gripper.pos" in target:
            self._pos = float(target["gripper.pos"])
            if abs(self._pos - self._close_pos) < 1e-6:
                return "stalled"
        return "reached"

    def move_to(self, target: dict[str, float], settle_s: float) -> str:
        if "wrist_flex.pos" in target:
            self._pose_bias = (float(target["wrist_flex.pos"]) - 75.0) / 100.0
        return "reached"

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_poses(text: str) -> list[float | None]:
    """``"60,75,90"`` → ``[60.0, 75.0, 90.0]``；``"current"`` → ``[None]``（不动姿态）。"""
    if text.strip().lower() in ("current", "none", ""):
        return [None]
    try:
        return [float(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"--poses 应为逗号分隔的角度或 current，收到 {text!r}") from exc


def collect_trials(reader_factory, poses: list[float | None], runs: int, repeat: int,
                   close_pos: float, release_pos: float, settle: float,
                   joint: str) -> list[dict]:
    """执行 R×P 次「张开→闭合→采样→退回」，返回逐次记录。"""
    pos_key, load_key = f"{joint}.pos", f"{joint}.load"
    trials: list[dict] = []
    for run_index in range(runs):
        reader = reader_factory(run_index)
        try:
            for pose in poses:
                if pose is not None:
                    reader.move_to({"wrist_flex.pos": pose}, settle)
                reader.command({pos_key: release_pos}, settle)
                close_status = reader.command({pos_key: close_pos}, settle)
                positions, loads = [], []
                for _ in range(repeat):
                    positions.append(float(reader.read_state()[pos_key]))
                    loads.append(float(reader.read_load()[load_key]))
                reader.command({pos_key: release_pos}, settle)
                trial = {
                    "run": run_index, "pose": "current" if pose is None else pose,
                    "close_status": close_status,
                    "positions": [round(v, 3) for v in positions],
                    "pos_mean": round(sum(positions) / len(positions), 3),
                    "pos_std": stats(positions)["std"],
                    "load_mean": round(sum(loads) / len(loads), 2),
                }
                trials.append(trial)
                flag = "" if close_status == "stalled" else f"  ⚠ close_status={close_status}"
                print(f"  连接#{run_index} 姿态 {trial['pose']}: "
                      f"pos {trial['pos_mean']:.2f} (σ={trial['pos_std']:.3f}) | "
                      f"load {trial['load_mean']:.1f}‰{flag}")
        finally:
            reader.close()
    return trials


def build_report(trials: list[dict], args, poses: list[float | None]) -> dict:
    all_positions = [p for t in trials for p in t["positions"]]
    by_run = spread_by(trials, "run")
    by_pose = spread_by(trials, "pose")
    within = [t["pos_std"] for t in trials]
    return {
        "config": {
            "runs": args.runs, "poses": [str(p) for p in poses], "repeat": args.repeat,
            "joint": args.joint, "close_pos": args.close_pos, "release_pos": args.release_pos,
            "settle_s": args.settle, "hold_pos": args.hold_pos,
            "load_margin": args.load_margin, "dry_run": bool(args.dry_run),
        },
        "trials": trials,
        "aggregate": {
            "all_samples": stats(all_positions),
            "trial_means": stats([t["pos_mean"] for t in trials]),
            "within_trial_std_mean": round(sum(within) / len(within), 4) if within else None,
            "across_run": by_run,
            "across_pose": by_pose,
        },
        "recommendation": suggest_threshold([t["pos_mean"] for t in trials],
                                            hold_pos=args.hold_pos,
                                            load_margin=args.load_margin),
    }


def print_summary(report: dict) -> None:
    agg = report["aggregate"]
    rec = report["recommendation"]
    print("\n=== 空闭基准标定结论 ===")
    print(f"空载闭合位置（全部采样 {agg['all_samples']['n']} 个）: "
          f"均值 {agg['all_samples']['mean']} | σ {agg['all_samples']['std']} | "
          f"范围 [{agg['all_samples']['min']}, {agg['all_samples']['max']}]")
    print(f"  各次试验均值范围 : [{agg['trial_means']['min']}, {agg['trial_means']['max']}] "
          f"(极差 {agg['trial_means']['range']})")
    print(f"  单次内采样噪声 σ : {agg['within_trial_std_mean']}")
    print(f"  跨运行波动(极差) : {agg['across_run']['range']}  组均值 {agg['across_run']['groups']}")
    print(f"  跨姿态波动(极差) : {agg['across_pose']['range']}  组均值 {agg['across_pose']['groups']}")
    print(f"建议 empty_grip_baseline = {rec.get('baseline_suggested')} "
          f"(当前在用 {rec.get('baseline_in_use')})")
    if rec.get("margin_suggested") is not None:
        print(f"建议 margin = {rec['margin_suggested']} → 阈值 {rec['threshold_suggested']} "
              f"(当前在用 {rec.get('load_margin_in_use')} 为负载口径；位置 margin 现为 3.0)")
    for line in (rec.get("note") or "").split("\n"):
        if line:
            print(f"  说明: {line}")
    print("建议判据: 位置 OR 负载 —— "
          f"`pos > baseline + margin` 或 `|load| > {report['config']['load_margin']}‰`"
          "（CP0 实测夹持负载 Δ≈151‰，约为负载阈值的 3 倍，余量充足）")
    print("注意: 本脚本**只测量与给建议**，未改动任何真机判定行为。")


def silence_expected_clamp_warnings() -> None:
    """屏蔽 lerobot 的"目标被限幅"告警。

    本脚本**就是要**一直下发同一个目标直到顶住限位，因此每次下发都会打一条
    ``Relative goal position magnitude had to be clamped to be safe``；
    不限流的话输出会被几千行淹没。只挡这一条，其他日志照常输出。
    """

    class _ClampFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "clamped to be safe" not in record.getMessage()

    logging.getLogger().addFilter(_ClampFilter())


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 空闭基准标定（0.6 门禁）")
    parser.add_argument("--runs", type=int, default=3, help="独立连接次数（跨运行）")
    parser.add_argument("--poses", default=DEFAULT_POSES,
                        help=f"wrist_flex 姿态列表（逗号分隔）或 current；默认 {DEFAULT_POSES}")
    parser.add_argument("--repeat", type=int, default=5, help="每次试验的采样次数")
    parser.add_argument("--joint", default="gripper", help="被测关节（默认夹爪）")
    parser.add_argument("--close-pos", type=float, default=DEFAULT_CLOSE_POS,
                        help=f"闭合目标位置（默认 {DEFAULT_CLOSE_POS}，勿用 0 以免顶死限位）")
    parser.add_argument("--release-pos", type=float, default=DEFAULT_RELEASE_POS,
                        help=f"张开/退回位置（默认 {DEFAULT_RELEASE_POS}）")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S, help="指令后稳定秒数")
    parser.add_argument("--hold-pos", type=float, default=None,
                        help="实测的夹持位置（如 CP0 薄件的 6.8），用于给出 margin 建议")
    parser.add_argument("--load-margin", type=float, default=LOAD_MARGIN, help="负载判据阈值（‰）")
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--max-relative-target", type=float, default=DEFAULT_MAX_RELATIVE_TARGET,
                        help="单步关节增量限幅（安全）")
    parser.add_argument("--json", default="runs/week38/grip_baseline.json")
    parser.add_argument("--dry-run", action="store_true", help="用合成读数验证链路，不连机械臂")
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs 至少为 1")
    if args.repeat < 1:
        raise SystemExit("--repeat 至少为 1")
    if args.close_pos <= 0:
        raise SystemExit("--close-pos 必须大于 0（0 会顶住机械限位，有堵转/过载风险）")

    poses = parse_poses(args.poses)
    print(f"[{'dry-run' if args.dry_run else '真机'}] 空闭基准标定 | "
          f"{args.runs} 次连接 × {len(poses)} 姿态 × 每姿态 {args.repeat} 采样")
    if not args.dry_run:
        silence_expected_clamp_warnings()
        print("  夹爪时序: 张开 → 闭合 → 稳定 → 采样 → 退回张开")
        print(f"  姿态: {poses}（wrist_flex；标定范围 49–94，单步限幅 {args.max_relative_target}°）")
        print("  请确认：钳口内**无任何物体**（空载），且工作区无人手/障碍物")

    if args.dry_run:
        def factory(run_index: int):
            return FakeGripReader(run_index=run_index, close_pos=args.close_pos,
                                  release_pos=args.release_pos)
    else:
        def factory(run_index: int):
            return RealGripReader(args.port, args.robot_id, args.max_relative_target)

    trials = collect_trials(factory, poses, args.runs, args.repeat, args.close_pos,
                            args.release_pos, args.settle, args.joint)
    report = build_report(trials, args, poses)
    print_summary(report)

    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n记录已写入 {path}")


if __name__ == "__main__":
    main()
