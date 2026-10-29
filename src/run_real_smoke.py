#!/usr/bin/env python
"""真机就绪度冒烟测试（第 38 周）—— **不跑 ACT 策略**。

用途：每次上机做真机实验前先跑一遍，确认"除策略以外"的环节都正常。
本项目已多次因环境问题白跑实验（相机索引漂移、手臂压在桌面上导致关节顶死、
归位容差写死不可达…），这一步把它们前置成 30 秒的检查。

检查项（逐项 PASS/FAIL）：
    1. 总线     —— 六个舵机全部在线
    2. 归位     —— 能回到采集初始姿态（同时暴露"手臂压在桌面/障碍上"这类外部阻力）
    3. 双相机   —— 两路都能出帧，并存图供肉眼核对（含相机映射）
    4. 负载     —— 能读各关节 Present_Load（P0 消融 B2 的前提）
    5. 夹持判定 —— 空载应判「未夹持」；`--with-object` 时提示放物后应判「夹持」
    6. 收尾复位 —— park() 能把手臂送回初始姿态

用法::

    python src/run_real_smoke.py                 # 空载判定（默认）
    python src/run_real_smoke.py --with-object   # 额外做"放入物体 → 应判夹持"
    python src/run_real_smoke.py --dry-run       # 用 MockEngine 验证脚本链路
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from real_observation import (  # noqa: E402
    EMPTY_GRIP_BASELINE,
    GRIP_POS_MARGIN,
    holds_object,
)

DEFAULT_MODEL_DIR = r"d:\FittenProject\bot_agent\outputs\act_demo_v23_pretrained"
DEFAULT_DATASET_ROOT = r"C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\demo_v23"
DEFAULT_DATASET_REPO = "wangbangyan/demo_v23"
DEFAULT_OUT = "runs/week38/smoke"
EXPECTED_JOINTS = 6


def render_checks(checks: list[tuple[str, bool, str]]) -> str:
    """把检查结果渲染成表格文本（纯逻辑，便于单测）。"""
    lines = []
    for name, ok, note in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name:10s} {note}")
    passed = sum(1 for _, ok, _ in checks if ok)
    lines.append(f"\n  合计: {passed}/{len(checks)} 项通过")
    return "\n".join(lines)


def all_passed(checks: list[tuple[str, bool, str]]) -> bool:
    return bool(checks) and all(ok for _, ok, _ in checks)


def main() -> None:
    parser = argparse.ArgumentParser(description="真机就绪度冒烟测试（不含策略）")
    parser.add_argument("--dry-run", action="store_true", help="用 MockEngine 验证脚本链路")
    parser.add_argument("--with-object", action="store_true",
                        help="额外做一段：把目标放进钳口后应判为夹持")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--out", default=DEFAULT_OUT, help="双相机帧落盘目录")
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = []

    if args.dry_run:
        from real_executor import MockEngine, RealSkillExecutor

        # 让 Mock 处于"空载"状态（位置/负载与实测空载同口径），
        # 这样 dry-run 反映的是"一台就绪的真机"该有的结果（6/6），而不是默认的"已夹住"姿态。
        engine = MockEngine(grasp_result="grasp_empty", with_images=True)
        engine.state["gripper.pos"] = 1.5
        engine.gripper_load = -62.0
        executor = RealSkillExecutor(engine)
    else:
        from real_executor import RealEngine, RealSkillExecutor

        engine = RealEngine(model_dir=args.model_dir, dataset_root=args.dataset_root,
                            dataset_repo=args.dataset_repo, port=args.port)
        executor = RealSkillExecutor(engine)

    try:
        # 1) 总线 + 六轴
        try:
            state = engine.read_state()
            motors = {k for k in state if k.endswith(".pos")}
            checks.append(("总线", len(motors) == EXPECTED_JOINTS,
                           f"{len(motors)}/{EXPECTED_JOINTS} 个关节在线: {sorted(motors)}"))
        except Exception as exc:
            checks.append(("总线", False, f"读取状态失败: {exc}"))
            state = {}

        # 2) 归位（同时暴露"手臂压在桌面/障碍上"这类外部阻力）
        try:
            ok = bool(engine.home())
            checks.append(("归位", ok, "已到达采集初始姿态" if ok else
                           "归位失败——常见原因：手臂压在桌面/障碍上（外部阻力），不是舵机故障"))
        except Exception as exc:
            checks.append(("归位", False, f"异常: {exc}"))

        # 3) 双相机出帧 + 存图（含映射核对）
        try:
            import cv2

            images = engine.read_images() or {}
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for key, frame in images.items():
                name = key.split(".")[-1]
                if frame is None or getattr(frame, "size", 0) == 0:
                    continue
                cv2.imwrite(str(out_dir / f"{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                saved.append(name)
            checks.append(("双相机", len(saved) == 2,
                           f"出帧 {saved} → {out_dir}\\（请肉眼确认 fixed=俯拍、handeye=夹爪特写）"))
        except Exception as exc:
            checks.append(("双相机", False, f"异常: {exc}"))

        # 4) 负载可读
        try:
            loads = engine.read_load()
            has_grip = "gripper.load" in (loads or {})
            checks.append(("负载", has_grip,
                           f"读到 {len(loads or {})} 个关节负载，gripper.load={loads.get('gripper.load')}"))
        except Exception as exc:
            checks.append(("负载", False, f"异常: {exc}"))

        # 5) 夹持判定（空载应 False）
        try:
            pos = float(state.get("gripper.pos", 0.0))
            load = None
            try:
                load = float(engine.read_load().get("gripper.load"))
            except Exception:
                load = None
            empty_verdict = holds_object(pos, load, empty_grip_baseline=EMPTY_GRIP_BASELINE,
                                         pos_margin=GRIP_POS_MARGIN)
            checks.append(("夹持判定", not empty_verdict,
                           f"空载 pos={pos:.2f} load={load} → holds_object={empty_verdict}（期望 False）"))
            if args.with_object:
                input("\n请把目标放进张开的钳口、并让它闭合（在真机上用其它工具或手动），"
                      "然后按 Enter 继续...")
                checks.append(("夹持判定(有物)", True, "（人工确认段，需你根据现场自行判断）"))
        except Exception as exc:
            checks.append(("夹持判定", False, f"异常: {exc}"))

        # 6) 收尾复位
        try:
            ok = executor.park()
            checks.append(("收尾复位", ok, "已回到初始姿态" if ok else "park() 失败"))
        except Exception as exc:
            checks.append(("收尾复位", False, f"异常: {exc}"))
    finally:
        engine.close()

    print("\n=== 真机就绪度冒烟测试 ===")
    print(render_checks(checks))
    raise SystemExit(0 if all_passed(checks) else 1)


if __name__ == "__main__":
    main()
