#!/usr/bin/env python
"""SmolVLA 推理 runner（阶段 F S12）：替换 run_real_agent 的 VLARunner 占位。

- --dry-run：FakeVLAPolicy + Mock 状态/帧，验证 prepost 与主循环链路（无需真实模型）
- 真实模式：SmolVLAPolicy.from_pretrained(model_dir) + RealEngine 观测，
  每步 select_action → send_action（关节量，dataset 口径）
- --reset-per-step：每步 policy.reset() 清空 action queue 强制重推
  （SmolVLA 默认 n_action_steps>1 复用整段 chunk，与 ACT 的 queue 问题同源，
  真机实测后决定默认值；代码先暴露开关）

用法：
  python src/run_vla_infer.py --dry-run --instruction "夹起螺丝刀"
  python src/run_vla_infer.py --model-dir <smolvla checkpoint> --instruction "..." \
      --dataset-root <demo_v23> [--reset-per-step]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_obs(engine) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """从引擎取 state + 两路相机帧（尽力而为，缺图给 None 键占位）。"""
    state = engine.read_state()
    state_arr = np.array([state.get(n, 0.0) for n in
                          ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                           "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]],
                         dtype=np.float32)
    images: dict[str, np.ndarray] = {}
    try:
        frames = engine.read_images() or {}
        images = {k: v for k, v in frames.items() if v is not None}
    except Exception:
        pass
    return state_arr, images


def run_real(model_dir, instruction, dataset_root, dataset_repo, reset_per_step, steps):
    """真实 SmolVLA 推理（待微调 checkpoint 就绪后启用）。

    链路（对应 docs/week37/design.md §3）：
      policy = SmolVLAPolicy.from_pretrained(model_dir).to(device)   # 需联网/本地 checkpoint
      engine = RealEngine(...)                                        # 取 state + fixed/handeye 帧
      每步: batch = smolvla_prepare(state, images, instruction, device=device)
            action = smolvla_post(policy.select_action(batch))        # 或先 policy.reset()
            engine.send 动作...
    """
    if not model_dir or not dataset_root:
        raise SystemExit(
            "真机推理需要 --model-dir（SmolVLA 微调 checkpoint）与 --dataset-root。"
            "SmolVLA checkpoint 需先微调或联网加载 smolvla_base（本机未缓存）。"
            "框架链路请用 --dry-run 验证。"
        )
    raise NotImplementedError(
        "SmolVLA 真机推理实现待 checkpoint 就绪后按上方注释补全（文档 §6 遗留）。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLA 推理（S12 框架）")
    parser.add_argument("--dry-run", action="store_true",
                        help="FakeVLAPolicy + Mock 状态，验证 prepost/循环链路")
    parser.add_argument("--model-dir", default="", help="SmolVLA checkpoint 目录")
    parser.add_argument("--instruction", default="夹起桌面上的螺丝刀")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--dataset-repo", default="")
    parser.add_argument("--reset-per-step", action="store_true",
                        help="每步 policy.reset() 强制重推（默认 false 用 action queue）")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    if args.dry_run:
        run_dry(args)
    else:
        run_real(args.model_dir, args.instruction, args.dataset_root,
                 args.dataset_repo, args.reset_per_step, args.steps)


def run_dry(args) -> None:
    from vla_prepost import FakeVLAPolicy, smolvla_post, smolvla_prepare

    policy = FakeVLAPolicy(action_dim=6)
    state = np.zeros(6, dtype=np.float32)
    images = {
        "observation.images.fixed": np.full((64, 64, 3), 128, dtype=np.uint8),
        "observation.images.handeye": np.full((64, 64, 3), 200, dtype=np.uint8),
    }
    action_trace = []
    for step in range(args.steps):
        if args.reset_per_step:
            policy.reset()
        batch = smolvla_prepare(state, images, args.instruction)
        act = smolvla_post(policy.select_action(batch), action_dim=6)
        action_trace.append(act)
        # 模拟推进：状态朝动作缓慢移动（保持闭环形态，不追求收敛）
        state = 0.5 * state + 0.5 * act
    print(f"[dry-run] 策略调用 {policy.calls} 次，reset {policy.reset_calls} 次，"
          f"输出 {len(action_trace)} 条动作，末步动作={[round(float(v), 2) for v in action_trace[-1]]}")
    print("[dry-run] SmolVLA 推理链路 OK（未加载真实模型，仅框架验证）")


if __name__ == "__main__":
    main()
