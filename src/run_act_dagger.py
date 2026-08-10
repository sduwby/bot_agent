"""DAgger round：补采 recovery 轨迹 → 合并重训 ACT → dev 评测。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from behavior_cloning import BehaviorCloningError
from dagger_collect import (
    collect_dagger_episodes,
    load_dagger_episodes,
    merge_dagger_payloads,
    save_dagger_episodes,
)
from lerobot_act import LeRobotACTError
from maniskill_intro import ManiSkillUnavailableError
from run_act_training import main as act_training_main

TRAIN_EPISODES_BASE = 80


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DAgger round：补采 + 合并重训 + dev 评测"
    )
    parser.add_argument("--base-policy", type=Path, required=True,
                        help="用于诱导状态的 ACT policy 目录（含 best-rollout-dev）")
    parser.add_argument("--dagger-output", type=Path, required=True,
                        help="补采 npz 输出/输入路径")
    parser.add_argument("--dagger-seeds", type=str,
                        default="300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319")
    parser.add_argument("--takeover-zone", type=float, default=0.10,
                        help="ACT 抓取后 |obj-goal| 低于此值即专家接管")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--training-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=2500)
    parser.add_argument("--random-seed", type=int, default=20260807)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--eval-episodes-per-task", type=int, default=20)
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=Path("runs/maniskill_bc/week12/checkpoint.pt"),
    )
    parser.add_argument("--collect-only", action="store_true",
                        help="只补采不训练")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # 提前解析绝对路径，避免 ManiSkill 改变 CWD 后相对路径失效。
        base_policy = args.base_policy.expanduser().resolve()
        dagger_output = args.dagger_output.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        bc_checkpoint = args.bc_checkpoint.expanduser().resolve()
        dataset_summary = (
            Path("runs/maniskill_expert/week11-200/summary.json")
            .expanduser()
            .resolve()
        )
        seeds = [
            int(item) for item in args.dagger_seeds.split(",") if item.strip()
        ]
        if args.device not in ("mps", "cpu"):
            raise ValueError("device 只支持 mps 或 cpu")

        if not dagger_output.exists():
            print("收集 recovery 补采轨迹 ...")
            payload = collect_dagger_episodes(
                base_policy,
                seeds=seeds,
                device=args.device,
                takeover_zone=args.takeover_zone,
            )
            save_dagger_episodes(payload, dagger_output)
        else:
            # DAgger 多轮累积：本轮用当前 base policy 在 seeds 上再采集，
            # 与已有补采合并，保证训练集跨轮保留。
            print(f"累积补采：用 base policy 再采集 ...")
            new_payload = collect_dagger_episodes(
                base_policy,
                seeds=seeds,
                device=args.device,
                takeover_zone=args.takeover_zone,
            )
            existing = load_dagger_episodes(dagger_output)
            payload = merge_dagger_payloads([existing, new_payload])
            save_dagger_episodes(payload, dagger_output)
        dagger = load_dagger_episodes(dagger_output)
        n_episodes = int(dagger["n_episodes"])
        n_transitions = int(dagger["total_labeled_transitions"])
        print(
            f"补采：{n_episodes} episode / {n_transitions} transition"
        )
        if args.collect_only:
            return 0
        if n_episodes == 0:
            raise ValueError("补采没有产生 episode，无法重训")

        # warm-start 需要直接含 model.safetensors 的目录（best-rollout-dev）。
        warm_start_dir = (
            base_policy / "PickCube-v1" / "best-rollout-dev"
        )
        if not (warm_start_dir / "model.safetensors").is_file():
            raise LeRobotACTError(
                f"warm-start 目录缺少 model.safetensors：{warm_start_dir}"
            )
        training_argv = [
            "--task", "PickCube-v1",
            "--dataset-summary", str(dataset_summary),
            "--chunk-size", str(args.chunk_size),
            "--n-action-steps", str(args.n_action_steps),
            "--training-steps", str(args.training_steps),
            "--batch-size", str(args.batch_size),
            "--validation-interval", str(args.validation_interval),
            "--checkpoint-interval", str(args.checkpoint_interval),
            "--random-seed", str(args.random_seed),
            "--device", args.device,
            "--use-vae", "False",
            "--use-qvel",
            "--warm-start", str(warm_start_dir),
            "--dagger-data", str(dagger_output),
            "--train-episodes-per-task", str(TRAIN_EPISODES_BASE + n_episodes),
            "--dev-seed-start", "150",
            "--dev-episodes", "20",
            "--eval-seed-start", "150",
            "--eval-episodes-per-task", str(args.eval_episodes_per_task),
            "--bc-checkpoint", str(bc_checkpoint),
            "--output-dir", str(output_dir),
        ]
        print("启动 DAgger 重训（原始 80 + 补采 N 个 episode）...")
        return act_training_main(training_argv)
    except (
        LeRobotACTError,
        BehaviorCloningError,
        ManiSkillUnavailableError,
        OSError,
        ValueError,
    ) as exc:
        print(f"DAgger 运行失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
