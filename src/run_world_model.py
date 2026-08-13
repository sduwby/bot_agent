"""第 21 周：世界模型支线 CLI（dynamics 训练 + PushCube 演示评测）。

用法示例：

    python src/run_world_model.py train \\
        --dataset-dir runs/maniskill_lerobot/week15-cam \\
        --repo-id bot-agent/maniskill-week11 \\
        --output-dir runs/maniskill_world_model/week21-pushcube

    python src/run_world_model.py eval \\
        --output-dir runs/maniskill_world_model/week21-pushcube \\
        --seeds 500-509
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from world_model import (
    WorldModelConfig,
    WorldModelError,
    evaluate_world_model,
    load_dynamics_data,
    train_dynamics,
)


def _parse_seeds(text: str) -> list[int]:
    """解析 "500-509" 或 "500,600" 形式的 seed 表达式。"""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError(f"seed 区间非法：{part!r}") from exc
            if start < 0 or end < start:
                raise ValueError(f"seed 区间非法：{part!r}")
            seeds.extend(range(start, end + 1))
        else:
            try:
                seeds.append(int(part))
            except ValueError as exc:
                raise ValueError(f"seed 非法：{part!r}") from exc
    return seeds


def _train(args: argparse.Namespace) -> int:
    try:
        hidden_dims = tuple(
            int(part) for part in args.hidden_dims.split(",") if part.strip()
        )
        config = WorldModelConfig(
            dataset_dir=args.dataset_dir,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            hidden_dims=hidden_dims,
            train_episodes=args.train_episodes,
            validation_episodes=args.validation_episodes,
            training_steps=args.training_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        data = load_dynamics_data(config.dataset_dir, config.repo_id)
        summary = train_dynamics(config, data)
    except (ValueError, WorldModelError) as exc:
        print(f"训练失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"dynamics 训练完成：{len(data.states)} transitions → "
        f"{summary['checkpoint_dir']}（best_step={summary['best_step']}，"
        f"val_loss={summary['best_validation_loss']:.6f}）"
    )
    return 0


def _eval(args: argparse.Namespace) -> int:
    try:
        seeds = tuple(_parse_seeds(args.seeds))
        config = WorldModelConfig(
            dataset_dir=args.output_dir,
            repo_id="eval-only",
            output_dir=args.output_dir,
            seeds=seeds,
        )
        report = evaluate_world_model(config)
    except (ValueError, WorldModelError) as exc:
        print(f"评测失败：{exc}", file=sys.stderr)
        return 1
    world_model = report["world_model"]
    baseline = report["scripted_baseline"]
    print(
        f"PushCube CEM MPC：{world_model['successes']}/{world_model['episodes']} "
        f"({world_model['success_rate']:.1%})，mean_steps={world_model['mean_steps']:.1f}"
    )
    print(
        f"scripted 基线：{baseline['successes']}/{baseline['episodes']} "
        f"({baseline['success_rate']:.1%})，mean_steps={baseline['mean_steps']:.1f}"
    )
    print(f"报告 → {args.output_dir}/demo_report.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练 dynamics 模型")
    train_parser.add_argument("--dataset-dir", type=Path, required=True)
    train_parser.add_argument("--repo-id", required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--hidden-dims", default="128,128")
    train_parser.add_argument("--train-episodes", type=int, default=80)
    train_parser.add_argument("--validation-episodes", type=int, default=20)
    train_parser.add_argument("--training-steps", type=int, default=4000)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--device", default="cpu")

    eval_parser = subparsers.add_parser("eval", help="PushCube 演示评测")
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.add_argument("--seeds", default="500-509")

    args = parser.parse_args(argv)
    if args.command == "train":
        return _train(args)
    return _eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
