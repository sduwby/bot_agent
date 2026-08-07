"""运行第 12 周低维 BC 训练和独立 rollout 验收。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from behavior_cloning import (
    BCTrainingConfig,
    BehaviorCloningError,
    evaluate_bc_checkpoint,
    train_behavior_cloning,
)
from maniskill_intro import ManiSkillUnavailableError


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / "maniskill_bc" / timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析数据、训练和未见 seed 评测参数。"""
    parser = argparse.ArgumentParser(
        description="训练并独立评测第 12 周低维 Behavior Cloning"
    )
    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=Path("runs/maniskill_expert/week11-200/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-episodes-per-task", type=int, default=80)
    parser.add_argument("--validation-episodes-per-task", type=int, default=20)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--random-seed", type=int, default=20260807)
    parser.add_argument("--eval-seed-start", type=int, default=100)
    parser.add_argument("--eval-episodes-per-task", type=int, default=50)
    parser.add_argument("--success-threshold", type=float, default=0.70)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """训练后必须从 checkpoint 重新加载并通过独立 rollout 门禁。"""
    args = parse_args(argv)
    output_dir = args.output_dir or _default_output_dir()
    try:
        if args.eval_seed_start < 0:
            raise ValueError("eval_seed_start 不能为负数")
        if args.eval_episodes_per_task <= 0:
            raise ValueError("eval_episodes_per_task 必须是正整数")
        config = BCTrainingConfig(
            dataset_summary=args.dataset_summary,
            output_dir=output_dir,
            train_episodes_per_task=args.train_episodes_per_task,
            validation_episodes_per_task=args.validation_episodes_per_task,
            hidden_dims=tuple(args.hidden_dims),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            random_seed=args.random_seed,
        )
        training = train_behavior_cloning(config)
        print(f"训练目录：{config.output_dir}")
        for task in training.tasks:
            print(
                f"{task.env_id}: train={task.train_transitions}, "
                f"validation={task.validation_transitions}, "
                f"best_epoch={task.best_epoch}, "
                f"validation_loss={task.best_validation_loss:.6f}"
            )
        seeds = range(
            args.eval_seed_start,
            args.eval_seed_start + args.eval_episodes_per_task,
        )
        evaluation = evaluate_bc_checkpoint(
            Path(training.checkpoint_path),
            seeds=seeds,
            threshold=args.success_threshold,
        )
        for task in evaluation.tasks:
            print(
                f"{task.env_id}: successes={task.successes}/"
                f"{task.episodes} ({task.success_rate:.1%})"
            )
        print(
            f"总体独立评测：{evaluation.total_successes}/"
            f"{evaluation.total_episodes} "
            f"({evaluation.success_rate:.1%})"
        )
        if not evaluation.passed:
            print(
                f"未达到成功率门禁 {evaluation.threshold:.1%}。",
                file=sys.stderr,
            )
            return 1
    except (
        BehaviorCloningError,
        ManiSkillUnavailableError,
        OSError,
        ValueError,
    ) as exc:
        print(f"第 12 周 BC 训练/评测失败：{exc}", file=sys.stderr)
        return 1

    print("第 12 周低维 BC 训练与独立评测验收通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
