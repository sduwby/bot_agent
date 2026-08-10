"""运行第 16 周视觉 ACT：训练、smoke 与固定 seed 评测。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from behavior_cloning import BehaviorCloningError
from maniskill_intro import SUPPORTED_TASKS, ManiSkillUnavailableError
from visual_act import (
    VisualACTError,
    VisualACTTrainingConfig,
    evaluate_visual_act,
    load_visual_act_checkpoints,
    train_visual_act,
    write_resource_report,
)


def _default_output_dir() -> Path:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (Path("runs") / "maniskill_act" / timestamp).expanduser().resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析视觉 ACT 数据、训练、固定评测和显存/时长记录参数。"""
    parser = argparse.ArgumentParser(
        description="训练并评测第 16 周视觉 ACT"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("runs/maniskill_lerobot/week15-cam"),
    )
    parser.add_argument("--repo-id", type=str, default="bot-agent/maniskill-week11")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--training-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--vision-backbone", type=str, default="resnet18"
    )
    parser.add_argument(
        "--pretrained-backbone",
        type=lambda v: v if v.lower() != "none" else None,
        default="ResNet18_Weights.IMAGENET1K_V1",
    )
    parser.add_argument("--dim-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=3200)
    parser.add_argument("--n-encoder-layers", type=int, default=4)
    parser.add_argument("--n-decoder-layers", type=int, default=1)
    parser.add_argument("--n-vae-encoder-layers", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--use-vae", type=lambda v: v.lower() in ("1", "true", "yes", "on"),
                        default=True)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--task", type=str, default="all",
                        choices=("all",) + SUPPORTED_TASKS)
    parser.add_argument("--random-seed", type=int, default=20260816)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--train-episodes-per-task", type=int, default=80)
    parser.add_argument("--validation-episodes-per-task", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=500)
    parser.add_argument("--eval-episodes-per-task", type=int, default=10)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.resolution <= 0:
            raise ValueError("resolution 必须是正整数")
        if args.training_steps <= 0:
            raise ValueError("training_steps 必须是正整数")
        if args.batch_size <= 0:
            raise ValueError("batch_size 必须是正整数")
        if args.eval_seed_start < 0:
            raise ValueError("eval_seed_start 不能为负数")
        if args.eval_episodes_per_task <= 0:
            raise ValueError("eval_episodes_per_task 必须是正整数")
        tasks = SUPPORTED_TASKS if args.task == "all" else (args.task,)
        output_dir = args.output_dir or _default_output_dir()
        config = VisualACTTrainingConfig(
            dataset_dir=args.dataset_dir,
            repo_id=args.repo_id,
            output_dir=output_dir,
            tasks=tasks,
            resolution=args.resolution,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            training_steps=args.training_steps,
            batch_size=args.batch_size,
            validation_interval=args.validation_interval,
            checkpoint_interval=args.checkpoint_interval,
            vision_backbone=args.vision_backbone,
            pretrained_backbone_weights=args.pretrained_backbone,
            dim_model=args.dim_model,
            n_heads=args.n_heads,
            dim_feedforward=args.dim_feedforward,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            n_vae_encoder_layers=args.n_vae_encoder_layers,
            latent_dim=args.latent_dim,
            use_vae=args.use_vae,
            kl_weight=args.kl_weight,
            random_seed=args.random_seed,
            device=args.device,
            train_episodes_per_task=args.train_episodes_per_task,
            validation_episodes_per_task=args.validation_episodes_per_task,
        )
        output_dir = config.output_dir
        peak_rss_before = _peak_rss()
        training_seconds = 0.0
        if args.skip_training:
            print(f"跳过训练，复用已有 checkpoint：{config.output_dir}")
        else:
            print(f"开始视觉 ACT 训练：resolution={config.resolution}, "
                  f"batch={config.batch_size}, device={config.device} ...")
            started = time.perf_counter()
            training = train_visual_act(config)
            training_seconds = time.perf_counter() - started
            print(f"训练目录：{config.output_dir}")
            for task in training.tasks:
                print(
                    f"{task.env_id}: train={task.train_transitions}, "
                    f"validation={task.validation_transitions}, "
                    f"best_step={task.best_step}, "
                    f"validation_loss={task.best_validation_loss:.6f}, "
                    f"peak_rss={task.peak_rss_mb:.1f} MB"
                )

        # 从磁盘重载完成最小闭环 smoke。
        policies = load_visual_act_checkpoints(
            config.output_dir,
            device=config.device,
            resolution=config.resolution,
            tasks=config.tasks,
        )
        print(f"checkpoint 重载：{sorted(policies)}")

        seeds = range(
            args.eval_seed_start,
            args.eval_seed_start + args.eval_episodes_per_task,
        )
        print("执行固定 seed 视觉 ACT rollout ...")
        started = time.perf_counter()
        evaluation = evaluate_visual_act(
            config.output_dir,
            seeds=seeds,
            device=config.device,
            resolution=config.resolution,
            tasks=config.tasks,
        )
        evaluation_seconds = time.perf_counter() - started
        for task in evaluation.tasks:
            print(
                f"  {task.env_id}: {task.successes}/{task.episodes} "
                f"({task.success_rate:.1%})，平均步数 {task.mean_steps:.1f}"
            )
        print(
            f"总体：{evaluation.total_successes}/{evaluation.total_episodes} "
            f"({evaluation.success_rate:.1%})，平均推理 "
            f"{evaluation.mean_inference_ms:.2f} ms/step"
        )
        report = write_resource_report(
            config.output_dir,
            device=config.device,
            resolution=config.resolution,
            batch_size=config.batch_size,
            training_seconds=training_seconds,
            evaluation_seconds=evaluation_seconds,
            evaluation=evaluation,
            peak_rss_before=peak_rss_before,
        )
        print(
            f"资源报告：train_rss={report.peak_rss_mb_train} MB，"
            f"total_rss={report.peak_rss_mb_total} MB，"
            f"train={report.training_seconds}s，"
            f"eval={report.evaluation_seconds}s"
        )
        return 0
    except (
        VisualACTError,
        BehaviorCloningError,
        ManiSkillUnavailableError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"第 16 周视觉 ACT 运行失败：{exc}", file=sys.stderr)
        return 1


def _peak_rss() -> float:
    """返回进程峰值常驻内存（MB），macOS 的 ru_maxrss 为字节。"""
    try:
        import platform
        import resource
    except ImportError:
        return 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return round(rss / (1024.0 * 1024.0), 3)
    return round(rss / 1024.0, 3)


if __name__ == "__main__":
    raise SystemExit(main())
