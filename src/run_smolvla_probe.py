"""运行第 18 周 SmolVLA 适配探针：零样本固定 seed 评测与 go/no-go 决策。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smolvla_probe import (
    SmolVLAProbeConfig,
    SmolVLAProbeError,
    TASK_TEXTS,
    evaluate_smolvla_probe,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第 18 周 SmolVLA 适配探针（零样本评测 + go/no-go）"
    )
    parser.add_argument("--model-id", type=str, default="lerobot/smolvla_base")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/smolvla/week18"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--seeds", type=str, default="500-519",
        help="固定 seed 集（逗号分隔，支持 start-end）",
    )
    parser.add_argument(
        "--task", type=str, default="all",
        choices=("all",) + tuple(TASK_TEXTS),
    )
    parser.add_argument("--perturbed", action="store_true")
    parser.add_argument("--max-steps", type=int, default=50)
    return parser.parse_args(argv)


def _parse_seeds(text: str) -> tuple[int, ...]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if not parts:
        raise ValueError("seeds 不能为空")
    seeds: list[int] = []
    for part in parts:
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return tuple(seeds)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        tasks = tuple(TASK_TEXTS) if args.task == "all" else (args.task,)
        config = SmolVLAProbeConfig(
            output_dir=args.output_dir,
            model_id=args.model_id,
            device=args.device,
            seeds=_parse_seeds(args.seeds),
            tasks=tasks,
            perturbed=args.perturbed,
            max_steps=args.max_steps,
        )
        print(
            f"SmolVLA 探针：model={config.model_id} device={config.device} "
            f"seeds={config.seeds} tasks={config.tasks}"
        )
        report = evaluate_smolvla_probe(config)
        for task in report.tasks:
            print(
                f"  {task.env_id}: {task.successes}/{task.episodes} "
                f"({task.success_rate:.1%}) xy_scale={task.mean_xy_scale:.3f} "
                f"z_scale={task.mean_z_scale:.3f} "
                f"gripper_close={task.gripper_close_ratio:.0%} "
                f"saturation={task.action_saturation:.0%} "
                f"truncated_all={task.truncated_all}"
            )
        print(f"go/no-go: {'GO' if report.go else 'NO-GO'} — {report.decision}")
        print(f"报告写入：{config.output_dir / 'probe_report.json'}")
        return 0
    except (SmolVLAProbeError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f"第 18 周 SmolVLA 探针失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
