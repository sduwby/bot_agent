"""运行第 14 周稳定性评测：初始位姿采样、三策略评测、平滑扫描、失败分析与门禁。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from act_stability import (
    StabilityEvalConfig,
    StabilityEvalError,
    analyze_failures,
    default_stability_seeds,
    evaluate_stability,
    record_initial_poses,
    run_smoothing_sweep,
    summarize_pose_coverage,
)
from behavior_cloning import BehaviorCloningError
from maniskill_intro import ManiSkillUnavailableError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析第 14 周稳定性评测参数。"""
    parser = argparse.ArgumentParser(
        description="第 14 周稳定性与阶段 B 评测"
    )
    parser.add_argument(
        "--act-pick-checkpoint",
        type=Path,
        default=Path(
            "runs/maniskill_act/week13-dagger-train-r3"
        ),
    )
    parser.add_argument(
        "--act-push-checkpoint",
        type=Path,
        default=Path("runs/maniskill_act/week13"),
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=Path("runs/maniskill_bc/week12/checkpoint.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/maniskill_act/week14"),
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=500,
        help="第 14 周评测起始 seed（默认 500）",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=50,
        help="每任务评测 episode 数（默认 50）",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument(
        "--smoothing-sweep",
        type=str,
        default=None,
        help="逗号分隔的 action_smoothing 扫描值，如 '0.5,0.75,0.9,1.0'",
    )
    parser.add_argument(
        "--skip-failure-analysis",
        action="store_true",
        help="跳过 ACT 失败相位分析（可显著缩短运行时间）",
    )
    parser.add_argument("--success-threshold", type=float, default=0.80)
    parser.add_argument("--max-steps", type=int, default=50)
    return parser.parse_args(argv)


def _parse_smoothing_values(value: str | None) -> list[float] | None:
    if value is None:
        return None
    items = [float(item) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("smoothing-sweep 不能为空")
    return items


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.seed_start < 0:
            raise ValueError("seed_start 不能为负数")
        if args.episodes_per_task <= 0:
            raise ValueError("episodes_per_task 必须是正整数")
        if args.device not in ("mps", "cpu"):
            raise ValueError("device 只支持 mps 或 cpu")
        if args.max_steps <= 0:
            raise ValueError("max_steps 必须为正数")
        smoothing_values = _parse_smoothing_values(args.smoothing_sweep)
        seeds = tuple(
            range(args.seed_start, args.seed_start + args.episodes_per_task)
        )
        config = StabilityEvalConfig(
            act_pick_checkpoint=args.act_pick_checkpoint,
            act_push_checkpoint=args.act_push_checkpoint,
            bc_checkpoint=args.bc_checkpoint,
            output_dir=args.output_dir,
            seeds=seeds,
            device=args.device,
            smoothing=args.smoothing,
            max_steps=args.max_steps,
            success_threshold=args.success_threshold,
        )
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print("记录初始位姿采样 ...")
        samples = record_initial_poses(config)
        coverage = summarize_pose_coverage(samples)
        from act_stability import _atomic_write_json

        _atomic_write_json(output_dir / "initial_poses.json", coverage)
        for env_id, stats in coverage.items():
            print(
                f"{env_id}: {stats['samples']} 个样本，"
                f"goal_z 范围 [{stats['goal_min'][2]:.3f}, "
                f"{stats['goal_max'][2]:.3f}]"
            )

        print("执行 ACT/BC/scripted 稳定性评测 ...")
        comparison = evaluate_stability(config)
        for policy in comparison.policies:
            print(
                f"{policy.policy_name}: {policy.total_successes}/"
                f"{policy.total_episodes} ({policy.success_rate:.1%})，"
                f"平均步数 {policy.mean_steps:.1f}"
            )
        print(
            f"门禁：ACT {comparison.act_rate:.1%} >= "
            f"max({comparison.threshold:.1%}, BC {comparison.bc_rate:.1%}) "
            f"= {comparison.effective_threshold:.1%}"
        )

        if smoothing_values is not None:
            print("执行 action_smoothing 扫描 ...")
            rows = run_smoothing_sweep(config, smoothing_values=smoothing_values)
            for row in rows:
                print(
                    f"smoothing={row.smoothing:.2f}: "
                    f"{row.total_successes}/{row.total_episodes} "
                    f"({row.success_rate:.1%})，nonfinite={row.nonfinite_actions}"
                )

        if not args.skip_failure_analysis:
            print("执行 ACT 失败相位分析 ...")
            failure_summaries = analyze_failures(config, seeds=seeds)
            for item in failure_summaries:
                print(
                    f"{item.env_id}: {item.total_failures} 个失败，"
                    f"相位分布 {item.phase_counts}"
                )

        if not comparison.act_passed:
            print(
                f"阶段 B 门禁未通过：ACT "
                f"{comparison.act_rate:.1%} < "
                f"{comparison.effective_threshold:.1%}"
            )
            return 1
        print("阶段 B 门禁通过：ACT 状态输入成功率达标且异常为 0。")
        return 0
    except (
        StabilityEvalError,
        BehaviorCloningError,
        ManiSkillUnavailableError,
        OSError,
        ValueError,
    ) as exc:
        print(f"第 14 周稳定性评测失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
