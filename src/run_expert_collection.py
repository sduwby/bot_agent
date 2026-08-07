"""运行第 11 周 ManiSkill 专家轨迹采集与回放验收。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from expert_trajectory import (
    TrajectoryCollectionConfig,
    TrajectoryDataError,
    collect_expert_dataset,
    replay_trajectory_dataset,
)
from maniskill_intro import (
    EpisodeContractError,
    ManiSkillUnavailableError,
    ObservationContractError,
)


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / "maniskill_expert" / timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析采集数量、seed、重试和回放参数。"""
    parser = argparse.ArgumentParser(
        description="采集并回放第 11 周 ManiSkill scripted expert 轨迹"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-attempts-per-task", type=int, default=300)
    parser.add_argument("--replay-count", type=int, default=50)
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="仅用于采集诊断；正式验收不得跳过回放",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """采集成功数据集并按要求抽检；任一门禁失败返回非零。"""
    args = parse_args(argv)
    output_dir = args.output_dir or _default_output_dir()
    try:
        config = TrajectoryCollectionConfig(
            output_dir=output_dir,
            episodes_per_task=args.episodes_per_task,
            seed_start=args.seed_start,
            max_attempts_per_task=args.max_attempts_per_task,
            replay_count=args.replay_count,
        )
        summary = collect_expert_dataset(config)
        print(f"轨迹目录：{config.output_dir}")
        for task in summary.tasks:
            print(
                f"{task.env_id}: successes={task.successes}/"
                f"{task.target_successes}, attempts={task.attempts}, "
                f"failures={task.failures}, steps={task.total_steps}"
            )
        print(
            f"静态检查：{summary.validated_episodes}/"
            f"{summary.total_successes} 条通过"
        )
        if not summary.complete:
            print(
                "成功轨迹数量或静态检查未达到门禁。",
                file=sys.stderr,
            )
            return 1

        if args.skip_replay:
            print(
                "已跳过真实回放；"
                "本次结果不能作为第 11 周最终验收。"
            )
            return 0

        replay = replay_trajectory_dataset(
            config.output_dir / "summary.json",
            replay_count=args.replay_count,
        )
        print(f"真实回放：{replay.successful}/{replay.requested} 条成功")
        if not replay.complete:
            failed = [
                f"{item.env_id}/traj_{item.episode_id}: {item.error}"
                for item in replay.episodes
                if not item.success
            ]
            print("回放失败：" + "; ".join(failed), file=sys.stderr)
            return 1
    except (
        EpisodeContractError,
        ManiSkillUnavailableError,
        ObservationContractError,
        TrajectoryDataError,
        OSError,
        ValueError,
    ) as exc:
        print(f"第 11 周轨迹采集失败：{exc}", file=sys.stderr)
        return 1

    print("第 11 周专家轨迹采集与回放验收通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
