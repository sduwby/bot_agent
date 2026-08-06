"""运行第 10 周 PickCube / PushCube scripted policy 验收。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from maniskill_intro import (
    SUPPORTED_TASKS,
    EpisodeContractError,
    ManiSkillTaskConfig,
    ManiSkillUnavailableError,
    ObservationContractError,
    run_episode,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析任务、随机种子和输出格式。"""
    parser = argparse.ArgumentParser(
        description="运行 ManiSkill PickCube / PushCube 脚本策略"
    )
    parser.add_argument(
        "--task",
        choices=("all", *SUPPORTED_TASKS),
        default="all",
        help="默认依次运行两个第 10 周任务",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json", action="store_true", help="输出完整 JSON 摘要"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """运行所选任务；任一 scripted episode 失败则返回非零。"""
    args = parse_args(argv)
    tasks = SUPPORTED_TASKS if args.task == "all" else (args.task,)
    results = []
    try:
        for env_id in tasks:
            result = run_episode(
                ManiSkillTaskConfig(env_id=env_id, seed=args.seed)
            )
            results.append(result)
            if not args.json:
                field_names = ", ".join(
                    field.path for field in result.observation_fields
                )
                phases = " → ".join(dict.fromkeys(result.phase_history))
                print(
                    f"{env_id}: success={result.success}, "
                    f"steps={result.steps}, reward={result.total_reward:.4f}, "
                    f"terminated={result.terminated}, "
                    f"truncated={result.truncated}"
                )
                print(
                    f"  action_shape={result.action_shape}, "
                    f"phases={phases}"
                )
                print(f"  observation_fields={field_names}")
    except (
        EpisodeContractError,
        ManiSkillUnavailableError,
        ObservationContractError,
        ValueError,
    ) as exc:
        print(f"第 10 周 ManiSkill 验收失败：{exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False))
    if not all(result.success for result in results):
        failed = ", ".join(
            result.env_id for result in results if not result.success
        )
        print(f"scripted policy 未完成任务：{failed}", file=sys.stderr)
        return 1
    print("第 10 周 ManiSkill 验收通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
