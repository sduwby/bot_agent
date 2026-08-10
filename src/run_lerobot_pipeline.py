"""运行第 15 周 LeRobot 数据管线：转换、检查、浏览与可视化。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lerobot_pipeline import (
    LerobotConversionConfig,
    LerobotPipelineError,
    browse_lerobot_episode,
    convert_maniskill_to_lerobot,
    inspect_lerobot_dataset,
    visualize_lerobot_episode,
)
from maniskill_intro import ManiSkillUnavailableError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="第 15 周 LeRobot 数据管线"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="转换 week11 → LeRobotDataset v3")
    convert.add_argument(
        "--source-summary",
        type=Path,
        default=Path("runs/maniskill_expert/week11-200/summary.json"),
    )
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.add_argument("--repo-id", type=str, default="bot-agent/maniskill-week11")
    convert.add_argument("--fps", type=int, default=20)
    convert.add_argument("--include-camera", action="store_true")
    convert.add_argument("--episodes-per-task", type=int, default=None)
    convert.add_argument("--task", type=str, default=None,
                         help="只转换指定任务（逗号分隔）")
    convert.add_argument("--device", type=str, default="mps",
                         help="保留参数（转换走 CPU 渲染，ManiSkill 内部决定）")

    inspect = subparsers.add_parser("inspect", help="字段完整性与异常帧检查")
    inspect.add_argument("--dataset-dir", type=Path, required=True)
    inspect.add_argument("--repo-id", type=str, default="bot-agent/maniskill-week11")

    browse = subparsers.add_parser("browse", help="浏览单个 episode")
    browse.add_argument("--dataset-dir", type=Path, required=True)
    browse.add_argument("--repo-id", type=str, default="bot-agent/maniskill-week11")
    browse.add_argument("--episode-index", type=int, required=True)

    visualize = subparsers.add_parser("visualize", help="可视化单个 episode 相机帧")
    visualize.add_argument("--dataset-dir", type=Path, required=True)
    visualize.add_argument("--repo-id", type=str, default="bot-agent/maniskill-week11")
    visualize.add_argument("--episode-index", type=int, required=True)
    visualize.add_argument("--image-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "convert":
            tasks = None
            if args.task is not None:
                tasks = tuple(
                    item.strip()
                    for item in args.task.split(",")
                    if item.strip()
                )
            config = LerobotConversionConfig(
                source_summary=args.source_summary,
                output_dir=args.output_dir,
                repo_id=args.repo_id,
                fps=args.fps,
                include_camera=args.include_camera,
                episodes_per_task=args.episodes_per_task,
                tasks=tasks if tasks is not None else ("PickCube-v1", "PushCube-v1"),
            )
            result = convert_maniskill_to_lerobot(config)
            print(
                f"转换完成：{result.total_episodes} episode / "
                f"{result.total_frames} 帧，errors={result.total_errors}"
            )
            for task in result.tasks:
                print(
                    f"  {task.env_id}: {task.episodes} episode / "
                    f"{task.frames} 帧，errors={task.errors}"
                )
            if not result.complete:
                return 1
            return 0

        if args.command == "inspect":
            result = inspect_lerobot_dataset(args.dataset_dir, args.repo_id)
            print(
                f"检查：{result.total_episodes} episode / "
                f"{result.total_frames} 帧，missing={result.missing_count}，"
                f"abnormal_frames={result.abnormal_frames}"
            )
            if not result.complete:
                return 1
            return 0

        if args.command == "browse":
            result = browse_lerobot_episode(
                args.dataset_dir, args.repo_id, args.episode_index
            )
            print(f"episode {result['episode_index']}：{result['frame_count']} 帧")
            for row in result["frames"]:
                print(
                    f"  frame {row['frame_index']:4d} "
                    f"ts={row['timestamp']:.2f} "
                    f"state_norm={row['state_norm']:.3f} "
                    f"action={[round(float(v), 2) for v in row['action']]} "
                    f"success={row['success']} done={row['next_done']}"
                )
            return 0

        if args.command == "visualize":
            image_dir = args.image_dir or (
                args.dataset_dir / "episode_browser"
            )
            result = visualize_lerobot_episode(
                args.dataset_dir, args.repo_id, args.episode_index, image_dir
            )
            print(
                f"episode {result['episode_index']}：{result['frame_count']} 帧，"
                f"{len(result['thumbnails'])} 张缩略图"
            )
            return 0
        return 1
    except (
        LerobotPipelineError,
        ManiSkillUnavailableError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"第 15 周 LeRobot 数据管线失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
