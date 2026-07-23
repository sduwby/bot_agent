#!/usr/bin/env python3
"""以明确参数运行“抓取指定物体并放入指定容器”的脚本化任务。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai2thor.controller import Controller

from first_trajectory import create_run_dir, inventory_object_ids
from task_skills import SkillExecutor


def parse_args() -> argparse.Namespace:
    """解析任务目标和运行配置。"""
    parser = argparse.ArgumentParser(
        description="执行 pick(物体类型) → place(容器类型) 的脚本化具身任务。"
    )
    parser.add_argument("--object-type", default="CreditCard", help="要抓取的物体类型")
    parser.add_argument("--receptacle-type", default="Drawer", help="目标容器类型")
    parser.add_argument("--scene", default="FloorPlan1", help="要加载的场景名称")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/scripted_task"),
        help="运行产物的根目录",
    )
    parser.add_argument("--width", type=int, default=480, help="输出图像宽度")
    parser.add_argument("--height", type=int, default=360, help="输出图像高度")
    return parser.parse_args()


def write_summary(run_dir: Path, summary: dict) -> None:
    """将任务层面的输入、结果和失败原因保存为 JSON。"""
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_task_once(
    scene: str,
    object_type: str,
    receptacle_type: str,
    output_root: Path,
    width: int = 480,
    height: int = 360,
) -> dict[str, Any]:
    """执行一次 pick→place 任务，返回可重用的任务摘要。"""
    run_dir = create_run_dir(output_root)
    trajectory_path = run_dir / "trajectory.jsonl"
    records: list[dict] = []
    pick_result = None
    place_result = None
    final_held_object_ids: list[str] = []
    controller = None
    task_success = False
    final_metadata = None
    error_message: str | None = None

    try:
        controller = Controller(
            scene=scene,
            width=width,
            height=height,
            quality="Low",
            gridSize=0.25,
            rotateStepDegrees=90,
        )
        with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
            skills = SkillExecutor(controller, run_dir, trajectory_file, records)
            skills.record_initial_observation()

            pick_result = skills.pick(object_type)
            if pick_result.success:
                place_result = skills.place(
                    pick_result.data["target"]["object_id"],
                    receptacle_type,
                    set(pick_result.data["source_parent_ids"]),
                )
            final_metadata = controller.last_event.metadata
            final_held_object_ids = inventory_object_ids(final_metadata)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
    finally:
        if controller is not None:
            controller.stop()
            if final_metadata is None:
                final_metadata = controller.last_event.metadata
                final_held_object_ids = inventory_object_ids(final_metadata)

    task_success = bool(
        pick_result and pick_result.success and place_result and place_result.success
    )
    summary = {
        "scene": scene,
        "task": {
            "skill_sequence": ["pick", "place"],
            "object_type": object_type,
            "receptacle_type": receptacle_type,
        },
        "task_success": task_success,
        "pick_result": asdict(pick_result) if pick_result else None,
        "place_result": asdict(place_result) if place_result else None,
        "final_held_object_ids": final_held_object_ids,
        "steps_recorded": len(records),
        "trajectory": trajectory_path.name,
        "error_message": error_message,
    }
    write_summary(run_dir, summary)
    return summary | {"run_dir": str(run_dir)}


def main() -> int:
    """按 pick → place 顺序调用标准技能，并保留所有轨迹证据。"""
    args = parse_args()
    summary = run_task_once(
        scene=args.scene,
        object_type=args.object_type,
        receptacle_type=args.receptacle_type,
        output_root=args.output_root,
        width=args.width,
        height=args.height,
    )
    run_dir = Path(summary["run_dir"])
    task_success = bool(summary["task_success"])

    print(f"运行目录：{run_dir}")
    print(f"任务：pick({args.object_type}) → place({args.receptacle_type})")
    print(f"任务结果：{'成功' if task_success else '失败'}")
    return 0 if task_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
