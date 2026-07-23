#!/usr/bin/env python3
"""扫描周围环境，抓取最近的可抓取物体，并验证任务结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai2thor.controller import Controller

from first_trajectory import create_run_dir, inventory_object_ids, record_event


def parse_args() -> argparse.Namespace:
    """解析场景、输出目录和渲染尺寸。"""
    parser = argparse.ArgumentParser(
        description="扫描 360°，选择最近的可抓取物体并执行 PickupObject。"
    )
    parser.add_argument("--scene", default="FloorPlan1", help="要加载的场景名称")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/pickup_nearest"),
        help="运行产物的根目录",
    )
    parser.add_argument("--width", type=int, default=480, help="输出图像宽度")
    parser.add_argument("--height", type=int, default=360, help="输出图像高度")
    return parser.parse_args()


def collect_pickup_candidates(event, rotation_y: float) -> list[dict[str, Any]]:
    """收集当前视角内所有可见、可抓取的物体。"""
    candidates = []
    for obj in event.metadata.get("objects", []):
        if not obj.get("visible") or not obj.get("pickupable"):
            continue
        candidates.append(
            {
                "object_id": obj["objectId"],
                "object_type": obj["objectType"],
                "distance": obj["distance"],
                "rotation_y": rotation_y,
            }
        )
    return candidates


def add_new_candidates(
    candidates_by_id: dict[str, dict[str, Any]],
    event,
) -> None:
    """将当前视角的新候选物体加入集合，避免重复记录同一物体。"""
    rotation_y = event.metadata["agent"]["rotation"]["y"]
    for candidate in collect_pickup_candidates(event, rotation_y):
        candidates_by_id.setdefault(candidate["object_id"], candidate)


def rotate_to(
    controller: Controller,
    target_rotation_y: float,
    run_dir: Path,
    trajectory_file,
    records: list[dict[str, Any]],
    step_index: int,
) -> int:
    """以 90° 为步长向右转，直到面向发现目标时的方向。"""
    current_rotation_y = controller.last_event.metadata["agent"]["rotation"]["y"]
    clockwise_degrees = (target_rotation_y - current_rotation_y) % 360
    rotate_steps = int(round(clockwise_degrees / 90)) % 4

    for _ in range(rotate_steps):
        action = {"action": "RotateRight"}
        event = controller.step(**action)
        records.append(
            record_event(
                run_dir,
                trajectory_file,
                step_index,
                action,
                event,
            )
        )
        step_index += 1
    return step_index


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    """以易读 JSON 格式保存任务级结果。"""
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """完成扫描、目标选择、转向、抓取和结果验证。"""
    args = parse_args()
    run_dir = create_run_dir(args.output_root)
    trajectory_path = run_dir / "trajectory.jsonl"
    records: list[dict[str, Any]] = []
    candidates_by_id: dict[str, dict[str, Any]] = {}
    controller = None

    try:
        controller = Controller(
            scene=args.scene,
            width=args.width,
            height=args.height,
            quality="Low",
            gridSize=0.25,
            rotateStepDegrees=90,
        )

        with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
            # 保存初始视角，并把该视角内的候选物体加入集合。
            initial_event = controller.last_event
            records.append(
                record_event(
                    run_dir,
                    trajectory_file,
                    0,
                    {"action": "Initialize"},
                    initial_event,
                )
            )
            add_new_candidates(candidates_by_id, initial_event)

            # 连续右转四次完成 360° 扫描，最后回到初始方向。
            step_index = 1
            for _ in range(4):
                action = {"action": "RotateRight"}
                event = controller.step(**action)
                records.append(
                    record_event(
                        run_dir,
                        trajectory_file,
                        step_index,
                        action,
                        event,
                    )
                )
                add_new_candidates(candidates_by_id, event)
                step_index += 1

            candidates = sorted(
                candidates_by_id.values(),
                key=lambda candidate: candidate["distance"],
            )
            if not candidates:
                summary = {
                    "scene": initial_event.metadata.get("sceneName"),
                    "task_success": False,
                    "failure_reason": "360° 扫描后没有发现可抓取物体",
                    "pickup_candidates": [],
                    "trajectory": trajectory_path.name,
                }
                write_summary(run_dir, summary)
                print(f"运行目录：{run_dir}")
                print("任务失败：360° 扫描后没有发现可抓取物体")
                return 2

            # 选择直线距离最近的候选物体，并恢复到发现它时的朝向。
            target = candidates[0]
            step_index = rotate_to(
                controller,
                target["rotation_y"],
                run_dir,
                trajectory_file,
                records,
                step_index,
            )

            pickup_action = {
                "action": "PickupObject",
                "objectId": target["object_id"],
            }
            pickup_event = controller.step(**pickup_action)
            records.append(
                record_event(
                    run_dir,
                    trajectory_file,
                    step_index,
                    pickup_action,
                    pickup_event,
                )
            )

            # 动作成功且目标出现在 inventoryObjects 中，才算任务真正完成。
            held_object_ids = inventory_object_ids(pickup_event.metadata)
            task_success = (
                pickup_event.metadata.get("lastActionSuccess") is True
                and target["object_id"] in held_object_ids
            )
            summary = {
                "scene": initial_event.metadata.get("sceneName"),
                "task_success": task_success,
                "target": target,
                "pickup_candidates": candidates,
                "pickup_action_success": pickup_event.metadata.get(
                    "lastActionSuccess"
                ),
                "pickup_error_message": pickup_event.metadata.get("errorMessage") or None,
                "held_object_ids": held_object_ids,
                "steps_recorded": len(records),
                "trajectory": trajectory_path.name,
            }
            write_summary(run_dir, summary)
    finally:
        # 始终关闭 Unity，避免残留进程占用端口。
        if controller is not None:
            controller.stop()

    print(f"运行目录：{run_dir}")
    print(f"发现候选物体：{len(candidates)} 个")
    print(
        "选择目标："
        f"{target['object_type']}，距离 {target['distance']:.3f} 米"
    )
    print(f"抓取结果：{'成功' if task_success else '失败'}")
    return 0 if task_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
