#!/usr/bin/env python3
"""抓取最近物体，将其放入合适容器，并验证完整任务状态。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ai2thor.controller import Controller

from first_trajectory import create_run_dir, inventory_object_ids, record_event
from pickup_nearest import add_new_candidates, rotate_to, write_summary


# 优先选择语义明确、通常适合收纳小物体的容器；未知类型排在最后尝试。
RECEPTACLE_PRIORITY = {
    "Drawer": 0,
    "Box": 1,
    "Bowl": 2,
    "Plate": 3,
    "SinkBasin": 4,
    "GarbageCan": 5,
    "CounterTop": 6,
}


def parse_args() -> argparse.Namespace:
    """解析场景、输出目录和渲染尺寸。"""
    parser = argparse.ArgumentParser(
        description="抓取最近物体，寻找合适容器，放置后验证并关闭容器。"
    )
    parser.add_argument("--scene", default="FloorPlan1", help="要加载的场景名称")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/pickup_and_place"),
        help="运行产物的根目录",
    )
    parser.add_argument("--width", type=int, default=480, help="输出图像宽度")
    parser.add_argument("--height", type=int, default=360, help="输出图像高度")
    return parser.parse_args()


def add_new_receptacles(
    receptacles_by_id: dict[str, dict[str, Any]],
    event,
    excluded_ids: set[str],
) -> None:
    """收集当前视角可见的容器，并排除地面和物体原来的承载面。"""
    rotation_y = event.metadata["agent"]["rotation"]["y"]
    for obj in event.metadata.get("objects", []):
        if not obj.get("visible") or not obj.get("receptacle"):
            continue
        if obj["objectType"] == "Floor" or obj["objectId"] in excluded_ids:
            continue
        receptacles_by_id.setdefault(
            obj["objectId"],
            {
                "object_id": obj["objectId"],
                "object_type": obj["objectType"],
                "distance": obj["distance"],
                "rotation_y": rotation_y,
                "openable": obj.get("openable", False),
                "is_open": obj.get("isOpen", False),
            },
        )


def placement_verified(event, object_id: str, receptacle_id: str) -> bool:
    """从持有状态、物体父容器和容器内容三个边界验证放置结果。"""
    if object_id in inventory_object_ids(event.metadata):
        return False

    object_metadata = next(
        obj
        for obj in event.metadata["objects"]
        if obj["objectId"] == object_id
    )
    receptacle_metadata = next(
        obj
        for obj in event.metadata["objects"]
        if obj["objectId"] == receptacle_id
    )
    parent_receptacles = object_metadata.get("parentReceptacles") or []
    receptacle_contents = receptacle_metadata.get("receptacleObjectIds") or []
    return (
        receptacle_id in parent_receptacles
        and object_id in receptacle_contents
    )


def main() -> int:
    """执行抓取、容器搜索、放置、验证和容器关闭。"""
    args = parse_args()
    run_dir = create_run_dir(args.output_root)
    trajectory_path = run_dir / "trajectory.jsonl"
    records: list[dict[str, Any]] = []
    pickup_candidates_by_id: dict[str, dict[str, Any]] = {}
    receptacles_by_id: dict[str, dict[str, Any]] = {}
    placement_attempts: list[dict[str, Any]] = []
    controller = None
    task_success = False
    selected_receptacle = None

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
            add_new_candidates(pickup_candidates_by_id, initial_event)

            # 第一轮 360° 扫描寻找可抓取物体。
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
                add_new_candidates(pickup_candidates_by_id, event)
                step_index += 1

            pickup_candidates = sorted(
                pickup_candidates_by_id.values(),
                key=lambda candidate: candidate["distance"],
            )
            if not pickup_candidates:
                summary = {
                    "scene": initial_event.metadata.get("sceneName"),
                    "task_success": False,
                    "failure_reason": "没有发现可抓取物体",
                    "trajectory": trajectory_path.name,
                }
                write_summary(run_dir, summary)
                print(f"运行目录：{run_dir}")
                print("任务失败：没有发现可抓取物体")
                return 2

            target = pickup_candidates[0]
            target_before_pickup = next(
                obj
                for obj in controller.last_event.metadata["objects"]
                if obj["objectId"] == target["object_id"]
            )
            source_parent_ids = set(
                target_before_pickup.get("parentReceptacles") or []
            )

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
            step_index += 1
            pickup_verified = (
                pickup_event.metadata.get("lastActionSuccess") is True
                and target["object_id"] in inventory_object_ids(pickup_event.metadata)
            )
            if not pickup_verified:
                summary = {
                    "scene": initial_event.metadata.get("sceneName"),
                    "task_success": False,
                    "failure_reason": "抓取动作或 inventoryObjects 验证失败",
                    "target": target,
                    "pickup_error_message": pickup_event.metadata.get("errorMessage") or None,
                    "trajectory": trajectory_path.name,
                }
                write_summary(run_dir, summary)
                print(f"运行目录：{run_dir}")
                print("任务失败：没有成功拿起目标物体")
                return 3

            # 持有物体后再次扫描，寻找与原承载面不同的容器。
            add_new_receptacles(receptacles_by_id, pickup_event, source_parent_ids)
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
                add_new_receptacles(receptacles_by_id, event, source_parent_ids)
                step_index += 1

            receptacle_candidates = sorted(
                receptacles_by_id.values(),
                key=lambda receptacle: (
                    RECEPTACLE_PRIORITY.get(receptacle["object_type"], 100),
                    receptacle["distance"],
                ),
            )

            # 依次尝试候选容器；放置失败时仍持有物体，可以继续尝试下一个。
            for receptacle in receptacle_candidates:
                step_index = rotate_to(
                    controller,
                    receptacle["rotation_y"],
                    run_dir,
                    trajectory_file,
                    records,
                    step_index,
                )
                attempt = {
                    "receptacle": receptacle,
                    "open_action_success": None,
                    "put_action_success": False,
                    "close_action_success": None,
                    "placement_verified": False,
                    "error_message": None,
                }

                if receptacle["openable"] and not receptacle["is_open"]:
                    open_action = {
                        "action": "OpenObject",
                        "objectId": receptacle["object_id"],
                    }
                    open_event = controller.step(**open_action)
                    records.append(
                        record_event(
                            run_dir,
                            trajectory_file,
                            step_index,
                            open_action,
                            open_event,
                        )
                    )
                    step_index += 1
                    attempt["open_action_success"] = open_event.metadata.get(
                        "lastActionSuccess"
                    )
                    if attempt["open_action_success"] is not True:
                        attempt["error_message"] = open_event.metadata.get(
                            "errorMessage"
                        )
                        placement_attempts.append(attempt)
                        continue

                put_action = {
                    "action": "PutObject",
                    "objectId": receptacle["object_id"],
                }
                put_event = controller.step(**put_action)
                records.append(
                    record_event(
                        run_dir,
                        trajectory_file,
                        step_index,
                        put_action,
                        put_event,
                    )
                )
                step_index += 1
                attempt["put_action_success"] = put_event.metadata.get(
                    "lastActionSuccess"
                )
                attempt["error_message"] = put_event.metadata.get("errorMessage") or None
                attempt["placement_verified"] = (
                    attempt["put_action_success"] is True
                    and placement_verified(
                        put_event,
                        target["object_id"],
                        receptacle["object_id"],
                    )
                )
                placement_attempts.append(attempt)
                if not attempt["placement_verified"]:
                    continue

                selected_receptacle = receptacle
                task_success = True

                # 对已经打开的容器执行关闭动作，完成“收纳”而非只完成“放入”。
                if receptacle["openable"]:
                    close_action = {
                        "action": "CloseObject",
                        "objectId": receptacle["object_id"],
                    }
                    close_event = controller.step(**close_action)
                    records.append(
                        record_event(
                            run_dir,
                            trajectory_file,
                            step_index,
                            close_action,
                            close_event,
                        )
                    )
                    step_index += 1
                    attempt["close_action_success"] = close_event.metadata.get(
                        "lastActionSuccess"
                    )
                    task_success = (
                        attempt["close_action_success"] is True
                        and placement_verified(
                            close_event,
                            target["object_id"],
                            receptacle["object_id"],
                        )
                    )
                break

            failure_reason = None
            if not receptacle_candidates:
                failure_reason = "没有发现与原承载面不同的容器"
            elif not task_success:
                failure_reason = "候选容器均无法完成经过验证的放置"

            final_object_parent_ids: list[str] = []
            final_receptacle_contains_object = False
            final_receptacle_is_open = None
            if selected_receptacle is not None:
                final_event = controller.last_event
                final_object = next(
                    obj
                    for obj in final_event.metadata["objects"]
                    if obj["objectId"] == target["object_id"]
                )
                final_receptacle = next(
                    obj
                    for obj in final_event.metadata["objects"]
                    if obj["objectId"] == selected_receptacle["object_id"]
                )
                final_object_parent_ids = final_object.get("parentReceptacles") or []
                final_receptacle_contains_object = target["object_id"] in (
                    final_receptacle.get("receptacleObjectIds") or []
                )
                final_receptacle_is_open = final_receptacle.get("isOpen")

            summary = {
                "scene": initial_event.metadata.get("sceneName"),
                "task_success": task_success,
                "failure_reason": failure_reason,
                "target": target,
                "source_parent_ids": sorted(source_parent_ids),
                "pickup_candidates": pickup_candidates,
                "receptacle_candidates": receptacle_candidates,
                "selected_receptacle": selected_receptacle,
                "placement_attempts": placement_attempts,
                "final_held_object_ids": inventory_object_ids(
                    controller.last_event.metadata
                ),
                "final_object_parent_ids": final_object_parent_ids,
                "final_receptacle_contains_object": final_receptacle_contains_object,
                "final_receptacle_is_open": final_receptacle_is_open,
                "steps_recorded": len(records),
                "trajectory": trajectory_path.name,
            }
            write_summary(run_dir, summary)
    finally:
        # 始终关闭 Unity，避免残留进程占用端口。
        if controller is not None:
            controller.stop()

    print(f"运行目录：{run_dir}")
    print(f"抓取目标：{target['object_type']}")
    if selected_receptacle is not None:
        print(f"放置容器：{selected_receptacle['object_type']}")
    print(f"任务结果：{'成功' if task_success else '失败'}")
    return 0 if task_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
