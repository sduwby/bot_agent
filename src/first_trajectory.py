#!/usr/bin/env python3
"""运行一条最小 AI2-THOR 轨迹，并保存可检查、可复现的运行产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai2thor.controller import Controller
from PIL import Image


# 第一条固定轨迹只验证移动、转向和视角控制，暂不加入物体交互。
DEFAULT_ACTIONS: tuple[dict[str, Any], ...] = (
    {"action": "MoveAhead"},
    {"action": "RotateRight"},
    {"action": "MoveAhead"},
    {"action": "RotateLeft"},
    {"action": "LookDown", "degrees": 30},
    {"action": "LookUp", "degrees": 30},
)


def parse_args() -> argparse.Namespace:
    """解析场景、输出目录和渲染尺寸等命令行参数。"""
    parser = argparse.ArgumentParser(
        description="运行固定的 AI2-THOR 动作序列，并保存 RGB 帧和 JSONL 轨迹。"
    )
    parser.add_argument("--scene", default="FloorPlan1", help="要加载的场景名称")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/first_trajectory"),
        help="运行产物的根目录",
    )
    parser.add_argument("--width", type=int, default=480, help="输出图像宽度")
    parser.add_argument("--height", type=int, default=360, help="输出图像高度")
    return parser.parse_args()


def create_run_dir(output_root: Path) -> Path:
    """为本次运行创建独立目录，避免覆盖之前的轨迹。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{timestamp}-{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def visible_object_summary(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """提取当前真正可见的物体，并按与智能体的距离排序。"""
    objects = []
    for obj in metadata.get("objects", []):
        if not obj.get("visible"):
            continue
        objects.append(
            {
                "objectId": obj.get("objectId"),
                "objectType": obj.get("objectType"),
                "distance": obj.get("distance"),
                "pickupable": obj.get("pickupable", False),
                "receptacle": obj.get("receptacle", False),
            }
        )
    return sorted(objects, key=lambda obj: (obj["distance"] or float("inf")))


def inventory_object_ids(metadata: dict[str, Any]) -> list[str]:
    """返回智能体当前持有物体的唯一标识。"""
    return [
        obj["objectId"]
        for obj in metadata.get("inventoryObjects", [])
        if obj.get("objectId")
    ]


def record_event(
    run_dir: Path,
    trajectory_file,
    step_index: int,
    requested_action: dict[str, Any],
    event,
) -> dict[str, Any]:
    """保存当前 RGB 帧，并把动作结果、位姿和可见物体写入 JSONL。"""
    frame_name = f"step_{step_index:03d}.png"
    frame_path = run_dir / frame_name
    Image.fromarray(event.frame).save(frame_path)

    metadata = event.metadata
    agent = metadata.get("agent", {})
    record = {
        "step": step_index,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scene": metadata.get("sceneName"),
        "requested_action": requested_action,
        "last_action": metadata.get("lastAction"),
        "success": metadata.get("lastActionSuccess"),
        "error_message": metadata.get("errorMessage") or None,
        "agent": {
            "position": agent.get("position"),
            "rotation": agent.get("rotation"),
            "camera_horizon": agent.get("cameraHorizon"),
            "is_standing": agent.get("isStanding"),
        },
        "held_object_ids": inventory_object_ids(metadata),
        "visible_objects": visible_object_summary(metadata),
        "frame": frame_name,
        "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        "frame_shape": list(event.frame.shape),
    }
    trajectory_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    trajectory_file.flush()
    return record


def main() -> int:
    """初始化场景、执行固定动作序列并生成运行汇总。"""
    args = parse_args()
    run_dir = create_run_dir(args.output_root)
    trajectory_path = run_dir / "trajectory.jsonl"
    controller = None
    records: list[dict[str, Any]] = []

    try:
        # 使用低画质降低首个实验的渲染开销；网格移动步长为 0.25 米。
        controller = Controller(
            scene=args.scene,
            width=args.width,
            height=args.height,
            quality="Low",
            gridSize=0.25,
            rotateStepDegrees=90,
        )
        with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
            # 第 0 步保存初始化后的观察，作为后续动作的基准状态。
            records.append(
                record_event(
                    run_dir,
                    trajectory_file,
                    0,
                    {"action": "Initialize"},
                    controller.last_event,
                )
            )
            # 每执行一个动作立即落盘，避免中途异常导致整条轨迹丢失。
            for step_index, action in enumerate(DEFAULT_ACTIONS, start=1):
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
    finally:
        # 即使动作或写盘失败，也要关闭 Unity 进程并释放端口。
        if controller is not None:
            controller.stop()

    # 汇总只统计初始化之后的实际动作，初始化帧不计入成功率。
    action_records = records[1:]
    summary = {
        "scene": records[0]["scene"],
        "steps_recorded": len(records),
        "actions_attempted": len(action_records),
        "actions_succeeded": sum(record["success"] is True for record in action_records),
        "actions_failed": [
            {
                "step": record["step"],
                "action": record["requested_action"],
                "error_message": record["error_message"],
            }
            for record in action_records
            if record["success"] is not True
        ],
        "trajectory": trajectory_path.name,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"运行目录：{run_dir}")
    print(f"记录状态数：{summary['steps_recorded']}")
    print(
        "动作成功数："
        f"{summary['actions_succeeded']}/{summary['actions_attempted']}"
    )
    return 0 if not summary["actions_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
