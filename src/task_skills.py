"""可被脚本或后续 Planner 调用的具身任务技能。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai2thor.controller import Controller

from first_trajectory import inventory_object_ids, record_event


@dataclass
class SkillResult:
    """技能执行的统一结果，便于上层根据成功或失败继续规划。"""

    success: bool
    data: dict[str, Any]
    failure_reason: str | None = None


class SkillExecutor:
    """将 AI2-THOR 原子动作封装成扫描、抓取和放置等高层技能。"""

    def __init__(
        self,
        controller: Controller,
        run_dir,
        trajectory_file,
        records: list[dict[str, Any]],
    ) -> None:
        self.controller = controller
        self.run_dir = run_dir
        self.trajectory_file = trajectory_file
        self.records = records
        self.step_index = 0

    def record_initial_observation(self) -> None:
        """将初始化后的观察作为轨迹第 0 步。"""
        self.records.append(
            record_event(
                self.run_dir,
                self.trajectory_file,
                self.step_index,
                {"action": "Initialize"},
                self.controller.last_event,
            )
        )
        self.step_index += 1

    def act(self, action: dict[str, Any]):
        """执行一个原子动作，并立即记录其结果。"""
        event = self.controller.step(**action)
        self.records.append(
            record_event(
                self.run_dir,
                self.trajectory_file,
                self.step_index,
                action,
                event,
            )
        )
        self.step_index += 1
        return event

    def rotate_to(self, target_rotation_y: float) -> None:
        """以 90° 网格右转到目标朝向。"""
        current_rotation_y = self.controller.last_event.metadata["agent"]["rotation"]["y"]
        clockwise_degrees = (target_rotation_y - current_rotation_y) % 360
        rotate_steps = int(round(clockwise_degrees / 90)) % 4
        for _ in range(rotate_steps):
            self.act({"action": "RotateRight"})

    def scan_pickupable(self) -> list[dict[str, Any]]:
        """原地扫描一周，收集所有可见、可抓取物体。"""
        candidates_by_id: dict[str, dict[str, Any]] = {}

        def collect(event) -> None:
            rotation_y = event.metadata["agent"]["rotation"]["y"]
            for obj in event.metadata.get("objects", []):
                if not obj.get("visible") or not obj.get("pickupable"):
                    continue
                candidates_by_id.setdefault(
                    obj["objectId"],
                    {
                        "object_id": obj["objectId"],
                        "object_type": obj["objectType"],
                        "distance": obj["distance"],
                        "rotation_y": rotation_y,
                    },
                )

        collect(self.controller.last_event)
        # 四次右转后回到初始朝向，使扫描不会改变后续技能的起始状态。
        for _ in range(4):
            collect(self.act({"action": "RotateRight"}))
        return sorted(candidates_by_id.values(), key=lambda item: item["distance"])

    def scan_receptacles(self, excluded_ids: set[str]) -> list[dict[str, Any]]:
        """扫描可见容器，忽略地面和目标物体原来的承载面。"""
        candidates_by_id: dict[str, dict[str, Any]] = {}

        def collect(event) -> None:
            rotation_y = event.metadata["agent"]["rotation"]["y"]
            for obj in event.metadata.get("objects", []):
                if not obj.get("visible") or not obj.get("receptacle"):
                    continue
                if obj["objectType"] == "Floor" or obj["objectId"] in excluded_ids:
                    continue
                candidates_by_id.setdefault(
                    obj["objectId"],
                    {
                        "object_id": obj["objectId"],
                        "object_type": obj["objectType"],
                        "distance": obj["distance"],
                        "rotation_y": rotation_y,
                        "openable": obj.get("openable", False),
                    },
                )

        collect(self.controller.last_event)
        for _ in range(4):
            collect(self.act({"action": "RotateRight"}))
        return sorted(candidates_by_id.values(), key=lambda item: item["distance"])

    def pick(self, object_type: str) -> SkillResult:
        """找到指定类型的最近物体，抓取后通过 inventoryObjects 验证。"""
        candidates = [
            candidate
            for candidate in self.scan_pickupable()
            if candidate["object_type"].casefold() == object_type.casefold()
        ]
        if not candidates:
            return SkillResult(
                success=False,
                data={"candidates": []},
                failure_reason=f"扫描后未发现可抓取的 {object_type}",
            )

        target = candidates[0]
        target_metadata = next(
            obj
            for obj in self.controller.last_event.metadata["objects"]
            if obj["objectId"] == target["object_id"]
        )
        source_parent_ids = target_metadata.get("parentReceptacles") or []

        self.rotate_to(target["rotation_y"])
        event = self.act(
            {"action": "PickupObject", "objectId": target["object_id"]}
        )
        success = (
            event.metadata.get("lastActionSuccess") is True
            and target["object_id"] in inventory_object_ids(event.metadata)
        )
        return SkillResult(
            success=success,
            data={
                "target": target,
                "source_parent_ids": source_parent_ids,
                "pickup_action_success": event.metadata.get("lastActionSuccess"),
                "held_object_ids": inventory_object_ids(event.metadata),
            },
            failure_reason=(
                None
                if success
                else event.metadata.get("errorMessage") or "抓取后未在 inventoryObjects 中找到目标"
            ),
        )

    def place(
        self,
        object_id: str,
        receptacle_type: str,
        excluded_ids: set[str],
    ) -> SkillResult:
        """放入指定类型容器；容器可打开时会自动打开和关闭。"""
        candidates = [
            candidate
            for candidate in self.scan_receptacles(excluded_ids)
            if candidate["object_type"].casefold() == receptacle_type.casefold()
        ]
        if not candidates:
            return SkillResult(
                success=False,
                data={"candidates": []},
                failure_reason=f"扫描后未发现容器 {receptacle_type}",
            )

        attempts: list[dict[str, Any]] = []
        for receptacle in candidates:
            self.rotate_to(receptacle["rotation_y"])
            attempt = {
                "receptacle": receptacle,
                "open_action_success": None,
                "put_action_success": False,
                "close_action_success": None,
                "placement_verified": False,
                "error_message": None,
            }

            latest_receptacle = self._object_metadata(receptacle["object_id"])
            # 上一个候选容器的开闭可能遮挡当前目标；不可见时不向 Unity 发出无效动作。
            if not latest_receptacle.get("visible", False):
                attempt["error_message"] = "目标容器当前不可见，已跳过"
                attempts.append(attempt)
                continue

            opened_by_this_attempt = False
            if receptacle["openable"] and not latest_receptacle.get("isOpen", False):
                open_event = self.act(
                    {"action": "OpenObject", "objectId": receptacle["object_id"]}
                )
                attempt["open_action_success"] = open_event.metadata.get(
                        "lastActionSuccess"
                    )
                if attempt["open_action_success"] is not True:
                    attempt["error_message"] = open_event.metadata.get("errorMessage")
                    attempts.append(attempt)
                    continue
                opened_by_this_attempt = True

            put_event = self.act(
                {"action": "PutObject", "objectId": receptacle["object_id"]}
            )
            attempt["put_action_success"] = put_event.metadata.get("lastActionSuccess")
            attempt["error_message"] = put_event.metadata.get("errorMessage") or None
            attempt["placement_verified"] = self._placement_verified(
                put_event,
                object_id,
                receptacle["object_id"],
            )
            if not attempt["placement_verified"]:
                # 放置失败后恢复本次主动打开的容器，避免影响后续候选或下一次任务。
                if opened_by_this_attempt:
                    close_event = self.act(
                        {"action": "CloseObject", "objectId": receptacle["object_id"]}
                    )
                    attempt["close_action_success"] = close_event.metadata.get(
                        "lastActionSuccess"
                    )
                attempts.append(attempt)
                continue

            attempts.append(attempt)
            if receptacle["openable"]:
                close_event = self.act(
                    {"action": "CloseObject", "objectId": receptacle["object_id"]}
                )
                attempt["close_action_success"] = close_event.metadata.get(
                    "lastActionSuccess"
                )
                if attempt["close_action_success"] is not True or not self._placement_verified(
                    close_event,
                    object_id,
                    receptacle["object_id"],
                ):
                    return SkillResult(
                        success=False,
                        data={"attempts": attempts},
                        failure_reason=close_event.metadata.get("errorMessage")
                        or "关闭容器后未能验证物体仍在容器内",
                    )

            final_event = self.controller.last_event
            return SkillResult(
                success=True,
                data={
                    "selected_receptacle": receptacle,
                    "attempts": attempts,
                    "final_held_object_ids": inventory_object_ids(
                        final_event.metadata
                    ),
                    "final_object_parent_ids": self._object_metadata(
                        object_id
                    ).get("parentReceptacles")
                    or [],
                    "final_receptacle_contains_object": object_id
                    in (
                        self._object_metadata(receptacle["object_id"]).get(
                            "receptacleObjectIds"
                        )
                        or []
                    ),
                    "final_receptacle_is_open": self._object_metadata(
                        receptacle["object_id"]
                    ).get("isOpen"),
                },
            )

        return SkillResult(
            success=False,
            data={"attempts": attempts},
            failure_reason="所有指定类型容器都无法完成经过验证的放置",
        )

    def _object_metadata(self, object_id: str) -> dict[str, Any]:
        """从最新状态中读取一个物体的完整结构化信息。"""
        return next(
            obj
            for obj in self.controller.last_event.metadata["objects"]
            if obj["objectId"] == object_id
        )

    def _placement_verified(self, event, object_id: str, receptacle_id: str) -> bool:
        """验证物体已离开手中，并已被目标容器实际持有。"""
        if object_id in inventory_object_ids(event.metadata):
            return False
        object_metadata = next(
            obj for obj in event.metadata["objects"] if obj["objectId"] == object_id
        )
        receptacle_metadata = next(
            obj
            for obj in event.metadata["objects"]
            if obj["objectId"] == receptacle_id
        )
        return (
            receptacle_id in (object_metadata.get("parentReceptacles") or [])
            and object_id in (receptacle_metadata.get("receptacleObjectIds") or [])
        )
