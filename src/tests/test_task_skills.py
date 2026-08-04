"""task_skills 的单元测试 — 通过 mock 隔离 AI2-THOR 依赖。"""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch
from io import StringIO
from pathlib import Path

# 在导入 task_skills 之前 mock 外部依赖，避免 ModuleNotFoundError。
sys.modules["ai2thor"] = MagicMock()
sys.modules["ai2thor.controller"] = MagicMock()

mock_pil = MagicMock()
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_skills import SkillResult, SkillExecutor


# ---------------------------------------------------------------------------
# Helpers — 构建最小可用的 mock 环境
# ---------------------------------------------------------------------------

def _make_metadata(
    *,
    rotation_y: float = 0.0,
    position: dict | None = None,
    objects: list[dict] | None = None,
    inventory_objects: list[dict] | None = None,
    last_action_success: bool = True,
    error_message: str | None = None,
) -> dict:
    """生成一份符合 AI2-THOR 结构的 metadata。"""
    return {
        "sceneName": "FloorPlan1",
        "agent": {
            "position": position or {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0, "y": rotation_y, "z": 0},
            "cameraHorizon": 0,
            "isStanding": True,
        },
        "objects": objects or [],
        "inventoryObjects": inventory_objects or [],
        "lastActionSuccess": last_action_success,
        "errorMessage": error_message,
    }


def _make_event(metadata: dict):
    """生成一个 mock Event，包含 frame 和 metadata。"""
    event = MagicMock()
    event.metadata = metadata
    event.frame = MagicMock()
    event.frame.shape = (360, 480, 3)
    return event


def _make_frame_path_mock():
    """生成一个 mock 的 frame_path，支持 .read_bytes()。"""
    fp = MagicMock()
    fp.read_bytes.return_value = b"fake_frame_data"
    return fp


def _make_run_dir_mock():
    """生成一个 mock 的 run_dir，支持 / operator 返回 frame_path mock。"""
    run_dir = MagicMock(spec=Path)
    run_dir.__truediv__ = lambda self, name: _make_frame_path_mock()
    return run_dir


def _make_executor(
    objects: list[dict] | None = None,
    inventory_objects: list[dict] | None = None,
    rotation_y: float = 0.0,
    step_success: bool = True,
) -> tuple[SkillExecutor, MagicMock, MagicMock, list]:
    """创建 SkillExecutor 及其依赖的 mock 对象。"""
    controller = MagicMock()
    metadata = _make_metadata(
        rotation_y=rotation_y,
        objects=objects or [],
        inventory_objects=inventory_objects or [],
        last_action_success=step_success,
    )
    controller.last_event = _make_event(metadata)

    # step() 返回一个新 event，默认与当前状态一致
    def step_side_effect(**kwargs):
        meta = _make_metadata(
            rotation_y=rotation_y,
            objects=objects or [],
            inventory_objects=inventory_objects or [],
            last_action_success=step_success,
        )
        return _make_event(meta)
    controller.step.side_effect = step_side_effect

    run_dir = _make_run_dir_mock()
    trajectory_file = StringIO()
    records: list[dict] = []

    executor = SkillExecutor(controller, run_dir, trajectory_file, records)
    return executor, controller, trajectory_file, records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSkillResult:
    """SkillResult 是简单的数据类，验证基本行为即可。"""

    def test_success_result(self) -> None:
        r = SkillResult(success=True, data={"key": "value"})
        assert r.success is True
        assert r.data == {"key": "value"}
        assert r.failure_reason is None

    def test_failure_result(self) -> None:
        r = SkillResult(success=False, data={}, failure_reason="not found")
        assert r.success is False
        assert r.failure_reason == "not found"


class TestSkillExecutorAct:
    def test_act_calls_step(self) -> None:
        executor, controller, _, records = _make_executor()
        initial_step = executor.step_index
        executor.act({"action": "MoveAhead"})
        controller.step.assert_called_once_with(action="MoveAhead")
        assert executor.step_index == initial_step + 1
        assert len(records) == 1

    def test_act_records_event(self) -> None:
        executor, controller, _, records = _make_executor()
        executor.act({"action": "RotateRight"})
        record = records[0]
        assert record["step"] == 0
        assert record["requested_action"]["action"] == "RotateRight"


class TestSkillExecutorRotateTo:
    def test_rotate_right_90(self) -> None:
        # 当前朝向 0°，目标 90°，需要右转 1 步
        executor, controller, _, _ = _make_executor(rotation_y=0.0)
        executor.rotate_to(90.0)
        # 右转 1 次
        calls = [c for c in controller.step.call_args_list if c.kwargs.get("action") == "RotateRight"]
        assert len(calls) == 1

    def test_rotate_right_180(self) -> None:
        executor, controller, _, _ = _make_executor(rotation_y=0.0)
        executor.rotate_to(180.0)
        calls = [c for c in controller.step.call_args_list if c.kwargs.get("action") == "RotateRight"]
        assert len(calls) == 2

    def test_no_rotation_when_aligned(self) -> None:
        executor, controller, _, _ = _make_executor(rotation_y=90.0)
        executor.rotate_to(90.0)
        calls = [c for c in controller.step.call_args_list if c.kwargs.get("action") == "RotateRight"]
        assert len(calls) == 0

    def test_rotate_clockwise_wraparound(self) -> None:
        # 当前 270°，目标 0°，顺时针 90° → 1 步
        executor, controller, _, _ = _make_executor(rotation_y=270.0)
        executor.rotate_to(0.0)
        calls = [c for c in controller.step.call_args_list if c.kwargs.get("action") == "RotateRight"]
        assert len(calls) == 1


class TestSkillExecutorScanPickupable:
    def test_finds_visible_pickupable_objects(self) -> None:
        objects = [
            {"objectId": "obj1", "objectType": "CreditCard", "visible": True, "pickupable": True, "distance": 1.5},
            {"objectId": "obj2", "objectType": "Apple", "visible": True, "pickupable": True, "distance": 3.0},
        ]
        executor, _, _, _ = _make_executor(objects=objects)
        result = executor.scan_pickupable()
        # scan_pickupable calls collect on last_event, then 4 RotateRight steps.
        # last_event has the objects; the rotated steps return empty (mock returns same metadata)
        # so we get 2 candidates from initial view.
        assert len(result) == 2
        # 结果按距离排序
        assert result[0]["object_id"] == "obj1"
        assert result[1]["object_id"] == "obj2"

    def test_filters_invisible_objects(self) -> None:
        objects = [
            {"objectId": "obj1", "objectType": "Apple", "visible": True, "pickupable": True, "distance": 1.0},
            {"objectId": "obj2", "objectType": "Apple", "visible": False, "pickupable": True, "distance": 2.0},
        ]
        executor, _, _, _ = _make_executor(objects=objects)
        result = executor.scan_pickupable()
        assert len(result) == 1
        assert result[0]["object_id"] == "obj1"

    def test_filters_non_pickupable_objects(self) -> None:
        objects = [
            {"objectId": "obj1", "objectType": "CreditCard", "visible": True, "pickupable": True, "distance": 1.0},
            {"objectId": "obj2", "objectType": "Cabinet", "visible": True, "pickupable": False, "distance": 2.0},
        ]
        executor, _, _, _ = _make_executor(objects=objects)
        result = executor.scan_pickupable()
        assert len(result) == 1
        assert result[0]["object_id"] == "obj1"

    def test_returns_empty_when_no_candidates(self) -> None:
        executor, _, _, _ = _make_executor(objects=[])
        result = executor.scan_pickupable()
        assert result == []


class TestSkillExecutorPick:
    def test_pick_fails_when_no_candidates(self) -> None:
        objects = [
            {"objectId": "obj1", "objectType": "Apple", "visible": True, "pickupable": True, "distance": 1.0},
        ]
        executor, _, _, _ = _make_executor(objects=objects)
        result = executor.pick("CreditCard")
        assert result.success is False
        assert "CreditCard" in (result.failure_reason or "")

    def test_pick_success(self) -> None:
        card_id = "CreditCard_0"
        objects = [
            {"objectId": card_id, "objectType": "CreditCard", "visible": True, "pickupable": True, "distance": 1.5, "rotation": {"y": 90}},
            {"objectId": card_id, "objectType": "CreditCard", "distance": 1.5},
        ]
        # The pick flow: scan_pickupable → rotate_to → PickupObject
        # We need to mock step to return success on PickupObject
        executor, controller, _, _ = _make_executor(objects=objects)

        def step_side_effect(**kwargs):
            action = kwargs.get("action")
            if action == "PickupObject":
                meta = _make_metadata(
                    inventory_objects=[{"objectId": card_id, "objectType": "CreditCard"}],
                    last_action_success=True,
                )
                return _make_event(meta)
            else:
                meta = _make_metadata(objects=objects, last_action_success=True)
                return _make_event(meta)

        controller.step.side_effect = step_side_effect
        # Update last_event after scan changes it
        result = executor.pick("CreditCard")
        assert result.success is True
        assert result.data["target"]["object_id"] == card_id

    def test_pick_returns_failure_reason_when_action_fails(self) -> None:
        card_id = "CreditCard_0"
        objects = [
            {"objectId": card_id, "objectType": "CreditCard", "visible": True, "pickupable": True, "distance": 1.5, "rotation": {"y": 90}},
            {"objectId": card_id, "objectType": "CreditCard", "distance": 1.5},
        ]
        executor, controller, _, _ = _make_executor(objects=objects)

        def step_side_effect(**kwargs):
            action = kwargs.get("action")
            if action == "PickupObject":
                meta = _make_metadata(
                    last_action_success=False,
                    error_message="Cannot pickup this object",
                )
                return _make_event(meta)
            else:
                meta = _make_metadata(objects=objects, last_action_success=True)
                return _make_event(meta)

        controller.step.side_effect = step_side_effect
        result = executor.pick("CreditCard")
        assert result.success is False
        assert result.failure_reason is not None


class TestSkillExecutorRecordInitialObservation:
    def test_records_step_zero(self) -> None:
        executor, _, _, records = _make_executor()
        executor.record_initial_observation()
        assert len(records) == 1
        assert records[0]["step"] == 0
        assert records[0]["requested_action"]["action"] == "Initialize"
