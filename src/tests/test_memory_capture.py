"""MemoryCapture 单元测试（不依赖 AI2-THOR）。"""

from pathlib import Path

from memory_capture import MemoryCapture
from memory_store import MemoryRecord
from observation import Observation


def _make_observation(
    scene_name: str = "FloorPlan1",
    position: dict | None = None,
    rotation_y: float = 90.0,
    pickupable: list[dict] | None = None,
    receptacles: list[dict] | None = None,
    held_ids: list[str] | None = None,
    held_types: list[str] | None = None,
) -> Observation:
    return Observation(
        scene_name=scene_name,
        position=position or {"x": 1.0, "y": 0.0, "z": 1.0},
        rotation_y=rotation_y,
        held_object_ids=held_ids or [],
        held_object_types=held_types or [],
        visible_pickupable=pickupable or [],
        visible_receptacles=receptacles or [],
    )


class TestMemoryCapture:
    """测试 MemoryCapture 捕获适配器。"""

    def test_observation_to_conversation_instruction(self) -> None:
        obs = _make_observation()
        records = MemoryCapture.observation_to_conversation(
            "把信用卡放进抽屉", "sess1", [obs], []
        )
        assert len(records) == 1
        assert records[0].role == "user"
        assert records[0].content == "把信用卡放进抽屉"
        assert records[0].session_key == "sess1"
        assert records[0].step == 0

    def test_observation_to_conversation_with_actions(self) -> None:
        obs = _make_observation()
        actions = [
            {"skill": "pick", "parameters": {"object_type": "CreditCard"}, "success": True, "result_summary": "成功"},
        ]
        records = MemoryCapture.observation_to_conversation(
            "拿信用卡", "sess2", [obs], actions
        )
        assert len(records) == 2
        assert records[0].role == "user"
        assert records[1].role == "assistant"
        assert records[1].step == 1
        assert "pick" in records[1].content
        assert "CreditCard" in records[1].content

    def test_observation_to_conversation_step_increment(self) -> None:
        obs1 = _make_observation()
        obs2 = _make_observation(position={"x": 2.0, "y": 0.0, "z": 1.0})
        actions = [
            {"skill": "pick", "parameters": {}, "success": True, "result_summary": "成功"},
            {"skill": "move_to", "parameters": {"direction": "forward"}, "success": True, "result_summary": "成功"},
        ]
        records = MemoryCapture.observation_to_conversation(
            "任务", "sess3", [obs1, obs2], actions
        )
        assert records[0].step == 0
        assert records[1].step == 1
        assert records[2].step == 2

    def test_extract_memories_empty_observation(self) -> None:
        obs = _make_observation()
        memories = MemoryCapture.extract_memories_from_observation(obs, 0)
        assert memories == []

    def test_extract_memories_pickupable_objects(self) -> None:
        obs = _make_observation(
            pickupable=[
                {"object_id": "cc1", "object_type": "CreditCard", "distance": 2.3},
                {"object_id": "ap1", "object_type": "Apple", "distance": 1.5},
            ]
        )
        memories = MemoryCapture.extract_memories_from_observation(obs, 0)
        assert len(memories) == 2
        assert memories[0].type == "object_location"
        assert "CreditCard" in memories[0].content
        assert "Apple" in memories[1].content
        assert memories[0].priority == 80

    def test_extract_memories_receptacles(self) -> None:
        obs = _make_observation(
            receptacles=[
                {"object_id": "dr1", "object_type": "Drawer", "distance": 1.5, "openable": True, "is_open": True},
                {"object_id": "cb1", "object_type": "Cabinet", "distance": 3.0, "openable": True, "is_open": False},
            ]
        )
        memories = MemoryCapture.extract_memories_from_observation(obs, 1)
        assert len(memories) == 2
        assert memories[0].type == "room_state"
        assert "已打开" in memories[0].content
        assert "已关闭" in memories[1].content
        assert memories[0].priority == 60

    def test_extract_memories_held_objects(self) -> None:
        obs = _make_observation(
            held_ids=["cc1"],
            held_types=["CreditCard"],
        )
        memories = MemoryCapture.extract_memories_from_observation(obs, 2)
        assert len(memories) == 1
        assert memories[0].type == "action_experience"
        assert "CreditCard" in memories[0].content
        assert "cc1" in memories[0].content
        assert memories[0].priority == 90

    def test_extract_memories_mixed(self) -> None:
        obs = _make_observation(
            pickupable=[{"object_id": "cc1", "object_type": "CreditCard", "distance": 2.0}],
            receptacles=[{"object_id": "dr1", "object_type": "Drawer", "distance": 1.0, "openable": True, "is_open": True}],
            held_ids=["ap1"],
            held_types=["Apple"],
        )
        memories = MemoryCapture.extract_memories_from_observation(obs, 3)
        assert len(memories) == 3
        types = [m.type for m in memories]
        assert "object_location" in types
        assert "room_state" in types
        assert "action_experience" in types

    def test_extract_failure_memory(self) -> None:
        mem = MemoryCapture.extract_failure_memory(
            skill_name="pick",
            params={"object_type": "CreditCard"},
            failure_reason="物体不在视野中",
            step=5,
            scene_name="FloorPlan1",
        )
        assert mem.type == "failure_lesson"
        assert "pick" in mem.content
        assert "CreditCard" in mem.content
        assert "物体不在视野中" in mem.content
        assert mem.id == "m_fail_5"
        assert mem.priority == 70