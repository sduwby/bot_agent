"""MemoryRecall 单元测试（不依赖 AI2-THOR）。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from memory_recall import MemoryRecall
from memory_store import MemoryRecord, MemoryStore
from observation import Observation


def _make_observation(
    scene_name: str = "FloorPlan1",
    position: dict | None = None,
    rotation_y: float = 90.0,
    pickupable: list[dict] | None = None,
    receptacles: list[dict] | None = None,
) -> Observation:
    return Observation(
        scene_name=scene_name,
        position=position or {"x": 1.0, "y": 0.0, "z": 1.0},
        rotation_y=rotation_y,
        held_object_ids=[],
        held_object_types=[],
        visible_pickupable=pickupable or [],
        visible_receptacles=receptacles or [],
    )


def _make_memory(content: str, mtype: str = "object_location", priority: int = 80) -> MemoryRecord:
    now = datetime.now(timezone.utc).isoformat()
    return MemoryRecord(
        id=f"m_{content[:10]}",
        content=content,
        type=mtype,
        priority=priority,
        scene_name="FloorPlan1",
        source_step=0,
        created_at=now,
        updated_at=now,
    )


class TestMemoryRecall:
    """测试 MemoryRecall 召回模块。"""

    def _make_store(self, tmp_path: Path) -> MemoryStore:
        return MemoryStore(str(tmp_path), "recall_test")

    def test_build_recall_query_default(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        recall = MemoryRecall(store)
        obs = _make_observation()

        query = recall.build_recall_query("把信用卡放进抽屉", obs, [])
        assert query == "CreditCard Drawer"

    def test_build_recall_query_pick_failure(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        recall = MemoryRecall(store)
        obs = _make_observation()
        history = [
            {"skill": "pick", "parameters": {"object_type": "CreditCard"}, "success": False, "result_summary": "失败"},
        ]

        query = recall.build_recall_query("把信用卡放进抽屉", obs, history)
        assert query == "CreditCard"

    def test_build_recall_query_place_failure(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        recall = MemoryRecall(store)
        obs = _make_observation()
        history = [
            {"skill": "place", "parameters": {"receptacle_type": "Drawer"}, "success": False, "result_summary": "失败"},
        ]

        query = recall.build_recall_query("把信用卡放进抽屉", obs, history)
        assert query == "Drawer"

    def test_build_recall_query_success_action(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        recall = MemoryRecall(store)
        obs = _make_observation()
        history = [
            {"skill": "pick", "parameters": {"object_type": "CreditCard"}, "success": True, "result_summary": "成功"},
        ]

        query = recall.build_recall_query("把信用卡放进抽屉", obs, history)
        assert query == "CreditCard Drawer"

    def test_build_recall_query_held_object_targets_receptacle(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        recall = MemoryRecall(store)
        obs = _make_observation()
        obs.held_object_types = ["CreditCard"]

        query = recall.build_recall_query("把信用卡放进抽屉", obs, [])
        assert query == "Drawer"

    def test_build_recall_query_no_history(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        recall = MemoryRecall(store)
        obs = _make_observation()

        query = recall.build_recall_query("拿苹果", obs, None)
        assert query == "拿苹果"

    def test_recall_with_store(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        store.append_memory(_make_memory("CreditCard 在客厅中可见，距离 2.3m"))
        store.append_memory(_make_memory("Apple 在厨房中可见，距离 1.5m"))
        store.append_memory(_make_memory("Drawer 容器已打开", mtype="room_state"))

        recall = MemoryRecall(store)
        res = recall.recall("CreditCard")
        assert len(res) >= 1
        assert any("CreditCard" in m.content for m in res)

    def test_recall_type_filter(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        store.append_memory(_make_memory("CreditCard 在客厅中可见"))
        store.append_memory(_make_memory("Drawer 容器已打开", mtype="room_state"))

        recall = MemoryRecall(store)
        res = recall.recall("CreditCard", type_filter="room_state")
        assert len(res) == 0

        res = recall.recall("CreditCard", type_filter="object_location")
        assert len(res) >= 1

    def test_recall_limit(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        for i in range(5):
            store.append_memory(_make_memory(f"Apple {i} 在房间中"))

        recall = MemoryRecall(store)
        res = recall.recall("Apple", limit=2)
        assert len(res) <= 2

    def test_inject_into_prompt_empty(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        recall = MemoryRecall(store)
        result = recall.inject_into_prompt([])
        assert result is None

    def test_inject_into_prompt_with_memories(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        recall = MemoryRecall(store)
        memories = [
            _make_memory("CreditCard 在 FloorPlan1 中可见，距离 2.3m，Agent 位置 (1.00, 3.00)，朝向 90°"),
            _make_memory("Drawer 容器在 FloorPlan1 中可见，距离 1.5m，已打开", mtype="room_state"),
        ]

        result = recall.inject_into_prompt(memories)
        assert result is not None
        assert "## 场景记忆" in result
        assert "CreditCard" in result
        assert "Drawer" in result

    def test_inject_into_prompt_deduplication(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        recall = MemoryRecall(store)
        content = "CreditCard 在房间中"
        memories = [
            _make_memory(content),
            _make_memory(content),  # duplicate content
        ]

        result = recall.inject_into_prompt(memories)
        assert result is not None
        # Should only appear once
        assert result.count(content) == 1
