"""MemoryStore 和 SearchStrategy 的单元测试（不依赖 AI2-THOR）。"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from memory_store import (
    ConversationRecord,
    ExploredPosition,
    MemoryRecord,
    MemoryStore,
    SearchStrategy,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_memory(
    obj_id: str = "obj1",
    content: str = "CreditCard 在 FloorPlan1 中可见，距离 2.3m",
    mtype: str = "object_location",
    priority: int = 80,
    step: int = 0,
    updated_at: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=f"m_{obj_id}",
        content=content,
        type=mtype,
        priority=priority,
        scene_name="FloorPlan1",
        source_step=step,
        created_at=_now(),
        updated_at=updated_at or _now(),
    )


class TestMemoryStore:
    """测试 MemoryStore 核心功能。"""

    def test_append_and_query_memory(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s1")
        assert store.query_memories() == []

        m = _make_memory()
        store.append_memory(m)
        all_m = store.query_memories()
        assert len(all_m) == 1
        assert all_m[0].id == "m_obj1"

    def test_append_conversation(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s2")
        ts = 1000
        rec = ConversationRecord(
            session_key="s2", role="user", content="放信用卡",
            timestamp=ts, step=0,
        )
        store.append_conversation(rec)

        # 验证文件写入
        conv_file = store.conversations_file
        assert conv_file.exists()
        with open(conv_file) as f:
            data = json.loads(f.readline())
        assert data["role"] == "user"
        assert data["content"] == "放信用卡"

    def test_append_memory_writes_file(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s3")
        m = _make_memory(obj_id="k1")
        store.append_memory(m)

        mem_file = store.memories_file
        assert mem_file.exists()
        with open(mem_file) as f:
            data = json.loads(f.readline())
        assert data["id"] == "m_k1"

    def test_search_keyword_match(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s4")
        store.append_memory(_make_memory(obj_id="a", content="CreditCard 在客厅中可见"))
        store.append_memory(_make_memory(obj_id="b", content="Apple 在厨房中可见"))

        res = store.search_memories("CreditCard")
        assert len(res) == 1
        assert "CreditCard" in res[0].content

        res = store.search_memories("Apple")
        assert len(res) == 1
        assert "Apple" in res[0].content

    def test_search_type_filter(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s5")
        store.append_memory(_make_memory(obj_id="a", content="CreditCard 在客厅中可见", mtype="object_location"))
        store.append_memory(_make_memory(obj_id="b", content="Drawer 容器已打开", mtype="room_state"))

        res = store.search_memories("CreditCard", type_filter="object_location")
        assert len(res) == 1

        res = store.search_memories("CreditCard", type_filter="room_state")
        assert len(res) == 0

    def test_search_priority_ordering(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s6")
        store.append_memory(_make_memory(obj_id="low", content="Card 在房间中", priority=20))
        store.append_memory(_make_memory(obj_id="high", content="Card 在客厅", priority=90))

        res = store.search_memories("Card", limit=1)
        assert res[0].id == "m_high"

    def test_search_limit(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s7")
        for i in range(5):
            store.append_memory(_make_memory(obj_id=str(i), content=f"Apple {i} 在房间中"))

        res = store.search_memories("Apple", limit=2)
        assert len(res) == 2

    def test_search_empty_query(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s8")
        store.append_memory(_make_memory())
        res = store.search_memories("")
        assert res == []

    def test_search_no_match(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s9")
        store.append_memory(_make_memory(content="CreditCard 在房间中"))
        res = store.search_memories("Watermelon")
        assert res == []

    def test_query_by_type(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s10")
        store.append_memory(_make_memory(mtype="object_location", content="CreditCard 可见"))
        store.append_memory(_make_memory(mtype="room_state", content="Drawer 已打开"))
        store.append_memory(_make_memory(mtype="failure_lesson", content="pick 失败"))

        loc = store.query_memories("object_location")
        assert len(loc) == 1
        assert loc[0].type == "object_location"

        all_m = store.query_memories()
        assert len(all_m) == 3

    def test_get_known_object_types(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s11")
        store.append_memory(_make_memory(content="CreditCard 在房间中可见"))
        store.append_memory(_make_memory(content="Apple 在厨房中可见"))
        store.append_memory(_make_memory(mtype="room_state", content="Drawer 已打开"))

        types = store.get_known_object_types()
        assert "CreditCard" in types
        assert "Apple" in types
        assert "Drawer" not in types

    def test_get_summary_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s12")
        assert store.get_summary() == ""

    def test_get_summary_with_memories(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s13")
        store.append_memory(_make_memory(mtype="object_location", content="CreditCard 可见"))
        store.append_memory(_make_memory(mtype="room_state", content="Drawer 已打开"))

        s = store.get_summary()
        assert "2 条记忆" in s
        assert "1 个物体位置" in s
        assert "1 个容器状态" in s

    def test_clear(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s14")
        store.append_memory(_make_memory())
        store.clear()
        assert store.query_memories() == []

    def test_load_existing_memories(self, tmp_path: Path) -> None:
        # 先写入
        store1 = MemoryStore(str(tmp_path), "s15")
        store1.append_memory(_make_memory(obj_id="persist"))
        del store1

        # 重新加载
        store2 = MemoryStore(str(tmp_path), "s15")
        all_m = store2.query_memories()
        assert len(all_m) == 1
        assert all_m[0].id == "m_persist"

    def test_append_multiple_memories(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path), "s16")
        for i in range(3):
            store.append_memory(_make_memory(obj_id=str(i)))
        assert len(store.query_memories()) == 3


class TestSearchStrategy:
    """测试 SearchStrategy 搜索策略。"""

    def _make_store(self, tmp_path: Path) -> MemoryStore:
        return MemoryStore(str(tmp_path), "test")

    def test_should_search_for_not_in_memory(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        search = SearchStrategy(store)
        assert search.should_search_for("CreditCard") is True

    def test_memory_does_not_prevent_search_when_target_not_visible(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        store.append_memory(_make_memory(content="CreditCard 在房间中可见"))
        search = SearchStrategy(store)
        assert search.should_search_for("CreditCard") is True

    def test_visible_target_does_not_need_search(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        search = SearchStrategy(store)
        assert search.should_search_for(
            "CreditCard", {"CreditCard", "Apple"}
        ) is False

    def test_next_move_directions(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        search = SearchStrategy(store, max_steps=4)

        d1 = search.next_move()
        assert d1["skill"] == "move_to"
        assert d1["parameters"]["direction"] == "forward"

        d2 = search.next_move()
        assert d2["parameters"]["direction"] == "right"

        d3 = search.next_move()
        assert d3["parameters"]["direction"] == "forward"

    def test_next_move_when_exhausted(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        search = SearchStrategy(store, max_steps=2)
        search.next_move()
        search.next_move()

        d = search.next_move()
        assert d["skill"] == "done"

    def test_is_exhausted(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        search = SearchStrategy(store, max_steps=3)
        assert search.is_exhausted() is False
        search.next_move()
        search.next_move()
        assert search.is_exhausted() is False
        search.next_move()
        assert search.is_exhausted() is True

    def test_is_position_explored(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        search = SearchStrategy(store)
        search.record_explored_position({"x": 1.0, "y": 0, "z": 1.0}, 0, 0)

        # Same position should be explored
        assert search.is_position_explored({"x": 1.0, "y": 0, "z": 1.0}) is True

        # Close position (within tolerance) should be explored
        assert search.is_position_explored({"x": 1.2, "y": 0, "z": 1.2}) is True

        # Far position should not be explored
        assert search.is_position_explored({"x": 5.0, "y": 0, "z": 5.0}) is False

    def test_reset(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        search = SearchStrategy(store, max_steps=3)
        search.next_move()
        search.next_move()
        assert search.is_exhausted() is False

        search.reset()
        assert search.is_exhausted() is False
        assert search._steps_taken == 0

    def test_next_move_has_thought(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        search = SearchStrategy(store)
        d = search.next_move()
        assert "thought" in d
        assert "搜索模式" in d["thought"]

    def test_next_move_avoids_explored_pose(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        search = SearchStrategy(store)
        current = {"x": 0.0, "y": 0.0, "z": 0.0}
        search.record_explored_position(
            {"x": 0.0, "y": 0.0, "z": 0.25}, 0.0, 1
        )

        decision = search.next_move(current, 0.0)

        assert decision["parameters"]["direction"] == "right"
