"""本地轻量记忆存储，兼容 TencentDB-Agent-Memory L0/L1 接口。

参考 TencentDB-Agent-Memory 的分层记忆架构：
- L0（原始对话）：JSONL append-only，记录每步交互的完整文本
- L1（原子记忆）：从 L0 提取的结构化事实（物体位置、房间状态等）
- 检索：初期使用关键词匹配，后续可扩展 Embedding
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ConversationRecord:
    """L0 原始对话记录（对应 TencentDB-Agent-Memory 的 L0）"""
    session_key: str
    role: str          # "user" | "assistant"
    content: str       # 本轮交互的文本描述
    timestamp: int     # 毫秒级时间戳
    step: int          # 在 Agent loop 中的步骤编号


@dataclass
class MemoryRecord:
    """L1 原子记忆（对应 TencentDB-Agent-Memory 的 L1）"""
    id: str                        # 唯一 ID
    content: str                   # 记忆内容（自然语言描述）
    type: str                      # object_location | room_state | action_experience | failure_lesson
    priority: int                  # 1-100，越高越重要
    scene_name: str                # 场景名称
    source_step: int               # 来源步骤编号
    created_at: str                # ISO 8601 时间
    updated_at: str                # 最后更新时间


class MemoryStore:
    """本地轻量记忆存储，兼容 TencentDB-Agent-Memory 接口。

    L0 和 L1 分别写入独立的 JSONL 文件（append-only）。
    检索使用关键词匹配 + 类型过滤，按优先级和新鲜度排序。
    """

    def __init__(self, base_dir: str, session_key: str) -> None:
        self.base_dir = Path(base_dir)
        self.session_key = session_key

        conversations_dir = self.base_dir / "conversations"
        memories_dir = self.base_dir / "memories"
        conversations_dir.mkdir(parents=True, exist_ok=True)
        memories_dir.mkdir(parents=True, exist_ok=True)

        self.conversations_file = conversations_dir / f"{session_key}.jsonl"
        self.memories_file = memories_dir / f"{session_key}.jsonl"

        # 内存缓存：避免每次检索都读磁盘
        self._memories: list[MemoryRecord] = []
        self._load_existing_memories()

    def _load_existing_memories(self) -> None:
        """加载已存在的记忆记录到内存缓存。"""
        if not self.memories_file.exists():
            return
        try:
            with open(self.memories_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            self._memories.append(MemoryRecord(**data))
                        except (json.JSONDecodeError, TypeError):
                            continue
        except OSError:
            pass

    # ── L0 写入 ──

    def append_conversation(self, record: ConversationRecord) -> None:
        """写入 L0 对话记录（append-only）"""
        data = asdict(record)
        with open(self.conversations_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # ── L1 写入 ──

    def append_memory(self, record: MemoryRecord) -> None:
        """写入 L1 记忆记录（append-only，去重在检索时处理）"""
        self._memories.append(record)
        data = asdict(record)
        with open(self.memories_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # ── 检索 ──

    def search_memories(
        self,
        query: str,
        type_filter: str | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """搜索记忆。

        使用关键词匹配 + 类型过滤，按优先级降序、时间降序排序。
        """
        if not query:
            return []

        query_lower = query.lower()
        # 将查询拆分为关键词（中英文兼容）
        keywords = [w.strip() for w in query_lower.split() if w.strip()]
        if not keywords:
            keywords = [query_lower]

        candidates: list[tuple[float, MemoryRecord]] = []
        for m in self._memories:
            # 类型过滤
            if type_filter and m.type != type_filter:
                continue
            # 关键词匹配：在 content 中搜索
            content_lower = m.content.lower()
            matched = any(kw in content_lower for kw in keywords)
            if not matched:
                continue
            # 评分：优先级为主，时间新鲜度为辅
            try:
                ts = datetime.fromisoformat(m.updated_at).timestamp()
            except (ValueError, TypeError):
                ts = 0
            score = m.priority * 1000 + ts
            candidates.append((score, m))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in candidates[:limit]]

    def query_memories(
        self,
        type_filter: str | None = None,
    ) -> list[MemoryRecord]:
        """按类型查询所有记忆"""
        if not type_filter:
            return list(self._memories)
        return [m for m in self._memories if m.type == type_filter]

    def get_known_object_types(self) -> set[str]:
        """返回所有已知的物体类型（从 object_location 记忆中提取）"""
        types: set[str] = set()
        for m in self._memories:
            if m.type == "object_location":
                # 从 content 中尝试提取物体类型
                # 格式: "{ObjectType} 在 {scene} 中可见..."
                parts = m.content.split()
                if parts:
                    # 第一个词通常是物体类型（如 CreditCard, Apple）
                    obj_type = parts[0]
                    types.add(obj_type)
        return types

    def get_summary(self) -> str:
        """生成记忆摘要供 LLM prompt 使用"""
        if not self._memories:
            return ""

        loc_count = sum(1 for m in self._memories if m.type == "object_location")
        room_count = sum(1 for m in self._memories if m.type == "room_state")
        fail_count = sum(1 for m in self._memories if m.type == "failure_lesson")
        parts = [f"已记录 {len(self._memories)} 条记忆"]
        detail_parts = []
        if loc_count:
            detail_parts.append(f"{loc_count} 个物体位置")
        if room_count:
            detail_parts.append(f"{room_count} 个容器状态")
        if fail_count:
            detail_parts.append(f"{fail_count} 次失败教训")
        return f"{parts[0]}（{', '.join(detail_parts)}）"

    def clear(self) -> None:
        """清空内存缓存（用于测试）"""
        self._memories.clear()


# ── SearchStrategy ──

@dataclass
class ExploredPosition:
    """某次探索的位姿快照"""
    position: dict        # {"x": float, "y": float, "z": float}
    rotation_y: float     # 朝向
    step: int             # 在哪个步骤探索的


# 搜索探索方向策略：forward → right → forward → left（螺旋式探索）
_SEARCH_DIRECTIONS = ["forward", "right", "forward", "left"]


class SearchStrategy:
    """基于记忆的搜索策略。

    当目标物体不在当前视野时，决定是否继续搜索及向哪个方向移动。
    """

    def __init__(self, store: MemoryStore, max_steps: int = 8) -> None:
        self.store = store
        self.max_steps = max_steps
        self._steps_taken = 0
        self._explored_positions: list[ExploredPosition] = []
        self._direction_index = 0

    def should_search_for(self, object_type: str) -> bool:
        """目标类型是否在记忆中？不在则需搜索。"""
        known = self.store.get_known_object_types()
        return object_type not in known

    def next_move(self) -> dict[str, Any]:
        """返回下一步建议（move_to 技能调用）。

        策略：按 _SEARCH_DIRECTIONS 顺序执行探索。
        """
        if self.is_exhausted():
            return {
                "thought": "搜索已耗尽所有选项",
                "skill": "done",
                "parameters": {},
            }

        direction = _SEARCH_DIRECTIONS[self._direction_index % len(_SEARCH_DIRECTIONS)]
        self._direction_index += 1
        self._steps_taken += 1

        return {
            "thought": f"搜索模式：向 {direction} 方向探索（第 {self._steps_taken} 步）",
            "skill": "move_to",
            "parameters": {"direction": direction},
        }

    def is_exhausted(self) -> bool:
        """搜索是否已耗尽所有选项"""
        return self._steps_taken >= self.max_steps

    def is_position_explored(
        self,
        position: dict,
        tolerance: float = 0.5,
    ) -> bool:
        """判断给定位置是否已经探索过（曼哈顿距离容差判断）"""
        px = position.get("x", 0)
        py = position.get("y", 0)
        pz = position.get("z", 0)
        for ep in self._explored_positions:
            dx = abs(px - ep.position.get("x", 0))
            dy = abs(py - ep.position.get("y", 0))
            dz = abs(pz - ep.position.get("z", 0))
            if dx + dy + dz < tolerance:
                return True
        return False

    def record_explored_position(
        self,
        position: dict,
        rotation_y: float,
        step: int,
    ) -> None:
        """记录当前位置已探索"""
        self._explored_positions.append(ExploredPosition(
            position=position,
            rotation_y=rotation_y,
            step=step,
        ))

    def reset(self) -> None:
        """重置搜索状态"""
        self._steps_taken = 0
        self._explored_positions.clear()
        self._direction_index = 0