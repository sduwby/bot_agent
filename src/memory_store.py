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

        candidates: list[tuple[tuple[int, float], MemoryRecord]] = []
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
            candidates.append(((m.priority, ts), m))

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
        self._reachable_positions: list[dict[str, float]] = []

    def set_reachable_positions(
        self, positions: list[dict[str, Any]]
    ) -> None:
        """缓存由 AI2-THOR 返回的可达网格点。"""
        valid: list[dict[str, float]] = []
        for position in positions:
            if not isinstance(position, dict):
                continue
            if any(
                not isinstance(position.get(axis), (int, float))
                for axis in ("x", "y", "z")
            ):
                continue
            valid.append({
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position["z"]),
            })
        self._reachable_positions = valid

    def should_search_for(
        self,
        object_type: str,
        visible_types: set[str] | None = None,
    ) -> bool:
        """目标当前不可见时继续搜索；历史记忆不能代替当前观察。"""
        visible = {item.casefold() for item in (visible_types or set())}
        return object_type.casefold() not in visible

    def next_move(
        self,
        position: dict[str, float] | None = None,
        rotation_y: float = 0.0,
        preferred_position: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """返回下一步建议（move_to 技能调用）。

        策略：按 _SEARCH_DIRECTIONS 顺序执行探索。
        """
        if self.is_exhausted():
            return {
                "thought": "搜索已耗尽所有选项",
                "skill": "done",
                "parameters": {},
            }

        waypoint = None
        if (
            preferred_position is not None
            and self.is_reachable_position(preferred_position)
        ):
            waypoint = dict(preferred_position)
        if waypoint is None:
            waypoint = self._next_reachable_position(position)
        self._steps_taken += 1

        if waypoint is not None:
            return {
                "thought": (
                    "搜索模式：前往未探索可达视点 "
                    f"({waypoint['x']:.2f}, {waypoint['z']:.2f})"
                    f"（第 {self._steps_taken} 步）"
                ),
                "skill": "move_to",
                "parameters": {
                    "direction": "reachable",
                    "target_position": waypoint,
                },
            }

        direction = self._next_unexplored_direction(position, rotation_y)

        return {
            "thought": f"搜索模式：向 {direction} 方向探索（第 {self._steps_taken} 步）",
            "skill": "move_to",
            "parameters": {"direction": direction},
        }

    def _next_reachable_position(
        self,
        current_position: dict[str, float] | None,
    ) -> dict[str, float] | None:
        """用最远点采样选择覆盖增益最大的未探索可达视点。"""
        if not self._reachable_positions:
            return None
        explored = [dict(item.position) for item in self._explored_positions]
        if current_position is not None:
            explored.append(dict(current_position))
        candidates = [
            point
            for point in self._reachable_positions
            if not any(
                self._positions_close(point, seen, tolerance=0.1)
                for seen in explored
            )
        ]
        if not candidates:
            return None

        def coverage_score(point: dict[str, float]) -> float:
            if not explored:
                return 0.0
            return min(
                (point["x"] - seen.get("x", 0.0)) ** 2
                + (point["z"] - seen.get("z", 0.0)) ** 2
                for seen in explored
            )

        selected = max(
            candidates,
            key=lambda point: (
                coverage_score(point),
                -point["x"],
                -point["z"],
            ),
        )
        return dict(selected)

    def is_reachable_position(
        self, position: dict[str, Any], tolerance: float = 0.01
    ) -> bool:
        """验证目标点确实来自缓存的 AI2-THOR 可达点集合。"""
        if not isinstance(position, dict):
            return False
        return any(
            self._positions_close(position, candidate, tolerance)
            for candidate in self._reachable_positions
        )

    def _next_unexplored_direction(
        self,
        position: dict[str, float] | None,
        rotation_y: float,
    ) -> str:
        """优先选择预测落点尚未探索的方向。"""
        for _ in range(len(_SEARCH_DIRECTIONS)):
            direction = _SEARCH_DIRECTIONS[
                self._direction_index % len(_SEARCH_DIRECTIONS)
            ]
            self._direction_index += 1
            if position is None:
                return direction
            predicted_position, predicted_rotation = self._predict_pose(
                position, rotation_y, direction
            )
            if not self.is_pose_explored(predicted_position, predicted_rotation):
                return direction
        return "right"

    @staticmethod
    def _predict_pose(
        position: dict[str, float],
        rotation_y: float,
        direction: str,
    ) -> tuple[dict[str, float], float]:
        """按 0.25m 网格和 90 度旋转预测一个搜索动作后的位姿。"""
        predicted = dict(position)
        predicted_rotation = rotation_y % 360
        if direction == "left":
            return predicted, (predicted_rotation - 90) % 360
        if direction == "right":
            return predicted, (predicted_rotation + 90) % 360

        step = 0.25 if direction == "forward" else -0.25
        heading = int(round(predicted_rotation / 90)) % 4
        if heading == 0:
            predicted["z"] = predicted.get("z", 0.0) + step
        elif heading == 1:
            predicted["x"] = predicted.get("x", 0.0) + step
        elif heading == 2:
            predicted["z"] = predicted.get("z", 0.0) - step
        else:
            predicted["x"] = predicted.get("x", 0.0) - step
        return predicted, predicted_rotation

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

    def is_pose_explored(
        self,
        position: dict[str, float],
        rotation_y: float,
        tolerance: float = 0.1,
    ) -> bool:
        """位置接近且朝向相同时视为已探索过同一视角。"""
        for ep in self._explored_positions:
            if not self._positions_close(position, ep.position, tolerance):
                continue
            rotation_delta = abs((rotation_y - ep.rotation_y + 180) % 360 - 180)
            if rotation_delta < 1.0:
                return True
        return False

    @staticmethod
    def _positions_close(
        first: dict[str, float],
        second: dict[str, float],
        tolerance: float,
    ) -> bool:
        return sum(
            abs(first.get(axis, 0.0) - second.get(axis, 0.0))
            for axis in ("x", "y", "z")
        ) < tolerance

    def record_explored_position(
        self,
        position: dict,
        rotation_y: float,
        step: int,
    ) -> None:
        """记录当前位置已探索"""
        self._explored_positions.append(ExploredPosition(
            position=dict(position),
            rotation_y=rotation_y,
            step=step,
        ))

    def reset(self, clear_explored: bool = True) -> None:
        """重置搜索状态"""
        self._steps_taken = 0
        if clear_explored:
            self._explored_positions.clear()
        self._direction_index = 0
