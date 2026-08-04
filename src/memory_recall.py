"""记忆召回：根据当前任务和观察，召回相关记忆。

参考 TencentDB-Agent-Memory 的 auto-recall 钩子：
原文本 Agent 在每轮对话前自动召回。
具身智能场景下，召回查询 = 任务指令 + 当前缺失的信息。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from observation import Observation

if TYPE_CHECKING:
    from memory_store import MemoryRecord, MemoryStore


class MemoryRecall:
    """根据任务+观察召回相关记忆。"""

    def __init__(self, store: "MemoryStore") -> None:
        self.store = store

    def build_recall_query(
        self,
        instruction: str,
        observation: Observation,
        action_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """构建召回查询。

        策略：
        - 如果最近一次 pick 失败 → 搜索目标物体的位置
        - 如果最近一次 place 失败 → 搜索目标容器
        - 默认 → 搜索任务指令中的关键词
        """
        if not action_history:
            return instruction

        last = action_history[-1]
        if not last.get("success"):
            params = last.get("parameters", {})
            if last["skill"] == "pick":
                return params.get("object_type", instruction)
            if last["skill"] == "place":
                return params.get("receptacle_type", "容器")
        return instruction

    def recall(
        self,
        query: str,
        type_filter: str | None = None,
        limit: int = 5,
    ) -> list["MemoryRecord"]:
        """执行记忆召回。"""
        return self.store.search_memories(query, type_filter, limit)

    def inject_into_prompt(
        self,
        memories: list["MemoryRecord"],
    ) -> str | None:
        """将召回的记忆格式化为 prompt 片段。

        返回格式：
        ```
        ## 场景记忆

        以下是在当前场景中已探索过的信息，供参考：

        - CreditCard 在 FloorPlan1 中可见，距离 2.3m，Agent 位置 (1.00, 3.00)，朝向 90°
        - Drawer 容器在 FloorPlan1 中可见，距离 1.5m，已打开
        ```
        """
        if not memories:
            return None
        lines = ["## 场景记忆", "", "以下是在当前场景中已探索过的信息，供参考：", ""]
        # 去重：按 content 去重
        seen: set[str] = set()
        for m in memories:
            if m.content not in seen:
                seen.add(m.content)
                lines.append(f"- {m.content}")
        return "\n".join(lines)