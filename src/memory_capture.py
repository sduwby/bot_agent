"""具身捕获适配器：将环境交互转换为 L0/L1 记忆格式。

参考 TencentDB-Agent-Memory 的 auto-capture 钩子：
原文本 Agent 的 auto-capture 接收用户/助手对话。
具身智能场景下，将任务指令转换为 user message，
将观察 + 动作 + 结果转换为 assistant message。
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memory_store import ConversationRecord, MemoryRecord
from observation import Observation, summarize

if TYPE_CHECKING:
    pass


class MemoryCapture:
    """将具身交互转换为记忆条目。"""

    @staticmethod
    def observation_to_conversation(
        instruction: str,
        session_key: str,
        observations: list[Observation],
        action_history: list[dict[str, Any]],
    ) -> list[ConversationRecord]:
        """将观察和动作历史转换为 L0 对话格式。

        转换规则：
        - 任务指令 → user message
        - 观察 + 动作 + 结果 → assistant message（描述本轮做了什么、看到了什么）
        """
        records: list[ConversationRecord] = []

        ts = int(time.time() * 1000)
        records.append(ConversationRecord(
            session_key=session_key,
            role="user",
            content=instruction,
            timestamp=ts,
            step=0,
        ))

        for i, action in enumerate(action_history):
            obs_text = ""
            if i < len(observations):
                obs_text = summarize(observations[i])
            result_text = action.get("result_summary", "")
            params = action.get("parameters", {})
            content = (
                f"[Step {i}] 观察到: {obs_text}\n"
                f"执行: {action['skill']}({params})\n"
                f"结果: {result_text}"
            )
            records.append(ConversationRecord(
                session_key=session_key,
                role="assistant",
                content=content,
                timestamp=int(time.time() * 1000),
                step=i + 1,
            ))
        return records

    @staticmethod
    def extract_memories_from_observation(
        observation: Observation,
        step: int,
    ) -> list[MemoryRecord]:
        """从当前观察中提取 L1 记忆。

        提取规则：
        - 每个可见可抓取物体 → object_location 类型记忆
        - 每个可见容器 → room_state 类型记忆
        - 当前持有的物体 → action_experience 类型记忆
        """
        memories: list[MemoryRecord] = []
        now = datetime.now(timezone.utc).isoformat()
        scene_name = observation.scene_name

        # 提取可见物体位置记忆
        for obj in observation.visible_pickupable:
            content = (
                f"{obj['object_type']} 在 {scene_name} 中可见，"
                f"距离 {obj['distance']}m，"
                f"Agent 位置 ({observation.position.get('x', 0):.2f}, {observation.position.get('z', 0):.2f})，"
                f"朝向 {observation.rotation_y:.0f}°"
            )
            memories.append(MemoryRecord(
                id=f"m_{step}_{obj['object_id']}",
                content=content,
                type="object_location",
                priority=80,
                scene_name=scene_name,
                source_step=step,
                created_at=now,
                updated_at=now,
            ))

        # 提取容器状态记忆
        for rc in observation.visible_receptacles:
            status = "已打开" if rc["is_open"] else "已关闭"
            content = (
                f"{rc['object_type']} 容器在 {scene_name} 中可见，"
                f"距离 {rc['distance']}m，{status}"
            )
            memories.append(MemoryRecord(
                id=f"m_{step}_{rc['object_id']}",
                content=content,
                type="room_state",
                priority=60,
                scene_name=scene_name,
                source_step=step,
                created_at=now,
                updated_at=now,
            ))

        # 提取持有状态记忆
        for i, obj_id in enumerate(observation.held_object_ids):
            obj_type = observation.held_object_types[i] if i < len(observation.held_object_types) else "Unknown"
            content = f"当前持有 {obj_type}(id={obj_id}) 在 {scene_name} 中"
            memories.append(MemoryRecord(
                id=f"m_{step}_held_{obj_id}",
                content=content,
                type="action_experience",
                priority=90,
                scene_name=scene_name,
                source_step=step,
                created_at=now,
                updated_at=now,
            ))

        return memories

    @staticmethod
    def extract_failure_memory(
        skill_name: str,
        params: dict[str, Any],
        failure_reason: str,
        step: int,
        scene_name: str,
    ) -> MemoryRecord:
        """从失败的执行中提取教训记忆。"""
        now = datetime.now(timezone.utc).isoformat()
        content = (
            f"执行 {skill_name}({params}) 在 {scene_name} 中失败: {failure_reason}"
        )
        return MemoryRecord(
            id=f"m_fail_{step}",
            content=content,
            type="failure_lesson",
            priority=70,
            scene_name=scene_name,
            source_step=step,
            created_at=now,
            updated_at=now,
        )