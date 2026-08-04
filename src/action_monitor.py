"""重复动作检测：跟踪 action_history 滑动窗口，发现循环模式。

循环类型：
- IDENTICAL_REPEAT: 连续 N 步（默认 3）相同的 (skill, params)
- OSCILLATION: 两个动作交替出现（A→B→A→B）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class LoopDetection:
    """循环检测结果"""
    is_loop: bool
    loop_type: str | None  # "IDENTICAL_REPEAT" | "OSCILLATION"
    window_size: int
    details: str  # 人类可读的描述


class ActionMonitor:
    """检测 action_history 中的循环模式。"""

    @staticmethod
    def detect_loop(
        history: list[dict[str, Any]],
        window: int = 3,
    ) -> LoopDetection:
        """检测最近 window 步是否存在循环。

        Args:
            history: action_history 列表，每项包含 "skill" 和 "parameters"。
            window: 滑动窗口大小。

        Returns:
            LoopDetection 结果。
        """
        if len(history) < window:
            return LoopDetection(
                is_loop=False,
                loop_type=None,
                window_size=window,
                details=f"历史步数 {len(history)} 少于窗口 {window}，无法检测循环",
            )

        window_items = history[-window:]
        keys = [_action_key(item) for item in window_items]

        # 检查 IDENTICAL_REPEAT：所有 key 相同
        if len(set(keys)) == 1:
            skill = window_items[0].get("skill", "?")
            params = window_items[0].get("parameters", {})
            return LoopDetection(
                is_loop=True,
                loop_type="IDENTICAL_REPEAT",
                window_size=window,
                details=f"连续 {window} 步重复执行 {skill}{params}",
            )

        # 检查 OSCILLATION：只有 2 个唯一 key 且交替出现
        unique_keys = list(dict.fromkeys(keys))  # 保持顺序去重
        if len(unique_keys) == 2 and window >= 3:
            is_alternating = all(keys[i] == keys[i % 2] for i in range(len(keys)))
            if is_alternating:
                skill_a = window_items[0].get("skill", "?")
                skill_b = window_items[1].get("skill", "?")
                return LoopDetection(
                    is_loop=True,
                    loop_type="OSCILLATION",
                    window_size=window,
                    details=f"检测到交替循环: {skill_a} ↔ {skill_b}，持续 {window} 步",
                )

        return LoopDetection(
            is_loop=False,
            loop_type=None,
            window_size=window,
            details=f"最近 {window} 步未发现循环模式",
        )


def _action_key(item: dict[str, Any]) -> str:
    """将动作转换为可比较的字符串键。"""
    skill = item.get("skill", "")
    params = json.dumps(item.get("parameters", {}), sort_keys=True)
    return f"{skill}:{params}"