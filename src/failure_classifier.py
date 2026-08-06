"""动作失败分类：将技能执行失败归类到预定义类型。

失败类型：
- OBJECT_NOT_FOUND: 目标物体未找到
- GRASP_FAILED:    抓取失败（物体可见但无法抓取）
- PLACEMENT_FAILED: 放置失败
- NAVIGATION_BLOCKED: 移动受阻
- UNKNOWN:         无法分类的失败
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FailureCategory(str, Enum):
    """失败类型枚举"""
    OBJECT_NOT_FOUND = "object_not_found"
    GRASP_FAILED = "grasp_failed"
    PLACEMENT_FAILED = "placement_failed"
    NAVIGATION_BLOCKED = "navigation_blocked"
    UNKNOWN = "unknown"


@dataclass
class FailureInfo:
    """失败分类结果"""
    category: FailureCategory
    skill_name: str
    params: dict[str, Any]
    original_reason: str


# 中文关键词映射
_NOT_FOUND_KEYWORDS = {"未发现", "not found", "candidates"}
_BLOCKED_KEYWORDS = {"blocked", "受阻", "无法移动", "collided", "碰撞"}


class FailureClassifier:
    """将技能执行失败归类到预定义类型。"""

    @staticmethod
    def classify_failure(
        skill_name: str,
        params: dict[str, Any],
        failure_reason: str,
    ) -> FailureInfo:
        """根据技能名称和失败原因分类。

        Args:
            skill_name: 失败的技能名称。
            params: 技能参数。
            failure_reason: 原始失败原因描述。

        Returns:
            FailureInfo 分类结果。
        """
        reason_lower = failure_reason.lower()

        if skill_name == "pick":
            if any(kw in reason_lower for kw in _NOT_FOUND_KEYWORDS):
                return FailureInfo(
                    category=FailureCategory.OBJECT_NOT_FOUND,
                    skill_name=skill_name,
                    params=params,
                    original_reason=failure_reason,
                )
            return FailureInfo(
                category=FailureCategory.GRASP_FAILED,
                skill_name=skill_name,
                params=params,
                original_reason=failure_reason,
            )

        if skill_name == "place":
            if any(kw in reason_lower for kw in _NOT_FOUND_KEYWORDS):
                return FailureInfo(
                    category=FailureCategory.OBJECT_NOT_FOUND,
                    skill_name=skill_name,
                    params=params,
                    original_reason=failure_reason,
                )
            return FailureInfo(
                category=FailureCategory.PLACEMENT_FAILED,
                skill_name=skill_name,
                params=params,
                original_reason=failure_reason,
            )

        if skill_name == "move_to":
            return FailureInfo(
                category=FailureCategory.NAVIGATION_BLOCKED,
                skill_name=skill_name,
                params=params,
                original_reason=failure_reason,
            )

        # scan, done 或其他未知技能
        return FailureInfo(
            category=FailureCategory.UNKNOWN,
            skill_name=skill_name,
            params=params,
            original_reason=failure_reason,
        )
