"""重新规划：根据失败信息生成恢复策略。

恢复策略：
- OBJECT_NOT_FOUND  → 触发搜索模式
- GRASP_FAILED     → 重试 1 次
- PLACEMENT_FAILED  → 重试 1 次
- NAVIGATION_BLOCKED → 换方向
- LOOP_DETECTED     → 强制 scan 重新观察
- 超过最大重试     → abort，交由 LLM 决策
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from action_monitor import LoopDetection
from failure_classifier import FailureCategory, FailureInfo

if TYPE_CHECKING:
    pass


@dataclass
class RecoveryAction:
    """恢复动作"""
    action: str  # "retry" | "scan" | "search" | "abort" | "llm_decision"
    reason: str
    skill_name: str | None   # action=retry 时指定要重试的技能
    params: dict[str, Any] | None  # action=retry 时的参数
    is_final: bool  # 是否已达到最大重试，之后不再自动恢复


class Replanner:
    """根据失败类型制定恢复策略。"""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self._retry_counts: dict[str, int] = {}

    def plan_recovery(
        self,
        failure_info: FailureInfo | None = None,
        loop_detection: LoopDetection | None = None,
        verification_failed: bool = False,
    ) -> RecoveryAction:
        """根据失败信息生成恢复策略。

        Args:
            failure_info: 失败分类结果。
            loop_detection: 循环检测结果。
            verification_failed: 验证失败标志。

        Returns:
            RecoveryAction 恢复动作。
        """
        # 优先处理循环检测
        if loop_detection and loop_detection.is_loop:
            if "scan" in loop_detection.details:
                return RecoveryAction(
                    action="retry",
                    reason="连续扫描没有产生进展，转向新视角",
                    skill_name="move_to",
                    params={"direction": "right"},
                    is_final=False,
                )
            return RecoveryAction(
                action="scan",
                reason=f"检测到循环（{loop_detection.loop_type}），重新观察环境",
                skill_name="scan",
                params={},
                is_final=False,
            )

        if not failure_info:
            return RecoveryAction(
                action="llm_decision",
                reason="无失败信息，交由 LLM 决策",
                skill_name=None,
                params=None,
                is_final=False,
            )

        category = failure_info.category
        skill = failure_info.skill_name
        params = failure_info.params

        # 分类处理
        if category == FailureCategory.OBJECT_NOT_FOUND:
            return RecoveryAction(
                action="search",
                reason="目标物体未找到，进入搜索模式",
                skill_name=None,
                params=None,
                is_final=False,
            )

        if category == FailureCategory.GRASP_FAILED:
            return self._handle_retry(skill, params, "抓取失败，重试")

        if category == FailureCategory.PLACEMENT_FAILED:
            return self._handle_retry(skill, params, "放置失败，重试")

        if category == FailureCategory.NAVIGATION_BLOCKED:
            return self._handle_navigation_blocked(params)

        # 未知类型 → 让 LLM 决策
        return RecoveryAction(
            action="llm_decision",
            reason=f"未知失败类型，交由 LLM 决策: {failure_info.original_reason}",
            skill_name=None,
            params=None,
            is_final=False,
        )

    def _handle_retry(
        self, skill: str, params: dict[str, Any], reason: str
    ) -> RecoveryAction:
        """处理重试逻辑。"""
        count = self._retry_counts.get(skill, 0)
        if count >= self.max_retries:
            return RecoveryAction(
                action="abort",
                reason=f"{skill} 已超过最大重试次数 ({self.max_retries})，放弃自动恢复",
                skill_name=skill,
                params=params,
                is_final=True,
            )
        self._retry_counts[skill] = count + 1
        return RecoveryAction(
            action="retry",
            reason=f"{reason}（第 {count + 1}/{self.max_retries} 次）",
            skill_name=skill,
            params=params,
            is_final=False,
        )

    def _handle_navigation_blocked(
        self, params: dict[str, Any]
    ) -> RecoveryAction:
        """处理导航受阻：换方向重试。"""
        direction = params.get("direction", "forward")
        opposite = {"forward": "backward", "backward": "forward", "left": "right", "right": "left"}
        new_direction = opposite.get(direction, "right")

        count = self._retry_counts.get("move_to", 0)
        if count >= self.max_retries:
            return RecoveryAction(
                action="abort",
                reason="move_to 已超过最大重试次数，放弃自动恢复",
                skill_name="move_to",
                params=params,
                is_final=True,
            )
        self._retry_counts["move_to"] = count + 1
        return RecoveryAction(
            action="retry",
            reason=f"移动受阻，换方向 {new_direction}（第 {count + 1}/{self.max_retries} 次）",
            skill_name="move_to",
            params={"direction": new_direction},
            is_final=False,
        )

    def record_retry(self, skill_name: str) -> None:
        """手动记录某技能的重试次数。"""
        self._retry_counts[skill_name] = self._retry_counts.get(skill_name, 0) + 1

    def should_retry(self, skill_name: str) -> bool:
        """检查某技能是否还有重试次数。"""
        return self._retry_counts.get(skill_name, 0) < self.max_retries

    def reset(self) -> None:
        """重置重试计数。"""
        self._retry_counts.clear()

    def record_success(self, skill_name: str) -> None:
        """技能取得进展后清除该技能累计的失败次数。"""
        self._retry_counts.pop(skill_name, None)
