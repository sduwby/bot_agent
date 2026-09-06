"""真机技能结果验证与失败归类（SO-101）。

纯逻辑：依据关节观测判定技能执行是否成功、归入哪类失败码。
结构与仿真版 `verifier.py` 的"独立验证层"思想一致，输入改为真机可观测信号。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from real_observation import is_holding, is_lifted
from real_skills import (
    FAIL_COMM,
    FAIL_GRASP_EMPTY,
    FAIL_NOT_HOLDING,
    FAIL_NOT_LIFTED,
)


@dataclass
class RealVerificationResult:
    """技能执行验证结果。"""

    success: bool
    failure_code: str | None = None
    details: str = ""


class RealVerifier:
    """基于关节/夹爪观测的真机结果验证器。"""

    def __init__(self, empty_grip_baseline: float, margin: float = 3.0) -> None:
        self.empty_grip_baseline = empty_grip_baseline
        self.margin = margin

    def verify_grasp_lift(self, final_gripper: float, final_wrist: float) -> RealVerificationResult:
        """验证 grasp_screwdriver：夹住 + 抬起。"""
        holding = is_holding(final_gripper, self.empty_grip_baseline, self.margin)
        lifted = is_lifted(final_wrist)
        if holding and lifted:
            return RealVerificationResult(
                True,
                details=f"gripper={final_gripper:.1f} wrist={final_wrist:.1f}",
            )
        if not holding and lifted:
            return RealVerificationResult(
                False,
                FAIL_GRASP_EMPTY,
                f"夹爪闭合但未被物体挡住(空闭基准 {self.empty_grip_baseline:.1f}+{self.margin})，疑似咬空",
            )
        if holding and not lifted:
            return RealVerificationResult(
                False,
                FAIL_NOT_LIFTED,
                f"已夹住但未抬离桌面(wrist={final_wrist:.1f})",
            )
        return RealVerificationResult(
            False,
            FAIL_GRASP_EMPTY,
            f"夹爪未夹住物体(gripper={final_gripper:.1f})",
        )

    def verify_holding(self, gripper_pos: float) -> RealVerificationResult:
        """验证 verify_holding。"""
        if is_holding(gripper_pos, self.empty_grip_baseline, self.margin):
            return RealVerificationResult(
                True, details=f"夹爪受阻位置 {gripper_pos:.1f}，判定为夹住物体"
            )
        return RealVerificationResult(
            False,
            FAIL_NOT_HOLDING,
            f"夹爪位置 {gripper_pos:.1f} 未超过空闭基准，未夹住物体",
        )


def classify_failure_code(failure_code: str | None, skill_name: str = "") -> str:
    """把失败码归一为可给 replanner 的恢复意图类别。

    Returns:
        恢复意图: retry / scan 视角 / abort / comm。
    """
    if failure_code in (FAIL_GRASP_EMPTY, FAIL_NOT_LIFTED):
        return "retry"
    if failure_code == FAIL_NOT_HOLDING:
        return "retry"
    if failure_code == FAIL_COMM:
        return "comm"
    return "abort"
