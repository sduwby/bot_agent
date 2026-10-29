"""真机技能结果验证与失败归类（SO-101）。

纯逻辑：依据关节观测判定技能执行是否成功、归入哪类失败码。
结构与仿真版 `verifier.py` 的"独立验证层"思想一致，输入改为真机可观测信号。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from real_observation import (
    EMPTY_GRIP_BASELINE,
    GRIP_LOAD_THRESHOLD,
    GRIP_POS_MARGIN,
    holds_object,
    is_lifted,
)
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
    """基于关节/夹爪观测的真机结果验证器。

    夹持判定用「位置 OR 负载」（`holds_object`）：位置判据对薄件会漏判，
    负载判据对位置抖动免疫。默认值取第 38 周 0.6 真机标定结果。
    """

    def __init__(self, empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
                 margin: float = GRIP_POS_MARGIN,
                 load_threshold: float = GRIP_LOAD_THRESHOLD) -> None:
        self.empty_grip_baseline = empty_grip_baseline
        self.margin = margin
        self.load_threshold = load_threshold

    def _holding(self, gripper_pos: float, load: float | None) -> bool:
        return holds_object(gripper_pos, load,
                            empty_grip_baseline=self.empty_grip_baseline,
                            pos_margin=self.margin,
                            load_threshold=self.load_threshold)

    def verify_grasp_lift(self, final_gripper: float, final_wrist: float,
                          final_load: float | None = None) -> RealVerificationResult:
        """验证 grasp_screwdriver：夹住 + 抬起。

        ``final_load`` 为夹爪负载读数（‰）；缺省或读取失败时退化为纯位置判据。
        """
        holding = self._holding(final_gripper, final_load)
        lifted = is_lifted(final_wrist)
        if holding and lifted:
            return RealVerificationResult(
                True,
                details=f"gripper={final_gripper:.1f} wrist={final_wrist:.1f} load={final_load}",
            )
        if not holding and lifted:
            return RealVerificationResult(
                False,
                FAIL_GRASP_EMPTY,
                f"夹爪闭合但未被物体挡住(空闭基准 {self.empty_grip_baseline:.1f}+{self.margin}"
                f"，负载 {final_load})，疑似咬空",
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
            f"夹爪未夹住物体(gripper={final_gripper:.1f} load={final_load})",
        )

    def verify_holding(self, gripper_pos: float,
                       load: float | None = None) -> RealVerificationResult:
        """验证 verify_holding；``load`` 为夹爪负载读数（可选）。"""
        if self._holding(gripper_pos, load):
            return RealVerificationResult(
                True, details=f"夹爪受阻位置 {gripper_pos:.1f}（负载 {load}），判定为夹住物体"
            )
        return RealVerificationResult(
            False,
            FAIL_NOT_HOLDING,
            f"夹爪位置 {gripper_pos:.1f} 未超过空闭基准且负载未超阈值，未夹住物体",
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
