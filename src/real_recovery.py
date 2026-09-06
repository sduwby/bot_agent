"""真机 Recovery Manager（阶段 F，SO-101）——S8。

把失败码（real_skills.FAIL_*）映射为结构化恢复动作（重试/复位重来/终止/交由
LLM 决策），并接入循环检测（action_monitor 的纯逻辑版），供 real_agent 消费。

纯逻辑：只依赖失败码与历史动作，不驱动机械臂。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from real_skills import (
    FAIL_COMM,
    FAIL_GRASP_EMPTY,
    FAIL_HOME_TIMEOUT,
    FAIL_NOT_HOLDING,
    FAIL_NOT_LIFTED,
    FAIL_TIMEOUT,
)

logger = logging.getLogger(__name__)

# 恢复动作取值
REC_RETRY = "retry"              # 原技能重试
REC_RESET = "reset_and_retry"    # 先复位（release+home）再重试
REC_ABORT = "abort"              # 安全终止（保持当前姿态待人工）
REC_LLM = "llm_decision"         # 交由 LLM 决策
REC_VERIFY = "verify"            # 先 verify_holding 再决定


@dataclass
class RecoveryAction:
    """一次恢复决策。"""

    action: str
    reason: str
    skill_name: str | None = None
    params: dict[str, Any] | None = None
    is_final: bool = False


@dataclass
class LoopSignal:
    """真机动作循环信号（简化版 action_monitor 契约）。"""

    is_loop: bool
    loop_type: str | None = None  # "IDENTICAL_REPEAT" | "OSCILLATION"
    details: str = ""


def detect_loop(history: list[dict[str, Any]], window: int = 3) -> LoopSignal:
    """检测最近 window 步是否为同一/振荡技能反复（无成功的 pick/grasp 进展）。"""
    if len(history) < window:
        return LoopSignal(False)
    recent = history[-window:]
    # 最近有成功的关键技能视为有进展
    if any(item.get("success") and item.get("skill") in {"grasp_screwdriver", "pick"}
           for item in recent):
        return LoopSignal(False)
    keys = [f'{item.get("skill", "")}:{item.get("params") or {}}' for item in recent]
    if len(set(keys)) == 1:
        return LoopSignal(True, "IDENTICAL_REPEAT", f"连续 {window} 步重复 {recent[-1].get('skill')}")
    if len(set(keys)) == 2 and all(keys[i] == keys[i % 2] for i in range(len(keys))):
        return LoopSignal(True, "OSCILLATION", f"最近 {window} 步在 2 个技能间振荡")
    return LoopSignal(False)


class RealRecoveryManager:
    """真机失败恢复：失败码 → RecoveryAction。"""

    def __init__(self, max_retries: int = 1) -> None:
        self.max_retries = max_retries
        self._retry_counts: dict[str, int] = {}

    def reset(self) -> None:
        self._retry_counts = {}

    def record_success(self, skill_name: str) -> None:
        self._retry_counts.pop(skill_name, None)

    def plan_recovery(
        self,
        *,
        skill_name: str,
        failure_code: str | None,
        loop: LoopSignal | None = None,
    ) -> RecoveryAction:
        """依据失败码与循环信号生成恢复动作。"""
        if loop is not None and loop.is_loop:
            return RecoveryAction(
                REC_RESET, f"检测到动作循环({loop.loop_type})，先复位再重试",
                skill_name="reset_pose", is_final=True,
            )
        if failure_code == FAIL_COMM:
            return RecoveryAction(REC_ABORT, "通信错误，终止并待人工接管", is_final=True)
        if failure_code in (FAIL_GRASP_EMPTY, FAIL_NOT_LIFTED):
            return self._retry_or_abort(skill_name, "抓取失败")
        if failure_code == FAIL_NOT_HOLDING:
            return RecoveryAction(REC_VERIFY, "确认未夹住，重新抓取")
        if failure_code == FAIL_HOME_TIMEOUT:
            return RecoveryAction(REC_ABORT, "归位超时，终止", is_final=True)
        if failure_code == FAIL_TIMEOUT:
            return RecoveryAction(REC_RESET, "技能超时，先复位再重试", skill_name="reset_pose")
        return RecoveryAction(REC_LLM, f"未知失败码 {failure_code}，交由 LLM 决策")

    def _retry_or_abort(self, skill_name: str, reason_prefix: str) -> RecoveryAction:
        count = self._retry_counts.get(skill_name, 0)
        if count >= self.max_retries:
            return RecoveryAction(
                REC_ABORT, f"{reason_prefix}且超出重试上限({self.max_retries})",
                is_final=True,
            )
        self._retry_counts[skill_name] = count + 1
        return RecoveryAction(
            REC_RETRY, f"{reason_prefix}，第 {count + 1}/{self.max_retries} 次重试",
            skill_name=skill_name,
        )
