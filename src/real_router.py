"""真机多策略路由与 scripted fallback（阶段 F，SO-101）。

S3/S4：
- `ScriptedPolicy`：确定性原语序列（reset→重试 ACT 的安全复位回退）。对当前
  单技能真机，scripted 提供两个确定性技能：reset_pose（home）与 release+reset
  （把机械臂带回初始并张开），不依赖目标物体定位，作为 ACT 失败后的安全回退。
- `RealSkillRouter`：在技能执行器之上做"主策略(ACT) → 回退策略(scripted)"路由，
  记录切换事件与失败原因；借鉴 `skill_router.py` 的编排思想，但观测契约改为
  真机 SkillResult/失败码，不依赖 ManiSkill env。

纯逻辑：策略均作用于注入的 RealSkillExecutor；测试用 MockEngine。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from real_executor import RealSkillExecutor, SkillResult

logger = logging.getLogger(__name__)

# 可路由策略位
ROUTE_ACT = "act"
ROUTE_SCRIPTED = "scripted"
SUPPORTED_ROUTE_KINDS = (ROUTE_ACT, ROUTE_SCRIPTED)


@dataclass(frozen=True)
class PolicySource:
    """路由中的策略位描述。"""

    kind: str
    name: str
    optional: bool = False

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_ROUTE_KINDS:
            raise ValueError(f"未知策略位 kind={self.kind}")
        if not self.name:
            raise ValueError("策略位 name 不能为空")


@dataclass(frozen=True)
class SwitchEvent:
    """策略切换事件。"""

    from_policy: str
    to_policy: str
    reason: str
    at_step: int


@dataclass(frozen=True)
class RouteResult:
    """一次任务路由结果。"""

    success: bool
    task: str
    steps: int
    policy_steps: dict[str, int] = field(default_factory=dict)
    switch_events: tuple[SwitchEvent, ...] = ()
    final_policy: str = ""
    exhausted: bool = False
    error: str | None = None
    outcome_message: str = ""


class ScriptedPolicy:
    """确定性回退策略：把机械臂复位到安全初始姿态。

    不依赖目标定位：作为 ACT 失败后安全回退，保证"不卡在异常姿态"。
    """

    def __init__(self, executor: RealSkillExecutor, retries_after_reset: int = 0) -> None:
        self.executor = executor
        self.retries_after_reset = retries_after_reset

    def act(self, skill: str, params: dict[str, Any] | None = None) -> SkillResult:
        """执行 scripted 原语；失败时回退 release+home 兜底复位。"""
        result = self.executor.execute(skill, params)
        if result.success:
            return result
        # 复位兜底：张开夹爪 + 归位
        self.executor.execute("release")
        home = self.executor.execute("reset_pose")
        if not home.success:
            return SkillResult(False, failure_reason=f"scripted 复位失败: {home.failure_reason}")
        return SkillResult(False, failure_reason=f"scripted 原语失败: {result.failure_reason}")


class RealSkillRouter:
    """真机技能路由：主策略(ACT)优先，失败按失败码切到回退策略。"""

    def __init__(
        self,
        executor: RealSkillExecutor,
        sources: tuple[PolicySource, ...],
        *,
        retryable_codes: tuple[str, ...] = ("grasp_empty", "not_lifted", "not_holding"),
        max_policy_attempts: int = 2,
    ) -> None:
        if not sources:
            raise ValueError("sources 不能为空")
        self.executor = executor
        self.sources = sources
        self.retryable_codes = retryable_codes
        self.max_policy_attempts = max_policy_attempts

    # -- 路由执行 ----------------------------------------------------------
    def run_skill(self, skill: str, params: dict[str, Any] | None = None) -> RouteResult:
        switches: list[SwitchEvent] = []
        policy_steps: dict[str, int] = {}
        step = 0
        active = self.sources[0].name
        last_failure = ""

        while step < self.max_policy_attempts * len(self.sources) * 2:
            step += 1
            policy_steps[active] = policy_steps.get(active, 0) + 1
            result = self._execute_active(active, skill, params)
            if result.success:
                return RouteResult(
                    True, task=skill, steps=step, policy_steps=policy_steps,
                    switch_events=tuple(switches), final_policy=active, exhausted=False,
                    outcome_message=f"{active} 执行 {skill} 成功",
                )
            last_failure = result.failure_reason or "unknown"

            # 当前策略还有重试机会（可重试失败码）
            if last_failure in self.retryable_codes and policy_steps[active] < 2:
                continue

            # 切到下一个可用策略位
            idx = next(i for i, s in enumerate(self.sources) if s.name == active)
            if idx + 1 >= len(self.sources):
                return RouteResult(
                    False, task=skill, steps=step, policy_steps=policy_steps,
                    switch_events=tuple(switches), final_policy=active, exhausted=True,
                    error=last_failure, outcome_message=f"全部策略位失败({last_failure})",
                )
            nxt = self.sources[idx + 1]
            switches.append(SwitchEvent(from_policy=active, to_policy=nxt.name,
                                        reason=last_failure, at_step=step))
            logger.info("路由切换 %s -> %s（原因 %s）", active, nxt.name, last_failure)
            active = nxt.name
            policy_steps[active] = policy_steps.get(active, 0)

        return RouteResult(
            False, task=skill, steps=step, policy_steps=policy_steps,
            switch_events=tuple(switches), final_policy=active, exhausted=True,
            error="达到最大尝试步数",
        )

    def _execute_active(self, active: str, skill: str, params: dict[str, Any] | None) -> SkillResult:
        """执行某一策略位：ACT 走 executor；scripted 走复位原语。"""
        if active == ROUTE_ACT:
            return self.executor.execute(skill, params)
        if active == ROUTE_SCRIPTED:
            # scripted 只支持确定性原语；对抓取任务回退到 release+reset
            if skill == "grasp_screwdriver":
                r = ScriptedPolicy(self.executor).act("reset_pose")
                return SkillResult(False, failure_reason="scripted 无抓取能力，仅复位")
            return ScriptedPolicy(self.executor).act(skill, params)
        return SkillResult(False, failure_reason=f"未实现策略位 {active}")


def default_real_route() -> tuple[PolicySource, ...]:
    """默认路由：ACT 主策略 → scripted 回退。"""
    return (PolicySource(kind=ROUTE_ACT, name=ROUTE_ACT),
            PolicySource(kind=ROUTE_SCRIPTED, name=ROUTE_SCRIPTED))
