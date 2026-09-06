"""真机 LLM Agent 主循环（阶段 F，SO-101）。

把阶段 A `executor.py` 的“LLM 决策 → 技能分发 → 验证 → 失败恢复”骨架迁移到
真机技能执行器。复用 `llm_planner` 的 chat/解析/提示构建；技能分发与验证走
`RealSkillExecutor`/`RealVerifier`；恢复策略为每技能有限重试（默认 1 次）后终止。

纯逻辑：技能执行器由外部注入（Real 或 Mock），本模块不直接驱动机械臂。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_planner import (
    LLMBackend,
    build_user_prompt,
    create_backend,
    parse_decision,
)
from real_executor import RealSkillExecutor, SkillResult
from real_recovery import RealRecoveryManager, detect_loop
from real_skills import get_real_skills_description

logger = logging.getLogger(__name__)

_REAL_SYSTEM_PROMPT = """\
你是一个具身智能 Agent，控制一台 SO-101 机械臂在真实桌面上执行任务。

## 你的能力

你可以调用以下真机技能：
{skills_description}

## 任务规则

- 目标是中文自然语言指令，通常涉及桌面上的物体（当前只有螺丝刀）。
- 若任务要求夹起/拿起物体：直接调用 grasp_screwdriver（无需参数）。该技能会
  自动完成靠近、张开、夹住并抬离；完成后机械臂保持夹持姿态。
- 若想确认是否夹住：调用 verify_holding。
- 若要把机械臂复位：调用 reset_pose。
- 不要连续重复调用同一技能超过需要的次数。

## 输出格式

返回 JSON，包含：
- thought: 简要思考
- skill: 技能名
- parameters: 参数对象（可为空 {{}}）

## 图像观测

每轮你可能会收到当前两路相机画面（fixed=工作台俯拍、handeye=手臂特写）与机械臂状态文本。
如画面显示任务已完成或无法继续，请用 verify_holding/done 结束；不要臆造画面中不存在的物体。
"""


@dataclass
class AgentStep:
    """一次技能决策与执行记录。"""

    skill: str
    params: dict[str, Any]
    result: SkillResult
    recovery_note: str = ""


@dataclass
class AgentOutcome:
    """任务结束汇总。"""

    success: bool
    message: str
    steps: list[AgentStep] = field(default_factory=list)
    plan_steps_used: int = 0


class RealRobotAgent:
    """真机 LLM Agent 主循环。"""

    def __init__(
        self,
        skill_executor: RealSkillExecutor,
        backend: LLMBackend | None = None,
        max_plan_steps: int = 6,
        max_retries: int = 1,
        vision: bool = True,
    ) -> None:
        self.executor = skill_executor
        self.backend = backend or create_backend()
        self.max_plan_steps = max_plan_steps
        self.max_retries = max_retries
        self.vision = vision
        self.history: list[dict[str, Any]] = []
        self.recovery = RealRecoveryManager(max_retries=max_retries)
        self.system_prompt = _REAL_SYSTEM_PROMPT.format(
            skills_description=get_real_skills_description()
        )

    # -- 观测 --------------------------------------------------------------
    def _observation_summary(self) -> str:
        """收集机械臂当前状态（尽力而为；读不到则提示）。"""
        try:
            state = self.executor.engine.read_state()
            from real_observation import summarize_state

            return summarize_state(state)
        except Exception as exc:  # 无真机/驱动未连
            return f"（无法读取机械臂状态: {exc}）"

    def _current_images(self) -> list[str] | None:
        """读取相机帧并编码为 data URL；失败或未开启时返回 None（降级纯文本）。"""
        if not self.vision:
            return None
        try:
            frames = self.executor.engine.read_images()
        except Exception as exc:
            logger.warning("读取图像失败，降级纯文本决策: %s", exc)
            return None
        if not frames:
            return None
        try:
            from llm_planner import image_to_data_url

            urls = []
            for key in ("observation.images.fixed", "observation.images.handeye"):
                frame = frames.get(key)
                if frame is not None:
                    urls.append(image_to_data_url(frame))
            return urls or None
        except Exception as exc:
            logger.warning("图像编码失败，降级纯文本决策: %s", exc)
            return None

    # -- 主循环 ------------------------------------------------------------
    def run(self, instruction: str) -> AgentOutcome:
        steps: list[AgentStep] = []
        plan_steps = 0
        while plan_steps < self.max_plan_steps:
            plan_steps += 1
            # 1) 决策（携带当前相机帧，视觉模型多模态）
            user_prompt = self._build_user_prompt(instruction)
            raw = self.backend.chat(self.system_prompt, user_prompt,
                                    images=self._current_images())
            decision = parse_decision(raw)
            if decision is None or "skill" not in decision:
                return AgentOutcome(False, f"LLM 输出无法解析: {raw[:200]}", steps, plan_steps)
            skill = decision["skill"]
            params = decision.get("parameters") or {}

            # 2) done → 结束
            if skill == "done":
                steps.append(AgentStep(skill, params, SkillResult(True, {"skill": "done"})))
                return AgentOutcome(True, "任务完成（Agent 判定已达成目标）", steps, plan_steps)

            # 3) 执行
            result = self.executor.execute(skill, params)
            step = AgentStep(skill, params, result)
            self.history.append(
                {"skill": skill, "params": params,
                 "success": result.success, "failure_reason": result.failure_reason}
            )

            # 4) 成功 / 失败恢复（RealRecoveryManager + 循环检测）
            if result.success:
                self.recovery.record_success(skill)
                steps.append(step)
                continue
            code = result.failure_reason or "unknown"
            loop = detect_loop(self.history)
            rec = self.recovery.plan_recovery(skill_name=skill, failure_code=code, loop=loop)
            if rec.action == "retry":
                steps.append(step)
                step.recovery_note = f"技能 {skill} 失败({code})，{rec.reason}"
                continue
            if rec.action == "reset_and_retry" or rec.action == "verify":
                # 复位重试 / 二次确认：先执行复位原语再重试一次
                reset_result = self.executor.execute("reset_pose")
                step.recovery_note = f"{rec.reason}；复位结果={reset_result.success}"
                steps.append(step)
                if reset_result.success:
                    continue
                return AgentOutcome(False, f"恢复失败(复位不成功): {rec.reason}", steps, plan_steps)
            # abort / comm / llm 兜底 → 终止
            steps.append(step)
            if code == "comm_error":
                return AgentOutcome(False, f"通信/驱动错误，终止: {result.failure_reason}", steps, plan_steps)
            return AgentOutcome(
                False,
                f"技能 {skill} 失败且超出重试次数: {result.failure_reason}",
                steps, plan_steps,
            )

        return AgentOutcome(
            False, f"达到最大规划步数 {self.max_plan_steps} 未完成", steps, plan_steps
        )

    def _build_user_prompt(self, instruction: str) -> str:
        action_history = self.history[-6:]
        return build_user_prompt(instruction, self._observation_summary(), action_history)
