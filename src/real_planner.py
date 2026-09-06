"""真机 LLM 计划层（阶段 F，SO-101）——S5。

把单步决策升级为"技能序列"计划：LLM 返回
{"plan": [{"skill": "...", "params": {...}}, ...], "reasoning": "..."}
随后做静态校验（技能存在、参数合法、无非法技能名），再逐条交给执行器。
纯逻辑：校验与解析不依赖机械臂。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from real_skills import validate_real_parameters

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    skill: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    reasoning: str
    steps: list[PlanStep]

    @property
    def is_empty(self) -> bool:
        return not self.steps


@dataclass
class PlanError:
    """计划解析/校验失败。"""

    message: str
    raw: str = ""


def parse_plan(raw: str) -> Plan | PlanError:
    """从 LLM 文本解析计划 JSON（容错提取花括号块）。"""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end])
            except json.JSONDecodeError:
                return PlanError("LLM 输出不是合法 JSON", raw[:200])
        else:
            return PlanError("LLM 输出未包含 JSON", raw[:200])
    if not isinstance(data, dict) or "plan" not in data:
        return PlanError("缺少 'plan' 字段", raw[:200])
    if not isinstance(data["plan"], list) or not data["plan"]:
        return PlanError("plan 必须是非空技能数组", raw[:200])
    reasoning = data.get("reasoning", "")
    steps = []
    for i, item in enumerate(data["plan"]):
        if not isinstance(item, dict) or "skill" not in item:
            return PlanError(f"plan[{i}] 缺少 skill", raw[:200])
        params = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
        steps.append(PlanStep(skill=item["skill"], params=params))
    return Plan(reasoning=reasoning, steps=steps)


def validate_plan(plan: Plan) -> str | None:
    """静态校验：技能存在 + 参数合法。None 表示通过。"""
    for i, step in enumerate(plan.steps):
        err = validate_real_parameters(step.skill, step.params)
        if err:
            return f"plan[{i}] {step.skill}: {err}"
    return None


def format_plan(plan: Plan) -> str:
    """给 LLM 看的计划回显（可注入下一次 prompt 作为执行跟踪）。"""
    lines = [f"{i + 1}. {s.skill}{s.params if s.params else ''}" for i, s in enumerate(plan.steps)]
    return "\n".join(lines)


def execute_plan(executor, plan: Plan) -> list[Any]:
    """顺序执行计划步骤（真实场景接入恢复重试由 Agent/Recovery 层负责）。

    Args:
        executor: 具备 execute(skill, params) -> SkillResult 的对象。
        plan: 已通过 validate_plan 的计划。

    Returns:
        每步 (PlanStep, SkillResult) 的列表，遇失败/非法即停。
    """
    from real_executor import SkillResult

    results: list[tuple[PlanStep, SkillResult]] = []
    for step in plan.steps:
        if step.skill == "done":
            results.append((step, SkillResult(True, data={"skill": "done"})))
            break
        result = executor.execute(step.skill, step.params)
        results.append((step, result))
        if not result.success:
            break
    return results
