"""LLM 后端接口与 System Prompt 构建。"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from skill_registry import get_skills_description
from task_parser import parse_instruction


class LLMBackend(Protocol):
    """LLM 后端统一接口。"""

    def chat(self, system: str, user: str) -> str: ...


_SYSTEM_PROMPT_TEMPLATE = """\
你是一个具身智能 Agent，在一个模拟的室内环境中执行任务。

## 你的能力

你可以调用以下技能：
{skills_description}

## 输出格式

请返回下一步操作。输出必须是合法的 JSON，包含以下字段：
- thought: 你的思考过程（简要说明）
- skill: 要调用的技能名称
- parameters: 技能参数（字典，可以为空）

## 失败恢复

当操作失败时，你会收到错误信息。请根据错误类型选择不同策略：
- 物体未找到：尝试 scan 重新观察，或 move_to 移动到新位置
- 抓取失败：可尝试重试，或寻找同一类型的其他物体
- 放置失败：尝试换另一个容器
"""


def build_user_prompt(
    instruction: str,
    observation_summary: str,
    action_history: list[dict[str, Any]] | None = None,
    memory_context: str | None = None,
) -> str:
    """构建发给 LLM 的 User Prompt。"""
    parts: list[str] = [
        f"## 当前观察\n\n{observation_summary}",
    ]

    if memory_context:
        parts.append(memory_context)

    parts.append(f"## 任务指令\n\n{instruction}")

    if action_history:
        history_text = []
        for i, entry in enumerate(action_history, 1):
            skill = entry.get("skill", "?")
            params = entry.get("parameters", {})
            success = entry.get("success")
            result_summary = entry.get("result_summary", "")
            status = "成功" if success else "失败"
            history_text.append(
                f"{i}. {skill}{params} → {status}"
                + (f" ({result_summary})" if result_summary else "")
            )
        history_joined = "\n".join(history_text)
        parts.append(f"## 已执行的操作\n\n{history_joined}")

    return "\n\n".join(parts)


class OpenAIBackend:
    """通过 OpenAI 兼容 API 调用 LLM。

    环境变量配置：
        LLM_API_BASE: API 端点（如 https://api.openai.com/v1）
        LLM_API_KEY:  API 密钥
        LLM_MODEL:    模型名称（如 gpt-4o-mini）
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_base = api_base or os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self._client: Any | None = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import openai  # type: ignore
        except ImportError:
            raise RuntimeError("openai 包未安装，无法使用 OpenAI 后端。运行: pip install openai")
        self._client = openai.OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

    def chat(self, system: str, user: str) -> str:
        self._ensure_client()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        return content.strip()


class RuleFallback:
    """规则引擎回退：按当前任务目标执行 scan → pick → place → done。"""

    def __init__(self) -> None:
        self._step = 0

    def chat(self, system: str, user: str) -> str:
        instruction = self._extract_instruction(user)
        spec = parse_instruction(instruction)
        if f"place{{'receptacle_type': '{spec.receptacle_type}'}} → 成功" in user:
            decision = {
                "thought": "放置已成功，请求验证任务状态",
                "skill": "done",
                "parameters": {},
            }
        elif f"当前持有: {spec.object_type}" in user:
            decision = {
                "thought": f"将目标物体放入 {spec.receptacle_type}",
                "skill": "place",
                "parameters": {"receptacle_type": spec.receptacle_type},
            }
        elif "scan{} → 成功" not in user:
            decision = {
                "thought": "先扫描环境了解可用物体",
                "skill": "scan",
                "parameters": {},
            }
        else:
            decision = {
                "thought": f"扫描完成后抓取 {spec.object_type}",
                "skill": "pick",
                "parameters": {"object_type": spec.object_type},
            }
        self._step += 1
        return json.dumps(decision, ensure_ascii=False)

    @staticmethod
    def _extract_instruction(user_prompt: str) -> str:
        marker = "## 任务指令\n\n"
        if marker not in user_prompt:
            raise ValueError("User Prompt 中缺少任务指令")
        remainder = user_prompt.split(marker, 1)[1]
        return remainder.split("\n\n## ", 1)[0].strip()


def parse_decision(raw: str) -> dict[str, Any] | None:
    """从 LLM 返回的文本中解析 JSON 决策。"""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # 尝试从包含 JSON 的文本中提取
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end])
            except json.JSONDecodeError:
                return None
        else:
            return None

    required_fields = ("thought", "skill", "parameters")
    for field in required_fields:
        if field not in data:
            return None

    if not isinstance(data["parameters"], dict):
        data["parameters"] = {}

    return data


def create_backend() -> LLMBackend:
    """根据环境变量创建 LLM 后端。

    如果 LLM_API_KEY 为空，则使用规则引擎回退。
    """
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        return RuleFallback()
    return OpenAIBackend()
