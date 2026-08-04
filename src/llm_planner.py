"""LLM 后端接口与 System Prompt 构建。"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from skill_registry import get_skills_description


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
"""


def build_user_prompt(
    instruction: str,
    observation_summary: str,
    action_history: list[dict[str, Any]] | None = None,
) -> str:
    """构建发给 LLM 的 User Prompt。"""
    parts: list[str] = [
        f"## 当前观察\n\n{observation_summary}",
        f"## 任务指令\n\n{instruction}",
    ]

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
        parts.append(f"## 已执行的操作\n\n{'\n'.join(history_text)}")

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
    """规则引擎回退：当 LLM 不可用时，执行预定义的 scan → pick → place → done 序列。"""

    def __init__(self) -> None:
        self._step = 0

    def chat(self, system: str, user: str) -> str:
        decision = self._next_decision()
        self._step += 1
        return json.dumps(decision, ensure_ascii=False)

    def _next_decision(self) -> dict[str, Any]:
        decisions = [
            {"thought": "先扫描环境了解可用物体", "skill": "scan", "parameters": {}},
            {
                "thought": "扫描完成后抓取目标物体",
                "skill": "pick",
                "parameters": {"object_type": "CreditCard"},
            },
            {
                "thought": "将物体放入容器",
                "skill": "place",
                "parameters": {"receptacle_type": "Drawer"},
            },
            {"thought": "任务完成", "skill": "done", "parameters": {}},
        ]
        if self._step < len(decisions):
            return decisions[self._step]
        return {"thought": "已达最大步骤，强制终止", "skill": "done", "parameters": {}}


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
