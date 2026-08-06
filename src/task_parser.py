"""将中文自然语言指令映射为 pick/place 任务参数。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TaskSpec:
    """一句中文指令映射后的结构化任务。"""

    object_type: str
    receptacle_type: str
    raw_instruction: str


_INSTRUCTION_PATTERNS = (
    re.compile(r"^把(?P<object>.+?)放(?:进|入|到)(?P<receptacle>.+)$"),
    re.compile(r"^将(?P<object>.+?)放(?:进|入|到)(?P<receptacle>.+)$"),
    re.compile(r"^请(?:帮我)?(?:把|将)?(?P<object>.+?)放(?:进|入|到)(?P<receptacle>.+)$"),
)


_OBJECT_ALIASES = {
    "信用卡": "CreditCard",
    "creditcard": "CreditCard",
    "apple": "Apple",
    "苹果": "Apple",
    "杯子": "Cup",
    "cup": "Cup",
    "书": "Book",
    "书本": "Book",
    "钥匙": "Key",
    "钥匙匙": "Key",
}


_RECEPTACLE_ALIASES = {
    "抽屉": "Drawer",
    "drawer": "Drawer",
    "垃圾桶": "GarbageCan",
    "垃圾箱": "GarbageCan",
    "cabinet": "Cabinet",
    "柜子": "Cabinet",
    "台面": "CounterTop",
    "桌子": "CounterTop",
    "桌面": "CounterTop",
    "微波炉": "Microwave",
    "microwave": "Microwave",
}


def _normalize(raw: str) -> str:
    """去掉常见空白、标点和停顿词，统一用于字典匹配。"""
    cleaned = re.sub(r"[\\s，。,.、!?！？]", "", raw.strip().lower())
    for stop_word in ("一个", "这", "这个", "那个", "把", "请", "一下", "进去", "里面"):
        cleaned = cleaned.replace(stop_word, "")
    for suffix in ("中的", "里的", "里面"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned


def _resolve(raw: str, alias_map: dict[str, str]) -> str:
    """把中文或英文名称解析为 AI2-THOR 的 objectType。"""
    normalized = _normalize(raw)
    for alias_key, target in alias_map.items():
        alias = _normalize(alias_key)
        if normalized == alias:
            return target
        if alias in normalized:
            return target
    raise ValueError(f"未支持的类别：{raw}")


def parse_instruction(instruction: str) -> TaskSpec:
    """解析“把X放进Y”指令，并返回 pick/place 的标准对象类型。"""
    text = instruction.strip()
    if not text:
        raise ValueError("指令不能为空")

    for pattern in _INSTRUCTION_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        object_raw = match.group("object")
        receptacle_raw = match.group("receptacle")
        if not object_raw or not receptacle_raw:
            continue
        object_type = _resolve(object_raw, _OBJECT_ALIASES)
        receptacle_type = _resolve(receptacle_raw, _RECEPTACLE_ALIASES)
        return TaskSpec(
            object_type=object_type,
            receptacle_type=receptacle_type,
            raw_instruction=instruction,
        )

    raise ValueError(
        "指令格式不支持，目前支持形如“把信用卡放进抽屉”这类中文短句。"
    )


def supported_terms() -> dict[str, list[str]]:
    """返回当前可解析的中文对象和容器名词。"""
    return {
        "objects": sorted(_OBJECT_ALIASES.keys(), key=str.casefold),
        "receptacles": sorted(_RECEPTACLE_ALIASES.keys(), key=str.casefold),
    }
