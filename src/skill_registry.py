"""定义 LLM 可调用的技能列表和参数 Schema。"""

from __future__ import annotations

from typing import Any

_SKILLS: list[dict[str, Any]] = [
    {
        "name": "scan",
        "description": "原地旋转一周，收集周围可见的可抓取物体和容器",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "pick",
        "description": "找到并抓取指定类型的最近物体",
        "parameters": {
            "type": "object",
            "properties": {
                "object_type": {
                    "type": "string",
                    "description": "AI2-THOR 物体类型，如 CreditCard, Apple, Book, Key",
                }
            },
            "required": ["object_type"],
        },
    },
    {
        "name": "place",
        "description": "将当前持有的物体放入指定类型的容器",
        "parameters": {
            "type": "object",
            "properties": {
                "receptacle_type": {
                    "type": "string",
                    "description": "AI2-THOR 容器类型，如 Drawer, Cabinet, GarbageCan, CounterTop",
                }
            },
            "required": ["receptacle_type"],
        },
    },
    {
        "name": "move_to",
        "description": "向指定方向移动，探索环境",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "left", "right"],
                    "description": "移动方向（相对当前朝向）",
                }
            },
            "required": ["direction"],
        },
    },
    {
        "name": "done",
        "description": "任务完成，不再执行任何操作",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


def get_skills() -> list[dict[str, Any]]:
    """返回所有可用技能的列表（含 schema）。"""
    return _SKILLS


def get_skill(name: str) -> dict[str, Any] | None:
    """按名称查找技能定义。"""
    for skill in _SKILLS:
        if skill["name"] == name:
            return skill
    return None


def get_skills_description() -> str:
    """生成供 System Prompt 使用的技能描述文本。"""
    lines: list[str] = []
    for skill in _SKILLS:
        params_desc = ""
        props = skill["parameters"].get("properties", {})
        required = skill["parameters"].get("required", [])
        if props:
            params = []
            for pname, pinfo in props.items():
                marker = " (必填)" if pname in required else ""
                params.append(f"{pname}: {pinfo.get('description', '')}{marker}")
            params_desc = f"\n  参数: {', '.join(params)}"
        lines.append(f"- {skill['name']}: {skill['description']}{params_desc}")
    return "\n".join(lines)


def validate_parameters(skill_name: str, params: dict[str, Any] | None) -> str | None:
    """校验技能参数是否符合 schema。

    Returns:
        None 表示校验通过，否则返回错误信息。
    """
    params = params or {}
    skill = get_skill(skill_name)
    if skill is None:
        return f"未知技能: {skill_name}"

    required = skill["parameters"].get("required", [])
    for field in required:
        if field not in params:
            return f"技能 '{skill_name}' 缺少必填参数: {field}"

    props = skill["parameters"].get("properties", {})
    for field, info in props.items():
        if field in params:
            enum = info.get("enum")
            if enum and params[field] not in enum:
                return f"参数 '{field}' 的值 '{params[field]}' 不在允许范围内: {enum}"

    return None
