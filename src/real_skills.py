"""定义 LLM 可调用的真机技能列表和参数 Schema（阶段 F，SO-101）。

结构复刻自 `skill_registry.py`（纯逻辑、无仿真依赖），技能清单按
真机 SO-101 可执行能力重建。真实语义：
- 当前数据/策略仅支持“夹起并抬离桌面上的螺丝刀”这一完整闭环技能，
  其余为辅助/控制/验证类原语。
"""

from __future__ import annotations

from typing import Any

_REAL_SKILLS: list[dict[str, Any]] = [
    {
        "name": "grasp_screwdriver",
        "description": "驱动机械臂执行 ACT 闭环：靠近桌面上的螺丝刀、张开夹爪、夹住并抬离桌面。"
        "成功后进入保持姿态（螺丝刀被夹住），等待后续指令",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "reset_pose",
        "description": "让机械臂回到采集初始姿态（低风险复位动作）",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "verify_holding",
        "description": "通过夹爪闭合位置与空闭基准检查当前是否夹住了物体，不移动机械臂",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "release",
        "description": "安全张开夹爪放下物体（仅限人工已确认夹爪位于安全高度的场景）",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "done",
        "description": "任务完成，不再执行任何操作，输出总结",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

# 真机失败码（real_verifier / failure 分类用）
FAIL_GRASP_EMPTY = "grasp_empty"          # 夹爪闭合但未被物体挡住（咬空）
FAIL_NOT_LIFTED = "not_lifted"            # 夹住但未抬离桌面
FAIL_HOME_TIMEOUT = "home_timeout"        # 归位超时
FAIL_TIMEOUT = "timeout"                  # 技能执行超时
FAIL_COMM = "comm_error"                  # 通信/驱动错误
FAIL_NOT_HOLDING = "not_holding"          # verify_holding 确认未夹住


def get_real_skills() -> list[dict[str, Any]]:
    """返回所有可用真机技能的列表（含 schema）。"""
    return _REAL_SKILLS


def get_real_skill(name: str) -> dict[str, Any] | None:
    """按名称查找真机技能定义。"""
    for skill in _REAL_SKILLS:
        if skill["name"] == name:
            return skill
    return None


def get_real_skills_description() -> str:
    """生成供 System Prompt 使用的真机技能描述文本。"""
    lines: list[str] = []
    for skill in _REAL_SKILLS:
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


def validate_real_parameters(skill_name: str, params: dict[str, Any] | None) -> str | None:
    """校验真机技能参数是否符合 schema。

    Returns:
        None 表示校验通过，否则返回错误信息。
    """
    params = params or {}
    skill = get_real_skill(skill_name)
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
