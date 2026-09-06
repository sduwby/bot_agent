"""真机观测收集与 LLM 摘要（SO-101）。

将机械臂关节状态/夹爪夹持状态/最近技能结果整理为喂给 LLM 的中文摘要。
纯逻辑：输入为 dict（由执行器提供），不直接驱动硬件。
"""

from __future__ import annotations

from typing import Any

# 与 outputs/infer_so101.py / deploy_eval 保持一致
STATE_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# 判定阈值（与数据标定一致）
GRASP_CLOSED = 10.0
LIFT_WRIST = 75.0


def summarize_state(state: dict[str, float] | None) -> str:
    """把关节状态 dict 转成紧凑中文描述。"""
    if not state:
        return "（无关节状态）"
    parts = []
    parts.append(f"手腕弯曲 {state.get('wrist_flex.pos', 0):.1f}" + ("(抬起)" if state.get("wrist_flex.pos", 0) >= LIFT_WRIST else "(未抬起)"))
    parts.append(f"夹爪位置 {state.get('gripper.pos', 0):.1f}" + ("(闭合)" if state.get("gripper.pos", 0) < GRASP_CLOSED else "(张开)"))
    return "；".join(parts)


def build_observation_summary(
    state: dict[str, float] | None,
    *,
    holding: bool | None = None,
    empty_grip_baseline: float | None = None,
    last_skill_result: str | None = None,
    recovery_hint: str | None = None,
    instruction: str | None = None,
) -> dict[str, Any]:
    """构造真机观测结构（喂给 build_user_prompt 的字段来源）。

    Returns:
        dict: 含 task_instruction / joint_summary / holding / last_skill_result /
              recovery_hint 等字段的中文摘要文本（用于 LLM user prompt）。
    """
    lines = []
    lines.append(f"## 当前机械臂状态\n{summarize_state(state)}")
    if empty_grip_baseline is not None:
        lines.append(f"- 空闭基准 gripper={empty_grip_baseline:.1f}（夹住判定 > {empty_grip_baseline + 3.0:.1f}）")
    if holding is not None:
        lines.append(f"- 是否夹住物体: {'是' if holding else '否'}")
    if last_skill_result:
        lines.append(f"- 最近技能结果: {last_skill_result}")
    if recovery_hint:
        lines.append(f"- 恢复建议: {recovery_hint}")

    return {
        "task_instruction": instruction or "",
        "summary_text": "\n".join(lines),
        "joint_summary": summarize_state(state),
        "holding": holding,
        "last_skill_result": last_skill_result,
        "recovery_hint": recovery_hint,
    }


def is_holding(gripper_pos: float, empty_grip_baseline: float, margin: float = 3.0) -> bool:
    """夹爪当前位置高于空闭基准即视为夹住物体。"""
    return gripper_pos > empty_grip_baseline + margin


def is_lifted(wrist_flex_pos: float, threshold: float = LIFT_WRIST) -> bool:
    """wrist_flex 是否达到抬起判定。"""
    return wrist_flex_pos >= threshold
