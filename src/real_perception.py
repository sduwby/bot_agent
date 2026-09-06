"""真机感知层（阶段 F，SO-101）——S6/S7。

目标：把"固定俯拍/手眼相机帧 + 关节状态"变成结构化感知，供 LLM 决策与验证。

设计边界（避免虚构能力）：
- 物体级别的检测/定位不内置"自研检测器"，而是通过已配置的多模态 LLM
  （deepseek-v4-flash-vision-exp）对当前帧做结构化问答；
- 夹持/抬起等可用关节阈值本地快判（复用 real_observation/real_verifier）；
- LLM 感知失败或未开视觉时返回 None/低置信，绝不编造物体。

纯逻辑：backend 注入，测试用 FakeBackend。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from real_observation import is_holding

logger = logging.getLogger(__name__)

_VISUAL_JSON_PROMPT = """\
你是机械臂工作台的视觉描述器。基于图片回答，输出严格 JSON：
{"objects": [{"name": "...", "region": "left|center|right", "visible": true}],
 "note": "..."}
objects 只列你确实看到的桌面物体；没看到任何明确物体就返回空数组。不要猜测。
"""


@dataclass
class PerceptionResult:
    """一次感知输出。"""

    objects: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    confidence: float = 0.0
    source: str = "vision_llm"


def describe_workspace(backend, images: list[str] | None) -> PerceptionResult:
    """用多模态 LLM 描述工作台画面中的可见物体。"""
    if not images:
        return PerceptionResult(note="无图像输入，感知为空", confidence=0.0, source="none")
    try:
        raw = backend.chat(_VISUAL_JSON_PROMPT, "请描述当前工作台画面。", images=images)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("非对象")
        objs = data.get("objects", []) if isinstance(data.get("objects"), list) else []
        clean = []
        for o in objs:
            if isinstance(o, dict) and o.get("name") and o.get("visible"):
                clean.append({"name": str(o["name"]),
                              "region": str(o.get("region", "center"))})
        return PerceptionResult(
            objects=clean,
            note=str(data.get("note", "")),
            confidence=0.9 if clean else 0.3,
            source="vision_llm",
        )
    except Exception as exc:
        logger.warning("视觉描述失败: %s", exc)
        return PerceptionResult(note=f"视觉感知失败: {exc}", confidence=0.0, source="none")


def local_holding_state(gripper_pos: float, empty_grip_baseline: float) -> dict[str, Any]:
    """本地夹持快判（关节阈值，无需视觉）。"""
    holding = is_holding(gripper_pos, empty_grip_baseline)
    return {"holding": holding, "gripper_pos": round(gripper_pos, 1), "source": "joint"}
