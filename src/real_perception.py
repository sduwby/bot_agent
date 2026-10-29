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
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

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


# ---------------------------------------------------------------------------
# 像素级 2D 定位（第 38 周 P0 / 子任务 0.3）
#
# 动机：现有 `describe_workspace` 只给 `region: left|center|right`，没有坐标；
# 而 P0 的几何链（`depth_from_known_size` 需 bbox 尺寸、`triangulate` 需像素中心）
# 都要求**像素级**输出。本段定义可插拔的定位源接口：
#   - `VLMObjectLocator`：用已配置的多模态 LLM 输出 bbox（零额外依赖）
#   - 后续可插入 GroundingDINO 类检测器（实现同一 Protocol 即可对比选型）
# 选型（VLM bbox 精度 vs 专用检测器）需模型 API/真机，本批只落接口与解析。
# ---------------------------------------------------------------------------

_LOCALIZE_JSON_PROMPT = """\
你是机械臂工作台的视觉定位器。基于图片，给出每个可见目标物体的像素级包围盒。
输出严格 JSON：
{"objects": [{"name": "...", "bbox_px": [x1, y1, x2, y2], "confidence": 0.0}],
 "note": "..."}
坐标以图像左上角为原点、像素为单位，x1<x2、y1<y2，包围盒需贴合物体。
只列你确实看到的物体；没看到就返回空数组。不要猜测，不要输出图像尺寸以外的坐标。
"""


@dataclass
class Detection:
    """单个目标的像素级检测结果（bbox 为图像像素坐标）。"""

    name: str
    bbox_px: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(float(v) for v in self.bbox_px)
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            raise ValueError("bbox_px 必须是 4 个有限数值")
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_px 必须满足 x2 > x1 且 y2 > y1")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence 必须在 [0, 1]")
        name = str(self.name).strip()
        if not name:
            raise ValueError("name 不能为空")
        self.name = name
        self.bbox_px = (x1, y1, x2, y2)
        self.confidence = float(self.confidence)

    @property
    def center_px(self) -> tuple[float, float]:
        """包围盒中心像素坐标（三角化与抓取点定位用）。"""
        x1, y1, x2, y2 = self.bbox_px
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def size_px(self) -> tuple[float, float]:
        """包围盒像素尺寸 ``(width, height)``（已知尺寸反解深度用）。"""
        x1, y1, x2, y2 = self.bbox_px
        return (x2 - x1, y2 - y1)


@dataclass
class LocalizationResult:
    """一次 2D 定位输出。"""

    detections: list[Detection] = field(default_factory=list)
    note: str = ""
    confidence: float = 0.0
    source: str = "vision_llm"
    skipped: int = 0  # 因解析失败、坐标不可换算或越界过多被丢弃的候选数
    clamped: int = 0  # 因边缘轻微越界被裁剪到画面内的候选数
    coord_space: str = "pixel"  # 实际采用的坐标约定（见 _resolve_coord_space）


class ObjectLocator(Protocol):
    """2D 定位源接口：便于对 VLM 与专用检测器做同口径选型对比。"""

    def locate(self, images: Sequence[str], query: str | None = None) -> LocalizationResult: ...


def _parse_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    """解析 bbox 字段；非法返回 None（由调用方计入 skipped）。"""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = tuple(float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# 显式指定像素约定时，允许的最大越界比例（超出则判为非法框）
_CLAMP_TOLERANCE = 0.2


def _clamp_to_frame(
    bbox: tuple[float, float, float, float],
    frame_size: tuple[int, int],
    tolerance: float = _CLAMP_TOLERANCE,
) -> tuple[tuple[float, float, float, float], bool] | None:
    """把像素框裁剪到画面内；越界过多或裁剪后退化则返回 None。

    返回值 ``(裁剪后 bbox, 是否发生过裁剪)``。

    真实帧上模型常有**边缘轻微过冲**（实测 `gui-plus` 给出 x2=644 而图宽 640），
    这类 4 px 过冲应视为正常误差并裁剪，而不是整条丢弃——丢弃会低估模型表现、
    污染选型对比。越界超过 ``tolerance`` 比例才判为非法框。
    """
    width, height = float(frame_size[0]), float(frame_size[1])
    x1, y1, x2, y2 = bbox
    if (x1 < -tolerance * width or y1 < -tolerance * height
            or x2 > (1.0 + tolerance) * width or y2 > (1.0 + tolerance) * height):
        return None
    cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
    cx2, cy2 = min(x2, width), min(y2, height)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    clamped = (cx1, cy1, cx2, cy2)
    return clamped, clamped != (x1, y1, x2, y2)


# 支持的坐标约定。**模型的坐标约定是"先验"，prompt 约束并不可靠**：
# Qwen-VL 系列官方即使用 [0, 1000) 归一化 bbox（Qwen-VL 论文 §2.2），
# 实测 `qwen3-vl-plus` 在我们明确要求"像素为单位"的情况下仍返回 0–1000，
# 若按像素解读会得到越界坐标、被误判为"模型不准"，因此必须在代码里判定并换算。
COORD_PIXEL = "pixel"
COORD_NORMALIZED_1000 = "normalized_1000"
COORD_NORMALIZED_1 = "normalized_1"
COORD_AUTO = "auto"
_COORD_SPACES = (COORD_PIXEL, COORD_NORMALIZED_1000, COORD_NORMALIZED_1, COORD_AUTO)

# `auto` 判定"坐标越出画面 ⇒ 只能是归一化"时的容差。
# 取 2% 是为了同时满足两个真实约束：
#   - 像素模型常有**几像素边缘过冲**（实测 `gui-plus` 给 x2=644 而图宽 640，仅 +0.6%），
#     不该因此被判成归一化 —— 否则垃圾框会被"换算"成看似合理的框（静默错误）；
#   - 归一化 0–1000 的真实越界通常远超 2%（实测 `qwen3-vl-plus` 给 y2=526 而图高 480，+9.6%），
#     仍能被正确识别。
# 即：容差落在 "+0.6%（须判像素）" 与 "+9.6%（须判归一化）" 之间。
_AUTO_OVERFLOW_TOLERANCE = 0.02


def _resolve_coord_space(
    requested: str,
    raw_bboxes: Sequence[tuple[float, float, float, float]],
    frame_size: tuple[int, int] | None,
) -> str:
    """确定一批 bbox 使用的坐标约定。

    ``requested`` 非 ``"auto"`` 时直接采用（调用方已知模型约定时最可靠）。
    ``"auto"`` 的确定性规则：

    1. 所有坐标都 ≤ 1 → ``normalized_1``；
    2. 给了 ``frame_size`` 且存在坐标**超出其对应轴的尺寸超过 ``_AUTO_OVERFLOW_TOLERANCE``**
       （x > 宽×(1+tol) 或 y > 高×(1+tol)）→ ``normalized_1000``：像素坐标不可能越出
       画面这么多，只能是归一化；
    3. 其余 → ``pixel``。

    规则 3 存在固有歧义：若 0–1000 坐标恰好都在图像尺寸以内（例如小目标位于
    左上角），会被误判为像素。**已知模型约定时请显式传 ``coord_space``**
    （或在 `run_perception_locate_eval.py` 里用 ``VLM_<源>_COORD_SPACE`` 配置）。
    """
    if requested != COORD_AUTO:
        return requested
    values = [v for bbox in raw_bboxes for v in bbox]
    if not values:
        return COORD_PIXEL
    if max(values) <= 1.0:
        return COORD_NORMALIZED_1
    if frame_size is not None:
        width, height = float(frame_size[0]), float(frame_size[1])
        limit_x = width * (1.0 + _AUTO_OVERFLOW_TOLERANCE)
        limit_y = height * (1.0 + _AUTO_OVERFLOW_TOLERANCE)
        for x1, y1, x2, y2 in raw_bboxes:
            if x1 > limit_x or x2 > limit_x or y1 > limit_y or y2 > limit_y:
                return COORD_NORMALIZED_1000
    return COORD_PIXEL


def _to_pixels(
    bbox: tuple[float, float, float, float],
    space: str,
    frame_size: tuple[int, int] | None,
) -> tuple[float, float, float, float] | None:
    """把 bbox 换算为像素坐标；无法换算（归一化但缺 frame_size）时返回 None。"""
    if space == COORD_PIXEL:
        return bbox
    if frame_size is None:
        return None
    width, height = float(frame_size[0]), float(frame_size[1])
    scale_x = width if space == COORD_NORMALIZED_1 else width / 1000.0
    scale_y = height if space == COORD_NORMALIZED_1 else height / 1000.0
    x1, y1, x2, y2 = bbox
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def _extract_json_object(text: Any) -> dict[str, Any] | None:
    """从模型回复中提取 JSON 对象。

    视觉模型普遍会用 ```json 围栏包裹，或在前后附加解释文字；严格 ``json.loads``
    会因此把**有效结果**判成解析失败。这里先直接解析，失败则回退到"截取第一个
    配对完整的 ``{...}``"，并在扫描时正确跳过字符串内的花括号。
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(stripped[start:index + 1])
                except json.JSONDecodeError:
                    return None
                return data if isinstance(data, dict) else None
    return None


def extract_json_object(text: Any) -> dict[str, Any] | None:
    """公开入口：从模型回复中提取 JSON 对象（容错 ```json 围栏与前后解释文字）。"""
    return _extract_json_object(text)


def locate_objects(
    backend: Any,
    images: Sequence[str] | None,
    *,
    frame_size: tuple[int, int] | None = None,
    query: str | None = None,
    coord_space: str = COORD_AUTO,
) -> LocalizationResult:
    """用多模态 LLM 输出像素级检测结果。

    Args:
        backend: 实现 ``chat(prompt, user, images=...)`` 的多模态后端（复用 `llm_planner` 契约）。
        images: 图像列表（约定为单帧或多视角；实际取用第一帧）。
        frame_size: ``(width, height)``；给出时越界检测会被丢弃并计入 ``skipped``，
            并把归一化坐标换算为像素。
        query: 目标描述（可选），用于引导定位特定物体。
        coord_space: 模型返回的坐标约定，取 ``auto`` / ``pixel`` /
            ``normalized_1000`` / ``normalized_1``。**已知模型约定时应显式指定**
            （如 Qwen-VL 系列用 ``normalized_1000``），``auto`` 仅作兜底。

    Returns:
        LocalizationResult: 解析成功的检测列表（坐标已是像素）；无图/调用失败/
        无法解析时返回空结果且 ``source="none"``，**绝不用低置信结果冒充检测**
        （与 `describe_workspace` 的降级原则一致）。
    """
    if coord_space not in _COORD_SPACES:
        raise ValueError(f"coord_space 必须是 {_COORD_SPACES} 之一")
    if not images:
        return LocalizationResult(note="无图像输入，定位为空", confidence=0.0, source="none")

    user_prompt = "请定位当前工作台画面中的所有目标物体。"
    if query:
        user_prompt = f"请定位当前工作台画面中的：{query}"
    try:
        raw = backend.chat(_LOCALIZE_JSON_PROMPT, user_prompt, images=list(images))
    except Exception as exc:  # VLM 不可用：明确降级
        logger.warning("2D 定位调用失败: %s", exc)
        return LocalizationResult(note=f"2D 定位失败: {exc}", confidence=0.0, source="none")

    data = _extract_json_object(raw)
    if data is None:
        logger.warning("2D 定位回复无法解析为 JSON 对象: %.200s", raw)
        return LocalizationResult(note=f"2D 定位回复无法解析为 JSON: {raw[:120]}",
                                  confidence=0.0, source="none")

    entries = data.get("objects")
    if not isinstance(entries, list):
        return LocalizationResult(note="objects 字段缺失或非数组", confidence=0.0,
                                  source="vision_llm")

    skipped = 0
    candidates: list[tuple[str, tuple[float, float, float, float], Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        bbox = _parse_bbox(entry.get("bbox_px"))
        name = str(entry.get("name", "")).strip()
        if bbox is None or not name:
            skipped += 1
            continue
        candidates.append((name, bbox, entry.get("confidence", 0.0)))

    space = _resolve_coord_space(coord_space, [bbox for _, bbox, _ in candidates], frame_size)
    if space != COORD_PIXEL and frame_size is None:
        return LocalizationResult(
            note=f"检测到 {space} 坐标但未提供 frame_size，无法换算为像素",
            confidence=0.0, source="vision_llm", skipped=skipped, coord_space=space,
        )

    detections: list[Detection] = []
    clamped = 0
    for name, raw_bbox, raw_confidence in candidates:
        pixels = _to_pixels(raw_bbox, space, frame_size)
        if pixels is None:
            skipped += 1
            continue
        if frame_size is not None:
            clamped_result = _clamp_to_frame(pixels, frame_size)
            if clamped_result is None:
                logger.warning("丢弃越界过多的检测 %s: %s（约定 %s，换算后 %s）",
                               name, raw_bbox, space, tuple(round(v, 1) for v in pixels))
                skipped += 1
                continue
            pixels, was_clamped = clamped_result
            clamped += int(was_clamped)
        try:
            detections.append(Detection(name=name, bbox_px=pixels,
                                        confidence=float(raw_confidence)))
        except (ValueError, TypeError):
            skipped += 1

    confidence = (sum(d.confidence for d in detections) / len(detections)) if detections else 0.0
    note = str(data.get("note", ""))
    if space != COORD_PIXEL:
        note = f"{note}（坐标约定：{space} → 已换算为像素）".strip()
    if clamped:
        note = f"{note}（{clamped} 个检测边缘越界已裁剪）".strip()
    if skipped:
        note = f"{note}（丢弃 {skipped} 个非法/越界候选）".strip()
    return LocalizationResult(detections=detections, note=note, confidence=confidence,
                              source="vision_llm", skipped=skipped, clamped=clamped,
                              coord_space=space)


class VLMObjectLocator:
    """基于多模态 LLM 的定位源（`ObjectLocator` 的一种实现）。

    ``coord_space`` 建议按模型显式指定（如 Qwen-VL 系列为 ``normalized_1000``），
    避免 ``auto`` 在"归一化坐标恰好落在图像尺寸内"时误判。
    """

    def __init__(self, backend: Any, frame_size: tuple[int, int] | None = None,
                 coord_space: str = COORD_AUTO) -> None:
        self.backend = backend
        self.frame_size = frame_size
        self.coord_space = coord_space

    def locate(self, images: Sequence[str], query: str | None = None) -> LocalizationResult:
        return locate_objects(self.backend, images, frame_size=self.frame_size,
                              query=query, coord_space=self.coord_space)


class FakeObjectLocator:
    """离线/测试用定位源：返回预设结果，可选抛错以验证降级。"""

    def __init__(self, detections: Sequence[Detection] = (), error: Exception | None = None) -> None:
        self.detections = list(detections)
        self.error = error
        self.calls = 0

    def locate(self, images: Sequence[str], query: str | None = None) -> LocalizationResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return LocalizationResult(detections=list(self.detections), source="fake")
