"""意图 taxonomy、指令→意图映射，以及 VLM 数据引擎标注链路（第 38 周 P1）。

自由文本指令不能直接计算 Cohen's κ，必须先映射到**离散意图集合**。
映射刻意做成**确定性关键词规则**，而不是再调一次 LLM：

- κ 衡量的是"VLM 标注 vs 人工标注"在同一 taxonomy 上的分歧；若映射本身也是 LLM，
  分歧就无法归因（会混入映射自身的不确定性），指标失去意义；
- 规则映射零成本、可复现、可审计，且不占用模型 API 配额。

标注链路的分层（便于纯逻辑测试）：

    LabelingBackend（VLM/桩）──► build_label_record（纯函数）──► LabelRecord
                                        ▲
                             sensor_state（成功判定）+ taxonomy（意图映射）

`label_dataset` 只依赖 `EpisodeSource` 协议，因此测试用假数据源即可覆盖
"批量 + 断点续跑 + jsonl 落盘"，不需要真数据集或 API。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from real_observation import (
    EMPTY_GRIP_BASELINE,
    GRIP_POS_MARGIN,
    is_holding,
)
from real_perception import extract_json_object

# 意图集合，**按匹配优先级排列**（越靠前越优先）：
# stack/sort 是复合意图（隐含 pick+place），故优先于单动作意图。
INTENT_PRIORITY = ("stack", "sort", "place", "pick")

# 参数槽名称。**值域开放**（物体是开放词表），具体取值由标注或泛化协议另行抽取，
# 本批不做抽取。
SLOT_NAMES = ("object", "target_zone")

# 关键词表：`map_instruction_to_intent` 默认使用；自定义意图需扩展此表或传 `keywords`。
# 匹配为**子串包含**，关键词本身会先归一化；空关键词被忽略。
INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pick": ("拿起", "拿住", "夹起", "夹住", "抓起", "拾起", "捡起", "取起",
             "拿", "夹", "抓", "取", "pick", "grasp", "grab"),
    "place": ("放到", "放进", "放入", "放在", "放下", "放置", "倒进", "送到", "移到",
              "place", "put", "move"),
    "sort": ("分类", "归类", "分拣", "按颜色", "按大小", "sort", "classify", "categorize"),
    "stack": ("堆叠", "摞起来", "垒起来", "叠起来", "叠放", "stack"),
}

_FULLWIDTH_LOWER = 0xFF01
_FULLWIDTH_UPPER = 0xFF5E
_FULLWIDTH_OFFSET = 0xFEE0
_IDEOGRAPHIC_SPACE = "\u3000"


@dataclass(frozen=True)
class IntentTaxonomy:
    """受限意图集合 + 参数槽 + 版本号。

    ``intents`` 的**顺序即匹配优先级**；``version`` 随标注产物一起记录，
    保证 κ 可复现（taxonomy 变了，历史 κ 不可比）。
    """

    intents: tuple[str, ...]
    slots: tuple[str, ...]
    version: str

    def __post_init__(self) -> None:
        if not self.intents:
            raise ValueError("intents 不能为空")
        for label, values in (("intents", self.intents), ("slots", self.slots)):
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{label} 的元素必须是非空字符串")
            if len(set(values)) != len(values):
                raise ValueError(f"{label} 的元素不能重复")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version 必须是非空字符串")

    def to_dict(self) -> dict[str, Any]:
        """序列化，便于与标注产物一起落盘。"""
        return {"intents": list(self.intents), "slots": list(self.slots), "version": self.version}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntentTaxonomy":
        """从 ``to_dict`` 的输出还原。"""
        if not isinstance(data, Mapping):
            raise ValueError("taxonomy 必须是映射类型")
        try:
            return cls(
                intents=tuple(data["intents"]),
                slots=tuple(data["slots"]),
                version=str(data["version"]),
            )
        except KeyError as exc:
            raise ValueError(f"taxonomy 缺少字段: {exc}") from exc
        except TypeError as exc:
            raise ValueError("taxonomy 字段类型不正确") from exc


def define_intent_taxonomy() -> IntentTaxonomy:
    """返回默认 taxonomy（v1）。"""
    return IntentTaxonomy(intents=INTENT_PRIORITY, slots=SLOT_NAMES, version="v1")


def normalize_instruction(text: str) -> str:
    """归一化指令文本：全角转半角、去掉多余空白、英文转小写。

    归一化让"１２３"与"123"、"Ａ"与"a"、"a　b"与"a b"得到同一表示，
    避免因输入法差异产生假的标注分歧。
    """
    if not isinstance(text, str):
        raise ValueError("text 必须是字符串")
    translated = "".join(
        chr(ord(char) - _FULLWIDTH_OFFSET)
        if _FULLWIDTH_LOWER <= ord(char) <= _FULLWIDTH_UPPER
        else char
        for char in text
    )
    return " ".join(translated.replace(_IDEOGRAPHIC_SPACE, " ").split()).lower()


def map_instruction_to_intent(
    text: str,
    taxonomy: IntentTaxonomy,
    keywords: Mapping[str, Sequence[str]] | None = None,
) -> str | None:
    """把自由文本指令映射到 taxonomy 中的离散意图。

    按 ``taxonomy.intents`` 的顺序逐意图做**子串包含**匹配，命中即返回；
    都不命中返回 ``None``（由 κ 计算方作为"未映射"类单独统计）。

    已知局限（v1，需用真实指令语料验证后再迭代）：
    - 不处理否定（"不要拿" 仍会映射到 pick）；
    - 一条指令含多个动作时只返回优先级最高的意图；
    - 关键词表与 ``define_intent_taxonomy()`` 对齐，自定义意图需扩展关键词表。

    Args:
        text: 原始指令文本。
        taxonomy: 意图集合（其顺序为优先级）。
        keywords: 关键词表覆盖项，默认使用 `INTENT_KEYWORDS`。

    Returns:
        命中的意图名，或 ``None``。
    """
    normalized = normalize_instruction(text)
    if not normalized:
        return None
    table = INTENT_KEYWORDS if keywords is None else keywords
    for intent in taxonomy.intents:
        for keyword in table.get(intent, ()):
            normalized_keyword = normalize_instruction(keyword)
            if normalized_keyword and normalized_keyword in normalized:
                return intent
    return None


# --------------------------------------------------------------------------- #
# 2.1 数据结构
# --------------------------------------------------------------------------- #

#: 子任务切分标签（设计 §4.1：approach | align | grasp | lift | place）
SUBTASK_NAMES = ("approach", "align", "grasp", "lift", "place")

LABEL_SCHEMA_VERSION = "week38_label_record_v1"

#: 成功判据取 episode **末尾若干帧**的夹爪位置（夹持发生在后段；避免把中途的
#: 偶发高位置误判为成功）。
SUCCESS_TAIL_FRAMES = 5


class LabelingError(RuntimeError):
    """标注链路契约错误。"""


@dataclass(frozen=True)
class LabelRecord:
    """单条 episode（或子任务切片）的标注记录（设计 §4.1）。

    `source` 区分来源：`vlm`（自动标注）/ `human`（人工复核）/ `sensor`（传感器）。
    只有 `vlm` 与 `human` 两条流会被 `label_audit` 拿来算 κ。
    """

    episode_ref: str
    task_text: str
    frame_range: tuple[int, int] | None = None
    intent: str | None = None
    subtask: str | None = None
    success: bool | None = None
    confidence: float = 0.0
    source: str = "vlm"
    taxonomy_version: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.episode_ref, str) or not self.episode_ref.strip():
            raise LabelingError("episode_ref 必须是非空字符串")
        if not isinstance(self.task_text, str):
            raise LabelingError("task_text 必须是字符串")
        if self.subtask is not None and self.subtask not in SUBTASK_NAMES:
            raise LabelingError(
                f"subtask 必须是 {SUBTASK_NAMES} 之一，实际 {self.subtask!r}"
            )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise LabelingError(f"confidence 必须在 [0,1]，实际 {self.confidence}")
        if self.frame_range is not None:
            start, end = self.frame_range
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
            ):
                raise LabelingError(f"frame_range 非法：{self.frame_range!r}")
        if self.source not in ("vlm", "human", "sensor"):
            raise LabelingError(f"source 必须是 vlm/human/sensor，实际 {self.source!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LABEL_SCHEMA_VERSION,
            "episode_ref": self.episode_ref,
            "task_text": self.task_text,
            "frame_range": list(self.frame_range) if self.frame_range else None,
            "intent": self.intent,
            "subtask": self.subtask,
            "success": self.success,
            "confidence": self.confidence,
            "source": self.source,
            "taxonomy_version": self.taxonomy_version,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelRecord":
        if not isinstance(data, Mapping):
            raise LabelingError("LabelRecord 必须是 mapping")
        try:
            raw_range = data.get("frame_range")
            return cls(
                episode_ref=str(data["episode_ref"]),
                task_text=str(data.get("task_text", "")),
                frame_range=(int(raw_range[0]), int(raw_range[1])) if raw_range else None,
                intent=data.get("intent"),
                subtask=data.get("subtask"),
                success=None if data.get("success") is None else bool(data["success"]),
                confidence=float(data.get("confidence", 0.0)),
                source=str(data.get("source", "vlm")),
                taxonomy_version=str(data.get("taxonomy_version", "")),
                note=str(data.get("note", "")),
            )
        except KeyError as exc:
            raise LabelingError(f"LabelRecord 缺少字段: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise LabelingError(f"LabelRecord 字段类型不正确: {exc}") from exc


# --------------------------------------------------------------------------- #
# 2.2 单 episode 标注
# --------------------------------------------------------------------------- #

def success_from_sensor(
    sensor_state: Mapping[str, Any] | None,
    *,
    empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
    pos_margin: float = GRIP_POS_MARGIN,
) -> bool | None:
    """用传感器状态判定是否**成功夹持**（设计：传感器优先，VLM 仅兜底）。

    **判据口径（必须与人工复核页一致）**：本函数回答的是「**夹爪是否夹住了物体**」，
    **不包含**"是否抬起"或"是否放到目标位"。批次 22 踩过口径不一致的坑——复核页当时
    问"是否夹住并提起"（更严），而判据只判夹住 → `κ_success` 被口径差异污染。此后
    复核页的问法改为"夹爪是否夹住物体"（见 `review_sheet`）。

    **已知局限**：真机演示数据集只记录关节位置、**没有负载通道**，所以这里只能用
    位置判据（弱于 `holds_object` 的「位置 OR 负载」）。该局限必须写进
    `label_scope`，不能把位置判据的"成功"当成夹持真值。

    Args:
        sensor_state: 需含 ``gripper_pos``（episode 各帧的夹爪位置序列）；
            缺失或为空时返回 ``None``（"未判定"，由审计方区分处理）。
    """
    if not isinstance(sensor_state, Mapping):
        return None
    positions = sensor_state.get("gripper_pos")
    if positions is None:
        return None
    try:
        values = [float(item) for item in positions]
    except (TypeError, ValueError) as exc:
        raise LabelingError(f"gripper_pos 含非数值：{exc}") from exc
    if not values:
        return None
    tail = values[-SUCCESS_TAIL_FRAMES:]
    return any(is_holding(value, empty_grip_baseline, pos_margin) for value in tail)


def build_label_record(
    episode_ref: str,
    payload: Mapping[str, Any],
    taxonomy: IntentTaxonomy,
    sensor_state: Mapping[str, Any] | None = None,
    *,
    default_task: str = "",
    source: str = "vlm",
) -> LabelRecord:
    """把后端返回的原始字段 + 传感器成功判定组装为 `LabelRecord`（纯函数）。

    - `intent` 由 `task_text` 经**确定性规则**映射（不交给 LLM，保证 κ 可归因）；
    - `success` 优先取传感器判定；无传感器数据时为 ``None``（不猜）。
    """
    if not isinstance(payload, Mapping):
        raise LabelingError("后端返回必须是 mapping")
    task_text = str(
        payload.get("task_text") or payload.get("task") or default_task or ""
    ).strip()
    raw_range = payload.get("frame_range")
    frame_range: tuple[int, int] | None = None
    if raw_range:
        try:
            frame_range = (int(raw_range[0]), int(raw_range[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise LabelingError(f"frame_range 非法：{raw_range!r}") from exc
    subtask = payload.get("subtask")
    if subtask is not None:
        subtask = str(subtask)
    return LabelRecord(
        episode_ref=episode_ref,
        task_text=task_text,
        frame_range=frame_range,
        intent=map_instruction_to_intent(task_text, taxonomy),
        subtask=subtask,
        success=success_from_sensor(sensor_state),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        source=source,
        taxonomy_version=taxonomy.version,
        note=str(payload.get("note", "")),
    )


class LabelingBackend(Protocol):
    """VLM（或桩）标注后端：给定若干帧，返回 JSON 化的标注字段。"""

    def label(
        self,
        frames: Sequence[str],
        *,
        task_hint: str | None = None,
        taxonomy: IntentTaxonomy | None = None,
    ) -> Mapping[str, Any]:
        ...


class FakeLabelingBackend:
    """测试用桩后端：按预设脚本依次返回，便于覆盖失败/断点续跑路径。"""

    def __init__(self, responses: Iterable[Mapping[str, Any] | Exception]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def label(
        self,
        frames: Sequence[str],
        *,
        task_hint: str | None = None,
        taxonomy: IntentTaxonomy | None = None,
    ) -> Mapping[str, Any]:
        if self.calls >= len(self._responses):
            raise LabelingError("桩后端脚本已用尽")
        item = self._responses[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return dict(item)


def label_episode(
    backend: LabelingBackend,
    frames: Sequence[str],
    sensor_state: Mapping[str, Any] | None,
    taxonomy: IntentTaxonomy,
    *,
    episode_ref: str = "",
    task_hint: str | None = None,
) -> LabelRecord:
    """标注单条 episode（设计 §5.1）。"""
    payload = backend.label(list(frames), task_hint=task_hint, taxonomy=taxonomy)
    return build_label_record(
        episode_ref,
        payload,
        taxonomy,
        sensor_state,
        default_task=task_hint or "",
    )


# --------------------------------------------------------------------------- #
# 2.2b VLM 标注后端（Qwen3-VL 等）
# --------------------------------------------------------------------------- #

#: 标注用 System Prompt。**不给**模型看数据集自带指令：VLM 流的 intent 要独立推断，
#: 否则 κ 只是"把已知答案抄一遍"，失去审计意义（对照设计 §7.2 的 L0/L1）。
#:
#: 阶段部分**要求输出序列**而非单个标签（批次 23 修正）：原 prompt 直接要"最靠后的主要阶段"，
#: 实测模型退化成整段恒答 `grasp`（人工 recall `lift`=0.00），且与它自己写的
#: task_text（"夹起桌上的螺丝刀"）自相矛盾。改为"先逐帧判断、再给出按时间排序的阶段序列"，
#: `subtask` 由代码取序列**末段**派生——把"判断"和"取末段"解耦，模型只需判断，取舍由代码负责。
VLM_LABEL_SYSTEM_PROMPT = """\
你是机器人示教数据的标注器。你会看到同一段演示的若干帧（按时间顺序，可能来自两个视角：
fixed=工作台俯拍、handeye=夹爪特写）。请输出严格 JSON：

{"task_text": "用一句话描述这段演示在做什么（中文，动词开头）",
 "stages": ["approach"],
 "confidence": 0.0,
 "note": "可选的简短说明"}

stages 是这段演示**按时间顺序实际发生的阶段序列**，每个取值只能是：
approach（只是靠近）、align（对准）、grasp（夹住）、lift（已抬起）、place（在放置）。
只列你确实从画面看到的阶段，个数不限——例如走了完整三步就写 ["approach","grasp","lift"]，
只做了一步就只写一个。**先逐步判断再列序列，不要笼统地只写一个阶段。**
只依据画面判断，不要猜测看不到的信息；不确定就把 confidence 调低。confidence 取 0–1 的小数。
"""


def subtask_from_payload(payload: Mapping[str, Any]) -> tuple[str | None, str]:
    """从后端回复解析「阶段」并取**末段**作为 `subtask`，返回 ``(subtask, note)``。

    优先读 ``stages``（时间序列，取最后一个合法阶段）；模型未给该字段时回退到旧的
    ``subtask`` 单值字段，并记 note（便于事后统计模型有多少次没按新 schema 输出）。
    非法阶段名**丢弃并记录**而不是让整条标注失败——单个坏值不该毁掉一次外网调用。
    """
    note_parts: list[str] = []
    stages = payload.get("stages")
    if isinstance(stages, list):
        cleaned = [str(item).strip().lower() for item in stages]
        kept = [item for item in cleaned if item in SUBTASK_NAMES]
        dropped = [item for item in cleaned if item and item not in SUBTASK_NAMES]
        if dropped:
            note_parts.append(f"stages 非法项 {dropped} 已忽略")
        if kept:
            if dropped:
                note_parts.append(f"有效阶段 {kept}")
            return kept[-1], "；".join(note_parts)
        note_parts.append("stages 中没有可用阶段")
    elif stages is not None:
        note_parts.append("stages 不是数组，已忽略")

    raw = str(payload.get("subtask") or "").strip().lower()
    if raw and raw not in SUBTASK_NAMES:
        note_parts.append(f"subtask 非法值 {raw!r} 已置空")
        return None, "；".join(note_parts)
    if raw:
        note_parts.append("回退 subtask 字段")
        return raw, "；".join(note_parts)
    return None, "；".join(note_parts)


def _default_image_loader(path: str) -> str:
    """把帧文件读成 data URL（交给 `llm_planner.image_to_data_url` 统一压缩/编码）。"""
    from PIL import Image

    from llm_planner import image_to_data_url

    try:
        with Image.open(path) as image:
            return image_to_data_url(image.convert("RGB"))
    except OSError as exc:
        raise LabelingError(f"读取帧失败 {path}: {exc}") from exc


class VlmLabelingBackend:
    """多模态 LLM（Qwen3-VL 等）标注后端：帧序列 → 任务文本 / 阶段 / 置信度。

    依赖注入：``chat_backend`` 只需实现 `llm_planner` 的
    ``chat(system, user, images=...)`` 契约，测试可注入假后端离线覆盖解析分支。

    ``intent`` **不**由本后端给出——它由 `build_label_record` 用确定性规则从
    ``task_text`` 映射（见模块头），保证 κ 的分歧可归因到"VLM 读视频读错了"，
    而不是"映射本身不确定"。

    ``subtask`` 也不再由模型直接给：模型只输出 ``stages`` 序列，末段由
    `subtask_from_payload` 派生（见该函数与 prompt 的说明）。
    """

    def __init__(
        self,
        chat_backend: Any,
        *,
        image_loader: Callable[[str], str] | None = None,
        prompt: str = VLM_LABEL_SYSTEM_PROMPT,
    ) -> None:
        if chat_backend is None:
            raise LabelingError("chat_backend 不能为空")
        self._chat = chat_backend
        self._load_image = image_loader or _default_image_loader
        self._prompt = prompt

    @staticmethod
    def _user_prompt(count: int) -> str:
        return (
            f"这是同一段演示按时间顺序抽取的 {count} 帧（可能含多个视角）。"
            "请按 system 给出的 JSON 格式输出标注。"
        )

    def label(
        self,
        frames: Sequence[str],
        *,
        task_hint: str | None = None,
        taxonomy: IntentTaxonomy | None = None,
    ) -> Mapping[str, Any]:
        if not frames:
            raise LabelingError("frames 为空，无法标注")
        images = [self._load_image(str(path)) for path in frames]
        raw = self._chat.chat(self._prompt, self._user_prompt(len(images)), images=images)
        data = extract_json_object(raw)
        if data is None:
            raise LabelingError(f"VLM 标注回复无法解析为 JSON: {str(raw)[:160]}")

        task_text = str(data.get("task_text") or "").strip()
        if not task_text:
            # 不能静默回退到数据集指令：那会把"模型没读出来"伪装成"读对了"，
            # 直接污染 intent 的 κ。宁可报错让调用方重试/续跑。
            raise LabelingError(f"VLM 未给出 task_text（原文: {str(raw)[:160]}）")

        subtask, stage_note = subtask_from_payload(data)
        note = str(data.get("note") or "").strip()
        if stage_note:
            note = f"{note}（{stage_note}）".strip()
        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        return {
            "task_text": task_text,
            "subtask": subtask,
            "confidence": confidence,
            "note": note,
        }


# --------------------------------------------------------------------------- #
# 2.3 批量标注（断点续跑 + jsonl 落盘）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EpisodeSample:
    """一条待标注 episode 的输入：帧路径 + 传感器状态 + 任务提示。"""

    episode_ref: str
    frames: tuple[str, ...]
    sensor_state: Mapping[str, Any] = field(default_factory=dict)
    task_hint: str | None = None


class EpisodeSource(Protocol):
    """待标注 episode 的来源（真数据集 / 假数据源都实现它）。"""

    def iter_episodes(self) -> Iterator[EpisodeSample]:
        ...


def read_label_records(path: Path) -> list[LabelRecord]:
    """读取 jsonl 标注产物（不存在则返回空表）。"""
    target = Path(path).expanduser()
    if not target.is_file():
        return []
    records: list[LabelRecord] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(LabelRecord.from_dict(json.loads(line)))
    return records


def label_dataset(
    backend: LabelingBackend,
    source: EpisodeSource,
    taxonomy: IntentTaxonomy,
    out_path: Path,
    *,
    limit: int | None = None,
    resume: bool = True,
    echo: bool = True,
) -> list[LabelRecord]:
    """批量标注：**逐条追加落盘** + 断点续跑。

    逐条 flush 而非最后一次性写盘：VLM 调用是外网请求、随时可能中断，逐条落盘
    让"跑到一半失败"不至于丢失已完成的标注（也是 `resume=True` 的前提）。
    """
    target = Path(out_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    records = read_label_records(target) if resume else []
    seen = {item.episode_ref for item in records}
    written = 0
    with target.open("a", encoding="utf-8") as handle:
        for sample in source.iter_episodes():
            if limit is not None and written >= limit:
                break
            if sample.episode_ref in seen:
                continue
            record = label_episode(
                backend,
                sample.frames,
                sample.sensor_state,
                taxonomy,
                episode_ref=sample.episode_ref,
                task_hint=sample.task_hint,
            )
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            seen.add(sample.episode_ref)
            written += 1
            if echo:
                print(
                    f"  [label] {record.episode_ref:16s} intent={record.intent} "
                    f"subtask={record.subtask} success={record.success} "
                    f"conf={record.confidence:.2f}",
                    flush=True,
                )
    return records


# --------------------------------------------------------------------------- #
# 2.4 标注产物 → LeRobotDataset 兼容元数据（不改 fields）
# --------------------------------------------------------------------------- #

def write_lerobot_metadata(
    records: Sequence[LabelRecord],
    out_path: Path,
    *,
    task_text_by_episode: Mapping[str, str] | None = None,
) -> Path:
    """产出 episode 级 sidecar 元数据（task/intent/subtask），**不改数据集 fields**。

    设计明确要求不动 LeRobotDataset 的 fields（否则旧 ACT 训练脚本会因 schema 变化
    全部失效），标注信息以旁挂 jsonl 的形式提供，训练脚本按需 join。
    """
    target = Path(out_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    hints = dict(task_text_by_episode or {})
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "episode_ref": record.episode_ref,
                "task": record.task_text or hints.get(record.episode_ref, ""),
                "intent": record.intent,
                "subtask": record.subtask,
                "success": record.success,
                "source": record.source,
                "taxonomy_version": record.taxonomy_version,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target

