"""逐集成功判据（第 38 周 A4 口径对照）。

## 为什么需要（背景，见 `vlm_labeler.success_from_sensor` 的已知局限）

现有标注链路的 `success` 来自 `success_from_sensor`——它只看**末尾若干帧的夹爪位置**
（`is_holding`），是**纯位置、集末、是否夹住**的口径。真机数据上这会系统性漏判：

- `place` 任务的正确终态是"**已松手**"，集末夹爪是**张开**的 → 纯位置判据一律判 False；
- 薄件（螺丝刀/黑色长条）夹持位只比空闭基准高一点 → 位置判据漏判。

而第 38 周 0.6 已实测标定出「位置 OR 负载」判据（`real_observation.holds_object`），
且采集阶段已用 `load_sidecar` **旁挂**录下负载通道（数据集本身不含负载）。本模块把这些
**逐帧**原语聚合成**逐集**成功，供 A4 机读口径对照使用。

## 三个可分开口径（便于对照，不做"一个数字盖棺"）

1. `position_only`：复刻现状口径（尾帧 `is_holding`，直接复用 `success_from_sensor`）；
2. `position_or_load`：尾帧 `holds_object`（位置 OR 负载，`|load| > GRIP_LOAD_THRESHOLD`）；
3. `task_aware`：任务感知——
   - `pick`：集内**曾**"夹住且抬起"（复用 `run_vla_infer.holding_and_lifted` 逐帧扫）；
   - `place`：在 pick 达成的**前提**下，集末**已松手**（夹爪张开且负载回落）。
     **"放到左边"不判**：数据只有 tcp/关节与夹爪，没有物体位置跟踪 →
     `left_verified` 恒为 False，绝不能拿夹爪位置冒充物体位置。

## 设计约束

- 判据函数**全部是纯函数**（序列进、布尔/结构出），便于离线单测；
- **不新增阈值魔法数**：一切阈值复用 `real_observation` 的实测标定常量
  （`EMPTY_GRIP_BASELINE` / `GRIP_POS_MARGIN` / `GRIP_LOAD_THRESHOLD` / `LIFT_WRIST`）；
- 任务类型从**指令文本**经 `vlm_labeler.map_instruction_to_intent` 映射，不另写关键词表。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from real_observation import (
    EMPTY_GRIP_BASELINE,
    GRIP_LOAD_THRESHOLD,
    GRIP_POS_MARGIN,
    holds_object,
    is_lifted,
)
from run_vla_infer import holding_and_lifted
from vlm_labeler import (
    SUCCESS_TAIL_FRAMES,
    IntentTaxonomy,
    define_intent_taxonomy,
    map_instruction_to_intent,
    success_from_sensor,
)

#: 三个口径名，供报告/CLI 统一引用（顺序即报告展示顺序）。
CRITERIA = ("position_only", "position_or_load", "task_aware")


class EpisodeSuccessError(RuntimeError):
    """逐集成功判据的契约错误。"""


@dataclass(frozen=True)
class EpisodeChannels:
    """单集**逐帧**通道序列（判据的输入；全部为等长序列或 ``None``）。

    通道命名对齐仓库既有口径，而不是照抄"tcp_pos+gripper_qpos"：

    - ``gripper_pos``：`observation.state` 的 ``gripper.pos``（夹爪位置，判定"是否夹住"）；
    - ``wrist_flex_pos``：``wrist_flex.pos``（`holding_and_lifted` 用的"抬起"通道）；
    - ``load``：旁挂 ``gripper.load`` 解码值（‰，可含 ``None`` 表示该帧未读到）。

    ``load`` 允许比 ``gripper_pos`` 短（缺帧）；越界下标按"缺"处理。
    """

    gripper_pos: tuple[float, ...]
    wrist_flex_pos: tuple[float, ...] | None = None
    load: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("gripper_pos", "wrist_flex_pos"):
            values = getattr(self, name)
            if values is None:
                continue
            if len(values) == 0:
                raise EpisodeSuccessError(f"{name} 不能是空序列（应给 None 表示缺失）")


def _load_at(load: Sequence[float | None] | None, index: int) -> float | None:
    """取某帧负载；``load`` 缺失或下标越界时返回 ``None``（由位置判据兜底）。"""
    if load is None or index < 0 or index >= len(load):
        return None
    value = load[index]
    return None if value is None else float(value)


def episode_position_only(
    channels: EpisodeChannels,
    *,
    empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
    pos_margin: float = GRIP_POS_MARGIN,
) -> bool | None:
    """口径 1：**复刻现状**——末尾 ``SUCCESS_TAIL_FRAMES`` 帧内任一帧 `is_holding`。

    直接复用 `vlm_labeler.success_from_sensor`（不重写），保证与 `labels.jsonl` 的
    `success` 字段**同源**，对照才有意义。空序列返回 ``None``（未判定）。
    """
    if not channels.gripper_pos:
        return None
    return success_from_sensor(
        {"gripper_pos": list(channels.gripper_pos)},
        empty_grip_baseline=empty_grip_baseline,
        pos_margin=pos_margin,
    )


def episode_position_or_load(
    channels: EpisodeChannels,
    *,
    empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
    pos_margin: float = GRIP_POS_MARGIN,
    load_threshold: float = GRIP_LOAD_THRESHOLD,
    tail: int = SUCCESS_TAIL_FRAMES,
) -> bool | None:
    """口径 2：末尾 ``tail`` 帧内任一帧 `holds_object`（**位置 OR 负载**）。

    与口径 1 只差"是否把负载纳入判据"，聚合方式（尾帧、any）保持一致，便于对照
    "负载通道到底有没有改变结论"。阈值复用 `GRIP_LOAD_THRESHOLD`（150‰）。
    """
    grip = channels.gripper_pos
    if not grip:
        return None
    start = max(0, len(grip) - int(tail))
    for index in range(start, len(grip)):
        if holds_object(
            grip[index],
            _load_at(channels.load, index),
            empty_grip_baseline=empty_grip_baseline,
            pos_margin=pos_margin,
            load_threshold=load_threshold,
        ):
            return True
    return False


def episode_held_any(
    channels: EpisodeChannels,
    *,
    empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
    pos_margin: float = GRIP_POS_MARGIN,
    load_threshold: float = GRIP_LOAD_THRESHOLD,
) -> bool | None:
    """诊断用：**整集**是否存在任一帧"夹住"（位置 OR 负载）。"""
    if not channels.gripper_pos:
        return None
    return any(
        holds_object(
            value,
            _load_at(channels.load, index),
            empty_grip_baseline=empty_grip_baseline,
            pos_margin=pos_margin,
            load_threshold=load_threshold,
        )
        for index, value in enumerate(channels.gripper_pos)
    )


def episode_lifted_any(
    channels: EpisodeChannels, *, lift_threshold: float | None = None
) -> bool | None:
    """诊断用：**整集**是否存在任一帧"抬起"。阈值缺省用 `real_observation.LIFT_WRIST`。"""
    wrist = channels.wrist_flex_pos
    if wrist is None:
        return None
    if lift_threshold is None:
        return any(is_lifted(value) for value in wrist)
    return any(is_lifted(value, lift_threshold) for value in wrist)


def episode_held_and_lifted(channels: EpisodeChannels) -> bool | None:
    """口径 3 的 pick 子判据：集内**曾有某一帧**同时"夹住且抬起"。

    逐帧复用 `run_vla_infer.holding_and_lifted`（单帧判据），"曾出现"= 逐帧扫到即真。
    这既覆盖了"先夹住、后抬起"，也避免自行规定"抬起的帧必须晚于夹住的帧"这种
    仓库里没有依据的时序约束。``wrist_flex_pos`` 缺失时返回 ``None``。
    """
    grip = channels.gripper_pos
    wrist = channels.wrist_flex_pos
    if not grip or wrist is None:
        return None
    for index in range(min(len(grip), len(wrist))):
        state = {"gripper.pos": grip[index], "wrist_flex.pos": wrist[index]}
        if holding_and_lifted(state, _load_at(channels.load, index)):
            return True
    return False


def episode_released_at_end(
    channels: EpisodeChannels,
    *,
    empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
    pos_margin: float = GRIP_POS_MARGIN,
    load_threshold: float = GRIP_LOAD_THRESHOLD,
) -> bool | None:
    """口径 3 的 place 子判据：集末（**末帧**）已松手。

    "已松手 = 夹爪张开 **且** 负载回落到空载附近"，两者合起来恰好等价于
    ``not holds_object``（`holds_object` 为「位置 OR 负载」，取反即"两条都不触发"）。
    复用该原语，而非另设"空载附近"的基准差值阈值（`EMPTY_LOAD_BASELINE` 的口径
    明确写了不可当夹持判据）。
    """
    grip = channels.gripper_pos
    if not grip:
        return None
    last = len(grip) - 1
    return not holds_object(
        grip[last],
        _load_at(channels.load, last),
        empty_grip_baseline=empty_grip_baseline,
        pos_margin=pos_margin,
        load_threshold=load_threshold,
    )


@dataclass(frozen=True)
class EpisodeSuccess:
    """单集的三种口径结果 + 诊断量。``left_verified`` 恒 False（无物体位置跟踪）。"""

    episode_ref: str
    task: str | None
    position_only: bool | None
    position_or_load: bool | None
    task_aware: bool | None
    pick_achieved: bool | None
    released_at_end: bool | None
    held_any: bool | None
    lifted_any: bool | None
    left_verified: bool = False
    note: str = ""

    def value(self, criterion: str) -> bool | None:
        """按口径名取结果（供报告/对照统一取值）。"""
        if criterion not in CRITERIA:
            raise EpisodeSuccessError(f"未知口径 {criterion!r}，可选 {CRITERIA}")
        return getattr(self, criterion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_ref": self.episode_ref,
            "task": self.task,
            "position_only": self.position_only,
            "position_or_load": self.position_or_load,
            "task_aware": self.task_aware,
            "pick_achieved": self.pick_achieved,
            "released_at_end": self.released_at_end,
            "held_any": self.held_any,
            "lifted_any": self.lifted_any,
            "left_verified": self.left_verified,
            "note": self.note,
        }


def task_from_instruction(
    instruction: str, taxonomy: IntentTaxonomy | None = None
) -> str | None:
    """由指令文本映射任务类型（复用 `vlm_labeler` 的确定性意图映射）。"""
    if not instruction:
        return None
    return map_instruction_to_intent(instruction, taxonomy or define_intent_taxonomy())


def assess_episode(
    episode_ref: str,
    task: str | None,
    channels: EpisodeChannels,
    *,
    empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
    pos_margin: float = GRIP_POS_MARGIN,
    load_threshold: float = GRIP_LOAD_THRESHOLD,
) -> EpisodeSuccess:
    """算出一集在三种口径下的成功（任务感知口径按 ``task`` 分派）。

    - ``pick``：`episode_held_and_lifted`；
    - ``place``：`episode_held_and_lifted` **且** `episode_released_at_end`
      （pick 未达成则 place 必然不成立）；
    - 其它/未知任务：任务感知口径记 ``None``（不猜），但前两种口径照常给出。

    ``left_verified`` 恒 False：本数据无物体位置跟踪，"物体是否到左侧"不可判。
    """
    holding_kwargs = {
        "empty_grip_baseline": empty_grip_baseline,
        "pos_margin": pos_margin,
        "load_threshold": load_threshold,
    }
    position_kwargs = {
        "empty_grip_baseline": empty_grip_baseline,
        "pos_margin": pos_margin,
    }
    position_only = episode_position_only(channels, **position_kwargs)
    position_or_load = episode_position_or_load(channels, **holding_kwargs)
    held_any = episode_held_any(channels, **holding_kwargs)
    lifted_any = episode_lifted_any(channels)
    pick_achieved = episode_held_and_lifted(channels)
    released = (
        episode_released_at_end(channels, **holding_kwargs) if task == "place" else None
    )

    if task == "pick":
        task_aware = pick_achieved
        note = ""
    elif task == "place":
        if pick_achieved is None or released is None:
            task_aware = None
        else:
            task_aware = bool(pick_achieved and released)
        note = "place = 先达成 pick（夹住并抬起）且集末已松手；不判“左边”"
    else:
        task_aware = None
        note = f"任务类型 {task!r} 不在 pick/place 内，任务感知口径不适用"

    return EpisodeSuccess(
        episode_ref=episode_ref,
        task=task,
        position_only=position_only,
        position_or_load=position_or_load,
        task_aware=task_aware,
        pick_achieved=pick_achieved,
        released_at_end=released,
        held_any=held_any,
        lifted_any=lifted_any,
        left_verified=False,
        note=note,
    )


def divergent_criteria(result: EpisodeSuccess) -> tuple[str, ...]:
    """列出该集在哪些口径上取值不一致（用于人工复核定位）。"""
    values: Mapping[str, bool | None] = {
        name: result.value(name) for name in CRITERIA
    }
    if any(value is None for value in values.values()):
        # 有口径未判定：把"未判定"也视作一种取值参与比较
        pass
    distinct = {repr(value) for value in values.values()}
    return CRITERIA if len(distinct) > 1 else ()
