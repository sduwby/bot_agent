"""真机观测收集与 LLM 摘要（SO-101）。

将机械臂关节状态/夹爪夹持状态/最近技能结果整理为喂给 LLM 的中文摘要。
纯逻辑：输入为 dict（由执行器提供），不直接驱动硬件。

另含负载/接触信号（第 38 周 P0）：读 STS3215 `Present_Load` 作接触判定，
与 `is_holding` 的"夹爪位置差"判据互补——位置差对薄片/小物体失效。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


# --- 夹持判定的实测标定值（第 38 周 0.6，SO-101 真机；见 docs/week38/plan.md 批次 9）---
# 空闭基准：3 次连接 × 3 姿态 × 5 采样实测空载闭合 3.921 ± 0.044（上限 4.045）→ 上限向上取整
EMPTY_GRIP_BASELINE = 4.1
# 位置余量：实测薄件（螺丝刀）夹持位 6.8，取 (6.8 − 4.1) 的一半 → 阈值 5.5
# （对空载上限净空 1.455 ≈ 33σ；旧值 margin=3.0 配旧基准 1.5 名义上"很严"，
#   实际对空载上限只有 0.455 净空，是靠基准写小了换来的假余量）
GRIP_POS_MARGIN = 1.4
# 夹持位置**上限**：超过它说明夹爪是"张开"而不是"夹住"。
# 实测依据（2026-10-18 三次真机闭环）：张开 16.9–22.7、夹住螺丝刀 6.7–9.8；
# 取 15.0 留出厚件余量。只给下界的旧判据把"张开 19.25 + 抬起"误判成成功。
GRIP_HOLD_MAX = 15.0


def is_holding(gripper_pos: float, empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
               margin: float = GRIP_POS_MARGIN, hold_max: float = GRIP_HOLD_MAX) -> bool:
    """夹爪位置落在"夹住带"内（空闭基准之上、张开位置之下）。

    **上下界都必要**：只有下界时，夹爪张开（≈17–22）也会被判成夹住 ——
    2026-10-18 真机闭环正是因此把"张开、没夹住、抬起"判成了成功。

    .. note::
       默认值已按 0.6 真机标定重设（baseline 1.5 → 4.1、margin 3.0 → 1.4）。
       位置判据对**薄件**仍可能漏判，故实际判定用 `holds_object`（位置 OR 负载）。
    """
    return empty_grip_baseline + margin < gripper_pos < hold_max


def is_lifted(wrist_flex_pos: float, threshold: float = LIFT_WRIST) -> bool:
    """wrist_flex 是否达到抬起判定。"""
    return wrist_flex_pos >= threshold


# ---------------------------------------------------------------------------
# 负载 / 接触信号（第 38 周 P0）
# ---------------------------------------------------------------------------

# 判定接触的默认负载增量阈值（**解码后**单位：千分比，1000 = 额定扭矩）
LOAD_MARGIN = 50.0

# 空载负载基准：0.6 实测"闭合顶住限位"时 |load| 在 55–69‰（CP0 时 −55‰）。
# 供探针 / `detect_contact` 的增量口径使用。
# **注意**：夹爪张开/空转时负载接近 0‰，故"相对基准做差"不能当夹持判据（会误判），
# 夹持判据请看 `GRIP_LOAD_THRESHOLD`。
EMPTY_LOAD_BASELINE = -69.0

# 夹持的**负载幅值**阈值（‰）。判据是 ``|load| > 该值``，不看基准差值。
# 实测依据：夹爪张开/空转 ≈ 0‰、空载闭合顶限位 55–69‰、抓住螺丝刀 −100‰（薄件）、
# CP0 夹持 −206‰。薄件的 100‰ 与空载的 69‰ 太接近，故阈值取 **150‰ 偏保守**——
# 它只在"明确受力"的夹持上生效（如 CP0 的 206‰），**宁可不触发也不许误判**；
# 薄件由位置判据兜（实测夹持位 6.69 > 阈值 5.5）。
GRIP_LOAD_THRESHOLD = 150.0

# Feetech STS 系列 Present_Load 的 2 字节编码：bit10 为方向位，低 10 位为幅值
LOAD_DIRECTION_BIT = 0x400
LOAD_MAGNITUDE_MASK = 0x3FF


def decode_feetech_load(raw: float) -> float:
    """解码 STS3215 ``Present_Load`` 原始读数为带符号负载。

    原始值**不可直接比较**：bit10 是方向位，低 10 位才是幅值。方向翻转会让原始值
    跳变 1024，直接做差会把"换向"误判成"巨大负载变化"，因此必须先解码。

    Args:
        raw: ``sync_read("Present_Load", normalize=False)`` 的原始读数。

    Returns:
        float: 带符号负载（千分比，绝对值 0–1000）。
    """
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("raw 必须是有限数值")
    encoded = int(round(value))
    if encoded < 0 or encoded > (LOAD_DIRECTION_BIT | LOAD_MAGNITUDE_MASK):
        raise ValueError(f"raw 超出 2 字节 Present_Load 取值范围: {raw}")
    magnitude = encoded & LOAD_MAGNITUDE_MASK
    return float(-magnitude if encoded & LOAD_DIRECTION_BIT else magnitude)


def load_state_names() -> list[str]:
    """与 STATE_NAMES 一一对应的负载键名（``<joint>.load``）。"""
    return [f"{name.rsplit('.', 1)[0]}.load" for name in STATE_NAMES]


@dataclass
class ContactSignal:
    """一次接触/夹持判定结果。"""

    is_contact: bool
    load: float
    baseline: float
    margin: float
    source: str = "present_load"

    @property
    def delta(self) -> float:
        """当前负载相对空载基准的增量。"""
        return self.load - self.baseline


@dataclass
class LoadProbeResult:
    """负载量程探针结论：Present_Load 能否分辨目标物体。"""

    empty_load: float
    loaded_load: float
    margin: float
    noise: float
    threshold: float
    resolvable: bool
    n_empty: int
    n_loaded: int
    note: str = ""

    @property
    def delta(self) -> float:
        """夹持与空载的平均负载差。"""
        return self.loaded_load - self.empty_load


@dataclass(frozen=True)
class EpisodeOutcome:
    """逐集成功判定及其支撑计数（供 A4 与人工标注对照）。"""

    success: bool
    n_frames: int
    held_frames: int
    lifted_frames: int
    holding_and_lifted_frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "n_frames": self.n_frames,
            "held_frames": self.held_frames,
            "lifted_frames": self.lifted_frames,
            "holding_and_lifted_frames": self.holding_and_lifted_frames,
        }


def read_load(engine: Any) -> dict[str, float]:
    """经注入引擎读取各关节负载（Present_Load）。

    引擎未实现 ``read_load``、抛异常或返回非 dict 时一律返回空 dict——
    调用方据此降级，**不得**把"读不到负载"当作"没有接触"。
    """
    reader = getattr(engine, "read_load", None)
    if reader is None:
        return {}
    try:
        raw = reader()
    except Exception:  # 通信/驱动层错误不应中断观测收集
        return {}
    if not isinstance(raw, dict):
        return {}
    values: dict[str, float] = {}
    for key, value in raw.items():
        try:
            values[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def detect_contact(load: float, baseline: float, margin: float = LOAD_MARGIN) -> ContactSignal:
    """按负载**增量幅值**判定是否接触/夹持（严格大于阈值）。

    用 ``|load − baseline|`` 而非带符号比较：STS3215 的负载方向由 bit10 给出，
    而"空载顶住机械限位"与"夹住物体"在真机上是**同向**扭矩、只有幅值不同
    （0.6 实测空载 −55 ~ −69‰、夹持 **−206‰**）。若按带符号比较，
    ``load(−206) > baseline(−69) + 50`` 恒为假 → **必然漏判**。幅值口径也与
    `summarize_load_probe` 的 ``|Δ|`` 判据一致。

    Args:
        load: 当前负载读数。
        baseline: 空载基准读数（由探针实测，默认取 `EMPTY_LOAD_BASELINE`）。
        margin: 判定阈值，必须非负。

    Returns:
        ContactSignal: 判定结果；``delta`` 为带符号的增量（供诊断）。
    """
    if margin < 0:
        raise ValueError("margin 不能为负数")
    load_value = float(load)
    baseline_value = float(baseline)
    if not math.isfinite(load_value) or not math.isfinite(baseline_value):
        raise ValueError("load 与 baseline 必须是有限数值")
    return ContactSignal(
        is_contact=abs(load_value - baseline_value) > margin,
        load=load_value,
        baseline=baseline_value,
        margin=margin,
    )


def holds_object(gripper_pos: float,
                 load: float | None = None,
                 *,
                 empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
                 pos_margin: float = GRIP_POS_MARGIN,
                 load_threshold: float = GRIP_LOAD_THRESHOLD) -> bool:
    """「位置 OR 负载」夹持判据（第 38 周 0.6 建议）。

    两条判据互补：
    - **位置**对极薄物体可能漏判（夹持位只比空载高一点）；
    - **负载**与位置抖动无关，作为"明确受力"的兜底。

    负载判据用**幅值** ``|load| > load_threshold``，**不是**"相对空载基准做差"：
    空载负载会随夹爪状态在 0‰（张开/空转）到 ~69‰（闭合顶限位）之间变化，
    用单一基准做差会在"张开"时把 0‰ 误判成接触（week38 实测踩到的假阳性）。

    ``load=None``（引擎无读负载能力或读取失败）时退化为纯位置判据。
    """
    if is_holding(gripper_pos, empty_grip_baseline, pos_margin):
        return True
    if load is None:
        return False
    return abs(float(load)) > float(load_threshold)


def aggregate_episode_success(
    states: Sequence[Sequence[float] | Mapping[str, float]],
    loads: Sequence[float | None] | None = None,
    *,
    lift_threshold: float = LIFT_WRIST,
    use_load: bool = True,
) -> "EpisodeOutcome":
    """把**逐帧**的夹持/抬起判据聚合成**逐集**结论（A4 的对照缺口）。

    规则与真机闭环一致：**存在某一帧同时满足"夹住"与"已抬起"** 即判该集成功。
    - ``states``：每帧的 6 维关节向量（顺序同 `STATE_NAMES`），或已命名的 dict；
    - ``loads``：每帧夹爪负载（‰），可为 `None`/缺省（此时退化为纯位置判据）；
    - ``use_load=False``：强制只按位置带判据（与 `success_from_sensor` 的纯位置口径对齐）。

    **为什么不直接看最后一帧**：抓取后可能松手/复位，末帧成功不代表整集成功；
    反之末帧失败也不代表过程中没成功。A4 要对照的是"这一集是否完成了任务"。
    """
    rows = list(states)
    if loads is not None and len(loads) != len(rows):
        raise ValueError(f"loads 与 states 帧数不一致：{len(loads)} vs {len(rows)}")
    if not rows:
        raise ValueError("states 不能为空")

    held_frames = 0
    lifted_frames = 0
    joint_frames = 0
    for index, state in enumerate(rows):
        if isinstance(state, Mapping):
            pos = float(state.get("gripper.pos", 0.0))
            wrist = float(state.get("wrist_flex.pos", 0.0))
        else:
            vector = [float(value) for value in state]
            if len(vector) != len(STATE_NAMES):
                raise ValueError(
                    f"第 {index} 帧维度 {len(vector)} 与 STATE_NAMES({len(STATE_NAMES)}) 不符"
                )
            pos = vector[STATE_NAMES.index("gripper.pos")]
            wrist = vector[STATE_NAMES.index("wrist_flex.pos")]

        load = None if loads is None else loads[index]
        held = holds_object(pos, load) if use_load else is_holding(pos)
        lifted = wrist >= lift_threshold
        held_frames += int(held)
        lifted_frames += int(lifted)
        joint_frames += int(held and lifted)

    return EpisodeOutcome(
        success=joint_frames > 0,
        n_frames=len(rows),
        held_frames=held_frames,
        lifted_frames=lifted_frames,
        holding_and_lifted_frames=joint_frames,
    )


def summarize_load_probe(
    empty_samples: Sequence[float],
    loaded_samples: Sequence[float],
    margin: float = LOAD_MARGIN,
) -> LoadProbeResult:
    """由空载/夹持两组采样给出量程探针结论。

    可分辨判据：``|Δ| > max(margin, 2×采样噪声)``，其中噪声取两组样本标准差
    的较大者。该判据同时要求"增量足够大"和"增量显著高于抖动"。
    """
    if margin < 0:
        raise ValueError("margin 不能为负数")
    empty = [float(v) for v in empty_samples]
    loaded = [float(v) for v in loaded_samples]
    if not empty or not loaded:
        raise ValueError("空载与夹持采样都不能为空")
    if not all(math.isfinite(v) for v in empty + loaded):
        raise ValueError("采样值必须是有限数值")

    empty_mean = statistics.fmean(empty)
    loaded_mean = statistics.fmean(loaded)
    noise = max(
        statistics.pstdev(empty) if len(empty) > 1 else 0.0,
        statistics.pstdev(loaded) if len(loaded) > 1 else 0.0,
    )
    threshold = max(float(margin), 2.0 * noise)
    delta = loaded_mean - empty_mean
    resolvable = abs(delta) > threshold
    note = (
        f"空载 {empty_mean:.2f}(n={len(empty)}) / 夹持 {loaded_mean:.2f}(n={len(loaded)}) / "
        f"Δ={delta:+.2f} / 噪声={noise:.2f} / 阈值={threshold:.2f}"
    )
    return LoadProbeResult(
        empty_load=empty_mean,
        loaded_load=loaded_mean,
        margin=float(margin),
        noise=noise,
        threshold=threshold,
        resolvable=resolvable,
        n_empty=len(empty),
        n_loaded=len(loaded),
        note=note,
    )
