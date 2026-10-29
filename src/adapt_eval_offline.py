"""P2 4.5：适配 checkpoint 的**离线**评测指标（纯逻辑，可离线测试）。

真机成功率（A6）才是最终判据；这里给的是**训练侧代理指标**，用于：

- 训练一结束就能横向比较 5 个单元，挑出值得上真机的 1–2 个；
- 发现"loss 降了但夹爪行为没学会"这类退化（夹爪夹住带命中率）。

代理指标**不能替代 A6**，报告里必须显式标注这一点。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from real_observation import GRIP_HOLD_MAX, EMPTY_GRIP_BASELINE, GRIP_POS_MARGIN, STATE_NAMES

#: 夹爪"夹住带"（与真机判据同源，避免两套口径）。
HOLD_BAND = (EMPTY_GRIP_BASELINE + GRIP_POS_MARGIN, GRIP_HOLD_MAX)
#: 夹爪在状态向量里的下标。
GRIPPER_INDEX = STATE_NAMES.index("gripper.pos")

CAVEAT = (
    "离线指标只反映示范数据上的动作拟合质量，不等于真机成功率；"
    "A6（相对 zero_shot 提升 ≥20pp）必须用真机闭环实测。"
)


@dataclass(frozen=True)
class OfflineMetrics:
    """一组 (预测, 真值) 的离线指标。"""

    n_frames: int
    action_l1: float
    gripper_l1: float
    hold_recall: float
    hold_precision: float
    hold_f1: float

    def to_dict(self) -> dict[str, object]:
        return {
            "n_frames": self.n_frames,
            "action_l1": self.action_l1,
            "gripper_l1": self.gripper_l1,
            "hold_recall": self.hold_recall,
            "hold_precision": self.hold_precision,
            "hold_f1": self.hold_f1,
            "caveat": CAVEAT,
        }


def _in_band(values: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    return (values > low) & (values < high)


def compute_offline_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    band: tuple[float, float] = HOLD_BAND,
) -> OfflineMetrics:
    """计算 action L1 与夹爪"夹住带"的分类指标。

    Args:
        predicted: ``(N, 6)`` 预测动作。
        truth: ``(N, 6)`` 示范动作。
        band: 夹住带区间（默认与真机 `is_holding` 同源）。

    Raises:
        ValueError: 形状不一致或帧数为 0。

    ``hold_f1`` 在分母为 0 时记 0.0（而不是 NaN），便于表格排序。
    """
    pred = np.asarray(predicted, dtype=float)
    gt = np.asarray(truth, dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"预测/真值形状不一致：{pred.shape} vs {gt.shape}")
    if pred.ndim != 2 or pred.shape[1] != len(STATE_NAMES):
        raise ValueError(f"期望 (N, {len(STATE_NAMES)})，实际 {pred.shape}")
    if pred.shape[0] == 0:
        raise ValueError("没有可评测帧")

    action_l1 = float(np.abs(pred - gt).mean())
    gripper_l1 = float(np.abs(pred[:, GRIPPER_INDEX] - gt[:, GRIPPER_INDEX]).mean())

    pred_hold = _in_band(pred[:, GRIPPER_INDEX], band)
    gt_hold = _in_band(gt[:, GRIPPER_INDEX], band)
    tp = int(np.sum(pred_hold & gt_hold))
    n_pred = int(pred_hold.sum())
    n_gt = int(gt_hold.sum())
    recall = tp / n_gt if n_gt else 0.0
    precision = tp / n_pred if n_pred else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0

    return OfflineMetrics(
        n_frames=int(pred.shape[0]),
        action_l1=action_l1,
        gripper_l1=gripper_l1,
        hold_recall=float(recall),
        hold_precision=float(precision),
        hold_f1=float(f1),
    )


def select_frame_ids(
    total: int,
    *,
    stride: int = 5,
    offset: int = 0,
    max_frames: int = 120,
    even: bool = False,
) -> list[int]:
    """选取评测帧下标。

    - 默认（``even=False``）沿用 A5 口径：从 ``offset`` 起每 ``stride`` 帧取一帧，
      最多 ``max_frames`` 帧。
    - ``even=True`` 时在 **[0, total-1] 全区间**均匀取 ``max_frames`` 帧（含首尾）。

    为什么需要 ``even``：A7 的 held-out 组合包含**整段任务**（几十集），若用 stride，
    前 ``max_frames`` 帧会全部落在最初几集里；而且夹住带在整集里只占约 1%–60%
    （实测中位数 5.8%），采样太稀会让 F1 退化（某集抽到的真值阳性为 0 → F1 恒 0）。
    """
    total = int(total)
    if total <= 0:
        raise ValueError(f"数据集帧数必须为正，实际 {total}")
    max_frames = max(1, int(max_frames))
    if even:
        count = min(max_frames, total)
        if count == 1:
            return [0]
        step = (total - 1) / (count - 1)
        return [int(round(index * step)) for index in range(count)]
    step = max(1, int(stride))
    start = int(offset)
    if start < 0 or start >= total:
        raise ValueError(f"offset 必须在 [0, {total}) 内，实际 {start}")
    return list(range(start, total, step))[:max_frames]


def parse_episode_list(raw: str) -> list[int]:
    """把 ``"[0,4,8]"`` / ``"0,4,8"`` 解析为去重、非负的 episode 列表。"""
    text = str(raw).strip().strip("[]")
    if not text:
        raise ValueError("episode 列表不能为空")
    out: list[int] = []
    for item in text.split(","):
        value = item.strip()
        if not value:
            continue
        index = int(value)
        if index < 0:
            raise ValueError(f"episode 必须是非负整数，实际 {index}")
        if index not in out:
            out.append(index)
    if not out:
        raise ValueError("episode 列表不能为空")
    return out


def metrics_table(results: Mapping[str, OfflineMetrics]) -> str:
    """把多个单元的结果渲染成 Markdown 表（按 action_l1 升序）。"""
    if not results:
        return "（无结果）\n"
    rows = sorted(results.items(), key=lambda item: item[1].action_l1)
    lines = [
        "| 单元 | 帧数 | action L1 | gripper L1 | 夹住 recall | 夹住 precision | 夹住 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m.n_frames} | {m.action_l1:.4f} | {m.gripper_l1:.4f} "
            f"| {m.hold_recall:.3f} | {m.hold_precision:.3f} | {m.hold_f1:.3f} |"
        )
    lines.append("")
    lines.append(f"> {CAVEAT}")
    return "\n".join(lines) + "\n"


def evaluate_pairs(pairs: Sequence[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, OfflineMetrics]:
    """对若干 ``(单元名, 预测, 真值)`` 批量计算指标。"""
    if not pairs:
        raise ValueError("pairs 不能为空")
    return {name: compute_offline_metrics(pred, gt) for name, pred, gt in pairs}
