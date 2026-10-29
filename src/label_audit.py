"""标注一致性审计（第 38 周 P1 / 子任务 2.5–2.6）。

κ 衡量"VLM 自动标注 vs 人工标注"在同一 taxonomy 上的一致性，是 P1 的验收指标 A3
（κ ≥ 0.6）。审计只在**两条独立标注流**之间做：VLM 流的 intent 由确定性规则映射、
人工流由人独立判断，二者不共享同一份推断，κ 才有意义。

**必须报告 `label_scope`**：κ 只在"标签真正有区分度"的数据上才有意义。例如
demo_v23 只有 2 条指令且都映射到 `pick`，intent 一栏必然全同 → κ 退化为 1.0，
这不代表标注质量高，而是**该数据集上 intent 维度没有信息量**。审计报告必须写明
适用边界，否则指标会被误读。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vlm_labeler import SUBTASK_NAMES, LabelRecord, read_label_records

AUDIT_SCHEMA_VERSION = "week38_label_audit_v1"

#: intent 为 None 时的占位类别。κ 需要离散类别，不能把"未映射"的样本丢掉
#: （丢掉会让 κ 只统计"能映射的"样本，系统性高估一致率）。
UNMAPPED = "未映射"


class LabelAuditError(RuntimeError):
    """审计契约错误。"""


def _as_class(value: Any) -> str:
    """把任意标签值规整成离散类别名。"""
    if value is None:
        return UNMAPPED
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's κ = (po − pe) / (1 − pe)。

    - 类别集合取两序列中出现过的**并集**；
    - ``None`` 视作独立类别（`UNMAPPED`），不丢样本；
    - ``pe >= 1``（双方都只出现同一类）时 κ 在数学上无定义 → 完全一致返回 1.0，
      否则返回 0.0（此时 po 必然也是 1，故实际只有"全同"一种情形）。
    """
    if len(a) != len(b):
        raise LabelAuditError(f"两组标注长度不一致：{len(a)} vs {len(b)}")
    if not a:
        raise LabelAuditError("至少需要一个样本才能算 κ")
    left = [_as_class(item) for item in a]
    right = [_as_class(item) for item in b]
    labels = sorted(set(left) | set(right))
    index = {label: position for position, label in enumerate(labels)}
    n = len(left)
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for x, y in zip(left, right):
        matrix[index[x], index[y]] += 1.0
    observed = float(np.trace(matrix)) / n
    expected = float((matrix.sum(axis=1) / n) @ (matrix.sum(axis=0) / n))
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return float((observed - expected) / (1.0 - expected))


@dataclass(frozen=True)
class LabelAuditReport:
    """审计报告（设计 §4.1 的 `LabelAuditReport`，另加 3 个补充字段）。"""

    n_total: int
    kappa_intent: float
    kappa_success: float
    confusion: list[list[int]]
    confusion_labels: list[str]
    disagreements: list[dict[str, Any]]
    per_subtask_recall: dict[str, float]
    label_scope: str
    #: 参与 success κ 的样本数（success 可能为 None，两侧同时有值才算一对）
    n_success_pairs: int = 0
    #: 两侧 taxonomy 版本，不一致时 κ 不可比
    taxonomy_versions: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "n_total": self.n_total,
            "kappa_intent": self.kappa_intent,
            "kappa_success": self.kappa_success,
            "n_success_pairs": self.n_success_pairs,
            "confusion": self.confusion,
            "confusion_labels": self.confusion_labels,
            "disagreements": self.disagreements,
            "per_subtask_recall": self.per_subtask_recall,
            "label_scope": self.label_scope,
            "taxonomy_versions": self.taxonomy_versions,
            "warnings": list(self.warnings),
            "verdict": self.verdict,
        }

    @property
    def verdict(self) -> str:
        """A3 判据：κ_intent ≥ 0.6 视为通过。"""
        return "pass" if self.kappa_intent >= 0.6 else "fail"

    def summary_lines(self) -> list[str]:
        return [
            f"样本数 {self.n_total}（success 可配对 {self.n_success_pairs}）",
            f"κ_intent = {self.kappa_intent:.3f}    κ_success = {self.kappa_success:.3f}",
            f"A3（κ_intent ≥ 0.6）判定：{self.verdict}",
            f"分歧样本 {len(self.disagreements)} 条",
            f"标签适用边界：{self.label_scope}",
        ]


def compare_labels(
    vlm: Sequence[LabelRecord],
    human: Sequence[LabelRecord],
    *,
    label_scope: str = "",
) -> LabelAuditReport:
    """按 `episode_ref` 对齐两条标注流，输出 κ / 混淆矩阵 / 分歧 / per-subtask recall。

    - 只比较两侧**都出现**的 episode（人工抽检通常只覆盖一部分）；
    - `intent` 一律参与（`None` → `UNMAPPED`）；
    - `success` 只在两侧都非 ``None`` 时配对（传感器未判定时不猜）。
    """
    vlm_by_ref = {item.episode_ref: item for item in vlm}
    human_by_ref = {item.episode_ref: item for item in human}
    shared = [ref for ref in human_by_ref if ref in vlm_by_ref]
    if not shared:
        raise LabelAuditError(
            "两条标注流没有共同的 episode_ref：无法审计（请确认抽检样本与自动标注同一批）"
        )
    shared.sort()

    warnings: list[str] = []
    versions = {
        "vlm": next((vlm_by_ref[ref].taxonomy_version for ref in shared), ""),
        "human": next((human_by_ref[ref].taxonomy_version for ref in shared), ""),
    }
    if versions["vlm"] != versions["human"]:
        warnings.append(
            f"两侧 taxonomy 版本不一致（{versions}）：κ 不可跨版本比较，请统一后重跑"
        )

    intent_vlm = [vlm_by_ref[ref].intent for ref in shared]
    intent_human = [human_by_ref[ref].intent for ref in shared]
    kappa_intent = cohen_kappa(intent_vlm, intent_human)
    confusion, labels = _confusion_matrix(intent_vlm, intent_human)

    success_pairs = [
        (vlm_by_ref[ref].success, human_by_ref[ref].success)
        for ref in shared
        if vlm_by_ref[ref].success is not None and human_by_ref[ref].success is not None
    ]
    if success_pairs:
        kappa_success = cohen_kappa(
            [item[0] for item in success_pairs], [item[1] for item in success_pairs]
        )
    else:
        kappa_success = 0.0
        warnings.append("没有任何 episode 在两侧同时有 success 判定：κ_success 记 0 并标注")

    # per-subtask recall：以**人工标注**为参照，各 subtask 被 VLM 正确认出的比例
    per_subtask: dict[str, float] = {}
    for subtask in SUBTASK_NAMES:
        total = sum(1 for ref in shared if human_by_ref[ref].subtask == subtask)
        if total == 0:
            continue
        hit = sum(
            1
            for ref in shared
            if human_by_ref[ref].subtask == subtask
            and vlm_by_ref[ref].subtask == subtask
        )
        per_subtask[subtask] = hit / total

    disagreements: list[dict[str, Any]] = []
    for ref in shared:
        left, right = vlm_by_ref[ref], human_by_ref[ref]
        diff: dict[str, Any] = {}
        for field_name in ("intent", "subtask", "success"):
            a, b = getattr(left, field_name), getattr(right, field_name)
            if _as_class(a) != _as_class(b):
                diff[field_name] = {"vlm": a, "human": b}
        if diff:
            disagreements.append(
                {
                    "episode_ref": ref,
                    "fields": diff,
                    "vlm_confidence": left.confidence,
                    "vlm_task_text": left.task_text,
                    "human_task_text": right.task_text,
                }
            )
    disagreements.sort(
        key=lambda item: (item["vlm_confidence"], item["episode_ref"])
    )  # 低置信度的分歧优先复核

    scope = label_scope or (
        "未声明适用边界：κ 只在标签具有区分度的数据上可解释，请显式说明"
    )
    return LabelAuditReport(
        n_total=len(shared),
        kappa_intent=kappa_intent,
        kappa_success=kappa_success,
        confusion=confusion,
        confusion_labels=labels,
        disagreements=disagreements,
        per_subtask_recall=per_subtask,
        label_scope=scope,
        n_success_pairs=len(success_pairs),
        taxonomy_versions=versions,
        warnings=tuple(warnings),
    )


def default_label_scope(records: Sequence[LabelRecord], *, dataset: str = "") -> str:
    """由标注产物自动生成 `label_scope` 说明（设计要求：κ 必须附适用边界）。

    只在无法人工指定时兜底；核心是把"这份数据上 κ 能不能被解释"写清楚：
    intent 维度退化（类别数 ≤ 1）时 κ_intent ≈ 1 只是没有区分度，不代表标注质量高。
    """
    intents = sorted({_as_class(record.intent) for record in records})
    parts: list[str] = [f"数据集 {dataset}" if dataset else "数据集未指定"]
    parts.append(f"自动标注 {len(records)} 集")
    if len(intents) <= 1:
        parts.append(
            f"intent 维度退化（仅 {intents or ['无']}）：κ_intent 不可解释为标注质量"
        )
    else:
        parts.append(f"intent 覆盖 {len(intents)} 类")
    parts.append("success 由夹爪位置判据给出（数据集无负载通道），薄件可能漏判，需人工抽检确认")
    return "；".join(parts)


def _confusion_matrix(
    a: Sequence[Any], b: Sequence[Any]
) -> tuple[list[list[int]], list[str]]:
    """行=第一条流（vlm）、列=第二条流（human）的混淆矩阵。"""
    left = [_as_class(item) for item in a]
    right = [_as_class(item) for item in b]
    labels = sorted(set(left) | set(right))
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for x, y in zip(left, right):
        matrix[index[x]][index[y]] += 1
    return matrix, labels


def write_audit_report(report: LabelAuditReport, out_dir: Path) -> tuple[Path, Path]:
    """落盘 `audit.json` + `disagreements.jsonl`（分歧样本供人工复核）。"""
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "audit.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    disagreements_path = directory / "disagreements.jsonl"
    with disagreements_path.open("w", encoding="utf-8") as handle:
        for item in report.disagreements:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return json_path, disagreements_path
