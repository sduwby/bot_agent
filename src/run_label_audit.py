#!/usr/bin/env python
"""标注一致性审计 CLI（第 38 周 P1 / 子任务 2.8）。

对 `run_vlm_label.py` 的自动标注与人工抽检（2.10，由复核页导出 `human.jsonl`）
计算 Cohen's κ、混淆矩阵、per-subtask recall，并导出分歧样本供复核。

用法::

    python src/run_label_audit.py --vlm runs/week38/labels/demo_v23 --human <目录>/human.jsonl
    python src/run_label_audit.py --vlm <labels.jsonl> --human human.jsonl --scope "..."

产物：`audit.json`（含 verdict / label_scope / warnings）+ `disagreements.jsonl`（按 VLM
置信度升序，低置信度的分歧优先复核）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from label_audit import (  # noqa: E402
    LabelAuditError,
    compare_labels,
    default_label_scope,
    write_audit_report,
)
from vlm_labeler import read_label_records  # noqa: E402

LABELS_NAME = "labels.jsonl"


def resolve_labels_path(value: str) -> Path:
    """`--vlm` 允许给目录（读 <dir>/labels.jsonl）或直接的 jsonl 文件。"""
    path = Path(value).expanduser()
    if path.is_dir():
        candidate = path / LABELS_NAME
        if not candidate.is_file():
            raise SystemExit(f"{path} 下没有 {LABELS_NAME}（请检查 --vlm 目录）")
        return candidate
    if not path.is_file():
        raise SystemExit(f"标注文件不存在: {path}")
    return path


def render_confusion(report) -> str:
    labels = report.confusion_labels
    if not labels:
        return "（无混淆矩阵）"
    width = max(len(label) for label in labels) + 2
    header = "vlm\\human".ljust(width) + "".join(label.rjust(max(width, 6)) for label in labels)
    lines = [header]
    for label, row in zip(labels, report.confusion):
        lines.append(label.ljust(width) + "".join(str(v).rjust(max(width, 6)) for v in row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="标注一致性审计（P1）")
    parser.add_argument("--vlm", required=True, help="自动标注目录或 labels.jsonl")
    parser.add_argument("--human", required=True, help="人工抽检 jsonl（复核页导出）")
    parser.add_argument("--out", default="", help="输出目录，默认 <自动标注目录>/audit")
    parser.add_argument("--dataset", default="", help="写入 label_scope 的数据集名")
    parser.add_argument("--scope", default="", help="显式指定 label_scope（覆盖自动说明）")
    args = parser.parse_args()

    vlm_path = resolve_labels_path(args.vlm)
    human_path = Path(args.human).expanduser()
    vlm = read_label_records(vlm_path)
    human = read_label_records(human_path)
    if not vlm:
        raise SystemExit(f"自动标注为空: {vlm_path}")
    if not human:
        raise SystemExit(f"人工标注为空: {human_path}")

    scope = args.scope or default_label_scope(vlm, dataset=args.dataset or vlm_path.parent.name)
    try:
        report = compare_labels(vlm, human, label_scope=scope)
    except LabelAuditError as exc:
        raise SystemExit(f"审计失败: {exc}") from exc

    out_dir = Path(args.out).expanduser() if args.out else vlm_path.parent / "audit"
    json_path, disagreements_path = write_audit_report(report, out_dir)

    print(f"[审计] 自动 {len(vlm)} 条 / 人工 {len(human)} 条 → 共同 {report.n_total} 条")
    for line in report.summary_lines():
        print(f"  {line}")
    print("\n混淆矩阵（行=自动 / 列=人工）：")
    print(render_confusion(report))
    if report.per_subtask_recall:
        recalls = "  ".join(f"{k}={v:.2f}" for k, v in sorted(report.per_subtask_recall.items()))
        print(f"\nper-subtask recall（以人工为参照）：{recalls}")
    for warning in report.warnings:
        print(f"  [警告] {warning}")
    print(f"\n报告 → {json_path}")
    print(f"分歧样本 → {disagreements_path}（{len(report.disagreements)} 条，低置信度优先）")


if __name__ == "__main__":
    main()
