#!/usr/bin/env python
"""A3/A4 机读侧汇总 CLI（第 38 周 P1 收尾）。

读 6 份 `labels.jsonl` + 旁挂负载文件，产出 `summary.json` + `summary.md`：

    python src/run_vla_label_summary.py
    python src/run_vla_label_summary.py --label-root runs/week38/label_vla \
        --out runs/week38/label_vla/summary
    python src/run_vla_label_summary.py --dry-run   # 只打印，不写盘

A4 口径对齐（旁挂负载 → 逐集成功）**如实标注为无法完成**：仓库内没有可直接复用的
"逐集成功"函数，故不产出混淆矩阵，只说明缺什么（见 `vla_label_summary.alignment_section`）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from vla_label_summary import (  # noqa: E402
    DEFAULT_LABEL_ROOT,
    DEFAULT_SIDECAR_DIR,
    build_summary,
    load_records_by_repo,
    write_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="A3/A4 机读侧汇总（P1）")
    parser.add_argument("--label-root", default=DEFAULT_LABEL_ROOT,
                        help=f"标注根目录（默认 {DEFAULT_LABEL_ROOT}）")
    parser.add_argument("--sidecar-dir", default=DEFAULT_SIDECAR_DIR,
                        help=f"旁挂负载目录（默认 {DEFAULT_SIDECAR_DIR}）")
    parser.add_argument("--out", default="",
                        help="输出目录（默认 <label-root>/summary）")
    parser.add_argument("--repos", default="",
                        help="逗号分隔的 repo 名；缺省则自动发现（跳过 dry-run 桩产物）")
    parser.add_argument("--dry-run", action="store_true", help="只打印摘要，不写文件")
    args = parser.parse_args()

    label_root = Path(args.label_root).expanduser()
    repos = [item.strip() for item in args.repos.split(",") if item.strip()] or None
    records_by_repo = load_records_by_repo(label_root, repos)
    if not records_by_repo:
        raise SystemExit(f"未在 {label_root} 下找到任何 labels.jsonl")

    summary = build_summary(records_by_repo, sidecar_dir=args.sidecar_dir)

    disc = summary["a3_intent"]["discriminability"]
    print(f"[汇总] repo {summary['n_repos']} 个｜标注 {summary['n_records']} 条")
    print(f"[A3] 预测类别分布 {json.dumps(disc['predicted_class_distribution'], ensure_ascii=False)}"
          f"｜期望 {json.dumps(disc['expected_class_distribution'], ensure_ascii=False)}")
    print(f"[A3] place 被预测 {disc['place_predicted_times']} 次"
          f"（期望 {disc['place_expected_count']}）"
          f"｜κ={disc['kappa_vs_repo_semantics']:.4f} 退化={disc['kappa_degenerate']}")
    print(f"[A4] success 分布 "
          f"{json.dumps(summary['a4_success']['success_distribution'], ensure_ascii=False)}")
    align = summary["a4_alignment"]
    print(f"[A4-对齐] 状态={align['status']}（{align['reason'][:40]}…）")

    if args.dry_run:
        print("[汇总] --dry-run：未写盘")
        return 0

    out_dir = Path(args.out).expanduser() if args.out else label_root / "summary"
    json_path, md_path = write_summary(summary, out_dir)
    print(f"\n[汇总] → {json_path}")
    print(f"[汇总] → {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
