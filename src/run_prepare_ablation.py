#!/usr/bin/env python
"""生成 B0–B3 各变体的数据集（第 38 周 P0 / 1.6 的真跑前置）。

把基础 v3 数据集的 `observation.environment_state`（13 维）扩展成 `13+n`，
额外列按**方案③**取自 `ablation_features`（B1/B2 用 oracle 特权量，纯算术、无需仿真）。

用法::

    # 先看会生成什么（不落盘）
    python src/run_prepare_ablation.py --base runs/maniskill_lerobot/week15-cam `
      --out-root runs/week38/ablation_data --variants B0,B1,B2 --dry-run

    # 真生成（默认硬链接视频，零拷贝）
    python src/run_prepare_ablation.py --base runs/maniskill_lerobot/week15-cam `
      --out-root runs/week38/ablation_data --variants B0,B1,B2

    # 生成完直接用消融 CLI 跑维度门禁
    python src/run_depth_ablation.py --dry-run `
      --dataset B0=runs/week38/ablation_data/B0 --dataset B1=runs/week38/ablation_data/B1

**为什么不生成 B3**：立体深度要两视角像素 + 内参/相对位姿，而数据集只有单相机通道 →
必须重新渲染第二视角；用物体真值位姿反推三角化属循环论证，故直接拒绝（见 `ablation_dataset`）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ablation_dataset import (  # noqa: E402
    AblationDatasetError,
    PrepareReport,
    prepare_variant_dataset,
)
from real_geometry import VARIANT_ORDER  # noqa: E402


def render_reports(reports: list[PrepareReport]) -> str:
    lines = ["变体   ENV维度   帧数   夹爪开合范围            额外列"]
    for item in reports:
        columns = ",".join(item.extra_features) if item.extra_features else "（无，基线）"
        span = f"{item.closed_width:.4f}~{item.open_width:.4f}"
        lines.append(f"{item.variant:<7}{item.env_dim:>7}{item.frames:>8}   {span:<22}{columns}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 B0–B3 变体数据集（ENV 列追加）")
    parser.add_argument("--base", required=True, help="基础 v3 数据集根目录（ENV = 13 维）")
    parser.add_argument("--out-root", required=True, help="输出根目录（每个变体一个子目录）")
    parser.add_argument("--variants", default="B0,B1,B2",
                        help=f"逗号分隔的变体（{VARIANT_ORDER} 的子集；B3 会被拒绝并说明理由）")
    parser.add_argument("--copy", action="store_true",
                        help="复制视频而不硬链接（默认硬链接，零拷贝；跨卷会自动退回复制）")
    parser.add_argument("--dry-run", action="store_true", help="只统计与校验，不落盘")
    args = parser.parse_args()

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not variants:
        parser.error("--variants 不能为空")
    out_root = Path(args.out_root).expanduser()
    reports: list[PrepareReport] = []
    for variant in variants:
        target = out_root / variant
        try:
            report = prepare_variant_dataset(
                args.base, target, variant, hardlink=not args.copy, dry_run=args.dry_run
            )
        except AblationDatasetError as exc:
            print(f"[准备] {variant} 失败：{exc}")
            return 1
        reports.append(report)

    print(render_reports(reports))
    for item in reports:
        for warning in item.warnings:
            print(f"[准备][警告] {item.variant}: {warning}")
    if args.dry_run:
        print("[准备] dry-run 结束（未落盘）")
        return 0

    summary_path = out_root / "prepare_summary.json"
    summary_path.write_text(
        json.dumps({"base": str(Path(args.base).expanduser()), "variants": [r.to_dict() for r in reports]},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[准备] 摘要 → {summary_path}")
    print(
        "[准备] 下一步：python src/run_depth_ablation.py --dry-run "
        + " ".join(f"--dataset {r.variant}={r.out_root}" for r in reports)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
