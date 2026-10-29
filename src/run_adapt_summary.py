#!/usr/bin/env python
"""P2 4.5：把各适配单元的离线评测结果汇总成 Markdown 表（A5 的离线部分）。

输入（每个单元一个目录）::

    <results-dir>/<unit>/offline_metrics.json   # run_adapt_eval_offline.py 的产物
    <train-stats-dir>/<unit>/adapt_stats.json   # run_vla_adapt_smolvla.py 的冻结统计（可选）

输出：Markdown 表 + 强制携带 `CAVEAT`（离线指标≠真机成功率，A6 必须实测）。

用法::

    python src/run_adapt_summary.py \
        --results-dir runs/week38/vla_adapt_eval \
        --train-stats-dir runs/week38/vla_adapt \
        --out runs/week38/vla_adapt_eval/A5_offline.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from adapt_eval_offline import CAVEAT  # noqa: E402

#: 重复评测目录命名：`<unit>_r1` / `<unit>_r2` …
_REPEAT = re.compile(r"^(?P<base>.+)_r(?P<idx>\d+)$")
_METRIC_KEYS = (
    "n_frames",
    "action_l1",
    "gripper_l1",
    "hold_recall",
    "hold_precision",
    "hold_f1",
)


def load_unit_rows(
    results_dir: str | Path,
    train_stats_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    """读取每个单元的 `offline_metrics.json`（以及可选的冻结统计）。

    Raises:
        FileNotFoundError: 一个单元结果都没有（避免输出空表被误当成"全失败"）。
    """
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"结果目录不存在：{root}")
    stats_root = Path(train_stats_dir) if train_stats_dir else None

    rows: list[dict[str, object]] = []
    for unit_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics_path = unit_dir / "offline_metrics.json"
        if not metrics_path.is_file():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        match = _REPEAT.match(unit_dir.name)
        base_name = match.group("base") if match else unit_dir.name
        stats_path = (
            stats_root / base_name / "adapt_stats.json" if stats_root else None
        )
        ratio = None
        if stats_path is not None and stats_path.is_file():
            ratio = json.loads(stats_path.read_text(encoding="utf-8")).get("trainable_ratio")
        elif base_name.startswith("zero_shot"):
            ratio = 0.0
        rows.append(
            {
                "unit": unit_dir.name,
                "n_frames": metrics.get("n_frames"),
                "action_l1": metrics.get("action_l1"),
                "gripper_l1": metrics.get("gripper_l1"),
                "hold_recall": metrics.get("hold_recall"),
                "hold_precision": metrics.get("hold_precision"),
                "hold_f1": metrics.get("hold_f1"),
                "trainable_ratio": ratio,
            }
        )
    if not rows:
        raise FileNotFoundError(f"{root} 下没有 <unit>/offline_metrics.json")
    return rows


def _fmt(value: object, digits: int = 4, pct: bool = False) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{number:.2%}" if pct else f"{number:.{digits}f}"


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """把 `<unit>_r1/_r2/...` 聚合成一行（均值 + 样本标准差）。

    单次结果保持原样（`repeats=1`、std 记 0），便于与旧表兼容。
    """
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        name = str(row["unit"])
        match = _REPEAT.match(name)
        base = match.group("base") if match else name
        groups.setdefault(base, []).append(row)

    aggregated: list[dict[str, object]] = []
    for base, items in groups.items():
        row: dict[str, object] = {
            "unit": base,
            "repeats": len(items),
            "trainable_ratio": next(
                (item["trainable_ratio"] for item in items
                 if item.get("trainable_ratio") is not None),
                None,
            ),
        }
        for key in _METRIC_KEYS:
            values = [float(item[key]) for item in items if item.get(key) is not None]
            if not values:
                row[key] = None
                row[f"{key}_std"] = None
                continue
            row[key] = statistics.fmean(values)
            row[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregated.append(row)
    return aggregated


def _cell(row: dict[str, object], key: str, digits: int = 4, pct: bool = False) -> str:
    value = row.get(key)
    if value is None:
        return "—"
    text = _fmt(value, digits=digits, pct=pct)
    std = row.get(f"{key}_std")
    if int(row.get("repeats", 1)) > 1 and std is not None:
        text += f" ± {_fmt(std, digits=digits, pct=pct)}"
    return text


def render_table(rows: list[dict[str, object]]) -> str:
    """按 action L1 升序渲染 Markdown 表（L1 缺失的排在最后）。"""
    ordered = sorted(
        rows,
        key=lambda row: (row["action_l1"] is None, float(row["action_l1"] or 0.0)),
    )
    lines = [
        "| 单元 | 重复 | 可训练参数 | 帧数 | action L1 | gripper L1 | 夹住 recall | 夹住 precision | 夹住 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['unit']} | {row.get('repeats', 1)} "
            f"| {_fmt(row['trainable_ratio'], pct=True)} "
            f"| {_fmt(row['n_frames'], 0)} | {_cell(row, 'action_l1')} "
            f"| {_cell(row, 'gripper_l1')} | {_cell(row, 'hold_recall', 3)} "
            f"| {_cell(row, 'hold_precision', 3)} | {_cell(row, 'hold_f1', 3)} |"
        )
    lines.append("")
    lines.append(f"> {CAVEAT}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="4.5 适配单元离线结果汇总")
    parser.add_argument("--results-dir", default="runs/week38/vla_adapt_eval")
    parser.add_argument("--train-stats-dir", default="runs/week38/vla_adapt")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = load_unit_rows(args.results_dir, args.train_stats_dir or None)
    table = render_table(aggregate_rows(rows))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
        print(f"[summary] 已写出 {out}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
