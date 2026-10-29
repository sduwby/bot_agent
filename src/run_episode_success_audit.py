#!/usr/bin/env python
"""A4 复评：用**旁挂负载 + 关节位置**把逐帧判据聚成逐集成功，再与 VLM 标签对照。

背景（`vla_label_summary` 的已知缺口）：仓库原来只有**逐帧**判据，而 `labels.jsonl` 的
`success` 是逐集标签且来自**纯位置**的 `success_from_sensor` → 直接自比无意义。
本次用 `real_observation.aggregate_episode_success`：
- 逐集规则：**存在某帧同时"夹住 + 已抬起"**；
- 两种口径：`use_load=True`（位置 OR 负载）与 `use_load=False`（纯位置）；
- 与 VLM 的逐集 `success` 算 κ、混淆矩阵与准确率。

数据来源：旁挂 `positions` 与 `load.gripper.load`（采集时逐帧校验过对齐，不依赖数据集格式）。

用法::

    python src/run_episode_success_audit.py \
        --labels-root runs/week38/label_vla \
        --sidecar-root runs/week38/collect/sidecars \
        --out runs/week38/label_vla/episode_success_audit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from label_audit import cohen_kappa  # noqa: E402
from real_observation import EpisodeOutcome, aggregate_episode_success  # noqa: E402
from vlm_labeler import LabelRecord, read_label_records  # noqa: E402

EPISODE_REF = re.compile(r"(\d+)\s*$")


def episode_index_of(ref: str) -> int:
    """从 `episode_000003` 之类的引用里取集号。"""
    match = EPISODE_REF.search(str(ref))
    if not match:
        raise ValueError(f"无法从 {ref!r} 解析集号")
    return int(match.group(1))


def episodes_from_sidecar(
    records: Iterable[Mapping[str, Any]],
) -> dict[int, list[dict[str, float]]]:
    """按集整理旁挂帧（按 `frame_index` 排序），返回每帧的 `positions` + `gripper.load`。"""
    buckets: dict[int, list[tuple[int, dict[str, float]]]] = {}
    for row in records:
        episode = int(row["episode_index"])
        frame = int(row["frame_index"])
        positions = row.get("positions") or {}
        if not positions:
            continue
        load = (row.get("load") or {}).get("gripper.load")
        frame_data = {str(k): float(v) for k, v in positions.items()}
        frame_data["gripper.load"] = None if load is None else float(load)
        buckets.setdefault(episode, []).append((frame, frame_data))
    return {
        episode: [data for _, data in sorted(items)]
        for episode, items in buckets.items()
    }


def sensor_success(
    frames: Sequence[Mapping[str, float]], *, use_load: bool
) -> EpisodeOutcome:
    """对单集帧序列跑逐集判定（返回 `EpisodeOutcome`）。"""
    lives = [dict(frame) for frame in frames]
    loads = [frame.get("gripper.load") for frame in lives]
    return aggregate_episode_success(lives, loads, use_load=use_load)


def audit_repo(
    sidecar_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[LabelRecord],
    *,
    use_load: bool,
) -> dict[str, Any] | None:
    """对照单 repo 的传感器逐集成功与 VLM 标签；无可用集时返回 None。"""
    episodes = episodes_from_sidecar(sidecar_rows)
    labelled = {
        episode_index_of(record.episode_ref): bool(record.success)
        for record in labels
        if record.success is not None
    }
    common = sorted(set(episodes) & set(labelled))
    if not common:
        return None

    sensor: list[bool] = []
    vlm: list[bool] = []
    detail: list[dict[str, Any]] = []
    for index in common:
        outcome = sensor_success(episodes[index], use_load=use_load)
        sensor.append(bool(outcome.success))
        vlm.append(labelled[index])
        detail.append(
            {
                "episode_index": index,
                "sensor_success": bool(outcome.success),
                "vlm_success": labelled[index],
                "held_frames": outcome.held_frames,
                "lifted_frames": outcome.lifted_frames,
                "holding_and_lifted_frames": outcome.holding_and_lifted_frames,
                "n_frames": outcome.n_frames,
            }
        )

    tp = sum(1 for s, v in zip(sensor, vlm) if s and v)
    tn = sum(1 for s, v in zip(sensor, vlm) if not s and not v)
    fp = sum(1 for s, v in zip(sensor, vlm) if s and not v)
    fn = sum(1 for s, v in zip(sensor, vlm) if not s and v)
    return {
        "use_load": use_load,
        "n_episodes": len(common),
        "kappa": cohen_kappa(sensor, vlm),
        "accuracy": (tp + tn) / len(common),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "sensor_success_count": sum(sensor),
        "vlm_success_count": sum(vlm),
        "episodes": detail,
    }


def pooled(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把多个 repo 的明细合并后再算一次 κ（避免把不同 repo 的 κ 简单平均）。"""
    episodes = [row for report in reports for row in report["episodes"]]
    if not episodes:
        return {"n_episodes": 0, "kappa": None, "accuracy": None}
    sensor = [bool(row["sensor_success"]) for row in episodes]
    vlm = [bool(row["vlm_success"]) for row in episodes]
    tp = sum(1 for s, v in zip(sensor, vlm) if s and v)
    tn = sum(1 for s, v in zip(sensor, vlm) if not s and not v)
    fp = sum(1 for s, v in zip(sensor, vlm) if s and not v)
    fn = sum(1 for s, v in zip(sensor, vlm) if not s and v)
    return {
        "n_episodes": len(episodes),
        "kappa": cohen_kappa(sensor, vlm),
        "accuracy": (tp + tn) / len(episodes),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "sensor_success_count": sum(sensor),
        "vlm_success_count": sum(vlm),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# A4 逐集成功复评（传感器 vs VLM 标签）",
        "",
        "> 传感器口径：存在某帧同时「夹住 + 已抬起」；数据来自旁挂 positions + gripper.load。",
        "",
        "| 口径 | 集数 | κ | 准确率 | TP | TN | FP | FN | 传感器成功 | VLM 成功 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for use_load, block in (("位置 OR 负载", payload["with_load"]), ("纯位置", payload["position_only"])):
        pooled_block = block["pooled"]
        conf = pooled_block.get("confusion", {})
        lines.append(
            f"| {use_load} | {pooled_block['n_episodes']} | {_fmt(pooled_block['kappa'])} "
            f"| {_fmt(pooled_block['accuracy'])} | {conf.get('tp', 0)} | {conf.get('tn', 0)} "
            f"| {conf.get('fp', 0)} | {conf.get('fn', 0)} | {pooled_block.get('sensor_success_count', 0)} "
            f"| {pooled_block.get('vlm_success_count', 0)} |"
        )
    lines.append("")
    lines.append("## 分 repo（位置 OR 负载）")
    lines.append("")
    lines.append("| repo | 集数 | κ | 准确率 | 传感器成功 | VLM 成功 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for repo, report in payload["with_load"]["repos"].items():
        lines.append(
            f"| {repo} | {report['n_episodes']} | {report['kappa']:.3f} "
            f"| {report['accuracy']:.3f} | {report['sensor_success_count']} "
            f"| {report['vlm_success_count']} |"
        )
    lines.append("")
    lines.append(
        "> 注意：VLM 标签本身未经人工全量复核（仅 14 集抽检），κ 只能说明"
        "「传感器逐集判据与 VLM 判断是否一致」，不能当作绝对真值。"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="A4 逐集成功复评")
    parser.add_argument("--labels-root", default="runs/week38/label_vla")
    parser.add_argument("--sidecar-root", default="runs/week38/collect/sidecars")
    parser.add_argument("--out", default="runs/week38/label_vla/episode_success_audit")
    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    sidecar_root = Path(args.sidecar_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "with_load": {"repos": {}, "pooled": {}},
        "position_only": {"repos": {}, "pooled": {}},
    }
    for repo_dir in sorted(p for p in labels_root.iterdir() if p.is_dir()):
        labels_path = repo_dir / "labels.jsonl"
        sidecar_path = sidecar_root / f"{repo_dir.name}.jsonl"
        if not labels_path.is_file() or not sidecar_path.is_file():
            print(f"[skip] {repo_dir.name}：缺 labels.jsonl 或旁挂", flush=True)
            continue
        labels = read_label_records(labels_path)
        sidecar_rows = [
            json.loads(line)
            for line in sidecar_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for use_load, key in ((True, "with_load"), (False, "position_only")):
            report = audit_repo(sidecar_rows, labels, use_load=use_load)
            if report is not None:
                payload[key]["repos"][repo_dir.name] = report
        print(f"[repo] {repo_dir.name} 完成", flush=True)

    for key in ("with_load", "position_only"):
        payload[key]["pooled"] = pooled(list(payload[key]["repos"].values()))

    (out_dir / "episode_success_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(payload)
    (out_dir / "episode_success_audit.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
