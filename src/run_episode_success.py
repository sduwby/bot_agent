#!/usr/bin/env python
"""逐集成功判据 CLI + A4 口径对照报告（第 38 周。

对 6 个真机 LeRobot repo 逐集计算三种口径的成功，并与人工抽检的 24 集比对，
产出 `report.json` + `report.md`。**纯本地**：只读数据集 parquet、旁挂负载、
`labels/` 人工抽检，不联网、不加载模型。

报告回答四件事：
1. 旁挂负载通道的实证统计（它第一次被真正用上）：逐帧 |load| 分布、超阈值帧数、
   逐集 max|load| —— 用来判断"负载通道在真机上到底有没有信号"；
2. 口径对照：`position_only` vs `position_or_load` 逐集成功，与人工 24 集的一致性
   （逐 repo + 整体：混淆矩阵 / 一致率 / Cohen κ，κ 复用 `label_audit.cohen_kappa`）；
3. 任务感知判据（`episode_success.assess_episode`）的结果与人工一致性；
4. 三种口径判断不同的集号（供人工复核），以及 `left_verified=False` 的影响。

用法::

    python src/run_episode_success.py
    python src/run_episode_success.py --dataset-root <root> --sidecar-dir <dir> \
        --human-dir <dir> --out runs/week38/episode_success
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from episode_success import (  # noqa: E402
    CRITERIA,
    EpisodeChannels,
    EpisodeSuccess,
    assess_episode,
    divergent_criteria,
    task_from_instruction,
)
from label_audit import cohen_kappa  # noqa: E402  # κ 复用仓库既有实现
from lerobot_label_source import (  # noqa: E402
    DatasetMeta,
    read_dataset_meta,
)
from load_sidecar import episode_loads, read_load_sidecar  # noqa: E402
from real_observation import GRIP_LOAD_THRESHOLD  # noqa: E402
from vlm_labeler import (  # noqa: E402
    LabelRecord,
    define_intent_taxonomy,
    read_label_records,
)

REPORT_SCHEMA_VERSION = "week38_episode_success_v1"

#: 默认数据集根 / 旁挂目录 / 输出目录（相对仓库根）。
DEFAULT_DATASET_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "wangbangyan"
DEFAULT_SIDECAR_DIR = "runs/week38/collect/sidecars"
DEFAULT_OUT = "runs/week38/episode_success"
#: 人工抽检 jsonl 默认目录（复核页导出到浏览器下载目录）。
DEFAULT_HUMAN_DIR = Path.home() / "Downloads"

#: `observation.state` 里夹爪位置与抬升通道的键名（与 `real_observation.STATE_NAMES` 一致）。
GRIPPER_STATE_NAME = "gripper.pos"
LIFT_STATE_NAME = "wrist_flex.pos"
#: 人工抽检文件名前缀（自动发现时用；仓库自带的 demo_v23 human.jsonl 因 task_text 不匹配而被忽略）。
HUMAN_FILE_PREFIX = "human"


class ReportError(RuntimeError):
    """报告生成的契约错误。"""


# --------------------------------------------------------------------------- #
# 读取：数据集状态 + 旁挂负载
# --------------------------------------------------------------------------- #

def read_episode_channels(
    meta: DatasetMeta,
    episode_index: int,
    load_series: list[float | None] | None,
) -> EpisodeChannels:
    """读单集 `observation.state` 的夹爪与抬升通道，接上旁挂负载。

    `observation.state` 的实际 schema 由 `meta.state_names` 决定（本批 6 个 repo 均为
    `shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll/gripper` 六轴），
    故按**名字**查下标，而不是写死 `[:3]` / `[3:5]`——真机 schema 变化时按名取列才稳。
    """
    import pyarrow.parquet as pq

    chunk = episode_index // max(meta.chunks_size, 1)
    path = meta.root / meta.data_path.format(
        episode_chunk=chunk, episode_index=episode_index
    )
    if not path.is_file():
        raise ReportError(f"缺少状态文件: {path}")
    table = pq.read_table(str(path))
    if "observation.state" not in table.column_names:
        raise ReportError(f"{path.name} 缺 observation.state 列")
    rows = table.column("observation.state").to_pylist()

    names = list(meta.state_names)
    if GRIPPER_STATE_NAME not in names:
        raise ReportError(
            f"observation.state 缺 {GRIPPER_STATE_NAME}（实际 {names}）"
        )
    grip_index = names.index(GRIPPER_STATE_NAME)
    lift_index = names.index(LIFT_STATE_NAME) if LIFT_STATE_NAME in names else None

    grip = tuple(float(row[grip_index]) for row in rows)
    wrist = (
        tuple(float(row[lift_index]) for row in rows)
        if lift_index is not None
        else None
    )
    load = tuple(load_series) if load_series else None
    return EpisodeChannels(gripper_pos=grip, wrist_flex_pos=wrist, load=load)


def load_channel_stats(
    loads_by_episode: dict[int, list[float | None]],
    *,
    threshold: float = GRIP_LOAD_THRESHOLD,
) -> dict[str, Any]:
    """旁挂负载通道的实证统计（逐帧 + 逐集）。"""
    frames = [abs(value) for series in loads_by_episode.values() for value in series
              if value is not None]
    over = sum(1 for value in frames if value > threshold)
    per_episode_max = {
        str(index): (max((abs(v) for v in series if v is not None), default=None))
        for index, series in sorted(loads_by_episode.items())
    }
    return {
        "episodes": len(loads_by_episode),
        "frames": len(frames),
        "load_abs_min": min(frames) if frames else None,
        "load_abs_median": statistics.median(frames) if frames else None,
        "load_abs_max": max(frames) if frames else None,
        "threshold": float(threshold),
        "frames_over_threshold": over,
        "frames_over_threshold_ratio": (over / len(frames)) if frames else None,
        "per_episode_max_abs_load": per_episode_max,
        "episodes_with_any_over_threshold": sum(
            1
            for value in per_episode_max.values()
            if value is not None and value > threshold
        ),
    }


# --------------------------------------------------------------------------- #
# 人工抽检对齐
# --------------------------------------------------------------------------- #

def load_human_by_task(human_dir: Path) -> dict[str, list[LabelRecord]]:
    """扫描人工抽检目录，按 `task_text` 归组（同一 repo 的 4 集共享同一 task_text）。"""
    grouped: dict[str, list[LabelRecord]] = {}
    if not human_dir.is_dir():
        return grouped
    for path in sorted(human_dir.glob(f"{HUMAN_FILE_PREFIX}*.jsonl")):
        for record in read_label_records(path):
            grouped.setdefault(record.task_text, []).append(record)
    return grouped


def evaluate_criterion(
    auto_by_ref: dict[str, EpisodeSuccess],
    human_by_ref: dict[str, LabelRecord],
    criterion: str,
) -> dict[str, Any]:
    """单一口径 vs 人工的混淆矩阵 / 一致率 / Cohen κ。

    只在两侧都有判定的集上配对（口径可能给 ``None``）。κ 复用 `label_audit.cohen_kappa`。
    """
    refs = sorted(set(auto_by_ref) & set(human_by_ref))
    pairs = [
        (auto_by_ref[ref].value(criterion), human_by_ref[ref].success)
        for ref in refs
    ]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    labels = ["False", "True"]
    matrix = [[0, 0], [0, 0]]
    for auto, human in pairs:
        matrix[0 if not auto else 1][0 if not human else 1] += 1
    total = len(pairs)
    agreement = (matrix[0][0] + matrix[1][1]) / total if total else None
    kappa = (
        float(cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]))
        if pairs
        else None
    )
    disagreements = sorted(
        ref for ref in refs
        if auto_by_ref[ref].value(criterion) is not None
        and human_by_ref[ref].success is not None
        and auto_by_ref[ref].value(criterion) != human_by_ref[ref].success
    )
    return {
        "criterion": criterion,
        "n_pairs": total,
        "confusion": matrix,
        "confusion_labels": labels,
        "agreement": agreement,
        "kappa": kappa,
        "disagreements": disagreements,
    }


# --------------------------------------------------------------------------- #
# 报告装配
# --------------------------------------------------------------------------- #

def repo_block(
    repo: str,
    meta: DatasetMeta,
    results: list[EpisodeSuccess],
    human_records: list[LabelRecord],
    load_stats: dict[str, Any],
) -> dict[str, Any]:
    """单 repo 的报告块。"""
    instruction = meta.episodes[0].task if meta.episodes else ""
    intent = task_from_instruction(instruction)
    auto_by_ref = {result.episode_ref: result for result in results}
    human_by_ref = {record.episode_ref: record for record in human_records}
    criteria = {
        criterion: evaluate_criterion(auto_by_ref, human_by_ref, criterion)
        for criterion in CRITERIA
    }
    divergences = [
        {
            "episode_ref": result.episode_ref,
            "position_only": result.position_only,
            "position_or_load": result.position_or_load,
            "task_aware": result.task_aware,
        }
        for result in results
        if divergent_criteria(result)
    ]
    return {
        "repo": repo,
        "instruction": instruction,
        "intent": intent,
        "n_episodes": len(results),
        "n_human": len(human_by_ref),
        "load_channel": load_stats,
        "criteria_comparison": criteria,
        "divergences": divergences,
    }


def pooled_comparison(
    pooled_auto: dict[str, EpisodeSuccess],
    pooled_human: dict[str, LabelRecord],
) -> dict[str, Any]:
    """6 个 repo 汇总后的逐口径一致性。"""
    return {
        criterion: evaluate_criterion(pooled_auto, pooled_human, criterion)
        for criterion in CRITERIA
    }


def build_report(
    dataset_root: Path,
    sidecar_dir: Path,
    human_dir: Path,
    repos: list[str],
    blocks: dict[str, dict[str, Any]],
    results_by_repo: dict[str, list[EpisodeSuccess]],
    pooled: dict[str, Any],
) -> dict[str, Any]:
    """汇总成完整报告 dict。"""
    pooled_auto: dict[str, EpisodeSuccess] = {}
    for repo, results in results_by_repo.items():
        for result in results:
            pooled_auto[f"{repo}/{result.episode_ref}"] = result

    state_names: dict[str, list[str]] = {}
    per_repo_public: dict[str, Any] = {}
    for repo, block in blocks.items():
        public = {key: value for key, value in block.items()
                  if not key.startswith("_")}
        per_repo_public[repo] = public
        state_names[repo] = list(block["_state_names"])

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "sidecar_dir": str(sidecar_dir),
        "human_dir": str(human_dir),
        "repos": repos,
        "observation_state_names": state_names,
        "load_threshold": GRIP_LOAD_THRESHOLD,
        "load_channel_summary": {
            repo: per_repo_public[repo]["load_channel"] for repo in repos
        },
        "per_repo": per_repo_public,
        "criteria_comparison_pooled": pooled,
        "kappa_caveat": (
            "本批 24 集人工抽检**全部 success=True**（正负样本极不平衡），κ 的二分类"
            "期望一致率 pe 退化为 1；`cohen_kappa` 按约定在‘非完全一致’时返回 0.0。"
            "因此这里三个口径的 κ 都读作 0 —— **不可解释为判据无价值**，应看‘一致率’"
            "与混淆矩阵。要得到有意义的 κ，需人工抽检里同时含成功与失败样本。"
        ),
        "left_verified": {
            "value": False,
            "reason": (
                "数据只有关节/tcp 通道与夹爪通道，**没有任何物体位置跟踪**；"
                "因此‘物体是否到了左边’无法从现有通道自动判定。任务感知判据的 place "
                "口径只判‘集末已松手’，不判‘左边’，结果里 left_verified 恒为 False。"
            ),
        },
        "place_criterion_ambiguity": (
            "place 口径本身仍有歧义：(1)‘放到左边’不可判（见 left_verified）；"
            "(2)‘集末已松手’以**末帧**为准，若演示在夹持状态下结束（如 01_place_left "
            "的个别集末帧夹爪仍闭合），会被判 False——这既可能是演示确实没松手，"
            "也可能是末帧取帧偏差，需人工复核。"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """把报告 dict 渲染成可读 Markdown。"""
    lines: list[str] = ["# A4 逐集成功判据 · 口径对照报告", ""]
    lines.append(f"- schema：`{report['schema_version']}`")
    lines.append(f"- 数据集根：`{report['dataset_root']}`")
    lines.append(f"- 旁挂目录：`{report['sidecar_dir']}`")
    lines.append(f"- 人工目录：`{report['human_dir']}`")
    lines.append(f"- 夹持负载阈值 `GRIP_LOAD_THRESHOLD` = {report['load_threshold']}‰")
    lines.append("")

    lines.append("## 1. 旁挂负载通道实证统计")
    lines.append("")
    lines.append("| repo | 集数 | 帧数 | \\|load\\| min/median/max | 超阈值帧 | 集级 max\\|load\\| 超阈值的集 |")
    lines.append("|---|---|---|---|---|---|")
    for repo in report["repos"]:
        stat = report["load_channel_summary"][repo]
        lines.append(
            f"| {repo} | {stat['episodes']} | {stat['frames']} | "
            f"{_num(stat['load_abs_min'])}/{_num(stat['load_abs_median'])}/{_num(stat['load_abs_max'])} | "
            f"{stat['frames_over_threshold']} ({_pct(stat['frames_over_threshold_ratio'])}) | "
            f"{stat['episodes_with_any_over_threshold']}/{stat['episodes']} |"
        )
    lines.append("")
    lines.append("逐集 `max|load|`：")
    for repo in report["repos"]:
        stat = report["load_channel_summary"][repo]
        pairs = ", ".join(
            f"{idx}:{_num(value)}" for idx, value in stat["per_episode_max_abs_load"].items()
        )
        lines.append(f"- **{repo}**：{pairs}")
    lines.append("")

    lines.append("## 2. 口径对照（vs 人工抽检 24 集）")
    lines.append("")
    lines.append("### 2.1 逐 repo")
    lines.append("")
    lines.append("| repo | intent | 口径 | n | 混淆(行=判据/列=人工) | 一致率 | κ | 分歧集 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for repo in report["repos"]:
        block = report["per_repo"][repo]
        for criterion in CRITERIA:
            comp = block["criteria_comparison"][criterion]
            lines.append(
                f"| {repo} | {block['intent']} | {criterion} | {comp['n_pairs']} | "
                f"{_conf(comp['confusion'])} | {_pct(comp['agreement'])} | "
                f"{_num(comp['kappa'], 3)} | {', '.join(comp['disagreements']) or '—'} |"
            )
    lines.append("")
    lines.append("### 2.2 整体（6 repo 汇总）")
    lines.append("")
    lines.append("| 口径 | n | 混淆(行=判据/列=人工) | 一致率 | κ |")
    lines.append("|---|---|---|---|---|")
    for criterion in CRITERIA:
        comp = report["criteria_comparison_pooled"][criterion]
        lines.append(
            f"| {criterion} | {comp['n_pairs']} | {_conf(comp['confusion'])} | "
            f"{_pct(comp['agreement'])} | {_num(comp['kappa'], 3)} |"
        )
    lines.append("")

    lines.append("## 3. 三种口径判断不同的集（供人工复核）")
    lines.append("")
    for repo in report["repos"]:
        block = report["per_repo"][repo]
        if not block["divergences"]:
            lines.append(f"- **{repo}**：无分歧")
            continue
        items = "；".join(
            f"{d['episode_ref']}(pos_only={d['position_only']}, "
            f"pos_or_load={d['position_or_load']}, aware={d['task_aware']})"
            for d in block["divergences"]
        )
        lines.append(f"- **{repo}**：{items}")
    lines.append("")

    lines.append("## 4. 已知边界")
    lines.append("")
    lines.append(f"- κ 退化说明：{report['kappa_caveat']}")
    left = report["left_verified"]
    lines.append(f"- `left_verified = {left['value']}`：{left['reason']}")
    lines.append(f"- place 口径歧义：{report['place_criterion_ambiguity']}")
    lines.append("")
    return "\n".join(lines)


def _num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def _conf(matrix: list[list[int]]) -> str:
    return f"[[{matrix[0][0]},{matrix[0][1]}],[{matrix[1][0]},{matrix[1][1]}]]"


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """落盘 `report.json` + `report.md`。"""
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    md_path = directory / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def discover_repos(sidecar_dir: Path) -> list[str]:
    """由旁挂目录的文件名发现 repo（每个 repo 一份 `<repo>.jsonl`）。"""
    return sorted(path.stem for path in sidecar_dir.glob("*.jsonl"))


def run(dataset_root: Path, sidecar_dir: Path, human_dir: Path, out_dir: Path) -> dict[str, Any]:
    """主流程：逐 repo 计算 + 与人工对照 + 落盘。"""
    repos = discover_repos(sidecar_dir)
    if not repos:
        raise ReportError(f"旁挂目录 {sidecar_dir} 下没有 *.jsonl")
    human_by_task = load_human_by_task(human_dir)
    taxonomy = define_intent_taxonomy()

    results_by_repo: dict[str, list[EpisodeSuccess]] = {}
    blocks: dict[str, dict[str, Any]] = {}
    pooled_auto: dict[str, EpisodeSuccess] = {}
    pooled_human: dict[str, LabelRecord] = {}

    for repo in repos:
        meta = read_dataset_meta(dataset_root / repo)
        rows = read_load_sidecar(sidecar_dir / f"{repo}.jsonl")
        loads_by_episode = episode_loads(rows)
        instruction = meta.episodes[0].task if meta.episodes else ""
        intent = task_from_instruction(instruction, taxonomy)

        results: list[EpisodeSuccess] = []
        for episode in meta.episodes:
            channels = read_episode_channels(
                meta, episode.episode_index, loads_by_episode.get(episode.episode_index)
            )
            results.append(
                assess_episode(f"episode_{episode.episode_index:06d}", intent, channels)
            )
        results_by_repo[repo] = results

        human_records = human_by_task.get(instruction, [])
        block = repo_block(
            repo, meta, results, human_records, load_channel_stats(loads_by_episode)
        )
        block["_results"] = results
        block["_state_names"] = list(meta.state_names)
        blocks[repo] = block

        for result in results:
            pooled_auto[f"{repo}/{result.episode_ref}"] = result
        for record in human_records:
            pooled_human[f"{repo}/{record.episode_ref}"] = record

        print(
            f"[逐集] {repo:32s} 集={len(results):2d} 人工={len(human_records)} "
            f"intent={intent}",
            flush=True,
        )

    pooled = pooled_comparison(pooled_auto, pooled_human)
    # 诊断块需要在 per_repo 里带上 `_results`，渲染前先补齐 position_only_true
    for repo, block in blocks.items():
        block["position_only_true"] = sum(1 for r in block["_results"] if r.position_only)
    report = build_report(
        dataset_root, sidecar_dir, human_dir, repos, blocks, results_by_repo, pooled
    )
    report["episode_results"] = {
        repo: [result.to_dict() for result in results]
        for repo, results in results_by_repo.items()
    }
    json_path, md_path = write_report(report, out_dir)
    # 人工抽检合并落盘（可追溯：报告中所有 κ 的参照）
    merged = out_dir / "human_merged.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for record in pooled_human.values():
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    print(f"\n[报告] → {json_path}")
    print(f"[报告] → {md_path}")
    print("[整体] 口径一致率 / κ（vs 人工 24 集）：")
    for criterion in CRITERIA:
        comp = pooled[criterion]
        print(
            f"  {criterion:16s} n={comp['n_pairs']:2d} "
            f"一致率={_pct(comp['agreement'])} κ={_num(comp['kappa'], 3)}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="逐集成功判据 + A4 口径对照报告")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--sidecar-dir", default=DEFAULT_SIDECAR_DIR)
    parser.add_argument("--human-dir", default=str(DEFAULT_HUMAN_DIR))
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    run(
        Path(args.dataset_root).expanduser(),
        Path(args.sidecar_dir).expanduser(),
        Path(args.human_dir).expanduser(),
        Path(args.out).expanduser(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
