"""A3/A4 机读侧汇总（第 38 周 P1 收尾）。

把 `run_vlm_label.py` 产出的 6 份 `labels.jsonl` 汇总成机读摘要，回答三件事：

1. **A3（意图判别力）**：`intent` 由确定性规则从 `task_text` 映射（见 `vlm_labeler`），
   而 6 个 repo 的 120 条标注里 `intent` 几乎全是 `pick`——本模块**不粉饰**，直接报告
   预测类别分布、`place` 是否被预测过，并在某类样本数为 0 时显式标注 κ 退化。
2. **A4（成功）**：`success` 逐 repo 分布，以及它和 repo 语义（`_pick_a` / `_place_left`）
   的一致性——用来判断 VLM 是否只是"照抄任务类型"。
3. **A4 口径对齐**：用旁挂负载数据算"物理侧成功判据"，与 VLM 的 `success` 逐集比对。
   **本模块不自行发明判据**：仓库里只有逐帧判据（`holds_object` 等）与逐集纯位置判据
   （`vlm_labeler.success_from_sensor`），没有任何"逐集负载成功"函数，故该比对
   **如实标注为无法完成**，并列出缺什么（见 `alignment_section`）。

纯逻辑：本模块只依赖 `vlm_labeler` / `load_sidecar` / `label_audit`（numpy），
不联网、不需要 GPU、不 import torch——因此可离线单测。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from label_audit import cohen_kappa
from load_sidecar import episode_loads, read_load_sidecar, verify_alignment
from vlm_labeler import LabelRecord, read_label_records

SUMMARY_SCHEMA_VERSION = "week38_vla_label_summary_v1"

#: `intent` 为 ``None``（规则未命中）时在分布里的占位键，避免 JSON 里出现 `null` 键。
NONE_KEY = "none"

#: 逐 repo 语义 → 期望意图。由 repo 名后缀推出（采集计划见 runs/week38/collect/collect_tasks.json）。
REPO_SEMANTICS: dict[str, str] = {
    "_pick_a": "pick",
    "_place_left": "place",
}

#: 旁挂负载文件默认目录（与 `run_vla_collect_record.py --sidecar` 一致）。
DEFAULT_SIDECAR_DIR = "runs/week38/collect/sidecars"
#: 标注产物默认根目录（每个子目录一份 labels.jsonl）。
DEFAULT_LABEL_ROOT = "runs/week38/label_vla"
#: 目录名以此开头的是 `run_vlm_label.py --dry-run` 的桩产物，**不**纳入汇总。
DRY_RUN_PREFIX = "dry_"

# --------------------------------------------------------------------------- #
# A4 口径对齐：仓库里到底有什么（用于说明"为什么无法完成"）
# --------------------------------------------------------------------------- #

#: 仓库里**已有的**夹持/接触判据原语（按文件:函数列出，全部逐帧或单帧）。
EXISTING_HOLDING_PRIMITIVES: tuple[str, ...] = (
    "real_observation.is_holding(gripper_pos) —— 逐帧、纯位置（空闭基准 + 余量）",
    "real_observation.holds_object(gripper_pos, load) —— 逐帧、位置 OR 负载（|load|>150‰）",
    "real_observation.detect_contact(load, baseline) —— 逐帧、相对基准的增量判定",
    "real_observation.GRIP_LOAD_THRESHOLD = 150.0 —— 负载幅值常量（‰）",
    "run_vla_infer.holding_and_lifted(state, load) —— 单帧、夹住且抬起",
    "real_verifier.RealVerifier.verify_holding / verify_grasp_lift —— 单帧、实时技能结果验证",
    "load_sidecar.episode_loads(records) —— 按集取负载**序列**（只整理、不下判定）",
    "load_sidecar.verify_alignment(...) —— 旁挂与数据集**帧对齐**校验（非成功判据）",
    "vlm_labeler.success_from_sensor(sensor_state) —— 逐集、但**纯位置**（gripper_pos 尾帧）",
)

#: 是否存在"直接复用即可得到逐集（负载）成功"的函数。经核查为否 → 见 `ALIGNMENT_MISSING`。
PER_EPISODE_SUCCESS_FROM_SIDECAR_AVAILABLE = False

#: 为什么无法完成逐集比对：缺的具体是什么。
ALIGNMENT_MISSING: tuple[str, ...] = (
    "没有任何函数把 `load_sidecar` 的逐集负载序列 + 位置序列判成逐集 success；",
    "`vlm_labeler.success_from_sensor` 虽是逐集函数，但只吃 `gripper_pos`（纯位置、不含负载），"
    "其口径正是 labels.jsonl 里 `success` 的来源——拿它和 VLM 的 success 比是同一判据自比，"
    "不构成独立的“物理侧负载判据”；",
    "`holds_object` / `detect_contact` / `holding_and_lifted` / `RealVerifier` 都是**单帧**判据，"
    "既有代码没有规定“逐集如何聚合”（尾帧？任意帧？多数票？）——自行规定聚合方式即为"
    "“临时拍判据”，本汇总不做。",
)


def infer_repo_semantics(repo: str) -> str | None:
    """由 repo 名后缀推出期望意图（`pick` / `place`），未知返回 ``None``。"""
    for suffix, intent in REPO_SEMANTICS.items():
        if repo.endswith(suffix):
            return intent
    return None


def _label_key(value: Any) -> str:
    """把标签值规整成可作 JSON 键的字符串（``None`` → ``NONE_KEY``）。"""
    if value is None:
        return NONE_KEY
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def count_values(values: Sequence[Any]) -> dict[str, int]:
    """统计取值分布，键统一规整为字符串（``None`` → ``"none"``）。"""
    counts: dict[str, int] = {}
    for value in values:
        key = _label_key(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def intent_distribution(records: Sequence[LabelRecord]) -> dict[str, int]:
    """逐条 `intent` 的分布。"""
    return count_values([record.intent for record in records])


def success_distribution(records: Sequence[LabelRecord]) -> dict[str, int]:
    """`success` 分布（键为 ``true`` / ``false`` / ``none``）。"""
    counts = {"true": 0, "false": 0, "none": 0}
    for record in records:
        if record.success is None:
            counts["none"] += 1
        elif record.success:
            counts["true"] += 1
        else:
            counts["false"] += 1
    return counts


def subtask_distribution(records: Sequence[LabelRecord]) -> dict[str, int]:
    """`subtask` 分布（``None`` → ``"none"``）。"""
    return count_values([record.subtask for record in records])


def confidence_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    """置信度统计；空序列时返回 n=0 与 ``None`` 统计量。"""
    items = [float(value) for value in values]
    if not items:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(items),
        "mean": sum(items) / len(items),
        "min": min(items),
        "max": max(items),
    }


def confidence_by_intent(records: Sequence[LabelRecord]) -> dict[str, dict[str, Any]]:
    """按 `intent` 分组的置信度分布。"""
    buckets: dict[str, list[float]] = {}
    for record in records:
        buckets.setdefault(_label_key(record.intent), []).append(float(record.confidence))
    return {key: confidence_stats(values) for key, values in sorted(buckets.items())}


def discriminability(
    records: Sequence[LabelRecord], expected_intents: Sequence[str | None]
) -> dict[str, Any]:
    """A3 判别力：预测分布 vs 期望分布 + κ（并显式标注退化）。

    `expected_intents` 为每条的 repo 语义（``pick`` / ``place`` / ``None``）。
    κ 用 `label_audit.cohen_kappa`，``None`` 作独立类别；当某一类在**期望**里存在、
    却在**预测**里一次都没出现时，κ 是退化值，必须显式报告而不是当成"一致率"。
    """
    predicted = [record.intent for record in records]
    expected = list(expected_intents)
    pred_dist = count_values(predicted)
    exp_dist = count_values(expected)
    place_predicted = pred_dist.get("place", 0)
    place_expected = exp_dist.get("place", 0)
    kappa = cohen_kappa(predicted, expected) if records else 0.0
    degenerate = place_expected > 0 and place_predicted == 0
    return {
        "predicted_class_distribution": pred_dist,
        "expected_class_distribution": exp_dist,
        "place_predicted_times": place_predicted,
        "place_expected_count": place_expected,
        "place_never_predicted": place_predicted == 0,
        "kappa_vs_repo_semantics": kappa,
        "kappa_degenerate": degenerate,
        "kappa_note": (
            "`place` 在期望里存在却在预测里一次未出现 → κ 退化（不代表一致或不一致），"
            "只能读作“该维度无判别力”"
            if degenerate
            else "两类均有样本，κ 可解释"
        ),
    }


def repo_summary(repo: str, records: Sequence[LabelRecord]) -> dict[str, Any]:
    """单个 repo 的汇总块。"""
    expected = infer_repo_semantics(repo)
    success = success_distribution(records)
    return {
        "repo": repo,
        "n_records": len(records),
        "expected_intent": expected,
        "intent_distribution": intent_distribution(records),
        "success_distribution": success,
        "success_true_ratio": (
            success["true"] / len(records) if records else None
        ),
        "subtask_distribution": subtask_distribution(records),
        "confidence": confidence_stats([record.confidence for record in records]),
        "all_success_true": bool(records) and success["false"] == 0 and success["none"] == 0,
        "all_success_false": bool(records) and success["true"] == 0 and success["none"] == 0,
    }


def sidecar_facts(sidecar_dir: str | Path, repos: Sequence[str]) -> dict[str, Any]:
    """读旁挂负载文件，给出覆盖情况与"逐集成功函数缺失"的结论。

    只做**描述性**统计（行数/集数/负载覆盖），**不产出任何成功判据**。
    """
    root = Path(sidecar_dir).expanduser()
    per_repo: dict[str, Any] = {}
    for repo in repos:
        rows = read_load_sidecar(root / f"{repo}.jsonl")
        episodes = sorted({int(row["episode_index"]) for row in rows})
        loads = episode_loads(rows)
        nonnull = sum(
            1
            for series in loads.values()
            for value in series
            if value is not None
        )
        per_repo[repo] = {
            "rows": len(rows),
            "episodes": len(episodes),
            "gripper_load_nonnull_frames": nonnull,
            "decode_error_rows": sum(1 for row in rows if row.get("decode_errors")),
        }
    return {
        "sidecar_dir": str(root),
        "per_repo": per_repo,
        "all_repos_present": all(item["rows"] > 0 for item in per_repo.values()),
    }


def alignment_section(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """A4 口径对齐块的结论：**无法完成**（附缺什么）。

    参数 `sidecar` 为 `sidecar_facts` 的输出，仅用于证明旁挂数据本身可用
    （有负载通道、逐帧对齐），从而把"无法完成"归因到**缺函数**而不是缺数据。
    """
    return {
        "status": "not_completed",
        "physical_success_criterion": None,
        "confusion_matrix": None,
        "agreement_rate": None,
        "reason": (
            "仓库内不存在可直接复用的“逐集成功”函数；仅有逐帧判据与逐集纯位置判据，"
            "缺少把旁挂负载序列聚合成逐集判定的既有实现。按任务约束（不自行发明判据）"
            "如实标注为无法完成。"
        ),
        "existing_primitives": list(EXISTING_HOLDING_PRIMITIVES),
        "missing": list(ALIGNMENT_MISSING),
        "sidecar_data_available": bool(sidecar.get("all_repos_present")),
        "sidecar": dict(sidecar),
    }


def build_summary(
    records_by_repo: Mapping[str, Sequence[LabelRecord]],
    *,
    sidecar_dir: str | Path = DEFAULT_SIDECAR_DIR,
) -> dict[str, Any]:
    """把逐 repo 的标注记录汇总成机读摘要（纯逻辑，不写盘）。"""
    repos = list(records_by_repo)
    all_records: list[LabelRecord] = [r for repo in repos for r in records_by_repo[repo]]
    expected = [infer_repo_semantics(repo) for repo in repos
                for _ in records_by_repo[repo]]

    per_repo = {repo: repo_summary(repo, records_by_repo[repo]) for repo in repos}
    sidecar = sidecar_facts(sidecar_dir, repos)

    consistency = [
        {
            "repo": repo,
            "expected_intent": per_repo[repo]["expected_intent"],
            "n_records": per_repo[repo]["n_records"],
            "success_true": per_repo[repo]["success_distribution"]["true"],
            "success_false": per_repo[repo]["success_distribution"]["false"],
            "success_none": per_repo[repo]["success_distribution"]["none"],
            "all_success_true": per_repo[repo]["all_success_true"],
            "all_success_false": per_repo[repo]["all_success_false"],
        }
        for repo in repos
    ]

    notes: list[str] = []
    disc = discriminability(all_records, expected)
    if disc["place_never_predicted"]:
        notes.append(
            f"`place` 期望 {disc['place_expected_count']} 条、被预测 0 条 → "
            "零样本 VLM 在双意图数据上意图判别力为零。"
        )

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "repos": repos,
        "n_repos": len(repos),
        "n_records": len(all_records),
        "per_repo": per_repo,
        "a3_intent": {
            "intent_distribution": intent_distribution(all_records),
            "discriminability": disc,
            "confidence_overall": confidence_stats([r.confidence for r in all_records]),
            "confidence_by_intent": confidence_by_intent(all_records),
        },
        "a4_success": {
            "success_distribution": success_distribution(all_records),
            "consistency_with_repo_semantics": consistency,
        },
        "subtask_distribution": subtask_distribution(all_records),
        "a4_alignment": alignment_section(sidecar),
        "notes": notes,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    """把汇总 dict 渲染成可读 Markdown。"""
    lines: list[str] = ["# A3/A4 机读侧汇总", ""]
    lines.append(f"- schema：`{summary['schema_version']}`")
    lines.append(f"- repo 数：{summary['n_repos']}｜标注总数：{summary['n_records']}")
    lines.append("")

    a3 = summary["a3_intent"]
    disc = a3["discriminability"]
    lines.append("## A3 意图（判别力）")
    lines.append(f"- 预测类别分布：`{json.dumps(disc['predicted_class_distribution'], ensure_ascii=False)}`")
    lines.append(f"- 期望类别分布（按 repo 语义）：`{json.dumps(disc['expected_class_distribution'], ensure_ascii=False)}`")
    lines.append(f"- `place` 被预测次数：{disc['place_predicted_times']}（期望 {disc['place_expected_count']}）"
                 f"｜完全未被预测：{disc['place_never_predicted']}")
    lines.append(f"- κ（预测 vs repo 语义）：{disc['kappa_vs_repo_semantics']:.4f}"
                 f"｜退化：{disc['kappa_degenerate']}")
    lines.append(f"  - {disc['kappa_note']}")
    lines.append("- 置信度（整体）："
                 f"`{json.dumps(a3['confidence_overall'], ensure_ascii=False)}`")
    for intent, stats in a3["confidence_by_intent"].items():
        lines.append(f"  - intent={intent}：`{json.dumps(stats, ensure_ascii=False)}`")
    lines.append("")

    lines.append("## A4 成功（分布与语义一致性）")
    lines.append(f"- 整体：`{json.dumps(summary['a4_success']['success_distribution'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("| repo | 期望意图 | 条数 | success True | False | None | True 比例 | 全 True | 全 False |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for item in summary["a4_success"]["consistency_with_repo_semantics"]:
        ratio = item["success_true"] / item["n_records"] if item["n_records"] else 0.0
        lines.append(
            f"| {item['repo']} | {item['expected_intent']} | {item['n_records']} | "
            f"{item['success_true']} | {item['success_false']} | {item['success_none']} | "
            f"{ratio:.3f} | {item['all_success_true']} | {item['all_success_false']} |"
        )
    lines.append("")

    lines.append("## subtask 分布")
    lines.append(f"- `{json.dumps(summary['subtask_distribution'], ensure_ascii=False)}`")
    lines.append("")

    align = summary["a4_alignment"]
    lines.append("## A4 口径对齐（物理侧 vs VLM success）")
    lines.append(f"- 状态：**{align['status']}**（未产出混淆矩阵/一致率）")
    lines.append(f"- 原因：{align['reason']}")
    lines.append(f"- 旁挂数据可用：{align['sidecar_data_available']}")
    lines.append("- 仓库已有原语：")
    for primitive in align["existing_primitives"]:
        lines.append(f"  - {primitive}")
    lines.append("- 缺什么：")
    for gap in align["missing"]:
        lines.append(f"  - {gap}")
    lines.append("")

    lines.append("## 附：逐 repo 明细")
    for repo, block in summary["per_repo"].items():
        lines.append(
            f"- **{repo}**（期望 {block['expected_intent']}）："
            f"intent `{json.dumps(block['intent_distribution'], ensure_ascii=False)}`，"
            f"success `{json.dumps(block['success_distribution'], ensure_ascii=False)}`，"
            f"subtask `{json.dumps(block['subtask_distribution'], ensure_ascii=False)}`"
        )
    if summary.get("notes"):
        lines.append("")
        lines.append("## 备注")
        for note in summary["notes"]:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def load_records_by_repo(
    label_root: str | Path, repos: Sequence[str] | None = None
) -> dict[str, list[LabelRecord]]:
    """扫描标注根目录，读回每个含 `labels.jsonl` 的子目录。

    - `repos` 给定时只读这些子目录（顺序即给定顺序）；
    - 否则自动发现，但**跳过** `DRY_RUN_PREFIX` 开头的 dry-run 桩产物。
    """
    root = Path(label_root).expanduser()
    result: dict[str, list[LabelRecord]] = {}
    if not root.is_dir():
        return result
    if repos is not None:
        candidates = [root / name for name in repos]
    else:
        candidates = [
            child for child in sorted(root.iterdir()) if not child.name.startswith(DRY_RUN_PREFIX)
        ]
    for child in candidates:
        labels = child / "labels.jsonl"
        if labels.is_file():
            result[child.name] = read_label_records(labels)
    return result


def write_summary(summary: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    """把汇总写成 `summary.json` + `summary.md`，返回两者路径。"""
    target = Path(out_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "summary.json"
    md_path = target / "summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, md_path


def verify_sidecar_alignment(
    sidecar_path: str | Path, positions_by_episode: Mapping[int, Sequence[float]]
) -> dict[str, Any]:
    """薄封装：用 `load_sidecar.verify_alignment` 校验旁挂与数据集逐帧对齐。

    仅供 CLI/人工复核使用；判定逻辑一律复用仓库既有实现，本函数不新增判据。
    """
    rows = read_load_sidecar(sidecar_path)
    report = verify_alignment(rows, positions_by_episode)
    return {
        "ok": report.ok,
        "episodes": len(report.episodes),
        "max_abs_diff": max((e.max_abs_diff for e in report.episodes), default=0.0),
    }
