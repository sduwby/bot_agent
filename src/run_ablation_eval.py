#!/usr/bin/env python
"""P0 感知消融的**评测** CLI（B0/B1/B2）：从磁盘 checkpoint 重载策略 → 实时 rollout → ΔGrasp。

训练侧（云上 A10）只产 checkpoint、不做评估，本脚本补上"训练之后拿分数"这一段：
对每个变体，用 `ablation_eval` 把仿真观测还原成与训练同序的额外观测通道，交给
`visual_act.evaluate_visual_act` 跑固定 seed 评测，最后用 `depth_ablation.summarize_ablation`
聚合成 ΔGrasp 对照表。

用法::

    # 只看计划（不需要 GPU / ManiSkill / pyarrow）
    python src/run_ablation_eval.py --dry-run

    # 真跑（需要 GPU + ManiSkill + 各变体数据集）
    python src/run_ablation_eval.py --root runs/week38/ablation_smoke `
      --dataset-root runs/week38/ablation_data `
      --variants B0,B1,B2 --tasks PickCube-v1,PushCube-v1 `
      --seeds 0,1,2,3,4 --device cuda

**前置**：`<root>/<variant>/<task>/<checkpoint-subdir>` 要有可加载的 checkpoint，
`<dataset-root>/<variant>` 要有变体数据集（用来取夹爪开合范围，评测要与训练同口径）。

**口径警告**：报告会原样带上 `depth_ablation.REDUNDANCY_CAVEAT` —— 在"B0 = 现状
（RGB + robot_state + env_state）"口径下，B1/B2 的列是 env_state 的确定性函数，
ΔGrasp 原则上只落在噪声内；此表只能作冗余性对照/噪声探针，不能论证几何信号的价值。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ablation_eval import (  # noqa: E402
    AblationEvalError,
    build_variant_result,
    closure_range_from_dataset,
    evaluation_episodes,
    make_oracle_provider,
)
from depth_ablation import summarize_ablation  # noqa: E402
from real_geometry import variant_feature_names  # noqa: E402

#: 默认根目录：第 38 周 P0 在云上产出的 6 个冒烟 checkpoint 所在处。
DEFAULT_ROOT = "runs/week38/ablation_smoke"

#: 本脚本评测的是 **`extra_only` 口径**（B0 完全无 ENV），所以 `depth_ablation.REDUNDANCY_CAVEAT`
#: （"B0 已含 env_state ⇒ B1/B2 的列是它的确定性函数"）**不适用**，必须换文案，否则报告自相矛盾。
EXTRA_ONLY_CAVEAT = (
    "B0 = extra_only + 空额外列 ⇒ 策略**完全无 ENV 输入**；B1/B2 在同一基线上追加 oracle 特权列。"
    "因此 B1/B2 相对 B0 的差值**同时含两件事**：① 多出的信息；② ACT 因为多了 ENV 通道而多一个 1D 输入"
    "token（结构变化）——不能把差值直接读成\"几何/深度信号的价值\"。另外 week15-cam 的 `is_grasped` "
    "从未为 1，故 B2 的 `load_gripper_permille`/`contact_flag` 两列**恒为 0、不承载信息**。"
)
DEFAULT_VARIANTS = "B0,B1,B2"
DEFAULT_TASKS = "PickCube-v1,PushCube-v1"
DEFAULT_SEEDS = "0,1,2,3,4"
#: 变体数据集根：用于取 open/closed 宽度（与训练同源，避免硬编码常数）。
DEFAULT_DATASET_ROOT = "runs/week38/ablation_data"
DEFAULT_CHECKPOINT_SUBDIR = "best-validation"
DEFAULT_BASELINE = "B0"


def _parse_csv(text: str) -> list[str]:
    """逗号分隔 → 去空白、去空的列表（保序）。"""
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_seeds(text: str) -> list[int]:
    """逗号分隔的种子 → 非负整数列表（重复/非法即报错，避免评测悄悄改变样本集）。"""
    seeds: list[int] = []
    for token in _parse_csv(text):
        try:
            value = int(token)
        except ValueError as exc:
            raise SystemExit(f"--seeds 含非整数 {token!r}") from exc
        if value < 0:
            raise SystemExit(f"--seeds 必须是非负整数，实际 {value}")
        seeds.append(value)
    if not seeds:
        raise SystemExit("--seeds 不能为空")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds 不能重复")
    return seeds


def render_eval_plan(
    variants: list[str],
    tasks: list[str],
    seeds: list[int],
    *,
    root: Path,
    dataset_root: Path,
    checkpoint_subdir: str,
) -> str:
    """渲染"将要做什么"的计划（纯函数，可离线测试）。

    dry-run 需要它的原因：**没有 GPU / ManiSkill 的机器也要能核对列序与组合**——
    列序一错，真跑的结论就是废的，所以先把每个变体的 provider 与列序打印出来。
    """
    lines = [
        f"评测根目录：{root}",
        f"数据集根目录：{dataset_root}（取 open/closed 宽度）",
        f"checkpoint 子目录：{checkpoint_subdir}",
        f"任务：{tasks}｜种子：{seeds}｜组合：{len(variants)}×{len(tasks)}×{len(seeds)}="
        f"{len(variants) * len(tasks) * len(seeds)} 条 episode",
        "",
        "变体     provider                    列序（= real_geometry.variant_feature_names）",
    ]
    for variant in variants:
        names = variant_feature_names(variant)
        provider = "None（无额外通道）" if not names else "oracle（特权量，与训练同序）"
        columns = ",".join(names) if names else "（无）"
        lines.append(
            f"{variant:<8}{provider:<28}[{columns}]"
        )
        lines.append(
            f"        checkpoint：{root / variant}/<task>/{checkpoint_subdir}"
            f"｜数据集：{dataset_root / variant}"
        )
    return "\n".join(lines)


def _write_markdown(report: object, path: Path) -> None:
    """把概要行写成 markdown 报告（与分析产物 eval_report.json 并存）。"""
    body = ["# P0 感知消融评测（ΔGrasp）", "", *[f"- {line}" for line in report.summary_lines()], ""]
    path.write_text("\n".join(body), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0 感知消融评测（B0/B1/B2，ΔGrasp 对照表）")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="checkpoint 根目录（<root>/<variant>/<task>）")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS, help="逗号分隔的变体，必须含基线 B0")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="逗号分隔的任务")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="逗号分隔的固定种子")
    parser.add_argument("--device", default="cuda", help="评测设备（默认 cuda）")
    parser.add_argument("--resolution", type=int, default=224, help="视觉输入分辨率")
    parser.add_argument("--checkpoint-subdir", default=DEFAULT_CHECKPOINT_SUBDIR,
                        help="每个任务下的 checkpoint 子目录名")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                        help="变体数据集根目录（取夹爪开合范围）")
    parser.add_argument("--out", default=None, help="报告输出目录（默认 <root>/eval_report）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划（变体×任务×种子、provider 与列序），不加载策略、不建 env")
    args = parser.parse_args(argv)

    variants = _parse_csv(args.variants)
    tasks = _parse_csv(args.tasks)
    seeds = _parse_seeds(args.seeds)
    if not variants:
        parser.error("--variants 不能为空")
    if not tasks:
        parser.error("--tasks 不能为空")
    if DEFAULT_BASELINE not in variants:
        parser.error(f"--variants 必须含基线 {DEFAULT_BASELINE}（否则没有 ΔGrasp 的分母）")

    root = Path(args.root).expanduser()
    dataset_root = Path(args.dataset_root).expanduser()
    out_dir = Path(args.out).expanduser() if args.out else root / "eval_report"

    print(render_eval_plan(
        variants, tasks, seeds,
        root=root, dataset_root=dataset_root, checkpoint_subdir=args.checkpoint_subdir,
    ))
    if args.dry_run:
        print("[评测] dry-run 结束（未加载策略、未建 env、未落盘）")
        return 0

    # 真跑：惰性导入 —— 本机无 GPU/ManiSkill 时 dry-run 路径不受影响。
    from visual_act import evaluate_visual_act

    results = []
    for variant in variants:
        try:
            open_width, closed_width = closure_range_from_dataset(dataset_root / variant)
        except AblationEvalError as exc:
            raise SystemExit(f"[评测] 变体 {variant} 无法确定夹爪开合范围：{exc}") from exc
        # B0 没有额外列 → provider=None（与训练侧 extra_features=() 一致）；
        # B1/B2 用 oracle provider，列序由 variant_feature_names 单点决定。
        provider = (
            make_oracle_provider(variant, open_width=open_width, closed_width=closed_width)
            if variant_feature_names(variant)
            else None
        )
        print(f"[评测] {variant} 开合范围 open={open_width:.6g} closed={closed_width:.6g}"
              f" provider={'oracle' if provider else 'None'}")
        summary = evaluate_visual_act(
            root / variant,
            seeds=seeds,
            device=args.device,
            resolution=args.resolution,
            tasks=tuple(tasks),
            checkpoint_subdir=args.checkpoint_subdir,
            extra_provider=provider,
        )
        results.append(build_variant_result(variant, evaluation_episodes(summary)))

    report = summarize_ablation(results, baseline=DEFAULT_BASELINE, caveat=EXTRA_ONLY_CAVEAT)
    for line in report.summary_lines():
        print(f"[评测] {line}")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_report.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path = out_dir / "eval_report.md"
    _write_markdown(report, markdown_path)
    print(f"[评测] 报告 → {json_path}")
    print(f"[评测]        → {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
