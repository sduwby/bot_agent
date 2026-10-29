#!/usr/bin/env python
"""P0 感知消融 CLI（B0–B3）：训练 + 评测 + ΔGrasp 归因表（第 38 周 P0 / 1.6）。

**实验口径（用户决策 2026-10-11）：B0 = 现状**（`RGB + robot_state(5) + env_state(13)`），
B1–B3 在其上追加 `ablation_features` 的额外列。**该口径的固有局限见
`depth_ablation.REDUNDANCY_CAVEAT`（B1/B2 的列是 env_state 的确定性函数 → 信息增量为 0）
——报告里会原样带上这句话，不允许只看数字不看前提。**

用法::

    # 只看计划与维度门禁（不训练、不落盘）
    python src/run_depth_ablation.py --dry-run --variants B0,B1,B2

    # 真跑（每个变体要有自己那份数据集：ENV 列数不同）
    python src/run_depth_ablation.py `
      --dataset B0=runs/maniskill_lerobot/week15-cam `
      --dataset B1=runs/ablation/B1 --dataset B2=runs/ablation/B2 `
      --out runs/week38/ablation --episodes 10 --steps 20000 --device cuda

**前置**（本机没有 GPU/仿真，故不做训练）：每个变体的数据集要把对应额外列**追加到
`observation.environment_state` 尾部**，维度必须等于 `13 + len(该变体的列)` —— 本脚本在开训前
逐个校验，列数不符直接报错（列数不符会让策略读到错位向量，指标只会"变差"而不报错）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from depth_ablation import (  # noqa: E402
    REDUNDANCY_CAVEAT,
    TARGET_DELTA_GRASP_PP,
    AblationError,
    AblationVariant,
    VariantResult,
    build_plan,
    read_env_state_dim,
    run_ablation,
    validate_dataset_dim,
    write_ablation_report,
)
from real_geometry import VARIANT_ORDER  # noqa: E402


def parse_dataset_args(items: list[str]) -> dict[str, Path]:
    """解析 ``--dataset B1=<路径>``；重复给同一变体即报错（不许静默覆盖）。"""
    datasets: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--dataset 需要 NAME=PATH 形式，实际 {item!r}")
        name, _, raw = item.partition("=")
        name = name.strip()
        if name in datasets:
            raise SystemExit(f"变体 {name} 的数据集重复指定")
        datasets[name] = Path(raw.strip()).expanduser()
    return datasets


def render_plan(plan: tuple[AblationVariant, ...]) -> str:
    lines = ["变体   追加列（进 ENV 尾部）                          ENV 维度"]
    for item in plan:
        columns = ",".join(item.extra_features) if item.extra_features else "（无，基线）"
        lines.append(f"{item.name:<7}{columns:<45}{item.env_state_dim:>6}")
    return "\n".join(lines)


def _default_train_fn(variant: AblationVariant, dataset: Path) -> float:
    """默认训练：走 `visual_act` 的真实训练入口（**需 GPU / mani_skill**，本机不可跑）。"""
    try:
        import visual_act
    except (ImportError, OSError) as exc:  # pragma: no cover - 依赖缺失时的明确提示
        raise AblationError(
            f"训练变体 {variant.name} 需要 visual_act 及其依赖（torch/mani_skill）：{exc}"
        ) from exc
    raise AblationError(
        f"默认训练路径未在本仓接通（需 GPU+仿真环境）；变体 {variant.name} 请用 "
        "visual_act 的训练入口并传入 extra_features="
        f"{list(variant.extra_features)}，本机只做编排与维度门禁"
    )


def _default_eval_fn(variant: AblationVariant, dataset: Path) -> VariantResult:  # pragma: no cover
    raise AblationError(
        f"默认评测路径未在本仓接通（需 GPU+仿真环境）；变体 {variant.name} 的评测应由调用方注入"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="P0 感知消融 B0–B3（ΔGrasp 归因表）")
    parser.add_argument("--variants", default=",".join(VARIANT_ORDER),
                        help=f"逗号分隔的变体（必须是 {VARIANT_ORDER} 的子集，且含 B0）")
    parser.add_argument("--dataset", action="append", default=[],
                        help="NAME=PATH，可重复；每个变体的数据集 ENV 列数不同，必须各给一份")
    parser.add_argument("--out", default="runs/week38/ablation", help="报告输出目录")
    parser.add_argument("--baseline", default="B0", help="作为 ΔGrasp 分母的基线变体")
    parser.add_argument("--target-pp", type=float, default=TARGET_DELTA_GRASP_PP,
                        help="A2 判据：相对基线提升的百分点")
    parser.add_argument("--episodes", type=int, default=10, help="每变体评测 episode 数")
    parser.add_argument("--steps", type=int, default=20000, help="每变体训练步数")
    parser.add_argument("--device", default="cuda", help="训练设备（默认 cuda）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划与维度门禁，不训练、不落盘")
    args = parser.parse_args()

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    plan = build_plan(variants)
    print(f"[消融] 口径：B0 = 现状（RGB + robot_state + env_state）｜判据 ΔGrasp ≥ {args.target_pp:g}pt")
    print(render_plan(plan))
    print(f"[消融] 注意：{REDUNDANCY_CAVEAT}")

    datasets = parse_dataset_args(args.dataset)
    if args.dry_run:
        if datasets:
            for item in plan:
                path = datasets.get(item.name)
                if path is None:
                    print(f"  [跳过] {item.name} 未提供数据集")
                    continue
                validate_dataset_dim(item, read_env_state_dim(path))
                print(f"  [维度门禁] {item.name} → {path} 通过")
        else:
            print("[消融] --dry-run 未提供数据集，跳过维度门禁")
        print("[消融] dry-run 结束（未训练、未落盘）")
        return 0

    missing = [item.name for item in plan if item.name not in datasets]
    if missing:
        parser.error(f"缺少这些变体的数据集：{missing}（用 --dataset NAME=PATH 指定）")
    for item in plan:  # 开训前的维度门禁：列数不符会让策略读到错位向量
        validate_dataset_dim(item, read_env_state_dim(datasets[item.name]))

    report = run_ablation(
        plan,
        datasets,
        train_fn=_default_train_fn,
        eval_fn=_default_eval_fn,
        baseline=args.baseline,
        target_pp=args.target_pp,
    )
    json_path, markdown_path = write_ablation_report(report, Path(args.out))
    for line in report.summary_lines():
        print(f"[消融] {line}")
    print(f"[消融] 报告 → {json_path}")
    print(f"[消融]        → {markdown_path}")
    return 0 if report.verdict != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
