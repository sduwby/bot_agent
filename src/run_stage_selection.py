"""第 22 周：阶段评测与选型 CLI（门禁 C）。

用法示例：

    python src/run_stage_selection.py \\
        --candidate scripted:scripted \\
        --candidate visual:visual_act:runs/maniskill_act/week17-c8-20k \\
        --candidate route:route:visual_act,scripted:runs/maniskill_act/week17-c8-20k \\
        --output-dir runs/maniskill_selection/week22 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stage_selection import (
    StageSelectionConfig,
    StageSelectionError,
    build_model_card,
    compute_gate_c,
    evaluate_candidates,
    parse_candidate,
)


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            try:
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError(f"seed 区间非法：{part!r}") from exc
            if start < 0 or end < start:
                raise ValueError(f"seed 区间非法：{part!r}")
            seeds.extend(range(start, end + 1))
        else:
            try:
                seeds.append(int(part))
            except ValueError as exc:
                raise ValueError(f"seed 非法：{part!r}") from exc
    return seeds


def _parse_candidate_text(text: str):
    """解析 ``name:kind:rest`` 形式；route 候选 rest 为 ``策略串:checkpoint``。"""
    parts = text.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"候选格式应为 name:kind[:rest]，实际：{text!r}")
    name, kind = parts[0].strip(), parts[1].strip()
    rest = parts[2].strip() if len(parts) > 2 else ""
    if kind == "route":
        if ":" not in rest:
            raise ValueError(
                "route 候选格式应为 name:route:策略串:checkpoint，"
                f"实际：{text!r}"
            )
        route, checkpoint = rest.rsplit(":", 1)
        return name, kind, route.strip(), Path(checkpoint).expanduser().resolve()
    return name, kind, rest, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        dest="candidates",
        help=(
            "候选：name:kind[:checkpoint]；route 候选为 "
            "name:route:策略串:checkpoint（可重复）"
        ),
    )
    parser.add_argument(
        "--seeds-in", default="500-529", help="分布内 seed（默认 500-529）"
    )
    parser.add_argument(
        "--seeds-ood", default="600-629", help="分布外 seed（默认 600-629）"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--gate-threshold", type=float, default=0.60)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-gate-exit",
        action="store_true",
        help="门禁未通过时不使用非零退出码",
    )
    args = parser.parse_args(argv)

    try:
        from stage_selection import PolicyCandidate

        candidates: list[PolicyCandidate] = []
        for text in args.candidates:
            name, kind, rest, checkpoint = _parse_candidate_text(text)
            if kind == "route":
                candidates.append(
                    PolicyCandidate(
                        name=name,
                        kind="route",
                        input_mode="state",
                        route=rest,
                        checkpoint_dir=checkpoint,
                    )
                )
            else:
                candidates.append(parse_candidate(f"{name}:{kind}:{rest}"))
        config = StageSelectionConfig(
            output_dir=args.output_dir,
            seeds_in=tuple(_parse_seeds(args.seeds_in)),
            seeds_ood=tuple(_parse_seeds(args.seeds_ood)),
            device=args.device,
            resolution=args.resolution,
            gate_threshold=args.gate_threshold,
        )
    except (ValueError, StageSelectionError) as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"输出目录必须为空或不存在：{output_dir}", file=sys.stderr)
        return 2

    try:
        results = evaluate_candidates(candidates, config)
        verdict = compute_gate_c(results, threshold=args.gate_threshold)
        card = build_model_card(
            results, verdict, output_dir=output_dir
        )
    except (ValueError, StageSelectionError, OSError) as exc:
        print(f"选型评测失败：{exc}", file=sys.stderr)
        return 1

    for result in results:
        in_split = result.in_distribution
        ood_split = result.out_of_distribution
        status = "可用" if result.available else f"不可用（{result.note}）"
        print(f"[{result.name}] {status}")
        if in_split is not None:
            print(
                f"  分布内 {in_split.overall_success_rate:.1%} "
                f"nonfinite={in_split.nonfinite} exceptions={in_split.exceptions}"
            )
        if ood_split is not None:
            print(
                f"  分布外 {ood_split.overall_success_rate:.1%} "
                f"nonfinite={ood_split.nonfinite} exceptions={ood_split.exceptions}"
            )
    print(
        f"门禁 C：{'通过' if verdict.passed else '未通过'} "
        f"（视觉 {verdict.visual_rate:.1%}，{verdict.notes}）"
    )
    print(f"模型卡 → {output_dir / 'model_card.json'}")
    print(f"选型报告 → {output_dir / 'selection_report.md'}")
    if not verdict.passed and not args.no_gate_exit:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
