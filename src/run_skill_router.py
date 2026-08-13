"""第 20 周：分层策略路由固定 seed 评测 CLI。

用法示例：

    python src/run_skill_router.py \\
        --route visual_act,state_act,scripted \\
        --visual-act-checkpoint runs/maniskill_act/week17-pick-grasp-30k \\
        --state-act-checkpoint runs/maniskill_act/week13-dagger-train-r3 \\
        --tasks PickCube-v1 --seeds 500-529 \\
        --device mps --output-dir runs/maniskill_act/week20-router
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from skill_router import (
    PolicySource,
    SkillRouteConfig,
    SkillRouterError,
    bc_source,
    evaluate_routes,
    scripted_source,
    smolvla_source,
    state_act_source,
    visual_act_source,
)

ROUTE_KINDS = ("scripted", "bc", "state_act", "visual_act", "smolvla")


def _parse_seeds(text: str) -> list[int]:
    """解析 "500-529" 或 "500" 形式的 seed 表达式。"""
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


def _parse_route(
    text: str,
    *,
    state_act_checkpoint: Path | None,
    visual_act_checkpoint: Path | None,
    bc_checkpoint: Path | None,
    smolvla_model: str | None,
    device: str,
) -> dict[str, PolicySource]:
    """解析逗号分隔的策略种类为有序 source 列表。"""
    kinds = [part.strip() for part in text.split(",") if part.strip()]
    if not kinds:
        raise ValueError("--route 不能为空")
    sources: list[PolicySource] = []
    for kind in kinds:
        if kind not in ROUTE_KINDS:
            raise ValueError(
                f"不支持的策略种类 {kind!r}，可选 {ROUTE_KINDS}"
            )
        if kind == "scripted":
            sources.append(scripted_source())
        elif kind == "state_act":
            if state_act_checkpoint is None:
                raise ValueError("--route 含 state_act 但未提供 --state-act-checkpoint")
            sources.append(
                state_act_source(state_act_checkpoint, device=device)
            )
        elif kind == "visual_act":
            if visual_act_checkpoint is None:
                raise ValueError(
                    "--route 含 visual_act 但未提供 --visual-act-checkpoint"
                )
            sources.append(
                visual_act_source(visual_act_checkpoint, device=device)
            )
        elif kind == "bc":
            if bc_checkpoint is None:
                raise ValueError("--route 含 bc 但未提供 --bc-checkpoint")
            sources.append(bc_source(bc_checkpoint, device=device))
        else:
            if smolvla_model is None:
                raise ValueError("--route 含 smolvla 但未提供 --smolvla-model")
            sources.append(
                smolvla_source(smolvla_model, device=device)
            )
    return sources


def _build_routes(
    tasks: list[str],
    sources: list[PolicySource],
) -> dict[str, SkillRouteConfig]:
    routes: dict[str, SkillRouteConfig] = {}
    for env_id in tasks:
        routes[env_id] = SkillRouteConfig(
            env_id=env_id, sources=tuple(sources)
        )
    return routes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        required=True,
        help="逗号分隔策略顺序，如 visual_act,state_act,scripted",
    )
    parser.add_argument(
        "--tasks",
        default="PickCube-v1,PushCube-v1",
        help="逗号分隔任务（默认 PickCube-v1,PushCube-v1）",
    )
    parser.add_argument(
        "--seeds",
        default="500-529",
        help="seed 表达式，如 500-529 或 500,600（默认 500-529）",
    )
    parser.add_argument(
        "--state-act-checkpoint",
        type=Path,
        default=None,
        help="状态 ACT checkpoint 目录",
    )
    parser.add_argument(
        "--visual-act-checkpoint",
        type=Path,
        default=None,
        help="视觉 ACT checkpoint 目录",
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=None,
        help="BC checkpoint 文件",
    )
    parser.add_argument(
        "--smolvla-model",
        default=None,
        help="SmolVLA 模型 id（可选位，如 lerobot/smolvla_base）",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="推理设备（mps/cuda/cpu，默认 mps）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="报告输出目录（必须为空或不存在）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="单 episode 最大步数（默认 50）",
    )
    args = parser.parse_args(argv)

    try:
        seeds = _parse_seeds(args.seeds)
        tasks = [part.strip() for part in args.tasks.split(",") if part.strip()]
        sources = _parse_route(
            args.route,
            state_act_checkpoint=args.state_act_checkpoint,
            visual_act_checkpoint=args.visual_act_checkpoint,
            bc_checkpoint=args.bc_checkpoint,
            smolvla_model=args.smolvla_model,
            device=args.device,
        )
        routes = {
            env_id: SkillRouteConfig(
                env_id=env_id,
                sources=tuple(sources),
                max_steps=args.max_steps,
            )
            for env_id in tasks
        }
    except (ValueError, SkillRouterError) as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            print(f"输出目录必须为空或不存在：{output_dir}", file=sys.stderr)
            return 2

    has_visual = any(source.input_mode == "visual" for source in sources)
    try:
        if has_visual:
            from lerobot_pipeline import make_render_env

            env_factory = make_render_env
        else:
            from maniskill_intro import make_maniskill_env

            env_factory = make_maniskill_env
        summary = evaluate_routes(
            routes,
            seeds,
            env_factory=env_factory,
            output_dir=output_dir,
        )
    except SkillRouterError as exc:
        print(f"评测失败：{exc}", file=sys.stderr)
        return 1

    for task in summary.tasks:
        print(
            f"{task.env_id:<12} 成功率 {task.success_rate:.1%} "
            f"({task.successes}/{task.episodes}) "
            f"mean_steps={task.mean_steps:.1f} "
            f"fallback={task.fallback_counts} exhausted={task.exhausted}"
        )
        print(f"  策略步数占比: {task.policy_step_fractions}")
    total_successes = sum(task.successes for task in summary.tasks)
    total_episodes = sum(task.episodes for task in summary.tasks)
    print(
        f"总计 {total_successes}/{total_episodes} "
        f"({total_successes / total_episodes:.1%}) "
        f"→ {output_dir / 'router_report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
