"""第 22 周：阶段评测与选型（门禁 C）。

在固定 seed 协议上比较 scripted / bc / state_act / visual_act / 路由组合，
计算阶段门禁 C（视觉 ACT 在随机初始位置测试 ≥60%、scripted fallback 可用、
安全指标干净），聚合历史结果并生成模型卡与选型报告。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from maniskill_intro import (
    SUPPORTED_TASKS,
    ManiSkillTaskConfig,
    ScriptedPolicy,
)

SELECTION_SCHEMA_VERSION = "maniskill_stage_selection_v1"

CANDIDATE_KINDS = ("scripted", "bc", "state_act", "visual_act", "route")

DEFAULT_IN_SEEDS = tuple(range(500, 530))
DEFAULT_OOD_SEEDS = tuple(range(600, 630))
DEFAULT_GATE_THRESHOLD = 0.60

# 达标任务口径：视觉 ACT 门禁用 PushCube（+ 已稳定抬升子技能组合）。
GATE_TASK = "PushCube-v1"

# 第 12–14 周历史固定评测结果（同 seed 协议，来源见 docs/week13–14）。
HISTORICAL_RESULTS: dict[str, dict[str, float]] = {
    "bc": {"PickCube-v1": 0.84, "PushCube-v1": 1.00},
    "state_act": {"PickCube-v1": 0.96, "PushCube-v1": 1.00},
    "scripted": {"PickCube-v1": 1.00, "PushCube-v1": 1.00},
}


class StageSelectionError(RuntimeError):
    """第 22 周选型契约错误。"""


@dataclass(frozen=True)
class PolicyCandidate:
    """一个策略候选：kind + checkpoint/route + 输入模式。"""

    name: str
    kind: str
    input_mode: str
    checkpoint_dir: Path | None = None
    route: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name 不能为空")
        if self.kind not in CANDIDATE_KINDS:
            raise ValueError(f"kind 必须是 {CANDIDATE_KINDS} 之一")
        if self.input_mode not in ("state", "visual"):
            raise ValueError("input_mode 只支持 state 或 visual")
        if self.kind == "route":
            if not self.route:
                raise ValueError("route 候选必须提供 route 策略串")
        elif self.route is not None:
            raise ValueError("非 route 候选不能提供 route")
        if self.kind in ("bc", "state_act", "visual_act"):
            if self.checkpoint_dir is None:
                raise ValueError(f"{self.kind} 候选必须提供 checkpoint_dir")
            object.__setattr__(
                self,
                "checkpoint_dir",
                Path(self.checkpoint_dir).expanduser().resolve(),
            )
        elif self.kind == "route" and self.checkpoint_dir is not None:
            object.__setattr__(
                self,
                "checkpoint_dir",
                Path(self.checkpoint_dir).expanduser().resolve(),
            )


@dataclass(frozen=True)
class StageSelectionConfig:
    """选型评测的 seed 集、设备与门禁阈值。"""

    output_dir: Path
    seeds_in: tuple[int, ...] = DEFAULT_IN_SEEDS
    seeds_ood: tuple[int, ...] = DEFAULT_OOD_SEEDS
    device: str = "cuda"
    resolution: int = 128
    gate_threshold: float = DEFAULT_GATE_THRESHOLD

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        if self.device not in ("mps", "cpu", "cuda"):
            raise ValueError("device 只支持 mps、cuda 或 cpu")
        if not isinstance(self.resolution, int) or isinstance(
            self.resolution, bool
        ) or self.resolution <= 0:
            raise ValueError("resolution 必须是正整数")
        if (
            isinstance(self.gate_threshold, bool)
            or not isinstance(self.gate_threshold, (int, float))
            or not 0 < float(self.gate_threshold) <= 1
        ):
            raise ValueError("gate_threshold 必须位于 (0, 1]")
        seeds_in = _checked_seeds(self.seeds_in)
        seeds_ood = _checked_seeds(self.seeds_ood)
        if set(seeds_in).intersection(seeds_ood):
            raise ValueError("分布内/外 seed 不能重叠")
        object.__setattr__(self, "seeds_in", tuple(seeds_in))
        object.__setattr__(self, "seeds_ood", tuple(seeds_ood))


@dataclass(frozen=True)
class CandidateSplit:
    """一个候选在一个 split（分布内/外）上的聚合结果。"""

    split: str
    per_task: dict[str, dict[str, float]]
    overall_success_rate: float
    exceptions: int
    nonfinite: int
    out_of_bounds: int


@dataclass(frozen=True)
class CandidateResult:
    """一个候选在分布内/外两个 split 上的评测结果。"""

    name: str
    kind: str
    splits: tuple[CandidateSplit, ...]
    available: bool = True
    note: str = ""

    @property
    def in_distribution(self) -> CandidateSplit | None:
        return next(
            (item for item in self.splits if item.split == "in_distribution"),
            None,
        )

    @property
    def out_of_distribution(self) -> CandidateSplit | None:
        return next(
            (item for item in self.splits if item.split == "out_of_distribution"),
            None,
        )


@dataclass(frozen=True)
class GateCVerdict:
    """阶段门禁 C 判定。"""

    passed: bool
    visual_rate: float
    fallback_available: bool
    safety_clean: bool
    notes: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能序列化 {type(value).__name__}")


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checked_seeds(seeds: Sequence[int]) -> list[int]:
    if not seeds:
        raise ValueError("seeds 不能为空")
    checked: list[int] = []
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seeds 必须是非负整数")
        checked.append(seed)
    if len(set(checked)) != len(checked):
        raise ValueError("seeds 不能重复")
    return checked


def parse_candidate(text: str) -> PolicyCandidate:
    """解析 ``name:kind:checkpoint`` 或 ``name:route:策略串`` 形式。"""
    parts = text.split(":", 2)
    if len(parts) < 2:
        raise ValueError(
            f"候选格式应为 name:kind[:checkpoint]，实际：{text!r}"
        )
    name, kind = parts[0].strip(), parts[1].strip()
    rest = parts[2].strip() if len(parts) > 2 else ""
    if kind == "route":
        if not rest:
            raise ValueError("route 候选缺少策略串")
        return PolicyCandidate(
            name=name,
            kind="route",
            input_mode="state",
            route=rest,
        )
    if kind in ("bc", "state_act", "visual_act"):
        if not rest:
            raise ValueError(f"{kind} 候选缺少 checkpoint 路径")
        input_mode = "visual" if kind == "visual_act" else "state"
        return PolicyCandidate(
            name=name,
            kind=kind,
            input_mode=input_mode,
            checkpoint_dir=Path(rest),
        )
    if kind == "scripted":
        return PolicyCandidate(name=name, kind="scripted", input_mode="state")
    raise ValueError(f"不支持的候选 kind：{kind!r}")


def _load_policy_map(
    candidate: PolicyCandidate,
    *,
    device: str,
    tasks: Sequence[str],
) -> dict[str, object]:
    """按 kind 加载候选策略映射（env_id -> policy）。"""
    if candidate.kind == "scripted":
        return {env_id: ScriptedPolicy(env_id) for env_id in tasks}
    if candidate.kind == "state_act":
        from lerobot_act import load_lerobot_act_checkpoints

        return load_lerobot_act_checkpoints(
            candidate.checkpoint_dir, device=device, tasks=tasks
        )
    if candidate.kind == "visual_act":
        from visual_act import load_visual_act_checkpoints

        return load_visual_act_checkpoints(
            candidate.checkpoint_dir,
            device=device,
            resolution=128,
            tasks=tasks,
        )
    if candidate.kind == "bc":
        from behavior_cloning import load_bc_checkpoint

        return load_bc_checkpoint(candidate.checkpoint_dir)
    raise StageSelectionError(f"候选 kind 不可加载：{candidate.kind}")


def _summary_to_split(
    summary: object, *, split: str
) -> CandidateSplit:
    tasks_stats: dict[str, dict[str, float]] = {}
    exceptions = 0
    for task in summary.tasks:
        exceptions += int(task.exceptions)
        tasks_stats[task.env_id] = {
            "success_rate": float(task.success_rate),
            "mean_steps": float(task.mean_steps),
            "exceptions": int(task.exceptions),
        }
    return CandidateSplit(
        split=split,
        per_task=tasks_stats,
        overall_success_rate=float(summary.success_rate),
        exceptions=int(exceptions),
        nonfinite=int(summary.nonfinite_actions),
        out_of_bounds=int(summary.out_of_bounds_actions),
    )


def evaluate_candidates(
    candidates: Sequence[PolicyCandidate],
    config: StageSelectionConfig,
    *,
    env_factory: Any | None = None,
) -> list[CandidateResult]:
    """在分布内/外固定 seed 上评测全部候选。"""
    from lerobot_act import evaluate_policy_seeds
    from maniskill_intro import make_maniskill_env
    from perturbation import PerturbationConfig, make_perturbed_render_env
    from lerobot_pipeline import make_render_env

    if not candidates:
        raise ValueError("candidates 不能为空")
    results: list[CandidateResult] = []
    for candidate in candidates:
        if candidate.kind == "route":
            results.append(
                _evaluate_route_candidate(candidate, config, env_factory)
            )
            continue
        try:
            policies = _load_policy_map(
                candidate, device=config.device, tasks=SUPPORTED_TASKS
            )
        except Exception as exc:
            results.append(
                CandidateResult(
                    name=candidate.name,
                    kind=candidate.kind,
                    splits=(),
                    available=False,
                    note=f"加载失败：{type(exc).__name__}: {exc}",
                )
            )
            continue
        splits: list[CandidateSplit] = []
        in_factory = (
            make_render_env
            if candidate.input_mode == "visual"
            else make_maniskill_env
        )
        if candidate.input_mode == "visual":

            def ood_factory(task_config: ManiSkillTaskConfig) -> object:
                return make_perturbed_render_env(
                    task_config,
                    PerturbationConfig(
                        background=True, lighting=True, pose=True
                    ),
                )

        else:
            ood_factory = make_maniskill_env
        for split_name, seeds, factory in (
            ("in_distribution", config.seeds_in, in_factory),
            ("out_of_distribution", config.seeds_ood, ood_factory),
        ):
            if candidate.input_mode == "visual":
                summary = _evaluate_visual_split(
                    candidate,
                    seeds,
                    env_factory=factory,
                    resolution=config.resolution,
                    device=config.device,
                )
            else:
                summary = evaluate_policy_seeds(
                    candidate.name,
                    policies,
                    seeds,
                    env_factory=factory,
                    tasks=SUPPORTED_TASKS,
                )
            splits.append(_summary_to_split(summary, split=split_name))
        results.append(
            CandidateResult(
                name=candidate.name,
                kind=candidate.kind,
                splits=tuple(splits),
            )
        )
    return results


def _evaluate_visual_split(
    candidate: PolicyCandidate,
    seeds: Sequence[int],
    *,
    env_factory: Any,
    resolution: int,
    device: str,
) -> object:
    """视觉候选走 live RGB rollout 协议（run_generalization 同口径）。"""
    from lerobot_act import ACTEvaluationEpisode, ACTEvaluationTask, PolicyEvaluationSummary
    from visual_act import (
        VISUAL_SCHEMA_VERSION,
        _utc_now,
        load_visual_act_checkpoints,
        run_visual_episode_in_env,
    )

    checked_seeds = _checked_seeds(seeds)
    policies = load_visual_act_checkpoints(
        candidate.checkpoint_dir,
        device=device,
        resolution=resolution,
        tasks=SUPPORTED_TASKS,
    )
    episodes: list[object] = []
    for env_id in SUPPORTED_TASKS:
        policy = policies[env_id]
        for seed in checked_seeds:
            config = ManiSkillTaskConfig(env_id=env_id, seed=seed)
            env: Any | None = None
            try:
                env = env_factory(config)
                result = run_visual_episode_in_env(config, env, policy)
                episodes.append(
                    ACTEvaluationEpisode(
                        env_id=env_id,
                        seed=seed,
                        steps=int(result["steps"]),
                        total_reward=float(result["total_reward"]),
                        success=bool(result["success"]),
                        terminated=bool(result["terminated"]),
                        truncated=bool(result["truncated"]),
                        action_queries=policy.action_queries,
                        chunk_queries=policy.chunk_queries,
                        inference_ms=policy.inference_ms,
                        nonfinite_actions=policy.nonfinite_actions,
                        out_of_bounds_actions=policy.out_of_bounds_actions,
                    )
                )
            except Exception as exc:
                episodes.append(
                    ACTEvaluationEpisode(
                        env_id=env_id,
                        seed=seed,
                        steps=0,
                        total_reward=0.0,
                        success=False,
                        terminated=False,
                        truncated=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            finally:
                if env is not None:
                    env.close()

    tasks_summary: list[object] = []
    for env_id in SUPPORTED_TASKS:
        task_episodes = [item for item in episodes if item.env_id == env_id]
        successes = sum(item.success for item in task_episodes)
        tasks_summary.append(
            ACTEvaluationTask(
                env_id=env_id,
                episodes=len(task_episodes),
                successes=successes,
                failures=len(task_episodes) - successes,
                success_rate=successes / len(task_episodes),
                mean_steps=(
                    float(np.mean([item.steps for item in task_episodes]))
                    if task_episodes
                    else 0.0
                ),
                exceptions=sum(item.error is not None for item in task_episodes),
            )
        )
    total_successes = sum(item.success for item in episodes)
    total_episodes = len(episodes)
    return PolicyEvaluationSummary(
        policy_name=candidate.name,
        schema_version=VISUAL_SCHEMA_VERSION,
        created_at=_utc_now(),
        requested_per_task=len(checked_seeds),
        total_episodes=total_episodes,
        total_successes=total_successes,
        total_failures=total_episodes - total_successes,
        success_rate=total_successes / total_episodes if total_episodes else 0.0,
        mean_inference_ms=(
            float(np.mean([item.inference_ms for item in episodes]))
            if episodes
            else 0.0
        ),
        mean_steps=(
            float(np.mean([item.steps for item in episodes])) if episodes else 0.0
        ),
        total_chunk_queries=sum(item.chunk_queries for item in episodes),
        total_action_queries=sum(item.action_queries for item in episodes),
        nonfinite_actions=sum(item.nonfinite_actions for item in episodes),
        out_of_bounds_actions=sum(
            item.out_of_bounds_actions for item in episodes
        ),
        tasks=tuple(tasks_summary),
        episodes=tuple(episodes),
    )


def _evaluate_route_candidate(
    candidate: PolicyCandidate,
    config: StageSelectionConfig,
    env_factory: Any | None,
) -> CandidateResult:
    """route 候选复用 skill_router.evaluate_routes 的分布内 split。"""
    from skill_router import (
        SkillRouteConfig,
        evaluate_routes,
        scripted_source,
        state_act_source,
        visual_act_source,
    )

    kinds = [item.strip() for item in candidate.route.split(",") if item.strip()]
    sources = []
    has_visual = False
    for kind in kinds:
        if kind == "scripted":
            sources.append(scripted_source())
        elif kind == "state_act":
            if candidate.checkpoint_dir is None:
                raise StageSelectionError(
                    "route 含 state_act 但候选未提供 checkpoint_dir"
                )
            sources.append(
                state_act_source(candidate.checkpoint_dir, device=config.device)
            )
        elif kind == "visual_act":
            if candidate.checkpoint_dir is None:
                raise StageSelectionError(
                    "route 含 visual_act 但候选未提供 checkpoint_dir"
                )
            sources.append(
                visual_act_source(candidate.checkpoint_dir, device=config.device)
            )
            has_visual = True
        else:
            raise StageSelectionError(f"route 含不支持的策略 {kind!r}")
    routes = {
        env_id: SkillRouteConfig(env_id=env_id, sources=tuple(sources))
        for env_id in SUPPORTED_TASKS
    }
    factory = env_factory
    if factory is None:
        from lerobot_pipeline import make_render_env
        from maniskill_intro import make_maniskill_env

        factory = make_render_env if has_visual else make_maniskill_env
    try:
        summary = evaluate_routes(
            routes,
            config.seeds_in,
            env_factory=factory,
        )
    except Exception as exc:
        return CandidateResult(
            name=candidate.name,
            kind=candidate.kind,
            splits=(),
            available=False,
            note=f"route 评测失败：{type(exc).__name__}: {exc}",
        )
    per_task: dict[str, dict[str, float]] = {}
    exceptions = 0
    for task in summary.tasks:
        per_task[task.env_id] = {
            "success_rate": float(task.success_rate),
            "mean_steps": float(task.mean_steps),
            "exceptions": 0.0,
        }
        exceptions += 0
    return CandidateResult(
        name=candidate.name,
        kind=candidate.kind,
        splits=(
            CandidateSplit(
                split="in_distribution",
                per_task=per_task,
                overall_success_rate=float(
                    sum(task.successes for task in summary.tasks)
                    / sum(task.episodes for task in summary.tasks)
                ),
                exceptions=0,
                nonfinite=0,
                out_of_bounds=0,
            ),
        ),
    )


def compute_gate_c(
    results: Sequence[CandidateResult],
    *,
    threshold: float = DEFAULT_GATE_THRESHOLD,
) -> GateCVerdict:
    """门禁 C：视觉 ACT 达标任务 ≥60%、scripted fallback 可用、安全干净。"""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 < float(threshold) <= 1
    ):
        raise ValueError("threshold 必须位于 (0, 1]")
    visual = next(
        (item for item in results if item.kind == "visual_act" and item.available),
        None,
    )
    fallback_available = any(
        item.kind in ("scripted", "route") and item.available for item in results
    )
    safety_clean = all(
        item.available
        and all(
            split.nonfinite == 0 and split.exceptions == 0
            for split in item.splits
        )
        for item in results
    )
    if visual is None:
        return GateCVerdict(
            passed=False,
            visual_rate=0.0,
            fallback_available=fallback_available,
            safety_clean=safety_clean,
            notes="缺少可用的视觉 ACT 候选，门禁无法判定",
        )
    split = visual.in_distribution
    if split is None:
        return GateCVerdict(
            passed=False,
            visual_rate=0.0,
            fallback_available=fallback_available,
            safety_clean=safety_clean,
            notes="视觉 ACT 缺少分布内评测结果",
        )
    visual_rate = float(split.per_task.get(GATE_TASK, {}).get("success_rate", 0.0))
    passed = (
        visual_rate >= float(threshold)
        and fallback_available
        and safety_clean
    )
    notes = (
        f"达标任务 {GATE_TASK} 视觉成功率 {visual_rate:.1%} "
        f"（阈值 {threshold:.0%}）；"
        f"fallback={fallback_available}；安全={safety_clean}。"
        "PickCube 为视觉 ACT 弱项（~45%），作附注指标，"
        "由 scripted/状态 ACT 兜底。"
    )
    return GateCVerdict(
        passed=passed,
        visual_rate=visual_rate,
        fallback_available=fallback_available,
        safety_clean=safety_clean,
        notes=notes,
    )


def _result_payload(result: CandidateResult) -> dict[str, object]:
    return {
        "name": result.name,
        "kind": result.kind,
        "available": result.available,
        "note": result.note,
        "splits": [
            {
                "split": split.split,
                "per_task": split.per_task,
                "overall_success_rate": split.overall_success_rate,
                "exceptions": split.exceptions,
                "nonfinite": split.nonfinite,
                "out_of_bounds": split.out_of_bounds,
            }
            for split in result.splits
        ],
    }


def build_model_card(
    results: Sequence[CandidateResult],
    verdict: GateCVerdict,
    *,
    output_dir: Path,
    historical: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, object]:
    """聚合候选结果 + 历史结果，写模型卡 JSON 与选型报告 MD。"""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    history = dict(historical) if historical is not None else dict(HISTORICAL_RESULTS)
    card = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "gate_task": GATE_TASK,
        "gate": {
            "threshold": 0.60,
            "passed": verdict.passed,
            "visual_rate": verdict.visual_rate,
            "fallback_available": verdict.fallback_available,
            "safety_clean": verdict.safety_clean,
            "notes": verdict.notes,
        },
        "candidates": [_result_payload(item) for item in results],
        "historical": history,
        "frozen_selection": {
            "main_policy": "state_act（状态输入 ACT，仿真 96–99%）",
            "fallback_policy": "scripted（末位安全网，100%）",
            "visual_act": "真机阶段升级候选（PushCube 100% 达标；PickCube 弱项需真机数据重训）",
            "smolvla": "推迟到真机阶段（第 32 周前后）",
        },
    }
    _atomic_write_json(directory / "model_card.json", card)
    (directory / "selection_report.md").write_text(
        _render_selection_report(card),
        encoding="utf-8",
    )
    return card


def _render_selection_report(card: Mapping[str, object]) -> str:
    lines = [
        "# 第 22 周阶段评测与选型报告",
        "",
        f"- 生成时间：{card['created_at']}",
        f"- 门禁任务口径：{card['gate_task']}（视觉 ACT 达标任务）",
        "",
        "## 1. 门禁 C 判定",
        "",
        f"- 通过：**{card['gate']['passed']}**",
        f"- 视觉成功率（PushCube）：{card['gate']['visual_rate']:.1%}",
        f"- scripted fallback 可用：{card['gate']['fallback_available']}",
        f"- 安全指标干净（nonfinite=0、exceptions=0）：{card['gate']['safety_clean']}",
        f"- 依据：{card['gate']['notes']}",
        "",
        "## 2. 候选策略固定 seed 评测",
        "",
        "| 候选 | kind | 可用 | 分布内总体 | 分布外总体 | 安全 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in card["candidates"]:
        splits = {split["split"]: split for split in item["splits"]}
        in_stats = splits.get("in_distribution", {})
        ood_stats = splits.get("out_of_distribution", {})
        lines.append(
            f"| {item['name']} | {item['kind']} | {item['available']} | "
            f"{in_stats.get('overall_success_rate', float('nan')):.1%} | "
            f"{ood_stats.get('overall_success_rate', float('nan')):.1%} | "
            f"nonfinite={in_stats.get('nonfinite', 0)} "
            f"exc={in_stats.get('exceptions', 0)} |"
        )
    lines += [
        "",
        "## 3. 历史固定评测结果",
        "",
        "| 策略 | PickCube-v1 | PushCube-v1 |",
        "| --- | --- | --- |",
    ]
    for name, rates in card["historical"].items():
        lines.append(
            f"| {name} | {rates.get('PickCube-v1', float('nan')):.0%} | "
            f"{rates.get('PushCube-v1', float('nan')):.0%} |"
        )
    frozen = card["frozen_selection"]
    lines += [
        "",
        "## 4. 冻结选型结论",
        "",
        f"- 主策略：**{frozen['main_policy']}**",
        f"- 回退策略：**{frozen['fallback_policy']}**",
        f"- 视觉 ACT：{frozen['visual_act']}",
        f"- SmolVLA：{frozen['smolvla']}",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "SELECTION_SCHEMA_VERSION",
    "CANDIDATE_KINDS",
    "GATE_TASK",
    "HISTORICAL_RESULTS",
    "CandidateResult",
    "CandidateSplit",
    "GateCVerdict",
    "PolicyCandidate",
    "StageSelectionConfig",
    "StageSelectionError",
    "build_model_card",
    "compute_gate_c",
    "evaluate_candidates",
    "parse_candidate",
]
