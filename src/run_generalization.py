"""第 17 周：视觉 ACT 泛化评测——分布内/外成功率、gap、失败分布与超参对比。

在 ``visual_act.py`` checkpoint 之上执行固定 seed 评测：
- 分布内：默认环境 + ``in_distribution_seeds``；
- 分布外：扰动环境（``perturbation.PerturbedRenderEnv``）+ ``out_of_distribution_seeds``；
- 失败分布：对分布外失败 episode 复用 ``act_stability.classify_failure_diagnosis`` 分类；
- 超参对比：action_smoothing 评测级扫描 + chunk 训练级对比（4060 回传产物）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lerobot_act import (
    ACTEvaluationEpisode,
    ACTEvaluationTask,
    PolicyEvaluationSummary,
)
from lerobot_pipeline import make_render_env
from maniskill_intro import SUPPORTED_TASKS, ManiSkillTaskConfig
from perturbation import (
    PERTURBATION_SCHEMA_VERSION,
    PerturbationConfig,
    PerturbationError,
    make_perturbed_render_env,
)
from visual_act import (
    VISUAL_SCHEMA_VERSION,
    VisualACTError,
    VisualACTPolicy,
    load_visual_act_checkpoints,
    run_visual_episode_in_env,
)

GENERALIZATION_SCHEMA_VERSION = "maniskill_act_generalization_v1"

# 默认分布内 seed 集（与第 16 周固定评测一致）与分布外 seed 集。
DEFAULT_IN_DISTRIBUTION_SEEDS = tuple(range(500, 510))
DEFAULT_OOD_SEEDS = tuple(range(600, 610))

DEFAULT_SMOOTHING_CANDIDATES = (1.0, 0.9, 0.75, 0.5)


class GeneralizationEvalError(RuntimeError):
    """泛化评测配置或执行契约错误。"""


@dataclass(frozen=True)
class GeneralizationEvalConfig:
    """分布内/外评测的 checkpoint、seed 集与扰动参数。"""

    checkpoint_dir: Path
    output_dir: Path
    device: str = "mps"
    resolution: int = 128
    tasks: tuple[str, ...] = SUPPORTED_TASKS
    in_distribution_seeds: tuple[int, ...] = DEFAULT_IN_DISTRIBUTION_SEEDS
    out_of_distribution_seeds: tuple[int, ...] = DEFAULT_OOD_SEEDS
    perturbation: PerturbationConfig | None = None
    max_steps: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checkpoint_dir", Path(self.checkpoint_dir).expanduser().resolve()
        )
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        if self.device not in ("mps", "cpu", "cuda"):
            raise ValueError("device 只支持 mps、cuda 或 cpu")
        if not isinstance(self.resolution, int) or isinstance(
            self.resolution, bool
        ) or self.resolution <= 0:
            raise ValueError("resolution 必须是正整数")
        if not isinstance(self.max_steps, int) or isinstance(
            self.max_steps, bool
        ) or self.max_steps <= 0:
            raise ValueError("max_steps 必须是正整数")
        if self.perturbation is None:
            raise ValueError("perturbation 不能为空，分布外评测需要扰动配置")
        checked_tasks: list[str] = []
        for env_id in self.tasks:
            if env_id not in SUPPORTED_TASKS:
                raise ValueError(
                    f"不支持的任务 {env_id!r}，仅支持 {SUPPORTED_TASKS}"
                )
            if env_id in checked_tasks:
                raise ValueError(f"任务重复：{env_id}")
            checked_tasks.append(env_id)
        object.__setattr__(self, "tasks", tuple(checked_tasks))
        in_seeds = _checked_seeds(self.in_distribution_seeds)
        ood_seeds = _checked_seeds(self.out_of_distribution_seeds)
        overlap = set(in_seeds).intersection(ood_seeds)
        if overlap:
            raise ValueError(f"分布内/外 seed 重叠：{sorted(overlap)}")


@dataclass(frozen=True)
class FailureDistributionSummary:
    """分布外失败 episode 的相位分类聚合。"""

    env_id: str
    total_failures: int
    phase_counts: dict[str, int]
    grasp_failures: int
    placed_but_not_static: int
    truncated_only: int


@dataclass(frozen=True)
class GeneralizationReport:
    """分布内/外评测、gap、失败分布与超参对比的聚合报告。"""

    schema_version: str
    created_at: str
    checkpoint_dir: str
    output_dir: str
    device: str
    resolution: int
    perturbation: dict[str, object]
    in_distribution: dict[str, object]
    out_of_distribution: dict[str, object]
    gap: dict[str, float]
    failure_distribution: dict[str, object]
    hyperparameters: dict[str, object]


def _checked_seeds(values: Sequence[int]) -> list[int]:
    if not values:
        raise ValueError("seeds 不能为空")
    checked: list[int] = []
    for seed in values:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seeds 必须是非负整数")
        checked.append(seed)
    if len(set(checked)) != len(checked):
        raise ValueError("seeds 不能重复")
    return checked


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
            payload, ensure_ascii=False, indent=2, default=_json_default
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _scalar_bool(value: object, name: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise GeneralizationEvalError(f"{name} 必须是标量")
    return bool(array.reshape(-1)[0])


def _scalar_obs_bool(observation: Mapping[str, object], key: str) -> bool:
    extra = observation.get("extra")
    if not isinstance(extra, Mapping) or key not in extra:
        return False
    array = np.asarray(extra[key]).reshape(-1)
    if array.size != 1:
        return False
    return bool(array[0])


def _info_flag(info: Mapping[str, object], key: str) -> bool | None:
    if key not in info:
        return None
    array = np.asarray(info[key]).reshape(-1)
    if array.size != 1:
        return None
    return bool(array[0])


def _evaluate_split(
    checkpoint_dir: Path,
    *,
    seeds: Sequence[int],
    device: str,
    resolution: int,
    tasks: Sequence[str],
    env_factory: Callable[[ManiSkillTaskConfig], object],
    action_smoothing: float = 1.0,
) -> PolicyEvaluationSummary:
    """对一组 seed 执行 rollout 并聚合摘要，不写任何产物。"""
    checked_seeds = _checked_seeds(seeds)
    checked_tasks = tuple(tasks)
    policies = load_visual_act_checkpoints(
        checkpoint_dir,
        device=device,
        resolution=resolution,
        tasks=checked_tasks,
        action_smoothing=action_smoothing,
    )
    episodes: list[ACTEvaluationEpisode] = []
    for env_id in checked_tasks:
        policy = policies[env_id]
        for seed in checked_seeds:
            config = ManiSkillTaskConfig(
                env_id=env_id, seed=seed, max_steps=50
            )
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

    tasks_summary: list[ACTEvaluationTask] = []
    for env_id in checked_tasks:
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
        policy_name="visual_act",
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


def _split_stats(summary: PolicyEvaluationSummary) -> dict[str, object]:
    overall = {
        "episodes": summary.total_episodes,
        "successes": summary.total_successes,
        "success_rate": summary.success_rate,
        "mean_steps": summary.mean_steps,
        "mean_inference_ms": summary.mean_inference_ms,
        "exceptions": sum(task.exceptions for task in summary.tasks),
        "nonfinite_actions": summary.nonfinite_actions,
    }
    per_task: dict[str, object] = {"overall": overall}
    for task in summary.tasks:
        per_task[task.env_id] = {
            "episodes": task.episodes,
            "successes": task.successes,
            "failures": task.failures,
            "success_rate": task.success_rate,
            "mean_steps": task.mean_steps,
            "exceptions": task.exceptions,
        }
    return per_task


def _rate(summary: PolicyEvaluationSummary, env_id: str) -> float:
    task = next(
        (item for item in summary.tasks if item.env_id == env_id), None
    )
    return float(task.success_rate) if task is not None else float("nan")


def _run_telemetry_episode(
    env: object,
    config: ManiSkillTaskConfig,
    policy: VisualACTPolicy,
) -> dict[str, object]:
    """带相位信号的 rollout，用于失败分布分类。"""
    observation, _ = env.reset(seed=config.seed)
    if not isinstance(observation, Mapping):
        raise GeneralizationEvalError("observation 必须是 mapping")
    policy.reset()
    steps = 0
    success = False
    terminated = False
    truncated = False
    ever_grasped = False
    ever_placed = False
    simultaneous = 0
    last_info: dict[str, object] = {}
    for step in range(1, config.max_steps + 1):
        is_grasped = _scalar_obs_bool(observation, "is_grasped")
        ever_grasped = ever_grasped or is_grasped
        action = policy.act(observation, env.render())
        result = env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise GeneralizationEvalError("env.step() 未返回五元组")
        observation, _, raw_terminated, raw_truncated, info = result
        if not isinstance(info, Mapping):
            info = {}
        last_info = info
        steps = step
        terminated = _scalar_bool(raw_terminated, "terminated")
        truncated = _scalar_bool(raw_truncated, "truncated")
        success = _scalar_bool(info.get("success", False), "info.success")
        is_obj_placed = _info_flag(info, "is_obj_placed")
        is_robot_static = _info_flag(info, "is_robot_static")
        if is_obj_placed:
            ever_placed = True
            if is_robot_static:
                simultaneous += 1
        if success or terminated or truncated:
            break
    return {
        "env_id": config.env_id,
        "seed": config.seed,
        "steps": steps,
        "success": success,
        "truncated": truncated,
        "ever_grasped": ever_grasped,
        "ever_placed": ever_placed,
        "simultaneously_valid_steps": simultaneous,
    }


def _failure_diagnosis(
    env_id: str,
    seed: int,
    *,
    ever_grasped: bool,
    ever_placed: bool,
    simultaneously_valid_steps: int,
    truncated: bool,
) -> str:
    """复用 act_stability 的失败相位分类规则。"""
    from act_stability import classify_failure_diagnosis

    class _Diagnosis:
        pass

    diagnosis = _Diagnosis()
    diagnosis.ever_grasped = ever_grasped
    diagnosis.ever_placed = ever_placed
    diagnosis.simultaneously_valid_steps = simultaneously_valid_steps
    diagnosis.truncated = truncated
    return classify_failure_diagnosis(diagnosis)


def analyze_failure_distribution(
    config: GeneralizationEvalConfig,
) -> list[FailureDistributionSummary]:
    """对分布外失败 episode 复用相位规则分类并聚合。"""
    policies = load_visual_act_checkpoints(
        config.checkpoint_dir,
        device=config.device,
        resolution=config.resolution,
        tasks=config.tasks,
    )
    ood_seeds = _checked_seeds(config.out_of_distribution_seeds)
    env_factory: Callable[[ManiSkillTaskConfig], object] = (
        lambda task_config: make_perturbed_render_env(
            task_config, config.perturbation
        )
    )
    summaries: list[FailureDistributionSummary] = []
    for env_id in config.tasks:
        policy = policies[env_id]
        phases: dict[str, int] = {}
        total_failures = 0
        grasp_failures = 0
        placed_but_not_static = 0
        truncated_only = 0
        for seed in ood_seeds:
            task_config = ManiSkillTaskConfig(
                env_id=env_id, seed=seed, max_steps=config.max_steps
            )
            env: Any | None = None
            try:
                env = env_factory(task_config)
                episode = _run_telemetry_episode(env, task_config, policy)
            except Exception:
                continue
            finally:
                if env is not None:
                    env.close()
            if bool(episode["success"]):
                continue
            total_failures += 1
            phase = _failure_diagnosis(
                env_id,
                seed,
                ever_grasped=bool(episode["ever_grasped"]),
                ever_placed=bool(episode["ever_placed"]),
                simultaneously_valid_steps=int(
                    episode["simultaneously_valid_steps"]
                ),
                truncated=bool(episode["truncated"]),
            )
            phases[phase] = phases.get(phase, 0) + 1
            if phase in ("grasp_failed", "grasped_not_placed"):
                grasp_failures += 1
            if phase == "placed_but_not_static":
                placed_but_not_static += 1
            if phase == "truncated_only":
                truncated_only += 1
        summaries.append(
            FailureDistributionSummary(
                env_id=env_id,
                total_failures=total_failures,
                phase_counts=phases,
                grasp_failures=grasp_failures,
                placed_but_not_static=placed_but_not_static,
                truncated_only=truncated_only,
            )
        )
    return summaries


def run_smoothing_sweep(
    config: GeneralizationEvalConfig,
    *,
    smoothing_values: Sequence[float],
) -> list[dict[str, object]]:
    """对同一 checkpoint 做 action_smoothing 的评测级对比（分布外）。"""
    if not smoothing_values:
        raise ValueError("smoothing_values 不能为空")
    rows: list[dict[str, object]] = []
    ood_seeds = _checked_seeds(config.out_of_distribution_seeds)
    env_factory: Callable[[ManiSkillTaskConfig], object] = (
        lambda task_config: make_perturbed_render_env(
            task_config, config.perturbation
        )
    )
    for value in smoothing_values:
        summary = _evaluate_split(
            config.checkpoint_dir,
            seeds=ood_seeds,
            device=config.device,
            resolution=config.resolution,
            tasks=config.tasks,
            env_factory=env_factory,
            action_smoothing=float(value),
        )
        rows.append(
            {
                "action_smoothing": float(value),
                "success_rate": summary.success_rate,
                "total_episodes": summary.total_episodes,
                "total_successes": summary.total_successes,
                "mean_steps": summary.mean_steps,
                "nonfinite_actions": summary.nonfinite_actions,
                "exceptions": sum(task.exceptions for task in summary.tasks),
            }
        )
    return rows


def run_chunk_compare(
    config: GeneralizationEvalConfig,
    *,
    candidates: Sequence[tuple[str, Path, int]],
) -> list[dict[str, object]]:
    """对多个（tag, checkpoint_dir, chunk_size）做训练级对比评测。

    用于 4060 回传的 week17-c4 checkpoint 与 Mac week16-visual（chunk 8）同口径对比。
    """
    if not candidates:
        raise ValueError("candidates 不能为空")
    ood_seeds = _checked_seeds(config.out_of_distribution_seeds)
    env_factory: Callable[[ManiSkillTaskConfig], object] = (
        lambda task_config: make_perturbed_render_env(
            task_config, config.perturbation
        )
    )
    rows: list[dict[str, object]] = []
    for tag, checkpoint_dir, chunk_size in candidates:
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须是正整数")
        summary = _evaluate_split(
            checkpoint_dir,
            seeds=ood_seeds,
            device=config.device,
            resolution=config.resolution,
            tasks=config.tasks,
            env_factory=env_factory,
        )
        rows.append(
            {
                "tag": str(tag),
                "chunk_size": int(chunk_size),
                "checkpoint_dir": str(checkpoint_dir),
                "success_rate": summary.success_rate,
                "total_episodes": summary.total_episodes,
                "total_successes": summary.total_successes,
                "mean_steps": summary.mean_steps,
                "nonfinite_actions": summary.nonfinite_actions,
                "exceptions": sum(task.exceptions for task in summary.tasks),
            }
        )
    return rows


def evaluate_generalization(
    config: GeneralizationEvalConfig,
    *,
    smoothing_candidates: Sequence[float] = DEFAULT_SMOOTHING_CANDIDATES,
    chunk_candidates: Sequence[tuple[str, Path, int]] | None = None,
) -> GeneralizationReport:
    """分布内/外评测 + gap + 失败分布 + 超参对比并落盘。"""
    if not config.checkpoint_dir.is_dir():
        raise GeneralizationEvalError(
            f"checkpoint 目录不存在：{config.checkpoint_dir}"
        )
    in_summary = _evaluate_split(
        config.checkpoint_dir,
        seeds=config.in_distribution_seeds,
        device=config.device,
        resolution=config.resolution,
        tasks=config.tasks,
        env_factory=make_render_env,
    )
    env_factory: Callable[[ManiSkillTaskConfig], object] = (
        lambda task_config: make_perturbed_render_env(
            task_config, config.perturbation
        )
    )
    ood_summary = _evaluate_split(
        config.checkpoint_dir,
        seeds=config.out_of_distribution_seeds,
        device=config.device,
        resolution=config.resolution,
        tasks=config.tasks,
        env_factory=env_factory,
    )
    gap: dict[str, float] = {
        env_id: float(
            np.round(_rate(in_summary, env_id) - _rate(ood_summary, env_id), 4)
        )
        for env_id in config.tasks
    }
    gap["overall"] = float(
        np.round(in_summary.success_rate - ood_summary.success_rate, 4)
    )

    failures = analyze_failure_distribution(config)
    smoothing_rows = run_smoothing_sweep(config, smoothing_values=smoothing_candidates)
    chunk_rows = (
        run_chunk_compare(config, candidates=chunk_candidates)
        if chunk_candidates
        else []
    )

    report = GeneralizationReport(
        schema_version=GENERALIZATION_SCHEMA_VERSION,
        created_at=_utc_now(),
        checkpoint_dir=str(config.checkpoint_dir),
        output_dir=str(config.output_dir),
        device=config.device,
        resolution=config.resolution,
        perturbation={
            "background": config.perturbation.background,
            "lighting": config.perturbation.lighting,
            "pose": config.perturbation.pose,
            "schema_version": PERTURBATION_SCHEMA_VERSION,
        },
        in_distribution=_split_stats(in_summary),
        out_of_distribution=_split_stats(ood_summary),
        gap=gap,
        failure_distribution={"tasks": [asdict(item) for item in failures]},
        hyperparameters={
            "action_smoothing": smoothing_rows,
            "chunk_compare": chunk_rows,
            "smoothing_candidates": list(smoothing_candidates),
        },
    )
    _atomic_write_json(
        config.output_dir / "generalization_report.json", asdict(report)
    )
    _atomic_write_json(
        config.output_dir / "failure_distribution.json",
        {"tasks": [asdict(item) for item in failures]},
    )
    return report


def _prepared_output_dir(path: Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"输出目录必须为空：{output_dir}")
    return output_dir


def _parse_seeds(text: str) -> tuple[int, ...]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if not parts:
        raise ValueError("seeds 不能为空")
    seeds: list[int] = []
    for part in parts:
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return tuple(seeds)


def _parse_chunk_candidates(
    entries: Sequence[str],
) -> list[tuple[str, Path, int]]:
    candidates: list[tuple[str, Path, int]] = []
    for entry in entries:
        tag, _, chunk, path = entry.split(":", 3)
        candidates.append((tag, Path(path), int(chunk)))
    return candidates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第 17 周视觉 ACT 泛化：分布内/外评测、失败分布与超参对比"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("runs/maniskill_act/week16-visual"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="默认 runs/maniskill_act/week17-<时间戳>，且必须为空",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument(
        "--in-seeds", type=str, default="500-509",
        help="分布内 seed 集（逗号分隔，支持 start-end）",
    )
    parser.add_argument(
        "--ood-seeds", type=str, default="600-609",
        help="分布外 seed 集（逗号分隔，支持 start-end）",
    )
    parser.add_argument(
        "--perturbation",
        type=str,
        default="background,lighting,pose",
        help="逗号分隔的扰动开关：background,lighting,pose",
    )
    parser.add_argument(
        "--task", type=str, default="all",
        choices=("all",) + SUPPORTED_TASKS,
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--smoothing-values", type=str, default="1.0,0.9,0.75,0.5",
        help="action_smoothing 评测级扫描取值（逗号分隔）",
    )
    parser.add_argument(
        "--chunk-compare",
        nargs="*",
        default=[],
        help="训练级 chunk 对比：tag:chunk:path，可多次给出",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from datetime import datetime

        perturbations = {
            name: name in args.perturbation.split(",")
            for name in ("background", "lighting", "pose")
        }
        perturbation = PerturbationConfig(
            background=perturbations["background"],
            lighting=perturbations["lighting"],
            pose=perturbations["pose"],
        )
        tasks = SUPPORTED_TASKS if args.task == "all" else (args.task,)
        output_dir = args.output_dir or Path(
            "runs/maniskill_act/"
            f"week17-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        output_dir = _prepared_output_dir(output_dir)
        config = GeneralizationEvalConfig(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=output_dir,
            device=args.device,
            resolution=args.resolution,
            tasks=tasks,
            in_distribution_seeds=_parse_seeds(args.in_seeds),
            out_of_distribution_seeds=_parse_seeds(args.ood_seeds),
            perturbation=perturbation,
            max_steps=args.max_steps,
        )
        smoothing = tuple(
            float(item) for item in args.smoothing_values.split(",")
        )
        chunk_candidates = (
            _parse_chunk_candidates(args.chunk_compare)
            if args.chunk_compare
            else None
        )
        print(
            f"分布内 seeds={config.in_distribution_seeds} "
            f"分布外 seeds={config.out_of_distribution_seeds} "
            f"扰动=background:{perturbation.background}, "
            f"lighting:{perturbation.lighting}, pose:{perturbation.pose}"
        )
        report = evaluate_generalization(
            config,
            smoothing_candidates=smoothing,
            chunk_candidates=chunk_candidates,
        )
        print("=== 分布内 ===")
        for env_id, stats in report.in_distribution.items():
            print(
                f"  {env_id}: {stats['successes']}/{stats['episodes']} "
                f"({stats['success_rate']:.1%})"
            )
        print("=== 分布外 ===")
        for env_id, stats in report.out_of_distribution.items():
            print(
                f"  {env_id}: {stats['successes']}/{stats['episodes']} "
                f"({stats['success_rate']:.1%})"
            )
        print("=== gap（分布内 - 分布外）===")
        for env_id, value in report.gap.items():
            print(f"  {env_id}: {value:+.1%}")
        print(
            f"报告已写入：{config.output_dir / 'generalization_report.json'}"
        )
        return 0
    except (
        GeneralizationEvalError,
        PerturbationError,
        VisualACTError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"第 17 周泛化评测失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GENERALIZATION_SCHEMA_VERSION",
    "DEFAULT_IN_DISTRIBUTION_SEEDS",
    "DEFAULT_OOD_SEEDS",
    "DEFAULT_SMOOTHING_CANDIDATES",
    "FailureDistributionSummary",
    "GeneralizationEvalConfig",
    "GeneralizationEvalError",
    "GeneralizationReport",
    "analyze_failure_distribution",
    "evaluate_generalization",
    "run_chunk_compare",
    "run_smoothing_sweep",
]
