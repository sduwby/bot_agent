"""第 14 周：稳定性与阶段 B 评测。

在状态输入 ACT（第 13 周 checkpoint）上验证随机初始位姿、多随机种子、
动作平滑扫描、失败相位分析与阶段 B 门禁。ACT 不参与 scripted fallback；
scripted policy 只作为专家/上界参考与安全回退描述。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from behavior_cloning import BehaviorCloningError, load_bc_checkpoint
from lerobot_act import (
    PolicyEvaluationSummary,
    evaluate_policy_seeds,
    load_lerobot_act_checkpoints,
)
from maniskill_intro import (
    SUPPORTED_TASKS,
    ManiSkillTaskConfig,
    ManiSkillUnavailableError,
    ScriptedPolicy,
    make_maniskill_env,
)

STABILITY_SCHEMA_VERSION = "maniskill_act_stability_v1"

# 第 14 周固定评测 seed 集（与历史 eval/dev/final/DAgger 均不重叠）。
STABILITY_SEED_START = 500
STABILITY_EPISODES_PER_TASK = 50


class StabilityEvalError(RuntimeError):
    """第 14 周稳定性评测契约错误。"""


@dataclass(frozen=True)
class StabilityEvalConfig:
    """稳定性评测的 checkpoint、seed 集与门禁参数。"""

    act_pick_checkpoint: Path
    act_push_checkpoint: Path
    bc_checkpoint: Path
    output_dir: Path
    seeds: tuple[int, ...]
    device: str = "mps"
    smoothing: float = 1.0
    tasks: tuple[str, ...] = SUPPORTED_TASKS
    max_steps: int = 50
    success_threshold: float = 0.80

    def __post_init__(self) -> None:
        for name in ("act_pick_checkpoint", "act_push_checkpoint", "bc_checkpoint"):
            value = getattr(self, name)
            object.__setattr__(
                self, name, Path(value).expanduser().resolve()
            )
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        if self.device not in ("mps", "cpu", "cuda"):
            raise ValueError("device 只支持 mps、cuda 或 cpu")
        if not isinstance(self.smoothing, (int, float)) or isinstance(
            self.smoothing, bool
        ):
            raise ValueError("smoothing 必须是数字")
        if not 0 < float(self.smoothing) <= 1:
            raise ValueError("smoothing 必须位于 (0, 1]")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须为正数")
        if not 0 <= self.success_threshold <= 1:
            raise ValueError("success_threshold 必须位于 [0, 1]")
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
        _checked_stability_seeds(self.seeds)


@dataclass(frozen=True)
class InitialPoseSample:
    """一个 seed 的初始 cube/goal 位姿。"""

    env_id: str
    seed: int
    obj_pos: tuple[float, float, float]
    goal_pos: tuple[float, float, float]


@dataclass(frozen=True)
class FailurePhaseSummary:
    """ACT 失败 episode 的相位分类聚合。"""

    env_id: str
    total_failures: int
    phase_counts: dict[str, int]
    grasp_failures: int
    placed_but_not_static: int
    overshoot_or_oscillation: int
    truncated_only: int


@dataclass(frozen=True)
class SmoothingSweepRow:
    """一个 action_smoothing 取值的评测结果。"""

    smoothing: float
    success_rate: float
    total_episodes: int
    total_successes: int
    mean_steps: float
    nonfinite_actions: int
    exceptions: int


@dataclass(frozen=True)
class StabilityComparisonSummary:
    """ACT/BC/scripted 同 seed 稳定性比较与门禁结果。"""

    schema_version: str
    created_at: str
    seeds: tuple[int, ...]
    act_pick_checkpoint: str
    act_push_checkpoint: str
    bc_checkpoint: str
    smoothing: float
    policies: tuple[PolicyEvaluationSummary, ...]
    act_rate: float
    bc_rate: float
    scripted_rate: float
    threshold: float
    effective_threshold: float
    act_passed: bool
    exceptions: int
    nonfinite: int


def _checked_stability_seeds(seeds: Sequence[int]) -> list[int]:
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


def default_stability_seeds() -> tuple[int, ...]:
    """返回第 14 周默认评测 seed 集（500–549，每任务 50 个）。"""
    return tuple(
        range(STABILITY_SEED_START, STABILITY_SEED_START + STABILITY_EPISODES_PER_TASK)
    )


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


def _required_vector(
    observation: Mapping[str, object], key: str, minimum_size: int
) -> np.ndarray:
    extra = observation.get("extra")
    if not isinstance(extra, Mapping) or key not in extra:
        raise StabilityEvalError(f"observation 缺少 extra.{key}")
    vector = np.asarray(extra[key], dtype=float).reshape(-1)
    if vector.size < minimum_size or not np.all(np.isfinite(vector)):
        raise StabilityEvalError(f"extra.{key} 不可用")
    return vector[:minimum_size]


def load_stability_checkpoints(
    pick_checkpoint: Path,
    push_checkpoint: Path,
    *,
    device: str,
    smoothing: float = 1.0,
    tasks: Sequence[str] = SUPPORTED_TASKS,
) -> dict[str, object]:
    """按 env 分别加载 ACT checkpoint（PickCube/PushCube 目录不同）。"""
    policies: dict[str, object] = {}
    sources = (
        ("PickCube-v1", pick_checkpoint),
        ("PushCube-v1", push_checkpoint),
    )
    for env_id, checkpoint in sources:
        if env_id not in tasks:
            continue
        loaded = load_lerobot_act_checkpoints(
            checkpoint,
            device=device,
            action_smoothing=smoothing,
            tasks=[env_id],
        )
        policies[env_id] = loaded[env_id]
    return policies


def record_initial_poses(
    config: StabilityEvalConfig,
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
) -> list[InitialPoseSample]:
    """对每个 seed 的 reset 记录初始 cube/goal 位姿。"""
    samples: list[InitialPoseSample] = []
    for env_id in config.tasks:
        for seed in config.seeds:
            task_config = ManiSkillTaskConfig(
                env_id=env_id, seed=seed, max_steps=config.max_steps
            )
            env: Any | None = None
            try:
                env = env_factory(task_config)
                observation, _ = env.reset(seed=seed)
                if not isinstance(observation, Mapping):
                    raise StabilityEvalError("reset observation 必须是 mapping")
                obj = _required_vector(observation, "obj_pose", 3)
                goal = _required_vector(observation, "goal_pos", 3)
                samples.append(
                    InitialPoseSample(
                        env_id=env_id,
                        seed=seed,
                        obj_pos=tuple(float(item) for item in obj),
                        goal_pos=tuple(float(item) for item in goal),
                    )
                )
            finally:
                if env is not None:
                    env.close()
    return samples


def summarize_pose_coverage(
    samples: Sequence[InitialPoseSample],
) -> dict[str, object]:
    """按任务聚合初始位姿范围与均值。"""
    by_task: dict[str, list[InitialPoseSample]] = {}
    for sample in samples:
        by_task.setdefault(sample.env_id, []).append(sample)
    coverage: dict[str, object] = {}
    for env_id, items in by_task.items():
        obj = np.asarray([item.obj_pos for item in items], dtype=float)
        goal = np.asarray([item.goal_pos for item in items], dtype=float)
        coverage[env_id] = {
            "samples": len(items),
            "obj_min": obj.min(axis=0).tolist(),
            "obj_max": obj.max(axis=0).tolist(),
            "obj_mean": obj.mean(axis=0).tolist(),
            "goal_min": goal.min(axis=0).tolist(),
            "goal_max": goal.max(axis=0).tolist(),
            "goal_mean": goal.mean(axis=0).tolist(),
        }
    return coverage


def _evaluate_policies(
    config: StabilityEvalConfig,
    policies: Mapping[str, object],
    policy_name: str,
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object],
) -> PolicyEvaluationSummary:
    return evaluate_policy_seeds(
        policy_name,
        policies,
        config.seeds,
        env_factory=env_factory,
        tasks=config.tasks,
    )


def _summary_exceptions(summary: PolicyEvaluationSummary) -> int:
    """聚合所有任务的异常 episode 数。"""
    return int(sum(task.exceptions for task in summary.tasks))


def evaluate_stability(
    config: StabilityEvalConfig,
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
) -> StabilityComparisonSummary:
    """在固定 seed 集上执行 ACT/BC/scripted 稳定性评测并落盘。"""
    act_policies = load_stability_checkpoints(
        config.act_pick_checkpoint,
        config.act_push_checkpoint,
        device=config.device,
        smoothing=config.smoothing,
        tasks=config.tasks,
    )
    bc_policies = load_bc_checkpoint(config.bc_checkpoint)
    scripted_policies = {
        env_id: ScriptedPolicy(env_id) for env_id in config.tasks
    }

    act = _evaluate_policies(
        config, act_policies, "act", env_factory=env_factory
    )
    bc = _evaluate_policies(
        config, bc_policies, "bc", env_factory=env_factory
    )
    scripted = _evaluate_policies(
        config, scripted_policies, "scripted", env_factory=env_factory
    )

    gate = compute_stability_gate(
        act_rate=act.success_rate,
        bc_rate=bc.success_rate,
        threshold=config.success_threshold,
        exceptions=_summary_exceptions(act),
        nonfinite=act.nonfinite_actions,
    )
    summary = StabilityComparisonSummary(
        schema_version=STABILITY_SCHEMA_VERSION,
        created_at=_utc_now(),
        seeds=tuple(config.seeds),
        act_pick_checkpoint=str(config.act_pick_checkpoint),
        act_push_checkpoint=str(config.act_push_checkpoint),
        bc_checkpoint=str(config.bc_checkpoint),
        smoothing=config.smoothing,
        policies=(act, bc, scripted),
        act_rate=act.success_rate,
        bc_rate=bc.success_rate,
        scripted_rate=scripted.success_rate,
        threshold=config.success_threshold,
        effective_threshold=float(gate["effective_threshold"]),
        act_passed=bool(gate["passed"]),
        exceptions=int(gate["exceptions"]),
        nonfinite=int(gate["nonfinite"]),
    )
    _atomic_write_json(
        config.output_dir / "stability_evaluation.json", asdict(summary)
    )
    return summary


def run_smoothing_sweep(
    config: StabilityEvalConfig,
    *,
    smoothing_values: Sequence[float],
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
) -> list[SmoothingSweepRow]:
    """受控扫描 action_smoothing，只诊断不重训。"""
    if not smoothing_values:
        raise ValueError("smoothing_values 不能为空")
    rows: list[SmoothingSweepRow] = []
    for value in smoothing_values:
        sweep_config = StabilityEvalConfig(
            act_pick_checkpoint=config.act_pick_checkpoint,
            act_push_checkpoint=config.act_push_checkpoint,
            bc_checkpoint=config.bc_checkpoint,
            output_dir=config.output_dir,
            seeds=config.seeds,
            device=config.device,
            smoothing=value,
            tasks=config.tasks,
            max_steps=config.max_steps,
            success_threshold=config.success_threshold,
        )
        act_policies = load_stability_checkpoints(
            sweep_config.act_pick_checkpoint,
            sweep_config.act_push_checkpoint,
            device=sweep_config.device,
            smoothing=value,
            tasks=sweep_config.tasks,
        )
        summary = _evaluate_policies(
            sweep_config, act_policies, "act", env_factory=env_factory
        )
        rows.append(
            SmoothingSweepRow(
                smoothing=float(value),
                success_rate=summary.success_rate,
                total_episodes=summary.total_episodes,
                total_successes=summary.total_successes,
                mean_steps=summary.mean_steps,
                nonfinite_actions=summary.nonfinite_actions,
                exceptions=_summary_exceptions(summary),
            )
        )
    _atomic_write_json(
        config.output_dir / "smoothing_sweep.json",
        [asdict(row) for row in rows],
    )
    return rows


def classify_failure_diagnosis(diagnosis: object) -> str:
    """将单个失败诊断对象归入失败相位。"""
    ever_grasped = bool(getattr(diagnosis, "ever_grasped", False))
    ever_placed = bool(getattr(diagnosis, "ever_placed", False))
    simultaneously_valid = int(
        getattr(diagnosis, "simultaneously_valid_steps", 0)
    )
    truncated_only = bool(
        getattr(diagnosis, "truncated", False)
        and not ever_grasped
        and not ever_placed
    )
    if truncated_only:
        return "truncated_only"
    if not ever_grasped:
        return "grasp_failed"
    if not ever_placed:
        return "grasped_not_placed"
    if simultaneously_valid == 0:
        return "placed_but_not_static"
    return "other"


def analyze_failures(
    config: StabilityEvalConfig,
    *,
    seeds: Sequence[int],
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
) -> list[FailurePhaseSummary]:
    """对 ACT 失败 seed 复用 run_act_diagnose 做全量相位分类。"""
    from run_act_diagnose import diagnose_episode

    checked = _checked_stability_seeds(seeds)
    policies = load_stability_checkpoints(
        config.act_pick_checkpoint,
        config.act_push_checkpoint,
        device=config.device,
        smoothing=config.smoothing,
        tasks=config.tasks,
    )
    summaries: list[FailurePhaseSummary] = []
    for env_id in config.tasks:
        policy = policies[env_id]
        phases: dict[str, int] = {}
        total_failures = 0
        grasp_failures = 0
        placed_but_not_static = 0
        overshoot_or_oscillation = 0
        truncated_only = 0
        for seed in checked:
            diagnosis = diagnose_episode(
                env_id,
                seed,
                policy,
                max_steps=config.max_steps,
                env_factory=env_factory,
            )
            if diagnosis.success:
                continue
            total_failures += 1
            phase = classify_failure_diagnosis(diagnosis)
            phases[phase] = phases.get(phase, 0) + 1
            if phase == "grasp_failed" or phase == "grasped_not_placed":
                grasp_failures += 1
            if phase == "placed_but_not_static":
                placed_but_not_static += 1
            if diagnosis.ever_placed and diagnosis.simultaneously_valid_steps == 0:
                overshoot_or_oscillation += 1
            if phase == "truncated_only":
                truncated_only += 1
        summaries.append(
            FailurePhaseSummary(
                env_id=env_id,
                total_failures=total_failures,
                phase_counts=phases,
                grasp_failures=grasp_failures,
                placed_but_not_static=placed_but_not_static,
                overshoot_or_oscillation=overshoot_or_oscillation,
                truncated_only=truncated_only,
            )
        )
    _atomic_write_json(
        config.output_dir / "failure_analysis.json",
        [asdict(item) for item in summaries],
    )
    return summaries


def compute_stability_gate(
    *,
    act_rate: float,
    bc_rate: float,
    threshold: float = 0.80,
    exceptions: int = 0,
    nonfinite: int = 0,
) -> dict[str, object]:
    """阶段 B 门禁：ACT ≥max(threshold, 同轮 BC) 且异常/非有限为 0。"""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold 必须是数字")
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError("threshold 必须位于 [0, 1]")
    if exceptions < 0 or nonfinite < 0:
        raise ValueError("exceptions/nonfinite 不能为负数")
    effective_threshold = max(float(threshold), float(bc_rate))
    passed = (
        float(act_rate) >= effective_threshold
        and int(exceptions) == 0
        and int(nonfinite) == 0
    )
    return {
        "threshold": float(threshold),
        "bc_rate": float(bc_rate),
        "act_rate": float(act_rate),
        "effective_threshold": effective_threshold,
        "exceptions": int(exceptions),
        "nonfinite": int(nonfinite),
        "passed": bool(passed),
    }


def check_dependencies() -> dict[str, str]:
    """返回稳定性评测依赖的可用性报告。"""
    import importlib.util

    report: dict[str, str] = {}
    for module in ("lerobot", "mani_skill", "gymnasium", "torch"):
        spec = importlib.util.find_spec(module)
        report[module] = "ok" if spec is not None else "missing"
    return report


__all__ = [
    "STABILITY_SCHEMA_VERSION",
    "STABILITY_SEED_START",
    "STABILITY_EPISODES_PER_TASK",
    "StabilityEvalConfig",
    "InitialPoseSample",
    "FailurePhaseSummary",
    "SmoothingSweepRow",
    "StabilityComparisonSummary",
    "StabilityEvalError",
    "analyze_failures",
    "check_dependencies",
    "classify_failure_diagnosis",
    "compute_stability_gate",
    "default_stability_seeds",
    "evaluate_stability",
    "load_stability_checkpoints",
    "record_initial_poses",
    "run_smoothing_sweep",
    "summarize_pose_coverage",
]
