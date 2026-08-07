"""第 11 周：ManiSkill 轻量专家轨迹采集、检查与回放。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from maniskill_intro import (
    SUPPORTED_TASKS,
    EpisodeContractError,
    ManiSkillTaskConfig,
    Policy,
    ScriptedPolicy,
    make_maniskill_env,
    run_episode_in_env,
)


SCHEMA_VERSION = "maniskill_expert_state_v1"
TASK_FILE_STEMS = {
    "PickCube-v1": "pick_cube",
    "PushCube-v1": "push_cube",
}


class TrajectoryDataError(RuntimeError):
    """轨迹文件缺失、错位或无法回放。"""


@dataclass(frozen=True)
class TrajectoryCollectionConfig:
    """两个 ManiSkill 任务的成功轨迹采集约束。"""

    output_dir: Path
    episodes_per_task: int = 100
    seed_start: int = 0
    max_attempts_per_task: int = 300
    replay_count: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_dir",
            Path(self.output_dir).expanduser().resolve(),
        )
        _require_positive_int(self.episodes_per_task, "episodes_per_task")
        _require_positive_int(
            self.max_attempts_per_task, "max_attempts_per_task"
        )
        if self.max_attempts_per_task < self.episodes_per_task:
            raise ValueError("max_attempts_per_task 不能小于 episodes_per_task")
        if not isinstance(self.seed_start, int) or isinstance(self.seed_start, bool):
            raise ValueError("seed_start 必须是整数")
        if self.seed_start < 0:
            raise ValueError("seed_start 不能为负数")
        _require_positive_int(self.replay_count, "replay_count")
        total_target = self.episodes_per_task * len(SUPPORTED_TASKS)
        if self.replay_count > total_target:
            raise ValueError("replay_count 不能超过目标成功轨迹总数")


@dataclass(frozen=True)
class TaskCollectionResult:
    """一个任务的采集统计和落盘文件。"""

    env_id: str
    target_successes: int
    attempts: int
    successes: int
    failures: int
    total_steps: int
    trajectory_path: str
    metadata_path: str


@dataclass(frozen=True)
class DatasetInspection:
    """HDF5 与伴随 JSON 的静态完整性检查结果。"""

    checked_episodes: int
    checked_steps: int
    task_counts: dict[str, int]


@dataclass(frozen=True)
class CollectionSummary:
    """一次完整采集运行的机器可读摘要。"""

    schema_version: str
    created_at: str
    obs_mode: str
    control_mode: str
    reward_mode: str
    sim_backend: str
    tasks: tuple[TaskCollectionResult, ...]
    total_attempts: int
    total_successes: int
    total_failures: int
    total_steps: int
    validated_episodes: int
    target_replays: int
    replay_summary_path: str
    duration_seconds: float
    complete: bool


@dataclass(frozen=True)
class ReplayEpisodeResult:
    """一条轨迹的真实 action replay 结果。"""

    env_id: str
    episode_id: int
    seed: int
    steps: int
    success: bool
    terminated: bool
    truncated: bool
    error: str | None = None


@dataclass(frozen=True)
class ReplaySummary:
    """抽检回放的总体结果。"""

    schema_version: str
    created_at: str
    requested: int
    replayed: int
    successful: int
    failed: int
    complete: bool
    episodes: tuple[ReplayEpisodeResult, ...]


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能序列化 {type(value).__name__}")


def _atomic_write_json(path: Path, payload: object) -> None:
    """同目录临时文件替换，避免留下半截 JSON。"""
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


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"输出路径不是目录：{path}")
        if any(path.iterdir()):
            raise ValueError(
                f"输出目录必须为空，拒绝覆盖已有数据：{path}"
            )
    else:
        path.mkdir(parents=True)


def make_recording_env(
    config: ManiSkillTaskConfig,
    output_dir: Path,
    trajectory_name: str,
) -> object:
    """创建不保存视频、记录 env state 的 ManiSkill 标准记录环境。"""
    try:
        from mani_skill.utils.wrappers.record import RecordEpisode
    except (ImportError, OSError) as exc:
        raise TrajectoryDataError(
            "ManiSkill RecordEpisode/HDF5 依赖不可用"
        ) from exc

    base_env: Any = make_maniskill_env(config)
    try:
        return RecordEpisode(
            base_env,
            output_dir=str(output_dir),
            trajectory_name=trajectory_name,
            save_trajectory=True,
            save_video=False,
            save_on_reset=False,
            clean_on_close=True,
            record_reward=True,
            record_env_state=True,
            source_type="scripted",
            source_desc="bot_agent week11 scripted expert state trajectories",
        )
    except Exception:
        base_env.close()
        raise


def _task_result(
    env_id: str,
    target: int,
    attempts: int,
    successes: int,
    total_steps: int,
) -> TaskCollectionResult:
    stem = TASK_FILE_STEMS[env_id]
    return TaskCollectionResult(
        env_id=env_id,
        target_successes=target,
        attempts=attempts,
        successes=successes,
        failures=attempts - successes,
        total_steps=total_steps,
        trajectory_path=f"{stem}.h5",
        metadata_path=f"{stem}.json",
    )


def collect_expert_dataset(
    config: TrajectoryCollectionConfig,
    *,
    env_factory: Callable[[ManiSkillTaskConfig, Path, str], object] = (
        make_recording_env
    ),
    policy_factory: Callable[[str], Policy] = ScriptedPolicy,
    inspector: Callable[[Path], DatasetInspection] | None = None,
) -> CollectionSummary:
    """采集两任务成功轨迹；失败 attempt 不写入监督数据集。"""
    started = time.perf_counter()
    _prepare_output_directory(config.output_dir)
    task_results: list[TaskCollectionResult] = []

    for env_id in SUPPORTED_TASKS:
        attempts = 0
        successes = 0
        total_steps = 0
        stem = TASK_FILE_STEMS[env_id]
        initial_config = ManiSkillTaskConfig(env_id, seed=config.seed_start)
        env: Any = env_factory(initial_config, config.output_dir, stem)
        try:
            while (
                successes < config.episodes_per_task
                and attempts < config.max_attempts_per_task
            ):
                seed = config.seed_start + attempts
                episode_config = ManiSkillTaskConfig(env_id, seed=seed)
                result = run_episode_in_env(
                    episode_config,
                    env,
                    policy_factory(env_id),
                )
                attempts += 1
                save_episode = result.success
                env.flush_trajectory(save=save_episode)
                if save_episode:
                    successes += 1
                    total_steps += result.steps
        finally:
            env.close()

        task_results.append(
            _task_result(
                env_id,
                config.episodes_per_task,
                attempts,
                successes,
                total_steps,
            )
        )

    total_successes = sum(item.successes for item in task_results)
    total_attempts = sum(item.attempts for item in task_results)
    complete = all(
        item.successes == item.target_successes for item in task_results
    )
    summary_path = config.output_dir / "summary.json"
    provisional = CollectionSummary(
        schema_version=SCHEMA_VERSION,
        created_at=_utc_now(),
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pos",
        reward_mode="normalized_dense",
        sim_backend="physx_cpu",
        tasks=tuple(task_results),
        total_attempts=total_attempts,
        total_successes=total_successes,
        total_failures=total_attempts - total_successes,
        total_steps=sum(item.total_steps for item in task_results),
        validated_episodes=0,
        target_replays=config.replay_count,
        replay_summary_path="replay_summary.json",
        duration_seconds=round(time.perf_counter() - started, 6),
        complete=complete,
    )
    _atomic_write_json(summary_path, asdict(provisional))

    inspection = (inspector or inspect_trajectory_dataset)(summary_path)
    validated = CollectionSummary(
        **{
            **asdict(provisional),
            "tasks": provisional.tasks,
            "validated_episodes": inspection.checked_episodes,
            "duration_seconds": round(time.perf_counter() - started, 6),
            "complete": complete
            and inspection.checked_episodes == total_successes,
        }
    )
    _atomic_write_json(summary_path, asdict(validated))
    return validated


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryDataError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise TrajectoryDataError(f"JSON 顶层必须是 object：{path}")
    return value


def _iter_leaf_datasets(group: object) -> Sequence[object]:
    leaves: list[object] = []

    def visit(node: object) -> None:
        if hasattr(node, "shape") and hasattr(node, "dtype"):
            leaves.append(node)
            return
        values = getattr(node, "values", None)
        if not callable(values):
            raise TrajectoryDataError("HDF5 env state/observation 节点格式错误")
        for child in values():
            visit(child)

    visit(group)
    return leaves


def inspect_trajectory_dataset(summary_path: Path) -> DatasetInspection:
    """校验元数据、轨迹数量、关键字段和 T/T+1 对齐。"""
    try:
        import h5py
    except (ImportError, OSError) as exc:
        raise TrajectoryDataError("静态检查需要 h5py") from exc

    summary_path = Path(summary_path)
    summary = _read_json_object(summary_path)
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise TrajectoryDataError("summary schema_version 不受支持")
    raw_tasks = summary.get("tasks")
    if not isinstance(raw_tasks, list):
        raise TrajectoryDataError("summary.tasks 必须是数组")

    checked_episodes = 0
    checked_steps = 0
    task_counts: dict[str, int] = {}
    for task in raw_tasks:
        if not isinstance(task, dict):
            raise TrajectoryDataError("summary.tasks 元素必须是 object")
        env_id = str(task.get("env_id"))
        metadata_path = summary_path.parent / str(task.get("metadata_path"))
        trajectory_path = summary_path.parent / str(task.get("trajectory_path"))
        metadata = _read_json_object(metadata_path)
        episodes = metadata.get("episodes")
        if not isinstance(episodes, list):
            raise TrajectoryDataError(f"{metadata_path.name} 缺少 episodes")
        expected = int(task.get("successes", -1))
        if len(episodes) != expected:
            raise TrajectoryDataError(
                f"{env_id} metadata={len(episodes)}，summary={expected}"
            )

        with h5py.File(trajectory_path, "r") as trajectories:
            if len(trajectories) != expected:
                raise TrajectoryDataError(
                    f"{env_id} HDF5={len(trajectories)}，summary={expected}"
                )
            for episode in episodes:
                if not isinstance(episode, dict):
                    raise TrajectoryDataError(f"{env_id} episode metadata 格式错误")
                episode_id = int(episode["episode_id"])
                group_name = f"traj_{episode_id}"
                if group_name not in trajectories:
                    raise TrajectoryDataError(f"{env_id} 缺少 {group_name}")
                group = trajectories[group_name]
                required = {
                    "actions",
                    "env_states",
                    "obs",
                    "rewards",
                    "terminated",
                    "truncated",
                    "success",
                }
                missing = sorted(required.difference(group.keys()))
                if missing:
                    raise TrajectoryDataError(
                        f"{env_id}/{group_name} 缺少字段：{missing}"
                    )

                steps = int(episode["elapsed_steps"])
                actions = np.asarray(group["actions"])
                rewards = np.asarray(group["rewards"])
                if actions.shape != (steps, 4):
                    raise TrajectoryDataError(
                        f"{env_id}/{group_name} actions shape={actions.shape}"
                    )
                if rewards.shape != (steps,):
                    raise TrajectoryDataError(
                        f"{env_id}/{group_name} rewards shape={rewards.shape}"
                    )
                if not np.all(np.isfinite(actions)) or not np.all(
                    np.isfinite(rewards)
                ):
                    raise TrajectoryDataError(
                        f"{env_id}/{group_name} 含非有限 action/reward"
                    )
                for key in ("terminated", "truncated", "success"):
                    if tuple(group[key].shape) != (steps,):
                        raise TrajectoryDataError(
                            f"{env_id}/{group_name}/{key} 长度错误"
                        )
                success_values = np.asarray(group["success"], dtype=bool)
                if steps == 0 or not bool(success_values[-1]):
                    raise TrajectoryDataError(
                        f"{env_id}/{group_name} 不是成功轨迹"
                    )
                if not bool(episode.get("success", False)):
                    raise TrajectoryDataError(
                        f"{env_id}/{group_name} metadata success=false"
                    )
                if episode.get("control_mode") != summary.get("control_mode"):
                    raise TrajectoryDataError(
                        f"{env_id}/{group_name} control_mode 不一致"
                    )
                for key in ("env_states", "obs"):
                    leaves = _iter_leaf_datasets(group[key])
                    if not leaves:
                        raise TrajectoryDataError(
                            f"{env_id}/{group_name}/{key} 没有叶子字段"
                        )
                    for dataset in leaves:
                        if int(dataset.shape[0]) != steps + 1:
                            raise TrajectoryDataError(
                                f"{env_id}/{group_name}/{key} "
                                "与 action 未按 T+1 对齐"
                            )

                checked_episodes += 1
                checked_steps += steps
        task_counts[env_id] = len(episodes)

    return DatasetInspection(
        checked_episodes=checked_episodes,
        checked_steps=checked_steps,
        task_counts=task_counts,
    )


def _read_h5_state(node: object, index: int) -> object:
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        return np.asarray(node[index])
    items = getattr(node, "items", None)
    if not callable(items):
        raise TrajectoryDataError("env_states 节点格式错误")
    return {str(key): _read_h5_state(value, index) for key, value in items()}


def _scalar_bool(value: object, name: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise TrajectoryDataError(f"{name} 必须是标量")
    return bool(array.reshape(-1)[0])


def _balanced_episode_indices(available: int, count: int) -> tuple[int, ...]:
    if count == 0:
        return ()
    if available < count:
        raise TrajectoryDataError(
            f"可用轨迹 {available} 条，不足以抽检 {count} 条"
        )
    if count == 1:
        return (0,)
    return tuple(int(value) for value in np.linspace(0, available - 1, count))


def replay_trajectory_dataset(
    summary_path: Path,
    *,
    replay_count: int,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
) -> ReplaySummary:
    """从初始 env state 重放原 action，并严格检查最终 success。"""
    _require_positive_int(replay_count, "replay_count")
    try:
        import h5py
    except (ImportError, OSError) as exc:
        raise TrajectoryDataError("轨迹回放需要 h5py") from exc

    summary_path = Path(summary_path)
    summary = _read_json_object(summary_path)
    raw_tasks = summary.get("tasks")
    if summary.get("schema_version") != SCHEMA_VERSION or not isinstance(
        raw_tasks, list
    ):
        raise TrajectoryDataError("summary 格式错误")
    if not raw_tasks:
        raise TrajectoryDataError("summary.tasks 不能为空")
    if replay_count > int(summary.get("total_successes", 0)):
        raise TrajectoryDataError("回放数量超过成功轨迹数")

    per_task = replay_count // len(raw_tasks)
    remainder = replay_count % len(raw_tasks)
    results: list[ReplayEpisodeResult] = []
    for task_index, task in enumerate(raw_tasks):
        if not isinstance(task, dict):
            raise TrajectoryDataError("summary.tasks 元素必须是 object")
        env_id = str(task["env_id"])
        task_replays = per_task + (1 if task_index < remainder else 0)
        metadata = _read_json_object(
            summary_path.parent / str(task["metadata_path"])
        )
        episodes = metadata.get("episodes")
        if not isinstance(episodes, list):
            raise TrajectoryDataError(f"{env_id} metadata 缺少 episodes")
        selected = _balanced_episode_indices(len(episodes), task_replays)
        trajectory_path = summary_path.parent / str(task["trajectory_path"])
        env: Any | None = None
        try:
            if selected:
                env = env_factory(ManiSkillTaskConfig(env_id))
            with h5py.File(trajectory_path, "r") as trajectories:
                for metadata_index in selected:
                    episode = episodes[metadata_index]
                    if not isinstance(episode, dict):
                        raise TrajectoryDataError(f"{env_id} episode metadata 错误")
                    episode_id = int(episode["episode_id"])
                    seed = int(episode.get("episode_seed", episode_id))
                    group = trajectories[f"traj_{episode_id}"]
                    actions = np.asarray(group["actions"], dtype=np.float32)
                    terminated = False
                    truncated = False
                    success = False
                    error: str | None = None
                    steps = 0
                    try:
                        reset_result = env.reset(seed=seed)
                        if not isinstance(reset_result, tuple) or len(
                            reset_result
                        ) != 2:
                            raise TrajectoryDataError(
                                "replay env.reset() 未返回二元组"
                            )
                        initial_state = _read_h5_state(group["env_states"], 0)
                        base_env = getattr(env, "base_env", None)
                        if base_env is None:
                            base_env = getattr(env, "unwrapped", None)
                        setter = getattr(base_env, "set_state_dict", None)
                        if not callable(setter):
                            raise TrajectoryDataError(
                                "replay 环境缺少 set_state_dict()"
                            )
                        setter(initial_state)
                        info: Mapping[str, object] = {}
                        for action in actions:
                            step_result = env.step(action)
                            if not isinstance(step_result, tuple) or len(
                                step_result
                            ) != 5:
                                raise TrajectoryDataError(
                                    "replay env.step() 未返回五元组"
                                )
                            _, _, raw_terminated, raw_truncated, raw_info = (
                                step_result
                            )
                            if not isinstance(raw_info, Mapping):
                                raise TrajectoryDataError(
                                    "replay info 必须是 mapping"
                                )
                            steps += 1
                            info = raw_info
                            terminated = _scalar_bool(
                                raw_terminated, "terminated"
                            )
                            truncated = _scalar_bool(raw_truncated, "truncated")
                        success = _scalar_bool(
                            info.get("success", False), "info.success"
                        )
                        if not success:
                            error = "原 action 回放未达到 success"
                    except (
                        EpisodeContractError,
                        TrajectoryDataError,
                        ValueError,
                    ) as exc:
                        error = str(exc)
                    results.append(
                        ReplayEpisodeResult(
                            env_id=env_id,
                            episode_id=episode_id,
                            seed=seed,
                            steps=steps,
                            success=success,
                            terminated=terminated,
                            truncated=truncated,
                            error=error,
                        )
                    )
        finally:
            if env is not None:
                env.close()

    successful = sum(item.success for item in results)
    replay_summary = ReplaySummary(
        schema_version=SCHEMA_VERSION,
        created_at=_utc_now(),
        requested=replay_count,
        replayed=len(results),
        successful=successful,
        failed=len(results) - successful,
        complete=len(results) == replay_count and successful == replay_count,
        episodes=tuple(results),
    )
    replay_path = summary_path.parent / str(
        summary.get("replay_summary_path", "replay_summary.json")
    )
    _atomic_write_json(replay_path, asdict(replay_summary))
    return replay_summary
