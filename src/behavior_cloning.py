"""第 12 周：低维 Behavior Cloning 数据、训练和独立 rollout。

稳定公共契约（第 13 周 ACT 复用，不复制实现）：
- ``TaskTransitions``：按 episode 连续排列的监督 transition 容器；
- ``load_bc_dataset()``：加载第 11 周 HDF5 并重建权威 18 维状态；
- ``split_task_transitions()``：按 episode 顺序切分 train/validation；
- ``observation_to_state()``：live observation → 18 维任务状态。
上述接口保持参数和异常类型稳定，其他模块只应依赖这些名字。
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from expert_trajectory import SCHEMA_VERSION as EXPERT_SCHEMA_VERSION
from maniskill_intro import (
    SUPPORTED_TASKS,
    ManiSkillTaskConfig,
    make_maniskill_env,
    run_episode_in_env,
)


BC_SCHEMA_VERSION = "maniskill_state_bc_v1"
STATE_DIM = 18
BASE_STATE_DIM = 28
ACTION_DIM = 4
PANDA_QPOS_SLICE = slice(13, 22)
PANDA_QVEL_SLICE = slice(22, 31)
GOAL_ACTORS = {
    "PickCube-v1": "goal_site",
    "PushCube-v1": "goal_region",
}


class BehaviorCloningError(RuntimeError):
    """BC 数据、checkpoint、训练或评测契约错误。"""


@dataclass(frozen=True)
class BCTrainingConfig:
    """低维 BC 的数据切分、网络和优化参数。"""

    dataset_summary: Path
    output_dir: Path
    train_episodes_per_task: int = 80
    validation_episodes_per_task: int = 20
    hidden_dims: tuple[int, ...] = (128, 128)
    epochs: int = 400
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    random_seed: int = 20260807

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_summary",
            Path(self.dataset_summary).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "output_dir",
            Path(self.output_dir).expanduser().resolve(),
        )
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        _require_positive_int(
            self.train_episodes_per_task, "train_episodes_per_task"
        )
        _require_positive_int(
            self.validation_episodes_per_task,
            "validation_episodes_per_task",
        )
        _require_positive_int(self.epochs, "epochs")
        _require_positive_int(self.batch_size, "batch_size")
        if not self.hidden_dims or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            for size in self.hidden_dims
        ):
            raise ValueError("hidden_dims 必须是非空正整数 tuple")
        for value, name in (
            (self.learning_rate, "learning_rate"),
            (self.weight_decay, "weight_decay"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} 必须是有限非负数")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if not isinstance(self.random_seed, int) or isinstance(
            self.random_seed, bool
        ):
            raise ValueError("random_seed 必须是整数")
        if self.random_seed < 0:
            raise ValueError("random_seed 不能为负数")


@dataclass(frozen=True)
class TaskTransitions:
    """一个任务按 episode 连续排列的监督 transition。"""

    env_id: str
    episode_ids: tuple[int, ...]
    episode_lengths: tuple[int, ...]
    states: np.ndarray
    actions: np.ndarray
    arm_qvel: np.ndarray | None = None


@dataclass(frozen=True)
class TaskTrainingResult:
    """一个任务的切分规模和最佳训练指标。"""

    env_id: str
    train_episode_ids: tuple[int, ...]
    validation_episode_ids: tuple[int, ...]
    train_transitions: int
    validation_transitions: int
    best_epoch: int
    best_train_loss: float
    best_validation_loss: float


@dataclass(frozen=True)
class BCTrainingSummary:
    """训练产物和分任务结果。"""

    schema_version: str
    created_at: str
    dataset_summary: str
    checkpoint_path: str
    state_dim: int
    action_dim: int
    duration_seconds: float
    tasks: tuple[TaskTrainingResult, ...]
    complete: bool


@dataclass(frozen=True)
class BCEvaluationEpisode:
    """一个独立 rollout 的结果。"""

    env_id: str
    seed: int
    steps: int
    total_reward: float
    success: bool
    terminated: bool
    truncated: bool
    error: str | None = None


@dataclass(frozen=True)
class BCEvaluationTask:
    """一个任务的独立评测聚合。"""

    env_id: str
    episodes: int
    successes: int
    failures: int
    success_rate: float


@dataclass(frozen=True)
class BCEvaluationSummary:
    """两个任务的独立 rollout 门禁结果。"""

    schema_version: str
    created_at: str
    checkpoint_path: str
    requested_per_task: int
    total_episodes: int
    total_successes: int
    total_failures: int
    success_rate: float
    threshold: float
    passed: bool
    tasks: tuple[BCEvaluationTask, ...]
    episodes: tuple[BCEvaluationEpisode, ...]


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


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BehaviorCloningError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise BehaviorCloningError(f"JSON 顶层必须是 object：{path}")
    return value


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"输出路径不是目录：{path}")
        if any(path.iterdir()):
            raise ValueError(
                f"输出目录必须为空，拒绝覆盖已有产物：{path}"
            )
    else:
        path.mkdir(parents=True)


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    array = np.asarray(candidate, dtype=np.float32)
    if array.shape != shape:
        raise BehaviorCloningError(
            f"{name} shape={array.shape}，期望 {shape}"
        )
    if not np.all(np.isfinite(array)):
        raise BehaviorCloningError(f"{name} 包含非有限数")
    return array


def _task_from_summary(
    summary: Mapping[str, object], env_id: str
) -> Mapping[str, object]:
    tasks = summary.get("tasks")
    if not isinstance(tasks, list):
        raise BehaviorCloningError("summary.tasks 必须是数组")
    matches = [
        item
        for item in tasks
        if isinstance(item, Mapping) and item.get("env_id") == env_id
    ]
    if len(matches) != 1:
        raise BehaviorCloningError(f"summary 中 {env_id} 必须且只能出现一次")
    return matches[0]


def _extract_episode_features(
    group: object,
    *,
    env_id: str,
    episode_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从完整 env state 提取与 live state_dict 一致的 28 维输入。"""
    prefix = f"{env_id}/traj_{episode_id}"
    try:
        actions = np.asarray(group["actions"], dtype=np.float32)
        env_states = group["env_states"]
        panda = np.asarray(
            env_states["articulations"]["panda"], dtype=np.float32
        )
        cube = np.asarray(env_states["actors"]["cube"], dtype=np.float32)
        goal = np.asarray(
            env_states["actors"][GOAL_ACTORS[env_id]], dtype=np.float32
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BehaviorCloningError(f"{prefix} env state 布局错误") from exc

    if actions.ndim != 2 or actions.shape[1:] != (ACTION_DIM,):
        raise BehaviorCloningError(
            f"{prefix} actions shape={actions.shape}，期望 [T, 4]"
        )
    steps = actions.shape[0]
    expected_frames = steps + 1
    for array, width, name in (
        (panda, 31, "panda"),
        (cube, 13, "cube"),
        (goal, 13, GOAL_ACTORS[env_id]),
    ):
        if array.shape != (expected_frames, width):
            raise BehaviorCloningError(
                f"{prefix}/{name} shape={array.shape}，"
                f"期望 {(expected_frames, width)}"
            )
        if not np.all(np.isfinite(array)):
            raise BehaviorCloningError(f"{prefix}/{name} 包含非有限数")
    if not np.all(np.isfinite(actions)):
        raise BehaviorCloningError(f"{prefix}/actions 包含非有限数")

    states = np.concatenate(
        (
            panda[:-1, PANDA_QPOS_SLICE],
            panda[:-1, PANDA_QVEL_SLICE],
            cube[:-1, :7],
            goal[:-1, :3],
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if states.shape != (steps, BASE_STATE_DIM):
        raise BehaviorCloningError(
            f"{prefix} state shape={states.shape}，"
            f"期望 {(steps, BASE_STATE_DIM)}"
        )

    # 第 11 周 obs 向量叶子被 RecordEpisode 压为标量；使用仍保留的
    # 首元素交叉校验 env state 布局，避免版本变化后静默提取错误 slice。
    try:
        observations = group["obs"]
        checks = (
            (
                np.asarray(observations["agent"]["qpos"]),
                panda[:, PANDA_QPOS_SLICE.start],
                "agent.qpos[0]",
            ),
            (
                np.asarray(observations["agent"]["qvel"]),
                panda[:, PANDA_QVEL_SLICE.start],
                "agent.qvel[0]",
            ),
            (
                np.asarray(observations["extra"]["obj_pose"]),
                cube[:, 0],
                "extra.obj_pose[0]",
            ),
            (
                np.asarray(observations["extra"]["goal_pos"]),
                goal[:, 0],
                "extra.goal_pos[0]",
            ),
        )
    except (KeyError, TypeError) as exc:
        raise BehaviorCloningError(f"{prefix} 缺少 observation 布局证据") from exc
    for observed, expected, field in checks:
        if observed.shape != (expected_frames,) or not np.allclose(
            observed, expected, rtol=1e-5, atol=1e-6
        ):
            raise BehaviorCloningError(
                f"{prefix} {field} 与 env state 映射不一致"
            )
    return states, actions


def _read_h5_state(node: object, index: int) -> object:
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        return np.asarray(node[index])
    items = getattr(node, "items", None)
    if not callable(items):
        raise BehaviorCloningError("env_states 节点格式错误")
    return {str(key): _read_h5_state(value, index) for key, value in items()}


def _unbatch_observation(value: object, path: str = "") -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _unbatch_observation(
                item, f"{path}.{key}" if path else str(key)
            )
            for key, item in value.items()
        }
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    array = np.asarray(candidate)
    if array.ndim == 0 or array.shape[0] != 1:
        raise BehaviorCloningError(
            f"重建 observation {path} 缺少单环境 batch 维：{array.shape}"
        )
    return array[0]


def _rebuild_episode_states(
    group: object,
    *,
    env_id: str,
    episode_id: int,
    env: object,
) -> np.ndarray:
    """从每帧 env state 恢复含 TCP 和抓取状态的 live observation。"""
    actions = np.asarray(group["actions"])
    base_env = getattr(env, "base_env", None)
    if base_env is None:
        base_env = getattr(env, "unwrapped", None)
    setter = getattr(base_env, "set_state_dict", None)
    getter = getattr(base_env, "get_obs", None)
    if not callable(setter) or not callable(getter):
        raise BehaviorCloningError(
            f"{env_id} observation 重建环境缺少 set_state_dict/get_obs"
        )
    states: list[np.ndarray] = []
    arm_qvels: list[np.ndarray] = []
    for frame_index in range(len(actions)):
        try:
            state_dict = _read_h5_state(group["env_states"], frame_index)
            setter(state_dict)
            observation = _unbatch_observation(getter())
            if not isinstance(observation, Mapping):
                raise BehaviorCloningError("重建 observation 必须是 mapping")
            states.append(observation_to_state(observation))
            arm_qvels.append(observation_to_arm_qvel(observation))
        except (KeyError, TypeError, ValueError) as exc:
            raise BehaviorCloningError(
                f"{env_id}/traj_{episode_id} 第 {frame_index} 帧重建失败"
            ) from exc
    result = np.stack(states).astype(np.float32, copy=False)
    if result.shape != (len(actions), STATE_DIM):
        raise BehaviorCloningError(
            f"{env_id}/traj_{episode_id} 重建 state shape={result.shape}"
        )
    arm_qvel = np.stack(arm_qvels).astype(np.float32, copy=False)
    if arm_qvel.shape != (len(actions), 7):
        raise BehaviorCloningError(
            f"{env_id}/traj_{episode_id} 重建 arm_qvel shape={arm_qvel.shape}"
        )
    return result, arm_qvel


def load_bc_dataset(
    summary_path: Path,
    *,
    observation_env_factory: Callable[[ManiSkillTaskConfig], object] = (
        make_maniskill_env
    ),
) -> dict[str, TaskTransitions]:
    """加载第 11 周成功轨迹并提取 BC 的完整低维状态。"""
    try:
        import h5py
    except (ImportError, OSError) as exc:
        raise BehaviorCloningError("BC 数据加载需要 h5py") from exc

    summary_path = Path(summary_path).expanduser().resolve()
    summary = _read_json_object(summary_path)
    if summary.get("schema_version") != EXPERT_SCHEMA_VERSION:
        raise BehaviorCloningError("专家数据 schema_version 不受支持")
    if not bool(summary.get("complete", False)):
        raise BehaviorCloningError("专家数据 summary 未标记 complete")

    result: dict[str, TaskTransitions] = {}
    for env_id in SUPPORTED_TASKS:
        task = _task_from_summary(summary, env_id)
        metadata_path = summary_path.parent / str(task.get("metadata_path"))
        trajectory_path = summary_path.parent / str(task.get("trajectory_path"))
        metadata = _read_json_object(metadata_path)
        episodes = metadata.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise BehaviorCloningError(f"{env_id} metadata.episodes 不能为空")
        expected = int(task.get("successes", -1))
        if len(episodes) != expected:
            raise BehaviorCloningError(
                f"{env_id} metadata={len(episodes)}，summary={expected}"
            )

        episode_ids: list[int] = []
        episode_lengths: list[int] = []
        state_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        arm_qvel_chunks: list[np.ndarray] = []
        env: Any | None = None
        try:
            env = observation_env_factory(ManiSkillTaskConfig(env_id))
            reset_result = env.reset(seed=0)
            if not isinstance(reset_result, tuple) or len(reset_result) != 2:
                raise BehaviorCloningError(
                    f"{env_id} observation 重建环境 reset 契约错误"
                )
            with h5py.File(trajectory_path, "r") as trajectories:
                for episode in episodes:
                    if not isinstance(episode, Mapping):
                        raise BehaviorCloningError(
                            f"{env_id} episode metadata 错误"
                        )
                    episode_id = int(episode["episode_id"])
                    group_name = f"traj_{episode_id}"
                    if group_name not in trajectories:
                        raise BehaviorCloningError(f"{env_id} 缺少 {group_name}")
                    _, actions = _extract_episode_features(
                        trajectories[group_name],
                        env_id=env_id,
                        episode_id=episode_id,
                    )
                    states, arm_qvel = _rebuild_episode_states(
                        trajectories[group_name],
                        env_id=env_id,
                        episode_id=episode_id,
                        env=env,
                    )
                    if int(episode.get("elapsed_steps", -1)) != len(actions):
                        raise BehaviorCloningError(
                            f"{env_id}/{group_name} metadata 步数不一致"
                        )
                    episode_ids.append(episode_id)
                    episode_lengths.append(len(actions))
                    state_chunks.append(states)
                    action_chunks.append(actions)
                    arm_qvel_chunks.append(arm_qvel)
        finally:
            if env is not None:
                env.close()

        if len(set(episode_ids)) != len(episode_ids):
            raise BehaviorCloningError(f"{env_id} episode_id 重复")
        result[env_id] = TaskTransitions(
            env_id=env_id,
            episode_ids=tuple(episode_ids),
            episode_lengths=tuple(episode_lengths),
            states=np.concatenate(state_chunks, axis=0),
            actions=np.concatenate(action_chunks, axis=0),
            arm_qvel=np.concatenate(arm_qvel_chunks, axis=0),
        )
    return result


def _episode_transition_slice(
    lengths: Sequence[int], episode_indices: Sequence[int]
) -> np.ndarray:
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    chunks = [
        np.arange(offsets[index], offsets[index + 1], dtype=np.int64)
        for index in episode_indices
    ]
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks)


def split_task_transitions(
    data: TaskTransitions,
    *,
    train_episodes: int,
    validation_episodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按 episode 顺序切分并返回两组 transition 索引。"""
    _require_positive_int(train_episodes, "train_episodes")
    _require_positive_int(validation_episodes, "validation_episodes")
    total = train_episodes + validation_episodes
    if total > len(data.episode_ids):
        raise BehaviorCloningError(
            f"{data.env_id} 仅有 {len(data.episode_ids)} 条 episode，"
            f"不足以切分 {total} 条"
        )
    train_indices = _episode_transition_slice(
        data.episode_lengths, range(train_episodes)
    )
    validation_indices = _episode_transition_slice(
        data.episode_lengths, range(train_episodes, total)
    )
    if np.intersect1d(train_indices, validation_indices).size:
        raise BehaviorCloningError(f"{data.env_id} train/validation 发生泄漏")
    return train_indices, validation_indices


def _normalization_stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(array.mean(axis=0), dtype=np.float32)
    std = np.asarray(array.std(axis=0), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def observation_to_state(observation: Mapping[str, object]) -> np.ndarray:
    """提取与 scripted expert 决策一致的 18 维任务状态。"""
    agent = observation.get("agent")
    extra = observation.get("extra")
    if not isinstance(agent, Mapping) or not isinstance(extra, Mapping):
        raise BehaviorCloningError("observation 缺少 agent/extra mapping")
    try:
        qpos = _finite_array(agent["qpos"], (9,), "agent.qpos")
        tcp = _finite_array(extra["tcp_pose"], (7,), "extra.tcp_pose")[:3]
        obj = _finite_array(extra["obj_pose"], (7,), "extra.obj_pose")[:3]
        goal = _finite_array(extra["goal_pos"], (3,), "extra.goal_pos")
        grasped = _finite_array(
            extra.get("is_grasped", False), (), "extra.is_grasped"
        ).reshape(1)
    except KeyError as exc:
        raise BehaviorCloningError(
            f"observation 缺少字段 {exc.args[0]}"
        ) from exc
    state = np.concatenate(
        (
            tcp,
            obj,
            goal,
            tcp - obj,
            obj - goal,
            qpos[-2:],
            grasped,
        )
    ).astype(np.float32, copy=False)
    if state.shape != (STATE_DIM,):
        raise BehaviorCloningError(f"live state shape={state.shape}")
    return state


def observation_to_arm_qvel(observation: Mapping[str, object]) -> np.ndarray:
    """提取 Panda 前 7 个 arm joint 速度（`is_static` 检查的关节）。"""
    agent = observation.get("agent")
    if not isinstance(agent, Mapping):
        raise BehaviorCloningError("observation 缺少 agent mapping")
    try:
        qvel = _finite_array(agent["qvel"], (9,), "agent.qvel")
    except KeyError as exc:
        raise BehaviorCloningError(
            f"observation 缺少字段 {exc.args[0]}"
        ) from exc
    return qvel[:7].astype(np.float32, copy=False)


def _torch_modules() -> tuple[object, object]:
    try:
        import torch
        from torch import nn
    except (ImportError, OSError) as exc:
        raise BehaviorCloningError("BC 训练需要 PyTorch") from exc
    return torch, nn


def _make_model(hidden_dims: Sequence[int]) -> object:
    torch, nn = _torch_modules()
    del torch
    layers: list[object] = []
    input_dim = STATE_DIM
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, ACTION_DIM))
    return nn.Sequential(*layers)


def _evaluate_loss(model: object, states: object, actions: object) -> float:
    torch, _ = _torch_modules()
    with torch.no_grad():
        loss = torch.mean((model(states) - actions) ** 2)
    return float(loss.item())


def _train_one_task(
    data: TaskTransitions,
    config: BCTrainingConfig,
) -> tuple[dict[str, object], TaskTrainingResult]:
    torch, _ = _torch_modules()
    train_indices, validation_indices = split_task_transitions(
        data,
        train_episodes=config.train_episodes_per_task,
        validation_episodes=config.validation_episodes_per_task,
    )
    train_states = data.states[train_indices]
    train_actions = data.actions[train_indices]
    validation_states = data.states[validation_indices]
    validation_actions = data.actions[validation_indices]
    state_mean, state_std = _normalization_stats(train_states)
    action_mean, action_std = _normalization_stats(train_actions)

    train_state_tensor = torch.from_numpy(
        (train_states - state_mean) / state_std
    )
    train_action_tensor = torch.from_numpy(
        (train_actions - action_mean) / action_std
    )
    validation_state_tensor = torch.from_numpy(
        (validation_states - state_mean) / state_std
    )
    validation_action_tensor = torch.from_numpy(
        (validation_actions - action_mean) / action_std
    )

    task_seed = config.random_seed + SUPPORTED_TASKS.index(data.env_id)
    torch.manual_seed(task_seed)
    model = _make_model(config.hidden_dims)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(task_seed)
    best_validation_loss = math.inf
    best_train_loss = math.inf
    best_epoch = 0
    best_state: dict[str, object] | None = None

    model.train()
    for epoch in range(1, config.epochs + 1):
        permutation = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            prediction = model(train_state_tensor[indices])
            loss = torch.mean((prediction - train_action_tensor[indices]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        train_loss = _evaluate_loss(
            model, train_state_tensor, train_action_tensor
        )
        validation_loss = _evaluate_loss(
            model, validation_state_tensor, validation_action_tensor
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        model.train()

    if best_state is None or not math.isfinite(best_validation_loss):
        raise BehaviorCloningError(f"{data.env_id} 未得到有限 validation loss")
    train_count = config.train_episodes_per_task
    validation_count = config.validation_episodes_per_task
    result = TaskTrainingResult(
        env_id=data.env_id,
        train_episode_ids=tuple(data.episode_ids[:train_count]),
        validation_episode_ids=tuple(
            data.episode_ids[train_count : train_count + validation_count]
        ),
        train_transitions=len(train_indices),
        validation_transitions=len(validation_indices),
        best_epoch=best_epoch,
        best_train_loss=best_train_loss,
        best_validation_loss=best_validation_loss,
    )
    payload = {
        "model_state": best_state,
        "state_mean": torch.from_numpy(state_mean),
        "state_std": torch.from_numpy(state_std),
        "action_mean": torch.from_numpy(action_mean),
        "action_std": torch.from_numpy(action_std),
        "train_episode_ids": list(result.train_episode_ids),
        "validation_episode_ids": list(result.validation_episode_ids),
        "train_transitions": result.train_transitions,
        "validation_transitions": result.validation_transitions,
        "best_epoch": result.best_epoch,
        "best_train_loss": result.best_train_loss,
        "best_validation_loss": result.best_validation_loss,
    }
    return payload, result


def train_behavior_cloning(
    config: BCTrainingConfig,
    *,
    observation_env_factory: Callable[[ManiSkillTaskConfig], object] = (
        make_maniskill_env
    ),
) -> BCTrainingSummary:
    """训练两个任务的小型 BC，并原子保存最佳 checkpoint。"""
    started = time.perf_counter()
    _prepare_output_directory(config.output_dir)
    dataset = load_bc_dataset(
        config.dataset_summary,
        observation_env_factory=observation_env_factory,
    )
    task_payloads: dict[str, object] = {}
    task_results: list[TaskTrainingResult] = []
    for env_id in SUPPORTED_TASKS:
        payload, result = _train_one_task(dataset[env_id], config)
        task_payloads[env_id] = payload
        task_results.append(result)

    checkpoint_path = config.output_dir / "checkpoint.pt"
    torch, _ = _torch_modules()
    checkpoint = {
        "schema_version": BC_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "dataset_summary": str(config.dataset_summary),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dims": list(config.hidden_dims),
        "config": {
            "train_episodes_per_task": config.train_episodes_per_task,
            "validation_episodes_per_task": (
                config.validation_episodes_per_task
            ),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "random_seed": config.random_seed,
        },
        "tasks": task_payloads,
    }
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)

    # 必须从磁盘安全加载一次；checkpoint 不可用时训练不能标记完成。
    load_bc_checkpoint(checkpoint_path)
    summary = BCTrainingSummary(
        schema_version=BC_SCHEMA_VERSION,
        created_at=str(checkpoint["created_at"]),
        dataset_summary=str(config.dataset_summary),
        checkpoint_path=str(checkpoint_path),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        duration_seconds=round(time.perf_counter() - started, 6),
        tasks=tuple(task_results),
        complete=True,
    )
    _atomic_write_json(config.output_dir / "training_summary.json", asdict(summary))
    return summary


class BCPolicy:
    """从 checkpoint 构造的单任务低维 BC policy。"""

    phase = "behavior_cloning"

    def __init__(
        self,
        *,
        env_id: str,
        model: object,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ) -> None:
        self.env_id = env_id
        self._model = model
        self._state_mean = state_mean
        self._state_std = state_std
        self._action_mean = action_mean
        self._action_std = action_std

    def reset(self) -> None:
        """无循环隐藏状态；保留统一 Policy 接口。"""

    def act(self, observation: Mapping[str, object]) -> np.ndarray:
        torch, _ = _torch_modules()
        state = observation_to_state(observation)
        normalized = (state - self._state_mean) / self._state_std
        with torch.no_grad():
            output = self._model(torch.from_numpy(normalized).unsqueeze(0))
        normalized_action = output.squeeze(0).cpu().numpy()
        action = normalized_action * self._action_std + self._action_mean
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise BehaviorCloningError("BC policy 产生非法 action")
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, object]:
    torch, _ = _torch_modules()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BehaviorCloningError(
            f"无法安全加载 checkpoint：{checkpoint_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise BehaviorCloningError("checkpoint 顶层必须是 dict")
    if payload.get("schema_version") != BC_SCHEMA_VERSION:
        raise BehaviorCloningError("checkpoint schema_version 不受支持")
    if payload.get("state_dim") != STATE_DIM or payload.get(
        "action_dim"
    ) != ACTION_DIM:
        raise BehaviorCloningError("checkpoint state/action 维度错误")
    return payload


def load_bc_checkpoint(checkpoint_path: Path) -> dict[str, BCPolicy]:
    """安全加载 checkpoint，验证所有模型和归一化 tensor。"""
    payload = _load_checkpoint_payload(checkpoint_path)
    hidden_dims = payload.get("hidden_dims")
    if not isinstance(hidden_dims, list) or not hidden_dims:
        raise BehaviorCloningError("checkpoint hidden_dims 错误")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, dict) or set(raw_tasks) != set(SUPPORTED_TASKS):
        raise BehaviorCloningError("checkpoint tasks 必须覆盖两个固定任务")

    policies: dict[str, BCPolicy] = {}
    for env_id in SUPPORTED_TASKS:
        task = raw_tasks[env_id]
        if not isinstance(task, dict):
            raise BehaviorCloningError(f"{env_id} checkpoint task 格式错误")
        model = _make_model(hidden_dims)
        try:
            model.load_state_dict(task["model_state"], strict=True)
        except (KeyError, RuntimeError, TypeError) as exc:
            raise BehaviorCloningError(f"{env_id} model_state 错误") from exc
        model.eval()

        arrays: dict[str, np.ndarray] = {}
        for name, shape in (
            ("state_mean", (STATE_DIM,)),
            ("state_std", (STATE_DIM,)),
            ("action_mean", (ACTION_DIM,)),
            ("action_std", (ACTION_DIM,)),
        ):
            try:
                value = task[name].detach().cpu().numpy()
            except (KeyError, AttributeError) as exc:
                raise BehaviorCloningError(
                    f"{env_id} checkpoint 缺少 {name} tensor"
                ) from exc
            arrays[name] = _finite_array(value, shape, f"{env_id}.{name}")
        if np.any(arrays["state_std"] <= 0) or np.any(
            arrays["action_std"] <= 0
        ):
            raise BehaviorCloningError(f"{env_id} normalization std 必须为正")
        policies[env_id] = BCPolicy(
            env_id=env_id,
            model=model,
            state_mean=arrays["state_mean"],
            state_std=arrays["state_std"],
            action_mean=arrays["action_mean"],
            action_std=arrays["action_std"],
        )
    return policies


def evaluate_bc_checkpoint(
    checkpoint_path: Path,
    *,
    seeds: Sequence[int],
    threshold: float = 0.70,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
) -> BCEvaluationSummary:
    """从磁盘 checkpoint 对两个任务执行未见 seed 的真实 rollout。"""
    if not seeds:
        raise ValueError("seeds 不能为空")
    checked_seeds: list[int] = []
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seeds 必须是非负整数")
        checked_seeds.append(seed)
    if len(set(checked_seeds)) != len(checked_seeds):
        raise ValueError("seeds 不能重复")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 <= float(threshold) <= 1
    ):
        raise ValueError("threshold 必须位于 [0, 1]")

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    policies = load_bc_checkpoint(checkpoint_path)
    episodes: list[BCEvaluationEpisode] = []
    task_results: list[BCEvaluationTask] = []
    for env_id in SUPPORTED_TASKS:
        task_episodes: list[BCEvaluationEpisode] = []
        policy = policies[env_id]
        for seed in checked_seeds:
            config = ManiSkillTaskConfig(env_id=env_id, seed=seed)
            env: Any | None = None
            try:
                env = env_factory(config)
                result = run_episode_in_env(config, env, policy)
                episode = BCEvaluationEpisode(
                    env_id=env_id,
                    seed=seed,
                    steps=result.steps,
                    total_reward=result.total_reward,
                    success=result.success,
                    terminated=result.terminated,
                    truncated=result.truncated,
                )
            # 固定评测不中断，逐条保存失败证据。
            except Exception as exc:
                episode = BCEvaluationEpisode(
                    env_id=env_id,
                    seed=seed,
                    steps=0,
                    total_reward=0.0,
                    success=False,
                    terminated=False,
                    truncated=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                if env is not None:
                    env.close()
            task_episodes.append(episode)
            episodes.append(episode)
        successes = sum(item.success for item in task_episodes)
        task_results.append(
            BCEvaluationTask(
                env_id=env_id,
                episodes=len(task_episodes),
                successes=successes,
                failures=len(task_episodes) - successes,
                success_rate=successes / len(task_episodes),
            )
        )

    total_successes = sum(item.success for item in episodes)
    success_rate = total_successes / len(episodes)
    summary = BCEvaluationSummary(
        schema_version=BC_SCHEMA_VERSION,
        created_at=_utc_now(),
        checkpoint_path=str(checkpoint_path),
        requested_per_task=len(checked_seeds),
        total_episodes=len(episodes),
        total_successes=total_successes,
        total_failures=len(episodes) - total_successes,
        success_rate=success_rate,
        threshold=float(threshold),
        passed=success_rate >= threshold,
        tasks=tuple(task_results),
        episodes=tuple(episodes),
    )
    _atomic_write_json(checkpoint_path.parent / "evaluation.json", asdict(summary))
    return summary
