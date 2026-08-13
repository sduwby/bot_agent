"""第 21 周：世界模型支线——小型 dynamics + CEM（PushCube 有限演示）。

从第 15 周 v3 数据集读回 PushCube 的 (state, action, next_state) transitions，
训练 22→128→128→18 的小型 MLP 动力学模型（预测状态增量），导出 NumPy 权重，
配合 CEM 规划在 live 环境做 MPC rollout。纯研究支线，不参与阶段门禁。
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from behavior_cloning import (
    ACTION_DIM,
    STATE_DIM,
    BehaviorCloningError,
    observation_to_state,
)
from maniskill_intro import (
    ManiSkillTaskConfig,
    ManiSkillUnavailableError,
    ScriptedPolicy,
    make_maniskill_env,
)

WORLD_MODEL_SCHEMA_VERSION = "maniskill_world_model_v1"

INPUT_DIM = STATE_DIM + ACTION_DIM  # 22
PUSH_TASK_TEXT = "PushCube-v1"

DEFAULT_HIDDEN_DIMS = (128, 128)
DEFAULT_TRAINING_STEPS = 4000
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_CEM_POPULATION = 128
DEFAULT_CEM_HORIZON = 8
DEFAULT_CEM_ELITE_RATIO = 0.1
DEFAULT_CEM_ITERATIONS = 4


class WorldModelError(RuntimeError):
    """第 21 周世界模型契约错误。"""


@dataclass(frozen=True)
class DynamicsDataset:
    """PushCube 的 (state, action, next_state) transitions。"""

    states: np.ndarray        # [N, 18]
    actions: np.ndarray       # [N, 4]
    next_states: np.ndarray   # [N, 18]
    episode_ids: np.ndarray   # [N]


@dataclass(frozen=True)
class WorldModelConfig:
    """dynamics 训练与 CEM 评测配置。"""

    dataset_dir: Path
    repo_id: str
    output_dir: Path
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    train_episodes: int = 80
    validation_episodes: int = 20
    training_steps: int = DEFAULT_TRAINING_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    validation_interval: int = 200
    device: str = "cpu"
    cem_population: int = DEFAULT_CEM_POPULATION
    cem_horizon: int = DEFAULT_CEM_HORIZON
    cem_elite_ratio: float = DEFAULT_CEM_ELITE_RATIO
    cem_iterations: int = DEFAULT_CEM_ITERATIONS
    seeds: tuple[int, ...] = tuple(range(500, 510))

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_dir", Path(self.dataset_dir).expanduser().resolve()
        )
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        if not self.repo_id:
            raise ValueError("repo_id 不能为空")
        if not self.hidden_dims or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in self.hidden_dims
        ):
            raise ValueError("hidden_dims 必须是非空正整数序列")
        for name in (
            "train_episodes",
            "validation_episodes",
            "training_steps",
            "batch_size",
            "validation_interval",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not 0 < float(self.learning_rate) < 1
        ):
            raise ValueError("learning_rate 必须位于 (0, 1)")
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("device 只支持 cpu、mps 或 cuda")
        for name in ("cem_population", "cem_horizon", "cem_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if (
            isinstance(self.cem_elite_ratio, bool)
            or not isinstance(self.cem_elite_ratio, (int, float))
            or not 0 < float(self.cem_elite_ratio) < 1
        ):
            raise ValueError("cem_elite_ratio 必须位于 (0, 1)")
        checked_seeds = _checked_seeds(self.seeds)
        object.__setattr__(self, "seeds", tuple(checked_seeds))


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


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise WorldModelError(
            f"{name} shape={array.shape} 期望 {shape} 或含非有限数"
        )
    return array


def load_dynamics_data(
    dataset_dir: Path,
    repo_id: str,
    *,
    task: str = PUSH_TASK_TEXT,
) -> DynamicsDataset:
    """从 v3 读回 PushCube 的 (state, action, next_state) transitions。

    state = robot_state[5] + env_state[13]（18D，与第 12 周口径一致）；
    每条 episode 的最后一帧没有 next_state，被丢弃。
    """
    from lerobot_act import (
        OBS_ENV_STATE_KEY,
        OBS_STATE_KEY,
        ACTION_KEY,
        ENV_STATE_DIM,
        ROBOT_STATE_DIM,
        combine_lerobot_act_state,
    )
    from lerobot_pipeline import open_lerobot_dataset

    try:
        dataset = open_lerobot_dataset(dataset_dir, repo_id)
    except Exception as exc:
        raise WorldModelError(
            f"v3 数据集不可用（需先跑第 15 周管线）：{type(exc).__name__}: {exc}"
        ) from exc

    frames: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        task_text = str(item.get("task", ""))
        if task_text != task:
            continue
        robot = _finite_array(
            item[OBS_STATE_KEY], (ROBOT_STATE_DIM,), "observation.state"
        )
        env = _finite_array(
            item[OBS_ENV_STATE_KEY],
            (ENV_STATE_DIM,),
            "environment_state",
        )
        state = combine_lerobot_act_state(robot, env)
        action = _finite_array(item[ACTION_KEY], (ACTION_DIM,), "action")
        frames.append(
            (
                int(item["episode_index"]),
                int(item["frame_index"]),
                state,
                action,
            )
        )
    if not frames:
        raise WorldModelError(f"数据集没有 {task} 的任何帧")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    episode_ids: list[int] = []
    by_episode: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for episode_index, frame_index, state, action in frames:
        by_episode.setdefault(episode_index, []).append(
            (frame_index, state, action)
        )
    for episode_index, items in sorted(by_episode.items()):
        items.sort(key=lambda entry: entry[0])
        for position in range(len(items) - 1):
            _, state, action = items[position]
            _, next_state, _ = items[position + 1]
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            episode_ids.append(episode_index)
    return DynamicsDataset(
        states=np.stack(states, axis=0),
        actions=np.stack(actions, axis=0),
        next_states=np.stack(next_states, axis=0),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )


def split_dynamics_data(
    data: DynamicsDataset,
    *,
    train_episodes: int,
    validation_episodes: int,
) -> tuple[DynamicsDataset, DynamicsDataset]:
    """按 episode 顺序切分 train/validation（无泄漏）。"""
    for name, value in (
        ("train_episodes", train_episodes),
        ("validation_episodes", validation_episodes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    episodes = sorted(set(int(item) for item in data.episode_ids))
    total = train_episodes + validation_episodes
    if total > len(episodes):
        raise WorldModelError(
            f"仅有 {len(episodes)} 条 episode，不足以切分 {total} 条"
        )
    train_episode_set = set(episodes[:train_episodes])
    validation_episode_set = set(
        episodes[train_episodes : train_episodes + validation_episodes]
    )
    train_mask = np.asarray(
        [int(item) in train_episode_set for item in data.episode_ids], dtype=bool
    )
    validation_mask = np.asarray(
        [int(item) in validation_episode_set for item in data.episode_ids],
        dtype=bool,
    )
    if np.any(train_mask & validation_mask):
        raise WorldModelError("train/validation 发生泄漏")
    return (
        _slice_dataset(data, train_mask),
        _slice_dataset(data, validation_mask),
    )


def _slice_dataset(data: DynamicsDataset, mask: np.ndarray) -> DynamicsDataset:
    return DynamicsDataset(
        states=data.states[mask],
        actions=data.actions[mask],
        next_states=data.next_states[mask],
        episode_ids=data.episode_ids[mask],
    )


def _normalization_stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(array.mean(axis=0), dtype=np.float32)
    std = np.asarray(array.std(axis=0), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _torch_module() -> object:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise WorldModelError("dynamics 训练需要 PyTorch") from exc
    return torch


def _select_device(device: str) -> str:
    torch = _torch_module()
    if device == "mps" and not torch.backends.mps.is_available():
        raise WorldModelError("mps 不可用，请改用 cpu 或 cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise WorldModelError("cuda 不可用，请改用 cpu 或 mps")
    return device


def _make_mlp(hidden_dims: Sequence[int], device: str) -> object:
    torch = _torch_module()
    layers: list[object] = []
    input_dim = INPUT_DIM
    for hidden in hidden_dims:
        layers.append(torch.nn.Linear(input_dim, hidden))
        layers.append(torch.nn.ReLU())
        input_dim = hidden
    layers.append(torch.nn.Linear(input_dim, STATE_DIM))
    model = torch.nn.Sequential(*layers)
    model.to(device)
    return model


def train_dynamics(
    config: WorldModelConfig,
    data: DynamicsDataset,
) -> dict[str, object]:
    """训练 MLP dynamics，导出 NumPy checkpoint 并返回训练摘要。"""
    torch = _torch_module()
    device = _select_device(config.device)
    started = time.perf_counter()
    train, validation = split_dynamics_data(
        data,
        train_episodes=config.train_episodes,
        validation_episodes=config.validation_episodes,
    )
    if len(train.states) == 0 or len(validation.states) == 0:
        raise WorldModelError("train/validation transitions 不能为空")

    state_mean, state_std = _normalization_stats(train.states)
    action_mean, action_std = _normalization_stats(train.actions)
    delta_mean, delta_std = _normalization_stats(
        train.next_states - train.states
    )

    def normalize_states(states: np.ndarray) -> np.ndarray:
        return (states - state_mean) / state_std

    def normalize_actions(actions: np.ndarray) -> np.ndarray:
        return (actions - action_mean) / action_std

    def normalize_deltas(deltas: np.ndarray) -> np.ndarray:
        return (deltas - delta_mean) / delta_std

    train_inputs = np.concatenate(
        (normalize_states(train.states), normalize_actions(train.actions)),
        axis=1,
    )
    train_targets = normalize_deltas(train.next_states - train.states)
    validation_inputs = np.concatenate(
        (
            normalize_states(validation.states),
            normalize_actions(validation.actions),
        ),
        axis=1,
    )
    validation_targets = normalize_deltas(
        validation.next_states - validation.states
    )

    model = _make_mlp(config.hidden_dims, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_tensor = torch.from_numpy(train_inputs).to(device)
    train_target_tensor = torch.from_numpy(train_targets).to(device)
    validation_tensor = torch.from_numpy(validation_inputs).to(device)
    validation_target_tensor = torch.from_numpy(validation_targets).to(device)

    count = len(train_inputs)
    best_validation_loss = math.inf
    best_step = 0
    best_state: dict[str, object] | None = None
    final_train_loss = math.inf
    step = 0
    generator = torch.Generator(device=device).manual_seed(20260816)

    model.train()
    while step < config.training_steps:
        permutation = torch.randperm(count, generator=generator, device=device)
        for start in range(0, count, config.batch_size):
            if step >= config.training_steps:
                break
            indices = permutation[start : start + config.batch_size]
            batch = train_tensor[indices]
            target = train_target_tensor[indices]
            prediction = model(batch)
            loss = loss_fn(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_train_loss = float(loss.item())
            step += 1
            if step % config.validation_interval == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss = float(
                        loss_fn(model(validation_tensor), validation_target_tensor)
                        .item()
                    )
                model.train()
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_step = step
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
    if best_state is None or not math.isfinite(best_validation_loss):
        raise WorldModelError("未得到有限 validation loss")
    if not math.isfinite(final_train_loss):
        raise WorldModelError("train loss 非有限")

    model.load_state_dict(best_state)
    weights = _export_mlp_weights(model)
    checkpoint_dir = config.output_dir / "dynamics_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        checkpoint_dir / "weights.npz",
        **{name: value for name, value in weights.items()},
    )
    metadata = {
        "schema_version": WORLD_MODEL_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "input_dim": INPUT_DIM,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dims": list(config.hidden_dims),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "delta_mean": delta_mean.tolist(),
        "delta_std": delta_std.tolist(),
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
        "final_train_loss": final_train_loss,
        "train_transitions": len(train.states),
        "validation_transitions": len(validation.states),
        "train_episodes": config.train_episodes,
        "validation_episodes": config.validation_episodes,
    }
    _atomic_write_json(checkpoint_dir / "metadata.json", metadata)
    summary = {
        "schema_version": WORLD_MODEL_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "checkpoint_dir": str(checkpoint_dir),
        "duration_seconds": round(time.perf_counter() - started, 6),
    }
    summary.update(
        {key: value for key, value in metadata.items() if key != "created_at"}
    )
    _atomic_write_json(config.output_dir / "training_summary.json", summary)
    return summary


def _export_mlp_weights(model: object) -> dict[str, np.ndarray]:
    """把 Sequential Linear/ReLU 的权重导出为 NumPy 数组。"""
    weights: dict[str, np.ndarray] = {}
    index = 0
    for module in model.modules():
        if not hasattr(module, "weight"):
            continue
        weight = module.weight.detach().cpu().numpy()
        bias = module.bias.detach().cpu().numpy()
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise WorldModelError(f"第 {index} 层权重包含非有限数")
        weights[f"layer_{index}_weight"] = weight.astype(np.float32)
        weights[f"layer_{index}_bias"] = bias.astype(np.float32)
        index += 1
    if not weights:
        raise WorldModelError("模型没有任何线性层")
    return weights


class DynamicsModel:
    """NumPy 前向的 MLP dynamics（预测状态增量）。"""

    def __init__(self, checkpoint_dir: Path) -> None:
        directory = Path(checkpoint_dir).expanduser().resolve()
        weights_path = directory / "weights.npz"
        metadata_path = directory / "metadata.json"
        if not weights_path.is_file() or not metadata_path.is_file():
            raise WorldModelError(
                f"checkpoint 缺少 weights.npz/metadata.json：{directory}"
            )
        try:
            loaded = np.load(weights_path)
        except (OSError, ValueError) as exc:
            raise WorldModelError(f"weights.npz 读取失败：{exc}") from exc
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorldModelError(f"metadata.json 读取失败：{exc}") from exc
        if not isinstance(metadata, dict):
            raise WorldModelError("metadata.json 顶层必须是 object")
        hidden_dims = list(metadata.get("hidden_dims", []))
        expected_layers = len(hidden_dims) + 1
        self._layers: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(expected_layers):
            weight_key = f"layer_{index}_weight"
            bias_key = f"layer_{index}_bias"
            if weight_key not in loaded or bias_key not in loaded:
                raise WorldModelError(f"checkpoint 缺少 {weight_key}/{bias_key}")
            self._layers.append(
                (
                    loaded[weight_key].astype(np.float32),
                    loaded[bias_key].astype(np.float32),
                )
            )
        self.state_mean = np.asarray(metadata["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(metadata["state_std"], dtype=np.float32)
        self.action_mean = np.asarray(metadata["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(metadata["action_std"], dtype=np.float32)
        self.delta_mean = np.asarray(metadata["delta_mean"], dtype=np.float32)
        self.delta_std = np.asarray(metadata["delta_std"], dtype=np.float32)

    def _forward(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float32)
        for weight, bias in self._layers[:-1]:
            values = np.maximum(values @ weight.T + bias, 0.0)
        weight, bias = self._layers[-1]
        return values @ weight.T + bias

    def predict_delta(
        self, states: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """输入 [B,18] states 与 [B,4] actions，输出 [B,18] 状态增量。"""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise WorldModelError(f"states shape={states.shape}，期望 [B, 18]")
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise WorldModelError(f"actions shape={actions.shape}，期望 [B, 4]")
        if states.shape[0] != actions.shape[0]:
            raise WorldModelError("states 与 actions batch 大小不一致")
        normalized_states = (states - self.state_mean) / self.state_std
        normalized_actions = (actions - self.action_mean) / self.action_std
        inputs = np.concatenate((normalized_states, normalized_actions), axis=1)
        normalized_deltas = self._forward(inputs)
        return (normalized_deltas * self.delta_std + self.delta_mean).astype(
            np.float32, copy=False
        )

    def predict_next(
        self, states: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """输入当前 states 与 actions，输出预测的下一帧 states。"""
        return states + self.predict_delta(states, actions)


class CEMPlanner:
    """CEM 规划：采样→动力学前向→精英重拟合→首步动作。"""

    def __init__(
        self,
        dynamics: DynamicsModel,
        *,
        population: int,
        horizon: int,
        elite_ratio: float,
        iterations: int,
    ) -> None:
        if not isinstance(population, int) or isinstance(population, bool) or population <= 0:
            raise ValueError("population 必须是正整数")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("horizon 必须是正整数")
        if (
            isinstance(elite_ratio, bool)
            or not isinstance(elite_ratio, (int, float))
            or not 0 < float(elite_ratio) < 1
        ):
            raise ValueError("elite_ratio 必须位于 (0, 1)")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
            raise ValueError("iterations 必须是正整数")
        self._dynamics = dynamics
        self._population = population
        self._horizon = horizon
        self._elite_count = max(1, int(population * float(elite_ratio)))
        self._iterations = iterations

    def plan(
        self, state: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """从当前 18D state 出发做 CEM，返回首步 4D 动作。"""
        state_array = np.asarray(state, dtype=np.float32).reshape(1, STATE_DIM)
        mean = np.zeros((self._horizon, ACTION_DIM), dtype=np.float32)
        std = np.full((self._horizon, ACTION_DIM), 0.5, dtype=np.float32)
        for _ in range(self._iterations):
            samples = rng.normal(
                loc=mean, scale=std, size=(self._population, self._horizon, ACTION_DIM)
            ).astype(np.float32)
            samples = np.clip(samples, -1.0, 1.0)
            scores = self._score_sequences(state_array, samples)
            elite_indices = np.argsort(scores)[-self._elite_count :]
            elite = samples[elite_indices]
            mean = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-4
            std = np.clip(std, 1e-3, 1.0)
        action = np.clip(mean[0], -1.0, 1.0).astype(np.float32)
        action[3] = -1.0  # PushCube 不需要夹爪变化，保持闭合。
        return action

    def _score_sequences(
        self, state: np.ndarray, sequences: np.ndarray
    ) -> np.ndarray:
        """批量预测每条序列的末帧 obj-goal 距离，得分取负。"""
        batch = len(sequences)
        current = np.repeat(state, batch, axis=0)
        for step in range(self._horizon):
            actions = sequences[:, step, :]
            current = self._dynamics.predict_next(current, actions)
            if not np.all(np.isfinite(current)):
                return np.full(batch, -np.inf, dtype=np.float32)
        obj = current[:, 3:6] if current.shape[1] >= 6 else current[:, :3]
        goal = current[:, 6:9] if current.shape[1] >= 9 else current[:, 3:6]
        distances = np.linalg.norm(obj - goal, axis=1)
        return -distances.astype(np.float32)


def run_world_model_episode(
    config: ManiSkillTaskConfig,
    env: object,
    planner: CEMPlanner,
    *,
    rng: np.random.Generator,
) -> dict[str, object]:
    """live 环境 + CEM MPC rollout，返回成功/步数。"""
    env_any: Any = env
    reset_result = env_any.reset(seed=config.seed)
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise WorldModelError("env.reset() 未返回二元组")
    observation, _ = reset_result
    if not isinstance(observation, Mapping):
        raise WorldModelError("observation 必须是 mapping")
    action_shape = tuple(int(size) for size in env_any.action_space.shape)
    steps = 0
    success = False
    terminated = False
    truncated = False
    for step_index in range(1, config.max_steps + 1):
        try:
            state = observation_to_state(observation)
            action = planner.plan(state, rng)
        except BehaviorCloningError:
            raise
        if action.shape != action_shape or not np.all(np.isfinite(action)):
            raise WorldModelError(
                f"规划动作 shape={action.shape} 或含非有限数"
            )
        step_result = env_any.step(action)
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise WorldModelError("env.step() 未返回五元组")
        observation, _, raw_terminated, raw_truncated, info = step_result
        if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
            raise WorldModelError("step observation/info 必须是 mapping")
        steps = step_index
        terminated = bool(np.asarray(raw_terminated).reshape(-1)[0])
        truncated = bool(np.asarray(raw_truncated).reshape(-1)[0])
        success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
        if success or terminated or truncated:
            break
    return {
        "steps": steps,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
    }


def run_scripted_episode(
    config: ManiSkillTaskConfig,
    env: object,
) -> dict[str, object]:
    """scripted 基线对照（复用第 10 周状态机）。"""
    from maniskill_intro import run_episode_in_env

    policy = ScriptedPolicy(config.env_id)
    result = run_episode_in_env(config, env, policy)
    return {
        "steps": result.steps,
        "success": result.success,
        "terminated": result.terminated,
        "truncated": result.truncated,
    }


def evaluate_world_model(
    config: WorldModelConfig,
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
) -> dict[str, object]:
    """固定 seed PushCube 演示评测（CEM MPC + scripted 对照）并落盘。"""
    dynamics = DynamicsModel(config.output_dir / "dynamics_checkpoint")
    planner = CEMPlanner(
        dynamics,
        population=config.cem_population,
        horizon=config.cem_horizon,
        elite_ratio=config.cem_elite_ratio,
        iterations=config.cem_iterations,
    )
    rng = np.random.default_rng(20260816)
    episodes: list[dict[str, object]] = []
    baseline: list[dict[str, object]] = []
    for seed in config.seeds:
        task_config = ManiSkillTaskConfig(env_id=PUSH_TASK_TEXT, seed=seed)
        env: Any | None = None
        try:
            env = env_factory(task_config)
            episodes.append(
                {"seed": seed, **run_world_model_episode(task_config, env, planner, rng=rng)}
            )
        finally:
            if env is not None:
                env.close()
        env = None
        try:
            env = env_factory(task_config)
            baseline.append(
                {"seed": seed, **run_scripted_episode(task_config, env)}
            )
        finally:
            if env is not None:
                env.close()

    def summarize(items: list[dict[str, object]]) -> dict[str, object]:
        total = len(items)
        successes = sum(bool(item["success"]) for item in items)
        mean_steps = (
            float(np.mean([int(item["steps"]) for item in items])) if items else 0.0
        )
        return {
            "episodes": total,
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "mean_steps": mean_steps,
        }

    report = {
        "schema_version": WORLD_MODEL_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "task": PUSH_TASK_TEXT,
        "seeds": list(config.seeds),
        "cem": {
            "population": config.cem_population,
            "horizon": config.cem_horizon,
            "elite_ratio": config.cem_elite_ratio,
            "iterations": config.cem_iterations,
        },
        "world_model": summarize(episodes),
        "scripted_baseline": summarize(baseline),
        "episodes": episodes,
        "baseline_episodes": baseline,
    }
    _atomic_write_json(config.output_dir / "demo_report.json", report)
    return report


__all__ = [
    "WORLD_MODEL_SCHEMA_VERSION",
    "INPUT_DIM",
    "PUSH_TASK_TEXT",
    "CEMPlanner",
    "DynamicsDataset",
    "DynamicsModel",
    "WorldModelConfig",
    "WorldModelError",
    "evaluate_world_model",
    "load_dynamics_data",
    "run_scripted_episode",
    "run_world_model_episode",
    "split_dynamics_data",
    "train_dynamics",
]
