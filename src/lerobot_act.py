"""第 13 周：基于 LeRobot 官方 ACT 的低维策略训练、checkpoint 与评测。

本模块只做数据适配、训练编排、checkpoint 管理和 ManiSkill policy wrapper，
不复制或改写 LeRobot ``ACTConfig`` / ``ACTPolicy`` 的实现。视觉输入、真实硬件、
``LeRobotDataset`` 转换和 Hub 上传不属于本周范围。
"""

from __future__ import annotations

import importlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from behavior_cloning import (
    ACTION_DIM,
    STATE_DIM,
    BehaviorCloningError,
    load_bc_dataset,
    observation_to_state,
    split_task_transitions,
)
from maniskill_intro import (
    SUPPORTED_TASKS,
    ManiSkillTaskConfig,
    ScriptedPolicy,
    make_maniskill_env,
    run_episode_in_env,
)

ACT_SCHEMA_VERSION = "maniskill_state_act_v1"
ACT_SCHEMA_VERSION_V2_QVEL = "maniskill_state_act_v2_qvel"
ROBOT_STATE_DIM = 5
ENV_STATE_DIM = 13
ARM_QVEL_DIM = 7
ROBOT_STATE_V2_DIM = 12

# LeRobot feature 键名（与 lerobot.utils.constants 保持一致，但避免模块导入 LeRobot）。
OBS_STATE_KEY = "observation.state"
OBS_ENV_STATE_KEY = "observation.environment_state"
ACTION_KEY = "action"
ACTION_IS_PAD_KEY = "action_is_pad"

# 18 维状态按 LeRobot 语义拆分的显式索引：
#   state[0:3]  -> robot[0:3]   tcp_pos
#   state[15:17] -> robot[3:5]  gripper_qpos
#   state[3:15]  -> env[0:12]   obj_pos+goal_pos+(tcp-obj)+(obj-goal)
#   state[17:18] -> env[12:13]  is_grasped
ROBOT_STATE_INDICES = (0, 1, 2, 15, 16)
ENV_STATE_INDICES = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17)

ACTION_SPACE_BOUND = 1.0


class LeRobotACTError(RuntimeError):
    """LeRobot ACT 数据、训练、checkpoint 或评测契约错误。"""


@dataclass(frozen=True)
class ACTTrainingConfig:
    """LeRobot ACT 的数据切分、chunk、架构和优化参数。"""

    dataset_summary: Path
    output_dir: Path
    train_episodes_per_task: int = 80
    validation_episodes_per_task: int = 20
    chunk_size: int = 8
    n_action_steps: int = 4
    training_steps: int = 20_000
    batch_size: int = 64
    validation_interval: int = 250
    checkpoint_interval: int | None = None
    tasks: tuple[str, ...] = SUPPORTED_TASKS
    normalization_mode: str = "unique"
    warm_start_dir: Path | None = None
    use_qvel: bool = False
    random_seed: int = 20260807
    device: str = "mps"
    # 与官方 ACTConfig 默认架构一致；测试可用更小取值加速。
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    n_vae_encoder_layers: int = 4
    latent_dim: int = 32
    use_vae: bool = True
    kl_weight: float = 10.0
    dropout: float = 0.1

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
        if self.normalization_mode not in ("unique", "chunk"):
            raise ValueError("normalization_mode 只支持 unique 或 chunk")
        if self.warm_start_dir is not None:
            object.__setattr__(
                self,
                "warm_start_dir",
                Path(self.warm_start_dir).expanduser().resolve(),
            )
        for value, name in (
            (self.train_episodes_per_task, "train_episodes_per_task"),
            (self.validation_episodes_per_task, "validation_episodes_per_task"),
            (self.chunk_size, "chunk_size"),
            (self.n_action_steps, "n_action_steps"),
            (self.training_steps, "training_steps"),
            (self.batch_size, "batch_size"),
            (self.validation_interval, "validation_interval"),
            (self.dim_model, "dim_model"),
            (self.n_heads, "n_heads"),
            (self.dim_feedforward, "dim_feedforward"),
            (self.n_encoder_layers, "n_encoder_layers"),
            (self.n_decoder_layers, "n_decoder_layers"),
            (self.n_vae_encoder_layers, "n_vae_encoder_layers"),
            (self.latent_dim, "latent_dim"),
        ):
            _require_positive_int(value, name)
        if self.checkpoint_interval is not None:
            if not isinstance(self.checkpoint_interval, int) or isinstance(
                self.checkpoint_interval, bool
            ):
                raise ValueError("checkpoint_interval 必须是正整数或 None")
            if self.checkpoint_interval <= 0:
                raise ValueError("checkpoint_interval 必须是正整数或 None")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps 不能超过 chunk_size")
        if self.chunk_size > 64:
            raise ValueError("chunk_size 超过 64，与正式轨迹尺度不符")
        if not isinstance(self.random_seed, int) or isinstance(
            self.random_seed, bool
        ):
            raise ValueError("random_seed 必须是整数")
        if self.random_seed < 0:
            raise ValueError("random_seed 不能为负数")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device 必须是字符串")
        for value, name in (
            (self.use_vae, "use_vae"),
            (self.kl_weight, "kl_weight"),
            (self.dropout, "dropout"),
        ):
            if isinstance(value, bool) and name != "use_vae":
                raise ValueError(f"{name} 必须是数值")
        if not math.isfinite(float(self.kl_weight)) or self.kl_weight <= 0:
            raise ValueError("kl_weight 必须是有限正数")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("dropout 必须位于 [0, 1)")


@dataclass(frozen=True)
class ACTTaskTrainingResult:
    """一个任务的切分规模和最佳训练指标。"""

    env_id: str
    train_episode_ids: tuple[int, ...]
    validation_episode_ids: tuple[int, ...]
    train_transitions: int
    validation_transitions: int
    best_step: int
    best_train_loss: float
    best_validation_loss: float
    device: str
    duration_seconds: float


@dataclass(frozen=True)
class ACTTrainingSummary:
    """训练产物与分任务结果。"""

    schema_version: str
    created_at: str
    dataset_summary: str
    output_dir: str
    device: str
    chunk_size: int
    n_action_steps: int
    training_steps: int
    batch_size: int
    validation_interval: int
    checkpoint_interval: int | None
    tasks_selected: tuple[str, ...]
    duration_seconds: float
    tasks: tuple[ACTTaskTrainingResult, ...]
    complete: bool


@dataclass(frozen=True)
class ACTEvaluationEpisode:
    """一次独立 rollout 的结果与策略运行证据。"""

    env_id: str
    seed: int
    steps: int
    total_reward: float
    success: bool
    terminated: bool
    truncated: bool
    action_queries: int = 0
    chunk_queries: int = 0
    inference_ms: float = 0.0
    nonfinite_actions: int = 0
    out_of_bounds_actions: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ACTEvaluationTask:
    """一个任务的独立评测聚合。"""

    env_id: str
    episodes: int
    successes: int
    failures: int
    success_rate: float
    mean_steps: float
    exceptions: int


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    """一个策略在固定 seed 集合上的评测摘要。"""

    policy_name: str
    schema_version: str
    created_at: str
    requested_per_task: int
    total_episodes: int
    total_successes: int
    total_failures: int
    success_rate: float
    mean_inference_ms: float
    mean_steps: float
    total_chunk_queries: int
    total_action_queries: int
    nonfinite_actions: int
    out_of_bounds_actions: int
    tasks: tuple[ACTEvaluationTask, ...]
    episodes: tuple[ACTEvaluationEpisode, ...]


@dataclass(frozen=True)
class ACTComparisonSummary:
    """ACT、BC、scripted policy 的同 seed 比较与门禁结果。"""

    schema_version: str
    created_at: str
    output_dir: str
    bc_checkpoint_path: str
    seeds: tuple[int, ...]
    policies: tuple[PolicyEvaluationSummary, ...]
    act_rate: float
    bc_rate: float
    scripted_rate: float
    threshold: float
    effective_threshold: float
    act_passed: bool
    n_action_steps: int | None = None


@dataclass(frozen=True)
class LeRobotACTChunkDataset:
    """一个任务某个切分的 LeRobot feature 与 action chunk。"""

    env_id: str
    robot_state: np.ndarray
    env_state: np.ndarray
    action_chunk: np.ndarray
    action_is_pad: np.ndarray
    chunk_size: int


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
        raise LeRobotACTError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise LeRobotACTError(f"JSON 顶层必须是 object：{path}")
    return value


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"输出路径不是目录：{path}")
        if any(path.iterdir()):
            raise ValueError(f"输出目录必须为空，拒绝覆盖已有产物：{path}")
    else:
        path.mkdir(parents=True)


def _write_metrics_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, default=_json_default)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _normalization_stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(array.mean(axis=0), dtype=np.float32)
    std = np.asarray(array.std(axis=0), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


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
        raise LeRobotACTError(f"{name} shape={array.shape}，期望 {shape}")
    if not np.all(np.isfinite(array)):
        raise LeRobotACTError(f"{name} 包含非有限数")
    return array


def _torch_module() -> object:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise LeRobotACTError("LeRobot ACT 训练需要 PyTorch") from exc
    return torch


def _lerobot_import_error(exc: Exception) -> LeRobotACTError:
    return LeRobotACTError(
        "LeRobot 运行环境不可用；请用 Python 3.12 创建 `.venv-lerobot` "
        "并安装 requirements-lerobot.txt"
    )


def _select_device(device: str) -> str:
    torch = _torch_module()
    if device == "mps" and torch.backends.mps.is_available():
        return "mps"
    if device == "cpu":
        return "cpu"
    return "cpu"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except OSError:
        pass
    return None


def _environment_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for module_name in (
        "lerobot",
        "torch",
        "torchvision",
        "numpy",
        "gymnasium",
        "mani_skill",
        "sapien",
        "pinocchio",
    ):
        try:
            versions[module_name] = getattr(
                importlib.import_module(module_name), "__version__", "unknown"
            )
        except Exception:
            versions[module_name] = "unavailable"
    return versions


def split_lerobot_act_state(
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """将 18 维状态拆为 5 维 robot state 和 13 维 env state。"""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise LeRobotACTError(
            f"state shape={state.shape}，期望 ({STATE_DIM},)"
        )
    if not np.all(np.isfinite(state)):
        raise LeRobotACTError("state 包含非有限数")
    robot = state[list(ROBOT_STATE_INDICES)].astype(np.float32, copy=False)
    env = state[list(ENV_STATE_INDICES)].astype(np.float32, copy=False)
    return robot, env


def split_lerobot_act_state_v2(
    state: np.ndarray, arm_qvel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """qvel 版本：18 维状态 + 7 维 arm qvel → 12 维 robot state + 13 维 env state。"""
    state = np.asarray(state, dtype=np.float32)
    arm_qvel = np.asarray(arm_qvel, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise LeRobotACTError(
            f"state shape={state.shape}，期望 ({STATE_DIM},)"
        )
    if arm_qvel.shape != (ARM_QVEL_DIM,):
        raise LeRobotACTError(
            f"arm_qvel shape={arm_qvel.shape}，期望 ({ARM_QVEL_DIM},)"
        )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(arm_qvel)):
        raise LeRobotACTError("state 或 arm_qvel 包含非有限数")
    robot = np.concatenate(
        (
            state[list(ROBOT_STATE_INDICES)],
            arm_qvel,
        )
    ).astype(np.float32, copy=False)
    env = state[list(ENV_STATE_INDICES)].astype(np.float32, copy=False)
    return robot, env


def combine_lerobot_act_state(
    robot: np.ndarray, env: np.ndarray
) -> np.ndarray:
    """逆向拼接 5/13 维特征，返回原始 18 维状态（信息无损校验）。"""
    robot = np.asarray(robot, dtype=np.float32)
    env = np.asarray(env, dtype=np.float32)
    if robot.shape != (ROBOT_STATE_DIM,):
        raise LeRobotACTError(
            f"robot state shape={robot.shape}，期望 ({ROBOT_STATE_DIM},)"
        )
    if env.shape != (ENV_STATE_DIM,):
        raise LeRobotACTError(
            f"env state shape={env.shape}，期望 ({ENV_STATE_DIM},)"
        )
    return np.concatenate(
        (robot[0:3], env[0:12], robot[3:5], env[12:13])
    ).astype(np.float32, copy=False)


def build_action_chunk(
    actions: np.ndarray,
    start: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 episode 内 action chunk 与 bool padding mask。

    ``action_is_pad`` 为 ``True`` 表示该位置是 episode 尾部填充。
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise LeRobotACTError(
            f"actions shape={actions.shape}，期望 [T, {ACTION_DIM}]"
        )
    if not np.all(np.isfinite(actions)):
        raise LeRobotACTError("actions 包含非有限数")
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise ValueError("start 必须是非负整数")
    if start >= len(actions):
        raise ValueError("start 超出 episode 范围")
    _require_positive_int(chunk_size, "chunk_size")

    chunk = actions[start : start + chunk_size]
    mask = np.zeros(chunk_size, dtype=bool)
    valid = len(chunk)
    if valid < chunk_size:
        chunk = np.concatenate(
            (
                chunk,
                np.zeros(
                    (chunk_size - valid, ACTION_DIM), dtype=np.float32
                ),
            ),
            axis=0,
        )
        mask[valid:] = True
    return chunk, mask


def build_chunk_dataset(
    data: object,
    indices: Sequence[int],
    chunk_size: int,
    *,
    use_qvel: bool = False,
) -> LeRobotACTChunkDataset:
    """按给定 transition 索引构造 state 特征、action chunk 与 padding mask。"""
    _require_positive_int(chunk_size, "chunk_size")
    indices = np.asarray(list(indices), dtype=np.int64)
    lengths = np.asarray(data.episode_lengths, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    if data.actions.ndim != 2 or data.actions.shape[1] != ACTION_DIM:
        raise LeRobotACTError(
            f"{data.env_id} actions shape={data.actions.shape}"
        )
    if not np.all(np.isfinite(data.actions)):
        raise LeRobotACTError(f"{data.env_id} actions 包含非有限数")
    robot_dim = ROBOT_STATE_V2_DIM if use_qvel else ROBOT_STATE_DIM
    if use_qvel:
        arm_qvel = np.asarray(data.arm_qvel, dtype=np.float32)
        if arm_qvel.shape != (len(data.actions), ARM_QVEL_DIM):
            raise LeRobotACTError(
                f"{data.env_id} arm_qvel shape={arm_qvel.shape}"
            )
        if not np.all(np.isfinite(arm_qvel)):
            raise LeRobotACTError(f"{data.env_id} arm_qvel 包含非有限数")

    count = len(indices)
    robot = np.empty((count, robot_dim), dtype=np.float32)
    env = np.empty((count, ENV_STATE_DIM), dtype=np.float32)
    chunks = np.zeros((count, chunk_size, ACTION_DIM), dtype=np.float32)
    masks = np.zeros((count, chunk_size), dtype=bool)
    for row, index in enumerate(indices):
        episode = int(np.searchsorted(offsets, index, side="right") - 1)
        start = int(index - offsets[episode])
        length = int(lengths[episode])
        if not (0 <= start < length):
            raise LeRobotACTError(
                f"{data.env_id} transition {index} 不在任何 episode 内"
            )
        if use_qvel:
            robot[row], env[row] = split_lerobot_act_state_v2(
                data.states[index], arm_qvel[index]
            )
        else:
            robot[row], env[row] = split_lerobot_act_state(data.states[index])
        base = int(offsets[episode]) + start
        valid = min(chunk_size, length - start)
        chunks[row, :valid] = data.actions[base : base + valid]
        masks[row, valid:] = True
    return LeRobotACTChunkDataset(
        env_id=data.env_id,
        robot_state=robot,
        env_state=env,
        action_chunk=chunks,
        action_is_pad=masks,
        chunk_size=chunk_size,
    )


def compute_act_dataset_stats(
    data: object,
    train_indices: Sequence[int],
    *,
    use_qvel: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """只从训练集的唯一 transition 计算 STATE/ENV/ACTION 的 mean/std。

    每个 transition 只统计一次，不随 chunk size 改变；同 train split 下
    chunk 4 与 chunk 8 的统计完全相同。
    """
    indices = np.asarray(list(train_indices), dtype=np.int64)
    if data.actions.ndim != 2 or data.actions.shape[1] != ACTION_DIM:
        raise LeRobotACTError(
            f"{data.env_id} actions shape={data.actions.shape}"
        )
    if not np.all(np.isfinite(data.actions[indices])):
        raise LeRobotACTError(f"{data.env_id} train actions 包含非有限数")
    if use_qvel:
        arm_qvel = np.asarray(data.arm_qvel, dtype=np.float32)
        if arm_qvel.shape != (len(data.actions), ARM_QVEL_DIM):
            raise LeRobotACTError(
                f"{data.env_id} arm_qvel shape={arm_qvel.shape}"
            )
    robot_list: list[np.ndarray] = []
    env_list: list[np.ndarray] = []
    for index in indices:
        if use_qvel:
            robot, env = split_lerobot_act_state_v2(
                data.states[index], arm_qvel[index]
            )
        else:
            robot, env = split_lerobot_act_state(data.states[index])
        robot_list.append(robot)
        env_list.append(env)
    robot = np.stack(robot_list)
    env = np.stack(env_list)
    robot_mean, robot_std = _normalization_stats(robot)
    env_mean, env_std = _normalization_stats(env)
    action_mean, action_std = _normalization_stats(data.actions[indices])
    return {
        OBS_STATE_KEY: {"mean": robot_mean, "std": robot_std},
        OBS_ENV_STATE_KEY: {"mean": env_mean, "std": env_std},
        ACTION_KEY: {"mean": action_mean, "std": action_std},
    }


def compute_chunk_dataset_stats(
    dataset: LeRobotACTChunkDataset,
) -> dict[str, dict[str, np.ndarray]]:
    """只从给定（训练）数据集计算 STATE/ENV/ACTION 的 mean/std。"""
    robot_mean, robot_std = _normalization_stats(dataset.robot_state)
    env_mean, env_std = _normalization_stats(dataset.env_state)
    valid_actions = dataset.action_chunk[~dataset.action_is_pad].reshape(
        -1, ACTION_DIM
    )
    if len(valid_actions) == 0:
        raise LeRobotACTError(f"{dataset.env_id} 没有有效 action 样本")
    action_mean, action_std = _normalization_stats(valid_actions)
    return {
        OBS_STATE_KEY: {"mean": robot_mean, "std": robot_std},
        OBS_ENV_STATE_KEY: {"mean": env_mean, "std": env_std},
        ACTION_KEY: {"mean": action_mean, "std": action_std},
    }


def make_lerobot_act_config(
    *,
    chunk_size: int,
    n_action_steps: int,
    device: str,
    robot_state_dim: int = ROBOT_STATE_DIM,
    dim_model: int = 512,
    n_heads: int = 8,
    dim_feedforward: int = 3200,
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 1,
    n_vae_encoder_layers: int = 4,
    latent_dim: int = 32,
    use_vae: bool = True,
    kl_weight: float = 10.0,
    dropout: float = 0.1,
) -> object:
    """创建 state-only 的 LeRobot ``ACTConfig``。"""
    try:
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
    except (ImportError, OSError) as exc:
        raise _lerobot_import_error(exc) from exc
    input_features = {
        OBS_STATE_KEY: PolicyFeature(
            FeatureType.STATE, (robot_state_dim,)
        ),
        OBS_ENV_STATE_KEY: PolicyFeature(
            FeatureType.ENV, (ENV_STATE_DIM,)
        ),
    }
    output_features = {
        ACTION_KEY: PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))
    }
    return ACTConfig(
        n_obs_steps=1,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        input_features=input_features,
        output_features=output_features,
        device=device,
        dim_model=dim_model,
        n_heads=n_heads,
        dim_feedforward=dim_feedforward,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_vae_encoder_layers=n_vae_encoder_layers,
        latent_dim=latent_dim,
        use_vae=use_vae,
        kl_weight=kl_weight,
        dropout=dropout,
        temporal_ensemble_coeff=None,
    )


def _stats_to_tensors(
    stats: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    torch = _torch_module()
    result: dict[str, dict[str, object]] = {}
    for key, item in stats.items():
        result[str(key)] = {
            "mean": torch.from_numpy(
                np.asarray(item["mean"], dtype=np.float32)
            ),
            "std": torch.from_numpy(
                np.asarray(item["std"], dtype=np.float32)
            ),
        }
    return result


def make_lerobot_act_policy(
    config: ACTTrainingConfig,
    dataset_stats: Mapping[str, Mapping[str, object]],
) -> tuple[object, object, object, object]:
    """创建官方 ACTPolicy 及其 pre/post processors。"""
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.processor_act import (
            make_act_pre_post_processors,
        )
    except (ImportError, OSError) as exc:
        raise _lerobot_import_error(exc) from exc
    device = _select_device(config.device)
    robot_state_dim = (
        ROBOT_STATE_V2_DIM if config.use_qvel else ROBOT_STATE_DIM
    )
    act_config = make_lerobot_act_config(
        chunk_size=config.chunk_size,
        n_action_steps=config.n_action_steps,
        device=device,
        robot_state_dim=robot_state_dim,
        dim_model=config.dim_model,
        n_heads=config.n_heads,
        dim_feedforward=config.dim_feedforward,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        n_vae_encoder_layers=config.n_vae_encoder_layers,
        latent_dim=config.latent_dim,
        use_vae=config.use_vae,
        kl_weight=config.kl_weight,
        dropout=config.dropout,
    )
    pre, post = make_act_pre_post_processors(
        act_config,
        dataset_stats=_stats_to_tensors(dataset_stats),
    )
    policy = ACTPolicy(act_config)
    policy.to(device)
    return act_config, policy, pre, post


def _batch_from_indices(
    dataset: LeRobotACTChunkDataset, indices: Sequence[int]
) -> dict[str, object]:
    torch = _torch_module()
    indices = np.asarray(list(indices), dtype=np.int64)
    return {
        OBS_STATE_KEY: torch.from_numpy(dataset.robot_state[indices]),
        OBS_ENV_STATE_KEY: torch.from_numpy(dataset.env_state[indices]),
        ACTION_KEY: torch.from_numpy(dataset.action_chunk[indices]),
        ACTION_IS_PAD_KEY: torch.from_numpy(dataset.action_is_pad[indices]),
    }


def _evaluate_validation_loss(
    policy: object,
    pre: object,
    dataset: LeRobotACTChunkDataset,
    config: ACTTrainingConfig,
) -> float:
    """对完整 validation 集计算总 loss（保持 eval 模式语义）。"""
    torch = _torch_module()
    total = 0.0
    count = 0
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(dataset.robot_state), config.batch_size):
            indices = np.arange(
                start,
                min(start + config.batch_size, len(dataset.robot_state)),
                dtype=np.int64,
            )
            processed = pre(_batch_from_indices(dataset, indices))
            loss, _ = policy.forward(processed)
            total += float(loss.item()) * len(indices)
            count += len(indices)
    policy.train()
    if count == 0:
        raise LeRobotACTError("validation 集为空")
    return total / count


def _save_checkpoint_dir(
    policy: object, pre: object, post: object, path: Path
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)


def _warm_start_policy(
    policy: object, warm_start_dir: Path, device: str
) -> tuple[int, list[str]]:
    """把已有 checkpoint 的共享权重载入当前 policy。

    qvel 场景：robot-state 投影层输入维 5→12，把旧 5 列复制到新权重前 5 列，
    其余共享权重按原名匹配加载。返回 (载入权重数, 前缀复制的键)。
    """
    from collections import OrderedDict

    from safetensors import safe_open

    torch = _torch_module()
    warm_start_dir = Path(warm_start_dir)
    model_file = warm_start_dir / "model.safetensors"
    if not model_file.is_file():
        raise LeRobotACTError(
            f"warm-start 缺 model.safetensors：{warm_start_dir}"
        )
    target = policy.state_dict()
    filtered: "OrderedDict[str, object]" = OrderedDict()
    copied_prefix: list[str] = []
    with safe_open(str(model_file), framework="pt") as handle:
        for key in handle.keys():
            if key not in target:
                continue
            tensor = handle.get_tensor(key)
            if target[key].shape == tuple(tensor.shape):
                filtered[key] = tensor
            elif (
                key == "model.encoder_robot_state_input_proj.weight"
                and tensor.shape[1] < target[key].shape[1]
            ):
                with torch.no_grad():
                    target[key][:, : tensor.shape[1]].copy_(tensor)
                copied_prefix.append(key)
    if not filtered and not copied_prefix:
        raise LeRobotACTError("warm-start 没有可加载的共享权重")
    policy.load_state_dict(filtered, strict=False)
    policy.to(device)
    return len(filtered), copied_prefix


def _train_one_task(
    data: object,
    config: ACTTrainingConfig,
) -> ACTTaskTrainingResult:
    """训练单个任务并保存周期/best-validation checkpoint 与 metadata。"""
    torch = _torch_module()
    started = time.perf_counter()
    train_indices, validation_indices = split_task_transitions(
        data,
        train_episodes=config.train_episodes_per_task,
        validation_episodes=config.validation_episodes_per_task,
    )
    train_dataset = build_chunk_dataset(
        data, train_indices, config.chunk_size, use_qvel=config.use_qvel
    )
    validation_dataset = build_chunk_dataset(
        data, validation_indices, config.chunk_size, use_qvel=config.use_qvel
    )
    # normalization：unique=每个训练 transition 统计一次（chunk 无关）；
    # chunk=按 chunk 展开统计（原始行为，用于隔离实验）。
    if config.normalization_mode == "chunk":
        dataset_stats = compute_chunk_dataset_stats(train_dataset)
    else:
        dataset_stats = compute_act_dataset_stats(
            data, train_indices, use_qvel=config.use_qvel
        )

    # task seed 必须在模型实例化之前生效，且每任务独立。
    task_index = config.tasks.index(data.env_id)
    task_seed = config.random_seed + task_index
    random.seed(task_seed)
    np.random.seed(task_seed)
    torch.manual_seed(task_seed)

    act_config, policy, pre, post = make_lerobot_act_policy(
        config, dataset_stats
    )
    device = act_config.device
    if config.warm_start_dir is not None:
        loaded_count, copied_prefix = _warm_start_policy(
            policy, config.warm_start_dir, device
        )
        print(
            f"[{data.env_id}] warm-start 载入 {loaded_count} 个共享权重"
            + (f"，前缀复制 {copied_prefix}" if copied_prefix else "")
            + f"（{config.warm_start_dir.name}）"
        )
    optimizer_preset = act_config.get_optimizer_preset()
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=optimizer_preset.lr,
        weight_decay=optimizer_preset.weight_decay,
    )
    generator = torch.Generator().manual_seed(task_seed)

    best_validation_loss = math.inf
    best_train_loss = math.inf
    best_step = 0
    best_state: dict[str, object] | None = None
    last_train_loss = math.inf
    metrics: list[dict[str, object]] = []
    step = 0
    count = len(train_dataset.robot_state)
    task_dir = config.output_dir / data.env_id

    policy.train()
    while step < config.training_steps:
        permutation = torch.randperm(count, generator=generator)
        for start in range(0, count, config.batch_size):
            if step >= config.training_steps:
                break
            indices = permutation[start : start + config.batch_size]
            processed = pre(_batch_from_indices(train_dataset, indices))
            loss, _ = policy.forward(processed)
            last_train_loss = float(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % config.validation_interval == 0:
                validation_loss = _evaluate_validation_loss(
                    policy, pre, validation_dataset, config
                )
                metrics.append(
                    {"step": step, "validation_loss": validation_loss}
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_train_loss = last_train_loss
                    best_step = step
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in policy.state_dict().items()
                    }
            if (
                config.checkpoint_interval is not None
                and step % config.checkpoint_interval == 0
            ):
                _save_checkpoint_dir(
                    policy,
                    pre,
                    post,
                    task_dir / f"step-{step:05d}",
                )

    # 至少执行一次 validation，避免 interval 未触发时缺少 best。
    final_validation_loss = _evaluate_validation_loss(
        policy, pre, validation_dataset, config
    )
    if best_state is None or final_validation_loss < best_validation_loss:
        best_validation_loss = final_validation_loss
        best_train_loss = last_train_loss
        best_step = step
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in policy.state_dict().items()
        }
        if not any(m["step"] == step for m in metrics):
            metrics.append({"step": step, "validation_loss": final_validation_loss})

    if best_state is None or not math.isfinite(best_validation_loss):
        raise LeRobotACTError(f"{data.env_id} 未得到有限 validation loss")
    if not math.isfinite(best_train_loss):
        raise LeRobotACTError(f"{data.env_id} 未得到有限 train loss")

    policy.load_state_dict(best_state)
    _save_checkpoint_dir(
        policy, pre, post, task_dir / "best-validation"
    )

    train_count = config.train_episodes_per_task
    validation_count = config.validation_episodes_per_task
    metadata = {
        "schema_version": (
            ACT_SCHEMA_VERSION_V2_QVEL
            if config.use_qvel
            else ACT_SCHEMA_VERSION
        ),
        "env_id": data.env_id,
        "created_at": _utc_now(),
        "dataset_summary": str(config.dataset_summary),
        "dataset_hash": _dataset_hash(config.dataset_summary),
        "git_commit": _git_commit(),
        "device": device,
        "random_seed": task_seed,
        "task_index": task_index,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "training_steps": config.training_steps,
        "checkpoint_interval": config.checkpoint_interval,
        "normalization_mode": config.normalization_mode,
        "use_qvel": config.use_qvel,
        "warm_start_dir": (
            str(config.warm_start_dir) if config.warm_start_dir else None
        ),
        "architecture": {
            "dim_model": config.dim_model,
            "n_heads": config.n_heads,
            "dim_feedforward": config.dim_feedforward,
            "n_encoder_layers": config.n_encoder_layers,
            "n_decoder_layers": config.n_decoder_layers,
            "n_vae_encoder_layers": config.n_vae_encoder_layers,
            "latent_dim": config.latent_dim,
            "use_vae": config.use_vae,
            "kl_weight": config.kl_weight,
            "dropout": config.dropout,
        },
        "train_episode_ids": list(data.episode_ids[:train_count]),
        "validation_episode_ids": list(
            data.episode_ids[train_count : train_count + validation_count]
        ),
        "train_transitions": len(train_indices),
        "validation_transitions": len(validation_indices),
        "best_step": best_step,
        "best_train_loss": best_train_loss,
        "best_validation_loss": best_validation_loss,
        "best_rollout_dev": None,
        "versions": _environment_versions(),
    }
    _atomic_write_json(task_dir / "task_metadata.json", metadata)
    _write_metrics_jsonl(task_dir / "metrics.jsonl", metrics)

    result = ACTTaskTrainingResult(
        env_id=data.env_id,
        train_episode_ids=tuple(data.episode_ids[:train_count]),
        validation_episode_ids=tuple(
            data.episode_ids[train_count : train_count + validation_count]
        ),
        train_transitions=len(train_indices),
        validation_transitions=len(validation_indices),
        best_step=best_step,
        best_train_loss=best_train_loss,
        best_validation_loss=best_validation_loss,
        device=device,
        duration_seconds=round(time.perf_counter() - started, 6),
    )
    return result


def _dataset_hash(summary_path: Path) -> str:
    import hashlib

    try:
        digest = hashlib.sha256(
            Path(summary_path).read_bytes()
        ).hexdigest()
        return f"sha256:{digest}"
    except OSError:
        return "unavailable"


def train_lerobot_act(
    config: ACTTrainingConfig,
    *,
    observation_env_factory: Callable[[ManiSkillTaskConfig], object] = (
        make_maniskill_env
    ),
) -> ACTTrainingSummary:
    """对配置指定的任务训练并保存周期与 best-validation checkpoint。"""
    started = time.perf_counter()
    _prepare_output_directory(config.output_dir)
    device = _select_device(config.device)
    dataset = load_bc_dataset(
        config.dataset_summary,
        observation_env_factory=observation_env_factory,
    )
    results: list[ACTTaskTrainingResult] = []
    for env_id in config.tasks:
        results.append(_train_one_task(dataset[env_id], config))

    # 从磁盘重载 best-validation 验证 checkpoint 契约。
    load_lerobot_act_checkpoints(
        config.output_dir,
        device=device,
        tasks=config.tasks,
        checkpoint_subdir="best-validation",
    )
    summary = ACTTrainingSummary(
        schema_version=ACT_SCHEMA_VERSION,
        created_at=_utc_now(),
        dataset_summary=str(config.dataset_summary),
        output_dir=str(config.output_dir),
        device=device,
        chunk_size=config.chunk_size,
        n_action_steps=config.n_action_steps,
        training_steps=config.training_steps,
        batch_size=config.batch_size,
        validation_interval=config.validation_interval,
        checkpoint_interval=config.checkpoint_interval,
        tasks_selected=config.tasks,
        duration_seconds=round(time.perf_counter() - started, 6),
        tasks=tuple(results),
        complete=True,
    )
    _atomic_write_json(
        config.output_dir / "training_summary.json", asdict(summary)
    )
    return summary


def _mean_success_steps(summary: PolicyEvaluationSummary) -> float:
    steps = [
        item.steps for item in summary.episodes if item.success
    ]
    if not steps:
        return math.inf
    return float(np.mean(steps))


def _patch_task_metadata(task_dir: Path, updates: Mapping[str, object]) -> None:
    metadata_path = task_dir / "task_metadata.json"
    payload = _read_json_object(metadata_path)
    payload.update(updates)
    _atomic_write_json(metadata_path, payload)


def select_best_rollout_checkpoint(
    output_dir: Path,
    *,
    env_id: str,
    dev_seeds: Sequence[int],
    device: str,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    n_action_steps: int | None = None,
    action_smoothing: float = 1.0,
) -> Path:
    """在 dev seeds 上评估周期 checkpoint，按固定规则选出 best-rollout-dev。

    排名规则：成功数最高 → 成功平均步数最少 → 训练 step 最早。
    validation loss 不参与排名。
    """
    output_dir = Path(output_dir).expanduser().resolve()
    task_dir = output_dir / env_id
    step_dirs = sorted(
        (d for d in task_dir.glob("step-*") if d.is_dir()),
        key=lambda path: int(path.name.split("-")[1]),
    )
    if not step_dirs:
        raise LeRobotACTError(f"{env_id} 没有周期 checkpoint")
    checked_seeds = _checked_seeds(dev_seeds)
    results: list[dict[str, object]] = []
    for step_dir in step_dirs:
        subdir = step_dir.name
        policies = load_lerobot_act_checkpoints(
            output_dir,
            device=device,
            tasks=[env_id],
            checkpoint_subdir=subdir,
            n_action_steps=n_action_steps,
            action_smoothing=action_smoothing,
        )
        summary = evaluate_policy_seeds(
            "act", policies, checked_seeds, env_factory=env_factory,
            tasks=[env_id],
        )
        results.append(
            {
                "checkpoint": subdir,
                "step": int(subdir.split("-")[1]),
                "successes": summary.total_successes,
                "episodes": summary.total_episodes,
                "success_rate": summary.success_rate,
                "mean_steps_success": _mean_success_steps(summary),
                "mean_inference_ms": summary.mean_inference_ms,
            }
        )

    def sort_key(item: Mapping[str, object]) -> tuple[int, float, int]:
        return (
            -int(item["successes"]),
            float(item["mean_steps_success"]),
            int(item["step"]),
        )

    ranked = sorted(results, key=sort_key)
    winner = ranked[0]
    winner_name = str(winner["checkpoint"])
    winner_dir = task_dir / "best-rollout-dev"
    shutil.rmtree(winner_dir, ignore_errors=True)
    shutil.copytree(task_dir / winner_name, winner_dir)
    selection = {
        "schema_version": ACT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "env_id": env_id,
        "dev_seeds": list(checked_seeds),
        "selection_rule": (
            "successes desc; then success mean steps asc; then step asc"
        ),
        "ranked": ranked,
        "selected": winner_name,
    }
    _atomic_write_json(task_dir / "rollout_selection.json", selection)
    _patch_task_metadata(task_dir, {"best_rollout_dev": winner_name})
    return winner_dir


def _resolve_checkpoint_subdir(
    output_dir: Path, env_id: str, checkpoint_subdir: str | None
) -> Path:
    """解析每任务 checkpoint 子目录；None 时按优先级自动选择。"""
    task_dir = output_dir / env_id
    if checkpoint_subdir is not None:
        path = task_dir / checkpoint_subdir
        if not path.is_dir():
            raise LeRobotACTError(
                f"{env_id} checkpoint 子目录不存在：{path}"
            )
        return path
    for candidate in ("best-rollout-dev", "best", "best-validation"):
        path = task_dir / candidate
        if path.is_dir():
            return path
    raise LeRobotACTError(f"{env_id} 没有可用 checkpoint 子目录")


def _require_checkpoint_directory(path: Path) -> None:
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        if not (path / name).is_file():
            raise LeRobotACTError(f"checkpoint 缺少 {name}：{path}")
    for config_name in (
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        config = _read_json_object(path / config_name)
        steps = config.get("steps")
        if not isinstance(steps, list):
            raise LeRobotACTError(
                f"{path / config_name} 缺少 processor steps"
            )
        for step in steps:
            state_file = step.get("state_file")
            if state_file is not None and not (path / str(state_file)).is_file():
                raise LeRobotACTError(
                    f"checkpoint 缺少 processor state："
                    f"{path / str(state_file)}"
                )


def load_lerobot_act_checkpoints(
    output_dir: Path,
    *,
    device: str,
    n_action_steps: int | None = None,
    action_smoothing: float = 1.0,
    tasks: Sequence[str] | None = None,
    checkpoint_subdir: str | None = None,
) -> dict[str, "LeRobotACTPolicy"]:
    """只从本地目录加载 policy 和 processors，并校验完整契约。

    ``n_action_steps`` 用于诊断迭代：默认沿用保存配置，可显式覆盖为
    ``1`` 以关闭开环陈旧动作，模型权重与 chunk 配置保持不变。
    ``action_smoothing`` 为 rollout 位置增量的 EMA 系数（(0,1]）。
    ``tasks`` 限定加载的任务；``checkpoint_subdir`` 选择每任务下的
    checkpoint 子目录，默认依次解析 ``best-rollout-dev`` / ``best`` /
    ``best-validation``。
    """
    device = _select_device(device)
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        raise LeRobotACTError(f"输出目录不存在：{output_dir}")
    if n_action_steps is not None:
        _require_positive_int(n_action_steps, "n_action_steps")
    _validate_smoothing(action_smoothing)
    checked_tasks = tuple(tasks) if tasks is not None else SUPPORTED_TASKS
    for env_id in checked_tasks:
        if env_id not in SUPPORTED_TASKS:
            raise LeRobotACTError(f"不支持的任务 {env_id!r}")
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor import (
            DataProcessorPipeline,
            policy_action_to_transition,
            transition_to_policy_action,
        )
    except (ImportError, OSError) as exc:
        raise _lerobot_import_error(exc) from exc

    policies: dict[str, LeRobotACTPolicy] = {}
    for env_id in checked_tasks:
        best_dir = _resolve_checkpoint_subdir(output_dir, env_id, checkpoint_subdir)
        _require_checkpoint_directory(best_dir)
        saved = _read_json_object(best_dir / "config.json")
        chunk_size = int(saved.get("chunk_size", -1))
        saved_n_action_steps = int(saved.get("n_action_steps", -1))
        if not 1 <= saved_n_action_steps <= chunk_size:
            raise LeRobotACTError(f"{env_id} 保存的 chunk 配置错误")
        saved_robot_dim = int(
            saved.get("input_features", {})
            .get(OBS_STATE_KEY, {})
            .get("shape", [ROBOT_STATE_DIM])[0]
        )
        if saved_robot_dim not in (ROBOT_STATE_DIM, ROBOT_STATE_V2_DIM):
            raise LeRobotACTError(
                f"{env_id} robot state 维度不受支持：{saved_robot_dim}"
            )
        eval_n_action_steps = (
            n_action_steps if n_action_steps is not None else saved_n_action_steps
        )
        if not 1 <= eval_n_action_steps <= chunk_size:
            raise LeRobotACTError(
                f"{env_id} 覆盖的 n_action_steps 必须位于 "
                f"[1, {chunk_size}]"
            )
        config = make_lerobot_act_config(
            chunk_size=chunk_size,
            n_action_steps=eval_n_action_steps,
            device=device,
            robot_state_dim=saved_robot_dim,
            dim_model=int(saved["dim_model"]),
            n_heads=int(saved["n_heads"]),
            dim_feedforward=int(saved["dim_feedforward"]),
            n_encoder_layers=int(saved["n_encoder_layers"]),
            n_decoder_layers=int(saved["n_decoder_layers"]),
            n_vae_encoder_layers=int(saved["n_vae_encoder_layers"]),
            latent_dim=int(saved["latent_dim"]),
            use_vae=bool(saved["use_vae"]),
            kl_weight=float(saved["kl_weight"]),
            dropout=float(saved["dropout"]),
        )
        policy = ACTPolicy.from_pretrained(best_dir, config=config)
        policy.to(device)
        pre = DataProcessorPipeline.from_pretrained(
            best_dir, config_filename="policy_preprocessor.json"
        )
        post = DataProcessorPipeline.from_pretrained(
            best_dir,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        policies[env_id] = LeRobotACTPolicy(
            env_id=env_id,
            policy=policy,
            preprocessor=pre,
            postprocessor=post,
            action_smoothing=action_smoothing,
        )
    return policies


def _validate_smoothing(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= 1
    ):
        raise ValueError("action_smoothing 必须位于 (0, 1]")
    return float(value)


def _to_float32_numpy(value: object) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    return np.asarray(candidate, dtype=np.float32).reshape(-1)


class LeRobotACTPolicy:
    """从磁盘 checkpoint 构造的单任务 ACT rollout wrapper。"""

    phase = "act"

    def __init__(
        self,
        *,
        env_id: str,
        policy: object,
        preprocessor: object,
        postprocessor: object,
        action_smoothing: float = 1.0,
    ) -> None:
        self.env_id = env_id
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self._smoothing = _validate_smoothing(action_smoothing)
        self._smoothed: np.ndarray | None = None
        config_obj = getattr(policy, "config", None)
        n_action_steps = (
            int(config_obj.n_action_steps)
            if config_obj is not None
            else None
        )
        if n_action_steps is None:
            queue = getattr(policy, "_action_queue", None)
            maxlen = getattr(queue, "maxlen", None)
            n_action_steps = int(maxlen) if maxlen else 1
        self._n_action_steps = n_action_steps
        robot_feature = getattr(config_obj, "robot_state_feature", None)
        self._use_qvel = bool(
            robot_feature is not None
            and tuple(robot_feature.shape) == (ROBOT_STATE_V2_DIM,)
        )
        self.action_queries = 0
        self.chunk_queries = 0
        self.chunk_query_id = 0
        self.chunk_offset = -1
        self.queue_length_before_action = 0
        self.queue_length_after_action = 0
        self.nonfinite_actions = 0
        self.out_of_bounds_actions = 0
        self.inference_ms = 0.0
        self.last_raw_action: np.ndarray | None = None

    def reset(self) -> None:
        """清空 LeRobot 内部 action queue，避免跨 episode 泄漏。"""
        self._policy.reset()
        self._smoothed = None
        self.action_queries = 0
        self.chunk_queries = 0
        self.chunk_query_id = 0
        self.chunk_offset = -1
        self.queue_length_before_action = 0
        self.queue_length_after_action = 0
        self.nonfinite_actions = 0
        self.out_of_bounds_actions = 0
        self.inference_ms = 0.0
        self.last_raw_action = None

    def _queue_empty(self) -> bool:
        queue = getattr(self._policy, "_action_queue", None)
        if queue is not None and hasattr(queue, "__len__"):
            return len(queue) == 0
        return False

    def act(self, observation: Mapping[str, object]) -> np.ndarray:
        """live observation → 18D → 5/13D → processor → 四维动作。"""
        torch = _torch_module()
        state = observation_to_state(observation)
        if self._use_qvel:
            from behavior_cloning import observation_to_arm_qvel

            robot, env = split_lerobot_act_state_v2(
                state, observation_to_arm_qvel(observation)
            )
        else:
            robot, env = split_lerobot_act_state(state)
        self.action_queries += 1
        queue = getattr(self._policy, "_action_queue", None)
        before = len(queue) if queue is not None and hasattr(queue, "__len__") else 0
        self.queue_length_before_action = before
        if before == 0:
            self.chunk_query_id += 1
            self.chunk_queries += 1
        batch = {
            OBS_STATE_KEY: torch.from_numpy(robot),
            OBS_ENV_STATE_KEY: torch.from_numpy(env),
        }
        started = time.perf_counter()
        processed = self._pre(batch)
        action = self._policy.select_action(processed)
        action = self._post(action)
        self.inference_ms += (time.perf_counter() - started) * 1000.0
        after = len(queue) if queue is not None and hasattr(queue, "__len__") else 0
        self.queue_length_after_action = after
        self.chunk_offset = max(0, self._n_action_steps - 1 - after)
        array = _to_float32_numpy(action)
        if array.shape != (ACTION_DIM,):
            raise LeRobotACTError(
                f"{self.env_id} ACT 输出 shape={array.shape}，期望 ({ACTION_DIM},)"
            )
        if not np.all(np.isfinite(array)):
            self.nonfinite_actions += 1
            raise LeRobotACTError(f"{self.env_id} ACT 输出包含非有限数")
        self.last_raw_action = array.copy()
        clipped = np.clip(array, -ACTION_SPACE_BOUND, ACTION_SPACE_BOUND)
        if self._smoothing < 1.0:
            position = clipped[:3]
            if self._smoothed is None:
                self._smoothed = position.copy()
            else:
                self._smoothed = (
                    self._smoothing * position
                    + (1.0 - self._smoothing) * self._smoothed
                )
            clipped = np.concatenate((self._smoothed, clipped[3:]))
        if not np.allclose(clipped, array, atol=1e-6):
            self.out_of_bounds_actions += 1
        return clipped.astype(np.float32)


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


def _policy_counter(policy: object, name: str) -> int:
    return int(getattr(policy, name, 0))


def _policy_float_counter(policy: object, name: str) -> float:
    return float(getattr(policy, name, 0.0))


def evaluate_policy_seeds(
    policy_name: str,
    policies: Mapping[str, object],
    seeds: Sequence[int],
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    tasks: Sequence[str] | None = None,
) -> PolicyEvaluationSummary:
    """在未见 seed 上执行无 fallback 的真实 rollout，并聚合证据。"""
    checked_seeds = _checked_seeds(seeds)
    checked_tasks = tuple(tasks) if tasks is not None else SUPPORTED_TASKS
    for env_id in checked_tasks:
        if env_id not in policies:
            raise LeRobotACTError(f"缺少任务策略：{env_id}")
    episodes: list[ACTEvaluationEpisode] = []
    for env_id in checked_tasks:
        policy = policies[env_id]
        for seed in checked_seeds:
            config = ManiSkillTaskConfig(env_id=env_id, seed=seed)
            env: Any | None = None
            try:
                env = env_factory(config)
                result = run_episode_in_env(config, env, policy)
                episode = ACTEvaluationEpisode(
                    env_id=env_id,
                    seed=seed,
                    steps=result.steps,
                    total_reward=result.total_reward,
                    success=result.success,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    action_queries=_policy_counter(policy, "action_queries"),
                    chunk_queries=_policy_counter(policy, "chunk_queries"),
                    inference_ms=_policy_float_counter(
                        policy, "inference_ms"
                    ),
                    nonfinite_actions=_policy_counter(
                        policy, "nonfinite_actions"
                    ),
                    out_of_bounds_actions=_policy_counter(
                        policy, "out_of_bounds_actions"
                    ),
                )
            except Exception as exc:
                episode = ACTEvaluationEpisode(
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
            episodes.append(episode)

    tasks: list[ACTEvaluationTask] = []
    for env_id in checked_tasks:
        task_episodes = [
            item for item in episodes if item.env_id == env_id
        ]
        successes = sum(item.success for item in task_episodes)
        steps_list = [item.steps for item in task_episodes]
        tasks.append(
            ACTEvaluationTask(
                env_id=env_id,
                episodes=len(task_episodes),
                successes=successes,
                failures=len(task_episodes) - successes,
                success_rate=successes / len(task_episodes),
                mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
                exceptions=sum(item.error is not None for item in task_episodes),
            )
        )

    total_successes = sum(item.success for item in episodes)
    total_episodes = len(episodes)
    mean_inference = (
        float(np.mean([item.inference_ms for item in episodes]))
        if episodes
        else 0.0
    )
    mean_steps = (
        float(np.mean([item.steps for item in episodes]))
        if episodes
        else 0.0
    )
    return PolicyEvaluationSummary(
        policy_name=policy_name,
        schema_version=ACT_SCHEMA_VERSION,
        created_at=_utc_now(),
        requested_per_task=len(checked_seeds),
        total_episodes=total_episodes,
        total_successes=total_successes,
        total_failures=total_episodes - total_successes,
        success_rate=total_successes / total_episodes,
        mean_inference_ms=mean_inference,
        mean_steps=mean_steps,
        total_chunk_queries=sum(item.chunk_queries for item in episodes),
        total_action_queries=sum(item.action_queries for item in episodes),
        nonfinite_actions=sum(item.nonfinite_actions for item in episodes),
        out_of_bounds_actions=sum(
            item.out_of_bounds_actions for item in episodes
        ),
        tasks=tuple(tasks),
        episodes=tuple(episodes),
    )


def evaluate_lerobot_act(
    output_dir: Path,
    *,
    seeds: Sequence[int],
    device: str,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    n_action_steps: int | None = None,
    action_smoothing: float = 1.0,
    tasks: Sequence[str] | None = None,
    checkpoint_subdir: str | None = None,
) -> PolicyEvaluationSummary:
    """从磁盘重载 ACT checkpoint 后执行固定 seed 评测并落盘。"""
    output_dir = Path(output_dir).expanduser().resolve()
    policies = load_lerobot_act_checkpoints(
        output_dir,
        device=device,
        n_action_steps=n_action_steps,
        action_smoothing=action_smoothing,
        tasks=tasks,
        checkpoint_subdir=checkpoint_subdir,
    )
    summary = evaluate_policy_seeds(
        "act", policies, seeds, env_factory=env_factory, tasks=tasks
    )
    _atomic_write_json(output_dir / "evaluation.json", asdict(summary))
    return summary


def evaluate_bc_for_comparison(
    checkpoint_path: Path,
    *,
    seeds: Sequence[int],
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    tasks: Sequence[str] | None = None,
) -> PolicyEvaluationSummary:
    """加载 Week 12 BC checkpoint，在相同 seed 上评测且不回写 BC 目录。"""
    from behavior_cloning import load_bc_checkpoint

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    policies = load_bc_checkpoint(checkpoint_path)
    return evaluate_policy_seeds(
        "bc", policies, seeds, env_factory=env_factory, tasks=tasks
    )


def evaluate_scripted_for_comparison(
    *,
    seeds: Sequence[int],
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    tasks: Sequence[str] | None = None,
) -> PolicyEvaluationSummary:
    """在相同 seed 上运行 scripted expert 作为上界参考。"""
    policies = {
        env_id: ScriptedPolicy(env_id) for env_id in SUPPORTED_TASKS
    }
    return evaluate_policy_seeds(
        "scripted", policies, seeds, env_factory=env_factory, tasks=tasks
    )


def _validate_threshold(threshold: float) -> float:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 <= float(threshold) <= 1
    ):
        raise ValueError("threshold 必须位于 [0, 1]")
    return float(threshold)


def _compute_effective_threshold(threshold: float, bc_rate: float) -> float:
    """门禁为 ``max(threshold, 同轮 BC 成功率)``，而不是固定值。"""
    return max(_validate_threshold(threshold), float(bc_rate))


def compare_act_bc_scripted(
    output_dir: Path,
    *,
    bc_checkpoint_path: Path,
    seeds: Sequence[int],
    device: str,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    threshold: float = 0.75,
    n_action_steps: int | None = None,
    action_smoothing: float = 1.0,
    tasks: Sequence[str] | None = None,
    checkpoint_subdir: str | None = None,
) -> ACTComparisonSummary:
    """在同一 checkout、同 seed、同环境上比较 ACT/BC/scripted。"""
    threshold = _validate_threshold(threshold)
    checked_seeds = tuple(_checked_seeds(seeds))
    output_dir = Path(output_dir).expanduser().resolve()
    bc_checkpoint_path = Path(bc_checkpoint_path).expanduser().resolve()
    act = evaluate_lerobot_act(
        output_dir,
        seeds=checked_seeds,
        device=device,
        env_factory=env_factory,
        n_action_steps=n_action_steps,
        action_smoothing=action_smoothing,
        tasks=tasks,
        checkpoint_subdir=checkpoint_subdir,
    )
    bc = evaluate_bc_for_comparison(
        bc_checkpoint_path,
        seeds=checked_seeds,
        env_factory=env_factory,
        tasks=tasks,
    )
    scripted = evaluate_scripted_for_comparison(
        seeds=checked_seeds, env_factory=env_factory, tasks=tasks
    )
    effective_threshold = _compute_effective_threshold(
        threshold, bc.success_rate
    )
    act_passed = act.success_rate >= effective_threshold
    summary = ACTComparisonSummary(
        schema_version=ACT_SCHEMA_VERSION,
        created_at=_utc_now(),
        output_dir=str(output_dir),
        bc_checkpoint_path=str(
            Path(bc_checkpoint_path).expanduser().resolve()
        ),
        seeds=checked_seeds,
        policies=(act, bc, scripted),
        act_rate=act.success_rate,
        bc_rate=bc.success_rate,
        scripted_rate=scripted.success_rate,
        threshold=threshold,
        effective_threshold=effective_threshold,
        act_passed=act_passed,
        n_action_steps=n_action_steps,
    )
    _atomic_write_json(output_dir / "comparison.json", asdict(summary))
    return summary


def probe_lerobot_act(device: str = "mps") -> dict[str, object]:
    """兼容性探针：import、device、forward/backward、save/load。"""
    device = _select_device(device)
    torch = _torch_module()
    report: dict[str, object] = {
        "device": device,
        "versions": _environment_versions(),
    }
    act_config = make_lerobot_act_config(
        chunk_size=8, n_action_steps=4, device=device
    )
    stats = {
        OBS_STATE_KEY: {
            "mean": torch.zeros(ROBOT_STATE_DIM),
            "std": torch.ones(ROBOT_STATE_DIM),
        },
        OBS_ENV_STATE_KEY: {
            "mean": torch.zeros(ENV_STATE_DIM),
            "std": torch.ones(ENV_STATE_DIM),
        },
        ACTION_KEY: {
            "mean": torch.zeros(ACTION_DIM),
            "std": torch.ones(ACTION_DIM),
        },
    }
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.processor_act import (
            make_act_pre_post_processors,
        )
    except (ImportError, OSError) as exc:
        raise _lerobot_import_error(exc) from exc
    pre, post = make_act_pre_post_processors(act_config, dataset_stats=stats)
    policy = ACTPolicy(act_config)
    policy.to(device)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    batch = _batch_from_indices(
        LeRobotACTChunkDataset(
            env_id="probe",
            robot_state=rng.standard_normal((4, ROBOT_STATE_DIM)).astype(
                np.float32
            ),
            env_state=rng.standard_normal((4, ENV_STATE_DIM)).astype(
                np.float32
            ),
            action_chunk=rng.standard_normal((4, 8, ACTION_DIM)).astype(
                np.float32
            ),
            action_is_pad=np.zeros((4, 8), dtype=bool),
            chunk_size=8,
        ),
        range(4),
    )
    processed = pre(batch)
    policy.train()
    loss, _ = policy.forward(processed)
    report["loss"] = float(loss.item())
    if not bool(torch.isfinite(loss)):
        raise LeRobotACTError("ACT forward loss 非有限")
    loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    if not gradients or not all(
        bool(torch.isfinite(g).all()) for g in gradients
    ):
        raise LeRobotACTError("ACT 梯度非有限")
    report["grad_finite"] = True

    import tempfile

    from lerobot.processor import (
        DataProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="lerobot_act_probe_"))
    policy.save_pretrained(tmpdir)
    pre.save_pretrained(tmpdir)
    post.save_pretrained(tmpdir)
    _require_checkpoint_directory(tmpdir)

    policy.eval()
    policy.reset()
    live = {
        OBS_STATE_KEY: torch.from_numpy(
            rng.standard_normal(ROBOT_STATE_DIM).astype(np.float32)
        ),
        OBS_ENV_STATE_KEY: torch.from_numpy(
            rng.standard_normal(ENV_STATE_DIM).astype(np.float32)
        ),
    }
    processed_live = pre(live)
    action = post(policy.select_action(processed_live))
    if tuple(action.shape) != (1, ACTION_DIM) or not bool(
        torch.isfinite(action).all()
    ):
        raise LeRobotACTError("ACT 推理输出非法")

    loaded = ACTPolicy.from_pretrained(tmpdir, config=act_config)
    loaded.to(device)
    with torch.no_grad():
        reloaded_loss = loaded(processed)[0]
    report["reload_loss"] = float(reloaded_loss.item())
    if not bool(torch.isfinite(reloaded_loss)):
        raise LeRobotACTError("ACT 重载后 forward loss 非有限")
    DataProcessorPipeline.from_pretrained(
        tmpdir, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    )
    DataProcessorPipeline.from_pretrained(
        tmpdir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    report["probe_passed"] = True
    return report
