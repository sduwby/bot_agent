"""第 16 周：视觉 ACT——单相机 RGB + proprioception 训练、checkpoint 与 rollout。

直接从第 15 周 LeRobotDataset v3（week15-cam）读回图像/状态/动作，构造含图像的
action chunk 数据集，训练 ResNet18 backbone + ACT transformer，并做 live RGB
渲染 rollout 与显存/时长记录。
"""

from __future__ import annotations

import json
import math
import random
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
    observation_to_state,
)
from lerobot_act import (
    ACTION_KEY,
    ENV_STATE_DIM,
    OBS_ENV_STATE_KEY,
    OBS_STATE_KEY,
    ROBOT_STATE_DIM,
    LeRobotACTError,
    PolicyEvaluationSummary,
    split_lerobot_act_state,
)
from lerobot_pipeline import (
    OBS_IMAGE_KEY,
    _image_hwc,
    make_render_env,
    open_lerobot_dataset,
)
from maniskill_intro import (
    SUPPORTED_TASKS,
    ManiSkillTaskConfig,
    ManiSkillUnavailableError,
)

VISUAL_SCHEMA_VERSION = "maniskill_visual_act_v1"

# LeRobot 内部图像 batch 键（与官方 ACTPolicy 一致）。
OBS_IMAGES_KEY = "observation.images"
ACTION_IS_PAD_KEY = "action_is_pad"

TASK_TEXT_TO_ENV_ID = {
    "PickCube-v1": "PickCube-v1",
    "PushCube-v1": "PushCube-v1",
}


class VisualACTError(RuntimeError):
    """视觉 ACT 数据、训练、checkpoint 或评测契约错误。"""


@dataclass(frozen=True)
class VisualTaskData:
    """一个任务从 v3 读回的、按 episode 连续排列的视觉 transition。"""

    env_id: str
    episode_ids: tuple[int, ...]
    episode_lengths: tuple[int, ...]
    images: np.ndarray      # [T, 3, H, W] float32 [0,1]
    robot_state: np.ndarray  # [T, 5] proprioception
    env_state: np.ndarray    # [T, 13] environment state
    actions: np.ndarray     # [T, 4]


@dataclass(frozen=True)
class VisualACTChunkDataset:
    """一个任务某个切分的视觉 feature 与 action chunk。"""

    env_id: str
    images: np.ndarray          # [N, 3, H, W] float32
    robot_state: np.ndarray     # [N, 5]
    env_state: np.ndarray       # [N, 13]
    action_chunk: np.ndarray    # [N, chunk, 4]
    action_is_pad: np.ndarray   # [N, chunk] bool
    chunk_size: int


@dataclass(frozen=True)
class VisualACTTrainingConfig:
    """视觉 ACT 的数据、架构与优化参数。"""

    dataset_dir: Path
    repo_id: str
    output_dir: Path
    tasks: tuple[str, ...] = SUPPORTED_TASKS
    resolution: int = 224
    chunk_size: int = 8
    n_action_steps: int = 4
    training_steps: int = 20_000
    batch_size: int = 16
    validation_interval: int = 250
    checkpoint_interval: int | None = 2500
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = (
        "ResNet18_Weights.IMAGENET1K_V1"
    )
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
    random_seed: int = 20260816
    device: str = "mps"
    train_episodes_per_task: int = 80
    validation_episodes_per_task: int = 20
    dagger_data: Path | None = None
    #: 第 38 周 P0/B0–B3 的**额外观测通道**：`real_geometry.variant_feature_names(variant)`
    #: 给出的特征名列表（`()` = B0，即与历史行为完全一致）。
    #: 这些列**追加在 ENV 辅助向量的尾部**（13 → 13+n），因此不改动 LeRobot 的
    #: FeatureType/归一化映射、也不新增 feature key —— 老 checkpoint 与老数据集不受影响。
    #: 语义与既有 ENV 通道一致：**不做归一化**，上游需自行标准化。
    extra_features: tuple[str, ...] = ()
    #: **ENV 通道口径**（第 38 周 P0 消融，2026-10-11 决策）：
    #:
    #: - ``"full"``：13 维特权 env_state + 额外列（历史行为 = 旧 B0）；
    #: - ``"extra_only"``：**只保留额外列**（``extra_features=()`` 时等于彻底不用 ENV 通道）。
    #:
    #: 这是让 B0–B3 消融**有信息量**的口径：`env_state` 本身已含
    #: `obj_pos / (tcp−obj) / is_grasped`，而 B1/B2 的列正是它们的确定性函数 ——
    #: B0 也带着这些特权量时，"补列"不可能带来增益（ΔGrasp 只可能是噪声）。
    #: 改成 B0 不带 ENV 后，B1–B3 才是在**真的补信息**。
    #: 该开关只改**策略侧用多少列**；数据集仍然存 13+n（见 `dataset_env_state_dim`）。
    env_state_mode: str = "full"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_dir", Path(self.dataset_dir).expanduser().resolve()
        )
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        if self.dagger_data is not None:
            object.__setattr__(
                self,
                "dagger_data",
                Path(self.dagger_data).expanduser().resolve(),
            )
            if not self.dagger_data.is_file():
                raise ValueError(f"dagger_data 不存在：{self.dagger_data}")
        _require_positive_int(self.resolution, "resolution")
        _require_positive_int(self.chunk_size, "chunk_size")
        _require_positive_int(self.n_action_steps, "n_action_steps")
        _require_positive_int(self.training_steps, "training_steps")
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(self.validation_interval, "validation_interval")
        _require_positive_int(
            self.train_episodes_per_task, "train_episodes_per_task"
        )
        _require_positive_int(
            self.validation_episodes_per_task, "validation_episodes_per_task"
        )
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps 不能大于 chunk_size")
        if self.device not in ("mps", "cpu", "cuda"):
            raise ValueError("device 只支持 mps、cuda 或 cpu")
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError("vision_backbone 必须是 ResNet 变体")
        checked: list[str] = []
        for env_id in self.tasks:
            if env_id not in SUPPORTED_TASKS:
                raise ValueError(
                    f"不支持的任务 {env_id!r}，仅支持 {SUPPORTED_TASKS}"
                )
            if env_id in checked:
                raise ValueError(f"任务重复：{env_id}")
            checked.append(env_id)
        object.__setattr__(self, "tasks", tuple(checked))
        checked_extra: list[str] = []
        for name in self.extra_features:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("extra_features 的元素必须是非空字符串")
            if name in checked_extra:
                raise ValueError(f"extra_features 重复：{name}")
            checked_extra.append(name)
        object.__setattr__(self, "extra_features", tuple(checked_extra))
        if self.env_state_mode not in ("full", "extra_only"):
            raise ValueError(
                "env_state_mode 必须是 'full' 或 'extra_only'，"
                f"实际 {self.env_state_mode!r}"
            )

    @property
    def dataset_env_state_dim(self) -> int:
        """数据集里 `observation.environment_state` 应有的列数（13 + 额外列）。"""
        return ENV_STATE_DIM + len(self.extra_features)

    @property
    def env_state_dim(self) -> int:
        """**策略侧** ENV 向量维度；``extra_only`` 模式下可为 0（表示不用 ENV 通道）。"""
        if self.env_state_mode == "extra_only":
            return len(self.extra_features)
        return self.dataset_env_state_dim


@dataclass(frozen=True)
class VisualTaskTrainingResult:
    """一个视觉 ACT 任务的切分规模与最佳训练指标。"""

    env_id: str
    train_episode_ids: tuple[int, ...]
    validation_episode_ids: tuple[int, ...]
    train_transitions: int
    validation_transitions: int
    best_step: int
    best_train_loss: float
    best_validation_loss: float
    device: str
    peak_rss_mb: float
    duration_seconds: float


@dataclass(frozen=True)
class VisualACTTrainingSummary:
    """视觉 ACT 训练产物与分任务结果。"""

    schema_version: str
    created_at: str
    dataset_dir: str
    repo_id: str
    output_dir: str
    device: str
    resolution: int
    chunk_size: int
    n_action_steps: int
    training_steps: int
    batch_size: int
    vision_backbone: str
    tasks_selected: tuple[str, ...]
    dagger_data: str | None
    duration_seconds: float
    peak_rss_mb: float
    tasks: tuple[VisualTaskTrainingResult, ...]
    complete: bool


@dataclass(frozen=True)
class ResourceReport:
    """训练/评测的显存与时长记录。"""

    schema_version: str
    created_at: str
    device: str
    resolution: int
    batch_size: int
    peak_rss_mb_train: float
    peak_rss_mb_total: float
    training_seconds: float
    evaluation_seconds: float
    mean_inference_ms: float
    total_episodes: int


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


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"输出路径不是目录：{path}")
        if any(path.iterdir()):
            raise ValueError(
                f"输出目录必须为空，拒绝覆盖已有数据：{path}"
            )


def _torch_module() -> object:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise VisualACTError("视觉 ACT 训练需要 PyTorch") from exc
    return torch


def _select_device(device: str) -> str:
    torch = _torch_module()
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise VisualACTError("mps 不可用，请改用 cpu 或 cuda")
        return "mps"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise VisualACTError("cuda 不可用，请改用 mps 或 cpu")
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise VisualACTError(f"device 不受支持：{device!r}")


def _lerobot_import_error(exc: Exception) -> VisualACTError:
    return VisualACTError(
        f"LeRobot/ManiSkill 依赖不可用：{type(exc).__name__}: {exc}"
    )


def _resize_image(image: np.ndarray, resolution: int) -> np.ndarray:
    """把 uint8 CHW 图像缩放到 resolution×resolution 并转 float[0,1]。"""
    try:
        from PIL import Image
    except (ImportError, OSError) as exc:
        raise VisualACTError("图像缩放需要 Pillow") from exc
    array = np.asarray(image)
    if array.ndim != 3:
        raise VisualACTError(f"图像维度错误：{array.shape}")
    h, w, channels = array.shape
    if channels != 3:
        # CHW → HWC。
        array = np.transpose(array, (1, 2, 0))
        h, w, channels = array.shape
    if channels != 3 or not (
        array.dtype == np.uint8 or array.dtype == np.float32
    ):
        raise VisualACTError(f"图像 dtype/channels 错误：{array.dtype} {channels}")
    pil = Image.fromarray(array) if array.dtype == np.uint8 else Image.fromarray(
        (array * 255).astype(np.uint8)
    )
    resized = pil.resize((resolution, resolution), Image.BILINEAR)
    result = np.asarray(resized, dtype=np.float32) / 255.0
    return np.transpose(result, (2, 0, 1))  # (3, H, W)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualACTError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise VisualACTError(f"JSON 顶层必须是 object：{path}")
    return value


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise VisualACTError(
            f"{name} shape={array.shape} 期望 {shape} 或含非有限数"
        )
    return array


def load_visual_dataset(
    config: VisualACTTrainingConfig,
) -> dict[str, VisualTaskData]:
    """从 v3 读回图像/状态/动作并按任务、episode 分组。"""
    dataset = open_lerobot_dataset(config.dataset_dir, config.repo_id)
    grouped: dict[str, dict[int, dict[str, list[np.ndarray]]]] = {
        env_id: {} for env_id in config.tasks
    }
    for index in range(len(dataset)):
        item = dataset[index]
        task_text = str(item.get("task", ""))
        env_id = TASK_TEXT_TO_ENV_ID.get(task_text)
        if env_id not in config.tasks:
            continue
        episode_index = int(item["episode_index"])
        image = _image_hwc(item[OBS_IMAGE_KEY])
        image_resized = _resize_image(image, config.resolution)
        robot_state = _finite_array(
            item[OBS_STATE_KEY], (ROBOT_STATE_DIM,), "observation.state"
        )
        env_state = _finite_array(
            item[OBS_ENV_STATE_KEY],
            (config.dataset_env_state_dim,),
            f"environment_state（B 变体额外 {len(config.extra_features)} 列）",
        )
        env_state = _select_env_state(env_state, config)
        action = _finite_array(item[ACTION_KEY], (ACTION_DIM,), "action")
        bucket = grouped[env_id].setdefault(
            episode_index,
            {"images": [], "robot_states": [], "env_states": [], "actions": []},
        )
        bucket["images"].append(image_resized)
        bucket["robot_states"].append(robot_state)
        bucket["env_states"].append(env_state)
        bucket["actions"].append(action)

    result: dict[str, VisualTaskData] = {}
    for env_id in config.tasks:
        episodes = grouped[env_id]
        if not episodes:
            raise VisualACTError(f"{env_id} 没有读回任何 episode")
        episode_ids: list[int] = []
        lengths: list[int] = []
        image_chunks: list[np.ndarray] = []
        robot_state_chunks: list[np.ndarray] = []
        env_state_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        for episode_index in sorted(episodes):
            bucket = episodes[episode_index]
            images = np.stack(bucket["images"], axis=0)
            robot_states = np.stack(bucket["robot_states"], axis=0)
            env_states = np.stack(bucket["env_states"], axis=0)
            actions = np.stack(bucket["actions"], axis=0)
            length = len(actions)
            if images.shape[0] != length:
                raise VisualACTError(
                    f"{env_id}/episode {episode_index} 图像与动作数不一致"
                )
            episode_ids.append(episode_index)
            lengths.append(length)
            image_chunks.append(images)
            robot_state_chunks.append(robot_states)
            env_state_chunks.append(env_states)
            action_chunks.append(actions)
        result[env_id] = VisualTaskData(
            env_id=env_id,
            episode_ids=tuple(episode_ids),
            episode_lengths=tuple(lengths),
            images=np.concatenate(image_chunks, axis=0),
            robot_state=np.concatenate(robot_state_chunks, axis=0),
            env_state=np.concatenate(env_state_chunks, axis=0),
            actions=np.concatenate(action_chunks, axis=0),
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
    data: VisualTaskData,
    *,
    train_episodes: int,
    validation_episodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按 episode 顺序切分并返回两组 transition 索引。"""
    _require_positive_int(train_episodes, "train_episodes")
    _require_positive_int(validation_episodes, "validation_episodes")
    total = train_episodes + validation_episodes
    if total > len(data.episode_ids):
        raise VisualACTError(
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
        raise VisualACTError(f"{data.env_id} train/validation 发生泄漏")
    return train_indices, validation_indices


def build_visual_chunk_dataset(
    data: VisualTaskData,
    indices: Sequence[int],
    chunk_size: int,
) -> VisualACTChunkDataset:
    """构造含图像的 action chunk 数据集。"""
    _require_positive_int(chunk_size, "chunk_size")
    index_array = np.asarray(list(indices), dtype=np.int64)
    if index_array.ndim != 1 or index_array.size == 0:
        raise VisualACTError("indices 不能为空")
    offsets = np.concatenate(([0], np.cumsum(data.episode_lengths, dtype=np.int64)))
    images: list[np.ndarray] = []
    robot: list[np.ndarray] = []
    env: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    pads: list[np.ndarray] = []
    for t in index_array:
        episode = int(np.searchsorted(offsets, t, side="right") - 1)
        start = int(offsets[episode])
        end = int(offsets[episode + 1])
        available = end - t
        pad = np.zeros(chunk_size, dtype=bool)
        if available >= chunk_size:
            action_t = data.actions[t : t + chunk_size]
        else:
            pad[available:] = True
            action_t = np.zeros((chunk_size, ACTION_DIM), dtype=np.float32)
            action_t[:available] = data.actions[t:end]
        images.append(data.images[t])
        robot.append(data.robot_state[t])
        env.append(data.env_state[t])
        chunks.append(action_t)
        pads.append(pad)
    return VisualACTChunkDataset(
        env_id=data.env_id,
        images=np.stack(images, axis=0).astype(np.float32, copy=False),
        robot_state=np.stack(robot, axis=0).astype(np.float32, copy=False),
        env_state=np.stack(env, axis=0).astype(np.float32, copy=False),
        action_chunk=np.stack(chunks, axis=0).astype(np.float32, copy=False),
        action_is_pad=np.stack(pads, axis=0),
        chunk_size=chunk_size,
    )


def _normalization_stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(array.mean(axis=0), dtype=np.float32)
    std = np.asarray(array.std(axis=0), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def compute_visual_act_stats(
    data: VisualTaskData,
    train_indices: Sequence[int],
) -> dict[str, dict[str, np.ndarray]]:
    """只从训练 transition 计算 VISUAL/STATE/ENV/ACTION 统计。"""
    index_array = np.asarray(list(train_indices), dtype=np.int64)
    if index_array.size == 0:
        raise VisualACTError("train_indices 不能为空")
    robot_states = data.robot_state[index_array]
    env_states = data.env_state[index_array]
    images = data.images[index_array]
    image_mean = images.mean(axis=(0, 2, 3))
    image_std = images.std(axis=(0, 2, 3))
    image_std = np.where(image_std < 1e-6, 1.0, image_std).astype(np.float32)
    state_mean, state_std = _normalization_stats(robot_states)
    action_mean, action_std = _normalization_stats(data.actions[index_array])
    stats: dict[str, dict[str, object]] = {
        OBS_IMAGE_KEY: {
            "mean": image_mean.reshape(-1, 1, 1),
            "std": image_std.reshape(-1, 1, 1),
        },
        OBS_STATE_KEY: {"mean": state_mean, "std": state_std},
        ACTION_KEY: {"mean": action_mean, "std": action_std},
    }
    # 0 维 = 该变体不用 ENV 通道（B0 新口径）：统计项**一并省掉**，
    # 否则会给归一化器塞一个 0 维张量
    if env_states.shape[-1] > 0:
        env_mean, env_std = _normalization_stats(env_states)
        stats[OBS_ENV_STATE_KEY] = {"mean": env_mean, "std": env_std}
    return stats


def _stats_to_tensors(
    stats: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    torch = _torch_module()
    result: dict[str, dict[str, object]] = {}
    for key, item in stats.items():
        result[str(key)] = {
            "mean": torch.from_numpy(np.asarray(item["mean"], dtype=np.float32)),
            "std": torch.from_numpy(np.asarray(item["std"], dtype=np.float32)),
        }
    return result


def make_visual_act_config(
    config: VisualACTTrainingConfig,
) -> object:
    """创建 ACTConfig：VISUAL + STATE，ENV 视 `env_state_mode` 决定是否注册。"""
    try:
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
    except (ImportError, OSError) as exc:
        raise _lerobot_import_error(exc) from exc
    input_features = {
        OBS_IMAGE_KEY: PolicyFeature(
            FeatureType.VISUAL, (3, config.resolution, config.resolution)
        ),
        OBS_STATE_KEY: PolicyFeature(FeatureType.STATE, (ROBOT_STATE_DIM,)),
    }
    # ENV 维度为 0（B0 新口径：不带特权几何）时**整块省掉**该 feature ——
    # 注册一个 0 维特征会让 ACT 的输入投影退化成无意义的噪声源。
    if config.env_state_dim > 0:
        input_features[OBS_ENV_STATE_KEY] = PolicyFeature(
            FeatureType.ENV, (config.env_state_dim,)
        )
    output_features = {
        ACTION_KEY: PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))
    }
    return ACTConfig(
        n_obs_steps=1,
        chunk_size=config.chunk_size,
        n_action_steps=config.n_action_steps,
        input_features=input_features,
        output_features=output_features,
        device=_select_device(config.device),
        vision_backbone=config.vision_backbone,
        pretrained_backbone_weights=config.pretrained_backbone_weights,
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
        temporal_ensemble_coeff=None,
        optimizer_lr=1e-4,
    )


def make_visual_act_policy(
    config: VisualACTTrainingConfig,
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
    act_config = make_visual_act_config(config)
    pre, post = make_act_pre_post_processors(
        act_config,
        dataset_stats=_stats_to_tensors(dataset_stats),
    )
    policy = ACTPolicy(act_config)
    policy.to(device)
    return act_config, policy, pre, post


def _visual_batch_from_indices(
    dataset: VisualACTChunkDataset, indices: Sequence[int]
) -> dict[str, object]:
    torch = _torch_module()
    indices = np.asarray(list(indices), dtype=np.int64)
    batch: dict[str, object] = {
        OBS_IMAGE_KEY: torch.from_numpy(dataset.images[indices]),
        OBS_STATE_KEY: torch.from_numpy(dataset.robot_state[indices]),
        ACTION_KEY: torch.from_numpy(dataset.action_chunk[indices]),
        ACTION_IS_PAD_KEY: torch.from_numpy(dataset.action_is_pad[indices]),
    }
    if dataset.env_state.shape[-1] > 0:  # 0 维 = 该变体不用 ENV 通道（B0 新口径）
        batch[OBS_ENV_STATE_KEY] = torch.from_numpy(dataset.env_state[indices])
    return batch


def _evaluate_validation_loss(
    policy: object,
    pre: object,
    dataset: VisualACTChunkDataset,
    config: VisualACTTrainingConfig,
) -> float:
    torch = _torch_module()
    total = 0.0
    count = 0
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(dataset.images), config.batch_size):
            indices = np.arange(
                start,
                min(start + config.batch_size, len(dataset.images)),
                dtype=np.int64,
            )
            processed = pre(_visual_batch_from_indices(dataset, indices))
            loss, _ = policy.forward(processed)
            total += float(loss.item()) * len(indices)
            count += len(indices)
    policy.train()
    if count == 0:
        raise VisualACTError("validation 集为空")
    return total / count


def _save_checkpoint_dir(
    policy: object, pre: object, post: object, path: Path
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)


def _peak_rss_mb() -> float:
    """返回进程峰值常驻内存（MB）。

    macOS 的 ``ru_maxrss`` 以字节为单位，Linux 以 KB 为单位。
    """
    try:
        import platform
        import resource
    except ImportError:
        return 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return round(rss / (1024.0 * 1024.0), 3)
    return round(rss / 1024.0, 3)


def _peak_memory_mb(device: str) -> float:
    """返回峰值显存（cuda）或峰值常驻内存（mps/cpu），单位为 MB。"""
    if device == "cuda":
        torch = _torch_module()
        if not torch.cuda.is_available():
            return 0.0
        return round(
            float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2), 3
        )
    return _peak_rss_mb()


def _git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _environment_versions() -> dict[str, str]:
    import importlib.metadata

    versions: dict[str, str] = {}
    for package in ("lerobot", "torch", "torchvision", "mani_skill", "sapien", "pin", "h5py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _train_one_task(
    data: VisualTaskData,
    config: VisualACTTrainingConfig,
) -> VisualTaskTrainingResult:
    torch = _torch_module()
    started = time.perf_counter()
    train_indices, validation_indices = split_task_transitions(
        data,
        train_episodes=config.train_episodes_per_task,
        validation_episodes=config.validation_episodes_per_task,
    )
    train_dataset = build_visual_chunk_dataset(
        data, train_indices, config.chunk_size
    )
    validation_dataset = build_visual_chunk_dataset(
        data, validation_indices, config.chunk_size
    )
    dataset_stats = compute_visual_act_stats(data, train_indices)

    task_index = config.tasks.index(data.env_id)
    task_seed = config.random_seed + task_index
    random.seed(task_seed)
    np.random.seed(task_seed)
    torch.manual_seed(task_seed)

    act_config, policy, pre, post = make_visual_act_policy(
        config, dataset_stats
    )
    device = act_config.device
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
    count = len(train_dataset.images)
    task_dir = config.output_dir / data.env_id

    policy.train()
    while step < config.training_steps:
        permutation = torch.randperm(count, generator=generator)
        for start in range(0, count, config.batch_size):
            if step >= config.training_steps:
                break
            indices = permutation[start : start + config.batch_size]
            processed = pre(_visual_batch_from_indices(train_dataset, indices))
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
                metrics.append({"step": step, "validation_loss": validation_loss})
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
                    policy, pre, post, task_dir / f"step-{step:05d}"
                )

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
        raise VisualACTError(f"{data.env_id} 未得到有限 validation loss")
    if not math.isfinite(best_train_loss):
        raise VisualACTError(f"{data.env_id} 未得到有限 train loss")

    policy.load_state_dict(best_state)
    _save_checkpoint_dir(policy, pre, post, task_dir / "best-validation")

    metadata = {
        "schema_version": VISUAL_SCHEMA_VERSION,
        "env_id": data.env_id,
        "created_at": _utc_now(),
        "dataset_dir": str(config.dataset_dir),
        "repo_id": config.repo_id,
        "git_commit": _git_commit(),
        "device": device,
        "random_seed": task_seed,
        "task_index": task_index,
        "resolution": config.resolution,
        "vision_backbone": config.vision_backbone,
        "pretrained_backbone_weights": config.pretrained_backbone_weights,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "training_steps": config.training_steps,
        "batch_size": config.batch_size,
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
        "train_episode_ids": list(data.episode_ids[: config.train_episodes_per_task]),
        "validation_episode_ids": list(
            data.episode_ids[
                config.train_episodes_per_task :
                config.train_episodes_per_task + config.validation_episodes_per_task
            ]
        ),
        "train_transitions": len(train_indices),
        "validation_transitions": len(validation_indices),
        "best_step": best_step,
        "best_train_loss": best_train_loss,
        "best_validation_loss": best_validation_loss,
        "image_stats": {
            key: {
                stat: np.asarray(value[stat]).reshape(-1).tolist()
                for stat in ("mean", "std")
            }
            for key, value in dataset_stats.items()
        },
        "extra_features": list(config.extra_features),
        "env_state_mode": config.env_state_mode,
        "versions": _environment_versions(),
    }
    _atomic_write_json(task_dir / "task_metadata.json", metadata)
    _write_metrics_jsonl(task_dir / "metrics.jsonl", metrics)

    return VisualTaskTrainingResult(
        env_id=data.env_id,
        train_episode_ids=tuple(
            data.episode_ids[: config.train_episodes_per_task]
        ),
        validation_episode_ids=tuple(
            data.episode_ids[
                config.train_episodes_per_task :
                config.train_episodes_per_task + config.validation_episodes_per_task
            ]
        ),
        train_transitions=len(train_indices),
        validation_transitions=len(validation_indices),
        best_step=best_step,
        best_train_loss=best_train_loss,
        best_validation_loss=best_validation_loss,
        device=device,
        peak_rss_mb=_peak_memory_mb(device),
        duration_seconds=round(time.perf_counter() - started, 6),
    )


def _write_metrics_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(row, ensure_ascii=False, default=_json_default)
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_visual_dagger(
    data: dict[str, VisualTaskData], config: VisualACTTrainingConfig
) -> None:
    """把视觉 DAgger 补采 episode 合并进对应任务的训练数据。"""
    from visual_dagger_collect import (
        build_visual_dagger_transitions,
        load_visual_dagger_episodes,
    )

    dagger = load_visual_dagger_episodes(config.dagger_data)
    env_id = str(dagger["env_id"])
    if env_id not in config.tasks:
        raise VisualACTError(
            f"dagger 数据针对 {env_id}，但训练任务只包含 {config.tasks}"
        )
    merged = build_visual_dagger_transitions(
        data[env_id],
        dagger,
        resolution=config.resolution,
    )
    print(
        f"视觉 DAgger 合并：{env_id} 追加 {dagger['n_episodes']} 个 episode / "
        f"{dagger['total_labeled_transitions']} 条 recovery transition"
    )
    data[env_id] = merged


def train_visual_act(
    config: VisualACTTrainingConfig,
) -> VisualACTTrainingSummary:
    """对配置指定的任务训练并保存周期与 best-validation checkpoint。"""
    started = time.perf_counter()
    _prepare_output_directory(config.output_dir)
    device = _select_device(config.device)
    data = load_visual_dataset(config)
    if config.dagger_data is not None:
        _merge_visual_dagger(data, config)
    results: list[VisualTaskTrainingResult] = []
    for env_id in config.tasks:
        results.append(_train_one_task(data[env_id], config))

    load_visual_act_checkpoints(
        config.output_dir,
        device=device,
        resolution=config.resolution,
        tasks=config.tasks,
        checkpoint_subdir="best-validation",
    )
    summary = VisualACTTrainingSummary(
        schema_version=VISUAL_SCHEMA_VERSION,
        created_at=_utc_now(),
        dataset_dir=str(config.dataset_dir),
        repo_id=config.repo_id,
        output_dir=str(config.output_dir),
        device=device,
        resolution=config.resolution,
        chunk_size=config.chunk_size,
        n_action_steps=config.n_action_steps,
        training_steps=config.training_steps,
        batch_size=config.batch_size,
        vision_backbone=config.vision_backbone,
        tasks_selected=config.tasks,
        dagger_data=str(config.dagger_data) if config.dagger_data else None,
        duration_seconds=round(time.perf_counter() - started, 6),
        peak_rss_mb=_peak_rss_mb(),
        tasks=tuple(results),
        complete=True,
    )
    _atomic_write_json(config.output_dir / "training_summary.json", asdict(summary))
    return summary


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
    """把 tensor/数组安全转成 CPU float32 一维数组（MPS/cuda 兼容）。"""
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    return np.asarray(candidate, dtype=np.float32).reshape(-1)


def _select_env_state(
    env_state: np.ndarray, config: VisualACTTrainingConfig
) -> np.ndarray:
    """按 `env_state_mode` 从数据集列里挑出**策略侧**要用的部分。

    ``extra_only`` 时只取**尾部 n 列**（额外列在数据集里就是追加在 13 维之后的）——
    这样同一个数据集既能喂 ``full`` 也能喂 ``extra_only``，不必重做数据。
    n=0 时返回空向量，后续装配会**整块省掉 ENV 通道**。
    """
    array = np.asarray(env_state, dtype=np.float32)
    if config.env_state_mode != "extra_only":
        return array
    count = len(config.extra_features)
    if count == 0:
        return np.zeros((0,), dtype=np.float32)
    if array.shape[-1] < count:
        raise VisualACTError(
            f"env_state 末维 {array.shape[-1]} 小于额外列数 {count}，无法切出额外列"
        )
    return array[..., -count:]


def _append_extra_features(
    env_state: np.ndarray,
    extras: np.ndarray,
    expected: int,
    env_id: str,
) -> np.ndarray:
    """把 B 变体的额外观测通道**追加**到 ENV 辅助向量尾部（13 → 13+n）。

    抽成纯函数是为了让"列序与维度"能被离线测试——它一旦错位，训练与推理仍会
    产出"有限"的数值，指标只是变差，事后几乎无法归因。
    """
    if extras.shape != (expected,):
        raise VisualACTError(
            f"{env_id} 额外观测通道维度不符：{extras.shape} 期望 ({expected},)"
        )
    if not np.all(np.isfinite(extras)):
        raise VisualACTError(f"{env_id} 额外观测通道含非有限数")
    return np.concatenate((np.asarray(env_state, dtype=np.float32), extras))


class VisualACTPolicy:
    """从磁盘 checkpoint 构造的单任务视觉 ACT rollout wrapper。"""

    phase = "visual_act"

    def __init__(
        self,
        *,
        env_id: str,
        policy: object,
        preprocessor: object,
        postprocessor: object,
        resolution: int,
        action_smoothing: float = 1.0,
        extra_features: Sequence[str] = (),
        extra_provider: Callable[[Mapping[str, object]], Sequence[float]] | None = None,
        env_state_mode: str = "full",
    ) -> None:
        self.env_id = env_id
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self._resolution = resolution
        self._smoothing = _validate_smoothing(action_smoothing)
        #: 额外观测通道（B1–B3）。**必须由调用方提供与训练同序的特征**：
        #: 缺特征时宁可报错，也不能补 0 —— 补 0 会让"特征没测到"变成"特征等于 0"，
        #: 消融结论会因此失真且无法事后区分。
        self._extra_features = tuple(str(name) for name in extra_features)
        self._extra_provider = extra_provider
        self._env_state_mode = str(env_state_mode)
        if self._env_state_mode not in ("full", "extra_only"):
            raise VisualACTError(
                f"env_state_mode 必须是 'full' 或 'extra_only'，实际 {self._env_state_mode!r}"
            )
        if self._extra_features and extra_provider is None:
            raise VisualACTError(
                f"{env_id} checkpoint 带额外观测通道 {self._extra_features}，"
                "但未提供 extra_provider：推理时无法构造与训练同序的特征"
            )
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

    def act(
        self,
        observation: Mapping[str, object],
        image: np.ndarray,
    ) -> np.ndarray:
        """live observation + 渲染图 → 5/13 state + 图像 → 四维动作。"""
        torch = _torch_module()
        state = observation_to_state(observation)
        robot, env_state = split_lerobot_act_state(state)
        if self._extra_features:
            extras = np.asarray(
                self._extra_provider(observation), dtype=np.float32
            ).reshape(-1)
            # `extra_only` 时**只喂派生列**：必须与训练侧 `_select_env_state` 同口径，
            # 否则策略收到 13+n 维而权重只认 n 维 —— 形状不报错、结果全错，最难查
            base = np.zeros((0,), dtype=np.float32) if self._env_state_mode == "extra_only" else env_state
            env_state = _append_extra_features(
                base, extras, len(self._extra_features), self.env_id
            )
        elif self._env_state_mode == "extra_only":
            env_state = np.zeros((0,), dtype=np.float32)
        image_resized = _resize_image(image, self._resolution)
        self.action_queries += 1
        batch: dict[str, object] = {
            OBS_IMAGE_KEY: torch.from_numpy(image_resized).unsqueeze(0),
            OBS_STATE_KEY: torch.from_numpy(robot).unsqueeze(0),
        }
        if env_state.shape[-1] > 0:  # 0 维 = 训练时就没用 ENV 通道
            batch[OBS_ENV_STATE_KEY] = torch.from_numpy(env_state).unsqueeze(0)
        started = time.perf_counter()
        processed = self._pre(batch)
        action = self._policy.select_action(processed)
        action = self._post(action)
        self.inference_ms += (time.perf_counter() - started) * 1000.0
        array = _to_float32_numpy(action)
        if array.shape != (ACTION_DIM,):
            raise VisualACTError(
                f"{self.env_id} 视觉 ACT 输出 shape={array.shape}"
            )
        if not np.all(np.isfinite(array)):
            self.nonfinite_actions += 1
            raise VisualACTError(f"{self.env_id} 视觉 ACT 输出包含非有限数")
        self.last_raw_action = array.copy()
        clipped = np.clip(array, -1.0, 1.0)
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


def _require_checkpoint_directory(path: Path) -> None:
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        if not (path / name).is_file():
            raise VisualACTError(f"checkpoint 缺少 {name}：{path}")


def _resolve_checkpoint_subdir(
    output_dir: Path, env_id: str, checkpoint_subdir: str | None
) -> Path:
    task_dir = output_dir / env_id
    if checkpoint_subdir is not None:
        path = task_dir / checkpoint_subdir
        if not path.is_dir():
            raise VisualACTError(
                f"{env_id} checkpoint 子目录不存在：{path}"
            )
        return path
    for candidate in ("best-rollout-dev", "best", "best-validation"):
        path = task_dir / candidate
        if path.is_dir():
            return path
    raise VisualACTError(f"{env_id} 没有可用 checkpoint 子目录")


def _read_optional_json_object(path: Path) -> dict[str, Any]:
    """读 JSON 对象；文件不存在返回空 dict（用于"可选元数据"）。"""
    return _read_json_object(path) if path.is_file() else {}


def env_feature_dim(features: Mapping[str, Any]) -> int | None:
    """从 `input_features` 映射里取 ENV 通道维度（没有该通道返回 `None`）。

    同一份 `input_features` 有两种形态，**必须都认**：磁盘 `config.json` 里是
    `{"shape": [n]}` 这样的 dict，而内存里 `ACTConfig.input_features` 是 `PolicyFeature`
    对象（`.shape` 属性）。只认 dict 会让下面"重建维度对账"在真正出问题时静默失效
    （`None` 被当成 0），把可定位的报错退化成晦涩的形状不符。
    """
    if not isinstance(features, Mapping):
        return None
    entry = features.get(OBS_ENV_STATE_KEY)
    if entry is None:
        return None
    if isinstance(entry, Mapping):
        shape = entry.get("shape")
    else:
        shape = getattr(entry, "shape", None)
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def saved_env_feature_dim(saved: Mapping[str, Any]) -> int | None:
    """从 checkpoint 的 `config.json` 取 ENV 通道维度（没有该通道返回 `None`）。

    这是**权重侧的事实**：`encoder_1d_feature_pos_embed` 的行数就由"注册了几个 1D 输入"决定，
    所以重建 config 后必须拿它对账，否则口径读错只会表现为晦涩的形状不匹配。
    """
    return env_feature_dim(saved.get("input_features"))


def resolve_checkpoint_env_config(
    saved: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[tuple[str, ...], str]:
    """决定重建 config 用的 ``(extra_features, env_state_mode)``。

    **口径的权威来源是训练时落盘的 `task_metadata.json`**，它和 checkpoint 同属一个任务目录。
    checkpoint 自己的 `config.json` 是 `ACTPolicy` 的配置，**不含**这两个字段 ——
    早期实现只看 `config.json`，于是 `env_state_mode` 退化成 `"full"`，
    给 B0（`extra_only` + 空列）多注册了一个 13 维 ENV 输入，加载时报
    `encoder_1d_feature_pos_embed` 行数不符（`[3,512]` vs `[2,512]`）。

    两者都缺时按 `input_features` 反推：能唯一确定"没有 ENV"（→ `extra_only` + 空列）
    与"ENV 恰好 13 维"（→ `full`）；其余情况说明它带额外列却丢了元数据，
    **必须报错**——额外列的**名字**无从恢复，猜一个就等于把列序错配埋进推理。
    """
    extra_features = tuple(
        str(item)
        for item in metadata.get("extra_features", saved.get("extra_features", ()) or ())
    )
    mode = metadata.get("env_state_mode", saved.get("env_state_mode"))
    if mode is not None:
        return extra_features, str(mode)

    dim = saved_env_feature_dim(saved)
    if dim is None or dim == 0:
        return extra_features, "extra_only"
    if dim == ENV_STATE_DIM:
        return extra_features, "full"
    raise VisualACTError(
        f"checkpoint 带 {dim} 维 ENV 通道，但找不到口径元数据（`task_metadata.json` 缺失）："
        f"无法还原额外列的**名字与顺序**。请把训练时的 `task_metadata.json` 放回任务目录，"
        f"不要靠猜列序来加载"
    )


def load_visual_act_checkpoints(
    output_dir: Path,
    *,
    device: str,
    resolution: int,
    tasks: Sequence[str] | None = None,
    checkpoint_subdir: str | None = None,
    action_smoothing: float = 1.0,
    extra_provider: Callable[[Mapping[str, object]], Sequence[float]] | None = None,
) -> dict[str, VisualACTPolicy]:
    """只从本地目录加载视觉 ACT policy 与 processors。"""
    device = _select_device(device)
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        raise VisualACTError(f"输出目录不存在：{output_dir}")
    _require_positive_int(resolution, "resolution")
    checked_tasks = tuple(tasks) if tasks is not None else SUPPORTED_TASKS
    for env_id in checked_tasks:
        if env_id not in SUPPORTED_TASKS:
            raise VisualACTError(f"不支持的任务 {env_id!r}")
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor import (
            DataProcessorPipeline,
            policy_action_to_transition,
            transition_to_policy_action,
        )
    except (ImportError, OSError) as exc:
        raise _lerobot_import_error(exc) from exc

    policies: dict[str, VisualACTPolicy] = {}
    for env_id in checked_tasks:
        best_dir = _resolve_checkpoint_subdir(output_dir, env_id, checkpoint_subdir)
        _require_checkpoint_directory(best_dir)
        saved = _read_json_object(best_dir / "config.json")
        chunk_size = int(saved.get("chunk_size", -1))
        saved_n_action_steps = int(saved.get("n_action_steps", -1))
        if not 1 <= saved_n_action_steps <= chunk_size:
            raise VisualACTError(f"{env_id} 保存的 chunk 配置错误")
        vision_backbone = str(saved.get("vision_backbone", "resnet18"))
        # 回读 B 变体的额外观测通道与 ENV 口径。**口径不在 checkpoint 自己的 config.json 里**
        # （那是 ACTPolicy 的配置），而在同级的 `task_metadata.json`；读错会让重建的
        # input_features 与权重不符（见 resolve_checkpoint_env_config 的说明）。
        metadata = _read_optional_json_object(best_dir.parent / "task_metadata.json")
        if not metadata:
            metadata = _read_optional_json_object(best_dir / "task_metadata.json")
        extra_features, env_state_mode = resolve_checkpoint_env_config(saved, metadata)
        config = make_visual_act_config(
            VisualACTTrainingConfig(
                dataset_dir=output_dir,
                repo_id="",
                output_dir=output_dir,
                tasks=(env_id,),
                resolution=resolution,
                chunk_size=chunk_size,
                n_action_steps=saved_n_action_steps,
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
                vision_backbone=vision_backbone,
                extra_features=extra_features,
                # **必须把回读到的口径传进去**：漏传会退回默认 `"full"`，
                # 于是 B0（`extra_only` + 空列）被重建出一个 13 维 ENV 输入，
                # 加载时在 `encoder_1d_feature_pos_embed` 上报 [3,512] vs [2,512]。
                env_state_mode=env_state_mode,
                pretrained_backbone_weights=None,
                device=device,
            )
        )
        # 对账：重建出来的 ENV 通道维度必须与 checkpoint 记录的**完全一致**。
        # 口径读错时这里会给出可定位的报错；否则它只会在 `encoder_1d_feature_pos_embed`
        # 上以"形状不符"的形式炸出来（[3,512] vs [2,512]），看的人得反推是哪一步错了。
        expected_env_dim = saved_env_feature_dim(saved) or 0
        rebuilt_env_dim = env_feature_dim(getattr(config, "input_features", {}) or {}) or 0
        if rebuilt_env_dim != expected_env_dim:
            raise VisualACTError(
                f"{env_id} 重建的 ENV 通道 {rebuilt_env_dim} 维与 checkpoint 记录的 "
                f"{expected_env_dim} 维不一致：`env_state_mode`/`extra_features` 读错了"
            )
        policy = ACTPolicy.from_pretrained(best_dir, config=config)
        policy.to(device)
        # 跨设备兼容：4060 上以 cuda 训练的 checkpoint，其 processor 的
        # device_processor 步骤记录了 "cuda"。在 Mac 上加载时用 overrides
        # 把设备重写为当前设备（内存级合并，不修改磁盘上的 checkpoint）。
        processor_overrides = {"device_processor": {"device": device}}
        pre = DataProcessorPipeline.from_pretrained(
            best_dir,
            config_filename="policy_preprocessor.json",
            overrides=processor_overrides,
        )
        post = DataProcessorPipeline.from_pretrained(
            best_dir,
            config_filename="policy_postprocessor.json",
            overrides=processor_overrides,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        policies[env_id] = VisualACTPolicy(
            env_id=env_id,
            policy=policy,
            preprocessor=pre,
            postprocessor=post,
            resolution=resolution,
            action_smoothing=action_smoothing,
            extra_features=extra_features,
            extra_provider=extra_provider,
            env_state_mode=env_state_mode,
        )
    return policies


def _scalar_bool(value: object, name: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise VisualACTError(f"{name} 必须是标量")
    return bool(array.reshape(-1)[0])


def run_visual_episode_in_env(
    config: ManiSkillTaskConfig,
    env: object,
    policy: VisualACTPolicy,
) -> dict[str, object]:
    """live RGB 渲染 rollout：observation + render → act → step。"""
    env_any: Any = env
    reset_result = env_any.reset(seed=config.seed)
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise VisualACTError("env.reset() 未返回二元组")
    observation, _ = reset_result
    if not isinstance(observation, Mapping):
        raise VisualACTError("state_dict observation 必须是 mapping")
    policy.reset()
    steps = 0
    success = False
    terminated = False
    truncated = False
    total_reward = 0.0
    for step_index in range(1, config.max_steps + 1):
        image = _render_live_image(env)
        action = policy.act(observation, image)
        step_result = env_any.step(action)
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise VisualACTError("env.step() 未返回五元组")
        observation, reward, raw_terminated, raw_truncated, info = step_result
        if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
            raise VisualACTError("step observation/info 必须是 mapping")
        steps = step_index
        total_reward += float(_to_float32_numpy(reward).reshape(-1)[0])
        terminated = _scalar_bool(raw_terminated, "terminated")
        truncated = _scalar_bool(raw_truncated, "truncated")
        success = _scalar_bool(info.get("success", False), "info.success")
        if success or terminated or truncated:
            break
    return {
        "steps": steps,
        "total_reward": total_reward,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
    }


def _render_live_image(env: object) -> np.ndarray:
    try:
        image = np.asarray(env.render())
    except Exception as exc:
        raise VisualACTError(f"live 渲染失败：{exc}") from exc
    return _image_hwc(image)


def evaluate_visual_act(
    output_dir: Path,
    *,
    seeds: Sequence[int],
    device: str,
    resolution: int,
    tasks: Sequence[str] | None = None,
    checkpoint_subdir: str | None = None,
    extra_provider: Callable[[Mapping[str, object]], Sequence[float]] | None = None,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_render_env,
) -> PolicyEvaluationSummary:
    """从磁盘重载视觉 ACT checkpoint 后执行固定 seed 评测并落盘。

    `extra_provider`：B1–B3 带额外观测通道，**必须**提供与训练同序的特征计算函数，
    否则 `VisualACTPolicy` 会直接报错（不允许补 0，见其构造注释）。B0 传 `None`。
    """
    from lerobot_act import (
        ACTEvaluationEpisode,
        ACTEvaluationTask,
    )

    checked_seeds = _checked_seeds(seeds)
    checked_tasks = tuple(tasks) if tasks is not None else SUPPORTED_TASKS
    policies = load_visual_act_checkpoints(
        output_dir,
        device=device,
        resolution=resolution,
        tasks=checked_tasks,
        checkpoint_subdir=checkpoint_subdir,
        extra_provider=extra_provider,
    )
    episodes: list[ACTEvaluationEpisode] = []
    for env_id in checked_tasks:
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
    summary = PolicyEvaluationSummary(
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
            float(np.mean([item.steps for item in episodes]))
            if episodes
            else 0.0
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
    _atomic_write_json(output_dir / "evaluation.json", asdict(summary))
    return summary


def write_resource_report(
    output_dir: Path,
    *,
    device: str,
    resolution: int,
    batch_size: int,
    training_seconds: float,
    evaluation_seconds: float,
    evaluation: PolicyEvaluationSummary,
    peak_rss_before: float,
) -> ResourceReport:
    """写出训练/评测的显存与时长记录。

    设备为 ``cuda`` 时 ``peak_rss_*`` 字段表示峰值显存（MB），
    其余设备表示峰值常驻内存。
    """
    peak_rss_total = _peak_memory_mb(device)
    report = ResourceReport(
        schema_version=VISUAL_SCHEMA_VERSION,
        created_at=_utc_now(),
        device=device,
        resolution=resolution,
        batch_size=batch_size,
        peak_rss_mb_train=round(peak_rss_total - peak_rss_before, 3),
        peak_rss_mb_total=peak_rss_total,
        training_seconds=round(training_seconds, 6),
        evaluation_seconds=round(evaluation_seconds, 6),
        mean_inference_ms=round(evaluation.mean_inference_ms, 6),
        total_episodes=evaluation.total_episodes,
    )
    _atomic_write_json(output_dir / "resource_report.json", asdict(report))
    return report


__all__ = [
    "VISUAL_SCHEMA_VERSION",
    "OBS_IMAGES_KEY",
    "VisualACTError",
    "VisualTaskData",
    "VisualACTChunkDataset",
    "VisualACTTrainingConfig",
    "VisualTaskTrainingResult",
    "VisualACTTrainingSummary",
    "ResourceReport",
    "build_visual_chunk_dataset",
    "compute_visual_act_stats",
    "evaluate_visual_act",
    "load_visual_act_checkpoints",
    "load_visual_dataset",
    "make_visual_act_config",
    "make_visual_act_policy",
    "run_visual_episode_in_env",
    "split_task_transitions",
    "train_visual_act",
    "write_resource_report",
]
