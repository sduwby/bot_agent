"""第 15 周：LeRobot 数据管线——转换、replay、检查与可视化。

把第 11 周 ManiSkill 成功轨迹（HDF5 + env_states）转换为 LeRobotDataset v3，
包含 episode、timestamp、camera、state、action；支持 v3 只读重开、episode
读回、字段完整性检查、异常帧检测与逐 episode 浏览/可视化。
"""

from __future__ import annotations

import json
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
from expert_trajectory import SCHEMA_VERSION as EXPERT_SCHEMA_VERSION
from lerobot_act import (
    ENV_STATE_DIM,
    LeRobotACTError,
    ROBOT_STATE_DIM,
    split_lerobot_act_state,
)
from maniskill_intro import (
    SUPPORTED_TASKS,
    ManiSkillTaskConfig,
    ManiSkillUnavailableError,
    make_maniskill_env,
)

PIPELINE_SCHEMA_VERSION = "maniskill_lerobot_pipeline_v1"

# LeRobotDataset v3 feature 键名。
OBS_STATE_KEY = "observation.state"
OBS_ENV_STATE_KEY = "observation.environment_state"
OBS_IMAGE_KEY = "observation.images.top"
ACTION_KEY = "action"
REWARD_KEY = "reward"
SUCCESS_KEY = "success"
NEXT_DONE_KEY = "next.done"

CAMERA_RESOLUTION = (512, 512)
DEFAULT_FPS = 20
DEFAULT_REPO_ID = "bot-agent/maniskill-week11"
ACTION_SPACE_BOUND = 1.0

# 第 11 周 HDF5 的 goal actor 名称。
GOAL_ACTORS = {"PickCube-v1": "goal_site", "PushCube-v1": "goal_region"}


class LerobotPipelineError(RuntimeError):
    """LeRobot 数据管线转换、检查或浏览契约错误。"""


@dataclass(frozen=True)
class LerobotConversionConfig:
    """LeRobotDataset v3 转换配置。"""

    source_summary: Path
    output_dir: Path
    repo_id: str = DEFAULT_REPO_ID
    fps: int = DEFAULT_FPS
    include_camera: bool = False
    tasks: tuple[str, ...] = SUPPORTED_TASKS
    robot_type: str = "panda"
    episodes_per_task: int | None = None
    max_steps: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_summary",
            Path(self.source_summary).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "output_dir",
            Path(self.output_dir).expanduser().resolve(),
        )
        if self.fps <= 0:
            raise ValueError("fps 必须是正整数")
        if self.episodes_per_task is not None and (
            not isinstance(self.episodes_per_task, int)
            or isinstance(self.episodes_per_task, bool)
            or self.episodes_per_task <= 0
        ):
            raise ValueError("episodes_per_task 必须是正整数")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须是正整数")
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


@dataclass(frozen=True)
class TaskConversionResult:
    """一个任务的转换统计。"""

    env_id: str
    episodes: int
    frames: int
    errors: int


@dataclass(frozen=True)
class ConversionSummary:
    """一次转换运行的机器可读摘要。"""

    schema_version: str
    created_at: str
    source_summary: str
    output_dir: str
    repo_id: str
    fps: int
    include_camera: bool
    tasks: tuple[TaskConversionResult, ...]
    total_episodes: int
    total_frames: int
    total_errors: int
    duration_seconds: float
    complete: bool


@dataclass(frozen=True)
class FrameInspection:
    """一帧的完整性与有效性检查结果。"""

    episode_index: int
    frame_index: int
    timestamp: float
    task: str
    missing: tuple[str, ...]
    nonfinite: tuple[str, ...]
    action_out_of_bounds: bool
    camera_missing: bool
    camera_black: bool


@dataclass(frozen=True)
class EpisodeInspection:
    """一个 episode 的聚合检查结果。"""

    episode_index: int
    frames: int
    missing_count: int
    abnormal_frames: int
    task: str


@dataclass(frozen=True)
class InspectionSummary:
    """全数据集的检查结果。"""

    schema_version: str
    created_at: str
    repo_id: str
    total_episodes: int
    total_frames: int
    missing_count: int
    abnormal_frames: int
    episodes: tuple[EpisodeInspection, ...]
    complete: bool


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


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LerobotPipelineError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise LerobotPipelineError(f"JSON 顶层必须是 object：{path}")
    return value


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def load_maniskill_summary(summary_path: Path) -> dict[str, Any]:
    """加载并校验第 11 周 summary。"""
    summary = _read_json_object(Path(summary_path))
    if summary.get("schema_version") != EXPERT_SCHEMA_VERSION:
        raise LerobotPipelineError("summary schema_version 不受支持")
    if not bool(summary.get("complete", False)):
        raise LerobotPipelineError("summary 未标记 complete")
    raw_tasks = summary.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise LerobotPipelineError("summary.tasks 必须是非空数组")
    for task in raw_tasks:
        if not isinstance(task, dict) or task.get("env_id") not in SUPPORTED_TASKS:
            raise LerobotPipelineError("summary.tasks 元素格式错误")
    return summary


def build_lerobot_features(
    include_camera: bool,
) -> dict[str, dict[str, object]]:
    """构造 LeRobotDataset v3 feature schema。"""
    features: dict[str, dict[str, object]] = {
        OBS_STATE_KEY: {
            "dtype": "float32",
            "shape": (ROBOT_STATE_DIM,),
            "names": None,
        },
        OBS_ENV_STATE_KEY: {
            "dtype": "float32",
            "shape": (ENV_STATE_DIM,),
            "names": None,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": None,
        },
        REWARD_KEY: {"dtype": "float32", "shape": (1,), "names": None},
        SUCCESS_KEY: {"dtype": "bool", "shape": (1,), "names": None},
        NEXT_DONE_KEY: {"dtype": "bool", "shape": (1,), "names": None},
    }
    if include_camera:
        features[OBS_IMAGE_KEY] = {
            "dtype": "video",
            "shape": (*CAMERA_RESOLUTION, 3),
            "names": None,
        }
    return features


def maniskill_state_to_lerobot_features(
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """18D → (robot_state[5], env_state[13])，与第 13 周语义一致。"""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise LerobotPipelineError(
            f"state shape={state.shape}，期望 ({STATE_DIM},)"
        )
    try:
        robot, env = split_lerobot_act_state(state)
    except LeRobotACTError as exc:
        raise LerobotPipelineError(str(exc)) from exc
    return robot, env


def make_render_env(config: ManiSkillTaskConfig) -> object:
    """创建带 RGB 渲染能力的 ManiSkill 环境。"""
    _configure_macos_vulkan()
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
    except (ImportError, OSError) as exc:
        raise ManiSkillUnavailableError(
            "ManiSkill 运行环境不可用；请使用 .venv-lerobot"
        ) from exc
    base_env = gym.make(
        config.env_id,
        num_envs=1,
        obs_mode=config.obs_mode,
        control_mode=config.control_mode,
        reward_mode=config.reward_mode,
        sim_backend=config.sim_backend,
        render_mode="rgb_array",
    )
    try:
        return CPUGymWrapper(base_env)
    except Exception:
        base_env.close()
        raise


def _configure_macos_vulkan() -> None:
    """在 Apple Silicon Homebrew 环境中定位 MoltenVK ICD。"""
    import os
    import sys

    if sys.platform != "darwin" or os.environ.get("VK_ICD_FILENAMES"):
        return
    candidates = (
        Path("/opt/homebrew/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"),
        Path("/usr/local/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"),
    )
    for candidate in candidates:
        if candidate.is_file():
            os.environ["VK_ICD_FILENAMES"] = str(candidate)
            return


def _prepare_output_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"输出路径不是目录：{path}")
        if any(path.iterdir()):
            raise ValueError(
                f"输出目录必须为空，拒绝覆盖已有数据：{path}"
            )


def _read_h5_state(node: object, index: int) -> object:
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        return np.asarray(node[index])
    items = getattr(node, "items", None)
    if not callable(items):
        raise LerobotPipelineError("env_states 节点格式错误")
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
        raise LerobotPipelineError(
            f"重建 observation {path} 缺少单环境 batch 维：{array.shape}"
        )
    return array[0]


def _episode_metadata(
    summary: Mapping[str, object], env_id: str, base_dir: Path
) -> list[dict[str, Any]]:
    tasks = summary["tasks"]
    task = next(
        (item for item in tasks if isinstance(item, Mapping) and item.get("env_id") == env_id),
        None,
    )
    if not isinstance(task, Mapping):
        raise LerobotPipelineError(f"{env_id} 不在 summary.tasks 中")
    metadata_path = base_dir / str(task["metadata_path"])
    metadata = _read_json_object(metadata_path)
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise LerobotPipelineError(f"{env_id} metadata.episodes 不能为空")
    return episodes


def _h5_path(summary: Mapping[str, object], env_id: str, base_dir: Path) -> Path:
    tasks = summary["tasks"]
    task = next(
        (item for item in tasks if isinstance(item, Mapping) and item.get("env_id") == env_id),
        None,
    )
    if not isinstance(task, Mapping):
        raise LerobotPipelineError(f"{env_id} 不在 summary.tasks 中")
    return base_dir / str(task["trajectory_path"])


def _task_text(env_id: str) -> str:
    return {
        "PickCube-v1": "PickCube-v1",
        "PushCube-v1": "PushCube-v1",
    }[env_id]


def convert_maniskill_to_lerobot(
    config: LerobotConversionConfig,
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_render_env,
) -> ConversionSummary:
    """把第 11 周 HDF5 轨迹转换为 LeRobotDataset v3。"""
    try:
        import h5py
        from lerobot.datasets import LeRobotDataset
    except (ImportError, OSError) as exc:
        raise LerobotPipelineError(
            "LeRobotDataset v3 需要 lerobot[dataset] 与 h5py"
        ) from exc

    started = time.perf_counter()
    summary = load_maniskill_summary(config.source_summary)
    base_dir = config.source_summary.parent
    _prepare_output_directory(config.output_dir)
    features = build_lerobot_features(config.include_camera)

    dataset = LeRobotDataset.create(
        config.repo_id,
        fps=config.fps,
        features=features,
        root=config.output_dir,
        use_videos=config.include_camera,
        robot_type=config.robot_type,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    task_results: list[TaskConversionResult] = []
    total_episodes = 0
    total_frames = 0
    total_errors = 0
    for env_id in config.tasks:
        episodes = _episode_metadata(summary, env_id, base_dir)
        if config.episodes_per_task is not None:
            episodes = episodes[: config.episodes_per_task]
        h5_path = _h5_path(summary, env_id, base_dir)
        env: Any | None = None
        episode_errors = 0
        frames = 0
        try:
            try:
                env = env_factory(ManiSkillTaskConfig(env_id))
            except (
                LerobotPipelineError,
                BehaviorCloningError,
                ManiSkillUnavailableError,
                ValueError,
                OSError,
                KeyError,
                TypeError,
            ) as exc:
                episode_errors = len(episodes)
                print(f"[{env_id}] 环境创建失败，跳过 {episode_errors} 条：{exc}")
            else:
                base_env = getattr(env, "base_env", None)
                if base_env is None:
                    base_env = getattr(env, "unwrapped", None)
                setter = getattr(base_env, "set_state_dict", None)
                getter = getattr(base_env, "get_obs", None)
                if not callable(setter) or not callable(getter):
                    raise LerobotPipelineError(
                        f"{env_id} 转换环境缺少 set_state_dict/get_obs"
                    )
                reset_result = env.reset(seed=0)
                if not isinstance(reset_result, tuple) or len(reset_result) != 2:
                    raise LerobotPipelineError(
                        f"{env_id} 转换环境 reset 契约错误"
                    )
                with h5py.File(str(h5_path), "r") as trajectories:
                    for episode in episodes:
                        episode_id = int(episode["episode_id"])
                        group = trajectories[f"traj_{episode_id}"]
                        actions = np.asarray(group["actions"], dtype=np.float32)
                        rewards = np.asarray(group["rewards"], dtype=np.float32)
                        terminated = np.asarray(group["terminated"], dtype=bool)
                        truncated = np.asarray(group["truncated"], dtype=bool)
                        success = np.asarray(group["success"], dtype=bool)
                        steps = int(episode["elapsed_steps"])
                        if actions.shape != (steps, ACTION_DIM):
                            raise LerobotPipelineError(
                                f"{env_id}/traj_{episode_id} actions shape 错误"
                            )
                        task_text = _task_text(env_id)
                        try:
                            for t in range(steps):
                                state_dict = _read_h5_state(group["env_states"], t)
                                setter(state_dict)
                                observation = _unbatch_observation(getter())
                                if not isinstance(observation, Mapping):
                                    raise LerobotPipelineError(
                                        "重建 observation 必须是 mapping"
                                    )
                                state = observation_to_state(observation)
                                robot, env_state = (
                                    maniskill_state_to_lerobot_features(state)
                                )
                                frame: dict[str, object] = {
                                    OBS_STATE_KEY: robot,
                                    OBS_ENV_STATE_KEY: env_state,
                                    ACTION_KEY: actions[t],
                                    REWARD_KEY: np.asarray(
                                        [rewards[t]], dtype=np.float32
                                    ),
                                    SUCCESS_KEY: np.asarray([bool(success[t])]),
                                    NEXT_DONE_KEY: np.asarray(
                                        [bool(terminated[t] or truncated[t])]
                                    ),
                                    "task": task_text,
                                }
                                if config.include_camera:
                                    image = _render_frame(env)
                                    if image is None:
                                        raise LerobotPipelineError(
                                            f"{env_id}/traj_{episode_id} "
                                            "camera 渲染返回空"
                                        )
                                    frame[OBS_IMAGE_KEY] = image
                                dataset.add_frame(frame)
                            dataset.save_episode()
                            frames += steps
                        except (
                            LerobotPipelineError,
                            BehaviorCloningError,
                            ValueError,
                            KeyError,
                            TypeError,
                        ) as exc:
                            episode_errors += 1
                            print(
                                f"[{env_id}] traj_{episode_id} 转换失败：{exc}"
                            )
        finally:
            if env is not None:
                env.close()
        task_results.append(
            TaskConversionResult(
                env_id=env_id,
                episodes=len(episodes),
                frames=frames,
                errors=episode_errors,
            )
        )
        total_episodes += len(episodes)
        total_frames += frames
        total_errors += episode_errors

    dataset.finalize()
    complete = total_errors == 0
    summary_payload = ConversionSummary(
        schema_version=PIPELINE_SCHEMA_VERSION,
        created_at=_utc_now(),
        source_summary=str(config.source_summary),
        output_dir=str(config.output_dir),
        repo_id=config.repo_id,
        fps=config.fps,
        include_camera=config.include_camera,
        tasks=tuple(task_results),
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_errors=total_errors,
        duration_seconds=round(time.perf_counter() - started, 6),
        complete=complete,
    )
    _atomic_write_json(
        config.output_dir / "conversion_summary.json", asdict(summary_payload)
    )
    return summary_payload


def _render_frame(env: object) -> np.ndarray | None:
    try:
        image = np.asarray(env.render())
    except Exception:
        return None
    if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
        return None
    return image.astype(np.uint8)


def open_lerobot_dataset(output_dir: Path, repo_id: str) -> object:
    """只读重开 v3 dataset。"""
    try:
        from lerobot.datasets import LeRobotDataset
    except (ImportError, OSError) as exc:
        raise LerobotPipelineError(
            "LeRobotDataset v3 需要 lerobot[dataset]"
        ) from exc
    output_dir = Path(output_dir).expanduser().resolve()
    return LeRobotDataset(repo_id, root=output_dir, return_uint8=True)


def _episode_frame_indices(dataset: object, episode_index: int) -> list[int]:
    """返回某个 episode 在数据集中的相对帧索引。"""
    indices: list[int] = []
    for index in range(len(dataset)):
        frame = dataset[index]
        if int(frame["episode_index"]) == episode_index:
            indices.append(index)
    if not indices:
        raise LerobotPipelineError(f"episode_index={episode_index} 不存在")
    return indices


def read_episode_frames(dataset: object, episode_index: int) -> list[dict[str, object]]:
    """读回单个 episode 的所有帧。"""
    indices = _episode_frame_indices(dataset, episode_index)
    frames = [dataset[index] for index in indices]
    frames.sort(key=lambda item: int(item["frame_index"]))
    return frames


def _as_numpy(value: object) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    return np.asarray(candidate)


def _image_hwc(value: object) -> np.ndarray:
    """将 video feature 归一化为 HWC uint8 数组（兼容 CHW/HWC）。"""
    array = _as_numpy(value)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[2] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] != 3 or array.size == 0:
        raise LerobotPipelineError("camera 帧不是有效 RGB 图像")
    return array.astype(np.uint8)


def _is_black_image(value: object) -> bool:
    try:
        array = _image_hwc(value)
    except Exception:
        return True
    return bool(array.min() == array.max())


def _inspect_frame(
    dataset: object, frame: Mapping[str, object], features: Mapping[str, object]
) -> FrameInspection:
    missing: list[str] = []
    nonfinite: list[str] = []
    action_out_of_bounds = False
    camera_missing = False
    camera_black = False
    for name in features:
        if name not in frame:
            missing.append(name)
            continue
        if name in (OBS_IMAGE_KEY,):
            if _as_numpy(frame[name]).size == 0:
                camera_missing = True
            elif _is_black_image(frame[name]):
                camera_black = True
            continue
        value = _as_numpy(frame[name])
        if name == ACTION_KEY:
            if value.shape != (ACTION_DIM,):
                nonfinite.append(name)
                continue
            if not np.all(np.isfinite(value)):
                nonfinite.append(name)
            if np.any(np.abs(value) > ACTION_SPACE_BOUND):
                action_out_of_bounds = True
        elif name == OBS_STATE_KEY and value.shape != (ROBOT_STATE_DIM,):
            nonfinite.append(name)
        elif name == OBS_ENV_STATE_KEY and value.shape != (ENV_STATE_DIM,):
            nonfinite.append(name)
        elif not np.all(np.isfinite(value)):
            nonfinite.append(name)
    return FrameInspection(
        episode_index=int(frame["episode_index"]),
        frame_index=int(frame["frame_index"]),
        timestamp=float(frame["timestamp"]),
        task=str(frame.get("task", "")),
        missing=tuple(missing),
        nonfinite=tuple(nonfinite),
        action_out_of_bounds=action_out_of_bounds,
        camera_missing=camera_missing,
        camera_black=camera_black,
    )


def inspect_lerobot_dataset(
    output_dir: Path,
    repo_id: str,
) -> InspectionSummary:
    """字段完整性（missing=0）+ 异常帧检测。"""
    dataset = open_lerobot_dataset(output_dir, repo_id)
    features = build_lerobot_features(
        dataset.meta.video_keys is not None and len(dataset.meta.video_keys) > 0
    )
    episodes_map: dict[int, list[FrameInspection]] = {}
    for index in range(len(dataset)):
        frame = dataset[index]
        ep_index = int(frame["episode_index"])
        inspection = _inspect_frame(dataset, frame, features)
        episodes_map.setdefault(ep_index, []).append(inspection)

    episode_summaries: list[EpisodeInspection] = []
    total_missing = 0
    total_abnormal = 0
    for ep_index in sorted(episodes_map):
        frames = episodes_map[ep_index]
        missing = sum(len(item.missing) for item in frames)
        abnormal = sum(
            1
            for item in frames
            if item.missing
            or item.nonfinite
            or item.action_out_of_bounds
            or item.camera_missing
            or item.camera_black
        )
        total_missing += missing
        total_abnormal += abnormal
        episode_summaries.append(
            EpisodeInspection(
                episode_index=ep_index,
                frames=len(frames),
                missing_count=missing,
                abnormal_frames=abnormal,
                task=frames[0].task if frames else "",
            )
        )
    summary = InspectionSummary(
        schema_version=PIPELINE_SCHEMA_VERSION,
        created_at=_utc_now(),
        repo_id=repo_id,
        total_episodes=len(episode_summaries),
        total_frames=len(dataset),
        missing_count=total_missing,
        abnormal_frames=total_abnormal,
        episodes=tuple(episode_summaries),
        complete=total_missing == 0 and total_abnormal == 0,
    )
    _atomic_write_json(Path(output_dir) / "inspection.json", asdict(summary))
    return summary


def _state_norm(value: object) -> float:
    try:
        array = _as_numpy(value).astype(float)
    except Exception:
        return float("nan")
    return float(np.linalg.norm(array))


def browse_lerobot_episode(
    output_dir: Path,
    repo_id: str,
    episode_index: int,
) -> dict[str, object]:
    """逐帧摘要（状态统计、动作、异常标记、camera 缩略图）。"""
    dataset = open_lerobot_dataset(output_dir, repo_id)
    frames = read_episode_frames(dataset, episode_index)
    rows: list[dict[str, object]] = []
    for frame in frames:
        rows.append(
            {
                "frame_index": int(frame["frame_index"]),
                "timestamp": float(frame["timestamp"]),
                "task": str(frame.get("task", "")),
                "state_norm": _state_norm(frame.get(OBS_STATE_KEY)),
                "env_state_norm": _state_norm(frame.get(OBS_ENV_STATE_KEY)),
                "action": _as_numpy(frame.get(ACTION_KEY)).tolist(),
                "reward": float(_as_numpy(frame.get(REWARD_KEY)).reshape(-1)[0])
                if REWARD_KEY in frame
                else None,
                "success": bool(_as_numpy(frame.get(SUCCESS_KEY)).reshape(-1)[0])
                if SUCCESS_KEY in frame
                else None,
                "next_done": bool(_as_numpy(frame.get(NEXT_DONE_KEY)).reshape(-1)[0])
                if NEXT_DONE_KEY in frame
                else None,
                "camera": (
                    OBS_IMAGE_KEY in frame
                    and _as_numpy(frame.get(OBS_IMAGE_KEY)).size > 0
                ),
            }
        )
    return {
        "episode_index": episode_index,
        "frames": rows,
        "frame_count": len(rows),
        "browse_schema": PIPELINE_SCHEMA_VERSION,
    }


def visualize_lerobot_episode(
    output_dir: Path,
    repo_id: str,
    episode_index: int,
    image_dir: Path,
) -> dict[str, object]:
    """生成相机帧缩略图网格与帧级 JSON 报告。"""
    dataset = open_lerobot_dataset(output_dir, repo_id)
    frames = read_episode_frames(dataset, episode_index)
    image_dir = Path(image_dir).expanduser().resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_paths: list[str] = []
    try:
        from PIL import Image
    except (ImportError, OSError) as exc:
        raise LerobotPipelineError("可视化需要 Pillow") from exc
    for index, frame in enumerate(frames):
        try:
            array = _image_hwc(frame.get(OBS_IMAGE_KEY))
        except LerobotPipelineError:
            continue
        image = Image.fromarray(array)
        thumbnail_path = image_dir / f"episode_{episode_index}_frame_{index:04d}.png"
        image.save(thumbnail_path)
        thumbnail_paths.append(str(thumbnail_path))
    report = browse_lerobot_episode(output_dir, repo_id, episode_index)
    report["thumbnails"] = thumbnail_paths
    _atomic_write_json(
        image_dir / f"episode_{episode_index}.json", report
    )
    return report


__all__ = [
    "PIPELINE_SCHEMA_VERSION",
    "OBS_STATE_KEY",
    "OBS_ENV_STATE_KEY",
    "OBS_IMAGE_KEY",
    "ACTION_KEY",
    "REWARD_KEY",
    "SUCCESS_KEY",
    "NEXT_DONE_KEY",
    "LerobotPipelineError",
    "LerobotConversionConfig",
    "TaskConversionResult",
    "ConversionSummary",
    "FrameInspection",
    "EpisodeInspection",
    "InspectionSummary",
    "browse_lerobot_episode",
    "build_lerobot_features",
    "convert_maniskill_to_lerobot",
    "inspect_lerobot_dataset",
    "load_maniskill_summary",
    "make_render_env",
    "maniskill_state_to_lerobot_features",
    "open_lerobot_dataset",
    "read_episode_frames",
    "visualize_lerobot_episode",
]
