"""ManiSkill 入门：环境契约、脚本策略与单 episode 验收。"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np


SUPPORTED_TASKS = ("PickCube-v1", "PushCube-v1")


class ManiSkillUnavailableError(RuntimeError):
    """ManiSkill 可选运行环境不可用。"""


class ObservationContractError(RuntimeError):
    """ManiSkill observation 缺少策略所需字段。"""


class EpisodeContractError(RuntimeError):
    """环境返回值或动作违反 Gymnasium episode 契约。"""


@dataclass(frozen=True)
class ManiSkillTaskConfig:
    """一个可复现的 ManiSkill CPU 单环境任务配置。"""

    env_id: str
    seed: int = 0
    obs_mode: str = "state_dict"
    control_mode: str = "pd_ee_delta_pos"
    reward_mode: str = "normalized_dense"
    sim_backend: str = "physx_cpu"
    max_steps: int = 50

    def __post_init__(self) -> None:
        if self.env_id not in SUPPORTED_TASKS:
            raise ValueError(
                f"不支持的 ManiSkill 任务 {self.env_id!r}，"
                f"仅支持 {SUPPORTED_TASKS}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed 必须是整数")
        if self.seed < 0:
            raise ValueError("seed 不能为负数")
        if not isinstance(self.max_steps, int) or isinstance(
            self.max_steps, bool
        ):
            raise ValueError("max_steps 必须是整数")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须为正数")
        if self.obs_mode != "state_dict":
            raise ValueError("第 10 周验收固定使用 state_dict observation")
        if self.control_mode != "pd_ee_delta_pos":
            raise ValueError("第 10 周验收固定使用 pd_ee_delta_pos control")
        if self.reward_mode != "normalized_dense":
            raise ValueError("第 10 周验收固定使用 normalized_dense reward")
        if self.sim_backend != "physx_cpu":
            raise ValueError("第 10 周验收固定使用 physx_cpu backend")


@dataclass(frozen=True)
class ObservationField:
    """一个 observation 叶子字段的路径、shape 和 dtype。"""

    path: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class EpisodeResult:
    """一次 ManiSkill episode 的可序列化验收摘要。"""

    env_id: str
    seed: int
    observation_fields: tuple[ObservationField, ...]
    action_shape: tuple[int, ...]
    steps: int
    total_reward: float
    success: bool
    terminated: bool
    truncated: bool
    final_info: dict[str, object]
    phase_history: tuple[str, ...]


class Policy(Protocol):
    """Episode runner 所需的最小策略接口。"""

    @property
    def phase(self) -> str: ...

    def reset(self) -> None: ...

    def act(self, observation: Mapping[str, object]) -> np.ndarray: ...


def summarize_observation(observation: object) -> tuple[ObservationField, ...]:
    """递归提取 observation 叶子字段，不复制大块数值。"""
    fields: list[ObservationField] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                child_path = f"{path}.{key}" if path else str(key)
                visit(value[key], child_path)
            return
        array = _as_numpy(value)
        fields.append(
            ObservationField(
                path=path or "<root>",
                shape=tuple(int(size) for size in array.shape),
                dtype=str(array.dtype),
            )
        )

    visit(observation, "")
    return tuple(fields)


def _as_numpy(value: object) -> np.ndarray:
    """将 NumPy、Torch 风格对象或标量转换为 CPU NumPy array。"""
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return np.asarray(candidate)


def _pythonize(value: object) -> object:
    """递归转换成 JSON 兼容的 Python 值。"""
    if isinstance(value, Mapping):
        return {str(key): _pythonize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_pythonize(item) for item in value]
    array = _as_numpy(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _scalar_float(value: object, name: str) -> float:
    array = _as_numpy(value)
    if array.size != 1:
        raise EpisodeContractError(
            f"{name} 必须是标量，实际 shape={array.shape}"
        )
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise EpisodeContractError(f"{name} 必须是有限数")
    return result


def _scalar_bool(value: object, name: str) -> bool:
    array = _as_numpy(value)
    if array.size != 1:
        raise EpisodeContractError(
            f"{name} 必须是标量，实际 shape={array.shape}"
        )
    return bool(array.reshape(-1)[0])


def _required_vector(
    observation: Mapping[str, object], key: str, minimum_size: int
) -> np.ndarray:
    extra = observation.get("extra")
    if not isinstance(extra, Mapping) or key not in extra:
        raise ObservationContractError(f"observation 缺少 extra.{key}")
    vector = _as_numpy(extra[key]).astype(float, copy=False).reshape(-1)
    if vector.size < minimum_size:
        raise ObservationContractError(
            f"extra.{key} 至少需要 {minimum_size} 个值，"
            f"实际只有 {vector.size} 个"
        )
    if not np.all(np.isfinite(vector)):
        raise ObservationContractError(f"extra.{key} 包含非有限数")
    return vector


def _optional_bool(
    observation: Mapping[str, object], key: str, default: bool = False
) -> bool:
    extra = observation.get("extra")
    if not isinstance(extra, Mapping) or key not in extra:
        return default
    array = _as_numpy(extra[key])
    if array.size != 1:
        raise ObservationContractError(f"extra.{key} 必须是标量")
    return bool(array.reshape(-1)[0])


class ScriptedPolicy:
    """用状态输入驱动 PickCube/PushCube 的有限状态机策略。"""

    _POSITION_SCALE = 0.10

    def __init__(self, env_id: str) -> None:
        if env_id not in SUPPORTED_TASKS:
            raise ValueError(f"没有任务 {env_id!r} 的 scripted policy")
        self.env_id = env_id
        self._phase = ""
        self._phase_steps = 0
        self.reset()

    @property
    def phase(self) -> str:
        """返回当前状态机阶段。"""
        return self._phase

    def reset(self) -> None:
        """重置到任务的首个阶段。"""
        self._phase = (
            "above_object"
            if self.env_id == "PickCube-v1"
            else "above_behind"
        )
        self._phase_steps = 0

    def act(self, observation: Mapping[str, object]) -> np.ndarray:
        """根据 observation 产生一个四维归一化末端增量动作。"""
        tcp = _required_vector(observation, "tcp_pose", 3)[:3]
        obj = _required_vector(observation, "obj_pose", 3)[:3]
        goal = _required_vector(observation, "goal_pos", 3)[:3]
        self._phase_steps += 1
        if self.env_id == "PickCube-v1":
            return self._act_pick(observation, tcp, obj, goal)
        return self._act_push(tcp, obj, goal)

    def _move_action(
        self, tcp: np.ndarray, target: np.ndarray, gripper: float
    ) -> np.ndarray:
        delta = (target - tcp) / self._POSITION_SCALE
        action = np.concatenate((np.clip(delta, -1.0, 1.0), [gripper]))
        return action.astype(np.float32)

    def _advance(self, phase: str) -> None:
        self._phase = phase
        self._phase_steps = 0

    def _act_pick(
        self,
        observation: Mapping[str, object],
        tcp: np.ndarray,
        obj: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        if self._phase == "above_object":
            target = obj + np.array([0.0, 0.0, 0.10])
            if np.linalg.norm(target - tcp) < 0.018 or self._phase_steps >= 8:
                self._advance("descend")
            return self._move_action(tcp, target, 1.0)

        if self._phase == "descend":
            target = obj.copy()
            if np.linalg.norm(target - tcp) < 0.012 or self._phase_steps >= 8:
                self._advance("close")
            return self._move_action(tcp, target, 1.0)

        if self._phase == "close":
            target = obj.copy()
            if _optional_bool(observation, "is_grasped") or self._phase_steps >= 6:
                self._advance("lift_to_goal")
            return self._move_action(tcp, target, -1.0)

        if self._phase == "lift_to_goal":
            target = goal.copy()
            if np.linalg.norm(target - obj) < 0.018 or self._phase_steps >= 18:
                self._advance("settle")
            return self._move_action(tcp, target, -1.0)

        return self._move_action(tcp, tcp, -1.0)

    def _act_push(
        self, tcp: np.ndarray, obj: np.ndarray, goal: np.ndarray
    ) -> np.ndarray:
        direction_xy = goal[:2] - obj[:2]
        distance = float(np.linalg.norm(direction_xy))
        if distance < 1e-9:
            direction_xy = np.array([1.0, 0.0])
        else:
            direction_xy /= distance
        direction = np.array([direction_xy[0], direction_xy[1], 0.0])
        behind = obj - direction * 0.055

        if self._phase == "above_behind":
            target = behind + np.array([0.0, 0.0, 0.08])
            if np.linalg.norm(target - tcp) < 0.018 or self._phase_steps >= 10:
                self._advance("behind_object")
            return self._move_action(tcp, target, -1.0)

        if self._phase == "behind_object":
            target = behind.copy()
            target[2] = obj[2]
            if np.linalg.norm(target - tcp) < 0.012 or self._phase_steps >= 10:
                self._advance("push_through_goal")
            return self._move_action(tcp, target, -1.0)

        if self._phase == "push_through_goal":
            target = goal + direction * 0.035
            target[2] = obj[2]
            if np.linalg.norm(goal[:2] - obj[:2]) < 0.025:
                self._advance("settle")
            return self._move_action(tcp, target, -1.0)

        return self._move_action(tcp, tcp, -1.0)


def make_maniskill_env(config: ManiSkillTaskConfig) -> object:
    """延迟导入 ManiSkill，并创建非批量 NumPy CPU Gym 环境。"""
    _configure_macos_vulkan()
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
    except (ImportError, OSError) as exc:
        raise ManiSkillUnavailableError(
            "ManiSkill 运行环境不可用；请用 Python 3.11 创建 "
            "`.venv-maniskill` 并安装 requirements-maniskill.txt"
        ) from exc

    base_env = gym.make(
        config.env_id,
        num_envs=1,
        obs_mode=config.obs_mode,
        control_mode=config.control_mode,
        reward_mode=config.reward_mode,
        sim_backend=config.sim_backend,
    )
    try:
        return CPUGymWrapper(base_env)
    except Exception:
        base_env.close()
        raise


def _configure_macos_vulkan() -> None:
    """在 Apple Silicon Homebrew 环境中定位 MoltenVK ICD。"""
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


def run_episode(
    config: ManiSkillTaskConfig,
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    policy_factory: Callable[[str], Policy] = ScriptedPolicy,
) -> EpisodeResult:
    """执行一个 scripted episode，并聚合四类核心运行时证据。"""
    env: Any = env_factory(config)
    try:
        policy = policy_factory(config.env_id)
        return run_episode_in_env(config, env, policy)
    finally:
        env.close()


def run_episode_in_env(
    config: ManiSkillTaskConfig,
    env: object,
    policy: Policy,
) -> EpisodeResult:
    """在调用方持有的环境中执行一次 episode，不关闭该环境。"""
    runtime_env: Any = env
    steps = 0
    total_reward = 0.0
    success = False
    terminated = False
    truncated = False
    final_info: dict[str, object] = {}
    phase_history: list[str] = []

    reset_result = runtime_env.reset(seed=config.seed)
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise EpisodeContractError("env.reset() 必须返回 (observation, info)")
    observation, _ = reset_result
    if not isinstance(observation, Mapping):
        raise EpisodeContractError("state_dict observation 必须是 mapping")
    observation_fields = summarize_observation(observation)

    action_space = runtime_env.action_space
    action_shape = tuple(int(size) for size in action_space.shape)
    policy.reset()
    for step_index in range(1, config.max_steps + 1):
        phase_history.append(policy.phase)
        action = np.asarray(policy.act(observation), dtype=np.float32)
        if action.shape != action_shape:
            raise EpisodeContractError(
                f"动作 shape={action.shape}，期望 {action_shape}"
            )
        if not np.all(np.isfinite(action)):
            raise EpisodeContractError("动作包含非有限数")
        if not action_space.contains(action):
            raise EpisodeContractError(f"动作超出 action space：{action}")

        step_result = runtime_env.step(action)
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise EpisodeContractError("env.step() 必须返回 Gymnasium 五元组")
        observation, reward, raw_terminated, raw_truncated, info = step_result
        if not isinstance(observation, Mapping):
            raise EpisodeContractError("step observation 必须是 mapping")
        if not isinstance(info, Mapping):
            raise EpisodeContractError("step info 必须是 mapping")

        steps = step_index
        total_reward += _scalar_float(reward, "reward")
        terminated = _scalar_bool(raw_terminated, "terminated")
        truncated = _scalar_bool(raw_truncated, "truncated")
        success = _scalar_bool(info.get("success", False), "info.success")
        converted_info = _pythonize(info)
        if not isinstance(converted_info, dict):
            raise EpisodeContractError("转换后的 info 必须是 dict")
        final_info = converted_info
        if success or terminated or truncated:
            break

    return EpisodeResult(
        env_id=config.env_id,
        seed=config.seed,
        observation_fields=observation_fields,
        action_shape=action_shape,
        steps=steps,
        total_reward=total_reward,
        success=success,
        terminated=terminated,
        truncated=truncated,
        final_info=final_info,
        phase_history=tuple(phase_history),
    )
