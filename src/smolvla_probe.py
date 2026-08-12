"""第 18 周：SmolVLA 适配探针。

目标：验证 SmolVLA 是否能作为 ManiSkill 视觉策略。包含：

1. ``make_smolvla_policy()`` 从 ``lerobot/smolvla_base`` 加载 policy + pre/post processors；
2. ``observation_to_smolvla_input()`` 把 ManiSkill observation 映射为 SmolVLA 输入；
3. ``run_smolvla_episode()`` 单 episode 评测并记录动作趋势；
4. ``evaluate_smolvla_probe()`` 固定 seed 评测 + go/no-go 决策。

注意：SmolVLA base 预训练于 SO-101（6D state、3 相机、6D action），与 ManiSkill
（5/13 state、单相机、4D action）维度不同。零样本评测验证"能加载 + 动作有方向
意义"；真正评测需微调后覆盖 input/output features（云上执行）。
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

from behavior_cloning import ACTION_DIM
from maniskill_intro import ManiSkillTaskConfig

# SmolVLA 常量。
SMOLVLA_SCHEMA_VERSION = "maniskill_smolvla_probe_v1"
DEFAULT_MODEL_ID = "lerobot/smolvla_base"
SMOLVLA_IMAGE_SIZE = 256
# 任务文本映射。
TASK_TEXTS = {
    "PickCube-v1": "Pick up the cube and place it at the target.",
    "PushCube-v1": "Push the cube to the target.",
}


class SmolVLAProbeError(RuntimeError):
    """SmolVLA 探针契约错误。"""


@dataclass(frozen=True)
class SmolVLAProbeConfig:
    """SmolVLA 零样本探针配置。"""

    output_dir: Path
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda"
    seeds: tuple[int, ...] = tuple(range(500, 520))
    tasks: tuple[str, ...] = ("PickCube-v1", "PushCube-v1")
    perturbed: bool = False
    max_steps: int = 50
    image_size: int = SMOLVLA_IMAGE_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        if self.device not in ("cuda", "mps", "cpu"):
            raise ValueError("device 只支持 cuda、mps 或 cpu")
        if self.image_size <= 0:
            raise ValueError("image_size 必须是正整数")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须是正整数")
        if not self.seeds:
            raise ValueError("seeds 不能为空")
        checked: list[str] = []
        for env_id in self.tasks:
            if env_id not in TASK_TEXTS:
                raise ValueError(f"不支持的任务 {env_id!r}")
            if env_id in checked:
                raise ValueError(f"任务重复：{env_id}")
            checked.append(env_id)
        object.__setattr__(self, "tasks", tuple(checked))


@dataclass(frozen=True)
class ProbeEpisode:
    """一次 SmolVLA 零样本 episode 的结果与动作趋势。"""

    env_id: str
    seed: int
    steps: int
    success: bool
    truncated: bool
    error: str | None
    # 动作趋势摘要。
    action_xy_mean: tuple[float, float]
    action_z_mean: float
    gripper_close_ratio: float
    raw_action_norm_mean: float
    saturating_ratio: float


@dataclass(frozen=True)
class TaskProbeResult:
    """一个任务的零样本探针聚合。"""

    env_id: str
    episodes: int
    successes: int
    success_rate: float
    truncated_all: bool
    mean_xy_scale: float
    mean_z_scale: float
    gripper_close_ratio: float
    action_saturation: float


@dataclass(frozen=True)
class ProbeReport:
    """SmolVLA 探针报告与 go/no-go 决策。"""

    schema_version: str
    created_at: str
    model_id: str
    device: str
    tasks: tuple[TaskProbeResult, ...]
    go: bool
    decision: str
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
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
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


def _to_float32_numpy(value: object) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    return np.asarray(candidate, dtype=np.float32).reshape(-1)


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    try:
        from PIL import Image
    except (ImportError, OSError) as exc:
        raise SmolVLAProbeError("图像缩放需要 Pillow") from exc
    array = np.asarray(image)
    if array.ndim != 3:
        raise SmolVLAProbeError(f"图像维度错误：{array.shape}")
    h, w, channels = array.shape
    if channels != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise SmolVLAProbeError(f"图像 channels 错误：{array.shape}")
    pil = Image.fromarray(array.astype(np.uint8))
    resized = pil.resize((size, size), Image.BILINEAR)
    result = np.asarray(resized, dtype=np.float32) / 255.0
    return np.transpose(result, (2, 0, 1))  # (3, H, W)


def make_smolvla_policy(
    model_id: str,
    device: str,
    preprocessor_overrides: Mapping[str, object] | None = None,
) -> tuple[object, object, object]:
    """加载 SmolVLAPolicy + pre/post processors。"""
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.processor_smolvla import (
            make_smolvla_pre_post_processors,
        )
    except (ImportError, OSError) as exc:
        raise SmolVLAProbeError(
            f"SmolVLA 依赖不可用（需 lerobot[smolvla]）：{exc}"
        ) from exc
    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy.to(device)
    pre, post = make_smolvla_pre_post_processors(policy.config)
    return policy, pre, post


def observation_to_smolvla_input(
    observation: Mapping[str, object],
    image: np.ndarray,
    task_text: str,
    image_size: int,
) -> dict[str, object]:
    """把 ManiSkill observation + 渲染图 + 任务文本构造为 SmolVLA 输入。"""
    extra = observation.get("extra")
    if not isinstance(extra, Mapping):
        raise SmolVLAProbeError("observation 缺少 extra mapping")
    try:
        tcp = np.asarray(extra["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
        qpos = np.asarray(observation["agent"]["qpos"], dtype=np.float32).reshape(-1)
        is_grasped = np.asarray(extra.get("is_grasped", False)).reshape(-1)[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise SmolVLAProbeError(f"observation 缺少 SmolVLA 输入字段：{exc}") from exc
    # proprioception：tcp_pos[3] + gripper_qpos[-2:] + is_grasped[1]（对齐 SO-101 6D）。
    gripper = qpos[-2:] if qpos.size >= 2 else np.zeros(2, dtype=np.float32)
    state = np.concatenate(
        (tcp, gripper, np.asarray([is_grasped], dtype=np.float32))
    ).astype(np.float32)
    image_resized = _resize_image(image, image_size)
    return {
        "observation.state": state,
        "observation.images.camera1": image_resized,
        "task": task_text,
    }


def _scalar_bool(value: object, name: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise SmolVLAProbeError(f"{name} 必须是标量")
    return bool(array.reshape(-1)[0])


def _info_flag(info: Mapping[str, object], key: str) -> bool:
    try:
        return bool(np.asarray(info.get(key, False)).reshape(-1)[0])
    except (ValueError, IndexError, TypeError):
        return False


def run_smolvla_episode(
    config: ManiSkillTaskConfig,
    env: object,
    policy: object,
    pre: object,
    post: object,
    task_text: str,
    image_size: int,
) -> ProbeEpisode:
    """单 episode 零样本评测，记录动作趋势。"""
    torch = __import__("torch")
    env_any: Any = env
    reset_result = env_any.reset(seed=config.seed)
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise SmolVLAProbeError("env.reset() 未返回二元组")
    observation, _ = reset_result
    if not isinstance(observation, Mapping):
        raise SmolVLAProbeError("observation 必须是 mapping")
    policy.reset()
    steps = 0
    success = False
    truncated = False
    xy_means: list[float] = []
    z_means: list[float] = []
    close_counts = 0
    norms: list[float] = []
    saturating = 0
    try:
        for step in range(1, config.max_steps + 1):
            image_raw = np.asarray(env_any.render())
            batch = observation_to_smolvla_input(
                observation, image_raw, task_text, image_size
            )
            processed = pre(batch)
            action = policy.select_action(processed)
            action = post(action)
            action_np = _to_float32_numpy(action)
            if action_np.size < ACTION_DIM:
                raise SmolVLAProbeError(
                    f"SmolVLA 输出维度不足：{action_np.shape}"
                )
            action_np = action_np[:ACTION_DIM]
            xy_means.append(float(np.mean(np.abs(action_np[:2]))))
            z_means.append(float(abs(action_np[2])))
            if action_np[3] < -0.5:
                close_counts += 1
            norms.append(float(np.linalg.norm(action_np)))
            if np.any(np.abs(action_np) > 0.99):
                saturating += 1
            step_result = env_any.step(action_np.astype(np.float32))
            if not isinstance(step_result, tuple) or len(step_result) != 5:
                raise SmolVLAProbeError("env.step() 未返回五元组")
            observation, _, raw_terminated, raw_truncated, info = step_result
            if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
                raise SmolVLAProbeError("step observation/info 必须是 mapping")
            steps = step
            truncated = _scalar_bool(raw_truncated, "truncated")
            success = _info_flag(info, "success")
            if success or truncated or _scalar_bool(raw_terminated, "terminated"):
                break
        return ProbeEpisode(
            env_id=config.env_id,
            seed=config.seed,
            steps=steps,
            success=success,
            truncated=truncated,
            error=None,
            action_xy_mean=(
                float(np.mean(xy_means)) if xy_means else 0.0,
                float(np.mean(z_means)) if z_means else 0.0,
            ),
            action_z_mean=float(np.mean(z_means)) if z_means else 0.0,
            gripper_close_ratio=close_counts / steps if steps else 0.0,
            raw_action_norm_mean=float(np.mean(norms)) if norms else 0.0,
            saturating_ratio=saturating / steps if steps else 0.0,
        )
    except Exception as exc:
        return ProbeEpisode(
            env_id=config.env_id,
            seed=config.seed,
            steps=steps,
            success=False,
            truncated=truncated,
            error=f"{type(exc).__name__}: {exc}",
            action_xy_mean=(0.0, 0.0),
            action_z_mean=0.0,
            gripper_close_ratio=0.0,
            raw_action_norm_mean=0.0,
            saturating_ratio=0.0,
        )


def _aggregate_task(env_id: str, episodes: Sequence[ProbeEpisode]) -> TaskProbeResult:
    eps = [e for e in episodes if e.env_id == env_id]
    successes = sum(e.success for e in eps)
    return TaskProbeResult(
        env_id=env_id,
        episodes=len(eps),
        successes=successes,
        success_rate=successes / len(eps) if eps else 0.0,
        truncated_all=all(e.truncated for e in eps) if eps else True,
        mean_xy_scale=float(np.mean([e.action_xy_mean[0] for e in eps])) if eps else 0.0,
        mean_z_scale=float(np.mean([e.action_z_mean for e in eps])) if eps else 0.0,
        gripper_close_ratio=float(np.mean([e.gripper_close_ratio for e in eps]))
        if eps
        else 0.0,
        action_saturation=float(np.mean([e.saturating_ratio for e in eps]))
        if eps
        else 0.0,
    )


def decide_go_no_go(
    tasks: Sequence[TaskProbeResult],
) -> tuple[bool, str]:
    """go/no-go：任一任务有成功 → go；否则依据动作趋势。"""
    any_success = any(t.successes > 0 for t in tasks)
    if any_success:
        return True, "存在成功 episode，具备继续微调的潜力"
    # 无成功：看动作是否有方向意义（非全零、非全饱和、夹爪有变化）。
    has_signal = any(
        t.mean_xy_scale > 0.01
        and t.action_saturation < 0.9
        and 0.05 < t.gripper_close_ratio < 0.95
        for t in tasks
    )
    if has_signal:
        return True, "零样本无成功但动作有方向意义，值得微调验证"
    return False, "零样本动作无有效信号（全零/全饱和/夹爪无变化），no-go"


def evaluate_smolvla_probe(
    config: SmolVLAProbeConfig,
    *,
    env_factory: Any | None = None,
) -> ProbeReport:
    """固定 seed 零样本评测 + go/no-go 决策并落盘。"""
    from lerobot_pipeline import make_render_env
    from maniskill_intro import make_maniskill_env
    from perturbation import PerturbationConfig, make_perturbed_render_env

    checked_seeds = _checked_seeds(config.seeds)
    policy, pre, post = make_smolvla_policy(config.model_id, config.device)

    def env_for(task_config: ManiSkillTaskConfig) -> object:
        if config.perturbed:
            return make_perturbed_render_env(
                task_config,
                PerturbationConfig(
                    background=True, lighting=True, pose=True
                ),
            )
        return make_render_env(task_config)

    episodes: list[ProbeEpisode] = []
    for env_id in config.tasks:
        for seed in checked_seeds:
            task_config = ManiSkillTaskConfig(
                env_id=env_id, seed=seed, max_steps=config.max_steps
            )
            env: Any | None = None
            try:
                env = env_for(task_config)
                episodes.append(
                    run_smolvla_episode(
                        task_config,
                        env,
                        policy,
                        pre,
                        post,
                        TASK_TEXTS[env_id],
                        config.image_size,
                    )
                )
            finally:
                if env is not None:
                    env.close()

    tasks = [
        _aggregate_task(env_id, episodes) for env_id in config.tasks
    ]
    go, decision = decide_go_no_go(tasks)
    report = ProbeReport(
        schema_version=SMOLVLA_SCHEMA_VERSION,
        created_at=_utc_now(),
        model_id=config.model_id,
        device=config.device,
        tasks=tuple(tasks),
        go=go,
        decision=decision,
        notes=(
            "SmolVLA base 预训练于 SO-101（6D state/3 相机/6D action），"
            "与 ManiSkill 维度不同；零样本仅验证加载与动作信号，"
            "正式评测需微调后覆盖 features。"
        ),
    )
    _atomic_write_json(config.output_dir / "probe_report.json", asdict(report))
    return report


__all__ = [
    "SMOLVLA_SCHEMA_VERSION",
    "DEFAULT_MODEL_ID",
    "SMOLVLA_IMAGE_SIZE",
    "TASK_TEXTS",
    "SmolVLAProbeError",
    "SmolVLAProbeConfig",
    "ProbeEpisode",
    "TaskProbeResult",
    "ProbeReport",
    "decide_go_no_go",
    "evaluate_smolvla_probe",
    "make_smolvla_policy",
    "observation_to_smolvla_input",
    "run_smolvla_episode",
]
