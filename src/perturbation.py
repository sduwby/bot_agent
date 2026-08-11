"""第 17 周：ACT 泛化评测环境——背景/照明/位姿扰动。

在 ``lerobot_pipeline.make_render_env()`` 之上提供可复现的扰动包装：
背景（新增彩色视觉面板）、照明（``set_ambient_light``）与物体初始位姿
（由 reset seed 派生）。三项扰动可独立开关，颜色由 episode seed 确定性派生，
不改 ManiSkill 任务、成功条件或动作空间。
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot_pipeline import make_render_env
from maniskill_intro import SUPPORTED_TASKS, ManiSkillTaskConfig

PERTURBATION_SCHEMA_VERSION = "maniskill_perturbation_v1"

# 默认扰动颜色调色板（RGB，[0,1]）。背景面板与照明共享确定性派生规则。
DEFAULT_BACKGROUND_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.30, 0.30),
    (0.30, 0.90, 0.30),
    (0.30, 0.30, 0.90),
    (0.95, 0.75, 0.20),
    (0.55, 0.25, 0.85),
)
DEFAULT_AMBIENT_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.30, 0.30),
    (0.30, 0.90, 0.30),
    (0.30, 0.30, 0.90),
    (0.95, 0.85, 0.30),
    (0.50, 0.20, 0.90),
)

# 背景面板放置位置（位于工作台后方，处于 PickCube/PushCube 相机视野内）。
_BACKGROUND_PANEL_POSE = (-0.9, 0.0, 0.9)
_BACKGROUND_PANEL_HALF_SIZE = (0.05, 2.0, 1.0)


class PerturbationError(RuntimeError):
    """扰动配置、环境工厂或扰动应用契约错误。"""


@dataclass(frozen=True)
class PerturbationConfig:
    """背景/照明/位姿扰动开关与确定性颜色调色板。

    ``pose`` 扰动通过 reset seed 派生：分布外评测使用不同于分布内的 seed 集，
    使物体初始位姿落入更广范围。``background`` / ``lighting`` 的颜色由每个
    episode 的 seed 经固定 RNG 派生，保证同 seed 可复现。
    """

    background: bool = False
    lighting: bool = False
    pose: bool = False
    background_colors: tuple[tuple[float, float, float], ...] = (
        DEFAULT_BACKGROUND_COLORS
    )
    ambient_colors: tuple[tuple[float, float, float], ...] = (
        DEFAULT_AMBIENT_COLORS
    )

    def __post_init__(self) -> None:
        if not (self.background or self.lighting or self.pose):
            raise ValueError("空扰动：background/lighting/pose 至少开启一项")
        for name, colors in (
            ("background_colors", self.background_colors),
            ("ambient_colors", self.ambient_colors),
        ):
            if not colors:
                raise ValueError(f"{name} 不能为空")
            for color in colors:
                _validate_color(color, name)


def _validate_color(color: object, name: str) -> None:
    if (
        not isinstance(color, (tuple, list))
        or len(color) != 3
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and np.isfinite(float(item))
            and 0.0 <= float(item) <= 1.0
            for item in color
        )
    ):
        raise ValueError(f"{name} 颜色必须是三个 [0,1] 数值组成的 RGB")


def _derive_color(seed: int, palette: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
    """按 seed 确定性派生一个调色板颜色。"""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed 必须是非负整数")
    if not palette:
        raise ValueError("调色板不能为空")
    return tuple(palette[random.Random(seed).randrange(len(palette))])


def _scene(env: object) -> object:
    base_env = getattr(env, "base_env", env)
    scene = getattr(base_env, "scene", None)
    if scene is None:
        raise PerturbationError("环境缺少 scene 引用")
    return scene


def _create_background_panel(scene: object, color: tuple[float, float, float]):
    """在场景中新增一个纯色背景面板并返回其 RenderMaterial 引用。"""
    try:
        import sapien
        import sapien.render
    except (ImportError, OSError) as exc:
        raise PerturbationError("背景扰动需要 sapien.render") from exc
    material = sapien.render.RenderMaterial()
    material.set_base_color([*color, 1.0])
    builder = scene.create_actor_builder()
    builder.add_box_visual(
        pose=sapien.Pose(p=list(_BACKGROUND_PANEL_POSE)),
        half_size=list(_BACKGROUND_PANEL_HALF_SIZE),
        material=material,
        name="background-panel",
    )
    builder.build_static(name="background-panel")
    scene.update_render()
    return material


def _recolor_background(scene: object, material: object, color: tuple[float, float, float]) -> None:
    material.set_base_color([*color, 1.0])
    scene.update_render()


class PerturbedRenderEnv:
    """包装 ManiSkill 渲染环境，在每次 reset 后应用配置的扰动。"""

    def __init__(self, env: object, perturbation: PerturbationConfig) -> None:
        self._env = env
        self._perturbation = perturbation
        self._background_material: object | None = None
        self._background_created = False
        self.last_applied: dict[str, object] = {}

    @property
    def base_env(self) -> object:
        """暴露底层 ManiSkill BaseEnv（含 scene 引用）。"""
        return self._env.base_env

    def reset(self, seed: int | None = None):
        result = self._env.reset(seed=seed)
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            raise PerturbationError("扰动环境 reset 必须返回二元组")
        applied: dict[str, object] = {}
        if seed is not None:
            scene = _scene(self)
            if self._perturbation.lighting:
                color = _derive_color(seed, self._perturbation.ambient_colors)
                scene.set_ambient_light(list(color))
                scene.update_render()
                applied["lighting"] = list(color)
            if self._perturbation.background:
                color = _derive_color(seed, self._perturbation.background_colors)
                if not self._background_created:
                    self._background_material = _create_background_panel(scene, color)
                    self._background_created = True
                else:
                    _recolor_background(scene, self._background_material, color)
                applied["background"] = list(color)
        self.last_applied = applied
        return observation, info

    def step(self, action: object) -> object:
        return self._env.step(action)

    def render(self) -> object:
        return self._env.render()

    def close(self) -> None:
        return self._env.close()

    def __getattr__(self, name: str) -> object:
        return getattr(self._env, name)


def make_perturbed_render_env(
    config: ManiSkillTaskConfig,
    perturbation: PerturbationConfig,
) -> object:
    """创建带背景/照明/位姿扰动的渲染环境。

    与 ``make_render_env`` 同签名约束：接收 ``ManiSkillTaskConfig``，
    返回带 ``reset/step/render/close`` 与 ``base_env.scene`` 的环境对象。
    """
    if config.env_id not in SUPPORTED_TASKS:
        raise PerturbationError(f"不支持的任务 {config.env_id!r}")
    env = make_render_env(config)
    try:
        return PerturbedRenderEnv(env, perturbation)
    except Exception:
        env.close()
        raise


def image_mean_abs_diff(a: object, b: object) -> float:
    """计算两帧渲染图像的通道平均绝对差，用于验证扰动生效。"""
    left = _image_hwc(a).astype(np.float32)
    right = _image_hwc(b).astype(np.float32)
    if left.shape != right.shape:
        raise PerturbationError(
            f"图像 shape 不一致：{left.shape} vs {right.shape}"
        )
    return float(np.abs(left - right).mean())


def _image_hwc(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[2] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] != 3 or array.size == 0:
        raise PerturbationError("渲染帧不是有效 RGB 图像")
    return array.astype(np.uint8)


__all__ = [
    "PERTURBATION_SCHEMA_VERSION",
    "DEFAULT_BACKGROUND_COLORS",
    "DEFAULT_AMBIENT_COLORS",
    "PerturbationConfig",
    "PerturbationError",
    "PerturbedRenderEnv",
    "image_mean_abs_diff",
    "make_perturbed_render_env",
]
