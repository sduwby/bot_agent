"""第 17 周测试：扰动配置、环境工厂、图像差异、分布内/外与失败分布逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lerobot_act import (
    ACTEvaluationTask,
    PolicyEvaluationSummary,
)
from maniskill_intro import ManiSkillTaskConfig
from perturbation import (
    DEFAULT_AMBIENT_COLORS,
    DEFAULT_BACKGROUND_COLORS,
    PerturbationConfig,
    PerturbationError,
    PerturbedRenderEnv,
    _derive_color,
    image_mean_abs_diff,
    make_perturbed_render_env,
)

import run_generalization as rg


# ---------------------------------------------------------------------------
# chunk-compare 解析
# ---------------------------------------------------------------------------


def test_parse_chunk_candidates_three_parts() -> None:
    candidates = rg._parse_chunk_candidates(
        ["c4:4:runs/maniskill_act/week17-c4-20k",
         "c8:8:runs/maniskill_act/week17-c8-20k"]
    )
    assert candidates == [
        ("c4", Path("runs/maniskill_act/week17-c4-20k"), 4),
        ("c8", Path("runs/maniskill_act/week17-c8-20k"), 8),
    ]


def test_parse_chunk_candidates_rejects_bad_entries() -> None:
    with pytest.raises(ValueError, match="tag:chunk:path"):
        rg._parse_chunk_candidates(["only_tag"])
    with pytest.raises(ValueError, match="必须是整数"):
        rg._parse_chunk_candidates(["c4:four:path"])
    with pytest.raises(ValueError, match="正整数"):
        rg._parse_chunk_candidates(["c4:0:path"])
    with pytest.raises(ValueError, match="tag 不能为空"):
        rg._parse_chunk_candidates([":4:path"])


# ---------------------------------------------------------------------------
# 扰动配置
# ---------------------------------------------------------------------------


def test_empty_perturbation_rejected() -> None:
    with pytest.raises(ValueError, match="空扰动"):
        PerturbationConfig(background=False, lighting=False, pose=False)


def test_single_flag_enabled_ok() -> None:
    assert PerturbationConfig(background=True).background is True
    assert PerturbationConfig(lighting=True).lighting is True
    assert PerturbationConfig(pose=True).pose is True


def test_invalid_color_rejected() -> None:
    with pytest.raises(ValueError, match="颜色"):
        PerturbationConfig(
            background=True, background_colors=((1.5, 0.0, 0.0),)
        )
    with pytest.raises(ValueError, match="颜色"):
        PerturbationConfig(lighting=True, ambient_colors=((0.0, 0.0),))
    with pytest.raises(ValueError, match="不能为空"):
        PerturbationConfig(background=True, background_colors=())


def test_derive_color_deterministic() -> None:
    first = _derive_color(600, DEFAULT_BACKGROUND_COLORS)
    second = _derive_color(600, DEFAULT_BACKGROUND_COLORS)
    assert first == second
    assert first in DEFAULT_BACKGROUND_COLORS


def test_derive_color_varies_across_seeds() -> None:
    colors = {_derive_color(seed, DEFAULT_BACKGROUND_COLORS) for seed in range(600, 620)}
    assert len(colors) > 1


def test_derive_color_rejects_bad_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        _derive_color(-1, DEFAULT_AMBIENT_COLORS)


# ---------------------------------------------------------------------------
# 图像差异
# ---------------------------------------------------------------------------


def test_image_diff_identical_zero() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert image_mean_abs_diff(image, image) == 0.0


def test_image_diff_different_positive() -> None:
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.full((8, 8, 3), 255, dtype=np.uint8)
    assert image_mean_abs_diff(a, b) == 255.0


def test_image_diff_chw_hwc_normalized() -> None:
    hwc = np.zeros((8, 8, 3), dtype=np.uint8)
    chw = np.transpose(hwc, (2, 0, 1))
    assert image_mean_abs_diff(hwc, chw) == 0.0


def test_image_diff_shape_mismatch_rejected() -> None:
    with pytest.raises(PerturbationError, match="shape"):
        image_mean_abs_diff(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 环境工厂
# ---------------------------------------------------------------------------


class _FakeMaterial:
    def __init__(self) -> None:
        self.colors: list[list[float]] = []

    def set_base_color(self, color: list[float]) -> None:
        self.colors.append(list(color))


class _FakeBuilder:
    def __init__(self, scene: "_FakeScene") -> None:
        self._scene = scene
        self._material: _FakeMaterial | None = None

    def add_box_visual(self, **kwargs: Any) -> "_FakeBuilder":
        self._material = _FakeMaterial()
        return self

    def build_static(self, name: str) -> object:
        self._scene.panels.append(name)
        return self._material


class _FakeScene:
    def __init__(self) -> None:
        self.ambient_calls: list[list[float]] = []
        self.update_calls = 0
        self.panels: list[str] = []
        self.actors: dict[str, object] = {"ground": object()}

    def set_ambient_light(self, color: list[float]) -> None:
        self.ambient_calls.append(list(color))

    def update_render(self) -> None:
        self.update_calls += 1

    def create_actor_builder(self) -> _FakeBuilder:
        return _FakeBuilder(self)


class _FakeEnv:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.reset_seeds: list[int | None] = []
        self._image = np.zeros((8, 8, 3), dtype=np.uint8)

    @property
    def base_env(self) -> "_FakeEnv":
        return self

    def reset(self, seed: int | None = None):
        self.reset_seeds.append(seed)
        return ({"extra": {}}, {})

    def step(self, action: object):
        return ({"extra": {}}, 0.0, False, False, {})

    def render(self) -> np.ndarray:
        return self._image

    def close(self) -> None:
        pass


def test_make_perturbed_render_env_wraps(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "perturbation._create_background_panel", _fake_create_background_panel
    )  # type: ignore[attr-defined]
    env = _FakeEnv()
    monkeypatch.setattr(
        "perturbation.make_render_env", lambda config: env
    )  # type: ignore[attr-defined]
    perturbed = make_perturbed_render_env(
        ManiSkillTaskConfig(env_id="PickCube-v1"),
        PerturbationConfig(background=True, lighting=True),
    )
    assert isinstance(perturbed, PerturbedRenderEnv)
    observation, info = perturbed.reset(seed=600)
    assert observation == {"extra": {}}
    assert info == {}
    assert env.reset_seeds == [600]


def test_lighting_applied_on_reset(monkeypatch: object) -> None:
    env = _FakeEnv()
    monkeypatch.setattr(
        "perturbation.make_render_env", lambda config: env
    )  # type: ignore[attr-defined]
    perturbed = make_perturbed_render_env(
        ManiSkillTaskConfig(env_id="PickCube-v1"),
        PerturbationConfig(lighting=True),
    )
    perturbed.reset(seed=600)
    assert len(env.scene.ambient_calls) == 1
    assert env.scene.ambient_calls[0] in [
        list(color) for color in DEFAULT_AMBIENT_COLORS
    ]


def test_lighting_off_does_not_apply(monkeypatch: object) -> None:
    env = _FakeEnv()
    monkeypatch.setattr(
        "perturbation.make_render_env", lambda config: env
    )  # type: ignore[attr-defined]
    perturbed = make_perturbed_render_env(
        ManiSkillTaskConfig(env_id="PickCube-v1"),
        PerturbationConfig(pose=True),
    )
    perturbed.reset(seed=600)
    assert env.scene.ambient_calls == []
    assert env.scene.panels == []


def _fake_create_background_panel(scene: object, color: tuple[float, float, float]):
    scene.panels.append("background-panel")
    material = _FakeMaterial()
    material.set_base_color([*color, 1.0])
    return material


def test_background_panel_created_once(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "perturbation._create_background_panel", _fake_create_background_panel
    )  # type: ignore[attr-defined]
    env = _FakeEnv()
    monkeypatch.setattr(
        "perturbation.make_render_env", lambda config: env
    )  # type: ignore[attr-defined]
    perturbed = make_perturbed_render_env(
        ManiSkillTaskConfig(env_id="PickCube-v1"),
        PerturbationConfig(background=True),
    )
    perturbed.reset(seed=600)
    perturbed.reset(seed=601)
    assert env.scene.panels == ["background-panel"]


def test_background_recolored_on_second_reset(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "perturbation._create_background_panel", _fake_create_background_panel
    )  # type: ignore[attr-defined]
    env = _FakeEnv()
    monkeypatch.setattr(
        "perturbation.make_render_env", lambda config: env
    )  # type: ignore[attr-defined]
    perturbed = make_perturbed_render_env(
        ManiSkillTaskConfig(env_id="PickCube-v1"),
        PerturbationConfig(background=True),
    )
    perturbed.reset(seed=600)
    assert perturbed.last_applied.get("background") is not None
    material = perturbed._background_material
    perturbed.reset(seed=600)
    assert perturbed._background_material is material


def test_reset_without_seed_skips_perturbation(monkeypatch: object) -> None:
    env = _FakeEnv()
    monkeypatch.setattr(
        "perturbation.make_render_env", lambda config: env
    )  # type: ignore[attr-defined]
    perturbed = make_perturbed_render_env(
        ManiSkillTaskConfig(env_id="PickCube-v1"),
        PerturbationConfig(lighting=True, background=True),
    )
    perturbed.reset(seed=None)
    assert env.scene.ambient_calls == []
    assert env.scene.panels == []


class _FakeConfig:
    def __init__(self, env_id: str) -> None:
        self.env_id = env_id


def test_factory_rejects_unsupported_task(monkeypatch: object) -> None:
    with pytest.raises(PerturbationError, match="不支持"):
        make_perturbed_render_env(
            _FakeConfig("Unknown-v1"),
            PerturbationConfig(pose=True),
        )


# ---------------------------------------------------------------------------
# 分布内/外与失败分布纯逻辑
# ---------------------------------------------------------------------------


def _summary(env_id: str, rate: float, episodes: int = 10) -> PolicyEvaluationSummary:
    successes = int(round(rate * episodes))
    return PolicyEvaluationSummary(
        policy_name="visual_act",
        schema_version="test",
        created_at="",
        requested_per_task=episodes,
        total_episodes=episodes,
        total_successes=successes,
        total_failures=episodes - successes,
        success_rate=rate,
        mean_inference_ms=1.0,
        mean_steps=10.0,
        total_chunk_queries=0,
        total_action_queries=0,
        nonfinite_actions=0,
        out_of_bounds_actions=0,
        tasks=(
            ACTEvaluationTask(
                env_id=env_id,
                episodes=episodes,
                successes=successes,
                failures=episodes - successes,
                success_rate=rate,
                mean_steps=10.0,
                exceptions=0,
            ),
        ),
        episodes=(),
    )


def test_split_stats_overall_and_per_task() -> None:
    summary = _summary("PickCube-v1", 0.8, episodes=10)
    stats = rg._split_stats(summary)
    assert stats["overall"]["success_rate"] == 0.8
    assert stats["PickCube-v1"]["successes"] == 8
    assert stats["PickCube-v1"]["episodes"] == 10


def test_rate_returns_task_rate() -> None:
    summary = _summary("PickCube-v1", 0.6)
    assert rg._rate(summary, "PickCube-v1") == 0.6
    assert np.isnan(rg._rate(summary, "PushCube-v1"))


def test_failure_diagnosis_phases() -> None:
    assert rg._failure_diagnosis(
        "PickCube-v1", 0, ever_grasped=False, ever_placed=False,
        simultaneously_valid_steps=0, truncated=True,
    ) == "truncated_only"
    assert rg._failure_diagnosis(
        "PickCube-v1", 0, ever_grasped=False, ever_placed=False,
        simultaneously_valid_steps=0, truncated=False,
    ) == "grasp_failed"
    assert rg._failure_diagnosis(
        "PickCube-v1", 0, ever_grasped=True, ever_placed=False,
        simultaneously_valid_steps=0, truncated=False,
    ) == "grasped_not_placed"
    assert rg._failure_diagnosis(
        "PickCube-v1", 0, ever_grasped=True, ever_placed=True,
        simultaneously_valid_steps=0, truncated=False,
    ) == "placed_but_not_static"


def test_generalization_config_rejects_seed_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="重叠"):
        rg.GeneralizationEvalConfig(
            checkpoint_dir=tmp_path,
            output_dir=tmp_path / "out",
            in_distribution_seeds=(600, 601),
            out_of_distribution_seeds=(601, 602),
            perturbation=PerturbationConfig(pose=True),
        )


def test_generalization_config_rejects_missing_perturbation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="perturbation"):
        rg.GeneralizationEvalConfig(
            checkpoint_dir=tmp_path,
            output_dir=tmp_path / "out",
            perturbation=None,
        )


def test_generalization_config_rejects_invalid_device(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        rg.GeneralizationEvalConfig(
            checkpoint_dir=tmp_path,
            output_dir=tmp_path / "out",
            device="tpu",
            perturbation=PerturbationConfig(pose=True),
        )
