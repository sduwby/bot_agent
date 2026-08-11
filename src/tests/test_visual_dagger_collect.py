"""视觉 ACT DAgger 补采：合并、save/load、配置校验与 fake env 采集测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from behavior_cloning import ACTION_DIM  # noqa: E402
from lerobot_act import ENV_STATE_DIM, ROBOT_STATE_DIM  # noqa: E402
from visual_act import (  # noqa: E402
    VisualACTTrainingConfig,
    VisualTaskData,
    split_task_transitions,
)
from visual_dagger_collect import (  # noqa: E402
    VISUAL_DAGGER_SCHEMA_VERSION,
    build_visual_dagger_transitions,
    collect_visual_dagger_episodes,
    load_visual_dagger_episodes,
    merge_visual_dagger_payloads,
    save_visual_dagger_episodes,
)

RES = 32


def _fake_task_data(
    episodes: int = 4, steps: int = 4, resolution: int = RES
) -> VisualTaskData:
    total = episodes * steps
    rng = np.random.default_rng(0)
    return VisualTaskData(
        env_id="PickCube-v1",
        episode_ids=tuple(range(episodes)),
        episode_lengths=tuple([steps] * episodes),
        images=rng.uniform(0, 1, (total, 3, resolution, resolution)).astype(
            np.float32
        ),
        robot_state=rng.uniform(-1, 1, (total, ROBOT_STATE_DIM)).astype(
            np.float32
        ),
        env_state=rng.uniform(-1, 1, (total, ENV_STATE_DIM)).astype(np.float32),
        actions=rng.uniform(-1, 1, (total, ACTION_DIM)).astype(np.float32),
    )


def _fake_dagger(
    lengths: tuple[int, ...] = (2, 3), resolution: int = RES
) -> dict[str, object]:
    n = sum(lengths)
    rng = np.random.default_rng(1)
    return {
        "schema_version": VISUAL_DAGGER_SCHEMA_VERSION,
        "env_id": "PickCube-v1",
        "resolution": resolution,
        "n_episodes": len(lengths),
        "episode_lengths": list(lengths),
        "total_labeled_transitions": n,
        "images": rng.integers(
            0, 255, (n, resolution, resolution, 3), dtype=np.uint8
        ),
        "robot_state": rng.uniform(-1, 1, (n, ROBOT_STATE_DIM)).astype(
            np.float32
        ),
        "env_state": rng.uniform(-1, 1, (n, ENV_STATE_DIM)).astype(np.float32),
        "actions": rng.uniform(-1, 1, (n, ACTION_DIM)).astype(np.float32),
    }


def _tiny_config(tmp_path: Path, **kwargs: Any) -> VisualACTTrainingConfig:
    defaults = dict(
        dataset_dir=tmp_path / "dataset",
        repo_id="test/repo",
        output_dir=tmp_path / "output",
        resolution=RES,
        chunk_size=4,
        n_action_steps=4,
        training_steps=2,
        batch_size=4,
        validation_interval=1,
        checkpoint_interval=1,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        pretrained_backbone_weights=None,
        device="cpu",
        train_episodes_per_task=3,
        validation_episodes_per_task=1,
    )
    defaults.update(kwargs)
    return VisualACTTrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------


def test_config_rejects_missing_dagger_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dagger_data"):
        _tiny_config(tmp_path, dagger_data=tmp_path / "missing.npz")


def test_config_accepts_existing_dagger_data(tmp_path: Path) -> None:
    dagger_path = tmp_path / "d.npz"
    dagger_path.write_bytes(b"placeholder")
    config = _tiny_config(tmp_path, dagger_data=dagger_path)
    assert config.dagger_data == dagger_path.resolve()


# ---------------------------------------------------------------------------
# 合并
# ---------------------------------------------------------------------------


def test_build_visual_dagger_transitions_merge() -> None:
    base = _fake_task_data(episodes=4, steps=4)
    dagger = _fake_dagger(lengths=(2, 3))
    merged = build_visual_dagger_transitions(
        base, dagger, resolution=RES, start_episode_id=100, insert_after=2
    )
    # 顺序：原始前 2 + dagger 2 + 原始后 2（validation 保持原始）
    assert merged.episode_ids == (0, 1, 100, 101, 2, 3)
    assert merged.episode_lengths == (4, 4, 2, 3, 4, 4)
    assert len(merged.images) == 4 * 4 + 5
    assert merged.images.dtype == np.float32
    assert float(merged.images.min()) >= 0.0
    assert float(merged.images.max()) <= 1.0
    # train = 原始前 2 + dagger，validation = 原始后 2（索引 13-20）
    train, validation = split_task_transitions(
        merged, train_episodes=4, validation_episodes=2
    )
    np.testing.assert_array_equal(validation, np.arange(13, 21))
    assert np.intersect1d(train, validation).size == 0


def test_build_visual_dagger_transitions_resize() -> None:
    base = _fake_task_data(episodes=2, steps=4)
    dagger = _fake_dagger(lengths=(2,), resolution=8)  # 补采图分辨率不同
    merged = build_visual_dagger_transitions(
        base, dagger, resolution=RES, start_episode_id=100, insert_after=1
    )
    assert merged.images.shape == (4 + 2 + 4, 3, RES, RES)


# ---------------------------------------------------------------------------
# save/load 与合并多轮
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    dagger = _fake_dagger()
    path = tmp_path / "dagger.npz"
    save_visual_dagger_episodes(dagger, path)
    loaded = load_visual_dagger_episodes(path)
    assert loaded["n_episodes"] == dagger["n_episodes"]
    assert loaded["episode_lengths"] == dagger["episode_lengths"]
    assert loaded["resolution"] == RES
    np.testing.assert_array_equal(loaded["images"], dagger["images"])
    np.testing.assert_array_equal(loaded["actions"], dagger["actions"])


def test_merge_visual_dagger_payloads() -> None:
    r1 = _fake_dagger(lengths=(2, 3))
    r2 = _fake_dagger(lengths=(4,))
    merged = merge_visual_dagger_payloads([r1, r2])
    assert merged["n_episodes"] == 3
    assert merged["episode_lengths"] == [2, 3, 4]
    assert merged["total_labeled_transitions"] == 9
    assert len(merged["images"]) == 9
    assert merged["resolution"] == RES


# ---------------------------------------------------------------------------
# fake env 采集（不依赖 Unity）
# ---------------------------------------------------------------------------


def _recovery_observation(
    *, tcp: list[float], obj: list[float], goal: list[float], grasped: bool
) -> dict[str, object]:
    return {
        "agent": {"qpos": np.zeros(9, dtype=np.float32)},
        "extra": {
            "tcp_pose": np.concatenate((np.asarray(tcp, float), np.ones(4))),
            "obj_pose": np.concatenate((np.asarray(obj, float), np.ones(4))),
            "goal_pos": np.asarray(goal, dtype=np.float32),
            "is_grasped": np.asarray(grasped),
        },
    }


def test_collect_visual_dagger_with_fake_env(monkeypatch: Any, tmp_path: Path) -> None:
    class FakePolicy:
        def reset(self) -> None:
            pass

        def act(self, observation: object, image: object) -> np.ndarray:
            return np.zeros(ACTION_DIM, dtype=np.float32)

    def fake_load(policy_dir: object, *, device: str, resolution: int, tasks: object) -> dict[str, object]:
        return {"PickCube-v1": FakePolicy()}

    class FakeEnv:
        def __init__(self) -> None:
            self.step_count = 0

        def reset(self, seed: int | None = None) -> tuple[dict[str, object], dict[str, object]]:
            return (
                _recovery_observation(
                    tcp=[0.0, 0.0, 0.1], obj=[0.0, 0.0, 0.1], goal=[0.1, 0.0, 0.1], grasped=False
                ),
                {},
            )

        def render(self) -> np.ndarray:
            return np.zeros((RES * 4, RES * 4, 3), dtype=np.uint8)

        def step(self, action: object) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
            self.step_count += 1
            if self.step_count >= 2:
                obs = _recovery_observation(
                    tcp=[0.05, 0.0, 0.1], obj=[0.05, 0.0, 0.1], goal=[0.1, 0.0, 0.1], grasped=True
                )
            else:
                obs = _recovery_observation(
                    tcp=[0.0, 0.0, 0.1], obj=[0.0, 0.0, 0.1], goal=[0.1, 0.0, 0.1], grasped=False
                )
            return obs, 0.0, False, False, {"success": self.step_count >= 6}

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "visual_dagger_collect.load_visual_act_checkpoints", fake_load
    )
    payload = collect_visual_dagger_episodes(
        tmp_path / "policy",
        seeds=[0, 1],
        resolution=RES,
        env_factory=lambda config: FakeEnv(),
        device="cpu",
        max_steps=10,
        takeover_zone=0.1,
        min_segment=2,
    )
    assert payload["schema_version"] == VISUAL_DAGGER_SCHEMA_VERSION
    assert payload["env_id"] == "PickCube-v1"
    assert payload["n_episodes"] >= 1
    assert payload["total_labeled_transitions"] >= 1
    assert payload["takeover_episodes"] >= 1
    assert payload["images"].shape[-1] == 3
    assert payload["robot_state"].shape[1] == ROBOT_STATE_DIM
    assert payload["env_state"].shape[1] == ENV_STATE_DIM
    assert payload["actions"].shape[1] == ACTION_DIM
