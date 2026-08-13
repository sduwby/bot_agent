"""第 21 周：世界模型支线定向测试。

覆盖：数据切分、DynamicsModel 前向、CEM 规划、MPC episode 与配置校验。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from behavior_cloning import ACTION_DIM, STATE_DIM
from world_model import (
    CEMPlanner,
    DynamicsDataset,
    DynamicsModel,
    WorldModelConfig,
    WorldModelError,
    run_world_model_episode,
    split_dynamics_data,
    evaluate_world_model,
)


def _dataset(n: int = 12, episodes: list[int] | None = None) -> DynamicsDataset:
    rng = np.random.default_rng(7)
    episode_ids = (
        np.asarray(episodes, dtype=np.int64)
        if episodes is not None
        else np.repeat(np.arange(3), n // 3)
    )
    count = len(episode_ids)
    return DynamicsDataset(
        states=rng.standard_normal((count, STATE_DIM)).astype(np.float32),
        actions=rng.standard_normal((count, ACTION_DIM)).astype(np.float32),
        next_states=rng.standard_normal((count, STATE_DIM)).astype(np.float32),
        episode_ids=episode_ids,
    )


class TestWorldModelConfig:
    def test_valid_defaults(self, tmp_path: Path) -> None:
        config = WorldModelConfig(
            dataset_dir=tmp_path / "data",
            repo_id="repo",
            output_dir=tmp_path / "out",
        )
        assert config.seeds == tuple(range(500, 510))

    def test_invalid_hidden_dims(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            WorldModelConfig(
                dataset_dir=tmp_path / "data",
                repo_id="repo",
                output_dir=tmp_path / "out",
                hidden_dims=(),
            )

    def test_invalid_cem_elite_ratio(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            WorldModelConfig(
                dataset_dir=tmp_path / "data",
                repo_id="repo",
                output_dir=tmp_path / "out",
                cem_elite_ratio=1.5,
            )

    def test_duplicate_seeds_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            WorldModelConfig(
                dataset_dir=tmp_path / "data",
                repo_id="repo",
                output_dir=tmp_path / "out",
                seeds=(500, 500),
            )


class TestSplitDynamicsData:
    def test_episode_boundaries_no_leak(self) -> None:
        data = _dataset(n=15, episodes=[0] * 5 + [1] * 5 + [2] * 5)
        train, validation = split_dynamics_data(
            data, train_episodes=1, validation_episodes=1
        )
        assert len(train.states) == 5
        assert len(validation.states) == 5
        assert set(int(item) for item in train.episode_ids) == {0}
        assert set(int(item) for item in validation.episode_ids) == {1}

    def test_not_enough_episodes_rejected(self) -> None:
        data = _dataset(n=5, episodes=[0] * 5)
        with pytest.raises(WorldModelError):
            split_dynamics_data(
                data, train_episodes=1, validation_episodes=1
            )

    def test_invalid_counts_rejected(self) -> None:
        data = _dataset()
        with pytest.raises(ValueError):
            split_dynamics_data(
                data, train_episodes=0, validation_episodes=1
            )


def _write_checkpoint(tmp_path: Path, *, layers: int = 2) -> Path:
    rng = np.random.default_rng(3)
    checkpoint = tmp_path / "dynamics_checkpoint"
    checkpoint.mkdir()
    arrays: dict[str, np.ndarray] = {}
    dims = [22, 8, STATE_DIM]
    for index in range(layers):
        arrays[f"layer_{index}_weight"] = (
            rng.standard_normal((dims[index + 1], dims[index])) * 0.1
        ).astype(np.float32)
        arrays[f"layer_{index}_bias"] = (
            rng.standard_normal((dims[index + 1],)) * 0.1
        ).astype(np.float32)
    np.savez(checkpoint / "weights.npz", **arrays)
    metadata = {
        "schema_version": "maniskill_world_model_v1",
        "hidden_dims": [8],
        "state_mean": np.zeros(STATE_DIM).tolist(),
        "state_std": np.ones(STATE_DIM).tolist(),
        "action_mean": np.zeros(ACTION_DIM).tolist(),
        "action_std": np.ones(ACTION_DIM).tolist(),
        "delta_mean": np.zeros(STATE_DIM).tolist(),
        "delta_std": np.ones(STATE_DIM).tolist(),
    }
    (checkpoint / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return checkpoint


class TestDynamicsModel:
    def test_load_and_predict_shapes(self, tmp_path: Path) -> None:
        checkpoint = _write_checkpoint(tmp_path)
        model = DynamicsModel(checkpoint)
        states = np.zeros((3, STATE_DIM), dtype=np.float32)
        actions = np.zeros((3, ACTION_DIM), dtype=np.float32)
        delta = model.predict_delta(states, actions)
        assert delta.shape == (3, STATE_DIM)
        assert np.all(np.isfinite(delta))
        next_states = model.predict_next(states, actions)
        np.testing.assert_allclose(next_states, states + delta, rtol=1e-5)

    def test_batch_mismatch_rejected(self, tmp_path: Path) -> None:
        checkpoint = _write_checkpoint(tmp_path)
        model = DynamicsModel(checkpoint)
        with pytest.raises(WorldModelError):
            model.predict_delta(
                np.zeros((3, STATE_DIM), dtype=np.float32),
                np.zeros((2, ACTION_DIM), dtype=np.float32),
            )

    def test_missing_checkpoint_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(WorldModelError):
            DynamicsModel(tmp_path / "nonexistent")

    def test_incomplete_checkpoint_rejected(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "dynamics_checkpoint"
        checkpoint.mkdir()
        np.savez(checkpoint / "weights.npz", layer_0_weight=np.zeros((2, 2)))
        (checkpoint / "metadata.json").write_text(
            json.dumps({"hidden_dims": [8]}), encoding="utf-8"
        )
        with pytest.raises(WorldModelError):
            DynamicsModel(checkpoint)


class TestCEMPlanner:
    def _planner(self, tmp_path: Path) -> CEMPlanner:
        checkpoint = _write_checkpoint(tmp_path)
        return CEMPlanner(
            DynamicsModel(checkpoint),
            population=64,
            horizon=4,
            elite_ratio=0.25,
            iterations=3,
        )

    def test_action_shape_and_bounds(self, tmp_path: Path) -> None:
        planner = self._planner(tmp_path)
        rng = np.random.default_rng(1)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = planner.plan(state, rng)
        assert action.shape == (ACTION_DIM,)
        assert np.all(np.isfinite(action))
        assert np.all(action[:3] >= -1.0) and np.all(action[:3] <= 1.0)
        assert action[3] == -1.0

    def test_deterministic_with_same_rng(self, tmp_path: Path) -> None:
        planner = self._planner(tmp_path)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        first = planner.plan(state, np.random.default_rng(5))
        second = planner.plan(state, np.random.default_rng(5))
        np.testing.assert_array_equal(first, second)

    def test_nonfinite_predictions_do_not_crash(self, tmp_path: Path) -> None:
        """权重含 NaN 时 planner 必须安全返回，不崩溃。"""
        checkpoint = _write_checkpoint(tmp_path)
        loaded = np.load(checkpoint / "weights.npz")
        arrays = dict(loaded)
        arrays["layer_1_weight"] = np.full_like(
            arrays["layer_1_weight"], np.nan
        )
        np.savez(checkpoint / "weights.npz", **arrays)
        planner = CEMPlanner(
            DynamicsModel(checkpoint),
            population=32,
            horizon=2,
            elite_ratio=0.5,
            iterations=2,
        )
        action = planner.plan(
            np.zeros(STATE_DIM, dtype=np.float32), np.random.default_rng(0)
        )
        assert action.shape == (ACTION_DIM,)
        assert np.all(np.isfinite(action))

    def test_invalid_parameters_rejected(self, tmp_path: Path) -> None:
        checkpoint = _write_checkpoint(tmp_path)
        model = DynamicsModel(checkpoint)
        with pytest.raises(ValueError):
            CEMPlanner(model, population=0, horizon=4, elite_ratio=0.1, iterations=3)
        with pytest.raises(ValueError):
            CEMPlanner(model, population=64, horizon=4, elite_ratio=0.0, iterations=3)


class _FakeEnv:
    """MPC episode 假环境。"""

    def __init__(self, *, success_after: int | None = None) -> None:
        self.steps = 0
        self._success_after = success_after
        self.action_space = type(
            "Space",
            (),
            {"shape": (ACTION_DIM,), "contains": lambda self, action: True},
        )()
        self._observation = {
            "agent": {"qpos": np.zeros(9, dtype=np.float32)},
            "extra": {
                "tcp_pose": np.zeros(7, dtype=np.float32),
                "obj_pose": np.zeros(7, dtype=np.float32),
                "goal_pos": np.zeros(3, dtype=np.float32),
                "is_grasped": np.asarray(False),
            },
        }

    def reset(self, seed: int | None = None):
        self.steps = 0
        return (self._observation, {})

    def step(self, action: object):
        self.steps += 1
        success = (
            self._success_after is not None and self.steps >= self._success_after
        )
        return (
            self._observation,
            0.0,
            success,
            False,
            {"success": success},
        )

    def close(self) -> None:
        pass


class TestRunWorldModelEpisode:
    def test_success_path(self, tmp_path: Path) -> None:
        from maniskill_intro import ManiSkillTaskConfig

        checkpoint = _write_checkpoint(tmp_path)
        planner = CEMPlanner(
            DynamicsModel(checkpoint),
            population=32,
            horizon=2,
            elite_ratio=0.5,
            iterations=2,
        )
        env = _FakeEnv(success_after=2)
        result = run_world_model_episode(
            ManiSkillTaskConfig(env_id="PushCube-v1", seed=500),
            env,
            planner,
            rng=np.random.default_rng(0),
        )
        assert result["success"] is True
        assert result["steps"] == 2

    def test_truncated_path(self, tmp_path: Path) -> None:
        from maniskill_intro import ManiSkillTaskConfig

        checkpoint = _write_checkpoint(tmp_path)
        planner = CEMPlanner(
            DynamicsModel(checkpoint),
            population=16,
            horizon=1,
            elite_ratio=0.5,
            iterations=1,
        )
        env = _FakeEnv()
        result = run_world_model_episode(
            ManiSkillTaskConfig(env_id="PushCube-v1", seed=500, max_steps=4),
            env,
            planner,
            rng=np.random.default_rng(0),
        )
        assert result["success"] is False
        assert result["steps"] == 4


class TestEvaluateWorldModel:
    def test_report_structure(self, tmp_path: Path) -> None:
        _write_checkpoint(tmp_path)
        from maniskill_intro import ManiSkillTaskConfig

        def env_factory(config: ManiSkillTaskConfig) -> _FakeEnv:
            return _FakeEnv(success_after=1)

        config = WorldModelConfig(
            dataset_dir=tmp_path,
            repo_id="repo",
            output_dir=tmp_path,
            seeds=(500, 501),
            cem_population=16,
            cem_horizon=1,
            cem_elite_ratio=0.5,
            cem_iterations=1,
        )
        report = evaluate_world_model(config, env_factory=env_factory)
        assert report["world_model"]["episodes"] == 2
        assert report["world_model"]["success_rate"] == 1.0
        assert report["scripted_baseline"]["episodes"] == 2
        report_path = tmp_path / "demo_report.json"
        assert report_path.is_file()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["schema_version"].startswith("maniskill_world_model")
        assert payload["world_model"]["episodes"] == 2

    def test_missing_checkpoint_rejected(self, tmp_path: Path) -> None:
        config = WorldModelConfig(
            dataset_dir=tmp_path,
            repo_id="repo",
            output_dir=tmp_path,
            seeds=(500,),
        )
        with pytest.raises(WorldModelError):
            evaluate_world_model(config, env_factory=lambda c: _FakeEnv())
