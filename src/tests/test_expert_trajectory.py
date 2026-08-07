"""第 11 周专家轨迹采集、检查与回放测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from expert_trajectory import (
    SCHEMA_VERSION,
    DatasetInspection,
    TrajectoryCollectionConfig,
    TrajectoryDataError,
    collect_expert_dataset,
    inspect_trajectory_dataset,
    replay_trajectory_dataset,
)


def make_observation() -> dict[str, object]:
    """构造 scripted runner 所需的最小 state_dict。"""
    return {
        "agent": {
            "qpos": np.zeros(9, dtype=np.float32),
            "qvel": np.zeros(9, dtype=np.float32),
        },
        "extra": {
            "tcp_pose": np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]),
            "obj_pose": np.array([0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]),
            "goal_pos": np.array([0.1, 0.0, 0.1]),
            "is_grasped": np.asarray(False),
        },
    }


class FakeActionSpace:
    shape = (4,)

    def contains(self, action: np.ndarray) -> bool:
        return action.shape == self.shape and bool(np.all(np.isfinite(action)))


class ZeroPolicy:
    phase = "collect"

    def __init__(self, env_id: str) -> None:
        self.env_id = env_id

    def reset(self) -> None:
        pass

    def act(self, observation: object) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)


class FakeRecordingEnv:
    """偶数 seed 一步成功、奇数 seed 一步失败的记录环境替身。"""

    action_space = FakeActionSpace()

    def __init__(self) -> None:
        self.seed = 0
        self.closed = False
        self.flushes: list[bool] = []

    def reset(self, *, seed: int) -> tuple[dict[str, object], dict[str, object]]:
        self.seed = seed
        return make_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        success = self.seed % 2 == 0
        return make_observation(), 1.0, True, False, {"success": success}

    def flush_trajectory(self, *, save: bool) -> None:
        self.flushes.append(save)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"episodes_per_task": 0}, "episodes_per_task"),
        ({"seed_start": -1}, "seed_start"),
        (
            {"episodes_per_task": 3, "max_attempts_per_task": 2},
            "max_attempts_per_task",
        ),
        ({"episodes_per_task": 1, "replay_count": 3}, "replay_count"),
    ],
)
def test_collection_config_rejects_invalid_values(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrajectoryCollectionConfig(output_dir=tmp_path / "data", **kwargs)


def test_collect_filters_failures_reuses_env_and_writes_summary(
    tmp_path: Path,
) -> None:
    environments: list[FakeRecordingEnv] = []
    output_path = (tmp_path / "data").resolve()

    def factory(config: object, output_dir: Path, name: str) -> FakeRecordingEnv:
        assert output_dir == output_path
        assert name in {"pick_cube", "push_cube"}
        env = FakeRecordingEnv()
        environments.append(env)
        return env

    def inspector(summary_path: Path) -> DatasetInspection:
        assert summary_path.is_file()
        return DatasetInspection(
            checked_episodes=4,
            checked_steps=4,
            task_counts={"PickCube-v1": 2, "PushCube-v1": 2},
        )

    summary = collect_expert_dataset(
        TrajectoryCollectionConfig(
            output_dir=tmp_path / "data",
            episodes_per_task=2,
            max_attempts_per_task=4,
            replay_count=2,
        ),
        env_factory=factory,
        policy_factory=ZeroPolicy,
        inspector=inspector,
    )

    assert summary.complete
    assert summary.total_successes == 4
    assert summary.total_failures == 2
    assert summary.validated_episodes == 4
    assert [item.attempts for item in summary.tasks] == [3, 3]
    assert all(env.flushes == [True, False, True] for env in environments)
    assert all(env.closed for env in environments)
    on_disk = json.loads((output_path / "summary.json").read_text())
    assert on_disk["schema_version"] == SCHEMA_VERSION
    assert on_disk["complete"] is True


def test_collect_marks_shortfall_incomplete(tmp_path: Path) -> None:
    summary = collect_expert_dataset(
        TrajectoryCollectionConfig(
            output_dir=tmp_path / "data",
            episodes_per_task=2,
            max_attempts_per_task=2,
            replay_count=1,
        ),
        env_factory=lambda config, output_dir, name: FakeRecordingEnv(),
        policy_factory=ZeroPolicy,
        inspector=lambda path: DatasetInspection(
            checked_episodes=2,
            checked_steps=2,
            task_counts={"PickCube-v1": 1, "PushCube-v1": 1},
        ),
    )

    assert not summary.complete
    assert summary.total_successes == 2
    assert summary.total_failures == 2


def test_collect_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "data"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="拒绝覆盖"):
        collect_expert_dataset(
            TrajectoryCollectionConfig(
                output_dir=output_dir,
                episodes_per_task=1,
                replay_count=1,
            )
        )

    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "user data"


def _write_dataset(
    root: Path,
    *,
    steps: int = 2,
    env_state_length: int | None = None,
) -> Path:
    h5py = pytest.importorskip("h5py")
    metadata = {
        "episodes": [
            {
                "episode_id": 0,
                "episode_seed": 7,
                "reset_kwargs": {"seed": 7},
                "control_mode": "pd_ee_delta_pos",
                "elapsed_steps": steps,
                "success": True,
            }
        ]
    }
    (root / "pick_cube.json").write_text(json.dumps(metadata), encoding="utf-8")
    with h5py.File(root / "pick_cube.h5", "w") as trajectories:
        group = trajectories.create_group("traj_0")
        group.create_dataset("actions", data=np.zeros((steps, 4), dtype=np.float32))
        group.create_dataset("rewards", data=np.ones(steps, dtype=np.float32))
        group.create_dataset("terminated", data=[False] * (steps - 1) + [True])
        group.create_dataset("truncated", data=np.zeros(steps, dtype=bool))
        group.create_dataset("success", data=[False] * (steps - 1) + [True])
        states = group.create_group("env_states")
        actors = states.create_group("actors")
        actors.create_dataset(
            "cube",
            data=np.zeros(
                (env_state_length if env_state_length is not None else steps + 1, 13),
                dtype=np.float32,
            ),
        )
        obs = group.create_group("obs")
        agent = obs.create_group("agent")
        agent.create_dataset("qpos", data=np.zeros((steps + 1, 9)))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "control_mode": "pd_ee_delta_pos",
        "total_successes": 1,
        "replay_summary_path": "replay_summary.json",
        "tasks": [
            {
                "env_id": "PickCube-v1",
                "successes": 1,
                "trajectory_path": "pick_cube.h5",
                "metadata_path": "pick_cube.json",
            }
        ],
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def test_inspect_trajectory_dataset_checks_t_plus_one_alignment(
    tmp_path: Path,
) -> None:
    summary_path = _write_dataset(tmp_path)

    inspection = inspect_trajectory_dataset(summary_path)

    assert inspection.checked_episodes == 1
    assert inspection.checked_steps == 2
    assert inspection.task_counts == {"PickCube-v1": 1}


def test_inspect_trajectory_dataset_rejects_misaligned_env_state(
    tmp_path: Path,
) -> None:
    summary_path = _write_dataset(tmp_path, env_state_length=2)

    with pytest.raises(TrajectoryDataError, match=r"T\+1"):
        inspect_trajectory_dataset(summary_path)


class FakeStateTarget:
    def __init__(self) -> None:
        self.states: list[object] = []

    def set_state_dict(self, state: object) -> None:
        self.states.append(state)


class FakeReplayEnv:
    def __init__(self) -> None:
        self.base_env = FakeStateTarget()
        self.step_count = 0
        self.closed = False
        self.reset_seeds: list[int] = []

    def reset(self, *, seed: int) -> tuple[dict[str, object], dict[str, object]]:
        self.step_count = 0
        self.reset_seeds.append(seed)
        return make_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        self.step_count += 1
        success = self.step_count == 2
        return make_observation(), 1.0, success, False, {"success": success}

    def close(self) -> None:
        self.closed = True


def test_replay_restores_initial_state_and_executes_actions(tmp_path: Path) -> None:
    summary_path = _write_dataset(tmp_path)
    environments: list[FakeReplayEnv] = []

    def factory(config: object) -> FakeReplayEnv:
        env = FakeReplayEnv()
        environments.append(env)
        return env

    replay = replay_trajectory_dataset(
        summary_path,
        replay_count=1,
        env_factory=factory,
    )

    assert replay.complete
    assert replay.successful == 1
    assert replay.episodes[0].steps == 2
    assert environments[0].reset_seeds == [7]
    assert len(environments[0].base_env.states) == 1
    assert environments[0].closed
    on_disk = json.loads((tmp_path / "replay_summary.json").read_text())
    assert on_disk["complete"] is True
