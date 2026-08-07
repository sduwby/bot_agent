"""第 12 周低维 Behavior Cloning 数据、训练和 rollout 测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from behavior_cloning import (  # noqa: E402
    ACTION_DIM,
    BCTrainingConfig,
    BehaviorCloningError,
    evaluate_bc_checkpoint,
    load_bc_checkpoint,
    load_bc_dataset,
    observation_to_state,
    split_task_transitions,
    train_behavior_cloning,
)
from expert_trajectory import SCHEMA_VERSION  # noqa: E402
from maniskill_intro import SUPPORTED_TASKS  # noqa: E402
from run_bc_training import main as training_main  # noqa: E402


def _state_frame(step: int, episode_id: int) -> tuple[np.ndarray, ...]:
    qpos = np.arange(9, dtype=np.float32) + episode_id + step * 0.1
    qvel = np.arange(9, dtype=np.float32) * 0.01 + step * 0.02
    cube = np.zeros(13, dtype=np.float32)
    cube[:7] = np.arange(7, dtype=np.float32) + episode_id + step * 0.03
    goal = np.zeros(13, dtype=np.float32)
    goal[:3] = np.asarray([0.2, -0.1, 0.3], dtype=np.float32) + episode_id
    panda = np.zeros(31, dtype=np.float32)
    panda[13:22] = qpos
    panda[22:31] = qvel
    return panda, cube, goal


def _write_task_dataset(
    root: Path,
    *,
    env_id: str,
    stem: str,
    goal_actor: str,
    episodes: int = 3,
    steps: int = 2,
) -> dict[str, object]:
    metadata_episodes = []
    with h5py.File(root / f"{stem}.h5", "w") as trajectories:
        for episode_id in range(episodes):
            group = trajectories.create_group(f"traj_{episode_id}")
            actions = np.stack(
                [
                    np.asarray(
                        [0.1 * step, -0.2, 0.3, 1.0], dtype=np.float32
                    )
                    for step in range(steps)
                ]
            )
            group.create_dataset("actions", data=actions)
            states = group.create_group("env_states")
            articulations = states.create_group("articulations")
            actors = states.create_group("actors")
            pandas = []
            cubes = []
            goals = []
            for step in range(steps + 1):
                panda, cube, goal = _state_frame(step, episode_id)
                pandas.append(panda)
                cubes.append(cube)
                goals.append(goal)
            articulations.create_dataset("panda", data=np.stack(pandas))
            actors.create_dataset("cube", data=np.stack(cubes))
            actors.create_dataset(goal_actor, data=np.stack(goals))

            obs = group.create_group("obs")
            agent = obs.create_group("agent")
            extra = obs.create_group("extra")
            panda_array = np.stack(pandas)
            cube_array = np.stack(cubes)
            goal_array = np.stack(goals)
            agent.create_dataset("qpos", data=panda_array[:, 13])
            agent.create_dataset("qvel", data=panda_array[:, 22])
            extra.create_dataset("obj_pose", data=cube_array[:, 0])
            extra.create_dataset("goal_pos", data=goal_array[:, 0])
            metadata_episodes.append(
                {
                    "episode_id": episode_id,
                    "episode_seed": episode_id,
                    "elapsed_steps": steps,
                    "success": True,
                }
            )
    (root / f"{stem}.json").write_text(
        json.dumps({"episodes": metadata_episodes}), encoding="utf-8"
    )
    return {
        "env_id": env_id,
        "successes": episodes,
        "trajectory_path": f"{stem}.h5",
        "metadata_path": f"{stem}.json",
    }


def _write_dataset(root: Path, *, episodes: int = 3) -> Path:
    tasks = [
        _write_task_dataset(
            root,
            env_id="PickCube-v1",
            stem="pick_cube",
            goal_actor="goal_site",
            episodes=episodes,
        ),
        _write_task_dataset(
            root,
            env_id="PushCube-v1",
            stem="push_cube",
            goal_actor="goal_region",
            episodes=episodes,
        ),
    ]
    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )
    return summary_path


class FakeObservationBaseEnv:
    def __init__(self) -> None:
        self.state: dict[str, object] | None = None

    def set_state_dict(self, state: dict[str, object]) -> None:
        self.state = state

    def get_obs(self) -> dict[str, object]:
        assert self.state is not None
        panda = self.state["articulations"]["panda"]
        actors = self.state["actors"]
        goal_key = "goal_site" if "goal_site" in actors else "goal_region"
        qpos = panda[13:22]
        qvel = panda[22:31]
        return {
            "agent": {"qpos": qpos[None], "qvel": qvel[None]},
            "extra": {
                "tcp_pose": np.concatenate((qpos[:3], np.ones(4)))[None],
                "obj_pose": actors["cube"][:7][None],
                "goal_pos": actors[goal_key][:3][None],
                "is_grasped": np.asarray([False]),
            },
        }


class FakeObservationEnv:
    def __init__(self) -> None:
        self.base_env = FakeObservationBaseEnv()
        self.closed = False

    def reset(self, *, seed: int) -> tuple[dict[str, object], dict[str, object]]:
        return _observation(), {}

    def close(self) -> None:
        self.closed = True


def _observation_factory(config: object) -> FakeObservationEnv:
    return FakeObservationEnv()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"train_episodes_per_task": 0}, "train_episodes_per_task"),
        ({"validation_episodes_per_task": 0}, "validation"),
        ({"hidden_dims": ()}, "hidden_dims"),
        ({"learning_rate": 0}, "learning_rate"),
    ],
)
def test_training_config_rejects_invalid_values(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BCTrainingConfig(
            dataset_summary=tmp_path / "summary.json",
            output_dir=tmp_path / "output",
            **kwargs,
        )


def test_load_dataset_maps_full_env_state_and_splits_by_episode(
    tmp_path: Path,
) -> None:
    dataset = load_bc_dataset(
        _write_dataset(tmp_path),
        observation_env_factory=_observation_factory,
    )
    pick = dataset["PickCube-v1"]

    assert pick.states.shape == (6, 18)
    assert pick.actions.shape == (6, ACTION_DIM)
    assert pick.episode_ids == (0, 1, 2)
    np.testing.assert_allclose(pick.states[0, :3], np.arange(3))
    np.testing.assert_allclose(pick.states[0, 3:6], np.arange(3))

    train, validation = split_task_transitions(
        pick, train_episodes=1, validation_episodes=2
    )
    np.testing.assert_array_equal(train, [0, 1])
    np.testing.assert_array_equal(validation, [2, 3, 4, 5])
    assert not np.intersect1d(train, validation).size


def test_load_dataset_rejects_changed_env_state_mapping(tmp_path: Path) -> None:
    summary_path = _write_dataset(tmp_path)
    with h5py.File(tmp_path / "pick_cube.h5", "r+") as trajectories:
        trajectories["traj_0/obs/agent/qpos"][0] = 999

    with pytest.raises(BehaviorCloningError, match="映射不一致"):
        load_bc_dataset(
            summary_path,
            observation_env_factory=_observation_factory,
        )


def _observation() -> dict[str, object]:
    return {
        "agent": {
            "qpos": np.arange(9, dtype=np.float32),
            "qvel": np.arange(9, dtype=np.float32) * 0.1,
        },
        "extra": {
            "tcp_pose": np.arange(7, dtype=np.float32),
            "obj_pose": np.arange(7, dtype=np.float32),
            "goal_pos": np.arange(3, dtype=np.float32),
            "is_grasped": np.asarray(False),
        },
    }


def test_observation_to_state_requires_exact_live_shapes() -> None:
    state = observation_to_state(_observation())
    assert state.shape == (18,)

    invalid = _observation()
    invalid["agent"]["qpos"] = np.zeros(8)
    with pytest.raises(BehaviorCloningError, match="agent.qpos"):
        observation_to_state(invalid)


class FakeActionSpace:
    shape = (4,)

    def contains(self, action: np.ndarray) -> bool:
        return action.shape == self.shape and bool(np.all(np.abs(action) <= 1))


class FakeEvaluationEnv:
    action_space = FakeActionSpace()

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.closed = False

    def reset(self, *, seed: int) -> tuple[dict[str, object], dict[str, object]]:
        assert seed == self.seed
        return _observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        return _observation(), 1.0, True, False, {"success": True}

    def close(self) -> None:
        self.closed = True


def test_train_reload_and_evaluate_checkpoint(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    summary_path = _write_dataset(dataset_dir)
    output_dir = tmp_path / "output"
    training = train_behavior_cloning(
        BCTrainingConfig(
            dataset_summary=summary_path,
            output_dir=output_dir,
            train_episodes_per_task=1,
            validation_episodes_per_task=1,
            hidden_dims=(8,),
            epochs=2,
            batch_size=2,
            random_seed=7,
        ),
        observation_env_factory=_observation_factory,
    )

    assert training.complete
    assert Path(training.checkpoint_path).is_file()
    assert set(load_bc_checkpoint(Path(training.checkpoint_path))) == set(
        SUPPORTED_TASKS
    )

    environments: list[FakeEvaluationEnv] = []

    def factory(config: object) -> FakeEvaluationEnv:
        env = FakeEvaluationEnv(config.seed)
        environments.append(env)
        return env

    evaluation = evaluate_bc_checkpoint(
        Path(training.checkpoint_path),
        seeds=[100, 101],
        env_factory=factory,
    )

    assert evaluation.passed
    assert evaluation.total_successes == 4
    assert all(env.closed for env in environments)
    assert (output_dir / "evaluation.json").is_file()


def test_training_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    summary_path = _write_dataset(dataset_dir)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="拒绝覆盖"):
        train_behavior_cloning(
            BCTrainingConfig(
                dataset_summary=summary_path,
                output_dir=output_dir,
                train_episodes_per_task=1,
                validation_episodes_per_task=1,
                epochs=1,
            ),
            observation_env_factory=_observation_factory,
        )
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_cli_rejects_zero_evaluation_episodes(capsys: object) -> None:
    assert training_main(["--eval-episodes-per-task", "0"]) == 1
    assert "eval_episodes_per_task" in capsys.readouterr().err
