"""第 13 周 LeRobot 低维 ACT 数据、训练、checkpoint 与评测测试。"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from behavior_cloning import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    TaskTransitions,
    load_bc_dataset,
)
from expert_trajectory import SCHEMA_VERSION  # noqa: E402
import lerobot_act as act  # noqa: E402
from lerobot_act import (  # noqa: E402
    ACT_SCHEMA_VERSION,
    ACTTrainingConfig,
    ENV_STATE_DIM,
    LeRobotACTChunkDataset,
    LeRobotACTError,
    LeRobotACTPolicy,
    ROBOT_STATE_DIM,
    ROBOT_STATE_V2_DIM,
    build_action_chunk,
    build_chunk_dataset,
    combine_lerobot_act_state,
    compare_act_bc_scripted,
    compute_act_dataset_stats,
    compute_chunk_dataset_stats,
    evaluate_lerobot_act,
    evaluate_policy_seeds,
    load_lerobot_act_checkpoints,
    make_lerobot_act_config,
    select_best_rollout_checkpoint,
    split_lerobot_act_state,
    split_lerobot_act_state_v2,
    train_lerobot_act,
)
from maniskill_intro import SUPPORTED_TASKS  # noqa: E402
from run_act_training import main as act_training_main  # noqa: E402


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


def _tiny_config(
    summary: Path, output: Path, *, training_steps: int = 3,
    checkpoint_interval: int | None = None,
    tasks: tuple[str, ...] = SUPPORTED_TASKS,
) -> ACTTrainingConfig:
    return ACTTrainingConfig(
        dataset_summary=summary,
        output_dir=output,
        train_episodes_per_task=1,
        validation_episodes_per_task=1,
        chunk_size=4,
        n_action_steps=2,
        training_steps=training_steps,
        batch_size=4,
        validation_interval=2,
        checkpoint_interval=checkpoint_interval,
        tasks=tasks,
        random_seed=7,
        device="cpu",
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
    )


# ---------------------------------------------------------------------------
# 状态拆分与 action chunk
# ---------------------------------------------------------------------------


def test_split_state_lossless_roundtrip() -> None:
    rng = np.random.default_rng(0)
    state = rng.standard_normal(STATE_DIM).astype(np.float32)
    robot, env = split_lerobot_act_state(state)
    assert robot.shape == (ROBOT_STATE_DIM,)
    assert env.shape == (ENV_STATE_DIM,)
    np.testing.assert_array_equal(
        combine_lerobot_act_state(robot, env), state
    )
    np.testing.assert_array_equal(robot, state[[0, 1, 2, 15, 16]])
    np.testing.assert_array_equal(
        env, state[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]]
    )


def test_split_state_rejects_bad_input() -> None:
    with pytest.raises(LeRobotACTError, match="shape"):
        split_lerobot_act_state(np.zeros(17, dtype=np.float32))
    bad = np.ones(STATE_DIM, dtype=np.float32)
    bad[0] = np.nan
    with pytest.raises(LeRobotACTError, match="非有限"):
        split_lerobot_act_state(bad)


def test_split_state_v2_qvel() -> None:
    rng = np.random.default_rng(0)
    state = rng.standard_normal(STATE_DIM).astype(np.float32)
    qvel = rng.standard_normal(7).astype(np.float32)
    robot, env = split_lerobot_act_state_v2(state, qvel)
    assert robot.shape == (ROBOT_STATE_V2_DIM,)
    assert env.shape == (ENV_STATE_DIM,)
    np.testing.assert_array_equal(robot[:5], state[[0, 1, 2, 15, 16]])
    np.testing.assert_array_equal(robot[5:], qvel)
    with pytest.raises(LeRobotACTError, match="arm_qvel"):
        split_lerobot_act_state_v2(state, np.zeros(6, dtype=np.float32))
    bad_qvel = qvel.copy()
    bad_qvel[0] = np.inf
    with pytest.raises(LeRobotACTError, match="非有限"):
        split_lerobot_act_state_v2(state, bad_qvel)


def test_build_and_stats_with_qvel() -> None:
    data = _transitions(lengths=(6,), with_arm_qvel=True)
    indices = [0, 1, 2, 3]
    dataset = build_chunk_dataset(data, indices, chunk_size=4, use_qvel=True)
    assert dataset.robot_state.shape == (4, ROBOT_STATE_V2_DIM)
    assert dataset.env_state.shape == (4, ENV_STATE_DIM)
    stats = compute_act_dataset_stats(data, indices, use_qvel=True)
    assert stats["observation.state"]["mean"].shape == (
        ROBOT_STATE_V2_DIM,
    )
    assert stats["observation.environment_state"]["mean"].shape == (
        ENV_STATE_DIM,
    )
    # 缺 arm_qvel 时必须明确失败
    missing = _transitions(lengths=(6,), with_arm_qvel=False)
    with pytest.raises(LeRobotACTError, match="arm_qvel"):
        build_chunk_dataset(missing, indices, chunk_size=4, use_qvel=True)


def test_build_action_chunk_full_and_padding() -> None:
    actions = np.arange(24, dtype=np.float32).reshape(6, 4)
    chunk, mask = build_action_chunk(actions, 1, chunk_size=8)
    assert chunk.shape == (8, ACTION_DIM)
    np.testing.assert_array_equal(chunk[:5], actions[1:6])
    np.testing.assert_array_equal(chunk[5:], 0)
    np.testing.assert_array_equal(
        mask, np.array([False] * 5 + [True] * 3, dtype=bool)
    )

    single = np.full((1, ACTION_DIM), 0.5, dtype=np.float32)
    chunk, mask = build_action_chunk(single, 0, chunk_size=8)
    np.testing.assert_array_equal(chunk[0], [0.5] * ACTION_DIM)
    assert not mask[0]
    assert mask[1:].all()


def test_build_action_chunk_rejects_invalid() -> None:
    actions = np.ones((4, ACTION_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="超出"):
        build_action_chunk(actions, 4, 8)
    bad = actions.copy()
    bad[0, 0] = np.nan
    with pytest.raises(LeRobotACTError, match="非有限"):
        build_action_chunk(bad, 0, 8)
    with pytest.raises(ValueError, match="chunk_size"):
        build_action_chunk(actions, 0, 0)
    with pytest.raises(LeRobotACTError, match="shape"):
        build_action_chunk(np.zeros((3, 3), dtype=np.float32), 0, 8)


def _transitions(
    *,
    lengths: tuple[int, ...],
    env_id: str = "PickCube-v1",
    with_arm_qvel: bool = False,
) -> TaskTransitions:
    total = sum(lengths)
    rng = np.random.default_rng(1)
    return TaskTransitions(
        env_id=env_id,
        episode_ids=tuple(range(len(lengths))),
        episode_lengths=lengths,
        states=rng.standard_normal((total, STATE_DIM)).astype(np.float32),
        actions=rng.standard_normal((total, ACTION_DIM)).astype(np.float32),
        arm_qvel=(
            rng.standard_normal((total, 7)).astype(np.float32)
            if with_arm_qvel
            else None
        ),
    )


def _reference_stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(array.mean(axis=0), dtype=np.float32)
    std = np.asarray(array.std(axis=0), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def test_chunk_dataset_no_episode_crossing() -> None:
    data = _transitions(lengths=(3, 2))
    dataset = build_chunk_dataset(data, [0, 1, 3], chunk_size=8)
    # transition 0: 起点在 episode 0 首帧，chunk 取 actions[0:3]。
    np.testing.assert_array_equal(dataset.action_chunk[0][:3], data.actions[0:3])
    np.testing.assert_array_equal(dataset.action_chunk[0][3:], 0)
    # transition 1: 起点在 episode 0 第二帧，chunk 取 actions[1:3]，不跨 episode。
    np.testing.assert_array_equal(dataset.action_chunk[1][:2], data.actions[1:3])
    np.testing.assert_array_equal(dataset.action_chunk[1][2:], 0)
    # transition 3: episode 1 首帧，chunk 取 actions[3:5]。
    np.testing.assert_array_equal(dataset.action_chunk[2][:2], data.actions[3:5])
    assert dataset.action_chunk[2][2:].sum() == 0.0
    assert dataset.action_is_pad.dtype == bool
    assert dataset.chunk_size == 8


def test_chunk_dataset_rejects_invalid() -> None:
    bad_actions = np.ones((4, ACTION_DIM), dtype=np.float32)
    bad_actions[1, 0] = np.inf
    data = TaskTransitions(
        env_id="PickCube-v1",
        episode_ids=(0,),
        episode_lengths=(4,),
        states=np.zeros((4, STATE_DIM), dtype=np.float32),
        actions=bad_actions,
    )
    with pytest.raises(LeRobotACTError, match="非有限"):
        build_chunk_dataset(data, [0], chunk_size=8)
    with pytest.raises(ValueError, match="chunk_size"):
        build_chunk_dataset(data, [0], chunk_size=0)


def test_compute_stats_ignores_padding_and_uses_train_only() -> None:
    dataset = LeRobotACTChunkDataset(
        env_id="PickCube-v1",
        robot_state=np.zeros((3, ROBOT_STATE_DIM), dtype=np.float32)
        + np.arange(ROBOT_STATE_DIM, dtype=np.float32),
        env_state=np.zeros((3, ENV_STATE_DIM), dtype=np.float32),
        action_chunk=np.stack(
            [
                np.full((8, ACTION_DIM), 2.0, dtype=np.float32),
                np.full((8, ACTION_DIM), 4.0, dtype=np.float32),
                np.zeros((8, ACTION_DIM), dtype=np.float32),
            ]
        ),
        action_is_pad=np.array(
            [[False] * 8, [False] * 8, [True] * 8], dtype=bool
        ),
        chunk_size=8,
    )
    stats = compute_chunk_dataset_stats(dataset)
    np.testing.assert_allclose(stats["action"]["mean"], 3.0)
    np.testing.assert_allclose(stats["action"]["std"], 1.0)
    assert stats["observation.state"]["mean"].shape == (ROBOT_STATE_DIM,)
    assert stats["observation.environment_state"]["std"].shape == (
        ENV_STATE_DIM,
    )


def test_compute_act_dataset_stats_unique_transitions() -> None:
    data = _transitions(lengths=(12,))
    train_indices = list(range(12))
    stats = compute_act_dataset_stats(data, train_indices)
    # action mean/std 必须等于训练集唯一 transition 的直接统计。
    expected_mean, expected_std = _reference_stats(data.actions[train_indices])
    np.testing.assert_allclose(stats["action"]["mean"], expected_mean)
    np.testing.assert_allclose(stats["action"]["std"], expected_std)
    # robot/env 由每个 transition 拆分一次得到，样本数等于 transition 数。
    assert stats["observation.state"]["mean"].shape == (ROBOT_STATE_DIM,)
    assert stats["observation.environment_state"]["mean"].shape == (
        ENV_STATE_DIM,
    )
    # 旧 chunk 展开统计随 chunk size 改变，新接口与 chunk size 无关。
    chunk4 = compute_chunk_dataset_stats(
        build_chunk_dataset(data, train_indices, chunk_size=4)
    )
    chunk8 = compute_chunk_dataset_stats(
        build_chunk_dataset(data, train_indices, chunk_size=8)
    )
    assert not np.allclose(
        chunk4["action"]["mean"], chunk8["action"]["mean"]
    )


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_episodes_per_task": 0},
        {"chunk_size": 0},
        {"n_action_steps": 0},
        {"training_steps": 0},
        {"batch_size": 0},
        {"n_action_steps": 9, "chunk_size": 8},
        {"random_seed": -1},
        {"dim_model": 0},
    ],
)
def test_training_config_rejects_invalid(tmp_path: Path, kwargs: Any) -> None:
    with pytest.raises(ValueError):
        ACTTrainingConfig(
            dataset_summary=tmp_path / "summary.json",
            output_dir=tmp_path / "output",
            **kwargs,
        )


def test_make_act_config_feature_shapes() -> None:
    config = make_lerobot_act_config(
        chunk_size=8, n_action_steps=4, device="cpu"
    )
    assert config.robot_state_feature.shape == (ROBOT_STATE_DIM,)
    assert config.env_state_feature.shape == (ENV_STATE_DIM,)
    assert config.action_feature.shape == (ACTION_DIM,)
    assert config.chunk_size == 8
    assert config.n_action_steps == 4
    assert config.use_vae is True
    assert config.temporal_ensemble_coeff is None
    with pytest.raises(ValueError):
        make_lerobot_act_config(chunk_size=8, n_action_steps=9, device="cpu")


# ---------------------------------------------------------------------------
# 训练、checkpoint
# ---------------------------------------------------------------------------


def test_train_reload_checkpoint_with_fake_env(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    summary = _write_dataset(dataset_dir)
    output = tmp_path / "output"
    config = _tiny_config(summary, output)
    training = train_lerobot_act(
        config, observation_env_factory=_observation_factory
    )
    assert training.complete
    assert len(training.tasks) == 2
    for env_id in SUPPORTED_TASKS:
        best_dir = output / env_id / "best-validation"
        assert (best_dir / "config.json").is_file()
        assert (best_dir / "model.safetensors").is_file()
        assert (best_dir / "policy_preprocessor.json").is_file()
        assert (best_dir / "policy_postprocessor.json").is_file()
        assert (output / env_id / "task_metadata.json").is_file()
        assert (output / env_id / "metrics.jsonl").is_file()
    assert (output / "training_summary.json").is_file()
    policies = load_lerobot_act_checkpoints(output, device="cpu")
    assert set(policies) == set(SUPPORTED_TASKS)


def test_training_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    summary = _write_dataset(dataset_dir)
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        train_lerobot_act(
            _tiny_config(summary, output),
            observation_env_factory=_observation_factory,
        )
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_checkpoint_missing_files_fail(tmp_path: Path) -> None:
    output = tmp_path / "output"
    (output / "PickCube-v1" / "best").mkdir(parents=True)
    (output / "PushCube-v1" / "best").mkdir(parents=True)
    with pytest.raises(LeRobotACTError, match="config.json"):
        load_lerobot_act_checkpoints(output, device="cpu")


def test_checkpoint_missing_processor_state_fail(tmp_path: Path) -> None:
    output = tmp_path / "output"
    best = output / "PickCube-v1" / "best"
    best.mkdir(parents=True)
    (output / "PushCube-v1" / "best").mkdir(parents=True)
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (best / name).write_text("{}" if name.endswith(".json") else "x")
    pre = best / "policy_preprocessor.json"
    pre.write_text(
        json.dumps({"steps": [{"state_file": "missing.safetensors"}]}),
        encoding="utf-8",
    )
    with pytest.raises(LeRobotACTError, match="processor state"):
        load_lerobot_act_checkpoints(output, device="cpu")


def test_load_checkpoints_with_n_action_steps_override(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    summary = _write_dataset(dataset_dir)
    output = tmp_path / "output"
    config = _tiny_config(summary, output, training_steps=2)
    train_lerobot_act(
        config, observation_env_factory=_observation_factory
    )
    policies = load_lerobot_act_checkpoints(
        output, device="cpu", n_action_steps=1
    )
    for env_id in SUPPORTED_TASKS:
        assert policies[env_id]._policy.config.n_action_steps == 1
    with pytest.raises(LeRobotACTError, match="n_action_steps"):
        load_lerobot_act_checkpoints(
            output, device="cpu", n_action_steps=99
        )


def test_single_task_training_only_creates_task_dir(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    summary = _write_dataset(dataset_dir)
    output = tmp_path / "output"
    config = _tiny_config(
        summary, output, training_steps=2, tasks=("PickCube-v1",)
    )
    training = train_lerobot_act(
        config, observation_env_factory=_observation_factory
    )
    assert len(training.tasks) == 1
    assert training.tasks_selected == ("PickCube-v1",)
    assert (output / "PickCube-v1" / "best-validation").is_dir()
    assert not (output / "PushCube-v1").exists()
    policies = load_lerobot_act_checkpoints(
        output, device="cpu", tasks=("PickCube-v1",)
    )
    assert set(policies) == {"PickCube-v1"}


def test_periodic_checkpoints_saved(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    summary = _write_dataset(dataset_dir)
    output = tmp_path / "output"
    config = _tiny_config(
        summary, output, training_steps=4, checkpoint_interval=2
    )
    train_lerobot_act(
        config, observation_env_factory=_observation_factory
    )
    task_dir = output / "PickCube-v1"
    steps = sorted(
        d.name for d in task_dir.glob("step-*") if d.is_dir()
    )
    assert steps == ["step-00002", "step-00004"]
    assert (task_dir / "best-validation" / "config.json").is_file()


def test_rollout_selection_with_fake_env(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    summary = _write_dataset(dataset_dir)
    output = tmp_path / "output"
    config = _tiny_config(
        summary, output, training_steps=4, checkpoint_interval=2
    )
    train_lerobot_act(
        config, observation_env_factory=_observation_factory
    )
    environments: list[FakeEvaluationEnv] = []

    def factory(cfg: object) -> FakeEvaluationEnv:
        env = FakeEvaluationEnv(cfg.seed)
        environments.append(env)
        return env

    selected = select_best_rollout_checkpoint(
        output,
        env_id="PickCube-v1",
        dev_seeds=[150, 151],
        device="cpu",
        env_factory=factory,
    )
    assert selected.name == "best-rollout-dev"
    assert (selected / "config.json").is_file()
    assert (output / "PickCube-v1" / "rollout_selection.json").is_file()
    assert all(env.closed for env in environments)


def test_wrapper_tracks_chunk_offset() -> None:
    policy = FakeLeRobotPolicy(n_action_steps=4)
    wrapper = LeRobotACTPolicy(
        env_id="PickCube-v1",
        policy=policy,
        preprocessor=FakePre(),
        postprocessor=FakePost(),
    )
    obs = _observation()
    offsets = []
    for _ in range(4):
        wrapper.act(obs)
        offsets.append(wrapper.chunk_offset)
    assert offsets == [0, 1, 2, 3]
    assert wrapper.chunk_query_id == 1
    wrapper.act(obs)
    assert wrapper.chunk_query_id == 2
    assert wrapper.chunk_offset == 0
    wrapper.reset()
    assert wrapper.chunk_query_id == 0
    assert wrapper.chunk_offset == -1


def test_config_rejects_duplicate_or_unknown_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="重复"):
        ACTTrainingConfig(
            dataset_summary=tmp_path / "s.json",
            output_dir=tmp_path / "o",
            tasks=("PickCube-v1", "PickCube-v1"),
        )
    with pytest.raises(ValueError, match="不支持"):
        ACTTrainingConfig(
            dataset_summary=tmp_path / "s.json",
            output_dir=tmp_path / "o",
            tasks=("Nope-v1",),
        )
    with pytest.raises(ValueError, match="checkpoint_interval"):
        ACTTrainingConfig(
            dataset_summary=tmp_path / "s.json",
            output_dir=tmp_path / "o",
            checkpoint_interval=0,
        )


def _recovery_observation(
    *, tcp: np.ndarray, obj: np.ndarray, goal: np.ndarray, grasped: bool
) -> dict[str, object]:
    return {
        "agent": {"qpos": np.zeros(9, dtype=np.float32)},
        "extra": {
            "tcp_pose": np.concatenate((tcp, np.ones(4))),
            "obj_pose": np.concatenate((obj, np.ones(4))),
            "goal_pos": goal,
            "is_grasped": np.asarray(grasped),
        },
    }


def test_recovery_expert_labels_grasped_near_goal() -> None:
    from dagger_collect import RecoveryExpert, SETTLE_DIST

    expert = RecoveryExpert()
    tcp = np.array([0.0, 0.0, 0.1], dtype=float)
    goal = np.array([0.1, 0.0, 0.1], dtype=float)
    obj = np.array([0.05, 0.0, 0.1], dtype=float)
    # 未抓取 → None
    assert (
        expert.act(
            _recovery_observation(tcp=tcp, obj=obj, goal=goal, grasped=False)
        )
        is None
    )
    # 抓取且远离 goal（> SETTLE_DIST）→ 向 goal 移动、闭爪
    action = expert.act(
        _recovery_observation(tcp=tcp, obj=obj, goal=goal, grasped=True)
    )
    assert action is not None and action.shape == (4,)
    assert np.all(np.abs(action[:3]) <= 1.0)
    assert action[3] == -1.0
    # 抓取且已到位（< SETTLE_DIST）→ 零增量保持
    obj_at_goal = goal.copy()
    settled = expert.act(
        _recovery_observation(tcp=tcp, obj=obj_at_goal, goal=goal, grasped=True)
    )
    np.testing.assert_allclose(settled[:3], 0.0, atol=1e-6)
    assert settled[3] == -1.0


def test_build_dagger_transitions_merge() -> None:
    from dagger_collect import build_dagger_transitions

    base = _transitions(lengths=(2, 2, 2, 2))  # 4 episodes：2 train + 2 val
    dagger = {
        "env_id": "PickCube-v1",
        "n_episodes": 2,
        "episode_lengths": [2, 3],
        "states": np.zeros((5, STATE_DIM), dtype=np.float32),
        "arm_qvel": np.zeros((5, 7), dtype=np.float32),
        "actions": np.zeros((5, ACTION_DIM), dtype=np.float32),
    }
    merged = build_dagger_transitions(base, dagger, start_episode_id=100, insert_after=2)
    # 顺序：原始前 2 + dagger 2 + 原始后 2（validation 保持原始）
    assert merged.episode_ids == (0, 1, 100, 101, 2, 3)
    assert merged.episode_lengths == (2, 2, 2, 3, 2, 2)
    assert merged.states.shape == (13, STATE_DIM)
    # train = 原始前 2 + dagger（episode 0-3），validation = 原始后 2（episode 4-5）
    train, validation = act.split_task_transitions(
        merged, train_episodes=4, validation_episodes=2
    )
    # validation 只含原始后 2 个 episode 的 transition（state 索引 9-12）
    np.testing.assert_array_equal(validation, [9, 10, 11, 12])


# ---------------------------------------------------------------------------
# wrapper 与 rollout
# ---------------------------------------------------------------------------


class FakePre:
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch


class FakePost:
    def __call__(self, action: Any) -> Any:
        return action


class FakeLeRobotPolicy:
    """模拟 LeRobot ACT 的 action queue 消费/重填充语义。"""

    def __init__(self, n_action_steps: int = 4) -> None:
        self._action_queue: deque[Any] = deque(maxlen=n_action_steps)
        self._n_action_steps = n_action_steps
        self.queries = 0

    def reset(self) -> None:
        self._action_queue.clear()

    def select_action(self, batch: dict[str, Any]) -> Any:
        import torch

        if len(self._action_queue) == 0:
            self.queries += 1
            for _ in range(self._n_action_steps):
                self._action_queue.append(
                    torch.full((1, ACTION_DIM), 0.25, dtype=torch.float32)
                )
        return self._action_queue.popleft()


def test_act_wrapper_queue_consumes_then_requeries() -> None:
    policy = FakeLeRobotPolicy(n_action_steps=4)
    wrapper = LeRobotACTPolicy(
        env_id="PickCube-v1",
        policy=policy,
        preprocessor=FakePre(),
        postprocessor=FakePost(),
    )
    observation = _observation()
    for _ in range(4):
        action = wrapper.act(observation)
        assert action.shape == (ACTION_DIM,)
        assert np.all(np.isfinite(action))
    assert policy.queries == 1
    assert wrapper.chunk_queries == 1
    assert wrapper.action_queries == 4

    wrapper.act(observation)
    assert policy.queries == 2
    assert wrapper.chunk_queries == 2

    wrapper.reset()
    assert len(policy._action_queue) == 0
    wrapper.act(observation)
    assert policy.queries == 3


class BadPost:
    def __call__(self, action: Any) -> Any:
        import torch

        return torch.full((1, ACTION_DIM), float("nan"))


def test_act_wrapper_rejects_non_finite_action() -> None:
    wrapper = LeRobotACTPolicy(
        env_id="PickCube-v1",
        policy=FakeLeRobotPolicy(n_action_steps=2),
        preprocessor=FakePre(),
        postprocessor=BadPost(),
    )
    with pytest.raises(LeRobotACTError, match="非有限"):
        wrapper.act(_observation())
    assert wrapper.nonfinite_actions == 1


def test_act_wrapper_action_smoothing_ema() -> None:
    import torch

    sequence = [
        torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
    ]

    class SeqPost:
        def __call__(self, action: Any) -> Any:
            return sequence.pop(0)

    wrapper = LeRobotACTPolicy(
        env_id="PickCube-v1",
        policy=FakeLeRobotPolicy(n_action_steps=1),
        preprocessor=FakePre(),
        postprocessor=SeqPost(),
        action_smoothing=0.5,
    )
    first = wrapper.act(_observation())
    second = wrapper.act(_observation())
    third = wrapper.act(_observation())
    assert first[0] == 1.0
    assert abs(second[0] - 0.5) < 1e-6
    assert abs(third[0] - 0.25) < 1e-6
    assert first[3] == 1.0 and second[3] == 1.0
    wrapper.reset()
    assert wrapper._smoothed is None


def test_action_smoothing_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="action_smoothing"):
        LeRobotACTPolicy(
            env_id="PickCube-v1",
            policy=FakeLeRobotPolicy(),
            preprocessor=FakePre(),
            postprocessor=FakePost(),
            action_smoothing=0.0,
        )


class FakeActionSpace:
    shape = (ACTION_DIM,)

    def contains(self, action: np.ndarray) -> bool:
        return action.shape == self.shape and bool(
            np.all(np.abs(action) <= 1)
        )


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


class SimplePolicy:
    phase = "fake"

    def reset(self) -> None:
        pass

    def act(self, observation: dict[str, object]) -> np.ndarray:
        return np.zeros(ACTION_DIM, dtype=np.float32)


def test_evaluate_policy_seeds_with_fake_env(tmp_path: Path) -> None:
    policies = {env_id: SimplePolicy() for env_id in SUPPORTED_TASKS}
    environments: list[FakeEvaluationEnv] = []

    def factory(config: object) -> FakeEvaluationEnv:
        env = FakeEvaluationEnv(config.seed)
        environments.append(env)
        return env

    summary = evaluate_policy_seeds(
        "fake", policies, [100, 101], env_factory=factory
    )
    assert summary.total_episodes == 4
    assert summary.total_successes == 4
    assert summary.success_rate == 1.0
    assert all(env.closed for env in environments)
    assert summary.tasks[0].success_rate == 1.0


def test_evaluate_lerobot_act_end_to_end_with_fake_env(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    summary = _write_dataset(dataset_dir)
    output = tmp_path / "output"
    train_lerobot_act(
        _tiny_config(summary, output),
        observation_env_factory=_observation_factory,
    )

    environments: list[FakeEvaluationEnv] = []

    def factory(config: object) -> FakeEvaluationEnv:
        env = FakeEvaluationEnv(config.seed)
        environments.append(env)
        return env

    result = evaluate_lerobot_act(
        output, seeds=[100, 101], device="cpu", env_factory=factory
    )
    assert (output / "evaluation.json").is_file()
    assert result.total_episodes == 4
    assert result.total_successes == 4
    assert all(env.closed for env in environments)


def test_evaluate_rejects_invalid_seeds() -> None:
    policies = {env_id: SimplePolicy() for env_id in SUPPORTED_TASKS}
    with pytest.raises(ValueError, match="不能为空"):
        evaluate_policy_seeds("fake", policies, [])
    with pytest.raises(ValueError, match="重复"):
        evaluate_policy_seeds("fake", policies, [100, 100])
    with pytest.raises(ValueError, match="非负"):
        evaluate_policy_seeds("fake", policies, [-1])


def test_effective_threshold_uses_bc_rate() -> None:
    assert act._compute_effective_threshold(0.75, 0.92) == 0.92
    assert act._compute_effective_threshold(0.80, 0.70) == 0.80


def _make_summary(name: str, rate: float) -> act.PolicyEvaluationSummary:
    successes = int(round(4 * rate))
    return act.PolicyEvaluationSummary(
        policy_name=name,
        schema_version=ACT_SCHEMA_VERSION,
        created_at="x",
        requested_per_task=2,
        total_episodes=4,
        total_successes=successes,
        total_failures=4 - successes,
        success_rate=rate,
        mean_inference_ms=1.0,
        mean_steps=1.0,
        total_chunk_queries=0,
        total_action_queries=0,
        nonfinite_actions=0,
        out_of_bounds_actions=0,
        tasks=(),
        episodes=(),
    )


def test_compare_gate_writes_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(
        act, "evaluate_lerobot_act", lambda *a, **k: _make_summary("act", 0.92)
    )
    monkeypatch.setattr(
        act,
        "evaluate_bc_for_comparison",
        lambda *a, **k: _make_summary("bc", 0.92),
    )
    monkeypatch.setattr(
        act,
        "evaluate_scripted_for_comparison",
        lambda *a, **k: _make_summary("scripted", 1.0),
    )
    summary = compare_act_bc_scripted(
        output,
        bc_checkpoint_path=tmp_path / "checkpoint.pt",
        seeds=[100, 101],
        device="cpu",
    )
    assert summary.act_rate == 0.92
    assert summary.bc_rate == 0.92
    assert summary.effective_threshold == 0.92
    assert summary.act_passed is True
    assert (output / "comparison.json").is_file()


def test_compare_gate_fails_below_bc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(
        act, "evaluate_lerobot_act", lambda *a, **k: _make_summary("act", 0.75)
    )
    monkeypatch.setattr(
        act,
        "evaluate_bc_for_comparison",
        lambda *a, **k: _make_summary("bc", 0.92),
    )
    monkeypatch.setattr(
        act,
        "evaluate_scripted_for_comparison",
        lambda *a, **k: _make_summary("scripted", 1.0),
    )
    summary = compare_act_bc_scripted(
        output,
        bc_checkpoint_path=tmp_path / "checkpoint.pt",
        seeds=[100, 101],
        device="cpu",
    )
    assert summary.effective_threshold == 0.92
    assert summary.act_passed is False


# ---------------------------------------------------------------------------
# 兼容性探针与 CLI
# ---------------------------------------------------------------------------


def test_probe_lerobot_act() -> None:
    report = act.probe_lerobot_act("mps")
    assert report["probe_passed"] is True
    assert report["device"] in ("mps", "cpu")
    assert report["grad_finite"] is True
    assert report["versions"]["lerobot"] == "0.6.1"


def test_cli_rejects_invalid_device(capsys: object) -> None:
    assert act_training_main(["--device", "tpu"]) == 1
    assert "device" in capsys.readouterr().err


def test_cli_rejects_zero_evaluation_episodes(capsys: object) -> None:
    assert act_training_main(["--eval-episodes-per-task", "0"]) == 1
    assert "eval_episodes_per_task" in capsys.readouterr().err
