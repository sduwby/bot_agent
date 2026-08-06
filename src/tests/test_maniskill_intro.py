"""第 10 周 ManiSkill 环境契约和 scripted policy 测试。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from maniskill_intro import (
    EpisodeContractError,
    ManiSkillTaskConfig,
    ObservationContractError,
    ScriptedPolicy,
    run_episode,
    summarize_observation,
)


def make_observation(
    *,
    tcp: tuple[float, float, float] = (0.0, 0.0, 0.12),
    obj: tuple[float, float, float] = (0.0, 0.0, 0.02),
    goal: tuple[float, float, float] = (0.15, 0.0, 0.10),
    is_grasped: bool = False,
) -> dict[str, object]:
    """构造与 ManiSkill state_dict extra 兼容的最小观察。"""
    return {
        "agent": {
            "qpos": np.zeros(9, dtype=np.float32),
            "qvel": np.zeros(9, dtype=np.float32),
        },
        "extra": {
            "tcp_pose": np.array((*tcp, 1.0, 0.0, 0.0, 0.0)),
            "obj_pose": np.array((*obj, 1.0, 0.0, 0.0, 0.0)),
            "goal_pos": np.asarray(goal, dtype=float),
            "is_grasped": np.asarray(is_grasped),
        },
    }


@pytest.mark.parametrize("env_id", ["PickCube-v1", "PushCube-v1"])
def test_task_config_accepts_week10_tasks(env_id: str) -> None:
    config = ManiSkillTaskConfig(env_id=env_id)
    assert config.sim_backend == "physx_cpu"
    assert config.control_mode == "pd_ee_delta_pos"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"env_id": "StackCube-v1"}, "不支持"),
        ({"env_id": "PickCube-v1", "seed": -1}, "seed"),
        ({"env_id": "PickCube-v1", "max_steps": 0}, "max_steps"),
        ({"env_id": "PickCube-v1", "obs_mode": "rgbd"}, "state_dict"),
        (
            {"env_id": "PickCube-v1", "control_mode": "pd_joint_pos"},
            "pd_ee_delta_pos",
        ),
        ({"env_id": "PickCube-v1", "reward_mode": "sparse"}, "reward"),
        ({"env_id": "PickCube-v1", "sim_backend": "auto"}, "physx_cpu"),
    ],
)
def test_task_config_rejects_out_of_scope_values(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ManiSkillTaskConfig(**kwargs)


def test_summarize_observation_records_sorted_leaf_contract() -> None:
    fields = summarize_observation(make_observation())
    paths = [field.path for field in fields]

    assert paths == sorted(paths)
    assert "agent.qpos" in paths
    assert "extra.tcp_pose" in paths
    tcp_field = next(field for field in fields if field.path == "extra.tcp_pose")
    assert tcp_field.shape == (7,)
    assert tcp_field.dtype == "float64"


def test_pick_policy_transitions_and_keeps_actions_bounded() -> None:
    policy = ScriptedPolicy("PickCube-v1")
    observations = [
        make_observation(tcp=(0.0, 0.0, 0.12)),
        make_observation(tcp=(0.0, 0.0, 0.02)),
        make_observation(tcp=(0.0, 0.0, 0.02), is_grasped=True),
        make_observation(
            tcp=(0.15, 0.0, 0.10),
            obj=(0.15, 0.0, 0.10),
            goal=(0.15, 0.0, 0.10),
            is_grasped=True,
        ),
    ]
    expected_phases = ["descend", "close", "lift_to_goal", "settle"]

    for observation, expected_phase in zip(observations, expected_phases):
        action = policy.act(observation)
        assert action.shape == (4,)
        assert np.all(np.isfinite(action))
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
        assert policy.phase == expected_phase


def test_push_policy_approaches_behind_then_pushes_toward_goal() -> None:
    policy = ScriptedPolicy("PushCube-v1")
    obj = (0.0, 0.0, 0.02)
    goal = (0.15, 0.0, 0.001)

    action = policy.act(
        make_observation(tcp=(-0.055, 0.0, 0.10), obj=obj, goal=goal)
    )
    assert policy.phase == "behind_object"
    assert action[3] == -1.0

    action = policy.act(
        make_observation(tcp=(-0.055, 0.0, 0.02), obj=obj, goal=goal)
    )
    assert policy.phase == "push_through_goal"

    action = policy.act(
        make_observation(
            tcp=(0.12, 0.0, 0.02),
            obj=(0.14, 0.0, 0.02),
            goal=goal,
        )
    )
    assert policy.phase == "settle"
    assert action[0] > 0.0


@pytest.mark.parametrize("missing_key", ["tcp_pose", "obj_pose", "goal_pos"])
def test_policy_rejects_missing_observation_fields(missing_key: str) -> None:
    observation = make_observation()
    del observation["extra"][missing_key]  # type: ignore[index]

    with pytest.raises(ObservationContractError, match=missing_key):
        ScriptedPolicy("PickCube-v1").act(observation)


class FakeActionSpace:
    """最小 Box action space 替身。"""

    shape = (4,)

    def contains(self, action: np.ndarray) -> bool:
        return (
            action.shape == self.shape
            and np.all(np.isfinite(action))
            and np.all(action >= -1.0)
            and np.all(action <= 1.0)
        )


class FakeEnv:
    """两步成功的 Gymnasium 环境替身。"""

    action_space = FakeActionSpace()

    def __init__(self) -> None:
        self.closed = False
        self.step_count = 0
        self.reset_seed: int | None = None

    def reset(self, *, seed: int) -> tuple[dict[str, object], dict[str, object]]:
        self.reset_seed = seed
        return make_observation(), {"reset_seed": seed}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        assert self.action_space.contains(action)
        self.step_count += 1
        success = self.step_count == 2
        return (
            make_observation(),
            0.25 * self.step_count,
            success,
            False,
            {"success": np.asarray(success), "elapsed_steps": self.step_count},
        )

    def close(self) -> None:
        self.closed = True


class ZeroPolicy:
    """始终返回合法零位移动、闭合夹爪动作。"""

    phase = "test"

    def __init__(self, env_id: str) -> None:
        self.env_id = env_id

    def reset(self) -> None:
        pass

    def act(self, observation: object) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def test_run_episode_collects_gymnasium_semantics_and_closes() -> None:
    env = FakeEnv()
    result = run_episode(
        ManiSkillTaskConfig("PickCube-v1", seed=7),
        env_factory=lambda config: env,
        policy_factory=ZeroPolicy,
    )

    assert env.reset_seed == 7
    assert env.closed
    assert result.success
    assert result.terminated
    assert not result.truncated
    assert result.steps == 2
    assert result.total_reward == pytest.approx(0.75)
    assert result.action_shape == (4,)
    assert result.phase_history == ("test", "test")
    assert result.final_info["elapsed_steps"] == 2


class InvalidPolicy(ZeroPolicy):
    def act(self, observation: object) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


def test_run_episode_closes_environment_on_contract_error() -> None:
    env = FakeEnv()

    with pytest.raises(EpisodeContractError, match="动作 shape"):
        run_episode(
            ManiSkillTaskConfig("PickCube-v1"),
            env_factory=lambda config: env,
            policy_factory=InvalidPolicy,
        )

    assert env.closed


class BrokenPolicy(ZeroPolicy):
    def __init__(self, env_id: str) -> None:
        raise RuntimeError("policy construction failed")


def test_run_episode_closes_environment_when_policy_construction_fails() -> None:
    env = FakeEnv()

    with pytest.raises(RuntimeError, match="policy construction failed"):
        run_episode(
            ManiSkillTaskConfig("PickCube-v1"),
            env_factory=lambda config: env,
            policy_factory=BrokenPolicy,
        )

    assert env.closed


class InvalidStepEnv(FakeEnv):
    def step(self, action: np.ndarray) -> tuple[object, ...]:
        return (make_observation(), 0.0, False, {})


def test_run_episode_rejects_non_gymnasium_step_tuple() -> None:
    env = InvalidStepEnv()

    with pytest.raises(EpisodeContractError, match="五元组"):
        run_episode(
            ManiSkillTaskConfig("PickCube-v1"),
            env_factory=lambda config: env,
            policy_factory=ZeroPolicy,
        )

    assert env.closed
