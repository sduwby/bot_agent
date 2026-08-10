"""第 15 周 LeRobot 数据管线：schema、拆分、读回、检查、浏览与错误路径测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from behavior_cloning import ACTION_DIM, STATE_DIM  # noqa: E402
from lerobot_act import ENV_STATE_DIM, ROBOT_STATE_DIM  # noqa: E402
from lerobot_pipeline import (  # noqa: E402
    ACTION_KEY,
    CAMERA_RESOLUTION,
    NEXT_DONE_KEY,
    OBS_ENV_STATE_KEY as ENV_STATE_KEY,
    OBS_IMAGE_KEY,
    OBS_STATE_KEY,
    REWARD_KEY,
    SUCCESS_KEY,
    LerobotConversionConfig,
    LerobotPipelineError,
    build_lerobot_features,
    browse_lerobot_episode,
    convert_maniskill_to_lerobot,
    inspect_lerobot_dataset,
    load_maniskill_summary,
    maniskill_state_to_lerobot_features,
    open_lerobot_dataset,
    read_episode_frames,
    visualize_lerobot_episode,
)
from maniskill_intro import SUPPORTED_TASKS  # noqa: E402


def _obs_state_frame(step: int, episode_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.arange(9, dtype=np.float32) + episode_id + step * 0.1
    cube = np.zeros(13, dtype=np.float32)
    cube[:7] = np.arange(7, dtype=np.float32) + episode_id + step * 0.03
    goal = np.zeros(13, dtype=np.float32)
    goal[:3] = np.asarray([0.2, -0.1, 0.3], dtype=np.float32) + episode_id
    panda = np.zeros(31, dtype=np.float32)
    panda[13:22] = qpos
    panda[22:31] = np.arange(9, dtype=np.float32) * 0.01
    return panda, cube, goal


def _write_week11_style(
    root: Path,
    *,
    env_id: str,
    stem: str,
    goal_actor: str,
    episodes: int = 2,
    steps: int = 3,
) -> dict[str, object]:
    metadata_episodes = []
    with h5py.File(root / f"{stem}.h5", "w") as trajectories:
        for episode_id in range(episodes):
            group = trajectories.create_group(f"traj_{episode_id}")
            actions = np.stack(
                [
                    np.asarray([0.1 * step, -0.2, 0.3, 1.0], dtype=np.float32)
                    for step in range(steps)
                ]
            )
            group.create_dataset("actions", data=actions)
            group.create_dataset("rewards", data=np.ones(steps, dtype=np.float32))
            group.create_dataset(
                "terminated", data=np.zeros(steps, dtype=bool)
            )
            group.create_dataset(
                "truncated", data=np.zeros(steps, dtype=bool)
            )
            success = np.zeros(steps, dtype=bool)
            success[-1] = True
            group.create_dataset("success", data=success)
            states = group.create_group("env_states")
            articulations = states.create_group("articulations")
            actors = states.create_group("actors")
            pandas = []
            cubes = []
            goals = []
            for step in range(steps + 1):
                panda, cube, goal = _obs_state_frame(step, episode_id)
                pandas.append(panda)
                cubes.append(cube)
                goals.append(goal)
            articulations.create_dataset("panda", data=np.stack(pandas))
            actors.create_dataset("cube", data=np.stack(cubes))
            actors.create_dataset(goal_actor, data=np.stack(goals))
            obs = group.create_group("obs")
            agent = obs.create_group("agent")
            extra = obs.create_group("extra")
            agent.create_dataset("qpos", data=np.stack(pandas)[:, 13])
            agent.create_dataset("qvel", data=np.stack(pandas)[:, 22])
            extra.create_dataset("obj_pose", data=np.stack(cubes)[:, 0])
            extra.create_dataset("goal_pos", data=np.stack(goals)[:, 0])
            metadata_episodes.append(
                {
                    "episode_id": episode_id,
                    "episode_seed": episode_id,
                    "elapsed_steps": steps,
                    "success": True,
                    "control_mode": "pd_ee_delta_pos",
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


def _write_summary(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tasks = [
        _write_week11_style(
            root,
            env_id="PickCube-v1",
            stem="pick_cube",
            goal_actor="goal_site",
        ),
        _write_week11_style(
            root,
            env_id="PushCube-v1",
            stem="push_cube",
            goal_actor="goal_region",
        ),
    ]
    summary = {
        "schema_version": "maniskill_expert_state_v1",
        "complete": True,
        "tasks": tasks,
    }
    path = root / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def _config(tmp_path: Path, **kwargs: Any) -> LerobotConversionConfig:
    source = _write_summary(tmp_path / "source")
    output = tmp_path / "output"
    defaults = dict(
        source_summary=source,
        output_dir=output,
        episodes_per_task=1,
        fps=20,
    )
    defaults.update(kwargs)
    return LerobotConversionConfig(**defaults)


def _observation(episode_id: int = 0, step: int = 0) -> dict[str, object]:
    qpos = np.arange(9, dtype=np.float32) + episode_id + step * 0.1
    cube = np.zeros(13, dtype=np.float32)
    cube[:7] = np.arange(7, dtype=np.float32) + episode_id + step * 0.03
    goal = np.zeros(13, dtype=np.float32)
    goal[:3] = np.asarray([0.2, -0.1, 0.3], dtype=np.float32) + episode_id
    # ManiSkill get_obs 返回带 batch 维 (1, ...) 的 observation。
    return {
        "agent": {
            "qpos": qpos[None],
            "qvel": (np.arange(9, dtype=np.float32) * 0.01)[None],
        },
        "extra": {
            "tcp_pose": np.concatenate((qpos[:3], np.ones(4)))[None],
            "obj_pose": np.concatenate((cube[:3], np.ones(4)))[None],
            "goal_pos": goal[:3][None],
            "is_grasped": np.asarray([False]),
        },
    }


class FakeRenderBaseEnv:
    def __init__(self) -> None:
        self.frame = 0

    def set_state_dict(self, state: dict[str, object]) -> None:
        self.state = state

    def get_obs(self) -> dict[str, object]:
        return _observation(step=self.frame)


class FakeRenderEnv:
    action_space = type("AS", (), {"shape": (ACTION_DIM,)})()

    def __init__(self, env_id: str) -> None:
        self.env_id = env_id
        self.base_env = FakeRenderBaseEnv()
        self.closed = False
        self.frame = 0

    def reset(self, *, seed: int) -> tuple[dict[str, object], dict[str, object]]:
        return _observation(), {}

    def render(self) -> np.ndarray:
        base = np.full((*CAMERA_RESOLUTION, 3), 200, dtype=np.uint8)
        base[..., 0] = (200 + self.frame) % 256
        return base

    def close(self) -> None:
        self.closed = True


def _fake_env_factory(config: object) -> FakeRenderEnv:
    return FakeRenderEnv(config.env_id)


def _read_frame_rewards(dataset: object, episode_index: int) -> list[float]:
    frames = read_episode_frames(dataset, episode_index)
    return [float(np.asarray(f[REWARD_KEY]).reshape(-1)[0]) for f in frames]


# ---------------------------------------------------------------------------
# schema 与状态拆分
# ---------------------------------------------------------------------------


def test_build_features_no_camera() -> None:
    features = build_lerobot_features(False)
    assert set(features) == {
        OBS_STATE_KEY,
        ENV_STATE_KEY,
        ACTION_KEY,
        REWARD_KEY,
        SUCCESS_KEY,
        NEXT_DONE_KEY,
    }
    assert features[OBS_STATE_KEY]["shape"] == (ROBOT_STATE_DIM,)
    assert features[ENV_STATE_KEY]["shape"] == (ENV_STATE_DIM,)
    assert features[ACTION_KEY]["shape"] == (ACTION_DIM,)
    assert features[REWARD_KEY]["shape"] == (1,)


def test_build_features_with_camera() -> None:
    features = build_lerobot_features(True)
    assert OBS_IMAGE_KEY in features
    assert features[OBS_IMAGE_KEY]["dtype"] == "video"
    assert features[OBS_IMAGE_KEY]["shape"] == (*CAMERA_RESOLUTION, 3)


def test_split_state_roundtrip() -> None:
    state = np.random.randn(STATE_DIM).astype(np.float32)
    robot, env = maniskill_state_to_lerobot_features(state)
    assert robot.shape == (ROBOT_STATE_DIM,)
    assert env.shape == (ENV_STATE_DIM,)
    from lerobot_act import combine_lerobot_act_state

    np.testing.assert_allclose(combine_lerobot_act_state(robot, env), state)


def test_split_state_rejects_bad_dim() -> None:
    with pytest.raises(LerobotPipelineError, match="期望"):
        maniskill_state_to_lerobot_features(np.zeros(STATE_DIM + 1))
    with pytest.raises(LerobotPipelineError):
        maniskill_state_to_lerobot_features(
            np.array([np.nan] * STATE_DIM, dtype=np.float32)
        )


# ---------------------------------------------------------------------------
# 转换配置与 summary
# ---------------------------------------------------------------------------


def test_load_summary_rejects_bad_schema(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "summary.json"
    path.write_text(
        json.dumps({"schema_version": "unknown", "complete": True}),
        encoding="utf-8",
    )
    with pytest.raises(LerobotPipelineError, match="schema_version"):
        load_maniskill_summary(path)


def test_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fps"):
        _config(tmp_path, fps=0)
    with pytest.raises(ValueError, match="episodes_per_task"):
        _config(tmp_path, episodes_per_task=0)
    with pytest.raises(ValueError, match="不支持"):
        _config(tmp_path, tasks=("Unknown-v1",))
    with pytest.raises(ValueError, match="重复"):
        _config(tmp_path, tasks=("PickCube-v1", "PickCube-v1"))


# ---------------------------------------------------------------------------
# 转换与读回（fake env）
# ---------------------------------------------------------------------------


def test_convert_with_fake_env_and_readback(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = convert_maniskill_to_lerobot(
        config, env_factory=_fake_env_factory
    )
    assert result.complete
    assert result.total_errors == 0
    assert result.total_episodes == 2  # 每任务 1 条
    assert result.total_frames == 6  # 每任务 1 条 × 3 步
    for task in result.tasks:
        assert task.frames == 3
        assert task.errors == 0

    dataset = open_lerobot_dataset(config.output_dir, config.repo_id)
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 6
    # 读回验证 reward 与 action。
    rewards = _read_frame_rewards(dataset, 0)
    assert rewards == [1.0, 1.0, 1.0]
    frames = read_episode_frames(dataset, 0)
    assert len(frames) == 3
    frame = frames[0]
    assert np.asarray(frame[OBS_STATE_KEY]).shape == (ROBOT_STATE_DIM,)
    assert np.asarray(frame[ENV_STATE_KEY]).shape == (ENV_STATE_DIM,)
    assert np.asarray(frame[ACTION_KEY]).shape == (ACTION_DIM,)
    assert bool(np.asarray(frame[SUCCESS_KEY]).reshape(-1)[0]) is False
    assert bool(np.asarray(frames[-1][SUCCESS_KEY]).reshape(-1)[0]) is True


def test_convert_with_camera_frames(tmp_path: Path) -> None:
    config = _config(tmp_path, include_camera=True)
    result = convert_maniskill_to_lerobot(
        config, env_factory=_fake_env_factory
    )
    assert result.complete
    dataset = open_lerobot_dataset(config.output_dir, config.repo_id)
    frames = read_episode_frames(dataset, 0)
    from lerobot_pipeline import _image_hwc

    image = _image_hwc(frames[0][OBS_IMAGE_KEY])
    assert image.shape == (*CAMERA_RESOLUTION, 3)
    assert image.dtype == np.uint8


def test_convert_refuses_nonempty_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_dir.mkdir()
    (config.output_dir / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        convert_maniskill_to_lerobot(
            config, env_factory=_fake_env_factory
        )
    assert (config.output_dir / "keep.txt").read_text() == "x"


def test_convert_skips_errors_and_reports(tmp_path: Path) -> None:
    config = _config(tmp_path, episodes_per_task=2)

    def broken_factory(config_obj: object) -> FakeRenderEnv:
        if config_obj.env_id == "PushCube-v1":
            raise LerobotPipelineError("故意失败")
        return FakeRenderEnv(config_obj.env_id)

    result = convert_maniskill_to_lerobot(
        config, env_factory=broken_factory
    )
    assert not result.complete
    assert result.total_errors == 2
    pick = next(t for t in result.tasks if t.env_id == "PickCube-v1")
    push = next(t for t in result.tasks if t.env_id == "PushCube-v1")
    assert pick.errors == 0
    assert push.errors == 2  # 每 task 的 episodes_per_task 全部失败


# ---------------------------------------------------------------------------
# 检查与异常帧
# ---------------------------------------------------------------------------


def test_inspect_clean_dataset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    convert_maniskill_to_lerobot(config, env_factory=_fake_env_factory)
    report = inspect_lerobot_dataset(config.output_dir, config.repo_id)
    assert report.total_episodes == 2
    assert report.total_frames == 6
    assert report.missing_count == 0
    assert report.abnormal_frames == 0
    assert report.complete
    payload = json.loads(
        (config.output_dir / "inspection.json").read_text(encoding="utf-8")
    )
    assert payload["missing_count"] == 0


def test_inspect_detects_missing_and_abnormal(tmp_path: Path, monkeypatch: Any) -> None:
    config = _config(tmp_path)
    convert_maniskill_to_lerobot(config, env_factory=_fake_env_factory)

    from lerobot_pipeline import _inspect_frame

    dataset = open_lerobot_dataset(config.output_dir, config.repo_id)
    features = build_lerobot_features(False)
    frame = dataset[0]
    # 人为制造字段缺失与越界 action。
    frame.pop(ENV_STATE_KEY)
    frame[ACTION_KEY] = np.asarray([5.0, 0, 0, 0], dtype=np.float32)
    inspection = _inspect_frame(dataset, frame, features)
    assert ENV_STATE_KEY in inspection.missing
    assert inspection.action_out_of_bounds is True
    # 非有限值检测。
    frame2 = dataset[1]
    frame2[ACTION_KEY] = np.asarray([np.nan, 0, 0, 0], dtype=np.float32)
    inspection2 = _inspect_frame(dataset, frame2, features)
    assert ACTION_KEY in inspection2.nonfinite


# ---------------------------------------------------------------------------
# 浏览与可视化
# ---------------------------------------------------------------------------


def test_browse_episode(tmp_path: Path) -> None:
    config = _config(tmp_path)
    convert_maniskill_to_lerobot(config, env_factory=_fake_env_factory)
    result = browse_lerobot_episode(config.output_dir, config.repo_id, 0)
    assert result["episode_index"] == 0
    assert result["frame_count"] == 3
    assert result["frames"][0]["state_norm"] > 0
    assert result["frames"][0]["action"] == pytest.approx(
        [0.0, -0.2, 0.3, 1.0], abs=1e-5
    )


def test_visualize_episode_with_camera(tmp_path: Path) -> None:
    config = _config(tmp_path, include_camera=True)
    convert_maniskill_to_lerobot(config, env_factory=_fake_env_factory)
    image_dir = tmp_path / "images"
    report = visualize_lerobot_episode(
        config.output_dir, config.repo_id, 0, image_dir
    )
    assert len(report["thumbnails"]) == 3
    assert Path(report["thumbnails"][0]).is_file()
    assert (image_dir / "episode_0.json").is_file()


def test_browse_unknown_episode_fails(tmp_path: Path) -> None:
    config = _config(tmp_path)
    convert_maniskill_to_lerobot(config, env_factory=_fake_env_factory)
    with pytest.raises(LerobotPipelineError, match="不存在"):
        browse_lerobot_episode(config.output_dir, config.repo_id, 99)
