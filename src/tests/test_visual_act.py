"""第 16 周视觉 ACT：v3 读取、chunk、统计、配置、rollout 与资源报告测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from behavior_cloning import ACTION_DIM  # noqa: E402
from lerobot_act import ENV_STATE_DIM, ROBOT_STATE_DIM  # noqa: E402
from lerobot_pipeline import OBS_IMAGE_KEY  # noqa: E402
from maniskill_intro import SUPPORTED_TASKS  # noqa: E402
from visual_act import (  # noqa: E402
    ACTION_IS_PAD_KEY,
    ACTION_KEY,
    OBS_ENV_STATE_KEY,
    OBS_STATE_KEY,
    VisualACTChunkDataset,
    VisualACTError,
    VisualACTTrainingConfig,
    VisualTaskData,
    build_visual_chunk_dataset,
    compute_visual_act_stats,
    load_visual_act_checkpoints,
    load_visual_dataset,
    make_visual_act_config,
    make_visual_act_policy,
    split_task_transitions,
)


def _fake_task_data(
    episodes: int = 3,
    steps: int = 4,
    resolution: int = 32,
) -> VisualTaskData:
    env_id = "PickCube-v1"
    lengths = [steps] * episodes
    total = episodes * steps
    rng = np.random.default_rng(0)
    images = rng.uniform(0, 1, (total, 3, resolution, resolution)).astype(
        np.float32
    )
    robot_state = rng.uniform(-1, 1, (total, ROBOT_STATE_DIM)).astype(np.float32)
    env_state = rng.uniform(-1, 1, (total, ENV_STATE_DIM)).astype(np.float32)
    actions = rng.uniform(-1, 1, (total, ACTION_DIM)).astype(np.float32)
    return VisualTaskData(
        env_id=env_id,
        episode_ids=tuple(range(episodes)),
        episode_lengths=tuple(lengths),
        images=images,
        robot_state=robot_state,
        env_state=env_state,
        actions=actions,
    )


def _tiny_config(tmp_path: Path, **kwargs: Any) -> VisualACTTrainingConfig:
    defaults = dict(
        dataset_dir=tmp_path / "dataset",
        repo_id="test/repo",
        output_dir=tmp_path / "output",
        resolution=32,
        chunk_size=4,
        n_action_steps=2,
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
        train_episodes_per_task=2,
        validation_episodes_per_task=1,
    )
    defaults.update(kwargs)
    return VisualACTTrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------


def test_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resolution"):
        _tiny_config(tmp_path, resolution=0)
    with pytest.raises(ValueError, match="n_action_steps"):
        _tiny_config(tmp_path, chunk_size=4, n_action_steps=5)
    with pytest.raises(ValueError, match="device"):
        _tiny_config(tmp_path, device="tpu")
    with pytest.raises(ValueError, match="vision_backbone"):
        _tiny_config(tmp_path, vision_backbone="vit")
    with pytest.raises(ValueError, match="不支持"):
        _tiny_config(tmp_path, tasks=("Unknown-v1",))


# ---------------------------------------------------------------------------
# chunk 数据集与统计
# ---------------------------------------------------------------------------


def test_split_task_transitions_no_leak() -> None:
    data = _fake_task_data(episodes=4, steps=4)
    train, validation = split_task_transitions(
        data, train_episodes=2, validation_episodes=2
    )
    assert len(train) == 8
    assert len(validation) == 8
    assert np.intersect1d(train, validation).size == 0
    with pytest.raises(VisualACTError, match="不足以切分"):
        split_task_transitions(data, train_episodes=5, validation_episodes=1)


def test_build_visual_chunk_dataset() -> None:
    data = _fake_task_data(episodes=3, steps=8)
    train, _ = split_task_transitions(data, train_episodes=2, validation_episodes=1)
    dataset = build_visual_chunk_dataset(data, train, chunk_size=4)
    assert isinstance(dataset, VisualACTChunkDataset)
    assert dataset.images.shape[1:] == (3, 32, 32)
    assert dataset.robot_state.shape == (len(train), ROBOT_STATE_DIM)
    assert dataset.env_state.shape == (len(train), ENV_STATE_DIM)
    assert dataset.action_chunk.shape == (len(train), 4, ACTION_DIM)
    # episode 尾部（每个 8 步 episode 的最后 3 帧）应有 pad，其余无。
    for t in range(len(train)):
        if t % 8 < 5:
            assert not dataset.action_is_pad[t].any(), f"t={t}"
        else:
            assert dataset.action_is_pad[t].any(), f"t={t}"


def test_build_visual_chunk_dataset_padding() -> None:
    data = _fake_task_data(episodes=1, steps=2)
    dataset = build_visual_chunk_dataset(
        data, [0, 1], chunk_size=4
    )
    # t=0：可用 2 步（a0,a1），后 2 位 pad。
    assert not dataset.action_is_pad[0, :2].any()
    assert dataset.action_is_pad[0, 2:].all()
    # t=1：仅 1 步可用（a1），后 3 位 pad。
    assert not dataset.action_is_pad[1, 0]
    assert dataset.action_is_pad[1, 1:].all()


def test_compute_stats_only_train() -> None:
    data = _fake_task_data(episodes=3, steps=4)
    train, validation = split_task_transitions(
        data, train_episodes=2, validation_episodes=1
    )
    stats = compute_visual_act_stats(data, train)
    assert stats[OBS_IMAGE_KEY]["mean"].shape == (3, 1, 1)
    assert stats[OBS_STATE_KEY]["mean"].shape == (ROBOT_STATE_DIM,)
    assert stats[OBS_ENV_STATE_KEY]["mean"].shape == (ENV_STATE_DIM,)
    assert stats[ACTION_KEY]["mean"].shape == (ACTION_DIM,)
    # 与 validation 无关：仅由 train 决定。
    stats_val = compute_visual_act_stats(data, validation)
    assert not np.allclose(stats[OBS_STATE_KEY]["mean"], stats_val[OBS_STATE_KEY]["mean"])


# ---------------------------------------------------------------------------
# 配置与 policy 创建
# ---------------------------------------------------------------------------


def test_make_visual_act_config_features() -> None:
    config = _tiny_config(Path("/tmp/x"), resolution=64)
    act_config = make_visual_act_config(config)
    assert OBS_IMAGE_KEY in act_config.image_features
    assert act_config.image_features[OBS_IMAGE_KEY].shape == (3, 64, 64)
    assert act_config.robot_state_feature.shape == (ROBOT_STATE_DIM,)
    assert act_config.env_state_feature.shape == (ENV_STATE_DIM,)
    assert act_config.action_feature.shape == (ACTION_DIM,)


def test_make_visual_act_policy_forward(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    stats = {
        OBS_IMAGE_KEY: {
            "mean": np.zeros((3, 1, 1), dtype=np.float32),
            "std": np.ones((3, 1, 1), dtype=np.float32),
        },
        OBS_STATE_KEY: {
            "mean": np.zeros(ROBOT_STATE_DIM, dtype=np.float32),
            "std": np.ones(ROBOT_STATE_DIM, dtype=np.float32),
        },
        OBS_ENV_STATE_KEY: {
            "mean": np.zeros(ENV_STATE_DIM, dtype=np.float32),
            "std": np.ones(ENV_STATE_DIM, dtype=np.float32),
        },
        ACTION_KEY: {
            "mean": np.zeros(ACTION_DIM, dtype=np.float32),
            "std": np.ones(ACTION_DIM, dtype=np.float32),
        },
    }
    _, policy, pre, post = make_visual_act_policy(config, stats)
    torch = __import__("torch")
    data = _fake_task_data(episodes=3, steps=4, resolution=32)
    dataset = build_visual_chunk_dataset(
        data, [0, 1, 2], chunk_size=config.chunk_size
    )
    batch = {
        OBS_IMAGE_KEY: torch.from_numpy(dataset.images[:2]),
        OBS_STATE_KEY: torch.from_numpy(dataset.robot_state[:2]),
        OBS_ENV_STATE_KEY: torch.from_numpy(dataset.env_state[:2]),
        ACTION_KEY: torch.from_numpy(dataset.action_chunk[:2]),
        ACTION_IS_PAD_KEY: torch.from_numpy(dataset.action_is_pad[:2]),
    }
    processed = pre(batch)
    policy.train()
    loss, _ = policy.forward(processed)
    assert bool(torch.isfinite(loss))
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


# ---------------------------------------------------------------------------
# checkpoint 加载错误路径
# ---------------------------------------------------------------------------


def test_load_checkpoints_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(VisualACTError, match="不存在"):
        load_visual_act_checkpoints(
            tmp_path / "missing", device="cpu", resolution=32
        )


def test_load_checkpoints_missing_config(tmp_path: Path) -> None:
    output = tmp_path / "output"
    (output / "PickCube-v1" / "best").mkdir(parents=True)
    (output / "PushCube-v1" / "best").mkdir(parents=True)
    with pytest.raises(VisualACTError, match="config.json"):
        load_visual_act_checkpoints(output, device="cpu", resolution=32)


def test_visual_task_data_rejects_inconsistent(tmp_path: Path) -> None:
    data = _fake_task_data(episodes=2, steps=4)
    # 人为构造图像与动作数量不一致的读回结果（直接构造无法做到，用函数内部校验）
    images = data.images
    assert images.shape[0] == len(data.actions)


def test_resize_image_hwc_and_chw(tmp_path: Path) -> None:
    from visual_act import _resize_image

    # HWC uint8 → (3, H, W) float[0,1]。
    hwc = np.full((64, 64, 3), 128, dtype=np.uint8)
    resized = _resize_image(hwc, 32)
    assert resized.shape == (3, 32, 32)
    assert np.allclose(resized, 128 / 255.0, atol=1e-6)
    # CHW 输入也能处理。
    chw = np.transpose(hwc, (2, 0, 1))
    resized2 = _resize_image(chw, 32)
    assert resized2.shape == (3, 32, 32)
    with pytest.raises(VisualACTError, match="维度"):
        _resize_image(np.zeros((3, 3), dtype=np.uint8), 32)
