"""第 18 周 SmolVLA 探针：配置、输入映射、go/no-go 与评测逻辑测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from smolvla_probe import (
    SMOLVLA_IMAGE_SIZE,
    SmolVLAProbeConfig,
    SmolVLAProbeError,
    ProbeEpisode,
    TaskProbeResult,
    decide_go_no_go,
    observation_to_smolvla_input,
)


def _observation() -> dict[str, object]:
    return {
        "agent": {"qpos": np.arange(9, dtype=np.float32)},
        "extra": {
            "tcp_pose": np.concatenate((np.arange(3, dtype=np.float32), np.ones(4))),
            "obj_pose": np.concatenate((np.arange(3, dtype=np.float32) + 1, np.ones(4))),
            "goal_pos": np.array([0.2, -0.1, 0.3], dtype=np.float32),
            "is_grasped": np.asarray(False),
        },
    }


def _image(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _episode(
    *,
    success: bool = False,
    truncated: bool = True,
    xy: float = 0.1,
    z: float = 0.1,
    gripper_close: float = 0.5,
    saturation: float = 0.5,
) -> ProbeEpisode:
    return ProbeEpisode(
        env_id="PickCube-v1",
        seed=0,
        steps=10,
        success=success,
        truncated=truncated,
        error=None,
        action_xy_mean=(xy, z),
        action_z_mean=z,
        gripper_close_ratio=gripper_close,
        raw_action_norm_mean=1.0,
        saturating_ratio=saturation,
    )


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------


def test_config_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        SmolVLAProbeConfig(output_dir=tmp_path, device="tpux")
    with pytest.raises(ValueError, match="seeds"):
        SmolVLAProbeConfig(output_dir=tmp_path, seeds=())
    with pytest.raises(ValueError, match="不支持"):
        SmolVLAProbeConfig(output_dir=tmp_path, tasks=("Unknown-v1",))
    with pytest.raises(ValueError, match="重复"):
        SmolVLAProbeConfig(
            output_dir=tmp_path, tasks=("PickCube-v1", "PickCube-v1")
        )


# ---------------------------------------------------------------------------
# 输入映射
# ---------------------------------------------------------------------------


def test_observation_to_input_shape() -> None:
    result = observation_to_smolvla_input(
        _observation(), _image(), "Pick up the cube.", SMOLVLA_IMAGE_SIZE
    )
    assert "observation.state" in result
    assert "observation.images.camera1" in result
    assert result["task"] == "Pick up the cube."
    state = result["observation.state"]
    assert state.shape == (6,)  # tcp[3] + gripper[2] + grasped[1]
    assert state.dtype == np.float32
    image = result["observation.images.camera1"]
    assert image.shape == (3, SMOLVLA_IMAGE_SIZE, SMOLVLA_IMAGE_SIZE)
    assert 0.0 <= image.min() <= image.max() <= 1.0


def test_observation_to_input_missing_field() -> None:
    bad = {"extra": {}}
    with pytest.raises(SmolVLAProbeError, match="tcp_pose"):
        observation_to_smolvla_input(bad, _image(), "x", SMOLVLA_IMAGE_SIZE)


# ---------------------------------------------------------------------------
# go/no-go 决策
# ---------------------------------------------------------------------------


def test_decide_go_with_success() -> None:
    task = TaskProbeResult(
        env_id="PickCube-v1", episodes=20, successes=1, success_rate=0.05,
        truncated_all=True, mean_xy_scale=0.1, mean_z_scale=0.1,
        gripper_close_ratio=0.5, action_saturation=0.5,
    )
    go, decision = decide_go_no_go([task])
    assert go is True
    assert "成功" in decision


def test_decide_go_with_signal() -> None:
    task = TaskProbeResult(
        env_id="PickCube-v1", episodes=20, successes=0, success_rate=0.0,
        truncated_all=True, mean_xy_scale=0.1, mean_z_scale=0.1,
        gripper_close_ratio=0.5, action_saturation=0.5,
    )
    go, _ = decide_go_no_go([task])
    assert go is True


def test_decide_no_go_no_signal() -> None:
    task = TaskProbeResult(
        env_id="PickCube-v1", episodes=20, successes=0, success_rate=0.0,
        truncated_all=True, mean_xy_scale=0.0, mean_z_scale=0.0,
        gripper_close_ratio=0.0, action_saturation=1.0,
    )
    go, decision = decide_go_no_go([task])
    assert go is False
    assert "no-go" in decision


# ---------------------------------------------------------------------------
# 聚合与 CLI 参数
# ---------------------------------------------------------------------------


def test_aggregate_task() -> None:
    from smolvla_probe import _aggregate_task

    eps = [
        _episode(success=False),
        _episode(success=True, truncated=False),
    ]
    agg = _aggregate_task("PickCube-v1", eps)
    assert agg.successes == 1
    assert agg.success_rate == 0.5


def test_parse_seeds_from_cli(tmp_path: Path) -> None:
    from run_smolvla_probe import _parse_seeds

    assert _parse_seeds("500-502") == (500, 501, 502)
    assert _parse_seeds("500,505") == (500, 505)
    with pytest.raises(ValueError, match="不能为空"):
        _parse_seeds("")
