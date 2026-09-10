"""vla_prepost 单元测试 — 纯逻辑（torch 组装/数值域/截断），不加载真实 SmolVLA。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from vla_prepost import FakeVLAPolicy, smolvla_post, smolvla_prepare


def test_prepare_batch_shape_and_range() -> None:
    state = np.zeros(6, dtype=np.float32)
    images = {
        "observation.images.fixed": np.full((64, 64, 3), 255, dtype=np.uint8),
        "observation.images.handeye": np.full((64, 64, 3), 0, dtype=np.uint8),
    }
    b = smolvla_prepare(state, images, "夹起螺丝刀", device="cpu")
    assert b["task"] == "夹起螺丝刀"
    assert tuple(b["observation.state"].shape) == (1, 6)
    assert tuple(b["observation.images.fixed"].shape) == (1, 3, 64, 64)
    # uint8 255 → float 1.0；0 → 0.0
    assert float(b["observation.images.fixed"].max()) == 1.0
    assert float(b["observation.images.handeye"].min()) == 0.0


def test_prepare_missing_image_skipped() -> None:
    state = np.zeros(6, dtype=np.float32)
    b = smolvla_prepare(state, {}, "task")
    assert "observation.images.fixed" not in b


def test_prepare_accepts_state_dict() -> None:
    state = {"wrist_flex.pos": 89.0, "gripper.pos": 1.5}
    b = smolvla_prepare(state, {}, "task")
    assert b["observation.state"].shape[1] == 6


def test_post_truncates_and_flattens() -> None:
    import torch

    action = torch.arange(12, dtype=torch.float32).view(1, 12)
    out = smolvla_post(action, action_dim=6)
    assert out.shape == (6,)
    assert out.dtype == np.float32
    assert float(out[0]) == 0.0 and float(out[5]) == 5.0


def test_fake_policy_contract() -> None:
    p = FakeVLAPolicy(action_dim=6)
    state = np.full(6, 3.0, dtype=np.float32)
    b = smolvla_prepare(state, {}, "task")
    act = smolvla_post(p.select_action(b), action_dim=6)
    assert p.calls == 1
    assert act.shape == (6,)
    p.reset()
    assert p.reset_calls == 1
