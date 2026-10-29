"""适配 checkpoint 离线评测指标测试 — 纯逻辑，无 torch/lerobot。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapt_eval_offline import (  # noqa: E402
    CAVEAT,
    GRIPPER_INDEX,
    HOLD_BAND,
    compute_offline_metrics,
    metrics_table,
    parse_episode_list,
    select_frame_ids,
)
from real_observation import EMPTY_GRIP_BASELINE, GRIP_HOLD_MAX, GRIP_POS_MARGIN  # noqa: E402


def _frames(grippers: list[float]) -> np.ndarray:
    """构造 (N, 6) 数组，前 5 维为 0，第 6 维为给定夹爪值。"""
    arr = np.zeros((len(grippers), 6))
    arr[:, GRIPPER_INDEX] = grippers
    return arr


class TestHoldBand:
    def test_matches_real_robot_judgement(self) -> None:
        assert HOLD_BAND == pytest.approx(
            (EMPTY_GRIP_BASELINE + GRIP_POS_MARGIN, GRIP_HOLD_MAX)
        )


class TestComputeMetrics:
    def test_perfect_prediction(self) -> None:
        truth = _frames([2.0, 7.0, 9.0, 20.0])
        m = compute_offline_metrics(truth.copy(), truth)
        assert m.action_l1 == pytest.approx(0.0)
        assert m.gripper_l1 == pytest.approx(0.0)
        assert m.hold_recall == pytest.approx(1.0)
        assert m.hold_precision == pytest.approx(1.0)
        assert m.hold_f1 == pytest.approx(1.0)

    def test_missed_hold(self) -> None:
        """真值夹住（7）但预测为空闭（2）→ recall 0。"""
        gt = _frames([7.0, 7.0])
        pred = _frames([2.0, 2.0])
        m = compute_offline_metrics(pred, gt)
        assert m.hold_recall == pytest.approx(0.0)
        assert m.hold_precision == pytest.approx(0.0)
        assert m.hold_f1 == pytest.approx(0.0)

    def test_false_positive_hold(self) -> None:
        """真值张开（20）但预测夹住（7）→ precision 0。"""
        gt = _frames([20.0])
        pred = _frames([7.0])
        m = compute_offline_metrics(pred, gt)
        assert m.hold_precision == pytest.approx(0.0)
        assert m.hold_recall == pytest.approx(0.0)

    def test_partial_recall_and_l1(self) -> None:
        gt = _frames([7.0, 7.0, 2.0, 2.0])
        pred = _frames([7.0, 2.0, 2.0, 2.0])
        m = compute_offline_metrics(pred, gt)
        assert m.hold_recall == pytest.approx(0.5)
        assert m.hold_precision == pytest.approx(1.0)
        assert m.hold_f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)
        assert m.gripper_l1 == pytest.approx(5.0 / 4)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_offline_metrics(np.zeros((3, 6)), np.zeros((2, 6)))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_offline_metrics(np.zeros((0, 6)), np.zeros((0, 6)))

    def test_caveat_is_reported(self) -> None:
        m = compute_offline_metrics(_frames([7.0]), _frames([7.0]))
        assert m.to_dict()["caveat"] == CAVEAT


class TestParseEpisodes:
    def test_accepts_bracket_and_plain(self) -> None:
        assert parse_episode_list("[0,4,8]") == [0, 4, 8]
        assert parse_episode_list("0, 4, 8") == [0, 4, 8]

    def test_dedupes(self) -> None:
        assert parse_episode_list("[0,0,4]") == [0, 4]

    def test_rejects_negative_and_empty(self) -> None:
        with pytest.raises(ValueError):
            parse_episode_list("[-1,2]")
        with pytest.raises(ValueError):
            parse_episode_list("[]")


class TestSelectFrameIds:
    """A7 需要在整个 episode 区间均匀取帧（stride 会全落在前几集）。"""

    def test_stride_matches_a5_protocol(self) -> None:
        assert select_frame_ids(100, stride=40, max_frames=120) == [0, 40, 80]
        assert select_frame_ids(100, stride=40, offset=10, max_frames=2) == [10, 50]

    def test_even_covers_both_ends(self) -> None:
        ids = select_frame_ids(1000, max_frames=5, even=True)
        assert ids[0] == 0 and ids[-1] == 999
        assert ids == sorted(set(ids))

    def test_even_respects_frame_budget(self) -> None:
        assert len(select_frame_ids(10000, max_frames=160, even=True)) == 160

    def test_even_caps_at_total(self) -> None:
        assert select_frame_ids(3, max_frames=120, even=True) == [0, 1, 2]
        assert select_frame_ids(1, max_frames=120, even=True) == [0]

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError):
            select_frame_ids(0)
        with pytest.raises(ValueError):
            select_frame_ids(10, offset=10)


class TestTable:
    def test_sorted_by_action_l1_and_has_caveat(self) -> None:
        good = compute_offline_metrics(_frames([7.0]), _frames([7.0]))
        bad = compute_offline_metrics(_frames([2.0]), _frames([7.0]))
        table = metrics_table({"bad": bad, "good": good})
        lines = [line for line in table.splitlines() if line.startswith("|")]
        assert "good" in lines[2] and "bad" in lines[3]  # 升序
        assert CAVEAT in table


class TestRenameCameras:
    """zero_shot 用 smolvla_base，必须把 fixed/handeye 改成 camera1/2。"""

    def test_renames_only_cameras(self) -> None:
        from run_adapt_eval_offline import RENAME_TO_BASE_CAMERAS, _rename_cameras

        item = {
            "observation.images.fixed": 1,
            "observation.images.handeye": 2,
            "observation.state": 3,
        }
        out = _rename_cameras(item, RENAME_TO_BASE_CAMERAS)
        assert out == {
            "observation.images.camera1": 1,
            "observation.images.camera2": 2,
            "observation.state": 3,
        }

    def test_empty_mapping_is_identity(self) -> None:
        from run_adapt_eval_offline import _rename_cameras

        item = {"observation.images.fixed": 1}
        assert _rename_cameras(item, {}) is item
