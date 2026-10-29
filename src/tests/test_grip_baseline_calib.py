"""空闭基准标定测试 — 纯逻辑，不连机械臂。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_grip_baseline_calib import (  # noqa: E402
    group_means,
    parse_poses,
    spread_by,
    stats,
    suggest_threshold,
)


class TestStats:
    def test_empty(self) -> None:
        assert stats([]) == {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                             "range": None}

    def test_single_value_has_zero_std(self) -> None:
        s = stats([3.8])
        assert (s["n"], s["mean"], s["std"], s["range"]) == (1, 3.8, 0.0, 0.0)

    def test_known_values(self) -> None:
        s = stats([3.0, 4.0])
        assert s["mean"] == pytest.approx(3.5)
        assert s["std"] == pytest.approx(0.5)
        assert (s["min"], s["max"], s["range"]) == (3.0, 4.0, 1.0)


class TestSpreadBy:
    def _trials(self):
        return [
            {"run": 0, "pose": "60", "pos_mean": 3.0},
            {"run": 0, "pose": "90", "pos_mean": 3.2},
            {"run": 1, "pose": "60", "pos_mean": 3.4},
            {"run": 1, "pose": "90", "pos_mean": 3.6},
        ]

    def test_groups_take_group_mean(self) -> None:
        assert group_means(self._trials(), "run") == {"0": [3.0, 3.2], "1": [3.4, 3.6]}

    def test_run_spread_range(self) -> None:
        result = spread_by(self._trials(), "run")
        assert result["groups"] == {"0": 3.1, "1": 3.5}
        assert result["range"] == pytest.approx(0.4)

    def test_pose_spread_range(self) -> None:
        result = spread_by(self._trials(), "pose")
        assert result["groups"] == {"60": 3.2, "90": 3.4}
        assert result["range"] == pytest.approx(0.2)

    def test_single_group_has_zero_range(self) -> None:
        trials = [{"run": 0, "pose": "60", "pos_mean": 3.0},
                  {"run": 0, "pose": "90", "pos_mean": 4.0}]
        assert spread_by(trials, "run")["range"] == 0.0

    def test_trials_without_pos_mean_skipped(self) -> None:
        assert spread_by([{"run": 0}], "run")["range"] == 0.0


class TestSuggestThreshold:
    def test_baseline_rounds_max_up_to_tenth(self) -> None:
        """判据是严格大于，baseline 必须取实测上限向上取整，否则空载会被误判为夹持。"""
        rec = suggest_threshold([3.7, 3.81, 3.62])
        assert rec["baseline_suggested"] == 3.9

    def test_without_hold_pos_only_gives_baseline(self) -> None:
        rec = suggest_threshold([3.6, 3.8])
        assert rec["baseline_suggested"] == 3.8
        assert "margin_suggested" not in rec
        assert "未提供 --hold-pos" in rec["note"]

    def test_margin_uses_half_headroom(self) -> None:
        """空载上限 3.9、薄件夹持 6.8 → 余量 2.9，margin 取一半 1.4（旧值 3.0 会漏判）。"""
        rec = suggest_threshold([3.7, 3.85], hold_pos=6.8)
        assert rec["baseline_suggested"] == 3.9
        assert rec["margin_max_allowed"] == pytest.approx(2.9)
        assert rec["margin_suggested"] == 1.4
        assert rec["threshold_suggested"] == pytest.approx(5.3)
        assert rec["threshold_suggested"] < 6.8

    def test_old_config_would_miss_thin_object(self) -> None:
        """对照：沿用旧 baseline=1.5 + margin=3.0 → 阈值 4.5 看似通过，
        但按实测上限 3.9 计算真实阈值应为 6.9 > 6.8 → 漏判。"""
        rec = suggest_threshold([3.7, 3.85], hold_pos=6.8)
        assert rec["baseline_suggested"] + 3.0 > 6.8

    def test_hold_pos_below_baseline_is_reported_unfeasible(self) -> None:
        """夹持位置没高于基准 → 位置判据对该件必然漏判，须靠负载判据。"""
        rec = suggest_threshold([3.0, 4.0], hold_pos=3.5)
        assert rec["margin_suggested"] is None
        assert "必然漏判" in rec["note"]

    def test_empty_samples_reported(self) -> None:
        rec = suggest_threshold([])
        assert rec["ok"] is False
        assert "没有空载位置样本" in rec["note"]

    def test_load_margin_is_echoed(self) -> None:
        assert suggest_threshold([3.0], load_margin=42.0)["load_margin_in_use"] == 42.0


class TestParsePoses:
    def test_comma_separated(self) -> None:
        assert parse_poses("60,75,90") == [60.0, 75.0, 90.0]

    def test_current_means_no_pose_change(self) -> None:
        assert parse_poses("current") == [None]
        assert parse_poses("") == [None]

    def test_whitespace_tolerated(self) -> None:
        assert parse_poses(" 60 , 75 ") == [60.0, 75.0]

    def test_invalid_raises(self) -> None:
        with pytest.raises(SystemExit):
            parse_poses("60,abc")
