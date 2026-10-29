"""A4 逐集复评测试 — 纯逻辑（旁挂行 + 标签），不读数据集。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_episode_success_audit import (  # noqa: E402
    audit_repo,
    episode_index_of,
    episodes_from_sidecar,
    pooled,
    render_markdown,
)
from vlm_labeler import LabelRecord  # noqa: E402


def _row(episode: int, frame: int, grip: float, wrist: float, load: float | None = None):
    return {
        "episode_index": episode,
        "frame_index": frame,
        "positions": {"gripper.pos": grip, "wrist_flex.pos": wrist},
        "load": {"gripper.load": load},
    }


def _label(episode: int, success: bool) -> LabelRecord:
    return LabelRecord(episode_ref=f"episode_{episode:06d}", task_text="t", success=success)


class TestEpisodeIndex:
    def test_parses_suffix(self) -> None:
        assert episode_index_of("episode_000003") == 3
        assert episode_index_of("demo/episode_12") == 12

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            episode_index_of("episode_abc")


class TestEpisodesFromSidecar:
    def test_sorts_frames_and_keeps_load(self) -> None:
        rows = [_row(0, 2, 7.0, 78.0, -100.0), _row(0, 1, 2.0, 72.0, 0.0)]
        episodes = episodes_from_sidecar(rows)
        assert [frame["gripper.pos"] for frame in episodes[0]] == [2.0, 7.0]
        assert episodes[0][1]["gripper.load"] == pytest.approx(-100.0)


class TestAuditRepo:
    def test_position_or_load_vs_pure_position(self) -> None:
        """ep2 位置不够但负载够：两种口径应当给出不同结论。"""
        rows = [
            _row(0, 0, 2.0, 72.0, 0.0),
            _row(0, 1, 7.0, 78.0, -100.0),
            _row(1, 0, 19.0, 78.0, 60.0),
            _row(2, 0, 3.9, 78.0, -206.0),
        ]
        labels = [_label(0, True), _label(1, False), _label(2, True)]

        with_load = audit_repo(rows, labels, use_load=True)
        assert with_load is not None
        assert with_load["sensor_success_count"] == 2
        assert with_load["confusion"] == {"tp": 2, "tn": 1, "fp": 0, "fn": 0}
        assert with_load["kappa"] == pytest.approx(1.0)

        position_only = audit_repo(rows, labels, use_load=False)
        assert position_only is not None
        assert position_only["sensor_success_count"] == 1
        assert position_only["confusion"] == {"tp": 1, "tn": 1, "fp": 0, "fn": 1}

    def test_no_common_episode_returns_none(self) -> None:
        assert audit_repo([_row(5, 0, 7.0, 78.0)], [_label(0, True)], use_load=True) is None


class TestPooled:
    def test_pools_episodes_instead_of_averaging_kappa(self) -> None:
        a = {"episodes": [
            {"sensor_success": True, "vlm_success": True},
            {"sensor_success": False, "vlm_success": False},
        ]}
        b = {"episodes": [
            {"sensor_success": True, "vlm_success": True},
        ]}
        result = pooled([a, b])
        assert result["n_episodes"] == 3
        assert result["kappa"] == pytest.approx(1.0)
        assert result["confusion"] == {"tp": 2, "tn": 1, "fp": 0, "fn": 0}

    def test_empty(self) -> None:
        assert pooled([])["kappa"] is None


class TestRender:
    def test_contains_both_rules(self) -> None:
        payload = {
            "with_load": {"repos": {}, "pooled": {
                "n_episodes": 0, "kappa": None, "accuracy": None, "confusion": {}}},
            "position_only": {"repos": {}, "pooled": {
                "n_episodes": 0, "kappa": None, "accuracy": None, "confusion": {}}},
        }
        text = render_markdown(payload)
        assert "位置 OR 负载" in text and "纯位置" in text
        assert "—" in text
