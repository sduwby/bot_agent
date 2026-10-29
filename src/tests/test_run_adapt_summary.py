"""4.5 离线结果汇总测试 — 纯文件 IO + 纯渲染，无 torch。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapt_eval_offline import CAVEAT  # noqa: E402
from run_adapt_summary import aggregate_rows, load_unit_rows, render_table  # noqa: E402


def _row(unit: str, action_l1: float, ratio: float | None = None) -> dict[str, object]:
    return {
        "unit": unit,
        "n_frames": 81,
        "action_l1": action_l1,
        "gripper_l1": 2.0,
        "hold_recall": 0.6,
        "hold_precision": 0.6,
        "hold_f1": 0.6,
        "trainable_ratio": ratio,
    }


def _write_unit(root, name: str, action_l1: float, ratio: float | None = None) -> None:
    unit = root / name
    unit.mkdir(parents=True)
    (unit / "offline_metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "n_frames": 10,
                    "action_l1": action_l1,
                    "gripper_l1": 0.1,
                    "hold_recall": 0.5,
                    "hold_precision": 0.4,
                    "hold_f1": 0.44,
                }
            }
        ),
        encoding="utf-8",
    )
    if ratio is not None:
        stats = root.parent / "stats" / name
        stats.mkdir(parents=True, exist_ok=True)
        (stats / "adapt_stats.json").write_text(
            json.dumps({"trainable_ratio": ratio}), encoding="utf-8"
        )


class TestLoadUnitRows:
    def test_reads_metrics_and_ratio(self, tmp_path) -> None:
        results = tmp_path / "results"
        results.mkdir()
        _write_unit(results, "head_ep30", 0.30, ratio=0.003633)
        _write_unit(results, "full_ep30", 0.40, ratio=None)
        rows = load_unit_rows(results, tmp_path / "stats")
        by_unit = {row["unit"]: row for row in rows}
        assert by_unit["head_ep30"]["trainable_ratio"] == pytest.approx(0.003633)
        assert by_unit["full_ep30"]["trainable_ratio"] is None
        assert by_unit["head_ep30"]["action_l1"] == pytest.approx(0.30)

    def test_zero_shot_ratio_is_zero(self, tmp_path) -> None:
        results = tmp_path / "results"
        results.mkdir()
        _write_unit(results, "zero_shot", 0.50, ratio=None)
        rows = load_unit_rows(results)
        assert rows[0]["trainable_ratio"] == 0.0

    def test_missing_dir_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_unit_rows(tmp_path / "nope")

    def test_empty_dir_raises(self, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            load_unit_rows(empty)

    def test_repeat_dir_uses_base_stats(self, tmp_path) -> None:
        """`head_ep30_r1` 的可训练占比应来自 `head_ep30/adapt_stats.json`。"""
        results = tmp_path / "results"
        _write_unit(results, "head_ep30_r1", 2.6, ratio=None)
        stats = tmp_path / "stats" / "head_ep30"
        stats.mkdir(parents=True)
        (stats / "adapt_stats.json").write_text(
            json.dumps({"trainable_ratio": 0.003633}), encoding="utf-8"
        )
        rows = load_unit_rows(results, tmp_path / "stats")
        assert rows[0]["trainable_ratio"] == pytest.approx(0.003633)


class TestAggregate:
    def test_groups_repeats_with_mean_and_std(self) -> None:
        rows = [
            _row("head_ep30_r1", 2.6, ratio=0.003633),
            _row("head_ep30_r2", 2.8, ratio=0.003633),
            _row("zero_shot", 32.0, ratio=0.0),
        ]
        aggregated = {row["unit"]: row for row in aggregate_rows(rows)}
        head = aggregated["head_ep30"]
        assert head["repeats"] == 2
        assert head["action_l1"] == pytest.approx(2.7)
        assert head["action_l1_std"] == pytest.approx(0.1414, rel=1e-3)
        assert head["trainable_ratio"] == pytest.approx(0.003633)
        assert aggregated["zero_shot"]["repeats"] == 1

    def test_single_result_has_no_band(self) -> None:
        table = render_table(aggregate_rows([_row("head_ep30", 2.6)]))
        assert "±" not in table

    def test_multi_repeat_table_shows_band(self) -> None:
        table = render_table(aggregate_rows([
            _row("head_ep30_r1", 2.6), _row("head_ep30_r2", 2.8),
        ]))
        assert "±" in table and "2" in table


class TestRenderTable:
    def test_sorted_and_formats_ratio(self) -> None:
        rows = [
            {"unit": "full_ep30", "n_frames": 10, "action_l1": 0.4, "gripper_l1": 0.1,
             "hold_recall": 0.5, "hold_precision": 0.4, "hold_f1": 0.44,
             "trainable_ratio": None},
            {"unit": "head_ep30", "n_frames": 10, "action_l1": 0.3, "gripper_l1": 0.1,
             "hold_recall": 0.5, "hold_precision": 0.4, "hold_f1": 0.44,
             "trainable_ratio": 0.003633},
        ]
        table = render_table(rows)
        lines = [line for line in table.splitlines() if line.startswith("|")]
        assert "head_ep30" in lines[2]
        assert "full_ep30" in lines[3]
        assert "0.36%" in table
        assert CAVEAT in table
