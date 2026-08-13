"""第 22 周：阶段评测与选型定向测试。

覆盖：候选解析、门禁 C 判定边界、模型卡聚合与落盘、CLI 退出码。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stage_selection import (
    GATE_TASK,
    CandidateResult,
    CandidateSplit,
    GateCVerdict,
    PolicyCandidate,
    StageSelectionConfig,
    StageSelectionError,
    build_model_card,
    compute_gate_c,
    parse_candidate,
)


def _split(
    *,
    split: str = "in_distribution",
    pick: float = 0.0,
    push: float = 1.0,
    exceptions: int = 0,
    nonfinite: int = 0,
    overall: float | None = None,
) -> CandidateSplit:
    if overall is None:
        overall = (pick + push) / 2
    return CandidateSplit(
        split=split,
        per_task={
            "PickCube-v1": {"success_rate": pick, "mean_steps": 30.0, "exceptions": 0.0},
            "PushCube-v1": {"success_rate": push, "mean_steps": 15.0, "exceptions": 0.0},
        },
        overall_success_rate=overall,
        exceptions=exceptions,
        nonfinite=nonfinite,
        out_of_bounds=0,
    )


def _visual(push_rate: float, *, nonfinite: int = 0, exceptions: int = 0) -> CandidateResult:
    return CandidateResult(
        name="visual",
        kind="visual_act",
        splits=(
            _split(
                pick=0.45,
                push=push_rate,
                nonfinite=nonfinite,
                exceptions=exceptions,
            ),
            _split(
                split="out_of_distribution",
                pick=0.30,
                push=push_rate,
                nonfinite=nonfinite,
                exceptions=exceptions,
            ),
        ),
    )


def _scripted() -> CandidateResult:
    return CandidateResult(
        name="scripted",
        kind="scripted",
        splits=(
            _split(pick=1.0, push=1.0),
            _split(split="out_of_distribution", pick=1.0, push=1.0),
        ),
    )


class TestPolicyCandidate:
    def test_parse_scripted(self) -> None:
        candidate = parse_candidate("scripted:scripted")
        assert candidate.kind == "scripted"
        assert candidate.input_mode == "state"

    def test_parse_visual_act(self, tmp_path: Path) -> None:
        candidate = parse_candidate(f"visual:visual_act:{tmp_path / 'ckpt'}")
        assert candidate.kind == "visual_act"
        assert candidate.input_mode == "visual"
        assert candidate.checkpoint_dir == (tmp_path / "ckpt").resolve()

    def test_parse_route(self) -> None:
        candidate = parse_candidate("route:route:visual_act,scripted")
        assert candidate.kind == "route"
        assert candidate.route == "visual_act,scripted"

    def test_parse_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_candidate("x:unknown")

    def test_visual_act_without_checkpoint_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_candidate("visual:visual_act:")

    def test_route_without_policies_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_candidate("route:route:")


class TestStageSelectionConfig:
    def test_valid(self, tmp_path: Path) -> None:
        config = StageSelectionConfig(
            output_dir=tmp_path / "out",
            seeds_in=(500, 501),
            seeds_ood=(600, 601),
        )
        assert config.seeds_in == (500, 501)

    def test_overlapping_seeds_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            StageSelectionConfig(
                output_dir=tmp_path / "out",
                seeds_in=(500, 501),
                seeds_ood=(501, 502),
            )

    def test_invalid_threshold(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            StageSelectionConfig(
                output_dir=tmp_path / "out", gate_threshold=1.5
            )


class TestComputeGateC:
    def test_passes(self) -> None:
        verdict = compute_gate_c([_visual(1.0), _scripted()])
        assert verdict.passed is True
        assert verdict.visual_rate == 1.0
        assert verdict.fallback_available is True
        assert verdict.safety_clean is True

    def test_visual_below_threshold_fails(self) -> None:
        verdict = compute_gate_c([_visual(0.4), _scripted()])
        assert verdict.passed is False
        assert verdict.visual_rate == 0.4

    def test_missing_fallback_fails(self) -> None:
        verdict = compute_gate_c([_visual(1.0)])
        assert verdict.passed is False
        assert verdict.fallback_available is False

    def test_nonfinite_dirty_fails(self) -> None:
        verdict = compute_gate_c([_visual(1.0, nonfinite=3), _scripted()])
        assert verdict.passed is False
        assert verdict.safety_clean is False

    def test_missing_visual_fails(self) -> None:
        verdict = compute_gate_c([_scripted()])
        assert verdict.passed is False
        assert "缺少可用" in verdict.notes

    def test_invalid_threshold_rejected(self) -> None:
        with pytest.raises(ValueError):
            compute_gate_c([_visual(1.0)], threshold=2.0)


class TestBuildModelCard:
    def test_card_and_report_written(self, tmp_path: Path) -> None:
        results = [_visual(1.0), _scripted()]
        verdict = compute_gate_c(results)
        card = build_model_card(
            results, verdict, output_dir=tmp_path
        )
        assert card["schema_version"].startswith("maniskill_stage_selection")
        assert card["gate"]["passed"] is True
        assert card["frozen_selection"]["main_policy"]
        card_path = tmp_path / "model_card.json"
        assert card_path.is_file()
        payload = json.loads(card_path.read_text(encoding="utf-8"))
        assert payload["candidates"][0]["name"] == "visual"
        report_path = tmp_path / "selection_report.md"
        assert report_path.is_file()
        text = report_path.read_text(encoding="utf-8")
        assert "门禁 C 判定" in text
        assert "冻结选型结论" in text

    def test_historical_merged(self, tmp_path: Path) -> None:
        results = [_visual(1.0), _scripted()]
        verdict = compute_gate_c(results)
        card = build_model_card(
            results,
            verdict,
            output_dir=tmp_path,
            historical={"bc": {"PickCube-v1": 0.84, "PushCube-v1": 1.0}},
        )
        assert card["historical"]["bc"]["PickCube-v1"] == 0.84

    def test_gate_task_in_card(self, tmp_path: Path) -> None:
        card = build_model_card(
            [_visual(1.0), _scripted()],
            compute_gate_c([_visual(1.0), _scripted()]),
            output_dir=tmp_path,
        )
        assert card["gate_task"] == GATE_TASK
