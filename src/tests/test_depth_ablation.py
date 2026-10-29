"""P0 感知消融（B0–B3 编排与聚合）测试 — 纯逻辑，不训练、不跑仿真。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from depth_ablation import (  # noqa: E402
    REDUNDANCY_CAVEAT,
    AblationError,
    AblationVariant,
    VariantResult,
    build_plan,
    read_env_state_dim,
    run_ablation,
    summarize_ablation,
    validate_dataset_dim,
    write_ablation_report,
)
from run_depth_ablation import parse_dataset_args, render_plan  # noqa: E402


class TestBuildPlan:
    def test_default_plan_dims_are_cumulative(self) -> None:
        plan = build_plan()
        assert [item.name for item in plan] == ["B0", "B1", "B2", "B3"]
        assert [item.env_state_dim for item in plan] == [13, 15, 17, 19]

    def test_plan_order_is_canonical(self) -> None:
        """跨运行可比的前提：即使调用方乱序传入，计划也按规范顺序。"""
        plan = build_plan(["B2", "B0"])
        assert [item.name for item in plan] == ["B0", "B2"]

    def test_subset_without_baseline_is_rejected(self) -> None:
        with pytest.raises(AblationError, match="B0"):
            build_plan(["B1", "B2"])

    def test_unknown_variant_rejected(self) -> None:
        with pytest.raises(AblationError, match="B9"):
            build_plan(["B0", "B9"])

    def test_empty_rejected(self) -> None:
        with pytest.raises(AblationError):
            build_plan([])

    def test_undeclared_feature_source_rejected(self) -> None:
        """列序必须来自单一事实来源，不允许"自己加列但不声明列数"。"""
        with pytest.raises(AblationError, match="derive_features"):
            build_plan(["B0"], derive_features=False)

    def test_base_dim_is_respected(self) -> None:
        plan = build_plan(["B0", "B1"], base_env_state_dim=13)
        assert plan[1].env_state_dim == 15


class TestVariantValidation:
    def test_variant_rejects_short_env_dim(self) -> None:
        with pytest.raises(AblationError, match="≥ 13"):
            AblationVariant(name="B0", extra_features=(), env_state_dim=12)

    def test_variant_rejects_unknown_name(self) -> None:
        with pytest.raises(AblationError, match="未知变体"):
            AblationVariant(name="B7", extra_features=(), env_state_dim=13)


class TestValidateDatasetDim:
    def _b2(self) -> AblationVariant:
        return build_plan(["B0", "B2"])[1]

    def test_matching_dim_passes(self) -> None:
        validate_dataset_dim(self._b2(), 17)

    def test_mismatch_is_rejected_with_numbers(self) -> None:
        with pytest.raises(AblationError) as excinfo:
            validate_dataset_dim(self._b2(), 13)
        message = str(excinfo.value)
        assert "13" in message and "17" in message


class TestReadEnvStateDim:
    def _dataset(self, tmp_path, shape):
        (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
        (tmp_path / "meta" / "info.json").write_text(
            json.dumps({"features": {"observation.environment_state": {"shape": shape}}}),
            encoding="utf-8",
        )
        return tmp_path

    def test_reads_first_axis(self, tmp_path) -> None:
        root = self._dataset(tmp_path, [15])
        assert read_env_state_dim(root) == 15

    def test_missing_meta_raises(self, tmp_path) -> None:
        with pytest.raises(AblationError, match="info.json"):
            read_env_state_dim(tmp_path)

    def test_missing_feature_raises(self, tmp_path) -> None:
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text(json.dumps({"features": {}}), encoding="utf-8")
        with pytest.raises(AblationError, match="environment_state"):
            read_env_state_dim(tmp_path)


class TestVariantResultValidation:
    def test_rate_out_of_range_rejected(self) -> None:
        with pytest.raises(AblationError, match="success_rate"):
            VariantResult(variant="B0", success_rate=1.5, n_episodes=10)

    def test_non_positive_episodes_rejected(self) -> None:
        with pytest.raises(AblationError, match="n_episodes"):
            VariantResult(variant="B0", success_rate=0.5, n_episodes=0)


class TestSummarizeAblation:
    def _results(self, rates: dict[str, float]) -> list[VariantResult]:
        return [
            VariantResult(variant=name, success_rate=rate, n_episodes=10)
            for name, rate in rates.items()
        ]

    def test_delta_grasp_is_in_percentage_points(self) -> None:
        report = summarize_ablation(self._results({"B0": 0.5, "B1": 0.65, "B2": 0.55}))
        assert report.delta_grasp_pp["B1"] == pytest.approx(15.0)
        assert report.delta_grasp_pp["B2"] == pytest.approx(5.0)
        assert "B0" not in report.delta_grasp_pp

    def test_verdict_uses_best_non_baseline(self) -> None:
        report = summarize_ablation(self._results({"B0": 0.5, "B1": 0.65, "B2": 0.55}))
        assert report.verdict == "pass"

    def test_verdict_fails_below_target(self) -> None:
        report = summarize_ablation(self._results({"B0": 0.5, "B1": 0.6}))
        assert report.verdict == "fail"

    def test_caveat_is_always_attached(self) -> None:
        report = summarize_ablation(self._results({"B0": 0.5, "B1": 0.5}))
        assert report.caveat == REDUNDANCY_CAVEAT
        assert any("冗余" in line for line in report.summary_lines())

    def test_missing_baseline_rejected(self) -> None:
        with pytest.raises(AblationError, match="B0"):
            summarize_ablation(self._results({"B1": 0.5}))

    def test_duplicate_variant_rejected(self) -> None:
        results = self._results({"B0": 0.5})
        with pytest.raises(AblationError, match="多次"):
            summarize_ablation(results + [VariantResult(variant="B0", success_rate=0.6, n_episodes=5)])

    def test_only_baseline_gives_not_applicable(self) -> None:
        assert summarize_ablation(self._results({"B0": 0.5})).verdict == "n/a"

    def test_nonfinite_action_warning(self) -> None:
        results = [
            VariantResult(variant="B0", success_rate=0.5, n_episodes=10),
            VariantResult(variant="B1", success_rate=0.5, n_episodes=10, nonfinite_actions=2),
        ]
        report = summarize_ablation(results)
        assert any("非有限" in note for note in report.notes)

    def test_failure_share_sum_warning(self) -> None:
        results = [
            VariantResult(variant="B0", success_rate=0.5, n_episodes=10),
            VariantResult(variant="B1", success_rate=0.5, n_episodes=10,
                          failure_shares={"truncated": 0.5}),
        ]
        report = summarize_ablation(results)
        assert any("归因" in note for note in report.notes)


class TestWriteReport:
    def test_writes_json_and_markdown(self, tmp_path) -> None:
        report = summarize_ablation([
            VariantResult(variant="B0", success_rate=0.5, n_episodes=10),
            VariantResult(variant="B1", success_rate=0.65, n_episodes=10,
                          closing_height_error_mm=3.5),
        ])
        json_path, markdown_path = write_ablation_report(report, tmp_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["delta_grasp_pp"]["B1"] == pytest.approx(15.0)
        assert payload["caveat"] == REDUNDANCY_CAVEAT
        assert "冗余" in markdown_path.read_text(encoding="utf-8")


class TestRunAblation:
    def test_order_and_baseline_skip_training(self, tmp_path) -> None:
        """B0 就是历史基线，不该再训一遍（省一个训练单元）。"""
        plan = build_plan(["B0", "B1"])
        trained: list[str] = []

        def train_fn(variant, dataset):
            trained.append(variant.name)
            return 1.0

        def eval_fn(variant, dataset):
            return VariantResult(variant=variant.name, success_rate=0.5, n_episodes=10)

        datasets = {"B0": tmp_path, "B1": tmp_path}
        run_ablation(plan, datasets, train_fn=train_fn, eval_fn=eval_fn)
        assert trained == ["B1"]

    def test_missing_dataset_rejected(self, tmp_path) -> None:
        plan = build_plan(["B0", "B1"])
        with pytest.raises(AblationError, match="B1"):
            run_ablation(plan, {"B0": tmp_path},
                         eval_fn=lambda v, d: VariantResult(v.name, 0.5, 10))

    def test_mismatched_variant_name_rejected(self, tmp_path) -> None:
        plan = build_plan(["B0", "B1"])
        with pytest.raises(AblationError, match="不一致"):
            run_ablation(plan, {"B0": tmp_path, "B1": tmp_path},
                         eval_fn=lambda v, d: VariantResult("B0", 0.5, 10))


class TestCliHelpers:
    def test_parse_dataset_args(self) -> None:
        datasets = parse_dataset_args(["B1=/tmp/a", "B2=/tmp/b"])
        assert set(datasets) == {"B1", "B2"}

    def test_duplicate_dataset_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_dataset_args(["B1=/tmp/a", "B1=/tmp/b"])

    def test_bad_format_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_dataset_args(["/tmp/a"])

    def test_render_plan_shows_dims(self) -> None:
        text = render_plan(build_plan(["B0", "B1"]))
        assert "B0" in text and "13" in text and "15" in text
