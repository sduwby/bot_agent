"""`vla_generalization` 的纯逻辑测试（不依赖 ManiSkill / PyTorch / 真机）。

覆盖：三种协议的 split 形状与语义、零泄漏（含交叉积留一）、固定 seed 的确定性、
非法 heldout_ratio 与空侧拒绝、`proportion_ci`（确定性 / total=0 / 区间包住点估计）、
`evaluate_generalization` 端到端（注入脚本 runner，手算 rate/CI/gap）、seed 重叠拒绝。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_vla_generalization import main as cli_main  # noqa: E402
from run_vla_generalization import parse_protocols  # noqa: E402
from run_vla_generalization import _load_combos  # noqa: E402
from vla_generalization import (  # noqa: E402
    CI_LEVEL,
    HELDOUT_SUCCESS_TARGET,
    PROTOCOLS,
    SCHEMA_VERSION,
    VLAGeneralizationError,
    GeneralizationSplit,
    _assert_disjoint_seeds,
    _checked_seeds,
    build_split,
    evaluate_generalization,
    format_generalization_markdown,
    proportion_ci,
    write_generalization_report,
)


def _overlap(split: GeneralizationSplit) -> set:
    return set(split.train_combos) & set(split.heldout_combos)


# ---------------------------------------------------------------------------
# combo 协议
# ---------------------------------------------------------------------------


class TestComboSplit:
    def test_cross_product_diagonal_keeps_all_on_both_sides(self) -> None:
        combos = [("A", "1"), ("A", "2"), ("B", "1"), ("B", "2")]
        split = build_split(combos, "combo", 0.3, seed=0)
        assert split.protocol == "combo"
        assert _overlap(split) == set(), "交叉积留一必须零泄漏"
        assert split.combo_counts == {"train": 2, "heldout": 2}
        # 每个物体、每条指令都出现在两侧（设计 §7.4 的经典例）。
        for side in (split.train_combos, split.heldout_combos):
            assert {obj for obj, _ in side} == {"A", "B"}
            assert {instruction for _, instruction in side} == {"1", "2"}

    def test_fallback_ratio_split_when_grid_incomplete(self) -> None:
        # 缺 (B,1)，不是完整笛卡尔积 → 走确定性比例切分。
        combos = [("A", "1"), ("A", "2"), ("B", "2")]
        split = build_split(combos, "combo", 0.34, seed=0)
        assert _overlap(split) == set()
        assert len(split.train_combos) + len(split.heldout_combos) == 3
        assert split.train_combos and split.heldout_combos

    @pytest.mark.parametrize("seed", [0, 1, 2, 7])
    def test_no_leakage_across_seeds(self, seed: int) -> None:
        combos = [(obj, instr) for obj in "ABCD" for instr in "12"]
        split = build_split(combos, "combo", 0.3, seed=seed)
        assert _overlap(split) == set()

    def test_deterministic_for_fixed_seed(self) -> None:
        combos = [(obj, instr) for obj in "ABCD" for instr in "12"]
        first = build_split(combos, "combo", 0.3, seed=3)
        second = build_split(combos, "combo", 0.3, seed=3)
        assert first.train_combos == second.train_combos
        assert first.heldout_combos == second.heldout_combos

    def test_result_independent_of_input_order(self) -> None:
        combos = [(obj, instr) for obj in "ABC" for instr in "12"]
        forward = build_split(combos, "combo", 0.3, seed=5)
        backward = build_split(list(reversed(combos)), "combo", 0.3, seed=5)
        assert forward.train_combos == backward.train_combos
        assert forward.heldout_combos == backward.heldout_combos

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_bad_heldout_ratio_rejected(self, ratio: float) -> None:
        with pytest.raises(VLAGeneralizationError, match="heldout_ratio"):
            build_split([("A", "1"), ("B", "2")], "combo", ratio, seed=0)

    def test_single_combo_cannot_split(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="< 2"):
            build_split([("A", "1")], "combo", 0.3, seed=0)

    def test_duplicate_input_combos_rejected(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="重复"):
            build_split([("A", "1"), ("A", "1")], "combo", 0.3, seed=0)

    def test_unknown_protocol_rejected(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="未知协议"):
            build_split([("A", "1"), ("B", "2")], "nope", 0.3, seed=0)


# ---------------------------------------------------------------------------
# paraphrase 协议
# ---------------------------------------------------------------------------


class TestParaphraseSplit:
    def test_holds_out_one_variant_and_flags_single_variant_object(self) -> None:
        combos = [
            ("A", "pick"), ("A", "grab"),
            ("B", "lift"), ("B", "raise"),
            ("C", "push"),
        ]
        split = build_split(combos, "paraphrase", 0.3, seed=0)
        assert _overlap(split) == set()
        # C 只有 1 个变体 → 整物体 held-out；A/B 各留 1 个变体。
        assert ("C", "push") in split.heldout_combos
        for obj in ("A", "B"):
            held = [pair for pair in split.heldout_combos if pair[0] == obj]
            train = [pair for pair in split.train_combos if pair[0] == obj]
            assert len(held) == 1 and len(train) == 1
        # 一条全局告警（无同义改写模型）+ 一条 C 的退化告警。
        assert len(split.warnings) == 2
        assert any("同义改写" in item for item in split.warnings)
        assert any("C" in item for item in split.warnings)

    def test_all_single_variant_rejected(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="paraphrase"):
            build_split([("A", "x"), ("B", "y")], "paraphrase", 0.3, seed=0)

    def test_deterministic_for_fixed_seed(self) -> None:
        combos = [("A", "pick"), ("A", "grab"), ("B", "lift"), ("B", "raise")]
        first = build_split(combos, "paraphrase", 0.3, seed=2)
        second = build_split(combos, "paraphrase", 0.3, seed=2)
        assert first.train_combos == second.train_combos
        assert first.heldout_combos == second.heldout_combos


# ---------------------------------------------------------------------------
# novel_object 协议
# ---------------------------------------------------------------------------


class TestNovelObjectSplit:
    def test_holds_out_whole_objects(self) -> None:
        combos = [(obj, instr) for obj in "ABC" for instr in "12"]
        split = build_split(combos, "novel_object", 0.34, seed=0)
        assert _overlap(split) == set()
        held_objects = {obj for obj, _ in split.heldout_combos}
        train_objects = {obj for obj, _ in split.train_combos}
        assert len(held_objects) == 1, "3 个物体 × 0.34 → 留 1 个物体"
        assert held_objects.isdisjoint(train_objects)
        assert train_objects and held_objects
        # 被留出物体的**全部**指令都进 held-out。
        for obj in held_objects:
            expected = {pair for pair in combos if pair[0] == obj}
            assert expected <= set(split.heldout_combos)

    def test_single_object_rejected(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="至少需要 2 个不同物体"):
            build_split([("A", "1"), ("A", "2")], "novel_object", 0.3, seed=0)

    def test_deterministic_for_fixed_seed(self) -> None:
        combos = [(obj, instr) for obj in "ABCD" for instr in "12"]
        first = build_split(combos, "novel_object", 0.3, seed=4)
        second = build_split(combos, "novel_object", 0.3, seed=4)
        assert first.train_combos == second.train_combos
        assert first.heldout_combos == second.heldout_combos


# ---------------------------------------------------------------------------
# GeneralizationSplit 数据结构
# ---------------------------------------------------------------------------


class TestGeneralizationSplitDataclass:
    def test_rejects_leakage(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="泄漏"):
            GeneralizationSplit("combo", (("A", "1"),), (("A", "1"),))

    def test_rejects_duplicates_within_side(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="重复"):
            GeneralizationSplit("combo", (("A", "1"), ("A", "1")), (("B", "2"),))

    def test_rejects_empty_side(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="不能为空"):
            GeneralizationSplit("combo", (), (("A", "1"),))

    def test_rejects_unknown_protocol(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="未知协议"):
            GeneralizationSplit("nope", (("A", "1"),), (("B", "2"),))

    def test_combo_counts_property(self) -> None:
        split = GeneralizationSplit("combo", (("A", "1"), ("B", "2")), (("A", "2"),))
        assert split.combo_counts == {"train": 2, "heldout": 1}

    def test_to_dict_from_dict_round_trip(self) -> None:
        split = GeneralizationSplit(
            "novel_object", (("A", "1"),), (("B", "2"),), warnings=("w",)
        )
        payload = split.to_dict()
        payload = json.loads(json.dumps(payload, ensure_ascii=False))
        restored = GeneralizationSplit.from_dict(payload)
        assert restored == split
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["combo_counts"] == {"train": 1, "heldout": 1}


# ---------------------------------------------------------------------------
# seed 校验
# ---------------------------------------------------------------------------


class TestSeedValidation:
    def test_disjoint_ok(self) -> None:
        _assert_disjoint_seeds([500, 501], [502, 503])

    def test_overlap_rejected(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="重叠"):
            _assert_disjoint_seeds([500, 501, 502], [502, 503])

    def test_duplicates_rejected(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="重复"):
            _checked_seeds([1, 1])

    def test_negative_rejected(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="非负整数"):
            _checked_seeds([-1])


# ---------------------------------------------------------------------------
# proportion_ci
# ---------------------------------------------------------------------------


class TestProportionCi:
    def test_deterministic_for_fixed_seed(self) -> None:
        assert proportion_ci(8, 10, seed=1) == proportion_ci(8, 10, seed=1)

    def test_total_zero_is_degenerate(self) -> None:
        assert proportion_ci(0, 0) == (0.0, 0.0)

    def test_interval_brackets_point_estimate(self) -> None:
        low, high = proportion_ci(8, 10, seed=0)
        assert low <= 0.8 <= high
        assert 0.0 <= low <= high <= 1.0

    def test_all_success_and_all_failure(self) -> None:
        assert proportion_ci(10, 10) == (1.0, 1.0)
        assert proportion_ci(0, 10) == (0.0, 0.0)

    def test_ci_level_widens_interval(self) -> None:
        narrow = proportion_ci(8, 10, level=0.80, seed=0)
        wide = proportion_ci(8, 10, level=0.99, seed=0)
        assert wide[1] - wide[0] >= narrow[1] - narrow[0]

    @pytest.mark.parametrize(
        "args,kwargs",
        [
            ((-1, 10), {}),
            ((11, 10), {}),
            ((1, -1), {}),
            ((1, 10), {"level": 1.0}),
            ((1, 10), {"samples": 0}),
        ],
    )
    def test_invalid_arguments_rejected(self, args: tuple, kwargs: dict) -> None:
        with pytest.raises(VLAGeneralizationError):
            proportion_ci(*args, **kwargs)


# ---------------------------------------------------------------------------
# evaluate_generalization
# ---------------------------------------------------------------------------


def _scripted_runner(succeed_on: set[tuple[str, str]]):
    """构造只对指定 (object, instruction) 成功的确定性 runner，并记录调用。"""
    calls: list[tuple[str, int, str]] = []

    def runner(env_id: str, seed: int, instruction: str) -> bool:
        calls.append((env_id, seed, instruction))
        return (env_id, instruction) in succeed_on

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def _manual_split() -> GeneralizationSplit:
    return GeneralizationSplit(
        "combo", (("A", "1"), ("B", "2")), (("A", "2"), ("B", "1"))
    )


class TestEvaluateGeneralization:
    def test_end_to_end_rates_ci_and_gap(self) -> None:
        split = _manual_split()
        # 除 ("B","2") 外全部成功：train 4/8，held-out 8/8。
        runner = _scripted_runner({("A", "1"), ("A", "2"), ("B", "1")})
        report = evaluate_generalization(None, split, 4, runner=runner, base_seed=500)
        assert report["schema_version"] == SCHEMA_VERSION
        assert report["protocol"] == "combo"
        assert report["n_episodes"] == 4
        assert report["train"] == {
            "episodes": 8, "successes": 4, "success_rate": 0.5,
            "ci_low": report["train"]["ci_low"], "ci_high": report["train"]["ci_high"],
        }
        assert report["train"]["ci_low"] <= 0.5 <= report["train"]["ci_high"]
        # held-out 全成功 → 退化 CI (1.0, 1.0)。
        assert report["heldout"]["successes"] == 8
        assert report["heldout"]["success_rate"] == 1.0
        assert (report["heldout"]["ci_low"], report["heldout"]["ci_high"]) == (1.0, 1.0)
        assert report["gap"] == -0.5
        assert report["heldout_meets_target"] is True
        assert report["heldout_target"] == HELDOUT_SUCCESS_TARGET

    def test_per_combo_details(self) -> None:
        runner = _scripted_runner({("A", "1"), ("A", "2"), ("B", "1")})
        report = evaluate_generalization(None, _manual_split(), 4, runner=runner)
        details = {
            (item["object"], item["instruction"]): item["successes"]
            for item in report["combos"]["train"] + report["combos"]["heldout"]
        }
        assert details == {("A", "1"): 4, ("B", "2"): 0, ("A", "2"): 4, ("B", "1"): 4}

    def test_train_and_heldout_seeds_are_disjoint(self) -> None:
        runner = _scripted_runner({("A", "1")})
        report = evaluate_generalization(None, _manual_split(), 3, runner=runner, base_seed=100)
        train_seeds = {seed for item in report["combos"]["train"] for seed in item["seeds"]}
        heldout_seeds = {seed for item in report["combos"]["heldout"] for seed in item["seeds"]}
        assert train_seeds and heldout_seeds
        assert train_seeds.isdisjoint(heldout_seeds)

    def test_runner_receives_object_seed_instruction(self) -> None:
        runner = _scripted_runner({("A", "1")})
        evaluate_generalization(None, _manual_split(), 2, runner=runner, base_seed=10)
        first = runner.calls[0]  # type: ignore[attr-defined]
        assert first == ("A", 10, "1")

    def test_gap_rounds_to_four_decimals(self) -> None:
        # train 2/3、held-out 1/3 → gap = round(0.33333…, 4) = 0.3333。
        split = GeneralizationSplit("combo", (("A", "1"),), (("B", "1"),))

        def even_seeds_only(env_id: str, seed: int, instruction: str) -> bool:
            return seed % 2 == 0 and (env_id, instruction) in {("A", "1"), ("B", "1")}

        report = evaluate_generalization(None, split, 3, runner=even_seeds_only, base_seed=500)
        assert report["train"]["success_rate"] == pytest.approx(2 / 3)
        assert report["heldout"]["success_rate"] == pytest.approx(1 / 3)
        assert report["gap"] == pytest.approx(0.3333)
        assert report["gap"] == round(
            report["train"]["success_rate"] - report["heldout"]["success_rate"], 4
        )

    def test_scope_note_is_honest_about_language(self) -> None:
        runner = _scripted_runner(set())
        report = evaluate_generalization(None, _manual_split(), 1, runner=runner)
        assert "忽略语言指令" in report["scope_note"]
        assert "runner" in report["scope_note"]

    def test_propagates_split_warnings(self) -> None:
        split = GeneralizationSplit(
            "paraphrase", (("A", "pick"),), (("A", "grab"),), warnings=("注意",)
        )
        runner = _scripted_runner(set())
        report = evaluate_generalization(None, split, 1, runner=runner)
        assert report["split_warnings"] == ["注意"]

    def test_missing_policy_for_default_runner_rejected(self) -> None:
        with pytest.raises(VLAGeneralizationError, match="policy"):
            evaluate_generalization(None, _manual_split(), 2)

    @pytest.mark.parametrize("episodes", [0, -1])
    def test_bad_n_episodes_rejected(self, episodes: int) -> None:
        runner = _scripted_runner(set())
        with pytest.raises(VLAGeneralizationError, match="n_episodes"):
            evaluate_generalization(None, _manual_split(), episodes, runner=runner)

    def test_bad_base_seed_rejected(self) -> None:
        runner = _scripted_runner(set())
        with pytest.raises(VLAGeneralizationError, match="base_seed"):
            evaluate_generalization(None, _manual_split(), 2, runner=runner, base_seed=-1)


# ---------------------------------------------------------------------------
# 报告落盘
# ---------------------------------------------------------------------------


class TestReportOutput:
    def test_markdown_contains_key_figures(self) -> None:
        runner = _scripted_runner({("A", "1"), ("A", "2"), ("B", "1")})
        report = evaluate_generalization(None, _manual_split(), 4, runner=runner)
        text = format_generalization_markdown(report)
        assert "泛化 gap" in text
        assert "口径边界" in text
        assert "A7 门槛" in text

    def test_write_report_creates_both_files(self, tmp_path) -> None:
        runner = _scripted_runner({("A", "1"), ("A", "2"), ("B", "1")})
        report = evaluate_generalization(None, _manual_split(), 4, runner=runner)
        json_path, markdown_path = write_generalization_report(report, tmp_path)
        assert json_path.is_file() and markdown_path.is_file()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["protocol"] == "combo"
        assert payload["gap"] == -0.5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_parse_protocols_dedupes_and_validates(self) -> None:
        assert parse_protocols("combo") == ["combo"]
        assert parse_protocols("combo,combo,paraphrase") == ["combo", "paraphrase"]
        assert parse_protocols(",".join(PROTOCOLS)) == list(PROTOCOLS)
        with pytest.raises(VLAGeneralizationError, match="未知协议"):
            parse_protocols("combo,nope")
        with pytest.raises(VLAGeneralizationError, match="不能为空"):
            parse_protocols(" , ")

    def test_load_combos_cross_product(self) -> None:
        args = argparse.Namespace(combos_json="", objects="A,B", instructions="1,2")
        assert _load_combos(args) == [
            ("A", "1"), ("A", "2"), ("B", "1"), ("B", "2")
        ]

    def test_load_combos_from_json(self, tmp_path) -> None:
        path = tmp_path / "combos.json"
        path.write_text(json.dumps([["A", "1"], ["B", "2"]]), encoding="utf-8")
        args = argparse.Namespace(combos_json=str(path), objects="", instructions="")
        assert _load_combos(args) == [("A", "1"), ("B", "2")]

    def test_load_combos_requires_a_source(self) -> None:
        args = argparse.Namespace(combos_json="", objects="", instructions="")
        with pytest.raises(VLAGeneralizationError, match="必须给出"):
            _load_combos(args)

    def test_load_combos_rejects_two_sources(self, tmp_path) -> None:
        path = tmp_path / "combos.json"
        path.write_text(json.dumps([["A", "1"]]), encoding="utf-8")
        args = argparse.Namespace(combos_json=str(path), objects="A", instructions="1")
        with pytest.raises(VLAGeneralizationError, match="二选一"):
            _load_combos(args)

    def test_dry_run_writes_nothing_and_passes(self, tmp_path, capsys) -> None:
        out_dir = tmp_path / "out"
        code = cli_main(
            [
                "--dry-run",
                "--objects", "A,B",
                "--instructions", "1,2",
                "--out", str(out_dir),
            ]
        )
        assert code == 0
        assert not out_dir.exists(), "dry-run 不应写任何产物"
        captured = capsys.readouterr().out
        assert "无泄漏断言" in captured
        for protocol in PROTOCOLS:
            assert f"[{protocol}]" in captured


def test_ci_level_default_matches_constant() -> None:
    """默认 level 与 CI_LEVEL 一致（95%），避免两侧漂移。"""
    assert CI_LEVEL == 0.95
    assert proportion_ci(5, 10) == proportion_ci(5, 10, level=CI_LEVEL)
