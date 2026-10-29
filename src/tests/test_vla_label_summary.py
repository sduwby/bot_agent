"""A3/A4 汇总（`vla_label_summary`）测试 — 纯逻辑，无网络 / 无 GPU / 无 torch。"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vlm_labeler import LabelRecord  # noqa: E402
from vla_label_summary import (  # noqa: E402
    ALIGNMENT_MISSING,
    DEFAULT_SIDECAR_DIR,
    NONE_KEY,
    alignment_section,
    build_summary,
    confidence_by_intent,
    confidence_stats,
    count_values,
    discriminability,
    infer_repo_semantics,
    intent_distribution,
    load_records_by_repo,
    render_markdown,
    repo_summary,
    sidecar_facts,
    subtask_distribution,
    success_distribution,
    verify_sidecar_alignment,
    write_summary,
)


def _record(ref: str, *, intent=None, subtask="grasp", success=None, conf=0.9,
            task="夹起桌上的螺丝刀") -> LabelRecord:
    return LabelRecord(episode_ref=ref, task_text=task, intent=intent,
                       subtask=subtask, success=success, confidence=conf)


class TestSemantics:
    def test_pick_suffix(self) -> None:
        assert infer_repo_semantics("demo_vla_01_pick_a") == "pick"

    def test_place_suffix(self) -> None:
        assert infer_repo_semantics("demo_vla_03_place_left") == "place"

    def test_unknown_suffix(self) -> None:
        assert infer_repo_semantics("demo_vla_09_whatever") is None


class TestDistributions:
    def test_intent_distribution_uses_none_key(self) -> None:
        records = [_record("e0", intent="pick"), _record("e1", intent=None),
                   _record("e2", intent="pick")]
        assert intent_distribution(records) == {"pick": 2, NONE_KEY: 1}

    def test_success_distribution_splits_three_ways(self) -> None:
        records = [_record("e0", success=True), _record("e1", success=False),
                   _record("e2", success=None), _record("e3", success=True)]
        assert success_distribution(records) == {"true": 2, "false": 1, "none": 1}

    def test_subtask_distribution(self) -> None:
        records = [_record("e0", subtask="grasp"), _record("e1", subtask="lift"),
                   _record("e2", subtask="grasp")]
        assert subtask_distribution(records) == {"grasp": 2, "lift": 1}

    def test_count_values_bool_and_none(self) -> None:
        assert count_values([True, False, None, True]) == {
            "true": 2, "false": 1, NONE_KEY: 1
        }


class TestConfidence:
    def test_empty_returns_zero_n(self) -> None:
        assert confidence_stats([]) == {"n": 0, "mean": None, "min": None, "max": None}

    def test_stats(self) -> None:
        stats = confidence_stats([0.8, 0.9, 1.0])
        assert stats["n"] == 3
        assert stats["mean"] == 0.9
        assert stats["min"] == 0.8
        assert stats["max"] == 1.0

    def test_by_intent_groups(self) -> None:
        records = [_record("e0", intent="pick", conf=0.9),
                   _record("e1", intent="pick", conf=0.7),
                   _record("e2", intent=None, conf=0.5)]
        grouped = confidence_by_intent(records)
        assert grouped["pick"]["n"] == 2
        assert grouped["pick"]["mean"] == 0.8
        assert grouped[NONE_KEY]["n"] == 1


class TestDiscriminability:
    def test_place_never_predicted_is_degenerate(self) -> None:
        # 预测全是 pick，期望里有一半是 place → κ 退化
        records = [_record(f"e{i}", intent="pick") for i in range(4)]
        result = discriminability(records, ["pick", "pick", "place", "place"])
        assert result["place_predicted_times"] == 0
        assert result["place_expected_count"] == 2
        assert result["place_never_predicted"] is True
        assert result["kappa_degenerate"] is True
        # 预测全同 → po 等于期望里 pick 的占比，κ 必然塌向 0
        assert result["kappa_vs_repo_semantics"] < 0.05

    def test_both_classes_predicted_not_degenerate(self) -> None:
        records = [_record("e0", intent="pick"), _record("e1", intent="place")]
        result = discriminability(records, ["pick", "place"])
        assert result["place_predicted_times"] == 1
        assert result["place_never_predicted"] is False
        assert result["kappa_degenerate"] is False
        assert result["kappa_vs_repo_semantics"] == 1.0


class TestRepoSummary:
    def test_flags_and_expected_intent(self) -> None:
        records = [_record("e0", intent="pick", success=True),
                   _record("e1", intent="pick", success=True)]
        block = repo_summary("demo_vla_01_pick_a", records)
        assert block["expected_intent"] == "pick"
        assert block["all_success_true"] is True
        assert block["all_success_false"] is False
        assert block["success_true_ratio"] == 1.0

    def test_all_false(self) -> None:
        records = [_record("e0", intent="pick", success=False)]
        block = repo_summary("demo_vla_03_place_left", records)
        assert block["expected_intent"] == "place"
        assert block["all_success_false"] is True


class TestSidecarFacts:
    def test_missing_sidecar_reports_zero(self, tmp_path) -> None:
        facts = sidecar_facts(tmp_path, ["demo_vla_01_pick_a"])
        assert facts["all_repos_present"] is False
        assert facts["per_repo"]["demo_vla_01_pick_a"]["rows"] == 0

    def test_reads_rows_and_loads(self, tmp_path) -> None:
        rows = [
            {"episode_index": 0, "frame_index": 0, "load": {"gripper.load": -206.0}},
            {"episode_index": 0, "frame_index": 1, "load": {"gripper.load": -210.0}},
            {"episode_index": 1, "frame_index": 0, "load": {"gripper.load": None},
             "decode_errors": ["gripper.load"]},
        ]
        (tmp_path / "demo_vla_01_pick_a.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        facts = sidecar_facts(tmp_path, ["demo_vla_01_pick_a"])
        block = facts["per_repo"]["demo_vla_01_pick_a"]
        assert block["rows"] == 3
        assert block["episodes"] == 2
        assert block["gripper_load_nonnull_frames"] == 2
        assert block["decode_error_rows"] == 1
        assert facts["all_repos_present"] is True


class TestAlignmentSection:
    def test_always_not_completed(self) -> None:
        section = alignment_section({"all_repos_present": True})
        assert section["status"] == "not_completed"
        assert section["confusion_matrix"] is None
        assert section["physical_success_criterion"] is None
        assert section["missing"] == list(ALIGNMENT_MISSING)
        assert section["sidecar_data_available"] is True

    def test_available_flag_follows_sidecar(self) -> None:
        assert alignment_section({"all_repos_present": False})["sidecar_data_available"] is False


class TestBuildSummary:
    def _two_repos(self):
        return {
            "demo_vla_01_pick_a": [_record("e0", intent="pick", success=True)],
            "demo_vla_01_place_left": [_record("e0", intent="pick", success=False)],
        }

    def test_headline_numbers(self) -> None:
        summary = build_summary(self._two_repos(), sidecar_dir="no_such_dir")
        assert summary["n_repos"] == 2
        assert summary["n_records"] == 2
        assert summary["a3_intent"]["intent_distribution"] == {"pick": 2}
        assert summary["a4_success"]["success_distribution"] == {
            "true": 1, "false": 1, "none": 0
        }
        disc = summary["a3_intent"]["discriminability"]
        assert disc["place_never_predicted"] is True
        assert disc["kappa_degenerate"] is True

    def test_notes_flag_zero_intent_power(self) -> None:
        summary = build_summary(self._two_repos(), sidecar_dir="no_such_dir")
        assert any("判别力为零" in note for note in summary["notes"])

    def test_consistency_table(self) -> None:
        summary = build_summary(self._two_repos(), sidecar_dir="no_such_dir")
        rows = {item["repo"]: item for item in
                summary["a4_success"]["consistency_with_repo_semantics"]}
        assert rows["demo_vla_01_pick_a"]["all_success_true"] is True
        assert rows["demo_vla_01_place_left"]["all_success_false"] is True

    def test_schema_and_alignment(self) -> None:
        summary = build_summary(self._two_repos(), sidecar_dir="no_such_dir")
        assert summary["schema_version"]
        assert summary["a4_alignment"]["status"] == "not_completed"


class TestRenderMarkdown:
    def test_contains_sections(self) -> None:
        summary = build_summary(
            {"demo_vla_01_place_left": [_record("e0", intent=None, success=False)]},
            sidecar_dir="no_such_dir",
        )
        md = render_markdown(summary)
        assert "## A3 意图（判别力）" in md
        assert "## A4 成功（分布与语义一致性）" in md
        assert "## A4 口径对齐（物理侧 vs VLM success）" in md
        assert "not_completed" in md


class TestLoadAndWrite:
    def _make_repo(self, root, name, records) -> None:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "labels.jsonl").write_text(
            "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )

    def test_skips_dry_run_artifacts(self, tmp_path) -> None:
        self._make_repo(tmp_path, "demo_vla_01_pick_a", [_record("e0", intent="pick")])
        self._make_repo(tmp_path, "dry_01_pick_a", [_record("e0", intent="pick")])
        found = load_records_by_repo(tmp_path)
        assert list(found) == ["demo_vla_01_pick_a"]

    def test_explicit_repos_filter(self, tmp_path) -> None:
        self._make_repo(tmp_path, "demo_vla_01_pick_a", [_record("e0")])
        self._make_repo(tmp_path, "demo_vla_01_place_left", [_record("e0")])
        found = load_records_by_repo(tmp_path, ["demo_vla_01_place_left"])
        assert list(found) == ["demo_vla_01_place_left"]

    def test_write_summary_writes_both_files(self, tmp_path) -> None:
        summary = build_summary(
            {"demo_vla_01_pick_a": [_record("e0", intent="pick", success=True)]},
            sidecar_dir="no_such_dir",
        )
        json_path, md_path = write_summary(summary, tmp_path / "summary")
        assert json_path.is_file() and md_path.is_file()
        reloaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert reloaded["n_records"] == 1


class TestVerifyAlignment:
    def test_ok_when_positions_match(self, tmp_path) -> None:
        rows = [
            {"episode_index": 0, "frame_index": 0, "load": {"gripper.load": -206.0},
             "positions": {"gripper.pos": 6.8}},
            {"episode_index": 0, "frame_index": 1, "load": {"gripper.load": -206.0},
             "positions": {"gripper.pos": 6.9}},
        ]
        path = tmp_path / "side.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
        result = verify_sidecar_alignment(path, {0: [6.8, 6.9]})
        assert result["ok"] is True
        assert result["episodes"] == 1

    def test_mismatch_detected(self, tmp_path) -> None:
        rows = [
            {"episode_index": 0, "frame_index": 0, "load": {"gripper.load": -206.0},
             "positions": {"gripper.pos": 6.8}},
        ]
        path = tmp_path / "side.jsonl"
        path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
        result = verify_sidecar_alignment(path, {0: [1.0]})
        assert result["ok"] is False


def test_default_sidecar_dir_constant() -> None:
    assert DEFAULT_SIDECAR_DIR == "runs/week38/collect/sidecars"
