"""label_audit 一致性审计测试 — 纯逻辑，无模型 API。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from label_audit import (  # noqa: E402
    LabelAuditError,
    cohen_kappa,
    compare_labels,
    default_label_scope,
    write_audit_report,
)
from vlm_labeler import LabelRecord  # noqa: E402


class TestCohenKappa:
    def test_perfect_agreement_is_one(self) -> None:
        assert cohen_kappa([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)

    def test_anti_correlated_is_minus_one(self) -> None:
        """完全反向：po=0、pe=0.5 → κ=-1。"""
        assert cohen_kappa([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(-1.0)

    def test_chance_level_is_zero(self) -> None:
        """po 与 pe 相等 → κ=0。"""
        assert cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)

    def test_hand_computed_case(self) -> None:
        """手算：a=5×0+5×1、b=4×0+6×1 → po=0.9、pe=0.5 → κ=0.8。"""
        a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        b = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        assert cohen_kappa(a, b) == pytest.approx(0.8)

    def test_strings_and_none_are_supported(self) -> None:
        assert cohen_kappa(["pick", "place"], ["pick", "place"]) == pytest.approx(1.0)
        # None 与缺失值同属 UNMAPPED 类（不丢样本）
        assert cohen_kappa([None, "pick"], [None, "pick"]) == pytest.approx(1.0)
        assert cohen_kappa([None, "pick"], ["pick", None]) == pytest.approx(-1.0)

    def test_unmapped_is_its_own_class(self) -> None:
        """一方未映射、另一方给了标签 → 记为一处不一致（而不是被丢掉）。"""
        assert cohen_kappa([None, "pick", "pick"], ["pick", "pick", "pick"]) < 1.0

    def test_all_same_class_is_trivially_one(self) -> None:
        """双方都只有同一类时 κ 无定义，按约定返回 1.0（并需在报告里说明退化）。"""
        assert cohen_kappa(["pick"] * 5, ["pick"] * 5) == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(LabelAuditError):
            cohen_kappa(["a"], ["a", "b"])

    def test_empty_raises(self) -> None:
        with pytest.raises(LabelAuditError):
            cohen_kappa([], [])


def _record(ref: str, *, source: str, intent=None, subtask=None, success=None, conf=0.9, task="t"):
    return LabelRecord(
        episode_ref=ref, task_text=task, intent=intent, subtask=subtask,
        success=success, confidence=conf, source=source, taxonomy_version="v1",
    )


class TestCompareLabels:
    def _pair(self):
        vlm = [
            _record("ep0", source="vlm", intent="pick", subtask="grasp", success=True),
            _record("ep1", source="vlm", intent="place", subtask="place", success=False, conf=0.2),
            _record("ep2", source="vlm", intent=None, subtask="lift", success=True, conf=0.5),
        ]
        human = [
            _record("ep0", source="human", intent="pick", subtask="grasp", success=True),
            _record("ep1", source="human", intent="pick", subtask="place", success=False),
            _record("ep2", source="human", intent=None, subtask="lift", success=True),
        ]
        return vlm, human

    def test_kappa_and_total(self) -> None:
        vlm, human = self._pair()
        report = compare_labels(vlm, human)
        assert report.n_total == 3
        # 3 条里 ep1 的 intent 不一致 → 不是完美一致
        assert 0.0 < report.kappa_intent < 1.0

    def test_confusion_matrix_shape_and_counts(self) -> None:
        vlm, human = self._pair()
        report = compare_labels(vlm, human)
        assert sum(sum(row) for row in report.confusion) == 3
        assert len(report.confusion) == len(report.confusion_labels)

    def test_confusion_counts_disagreement(self) -> None:
        vlm = [_record("ep0", source="vlm", intent="place")]
        human = [_record("ep0", source="human", intent="pick")]
        report = compare_labels(vlm, human)
        labels = report.confusion_labels
        row = report.confusion[labels.index("place")]
        assert row[labels.index("pick")] == 1

    def test_per_subtask_recall(self) -> None:
        vlm, human = self._pair()
        report = compare_labels(vlm, human)
        # 人工标的 grasp 1 条、VLM 也对上 → recall 1.0
        assert report.per_subtask_recall["grasp"] == pytest.approx(1.0)
        # 人工未出现的 subtask 不进报告
        assert "approach" not in report.per_subtask_recall

    def test_disagreements_sorted_by_confidence(self) -> None:
        vlm = [
            _record("epA", source="vlm", intent="pick", conf=0.9),
            _record("epB", source="vlm", intent="place", conf=0.1),
        ]
        human = [
            _record("epA", source="human", intent="place"),
            _record("epB", source="human", intent="pick"),
        ]
        report = compare_labels(vlm, human)
        assert [item["episode_ref"] for item in report.disagreements] == ["epB", "epA"]
        assert report.disagreements[0]["fields"]["intent"] == {"vlm": "place", "human": "pick"}

    def test_success_pairs_counted(self) -> None:
        vlm = [
            _record("ep0", source="vlm", success=True),
            _record("ep1", source="vlm", success=None),
        ]
        human = [
            _record("ep0", source="human", success=False),
            _record("ep1", source="human", success=True),
        ]
        report = compare_labels(vlm, human)
        assert report.n_success_pairs == 1
        # 只有一对且相反时 pe=0 → κ=0（数学上如此，不是 −1）
        assert report.kappa_success == pytest.approx(0.0)

    def test_kappa_success_uses_paired_samples(self) -> None:
        """4 对里 3 对一致 → po=0.5、pe=0.625 → κ=-1/3。"""
        flags = [(True, True), (True, True), (True, False), (False, True)]
        vlm = [_record(f"ep{i}", source="vlm", success=a) for i, (a, _) in enumerate(flags)]
        human = [_record(f"ep{i}", source="human", success=b) for i, (_, b) in enumerate(flags)]
        report = compare_labels(vlm, human)
        assert report.n_success_pairs == 4
        assert report.kappa_success == pytest.approx(-1.0 / 3.0)

    def test_no_success_pairs_warns(self) -> None:
        vlm = [_record("ep0", source="vlm", intent="pick")]
        human = [_record("ep0", source="human", intent="pick")]
        report = compare_labels(vlm, human)
        assert report.n_success_pairs == 0
        assert any("success" in item for item in report.warnings)

    def test_taxonomy_version_mismatch_warns(self) -> None:
        vlm = [LabelRecord(episode_ref="ep0", task_text="t", intent="pick", source="vlm",
                           taxonomy_version="v1")]
        human = [LabelRecord(episode_ref="ep0", task_text="t", intent="pick", source="human",
                             taxonomy_version="v2")]
        report = compare_labels(vlm, human)
        assert any("版本" in item for item in report.warnings)

    def test_label_scope_passthrough(self) -> None:
        vlm, human = self._pair()
        assert compare_labels(vlm, human, label_scope="grasp_lift_only").label_scope == "grasp_lift_only"
        # 未声明时必须给出默认提示，避免"边界消失"
        assert "未声明" in compare_labels(vlm, human).label_scope

    def test_only_shared_episodes_are_compared(self) -> None:
        vlm = [_record("ep0", source="vlm", intent="pick")]
        human = [_record("ep0", source="human", intent="pick"), _record("ep9", source="human", intent="pick")]
        assert compare_labels(vlm, human).n_total == 1

    def test_no_shared_episodes_raises(self) -> None:
        with pytest.raises(LabelAuditError, match="共同的 episode_ref"):
            compare_labels(
                [_record("ep0", source="vlm")], [_record("ep1", source="human")]
            )

    def test_verdict_uses_a3_threshold(self) -> None:
        vlm = [_record(f"ep{i}", source="vlm", intent="pick") for i in range(5)]
        human = [_record(f"ep{i}", source="human", intent="pick") for i in range(5)]
        assert compare_labels(vlm, human).verdict == "pass"

        human_bad = []
        for index in range(5):
            human_bad.append(
                _record(f"ep{index}", source="human", intent="place" if index < 2 else "pick")
            )
        assert compare_labels(vlm, human_bad).verdict == "fail"

    def test_degenerate_scope_is_flagged_by_caller(self) -> None:
        """单意图数据集上 intent κ 必然=1.0（退化）——测试固化这个事实，
        提醒调用方必须用 label_scope 说明边界，而不是把 κ=1 当成标注质量高。"""
        vlm = [_record(f"ep{i}", source="vlm", intent="pick") for i in range(4)]
        human = [_record(f"ep{i}", source="human", intent="pick") for i in range(4)]
        report = compare_labels(vlm, human, label_scope="single_intent_dataset_degenerate")
        assert report.kappa_intent == pytest.approx(1.0)
        assert "degenerate" in report.label_scope


class TestDefaultLabelScope:
    def test_mentions_dataset_and_degeneracy(self) -> None:
        vlm = [_record(f"ep{i}", source="vlm", intent="pick") for i in range(3)]
        scope = default_label_scope(vlm, dataset="demo_v23")
        assert "demo_v23" in scope
        assert "退化" in scope

    def test_multi_intent_not_flagged_degenerate(self) -> None:
        vlm = [_record("ep0", source="vlm", intent="pick"),
               _record("ep1", source="vlm", intent="place")]
        scope = default_label_scope(vlm, dataset="multi")
        assert "退化" not in scope
        assert "2 类" in scope

    def test_always_notes_position_only_success(self) -> None:
        scope = default_label_scope([_record("ep0", source="vlm", intent="pick")])
        assert "位置判据" in scope


class TestWriteAuditReport:
    def test_writes_json_and_disagreements(self, tmp_path) -> None:
        vlm = [_record("ep0", source="vlm", intent="pick", conf=0.3)]
        human = [_record("ep0", source="human", intent="place")]
        report = compare_labels(vlm, human, label_scope="unit_test")
        json_path, disagreements_path = write_audit_report(report, tmp_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["label_scope"] == "unit_test"
        assert payload["verdict"] in ("pass", "fail")
        rows = [json.loads(line) for line in disagreements_path.read_text(encoding="utf-8").splitlines() if line]
        assert rows and rows[0]["episode_ref"] == "ep0"

    def test_summary_lines_mention_scope(self) -> None:
        vlm = [_record("ep0", source="vlm", intent="pick")]
        human = [_record("ep0", source="human", intent="pick")]
        lines = compare_labels(vlm, human, label_scope="scoped").summary_lines()
        assert any("scoped" in line for line in lines)
        assert any("A3" in line for line in lines)
