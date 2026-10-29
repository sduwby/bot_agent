"""真值标注工具测试 — 纯逻辑，不弹窗、无 OpenCV 依赖。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_perception_gt_label import (  # noqa: E402
    candidates_from_report,
    iter_frames,
    load_boxes,
    match_target,
    median_bbox,
    parse_bbox,
    save_ground_truth,
)


class TestParseBbox:
    def test_valid(self) -> None:
        assert parse_bbox([1, 2, 3, 4]) == [1.0, 2.0, 3.0, 4.0]

    @pytest.mark.parametrize("raw", [[1, 2, 3], [1, 2, 3, 4, 5], "1234", None, {"a": 1}])
    def test_wrong_shape_returns_none(self, raw) -> None:
        assert parse_bbox(raw) is None

    def test_non_numeric_returns_none(self) -> None:
        assert parse_bbox([1, 2, 3, "x"]) is None

    def test_degenerate_returns_none(self) -> None:
        assert parse_bbox([10, 0, 10, 5]) is None
        assert parse_bbox([10, 5, 20, 5]) is None

    def test_non_finite_returns_none(self) -> None:
        assert parse_bbox([0, 0, float("nan"), 5]) is None
        assert parse_bbox([0, 0, float("inf"), 5]) is None


class TestMedianBbox:
    def test_median_of_three(self) -> None:
        boxes = [[0, 0, 10, 10], [2, 2, 12, 12], [4, 4, 14, 14]]
        assert median_bbox(boxes) == [2.0, 2.0, 12.0, 12.0]

    def test_outlier_is_suppressed(self) -> None:
        """单个来源给出离谱框时，中位数不应被带偏（这正是选它而非均值的原因）。"""
        boxes = [[100, 100, 200, 200], [102, 101, 201, 199], [5000, 5000, 5200, 5200]]
        assert median_bbox(boxes) == [102.0, 101.0, 201.0, 200.0]

    def test_invalid_entries_ignored(self) -> None:
        assert median_bbox([[0, 0, 10, 10], "bad", [4, 4, 14, 14]]) == [2.0, 2.0, 12.0, 12.0]

    def test_all_invalid_returns_none(self) -> None:
        assert median_bbox(["bad", None, [1, 2, 3]]) is None

    def test_empty_returns_none(self) -> None:
        assert median_bbox([]) is None


class TestMatchTarget:
    def test_chinese_name_matches_english_target(self) -> None:
        assert match_target("橙色手柄螺丝刀", "screwdriver") is True

    def test_english_name_matches_chinese_target(self) -> None:
        assert match_target("screwdriver with grip", "螺丝刀") is True

    def test_unrelated_name_does_not_match(self) -> None:
        assert match_target("键盘", "screwdriver") is False
        assert match_target("mouse", "screwdriver") is False


class TestCandidatesFromReport:
    def _report(self, tmp_path):
        payload = {
            "sources": {
                "src-a": {"frames": [
                    {"file": "f0.png", "detections": [
                        {"name": "螺丝刀", "bbox_px": [100, 100, 200, 200]},
                        {"name": "键盘", "bbox_px": [0, 0, 640, 480]},
                    ]},
                    {"file": "f1.png", "detections": [{"name": "鼠标", "bbox_px": [0, 0, 10, 10]}]},
                ]},
                "src-b": {"frames": [
                    {"file": "f0.png", "detections": [
                        {"name": "screwdriver", "bbox_px": [102, 101, 201, 199]},
                    ]},
                ]},
                "src-c": {"frames": [
                    {"file": "f0.png", "detections": [
                        {"name": "螺丝刀", "bbox_px": [104, 104, 214, 214]},
                    ]},
                ]},
            }
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_cross_source_median(self, tmp_path) -> None:
        out = candidates_from_report(self._report(tmp_path), "screwdriver")
        assert list(out) == ["f0.png"]
        # 逐轴中位数：x1{100,102,104}=102, y1{100,101,104}=101,
        #            x2{200,201,214}=201, y2{200,199,214}=200
        assert out["f0.png"] == pytest.approx([102.0, 101.0, 201.0, 200.0])

    def test_frame_without_match_excluded(self, tmp_path) -> None:
        """f1 只有"鼠标" → 不该产生候选。"""
        out = candidates_from_report(self._report(tmp_path), "screwdriver")
        assert "f1.png" not in out

    def test_keyboard_larger_box_is_not_picked(self, tmp_path) -> None:
        """取面积最大的**命中目标**，而不是全帧面积最大的框。"""
        out = candidates_from_report(self._report(tmp_path), "screwdriver")
        assert out["f0.png"][2] < 300  # 没有被 0..640 的键盘框污染


class TestLoadBoxes:
    def test_missing_file_returns_empty(self, tmp_path) -> None:
        assert load_boxes(tmp_path / "nope.json") == {}
        assert load_boxes(None) == {}

    def test_ground_truth_shape(self, tmp_path) -> None:
        path = tmp_path / "gt.json"
        path.write_text(json.dumps({"target": "screwdriver",
                                    "frames": {"a.png": {"bbox_px": [1, 2, 3, 4]}}}),
                        encoding="utf-8")
        assert load_boxes(path) == {"a.png": [1.0, 2.0, 3.0, 4.0]}

    def test_bare_list_shape(self, tmp_path) -> None:
        path = tmp_path / "cand.json"
        path.write_text(json.dumps({"a.png": [1, 2, 3, 4]}), encoding="utf-8")
        assert load_boxes(path) == {"a.png": [1.0, 2.0, 3.0, 4.0]}

    def test_invalid_entries_dropped(self, tmp_path) -> None:
        path = tmp_path / "gt.json"
        path.write_text(json.dumps({"frames": {"ok.png": {"bbox_px": [1, 2, 3, 4]},
                                               "bad.png": {"bbox_px": [0, 0, 0, 0]}}}),
                        encoding="utf-8")
        assert load_boxes(path) == {"ok.png": [1.0, 2.0, 3.0, 4.0]}


class TestSaveGroundTruth:
    def test_round_trip(self, tmp_path) -> None:
        path = tmp_path / "sub" / "gt.json"
        save_ground_truth(path, "screwdriver", {"a.png": [1.4, 2.6, 3.0, 4.0]})
        loaded = load_boxes(path)
        assert loaded == {"a.png": [1.4, 2.6, 3.0, 4.0]}
        text = path.read_text(encoding="utf-8")
        assert '"target": "screwdriver"' in text

    def test_written_in_eval_compatible_format(self, tmp_path) -> None:
        """必须是 run_perception_locate_eval.py 能直接吃的结构。"""
        path = tmp_path / "gt.json"
        save_ground_truth(path, "screwdriver", {"a.png": [1, 2, 3, 4]})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"target", "frames"}
        assert data["frames"]["a.png"]["bbox_px"] == [1.0, 2.0, 3.0, 4.0]


class TestIterFrames:
    def test_only_top_level_png(self, tmp_path) -> None:
        """<frames>/eval 里的叠加图不能被当成新帧。"""
        (tmp_path / "fixed_00.png").write_bytes(b"x")
        (tmp_path / "handeye_00.png").write_bytes(b"x")
        (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "fixed_00__gui-plus.png").write_bytes(b"x")
        names = [p.name for p in iter_frames(tmp_path)]
        assert names == ["fixed_00.png", "handeye_00.png"]
