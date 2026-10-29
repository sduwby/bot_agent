"""real_perception 像素级 2D 定位测试 — 纯逻辑，无机械臂 / 无模型 API。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from real_perception import (  # noqa: E402
    Detection,
    FakeObjectLocator,
    LocalizationResult,
    VLMObjectLocator,
    locate_objects,
)


class FakeBackend:
    """按预设字符串回复的多模态后端。"""

    def __init__(self, reply: str | None = None, error: Exception | None = None,
                 payload: dict | None = None) -> None:
        self.reply = reply if reply is not None else json.dumps(payload or {})
        self.error = error
        self.calls = 0
        self.last_images: list | None = None

    def chat(self, prompt: str, user: str, images=None):
        self.calls += 1
        self.last_images = images
        if self.error is not None:
            raise self.error
        return self.reply


class TestDetection:
    def test_center_and_size(self) -> None:
        det = Detection(name="cube", bbox_px=(10.0, 20.0, 30.0, 60.0))
        assert det.center_px == pytest.approx((20.0, 40.0))
        assert det.size_px == pytest.approx((20.0, 40.0))

    def test_name_is_stripped(self) -> None:
        assert Detection(name="  cube  ", bbox_px=(0.0, 0.0, 1.0, 1.0)).name == "cube"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError):
            Detection(name="   ", bbox_px=(0.0, 0.0, 1.0, 1.0))

    def test_bad_bbox_length_raises(self) -> None:
        with pytest.raises(ValueError):
            Detection(name="cube", bbox_px=(0.0, 0.0, 1.0))

    def test_reversed_bbox_raises(self) -> None:
        with pytest.raises(ValueError):
            Detection(name="cube", bbox_px=(10.0, 0.0, 10.0, 5.0))
        with pytest.raises(ValueError):
            Detection(name="cube", bbox_px=(10.0, 5.0, 20.0, 5.0))

    def test_non_finite_bbox_raises(self) -> None:
        with pytest.raises(ValueError):
            Detection(name="cube", bbox_px=(0.0, 0.0, float("nan"), 5.0))

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            Detection(name="cube", bbox_px=(0.0, 0.0, 1.0, 1.0), confidence=1.5)
        with pytest.raises(ValueError):
            Detection(name="cube", bbox_px=(0.0, 0.0, 1.0, 1.0), confidence=-0.1)


class TestLocateObjects:
    def test_no_images_returns_none_source(self) -> None:
        result = locate_objects(FakeBackend(payload={}), None)
        assert result.detections == []
        assert result.source == "none"
        assert result.confidence == 0.0

    def test_backend_error_degrades(self) -> None:
        result = locate_objects(FakeBackend(error=OSError("网络不可达")), ["f"])
        assert result.detections == []
        assert result.source == "none"
        assert "2D 定位失败" in result.note

    def test_non_json_reply_degrades(self) -> None:
        result = locate_objects(FakeBackend(reply="我看不清"), ["f"])
        assert result.source == "none"

    def test_markdown_fenced_json_is_parsed(self) -> None:
        """视觉模型普遍用 ```json 围栏包裹回复，不能被判成解析失败。"""
        fenced = '```json\n{"objects": [{"name": "cube", "bbox_px": [1, 2, 3, 4]}]}\n```'
        result = locate_objects(FakeBackend(reply=fenced), ["f"])
        assert [d.name for d in result.detections] == ["cube"]

    def test_json_with_surrounding_prose_is_parsed(self) -> None:
        messy = '好的，我看到：\n{"objects": [{"name": "cube", "bbox_px": [1, 2, 3, 4]}]}\n以上。'
        result = locate_objects(FakeBackend(reply=messy), ["f"])
        assert [d.name for d in result.detections] == ["cube"]

    def test_braces_inside_string_do_not_break_scan(self) -> None:
        messy = 'note\n{"objects": [{"name": "a{b}c", "bbox_px": [1, 2, 3, 4]}]}\ntail'
        result = locate_objects(FakeBackend(reply=messy), ["f"])
        assert [d.name for d in result.detections] == ["a{b}c"]

    def test_truncated_json_degrades(self) -> None:
        result = locate_objects(FakeBackend(reply='{"objects": [{"name": "cube"'), ["f"])
        assert result.detections == []
        assert result.source == "none"

    def test_json_not_object_degrades(self) -> None:
        result = locate_objects(FakeBackend(reply="[1, 2]"), ["f"])
        assert result.source == "none"

    def test_missing_objects_field(self) -> None:
        result = locate_objects(FakeBackend(payload={"note": "空"}), ["f"])
        assert result.detections == []
        assert result.source == "vision_llm"
        assert "objects 字段缺失" in result.note

    def test_objects_not_list(self) -> None:
        result = locate_objects(FakeBackend(payload={"objects": "cube"}), ["f"])
        assert result.detections == []

    def test_valid_entries_are_parsed(self) -> None:
        payload = {"objects": [
            {"name": "screwdriver", "bbox_px": [100, 200, 140, 210], "confidence": 0.8},
            {"name": "cube", "bbox_px": [300, 250, 340, 290], "confidence": 0.6},
        ], "note": "两个物体"}
        result = locate_objects(FakeBackend(payload=payload), ["f"])
        assert [d.name for d in result.detections] == ["screwdriver", "cube"]
        assert result.detections[0].center_px == pytest.approx((120.0, 205.0))
        assert result.confidence == pytest.approx(0.7)
        assert result.skipped == 0
        assert result.note == "两个物体"

    def test_missing_confidence_defaults_to_zero(self) -> None:
        payload = {"objects": [{"name": "cube", "bbox_px": [0, 0, 10, 10]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"])
        assert result.detections[0].confidence == pytest.approx(0.0)
        assert result.confidence == pytest.approx(0.0)

    def test_invalid_entries_are_skipped_and_counted(self) -> None:
        payload = {"objects": [
            {"name": "ok", "bbox_px": [0, 0, 10, 10]},
            {"name": "bad_len", "bbox_px": [0, 0, 10]},
            {"name": "bad_order", "bbox_px": [10, 10, 0, 0]},
            {"name": "   ", "bbox_px": [0, 0, 10, 10]},
            {"bbox_px": [0, 0, 10, 10]},
            "not-a-dict",
        ]}
        result = locate_objects(FakeBackend(payload=payload), ["f"])
        assert [d.name for d in result.detections] == ["ok"]
        assert result.skipped == 5
        assert "丢弃 5 个" in result.note

    def test_mild_edge_overshoot_is_clamped_not_dropped(self) -> None:
        """实测真实帧上模型常有几像素边缘过冲（gui-plus 给 x2=644 而图宽 640）
        → 应裁剪保留，整条丢弃会低估模型。」"""
        payload = {"objects": [{"name": "bench", "bbox_px": [0, 339, 644, 476]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"], frame_size=(640, 480),
                                coord_space="pixel")
        assert result.skipped == 0
        assert result.clamped == 1
        assert result.detections[0].bbox_px == pytest.approx((0.0, 339.0, 640.0, 476.0))
        assert "裁剪" in result.note

    def test_excessive_overshoot_is_dropped(self) -> None:
        payload = {"objects": [{"name": "far", "bbox_px": [2000, 0, 2200, 100]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"], frame_size=(640, 480),
                                coord_space="pixel")
        assert result.detections == []
        assert result.skipped == 1

    def test_degenerate_after_clamp_is_dropped(self) -> None:
        """整框落在画面外 → 裁剪后退化，丢弃。"""
        payload = {"objects": [{"name": "outside", "bbox_px": [700, 10, 750, 60]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"], frame_size=(640, 480),
                                coord_space="pixel")
        assert result.detections == []
        assert result.skipped == 1

    def test_frame_boundary_is_inclusive(self) -> None:
        payload = {"objects": [{"name": "edge", "bbox_px": [0, 0, 640, 480]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"], frame_size=(640, 480))
        assert len(result.detections) == 1

    def test_no_frame_size_skips_bounds_check(self) -> None:
        payload = {"objects": [{"name": "far", "bbox_px": [9000, 9000, 9100, 9100]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"])
        assert len(result.detections) == 1

    def test_query_is_forwarded(self) -> None:
        backend = FakeBackend(payload={"objects": []})
        locate_objects(backend, ["f"], query="红色方块")
        assert backend.last_images == ["f"]
        assert backend.calls == 1


class TestCoordSpace:
    """模型坐标约定是"先验"，prompt 约束不可靠（Qwen-VL 官方即用 [0,1000)）。"""

    def test_normalized_1000_is_deduced_when_pixel_is_impossible(self) -> None:
        """坐标 526 > 图高 480 —— 不可能是像素，只能是 [0,1000) 归一化。"""
        payload = {"objects": [{"name": "cube", "bbox_px": [304, 304, 438, 526]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"], frame_size=(640, 480))
        assert result.coord_space == "normalized_1000"
        assert result.skipped == 0
        det = result.detections[0]
        assert det.bbox_px == pytest.approx((194.56, 145.92, 280.32, 252.48), abs=0.01)
        assert det.center_px == pytest.approx((237.44, 199.2), abs=0.01)
        assert "normalized_1000" in result.note

    def test_explicit_normalized_1000_resolves_ambiguity(self) -> None:
        """归一化坐标恰好落在图像尺寸内时，auto 会误判为像素 → 已知约定须显式指定。"""
        payload = {"objects": [{"name": "cube", "bbox_px": [100, 100, 400, 300]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"],
                                frame_size=(640, 480), coord_space="normalized_1000")
        assert result.coord_space == "normalized_1000"
        assert result.detections[0].bbox_px == pytest.approx((64.0, 48.0, 256.0, 144.0))

    def test_auto_keeps_ambiguous_case_as_pixel(self) -> None:
        """记录已知局限：auto 在歧义场景按像素解读。"""
        payload = {"objects": [{"name": "cube", "bbox_px": [100, 100, 400, 300]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"], frame_size=(640, 480))
        assert result.coord_space == "pixel"
        assert result.detections[0].bbox_px == pytest.approx((100.0, 100.0, 400.0, 300.0))

    def test_auto_tolerates_small_overshoot_as_pixel(self) -> None:
        """4 px 边缘过冲（644 vs 640，仅 +0.6%）在 auto 下必须仍判像素并裁剪，
        而不是被误判成 normalized_1000（那会把垃圾框"救"成看似合理的框）。"""
        payload = {"objects": [{"name": "bench", "bbox_px": [0, 339, 644, 476]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"], frame_size=(640, 480))
        assert result.coord_space == "pixel"
        assert result.clamped == 1
        assert result.detections[0].bbox_px == pytest.approx((0.0, 339.0, 640.0, 476.0))

    def test_normalized_1_is_detected(self) -> None:
        payload = {"objects": [{"name": "cube", "bbox_px": [0.25, 0.5, 0.5, 0.75]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"], frame_size=(640, 480))
        assert result.coord_space == "normalized_1"
        assert result.detections[0].bbox_px == pytest.approx((160.0, 240.0, 320.0, 360.0))

    def test_normalized_without_frame_size_is_rejected(self) -> None:
        """归一化坐标缺 frame_size 时无法换算，明确拒绝而不是猜一个尺寸。"""
        payload = {"objects": [{"name": "cube", "bbox_px": [0.25, 0.5, 0.5, 0.75]}]}
        result = locate_objects(FakeBackend(payload=payload), ["f"])
        assert result.detections == []
        assert result.coord_space == "normalized_1"
        assert "无法换算" in result.note

    def test_invalid_coord_space_raises(self) -> None:
        with pytest.raises(ValueError):
            locate_objects(FakeBackend(payload={}), ["f"], coord_space="magic")

    def test_vlm_locator_forwards_coord_space(self) -> None:
        payload = {"objects": [{"name": "cube", "bbox_px": [100, 100, 400, 300]}]}
        locator = VLMObjectLocator(FakeBackend(payload=payload), frame_size=(640, 480),
                                   coord_space="normalized_1000")
        result = locator.locate(["f"])
        assert result.coord_space == "normalized_1000"
        assert result.detections[0].bbox_px == pytest.approx((64.0, 48.0, 256.0, 144.0))


class TestLocatorImplementations:
    def test_vlm_locator_delegates(self) -> None:
        payload = {"objects": [{"name": "cube", "bbox_px": [0, 0, 4, 4], "confidence": 0.5}]}
        locator = VLMObjectLocator(FakeBackend(payload=payload), frame_size=(640, 480))
        result = locator.locate(["f"])
        assert isinstance(result, LocalizationResult)
        assert result.detections[0].name == "cube"

    def test_fake_locator_returns_preset(self) -> None:
        preset = [Detection(name="cube", bbox_px=(0.0, 0.0, 4.0, 4.0))]
        locator = FakeObjectLocator(preset)
        result = locator.locate(["f"])
        assert locator.calls == 1
        assert result.detections == preset
        assert result.source == "fake"

    def test_fake_locator_can_raise(self) -> None:
        locator = FakeObjectLocator(error=RuntimeError("检测器不可用"))
        with pytest.raises(RuntimeError):
            locator.locate(["f"])
