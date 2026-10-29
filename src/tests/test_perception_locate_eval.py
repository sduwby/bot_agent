"""定位源评测脚本测试 — 纯逻辑，不调 API、不需要 OpenCV。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_perception_locate_eval import (  # noqa: E402
    DEFAULT_COORD_SPACE,
    bbox_error,
    default_models,
    resolve_source,
    source_bgr,
    source_env_prefix,
)


class TestBboxError:
    def test_perfect_match(self) -> None:
        center, size = bbox_error([10, 20, 30, 40], [10, 20, 30, 40])
        assert center == pytest.approx(0.0)
        assert size == pytest.approx([0.0, 0.0])

    def test_center_offset(self) -> None:
        """框整体右移 4 px、下移 3 px → 中心误差 5 px，尺寸不变。"""
        center, size = bbox_error([14, 23, 34, 43], [10, 20, 30, 40])
        assert center == pytest.approx(5.0)
        assert size == pytest.approx([0.0, 0.0])

    def test_size_error_is_detection_minus_gt(self) -> None:
        """尺寸差 = 检测 − 真值：框偏大取正（模型外扩），偏小取负。"""
        _, bigger = bbox_error([0, 0, 40, 60], [0, 0, 30, 40])
        assert bigger == pytest.approx([10.0, 20.0])
        _, smaller = bbox_error([0, 0, 20, 30], [0, 0, 30, 40])
        assert smaller == pytest.approx([-10.0, -10.0])

    def test_center_error_is_euclidean(self) -> None:
        """3-4-5 直角三角形：dx=3, dy=4 → 中心误差 5。"""
        center, _ = bbox_error([3, 4, 13, 14], [0, 0, 10, 10])
        assert center == pytest.approx(5.0)

    def test_accepts_ints_and_tuples(self) -> None:
        center, size = bbox_error((0, 0, 10, 10), (0, 0, 10, 10))
        assert (center, size) == (0.0, [0.0, 0.0])


class TestSourceConfig:
    def test_prefix_sanitizes_model_name(self) -> None:
        assert source_env_prefix("gui-plus") == "VLM_GUI_PLUS"
        assert source_env_prefix("qwen3-vl-plus") == "VLM_QWEN3_VL_PLUS"

    def test_known_space_used_when_env_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("VLM_GUI_PLUS_COORD_SPACE", raising=False)
        assert resolve_source("gui-plus")["coord_space"] == DEFAULT_COORD_SPACE["gui-plus"]
        monkeypatch.delenv("VLM_QWEN3_VL_PLUS_COORD_SPACE", raising=False)
        assert resolve_source("qwen3-vl-plus")["coord_space"] == "normalized_1000"

    def test_env_overrides_default_space(self, monkeypatch) -> None:
        """未知模型不必改代码：只要配 VLM_<源>_COORD_SPACE 即可。"""
        monkeypatch.setenv("VLM_GUI_PLUS_COORD_SPACE", "normalized_1")
        assert resolve_source("gui-plus")["coord_space"] == "normalized_1"

    def test_unknown_model_falls_back_to_auto(self, monkeypatch) -> None:
        monkeypatch.delenv("VLM_NEW_MODEL_COORD_SPACE", raising=False)
        cfg = resolve_source("new-model")
        assert cfg["coord_space"] == "auto"
        assert cfg["model"] == "new-model"

    def test_invalid_space_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("VLM_GUI_PLUS_COORD_SPACE", "magic")
        with pytest.raises(SystemExit):
            resolve_source("gui-plus")

    def test_api_config_falls_back_to_global(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_API_BASE", "https://global/v1")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.delenv("VLM_GUI_PLUS_API_BASE", raising=False)
        monkeypatch.delenv("VLM_GUI_PLUS_API_KEY", raising=False)
        cfg = resolve_source("gui-plus")
        assert cfg["api_base"] == "https://global/v1"
        assert cfg["api_key"] == "k"

    def test_default_models_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("VLM_SOURCES", "a,b , c")
        assert default_models() == ["a", "b", "c"]
        monkeypatch.setenv("VLM_SOURCES", "")
        assert default_models() == ["gui-plus"]


class TestSourceBgr:
    def test_known_source_is_stable(self) -> None:
        assert source_bgr("gui-plus") == source_bgr("gui-plus")

    def test_unknown_source_is_deterministic_and_colored(self) -> None:
        """未知源也必须"同源同色"，否则叠加图无法人工核对。"""
        first = source_bgr("brand-new-model")
        assert first == source_bgr("brand-new-model")
        assert isinstance(first, tuple) and len(first) == 3
