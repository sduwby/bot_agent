"""固定相机对齐工具测试 — 纯逻辑（相似度/分档/图像格式转换）。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_camera_align import (  # noqa: E402
    CLOSE,
    GOOD,
    frame_similarity,
    to_hwc_u8,
    verdict,
)


def _gradient(h: int = 32, w: int = 40) -> np.ndarray:
    """非恒定图像（避免相关系数退化为 nan）。"""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    col = np.linspace(0, 255, h, dtype=np.uint8)
    gray = ((row[None, :].astype(int) + col[:, None].astype(int)) // 2).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


class TestFrameSimilarity:
    def test_identical_is_perfect(self) -> None:
        img = _gradient()
        mae, corr = frame_similarity(img, img)
        assert mae == pytest.approx(0.0)
        assert corr == pytest.approx(1.0)

    def test_inverted_is_anticorrelated(self) -> None:
        img = _gradient()
        mae, corr = frame_similarity(img, 255 - img)
        assert mae > 0
        assert corr == pytest.approx(-1.0)

    def test_resizes_mismatched_shapes(self) -> None:
        """尺寸不同时先 resize，不应抛异常。"""
        mae, corr = frame_similarity(_gradient(32, 40), _gradient(64, 80))
        assert mae >= 0.0
        assert -1.0 <= corr <= 1.0

    def test_constant_shift_lowers_correlation(self) -> None:
        img = _gradient()
        shifted = np.clip(img.astype(int) + 40, 0, 255).astype(np.uint8)
        _, corr_same = frame_similarity(img, img)
        _, corr_shift = frame_similarity(img, shifted)
        assert corr_shift < corr_same


class TestVerdict:
    def test_thresholds(self) -> None:
        assert "良好" in verdict(GOOD)
        assert "接近" in verdict((GOOD + CLOSE) / 2)
        assert "明显不一致" in verdict(CLOSE - 0.01)

    def test_boundaries_are_inclusive(self) -> None:
        assert "良好" in verdict(GOOD)
        assert "接近" in verdict(CLOSE)
        assert "明显不一致" in verdict(0.4999)


class TestToHwcU8:
    def test_chw_becomes_hwc(self) -> None:
        chw = np.zeros((3, 8, 10), dtype=np.uint8)
        assert to_hwc_u8(chw).shape == (8, 10, 3)

    def test_float_zero_one_scaled(self) -> None:
        arr = np.zeros((4, 5, 3), dtype=np.float32)
        arr[..., 0] = 1.0
        out = to_hwc_u8(arr)
        assert out.dtype == np.uint8
        assert out[..., 0].max() == 255

    def test_float_zero_255_not_rescaled(self) -> None:
        """已是 0–255 的浮点不应再被 ×255。"""
        arr = np.full((4, 5, 3), 200.0, dtype=np.float32)
        assert int(to_hwc_u8(arr).max()) == 200

    def test_uint8_passthrough(self) -> None:
        arr = np.full((4, 5, 3), 7, dtype=np.uint8)
        out = to_hwc_u8(arr)
        assert out.shape == (4, 5, 3) and out.dtype == np.uint8
