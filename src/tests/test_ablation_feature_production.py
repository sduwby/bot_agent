"""B0–B3 特征生产测试（方案③）— 纯逻辑，不跑仿真、不训练。

重点覆盖三件事：
1. **列序与变体前缀**与 `real_geometry` 的单点定义一致；
2. **特权/检测两条来源的边界**——B1/B2 是上界（sigma=0），B3 的深度必须来自检测输入，
   不能拿特权位姿反推（那是循环论证，会让"真做感知能拿多少"的对照失效）；
3. 三角化在**合成几何**上回代得到已知深度。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ablation_features import (  # noqa: E402
    GRASP_LOAD_PROXY_PERMILLE,
    DetectedInputs,
    OracleInputs,
    depth_from_pixel_size,
    oracle_features,
    stereo_depth,
    u_close,
    variant_features,
)
from real_geometry import (  # noqa: E402
    DepthEstimate,
    GeometryError,
    depth_from_known_size,
    feature_values,
    variant_feature_names,
)

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
POINT_CAM1 = np.array([0.02, 0.01, 0.5])
T_AB = np.array([[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def _project(point_cam: np.ndarray) -> list[float]:
    pixels = K @ point_cam
    return [float(pixels[0] / pixels[2]), float(pixels[1] / pixels[2])]


def _detected() -> DetectedInputs:
    point_cam2 = POINT_CAM1 - T_AB[:3, 3]
    return DetectedInputs(
        center_px_a=_project(POINT_CAM1),
        center_px_b=_project(point_cam2),
        intrinsics_a=K,
        intrinsics_b=K,
        T_ab=T_AB,
    )


def _oracle(*, grasped: bool = True, width: float = 0.01) -> OracleInputs:
    return OracleInputs(
        tcp_pos_m=(0.1, 0.0, 0.40),
        obj_pos_m=(0.1, 0.0, 0.45),
        is_grasped=grasped,
        gripper_width=width,
        open_width=0.08,
        closed_width=0.0,
    )


class TestUClose:
    def test_endpoints(self) -> None:
        assert u_close(0.08, 0.08, 0.0) == pytest.approx(0.0)
        assert u_close(0.0, 0.08, 0.0) == pytest.approx(1.0)

    def test_clamped_outside_range(self) -> None:
        assert u_close(0.2, 0.08, 0.0) == 0.0
        assert u_close(-0.5, 0.08, 0.0) == 1.0


class TestOracleInputsValidation:
    def test_bad_tcp_shape(self) -> None:
        with pytest.raises(GeometryError, match="tcp_pos_m"):
            OracleInputs(tcp_pos_m=(1.0, 2.0), obj_pos_m=(0, 0, 0), is_grasped=False,
                         gripper_width=0.01, open_width=0.08, closed_width=0.0)

    def test_open_must_exceed_closed(self) -> None:
        with pytest.raises(GeometryError, match="open_width"):
            OracleInputs(tcp_pos_m=(0, 0, 0), obj_pos_m=(0, 0, 0), is_grasped=False,
                         gripper_width=0.01, open_width=0.0, closed_width=0.1)


class TestOracleFeatures:
    def test_b1_only_produces_depth_columns(self) -> None:
        values = oracle_features("B1", _oracle())
        assert set(values) == {"depth_known_size_m", "depth_known_size_sigma_m"}

    def test_b1_is_the_tcp_object_distance_with_zero_sigma(self) -> None:
        """特权上界：不确定性记 0 —— 这正是"上界"的含义，必须与检测路径的 sigma 区分。"""
        values = oracle_features("B1", _oracle())
        assert values["depth_known_size_m"] == pytest.approx(0.05)
        assert values["depth_known_size_sigma_m"] == 0.0

    def test_b2_adds_contact_and_load_proxy(self) -> None:
        values = oracle_features("B2", _oracle(grasped=True, width=0.04))
        assert values["contact_flag"] == 1.0
        assert values["load_gripper_permille"] == pytest.approx(
            GRASP_LOAD_PROXY_PERMILLE * 0.5
        )

    def test_not_grasped_means_zero_load(self) -> None:
        """未夹住时负载是 0（没有负载），不是"更小的负数"。"""
        values = oracle_features("B2", _oracle(grasped=False, width=0.0))
        assert values["contact_flag"] == 0.0
        assert values["load_gripper_permille"] == 0.0

    def test_b0_has_no_oracle_columns(self) -> None:
        assert oracle_features("B0", _oracle()) == {}

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(GeometryError):
            oracle_features("B9", _oracle())


class TestDetectedPaths:
    def test_depth_from_bbox_matches_real_geometry(self) -> None:
        """检测路径只是 `real_geometry` 的薄封装，数值必须逐位一致（否则两处口径会漂移）。"""
        size_px = (100.0, 100.0)
        values = depth_from_pixel_size(size_px, real_size_m=0.15, fx_px=500.0)
        reference = feature_values(depth=depth_from_known_size(size_px, 0.15, 500.0))
        assert values == pytest.approx(reference)

    def test_stereo_recovers_known_depth(self) -> None:
        """合成几何回代：三角化应恢复出 0.5 m。"""
        values = stereo_depth(_detected())
        assert values["depth_stereo_m"] == pytest.approx(0.5, abs=1e-6)
        assert values["depth_stereo_sigma_m"] >= 0.0

    def test_detected_inputs_validation(self) -> None:
        with pytest.raises(GeometryError, match="center_px_a"):
            DetectedInputs(center_px_a=(1.0,), center_px_b=(1.0, 2.0),
                           intrinsics_a=K, intrinsics_b=K, T_ab=T_AB)


class TestVariantFeatures:
    def test_b0_is_empty(self) -> None:
        assert variant_features("B0").shape == (0,)

    def test_b1_shape_and_order(self) -> None:
        vector = variant_features("B1", oracle=_oracle())
        assert vector.shape == (len(variant_feature_names("B1")),)

    def test_b2_without_oracle_raises(self) -> None:
        with pytest.raises(GeometryError, match="oracle"):
            variant_features("B2")

    def test_b3_without_detected_raises(self) -> None:
        """核心防循环论证检查：B3 的深度列不许由特权位姿顶替。"""
        with pytest.raises(GeometryError, match="B3 需要检测路径输入"):
            variant_features("B3", oracle=_oracle())

    def test_b3_is_mixed_source(self) -> None:
        """方案③：B1/B2 的列来自特权量、B3 自己的深度列来自检测。"""
        vector = variant_features("B3", oracle=_oracle(), detected=_detected())
        names = variant_feature_names("B3")
        assert vector.shape == (6,)
        index = {name: position for position, name in enumerate(names)}
        assert vector[index["depth_known_size_m"]] == pytest.approx(0.05)
        assert vector[index["depth_known_size_sigma_m"]] == pytest.approx(0.0)
        assert vector[index["depth_stereo_m"]] == pytest.approx(0.5, abs=1e-6)
        assert vector[index["depth_stereo_sigma_m"]] > 0.0

    def test_perturbing_oracle_does_not_move_the_stereo_columns(self) -> None:
        """B3 的检测列必须**只**由检测输入决定：动了特权位姿它们不该变。"""
        first = variant_features("B3", oracle=_oracle(), detected=_detected())
        moved = OracleInputs(tcp_pos_m=(1.0, 1.0, 0.1), obj_pos_m=(0.0, 0.0, 0.0),
                             is_grasped=True, gripper_width=0.01,
                             open_width=0.08, closed_width=0.0)
        second = variant_features("B3", oracle=moved, detected=_detected())
        names = variant_feature_names("B3")
        index = {name: position for position, name in enumerate(names)}
        for name in ("depth_stereo_m", "depth_stereo_sigma_m"):
            assert second[index[name]] == pytest.approx(first[index[name]])
        assert second[index["depth_known_size_m"]] != pytest.approx(
            first[index["depth_known_size_m"]]
        )
