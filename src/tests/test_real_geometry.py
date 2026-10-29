"""real_geometry 几何深度与手眼标定测试 — 纯逻辑，无真机/仿真框架。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from real_geometry import (  # noqa: E402
    DepthEstimate,
    GeometryError,
    HandEyeCalibration,
    depth_from_known_size,
    solve_handeye,
    triangulate,
)
from se3 import rotation_matrix_from_rpy  # noqa: E402

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _rigid(rotation: np.ndarray, translation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def _project(intrinsics: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = intrinsics @ point
    return homogeneous[:2] / homogeneous[2]


class TestDepthFromKnownSize:
    def test_pinhole_formula(self) -> None:
        estimate = depth_from_known_size((60.0, 60.0), real_size_m=0.05, fx=600.0)
        assert estimate.z_m == pytest.approx(0.5)
        assert estimate.method == "known_size"
        assert estimate.point_m is None

    def test_uses_larger_bbox_side(self) -> None:
        """像素参考尺寸取包围盒较大边。"""
        estimate = depth_from_known_size((40.0, 60.0), real_size_m=0.05, fx=600.0)
        assert estimate.z_m == pytest.approx(600.0 * 0.05 / 60.0)

    def test_farther_object_is_deeper(self) -> None:
        near = depth_from_known_size((80.0, 80.0), real_size_m=0.05, fx=600.0)
        far = depth_from_known_size((20.0, 20.0), real_size_m=0.05, fx=600.0)
        assert far.z_m > near.z_m

    def test_uncertainty_scales_with_bbox_sigma(self) -> None:
        estimate = depth_from_known_size((50.0, 50.0), 0.05, 600.0, bbox_sigma_px=2.0)
        assert estimate.uncertainty_m == pytest.approx(estimate.z_m * 2.0 / 50.0)

    def test_zero_sigma_gives_zero_uncertainty(self) -> None:
        estimate = depth_from_known_size((50.0, 50.0), 0.05, 600.0, bbox_sigma_px=0.0)
        assert estimate.uncertainty_m == pytest.approx(0.0)

    def test_non_positive_bbox_raises(self) -> None:
        with pytest.raises(ValueError):
            depth_from_known_size((0.0, 10.0), 0.05, 600.0)
        with pytest.raises(ValueError):
            depth_from_known_size((10.0, -1.0), 0.05, 600.0)

    def test_non_positive_physical_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            depth_from_known_size((10.0, 10.0), 0.0, 600.0)
        with pytest.raises(ValueError):
            depth_from_known_size((10.0, 10.0), 0.05, 0.0)

    def test_malformed_bbox_raises(self) -> None:
        with pytest.raises(ValueError):
            depth_from_known_size((10.0,), 0.05, 600.0)
        with pytest.raises(ValueError):
            depth_from_known_size((10.0, 10.0, 10.0), 0.05, 600.0)


class TestTriangulate:
    def test_recovers_known_point(self) -> None:
        """相机 2 相对相机 1 平移 5 cm、姿态一致 → 还原相机 1 系三维点。"""
        point = np.array([0.01, 0.02, 0.5])
        transform = _rigid(np.eye(3), [0.05, 0.0, 0.0])
        p1 = _project(K, point)
        p2 = _project(K, point - transform[:3, 3])

        estimate = triangulate(p1, p2, K, K, transform)

        assert estimate.method == "stereo"
        assert estimate.z_m == pytest.approx(0.5, abs=1e-9)
        assert estimate.point_m is not None
        assert np.allclose(estimate.point_m, point, atol=1e-9)
        assert estimate.uncertainty_m == pytest.approx(0.0, abs=1e-9)

    def test_depth_increases_with_distance(self) -> None:
        transform = _rigid(np.eye(3), [0.05, 0.0, 0.0])
        depths = []
        for z in (0.3, 0.6, 0.9):
            point = np.array([0.0, 0.0, z])
            p1 = _project(K, point)
            p2 = _project(K, point - transform[:3, 3])
            depths.append(triangulate(p1, p2, K, K, transform).z_m)
        assert depths[0] < depths[1] < depths[2]

    def test_uncertainty_reflects_ray_inconsistency(self) -> None:
        """人为在下方向打偏一路像素（离开极线平面）→ 两射线不再相交，不确定度应 > 0。"""
        point = np.array([0.0, 0.0, 0.5])
        transform = _rigid(np.eye(3), [0.05, 0.0, 0.0])
        p1 = _project(K, point)
        p2 = _project(K, point - transform[:3, 3]) + np.array([0.0, 8.0])

        estimate = triangulate(p1, p2, K, K, transform)
        assert estimate.z_m > 0
        assert estimate.uncertainty_m > 1e-6

    def test_parallel_rays_behind_camera_raise(self) -> None:
        """两光心仅横向平移、射线平行 → 深度不可解（z<=0）。"""
        transform = _rigid(np.eye(3), [0.05, 0.0, 0.0])
        principal = np.array([320.0, 240.0])
        with pytest.raises(GeometryError):
            triangulate(principal, principal, K, K, transform)

    def test_non_rigid_transform_raises(self) -> None:
        bad = np.eye(4)
        bad[:3, :3] = np.eye(3) * 2.0  # 非正交
        with pytest.raises(ValueError):
            triangulate((320.0, 240.0), (320.0, 240.0), K, K, bad)

    def test_singular_intrinsics_raise(self) -> None:
        singular = np.zeros((3, 3))
        with pytest.raises(GeometryError):
            triangulate((320.0, 240.0), (320.0, 240.0), singular, K, np.eye(4))

    def test_malformed_pixel_raises(self) -> None:
        with pytest.raises(ValueError):
            triangulate((320.0,), (320.0, 240.0), K, K, np.eye(4))


class TestSolveHandEye:
    @staticmethod
    def _make_pairs(rotation_x: np.ndarray, translation_x, count: int = 5, seed: int = 0):
        """构造满足 A X = X B 的相对运动对：B = X⁻¹ A X。"""
        rng = np.random.default_rng(seed)
        transform_x = _rigid(rotation_x, translation_x)
        inverse_x = np.linalg.inv(transform_x)
        pairs = []
        for _ in range(count):
            roll, pitch, yaw = rng.uniform(-0.6, 0.6, 3)
            motion_a = _rigid(rotation_matrix_from_rpy(roll, pitch, yaw),
                              rng.uniform(-0.05, 0.05, 3))
            pairs.append((motion_a, inverse_x @ motion_a @ transform_x))
        return pairs

    def test_recovers_known_transform(self) -> None:
        rotation_x = rotation_matrix_from_rpy(0.1, -0.2, 0.3)
        translation_x = np.array([0.03, -0.02, 0.05])
        pairs = self._make_pairs(rotation_x, translation_x)

        result = solve_handeye(pairs)

        assert np.allclose(result.R, rotation_x, atol=1e-8)
        assert np.allclose(result.t, translation_x, atol=1e-8)
        assert result.residual_m == pytest.approx(0.0, abs=1e-9)
        assert result.n_samples == 5
        assert result.method == "ax_xb"
        assert result.matrix().shape == (4, 4)

    def test_method_is_recorded(self) -> None:
        rotation_x = rotation_matrix_from_rpy(0.0, 0.0, 0.2)
        pairs = self._make_pairs(rotation_x, [0.01, 0.0, 0.0])
        assert solve_handeye(pairs, method="aruco").method == "aruco"

    def test_too_few_pairs_raise(self) -> None:
        rotation_x = rotation_matrix_from_rpy(0.0, 0.0, 0.2)
        pairs = self._make_pairs(rotation_x, [0.01, 0.0, 0.0], count=1)
        with pytest.raises(GeometryError):
            solve_handeye(pairs)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError):
            solve_handeye([(np.eye(4), np.eye(4))] * 2, method="magic")

    def test_translation_only_motions_are_degenerate(self) -> None:
        """全是纯平移（无转动分量）→ 旋转不可辨识。"""
        transform_x = _rigid(rotation_matrix_from_rpy(0.1, 0.1, 0.1), [0.01, 0.02, 0.03])
        inverse_x = np.linalg.inv(transform_x)
        pairs = []
        for shift in (0.01, -0.02, 0.03):
            motion_a = _rigid(np.eye(3), [shift, 0.0, 0.0])
            pairs.append((motion_a, inverse_x @ motion_a @ transform_x))
        with pytest.raises(GeometryError):
            solve_handeye(pairs)

    def test_malformed_pair_raises(self) -> None:
        with pytest.raises(ValueError):
            solve_handeye([np.eye(4), np.eye(4)])  # 不是 (A, B) 二元组

    def test_non_rigid_motion_raises(self) -> None:
        bad = np.eye(4)
        bad[0, 0] = 2.0
        with pytest.raises(ValueError):
            solve_handeye([(bad, np.eye(4)), (np.eye(4), np.eye(4))])


class TestResultValidation:
    def test_depth_estimate_rejects_negative_uncertainty(self) -> None:
        with pytest.raises(ValueError):
            DepthEstimate(z_m=0.5, method="known_size", uncertainty_m=-1.0)

    def test_handeye_rejects_non_orthogonal_rotation(self) -> None:
        with pytest.raises(ValueError):
            HandEyeCalibration(R=np.eye(3) * 2.0, t=np.zeros(3),
                               residual_m=0.0, n_samples=3, method="ax_xb")

    def test_handeye_rejects_reflection(self) -> None:
        reflection = np.diag([1.0, 1.0, -1.0])
        with pytest.raises(ValueError):
            HandEyeCalibration(R=reflection, t=np.zeros(3),
                               residual_m=0.0, n_samples=3, method="ax_xb")

    def test_handeye_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError):
            HandEyeCalibration(R=np.eye(3), t=np.zeros(3),
                               residual_m=0.0, n_samples=3, method="magic")

    def test_handeye_rejects_single_sample(self) -> None:
        with pytest.raises(ValueError):
            HandEyeCalibration(R=np.eye(3), t=np.zeros(3),
                               residual_m=0.0, n_samples=1, method="ax_xb")
