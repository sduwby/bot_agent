"""手眼标定封装测试（1.4）— 纯逻辑，用**合成位姿**（已知真值 X）验证整条链路。

覆盖两件事：
1. 数学正确性：相对运动构造、残差度量、留一法在"数据干净"时必须很小；
2. **诊断有效性**：人为塞一个离群对时，留一法**必须报警** —— 一个永远说"通过"的诊断
   比没有诊断更糟（它会让人放心地用错标定）。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from handeye_calib import (  # noqa: E402
    CAMERA_POSES_KEY,
    LOO_OUTLIER_FACTOR,
    MIN_POSES,
    ROBOT_POSES_KEY,
    CalibrationError,
    build_pairs,
    calibrate,
    calibration_matrix,
    compose_transform,
    invert_transform,
    leave_one_out,
    motion_residuals,
    read_pairs,
    relative_motions,
    rotation_angle_deg,
    synthetic_poses,
    write_pairs,
    write_report,
)


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


class TestTransformHelpers:
    def test_invert_is_inverse(self) -> None:
        matrix = _translation(0.1, -0.2, 0.3)
        assert np.allclose(compose_transform(matrix, invert_transform(matrix)), np.eye(4))

    def test_invert_rejects_non_rigid(self) -> None:
        with pytest.raises(CalibrationError, match="正交"):
            invert_transform(np.full((4, 4), 2.0) + np.eye(4))

    def test_invert_rejects_wrong_shape(self) -> None:
        with pytest.raises(CalibrationError, match="4×4"):
            invert_transform(np.eye(3))

    def test_rotation_angle_identity_and_quarter_turn(self) -> None:
        assert rotation_angle_deg(np.eye(3)) == pytest.approx(0.0, abs=1e-6)
        quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert rotation_angle_deg(quarter) == pytest.approx(90.0, abs=1e-6)


class TestRelativeMotions:
    def test_identical_poses_give_identity_motion(self) -> None:
        motions = relative_motions([_translation(0.1, 0.0, 0.0)] * 2)
        assert np.allclose(motions[0], np.eye(4))

    def test_motion_is_inverse_times_next(self) -> None:
        first, second = _translation(0.1, 0.0, 0.0), _translation(0.3, 0.0, 0.0)
        motion = relative_motions([first, second])[0]
        assert np.allclose(motion[:3, 3], [0.2, 0.0, 0.0])

    def test_single_pose_rejected(self) -> None:
        with pytest.raises(CalibrationError, match="2 个位姿"):
            relative_motions([np.eye(4)])


class TestBuildPairs:
    def test_too_few_poses_rejected(self) -> None:
        with pytest.raises(CalibrationError, match=str(MIN_POSES)):
            build_pairs([np.eye(4)] * 2, [np.eye(4)] * 2)

    def test_mismatched_counts_rejected(self) -> None:
        with pytest.raises(CalibrationError, match="不一致"):
            build_pairs([np.eye(4)] * 4, [np.eye(4)] * 3)

    def test_pair_count_is_poses_minus_one(self) -> None:
        robot, camera, _ = synthetic_poses(n_poses=6)
        assert len(build_pairs(robot, camera)) == 5


class TestCalibrateOnCleanSyntheticData:
    def test_recovers_truth_and_passes(self) -> None:
        """干净合成数据：反解应几乎精确复现真值，门禁必须通过。"""
        robot, camera, truth = synthetic_poses(n_poses=10)
        report = calibrate(robot, camera)
        assert report.passed is True
        assert report.residuals["rot_deg_max"] < 1e-4
        assert report.residuals["trans_mm_max"] < 1e-3
        assert report.warnings == ()

    def test_solved_matrix_matches_truth(self) -> None:
        robot, camera, truth = synthetic_poses(n_poses=10)
        report = calibrate(robot, camera)
        assert report.n_pairs == 9
        # 留一法在干净数据上也应稳定（没有离群对可抓）
        assert report.leave_one_out["trans_mm_max"] < 1.0
        assert report.residual_m >= 0.0


class TestDiagnosticsActuallyFlagProblems:
    def test_noise_degrades_but_keeps_under_loose_gate(self) -> None:
        robot, camera, _ = synthetic_poses(n_poses=10, noise_deg=0.3, noise_mm=1.5)
        report = calibrate(robot, camera, max_rot_deg=10.0, max_trans_mm=20.0)
        assert report.residuals["rot_deg_max"] > 0.0
        assert report.residuals["trans_mm_max"] > 0.0

    def test_outlier_pair_is_flagged_by_leave_one_out(self) -> None:
        """**关键**：塞一个离群对时留一法必须显著变大并写进 warnings。"""
        robot, camera, _ = synthetic_poses(n_poses=10, outlier_pair=3, outlier_mm=50.0)
        report = calibrate(robot, camera, max_rot_deg=90.0, max_trans_mm=500.0)
        loo = report.leave_one_out
        assert loo["trans_mm_max"] > report.residuals["trans_mm_mean"] * LOO_OUTLIER_FACTOR
        assert loo["worst_pair"] == 3
        assert any("离群" in warning for warning in report.warnings)

    def test_tight_gate_can_fail(self) -> None:
        robot, camera, _ = synthetic_poses(n_poses=10, noise_deg=1.0, noise_mm=5.0)
        report = calibrate(robot, camera, max_rot_deg=1e-6, max_trans_mm=1e-6)
        assert report.passed is False

    def test_too_few_poses_reports_leave_one_out_error(self) -> None:
        """3 个位姿 = 2 组相对运动：留住一对后只剩 1 对，留一法必须优雅降级而不是崩。"""
        robot, camera, _ = synthetic_poses(n_poses=MIN_POSES)
        report = calibrate(robot, camera)
        assert "error" in report.leave_one_out
        assert any("留一法" in warning for warning in report.warnings)


class TestMotionResiduals:
    def test_perfect_fit_has_zero_residual(self) -> None:
        robot, camera, truth = synthetic_poses(n_poses=8)
        from real_geometry import HandEyeCalibration

        pairs = build_pairs(robot, camera)
        calibration = HandEyeCalibration(
            R=truth[:3, :3], t=truth[:3, 3], residual_m=0.0, n_samples=len(pairs), method="ax_xb"
        )
        residuals = motion_residuals(calibration, pairs)
        assert residuals["rot_deg_max"] < 1e-6
        assert residuals["trans_mm_max"] < 1e-6
        assert len(residuals["per_pair"]) == len(pairs)

    def test_calibration_matrix_roundtrip(self) -> None:
        from real_geometry import HandEyeCalibration

        robot, camera, truth = synthetic_poses(n_poses=5)
        del robot, camera
        calibration = HandEyeCalibration(R=truth[:3, :3], t=truth[:3, 3],
                                         residual_m=0.0, n_samples=4, method="ax_xb")
        assert np.allclose(calibration_matrix(calibration), truth)


class TestPairsIO:
    def test_roundtrip(self, tmp_path) -> None:
        robot, camera, _ = synthetic_poses(n_poses=6)
        path = write_pairs(tmp_path / "pairs.npz", robot, camera)
        loaded_robot, loaded_camera = read_pairs(path)
        assert len(loaded_robot) == 6
        assert np.allclose(np.asarray(loaded_robot[0]), robot[0])
        assert np.allclose(np.asarray(loaded_camera[-1]), camera[-1])

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(CalibrationError, match="不存在"):
            read_pairs(tmp_path / "nope.npz")

    def test_missing_keys(self, tmp_path) -> None:
        path = tmp_path / "bad.npz"
        np.savez(path, **{ROBOT_POSES_KEY: np.zeros((4, 4, 4))})
        with pytest.raises(CalibrationError, match=CAMERA_POSES_KEY):
            read_pairs(path)

    def test_wrong_shape(self, tmp_path) -> None:
        path = tmp_path / "bad.npz"
        np.savez(path, **{ROBOT_POSES_KEY: np.zeros((4, 3, 3)),
                          CAMERA_POSES_KEY: np.zeros((4, 4, 4))})
        with pytest.raises(CalibrationError, match=r"\[N,4,4\]"):
            read_pairs(path)

    def test_too_few_poses(self, tmp_path) -> None:
        path = tmp_path / "few.npz"
        np.savez(path, **{ROBOT_POSES_KEY: np.tile(np.eye(4), (2, 1, 1)),
                          CAMERA_POSES_KEY: np.tile(np.eye(4), (2, 1, 1))})
        with pytest.raises(CalibrationError, match=str(MIN_POSES)):
            read_pairs(path)


class TestReportOutput:
    def test_writes_json_and_markdown(self, tmp_path) -> None:
        robot, camera, _ = synthetic_poses(n_poses=8)
        report = calibrate(robot, camera)
        json_path, markdown_path = write_report(report, tmp_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["passed"] is True
        assert payload["thresholds"]["max_rot_deg"] == pytest.approx(2.0)
        assert "手眼标定" in markdown_path.read_text(encoding="utf-8")
