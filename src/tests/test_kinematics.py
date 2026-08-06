"""二维双连杆机械臂正逆运动学测试。"""

from __future__ import annotations

from math import pi

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kinematics import (
    IKSolution,
    JointLimit,
    JointLimitError,
    PlanarArm2D,
    UnreachableTargetError,
)
from run_ik_demo import DEMO_TARGETS, build_demo_arm, validate_ik_demo


def _arm(
    limits: tuple[JointLimit, JointLimit] | None = None,
) -> PlanarArm2D:
    return PlanarArm2D(
        (0.6, 0.4),
        limits or (JointLimit(-pi, pi), JointLimit(-pi, pi)),
    )


def test_joint_limit_contains_boundaries() -> None:
    limit = JointLimit(-1.0, 1.0)

    assert limit.contains(-1.0)
    assert limit.contains(1.0)
    assert not limit.contains(1.1)
    assert not limit.contains(np.nan)


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (0.0, 0.0, "下限"),
        (1.0, -1.0, "下限"),
        (-pi - 0.1, 0.0, r"\[-π, π\]"),
        (0.0, pi + 0.1, r"\[-π, π\]"),
        (np.nan, 1.0, "有限"),
    ],
)
def test_joint_limit_rejects_invalid_bounds(
    lower: float,
    upper: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        JointLimit(lower, upper)


def test_ik_solution_normalizes_storage_types() -> None:
    solution = IKSolution([0, 1], "q2_positive", np.float64(1e-12))  # type: ignore[arg-type]

    assert solution.joint_positions == (0.0, 1.0)
    assert isinstance(solution.residual, float)


@pytest.mark.parametrize(
    "lengths",
    [(1.0,), (1.0, 0.0), (1.0, np.inf)],
)
def test_arm_rejects_invalid_link_lengths(lengths: object) -> None:
    with pytest.raises(ValueError, match="link_lengths"):
        PlanarArm2D(lengths, (JointLimit(-pi, pi), JointLimit(-pi, pi)))  # type: ignore[arg-type]


def test_arm_rejects_invalid_joint_limit_count() -> None:
    with pytest.raises(ValueError, match="joint_limits"):
        PlanarArm2D((0.6, 0.4), (JointLimit(-pi, pi),))  # type: ignore[arg-type]


def test_forward_kinematics_zero_pose() -> None:
    pose = _arm().forward_kinematics([0.0, 0.0])

    assert pose.target_frame == "base"
    assert pose.source_frame == "end_effector"
    assert_allclose(pose.matrix[:3, 3], [1.0, 0.0, 0.0], atol=1e-12)
    assert_allclose(pose.matrix[:3, :3], np.eye(3), atol=1e-12)


def test_forward_kinematics_first_joint_ninety_degrees() -> None:
    pose = _arm().forward_kinematics([pi / 2, 0.0])

    assert_allclose(pose.matrix[:2, 3], [0.0, 1.0], atol=1e-12)
    assert_allclose(
        pose.matrix[:3, :3] @ np.array([1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_forward_kinematics_second_joint_ninety_degrees() -> None:
    pose = _arm().forward_kinematics([0.0, pi / 2])

    assert_allclose(pose.matrix[:2, 3], [0.6, 0.4], atol=1e-12)


@pytest.mark.parametrize("joints", [[0.0], [0.0, np.nan], "bad"])
def test_forward_kinematics_rejects_invalid_joint_vector(joints: object) -> None:
    with pytest.raises(ValueError, match="joint_positions"):
        _arm().forward_kinematics(joints)  # type: ignore[arg-type]


def test_forward_kinematics_rejects_joint_limit_violation() -> None:
    arm = _arm((JointLimit(-0.5, 0.5), JointLimit(-0.5, 0.5)))

    with pytest.raises(JointLimitError, match="关节 1"):
        arm.forward_kinematics([0.75, 0.0])


def test_inverse_kinematics_returns_two_valid_branches() -> None:
    arm = _arm()
    target = np.array([0.6, 0.2])

    solutions = arm.inverse_kinematics_all(target)

    assert [solution.branch for solution in solutions] == [
        "q2_positive",
        "q2_negative",
    ]
    for solution in solutions:
        pose = arm.forward_kinematics(solution.joint_positions)
        assert_allclose(pose.matrix[:2, 3], target, atol=1e-12)
        assert solution.residual < 1e-12


def test_inverse_kinematics_recovers_forward_target() -> None:
    arm = _arm()
    expected_joints = (0.4, -0.8)
    target = arm.forward_kinematics(expected_joints).matrix[:2, 3]

    solutions = arm.inverse_kinematics_all(target)

    assert any(
        np.allclose(solution.joint_positions, expected_joints, atol=1e-12)
        for solution in solutions
    )


def test_inverse_kinematics_deduplicates_singular_extension() -> None:
    solutions = _arm().inverse_kinematics_all([1.0, 0.0])

    assert len(solutions) == 1
    assert solutions[0].branch == "singular"
    assert_allclose(solutions[0].joint_positions, [0.0, 0.0], atol=1e-12)


def test_inverse_kinematics_handles_fully_folded_singularity() -> None:
    arm = _arm()
    solutions = arm.inverse_kinematics_all([0.2, 0.0])

    assert len(solutions) == 1
    assert solutions[0].branch == "singular"
    reached = arm.forward_kinematics(solutions[0].joint_positions).matrix[:2, 3]
    assert_allclose(reached, [0.2, 0.0], atol=1e-12)


@pytest.mark.parametrize("target", [[1.01, 0.0], [0.19, 0.0]])
def test_inverse_kinematics_rejects_unreachable_target(target: list[float]) -> None:
    with pytest.raises(UnreachableTargetError, match="超出工作空间"):
        _arm().inverse_kinematics_all(target)


def test_inverse_kinematics_distinguishes_joint_limit_failure() -> None:
    arm = _arm((JointLimit(-0.1, 0.1), JointLimit(-0.1, 0.1)))

    with pytest.raises(JointLimitError, match="所有 IK 解"):
        arm.inverse_kinematics_all([0.0, 1.0])


def test_seed_selects_nearest_ik_branch() -> None:
    arm = _arm()
    target = [0.6, 0.2]
    solutions = arm.inverse_kinematics_all(target)

    selected_positive = arm.solve_ik(
        target,
        seed=solutions[0].joint_positions,
    )
    selected_negative = arm.solve_ik(
        target,
        seed=solutions[1].joint_positions,
    )

    assert selected_positive.branch == "q2_positive"
    assert selected_negative.branch == "q2_negative"


@pytest.mark.parametrize("target", [[0.5], [0.5, np.inf], "bad"])
def test_inverse_kinematics_rejects_invalid_target(target: object) -> None:
    with pytest.raises(ValueError, match="target_xy"):
        _arm().solve_ik(target)  # type: ignore[arg-type]


def test_solve_ik_rejects_invalid_seed_shape() -> None:
    with pytest.raises(ValueError, match="seed"):
        _arm().solve_ik([0.6, 0.2], seed=[0.0])


def test_solve_ik_rejects_seed_outside_limits() -> None:
    arm = _arm((JointLimit(-0.5, 0.5), JointLimit(-pi, pi)))

    with pytest.raises(JointLimitError, match="关节 1"):
        arm.solve_ik([1.0, 0.0], seed=[0.75, 0.0])


def test_demo_defines_five_distinct_reachable_targets() -> None:
    arm = build_demo_arm()

    assert len(DEMO_TARGETS) == 5
    assert len(set(DEMO_TARGETS)) == 5
    for target in DEMO_TARGETS:
        assert arm.inverse_kinematics_all(target)


def test_demo_reaches_all_targets_within_limits() -> None:
    report = validate_ik_demo()

    assert report["successful_targets"] == 5
    assert len(report["results"]) == 5
    assert report["maximum_error"] < 1e-9
    assert all(item["within_limits"] for item in report["results"])
    for item in report["results"]:
        assert_allclose(item["reached"], item["target"], atol=1e-12)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.inf])
def test_demo_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        validate_ik_demo(tolerance=tolerance)
