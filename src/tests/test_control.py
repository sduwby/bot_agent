"""统一控制接口、动作限幅与失败原子性测试。"""

from __future__ import annotations

from math import pi

import numpy as np
import pytest
from numpy.testing import assert_allclose

from control import (
    ControlLimits,
    ControlMode,
    ControlResult,
    EndEffectorDeltaCommand,
    GripperPositionCommand,
    JointPositionCommand,
    PlanarArmController,
    RobotState,
)
from kinematics import JointLimit, JointLimitError, PlanarArm2D
from run_control_demo import (
    build_demo_commands,
    build_demo_controller,
    validate_control_demo,
)


def _arm() -> PlanarArm2D:
    return PlanarArm2D(
        (0.6, 0.4),
        (JointLimit(-pi, pi), JointLimit(-pi, pi)),
    )


def _controller(
    *,
    state: RobotState | None = None,
    limits: ControlLimits | None = None,
) -> PlanarArmController:
    return PlanarArmController(
        _arm(),
        state or RobotState((0.4, -0.8), 1.0),
        limits or ControlLimits((0.2, 0.15), 0.05, 0.2),
    )


def test_commands_normalize_values_to_tuples_and_float() -> None:
    joint = JointPositionCommand([1, -1])  # type: ignore[arg-type]
    delta = EndEffectorDeltaCommand(np.array([0.1, 0.2]))  # type: ignore[arg-type]
    gripper = GripperPositionCommand(np.float64(0.5))

    assert joint.positions == (1.0, -1.0)
    assert delta.delta_xy == (0.1, 0.2)
    assert gripper.position == 0.5


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: JointPositionCommand([0.0]), "positions"),
        (lambda: JointPositionCommand([0.0, np.nan]), "positions"),
        (lambda: EndEffectorDeltaCommand([0.0]), "delta_xy"),
        (lambda: EndEffectorDeltaCommand([0.0, np.inf]), "delta_xy"),
        (lambda: GripperPositionCommand(np.nan), "position"),
    ],
)
def test_commands_reject_invalid_values(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize("gripper", [-0.1, 1.1, np.nan])
def test_robot_state_rejects_invalid_gripper(gripper: float) -> None:
    with pytest.raises(ValueError, match="gripper_position"):
        RobotState((0.0, 0.0), gripper)


@pytest.mark.parametrize(
    ("joint_delta", "ee_delta", "gripper_delta", "message"),
    [
        ((0.0, 0.1), 0.1, 0.1, "max_joint_delta"),
        ((0.1, 0.1), 0.0, 0.1, "max_end_effector_delta"),
        ((0.1, 0.1), 0.1, -0.1, "max_gripper_delta"),
        ((0.1,), 0.1, 0.1, "max_joint_delta"),
    ],
)
def test_control_limits_reject_invalid_values(
    joint_delta: object,
    ee_delta: float,
    gripper_delta: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ControlLimits(joint_delta, ee_delta, gripper_delta)  # type: ignore[arg-type]


def test_control_result_enforces_mode_value_dimensions() -> None:
    with pytest.raises(ValueError, match="requested"):
        ControlResult(
            success=True,
            mode=ControlMode.JOINT_POSITION,
            state=RobotState((0.0, 0.0), 0.5),
            requested=(0.0,),
            applied=(0.0, 0.0),
            clipped=False,
            message="invalid",
        )


def test_controller_rejects_initial_joint_limit_violation() -> None:
    with pytest.raises(JointLimitError):
        _controller(state=RobotState((pi + 0.1, 0.0), 0.5))


def test_joint_position_applies_per_step_limits() -> None:
    controller = _controller(
        state=RobotState((0.0, 0.0), 0.5),
        limits=ControlLimits((0.2, 0.1), 0.05, 0.2),
    )

    result = controller.step(JointPositionCommand((1.0, -1.0)))

    assert result.success
    assert result.mode is ControlMode.JOINT_POSITION
    assert result.clipped
    assert result.state.joint_positions == pytest.approx((0.2, -0.1))
    assert result.applied == pytest.approx((0.2, -0.1))


def test_joint_position_saturates_hard_limits() -> None:
    controller = _controller(
        state=RobotState((0.0, 0.0), 0.5),
        limits=ControlLimits((10.0, 10.0), 0.05, 0.2),
    )

    result = controller.step(JointPositionCommand((10.0, -10.0)))

    assert result.success
    assert result.clipped
    assert result.state.joint_positions == pytest.approx((pi, -pi))


def test_joint_position_without_limiting_is_not_marked_clipped() -> None:
    controller = _controller(state=RobotState((0.0, 0.0), 0.5))

    result = controller.step(JointPositionCommand((0.1, -0.1)))

    assert result.success
    assert not result.clipped
    assert result.applied == pytest.approx((0.1, -0.1))


def test_end_effector_delta_is_limited_and_keeps_joint_step_safe() -> None:
    controller = _controller()
    before = controller.state

    result = controller.step(EndEffectorDeltaCommand((0.20, 0.10)))

    assert result.success
    assert result.mode is ControlMode.END_EFFECTOR_DELTA
    assert result.clipped
    assert np.linalg.norm(result.applied) <= 0.05 + 1e-12
    joint_delta = np.abs(
        np.asarray(result.state.joint_positions)
        - np.asarray(before.joint_positions)
    )
    assert np.all(joint_delta <= np.array([0.2, 0.15]) + 1e-12)


def test_reachable_end_effector_delta_reports_actual_fk_delta() -> None:
    controller = _controller(
        limits=ControlLimits((pi, pi), 0.2, 0.2),
    )
    before_pose = controller.arm.forward_kinematics(
        controller.state.joint_positions
    )

    result = controller.step(EndEffectorDeltaCommand((-0.02, 0.01)))
    after_pose = controller.arm.forward_kinematics(
        controller.state.joint_positions
    )

    assert result.success
    assert_allclose(
        result.applied,
        after_pose.matrix[:2, 3] - before_pose.matrix[:2, 3],
        atol=1e-12,
    )
    assert_allclose(result.applied, [-0.02, 0.01], atol=1e-12)
    assert not result.clipped


def test_unreachable_end_effector_delta_is_atomic_failure() -> None:
    controller = _controller(
        state=RobotState((0.0, 0.0), 0.5),
        limits=ControlLimits((0.2, 0.2), 0.1, 0.2),
    )
    before = controller.state

    result = controller.step(EndEffectorDeltaCommand((0.1, 0.0)))

    assert not result.success
    assert result.mode is ControlMode.END_EFFECTOR_DELTA
    assert result.state == before
    assert controller.state == before
    assert result.applied == (0.0, 0.0)
    assert "安全拒绝" in result.message


def test_gripper_position_applies_range_and_step_limits() -> None:
    controller = _controller(state=RobotState((0.0, 0.0), 0.5))

    result = controller.step(GripperPositionCommand(-1.0))

    assert result.success
    assert result.mode is ControlMode.GRIPPER_POSITION
    assert result.clipped
    assert result.state.gripper_position == pytest.approx(0.3)
    assert result.state.joint_positions == (0.0, 0.0)
    assert result.applied == pytest.approx((0.3,))


def test_gripper_position_without_limiting_is_not_marked_clipped() -> None:
    controller = _controller(state=RobotState((0.0, 0.0), 0.5))

    result = controller.step(GripperPositionCommand(0.6))

    assert result.success
    assert not result.clipped
    assert result.state.gripper_position == pytest.approx(0.6)


def test_step_rejects_unknown_command_type() -> None:
    with pytest.raises(TypeError, match="控制命令"):
        _controller().step({"mode": "joint_position"})  # type: ignore[arg-type]


def test_demo_sequence_contains_all_control_modes() -> None:
    controller = build_demo_controller()
    modes = {
        controller.step(command).mode  # type: ignore[arg-type]
        for command in build_demo_commands()[:3]
    }

    assert modes == {
        ControlMode.JOINT_POSITION,
        ControlMode.END_EFFECTOR_DELTA,
        ControlMode.GRIPPER_POSITION,
    }


def test_demo_preserves_all_safety_invariants() -> None:
    report = validate_control_demo()

    assert report["modes"] == {
        ControlMode.JOINT_POSITION,
        ControlMode.END_EFFECTOR_DELTA,
        ControlMode.GRIPPER_POSITION,
    }
    assert report["clipped_steps"] >= 1
    assert report["rejected_steps"] == 1
    assert report["steps"][-1]["result"].success is False
    assert report["steps"][-1]["result"].state == report["final_state"]
    assert report["final_state"].joint_positions == pytest.approx((0.0, 0.0))
    assert 0.0 <= report["final_state"].gripper_position <= 1.0
