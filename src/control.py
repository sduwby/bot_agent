"""二维机械臂的统一控制命令、状态转移与动作限幅。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from kinematics import (
    JointLimitError,
    PlanarArm2D,
    UnreachableTargetError,
)


_CONTROL_ATOL = 1e-12


def _finite_tuple(value: ArrayLike, *, size: int, name: str) -> tuple[float, ...]:
    """将输入转换为指定长度的有限浮点 tuple。"""
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须包含 {size} 个有限数值") from exc
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须包含 {size} 个有限数值")
    return tuple(float(item) for item in vector)


def _finite_scalar(value: float, *, name: str) -> float:
    """将输入转换为有限浮点数。"""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数值") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值")
    return result


class ControlMode(str, Enum):
    """控制命令模式。"""

    JOINT_POSITION = "joint_position"
    END_EFFECTOR_DELTA = "end_effector_delta"
    GRIPPER_POSITION = "gripper_position"


@dataclass(frozen=True)
class JointPositionCommand:
    """两个旋转关节的绝对目标角命令。"""

    positions: tuple[float, float]

    def __post_init__(self) -> None:
        positions = _finite_tuple(self.positions, size=2, name="positions")
        object.__setattr__(self, "positions", positions)


@dataclass(frozen=True)
class EndEffectorDeltaCommand:
    """基座 XY 平面中的末端位置增量命令。"""

    delta_xy: tuple[float, float]

    def __post_init__(self) -> None:
        delta = _finite_tuple(self.delta_xy, size=2, name="delta_xy")
        object.__setattr__(self, "delta_xy", delta)


@dataclass(frozen=True)
class GripperPositionCommand:
    """夹爪归一化绝对位置命令，0 闭合、1 张开。"""

    position: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            _finite_scalar(self.position, name="position"),
        )


@dataclass(frozen=True)
class RobotState:
    """控制器持有的关节与夹爪状态。"""

    joint_positions: tuple[float, float]
    gripper_position: float

    def __post_init__(self) -> None:
        joints = _finite_tuple(
            self.joint_positions,
            size=2,
            name="joint_positions",
        )
        gripper = _finite_scalar(
            self.gripper_position,
            name="gripper_position",
        )
        if not 0.0 <= gripper <= 1.0:
            raise ValueError("gripper_position 必须位于 [0, 1]")
        object.__setattr__(self, "joint_positions", joints)
        object.__setattr__(self, "gripper_position", gripper)


@dataclass(frozen=True)
class ControlLimits:
    """单步关节、末端和夹爪动作限幅。"""

    max_joint_delta: tuple[float, float]
    max_end_effector_delta: float
    max_gripper_delta: float

    def __post_init__(self) -> None:
        joint_delta = _finite_tuple(
            self.max_joint_delta,
            size=2,
            name="max_joint_delta",
        )
        end_effector_delta = _finite_scalar(
            self.max_end_effector_delta,
            name="max_end_effector_delta",
        )
        gripper_delta = _finite_scalar(
            self.max_gripper_delta,
            name="max_gripper_delta",
        )
        if any(value <= 0 for value in joint_delta):
            raise ValueError("max_joint_delta 必须包含两个正数")
        if end_effector_delta <= 0:
            raise ValueError("max_end_effector_delta 必须是正数")
        if gripper_delta <= 0:
            raise ValueError("max_gripper_delta 必须是正数")
        object.__setattr__(self, "max_joint_delta", joint_delta)
        object.__setattr__(
            self,
            "max_end_effector_delta",
            end_effector_delta,
        )
        object.__setattr__(self, "max_gripper_delta", gripper_delta)


@dataclass(frozen=True)
class ControlResult:
    """一条控制命令的执行结果和更新后状态。"""

    success: bool
    mode: ControlMode
    state: RobotState
    requested: tuple[float, ...]
    applied: tuple[float, ...]
    clipped: bool
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool) or not isinstance(self.clipped, bool):
            raise ValueError("success 和 clipped 必须是布尔值")
        if not isinstance(self.mode, ControlMode):
            raise ValueError("mode 必须是 ControlMode")
        if not isinstance(self.state, RobotState):
            raise ValueError("state 必须是 RobotState")
        value_size = (
            1 if self.mode is ControlMode.GRIPPER_POSITION else 2
        )
        requested = _finite_tuple(
            self.requested,
            size=value_size,
            name="requested",
        )
        applied = _finite_tuple(
            self.applied,
            size=value_size,
            name="applied",
        )
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message 必须是非空字符串")
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "applied", applied)


ControlCommand = Union[
    JointPositionCommand,
    EndEffectorDeltaCommand,
    GripperPositionCommand,
]


class PlanarArmController:
    """对二维双连杆机械臂执行带限幅的离散控制命令。"""

    def __init__(
        self,
        arm: PlanarArm2D,
        initial_state: RobotState,
        limits: ControlLimits,
    ) -> None:
        if not isinstance(arm, PlanarArm2D):
            raise TypeError("arm 必须是 PlanarArm2D")
        if not isinstance(initial_state, RobotState):
            raise TypeError("initial_state 必须是 RobotState")
        if not isinstance(limits, ControlLimits):
            raise TypeError("limits 必须是 ControlLimits")
        arm.forward_kinematics(initial_state.joint_positions)
        self.arm = arm
        self.limits = limits
        self._state = initial_state

    @property
    def state(self) -> RobotState:
        """返回当前不可变状态。"""
        return self._state

    def step(self, command: ControlCommand) -> ControlResult:
        """按命令类型执行一次原子状态转移。"""
        if isinstance(command, JointPositionCommand):
            return self._apply_joint_position(command)
        if isinstance(command, EndEffectorDeltaCommand):
            return self._apply_end_effector_delta(command)
        if isinstance(command, GripperPositionCommand):
            return self._apply_gripper_position(command)
        raise TypeError("command 必须是受支持的控制命令")

    def _limited_joint_target(
        self,
        desired_positions: Sequence[float],
    ) -> tuple[tuple[float, float], bool]:
        """应用关节硬限位和相对当前状态的单步限幅。"""
        desired = np.asarray(desired_positions, dtype=float)
        current = np.asarray(self._state.joint_positions, dtype=float)
        lower = np.asarray(
            [limit.lower for limit in self.arm.joint_limits],
            dtype=float,
        )
        upper = np.asarray(
            [limit.upper for limit in self.arm.joint_limits],
            dtype=float,
        )
        bounded = np.clip(desired, lower, upper)
        maximum_delta = np.asarray(self.limits.max_joint_delta, dtype=float)
        limited_delta = np.clip(
            bounded - current,
            -maximum_delta,
            maximum_delta,
        )
        candidate = np.clip(current + limited_delta, lower, upper)
        clipped = not np.allclose(
            candidate,
            desired,
            rtol=0.0,
            atol=_CONTROL_ATOL,
        )
        return (float(candidate[0]), float(candidate[1])), clipped

    def _apply_joint_position(
        self,
        command: JointPositionCommand,
    ) -> ControlResult:
        candidate_joints, clipped = self._limited_joint_target(
            command.positions
        )
        self.arm.forward_kinematics(candidate_joints)
        self._state = RobotState(
            candidate_joints,
            self._state.gripper_position,
        )
        return ControlResult(
            success=True,
            mode=ControlMode.JOINT_POSITION,
            state=self._state,
            requested=command.positions,
            applied=candidate_joints,
            clipped=clipped,
            message="关节位置命令已应用" + ("（已限幅）" if clipped else ""),
        )

    def _apply_end_effector_delta(
        self,
        command: EndEffectorDeltaCommand,
    ) -> ControlResult:
        requested_delta = np.asarray(command.delta_xy, dtype=float)
        requested_norm = float(np.linalg.norm(requested_delta))
        limited_delta = requested_delta.copy()
        cartesian_clipped = False
        if requested_norm > self.limits.max_end_effector_delta:
            limited_delta *= (
                self.limits.max_end_effector_delta / requested_norm
            )
            cartesian_clipped = True

        current_pose = self.arm.forward_kinematics(
            self._state.joint_positions
        )
        current_xy = current_pose.matrix[:2, 3]
        target_xy = current_xy + limited_delta
        try:
            solution = self.arm.solve_ik(
                target_xy,
                seed=self._state.joint_positions,
            )
        except (UnreachableTargetError, JointLimitError) as exc:
            return ControlResult(
                success=False,
                mode=ControlMode.END_EFFECTOR_DELTA,
                state=self._state,
                requested=command.delta_xy,
                applied=(0.0, 0.0),
                clipped=cartesian_clipped,
                message=f"末端增量命令被安全拒绝：{exc}",
            )

        candidate_joints, joint_clipped = self._limited_joint_target(
            solution.joint_positions
        )
        candidate_pose = self.arm.forward_kinematics(candidate_joints)
        actual_delta = candidate_pose.matrix[:2, 3] - current_xy
        self._state = RobotState(
            candidate_joints,
            self._state.gripper_position,
        )
        clipped = (
            cartesian_clipped
            or joint_clipped
            or not np.allclose(
                actual_delta,
                requested_delta,
                rtol=0.0,
                atol=_CONTROL_ATOL,
            )
        )
        return ControlResult(
            success=True,
            mode=ControlMode.END_EFFECTOR_DELTA,
            state=self._state,
            requested=command.delta_xy,
            applied=(float(actual_delta[0]), float(actual_delta[1])),
            clipped=clipped,
            message=(
                f"末端增量命令已应用（IK 分支 {solution.branch}）"
                + ("（已限幅）" if clipped else "")
            ),
        )

    def _apply_gripper_position(
        self,
        command: GripperPositionCommand,
    ) -> ControlResult:
        requested = command.position
        bounded = float(np.clip(requested, 0.0, 1.0))
        current = self._state.gripper_position
        delta = float(
            np.clip(
                bounded - current,
                -self.limits.max_gripper_delta,
                self.limits.max_gripper_delta,
            )
        )
        applied = float(np.clip(current + delta, 0.0, 1.0))
        clipped = not np.isclose(
            applied,
            requested,
            rtol=0.0,
            atol=_CONTROL_ATOL,
        )
        self._state = RobotState(self._state.joint_positions, applied)
        return ControlResult(
            success=True,
            mode=ControlMode.GRIPPER_POSITION,
            state=self._state,
            requested=(requested,),
            applied=(applied,),
            clipped=clipped,
            message="夹爪位置命令已应用" + ("（已限幅）" if clipped else ""),
        )
