"""二维双连杆机械臂的正逆运动学与关节限位。"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, pi, sin, sqrt
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from se3 import RigidTransform


_KINEMATICS_TOLERANCE = 1e-9
_IK_BRANCHES = {"q2_positive", "q2_negative", "singular"}


class UnreachableTargetError(ValueError):
    """目标位于机械臂几何工作空间之外。"""


class JointLimitError(ValueError):
    """关节状态或 IK 解违反关节限位。"""


def _finite_vector(value: ArrayLike, *, size: int, name: str) -> np.ndarray:
    """将输入转换为指定长度的有限浮点向量。"""
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须包含 {size} 个有限数值") from exc
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须包含 {size} 个有限数值")
    return vector


def _normalize_angle(angle: float) -> float:
    """将角度规范化到 [-π, π] 主值区间。"""
    return atan2(sin(angle), cos(angle))


def _angle_distance(first: float, second: float) -> float:
    """返回两个角度之间的最短有符号差。"""
    return _normalize_angle(first - second)


@dataclass(frozen=True)
class JointLimit:
    """单个旋转关节在主值区间内的闭区间限位。"""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        try:
            lower = float(self.lower)
            upper = float(self.upper)
        except (TypeError, ValueError) as exc:
            raise ValueError("关节限位必须是有限数值") from exc
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("关节限位必须是有限数值")
        if lower < -pi or upper > pi:
            raise ValueError("关节限位必须位于 [-π, π]")
        if lower >= upper:
            raise ValueError("关节下限必须小于上限")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(
        self,
        angle: float,
        *,
        tolerance: float = _KINEMATICS_TOLERANCE,
    ) -> bool:
        """判断角度是否位于闭区间限位内。"""
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance 必须是非负有限数值")
        try:
            numeric_angle = float(angle)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(numeric_angle):
            return False
        return (
            self.lower - tolerance
            <= numeric_angle
            <= self.upper + tolerance
        )


@dataclass(frozen=True)
class IKSolution:
    """一个满足关节限位的解析 IK 解。"""

    joint_positions: tuple[float, float]
    branch: str
    residual: float

    def __post_init__(self) -> None:
        joints = _finite_vector(
            self.joint_positions,
            size=2,
            name="joint_positions",
        )
        if self.branch not in _IK_BRANCHES:
            raise ValueError(f"未知 IK 分支：{self.branch}")
        try:
            residual = float(self.residual)
        except (TypeError, ValueError) as exc:
            raise ValueError("residual 必须是非负有限数值") from exc
        if not np.isfinite(residual) or residual < 0:
            raise ValueError("residual 必须是非负有限数值")
        object.__setattr__(
            self,
            "joint_positions",
            (float(joints[0]), float(joints[1])),
        )
        object.__setattr__(self, "residual", residual)


@dataclass(frozen=True)
class PlanarArm2D:
    """位于基座 XY 平面的二维双旋转关节机械臂。"""

    link_lengths: tuple[float, float]
    joint_limits: tuple[JointLimit, JointLimit]

    def __post_init__(self) -> None:
        lengths = _finite_vector(
            self.link_lengths,
            size=2,
            name="link_lengths",
        )
        if np.any(lengths <= 0):
            raise ValueError("link_lengths 必须是两个正数")
        if (
            not isinstance(self.joint_limits, (tuple, list))
            or len(self.joint_limits) != 2
            or not all(
                isinstance(limit, JointLimit)
                for limit in self.joint_limits
            )
        ):
            raise ValueError("joint_limits 必须包含两个 JointLimit")
        object.__setattr__(
            self,
            "link_lengths",
            (float(lengths[0]), float(lengths[1])),
        )
        object.__setattr__(self, "joint_limits", tuple(self.joint_limits))

    def _validated_joints(
        self,
        joint_positions: Sequence[float],
        *,
        name: str = "joint_positions",
    ) -> tuple[float, float]:
        joints = _finite_vector(joint_positions, size=2, name=name)
        for index, (angle, limit) in enumerate(
            zip(joints, self.joint_limits),
            start=1,
        ):
            if not limit.contains(float(angle)):
                raise JointLimitError(
                    f"关节 {index} 角度 {angle:.6f} 超出 "
                    f"[{limit.lower:.6f}, {limit.upper:.6f}]"
                )
        return float(joints[0]), float(joints[1])

    def forward_kinematics(
        self,
        joint_positions: Sequence[float],
    ) -> RigidTransform:
        """由两个关节角计算 ``base ← end_effector`` 末端位姿。"""
        q1, q2 = self._validated_joints(joint_positions)
        length_1, length_2 = self.link_lengths
        combined_angle = q1 + q2
        x = length_1 * cos(q1) + length_2 * cos(combined_angle)
        y = length_1 * sin(q1) + length_2 * sin(combined_angle)
        return RigidTransform.from_rpy(
            "base",
            "end_effector",
            [x, y, 0.0],
            yaw=_normalize_angle(combined_angle),
        )

    def inverse_kinematics_all(
        self,
        target_xy: Sequence[float],
    ) -> tuple[IKSolution, ...]:
        """返回目标位置所有满足主值关节限位的解析 IK 解。"""
        target = _finite_vector(target_xy, size=2, name="target_xy")
        x, y = float(target[0]), float(target[1])
        length_1, length_2 = self.link_lengths
        radius_squared = x * x + y * y
        radius = sqrt(radius_squared)
        minimum_reach = abs(length_1 - length_2)
        maximum_reach = length_1 + length_2
        if (
            radius < minimum_reach - _KINEMATICS_TOLERANCE
            or radius > maximum_reach + _KINEMATICS_TOLERANCE
        ):
            raise UnreachableTargetError(
                f"目标 ({x:.6f}, {y:.6f}) 超出工作空间 "
                f"[{minimum_reach:.6f}, {maximum_reach:.6f}]"
            )

        cosine_q2 = (
            radius_squared - length_1 * length_1 - length_2 * length_2
        ) / (2.0 * length_1 * length_2)
        if (
            cosine_q2 < -1.0 - _KINEMATICS_TOLERANCE
            or cosine_q2 > 1.0 + _KINEMATICS_TOLERANCE
        ):
            raise UnreachableTargetError("目标因数值误差落在工作空间之外")
        cosine_q2 = float(np.clip(cosine_q2, -1.0, 1.0))
        q2_magnitude = acos(cosine_q2)

        if abs(sin(q2_magnitude)) <= _KINEMATICS_TOLERANCE:
            branches = (("singular", q2_magnitude),)
        else:
            branches = (
                ("q2_positive", q2_magnitude),
                ("q2_negative", -q2_magnitude),
            )

        solutions: list[IKSolution] = []
        for branch, q2 in branches:
            q1 = atan2(y, x) - atan2(
                length_2 * sin(q2),
                length_1 + length_2 * cos(q2),
            )
            joints = (_normalize_angle(q1), _normalize_angle(q2))
            if not all(
                limit.contains(angle)
                for angle, limit in zip(joints, self.joint_limits)
            ):
                continue
            pose = self.forward_kinematics(joints)
            reached = pose.matrix[:2, 3]
            residual = float(np.linalg.norm(reached - target))
            solutions.append(IKSolution(joints, branch, residual))

        if not solutions:
            raise JointLimitError(
                f"目标 ({x:.6f}, {y:.6f}) 的所有 IK 解均违反关节限位"
            )
        return tuple(solutions)

    def solve_ik(
        self,
        target_xy: Sequence[float],
        *,
        seed: Sequence[float] | None = None,
    ) -> IKSolution:
        """返回稳定首解，或返回与 seed 最接近的合法 IK 解。"""
        solutions = self.inverse_kinematics_all(target_xy)
        if seed is None:
            return solutions[0]

        seed_joints = self._validated_joints(seed, name="seed")

        def distance(solution: IKSolution) -> float:
            return sum(
                _angle_distance(angle, seed_angle) ** 2
                for angle, seed_angle in zip(
                    solution.joint_positions,
                    seed_joints,
                )
            )

        return min(solutions, key=distance)
