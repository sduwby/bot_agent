"""三维刚体变换（SE(3)）的构造、校验与坐标系链组合。"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


_VALIDATION_ATOL = 1e-9


def _frame_name(value: str, field_name: str) -> str:
    """校验并规范化坐标系名称。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _finite_vector(value: ArrayLike, *, name: str) -> np.ndarray:
    """将输入转换为包含三个有限浮点数的一维向量。"""
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须包含 3 个有限数值") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 必须包含 3 个有限数值")
    return vector


def rotation_matrix_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """按 ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` 生成旋转矩阵。"""
    try:
        angles = np.asarray([roll, pitch, yaw], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("roll、pitch、yaw 必须是有限数值") from exc
    if not np.all(np.isfinite(angles)):
        raise ValueError("roll、pitch、yaw 必须是有限数值")

    cr, cp, cy = cos(angles[0]), cos(angles[1]), cos(angles[2])
    sr, sp, sy = sin(angles[0]), sin(angles[1]), sin(angles[2])

    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=float,
    )
    rotation_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=float,
    )
    rotation_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return rotation_z @ rotation_y @ rotation_x


@dataclass(frozen=True)
class RigidTransform:
    """将 ``source_frame`` 中的三维坐标转换到 ``target_frame``。"""

    target_frame: str
    source_frame: str
    matrix: np.ndarray

    def __post_init__(self) -> None:
        target_frame = _frame_name(self.target_frame, "target_frame")
        source_frame = _frame_name(self.source_frame, "source_frame")
        try:
            matrix = np.array(self.matrix, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("matrix 必须是 4×4 有限数值矩阵") from exc

        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("matrix 必须是 4×4 有限数值矩阵")
        if not np.allclose(
            matrix[3],
            np.array([0.0, 0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=_VALIDATION_ATOL,
        ):
            raise ValueError("齐次变换矩阵最后一行必须是 [0, 0, 0, 1]")

        rotation = matrix[:3, :3]
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            rtol=0.0,
            atol=_VALIDATION_ATOL,
        ):
            raise ValueError("旋转矩阵必须正交")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(
            determinant,
            1.0,
            rtol=0.0,
            atol=_VALIDATION_ATOL,
        ):
            raise ValueError("旋转矩阵行列式必须为 +1")
        if target_frame == source_frame and not np.allclose(
            matrix,
            np.eye(4),
            rtol=0.0,
            atol=_VALIDATION_ATOL,
        ):
            raise ValueError("同名坐标系之间只能使用单位变换")

        matrix.setflags(write=False)
        object.__setattr__(self, "target_frame", target_frame)
        object.__setattr__(self, "source_frame", source_frame)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def identity(cls, frame: str) -> "RigidTransform":
        """构造指定坐标系到自身的单位变换。"""
        return cls(frame, frame, np.eye(4))

    @classmethod
    def from_rpy(
        cls,
        target_frame: str,
        source_frame: str,
        translation: Sequence[float],
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> "RigidTransform":
        """由目标系中的平移和 RPY 弧度角构造刚体变换。"""
        vector = _finite_vector(translation, name="translation")
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix_from_rpy(roll, pitch, yaw)
        matrix[:3, 3] = vector
        return cls(target_frame, source_frame, matrix)

    @classmethod
    def from_matrix(
        cls,
        target_frame: str,
        source_frame: str,
        matrix: ArrayLike,
    ) -> "RigidTransform":
        """由 4×4 齐次矩阵构造并校验刚体变换。"""
        return cls(target_frame, source_frame, matrix)  # type: ignore[arg-type]

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        """计算 ``self @ other``，并校验中间坐标系连续。"""
        if not isinstance(other, RigidTransform):
            raise TypeError("other 必须是 RigidTransform")
        if self.source_frame != other.target_frame:
            raise ValueError(
                "坐标变换链不连续："
                f"{self.target_frame} ← {self.source_frame} 无法连接 "
                f"{other.target_frame} ← {other.source_frame}"
            )
        return RigidTransform.from_matrix(
            self.target_frame,
            other.source_frame,
            self.matrix @ other.matrix,
        )

    def inverse(self) -> "RigidTransform":
        """使用刚体变换解析公式返回逆变换。"""
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]
        inverse_matrix = np.eye(4)
        inverse_matrix[:3, :3] = rotation.T
        inverse_matrix[:3, 3] = -rotation.T @ translation
        return RigidTransform.from_matrix(
            self.source_frame,
            self.target_frame,
            inverse_matrix,
        )

    def transform_point(self, point: Sequence[float]) -> np.ndarray:
        """将一个三维点从源坐标系转换到目标坐标系。"""
        vector = _finite_vector(point, name="point")
        return self.matrix[:3, :3] @ vector + self.matrix[:3, 3]


def compose_chain(*transforms: RigidTransform) -> RigidTransform:
    """按给定顺序组合至少一个连续的刚体变换。"""
    if not transforms:
        raise ValueError("compose_chain 至少需要一个变换")
    if not all(isinstance(transform, RigidTransform) for transform in transforms):
        raise TypeError("compose_chain 的所有输入都必须是 RigidTransform")

    first = transforms[0]
    result = RigidTransform.from_matrix(
        first.target_frame,
        first.source_frame,
        first.matrix,
    )
    for transform in transforms[1:]:
        result = result.compose(transform)
    return result
