"""几何深度与手眼标定（第 38 周 P0）——纯逻辑，不依赖真机/仿真框架。

三个能力：

- ``depth_from_known_size``：已知物体尺寸 + 单目成像几何反解深度（**主路径，免 FK**）；
- ``triangulate``：双视角三角化，得到度量深度与相机系三维点（条件路径，仅 B3 使用）；
- ``solve_handeye``：AX=XB 手眼标定（条件路径；需先补齐运动学或采用 marker 方案）。

单位约定：长度用米，像素用 px，焦距用 px。
坐标系约定：变换 ``T_12`` 把 camera-2 系坐标映射到 camera-1 系（``X_1 = R X_2 + t``）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

_ATOL = 1e-9
_HANDEYE_METHODS = ("ax_xb", "board", "aruco")


class GeometryError(ValueError):
    """几何求解失败：退化输入、目标位于相机后方、标定不可解等。"""


@dataclass
class DepthEstimate:
    """一次深度估计结果。"""

    z_m: float
    method: str  # "known_size" | "stereo" | "sensor"
    uncertainty_m: float
    point_m: np.ndarray | None = field(default=None)  # 三角化时的相机系三维点；单目反解为 None

    def __post_init__(self) -> None:
        if not math.isfinite(self.z_m):
            raise ValueError("z_m 必须是有限数值")
        if not math.isfinite(self.uncertainty_m) or self.uncertainty_m < 0:
            raise ValueError("uncertainty_m 必须是非负有限数值")
        if self.point_m is not None:
            point = np.asarray(self.point_m, dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError("point_m 必须是 3 个有限数值")
            self.point_m = point


@dataclass
class HandEyeCalibration:
    """AX=XB 手眼标定结果。"""

    R: np.ndarray  # (3,3) 旋转
    t: np.ndarray  # (3,) 平移（米）
    residual_m: float
    n_samples: int
    method: str

    def __post_init__(self) -> None:
        if self.method not in _HANDEYE_METHODS:
            raise ValueError(f"method 必须是 {_HANDEYE_METHODS} 之一")
        rotation = np.asarray(self.R, dtype=float)
        translation = np.asarray(self.t, dtype=float).reshape(-1)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("R 必须是 3×3 有限数值矩阵")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("t 必须是 3 个有限数值")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6):
            raise ValueError("R 必须是正交矩阵")
        if not np.isclose(float(np.linalg.det(rotation)), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("R 的行列式必须为 +1")
        if not math.isfinite(self.residual_m) or self.residual_m < 0:
            raise ValueError("residual_m 必须是非负有限数值")
        if self.n_samples < 2:
            raise ValueError("n_samples 至少为 2")
        self.R = rotation
        self.t = translation

    def matrix(self) -> np.ndarray:
        """返回 4×4 齐次变换矩阵。"""
        out = np.eye(4)
        out[:3, :3] = self.R
        out[:3, 3] = self.t
        return out


def depth_from_known_size(
    bbox_px: Sequence[float],
    real_size_m: float,
    fx: float,
    bbox_sigma_px: float = 1.0,
) -> DepthEstimate:
    """由目标像素包围盒与已知物理尺寸反解深度（针孔模型）。

    取包围盒的**较大边**作为像素参考尺寸，``real_size_m`` 应为目标对应的
    最大物理边长；窄基线与单目尺度歧义由此被先验尺寸消除。

    Args:
        bbox_px: ``(width_px, height_px)``，两者均须为正。
        real_size_m: 目标最大物理边长（米），须为正。
        fx: 相机 x 方向焦距（像素），须为正。
        bbox_sigma_px: 包围盒测量标准差（像素），用于一阶误差传播。

    Returns:
        DepthEstimate: ``method="known_size"``，不确定度按 ``z·σ/size_px`` 估计。
    """
    width, height = _finite_pair(bbox_px, "bbox_px")
    if width <= 0 or height <= 0:
        raise ValueError("bbox_px 的两个维度都必须为正")
    if not math.isfinite(real_size_m) or real_size_m <= 0:
        raise ValueError("real_size_m 必须为正的有限数值")
    if not math.isfinite(fx) or fx <= 0:
        raise ValueError("fx 必须为正的有限数值")
    if not math.isfinite(bbox_sigma_px) or bbox_sigma_px < 0:
        raise ValueError("bbox_sigma_px 必须是非负有限数值")

    size_px = max(width, height)
    z_m = fx * real_size_m / size_px
    uncertainty_m = z_m * bbox_sigma_px / size_px
    return DepthEstimate(z_m=z_m, method="known_size", uncertainty_m=uncertainty_m)


def triangulate(
    p1: Sequence[float],
    p2: Sequence[float],
    K1: ArrayLike,
    K2: ArrayLike,
    T_12: ArrayLike,
) -> DepthEstimate:
    """双视角三角化（中点法），返回 camera-1 系下的三维点与深度。

    Args:
        p1: camera-1 中的像素坐标 ``(u, v)``。
        p2: camera-2 中的像素坐标 ``(u, v)``。
        K1 / K2: 两相机内参 (3×3)。
        T_12: 4×4 刚体变换，把 camera-2 系坐标映射到 camera-1 系。

    Returns:
        DepthEstimate: ``method="stereo"``，``point_m`` 为 camera-1 系三维点，
        不确定度取两条射线最近点距离的一半。

    Raises:
        GeometryError: 目标落在 camera-1 后方（``z <= 0``）。
    """
    k1 = _intrinsics(K1, "K1")
    k2 = _intrinsics(K2, "K2")
    transform = _rigid_matrix(T_12, "T_12")
    d1 = _pixel_ray(p1, k1, "p1")
    d2_cam1 = transform[:3, :3] @ _pixel_ray(p2, k2, "p2")

    origin1 = np.zeros(3)
    origin2 = transform[:3, 3]  # camera-2 光心在 camera-1 系中的位置

    # 最小化 |origin1 + s1·d1 − (origin2 + s2·d2_cam1)|
    lhs = np.column_stack([d1, -d2_cam1])
    rhs = origin2 - origin1
    depths, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    near1 = origin1 + float(depths[0]) * d1
    near2 = origin2 + float(depths[1]) * d2_cam1
    point = 0.5 * (near1 + near2)

    if point[2] <= 0:
        raise GeometryError("三角化结果位于 camera-1 后方（z<=0）：请检查 T_12 与像素输入")

    return DepthEstimate(
        z_m=float(point[2]),
        method="stereo",
        uncertainty_m=0.5 * float(np.linalg.norm(near1 - near2)),
        point_m=point,
    )


def solve_handeye(
    pairs: Sequence[tuple[ArrayLike, ArrayLike]],
    method: str = "ax_xb",
) -> HandEyeCalibration:
    """求解 ``A X = X B`` 手眼标定。

    旋转部分用四元数表示的零空间法（``L(q_A) − R(q_B)`` 的最小奇异向量），
    天然规避了旋转矩阵取负带来的符号歧义；平移部分随后最小二乘求解。

    Args:
        pairs: ``(A_i, B_i)`` 序列，每个元素是 4×4 刚体变换；至少 2 组。
        method: ``"ax_xb"`` / ``"board"`` / ``"aruco"``，仅作来源记录；
            三种来源产出的都是相对运动对，求解方式一致。

    Returns:
        HandEyeCalibration: 估计的 X 与平均平移残差（米）。

    Raises:
        GeometryError: 组数不足 2，或旋转方程退化（相对运动过小/共面）。
    """
    if method not in _HANDEYE_METHODS:
        raise ValueError(f"method 必须是 {_HANDEYE_METHODS} 之一")
    if len(pairs) < 2:
        raise GeometryError(f"手眼标定至少需要 2 组相对运动，当前 {len(pairs)} 组")

    motion_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"pairs[{index}] 必须是 (A, B) 二元组")
        motion_pairs.append(
            (
                _rigid_matrix(pair[0], f"pairs[{index}].A"),
                _rigid_matrix(pair[1], f"pairs[{index}].B"),
            )
        )

    rows = []
    for motion_a, motion_b in motion_pairs:
        quat_a = _quaternion_from_matrix(motion_a[:3, :3])
        quat_b = _quaternion_from_matrix(motion_b[:3, :3])
        rows.append(_quat_left(quat_a) - _quat_right(quat_b))
    stacked = np.vstack(rows)

    eigenvalues, eigenvectors = np.linalg.eigh(stacked.T @ stacked)
    # 零空间必须恰为一维：若第二小特征值也≈0，说明约束不足（全平移/共面），
    # 此时最小特征向量任意，解无意义。允许真实数据的非零最小特征值（最小二乘）。
    scale = max(float(eigenvalues[-1]), 1.0)
    if eigenvalues[1] <= 1e-8 * scale:
        raise GeometryError(
            "旋转方程退化：相对运动不足/共面/全为纯平移，无法唯一确定旋转，"
            "请增加转动分量充分的多组采集"
        )
    quat_x = eigenvectors[:, 0]
    rotation_x = _matrix_from_quaternion(quat_x)

    blocks = []
    targets = []
    for motion_a, motion_b in motion_pairs:
        blocks.append(motion_a[:3, :3] - np.eye(3))
        targets.append(rotation_x @ motion_b[:3, 3] - motion_a[:3, 3])
    translation_x, *_ = np.linalg.lstsq(np.vstack(blocks), np.concatenate(targets), rcond=None)

    transform_x = np.eye(4)
    transform_x[:3, :3] = rotation_x
    transform_x[:3, 3] = translation_x
    residuals = [
        float(np.linalg.norm((motion_a @ transform_x - transform_x @ motion_b)[:3, 3]))
        for motion_a, motion_b in motion_pairs
    ]

    return HandEyeCalibration(
        R=rotation_x,
        t=translation_x,
        residual_m=float(np.mean(residuals)),
        n_samples=len(motion_pairs),
        method=method,
    )


# ---------------------------------------------------------------------------
# 内部校验与四元数工具
# ---------------------------------------------------------------------------


def _finite_pair(value: Sequence[float], name: str) -> tuple[float, float]:
    """校验并返回两个有限浮点数。"""
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须包含两个有限数值") from exc
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须包含两个有限数值")
    return float(array[0]), float(array[1])


def _intrinsics(value: ArrayLike, name: str) -> np.ndarray:
    """校验 3×3 可逆内参矩阵。"""
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 3×3 有限数值矩阵") from exc
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 必须是 3×3 有限数值矩阵")
    if abs(float(np.linalg.det(matrix))) <= _ATOL:
        raise GeometryError(f"{name} 不可逆：请检查焦距与主点")
    return matrix


def _rigid_matrix(value: ArrayLike, name: str) -> np.ndarray:
    """校验 4×4 刚体变换矩阵。"""
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 4×4 有限数值矩阵") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 必须是 4×4 有限数值矩阵")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=_ATOL):
        raise ValueError(f"{name} 的最后一行必须是 [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} 的旋转部分必须正交")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} 的旋转部分行列式必须为 +1")
    return matrix


def _pixel_ray(pixel: Sequence[float], intrinsics: np.ndarray, name: str) -> np.ndarray:
    """把像素坐标转为单位长度的相机系射线方向。"""
    u, v = _finite_pair(pixel, name)
    ray = np.linalg.inv(intrinsics) @ np.array([u, v, 1.0])
    norm = float(np.linalg.norm(ray))
    if norm <= _ATOL:
        raise GeometryError(f"{name} 反投影得到零向量")
    return ray / norm


def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 单位四元数 ``[w, x, y, z]``（Shepperd 分支法）。"""
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            0.25 * scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
        ])
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array([
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            0.25 * scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
        ])
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array([
            (rotation[1, 0] - rotation[0, 1]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            0.25 * scale,
        ])
    return quat / np.linalg.norm(quat)


def _matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """单位四元数 ``[w, x, y, z]`` → 旋转矩阵。"""
    vector = np.asarray(quat, dtype=float).reshape(-1)
    if vector.shape != (4,):
        raise ValueError("四元数必须包含 4 个数值")
    norm = float(np.linalg.norm(vector))
    if norm <= _ATOL:
        raise GeometryError("四元数范数为零")
    w, x, y, z = vector / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat_left(quat: np.ndarray) -> np.ndarray:
    """左乘矩阵 ``L(q)``，使 ``q ⊗ p = L(q) p``。"""
    w, x, y, z = quat
    return np.array([
        [w, -x, -y, -z],
        [x, w, -z, y],
        [y, z, w, -x],
        [z, -y, x, w],
    ])


def _quat_right(quat: np.ndarray) -> np.ndarray:
    """右乘矩阵 ``R(q)``，使 ``p ⊗ q = R(q) p``。"""
    w, x, y, z = quat
    return np.array([
        [w, -x, -y, -z],
        [x, w, z, -y],
        [y, -z, w, x],
        [z, y, -x, w],
    ])


# --------------------------------------------------------------------------- #
# 1.3 几何量 → 策略输入特征（第 38 周 P0；供 visual_act 的额外观测通道注入）
# --------------------------------------------------------------------------- #

#: 消融变体，**按累积顺序**：``B_k ⊇ B_{k-1}``（设计 §7.1 —— B0 仅 RGB，逐级加通道）。
VARIANT_ORDER: tuple[str, ...] = ("B0", "B1", "B2", "B3")

#: 每个变体**自身新增**的特征列（累积由 `variant_feature_names` 负责）。
#: 名字即契约：训练与推理两侧必须用**同一顺序**，故这里集中定义、不散落。
VARIANT_EXTRA_FEATURES: dict[str, tuple[str, ...]] = {
    "B0": (),
    "B1": ("depth_known_size_m", "depth_known_size_sigma_m"),
    "B2": ("load_gripper_permille", "contact_flag"),
    "B3": ("depth_stereo_m", "depth_stereo_sigma_m"),
}


def variant_feature_names(variant: str) -> tuple[str, ...]:
    """返回某变体的**累积**特征名（B0 → ``()``；B3 → 6 列）。

    累积展开刻意放在这里而不是让调用方自己拼：B1/B2/B3 的列顺序一旦不一致，
    训练与推理会静默错位（数值仍然"有限"，只是全错），这是最难查的一类错。
    """
    if variant not in VARIANT_ORDER:
        raise GeometryError(f"variant 必须是 {VARIANT_ORDER} 之一，实际 {variant!r}")
    names: list[str] = []
    for item in VARIANT_ORDER:
        names.extend(VARIANT_EXTRA_FEATURES[item])
        if item == variant:
            break
    return tuple(names)


def feature_values(
    *,
    depth: DepthEstimate | None = None,
    stereo: DepthEstimate | None = None,
    load_permille: float | None = None,
    contact: bool | None = None,
) -> dict[str, float]:
    """把几何/负载量整理成"特征名 → 值"。

    **不可用的量就缺键**（例如立体匹配失败时不给 `stereo`），由 `build_feature_vector`
    统一报错——不在这里填 0 或均值：把"没测到"伪装成一个合法数值，会污染训练且事后无法区分。
    """
    values: dict[str, float] = {}
    if depth is not None:
        if depth.method != "known_size":
            raise GeometryError(f"depth 的 method 应为 known_size，实际 {depth.method!r}")
        values["depth_known_size_m"] = float(depth.z_m)
        values["depth_known_size_sigma_m"] = float(depth.uncertainty_m)
    if stereo is not None:
        if stereo.method != "stereo":
            raise GeometryError(f"stereo 的 method 应为 stereo，实际 {stereo.method!r}")
        values["depth_stereo_m"] = float(stereo.z_m)
        values["depth_stereo_sigma_m"] = float(stereo.uncertainty_m)
    if load_permille is not None:
        value = float(load_permille)
        if not math.isfinite(value):
            raise GeometryError("load_permille 必须是有限数值")
        values["load_gripper_permille"] = value
    if contact is not None:
        values["contact_flag"] = 1.0 if contact else 0.0
    return values


def build_feature_vector(
    variant: str,
    values: Mapping[str, float],
) -> np.ndarray:
    """按变体把特征整理成**定序**的 float32 向量（供策略的额外观测通道）。

    **缺键即报错**（并列出缺了哪些）：缺特征的训练/推理数据必须显式失败，
    否则模型会拿到错位的向量而指标看起来"只是变差"，无法归因。
    """
    names = variant_feature_names(variant)
    if not names:
        return np.zeros((0,), dtype=np.float32)
    missing = [name for name in names if name not in values]
    if missing:
        raise GeometryError(
            f"变体 {variant} 缺少特征 {missing}（已有 {sorted(values)}）"
            "：不可用的量必须显式失败，不能填 0 冒充"
        )
    vector = np.asarray([float(values[name]) for name in names], dtype=np.float32)
    if not np.all(np.isfinite(vector)):
        raise GeometryError(f"变体 {variant} 的特征向量含非有限数：{vector}")
    return vector
