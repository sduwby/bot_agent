"""B0–B3 消融的**特征生产**（方案③：B1/B2 用特权量作上界，B3 用几何/检测作真实对照）。

设计 §7.1 只写了"加什么能力"，没规定这些数值**由谁产生**。用户决策（方案③）后本模块把来源钉死：

| 变体 | 自身新增列 | 来源 | 含义 |
|---|---|---|---|
| B1 | `depth_known_size_m` / `_sigma_m` | **oracle（特权）** | 物体→夹爪距离，代表"已知尺寸反解"能提供的**上界** |
| B2 | `load_gripper_permille` / `contact_flag` | **oracle（特权）** | `is_grasped`（真接触）+ 归一化闭合度（负载代理） |
| B3 | `depth_stereo_m` / `_sigma_m` | **detector（真实）** | 两视角观测 → `real_geometry.triangulate`，代表**真做感知**能拿到的 |

**为什么 B3 必须走检测路径**：三角化需要相机内参 + 两视角相对位姿 + 对应像素，
不可能由特权位姿"算出来"而不循环论证（用真值位姿反推三角化结果 ≡ 直接给答案）。
所以 B3 的**几何核心**放在这里（纯函数、可离线测），而"从渲染帧里拿到对应像素"
那一步留在仿真侧 `run_depth_ablation.py`（依赖 mani_skill，本机无法验证）。

分层：
1. `oracle_*` / `depth_from_bbox` / `stereo_depth`：**纯函数**，只吃数值；
2. `variant_features(...)`：把变体名 + 两类输入装配成 `real_geometry.build_feature_vector`
   所需的"特征名 → 值"字典（列序由 `real_geometry` 单点定义，这里不重复）。

**已知局限（必须随结论一起报告）**：B1/B2 是**特权量**，其 ΔGrasp 是**上界**、**不能**解释为
可部署能力；B2 的 `load_gripper_permille` 在仿真里**没有真实扭矩通道**，是个单调代理，
真正有信息量的是 `contact_flag`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from real_geometry import (
    VARIANT_ORDER,
    DepthEstimate,
    GeometryError,
    build_feature_vector,
    depth_from_known_size,
    feature_values,
    triangulate,
    variant_feature_names,
)

#: 仿真里"夹住"时给负载代理的幅值（‰）。仅作**单调代理**：仿真无扭矩通道，
#: 所以这个数值不承载物理量纲，只让"夹住 vs 未夹住"在负载列上可分。
GRASP_LOAD_PROXY_PERMILLE = -200.0


@dataclass(frozen=True)
class OracleInputs:
    """特权量输入（仿真真值）。**仅用于消融上界**，不代表可部署能力。"""

    tcp_pos_m: Sequence[float]
    obj_pos_m: Sequence[float]
    is_grasped: bool
    gripper_width: float
    open_width: float
    closed_width: float

    def __post_init__(self) -> None:
        for name, value in (("tcp_pos_m", self.tcp_pos_m), ("obj_pos_m", self.obj_pos_m)):
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise GeometryError(f"{name} 必须是 3 个有限数值")
        if not self.open_width > self.closed_width:
            raise GeometryError("open_width 必须大于 closed_width（否则无法归一化闭合度）")


@dataclass(frozen=True)
class DetectedInputs:
    """检测路径输入：两视角的**目标像素中心** + 相机内参/相对位姿。"""

    center_px_a: Sequence[float]
    center_px_b: Sequence[float]
    intrinsics_a: np.ndarray
    intrinsics_b: np.ndarray
    T_ab: np.ndarray

    def __post_init__(self) -> None:
        for name, value in (("center_px_a", self.center_px_a), ("center_px_b", self.center_px_b)):
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.shape != (2,) or not np.all(np.isfinite(array)):
                raise GeometryError(f"{name} 必须是 2 个有限数值（像素中心）")


def u_close(width: float, open_width: float, closed_width: float) -> float:
    """归一化闭合度：0 = 完全张开、1 = 完全闭合并顶到限位（越界会被截断）。"""
    value = (open_width - float(width)) / (open_width - closed_width)
    return float(min(1.0, max(0.0, value)))


def oracle_features(variant: str, inputs: OracleInputs) -> dict[str, float]:
    """B1/B2 的**特权**特征（上界），按变体只产出它自己那几列。

    B1 的"已知尺寸反解深度"在仿真里用**物体→夹爪距离**作为上界：
    真实 photogrammetry 受尺度误差/遮挡/检测抖动影响，绝不可能优于这个值，
    所以拿它当"几何量值多少分"的天花板是自洽的。
    """
    if variant not in VARIANT_ORDER:
        raise GeometryError(f"variant 必须是 {VARIANT_ORDER} 之一，实际 {variant!r}")
    values: dict[str, float] = {}
    if variant in ("B1", "B2", "B3"):
        distance = float(
            np.linalg.norm(
                np.asarray(inputs.tcp_pos_m, dtype=float) - np.asarray(inputs.obj_pos_m, dtype=float)
            )
        )
        # 特权量：不确定性记 0（这正是"上界"的含义，必须与检测路径的 sigma 区分开）
        estimate = DepthEstimate(z_m=distance, method="known_size", uncertainty_m=0.0)
        values.update(feature_values(depth=estimate))
    if variant in ("B2", "B3"):
        closure = u_close(inputs.gripper_width, inputs.open_width, inputs.closed_width)
        # 未夹住 → 0‰（不是"负得很小"，而是**没有负载**）；夹住时用闭合度缩放代理幅值
        load = GRASP_LOAD_PROXY_PERMILLE * closure if inputs.is_grasped else 0.0
        values.update(feature_values(load_permille=load, contact=inputs.is_grasped))
    return values


def depth_from_pixel_size(
    size_px: Sequence[float],
    real_size_m: float,
    fx_px: float,
) -> dict[str, float]:
    """已知尺寸 + 单目成像几何 → B1 的**检测路径**特征（真实，非特权）。

    ``size_px`` 是目标的**像素尺寸 (宽, 高)**（不是四元 bbox 角点）；检测器给出角点后
    自行相减即可。供 B3 的对照实验使用（方案③：B1 用特权上界、B3 用检测，
    因此需要这条真实路径同在）。
    """
    estimate = depth_from_known_size(size_px, real_size_m, fx_px)
    return feature_values(depth=estimate)


def stereo_depth(inputs: DetectedInputs) -> dict[str, float]:
    """两视角像素中心 → 三角化深度（B3 的**真实**检测路径）。"""
    estimate = triangulate(
        np.asarray(inputs.center_px_a, dtype=float),
        np.asarray(inputs.center_px_b, dtype=float),
        np.asarray(inputs.intrinsics_a, dtype=float),
        np.asarray(inputs.intrinsics_b, dtype=float),
        np.asarray(inputs.T_ab, dtype=float),
    )
    return feature_values(stereo=estimate)


def variant_features(
    variant: str,
    *,
    oracle: OracleInputs | None = None,
    detected: DetectedInputs | None = None,
) -> np.ndarray:
    """把变体所需的全部列装配成策略可吃的定序向量（方案③的来源分配）。

    - B0 → 空向量；
    - B1/B2 → 只用 `oracle`；
    - B3 → B1/B2 列用 `oracle`、自己的深度列用 `detected`（**混合来源**，即方案③）。

    缺哪一类输入就在需要它的变体上**直接报错**（由 `build_feature_vector` 的缺键校验兜底），
    不允许用 0 或特权值顶替检测结果 —— 那会让"真做感知能拿多少"这个对照失去意义。
    """
    names = variant_feature_names(variant)
    if not names:
        return np.zeros((0,), dtype=np.float32)
    values: dict[str, float] = {}
    needs_oracle = variant in ("B1", "B2", "B3")
    if needs_oracle:
        if oracle is None:
            raise GeometryError(f"变体 {variant} 需要特权量输入（oracle）")
        values.update(oracle_features(variant, oracle))
    if variant == "B3":
        if detected is None:
            raise GeometryError(
                "变体 B3 需要检测路径输入（两视角像素 + 内参/相对位姿）；"
                "不给就从特权位姿反推 —— 那是循环论证，等于没有对照"
            )
        values.update(stereo_depth(detected))
    return build_feature_vector(variant, values)
