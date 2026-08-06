"""运行世界、基座、末端、相机和物体坐标系的 SE(3) 变换演示。"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from se3 import RigidTransform, compose_chain


_ROUND_TRIP_TOLERANCE = 1e-9


def build_demo_transforms() -> tuple[RigidTransform, ...]:
    """构造 ``world ← base ← end_effector ← camera ← object`` 固定链。"""
    return (
        RigidTransform.from_rpy(
            "world",
            "base",
            [0.6, -0.25, 0.0],
            yaw=np.deg2rad(30.0),
        ),
        RigidTransform.from_rpy(
            "base",
            "end_effector",
            [0.35, 0.10, 0.55],
            roll=np.deg2rad(10.0),
            pitch=np.deg2rad(-20.0),
            yaw=np.deg2rad(15.0),
        ),
        RigidTransform.from_rpy(
            "end_effector",
            "camera",
            [0.05, 0.0, 0.08],
            pitch=np.deg2rad(5.0),
            yaw=np.deg2rad(-10.0),
        ),
        RigidTransform.from_rpy(
            "camera",
            "object",
            [0.02, -0.01, 0.45],
            yaw=np.deg2rad(25.0),
        ),
    )


def _transform_sequentially(
    transforms: tuple[RigidTransform, ...],
    point: np.ndarray,
) -> np.ndarray:
    """从变换链最右端开始逐段转换一个点。"""
    result = point
    for transform in reversed(transforms):
        result = transform.transform_point(result)
    return result


def validate_demo(
    *,
    tolerance: float = _ROUND_TRIP_TOLERANCE,
) -> dict[str, Any]:
    """运行固定坐标链并返回逐段等价与逆变换校验结果。"""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance 必须是正的有限数值")

    transforms = build_demo_transforms()
    expected_frames = (
        ("world", "base"),
        ("base", "end_effector"),
        ("end_effector", "camera"),
        ("camera", "object"),
    )
    actual_frames = tuple(
        (transform.target_frame, transform.source_frame)
        for transform in transforms
    )
    if actual_frames != expected_frames:
        raise RuntimeError(f"坐标链名称错误：{actual_frames!r}")

    world_from_object = compose_chain(*transforms)
    object_points = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.05, -0.02, 0.03]),
    )
    world_points = tuple(
        world_from_object.transform_point(point)
        for point in object_points
    )
    sequential_points = tuple(
        _transform_sequentially(transforms, point)
        for point in object_points
    )
    sequential_error = max(
        float(np.max(np.abs(composed - sequential)))
        for composed, sequential in zip(world_points, sequential_points)
    )

    object_from_world = world_from_object.inverse()
    recovered_points = tuple(
        object_from_world.transform_point(point)
        for point in world_points
    )
    round_trip_error = max(
        float(np.max(np.abs(original - recovered)))
        for original, recovered in zip(object_points, recovered_points)
    )

    if sequential_error >= tolerance:
        raise RuntimeError(
            f"逐段转换与组合转换不一致：{sequential_error:.3e}"
        )
    if round_trip_error >= tolerance:
        raise RuntimeError(f"逆变换往返误差超限：{round_trip_error:.3e}")

    return {
        "transforms": transforms,
        "world_from_object": world_from_object,
        "object_points": object_points,
        "world_points": world_points,
        "sequential_error": sequential_error,
        "round_trip_error": round_trip_error,
    }


def main() -> int:
    """打印完整坐标链及数值自检结果。"""
    try:
        result = validate_demo()
    except (RuntimeError, ValueError) as exc:
        print(f"坐标变换演示失败：{exc}", file=sys.stderr)
        return 1

    print("坐标变换链：world ← base ← end_effector ← camera ← object")
    for transform in result["transforms"]:
        print(f"\nT_{transform.target_frame}_{transform.source_frame}")
        print(np.array2string(transform.matrix, precision=6, suppress_small=True))

    world_from_object = result["world_from_object"]
    print("\n组合结果 T_world_object")
    print(np.array2string(world_from_object.matrix, precision=6, suppress_small=True))

    for index, (point_object, point_world) in enumerate(
        zip(result["object_points"], result["world_points"]),
        start=1,
    ):
        print(
            f"点 {index}: object {np.array2string(point_object, precision=6)} "
            f"→ world {np.array2string(point_world, precision=6)}"
        )
    print(f"逐段/组合最大误差：{result['sequential_error']:.3e}")
    print(f"逆变换往返最大误差：{result['round_trip_error']:.3e}")
    print("第 7 周坐标变换链验证通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
