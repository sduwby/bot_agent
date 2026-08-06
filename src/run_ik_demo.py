"""使用二维双连杆机械臂调用 IK 到达五个固定目标点。"""

from __future__ import annotations

import sys
from math import pi
from typing import Any

import numpy as np

from kinematics import JointLimit, PlanarArm2D


DEMO_TARGETS: tuple[tuple[float, float], ...] = (
    (0.80, 0.10),
    (0.60, 0.50),
    (0.25, 0.55),
    (0.45, -0.45),
    (0.85, -0.15),
)
_POSITION_TOLERANCE = 1e-9


def build_demo_arm() -> PlanarArm2D:
    """构造连杆长度 0.6/0.4 米、全主值限位的 2R 机械臂。"""
    return PlanarArm2D(
        link_lengths=(0.6, 0.4),
        joint_limits=(JointLimit(-pi, pi), JointLimit(-pi, pi)),
    )


def validate_ik_demo(
    *,
    tolerance: float = _POSITION_TOLERANCE,
) -> dict[str, Any]:
    """逐个求解五个目标，用 FK 回代并返回验收数据。"""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance 必须是正的有限数值")

    arm = build_demo_arm()
    seed: tuple[float, float] | None = None
    results: list[dict[str, Any]] = []
    for index, target in enumerate(DEMO_TARGETS, start=1):
        solution = arm.solve_ik(target, seed=seed)
        pose = arm.forward_kinematics(solution.joint_positions)
        reached = pose.matrix[:2, 3]
        error = float(np.linalg.norm(reached - np.asarray(target)))
        within_limits = all(
            limit.contains(angle)
            for angle, limit in zip(
                solution.joint_positions,
                arm.joint_limits,
            )
        )
        if not within_limits:
            raise RuntimeError(f"目标 {index} 的 IK 解违反关节限位")
        if error >= tolerance:
            raise RuntimeError(
                f"目标 {index} 的 FK 回代误差超限：{error:.3e}"
            )
        results.append(
            {
                "index": index,
                "target": target,
                "solution": solution,
                "reached": (float(reached[0]), float(reached[1])),
                "error": error,
                "within_limits": within_limits,
            }
        )
        seed = solution.joint_positions

    maximum_error = max(item["error"] for item in results)
    return {
        "arm": arm,
        "targets": DEMO_TARGETS,
        "results": tuple(results),
        "successful_targets": len(results),
        "maximum_error": maximum_error,
    }


def main() -> int:
    """打印五个目标的 IK 解、FK 回代位置和误差。"""
    try:
        report = validate_ik_demo()
    except (RuntimeError, ValueError) as exc:
        print(f"第 8 周 IK 演示失败：{exc}", file=sys.stderr)
        return 1

    arm = report["arm"]
    print(
        "二维 2R 机械臂："
        f"l1={arm.link_lengths[0]:.3f} m, "
        f"l2={arm.link_lengths[1]:.3f} m"
    )
    print("关节限位：q1/q2 ∈ [-π, π]")
    for item in report["results"]:
        solution = item["solution"]
        q_degrees = np.rad2deg(solution.joint_positions)
        print(
            f"目标 {item['index']}: target=({item['target'][0]:.3f}, "
            f"{item['target'][1]:.3f}) m, "
            f"q=({solution.joint_positions[0]:.6f}, "
            f"{solution.joint_positions[1]:.6f}) rad / "
            f"({q_degrees[0]:.2f}°, {q_degrees[1]:.2f}°), "
            f"branch={solution.branch}, "
            f"FK=({item['reached'][0]:.6f}, {item['reached'][1]:.6f}), "
            f"error={item['error']:.3e}"
        )
    print(
        f"验收结果：{report['successful_targets']}/{len(DEMO_TARGETS)} "
        f"目标到达，最大误差 {report['maximum_error']:.3e}"
    )
    print("第 8 周运动学验证通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
