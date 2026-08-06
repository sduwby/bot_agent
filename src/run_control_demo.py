"""演示三种控制模式、动作限幅和不可达命令的安全拒绝。"""

from __future__ import annotations

import sys
from math import pi
from typing import Any

import numpy as np

from control import (
    ControlLimits,
    ControlMode,
    EndEffectorDeltaCommand,
    GripperPositionCommand,
    JointPositionCommand,
    PlanarArmController,
    RobotState,
)
from kinematics import JointLimit, PlanarArm2D


def build_demo_controller() -> PlanarArmController:
    """构造固定 2R 机械臂、初始状态和单步控制限幅。"""
    arm = PlanarArm2D(
        (0.6, 0.4),
        (JointLimit(-pi, pi), JointLimit(-pi, pi)),
    )
    return PlanarArmController(
        arm=arm,
        initial_state=RobotState((0.4, -0.8), 1.0),
        limits=ControlLimits(
            max_joint_delta=(0.25, 0.25),
            max_end_effector_delta=0.05,
            max_gripper_delta=0.20,
        ),
    )


def build_demo_commands() -> tuple[object, ...]:
    """构造覆盖三模式、饱和与安全拒绝的固定命令序列。"""
    commands: list[object] = [
        JointPositionCommand((1.0, -1.2)),
        EndEffectorDeltaCommand((-0.02, 0.04)),
        GripperPositionCommand(-1.0),
        JointPositionCommand((10.0, -10.0)),
    ]
    commands.extend(JointPositionCommand((0.0, 0.0)) for _ in range(6))
    commands.append(EndEffectorDeltaCommand((0.10, 0.0)))
    return tuple(commands)


def _validate_transition(
    controller: PlanarArmController,
    previous: RobotState,
    result: Any,
    step_index: int,
) -> None:
    """验证一次状态转移满足硬限位、单步限幅和失败原子性。"""
    state = result.state
    values = (*state.joint_positions, state.gripper_position)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"步骤 {step_index} 产生非有限状态")
    if not all(
        limit.contains(angle)
        for angle, limit in zip(
            state.joint_positions,
            controller.arm.joint_limits,
        )
    ):
        raise RuntimeError(f"步骤 {step_index} 关节越界")
    if not 0.0 <= state.gripper_position <= 1.0:
        raise RuntimeError(f"步骤 {step_index} 夹爪越界")
    if state != controller.state:
        raise RuntimeError(f"步骤 {step_index} 结果状态与控制器状态不一致")

    if not result.success:
        if state != previous:
            raise RuntimeError(f"步骤 {step_index} 失败后状态发生变化")
        return

    joint_delta = np.abs(
        np.asarray(state.joint_positions)
        - np.asarray(previous.joint_positions)
    )
    if np.any(
        joint_delta
        > np.asarray(controller.limits.max_joint_delta) + 1e-12
    ):
        raise RuntimeError(f"步骤 {step_index} 关节单步变化超限")
    gripper_delta = abs(
        state.gripper_position - previous.gripper_position
    )
    if gripper_delta > controller.limits.max_gripper_delta + 1e-12:
        raise RuntimeError(f"步骤 {step_index} 夹爪单步变化超限")


def validate_control_demo() -> dict[str, Any]:
    """执行固定命令序列并返回安全不变量验收结果。"""
    controller = build_demo_controller()
    commands = build_demo_commands()
    steps: list[dict[str, Any]] = []
    for index, command in enumerate(commands, start=1):
        previous = controller.state
        result = controller.step(command)  # type: ignore[arg-type]
        _validate_transition(controller, previous, result, index)
        steps.append(
            {
                "index": index,
                "command": command,
                "result": result,
                "previous_state": previous,
            }
        )

    modes = {item["result"].mode for item in steps}
    expected_modes = {
        ControlMode.JOINT_POSITION,
        ControlMode.END_EFFECTOR_DELTA,
        ControlMode.GRIPPER_POSITION,
    }
    if modes != expected_modes:
        raise RuntimeError(f"控制模式覆盖不完整：{modes}")
    clipped_steps = sum(item["result"].clipped for item in steps)
    rejected_steps = sum(not item["result"].success for item in steps)
    if clipped_steps < 1:
        raise RuntimeError("演示未触发动作限幅")
    if rejected_steps != 1:
        raise RuntimeError(f"安全拒绝次数错误：{rejected_steps}")

    return {
        "controller": controller,
        "steps": tuple(steps),
        "modes": modes,
        "clipped_steps": clipped_steps,
        "rejected_steps": rejected_steps,
        "final_state": controller.state,
    }


def main() -> int:
    """打印固定控制序列的模式、状态、限幅和拒绝结果。"""
    try:
        report = validate_control_demo()
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"第 9 周控制演示失败：{exc}", file=sys.stderr)
        return 1

    for item in report["steps"]:
        result = item["result"]
        state = result.state
        print(
            f"步骤 {item['index']:02d}: mode={result.mode.value}, "
            f"success={result.success}, clipped={result.clipped}, "
            f"joints=({state.joint_positions[0]:.6f}, "
            f"{state.joint_positions[1]:.6f}), "
            f"gripper={state.gripper_position:.3f}, "
            f"message={result.message}"
        )
    print(
        f"验收结果：覆盖 {len(report['modes'])} 种模式，"
        f"限幅 {report['clipped_steps']} 步，"
        f"安全拒绝 {report['rejected_steps']} 步"
    )
    print("第 9 周控制接口验证通过，全程无状态越界。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
