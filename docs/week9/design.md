# 第 9 周设计文档 — 统一控制接口与动作限幅

## 1. 概述

**本周目标：** 在第 8 周二维双连杆运动学之上，明确区分关节位置、末端增量和夹爪位置三类控制命令，建立统一的控制器状态转移接口，并通过硬限位、单步限幅和失败原子性保证机械臂状态稳定且不越界。

**规划原文引用：**

> 第 9 周 | 控制接口 | 区分 joint position、end-effector delta、gripper action；加入动作限幅。 | 机械臂可稳定移动且不越界。

**前置依赖：**

- 第 7 周 `RigidTransform`：表达末端位姿。
- 第 8 周 `PlanarArm2D`：FK、IK 和关节硬限位。
- NumPy 与 `pytest`：数值处理和纯逻辑测试。

## 2. 设计边界与原则

- 本周继续使用确定性的二维 2R 教学模型，不提前接入第 10 周 ManiSkill。
- 三种命令使用不同的数据类型，禁止用一个无约束浮点数组猜测控制模式。
- 关节硬限位是不可突破的最终边界；单步关节增量、末端增量和夹爪增量是控制平滑约束。
- 输入超出可裁剪范围时进行饱和并在结果中标记 `clipped=True`，不能静默丢失限幅事实。
- 末端目标几何不可达或受关节限位排除时，命令失败且控制器状态完全不变。
- 本周控制器只计算离散目标状态，不处理真实时间、速度、加速度、动力学、碰撞或电机通信。
- 不修改阶段 A Agent、AI2-THOR 技能和评测路径，不调用外部 API。

## 3. 控制模式

| 模式 | 命令 | 语义 | 限幅顺序 |
| --- | --- | --- | --- |
| `joint_position` | `JointPositionCommand` | 两个关节的绝对目标角 | 关节硬限位 → 相对当前状态的单步关节增量 |
| `end_effector_delta` | `EndEffectorDeltaCommand` | 基座 XY 平面中的末端位置增量 | 增量范数 → IK → 单步关节增量 |
| `gripper_position` | `GripperPositionCommand` | 夹爪归一化绝对位置，0 闭合、1 张开 | `[0, 1]` 硬限位 → 单步夹爪增量 |

### 3.1 为什么末端增量仍需关节限幅

末端增量经过 IK 后可能对应较大的关节变化，尤其在奇异点附近。因此即使输入的笛卡尔增量已经限幅，也必须再次限制每个关节相对当前状态的单步变化。最终实际末端位移可能小于请求值，控制结果必须返回实际应用值。

### 3.2 失败原子性

每条命令先在局部变量中完成校验、IK 和候选状态计算，只有全部成功后才替换控制器当前状态。任何异常或不可达结果都返回失败 `ControlResult`，其 `state` 与命令前一致，`applied` 为零增量。

## 4. 架构设计

```text
src/control.py
    ├── ControlMode
    ├── JointPositionCommand
    ├── EndEffectorDeltaCommand
    ├── GripperPositionCommand
    ├── RobotState
    ├── ControlLimits
    ├── ControlResult
    └── PlanarArmController
        ├── step()
        ├── _apply_joint_position()
        ├── _apply_end_effector_delta()
        └── _apply_gripper_position()

src/run_control_demo.py
    └── 三模式固定命令序列 → 每步状态/限幅/误差检查

src/tests/test_control.py
    └── 命令校验、三模式、限幅、不可达回滚和演示验收
```

### 4.1 与第 8 周的连接

- `PlanarArmController` 持有一个 `PlanarArm2D`，不复制运动学公式。
- 每个候选关节状态都由 `forward_kinematics()` 验证。
- 末端增量通过 `solve_ik(target_xy, seed=current_joint_positions)` 转为最近关节解。
- `UnreachableTargetError` 和 `JointLimitError` 被转换为失败结果，不泄漏成部分状态更新。

## 5. 数据结构

```python
class ControlMode(str, Enum):
    JOINT_POSITION = "joint_position"
    END_EFFECTOR_DELTA = "end_effector_delta"
    GRIPPER_POSITION = "gripper_position"


@dataclass(frozen=True)
class JointPositionCommand:
    positions: tuple[float, float]


@dataclass(frozen=True)
class EndEffectorDeltaCommand:
    delta_xy: tuple[float, float]


@dataclass(frozen=True)
class GripperPositionCommand:
    position: float


@dataclass(frozen=True)
class RobotState:
    joint_positions: tuple[float, float]
    gripper_position: float


@dataclass(frozen=True)
class ControlLimits:
    max_joint_delta: tuple[float, float]
    max_end_effector_delta: float
    max_gripper_delta: float


@dataclass(frozen=True)
class ControlResult:
    success: bool
    mode: ControlMode
    state: RobotState
    requested: tuple[float, ...]
    applied: tuple[float, ...]
    clipped: bool
    message: str
```

所有命令、状态和限幅输入在构造时转换为不可变的浮点 tuple，并拒绝错误维度与非有限数值。`RobotState.gripper_position` 必须位于 `[0, 1]`；机械臂关节限位由控制器结合 `PlanarArm2D` 校验。

## 6. 接口设计

```python
ControlCommand = (
    JointPositionCommand
    | EndEffectorDeltaCommand
    | GripperPositionCommand
)


class PlanarArmController:
    def __init__(
        self,
        arm: PlanarArm2D,
        initial_state: RobotState,
        limits: ControlLimits,
    ) -> None: ...

    @property
    def state(self) -> RobotState: ...

    def step(self, command: ControlCommand) -> ControlResult: ...
```

`step()` 根据命令实际类型分派，不接受字典或裸数组；未知类型抛出 `TypeError`，表示调用契约错误。合法命令的运行失败则返回 `success=False`。

## 7. 数据流

### 7.1 关节位置命令

1. 读取绝对目标角。
2. 饱和到机械臂关节硬限位。
3. 计算与当前关节状态的差值。
4. 将每个差值裁剪到对应 `max_joint_delta`。
5. FK 验证候选状态，成功后原子更新。
6. `applied` 返回实际绝对关节位置。

### 7.2 末端增量命令

1. 用 FK 读取当前末端 XY。
2. 若请求增量范数超过 `max_end_effector_delta`，按比例缩放。
3. 计算目标 XY，并以当前关节状态为 seed 调用 IK。
4. 将 IK 目标与当前关节的差值按 `max_joint_delta` 再次裁剪。
5. FK 计算候选状态的实际末端增量并原子更新。
6. 不可达时返回失败，状态不变，实际增量为 `(0, 0)`。

### 7.3 夹爪位置命令

1. 将绝对请求裁剪到 `[0, 1]`。
2. 相对当前夹爪位置的变化裁剪到 `max_gripper_delta`。
3. 只更新夹爪位置，保持关节不变。
4. `applied` 返回实际夹爪绝对位置。

## 8. 错误处理与不变量

每次成功或失败后均需满足：

- 两个关节处于各自 `JointLimit` 内。
- 单步关节变化不超过 `max_joint_delta`。
- 夹爪位置处于 `[0, 1]`，单步变化不超过 `max_gripper_delta`。
- 所有状态数值有限。
- 失败命令不改变关节或夹爪状态。

| 场景 | 行为 |
| --- | --- |
| 命令/状态维度错误或含 NaN/Inf | `ValueError` |
| 限幅参数非正或维度错误 | `ValueError` |
| 初始关节违反硬限位 | `JointLimitError` |
| 末端目标不可达或无合法 IK | `ControlResult(success=False)`，状态不变 |
| 未知命令类型 | `TypeError` |

## 9. 测试与验收策略

- 验证三种命令是不同类型并映射到正确 `ControlMode`。
- 验证关节绝对目标先满足硬限位，再满足单步关节增量。
- 验证末端增量范数限幅、IK seed 选解和 IK 后关节增量限幅。
- 验证不可达末端命令失败且完整状态不变。
- 验证夹爪 `[0,1]` 硬限位和单步增量限幅。
- 验证所有非法数据类型、维度、非有限数值和初始状态。
- 固定演示同时覆盖三种模式、成功/失败、至少一次限幅，所有状态均不越界。
- 运行完整仓库测试与 `compileall`。

## 10. 验收标准

1. 三种控制模式有独立、明确且可校验的命令结构。
2. 所有成功步骤满足关节、夹爪硬限位和单步变化限制。
3. 末端增量通过 IK 转换，实际应用位移可追溯。
4. 不可达命令失败且状态原子回滚。
5. 演示序列覆盖三种模式、至少一次限幅和一次安全拒绝，最终无越界或非有限状态。
6. 定向测试、完整仓库测试和语法检查通过。
7. 总计划、实施状态和“实际偏差”按结果同步。

## 11. 实际偏差

- 三类命令、状态、限幅、结果和 `PlanarArmController.step()` 均按设计完成，无接口级偏差。
- 末端增量路径按“笛卡尔范数限幅 → IK seed 选解 → 关节单步限幅”实现；不可达命令返回失败结果且状态不变。
- 固定演示共 11 步，覆盖 3 种控制模式，触发 9 次限幅和 1 次工作空间外安全拒绝，全程无关节、夹爪或单步变化越界。
- `src/tests/test_control.py` 定向测试 26 项通过；完整仓库测试在项目虚拟环境和系统 Python 下均为 266 项通过，`compileall` 通过。
- 未修改 `se3.py`、`kinematics.py`、阶段 A Agent 或依赖文件，未接入 ManiSkill 和外部 API。
