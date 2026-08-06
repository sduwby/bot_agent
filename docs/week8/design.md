# 第 8 周设计文档 — 正逆运动学与关节限位

## 1. 概述

**本周目标：** 在第 7 周坐标变换基础上，用二维双连杆（2R）机械臂建立关节空间与末端笛卡尔空间之间的映射，实现正运动学、解析逆运动学、解分支选择和关节限位校验，并调用 IK 到达 5 个固定目标点。

**规划原文引用：**

> 第 8 周 | 运动学 | 理解关节状态、末端位姿、正逆运动学和关节限位。 | 调用 IK 到达 5 个目标点。

**前置依赖：**

- 第 7 周 `RigidTransform`：统一表示 `base ← end_effector` 末端位姿。
- NumPy：向量、距离和数值校验。
- `pytest`：纯逻辑单元测试。

## 2. 设计边界与原则

- 本周采用二维 2R 教学模型，重点验证运动学数学和接口契约，不声称等价于真实 6/7 自由度机械臂。
- 模型位于基座 XY 平面，两个旋转关节都绕 +Z 轴；零角时两根连杆沿 +X 方向伸直。
- 长度单位为米，关节角单位为弧度；公开接口不接受隐式角度制。
- IK 只求末端位置 `(x, y)`，末端 yaw 由关节解 `q1 + q2` 决定；不求任意位置加姿态约束。
- 解析 IK 返回所有满足关节限位的主值解，并提供基于当前关节状态（seed）的最近解选择。
- 新模块保持纯数学，不启动 AI2-THOR，不调用外部 API，不修改阶段 A Agent，也不提前接入 ManiSkill。

## 3. 运动学约定

### 3.1 关节状态

关节向量为：

```text
q = [q1, q2]
```

- `q1`：第一连杆相对基座 +X 轴的角度。
- `q2`：第二连杆相对第一连杆的角度。
- 正角方向遵循右手定则，即从 +Z 方向观察为逆时针。
- 求解器使用主值角 `[-π, π]`；本周关节限位也必须位于该范围内。

### 3.2 正运动学

两根连杆长度分别为 `l1`、`l2`：

```text
x   = l1 cos(q1) + l2 cos(q1 + q2)
y   = l1 sin(q1) + l2 sin(q1 + q2)
yaw = q1 + q2
```

返回 `RigidTransform("base", "end_effector", ...)`，其平移为 `[x, y, 0]`，旋转为绕 Z 轴的 `yaw`。

### 3.3 解析逆运动学

目标位置为 `(x, y)`，令 `r² = x² + y²`：

```text
cos(q2) = (r² - l1² - l2²) / (2 l1 l2)
q2      = ±acos(cos(q2))
q1      = atan2(y, x) - atan2(l2 sin(q2), l1 + l2 cos(q2))
```

`q2` 正、负对应两个解析分支，分别命名为 `q2_positive` 和 `q2_negative`，避免用视角相关的“肘上/肘下”名称产生歧义。奇异位置两个分支重合时只保留一个解。

### 3.4 工作空间与关节限位

几何可达条件为：

```text
|l1 - l2| ≤ sqrt(x² + y²) ≤ l1 + l2
```

- 超出几何工作空间时抛出 `UnreachableTargetError`。
- 几何可达但所有主值解都违反关节限位时抛出 `JointLimitError`。
- FK 输入只要有一个关节越界就立即失败，不执行静默裁剪。
- 边界比较使用小容差；`acos` 前只对容差内的浮点越界执行 `clip`。

## 4. 架构设计

```text
src/kinematics.py
    ├── JointLimit                 单关节上下界与包含判断
    ├── IKSolution                 关节解、分支、FK 残差
    ├── UnreachableTargetError     几何不可达
    ├── JointLimitError            解违反关节限位
    └── PlanarArm2D
        ├── forward_kinematics()
        ├── inverse_kinematics_all()
        └── solve_ik()

src/run_ik_demo.py
    └── 对 5 个固定目标逐个执行 solve_ik → FK 回代 → 误差/限位验收

src/tests/test_kinematics.py
    └── FK、IK 双分支、seed 选择、边界、限位和 5 点演示测试
```

### 4.1 与第 7 周的连接

`forward_kinematics()` 复用 `se3.RigidTransform` 输出末端位姿，不另建不兼容的 Pose 类型。这样第 8 周得到的 `T_base_end_effector` 可以直接参与第 7 周的坐标链组合。

### 4.2 与后续周次的边界

- 第 9 周才增加 joint position、end-effector delta、gripper action 等控制命令和动作限幅。
- 第 10 周才接入 ManiSkill 的具体机器人、环境 observation/action 和 episode。
- 本周 2R 模型作为解析真值和回归基线，不作为真实机械臂控制器。

## 5. 数据结构

### 5.1 `JointLimit`

```python
@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float

    def contains(self, angle: float, *, tolerance: float = 1e-9) -> bool: ...
```

构造时要求上下界有限、`-π ≤ lower < upper ≤ π`。

### 5.2 `IKSolution`

```python
@dataclass(frozen=True)
class IKSolution:
    joint_positions: tuple[float, float]
    branch: str
    residual: float
```

- `joint_positions`：满足限位的 `(q1, q2)`。
- `branch`：`q2_positive`、`q2_negative` 或奇异点的 `singular`。
- `residual`：用 FK 回代后末端 `(x, y)` 与目标之间的欧氏距离。

### 5.3 `PlanarArm2D`

```python
@dataclass(frozen=True)
class PlanarArm2D:
    link_lengths: tuple[float, float]
    joint_limits: tuple[JointLimit, JointLimit]
```

连杆长度必须是两个有限正数；关节限位必须恰好两个。

## 6. 公开接口设计

```python
class UnreachableTargetError(ValueError): ...

class JointLimitError(ValueError): ...


class PlanarArm2D:
    def forward_kinematics(
        self,
        joint_positions: Sequence[float],
    ) -> RigidTransform: ...

    def inverse_kinematics_all(
        self,
        target_xy: Sequence[float],
    ) -> tuple[IKSolution, ...]: ...

    def solve_ik(
        self,
        target_xy: Sequence[float],
        *,
        seed: Sequence[float] | None = None,
    ) -> IKSolution: ...
```

`inverse_kinematics_all()` 稳定地按 `q2_positive`、`q2_negative` 排序。`solve_ik()` 无 seed 时选择第一个有效解；有 seed 时使用各关节最短角距离平方和选择最近解，相同距离保持稳定顺序。

## 7. 数据流

### 7.1 FK

1. 接收两个关节角并校验形状、有限性和关节限位。
2. 使用解析公式计算 `(x, y, yaw)`。
3. 调用 `RigidTransform.from_rpy()` 构造 `T_base_end_effector`。
4. 返回可继续参与坐标链组合的末端位姿。

### 7.2 IK

1. 接收二维目标并校验形状和有限性。
2. 根据半径判断是否位于 2R 几何工作空间。
3. 计算 `q2` 正负两个分支和对应 `q1`，规范化到主值区间。
4. 去除奇异点重复解，再按关节限位过滤。
5. 对每个有效解调用 FK 回代并计算残差。
6. `solve_ik()` 根据 seed 选择解；没有 seed 时使用稳定的首个分支。

### 7.3 五目标点验收

1. 创建 `l1=0.6 m`、`l2=0.4 m`、两个关节限位均为 `[-π, π]` 的模型。
2. 固定 5 个处于工作空间不同象限/半径的目标点。
3. 以上一个目标的解作为下一个目标的 seed，模拟连续关节状态。
4. 每个目标调用 `solve_ik()`，再用 FK 回代。
5. 校验 5/5 目标均满足关节限位且位置误差 `< 1e-9`。

## 8. 错误处理

| 场景 | 行为 |
| --- | --- |
| 连杆数量错误、非有限或非正 | `ValueError` |
| 关节限位数量错误或边界非法 | `ValueError` |
| FK 关节向量不是两个有限数 | `ValueError` |
| FK 关节越界 | `JointLimitError` |
| IK 目标不是两个有限数 | `ValueError` |
| 目标超出几何工作空间 | `UnreachableTargetError` |
| 几何解全部违反限位 | `JointLimitError` |
| seed 不是两个有限数或越界 | `ValueError` 或 `JointLimitError` |

## 9. 测试与验收策略

- 用零角、单关节 90° 和已知非零角验证 FK 位置与 yaw。
- 用 FK 生成目标再做 IK，验证每个返回解的 FK 残差。
- 验证一般目标存在两个分支，奇异伸直点只返回一个解。
- 验证 seed 能稳定选择更近的关节解。
- 验证外工作空间、内工作空间空洞、非法输入和关节越界。
- 验证几何可达但被限位排除与几何不可达使用不同异常。
- 运行 5 目标点演示，要求 5/5 成功且最大误差 `< 1e-9`。
- 运行完整仓库测试与 `compileall`，确保第 1–7 周功能无回归。

## 10. 验收标准

1. 能解释关节空间、末端位姿、FK、IK 双分支、奇异点和关节限位。
2. FK 返回正确的 `T_base_end_effector`。
3. IK 返回全部合法主值解，并能根据 seed 选择最近解。
4. 几何不可达和限位不可达可区分诊断。
5. 5 个固定目标全部通过 IK→FK 回代，最大位置误差 `< 1e-9`，所有关节均在限位内。
6. 新增测试、完整仓库测试和语法检查通过。
7. 总开发计划、实施状态和本文“实际偏差”按结果更新。

## 11. 实际偏差

- 核心数据结构、公开接口、2R 解析公式、异常分类和 seed 选解均按设计完成，无接口级偏差。
- 五个固定目标全部通过 IK→FK 回代，最大位置误差为 `1.777e-16`，低于 `1e-9` 验收阈值。
- `src/tests/test_kinematics.py` 定向测试 36 项通过；完整仓库测试在项目虚拟环境和系统 Python 下均为 240 项通过，`compileall` 通过。
- 实现直接复用第 7 周 `RigidTransform` 输出 `T_base_end_effector`，未修改 `se3.py` 接口。
- 未新增依赖、未启动 AI2-THOR、未接入 ManiSkill 或外部 API，保持了第 8 周规划边界。
