# 第 7 周设计文档 — 坐标系与 SE(3) 变换

## 1. 概述

**本周目标：** 从 AI2-THOR 提供的抽象抓取、放置动作过渡到机器人学中的连续空间表示，建立世界、基座、末端、相机和物体五类坐标系的统一约定，实现可验证的 SE(3) 刚体变换，并用一条完整坐标变换链说明物体位姿如何在不同坐标系之间转换。

**规划原文引用：**

> 第 7 周 | 坐标系 | 学习世界、基座、相机、末端和物体坐标系；完成 SE(3) 变换练习。 | 能解释并代码验证坐标变换链。

**前置依赖：**

- 阶段 A 门禁已通过：现有 Agent 在结构化环境状态下完成 12 条固定任务评测，规则和真实 LLM 后端均达到 10/12（83.3%）。
- 项目继续使用 Python 和 `pytest`；矩阵运算使用 NumPy，并将其声明为直接依赖。
- 本周只建立机器人学数学基础，不要求机械臂模型、ManiSkill、逆运动学或真实硬件。

## 2. 设计边界与原则

- 新模块保持纯数学、无模拟器状态和文件 I/O，单元测试不启动 Unity，也不依赖 API。
- 不修改阶段 A 的 `AgentExecutor`、技能、搜索、记忆和评测行为，避免把学习性代码耦合进已通过门禁的闭环。
- 所有变换都携带源坐标系和目标坐标系名称，组合时校验链路连续性，防止只看矩阵形状而把乘法顺序写反。
- 只支持三维刚体变换：旋转属于 SO(3)，齐次矩阵属于 SE(3)；不支持缩放、错切和投影变换。
- 平移统一使用米，角度 API 统一使用弧度。示例中的角度通过 `numpy.deg2rad` 显式转换。
- 本周实现 RPY 构造，不实现欧拉角反解、四元数、插值、雅可比、正逆运动学；这些内容不属于第 7 周验收范围。

## 3. 坐标系与数学约定

### 3.1 坐标系

| 名称 | 标识 | 含义 |
| --- | --- | --- |
| 世界坐标系 | `world` | 场景中的全局固定参考系 |
| 基座坐标系 | `base` | 机械臂底座固定参考系 |
| 末端坐标系 | `end_effector` | 机械臂末端执行器参考系 |
| 相机坐标系 | `camera` | 眼在手上相机的光学参考系；本周只处理已知外参 |
| 物体坐标系 | `object` | 固连于目标物体的局部参考系 |

### 3.2 变换记号

使用 `T_target_source` 表示“把源坐标系中的坐标转换到目标坐标系”：

```text
p_target = T_target_source @ p_source
```

点采用齐次列向量 `[x, y, z, 1]^T`。齐次变换矩阵为：

```text
                  [ R_target_source  t_target_source ]
T_target_source = [        0 0 0              1       ]
```

其中 `R` 是 3×3 旋转矩阵，`t` 是在目标坐标系中表达的三维平移。

### 3.3 旋转约定

- 使用右手坐标系和列向量。
- RPY 输入顺序为 `roll, pitch, yaw`，分别绕固定的 X、Y、Z 轴旋转。
- 组合矩阵为 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`；对点的实际作用顺序是先 roll、再 pitch、最后 yaw。
- 合法旋转必须满足 `R.T @ R ≈ I` 且 `det(R) ≈ +1`。

### 3.4 组合与逆变换

连续链路按从左到右的坐标系名称匹配、按矩阵从左到右相乘：

```text
T_world_object =
    T_world_base
    @ T_base_end_effector
    @ T_end_effector_camera
    @ T_camera_object
```

逆变换满足：

```text
R_source_target = R_target_source.T
t_source_target = -R_target_source.T @ t_target_source
```

必须能通过 `T_source_target @ T_target_source ≈ I` 和点的往返变换验证。

## 4. 架构设计

```text
src/se3.py
    ├── rotation_matrix_from_rpy()  RPY → SO(3)
    ├── RigidTransform              带坐标系语义的 SE(3) 变换
    │   ├── identity()/from_rpy()/from_matrix()
    │   ├── compose()
    │   ├── inverse()
    │   └── transform_point()
    └── compose_chain()             校验并组合连续变换链

src/run_transform_demo.py
    └── 构造 world ← base ← end_effector ← camera ← object
        ├── 输出各段与最终齐次矩阵
        ├── 验证逐段转换和一次组合结果一致
        └── 验证逆变换往返误差

src/tests/test_se3.py
    └── 纯逻辑单元测试，不启动 AI2-THOR
```

### 4.1 与现有模块的关系

`se3.py` 在第 7 周保持独立。现有 `observation.py` 中的位置字典和 `rotation_y` 仍表示 AI2-THOR Agent 的简化状态，不在本周转换为完整机械臂位姿。后续第 8–10 周接入机械臂仿真时，再由适配层把模拟器位姿转换成 `RigidTransform`，而不是让数学模块直接读取某个模拟器的 metadata。

## 5. 数据结构

### 5.1 `RigidTransform`

```python
@dataclass(frozen=True)
class RigidTransform:
    """将 source_frame 中的三维坐标转换到 target_frame。"""

    target_frame: str
    source_frame: str
    matrix: np.ndarray
```

构造时执行以下验证：

- 两个坐标系名称必须是非空字符串；名称相同时矩阵必须是单位矩阵，并优先由 `identity(frame)` 构造。
- 矩阵形状必须是 `(4, 4)`，所有元素必须有限。
- 最后一行必须近似 `[0, 0, 0, 1]`。
- 左上角旋转块必须正交且行列式近似 `+1`，拒绝反射和缩放。
- 内部保存只读矩阵副本，避免调用者修改原始数组或对象属性中的数组内容而改变变换。

### 5.2 组合结果

`T_a_b.compose(T_b_c)` 返回新的 `T_a_c`。如果第一段的 `source_frame` 与第二段的 `target_frame` 不相同，则抛出 `ValueError`，错误信息包含实际链路名称。

`compose_chain()` 至少接收一个变换，依次调用相同的组合校验；单个输入返回其等价副本。

## 6. 公开接口设计

```python
def rotation_matrix_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """按 Rz @ Ry @ Rx 约定生成 3×3 旋转矩阵。"""


@dataclass(frozen=True)
class RigidTransform:
    target_frame: str
    source_frame: str
    matrix: np.ndarray

    @classmethod
    def identity(cls, frame: str) -> "RigidTransform": ...

    @classmethod
    def from_rpy(
        cls,
        target_frame: str,
        source_frame: str,
        translation: Sequence[float],
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> "RigidTransform": ...

    @classmethod
    def from_matrix(
        cls,
        target_frame: str,
        source_frame: str,
        matrix: ArrayLike,
    ) -> "RigidTransform": ...

    def compose(self, other: "RigidTransform") -> "RigidTransform": ...

    def inverse(self) -> "RigidTransform": ...

    def transform_point(self, point: Sequence[float]) -> np.ndarray: ...


def compose_chain(*transforms: RigidTransform) -> RigidTransform: ...
```

目标和源名称相同时只允许单位矩阵，`identity(frame)` 是推荐构造入口。`transform_point()` 本周只接受形状为 `(3,)` 的单点，非法形状和非有限数值抛出 `ValueError`，不隐式补零或截断。

## 7. 数据流

### 7.1 构造变换

1. 调用方提供目标坐标系、源坐标系、三维平移和 RPY 角。
2. `rotation_matrix_from_rpy()` 生成旋转块。
3. `from_rpy()` 组装 4×4 齐次矩阵。
4. `RigidTransform` 校验坐标系名称、矩阵形状、有限性和 SE(3) 约束。
5. 返回不可重新绑定字段、内部矩阵不受原始输入后续修改影响的变换对象。

### 7.2 五坐标系示例

1. 定义 `T_world_base`：机械臂基座在世界中的位姿。
2. 定义 `T_base_end_effector`：末端相对基座的位姿。
3. 定义 `T_end_effector_camera`：眼在手上相机外参。
4. 定义 `T_camera_object`：视觉系统给出的物体相对相机位姿。
5. 调用 `compose_chain()` 得到 `T_world_object`。
6. 将物体原点转换到世界坐标，分别用“逐段应用”和“组合矩阵一次应用”计算，结果必须一致。
7. 使用 `T_world_object.inverse()` 把世界坐标转回物体坐标，往返误差必须小于阈值。

## 8. 错误处理

| 输入或操作 | 行为 |
| --- | --- |
| 空坐标系名称 | `ValueError` |
| 目标和源名称相同但矩阵不是单位矩阵 | `ValueError`，提示坐标系语义冲突 |
| 非 4×4、含 NaN/Inf 的矩阵 | `ValueError` |
| 非正交旋转、行列式非 +1、齐次末行错误 | `ValueError` |
| 平移不是 3 个有限数 | `ValueError` |
| 点不是 3 个有限数 | `ValueError` |
| 组合链名称不连续 | `ValueError` |
| 空 `compose_chain()` | `ValueError` |

## 9. 测试与验收策略

- 单元测试验证单位变换、纯平移和绕 Z 轴 90° 的已知结果。
- 验证 RPY 组合顺序，避免只测试单位矩阵而掩盖轴顺序错误。
- 验证组合律的数值结果和坐标系名称传播。
- 验证逆变换、矩阵乘积单位性和点的往返恢复。
- 验证五坐标系示例中逐段转换与一次组合等价。
- 验证所有主要非法输入和链路不连续失败路径。
- 运行完整仓库测试，确保阶段 A 的 173 个既有测试无回归。
- 运行演示脚本；脚本打印坐标链、最终物体世界坐标和最大往返误差，并在数值校验失败时返回非零退出码。

## 10. 验收标准

第 7 周只有同时满足以下条件才标记完成：

1. 能明确解释 `T_target_source`、矩阵乘法顺序以及平移向量在哪个坐标系中表达。
2. `RigidTransform` 能构造、组合、求逆和转换三维点，并主动拒绝非法 SE(3) 矩阵。
3. `world ← base ← end_effector ← camera ← object` 五坐标系链可由代码运行验证。
4. 逐段转换与组合转换结果一致，逆变换往返最大绝对误差 `< 1e-9`。
5. 新增测试和完整仓库测试全部通过，`compileall` 通过。
6. 总开发计划更新第 7 周状态、当前代码结构和下一步；若实际实现偏离本设计，在本文末尾补充“实际偏差”。

## 11. 实际偏差

- 核心接口、坐标系命名、矩阵约定和错误处理均按设计完成，无接口级偏差。
- `requirements.txt` 将 NumPy 显式约束为 `numpy>=1.24,<3`；当前项目虚拟环境使用 NumPy 2.0.2。
- 演示链的逐段/组合最大误差为 `2.220e-16`，逆变换往返最大误差为 `4.718e-16`，均低于 `1e-9` 验收阈值。
- `src/tests/test_se3.py` 定向测试 31 项通过；完整仓库测试 204 项通过，`compileall` 通过。
- 本周未接入 `AgentExecutor`、AI2-THOR metadata、ManiSkill、IK 或外部 API，保持了规划中的阶段边界。
