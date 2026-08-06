# 第 8 周实施计划 — 正逆运动学与关节限位

## 1. 任务分解

| 编号 | 任务 | 文件 | 依赖 |
| --- | --- | --- | --- |
| 1 | 实现关节限位、IK 结果和异常类型 | `src/kinematics.py` | 第 7 周 `se3.py` |
| 2 | 实现 2R 正运动学 | `src/kinematics.py` | 1 |
| 3 | 实现解析 IK、双分支、限位过滤和 seed 选解 | `src/kinematics.py` | 1、2 |
| 4 | 编写运动学单元测试 | `src/tests/test_kinematics.py` | 1–3 |
| 5 | 实现并测试 5 目标点 IK 演示 | `src/run_ik_demo.py`、`src/tests/test_kinematics.py` | 3、4 |
| 6 | 运行完整验证并同步文档 | `docs/week8/*`、`具身智能完整项目_业余开发计划.md` | 5 |

## 2. 文件结构

**新增文件：**

- `src/kinematics.py` — 2R 机械臂 FK、解析 IK、关节限位和解选择。
- `src/run_ik_demo.py` — 5 个固定目标点的 IK→FK 验收脚本。
- `src/tests/test_kinematics.py` — 运动学正常路径、边界和失败路径测试。
- `docs/week8/design.md` — 第 8 周设计文档。
- `docs/week8/plan.md` — 第 8 周实施计划。

**修改文件：**

- `具身智能完整项目_业余开发计划.md` — 验收后更新第 8 周状态、当前代码结构和下一步。
- `docs/week8/design.md` — 验收后记录实际偏差。

**明确不修改：**

- `requirements.txt`：第 7 周的 NumPy 已满足本周需求，不增加 SciPy 或机器人框架。
- 阶段 A Agent、技能、任务集和报告。
- 第 7 周 `se3.py` 接口；本周作为稳定依赖复用。

## 3. 实施步骤

### 步骤 1：数据结构与输入校验

- 实现 `JointLimit`，校验有限上下界和 `[-π, π]` 主值范围。
- 实现 `IKSolution`，保存只读关节 tuple、分支和残差。
- 实现 `UnreachableTargetError` 与 `JointLimitError`。
- 实现二维有限向量、主值角和关节状态校验辅助函数。

### 步骤 2：正运动学

- 校验 `(q1, q2)` 数量、有限性与限位。
- 根据 2R 解析公式计算末端 `(x, y, yaw)`。
- 复用 `RigidTransform.from_rpy()` 返回 `base ← end_effector`。
- 增加零角、90° 和一般角度测试。

### 步骤 3：逆运动学

- 计算目标半径和 `cos(q2)`，区分几何不可达。
- 计算 `q2_positive`、`q2_negative` 及对应 `q1`。
- 将角度规范化到 `[-π, π]`，奇异解去重。
- 按关节限位过滤，并用 FK 回代生成 residual。
- 无有效限位解时抛出 `JointLimitError`。
- `solve_ik()` 无 seed 稳定选首解；有 seed 时选择最短角距离最近解。

### 步骤 4：边界与失败测试

- 测试工作空间外边界和内空洞。
- 测试最大伸展、折叠附近的浮点容差和解去重。
- 测试 FK 越界、IK 全部被限位排除、非法目标和非法 seed。
- 测试双分支均能通过 FK 回代并满足限位。

### 步骤 5：5 点演示

- 固定模型参数和 5 个目标点，不读取隐藏模拟器状态。
- 第一个目标无 seed，后续目标以上一解作为 seed。
- 输出目标、关节角（弧度/角度）、分支、FK 位置和误差。
- 5/5 成功且最大误差 `< 1e-9` 时返回 0，否则返回非零。
- 单元测试锁定目标数量、成功数、最大误差和所有关节限位。

### 步骤 6：完整验证与文档同步

- 运行运动学定向测试和演示脚本。
- 运行完整 `pytest` 与 `compileall`。
- 检查 Git 差异，不包含运行产物、缓存或无关文件。
- 回填设计偏差与本计划执行状态。
- 将总计划第 8 周标记为完成或部分完成，并按实际结果更新下一步。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向单元测试 | `.venv/bin/python -m pytest -q src/tests/test_kinematics.py` | FK、IK、分支、seed、限位和失败路径 |
| 5 点演示 | `.venv/bin/python src/run_ik_demo.py` | 5/5 目标 IK→FK 回代 |
| 完整回归 | `.venv/bin/python -m pytest -q` | 第 1–8 周功能无回归 |
| 语法检查 | `.venv/bin/python -m compileall -q src` | 所有 Python 文件可编译 |

## 5. 验收标准

- [x] `JointLimit`、`IKSolution` 和两类运动学异常实现完成。
- [x] FK 能返回正确的 `T_base_end_effector`。
- [x] IK 能返回一般目标的两个主值分支并去除奇异重复解。
- [x] `solve_ik()` 能根据 seed 选择最近合法解。
- [x] 几何不可达、限位不可达、非法输入均有测试。
- [x] 5 个固定目标全部到达，最大 FK 回代误差 `< 1e-9`。
- [x] 定向测试、完整仓库测试和 `compileall` 通过。
- [x] 阶段 A 与第 7 周接口无行为性改动。
- [x] 总开发计划、设计偏差和执行状态已同步。

## 6. 执行状态

第 8 周实现与验收已完成：

- 新增 `src/kinematics.py`、`src/run_ik_demo.py` 和 `src/tests/test_kinematics.py`。
- 五个固定目标全部到达，最大 FK 回代误差为 `1.777e-16`，所有关节解均满足限位。
- 定向测试 36 项通过；完整仓库测试在项目虚拟环境和系统 Python 下均为 240 项通过，`compileall` 通过。
- 未修改阶段 A 与第 7 周接口，可以进入第 9 周控制接口工作。
