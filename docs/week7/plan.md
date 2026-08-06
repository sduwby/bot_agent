# 第 7 周实施计划 — 坐标系与 SE(3) 变换

## 1. 任务分解

| 编号 | 任务 | 文件 | 依赖 |
| --- | --- | --- | --- |
| 1 | 将 NumPy 声明为直接运行依赖 | `requirements.txt` | 无 |
| 2 | 实现 RPY 旋转、带坐标系语义的刚体变换和链式组合 | `src/se3.py` | 1 |
| 3 | 为单位变换、构造、组合、求逆和非法输入编写单元测试 | `src/tests/test_se3.py` | 2 |
| 4 | 实现五坐标系变换链演示和数值自检 | `src/run_transform_demo.py` | 2 |
| 5 | 运行演示、全量测试和语法检查 | 全仓库 | 3、4 |
| 6 | 根据实际实现回溯设计文档并同步总开发计划 | `docs/week7/design.md`、`具身智能完整项目_业余开发计划.md` | 5 |

## 2. 文件结构

**新增文件：**

- `src/se3.py` — SO(3)/SE(3) 构造、校验、组合、求逆和点变换。
- `src/run_transform_demo.py` — 世界、基座、末端、相机、物体坐标链演示。
- `src/tests/test_se3.py` — 坐标变换纯逻辑单元测试。
- `docs/week7/design.md` — 第 7 周设计文档。
- `docs/week7/plan.md` — 第 7 周实施计划。

**修改文件：**

- `requirements.txt` — 显式增加 NumPy 直接依赖，不再只依赖 AI2-THOR 的传递安装。
- `具身智能完整项目_业余开发计划.md` — 实现验收后更新第 7 周状态、代码结构和下一步。
- `docs/week7/design.md` — 实现验收后记录实际偏差。

**明确不修改：**

- `src/executor.py`、`src/task_skills.py`、`src/observation.py`：本周不把坐标数学接入阶段 A Agent。
- `evaluation/stage_a_tasks.json` 和第 6 周报告：本周不改变既有门禁口径。

## 3. 实施步骤

### 步骤 1：直接依赖

- 在 `requirements.txt` 增加兼容当前 Python 3.9/3.13 环境的 NumPy 版本范围。
- 使用当前 `.venv` 检查实际 NumPy 版本和导入状态。
- 不引入 SciPy；RPY 构造、矩阵校验和逆变换只使用 NumPy。

### 步骤 2：实现 SE(3) 核心模块

- 实现 `rotation_matrix_from_rpy(roll, pitch, yaw)`，固定使用 `Rz @ Ry @ Rx`。
- 实现 `RigidTransform.identity()`、`from_rpy()` 和 `from_matrix()`。
- 构造时校验坐标系名称、矩阵形状、有限数、齐次末行、旋转正交性和行列式。
- 实现 `compose()`，同时检查 `source_frame == other.target_frame`。
- 实现 `inverse()`，使用刚体变换解析逆，不调用通用矩阵求逆掩盖非法旋转。
- 实现 `transform_point()`，仅接受一个三维有限点。
- 实现 `compose_chain()`，拒绝空链和不连续链路。
- 所有公开函数保留类型标注和简短中文文档字符串。

### 步骤 3：先完成核心单元测试

按每个功能完成后立即增加对应测试：

1. 单位变换保持点不变。
2. 纯平移给出人工可计算结果。
3. 绕 Z 轴 90° 将 X 轴单位向量变到 Y 轴。
4. 非零 roll、pitch、yaw 的结果与显式 `Rz @ Ry @ Rx` 一致。
5. 两段和多段变换组合结果与逐段点变换一致。
6. 求逆后矩阵乘积近似单位矩阵，点往返恢复。
7. 输入矩阵副本隔离且对象中的矩阵只读：外部修改不能改变已构造对象。
8. 空名称、同名非单位变换、错误形状、NaN/Inf、错误末行、缩放、反射全部被拒绝。
9. 非三维点、空组合链和坐标系名称不连续全部被拒绝。

数值比较统一使用 `numpy.testing.assert_allclose`；默认绝对误差按实现验证容差设置，最终往返验收使用 `< 1e-9`。

### 步骤 4：实现五坐标系演示

在 `src/run_transform_demo.py` 中使用固定、可手算或可由测试锁定的变换：

```text
world ← base ← end_effector ← camera ← object
```

脚本必须：

- 打印每段的 `target ← source` 名称和齐次矩阵。
- 组合并打印 `T_world_object`。
- 计算物体原点和至少一个非原点局部点的世界坐标。
- 分别计算逐段转换和组合转换，报告两者最大绝对误差。
- 用逆变换恢复物体局部点，报告往返最大绝对误差。
- 当链路名称、逐段等价性或往返误差不满足验收条件时返回非零退出码。

### 步骤 5：验证

按以下顺序执行：

1. 新模块定向测试。
2. 演示脚本数值自检。
3. 完整仓库测试，确认阶段 A 无回归。
4. 全部 Python 文件语法检查。
5. 检查 Git 差异，确认没有 `runs/`、缓存、密钥或无关文件进入变更。

### 步骤 6：进度同步

只有步骤 5 全部通过后才进行：

- 将总开发计划中的第 7 周标记为“✅ 完成”；如果只有部分验收通过，则标记为“⚠️ 部分完成”并列出缺口。
- 在“当前代码结构”加入 `se3.py`、`run_transform_demo.py` 和 `test_se3.py`。
- 将“下一步”更新为第 8 周运动学，前提是第 7 周验收全部通过。
- 在 `docs/week7/design.md` 的“实际偏差”记录最终实现与设计的差异。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向单元测试 | `.venv/bin/python -m pytest -q src/tests/test_se3.py` | SE(3) 正常路径、边界和失败路径 |
| 演示验收 | `.venv/bin/python src/run_transform_demo.py` | 五坐标系链、逐段等价、逆变换往返 |
| 完整回归 | `.venv/bin/python -m pytest -q` | 新功能通过且阶段 A 既有行为无回归 |
| 语法检查 | `.venv/bin/python -m compileall -q src` | 所有 Python 文件可编译 |

本周验收不需要启动 AI2-THOR Controller；完整回归中的既有测试继续使用 mock。第 7 周的数学正确性不能用“模拟器能启动”替代。

## 5. 验收标准

- [x] `requirements.txt` 显式声明 NumPy。
- [x] `RigidTransform` 支持单位变换、RPY 构造、矩阵构造、组合、求逆和三维点转换。
- [x] 非法旋转矩阵、非法齐次矩阵、非有限输入和不连续坐标链均有明确失败行为。
- [x] `world ← base ← end_effector ← camera ← object` 示例可运行并输出完整变换链。
- [x] 逐段点转换与组合矩阵点转换结果一致。
- [x] 逆变换往返最大绝对误差 `< 1e-9`。
- [x] 新增测试、完整仓库测试和 `compileall` 全部通过。
- [x] 现有 Agent、评测任务和阶段 A 报告未发生行为性改动。
- [x] 总开发计划和设计偏差已按实际结果同步。

## 6. 执行状态

第 7 周实现与验收已完成：

- 新增 `src/se3.py`、`src/run_transform_demo.py` 和 `src/tests/test_se3.py`。
- 定向测试 31 项通过，完整仓库测试 204 项通过，`compileall` 通过。
- 演示链逐段/组合最大误差 `2.220e-16`，逆变换往返最大误差 `4.718e-16`。
- 未修改阶段 A Agent、技能、任务集或评测行为，可以进入第 8 周运动学工作。
