# 第 10 周实施计划 — ManiSkill 入门与脚本策略验收

## 1. 任务分解

| 编号 | 任务 | 文件 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固定可复现的 ManiSkill 可选依赖与独立环境说明 | `requirements-maniskill.txt` | Python 3.11 |
| 2 | 定义任务配置、观察摘要和 episode 结果契约 | `src/maniskill_intro.py` | NumPy |
| 3 | 实现 PickCube/PushCube scripted policy | `src/maniskill_intro.py` | 2、第 9 周控制概念 |
| 4 | 实现延迟导入、CPU 环境创建和 episode runner | `src/maniskill_intro.py` | 1–3 |
| 5 | 编写契约、状态机和生命周期测试 | `src/tests/test_maniskill_intro.py` | 2–4 |
| 6 | 实现两任务 CLI 演示与机器可读摘要 | `src/run_maniskill_demo.py` | 4、5 |
| 7 | 安装独立环境并完成真实 CPU 仿真验收 | `.venv-maniskill/`（忽略） | 1–6 |
| 8 | 完整验证与进度文档同步 | `docs/week10/*`、总计划 | 7 |

## 2. 文件结构

**新增文件：**

- `requirements-maniskill.txt` — 第 10 周可选重依赖。
- `src/maniskill_intro.py` — 环境适配、观察摘要、scripted policy 和 episode runner。
- `src/run_maniskill_demo.py` — PickCube/PushCube 验收入口。
- `src/tests/test_maniskill_intro.py` — 不依赖真实仿真的纯逻辑回归测试。
- `docs/week10/design.md` — 第 10 周设计文档。
- `docs/week10/plan.md` — 第 10 周实施计划。

**修改文件：**

- `.gitignore` — 忽略 `.venv-maniskill/`（若现有规则未覆盖）。
- `具身智能完整项目_业余开发计划.md` — 验收后更新第 10 周状态、代码结构和下一步。
- `docs/week10/design.md`、`docs/week10/plan.md` — 回填实际偏差和执行状态。

**明确不修改：**

- `requirements.txt`：AI2-THOR 基础运行环境不承担 ManiSkill 重依赖。
- `src/control.py`、`src/kinematics.py`、`src/se3.py`：作为概念前置，不与仿真框架耦合。
- 阶段 A Agent、技能、任务集、运行产物和历史周次文档。

## 3. 实施步骤

### 步骤 1：运行环境与数据契约

- 新建 Python 3.11 `.venv-maniskill`，安装 `requirements-maniskill.txt`。
- 配置 `ManiSkillTaskConfig`，只开放本周两个任务与 CPU backend。
- 定义观察字段和 episode 结果，保证 CLI 可序列化。
- 为可选依赖实现延迟导入与明确错误信息。

### 步骤 2：观察与动作契约

- 递归展开 mapping observation，记录叶子字段的 path、shape、dtype。
- 从 `extra` 安全读取 TCP、物体和目标位置。
- 把 Torch/NumPy 标量统一转换为 Python 标量。
- 每步检查 action 维度、有限性和 action space 包含关系。

### 步骤 3：两任务 scripted policy

- 实现复用的 TCP 目标追踪和动作裁剪。
- PickCube 依次执行接近、下探、夹持、搬运、稳定。
- PushCube 根据实时物体—目标方向选择背后接近点并推动。
- 为阶段切换设置位置阈值与最大停留步数，避免策略死锁。

### 步骤 4：episode runner 与 CLI

- `run_episode()` 负责 reset、policy reset、step 循环、指标累计和 close。
- 捕获初始 observation schema 和 action space shape。
- CLI 默认依次运行两个任务，打印 reward、success、终止类型和阶段轨迹。
- 任一任务失败时返回非零退出码。

### 步骤 5：测试与真实调试

- 先用 fake environment 完成纯逻辑测试。
- 在真实 `physx_cpu` 环境检查实际 state_dict 路径和 action shape。
- 用 seed=0 调整阶段阈值与控制尺度，直到两任务都成功。
- 若实际 API 与设计不同，先更新设计文档再调整实现。

### 步骤 6：验证与同步

- 运行定向测试、两任务真实演示、完整 `pytest` 与 `compileall`。
- 检查 Git 差异，确保无 `.venv-maniskill`、缓存、视频或轨迹进入版本控制。
- 回填本计划执行状态、设计实际偏差和总开发计划。
- 全部验收通过后将下一步更新为第 11 周专家轨迹采集。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向测试 | `.venv/bin/python -m pytest -q src/tests/test_maniskill_intro.py` | 数据契约、策略状态机、生命周期和错误路径 |
| 真实演示 | `.venv-maniskill/bin/python src/run_maniskill_demo.py --task all --seed 0` | 两任务真实 CPU episode 成功 |
| 完整回归 | `.venv/bin/python -m pytest -q` | 第 1–9 周功能无回归 |
| 语法检查 | `.venv/bin/python -m compileall -q src` | 所有 Python 文件可编译 |

## 5. 验收标准

- [x] ManiSkill 独立环境安装并通过版本导入检查。
- [x] PickCube/PushCube observation 字段和 action shape 可追溯。
- [x] 两个 scripted policy 的 action 均合法且有限。
- [x] PickCube 固定 seed episode 成功。
- [x] PushCube 固定 seed episode 成功。
- [x] reward、success、terminated、truncated 语义在结果中分离。
- [x] 定向测试、完整回归和 `compileall` 通过。
- [x] 未修改第 1–9 周稳定接口，未提交运行产物或独立环境。
- [x] 总计划、设计偏差与执行状态完成同步。

## 6. 执行状态

第 10 周实现与验收已完成：

- 新增独立 `requirements-maniskill.txt`，实际环境为 Python 3.11、ManiSkill 3.0.1、SAPIEN 3.0.3、Torch 2.13.0 和 macOS Pinocchio binding 4.1.0；Homebrew MoltenVK 1.4.2 提供 Vulkan→Metal 驱动。
- 新增 `src/maniskill_intro.py`，完成环境延迟导入、macOS ICD 定位、观察字段摘要、动作契约、两任务 scripted policy 和 Gymnasium episode runner。
- 新增 `src/run_maniskill_demo.py`；组合运行中 PickCube 20 步、PushCube 14 步成功，两者均正常 terminated 且未 truncated。
- 新增 19 项定向测试；完整仓库 285 项通过，`compileall` 和独立环境 `pip check` 通过；额外 seed 1 组合运行也通过。
- 未修改第 1–9 周稳定接口，未生成或提交视频、轨迹与运行目录，可以进入第 11 周专家轨迹采集。
