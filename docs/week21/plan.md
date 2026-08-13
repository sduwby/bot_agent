# 第 21 周实施计划 — 世界模型支线（小型 dynamics + CEM）

## 1. 任务分解

| 序号 | 子任务 | 文件/产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化设计（dynamics 输入/输出、CEM 参数、评测协议） | `docs/week21/design.md` | 第 15/20 周 |
| 2 | 实现 v3 数据读回与 transition 构造 | `src/world_model.py` | 1 |
| 3 | 实现 PyTorch MLP 训练与 NumPy checkpoint 导出 | `src/world_model.py` | 2 |
| 4 | 实现 NumPy DynamicsModel + CEMPlanner | `src/world_model.py` | 3 |
| 5 | 实现 live MPC episode 与演示评测 | `src/world_model.py`、`src/run_world_model.py` | 4 |
| 6 | 编写定向测试（数据管线、CEM、episode 执行、CLI 校验） | `src/tests/test_world_model.py` | 4–5 |
| 7 | 4060/本地训练 + 10 seed PushCube 演示评测 | `runs/maniskill_world_model/` | 5 |
| 8 | 回填实际偏差并同步总开发计划 | `docs/week21/*`、总计划 | 7 |

## 2. 文件结构

### 2.1 计划新增

- `src/world_model.py` — 第 21 周：dynamics 训练、NumPy 前向、CEM 规划与 MPC episode。
- `src/run_world_model.py` — 第 21 周：训练与演示评测 CLI。
- `src/tests/test_world_model.py` — 第 21 周定向测试。
- `docs/week21/design.md` / `docs/week21/plan.md`。

### 2.2 计划修改

- `具身智能完整项目_业余开发计划.md` — 第 21 周状态、代码结构与下一步。

### 2.3 明确不修改

- 第 15–20 周数据、checkpoint、评测协议与 `skill_router.py`；
- ManiSkill 环境契约、动作空间。

## 3. 实施步骤

### 步骤 1：设计固化

1. 确认 dynamics 为 22→18 的 MLP（增量预测），状态/动作口径与第 12 周一致。
2. 确认 CEM 超参（N=128、H=8、精英 10%、4 轮）与得分函数（终点 obj-goal 距离）。
3. 记录到 `docs/week21/design.md`。

### 步骤 2：数据读回

1. `load_dynamics_data`：从 week15-cam v3 读 PushCube episode，构造 (state, action, next_state)。
2. episode 边界处理：每条 episode 最后一步无 next_state，丢弃。
3. 按 episode 顺序切分 train/validation（80/20）。

### 步骤 3：训练与导出

1. PyTorch MLP：输入 22（state+action 归一化后拼接）→ 128 → 128 → 18，MSE(Δstate)。
2. 训练循环 + validation loss 记录 + best checkpoint。
3. 导出 NumPy：每层权重/偏置存 `weights.npz`，归一化参数与超参存 `metadata.json`。

### 步骤 4：NumPy 前向与 CEM

1. `DynamicsModel.predict_delta/predict_next`（ReLU MLP 前向）。
2. `CEMPlanner.plan`：高斯采样 → 批量前向 → 精英重拟合 → 迭代 → 首步动作。
3. 得分 = 预测序列末帧 `obj-goal` 距离取负；非有限预测给最低分。

### 步骤 5：MPC episode 与 CLI

1. `run_world_model_episode`：live env（state_dict）+ `observation_to_state` 提取 18D + 每步 CEM 规划 + step。
2. `run_world_model.py`：`train` / `eval` 两个子命令；`eval` 跑固定 seed 并落盘 `demo_report.json`（含 scripted 对照）。

### 步骤 6：测试

1. 纯逻辑：数据构造（边界丢弃、切分）、归一化、CEM 精英选择与动作钳制、非有限处理。
2. fake env：MPC episode 执行路径、成功率聚合、报告结构。
3. 完整回归、`compileall`、`pip check`、`git diff --check`。

### 步骤 7：真实演示

1. `.venv-lerobot` 训练 dynamics（PushCube，CPU/MPS 均可）。
2. 固定 seed 500–509 跑 CEM MPC 演示 + scripted 基线对照。

### 步骤 8：文档同步

1. 回填 design.md 实际偏差与本计划执行状态。
2. 更新总计划第 21 周状态、代码结构与下一步（第 22 周阶段评测）。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向测试 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_world_model.py` | 数据管线、CEM、episode、CLI |
| 完整回归 | `.venv-lerobot/bin/python -m pytest -q --ignore=src/tests/test_llm_planner.py` | 第 1–21 周无回归 |
| 语法检查 | `.venv-lerobot/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 依赖检查 | `.venv-lerobot/bin/python -m pip check` | 独立环境依赖一致 |
| 差异检查 | `git diff --check` | 无空白和补丁格式错误 |

## 5. 单元测试覆盖

- `load_dynamics_data`：episode 末步丢弃、shape 校验、train/validation 无泄漏；
- 归一化与 checkpoint 读写往返一致；
- `DynamicsModel.predict_delta` 数值正确（小权重手算）；
- `CEMPlanner`：采样 shape、精英占比、迭代收敛方向（得分不降）、动作钳制、非有限防御；
- `run_world_model_episode`：fake env 执行路径与成功标记；
- CLI 参数校验与退出码。

## 6. 验收清单

- [ ] dynamics 训练完成，checkpoint 可重载。
- [ ] CEM 规划器纯 NumPy 可用且可测试。
- [ ] PushCube 10 次固定 seed 演示完成并落盘。
- [ ] 报告含 scripted 基线对照。
- [ ] 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
- [ ] 运行产物只写入已忽略的 `runs/`。

## 7. 执行状态

### 已完成

- **步骤 1–5（设计 + 实现）**：`docs/week21/design.md`、`src/world_model.py`（数据读回/训练/NumPy 前向/CEM/MPC episode）、`src/run_world_model.py`（train/eval CLI）完成。
- **步骤 6（测试）**：`src/tests/test_world_model.py` 19 项定向测试通过；完整回归 482 通过（1 项既有 mps 环境差异）；`compileall`、`pip check` 通过。
- **步骤 7（真实演示）**：4060 上 PushCube 1253 transitions 训练完成（best_step=600、val_loss 0.103）；CEM MPC 0/10（0%）vs scripted 10/10（100%）。结论：小型 dynamics + CEM 不优于 scripted 基线，详见 design.md §7.1。
- **步骤 8（文档同步）**：本文件与 design.md 实际偏差已回填；总计划待同步。

### 备注

- 该支线为可选研究记录，不参与第 22 周门禁 C；训练产物仅存于已忽略的 `runs/maniskill_world_model/`。
