# 第 18 周实施计划 — SmolVLA 适配探针

## 1. 任务分解

| 序号 | 子任务 | 文件/产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化 SmolVLA 输入/输出映射与 go/no-go 判据 | `docs/week18/design.md` | Week 15–17 |
| 2 | 确认腾讯云 GPU 规格与 `lerobot[smolvla]` 安装 | 云环境、`pip check` | 1 |
| 3 | 实现 SmolVLA policy 加载与输入/输出映射 | `src/smolvla_probe.py` | 2 |
| 4 | 实现单 episode 评测与动作趋势记录 | `src/smolvla_probe.py` | 3 |
| 5 | 实现固定 20 次评测 CLI 与 go/no-go 决策 | `src/run_smolvla_probe.py` | 4 |
| 6 | 编写纯逻辑、fake env、真实 smoke 与错误路径测试 | `src/tests/test_smolvla_probe.py` | 3–5 |
| 7 | 腾讯云零样本评测（PickCube + PushCube 各 20 次） | `runs/smolvla/week18/` | 2–6 |
| 8 | 可选：零样本有正向信号时小规模微调 | 云环境、`runs/smolvla/` | 7 |
| 9 | 回填真实结果、实际偏差并同步总开发计划 | `docs/week18/*`、总计划 | 7–8 |

## 2. 文件结构

### 2.1 计划新增

- `src/smolvla_probe.py` — SmolVLA 适配、评测、动作趋势与 go/no-go。
- `src/run_smolvla_probe.py` — 第 18 周探针 CLI。
- `src/tests/test_smolvla_probe.py` — 第 18 周定向测试。
- `docs/week18/design.md` — 本周设计与实际偏差。
- `docs/week18/plan.md` — 本实施计划与执行状态。

### 2.2 计划修改

- `requirements-lerobot.txt` — 记录 `lerobot[smolvla]` 相关依赖（如在 Mac 上安装）。
- `具身智能完整项目_业余开发计划.md` — 更新第 18 周状态、当前代码结构和下一步。

### 2.3 明确不修改

- Week 15–17 数据、checkpoint、评测协议；
- `visual_act.py`、`lerobot_pipeline.py`、`perturbation.py`、`run_generalization.py` 的既有契约；
- ManiSkill 环境、成功条件或动作空间。

## 3. 实施步骤

### 步骤 1：设计固化

1. 确认 SmolVLA 输入尺寸、state/action 维度、任务文本。
2. 确认 go/no-go 判据（存在成功 或 失败模式≠视觉 ACT 的闭合高度错误）。
3. 记录到 `docs/week18/design.md`。

完成条件：映射与判据明确。

### 步骤 2：环境与硬件

1. 腾讯云租用 GPU（L20/A10/4090 等价），确认 CUDA + Python 3.12。
2. 安装 `lerobot[smolvla]`、`mani_skill`、`h5py`、`pillow`；`pip check` 通过。
3. Mac 侧确认 `.venv-lerobot` 是否需补 `[smolvla]`（本地 smoke 用）。

完成条件：云与本地环境均可用。

### 步骤 3：policy 加载与映射

1. `make_smolvla_policy()`：`SmolVLAPolicy.from_pretrained` + pre/post。
2. `observation_to_smolvla_input()`：图像 resize + state + 任务文本。
3. `action` 输出 → ManiSkill 连续 4D。

完成条件：模型可加载，输入/输出 shape 校验通过。

### 步骤 4：单 episode 评测

1. 复用 `run_episode_in_env`（或自写）。
2. 记录每帧动作趋势（接近方向、夹爪、z 变化）。
3. 失败模式分类（复用 `act_stability.classify_failure_diagnosis` 思路）。

完成条件：单 episode 可跑通并输出趋势。

### 步骤 5：CLI 与 go/no-go

1. `run_smolvla_probe.py`：固定 seed（分布内 500–519 或 600–629 各 20 次）。
2. go/no-go 决策输出 `go_no_go.json`。
3. 以退出码表达。

完成条件：CLI 全流程可用。

### 步骤 6：测试

1. 纯逻辑：输入/输出映射、seed 集、go/no-go 判定。
2. fake env/policy：单 episode 评测、失败分类。
3. 完整回归、`compileall`、`pip check`、`git diff --check`。

完成条件：定向测试与完整回归通过。

### 步骤 7：腾讯云零样本评测

1. 云上执行 PickCube + PushCube 各 20 次零样本评测。
2. 记录成功率、动作趋势、失败模式。
3. 若 MPS 可跑，先在 Mac smoke 1–2 seed 验证链路。

完成条件：`probe_report.json`、`go_no_go.json` 齐全。

### 步骤 8：可选微调

1. 仅当 go 时：用 `lerobot-train` + 第 15 周 v3 数据，20k step 微调。
2. 产物回传 Mac，复用第 16/17 周评测协议对比。

完成条件：微调产物回传（可选）。

### 步骤 9：同步文档

1. 回填 `docs/week18/design.md` 的真实映射、结果与偏差。
2. 更新本计划复选框与执行状态。
3. 更新总计划第 18 周状态、已完成/未完成、当前代码结构与下一步。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向测试 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_smolvla_probe.py` | 映射、seed、go/no-go、评测 |
| 真实 smoke | `run_smolvla_probe.py --episodes-per-task 1 ...` | 模型加载、单 episode 闭环 |
| 完整回归 | `.venv-lerobot/bin/python -m pytest -q --ignore=src/tests/test_llm_planner.py` | 第 1–18 周无回归 |
| 语法检查 | `.venv-lerobot/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 依赖检查 | `.venv-lerobot/bin/python -m pip check` | 独立环境依赖一致 |
| 差异检查 | `git diff --check` | 无空白和补丁格式错误 |

## 5. 单元测试覆盖

### 5.1 输入/输出映射

- observation → SmolVLA feature dict（图像/state/任务文本）；
- action 输出 shape/有限性校验；
- 非法输入拒绝。

### 5.2 评测与 go/no-go

- 单 episode 评测结构正确；
- 动作趋势记录（接近/夹爪/z）；
- go/no-go 判定（有成功→go，全失败同视觉 ACT 模式→no-go）。

## 6. 验收清单

- [ ] `lerobot[smolvla]` 安装且模型可加载。
- [ ] 相机/state/action/任务文本映射完成。
- [ ] 固定 20 次零样本评测完成并落盘。
- [ ] 每 episode 记录动作趋势与失败模式。
- [ ] go/no-go 决策及依据输出。
- [ ] 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
- [ ] 运行产物只写入已忽略的 `runs/`。

## 7. 执行状态

### 已完成

- **步骤 1（设计固化）**：确认 SmolVLA base 契约（6D state/3 相机/6D action，与 ManiSkill 维度不同），go/no-go 判据固化到 `docs/week18/design.md`。
- **步骤 2（环境）**：本地 `.venv-lerobot` 补装 `lerobot[smolvla]`（transformers/timm/tokenizers），`pip check` 通过；`requirements-lerobot.txt` 更新为 `lerobot[dataset,smolvla]==0.6.1`。
- **步骤 3–5（代码）**：`smolvla_probe.py`（`make_smolvla_policy`、`observation_to_smolvla_input` 6D state 映射、`run_smolvla_episode` 动作趋势记录、`decide_go_no_go`）+ `run_smolvla_probe.py` CLI 完成。
- **步骤 6（测试）**：`tests/test_smolvla_probe.py` 8 项定向测试通过；完整回归 434 项通过；`compileall`、`pip check`、`git diff --check` 通过。
- **云清单**：`docs/week18/cloud_runbook.md` 含腾讯云 80G/48G 安装、零样本 20 次、go/no-go 判据、微调 20k step 命令。

### 待云上执行（本机无法下载 0.5B 模型）

- **步骤 7**：腾讯云零样本评测（PickCube + PushCube 各 20 次）→ `probe_report.json`。
- **步骤 8**：若 go，用第 15 周 v3 数据微调 20k step → 回传 Mac 评测。
- **步骤 9**：回填结果、实际偏差（design.md 已先记）并同步总计划。
