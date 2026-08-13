# 第 22 周实施计划 — 阶段评测与选型（门禁 C）

## 1. 任务分解

| 序号 | 子任务 | 文件/产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化选型设计（达标口径、冻结方案、模型卡结构） | `docs/week22/design.md` | 第 17/20/21 周 |
| 2 | 实现候选描述与评测执行 | `src/stage_selection.py` | 1 |
| 3 | 实现门禁 C 判定与模型卡/报告落盘 | `src/stage_selection.py` | 2 |
| 4 | 实现选型 CLI | `src/run_stage_selection.py` | 3 |
| 5 | 编写定向测试 | `src/tests/test_stage_selection.py` | 2–4 |
| 6 | 4060 固定 seed 评测（scripted/视觉 ACT/路由） | `runs/maniskill_selection/week22/` | 4 |
| 7 | 回填实际偏差、同步总计划与门禁评审 | `docs/week22/*`、总计划 | 6 |

## 2. 文件结构

### 2.1 计划新增

- `src/stage_selection.py` — 第 22 周：候选评测、门禁 C、模型卡与选型报告。
- `src/run_stage_selection.py` — 第 22 周选型 CLI。
- `src/tests/test_stage_selection.py` — 第 22 周定向测试。
- `docs/week22/design.md` / `docs/week22/plan.md`。
- `docs/week22/stage_c_gate_report.md` — 门禁评审报告（评测完成后生成）。

### 2.2 计划修改

- `具身智能完整项目_业余开发计划.md` — 第 22 周状态、门禁结论、代码结构与下一步（阶段 D 采购）。

### 2.3 明确不修改

- 第 14/17/20 周评测协议、策略实现与 checkpoint；
- `skill_router.py` 路由语义（只复用）。

## 3. 实施步骤

### 步骤 1：设计固化

1. 确认门禁 C 达标口径：视觉 ACT 用 PushCube（100%）+ 抬升子技能组合，PickCube 作附注。
2. 确认冻结方案：主 = 状态 ACT + scripted 兜底；视觉 ACT 为真机升级候选。
3. 记录到 `docs/week22/design.md`。

### 步骤 2–3：核心模块

1. `PolicyCandidate`/`StageSelectionConfig` 校验。
2. `evaluate_candidates`：scripted/bc/state_act/visual_act 走 `evaluate_policy_seeds`；route 走 `skill_router.evaluate_routes`；均按分布内/外两组 seed 执行。
3. `compute_gate_c`：视觉 ACT PushCube 分布内 ≥60%、scripted fallback 可用、nonfinite=0 且 exceptions=0。
4. `build_model_card`：候选结果 + 历史结果（BC 92%、状态 ACT 96%、scripted 100%）聚合，写 `model_card.json` + `selection_report.md`。

### 步骤 4：CLI

1. 参数：`--candidates`（重复给定 `name:kind:checkpoint` 或 `route:...`）、`--seeds-in`、`--seeds-ood`、`--device`、`--output-dir`。
2. 输出模型卡与门禁判定，门禁失败退出码非 0（可 `--no-gate-exit` 覆盖）。

### 步骤 5：测试

1. 纯逻辑：候选解析、门禁判定边界（通过/不通过/安全指标）、模型卡聚合与历史结果合并。
2. fake env：评测执行路径（含 route 候选）、落盘结构。
3. 完整回归、`compileall`、`pip check`、`git diff --check`。

### 步骤 6：4060 评测

1. 候选：scripted、visual_act（week17-c8-20k）、route `visual_act,scripted`；分布内 500–529 / 分布外 600–629。
2. BC/状态 ACT checkpoint 若在 4060 不可用，用历史结果入模型卡并在报告中注明。

### 步骤 7：文档与门禁

1. 生成 `docs/week22/stage_c_gate_report.md` 门禁评审。
2. 回填 design.md 实际偏差、plan.md 执行状态。
3. 更新总计划：第 22 周完成/门禁 C 通过、下一步 = 阶段 D 采购启动。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向测试 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_stage_selection.py` | 候选、门禁、模型卡、CLI |
| 完整回归 | `.venv-lerobot/bin/python -m pytest -q --ignore=src/tests/test_llm_planner.py` | 第 1–22 周无回归 |
| 语法检查 | `.venv-lerobot/bin/python -m compileall -q src` | 全部可编译 |
| 依赖检查 | `.venv-lerobot/bin/python -m pip check` | 依赖一致 |
| 差异检查 | `git diff --check` | 无空白错误 |

## 5. 单元测试覆盖

- `PolicyCandidate` 解析与非法输入拒绝；
- `compute_gate_c`：达标/不达标、fallback 缺失、安全指标脏的边界；
- `build_model_card`：历史结果合并、JSON/MD 结构、原子落盘；
- route 候选通过 fake env 的评测路径；
- CLI 参数校验与退出码。

## 6. 验收清单

- [ ] 固定 seed 评测完成（至少三组候选）。
- [ ] 门禁 C 判定与依据落盘。
- [ ] 模型卡 JSON + 选型报告 MD 生成。
- [ ] 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
- [ ] 运行产物只写入已忽略的 `runs/`。

## 7. 执行状态

### 已完成

- **步骤 1（设计固化）**：`docs/week22/design.md`（达标口径、冻结方案、模型卡结构）。
- **步骤 2–4（核心模块 + CLI）**：`src/stage_selection.py`（`PolicyCandidate`/`StageSelectionConfig`/`evaluate_candidates`/`_evaluate_visual_split`/`compute_gate_c`/`build_model_card`）+ `src/run_stage_selection.py`。
- **步骤 5（测试）**：`src/tests/test_stage_selection.py` 18 项定向测试通过；完整回归 500 通过（1 项既有 mps 环境差异）；`compileall`、`pip check` 通过。
- **步骤 6（4060 评测）**：scripted 100%/100%、visual_act 60.0%（PushCube 100%）/51.7%、route 78.3%；模型卡与选型报告落盘 `runs/maniskill_selection/week22/`。
- **步骤 7（门禁评审）**：`docs/week22/stage_c_gate_report.md` 生成，**门禁 C 通过**；总计划待同步。

### 备注

- BC/状态 ACT checkpoint 在 4060 不可用，以第 12–14 周历史结果（同协议）入模型卡并在报告中注明。
- 门禁通过后按决策门 2 可启动 SO-101 采购（第 23 周）。
