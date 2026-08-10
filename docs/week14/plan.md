# 第 14 周实施计划 — 稳定性与阶段 B 评测

## 1. 任务分解

| 序号 | 子任务 | 文件/产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化第 14 周评测 seed 集与门禁口径 | `docs/week14/design.md` | Week 13 |
| 2 | 校验 `.venv-lerobot` 与既有 checkpoint 契约 | `pip check`、`docs/week14/design.md` | 1 |
| 3 | 实现 checkpoint 按 env 分别加载与初始位姿采样 | `src/act_stability.py` | 2 |
| 4 | 实现三策略稳定性评测与门禁判定 | `src/act_stability.py` | 3 |
| 5 | 实现 `action_smoothing` 扫描与失败相位分析 | `src/act_stability.py` | 3–4 |
| 6 | 实现稳定性评测 CLI | `src/run_stability_eval.py` | 4–5 |
| 7 | 编写纯逻辑、fake env、真实 smoke 和错误路径测试 | `src/tests/test_act_stability.py` | 3–6 |
| 8 | 真实运行稳定性评测与门禁 | `runs/maniskill_act/week14/` | 2–7 |
| 9 | 编写硬件采购意向与预算文档 | `docs/week14/procurement.md` | 门禁结果 |
| 10 | 回填真实结果、实际偏差并同步总开发计划 | `docs/week14/*`、总计划 | 8–9 |

## 2. 文件结构

### 2.1 计划新增

- `src/act_stability.py` — 第 14 周稳定性评测：checkpoint 加载、初始位姿采样、三策略评测、平滑扫描、失败分析与门禁。
- `src/run_stability_eval.py` — 第 14 周评测 CLI。
- `src/tests/test_act_stability.py` — 第 14 周定向测试。
- `docs/week14/design.md` — 本周设计与实际偏差。
- `docs/week14/plan.md` — 本实施计划与执行状态。
- `docs/week14/procurement.md` — 硬件采购意向与预算（门禁达标后编写）。

### 2.2 计划修改

- `具身智能完整项目_业余开发计划.md` — 阶段 B 门禁验收后更新第 14 周状态、当前代码结构和下一步。
- 复用模块按需做最小公开化：`lerobot_act.py`、`maniskill_intro.py`、`run_act_diagnose.py` 原则上不改，仅在确实需要时公开稳定接口。

### 2.3 明确不修改

- Week 11 正式 HDF5/JSON 数据；
- Week 12 BC checkpoint；
- Week 13 ACT checkpoint（PickCube dagger-train-r3、PushCube week13）；
- `maniskill_intro.py` 中 scripted policy 的状态机、阈值和控制模式；
- ManiSkill 环境、成功条件或动作空间。

## 3. 实施步骤

### 步骤 1：确认前置产物

1. `.venv-lerobot/bin/python -m pip check` 通过。
2. 确认 `runs/maniskill_act/week13-dagger-train-r3/PickCube-v1/best-rollout-dev/` 与 `runs/maniskill_act/week13/PushCube-v1/best/` 均含完整 checkpoint。
3. 确认 `runs/maniskill_bc/week12/checkpoint.pt` 存在。

完成条件：三个 checkpoint 均可被现有 `load_lerobot_act_checkpoints()` / BC loader 加载。

### 步骤 2：评测 seed 集

1. 选择第 14 周独立 seed 集（建议 500–549 每任务 50 个），验证与 100–149、150–169、200–249、300–359 均不重叠。
2. 在 `StabilityEvalConfig` 中固化，并写 `docs/week14/design.md`。

完成条件：seed 集唯一、非负、不重叠。

### 步骤 3：checkpoint 加载与初始位姿采样

1. `load_stability_checkpoints()` 分别从 PickCube/PushCube 目录加载 ACT policy，BC 独立加载。
2. `record_initial_poses()` 对每个 seed reset 后记录 `obj_pose[:3]` / `goal_pos[:3]`。
3. `summarize_pose_coverage()` 输出每任务位姿范围与均值。
4. 校验 observation 字段缺失时记为异常，不允许静默跳过。

完成条件：采样输出每 seed 的初始位姿，位姿分布与任务随机化范围一致。

### 步骤 4：三策略稳定性评测与门禁

1. 复用 `lerobot_act.evaluate_policy_seeds()`，对 ACT/BC/scripted 在相同 seed 集上执行真实 rollout。
2. ACT 禁止 scripted fallback；BC/scripted 同条件。
3. `compute_stability_gate()` 判定：ACT 总体 ≥max(0.80, 同轮 BC)、exceptions=0、nonfinite=0。
4. 输出 `stability_evaluation.json`、`comparison_stability.json`。

完成条件：三策略逐 seed 可对齐，门禁判定可复现。

### 步骤 5：平滑扫描与失败分析

1. 固定 checkpoint 与 seed 集，扫描 `action_smoothing` ∈ `{0.5, 0.75, 0.9, 1.0}`。
2. 记录各平滑值成功率、步数、非有限动作与异常，写 `smoothing_sweep.json`。
3. 对 ACT 失败 episode 复用 `run_act_diagnose.diagnose_episode()` 做全量相位分类。
4. 输出 `FailurePhaseSummary` 并写 `failure_analysis.json`。

完成条件：平滑扫描无重训，失败分类覆盖全部失败 episode。

### 步骤 6：CLI

1. `run_stability_eval.py` 提供 `--act-pick-checkpoint`、`--act-push-checkpoint`、`--bc-checkpoint`、`--output-dir`、`--seed-start`、`--episodes-per-task`、`--device`、`--smoothing`、`--smoothing-sweep`、`--skip-failure-analysis` 等参数。
2. 提前解析绝对路径，避免 ManiSkill 改变 CWD 后相对路径失效。
3. 以退出码表达门禁是否通过。

完成条件：CLI 覆盖采样、评测、扫描、分析与门禁全链路。

### 步骤 7：测试

1. 纯逻辑：seed 集校验、门禁阈值、位姿覆盖统计、失败相位分类。
2. fake env/policy：三策略评测汇总、平滑扫描、异常隔离。
3. 真实 smoke：每任务 1–2 个 seed 的 ACT/BC/scripted rollout。
4. 错误路径：checkpoint 缺失、seed 重叠、字段缺失、非有限动作。

完成条件：定向测试通过；基础 `.venv` 全量回归无回退；`.venv-lerobot` 全量回归通过。

### 步骤 8：真实运行

1. 在 `.venv-lerobot` 中运行 `run_stability_eval.py` 完成第 14 周 seed 集评测。
2. 记录成功率、步数、推理时延、异常与失败分类。
3. 按门禁判定阶段 B 是否通过。

完成条件：输出 JSON 齐全，门禁判定明确。

### 步骤 9：采购意向与预算

1. 门禁达标后编写 `docs/week14/procurement.md`，只形成采购意向和预算，不下单。
2. 参考总计划 8.1 硬件路线、预算方法与 8.2 采购前检查表。

完成条件：文档含设备清单、预算区间、采购条件与风险。

### 步骤 10：同步文档

1. 回填 `docs/week14/design.md` 的真实 seed 集、环境行为、门禁结果与偏差。
2. 更新本计划复选框与执行状态。
3. 更新总计划第 14 周状态、已完成/未完成、当前代码结构与下一步。
4. 若门禁未达标，不得把阶段 C（第 15 周起）写成可直接开始。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向测试 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_act_stability.py` | seed 集、位姿、门禁、失败分类、CLI |
| 真实 smoke | `.venv-lerobot/bin/python src/run_stability_eval.py --episodes-per-task 1 ...` | 真实 checkpoint、环境与 rollout 最小闭环 |
| 完整回归 | `.venv/bin/python -m pytest -q` | 第 1–14 周无回归 |
| LeRobot 回归 | `.venv-lerobot/bin/python -m pytest -q --ignore=src/tests/test_llm_planner.py` | ManiSkill、HDF5、Torch、LeRobot 路径 |
| 语法检查 | `.venv/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 依赖检查 | `.venv-lerobot/bin/python -m pip check` | 独立环境依赖一致 |
| 差异检查 | `git diff --check` | 无空白和补丁格式错误 |

## 5. 单元测试覆盖

### 5.1 配置与 seed 集

- 与历史 eval/dev/final/DAgger seed 集不重叠；
- seed 非负、唯一、不为空；
- 门禁阈值 `[0, 1]` 校验。

### 5.2 初始位姿采样与覆盖

- 对 fake reset observation 正确读取 `obj_pose[:3]` / `goal_pos[:3]`；
- 字段缺失时记为异常；
- 覆盖统计的 min/max/mean 计算正确。

### 5.3 三策略评测与门禁

- fake env/policy 的 rollout 汇总；
- ACT 总体成功率、BC 成功率、scripted 成功率计算；
- 门禁 = max(threshold, 同轮 BC)，exceptions=0、nonfinite=0 判定；
- 禁 scripted fallback。

### 5.4 平滑扫描与失败分析

- 各平滑值行结构正确；
- 失败相位分类的 `ever_grasped` / `ever_placed` / `simultaneously_valid_steps` 聚合；
- 空失败集不崩溃。

## 6. 验收清单

- [ ] 第 14 周 seed 集与 train/dev/final/DAgger 不重叠。
- [ ] 初始位姿采样输出每任务覆盖统计。
- [ ] ACT/BC/scripted 同 seed 稳定性评测落盘。
- [ ] `action_smoothing` 扫描完成。
- [ ] ACT 失败 episode 全量相位分类。
- [ ] 阶段 B 门禁判定：ACT ≥max(0.80, 同轮 BC)、exceptions=0、nonfinite=0、禁 scripted fallback。
- [ ] 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
- [ ] 运行产物只写入已忽略的 `runs/`。
- [ ] 门禁达标后形成 `docs/week14/procurement.md`。
- [ ] 总计划、实际偏差和执行状态同步完成。

## 7. 执行状态

### 已完成

- **步骤 1–2（前置产物与 seed 集）**：`.venv-lerobot` 补装 `ai2thor` 后 `pip check` 通过；三个 checkpoint 均可加载；seed 集固定 500–549（与历史 eval/dev/final/DAgger 均不重叠）。
- **步骤 3–5（模块）**：`src/act_stability.py` 实现 `load_stability_checkpoints()`（按 env 分别加载）、`record_initial_poses()`、`summarize_pose_coverage()`、`evaluate_stability()`、`compute_stability_gate()`（max(threshold, 同轮 BC) + exceptions/nonfinite 判零）、`run_smoothing_sweep()`、`analyze_failures()`（复用 `run_act_diagnose.diagnose_episode`）与 `classify_failure_diagnosis()`。
- **步骤 6–7（CLI 与测试）**：`src/run_stability_eval.py` 提供完整 CLI；`src/tests/test_act_stability.py` 11 项定向测试全部通过；完整回归 358 项通过；`compileall`、`pip check`、`git diff --check` 通过。
- **步骤 8（真实运行）**：seed 500–549 每任务 50 个，ACT 99/100（99%）> BC 96% > scripted 100%；exceptions=0、nonfinite=0；阶段 B 门禁通过。平滑扫描 1.0 最优。失败分析：PickCube 1 个 `grasped_not_placed`、PushCube 0。产物在 `runs/maniskill_act/week14/`、`runs/maniskill_act/week14-sweep/`。
- **步骤 9（采购意向与预算）**：`docs/week14/procurement.md` 已编写，只形成意向与预算，不下单。
- **步骤 10（文档同步）**：`docs/week14/design.md` 已回填实际偏差与结果；总计划已同步。

### 验收清单状态

- [x] 第 14 周 seed 集与 train/dev/final/DAgger 不重叠。
- [x] 初始位姿采样输出每任务覆盖统计。
- [x] ACT/BC/scripted 同 seed 稳定性评测落盘。
- [x] `action_smoothing` 扫描完成。
- [x] ACT 失败 episode 全量相位分类。
- [x] 阶段 B 门禁判定：ACT ≥max(0.80, 同轮 BC)、exceptions=0、nonfinite=0、禁 scripted fallback。
- [x] 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
- [x] 运行产物只写入已忽略的 `runs/`。
- [x] 门禁达标后形成 `docs/week14/procurement.md`。
- [x] 总计划、实际偏差和执行状态同步完成。
