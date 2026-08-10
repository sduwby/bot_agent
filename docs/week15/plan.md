# 第 15 周实施计划 — LeRobot 数据管线

## 1. 任务分解

| 序号 | 子任务 | 文件/产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化 feature schema、fps、camera 与转换契约 | `docs/week15/design.md` | Week 11/13 |
| 2 | 安装 `lerobot[dataset]` 并验证 v3 读写 | `requirements-lerobot.txt`、`pip check` | 1 |
| 3 | 实现 summary/元数据加载与 18D→5/13 拆分 | `src/lerobot_pipeline.py` | 2 |
| 4 | 实现 HDF5 → LeRobotDataset v3 转换（含可选 camera） | `src/lerobot_pipeline.py` | 3 |
| 5 | 实现 v3 只读重开、episode 读回（replay） | `src/lerobot_pipeline.py` | 4 |
| 6 | 实现字段完整性检查与异常帧检测 | `src/lerobot_pipeline.py` | 5 |
| 7 | 实现逐 episode 浏览与可视化 | `src/lerobot_pipeline.py` | 5–6 |
| 8 | 实现 CLI（convert/inspect/browse/visualize） | `src/run_lerobot_pipeline.py` | 4–7 |
| 9 | 编写纯逻辑、fake env、真实 v3 smoke 与错误路径测试 | `src/tests/test_lerobot_pipeline.py` | 3–8 |
| 10 | 真实转换 200 条轨迹并检查 | `runs/maniskill_lerobot/week15-*/` | 2–9 |
| 11 | 回填真实结果、实际偏差并同步总开发计划 | `docs/week15/*`、总计划 | 10 |

## 2. 文件结构

### 2.1 计划新增

- `src/lerobot_pipeline.py` — LeRobotDataset v3 转换、replay、检查、浏览与可视化。
- `src/run_lerobot_pipeline.py` — 第 15 周 CLI。
- `src/tests/test_lerobot_pipeline.py` — 第 15 周定向测试。
- `docs/week15/design.md` — 本周设计与实际偏差。
- `docs/week15/plan.md` — 本实施计划与执行状态。

### 2.2 计划修改

- `requirements-lerobot.txt` — 记录 `lerobot[dataset]==0.6.1` 或新增的依赖（如 `datasets`、`pyarrow`）。
- `具身智能完整项目_业余开发计划.md` — 更新第 15 周状态、当前代码结构和下一步。

### 2.3 明确不修改

- Week 11 正式 HDF5/JSON 数据；
- Week 12 BC checkpoint、Week 13 ACT checkpoint；
- `maniskill_intro.py` 中 scripted policy 与 `make_maniskill_env()` 契约；
- ManiSkill 环境、成功条件或动作空间。

## 3. 实施步骤

### 步骤 1：确认前置与依赖

1. 确认 `runs/maniskill_expert/week11-200/` 的 summary 与两个 HDF5/JSON 完整。
2. `.venv-lerobot/bin/python -m pip install "lerobot[dataset]==0.6.1"`。
3. `.venv-lerobot/bin/python -m pip check` 通过。
4. 记录 Python、LeRobot、ManiSkill、SAPIEN、Pinocchio、h5py 版本。

完成条件：LeRobotDataset v3 可本地创建/写入/重开（用最小 smoke 验证）。

### 步骤 2：schema 与状态拆分

1. `build_lerobot_features(include_camera)` 返回 v3 feature dict。
2. `maniskill_state_to_lerobot_features()` 完成 18D→(5,13) 拆分与逆向校验。
3. 标量 feature 使用 shape `(1,)`；action 越界阈值为 `[-1,1]`。

完成条件：拆分结果与 `lerobot_act.split_lerobot_act_state` 一致。

### 步骤 3：转换主流程

1. `load_maniskill_summary()` 校验 schema、任务数与 episode 数。
2. 对每个 episode：逐帧重建 observation → 拆分；camera 开启时逐帧渲染 RGB。
3. 组装 action/reward/success/next.done/task 后 `add_frame`，每 episode `save_episode`，最后 `finalize`。
4. 写 `conversion_summary.json`。

完成条件：两任务各 100 episode 转换成功，字段无缺失。

### 步骤 4：只读重开与 episode 读回

1. `open_lerobot_dataset()` 只读重开 v3。
2. `read_episode_frames()` 读回单 episode 完整 state/action/reward/success/done。
3. 校验读回数据与转换时输入逐元素一致。

完成条件：任意 episode 可独立读回。

### 步骤 5：检查与异常帧检测

1. 遍历所有 episode 所有帧，检查字段完整性（missing=0）。
2. 检测 NaN/Inf、shape 错误、越界 action、camera 缺失/全黑。
3. 输出 `inspection.json`。

完成条件：missing 总数必须为 0；异常帧被准确标记。

### 步骤 6：浏览与可视化

1. `browse_lerobot_episode()` 输出逐帧摘要。
2. `visualize_lerobot_episode()` 保存相机帧缩略图网格与帧级 JSON。

完成条件：任意轨迹可独立浏览并发现异常帧。

### 步骤 7：CLI

1. `run_lerobot_pipeline.py` 提供 `convert`、`inspect`、`browse`、`visualize` 子命令。
2. 提前解析绝对路径，避免 ManiSkill 改变 CWD。
3. 以退出码表达转换/检查是否通过。

完成条件：CLI 覆盖全流程。

### 步骤 8：测试

1. 纯逻辑：schema、拆分/逆向、读回、检查、异常检测、浏览。
2. fake env/fake dataset：转换 smoke、字段缺失、错误路径。
3. 真实 v3 smoke：最小 dataset 创建/写入/重开。
4. 完整回归、`compileall`、`pip check`、`git diff --check`。

完成条件：定向测试通过；完整回归无回退。

### 步骤 9：真实转换与检查

1. 转换 200 条轨迹（先做每任务 1 条 smoke，再做全量）。
2. 运行检查，确认 missing=0。
3. 浏览几个代表 episode（含成功/异常）。

完成条件：`conversion_summary.json`、`inspection.json` 齐全。

### 步骤 10：同步文档

1. 回填 `docs/week15/design.md` 的真实版本、结果与偏差。
2. 更新本计划复选框与执行状态。
3. 更新总计划第 15 周状态、已完成/未完成、当前代码结构与下一步。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向测试 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_lerobot_pipeline.py` | schema、拆分、读回、检查、浏览、CLI |
| 真实 smoke | `.venv-lerobot/bin/python src/run_lerobot_pipeline.py convert --episodes-per-task 1 ...` | 真实 HDF5、v3 写入、重开最小闭环 |
| 完整回归 | `.venv-lerobot/bin/python -m pytest -q --ignore=src/tests/test_llm_planner.py` | 第 1–15 周无回归 |
| 语法检查 | `.venv-lerobot/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 依赖检查 | `.venv-lerobot/bin/python -m pip check` | 独立环境依赖一致 |
| 差异检查 | `git diff --check` | 无空白和补丁格式错误 |

## 5. 单元测试覆盖

### 5.1 schema 与拆分

- `build_lerobot_features(False/True)` 精确 feature 集；
- 18D 拆 5/13 与逆向无损；
- 非法输入（错误维度、非有限值）拒绝。

### 5.2 转换与读回

- fake env 重建 observation → 拆分 → 帧组装；
- camera 开启/关闭的帧内容差异；
- 输出目录已存在且非空时拒绝覆盖；
- v3 只读重开后读回与输入一致。

### 5.3 检查与异常帧

- 字段缺失计数（正常=0，人为删字段>0）；
- NaN/Inf、shape 错误、越界 action 检测；
- camera 缺失/全黑检测；
- 检查报告聚合正确。

### 5.4 浏览与可视化

- 逐帧摘要结构正确；
- 缩略图网格生成（camera 开启时）。

## 6. 验收清单

- [ ] `lerobot[dataset]==0.6.1` 安装且 `pip check` 通过。
- [ ] 第 11 周 200 条轨迹可转换为 LeRobotDataset v3。
- [ ] 18D 无损拆 5/13，逆向校验一致。
- [ ] 任意 episode 可只读读回完整 state/action/reward/success/done。
- [ ] 字段完整性 missing=0。
- [ ] 异常帧检测覆盖 NaN/Inf、shape、越界 action、camera 缺失/全黑。
- [ ] 逐 episode 浏览与可视化报告可生成。
- [ ] 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
- [ ] 运行产物只写入已忽略的 `runs/`。

## 7. 执行状态

### 已完成

- **步骤 1–2（前置与依赖）**：`.venv-lerobot` 安装 `lerobot[dataset]==0.6.1`，`pip check` 通过；`requirements-lerobot.txt` 更新为 `lerobot[dataset]==0.6.1`。
- **步骤 3–4（schema 与转换）**：`src/lerobot_pipeline.py` 实现 `build_lerobot_features()`、`maniskill_state_to_lerobot_features()`（18D→5/13 无损）、`convert_maniskill_to_lerobot()`（HDF5→v3，含可选 RGB 渲染）。
- **步骤 5–6（读回与检查）**：`open_lerobot_dataset()`、`read_episode_frames()`、`inspect_lerobot_dataset()`（missing=0 + 异常帧检测）完成。
- **步骤 7–8（浏览/可视化/CLI）**：`browse_lerobot_episode()`、`visualize_lerobot_episode()` 与 `run_lerobot_pipeline.py`（convert/inspect/browse/visualize）完成。
- **步骤 9（测试）**：`src/tests/test_lerobot_pipeline.py` 15 项定向测试全部通过；完整回归 373 项通过；`compileall`、`pip check`、`git diff --check` 通过。
- **步骤 10（真实转换）**：无相机与带相机两种全量转换均 200 episode / 3257 帧、errors=0；`inspection.json` 均 **missing=0、abnormal_frames=0**。产物在 `runs/maniskill_lerobot/week15-full/`、`runs/maniskill_lerobot/week15-cam/`。
- **步骤 11（文档同步）**：`docs/week15/design.md` 已回填实现偏差与结果；总计划已同步。

### 验收清单状态

- [x] `lerobot[dataset]==0.6.1` 安装且 `pip check` 通过。
- [x] 第 11 周 200 条轨迹可转换为 LeRobotDataset v3。
- [x] 18D 无损拆 5/13，逆向校验一致。
- [x] 任意 episode 可只读读回完整 state/action/reward/success/done。
- [x] 字段完整性 missing=0。
- [x] 异常帧检测覆盖 NaN/Inf、shape、越界 action、camera 缺失/全黑。
- [x] 逐 episode 浏览与可视化报告可生成。
- [x] 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
- [x] 运行产物只写入已忽略的 `runs/`。
