# 第 6 周实施计划 — 批量评测与阶段 A 门禁

## 1. 任务分解

| 编号 | 任务 | 文件 | 依赖 |
| --- | --- | --- | --- |
| 1 | 建立 12 条固定阶段 A 任务集 | `evaluation/stage_a_tasks.json` | 无 |
| 2 | 实现任务加载、失败归类、指标聚合和报告渲染 | `src/evaluation.py` | 1、既有失败分类器 |
| 3 | 为 Planner 与真实 LLM 调用增加精确计数 | `src/executor.py`、`src/llm_planner.py` | 既有 Agent loop |
| 4 | 实现批量运行 CLI、任务隔离与产物落盘 | `src/run_evaluation.py` | 1–3 |
| 5 | 增加纯逻辑与 Executor 回归测试 | `src/tests/test_evaluation.py`、`src/tests/test_executor.py` | 2–4 |
| 6 | 运行全量单元测试和语法检查 | 全仓库 | 5 |
| 7 | 执行真实 AI2-THOR 批量评测并生成阶段报告 | `runs/evaluation/`、`docs/week6/stage_a_gate_report.md` | 6 |
| 8 | 同步总开发计划与设计偏差 | `具身智能完整项目_业余开发计划.md`、`docs/week6/design.md` | 7 |

## 2. 文件结构

**新增文件：**

- `evaluation/stage_a_tasks.json` — 版本化固定任务集。
- `src/evaluation.py` — 评测数据模型、校验、聚合与报告。
- `src/run_evaluation.py` — 批量评测 CLI。
- `src/tests/test_evaluation.py` — 评测纯逻辑回归测试。
- `docs/week6/design.md` — 第 6 周设计文档。
- `docs/week6/plan.md` — 第 6 周实施计划。
- `docs/week6/stage_a_gate_report.md` — 实际批量运行后的阶段 A 门禁报告。
- `requirements-llm.txt` — 仅在运行真实 OpenAI 兼容 API 后端时安装的可选依赖。

**修改文件：**

- `src/executor.py` — 输出 Planner/LLM 调用计数。
- `src/llm_planner.py` — 后端声明是否是真实 LLM。
- `src/tests/test_executor.py` — 计数行为回归。
- `具身智能完整项目_业余开发计划.md` — 第 6 周状态、代码结构和下一步。

## 3. 实施步骤

### 步骤 1：固定任务集

- 选择 12 条可由现有 `TaskSpec` 表达的 pick-and-place 任务。
- 覆盖多场景、多对象、多容器和搜索路径。
- 通过 loader 校验任务数量、唯一 ID、字段和指令解析。

### 步骤 2：评测纯逻辑

- 定义 `EvaluationTask`。
- 实现失败动作与终止原因的标准化分类。
- 聚合成功率、动作数分布、Planner/LLM 调用和失败类型。
- 使用固定 70% 阈值和最少 10 任务判定门禁。
- 渲染包含配置、总览、逐任务结果、失败分析和结论的 Markdown 报告。

### 步骤 3：Executor 指标

- 在唯一的 `backend.chat` 调用点计数。
- 后端通过只读属性声明是否属于真实 LLM。
- 将计数加入所有正常终止摘要；异常 episode 由 runner 补 0 并记录异常。

### 步骤 4：批量 CLI

- 加载 `.env`，但不打印密钥。
- 为每个任务创建独立 Controller 和产物目录。
- 每完成一个任务立即 append 到 `episodes.jsonl`，降低中途异常的数据损失。
- 批次完成后生成 JSON/Markdown 报告和配置快照。
- 门禁失败返回非零，便于 CI 或手动脚本识别。

### 步骤 5：测试

- 测任务加载正常与所有主要非法输入。
- 测失败动作分类与终止原因分类。
- 测 7/10 恰好通过、少于 10 条不通过、批次不完整不通过。
- 测动作数与调用数统计。
- 测 Markdown 包含后端、逐任务、失败和门禁结论。
- 测规则后端调用只增加 Planner 次数，真实 LLM 标记后端同时增加两项。

### 步骤 6：真实批量运行与回溯

- 先跑单任务 smoke，再跑 12 条固定任务。
- 检查 `episodes.jsonl` 行数与任务数一致，逐任务轨迹可读取。
- 生成并审阅阶段报告。
- 将报告结论同步进总计划；如门禁未通过，第 6 周标记部分完成并明确缺口，不提前进入第 7 周。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 单元测试 | `.venv/bin/python -m pytest -q` | loader、指标、报告、Executor 回归 |
| 语法检查 | `.venv/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 单任务 smoke | `run_evaluation.py --task-id a01 --backend rule` | Controller、Agent、产物链路 |
| 批量 E2E | `run_evaluation.py --backend rule` | 12 条真实 AI2-THOR 完整闭环 |
| 可选 LLM 评测 | `run_evaluation.py --backend auto` | 在已配置 API 时验证真实 LLM 表现 |

## 5. 验收标准

- [x] 固定任务集包含 10–20 条、可解析且 ID 唯一的任务（12 条）。
- [x] 每条任务运行独立完整 Agent 闭环并保存完整轨迹（12/12）。
- [x] 报告包含成功率、动作数、Planner 调用数、LLM 调用数和失败类型。
- [x] 批次异常不会阻止其余任务运行，且失败可追溯（含 Controller 异常回归测试）。
- [x] 门禁判定同时检查任务数、批次完整性和 70% 成功率。
- [x] 所有单元测试与语法检查通过（172 passed）。
- [x] 完成真实 AI2-THOR 阶段报告，并明确规则后端与验证边界。
- [x] 同步总开发计划和设计偏差记录。

## 6. 实际执行结果（2026-08-06）

- smoke：a01 任务状态验证通过；完整 RGB、轨迹、L0/L1、episode 和报告链路正常。
- 固定批次：12 条全部完成运行，4 条成功，成功率 33.3%。
- 失败动作：`object_not_found` 46、`placement_failed` 30、`navigation_blocked` 4。
- 失败终止：`search_exhausted` 4、`max_steps` 4。
- 阶段 A 门禁未通过；第 6 周评测工程完成，但被评 Agent 指标未达标，状态为部分完成。

上述为首轮基线。收敛修复后的最终结果：

- 规则基线：10/12，83.3%，111 个动作，51 次 Planner 调用。
- 真实 LLM：10/12，83.3%，100 个动作，39 次 Planner/真实 LLM 调用。
- 两组 `episodes.jsonl` 均为 12 行、轨迹目录均为 12/12，阶段 A 门禁通过。

## 7. 门禁收敛修复计划

- [x] `SearchStrategy` 支持缓存 AI2-THOR 可达位置并确定性选择未探索视点。
- [x] `move_to` 支持仅供搜索策略使用的可达点导航，同时保留四个相对方向行为。
- [x] `AgentExecutor` 只查询一次可达位置，查询和导航均保留完整原子轨迹。
- [x] 搜索抓取目标时打开新视点下可交互的关闭容器，使容器内物体可被后续扫描发现。
- [x] `PLACEMENT_FAILED` 进入强制换位搜索，不再原地重试同一容器。
- [x] 普通放置仅在“无有效落点”错误时执行一次受控 `forceAction`，并继续要求完整状态验证。
- [x] 单元测试覆盖视点选择、回退、可达点导航、强制换位和旧路径兼容。
- [x] 全量测试和语法检查通过（172 passed）。
- [x] 同一 12 条任务、同一 20/8 步骤上限完成真实 AI2-THOR 复评。
- [x] 真实 LLM 对照配置有限请求超时和重试，避免单次 API 挂起阻塞整个批次。
- [x] 根据真实 LLM 失败轨迹补充内置搜索契约 Prompt，避免模型绕开 `pick/place` 恢复入口。
- [x] 根据复评结果更新阶段报告与总计划；规则和 LLM 均达到 83.3%，门禁通过。
