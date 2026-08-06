# 第 5 周实施计划 — 验证与恢复模块

## 1. 任务分解

| 编号 | 任务 | 文件 | 依赖 |
|------|------|------|------|
| 1 | `LoopDetection` 数据类 + `ActionMonitor.detect_loop` | `src/action_monitor.py` | 无 |
| 2 | `FailureCategory` 枚举 + `FailureInfo` 数据类 + `FailureClassifier.classify_failure` | `src/failure_classifier.py` | 无 |
| 3 | `VerificationResult` 数据类 + `SubtaskVerifier` 三个验证方法 | `src/verifier.py` | 无 |
| 4 | `RecoveryAction` 数据类 + `Replanner`（恢复策略 + 重试计数） | `src/replanner.py` | 1, 2 |
| 5 | 修改 `AgentExecutor.__init__`：初始化 monitor, classifier, verifier, replanner | `src/executor.py` | 4 |
| 6 | 修改 `AgentExecutor.run`：循环检测、二次验证、失败分类 + 恢复 | `src/executor.py` | 5 |
| 7 | 更新 `_SYSTEM_PROMPT_TEMPLATE`：增加失败恢复提示 | `src/llm_planner.py` | 无 |
| 8 | `ActionMonitor` 单元测试 | `src/tests/test_action_monitor.py` | 1 |
| 9 | `FailureClassifier` 单元测试 | `src/tests/test_failure_classifier.py` | 2 |
| 10 | `SubtaskVerifier` 单元测试 | `src/tests/test_verifier.py` | 3 |
| 11 | `Replanner` 单元测试 | `src/tests/test_replanner.py` | 4 |
| 12 | 更新开发计划文档 | `具身智能完整项目_业余开发计划.md` | 11 |

## 2. 文件结构

**新增文件：**
- `src/action_monitor.py` — 重复动作检测
- `src/failure_classifier.py` — 动作失败分类
- `src/verifier.py` — 子任务验证
- `src/replanner.py` — 重新规划
- `src/tests/test_action_monitor.py` — ActionMonitor 单元测试
- `src/tests/test_failure_classifier.py` — FailureClassifier 单元测试
- `src/tests/test_verifier.py` — SubtaskVerifier 单元测试
- `src/tests/test_replanner.py` — Replanner 单元测试

**修改文件：**
- `src/executor.py` — AgentExecutor 集成验证与恢复
- `src/llm_planner.py` — System Prompt 增加失败恢复提示
- `具身智能完整项目_业余开发计划.md` — 更新进度

## 3. 实施步骤

### 步骤 1：ActionMonitor（约 1h）

创建 `src/action_monitor.py`：
- `LoopDetection` 数据类
- `ActionMonitor.detect_loop(history, window=3)` 静态方法
- 检测 `IDENTICAL_REPEAT`：连续 N 步相同 (skill, params)
- 检测 `OSCILLATION`：两动作交替（A→B→A）

### 步骤 2：FailureClassifier（约 1h）

创建 `src/failure_classifier.py`：
- `FailureCategory` 枚举（5 类）
- `FailureInfo` 数据类
- `FailureClassifier.classify_failure(skill_name, params, failure_reason)` 静态方法
- 规则：pick → OBJECT_NOT_FOUND/GRASP_FAILED，place → PLACEMENT_FAILED，move_to → NAVIGATION_BLOCKED

### 步骤 3：SubtaskVerifier（约 1.5h）

创建 `src/verifier.py`：
- `VerificationResult` 数据类
- `verify_pick(metadata, object_type)` → 检查 inventoryObjects
- `verify_place(metadata, object_id, receptacle_type)` → 检查 parentReceptacles
- `verify_task_complete(instruction, metadata)` → 综合判断

### 步骤 4：Replanner（约 1.5h）

创建 `src/replanner.py`：
- `RecoveryAction` 数据类
- `Replanner` 类，含 retry_counts 跟踪
- `plan_recovery(failure_info, loop_detection, verification_failed)` 方法
- 恢复策略表实现（参考 design.md 3.4 节）
- `record_retry`, `should_retry`, `reset` 辅助方法

### 步骤 5-6：集成到 AgentExecutor（约 2.5h）

修改 `executor.py`：
- 构造函数新增 ActionMonitor, FailureClassifier, SubtaskVerifier, Replanner
- 主循环开头：循环检测 → 如果检测到循环，执行 scan 恢复
- 技能执行后：成功时二次验证 pick/place
- 失败时：分类 → replanner → 根据 RecoveryAction 决定 retry/scan/llm_decision
- 记录恢复操作到 action_history

### 步骤 7：System Prompt 更新（约 0.5h）

在 `llm_planner.py` 的 `_SYSTEM_PROMPT_TEMPLATE` 末尾追加失败恢复指导。

### 步骤 8-11：单元测试（约 3h）

**ActionMonitor 测试（`test_action_monitor.py`）：**
- 无循环：随机不同动作
- IDENTICAL_REPEAT：连续 3 次相同 pick
- OSCILLATION：pick → place → pick → place
- 边界：history 少于 window

**FailureClassifier 测试（`test_failure_classifier.py`）：**
- pick + "未发现" → OBJECT_NOT_FOUND
- pick + 其他错误 → GRASP_FAILED
- place 失败 → PLACEMENT_FAILED
- move_to 失败 → NAVIGATION_BLOCKED
- scan 失败 → UNKNOWN

**SubtaskVerifier 测试（`test_verifier.py`）：**
- verify_pick 成功：inventoryObjects 包含目标
- verify_pick 失败：inventoryObjects 为空
- verify_place 成功：parentReceptacles 包含目标容器
- verify_place 失败：物体不在容器内
- verify_task_complete：组合验证

**Replanner 测试（`test_replanner.py`）：**
- OBJECT_NOT_FOUND → search 动作
- GRASP_FAILED → retry 动作
- PLACEMENT_FAILED → retry 动作
- LOOP_DETECTED → scan 动作
- 超过最大重试 → abort
- 重试计数正确递增

### 步骤 12：更新文档（约 0.5h）

更新 `具身智能完整项目_业余开发计划.md`：
- 第 5 周状态改为完成
- 更新代码结构列表
- 更新下一步

## 4. 测试计划

| 测试类型 | 覆盖范围 | 文件 |
|----------|---------|------|
| 单元测试 | `ActionMonitor.detect_loop` | `test_action_monitor.py` |
| 单元测试 | `FailureClassifier.classify_failure` | `test_failure_classifier.py` |
| 单元测试 | `SubtaskVerifier` 所有方法 | `test_verifier.py` |
| 单元测试 | `Replanner.plan_recovery` | `test_replanner.py` |
| 端到端 | 重复动作触发恢复 | 手动 `run_e2e.py` |
| 端到端 | 失败分类正确 | 手动 `run_e2e.py` |

## 5. 验收标准

- [x] `ActionMonitor` 能检测连续重复和振荡两种循环模式（单元测试通过率 100%）
- [x] `FailureClassifier` 能正确分类 3+ 类失败（OBJECT_NOT_FOUND, GRASP_FAILED, PLACEMENT_FAILED）
- [x] `SubtaskVerifier` 能通过 metadata 独立验证 pick/place 和最终任务状态
- [x] `Replanner` 能根据失败类型生成正确的恢复动作
- [x] 重试次数有限制，不会无限循环
- [x] 集成后正常流程不受影响（真实 E2E 默认套件 3/3）
- [x] 所有测试通过（`.venv/bin/python -m pytest -q`，144 passed）

## 6. 审计修复计划（2026-08-06）

- [x] 复用 `task_parser.TaskSpec`，让 `done` 通过精确目标状态验证后才能成功。
- [x] 在 place 前保存被放置物体 ID，place 后执行独立验证并让失败进入恢复流程。
- [x] 抽取统一动作执行/观察刷新/捕获/记录方法，普通动作、搜索和自动重试全部复用。
- [x] 自动重试后使用最新 Observation，并对 retry 结果再次分类或验证。
- [x] 修复重复 scan 恢复仍执行 scan 的循环问题。
- [x] 新增 `test_executor.py`，覆盖假成功、放置验证、重试状态刷新、L0 收尾和循环恢复。
- [x] 让 `run_e2e.py` 校验 TaskSpec 对应的最终状态，失败时返回非零并保存汇总。
