# 第 4 周实施计划 — 记忆与搜索模块

## 1. 任务分解

| 编号 | 任务 | 文件 | 依赖 |
|------|------|------|------|
| 1 | `ConversationRecord` / `MemoryRecord` 数据类 | `src/memory_store.py` | 无 |
| 2 | `MemoryStore` 核心类：append_conversation, append_memory, search_memories, query_memories, get_summary | `src/memory_store.py` | 1 |
| 3 | `MemoryCapture` 类：observation_to_conversation, extract_memories_from_observation, extract_failure_memory | `src/memory_capture.py` | 1 |
| 4 | `MemoryRecall` 类：build_recall_query, recall, inject_into_prompt | `src/memory_recall.py` | 2 |
| 5 | `SearchStrategy` 类：should_search_for, next_move, is_exhausted, is_position_explored, reset | `src/memory_store.py` | 2 |
| 6 | 修改 `AgentExecutor.__init__`：初始化 store, capture, recall, search | `src/executor.py` | 5 |
| 7 | 修改 `AgentExecutor.run`：每步 capture + recall、搜索模式逻辑、记忆摘要注入 prompt | `src/executor.py` | 6 |
| 8 | 修改 `build_user_prompt`：接收并注入 memory_context 参数 | `src/llm_planner.py` | 7 |
| 9 | `MemoryStore` 单元测试（纯逻辑，不依赖 AI2-THOR） | `src/tests/test_memory_store.py` | 2 |
| 10 | `MemoryCapture` 单元测试 | `src/tests/test_memory_capture.py` | 3 |
| 11 | `MemoryRecall` 单元测试 | `src/tests/test_memory_recall.py` | 4 |
| 12 | `SearchStrategy` 单元测试 | `src/tests/test_memory_store.py` | 5 |
| 13 | 端到端测试：运行"找不到的物体"场景，验证搜索行为 | 手动测试 | 8 |
| 14 | 更新开发计划文档 | `具身智能完整项目_业余开发计划.md` | 13 |

## 2. 文件结构

**新增文件：**
- `src/memory_store.py` — MemoryStore（L0/L1 本地轻量存储） + SearchStrategy
- `src/memory_capture.py` — MemoryCapture（具身捕获适配器）
- `src/memory_recall.py` — MemoryRecall（记忆召回）
- `src/tests/test_memory_store.py` — MemoryStore + SearchStrategy 单元测试
- `src/tests/test_memory_capture.py` — MemoryCapture 单元测试
- `src/tests/test_memory_recall.py` — MemoryRecall 单元测试

**修改文件：**
- `src/executor.py` — AgentExecutor 集成记忆和搜索
- `src/llm_planner.py` — build_user_prompt 支持 memory_context
- `具身智能完整项目_业余开发计划.md` — 更新进度

## 3. 实施步骤

### 步骤 1：数据结构（约 1h）

创建 `src/memory_store.py`，实现 `ConversationRecord` 和 `MemoryRecord` 数据类，参考 TencentDB-Agent-Memory 的 L0/L1 模型。

### 步骤 2：MemoryStore 核心（约 2.5h）

实现以下方法：
- `append_conversation(record)` — 追加 L0 对话记录到 JSONL 文件
- `append_memory(record)` — 追加 L1 记忆记录到 JSONL 文件
- `search_memories(query, type_filter, limit)` — 关键词匹配 + 类型过滤，按优先级和新鲜度排序
- `query_memories(type_filter)` — 按类型查询所有记忆
- `get_summary()` — 格式化输出："已记录 N 条记忆（M 个物体，K 个容器）"

### 步骤 3：MemoryCapture 捕获适配器（约 2h）

创建 `src/memory_capture.py`，实现以下方法：
- `observation_to_conversation(instruction, observations, action_history)` — 将具身交互转换为 L0 对话格式
- `extract_memories_from_observation(observation, step, scene_name)` — 从当前观察中提取 L1 记忆（物体位置、容器状态）
- `extract_failure_memory(skill_name, params, failure_reason, step, scene_name)` — 从失败中提取教训

### 步骤 4：MemoryRecall 记忆召回（约 1.5h）

创建 `src/memory_recall.py`，实现以下方法：
- `build_recall_query(instruction, observation, action_history)` — 根据最近失败提取关键词
- `recall(query, type_filter, limit)` — 调用 MemoryStore 搜索
- `inject_into_prompt(memories)` — 格式化为 LLM prompt 片段

### 步骤 5：SearchStrategy（约 1.5h）

在 `src/memory_store.py` 中实现 SearchStrategy：
- `should_search_for(object_type)` — 搜索记忆中是否有该类型的物体
- `next_move()` — 策略：forward → right → forward → left（贪心探索），避免重复访问已探索位置
- `is_exhausted()` — max_steps 用尽或所有方向已探索
- `is_position_explored(pos, tolerance)` — 位置距离容差判断（曼哈顿距离 < 0.5）
- `reset()` — 清零计数器

### 步骤 6-7：集成到 AgentExecutor（约 2.5h）

修改 `executor.py`：
- 构造函数新增 MemoryStore、MemoryCapture、MemoryRecall、SearchStrategy
- 每步循环开始时调用 recall，将记忆摘要注入 prompt
- 每步执行后调用 capture，将新观察写入 L0/L1
- pick/place 失败时检查 search.should_search_for()，进入搜索模式
- 搜索模式下使用 search.next_move() 替代 LLM 决策
- 任务结束后调用 observation_to_conversation 写入完整 L0 记录
- `--max-search-steps` CLI 参数

### 步骤 8：LLM Prompt 更新（约 0.5h）

`build_user_prompt` 新增 `memory_context` 参数，格式：

```markdown
## 场景记忆

以下是在当前场景中已探索过的信息，供参考：

- CreditCard 在 FloorPlan1 中可见，距离 2.3m，Agent 位置 (1.00, 3.00)，朝向 90°
- Drawer 容器在 FloorPlan1 中可见，距离 1.5m，已打开
```

### 步骤 9-12：单元测试（约 3h）

纯逻辑测试（不依赖 AI2-THOR）：

**MemoryStore 测试（`test_memory_store.py`）：**
- `append_conversation` — 写入 JSONL 并回读验证
- `append_memory` — 写入并查询验证
- `search_memories` — 关键词匹配、类型过滤、优先级排序
- `query_memories` — 按类型查询

**MemoryCapture 测试（`test_memory_capture.py`）：**
- `observation_to_conversation` — 格式正确、step 编号递增
- `extract_memories_from_observation` — 提取物体位置、容器状态
- `extract_failure_memory` — 失败教训格式正确

**MemoryRecall 测试（`test_memory_recall.py`）：**
- `build_recall_query` — 从失败中提取关键词
- `recall` — 正确调用 store.search
- `inject_into_prompt` — 格式化输出正确

**SearchStrategy 测试（`test_memory_store.py`）：**
- `next_move` — 按策略顺序返回方向
- `is_exhausted` — 步数耗尽判断
- `is_position_explored` — 容差判断

### 步骤 13：端到端测试（约 1h）

手动运行测试场景：
- 场景 A：物体在当前视野 → 直接抓取（验证记忆不干扰正常流程）
- 场景 B：物体在视野外 → 触发搜索 → 移动找到后抓取

### 步骤 14：更新文档（约 0.5h）

更新 `具身智能完整项目_业余开发计划.md` 中第 4 周进度。

## 4. 测试计划

| 测试类型 | 覆盖范围 | 文件 |
|----------|---------|------|
| 单元测试 | `MemoryStore` 所有公开方法 | `test_memory_store.py` |
| 单元测试 | `MemoryCapture` 所有公开方法 | `test_memory_capture.py` |
| 单元测试 | `MemoryRecall` 所有公开方法 | `test_memory_recall.py` |
| 单元测试 | `SearchStrategy` 所有公开方法 | `test_memory_store.py` |
| 端到端 | "物体在视野内"场景 | 手动 `run_agent.py` |
| 端到端 | "物体在视野外"场景 | 手动 `run_agent.py` |

## 5. 验收标准

- [x] `MemoryStore` 能正确写入 L0/L1 记录并支持关键词搜索（单元测试通过率 100%）
- [x] `MemoryCapture` 能从观察中提取结构化记忆（单元测试通过率 100%）
- [x] `MemoryRecall` 能根据失败提取关键词并召回相关记忆（单元测试通过率 100%）
- [x] `SearchStrategy` 能按策略顺序生成移动指令（单元测试通过率 100%）
- [x] Agent 在目标物体不在当前视野时，能触发搜索并移动找到（FloorPlan6：Cup → CounterTop）
- [x] 记忆中已记录的物体会在 LLM prompt 中正确呈现
- [x] 搜索模式不会覆盖正常 pick 流程（FloorPlan1：CreditCard → Drawer）
- [x] `max_search_steps` 参数生效，防止无限搜索
- [x] 任务结束后 L0 完整记录写入 JSONL 文件

## 6. 审计修复计划（2026-08-06）

- [x] 修复所有提前返回路径缺失 L0 的问题，并增加成功、失败、搜索耗尽回归测试。
- [x] 将搜索触发条件改为基于当前观察和任务目标，不再因历史中曾见过目标而禁止搜索。
- [x] 让 `next_move` 使用当前位置和已探索位置选择方向，并验证搜索步数上限与 reset 行为。
- [x] 让搜索动作进入统一的观察刷新、记忆捕获和轨迹记录流程。
- [x] 在 `run_agent.py` 暴露 `--max-search-steps`。
- [x] 使用真实 AI2-THOR 运行“当前可见”和“需要移动搜索”两个场景，检查目标状态、L0/L1 和轨迹。
