# bot_agent 集成 TencentDB-Agent-Memory 方案

> 分析日期：2026-08-04
> 前置文档：[记忆系统分析](./memory_system_analysis.md)

---

## 1. 集成目标

将 TencentDB-Agent-Memory 的记忆能力引入 bot_agent，使其具备：

1. **已见物体记忆**：记录探索过程中见过的物体及其位置，即使物体离开视野也能检索
2. **房间状态维护**：记录已探索区域和容器状态
3. **记忆辅助决策**：在 LLM 规划时注入相关记忆，减少盲目探索
4. **跨任务经验积累**：不同任务之间可复用已获得的场景知识

---

## 2. bot_agent 需要做的修改

### 2.1 新增模块

| 文件 | 职责 | 说明 |
|------|------|------|
| `src/memory_store.py` | 记忆客户端封装 | 对接 MemoryCore Gateway 或本地轻量存储 |
| `src/memory_capture.py` | 具身交互捕获 | 将环境观察+动作结果转换为可记忆文本 |
| `src/memory_recall.py` | 具身记忆召回 | 根据当前任务+观察召回相关记忆 |
| `src/memory_tools.py` | LLM 可调用搜索工具 | 扩展技能注册表，增加 `memory_search` 技能 |
| `tests/test_memory_store.py` | 记忆模块单元测试 | Mock Gateway 测试读写/召回 |

### 2.2 修改现有模块

#### `executor.py` — 主循环增强

```python
# 修改前
class AgentExecutor:
    def run(self, instruction: str):
        obs = build_observation(metadata)
        obs_summary = summarize(obs)
        # ... LLM plan -> execute

# 修改后
class AgentExecutor:
    def __init__(self, ..., memory_client=None):
        self.memory_client = memory_client  # 可选，None 时跳过记忆

    def run(self, instruction: str):
        # 1. 记忆召回（修改前没有这一步）
        recalled = self.recall_memories(instruction) if self.memory_client else ""

        # 2. 构建增强版观察摘要
        obs = build_observation(metadata)
        obs_summary = summarize(obs)
        # 注入回忆内容
        if recalled:
            obs_summary = f"{recalled}\n\n{obs_summary}"

        # 3. LLM plan -> execute（无变化）

        # 4. 每步执行后捕获（修改前没有这一步）
        if self.memory_client:
            self.capture_step(obs, action, result)

        # 5. 任务结束后批量捕获
        if self.memory_client:
            self.capture_task_complete(instruction, history)
```

#### `observation.py` — 增加记忆增强摘要

```python
def summarize_with_memory(
    observation: Observation,
    recalled_memories: str,
) -> str:
    """将当前观察与回忆内容合并为增强摘要。"""
    parts = [summarize(observation)]
    if recalled_memories:
        parts.insert(0, recalled_memories)  # 回忆放在观察之前
    return "\n\n".join(parts)
```

#### `llm_planner.py` — System Prompt 增加记忆工具说明

```python
# 在 _SYSTEM_PROMPT_TEMPLATE 末尾追加
"""
## 记忆工具

当你需要查找之前见过的物体或位置信息时，可以调用 memory_search 技能。
注意：memory_search 每轮最多调用 2 次。
"""
```

#### `skill_registry.py` — 新增 memory_search 技能

```python
{
    "name": "memory_search",
    "description": "搜索已存储的记忆，查找已见过的物体、位置或经验",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {"type": "STRING", "description": "搜索关键词"}
        },
        "required": ["query"]
    }
}
```

### 2.3 依赖变更

```bash
pip install tencentdb-agent-memory  # 或 pip install -e /path/to/MemoryCore/sdk/memory-core/python
```

---

## 3. TencentDB-Agent-Memory 需要的适配

### 3.1 记忆类型扩展

现有类型：`preference`, `fact`, `rule`, `event`

需要扩展：
```python
# 在 l1-extractor 的 MemoryType 枚举中增加
EMBODYED_MEMORY_TYPES = [
    "object_location",   # 物体位置信息
    "room_state",        # 房间状态
    "action_experience", # 动作经验
    "failure_lesson",    # 失败教训
]
```

### 3.2 具身交互捕获适配器

TencentDB-Agent-Memory 的 `auto-capture` 钩子假设输入是用户/助手对话。需要编写**具身捕获适配器**：

```python
# src/memory_capture.py
def embodiment_to_conversation_messages(
    observations: list[Observation],
    action_history: list[dict],
    instruction: str
) -> list[dict]:
    """将具身交互转换为 L0 对话格式。

    转换规则：
    - 任务指令 -> user message
    - 观察+动作+结果 -> assistant message（描述本轮做了什么、看到了什么）
    """
    messages = [{"role": "user", "content": instruction}]
    for i, action in enumerate(action_history):
        obs_text = summarize(observations[i]) if i < len(observations) else ""
        result_text = action.get("result_summary", "")
        msg = (
            f"[Step {i}] 观察到: {obs_text}\n"
            f"执行: {action['skill']}({action.get('parameters', {})})\n"
            f"结果: {result_text}"
        )
        messages.append({"role": "assistant", "content": msg})
    return messages
```

### 3.3 检索触发条件调整

原文本 Agent 的检索触发是用户提问文本。具身智能场景下，需要调整为：

```python
# 检索查询 = 任务指令 + 当前缺失的信息
def build_recall_query(
    instruction: str,
    current_observation: Observation,
    action_history: list[dict],
) -> str:
    """构建回忆查询。

    例如任务"把苹果放进抽屉"：
    - 如果已持有苹果但未见抽屉 -> query="抽屉 放置 容器"
    - 如果什么都没找到 -> query="苹果 位置"
    """
    pass
```

### 3.4 空间信息编码

TencentDB-Agent-Memory 的记忆内容是纯文本。空间信息（位置坐标、方向、距离）需要编码为可读文本：

```python
def encode_object_location(obs: dict) -> str:
    """将物体位置编码为自然语言描述。"""
    return (
        f"{obs['object_type']} 位于 {obs['distance']}m 处 "
        f"(坐标 {obs.get('position', {})})"
    )
```

---

## 4. 实施路线图

### 阶段 1：本地轻量存储（1 周）

目标：不依赖 MemoryCore Gateway，实现基础记忆能力

- 实现 `memory_store.py`：基于 SQLite + 本地 JSONL 的轻量存储
- 实现 `memory_capture.py`：将环境观察转换为记忆条目
- 实现 `memory_recall.py`：基于关键词的简单搜索（暂不做 Embedding）
- 集成到 `executor.py`：每步捕获、每轮召回
- 编写 `test_memory_store.py`

**预期效果**：Agent 能记住已见过的物体位置，减少重复探索。

### 阶段 2：接入 MemoryCore Gateway（1 周）

目标：启用完整的混合检索和 L1 提取能力

- 本地启动 MemoryCore Gateway（SQLite 模式）
- 将 `memory_store.py` 替换为 SDK 客户端
- 接入 L1 自动提取：将具身交互作为"对话"提交
- 接入混合检索（FTS5）
- 扩展 `memory_search` 为 LLM 可调用的技能

**预期效果**：Agent 能完成"先找再拿"的任务（规划中第 4 周目标）。

### 阶段 3：空间感知增强（1 周，可选）

目标：使记忆系统理解空间关系

- 扩展记忆类型，支持 `object_location`
- 在检索时加入空间过滤（如"附近"、"同一房间"）
- 在 L1 提取 Prompt 中加入空间信息模板
- 实现场景归纳（L2）：自动总结房间布局

---

## 5. 架构对比

### 修改前（当前 bot_agent）

```
┌──────────┐     ┌──────────────┐
│  LLM      │────▶│ SkillExecutor │
│  Planner  │     │  (pick/place) │
└────┬──────┘     └──────┬───────┘
     │                   │
     │  build_observation│  AI2-THOR
     │  (仅当前视野)     │  环境
     ▼                   │
┌────────────────────────┘
│  summarize()
└────────────────────────┘
```

### 修改后（集成记忆）

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│  LLM      │────▶│ SkillExecutor │────▶│ AI2-THOR │
│  Planner  │     │  (pick/place) │     │  环境    │
└────┬──────┘     └──────┬───────┘     └──────────┘
     │                   │
     │◀─ 增强观察 ────────┘
     │
     │  build_observation()
     │  + recall_memories()    ← 新增
     ▼                         │
┌────────────────────────┐     │
│  观察 + 记忆增强摘要    │     │
│  (当前视野 + 已见物体)  │     │
└────────┬───────────────┘     │
         │                     │
         │ capture_step()      │
         ▼                     │
┌────────────────────────┐     │
│  Memory Store          │     │
│  (SQLite + JSONL)      │     │
└────────────────────────┘     │
```

---

## 6. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| MemoryCore 依赖 Node.js | 本地运行需要额外环境 | 阶段 1 使用纯 Python 轻量存储，阶段 2 再引入 Gateway |
| L1 提取延迟 | 同步提取会阻塞 Agent loop | 异步捕获，LLM 提取失败不阻塞主流程 |
| Embedding 需要额外 API | 增加成本和延迟 | 初期仅用 FTS5 关键词搜索，Embedding 可选 |
| 记忆污染 | 过时位置信息误导决策 | 记忆附带时间戳，召回时按新鲜度排序 |
| SDK Python 版本要求 | 可能不兼容现有环境 | Python SDK 无外部依赖（仅 httpx），兼容性好 |