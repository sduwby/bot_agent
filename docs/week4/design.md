# 第 4 周设计文档 — 记忆与搜索模块

## 1. 概述

**本周目标：** 让 Agent 具备"先找再拿"的能力——在目标物体不在当前视野中时，能够记录已探索区域、维护已见物体记忆，并执行有策略的搜索。

**规划原文引用（阶段 A 第 4 周）：**
> 只向模型提供当前可见对象；维护已见物体、房间和持有状态；加入搜索策略。可完成"先找再拿"的任务。

**前置依赖：**
- 第 3 周 LLM Agent loop（`executor.py`、`observation.py`、`llm_planner.py`）
- 第 2 周技能接口（`task_skills.py`、`skill_registry.py`）

**参考文档：**
- [TencentDB-Agent-Memory 记忆系统分析](../memory_system_analysis.md)
- [bot_agent 集成 TencentDB-Agent-Memory 方案](../integration_plan.md)

**设计决策：**
采用 TencentDB-Agent-Memory 的分层记忆架构（L0/L1），但本地轻量实现——阶段 1 使用纯 Python 的 SQLite + JSONL 存储，不依赖 MemoryCore Gateway。保留后续接入完整 Gateway 的接口。

## 2. 架构设计

```
┌──────────────────────────────────────────────────────┐
│              AgentExecutor                           │
│                                                      │
│  ┌──────────┐    ┌───────────────────────┐          │
│  │  Planner  │──▶│   SkillExecutor       │          │
│  │ (LLM)     │   │   (scan/pick/place)   │          │
│  └────┬──────┘   └───────────────────────┘          │
│       │                                              │
│  ┌────▼─────────────────────────────────────────┐    │
│  │      MemoryStore（新增 — 本地轻量存储）         │    │
│  │  - L0: conversation.jsonl (append-only)      │    │
│  │  - L1: memories.jsonl (原子记忆)              │    │
│  │  - search(query) → 关键词匹配 + 类型过滤       │    │
│  └──────────────────────────────────────────────┘    │
│       │                                              │
│  ┌────▼─────────────────────────────────────────┐    │
│  │      MemoryCapture（新增 — 具身捕获适配器）     │    │
│  │  - observation → L0 对话格式                  │    │
│  │  - 提取物体位置、房间状态为 L1 记忆             │    │
│  └──────────────────────────────────────────────┘    │
│       │                                              │
│  ┌────▼─────────────────────────────────────────┐    │
│  │      MemoryRecall（新增 — 记忆召回）           │    │
│  │  - 根据任务+当前观察构建查询                   │    │
│  │  - 召回相关记忆注入 prompt                     │    │
│  └──────────────────────────────────────────────┘    │
│       │                                              │
│  ┌────▼─────────────────────────────────────────┐    │
│  │     SearchStrategy（新增 — 搜索策略）          │    │
│  │  - 选择下一个探索方向                          │    │
│  │  - 判断是否需要继续搜索                        │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

### 2.1 MemoryStore（记忆存储 — 本地轻量实现）

**职责：** 提供 L0/L1 记忆的双写和检索能力。

**参考 TencentDB-Agent-Memory 架构：**
- L0（原始对话）：JSONL append-only，记录每步交互的完整文本
- L1（原子记忆）：从 L0 提取结构化事实（物体位置、房间状态等），支持去重和更新
- 检索：初期使用关键词匹配（FTS5 备选），后续可扩展 Embedding

**核心数据结构：**

```python
@dataclass
class ConversationRecord:
    """L0 原始对话记录（对应 TencentDB-Agent-Memory 的 L0）"""
    session_key: str
    role: str          # "user" | "assistant"
    content: str       # 本轮交互的文本描述
    timestamp: int     # 毫秒级时间戳
    step: int          # 在 Agent loop 中的步骤编号


@dataclass
class MemoryRecord:
    """L1 原子记忆（对应 TencentDB-Agent-Memory 的 L1）"""
    id: str                        # 唯一 ID
    content: str                   # 记忆内容（自然语言描述）
    type: str                      # object_location | room_state | action_experience | failure_lesson
    priority: int                  # 1-100，越高越重要
    scene_name: str                # 场景名称
    source_step: int               # 来源步骤编号
    created_at: str                # ISO 8601 时间
    updated_at: str                # 最后更新时间


class MemoryStore:
    """本地轻量记忆存储，兼容 TencentDB-Agent-Memory 接口。"""

    def __init__(self, base_dir: str, session_key: str) -> None:
        self.base_dir = Path(base_dir)
        self.session_key = session_key
        self.conversations_file = self.base_dir / "conversations" / f"{session_key}.jsonl"
        self.memories_file = self.base_dir / "memories" / f"{session_key}.jsonl"

    def append_conversation(self, record: ConversationRecord) -> None:
        """写入 L0 对话记录（append-only）"""

    def append_memory(self, record: MemoryRecord) -> None:
        """写入 L1 记忆记录（append-only，去重在检索时处理）"""

    def search_memories(
        self,
        query: str,
        type_filter: str | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """搜索记忆。初期使用关键词匹配，按优先级和新鲜度排序。"""

    def query_memories(
        self,
        type_filter: str | None = None,
    ) -> list[MemoryRecord]:
        """按类型查询所有记忆"""

    def get_summary(self) -> str:
        """生成记忆摘要供 LLM prompt 使用"""
```

### 2.2 MemoryCapture（具身捕获适配器）

**职责：** 将具身交互（观察 → 动作 → 结果）转换为 L0/L1 记忆格式。

**参考 TencentDB-Agent-Memory 的 auto-capture 钩子：**
原文本 Agent 的 auto-capture 接收用户/助手对话。具身智能场景下，需要：
- 将任务指令转换为 user message
- 将观察 + 动作 + 结果转换为 assistant message
- 从交互中提取结构化 L1 记忆

```python
class MemoryCapture:
    """将具身交互转换为记忆条目。"""

    @staticmethod
    def observation_to_conversation(
        instruction: str,
        observations: list[Observation],
        action_history: list[dict],
    ) -> list[ConversationRecord]:
        """将观察和动作历史转换为 L0 对话格式。"""
        records: list[ConversationRecord] = []
        records.append(ConversationRecord(
            session_key="", role="user", content=instruction,
            timestamp=int(time.time() * 1000), step=0,
        ))
        for i, action in enumerate(action_history):
            obs_text = summarize(observations[i]) if i < len(observations) else ""
            result_text = action.get("result_summary", "")
            content = (
                f"[Step {i}] 观察到: {obs_text}\n"
                f"执行: {action['skill']}({action.get('parameters', {})})\n"
                f"结果: {result_text}"
            )
            records.append(ConversationRecord(
                session_key="", role="assistant", content=content,
                timestamp=int(time.time() * 1000), step=i + 1,
            ))
        return records

    @staticmethod
    def extract_memories_from_observation(
        observation: Observation,
        step: int,
        scene_name: str,
    ) -> list[MemoryRecord]:
        """从当前观察中提取 L1 记忆。

        提取规则：
        - 每个可见可抓取物体 → object_location 类型记忆
        - 每个可见容器 → room_state 类型记忆
        - 当前持有的物体 → object_location 类型记忆
        """
        memories: list[MemoryRecord] = []
        now = datetime.now(timezone.utc).isoformat()

        # 提取可见物体位置记忆
        for obj in observation.visible_pickupable:
            content = (
                f"{obj['object_type']} 在 {observation.scene_name} 中可见，"
                f"距离 {obj['distance']}m，"
                f"Agent 位置 ({observation.position['x']:.2f}, {observation.position['z']:.2f})，"
                f"朝向 {observation.rotation_y:.0f}°"
            )
            memories.append(MemoryRecord(
                id=f"m_{step}_{obj['object_id']}",
                content=content,
                type="object_location",
                priority=80,
                scene_name=scene_name,
                source_step=step,
                created_at=now,
                updated_at=now,
            ))

        # 提取容器状态记忆
        for rc in observation.visible_receptacles:
            status = "已打开" if rc["is_open"] else "已关闭"
            content = (
                f"{rc['object_type']} 容器在 {observation.scene_name} 中可见，"
                f"距离 {rc['distance']}m，{status}"
            )
            memories.append(MemoryRecord(
                id=f"m_{step}_{rc['object_id']}",
                content=content,
                type="room_state",
                priority=60,
                scene_name=scene_name,
                source_step=step,
                created_at=now,
                updated_at=now,
            ))

        return memories

    @staticmethod
    def extract_failure_memory(
        skill_name: str,
        params: dict,
        failure_reason: str,
        step: int,
        scene_name: str,
    ) -> MemoryRecord:
        """从失败的执行中提取教训记忆。"""
        now = datetime.now(timezone.utc).isoformat()
        content = (
            f"执行 {skill_name}({params}) 在 {scene_name} 中失败: {failure_reason}"
        )
        return MemoryRecord(
            id=f"m_fail_{step}",
            content=content,
            type="failure_lesson",
            priority=70,
            scene_name=scene_name,
            source_step=step,
            created_at=now,
            updated_at=now,
        )
```

### 2.3 MemoryRecall（记忆召回）

**职责：** 根据当前任务和观察，召回相关记忆。

**参考 TencentDB-Agent-Memory 的 auto-recall 钩子：**
原文本 Agent 在每轮对话前自动召回。具身智能场景下：
- 召回查询 = 任务指令 + 当前缺失的信息（如找不到目标物体）
- 注入 prompt 时区分稳定部分（system prompt）和动态部分（user prompt）

```python
class MemoryRecall:
    """根据任务+观察召回相关记忆。"""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def build_recall_query(
        self,
        instruction: str,
        observation: Observation,
        action_history: list[dict],
    ) -> str:
        """构建召回查询。

        策略：
        - 如果最近一次 pick 失败 → 搜索目标物体的位置
        - 如果最近一次 place 失败 → 搜索目标容器
        - 默认 → 搜索任务指令中的关键词
        """
        # 从最近失败中提取关键词
        if action_history:
            last = action_history[-1]
            if not last.get("success"):
                params = last.get("parameters", {})
                if last["skill"] == "pick":
                    return params.get("object_type", instruction)
                if last["skill"] == "place":
                    return params.get("receptacle_type", "容器")
        return instruction

    def recall(
        self,
        query: str,
        type_filter: str | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """执行记忆召回。"""
        return self.store.search_memories(query, type_filter, limit)

    def inject_into_prompt(
        self,
        memories: list[MemoryRecord],
    ) -> str | None:
        """将召回的记忆格式化为 prompt 片段。

        返回格式：
        ```
        ## 场景记忆

        以下是在当前场景中已探索过的信息，供参考：

        - CreditCard 在 FloorPlan1 中可见，距离 2.3m，Agent 位置 (1.00, 3.00)，朝向 90°
        - Drawer 容器在 FloorPlan1 中可见，距离 1.5m，已打开
        ```
        """
        if not memories:
            return None
        lines = ["## 场景记忆\n\n以下是在当前场景中已探索过的信息，供参考：\n"]
        for m in memories:
            lines.append(f"- {m.content}")
        return "\n".join(lines)
```

### 2.4 SearchStrategy（搜索策略）

**职责：** 基于记忆和当前状态，决定下一步是否继续搜索、向哪个方向移动。

**策略设计（简化版）：**
1. 检查记忆中是否记录过目标类型的物体 → 如果有，导航到记录的位置并尝试抓取
2. 如果没有，选择最少探索次数的前方位置进行探索
3. 达到最大搜索步数或所有可达位置都探索过时，返回搜索失败

```python
class SearchStrategy:
    def __init__(self, store: MemoryStore, max_steps: int = 8) -> None: ...

    def should_search_for(self, object_type: str) -> bool:
        """目标类型是否在记忆中？不在则需搜索"""

    def next_move(self) -> dict[str, Any]:
        """返回下一步建议（move_to 技能调用）"""

    def is_exhausted(self) -> bool:
        """搜索是否已耗尽所有选项"""

    def is_position_explored(self, position: dict, tolerance: float = 0.5) -> bool:
        """判断给定位置是否已经探索过"""

    def reset(self) -> None:
        """重置搜索状态"""
```

## 3. 数据结构

### 3.1 L0 对话记录

```python
@dataclass
class ConversationRecord:
    session_key: str
    role: str          # "user" | "assistant"
    content: str
    timestamp: int
    step: int
```

### 3.2 L1 原子记忆

```python
@dataclass
class MemoryRecord:
    id: str
    content: str
    type: str          # object_location | room_state | action_experience | failure_lesson
    priority: int      # 1-100
    scene_name: str
    source_step: int
    created_at: str
    updated_at: str
```

## 4. 接口设计

### 4.1 MemoryStore 公开 API

| 方法 | 说明 |
|------|------|
| `append_conversation(record)` | 写入 L0 对话记录 |
| `append_memory(record)` | 写入 L1 记忆记录 |
| `search_memories(query, type_filter, limit)` | 关键词搜索 |
| `query_memories(type_filter)` | 按类型查询 |
| `get_summary()` | 生成记忆摘要 |

### 4.2 MemoryCapture 公开 API

| 方法 | 说明 |
|------|------|
| `observation_to_conversation(...)` | 将观察转换为 L0 对话格式 |
| `extract_memories_from_observation(...)` | 从观察中提取 L1 记忆 |
| `extract_failure_memory(...)` | 从失败中提取教训 |

### 4.3 MemoryRecall 公开 API

| 方法 | 说明 |
|------|------|
| `build_recall_query(...)` | 构建召回查询 |
| `recall(query, type_filter, limit)` | 执行召回 |
| `inject_into_prompt(memories)` | 格式化为 prompt 片段 |

### 4.4 SearchStrategy 公开 API

| 方法 | 说明 |
|------|------|
| `should_search_for(type)` | 是否需要搜索 |
| `next_move()` | 下一步移动建议 |
| `is_exhausted()` | 搜索是否耗尽 |
| `is_position_explored(pos)` | 位置是否已探索 |
| `reset()` | 重置状态 |

### 4.5 AgentExecutor 修改

```python
class AgentExecutor:
    def __init__(self, ..., max_steps=20, max_search_steps=8) -> None:
        # 新增：
        self.store = MemoryStore(run_dir, session_key)
        self.capture = MemoryCapture()
        self.recall = MemoryRecall(self.store)
        self.search = SearchStrategy(self.store, max_search_steps)
        self.searching = False  # 当前是否在搜索模式

    def run(self, instruction: str) -> dict:
        # 主循环中新增逻辑：
        # 1. 每步执行后调用 capture 写入 L0/L1
        # 2. 每轮开始前调用 recall 注入 prompt
        # 3. 如果 skill 执行失败且目标物体未找到，进入搜索模式
        # 4. 搜索模式下由 SearchStrategy 决定下一步
```

## 5. 数据流

```
1. Agent 初始化
   → MemoryStore, MemoryCapture, MemoryRecall, SearchStrategy 初始化

2. 每步循环：
   a. recall.recall(build_recall_query(...))  — 召回相关记忆
   b. recall.inject_into_prompt(memories)     — 注入 prompt
   c. 获取当前 Observation
   d. LLM 返回技能决策

3. 技能执行后：
   a. capture.extract_memories_from_observation(obs)  → 提取 L1 记忆
   b. store.append_memory(memories)                     → 写入存储
   c. 如果失败：capture.extract_failure_memory(...)     → 记录教训

4. 如果 pick 失败且目标物体不在记忆中：
   → 进入搜索模式
   → SearchStrategy.next_move() 返回 move_to 决策
   → 移动后回到步骤 2

5. 如果 pick 失败但目标物体在记忆中：
   → 导航到记录位置并重新尝试

6. 搜索耗尽或达到 max_search_steps：
   → 退出搜索模式，报告任务失败

7. 任务结束后：
   → capture.observation_to_conversation(...)  → 完整 L0 记录
   → store.append_conversation(records)        → 写入 JSONL
```
