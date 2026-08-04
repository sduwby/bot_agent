# TencentDB-Agent-Memory 记忆系统分析

> 分析对象：`/Users/fittenwby/TencentDB-Agent-Memory`
> 分析日期：2026-08-04

---

## 1. 架构总览

TencentDB-Agent-Memory 是一个为 AI Agent 设计的**多层次记忆系统**，核心组件包括：

```
┌──────────────────────────────────────────────────────┐
│                    Agent (调用方)                     │
│  (OpenClaw / Hermes / 自定义)                        │
└───────────┬──────────────────────────────────────────┘
            │ HTTP API / SDK
            ▼
┌──────────────────────────────────────────────────────┐
│              MemoryCore Gateway (:8420)               │
│                                                      │
│  ┌─────────────┐   ┌──────────────────────────────┐  │
│  │   Hooks      │   │       Core Modules           │  │
│  │  auto-capture│──▶│  L0  l0-recorder.ts          │  │
│  │  auto-recall │   │  L1  l1-extractor + writer   │  │
│  └─────────────┘   │  L2  scene writer/index       │  │
│                    │  L3  persona writer            │  │
│  ┌─────────────┐   └──────────────────────────────┘  │
│  │  Tools      │   ┌──────────────────────────────┐  │
│  │  memory-    │   │       Storage Layer           │  │
│  │  search     │   │  SQLite + FTS5               │  │
│  │  conv-      │   │  JSONL (append-only)         │  │
│  │  search     │   │  Markdown (L2/L3 profiles)   │  │
│  └─────────────┘   │  VectorStore (optional)      │  │
│                    └──────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ MemoryKnowledge   │  (Wiki / CodeGraph 等知识源)
    └──────────────────┘
```

**关键设计原则：**
- **分层沉淀**：原始对话 → 原子事实 → 场景知识 → 长期画像，逐层提炼
- **按需召回**：记忆不作为全局 Prompt 注入，而是根据当前问题按需检索
- **稳定/动态分离**：L2/L3 等不变内容放入 system prompt 尾部利用 prompt cache；L1 动态内容放在 user prompt 前缀
- **错误降级**：记忆服务不可用时不影响主流程

---

## 2. 四层记忆模型

| 层级 | 存储位置 | 触发方式 | 用途 |
|------|---------|---------|------|
| **L0 原始对话** | `conversations/` JSONL 文件 + SQLite | 每轮对话结束自动捕获 | 完整上下文记录，可追溯原文 |
| **L1 原子记忆** | `records/` JSONL + SQLite/VectorStore | L0 捕获后异步 LLM 提取 | 结构化事实、偏好、规则，可精确召回 |
| **L2 场景知识** | `scene_blocks/` Markdown 文件 + `scene_index.json` | L1 积累后 LLM 归纳 | 按场景组织的知识块，快速恢复工作场景 |
| **L3 用户画像** | `persona.md` | L1 持续积累后 LLM 归纳 | 长期稳定的用户特征和模式 |

### 2.1 L0 — 原始对话

- 文件：`src/core/conversation/l0-recorder.ts`
- 按日期分片 JSONL 格式，每条记录包含 `sessionKey`、`sessionId`、`role`、`content`、`timestamp`
- **增量捕获**：通过位置切片 + 时间戳游标双重保护，避免重复记录
- **污染替换**：召回时 prepend 的记忆文本会污染用户消息，写入前会用缓存的原始文本替换
- **文本清洗**：去除 base64 图片、代码块，过滤纯符号消息
- **向量索引**：支持异步后台 Embedding，写入 metadata+FTS 后立即返回

### 2.2 L1 — 原子记忆

- 提取文件：`src/core/record/l1-extractor.ts`
- 写入文件：`src/core/record/l1-writer.ts`
- 读取文件：`src/core/record/l1-reader.ts`

**提取流程：**
1. LLM 对 L0 对话进行场景分割（scene segmentation）
2. 每个场景内提取记忆（事实、偏好、约束等）
3. 为每条记忆分配类型（preference/fact/rule/event/...）和优先级（0-100）

**去重机制（batch dedup）：**
- 对新提取的记忆与已有记忆做冲突检测
- 决策类型：`store`（新增）、`update`（覆盖）、`merge`（合并）、`skip`（跳过）
- 使用向量搜索召回候选，LLM 判断冲突关系

**存储结构：**
```typescript
interface MemoryRecord {
  id: string;                    // 唯一 ID
  content: string;               // 记忆内容
  type: MemoryType;              // 类型
  priority: number;              // 优先级 0-100
  scene_name: string;            // 所属场景
  source_message_ids: string[];  // 来源 L0 消息 ID
  metadata: object;              // 扩展字段
  timestamps: string[];          // 时间戳列表（多版本）
  version: number;               // 版本号
  sessionKey: string;
  teamId?: string;
  userId?: string;
  agentId?: string;
}
```

### 2.3 L2 — 场景知识

- 按场景（如"项目 A"、"产品讨论"）组织的 Markdown 文件
- `scene_index.json` 维护场景列表和路径索引
- 通过 `scene_writers.ts` 由 LLM 归纳 L1 记忆生成

### 2.4 L3 — 用户画像

- `persona.md` 文件，包含长期稳定的用户特征
- 定期由 L1 记忆归纳更新
- 召回时完整注入 system prompt（利用 prompt cache）

---

## 3. 核心模块分析

### 3.1 自动捕获（auto-capture）

`src/core/hooks/auto-capture.ts`

- 触发时机：Agent 对话轮次结束
- 流程：L0 写入 → L0 向量索引（异步） → L1 提取 → L1 去重 → 写入
- 原子性：文件锁保证 read-cursor → record → advance-cursor 不重复
- 降级：LLM 提取失败时 L0 仍可正常写入

### 3.2 自动召回（auto-recall）

`src/core/hooks/auto-recall.ts`

- 触发时机：每轮对话开始前（before_prompt_build）
- 三路并行：
  1. **L1 搜索**：根据用户文本搜索相关原子记忆
  2. **L2 加载**：读取场景索引生成导航
  3. **L3 加载**：读取 persona.md
- **预算控制**：条数限制（maxMemories ≤ 5）、字符限制（maxChars）、超时（5s）
- **稳定/动态拆分**：L2/L3 放入 system prompt（stable），L1 放入 user prompt（dynamic）

### 3.3 混合检索（memory-search）

`src/core/tools/memory-search.ts`

- **FTS5 关键词搜索**：SQLite 内置全文检索
- **向量搜索**：Embedding 生成后向量相似度排序
- **RRF 融合**：Reciprocal Rank Fusion 合并两路结果
- **原生混合搜索**：TCVDB 后端支持单次调用混合检索
- **次级过滤**：按类型、场景过滤

### 3.4 存储层

- **SQLite + FTS5**：本地轻量存储，支持关键词搜索
- **VectorStore**：可插拔接口（`IMemoryStore`），支持 SQLite 本地、TCVDB 远程等
- **JSONL 日志**：Append-only 持久化，按日期分片
- **Markdown 文件**：L2/L3 人类可读

---

## 4. SDK 接口（Python）

`sdk/memory-core/python/tencentdb_agent_memory/`

提供同步 `MemoryClient` 和异步 `AsyncMemoryClient`，按 v3 隔离（team_id/agent_id/user_id/session_id）：

| 接口 | 方法 | 说明 |
|------|------|------|
| **L0** | `add_conversation()` | 写入对话 |
|  | `query_conversation()` | 查询历史 |
|  | `search_conversation()` | 搜索对话 |
| **L1** | `query_atomic()` | 查询原子记忆 |
|  | `search_atomic()` | 搜索原子记忆 |
|  | `update_atomic()` | 更新/删除记忆 |
| **L2** | `read_file(scene_blocks/...)` | 读取场景文件 |
| **L3** | `read_file(persona.md)` | 读取用户画像 |

**集成模式（4 步）：**
1. **召回**：对话开始前调用 `recall`，将结果注入 prompt
2. **对话**：正常 LLM 循环
3. **捕获**：对话结束后调用 `add_conversation` 写入 L0
4. **工具**：可选暴露搜索工具让 LLM 主动检索

---

## 5. 与具身智能的适配点

### 5.1 可直接复用的能力

- **L1 提取/去重机制**：将具身智能交互（观察→动作→结果）视为"对话"，LLM 提取关键事实
- **混合检索**：BM25 + Embedding 方案适用于搜索已见物体
- **场景（L2）归纳**：可将房间布局、物体位置归纳为场景知识
- **预算控制**：条数、字符、超时控制可直接沿用
- **稳定/动态拆分**：系统不变知识放 system，当前观察放 user

### 5.2 需要适配的差异

| 维度 | 原文本记忆 | 具身智能场景 |
|------|-----------|-------------|
| **输入** | 用户/助手对话文本 | 环境观察（metadata）+ 动作历史 + 任务指令 |
| **记忆类型** | preference/fact/rule/event | 物体位置/房间状态/动作经验/失败教训 |
| **检索触发** | 用户提问文本 | 当前观察 + 任务进度 |
| **实时性** | 异步提取可接受 | 需要在 Agent loop 中同步完成 |
| **空间信息** | 无 | 需要记录位置、方向、距离 |

---

## 6. 总结

TencentDB-Agent-Memory 的**分层记忆 + 混合检索 + 自动捕获/召回**架构非常适合作为具身智能 Agent 的记忆基础。核心差异在于：

1. 输入源不同（对话 vs 环境观察），需要改造 capture 模块
2. 记忆类型需要扩展（增加 spatial/object_state 等类型）
3. 检索触发时机不同（需要在 Agent loop 中集成）
4. 需要保持轻量（bot_agent 是 Python 本地项目，不需要完整的 Gateway+Panel 架构）

建议采用**SDK 直连**模式（而非 Plugin），将 MemoryCore 作为本地侧车运行，通过 Python SDK 交互。