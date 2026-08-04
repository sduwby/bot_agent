# LLM Agent Loop 设计文档

## 1. 概述

将当前硬编码的 `pick → place` 技能链升级为 **LLM 驱动的 Agent 循环**：LLM 根据自然语言指令和当前环境观察，动态选择下一步技能。

### 1.1 验收目标

- 模型能完成**物体已可见的简单任务**（如"把苹果放进抽屉"、"把书放到柜子上"）
- 每次循环输出结构化决策（JSON），可通过 schema 校验
- 保留完整的轨迹日志（观察、决策、执行结果）

### 1.2 设计原则

- **LLM 不直接输出原子动作**（不输出 MoveAhead、RotateRight 等）
- **LLM 只调用高层技能**（scan、pick、place、move_to）
- **执行结果由环境返回，不依赖 LLM 主观判断**
- **保持与现有 SkillExecutor 的兼容**

---

## 2. 架构

```
┌─────────────────────────────────────────────────────┐
│                   LLM Agent Loop                    │
│                                                      │
│  ┌──────────┐       ┌──────────────┐                │
│  │  Planner  │──────▶│ SkillExecutor│                │
│  │(LLM 规划) │       │(技能+原子动作)│                │
│  └─────┬────┘       └──────┬───────┘                │
│        │                   │                         │
│        │            ┌──────▼───────┐                 │
│        │            │  AI2-THOR   │                 │
│        │            │   环境       │                 │
│        │            └──────┬───────┘                 │
│  ┌─────▼──────────────────▼──────┐                  │
│  │     Observation Summarizer    │                  │
│  └───────────────────────────────┘                  │
│                      │                              │
│  ┌───────────────────▼──────────┐                   │
│  │       Trajectory Logger      │                   │
│  └──────────────────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

**Planner 直接调用 SkillExecutor**，无中间调度层。Planner 输出 JSON 决策，`executor.py` 中的 `AgentExecutor` 负责校验参数、分发调用并收集结果。

---

## 3. 组件设计

### 3.1 Observation Summarizer

**输入：** AI2-THOR `event.metadata`
**输出：** 结构化文本（给 LLM 的观察摘要）

只保留对技能调用有用的信息，不发送原始 metadata。

```python
@dataclass
class Observation:
    """当前环境状态的精简表示。"""
    held_object_ids: list[str]           # 当前持有的物体 ID
    held_object_types: list[str]         # 持有的物体类型
    visible_pickupable: list[dict]       # 可见可抓取物体（ID、类型、距离）
    visible_receptacles: list[dict]      # 可见容器（ID、类型、距离、是否可打开）
    position: dict                       # 当前坐标 (x, y, z)
    rotation_y: float                    # 当前朝向（度）
```

### 3.2 Skill Registry

定义 LLM 可调用的技能列表和参数 schema。每个技能声明：

- 名称和描述
- 参数（JSON Schema）
- 返回类型（`SkillResult`）

```python
SKILLS = [
    {
        "name": "scan",
        "description": "原地旋转一周，收集周围可见的可抓取物体和容器",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "pick",
        "description": "找到并抓取指定类型的最近物体",
        "parameters": {
            "type": "object",
            "properties": {
                "object_type": {"type": "string", "description": "AI2-THOR 物体类型，如 CreditCard, Apple, Book"}
            },
            "required": ["object_type"],
        },
    },
    {
        "name": "place",
        "description": "将当前持有的物体放入指定类型的容器",
        "parameters": {
            "type": "object",
            "properties": {
                "receptacle_type": {"type": "string", "description": "AI2-THOR 容器类型，如 Drawer, Cabinet, GarbageCan"}
            },
            "required": ["receptacle_type"],
        },
    },
    {
        "name": "move_to",
        "description": "向指定方向移动，探索环境",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["forward", "backward", "left", "right"],
                             "description": "移动方向（相对当前朝向）"}
            },
            "required": ["direction"],
        },
    },
    {
        "name": "done",
        "description": "任务完成，不再执行任何操作",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]
```

### 3.3 LLM Planner

**输入：**
- 用户指令
- 当前 Observation 摘要
- 已执行技能的历史（技能名 + 结果摘要）
- 技能列表

**输出：** JSON 格式的技能调用决策

```json
{
  "thought": "扫描后发现一个 CreditCard 在距离 1.5 米处，我需要先抓取它",
  "skill": "pick",
  "parameters": {"object_type": "CreditCard"},
  "max_steps": 20
}
```

**System Prompt 模板：**

```
你是一个具身智能 Agent，在一个模拟的室内环境中执行任务。

## 你的能力

你可以调用以下技能：
{skills_description}

## 当前观察

{observation_summary}

## 已执行的操作

{action_history}

## 任务指令

{user_instruction}

请返回下一步操作。输出必须是合法的 JSON，包含以下字段：
- thought: 你的思考过程（简要说明）
- skill: 要调用的技能名称
- parameters: 技能参数
- max_steps: 本任务最大步骤数（避免无限循环，建议 10-20）
```

### 3.4 AgentExecutor（主循环）

整合在 `executor.py` 中。接收 Planner 的 JSON 输出，校验参数后直接调用 `SkillExecutor`：

1. 校验 JSON schema
2. 调用 `SkillExecutor` 对应方法
3. 将结果转换为摘要文本，回传给 Planner
4. 如果技能是 `done`，终止循环

### 3.5 Trajectory Logger

在每一步记录：

```json
{
  "step": 5,
  "phase": "planner",
  "planner_input": { "instruction": "...", "observation": { ... }, "history": [ ... ] },
  "planner_output": { "thought": "...", "skill": "pick", "parameters": { ... } },
  "execution_result": { "success": true, "data": { ... }, "failure_reason": null },
  "timestamp_utc": "..."
}
```

---

## 4. 数据流

```
用户指令
  → 初始化 AI2-THOR 环境
  → 获取初始 Observation
  → LLM Planner（指令 + 观察 + 技能列表）→ 返回技能调用
  → AgentExecutor 校验参数 → 直接调用 SkillExecutor → 返回 SkillResult
  → 更新 Observation
  → 追加到 action_history
  → 判断是否 done 或达到 max_steps
  → 循环或终止
  → 输出最终 summary.json
```

没有单独的 Executor 调度层。`AgentExecutor` 负责参数校验、SkillExecutor 分发和主循环编排。

---

## 5. 文件结构

新增文件：

```
src/
├── observation.py        # Observation 数据结构和 Summarizer
├── skill_registry.py     # 技能列表和参数 Schema
├── llm_planner.py        # LLM 调用（支持多种后端）
├── executor.py           # 主循环：AgentExecutor（含 Planner 调用 + JSON 校验 + SkillExecutor 分发）
└── run_agent.py          # 入口脚本（CLI，接受中文指令）
```

---

## 6. 实施计划

| 步骤 | 内容 | 验收 |
|------|------|------|
| 1 | `observation.py` — Observation 数据结构 + Summarizer | 能从 metadata 构建 Observation 并输出文本摘要 |
| 2 | `skill_registry.py` — 技能定义和参数校验 | 校验合法/非法参数 |
| 3 | `llm_planner.py` — LLM 调用接口 | 能接收 Observation，返回 JSON 技能决策 |
| 4 | `executor.py` — 主循环 + AgentExecutor | 能完成一次完整的 observe → plan → execute 循环 |
| 5 | `run_agent.py` — CLI 入口 | 支持 `--instruction "把信用卡放进抽屉"` 端到端运行 |
| 6 | 测试与调优 | 5 条指令手动测试，成功率 ≥70% |

---

## 7. LLM 后端策略

### 7.1 接口抽象

`llm_planner.py` 定义统一接口：

```python
class LLMBackend(Protocol):
    def chat(self, system: str, user: str, response_format: dict | None) -> str: ...
```

### 7.2 实现方案

- **OpenAI / 兼容 API**（默认）：通过环境变量 `LLM_API_BASE`、`LLM_API_KEY`、`LLM_MODEL` 配置
- **规则引擎回退**：如果 LLM 不可用，`RuleFallback` 实现简单的 scan → pick → place → done 序列

### 7.3 成本控制

- 每轮只发送精简 Observation（不是完整 metadata）
- 使用较小的模型（gpt-4o-mini 或同等级别）
- 设置 max_steps 上限防止无限循环