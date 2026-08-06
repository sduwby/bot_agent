# 第 6 周设计文档 — 批量评测与阶段 A 门禁

## 1. 概述

**本周目标：** 建立固定、可复现的 AI2-THOR 批量任务集，逐任务运行完整
`AgentExecutor` 闭环，统计成功率、动作数、Planner/LLM 调用数和失败类型，最后输出阶段 A
报告并完成 70% 成功率门禁评审。

**规划原文引用：**

> 第 6 周 | 批量评测 | 建立 10–20 条任务集；统计成功率、动作数、LLM 调用数、失败类型。 | 完成阶段报告和门禁评审。

> 阶段门禁 A：在结构化环境状态下，至少 10 个测试任务总体成功率达到 70%；能输出失败分类和完整轨迹。

**前置依赖：**

- 第 3 周：`AgentExecutor`、Planner 后端和逐步轨迹记录。
- 第 4 周：搜索与 L0/L1 记忆。
- 第 5 周：精确任务验证、失败分类和有限恢复。

## 2. 设计边界与原则

- 评测必须调用真实 `AgentExecutor`，不能直接调用技能或用预写结果代替闭环。
- 固定任务集纳入版本控制；运行截图、轨迹和报告实例继续写入已忽略的 `runs/`。
- 每个任务初始化独立 Controller 和独立输出目录，避免场景状态与 Agent 内部状态串扰。
- 成功只采用 `AgentExecutor` 的精确最终状态验证结果；初始化成功、技能成功或脚本退出均不能替代任务成功。
- `planner_calls` 统计所有 Planner 后端调用；`llm_calls` 只统计真实 API LLM 调用。规则基线的 `llm_calls` 为 0，并在报告中明确后端类型。
- 单个任务异常必须转为失败 episode 并继续后续任务，保证批次报告完整。
- 门禁只在任务数不少于 10 且成功率不低于 70% 时通过；测试/配置验证成功不等于门禁通过。

## 3. 架构设计

```text
evaluation/stage_a_tasks.json (固定任务集)
                    │
                    ▼
src/run_evaluation.py (CLI、批次生命周期、Controller 隔离)
                    │
                    ├── 每个任务 ──▶ AgentExecutor.run()
                    │                   ├── summary.json
                    │                   └── trajectory.jsonl + RGB/L0/L1
                    │
                    ▼
src/evaluation.py (任务加载、结果归一化、指标聚合、报告渲染)
                    │
                    ├── episodes.jsonl
                    ├── report.json
                    └── report.md
```

### 3.1 `AgentExecutor` 计数扩展

在不改变决策行为的前提下增加两个运行级计数器：

- `planner_calls`：每次执行 `backend.chat(...)` 前递增。
- `llm_calls`：仅当后端声明 `counts_as_llm = True` 时递增。

两项计数写入任务 `summary.json`。自动搜索、重试和循环恢复不调用 Planner，因此不计入。

### 3.2 固定任务集

任务集使用 JSON，包含 12 条固定任务，覆盖：

- 多个厨房场景；
- 可见目标与需要搜索的目标；
- 开放承载面和可开闭容器；
- 多种可抓取物体与自然语言表达。

每条任务包含：

```json
{
  "id": "a01",
  "name": "信用卡放入抽屉",
  "scene": "FloorPlan1",
  "instruction": "把信用卡放进抽屉",
  "tags": ["direct", "openable_receptacle"]
}
```

加载时校验版本、任务数量（10–20）、ID 唯一性、必填字符串、标签格式和指令可解析性。

## 4. 数据结构与公开接口

### 4.1 核心数据类型

```python
@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    name: str
    scene: str
    instruction: str
    tags: tuple[str, ...]

@dataclass
class EpisodeResult:
    task_id: str
    scene: str
    instruction: str
    task_success: bool
    termination_reason: str
    steps_taken: int
    planner_calls: int
    llm_calls: int
    failure_counts: dict[str, int]
    elapsed_seconds: float
    run_dir: str
```

### 4.2 函数接口

```python
def load_task_set(path: Path) -> list[EvaluationTask]: ...

def classify_episode_failures(
    action_history: list[dict[str, Any]],
    termination_reason: str,
) -> dict[str, int]: ...

def aggregate_results(
    results: list[dict[str, Any]],
    *,
    minimum_tasks: int = 10,
    success_threshold: float = 0.70,
) -> dict[str, Any]: ...

def render_markdown_report(report: dict[str, Any]) -> str: ...
```

## 5. 指标与失败口径

| 指标 | 计算方式 |
| --- | --- |
| 任务成功率 | `task_success=True` 的 episode 数 / 总 episode 数 |
| 动作数 | `AgentExecutor.action_history` 长度；同时输出总计、均值、中位数、最小值和最大值 |
| Planner 调用数 | 实际进入 `backend.chat` 的次数 |
| LLM 调用数 | 使用真实 LLM 后端的 Planner 调用次数；规则后端为 0 |
| 失败类型 | 对每个失败动作调用既有 `FailureClassifier`，并补充参数校验、任务验证、解析、搜索耗尽、步骤上限和异常等终止类别 |

失败统计同时保留两层：

1. `action_failures`：episode 中所有失败动作，包含后续恢复成功的失败。
2. `failed_episode_terminations`：仅失败 episode 的最终终止原因。

这样可以区分“发生过失败但恢复成功”和“最终任务失败”。

## 6. 数据流

1. CLI 读取任务集并应用可选 `--task-id` 过滤。
2. 创建批次目录，记录开始时间、代码提交、任务集 SHA-256、后端、模型和运行参数。
3. 对每个任务创建独立目录与 Controller，运行完整 Agent 闭环。
4. 落盘 episode 原始摘要、完整轨迹、记忆和归一化结果；异常只终止当前 episode。
5. 聚合所有 episode，计算指标与门禁结论。
6. 原子性地生成 `episodes.jsonl`、`report.json` 和人类可读的 `report.md`。
7. CLI 仅在批次完整且门禁通过时返回 0；任务不足、批次不完整或成功率不足均返回非零。

## 7. CLI 设计

```bash
python src/run_evaluation.py \
  --task-set evaluation/stage_a_tasks.json \
  --backend rule
```

关键参数：

- `--backend rule|auto`：`rule` 是确定性回归基线；`auto` 根据本地环境选择真实 LLM 或规则回退。
- `--task-id ID`：可重复指定，用于快速调试；过滤后少于 10 条时报告可生成但门禁不通过。
- `--max-steps`、`--max-search-steps`、`--width`、`--height`：固定写入配置快照。
- `--output-root`：默认 `runs/evaluation`。

## 8. 测试与验收策略

- 单元测试覆盖任务集加载、重复 ID/数量/指令错误、失败分类、聚合计算、门禁边界和 Markdown 输出。
- Executor 回归测试验证 Planner/LLM 调用计数不会把自动恢复误计为 LLM 调用。
- 运行全量 `pytest` 和 `compileall`。
- 使用真实 AI2-THOR 跑完固定任务集；报告必须包含每条任务的可追溯目录。
- 阶段 A 门禁报告记录实际后端和验证边界，不能把规则基线结果表述为真实 LLM 结果。

## 9. 实际偏差

- 任务集和框架按设计完成，无接口级偏差。
- 首轮规则基线为 4/12（33.3%）；按第 10 节修复搜索、关闭容器检查、放置恢复后，最终规则基线为 10/12（83.3%）。
- 首轮完整真实 LLM 对照为 7/12（58.3%）；根据轨迹补充内置搜索契约 Prompt 后，`qwen3.7-plus` 最终为 10/12（83.3%），39 次真实调用。
- `a08` 是当前 AI2-THOR 的 Cup→Drawer 兼容性缺陷，`a12` 是持有物体后的放置搜索未收敛；两项按失败保留，未修改任务集或放宽门禁。
- 实现中补充了 `git_dirty` 元数据，避免只记录 HEAD 时把未提交工作区误表述为该提交的可复现结果。
- 两组最终批次均通过阶段 A 门禁，第 6 周状态更新为完成，可以进入第 7 周。

## 10. 门禁未通过后的收敛修复设计

### 10.1 失败证据

首轮失败轨迹显示两条确定的调用链问题：

1. `_SEARCH_DIRECTIONS` 依次执行前进和转向，8 个搜索步骤中有多步只改变朝向；导航受阻后自动反向又可能回到已探索位置，无法覆盖房间。
2. `PLACEMENT_FAILED` 只做有限原地重试；重试耗尽后 `_apply_recovery` 不处理 `abort`，规则 Planner 仍根据持有状态继续输出同一 `place`，循环恢复也主要产生 scan/转向而非新视点。
3. 真实场景元数据显示 a03 的 Cup 位于关闭 Cabinet、a05 的 Apple 位于关闭 Fridge、a06 的 Cup 位于关闭 Cabinet；只移动和扫描无法让这些目标变为可见。
4. a08 的原始 Unity 错误为 `Cup ... cannot be placed in Drawer`，属于当前 AI2-THOR 物体—容器兼容性限制，不是搜索或恢复可解决的执行失败。

### 10.2 可达视点搜索

- 第一次进入搜索时通过 AI2-THOR `GetReachablePositions` 获取当前场景的可达网格点；该查询不读取目标物体位置，不破坏“只向 Planner 提供当前可见对象”的边界。
- `SearchStrategy` 缓存可达点，以已探索位置为种子使用最远点采样：每轮选择与所有已探索位置最小平面距离最大的点。相同距离按坐标稳定排序，保证规则基线可复现。
- 每次全向扫描同时记录“容器类型 → 当前可达视点”；抓取后搜索放置目标时，优先回访尚未使用的已知目标容器视点，再退回最远点采样，避免已经看到 Microwave/Cabinet 后又盲目探索。
- 搜索决策仍复用 `move_to`，增加内部参数 `direction="reachable"` 和 `target_position`；Executor 使用 `TeleportFull` 到 AI2-THOR 已声明可达的点。它是阶段 A 的抽象导航技能，不代表阶段 B 机械臂连续控制。
- 查询和传送均写入原子轨迹；高层动作数仍按 Agent `action_history` 统计。
- 若可达位置查询失败或没有未探索点，回退到现有相对方向策略并继续受最大搜索步数约束。
- 当搜索目标是待抓取物体时，每个新视点先扫描当前可见容器，并逐个打开可交互的关闭容器（Cabinet、Drawer、Fridge、Microwave 等）；每次打开后在水平、向下 30°、向上 30° 三个相机俯仰角全向探测目标，并恢复原俯仰角。找到目标则保留容器开启，未找到则关闭并记录为已检查，避免开启的门板遮挡后续容器。开闭和扫描动作均写入原子轨迹。搜索放置目标时不做该操作，因为 `place` 已负责目标容器开闭。

### 10.3 放置恢复

- `PLACEMENT_FAILED` 不再在同一视点重复放置，直接返回 `search` 恢复。
- 当失败技能为 `place` 时，Executor 将目标设为 `receptacle_type` 并设置“强制移动”标志；即使同类型容器当前可见，下一轮也必须先去新可达视点，再执行 place 探测。
- 每次失败后的新视点由同一最远点采样器选择；成功放置后仍按原逻辑清除搜索状态。
- 若目标容器当前可见且已按原流程打开，普通 `PutObject` 仅因 `No valid positions to place object found` 失败，则在同一候选上执行一次 `forceAction=True` 语义放置兜底；兜底结果仍必须同时满足 inventory 清空、物体 `parentReceptacles` 和容器 `receptacleObjectIds` 三重状态验证。AI2-THOR 仍会拒绝不兼容关系（例如 Cup→Drawer），因此不能绕过任务语义兼容性。

### 10.4 保持不变的验收边界

- 不修改 `evaluation/stage_a_tasks.json`。
- 不降低 70% 成功率门禁。
- 复评仍使用 `max_steps=20`、`max_search_steps=8` 和真实 AI2-THOR。
- 先验证规则基线；通过后再执行真实 LLM 对照。
- a08 暂时保留在同一任务集中并按失败计入门禁，同时在报告中标记为任务集兼容性缺陷；即使它失败，12 条任务仍可通过 70% 门禁。

### 10.5 真实 LLM 对照的运行边界

- OpenAI 兼容客户端默认单请求超时 60 秒、最多重试 1 次，可通过 `LLM_TIMEOUT_SECONDS` 和 `LLM_MAX_RETRIES` 覆盖。
- 超时或 API 异常只终止当前 episode，批量 runner 继续执行后续任务并归类为 `exception`。
- 规则基线门禁与真实 LLM 对照分别报告；外部模型失败不抹去已经通过的确定性执行基线，但必须作为 Planner 缺口记录。
- 首轮真实 LLM 失败轨迹若表现为重复 `scan/move_to`、迟迟不调用目标 `pick/place`，System Prompt 必须明确 Executor 的内置搜索契约：不可见目标也应调用目标技能触发恢复，禁止手工重复探索，并严格限制方向枚举。
