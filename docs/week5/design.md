# 第 5 周设计文档 — 验证与恢复模块

## 1. 概述

**本周目标：** 实现重复动作检测、动作失败分类、子任务验证、重新规划，至少处理 3 类常见失败。

**规划原文引用：**
> 第 5 周 | 验证与恢复 | 实现重复动作检测、动作失败分类、子任务验证、重新规划 | 至少处理 3 类常见失败

**前置依赖：**
- 第 3 周：AgentExecutor 主循环（executor.py）
- 第 4 周：记忆模块与搜索策略（memory_store.py 等）

## 2. 架构设计

### 2.1 模块总览

新增 4 个独立模块，通过 AgentExecutor 集成到主循环：

```
executor.py (主循环)
    ├── action_monitor.py  ← 检测重复动作/循环
    ├── failure_classifier.py ← 分类失败类型
    ├── verifier.py        ← 验证子任务是否完成
    └── replanner.py       ← 根据失败类型制定恢复策略
```

### 2.2 数据流

每轮 Agent 循环中的新流程：

```
1. [新增] ActionMonitor 检查 action_history → 是否循环？
   → 是：触发 Replanner (LOOP_DETECTED)
   → 否：继续

2. 执行技能 (skill: pick/place/move_to/scan/done)

3. 技能返回 SkillResult
   ┗━ 成功：
      [新增] SubtaskVerifier 二次验证 → 实际完成？
      → 验证失败：当作处理失败，继续下方流程
   ┗━ 失败：
      [新增] FailureClassifier 分类 → FailureCategory
      [新增] Replanner 根据分类 + 历史 → 恢复决策
      → 恢复决策 = retry/skip/scan/search/abort
```

## 3. 模块详细设计

### 3.1 ActionMonitor（重复动作检测）

**职责：** 跟踪 action_history 滑动窗口，检测循环模式。

**循环类型：**
- `IDENTICAL_REPEAT`：连续 N 步（默认 3）相同的 (skill, params)
- `OSCILLATION`：两个动作交替出现（A→B→A→B）

```python
@dataclass
class LoopDetection:
    """循环检测结果"""
    is_loop: bool
    loop_type: str | None  # "IDENTICAL_REPEAT" | "OSCILLATION"
    window_size: int
    details: str  # 人类可读的描述

class ActionMonitor:
    @staticmethod
    def detect_loop(
        history: list[dict[str, Any]],
        window: int = 3,
    ) -> LoopDetection:
        ...
```

**检测算法：**
1. 取最近 `window` 条 action_history
2. 提取每条的 key = (skill, json.dumps(params, sort_keys=True))
3. 如果所有 key 相同 → `IDENTICAL_REPEAT`
4. 如果只有 2 个唯一 key 且交替出现 → `OSCILLATION`
5. 否则 → 无循环

**与现有模块的关系：**
- 输入：action_history（executor.py 已有）
- 输出：LoopDetection，传递给 Replanner

### 3.2 FailureClassifier（动作失败分类）

**职责：** 将技能执行失败归类到预定义类型，便于 Replanner 选择恢复策略。

**失败类型枚举：**

| 类别 | 描述 | 触发条件 |
|------|------|---------|
| `OBJECT_NOT_FOUND` | 目标物体未找到 | pick 失败，reason 包含"未发现"或"not found" |
| `GRASP_FAILED` | 抓取失败 | pick 失败，reason 不包含"未发现" |
| `PLACEMENT_FAILED` | 放置失败 | place 失败 |
| `NAVIGATION_BLOCKED` | 移动受阻 | move_to 失败 |
| `UNKNOWN` | 未知失败 | 无法匹配以上规则 |

```python
from enum import Enum

class FailureCategory(str, Enum):
    OBJECT_NOT_FOUND = "object_not_found"
    GRASP_FAILED = "grasp_failed"
    PLACEMENT_FAILED = "placement_failed"
    NAVIGATION_BLOCKED = "navigation_blocked"
    UNKNOWN = "unknown"

@dataclass
class FailureInfo:
    """失败分类结果"""
    category: FailureCategory
    skill_name: str
    params: dict[str, Any]
    original_reason: str

class FailureClassifier:
    @staticmethod
    def classify_failure(
        skill_name: str,
        params: dict[str, Any],
        failure_reason: str,
    ) -> FailureInfo:
        ...
```

**分类规则：**
- pick + "未发现" / "candidates" → OBJECT_NOT_FOUND
- place → PLACEMENT_FAILED
- move_to → NAVIGATION_BLOCKED
- 其余 → UNKNOWN

**与现有模块的关系：**
- 输入：skill_name, params, failure_reason（来自 executor.py 的 SkillResult）
- 输出：FailureInfo，传递给 Replanner

### 3.3 SubtaskVerifier（子任务验证）

**职责：** 基于 AI2-THOR metadata 二次验证子任务是否实际完成，不只依赖 lastActionSuccess。

**验证方法：**

| 方法 | 验证条件 | 数据源 |
|------|---------|--------|
| `verify_pick` | inventoryObjects 包含目标物体 | metadata.inventoryObjects |
| `verify_place` | 目标物体的 parentReceptacles 包含目标容器 | metadata.objects |
| `verify_task_complete` | 组合指令 + pick/place 验证 | metadata + instruction |

```python
@dataclass
class VerificationResult:
    """验证结果"""
    success: bool
    confidence: float  # 0.0 - 1.0
    details: str

class SubtaskVerifier:
    @staticmethod
    def verify_pick(
        metadata: dict[str, Any],
        object_type: str,
    ) -> VerificationResult:
        """验证物体是否真的被抓取。

        条件：inventoryObjects 中有目标类型的物体。
        """
        ...

    @staticmethod
    def verify_place(
        metadata: dict[str, Any],
        object_id: str,
        receptacle_type: str,
    ) -> VerificationResult:
        """验证物体是否真的被放置到容器中。

        条件：目标物体的 parentReceptacles 中有目标类型容器。
        """
        ...

    @staticmethod
    def verify_task_complete(
        instruction: str,
        metadata: dict[str, Any],
    ) -> VerificationResult:
        """综合验证任务是否完成。

        策略：检查当前 metadata 是否满足任务要求。
        对于"把 X 放进 Y"类型指令，验证 X 是否在 Y 内。
        """
        ...
```

**与现有模块的关系：**
- 输入：controller.last_event.metadata（executor.py 已有）
- 输出：VerificationResult，失败时触发 FailureClassifier + Replanner 流程
- 与 task_skills.py 的区别：SkillResult 验证单个动作，SubtaskVerifier 是独立的第二层验证

### 3.4 Replanner（重新规划）

**职责：** 根据 FailureInfo 或 LoopDetection，选择恢复策略。

**恢复策略表：**

| 触发条件 | 恢复动作 | 最大重试 | 超时报错 |
|---------|---------|---------|---------|
| `OBJECT_NOT_FOUND` | 触发搜索模式 (move_to) | 不限制（由 SearchStrategy.max_steps 控制） | 搜索耗尽 |
| `GRASP_FAILED` | 重试 pick 同目标 1 次 | 1 | 标记为 skip，提示 LLM 换目标 |
| `PLACEMENT_FAILED` | 重试 place 下一个容器 | 1 | 标记为 skip，提示 LLM 换容器 |
| `NAVIGATION_BLOCKED` | 换方向 (rotate + forward) | 2 | 提示 LLM 换位置 |
| `LOOP_DETECTED` | 强制 scan 重新观察 | 不限制 | 提示 LLM 换策略 |
| `VERIFICATION_FAILED` | 重试原技能 1 次 | 1 | 按原失败类型分类处理 |

```python
@dataclass
class RecoveryAction:
    """恢复动作"""
    action: str  # "retry" | "scan" | "search" | "abort" | "llm_decision"
    reason: str
    skill_name: str | None   # action=retry 时指定
    params: dict[str, Any] | None
    is_final: bool  # 是否已达到最大重试次数

class Replanner:
    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries
        self._retry_counts: dict[str, int] = {}

    def plan_recovery(
        self,
        failure_info: FailureInfo | None = None,
        loop_detection: LoopDetection | None = None,
        verification_failed: bool = False,
    ) -> RecoveryAction:
        """根据失败信息生成恢复策略。"""
        ...

    def record_retry(self, skill_name: str) -> None:
        """记录某技能的重试次数。"""
        ...

    def should_retry(self, skill_name: str) -> bool:
        """检查某技能是否还有重试次数。"""
        ...

    def reset(self) -> None:
        """重置重试计数。"""
        ...
```

**与现有模块的关系：**
- 输入：FailureInfo（来自 FailureClassifier）、LoopDetection（来自 ActionMonitor）
- 输出：RecoveryAction，executor.py 根据 action 字段决定如何继续
- 与 SearchStrategy 的协作：OBJECT_NOT_FOUND 触发搜索时，复用 SearchStrategy

## 4. 集成设计

### 4.1 AgentExecutor 修改

在 executor.py 的 `run()` 主循环中新增：

```python
# 伪代码
for step in range(max_steps):
    # [新增] 循环检测
    loop = self.monitor.detect_loop(self.action_history)
    if loop.is_loop:
        recovery = self.replanner.plan_recovery(loop_detection=loop)
        # 执行恢复动作
        continue

    # [新增] 如果上一步是恢复策略，直接执行
    if pending_recovery:
        # 执行 recovery.skill_name + recovery.params
        continue

    # 原有：LLM 决策 → 执行 → 更新观察

    # [新增] 成功后二次验证
    if result.success and skill_name in ("pick", "place"):
        verified = self.verifier.verify_pick(...) or self.verifier.verify_place(...)
        if not verified.success:
            # 验证失败，进入恢复流程
            failure = self.classifier.classify_failure(...)
            recovery = self.replanner.plan_recovery(failure_info=failure)
            continue

    # [新增] 失败时分类 + 恢复
    if not result.success:
        failure = self.classifier.classify_failure(...)
        recovery = self.replanner.plan_recovery(failure_info=failure)
        # 根据 recovery 决定是 retry/search/scan 还是继续让 LLM 决策
        continue
```

### 4.2 System Prompt 更新

在 llm_planner.py 的 `_SYSTEM_PROMPT_TEMPLATE` 中增加失败恢复提示：

```
## 失败恢复

当操作失败时，你会收到错误信息。请根据错误类型选择不同策略：
- 物体未找到：尝试 scan 重新观察，或 move_to 移动到新位置
- 抓取失败：可尝试重试，或寻找同一类型的其他物体
- 放置失败：尝试换另一个容器
```

## 5. 数据结构

### 5.1 新增数据类型

```python
@dataclass
class LoopDetection:
    is_loop: bool
    loop_type: str | None
    window_size: int
    details: str

class FailureCategory(str, Enum):
    OBJECT_NOT_FOUND = "object_not_found"
    GRASP_FAILED = "grasp_failed"
    PLACEMENT_FAILED = "placement_failed"
    NAVIGATION_BLOCKED = "navigation_blocked"
    UNKNOWN = "unknown"

@dataclass
class FailureInfo:
    category: FailureCategory
    skill_name: str
    params: dict[str, Any]
    original_reason: str

@dataclass
class VerificationResult:
    success: bool
    confidence: float
    details: str

@dataclass
class RecoveryAction:
    action: str
    reason: str
    skill_name: str | None
    params: dict[str, Any] | None
    is_final: bool
```