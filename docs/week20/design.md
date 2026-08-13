# 第 20 周设计文档 — 分层策略路由

## 1. 概述

**本周目标：** 把 scripted、状态 ACT、视觉 ACT 统一为技能执行器：同一技能接口可切换策略、定义选择顺序、超时和 fallback，失败可安全回退。SmolVLA 预留为可选策略位，加载失败不影响主线（探针与微调已按决策推迟到真机阶段，见总计划"下一步"）。

总计划原文：

> 第 20 周 | 分层策略路由 | 把 scripted、ACT、可选 SmolVLA 统一为技能执行器；定义选择、超时和 fallback。 | 同一技能接口可切换策略，失败可安全回退。

### 1.1 前置依赖

- 第 10 周 `maniskill_intro.py`：`ScriptedPolicy`（状态输入 FSM，`act(observation) -> 4D`）、`run_episode_in_env`。
- 第 12 周 `behavior_cloning.py`：`BCPolicy`（状态输入 MLP）。
- 第 13 周 `lerobot_act.py`：`LeRobotACTPolicy`（状态输入 ACT）、`evaluate_policy_seeds` 固定 seed 评测协议、`PolicyEvaluationSummary`。
- 第 16 周 `visual_act.py`：`VisualACTPolicy`（视觉输入 ACT，`act(observation, image)`）、`run_visual_episode_in_env`。
- 第 17 周 `perturbation.py`：扰动环境工厂；`run_pickcube_hybrid.py`：视觉 ACT→状态 ACT 混合切换的实验原型。
- 第 18 周 `smolvla_probe.py`：`make_smolvla_policy`（可选策略位复用的加载入口）。

### 1.2 本周关键决策

1. **技能粒度 = 任务粒度**：技能就是 `PickCube-v1`/`PushCube-v1` 的单 episode 执行。路由层不拆更细的子技能（approach/align/grasp），子技能切换已在第 17 周混合实验证明无法突破 PickCube 瓶颈，且门禁 C 用 PushCube + 抬升子技能组合达成。
2. **统一 Policy 接口 = 既有最小接口**：`phase` 属性 + `reset()` + `act(observation)`，视觉策略额外接受 `image`。路由层用 `input_mode`（`state`/`visual`）区分，不重写各策略。
3. **fallback 触发三类**：① 策略异常（非有限动作、shape 错误、推理崩溃）；② 停滞检测（连续 N 步 TCP 位移 < 阈值且无抓取状态变化）；③ 超时（skill 级 max_steps 用尽）。触发后按序切换到下一策略，**scripted 恒为末位安全网**。
4. **每个 seed 只重置一次环境**：fallback 切换在同一个 episode 内继续执行，不 reset 环境（状态连续性允许 ACT 中途接管）；若全部策略用尽仍失败，记 `exhausted`。
5. **SmolVLA 为可选位**：`PolicySource` 支持 `smolvla` 类型，但默认路由不含它；加载失败只警告并跳过该位，不抛错中断评测。
6. **评测协议复用第 14 周固定 seed 集**（500–549 分布内、600–649 分布外），保证与 `act_stability`/`run_generalization` 可比。

### 1.3 范围边界

本周包含：

- 统一的策略来源描述（`PolicySource`）、技能路由配置（`SkillRoute`）与执行器（`SkillRouter`）；
- 三类 fallback 触发（异常/停滞/超时）与切换执行；
- 固定 seed 评测 CLI 与路由报告（每策略执行率、fallback 次数、最终成功率）；
- SmolVLA 可选策略位（加载失败跳过）；
- 定向测试与完整回归。

本周不包含：

- 训练任何新模型；修改既有策略内部实现；
- 子技能级切换（第 17 周已否定）；
- 真实硬件、ROS、Safety Supervisor 实现（第 22 周选型与阶段 F 再做）；
- 修改第 14/17 周固定评测协议本身（只复用）。

## 2. 版本与设备设计

- 开发与纯逻辑测试：基础环境（无 ManiSkill 也可跑路由逻辑测试）。
- 真实评测：`.venv-lerobot`（Mac MPS 或 4060 cuda），复用第 16/17 周 checkpoint：
  - 状态 ACT：`runs/maniskill_act/week13-dagger-train-r3`（PickCube 96%）
  - 视觉 ACT：`runs/maniskill_act/week17-pick-grasp-30k`（PickCube 视觉）/ PushCube 视觉 checkpoint
  - scripted：内置 `ScriptedPolicy`
- 输出只写入已忽略的 `runs/`。

## 3. 架构设计

```text
SkillRoute(task, policies=[visual_act?, state_act?, scripted])
        │
        ▼
src/skill_router.py
   ├─ PolicySource：策略种类（scripted/bc/state_act/visual_act/smolvla）+ 工厂 + 输入模式
   ├─ SkillRouteConfig：任务 → 有序策略源列表 + 超时/停滞阈值 + 切换条件
   ├─ StagnationMonitor：TCP 位移与夹爪状态变化监测
   ├─ SkillRouter：单 episode 执行、异常捕获、fallback 切换、结果聚合
   └─ RouterEvaluationSummary：每策略执行/成功/fallback 统计与 JSON 落盘
        │
        ▼
src/run_skill_router.py（固定 seed 评测 CLI）
        │
        ▼
runs/maniskill_act/week20-router/
   ├─ router_report.json（每任务成功率、fallback 分布、逐 seed 明细）
   └─ ...
```

### 3.1 模块职责

新增 `src/skill_router.py`：

- `PolicySource`：不可变描述（`name`、`kind`、`input_mode`、`factory`、`optional`）。
- `SkillRouteConfig`：一个任务的策略顺序、`max_steps`、停滞窗口与阈值、`switch_on_exception` 等。
- `StagnationMonitor`：按观测帧更新，判定停滞（窗口内 TCP 位移均值 < 阈值且无 `is_grasped` 变化）。
- `SkillRouter`：`run_episode(env_id, seed)` 执行单 episode，返回 `RouterEpisodeResult`（成功与否、每策略执行的步数、fallback 原因链）。
- `evaluate_routes(config)`：多任务 × 固定 seed 批量评测，聚合 `RouterEvaluationSummary` 并落盘 JSON。

新增 `src/run_skill_router.py`：

- CLI：`--checkpoint-dir`（状态/视觉 ACT 目录）、`--tasks`、`--seeds`、`--route`（如 `state_act,scripted` / `visual_act,state_act,scripted`）、`--device`、`--output-dir`、`--perturbed`。

复用（不修改）：

- `maniskill_intro.py`：`ScriptedPolicy`、`ManiSkillTaskConfig`、`make_maniskill_env`。
- `lerobot_act.py`：`load_lerobot_act_checkpoints`。
- `visual_act.py`：`load_visual_act_checkpoints`、`_render_live_image`。
- `lerobot_pipeline.py`：`make_render_env`。
- `perturbation.py`：`make_perturbed_render_env`（分布外评测）。
- `behavior_cloning.py`：`load_bc_checkpoint`（可选 bc 位）。

## 4. 数据结构

```python
@dataclass(frozen=True)
class PolicySource:
    name: str                       # 报告中的策略名
    kind: str                       # scripted | bc | state_act | visual_act | smolvla
    input_mode: str                 # state | visual
    factory: Callable[[str], object]  # env_id -> policy 实例
    optional: bool = False          # 加载失败是否跳过

@dataclass(frozen=True)
class SkillRouteConfig:
    env_id: str
    sources: tuple[PolicySource, ...]
    max_steps: int = 50
    stagnation_window: int = 8
    stagnation_move_threshold: float = 0.005

@dataclass(frozen=True)
class RouterEpisodeResult:
    env_id: str
    seed: int
    success: bool
    steps: int
    policy_steps: dict[str, int]        # 每个策略实际执行的步数
    switch_events: tuple[SwitchEvent, ...]
    final_policy: str                   # 最后一个执行过动作的策略名
    exhausted: bool                     # 全部策略用尽仍失败

@dataclass(frozen=True)
class SwitchEvent:
    from_policy: str
    to_policy: str
    reason: str            # exception | stagnation | timeout
    at_step: int
```

## 5. 接口设计

```python
def build_default_route(
    env_id: str,
    *,
    state_act_checkpoint: Path | None = None,
    visual_act_checkpoint: Path | None = None,
    bc_checkpoint: Path | None = None,
    device: str = "mps",
    include_scripted: bool = True,
) -> SkillRouteConfig:
    """按可用 checkpoint 构造默认路由（可选位缺失即跳过）。"""


class StagnationMonitor:
    def reset(self) -> None: ...
    def update(self, observation: Mapping[str, object]) -> None: ...
    @property
    def is_stagnant(self) -> bool: ...


class SkillRouter:
    def run_episode(self, config: SkillRouteConfig, seed: int) -> RouterEpisodeResult: ...


def evaluate_routes(
    routes: Mapping[str, SkillRouteConfig],
    seeds: Sequence[int],
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    render_env: bool = False,
    output_dir: Path | None = None,
) -> RouterEvaluationSummary:
    """固定 seed 批量评测并落盘 router_report.json。"""
```

fallback 语义：

- `exception`：当前策略 `act()` 抛异常（含非有限动作），该步不执行，直接切换。
- `stagnation`：连续 `stagnation_window` 步停滞，切换并重置 monitor。
- `timeout`：当前策略执行步数达到该 skill 分配的步数上限。为简化，先按 episode 级 `max_steps` 与停滞共用：timeout 触发的场景 = 上一位策略耗尽 `max_steps` 且未成功（此时若还有下一位且 `max_steps` 未到总上限，则切换）。实现按"每策略步数上限 = 剩余总步数"处理。
- 切换后 monitor 与策略状态都 `reset()`；视觉→状态策略切换时无需额外对齐（第 17 周混合实验已验证状态 ACT 可在中途接管）。

## 6. 数据流

1. `evaluate_routes` 加载各策略（缺失的 optional 位跳过并警告）。
2. 对每个 `(env_id, seed)`：创建环境 → `SkillRouter.run_episode`。
3. 每步：按当前策略输入模式取 `act(obs)` 或 `act(obs, image)` → 校验动作 → `env.step` → `StagnationMonitor.update`。
4. 命中 fallback 条件则记录 `SwitchEvent` 并切到下一位；位耗尽且未成功 → `exhausted=True`。
5. 汇总每任务成功率、fallback 触发分布（按原因）、每策略步数占比，写 `router_report.json`。

## 7. 失败处理与降级

| 失败 | 处理 |
| --- | --- |
| 某策略 checkpoint 缺失 | 若 `optional=False` 且属于默认路由必需位则报错；可选位（smolvla）跳过 |
| 策略 act() 抛异常 | 记录 `SwitchEvent(exception)`，切下一位；最后一位也异常则 episode 记失败 |
| 动作非有限/shape 不符 | 视同策略异常，切换 |
| 视觉渲染失败 | 该 episode 记失败（环境级错误，不切换策略） |
| 全部策略耗尽 | `exhausted=True`，成功率为 0 计入报告 |

## 8. 验收标准

1. 同一技能接口可切换 scripted/状态 ACT/视觉 ACT，SmolVLA 位可选且加载失败不中断。
2. 三类 fallback（异常/停滞/超时）可触发并安全切换到下一位，scripted 恒为末位。
3. 固定 seed 评测完成并落盘：每策略执行率、fallback 原因分布、最终成功率。
4. 路由评测成功率不低于同 seed 下单一 scripted 基线（fallback 不应负优化）。
5. 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
6. 运行产物只写入已忽略的 `runs/`。

## 9. 参考资料

- 第 14 周 `docs/week14/design.md`（三策略同 seed 稳定性评测协议）
- 第 17 周 `docs/week17/pickcube_diagnosis.md` §8.7（混合切换结论：状态 ACT 可中途接管）
- 总计划 §7.1（模型职责与停止条件）

## 10. 实际偏差

### 10.1 实现偏差（2026-08-13）

1. **超时语义简化**：设计中原有"每策略步数上限"的 timeout 触发位。实现后简化为：episode 级 `max_steps` 全局共用（不按策略分配额），timeout 场景并入"episode 用尽 max_steps 且末位策略停滞 → exhausted"；fallback 触发收敛为异常（exception）与停滞（stagnation）两类，语义更简单且与 scripted 内部相位步数上限兼容。
2. **末位策略停滞不中断**：末位（scripted）停滞时不再切换（无处可切），继续执行到成功/截断/步数耗尽，并以 `exhausted=True` 标记"全部策略用尽仍失败"；末位策略异常才立即终止该 episode。
3. **smolvla 适配器内联**：`_SmolvlaAdapter` 直接放 `skill_router.py` 中（非独立模块），只做统一接口包装（6D state/单相机输入、4D action 截取），加载失败由 `optional=True` 跳过。
4. **真实 smoke 结果（4060，2026-08-13）**：`visual_act,scripted` 路由用 week17-c8-20k：PickCube 0/2（1 次 stagnation fallback 切到 scripted，视觉步数占比 0.78）、PushCube 2/2（纯视觉完成，无 fallback）；纯 scripted 基线 4/4。**fallback 链路真实可用且不产生负优化**。
5. **测试环境补装**：4060 的 `.venv-lerobot` 补装 `ai2thor` 以支持完整回归（此前缺失导致 test_task_skills/test_executor/test_evaluation 收集失败）。
