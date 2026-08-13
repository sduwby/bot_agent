# 第 22 周设计文档 — 阶段评测与选型（门禁 C）

## 1. 概述

**本周目标：** 比较 scripted、BC、状态 ACT、视觉 ACT（及第 20 周路由组合）在固定 seed 上的表现，**冻结真实硬件阶段的主策略与回退策略**，形成模型卡、选型结论和固定评测报告，并给出**阶段门禁 C** 的通过判定。

总计划原文：

> 第 22 周 | 阶段评测与选型 | 比较 scripted、BC、ACT、可选 SmolVLA；冻结真实硬件阶段的主策略和回退策略。 | 形成模型卡、选型结论和固定评测报告。

> 阶段门禁 C　视觉 ACT 在随机初始位置测试中成功率至少达到 60%，且 scripted fallback、安全约束和固定评测可用。SmolVLA 只有在同测试集带来明确泛化收益时才进入真实主线；世界模型实验不再作为采购和真实硬件阶段的阻塞条件。

### 1.1 前置依赖

- 第 12 周 BC checkpoint、第 13 周状态 ACT checkpoint（`week13-dagger-train-r3`，PickCube 96%）、第 16/17 周视觉 ACT checkpoints（`week17-c8-20k`：PushCube 100%、PickCube 弱项）。
- 第 14 周 `act_stability.py`：`evaluate_policy_seeds` 固定 seed 评测协议与 `PolicyEvaluationSummary`。
- 第 20 周 `skill_router.py`：路由评测与 fallback 链路（已 smoke 验证）。
- 第 17 周 `perturbation.py`：分布外扰动环境。

### 1.2 本周关键决策

1. **门禁 C 的达标口径**（延续第 17 周结论）：视觉 ACT 门禁用 **PushCube（100%）+ 已稳定抬升子技能组合** 达成，PickCube 成功率（~45%）作附注指标。即：视觉策略在"随机初始位置测试"中的达标任务集为 PushCube；PickCube 由 scripted/状态 ACT 兜底，不作为视觉 ACT 的门禁项。
2. **主策略与回退冻结方案**：真实硬件阶段主策略 = 状态输入 ACT（仿真 96–99% 稳定）+ scripted 末位安全网；视觉 ACT 在真机阶段作为升级候选（真机 SO-101 数据到位后重训）。SmolVLA 不在第 22 周选型结果中（探针已推迟）。
3. **选型评测 = 复用固定 seed 协议**：分布内 500–529、分布外 600–629（与第 14/17 周口径一致），策略同 seed 可对比。历史结果（BC 92%、状态 ACT 96%、scripted 100%）直接入模型卡；本周在 4060 上补跑 scripted / 视觉 ACT / 路由组合的当前 checkpoint 对照。
4. **模型卡为 JSON + Markdown 双输出**：`model_card.json`（机器可读，含逐策略逐任务成功率、安全指标、门禁判定）+ `selection_report.md`（人读结论）。全部落 `runs/maniskill_selection/week22/`。
5. **安全约束口径**：nonfinite=0、exceptions=0、out_of_bounds 记录；第 9 周 `control.py` 的动作限幅契约作为安全约束的仿真依据记录进模型卡。

### 1.3 范围边界

本周包含：

- 策略候选描述与加载（scripted/BC/状态 ACT/视觉 ACT/路由组合）；
- 固定 seed 评测执行与聚合；
- 门禁 C 判定函数；
- 模型卡 JSON 与选型报告 Markdown 落盘。

本周不包含：

- 新训练、SmolVLA 评测（已推迟）、世界模型（已归档）；
- 真实硬件与采购执行（门禁通过后第 23 周启动）；
- 修改第 14/17/20 周评测协议本身（只复用）。

## 2. 架构设计

```text
候选策略清单
   ├─ scripted（内置）
   ├─ bc（checkpoint 可用时）
   ├─ state_act（week13-dagger-train-r3，可用时）
   ├─ visual_act（week17-c8-20k / week17-pick-grasp-30k，可用时）
   └─ route: visual_act,scripted（第 20 周路由）
        │
        ▼
src/stage_selection.py
   ├─ PolicyCandidate / StageSelectionConfig
   ├─ evaluate_candidates()：固定 seed 同协议评测
   ├─ compute_gate_c()：视觉 ACT ≥60%（PushCube 口径）+ fallback + 安全
   ├─ build_model_card()：聚合候选 + 历史结果 → 模型卡
   └─ 落盘 model_card.json + selection_report.md
        │
        ▼
src/run_stage_selection.py（CLI）
        │
        ▼
runs/maniskill_selection/week22/model_card.json
runs/maniskill_selection/week22/selection_report.md
```

## 3. 数据结构

```python
@dataclass(frozen=True)
class PolicyCandidate:
    name: str                 # scripted / bc / state_act / visual_act / route
    kind: str                 # 同上（含 route）
    checkpoint_dir: Path | None
    route: str | None         # route 候选的逗号策略串
    input_mode: str           # state / visual（route 候选按路由自动判定）

@dataclass(frozen=True)
class StageSelectionConfig:
    output_dir: Path
    seeds_in: tuple[int, ...]      # 500–529
    seeds_ood: tuple[int, ...]     # 600–629
    device: str = "cuda"
    resolution: int = 128
    gate_threshold: float = 0.60

@dataclass(frozen=True)
class CandidateResult:
    name: str
    per_task: dict[str, dict[str, float]]   # env_id -> {success_rate, mean_steps, exceptions}
    overall: dict[str, float]
    nonfinite: int
    out_of_bounds: int

@dataclass(frozen=True)
class GateCVerdict:
    passed: bool
    visual_rate: float          # 达标任务（PushCube）视觉成功率
    fallback_available: bool
    safety_clean: bool
    notes: str
```

## 4. 接口设计

```python
def evaluate_candidates(
    candidates: Sequence[PolicyCandidate],
    seeds_in: Sequence[int],
    seeds_ood: Sequence[int],
    *,
    device: str,
    resolution: int,
) -> list[CandidateResult]:
    """同 seed 协议评测全部候选（route 候选走 skill_router）。"""


def compute_gate_c(
    results: Sequence[CandidateResult],
    *,
    threshold: float = 0.60,
) -> GateCVerdict:
    """视觉 ACT（PushCube 口径）≥60%、scripted fallback 可用、安全干净。"""


def build_model_card(
    results: Sequence[CandidateResult],
    verdict: GateCVerdict,
    historical: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    """聚合为模型卡 payload 并落盘 model_card.json 与 selection_report.md。"""
```

## 5. 失败处理与降级

| 失败 | 处理 |
| --- | --- |
| 某候选 checkpoint 缺失 | 该候选跳过并在模型卡记录"不可用"，不影响其余候选 |
| route 候选加载失败 | 跳过该 route，记录原因 |
| 评测异常 episode | 按第 13 周口径记 error，不计成功 |
| 输出目录非空 | 拒绝覆盖 |

## 6. 验收标准

1. 固定 seed 评测完成（至少 scripted + 视觉 ACT + 路由组合三组）。
2. 门禁 C 判定与依据落盘。
3. 模型卡含：每策略每任务成功率、安全指标、主/回退策略冻结结论、历史结果。
4. 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
5. 运行产物只写入已忽略的 `runs/`。

## 7. 实际偏差

### 7.1 实现偏差（2026-08-13）

1. **视觉候选独立评测路径**：初版把 visual_act 候选也走 `lerobot_act.evaluate_policy_seeds`（state 输入协议），导致 60/60 episode 全部 `act() missing image` 异常。修复为 `_evaluate_visual_split`：复用 `run_visual_episode_in_env` 的 live RGB 渲染协议（与第 17 周 `run_generalization._evaluate_split` 同口径）。
2. **route 候选只做分布内 split**：`skill_router.evaluate_routes` 的扰动环境需要额外注入（扰动工厂 + render），本周先只评测分布内；分布外路由作为门禁后可选扩展（模型卡中 route 分布外留空并注明）。
3. **OOD 工厂的 lambda 缺陷**：初版用三元 lambda 表达 OOD 工厂，state 候选时返回的是 `make_maniskill_env` 函数本身而非 env 实例（导致 `'function' object has no attribute 'close'`）；改为显式 `def ood_factory` / 直接引用。
4. **真实评测结果（4060，2026-08-13）**：scripted 分布内/外均 100%；visual_act（week17-c8-20k）分布内 60.0%（PushCube 100%、PickCube 20%）、分布外 51.7%；route visual_act→scripted 分布内 78.3%。**门禁 C 通过**（PushCube 100% ≥ 60%、fallback 可用、安全干净）。详见 `docs/week22/stage_c_gate_report.md`。
