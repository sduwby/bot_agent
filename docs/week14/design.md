# 第 14 周设计文档 — 稳定性与阶段 B 评测

## 1. 概述

**本周目标：** 在第 13 周 ACT 通过门禁（final seeds 200–249：ACT 96% > BC 92%，`docs/week13/design.md` 10.24）之后，将评测推进到稳定性和阶段 B 门禁：随机初始位姿、多随机种子、动作平滑与错误分析，保留 scripted fallback。若 ACT 状态输入成功率 ≥80% 且安全违规为 0，则形成硬件采购意向与预算。

总计划原文：

> 第 14 周 | 稳定性与阶段评测 | 随机初始位姿、多随机种子、动作平滑和错误分析；保留 scripted fallback。 | ACT 状态输入成功率 ≥80%，安全违规为 0。

总计划"下一步"原文：

> 进入第 14 周：随机初始位姿、多随机种子、动作平滑与错误分析，保留 scripted fallback；若阶段 B 门禁（ACT 状态输入成功率 ≥80%）达标，则形成硬件采购意向与预算。

### 1.1 前置依赖

- 第 13 周正式 checkpoint：
  - PickCube-v1：`runs/maniskill_act/week13-dagger-train-r3/PickCube-v1/best-rollout-dev`（qvel + no-VAE + warm-start + DAgger r3，schema `maniskill_state_act_v2_qvel`）。
  - PushCube-v1：`runs/maniskill_act/week13/PushCube-v1/best`（历史 20k，schema `maniskill_state_act_v1`）。
- 第 12 周 BC checkpoint：`runs/maniskill_bc/week12/checkpoint.pt`（18 维状态）。
- 第 10 周 scripted policy、`ManiSkillTaskConfig`、`make_maniskill_env()`、`run_episode_in_env()` 继续作为环境与 rollout 契约。
- 第 13 周 `lerobot_act.py` 已提供 `load_lerobot_act_checkpoints()`、`evaluate_lerobot_act()`、`evaluate_bc_for_comparison()`、`evaluate_scripted_for_comparison()`、`compare_act_bc_scripted()`、`LeRobotACTPolicy`（含 `action_smoothing` EMA）。
- `.venv-lerobot`（Python 3.12 + LeRobot 0.6.1 + ManiSkill 3.0.1）复用；`pip check` 在开发开始前必须通过。

### 1.2 本周关键决策

1. **阶段 B 门禁以状态输入 ACT 为准**：只验证 state-only ACT（+可选 qvel）的稳定成功率，不引入 RGB 或视觉 ACT。
2. **"随机初始位姿"落实为环境级随机化**：ManiSkill PickCube/PushCube 的 `_initialize_episode()` 本来就按 seed 随机化 cube/goal 初始位姿。本周在其上显式记录初始位姿样本，验证随机化确实覆盖目标区域；不做额外采样扰动，避免破坏与训练分布的可比性。
3. **"多随机种子" = 更大、且与 train/dev/final 不重叠的 seed 集**：新增第 14 周评测 seed 集（如 300–399 或 500–599），与第 12 周 eval（100–149）、第 13 周 dev（150–169）、第 13 周 final（200–249）、DAgger 采集（300–359）区分，避免泄漏。
4. **"动作平滑" = 复用已有 `action_smoothing` EMA 参数**，在固定 seed 上做受控扫描并记录成功率/异常，不改变训练数据或 checkpoint。
5. **"错误分析" = 在第 14 周 seed 集上对 ACT 失败 episode 做全量失败相位分类**，复用 `run_act_diagnose.py` 的逐帧证据字段，形成失败分布。
6. **"保留 scripted fallback" = 评测中同时报告 scripted 基线**，并验证统一评测入口可以在 ACT/BC/scripted 间切换；正式门禁仍以无 fallback 的 ACT 独立成功率计，scripted 只作为专家/上界与安全回退参考。
7. **硬件采购**：阶段 B 门禁达标后只形成采购意向与预算文档，不实际下单（最终下单仍以第 22 周视觉 ACT 门禁为准）。

### 1.3 范围边界

本周包含：

- 第 14 周评测 seed 集的初始位姿采样记录与随机化覆盖验证；
- ACT/BC/scripted 在同一 seed 集上的稳定性评测（成功率、步数、异常、非有限动作）；
- `action_smoothing` 受控扫描与平滑对成功率/异常的影响记录；
- ACT 失败 episode 的全量相位分类与错误分析报告；
- 硬件采购意向与预算文档（`docs/week14/procurement.md`）。

本周不包含：

- RGB、视觉 ACT、SmolVLA、语言条件或多任务 ACT；
- 重训 ACT/BC、修改 Week 11/12/13 checkpoint 或数据；
- 修改 ManiSkill 成功条件、`goal_thresh`、动作空间或环境；
- 真实机械臂、相机、ROS 2、CUDA 或云 GPU；
- 采购下单或任何真实硬件支出。

## 2. 版本与设备设计

### 2.1 运行环境

- 复用 `.venv-lerobot`（Python 3.12.13、LeRobot 0.6.1、ManiSkill 3.0.1、SAPIEN 3.0.3、Pinocchio 4.1.0、Torch 2.11.0）。
- 若新增依赖，必须先在 `requirements-lerobot.txt` 记录并执行 `pip check`；本周预计不需要新依赖。
- macOS 仍复用 Homebrew MoltenVK 定位 ICD。

### 2.2 设备要求

- 当前 Apple M4、16 GB 内存足够完成 100–200 次 rollout 评测与逐帧诊断。
- 评测必须记录 device、耗时和每个 episode 的推理时延；不能只报告成功率。

## 3. 架构设计

```text
runs/maniskill_act/week13-dagger-train-r3  (PickCube best-rollout-dev)
runs/maniskill_act/week13                   (PushCube best)
runs/maniskill_bc/week12/checkpoint.pt      (BC)
        │
        ▼
act_stability.load_stability_checkpoints()
        │  按 env 分别加载 ACT，BC 独立加载
        ▼
初始位姿采样记录（record_initial_poses）
        │  每个 seed reset 后记录 cube/goal 位姿，验证随机化覆盖
        ▼
evaluate_policy_seeds()（复用 lerobot_act）
        ├── ACT（无 fallback）
        ├── BC
        └── scripted（专家/上界）
        │
        ▼
compare_act_bc_scripted_stability()（同 seed 门禁）
        │  稳定性门禁：ACT 总体 ≥80%，异常为 0
        ▼
错误分析（analyze_failures）
        │  对 ACT 失败 episode 复用 run_act_diagnose 逐帧证据
        ▼
runs/maniskill_act/week14/
        ├── stability_evaluation.json
        ├── comparison_stability.json
        ├── failure_analysis.json
        ├── smoothing_sweep.json
        └── initial_poses.json
```

### 3.1 模块职责

新增 `src/act_stability.py`：

- 加载第 13 周 ACT checkpoint（PickCube/PushCube 分别指向不同目录）、第 12 周 BC checkpoint；
- 初始位姿采样记录：每个 seed 的 `reset()` 后记录 `obj_pose` / `goal_pos`，输出覆盖统计；
- 复用 `lerobot_act.evaluate_policy_seeds()` 执行 ACT/BC/scripted 的同 seed 稳定性评测；
- `action_smoothing` 受控扫描（如 `{0.5, 0.75, 0.9, 1.0}`）并记录成功率/异常；
- 对 ACT 失败 episode 运行失败相位诊断并输出错误分析 JSON；
- 稳定性门禁判定与报告生成。

新增 `src/run_stability_eval.py`：

- 参数校验和默认输出目录；
- 执行初始位姿采样、三策略稳定性评测、平滑扫描、失败分析与门禁；
- 以退出码表达阶段 B 门禁是否通过。

新增 `src/tests/test_act_stability.py`：

- 不依赖 Unity 的初始位姿记录、seed 集、门禁阈值、失败相位分类和错误路径测试；
- fake policy/fake env 的 rollout 汇总与平滑扫描测试；
- LeRobot/ManiSkill 可用时的真实 smoke 评测（每任务 1–2 个 seed）。

复用（不修改或仅最小修改）：

- `src/lerobot_act.py`：`load_lerobot_act_checkpoints()`、`evaluate_policy_seeds()`、`LeRobotACTPolicy`、`PolicyEvaluationSummary` 等。
- `src/maniskill_intro.py`：环境工厂与 rollout 契约。
- `src/run_act_diagnose.py`：`diagnose_episode()` 失败相位诊断。

## 4. 数据结构

### 4.1 评测配置

```python
@dataclass(frozen=True)
class StabilityEvalConfig:
    act_pick_checkpoint: Path
    act_push_checkpoint: Path
    bc_checkpoint: Path
    output_dir: Path
    seeds: tuple[int, ...]           # 第 14 周独立评测 seed 集
    device: str = "mps"
    smoothing: float = 1.0
    tasks: tuple[str, ...] = SUPPORTED_TASKS
    max_steps: int = 50
    success_threshold: float = 0.80
```

### 4.2 初始位姿样本

```python
@dataclass(frozen=True)
class InitialPoseSample:
    env_id: str
    seed: int
    obj_pos: tuple[float, float, float]
    goal_pos: tuple[float, float, float]
```

### 4.3 失败相位分析

复用 `run_act_diagnose.StepTelemetry` / `EpisodeDiagnosis` 的字段，额外聚合：

```python
@dataclass(frozen=True)
class FailurePhaseSummary:
    total_failures: int
    phase_counts: dict[str, int]     # 按 final_phase 分类
    grasp_failures: int              # ever_grasped=False
    placed_but_not_static: int       # ever_placed=True 且 simultaneously_valid_steps=0
    overshoot_or_oscillation: int    # 进入目标区但未同时满足静止
    truncated_only: int              # 纯截断无抓取迹象
```

### 4.4 平滑扫描结果

```python
@dataclass(frozen=True)
class SmoothingSweepRow:
    smoothing: float
    success_rate: float
    total_episodes: int
    total_successes: int
    mean_steps: float
    nonfinite_actions: int
    exceptions: int
```

## 5. 接口设计

```python
def load_stability_checkpoints(
    pick_checkpoint: Path,
    push_checkpoint: Path,
    *,
    device: str,
    smoothing: float = 1.0,
) -> dict[str, object]:
    """按 env 分别加载 ACT checkpoint，返回 policies 映射。"""


def record_initial_poses(
    config: StabilityEvalConfig,
    *,
    env_factory: Callable = make_maniskill_env,
) -> list[InitialPoseSample]:
    """对每个 seed reset 后记录初始 cube/goal 位姿。"""


def summarize_pose_coverage(
    samples: Sequence[InitialPoseSample],
) -> dict[str, object]:
    """聚合每任务的位姿范围、均值与覆盖判定。"""


def evaluate_stability(
    config: StabilityEvalConfig,
    *,
    env_factory: Callable = make_maniskill_env,
) -> StabilityEvaluationSummary:
    """在固定 seed 集上执行 ACT/BC/scripted 评测并落盘。"""


def run_smoothing_sweep(
    config: StabilityEvalConfig,
    *,
    smoothing_values: Sequence[float],
    env_factory: Callable = make_maniskill_env,
) -> list[SmoothingSweepRow]:
    """受控扫描 action_smoothing 并记录成功率/异常。"""


def analyze_failures(
    config: StabilityEvalConfig,
    *,
    seeds: Sequence[int],
    env_factory: Callable = make_maniskill_env,
) -> FailurePhaseSummary:
    """对 ACT 失败 episode 复用 run_act_diagnose 做全量相位分类。"""


def compute_stability_gate(
    summary: StabilityEvaluationSummary,
    *,
    threshold: float = 0.80,
) -> dict[str, object]:
    """判定阶段 B 门禁：ACT 总体 ≥threshold 且异常为 0。"""
```

## 6. 评测数据流

### 6.1 初始位姿采样

1. 对每个 seed，用 `ManiSkillTaskConfig(env_id, seed)` 创建环境并 `reset(seed)`。
2. 从 observation `extra.obj_pose[:3]`、`extra.goal_pos[:3]` 记录初始位姿。
3. 关闭环境；汇总每任务 x/y/z 范围、均值与标准差，判定位姿随机化覆盖目标区域。

### 6.2 三策略稳定性评测

- 固定 seed 集（第 14 周独立，如 500–599 每任务 50 个）。
- ACT、BC、scripted 使用完全相同的环境工厂、任务配置、最大步数和 success 判据。
- ACT 禁止 scripted fallback；scripted 仅作为专家/上界参考。
- 输出 `stability_evaluation.json` 与 `comparison_stability.json`，包括成功率、步数、推理时延、非有限动作、异常与失败 episode。

### 6.3 平滑扫描

- 保持 checkpoint、seed 集不变，仅改变 `action_smoothing` ∈ `{0.5, 0.75, 0.9, 1.0}`。
- 记录各平滑值下的成功率、步数、非有限动作与异常。
- 结果写入 `smoothing_sweep.json`；不据此重训，只做诊断。

### 6.4 失败分析

- 对稳定性评测中的 ACT 失败 episode，复用 `run_act_diagnose.diagnose_episode()` 重新 rollout 并输出逐帧证据。
- 按 `final_phase`、`ever_grasped`、`ever_placed`、`simultaneously_valid_steps` 分类失败相位。
- 结果写入 `failure_analysis.json`。

### 6.5 门禁

```text
阶段 B 门禁通过 <=>
   ACT 总体成功率 >= max(0.80, 同轮 BC 成功率)   （取较高者）
   且 exceptions = 0
   且 nonfinite actions = 0
   且禁 scripted fallback
```

## 7. 失败处理与降级

| 失败 | 处理 |
| --- | --- |
| checkpoint 目录或 processor 缺失 | 拒绝评测，不使用内存中模型替代 |
| seed 与历史 eval/dev/final/DAgger 集重叠 | 拒绝并提示更换 seed 集 |
| 位姿采样中 observation 缺少字段 | 记为异常 episode，不允许静默跳过 |
| 非有限 action 或 shape 不为 `(4,)` | 当前 episode 失败并记录错误 |
| ACT 未达到门禁 | 标记阶段 B 部分完成；保留 BC/scripted，不宣称采购 |
| ManiSkill 不可用 | 仅运行纯逻辑测试，真实评测明确失败并给出安装指引 |

## 8. 验收标准

1. 第 14 周评测 seed 集与 train/dev/final/DAgger 均不重叠。
2. 初始位姿采样能输出每任务位姿覆盖统计，验证随机化生效。
3. ACT/BC/scripted 在同一 seed 集上完成稳定性评测并落盘。
4. `action_smoothing` 扫描完成并记录成功率/异常。
5. ACT 失败 episode 完成全量相位分类，输出错误分析报告。
6. 稳定性门禁：ACT 总体 ≥max(0.80, 同轮 BC)，exceptions=0，nonfinite=0，禁 scripted fallback。
7. 定向测试、完整回归、`compileall`、`.venv-lerobot` 的 `pip check` 和 `git diff --check` 通过。
8. 运行产物只写入已忽略的 `runs/`，不提交模型、轨迹或密钥。
9. 若门禁达标，形成 `docs/week14/procurement.md`（采购意向与预算，不下单）。

## 9. 参考资料

- [ManiSkill 3 文档](https://maniskill.readthedocs.io/en/latest/)
- [LeRobot ACT 文档](https://huggingface.co/docs/lerobot/main/act)
- 第 13 周 `docs/week13/design.md`、`docs/week13/fix_plan.md`、`docs/week13/dagger_plan.md`
- 总开发计划"阶段 B：机器人学基础与仿真机械臂（第 7–14 周）"

## 10. 实际偏差

### 10.1 实现偏差（2026-08-10）

1. **`.venv-lerobot` 补充安装 `ai2thor`**：本机只有 `.venv-lerobot`（Python 3.12 + LeRobot 0.6.1 + ManiSkill 3.0.1），基础 AI2-THOR 环境未保留。为运行完整回归，在 `.venv-lerobot` 中补装 `ai2thor==5.0.0`（`pip check` 通过），纯逻辑测试仍 mock，不连接 Unity。
2. **seed 集固定为 500–549（每任务 50 个）**：与历史 eval（100–149）、dev（150–169）、final（200–249）、DAgger（300–359）均不重叠，见 `default_stability_seeds()`。
3. **PickCube/PushCube checkpoint 目录不同**：PickCube 用 `runs/maniskill_act/week13-dagger-train-r3`（best-rollout-dev，qvel v2），PushCube 用 `runs/maniskill_act/week13`（best，v1）。`load_stability_checkpoints()` 按 env 分别解析。
4. **`PolicyEvaluationSummary` 无 `total_failures_for_metrics()`**：异常数改由 `_summary_exceptions()` 聚合各任务 `task.exceptions` 得出，两个调用点均已替换。
5. **`analyze_failures()` 依赖函数内 `from run_act_diagnose import diagnose_episode`**：测试通过 monkeypatch `run_act_diagnose.diagnose_episode` 注入 fake，而非补丁模块内部名字。

### 10.2 评测结果（2026-08-10）

**初始位姿采样**（seed 500–549，每任务 50 个）：PickCube goal_z 覆盖 [0.029, 0.318]，PushCube goal_z 恒定 0.001（Push 目标在桌面），验证任务随机化确实生效。产物 `runs/maniskill_act/week14/initial_poses.json`。

**三策略稳定性评测**（seed 500–549，每任务 50 个，共 100/策略）：

| 策略 | 成功率 | PickCube | PushCube | 平均步数 |
| --- | ---: | ---: | ---: | ---: |
| ACT（无 fallback） | 99/100（99.0%） | 49/50 | 50/50 | 16.0 |
| Week 12 BC | 96/100（96.0%） | 46/50 | 50/50 | 18.1 |
| scripted | 100/100（100.0%） | 50/50 | 50/50 | 16.6 |

- exceptions=0、nonfinite=0、禁 scripted fallback。
- 门禁：ACT 99.0% ≥ max(80.0%, BC 96.0%) = 96.0%，**阶段 B 门禁通过**。

**action_smoothing 扫描**（seed 500–519，每任务 20 个，共 40/值）：

| smoothing | 成功率 | 平均步数 | nonfinite |
| --- | ---: | ---: | ---: |
| 0.50 | 16/40（40.0%） | 36.7 | 0 |
| 0.75 | 34/40（85.0%） | 20.6 | 0 |
| 0.90 | 37/40（92.5%） | 17.7 | 0 |
| 1.00 | 39/40（97.5%） | 16.3 | 0 |

**结论：强平滑显著降低成功率**（0.5 降到 40%），1.0（不平滑）最优。第 14 周正式评测保持 `smoothing=1.0`；不引入动作平滑作为默认。

**失败相位分析**（seed 500–549，全量 ACT 失败）：PickCube 1 个失败为 `grasped_not_placed`（抓到但未放入），PushCube 0 个失败。产物 `runs/maniskill_act/week14/failure_analysis.json`。

**阶段 B 门禁结论**：ACT 状态输入成功率 99% ≥80%，exceptions=0、nonfinite=0、无 scripted fallback，门禁达标。形成硬件采购意向与预算：`docs/week14/procurement.md`。
