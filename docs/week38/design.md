# 第 38 周 设计文档：深度化主线 —— VLM 数据引擎 + VLA 适配 + 世界模型评估

## 1. 概述

### 1.1 背景与问题

项目在广度上已完整（AI2-THOR 语义 Agent → ManiSkill 策略学习 → SO-101 真机采集部署 → LLM/VLM 决策层 → 评测 harness），但从外部评审视角看存在结构性问题：

- **每个环节都是"调用现成方法 + 跑通链路"**：ACT 来自 LeRobot、SmolVLA 来自 LeRobot、感知与规划是调 LLM API，没有一样是自研方法；
- **没有受控实验**：有评测脚本，但没有"固定变量、改变一个因素、给出对照表"的实验；
- **机理分析半途而止**：PickCube 单目深度歧义已定位到根因，但未做归因量化与解决方案；
- **没有可复现的结论性产出**：缺少报告、模型卡、数据卡、可复现脚本。

这四条正是"项目太基础"的实际含义。本周起**不再新增链路**，而是把现有资产收敛成一条有研究形状的主线。

### 1.2 本周目标

把「真机语言条件多任务操作」做成一条可量化的研究闭环，各部件各司其职、互为因果：

```text
VLM 数据引擎（降低标注成本） → VLA 适配与泛化（提供泛化能力） → 世界模型离线评估（降低真机试错）
                     深度感知（必配子问题：抬高物理成功率上限）
```

**本周不追求"再接入一个模型"，而追求"一个变量 + 一张对照表 + 一个归因结论"。**

### 1.3 规划原文引用

- 第 17 周：PickCube 视觉 ACT 确认结构性瓶颈为**单目深度模糊导致闭合高度错误**，成功率上限 ~45–53%（`docs/week17/pickcube_diagnosis.md` §8）
- 第 18 周：SmolVLA 探针"推迟到真机阶段（第 32 周前后）"
- 第 21 周：小型 dynamics + CEM 规划 0/10（scripted 100%），归因=CEM 利用 OOD 误差产生"幻觉最优"
- 第 22 周：冻结选型"主 = 状态 ACT + scripted 兜底，视觉 ACT 为真机升级候选"
- 第 30/31 周：真机 20 轮正式评测与定向补采挂起
- 第 37 周：SmolVLA 框架先行（S12 框架），**S12 真机闭环未做**，`run_vla_infer.py` 真实模式仍为 stub
- 第 36 周：真机感知为 VLM 零样本问答（无坐标/无 grounding）、verifier 为关节阈值、recovery 为硬编码失败码表

### 1.4 前置依赖

| 类别 | 内容 |
|---|---|
| 代码 | week35 技能框架（`real_*`）、week36 视觉/路由/评测（S1–S11）、week37 VLA 框架（`vla_prepost.py`/`run_vla_*.py`）、`world_model.py` |
| 数据 | `demo_v23`（真机 68 集，单任务）、`runs/maniskill_lerobot/week15-cam`（仿真 v3，含图像）、`runs/maniskill_act/week17-*`（checkpoint 与评测） |
| 硬件 | SO-101 follower(COM6)/leader(COM5)、`fixed`/`handeye` 双相机（**索引随 Windows 枚举漂移**：2026-10-10 为 `fixed=2/handeye=0`，**2026-10-17 重插+重启后核图为 `fixed=1/handeye=2`**；唯一事实来源已收敛到 `src/camera_slots.py`，上机前必须核图，见 §13 #40/#82）、稳定工作台 |
| 算力 | 阿里云 A10（SmolVLA 微调）、本机 CPU（仿真与纯逻辑） |
| 环境 | 真机 conda 环境 `lerobot`（Python 3.10.18 + lerobot 0.3.4）；仿真独立环境（Python 3.12 + lerobot 0.6.1） |

---

## 2. 深度化判定标准（贯穿全文，作为准入闸门）

任何新增内容必须同时满足以下五条，否则不做：

1. **有一个明确的、可证伪的研究问题**（不是"把 X 接进来"）
2. **有受控变量**（同数据、同 seed、同评测协议）
3. **有一张对照表**（≥2 个可比条件）
4. **有机理归因**（回答"为什么是这个结果"，而非只报数字）
5. **有可复现产出**（脚本 + 配置 + 报告 + 固定随机种子）

> 反例（明确禁止）：`加载 smolvla_base → 喂 demo 数据 → 报成功率`。这是集成，不是深度。

---

## 3. 总体架构

```text
                         ┌───────────────────────────────────────┐
  真机 / 仿真原始数据 ──► │ P1  VLM 数据引擎                       │──► 语言条件多任务数据集
  (demo_v23 / week15)    │ vlm_labeler.py + label_audit.py        │    (带语言标注 / 成功标签 / 子任务切片)
                         └───────────────────────────────────────┘
                                          │
                                          ▼
                         ┌───────────────────────────────────────┐
  smolvla_base ────────► │ P2  VLA 适配与泛化                     │──► 适配对照表 + 泛化协议报告
  (0.5B, 预训练于 SO-101)│ vla_adapt.py + vla_generalization.py   │
                         └───────────────────────────────────────┘
                                          │
                                          ▼
                         ┌───────────────────────────────────────┐
  world_model.py ──────► │ P3  世界模型离线策略评估 (OPE)          │──► 预测成功率 vs 仿真实测（Spearman ρ）
                         │ wm_ope.py                             │
                         └───────────────────────────────────────┘
                                          │
                         ┌────────────────┴──────────────┐
                         ▼                               ▼
              P3 失败样本回流 ──────────► P1 自动补采（闭环）
                         │
  P0 几何/负载特征（必配）│ real_geometry.py + Present_Load 读取
  ───────────────────────┴──► 作为策略额外输入通道，抬高 P2/P3 可达成功率上限
```

### 3.1 模块职责与交互

| 模块 | 职责 | 复用已有 | 变更 |
|---|---|---|---|
| P1 `vlm_labeler` | 调 VLM 自动产出任务文本、子任务切片；成功判定以传感器为准；纯逻辑可 Mock | `llm_planner` 的多模态 backend 接口、`real_perception` 的注入式 backend | 新增 |
| P1 `label_audit` | 与人工标注比对：一致率 κ、混淆矩阵、分歧样本导出 | `evaluation.py` 的指标聚合风格 | 新增 |
| P1 意图 taxonomy | 定义受限意图集合 schema，使自由文本可计算 κ | — | 新增 |
| P0 `real_geometry` | 已知尺寸反解深度（主）；双视角三角化与标定（条件启用） | `se3.py`（SE(3) 语义） | 新增 |
| P0 负载信号 | 读 STS3215 `Present_Load`（寄存器 60）作接触/夹持信号 | `real_executor` 引擎、`real_observation` | 改造 |
| P0 2D 定位 | 提供目标像素级 bbox/中心点（三角化与已知尺寸反解的输入） | `real_perception.py` | 改造 |
| P2 `vla_adapt` | 适配策略配置与训练接入（head / full；LoRA 延后），产出对照表 | `run_vla_train.py`、`vla_prepost.py` | 新增 |
| P2 `vla_generalization` | 组合泛化 split 生成与 held-out 评测、置信区间 | `evaluation.py`、`stage_selection.py` | 新增 |
| P3 `wm_ope` | 用 world model rollout 候选策略 → 预测成功率 → 与仿真实测做排序相关性 | `world_model.py`、`run_world_model.py` | 新增 |

### 3.2 关键设计决策

1. **VLM 只做"语言侧"，成功判定以传感器为准。** VLM 用于自动生成任务文本与子任务切片；成功率以夹爪位置/负载与抬升校验为准（复用 `real_verifier`）。避免把 VLM 幻觉引入监督标签。
   *标签适用边界*：现有 `is_holding` + `is_lifted` 只能判"夹住并抬起"，**判不了"放置位置是否正确"**。故成功标签对 pick 类可靠，place 类需补区域/视觉规则；报告须写明适用范围。
2. **P0 采用"几何作为策略输入"路线（R2），不做显式笛卡尔覆盖。** 真机策略是**图像 → 关节**的端到端映射；把几何/负载量作为**额外输入通道**即可，避免笛卡尔→关节的转换，从而**不需要 FK/IK**（见 §3.3 阻塞点 B）。
3. **接触信号直接读负载，不用位置差间接推断。** Feetech STS3215 寄存器表含只读 `Present_Load`（60）与 `Present_Current`（69），lerobot 内已有 `sync_read("Present_Current")` 用例；需在 SO-101 驱动侧扩一个读取方法，并先做量程实测。
4. **WM 不做策略，只做评估。** 第 21 周已证伪"WM 直接规划"，本周把 WM 的研究问题改为 Offline Policy Evaluation（OPE）。候选集合须**跨行为族**，否则成功率差异被噪声淹没、ρ 无统计意义；口径限定为**仿真 PushCube 策略族**（`world_model.py` 仅 18 维状态空间，无法 rollout 图像输入的真机策略），报告须写明外推边界。
5. **适配是研究对象，不是流程步骤。** P2 把适配策略与数据量当作自变量，把成功率 / 原能力保留 / 样本效率 / 训练成本当作因变量；首轮裁剪为 `zero_shot / head / full` × 2 数据量（LoRA 无现成入口，延后）。
6. **全部可离线完成。** 本设计不要求真机在线 RL，所有实验可在仿真或离线数据上完成，真机只用于最终验证，风险可控。

### 3.3 前置阻塞点与解除状态

| # | 阻塞点 | 状态 | 解除方式 |
|---|---|---|---|
| A | 感知只有 `region: left`，**无像素级坐标**——三角化与已知尺寸反解都缺输入 | **部分解除** | 像素级 2D 定位接口与坐标约定换算已落地；**定位精度实测待真机帧**（合成图冒烟见 §13 #26） |
| B | **项目没有 SO-101 的运动学模型**（`kinematics.py` 为平面 2 连杆教学模型，真机侧无 FK/IK），AX=XB 手眼标定与腕部相机三角化都需末端位姿 | **已绕过** | P0 走 R2 路线 + 采用免 FK 的 B1/B2；双视角三角化（B3）降为条件任务，前置 FK 建模或 marker 方案 |
| C | B3 原计划用位置差间接推断接触，量程差小时会失效 | **已解除（低成本）** | 直接读 STS3215 `Present_Load`（寄存器 60），先做量程探针 |
| D | 多任务语言数据不存在（`demo_v23` 为单物体单任务） | **待补** | 阶段 2 的采集任务（≥3 物体 × 2 指令 × 30 集） |
| E | 任务文本为自由文本，**无法直接计算 Cohen's κ** | **待补** | 前置子任务：定义意图 taxonomy（受限意图集合 + 参数槽） |
| F | `run_vla_infer.py` 真机推理未打通（第 37 周 stub） | **待补** | P2 门槛任务：先完成真机闭环，或首轮全仿真评测 |
| G | SmolVLA 的 LoRA / 按模块冻结在 lerobot 0.3.4 无现成入口 | **已知** | 首轮裁掉 LoRA，只做 `zero_shot / head / full` |

---

## 4. 数据结构

```python
# P1 —— 意图 taxonomy 与标注
@dataclass
class IntentTaxonomy:
    intents: list[str]                       # 受限意图集合：pick | place | sort | stack
    slots: dict[str, list[str]]              # 参数槽：object / target_zone
    version: str

@dataclass
class LabelRecord:
    episode_ref: str
    frame_range: tuple[int, int] | None      # 子任务切片；None=整段
    task_text: str                           # 自动生成/规范化的语言指令
    intent: str | None                       # 映射到 taxonomy 后的离散标签（供 κ 计算）
    subtask: str | None                      # approach | align | grasp | lift | place
    success: bool | None                     # 传感器判定优先；VLM 仅作兜底
    confidence: float
    source: str                              # "vlm" | "human" | "sensor"

@dataclass
class LabelAuditReport:
    n_total: int
    kappa_intent: float                      # 离散意图一致率（Cohen's kappa）
    kappa_success: float
    confusion: list[list[int]]
    disagreements: list[dict]
    per_subtask_recall: dict[str, float]
    label_scope: str                         # 标签适用边界说明，如 "grasp_lift_only"

# P0 —— 几何与负载
@dataclass
class DepthEstimate:
    z_m: float
    method: str                              # "known_size" | "stereo" | "sensor"
    uncertainty_m: float

@dataclass
class HandEyeCalibration:                    # 条件启用（B3 前置）
    R: np.ndarray
    t: np.ndarray
    residual_m: float
    n_samples: int
    method: str                              # "ax_xb" | "board" | "aruco"

@dataclass
class LoadProbeResult:                       # 负载量程探针
    empty_load: float
    loaded_load: float
    resolvable: bool                         # 能否分辨目标物体
    note: str

@dataclass
class ContactSignal:
    is_contact: bool
    load: float
    source: str                              # "present_load" | "position_delta"

# P2 —— VLA 适配与泛化
@dataclass
class AdaptConfig:
    strategy: str                            # 首轮："zero_shot" | "head" | "full"（"lora" 延后）
    trainable_ratio: float
    n_episodes: int
    steps: int
    seed: int

@dataclass
class AdaptResult:
    config: AdaptConfig
    success_rate: float
    retention_rate: float                    # 微调后在基线任务的成功率 / 微调前
    train_seconds: float
    peak_mem_mb: float

@dataclass
class GeneralizationSplit:
    train_combos: list[tuple[str, str]]      # (object, instruction) 组合
    heldout_combos: list[tuple[str, str]]
    protocol: str                            # "combo" | "paraphrase" | "novel_object"

# P3 —— 世界模型离线评估
@dataclass
class OPEReport:
    candidates: list[str]
    candidate_family: dict[str, str]         # 候选 → 行为族（scripted/bc/act/...）
    real_success: list[float]                # 仿真实测成功率
    wm_pred_success: list[float]
    real_success_var: list[float]            # 每候选方差（ρ 可信度前提）
    spearman_rho: float
    ci_low: float
    ci_high: float
    n_rollouts_per_candidate: int
    scope_note: str                          # 外推边界说明（仅仿真 PushCube 策略族）
```

状态管理：
- 标注产物写入 `runs/week38/labels/<dataset>/<episode>.jsonl`，人工复核结果写入 `.../human/`；
- 几何/负载探针产物入 `runs/week38/geometry/` 与 `runs/week38/load_probe.json`；
- 适配实验产出入 `runs/week38/vla_adapt/<job>/`（config.json、metrics.jsonl、adapt_result.json）；
- OPE 结果入 `runs/week38/wm_ope/<job>/report.json` + `report.md`；
- 所有产物沿用已忽略的 `runs/` 约定，不提交。

---

## 5. 接口设计

### 5.1 核心函数签名

```python
# P1 vlm_labeler.py
def define_intent_taxonomy() -> IntentTaxonomy: ...
def map_instruction_to_intent(text: str, taxonomy: IntentTaxonomy) -> str | None: ...
def label_episode(backend, frames: list[str], sensor_state: dict,
                  taxonomy: IntentTaxonomy, task_hint: str | None = None) -> LabelRecord: ...
def label_dataset(backend, dataset_root: str, taxonomy: IntentTaxonomy,
                  limit: int | None = None) -> list[LabelRecord]: ...

# P1 label_audit.py
def compare_labels(vlm: list[LabelRecord], human: list[LabelRecord]) -> LabelAuditReport: ...
def cohen_kappa(a: list, b: list) -> float: ...

# P0 real_geometry.py
def depth_from_known_size(bbox_px: tuple[float, float], real_size_m: float,
                          fx: float) -> DepthEstimate: ...          # 主路径，免 FK
def triangulate(p1: np.ndarray, p2: np.ndarray, K1: np.ndarray, K2: np.ndarray,
                T_12: np.ndarray) -> DepthEstimate: ...             # 条件启用
def solve_handeye(pairs: list[tuple[np.ndarray, np.ndarray]],
                  method: str = "ax_xb") -> HandEyeCalibration: ... # 条件启用

# P0 负载信号（扩展真机引擎）
def read_load() -> dict[str, float]: ...                            # 各关节 Present_Load
def probe_load_range(...) -> LoadProbeResult: ...
def detect_contact(load: float, baseline: float, margin: float) -> ContactSignal: ...

# P2 vla_adapt.py
def build_adapt_config(strategy: str, n_episodes: int, steps: int, seed: int) -> AdaptConfig: ...
def apply_adaptation(policy, config: AdaptConfig) -> dict: ...
def run_adapt_experiment(config: AdaptConfig, dataset_root: str,
                         eval_seeds: list[int]) -> AdaptResult: ...

# P2 vla_generalization.py
def build_split(dataset_root: str, protocol: str, heldout_ratio: float,
                seed: int) -> GeneralizationSplit: ...
def evaluate_generalization(policy, split: GeneralizationSplit, n_episodes: int) -> dict: ...

# P3 wm_ope.py
def rollout_policy_in_world_model(policy, wm, init_states: list[np.ndarray],
                                  horizon: int) -> list[float]: ...
def evaluate_ope(candidates: dict[str, list[float]], candidate_family: dict[str, str],
                 real_success: list[float]) -> OPEReport: ...
```

### 5.2 CLI

```bash
# P0 负载量程探针（新增，先跑）
python src/run_load_probe.py --port COM6 --repeats 5

# P0 几何（条件启用）
python src/run_handeye_calib.py --pairs runs/week38/calib/pairs.npz --method ax_xb

# P1 自动标注 + 审计
python src/run_vlm_label.py --dataset-root <demo_v23> --out runs/week38/labels/demo_v23
python src/run_label_audit.py --vlm runs/week38/labels/demo_v23 --human runs/week38/labels/human

# P2 适配对照（云上 A10）
python src/run_vla_adapt.py --strategy head --n-episodes 30 --steps 2000 \
    --dataset-repo <multi_task> --output-dir <远程> --device cuda
python src/run_vla_generalization.py --model-dir <ckpt> --protocol combo --heldout-ratio 0.3

# P3 世界模型 OPE（仿真口径）
python src/run_wm_ope.py --wm runs/week38/wm/pushcube --candidates <ckpt_a>,<ckpt_b> \
    --real runs/week38/sim_eval.json
```

### 5.3 参数说明（关键项）

| 参数 | 含义 | 取值 |
|---|---|---|
| `--strategy` | 适配策略 | 首轮 `zero_shot` / `head` / `full`（`lora` 延后） |
| `--n-episodes` | 微调数据量（自变量） | 首轮 30 / 60（10 作为可选） |
| `--protocol` | 泛化协议 | `combo` / `paraphrase` / `novel_object` |
| `--heldout-ratio` | held-out 组合比例 | 0.2–0.4 |
| `--horizon` | WM rollout 步数 | 与任务平均步数同量级 |
| `--device` | 训练设备 | `cpu` / `cuda`（A10） |

---

## 6. 数据流

### 6.1 P1 数据引擎流

```text
意图 taxonomy（受限意图集合 + 参数槽）
   │
   ▼
原始 episode（demo_v23 / week15-cam）→ 逐帧 RGB + 关节状态
   │
   ▼
vlm_labeler.label_episode
   │  ├─ VLM backend → 任务文本 / 子任务切片 / 置信度
   │  └─ map_instruction_to_intent → 离散 intent（供 κ 计算）
   │  └─ sensor_state → 成功判定（夹爪位置/负载 + 抬升校验）
   ▼
LabelRecord（jsonl）→ label_audit.compare_labels（对比人工抽检 20%）
   │
   ▼
LabelAuditReport（κ / 混淆矩阵 / 分歧样本 / 标签适用边界）
   │
   ▼
语言条件多任务数据集（LeRobotDataset v3，fields 不变，仅补 task/intent/subtask 元数据）
```

### 6.2 P0 几何/负载流（R2 路线：特征注入，不做显式覆盖）

```text
分支一（负载探针，先行）：
  读 Present_Load → empty/loaded 量程实测 → LoadProbeResult（决定 B2 是否成立）

分支二（几何特征，免 FK）：
  2D 定位（选型后的检测源）→ 目标 bbox → depth_from_known_size → DepthEstimate(z_m, σ)
  条件启用：固定相机同时观测夹爪 marker + 目标 → 相机系内相对几何

汇合：
  几何/负载量 → 策略额外输入通道（R2）→ 策略自行学习映射
  │
  ▼
对照评测（累积式输入变体）：
  B0 仅 RGB → B1 +已知尺寸反解 → B2 +负载接触 → B3 +双视角三角化（条件）
  输出：成功率 + 闭合高度误差 + 失败归因
```

### 6.3 P2 适配与泛化流

```text
多任务数据集 ──► vla_adapt.build_adapt_config(strategy, n_episodes, seed)
   │
   ▼  同一 checkpoint 基线，仅改一个自变量
run_vla_train（云端）→ checkpoint → run_vla_infer / 仿真评测
   │
   ▼
AdaptResult（成功率 / 保留率 / 训练成本）→ 汇总成对照表
   │
   ▼
vla_generalization.build_split(protocol) → held-out 组合评测 → 泛化报告（含 CI）
```

### 6.4 P3 OPE 流（仿真口径）

```text
候选策略集合（跨行为族：scripted / BC / ACT / 状态 vs 视觉 / 扰动版）
   │
   ├─► 世界模型内 rollout（world_model.DynamicsModel + 初始状态分布）→ wm_pred_success
   └─► 仿真 PushCube 实测（stage_selection / act_stability）→ real_success
   │
   ▼
wm_ope.evaluate_ope → OPEReport（Spearman ρ + 置信区间 + 候选方差 + 外推边界）
   │
   ├─ ρ ≥ 阈值 → 产出"可省实测试错的选型工具"
   └─ ρ < 阈值 → 产出归因报告（主归因：WM 训练分布与候选访问分布的 shift）
```

---

## 7. 实验设计（研究核心）

> **执行类型标注**（与 `plan.md` §0 一致）：
> **【代码】** 纯离线开发；**【数据】** 需采集数据/人工标注复核；**【真机】** 需真机操作；**【云端】** 需云 GPU 训练。
> 每个实验小节末尾的"执行类型"给出该实验实际所需的资源组合。

### 7.1 P0 感知消融（累积式输入变体）

| 组 | 策略输入 | 新增变量 | 是否需 FK |
|---|---|---|---|
| B0 | 仅 RGB | 基线 | 否 |
| B1 | + 已知尺寸反解深度 | +先验尺寸几何 | 否 |
| B2 | + 负载/接触特征（`Present_Load`） | +力/接触信号 | 否 |
| B3 | + 双视角三角化深度 | +多视角几何 | 是（条件任务） |

指标：抓取成功率、闭合高度误差 (mm)、失败归因占比。

**执行类型**：【代码】+【云端】（训练与评测）；**B2 额外需【真机】**负载探针，**B3 额外需【真机】**标定采图。

### 7.2 P1 标注可靠性

| 组 | 任务文本 | 成功判定 | 训练数据 |
|---|---|---|---|
| L0 | 人工 | 人工 | 人工标注集 |
| L1 | VLM | 传感器 | VLM 标注集 |
| L2 | 人工 | VLM | 混合集 |

指标：κ（离散 intent / 成功判定）、下游 VLA 成功率、标注吞吐（集/小时）。

**执行类型**：L0 / L2 依赖 **【数据】**（人工标注 + 抽检复核）；L1 为【代码】+ VLM API 调用。

### 7.3 P2 适配消融（首轮 3 策略 × 2 数据量，封顶 6 单元）

| 策略 | 可训参数 | 数据量 | 备注 |
|---|---|---|---|
| zero_shot | 0 | 0 | 基线锚点 |
| head | action head | 30 / 60 | 是否解冻 vision |
| full | 100% | 30 / 60 | lr 分组 |
| ~~lora~~ | — | — | lerobot 0.3.4 无现成入口，延后 |

指标：仿真/真机成功率、原任务保留率（遗忘）、样本效率曲线、训练成本。

**执行类型**：【代码】+**【云端】**（6 个训练单元）；输入数据来自 **【数据】**多任务采集。

### 7.4 P2 组合泛化协议

- `combo`：训练见 (物体 A, 指令 1)、(物体 B, 指令 2)，测试 (A, 2)、(B, 1)
- `paraphrase`：同义改写指令，测试改写鲁棒性
- `novel_object`：零样本新物体

指标：held-out 成功率 + 95% 置信区间、相对训练组合的泛化 gap。

**执行类型**：【代码】+【云端】；前提是 **【数据】**多任务集就绪（否则降级为仿真协议验证）。

### 7.5 P3 OPE 排序相关性

- 候选 ≥5 个，**必须跨行为族**（scripted / BC / ACT / 状态 vs 视觉 / 扰动版），含 scripted 作锚点
- 每候选 ≥10 次 rollout（WM 内）/ 仿真实测
- 同时报告每候选成功率方差，作为 ρ 可信度前提
- 指标：Spearman ρ + bootstrap 置信区间；ρ ≥ 0.6 视为工具可用

**执行类型**：**【代码】**（仿真 PushCube 口径，本机可完成，无数据采集需求）。

---

## 8. 指标定义

| 指标 | 计算 | 用途 |
|---|---|---|
| κ | Cohen's kappa（离散意图/成功判定 vs 人工） | P1 标注质量 |
| ΔSucc | 自动标注训练 − 人工标注训练 | P1 下游代价 |
| e_loc | 已知尺寸反解/三角化位置与真值距离（cm） | P0 几何精度 |
| ΔGrasp | 消融组 − 基线成功率（百分点） | P0 归因 |
| LoadRes | 空载与夹持负载差 / 量程 | P0 接触信号可用性 |
| Retention | 微调后在基线任务成功率 / 微调前 | P2 遗忘 |
| SampleEff | 达 70% 成功率所需 episode 数 | P2 样本效率 |
| ρ / Var | Spearman(WM 预测, 仿真实测) 与候选方差 | P3 工具可用性 |

---

## 9. 验收标准

| 编号 | 验收项 | 阈值 |
|---|---|---|
| A1 | P0 几何链路定位误差（工作区 30–60 cm） | ≤ 1.0 cm（标定残差 5–8 mm 亦可进入消融，去留由消融结果决定） |
| A2 | P0 抓取成功率相对 B0 提升 | ≥ 15 个百分点 |
| A3 | P1 离散意图标注一致率 κ | ≥ 0.6 |
| A4 | P1 用 VLM 标注训练相对人工标注下降 | ≤ 10 个百分点 |
| A5 | P2 适配对照表（首轮） | ≥ 3 策略 × 2 数据量（LoRA 为后续） |
| A6 | P2 最优策略相对 zero_shot 提升 | ≥ 20 个百分点（以 zero_shot 实测基线校准） |
| A7 | P2 held-out 组合成功率 | ≥ 40%（或降级为仿真协议验证） |
| A8 | P3 Spearman ρ | ≥ 0.6；若 < 0.6 则提交归因报告（负结果亦算通过） |
| A9 | 前置产物 | 意图 taxonomy 定义完成；负载量程探针完成 |
| A10 | 全部实验可复现 | 固定 seed、脚本、配置、报告齐备 |

---

## 10. 风险与降级

| 风险 | 早期信号 | 应对 | 降级 |
|---|---|---|---|
| 2D 定位精度不足 | 检测 bbox 抖动大 | 实测 VLM bbox，必要时引入 GroundingDINO 类检测器 | 退回"目标区域级"定位，P0 只做 B2（负载） |
| 负载量程不足 | 空载/夹持负载差接近噪声 | 提高采样均值、改变夹持力 | B2 降级为"位置差 + 负载"联合判据 |
| VLM 标注噪声大 | κ < 0.5 | 只用于语言标注与粗筛，成功判定以传感器为准；人工抽检 20% | 退回全人工标注，P1 只做"标注成本对比" |
| 多任务真机数据不足 | 采集周期长 | 先用 ManiSkill + week15 数据在仿真验证协议 | P1/P2 在仿真完成，真机只做最终验证 |
| FK 建模成本高 | 0 号任务评估超预期 | 走 R2 + B1/B2 免 FK 路径 | B3 三角化整组取消，不影响 B0–B2 |
| WM OPE 相关性低 | ρ < 0.3 | 分析 WM 训练分布与候选访问分布的 shift | 转为负结果 + 归因报告（仍有产出） |
| LoRA 无训练入口 | 实现量超预期 | 首轮不做 LoRA | 只交 `zero_shot / head / full` 对照表 |
| GPU/联网受限 | smolvla_base 拉不下 | 先做 P0/P1/P3 无 GPU 部分 | P2 延后，不阻塞其他支柱 |
| 环境版本分裂 | 真机 0.3.4 / 仿真 0.6.1 混用 | 明确各实验所属环境并写入报告 | 真机实验只在 `lerobot` 环境执行 |

---

## 11. 交付物

| 交付物 | 类型 | 说明 |
|---|---|---|
| P0 / P1 / P2 / P3 对照与结论报告（`runs/week38/` 下 report.json + report.md） | 【代码】 | 脚本生成；首轮以 **P0 + P3** 两份为保底交付 |
| 数据集卡片（多任务 + 语言标注 + 失败样本） | **【数据】** | 依赖多任务采集与标注产物 |
| 模型卡片（适配策略对照） | 【代码】+【云端】 | 依赖适配实验结果汇总 |
| 3 分钟演示视频（问题 → 方案 → 结果） | **【数据】**+【真机】 | 需真机录制 |
| 可复现脚本链：`run_load_probe` → `run_vlm_label` → `run_label_audit` → `run_vla_adapt` → `run_vla_generalization` → `run_wm_ope` | 【代码】 | 其中 `run_load_probe` 运行时需【真机】 |
| workshop paper 骨架："单目操作中的深度歧义：几何与力反馈的赋能" | 【代码】 | 前提是 P0 + P3 结论完整 |

---

## 12. 与"集成"的区别（自查清单）

| 集成式做法（面试会被否） | 本设计的做法 |
|---|---|
| 用 VLM 描述场景 + LLM 规划 | 把 VLM 标注可靠性当自变量，量化 κ 与下游代价 |
| 加载 smolvla_base 微调，报成功率 | 适配策略 × 数据量对照，量化遗忘与样本效率 |
| 用 VLM 做 verifier（调 API 看输出） | 传感器判定优先并写明标签适用边界，VLM 仅兜底 |
| 用位置差猜夹持 | 直接读 `Present_Load` 作接触信号，并先做量程探针 |
| 用世界模型直接规划 | 世界模型做 OPE，量化排序相关性（ρ）与候选方差 |
| 再加一条新链路 | 只做收敛，不新增链路 |

---

## 13. 实际偏差

### 2026-10-10 · 批次 1（阶段 0 部分 + 阶段 1 部分）

1. **`DepthEstimate` 增加 `point_m` 字段**（§4 原为 3 字段）：三角化天然产出相机系三维点，只保留 `z_m` 会丢失调用方需要的信息；`depth_from_known_size` 时该字段为 `None`。
2. **`HandEyeCalibration` 增加构造期校验**（R 正交且行列式为 +1、`n_samples ≥ 2`、`method` 枚举），与 `se3.RigidTransform` 的校验风格一致。
3. **手眼求解改用四元数零空间法**（原计划未指定算法）：`q` 与 `-q` 表示同一旋转，天然规避旋转矩阵取负的符号歧义；退化判定改为"零空间维数是否恰为 1"（可捕获全平移/共面），同时允许真实数据的非零最小特征值走最小二乘。
4. **暂未实现 §5.1 的 1.3 特征构造函数**：特征维度须与 `visual_act.py` 的额外观测通道一致，留待 1.5 一并定义，避免先定契约再返工。
5. **`LoadProbeResult` 字段多于 §4 设计**（增加 `noise` / `threshold` / `n_empty` / `n_loaded`）：探针结论需要可复现的判据依据，而非只给一个布尔值。
6. **新增 `run_load_probe.py --dry-run`**（设计未列）：与仓库既有 runner 的 dry-run 约定一致，可在无机械臂时验证链路。
7. **`MockEngine` 增加 `gripper_load` 构造参数**：供 dry-run 与单测模拟空载/夹持两态。
8. **`real_observation.read_load` 吞掉引擎异常并返回空 dict**：与 §3.2 的降级原则一致（"读不到负载"≠"没有接触"）；探针 CLI 则直接调用引擎的 `read_load` 以便暴露真实读取问题。
9. **`run_load_probe.py` 改为自行下发夹爪闭合指令**（原实现为只读）：lerobot 的 `disable_torque_on_disconnect` 默认 `True`，任何控制进程退出都会松力矩、夹爪回弹，因此"先用其他工具摆好状态、再跑探针"在时序上不可靠；同时该流程与"COM6 单进程独占"冲突。新增 `--no-command` 供已自行保证力矩/位姿的场景退回只读。
10. **探针同时记录夹爪位置**：CP0 的降级判据是"位置差 + 负载"联合判据，一次采集拿齐两项证据，避免二次出机。
11. **`connect(calibrate=False)` + `is_calibrated` 校验**：避免标定文件缺失时 `connect()` 进入交互式 `input()` 造成阻塞；改为明确报错并给出 `lerobot-calibrate` 命令。
12. **`--close-pos` 强制大于 0**：0 会顶住机械限位造成堵转（week24 有夹爪过载先例），默认 3.0（依据 week24 标定"闭≈0–10"）。
13. **`Present_Load` 原始读数含方向位（bit10），必须先解码再比较**：真机实测空载闭合读数为 `1079 = 0x437`（方向位=1、幅值 55 → 实际 **−55‰**）。原始值直接做差会在负载换向时虚增 1024，把"换向"误判成"巨大负载差异"。新增 `decode_feetech_load`，`RealEngine.read_load` 与探针均改为解码后输出；`LOAD_MARGIN` 由 3.0 修正为 **50.0**（解码单位为千分比，1000 = 额定扭矩）。
14. **实测确认空载闭合被机械限位阻挡（非执行不到位）**：指令 `--close-pos 3.0`，实际停在 `gripper.pos = 3.77`，5 次采样完全一致（稳态），负载 −55‰ → **3.77 即该夹爪的闭合极限**，`close-pos` 不可再往小调（只会加大顶死推力、逼近过载）。探针结论段新增"空载是否被限位挡住 / 夹持是否真的挡住钳口"的位置判定说明。
15. **探针连接后显式先张开再闭合**（设计未列）：连接时舵机会驱动到上次残留的 `Goal_Position`，若中途跑过推理/遥操作脚本，该值可能不是张开位，可能撞飞已放好的物体。改为"张开到 release_pos → 闭合到 close_pos"，使时序与残留状态无关。
16. **CP0 实测通过（B2 可执行）**：空载 −55‰ / 夹持 −206‰ / Δ=−151‰（阈值 50‰，**3 倍余量**），夹爪位置 3.8 → 6.8（Δ=+3.0），5 次采样噪声为 0。
17. **由 #16 发现基准不一致（待决策，未改代码）**：实测空载闭合位置为 **3.8**，而 `empty_grip_baseline` 在 `real_executor.py` / `run_real_agent.py` / `run_real_eval.py` 中默认 **1.5**。按实测基准校准后，判据 `pos > 3.8 + 3.0` 对夹持值 6.8 为假 → **位置判据会漏判**；当前位置差恰好等于 `margin=3.0`，处于临界。这正面支持 §3.2 决策 3 引入 `Present_Load` 作为接触信号（负载余量为位置的数十倍）。是否重标基准与 margin 需单独决策，因其会改变真机夹持判定行为。
18. **`IntentTaxonomy.slots` 由 `dict[str, list[str]]` 改为 `tuple[str, ...]`**：§4 用"槽名 → 允许取值"的闭集表示，但物体是**开放词表**（新物体随时出现），闭集值域会把 taxonomy 写死；改为只声明槽名、值域开放。本批亦不实现槽抽取（κ 只需要 intent）。
19. **`IntentTaxonomy` 增加 `to_dict` / `from_dict` 与构造期校验**（设计未列）：taxonomy 必须与标注产物一起落盘才能复现 κ（taxonomy 变了历史 κ 不可比）。
20. **指令→意图映射采用确定性关键词规则，而非再调一次 LLM**（设计未指定）：κ 衡量的是"VLM 标注 vs 人工标注"在同一 taxonomy 上的分歧，若映射本身是 LLM，分歧无法归因。已知局限（不处理否定、多动作只取最高优先级）写入 docstring。
21. **2D 定位新增独立数据结构与可插拔接口**（§4/§5 未细化）：`Detection`（bbox 校验 + `center_px` / `size_px`）、`LocalizationResult`（含 `skipped`）、`ObjectLocator` 协议、`VLMObjectLocator`、`FakeObjectLocator`。`ObjectLocator` 协议即为 0.3"选型"的插入点——专用检测器实现同一 Protocol 即可同口径对比。
22. **`describe_workspace` / `local_holding_state` 保持不变**（新增而非替换）:week36 已验收链路依赖它们，2D 定位以并存方式引入，避免破坏既有行为。
23. **定位解析对非法候选采取"丢弃并计数"而非整体失败**:单个坏 bbox 不应让整次观测失败；越界候选在给出 `frame_size` 时丢弃并计入 `skipped`，便于事后评估定位源质量。
24. **`locate_objects` 增加 `coord_space` 参数与自动判定**（设计未列）：**模型的坐标约定是"先验"，prompt 约束不可靠**——Qwen-VL 系列官方即使用 `[0, 1000)` 归一化 bbox（Qwen-VL 论文 §2.2），实测在明确要求"像素为单位"的情况下 `qwen3-vl-plus` 仍返回 0–1000，若按像素解读会被当成越界丢弃，从而**误判为"模型不准"**。新增 `auto` / `pixel` / `normalized_1000` / `normalized_1`；`auto` 的判定依据是"像素坐标不可能越出**对应轴**尺寸"（越界只可能是归一化），并在 `LocalizationResult.coord_space` 记录实际约定供审计。已知局限：归一化坐标恰好落在图像尺寸内时 `auto` 会误判为像素，故**已知模型约定应显式指定**。
25. **新增 `_extract_json_object` 容错解析**（设计未列）：视觉模型普遍用 ` ```json ` 围栏包裹回复或在前后附加解释文字，严格 `json.loads` 会把**有效结果**判为解析失败（实测 `qwen3-vl-plus` 即返回围栏内容）。改为"直接解析失败 → 截取首个配对完整的 `{...}`"，扫描时正确跳过字符串内的花括号。
26. **2D 定位精度冒烟（合成图，仅作方向性参考，不构成选型结论）**：640×480 高对比合成图、真值 bbox=(200,150,280,250)，中心误差 —— DeepSeek v4-flash-vision **87.7 px**（像素约定）；`qwen3-vl-plus` **2.7 px**（0–1000 归一化，`auto` 正确判定）；`gui-plus` **2.2 px**（像素约定）。**均为单样本、无光照/遮挡/背景干扰**；且 `gui-plus` 为 GUI 定位专用模型、测试图形似 UI 元素，成绩很可能不可外推。真实相机帧（细长低对比螺丝刀 + 真实光照）上的精度未知，**必须在真机帧上复测后方可选型**。
27. **评测脚本不使用 `coord_space="auto"`，改为按源显式钉死**：真实帧上 `gui-plus` 在不同帧被 `auto` 分别判成 `normalized_1000` 与 `pixel`——它实际用像素，那几帧是**越界像素框触发误判**，垃圾框被"救"成看似合理的归一化框（**静默错误**）。故 `run_perception_locate_eval.py` 用 `SOURCE_COORD_SPACE` 显式固定约定；`auto` 降级为"发现/诊断"用途，不用于产出结论。
28. **越界框由"整条丢弃"改为"轻微越界裁剪 + 计数"**：真实帧上模型常有边缘过冲（实测 `gui-plus` 给 x2=644 而图宽 640，仅超 4 px），整条丢弃会**低估模型表现、污染选型对比**。新增 `_clamp_to_frame`（越界 >20% 才判非法）与 `LocalizationResult.clamped` 计数，并在 note 中提示。
29. **新增两个 0.3 工具**（设计未列）：`run_perception_capture.py`（真机抓帧，直接用 `cv2.VideoCapture(idx, CAP_MSMF)`，**不连机械臂、不占 COM6**）与 `run_perception_locate_eval.py`（多源同帧评测，出叠加图供人工核对 + `report.json`，支持 `--gt` 与 `--dry-run`）。
30. **`auto` 判定加入 2% 容差，修正 #27 的根因（设计未列）**：#27 之所以不得不"按源显式钉死"，根因是 `auto` 把 `gui-plus` 的 **4 px 边缘过冲**（x2=644 vs 图宽 640，仅 +0.6%）当成"像素不可能越界"的证据，判定为 `normalized_1000`，把垃圾框缩放成看似合理的框。现 `_resolve_coord_space` 改为"越界需**超过 `_AUTO_OVERFLOW_TOLERANCE=2%`** 才算归一化证据"，容差落在真实观测的 **+0.6%（须判像素）** 与 **+9.6%（`qwen3-vl-plus` 的 y2=526 vs 图高 480，须判归一化）** 之间。已知局限（归一化坐标恰好落在图像尺寸内仍会误判为像素）不变，故显式配置仍优先。
31. **定位源配置改为 env 驱动注册表，取代硬编码的 `SOURCE_COORD_SPACE`（设计未列）**：#27 的"钉死"以代码常量形式写死，新增一个归一化输出模型必须改代码。现改为按源读取 `VLM_<源名大写非字母数字转下划线>_API_BASE / _API_KEY / _MODEL / _COORD_SPACE`，逐级回退到全局 `LLM_API_*` → 源名本身 → 内置默认表（`gui-plus=pixel`、`qwen3-vl-plus=normalized_1000`）→ `auto`；`--models` 缺省值取自 `VLM_SOURCES`；叠加图颜色对未知源按名字稳定散列取色。**新增归一化坐标模型只需在 `.env` 加几行，无需改代码**（兼容诉求见 #30/#31 的组合）。
32. **`.env` 主模型由 `deepseek-v4-flash-vision-exp` 改为 `deepseek-flash`**：实测该端点（`https://api.deepseek.com/v1`）仅接受 `deepseek-flash` / `deepseek-v4-pro`，原模型名直接 400；其中仅 `deepseek-flash` 具备视觉能力（`deepseek-v4-pro` 对图片返回"未收到可解析图像"、检出为空）。Qwen 源（`gui-plus` / `qwen3-vl-plus`）指向 `https://maas.qianwenaiapi.com/compatible-mode/v1`，需在 `.env` 填入各自 `VLM_*_API_KEY`（DeepSeek 的 key 在该端点返回 401）。
33. **第二轮真机帧（换位后）两路相机均检出螺丝刀**（`runs/week38/perception_frames/p0_v2`，`deepseek-flash` 单源）：fixed×3 与 handeye×3 全部检出，且 `auto` 均判为 `pixel`（#30 未引入误判）。**但 fixed 视角的框贴画面上边缘且异常宽**（如 fixed_00 `[125,0,595,135]`，占宽 470 px，模型自述"仰视、手柄被上边缘裁切"），与 handeye 视角"躺在桌面上"的表观尺度差数倍——**框质量存疑，须人工核对叠加图**；且单源、无真值，不构成选型结论。
34. **"框贴边（尺寸被裁）"根因取证：排除"API 压缩图片"假设（设计未列）**：用户假设"视觉模型 API 内部压缩图片导致框出界"。取证否定：① `image_to_data_url` 仅在 `max(img.size) > 640` 时缩放，帧正好 640×480 → **不缩放**，仅 JPEG q85 重编码；② 降采样/JPEG 不改变像素位置，无法把内容"推"到画面边缘；③ 早前合成图冒烟（真值 200,150–280,250）误差仅 2.2/2.7 px，若 API 缩小后按缩小空间给坐标会出现 ~20% 成比例缩放误差，实测没有。真因是**真实取景裁切**（像素取证：`p0_v2/fixed_00` 第 0 行 `x≈348–397` 有 46 个前景像素）+ **模型框过度外扩**（同帧报 `x∈[123,590]`，实际内容只到 `x≈584`）。取景问题已由用户调整相机解决（`y1: 0 → 115–129`）。
35. **0.3 选型落地与 `bbox_error` 补齐（#27/#31 的收口）**：新增 `run_perception_gt_label.py`（多源中位数候选 + 人工拖拽确认，**真值必须独立于被评模型**，避免循环论证）产出 `gt.json`；评测脚本 docstring 承诺"中心/尺寸误差"但实现只有中心误差 → 补纯函数 `bbox_error`。选型结果（6 帧）：`qwen3-vl-plus` **1.31 px / 6-6** 选定、`gui-plus` 3.37 px / 6-6 次选、`deepseek-flash` 22.81 px / 3-6（跨运行 0.83→0.50，不稳定）不采用。
36. **0.6 空闭基准标定实测，并暴露"限幅导致单发指令失效"的坑（设计未列）**：新增 `run_grip_baseline_calib.py`（R 次独立连接 × P 姿态 × N 采样；**只测量只建议**）。实测空载闭合 **3.92 ± 0.04**，跨运行/跨姿态极差均 **0.047**、采样噪声 **0**，9/9 次 `close_status=stalled`。结论：① 代码默认 `empty_grip_baseline=1.5` 与实测差 2.4，旧配置名义 `margin=3.0` 而**实际净空只有 0.455**；② 建议 `baseline=4.1` + `margin=1.4`（阈值 5.5，对空载上限净空 1.455≈33σ，仍低于薄件夹持位 6.8）；③ 判据建议「位置 OR 负载」（负载余量 ≈3 倍）。**工程坑**：`max_relative_target=8.0` 会把**单次** `send_action` 的目标裁到"当前位置 ±8"（`goal_pos 3.0 → safe 11.18`），只发一次会让夹爪停在 11.86 并测出完全失真的"空闭位置"——必须像 `RealEngine.move_to` 那样**反复下发直到「到位/卡住」**。**未改动任何真机判定行为**（是否采纳默认值另需决策）。
37. **落地 0.6 结论：重标默认值 + 「位置 OR 负载」判据 + 修正 `detect_contact` 语义（设计未列）**：用户决策"从实际出发"后采纳。`real_observation` 新增 `EMPTY_GRIP_BASELINE=4.1` / `GRIP_POS_MARGIN=1.4` / `EMPTY_LOAD_BASELINE=-69.0` 与 `holds_object()`；`is_holding` 默认值改为标定值；`real_verifier` 的 `verify_grasp_lift` / `verify_holding` 增加可选 `load`；`RealSkillExecutor` 默认 baseline → 4.1 并新增 `_read_gripper_load()`（读不到 → 退化纯位置判据）；`MockEngine` 负载与 `grasp_result` 同口径（咬空 −62‰ / 夹持 −206‰）；`run_real_agent` / `run_real_eval` 默认值同步。**同时修掉一处语义 bug**：`detect_contact` 原为 ``load > baseline + margin``，而本硬件空载与夹持是**同向**扭矩（−62‰ vs −206‰），带符号比较 ``−206 > −69+50`` **恒为假 → 负载判据完全失效**；改为增量幅值 `|load−baseline| > margin`，与 `summarize_load_probe` 的 `|Δ|` 口径一致（既有 4 项测试在新语义下仍全过）。安全性质：空载上限 4.045 < 阈值 5.5 不误判；薄件 6.8 > 5.5 可检出；负载缺失时绝不判成夹住。
38. **真机判定复验通过，并修掉真机链路 3 个问题（设计未列）**：真机实测——咬空 pos 3.91 / load −62‰ → `holds_object=False`；夹住 pos **6.69** / load **−100‰** → `True`（**注意：这次负载只有 −100‰，差值 31 < `LOAD_MARGIN` 50 → 纯负载判据会漏判，是位置判据兜住的**；`LOAD_MARGIN` 是否下调待更多样本）。端到端实抓暴露 3 个既存缺陷并修复：① `grasp_and_lift` 缺归位（实测起跑 `shoulder_lift=−100` vs 初始 `44.9`，偏差 145）→ `_do_grasp_screwdriver` 先 `home()`，失败即 `FAIL_HOME_TIMEOUT` 且不进策略段；② 策略段超时写死 120 s 而默认 600 步必然超时 → 新增 `GRASP_STEP_BUDGET_S=0.5` 按步数缩放；③ `infer_so101.HOME_TOL=0.03` 按"弧度"写但状态实际是 lerobot 整流量（±100），**低于舵机分辨率 → 归位永远不可能成功** → 改 3.0，`HOME_TIMEOUT` 10 → 45 s，`home_to_initial` 增 `timeout_s` 参数。**真机阻塞**：`shoulder_lift` 卡在恰好 −100.00、双方向均无响应（同总线 `wrist_flex`/`elbow_flex` 正常）→ 关节级硬件问题，端到端实抓待现场处置后再跑。
39. **新增收尾自动复位 `park()`，并取证否定"位置残留"假设（设计未列）**：用户诉求"运行结束后应自动回到起始位"。先前三个候选成因均用真机否定：**残留 `Goal_Position` 驱动**（连接后静置 2 s 六轴漂移全 0）、**标定零点错**（标定文件齐全、原始值→归一化一致）、**归位不到位**（`home()` 1.0→0.3 s、偏差 ≤2.57）、**断开松力矩下垂**（保持 25 s 稳定、断开 6 s 后重连偏差 ≤0.29，减速比自锁）。真实成因就是**缺收尾复位**。实现：`RealSkillExecutor.park()`（失败返回 `False` 不抛异常；docstring 警示"夹着物体会把物体一起拖走"）；`run_real_agent.py` 在 `finally` 中调用并打印，新增 `--no-park` 逃生开关；`run_real_eval.py` 每轮任务 `finally` 中调用、引擎关闭移入 `finally`。真机验收：完整跑一次结束后位姿 vs 采集初始姿态最大偏差 **1.82**（容差 3.0）。
40. **相机索引映射漂移导致策略腕部通道拿到笔记本摄像头画面（根因，已修）**：用户观察到"能走到螺丝刀但夹爪不张开、后续无动作"；历史闭环日志证明夹爪动作链路正常（`action_gripper.pos` 1.35–20.27），故问题在**策略输入**。逐帧人工确认 + 与数据集训练帧比对后定位：**Windows 枚举顺序漂移**——原映射 `handeye=1` 时 index 1 已变成**笔记本摄像头**，正确枚举为 `0=夹爪特写(handeye)、1=笔记本摄像头、2=工作台俯拍(fixed)`。策略的固定视角一直是对的，但腕部视角长期是"对着人的笔记本画面"→ 动作彻底失效。处置：`CAM_IDX`/`DEFAULT_CAM_IDX`/`DEFAULT_CAMERAS`/`run_vla_collect` 录制模板四处改为 **`{"fixed": 2, "handeye": 0}`**，并在四处注释中写明"枚举会漂移、运行前用 `--camera-check` 抓帧核对"。**教训（写在注释里）**：像素相关性自动判映射**不可靠**（腕部画面随手臂姿态变化，非初始位姿时 index0 得分反而低于无关的笔记本摄像头）——**最终判据必须是人眼**。**附带发现**：`disable_torque_on_disconnect=True` 下，手臂在**有下垂空间**时断开 10 s 会掉 **9.61** 个单位（此前"不下垂"的结论无效，因为当时手臂压在桌面上）；用户决定**保持默认 True**（安全优先）。**另一待办**：fixed 相机（index 2）取景与训练时仍不一致（与 `dataset.fixed` 仅 r=0.303），需物理调回。
41. **真机硬件故障取证：`shoulder_lift`（id 2）力矩不足/效率衰减（阻塞实抓）**：重跑实抓失败码变为 `home_timeout`。逐项排除后定位到该舵机：① 六轴寄存器**配置完全一致**（`Max_Torque_Limit`/`Torque_Limit`=1000、`Protection_Current`=3000、PID 16/32/0）→ 非参数问题；② 抬升时**电压稳定 12.1 V**（其他关节同）→ 非供电/排线压降（尽管连接时曾出现 `id 6` 掉线，排线已由用户检查恢复）；③ 当前位置上下 ±4 均能自由运动 → 非"卡死"；④ **堵转时负载恒定 347‰（≈35% 额定）而电流 256–263 接近 `Protection_Current`=3000，位置在 −42.46/−35.40 完全冻结**；对照手腕下垂时电流仅 4、健康关节 `wrist_roll` 电流 11 → **电流拉满却输出力矩很小 = 效率衰减**（电机/减速箱），抬不动整条手臂。**已提示停止继续堵转驱动**（电流接近保护值、温度 47→49℃）。**判别测试（待用户做）**：松力矩后用手抬该关节——顺滑 ⇒ 换件（BOM 有备件，换后**必须重新 `lerobot-calibrate`**）；卡涩/异响 ⇒ 机械问题。
42. **⚠️ 上条结论被推翻：舵机正常，"舵机衰减"是误判（外部接触阻力伪装成舵机能力上限）**。用户**手动把其他关节角度调开、让手臂离开桌面**后，同一个 `shoulder_lift` 跑出健康曲线：`-99.83 → 45.86`（目标 44.9，偏差 1.0），保持 5 s 不回弹；**电流 232→190→140→92→87→38→8→0→8、负载 276→…→100→−58‰**——电流在最低点（重力力矩最大）最高、越往上越小、到顶近 0，负载随重力正常变化。**无需换件、无需重新标定。** 真因是**手臂完全塌下时夹爪压在桌面上**，上抬等于把整条手臂从桌面撬起，外部阻力极大 → 表现为"电流拉满、位置不动"，那个恒定的 **347‰ 实为桌面反作用力**；本批次最早那次 `shoulder_lift` 卡在 −100.00 双方向不动亦同源。**教训（写入文档）**：① 运行/归位前必须确认手臂**未压在桌面或卡在障碍上**；② 诊断"关节不动"**先排除外部接触阻力**再看电气量——仅凭"电流高 + 位置不动"极易误判为舵机故障（本次误判成本 = 一整轮排查）。
43. **新增真机就绪度冒烟测试 `run_real_smoke.py`，并抓到负载判据的假阳性（设计未列）**：本项目多次因环境问题白跑实验（相机索引漂移、手臂压桌面顶死关节、归位容差写死不可达），故新增"不含策略"的 30 秒就绪检查（总线/归位/双相机/负载/夹持判定/收尾复位，逐项 PASS/FAIL + 退出码）。**真机首跑 5/6 立刻抓到真 bug**：夹爪张开/空转时 `load=0‰`，而旧负载判据是"相对空载基准（−69‰）做差"→ `|0−(−69)|=69 > 50` → **误判成夹持**。根因是**空载负载不是常数**（随夹爪状态在 0‰~69‰ 变化），单一基准做差必然出错。修复：`holds_object` 负载判据改为**幅值** `|load| > GRIP_LOAD_THRESHOLD`，阈值 **150‰**（偏保守——薄件实测 ~100‰ 与空载 69‰ 太接近，故宁可不触发也不误判，薄件由位置判据兜）；`real_verifier` 的 `load_threshold` 替换 `empty_load_baseline/load_margin`。修复后真机冒烟 **6/6 通过（exit 0）**。
44. **P3 OPE 实际结果：ρ = 0.398（CI [0.322, 0.398]）→ marginal → 按设计交付归因报告（负结果）**。7 候选 / 5 族、每候选 10 rollout；仿真实测 [1.0, 0.9, 0.5, 0, 0, 0, 0] vs WM 预测 [0.1, 0, 0.1, 0, 0, 0.1, 0]。**归因**：① WM 单步动力学基本正确（单步 MAE 0.00655、obj xy 位移比值 1.03），但**多步开环 rollout 指数发散**（40 步内 obj-goal xy 距离 0.20→1629，真实恒定的 goal 漂到 437×4468）——MLP 对恒定维度的噪声被反馈放大 = rollout error compounding；② WM 想象 rollout 的**访问分布随候选退化急剧偏离训练分布**（|z| 230→1190，OOD 0.73→0.98），预测不可信。结论：**1-step MSE 训练的 MLP + 开环 rollout 不足以做 OPE 筛选器**，需多步/rollout 训练目标、恒定维度结构化处理或闭环校正。
   **与设计偏差 6 项**（详见 `plan.md` 批次 19）：① WM 数据源由 week15 v3 改为**实时仿真采集**（v3 的 PushCube 演示**不含成功轨迹**：物体每集只移动 5–7 cm 而目标在 20 cm 外）；② horizon 40→25；③ 候选退化档强化（σ≤0.2/lag-3 无梯度）；④ 成功判据 3D→**xy**（z 恒有 ~0.019 偏移）；⑤ 归因改用**实际访问状态**（原用初始状态→无区分度）；⑥ 归因**跳过训练中恒定维度**（原实现全候选 OOD=1.0）。

### 2026-10-11 · 批次 21（P1 的 2.7/2.8/2.9 前置 + 2.10 抽检工具）

45. **新增 `src/lerobot_label_source.py`（§5 未列）**：`label_dataset` 只依赖 `EpisodeSource` 协议，但真数据集实现需要处理"**AV1 编码的 mp4**（`demo_v23` 的 `video.codec=av1`）"——OpenCV 默认构建无 AV1 解码器，改用 **PyAV**（libdav1d）。该模块同时负责"读 meta/parquet 取 `gripper.pos`"与"等间隔抽帧落盘"，并把 `pyarrow`/`av`/`cv2` 做成**函数内惰性导入**，使 `sample_indices`、`read_dataset_meta` 等纯函数可离线测试。
46. **抽帧策略：单帧取末帧**（§5 未指定）：演示的**终态**最能体现"是否夹住/抬起"，比取中间帧更利于 subtask 与 success 判定；多帧时等间隔且含首尾。
47. **帧清单采用"合并式"写入**（§5 未指定）：断点续跑时本次只迭代了部分集，覆盖式写清单会让复核页看不到历史集 → `write_manifest` 与既有文件合并。
48. **新增 `src/review_sheet.py`（§5 未列）**：2.10 原设计只写"人工执行"，但**没有工具就无法产出 `human.jsonl`**（第二份独立标注流），κ 无从计算。实现为自包含 HTML（内联 CSS/JS、仅相对路径引图），人工只填 intent/subtask/success 三栏，导出**直接是 `LabelRecord` schema 的 jsonl**（不需再转换）。**页面刻意不显示模型答案**——否则人工会照抄，κ 被系统性抬高，审计失去意义。
49. **`VlmLabelingBackend` 刻意不向模型提供数据集指令**（§5 未明确）：若把已知指令喂给模型，VLM 流的 intent 只是"把答案抄一遍"，κ 变成自证。故 System Prompt 只给帧与 JSON schema，让 intent 成为**独立推断**；同时约定"**模型未给出 `task_text` 必须报错**"，不许静默回退到数据集指令（否则"没读出来"会被伪装成"读对了"）。
50. **`extract_json_object` 由私有提升为公开**（承接偏差 25）：标注链路同样要解析围栏 JSON，复制一份会让两处容错逻辑漂移；改为 `real_perception.extract_json_object` 作为唯一实现，`_extract_json_object` 保留为内部别名。
51. **`run_vlm_label.py` 复用 0.3 的 `VLM_<源>_*` 注册表并显式 `load_local_env`**（§5 未列）：设计只写"调 VLM"，未规定配置来源；沿用 0.3 的 env 驱动注册表可避免模型名/端点/约定硬编码，并让 `--source` 与定位评测同口径。CLI 另加 `--dry-run`（桩 chat 后端，仍走真实解析/校验路径），与仓库既有 runner 的 dry-run 约定一致。
52. **`default_label_scope` 自动生成适用边界说明（§5 未列）**：设计要求"必须报告 label_scope"，但只给了透传参数；实跑发现 demo_v23 上 **intent（68/68=`pick`）与 subtask（66/68=`grasp`）两维都退化**、`confidence` 恒 0.90–0.95，若不带边界说明，κ≈1.0 会被误读为"标注质量高"。故新增纯函数自动写入"维度退化 + success 仅位置判据"两条边界。
53. **demo_v23 的 `gripper.pos` 量纲与当前真机标定不一致（重要局限，未改代码）**：该数据集实测张开 ≈1.5–4.0、闭合夹持 ≈7–10，而当前真机空载闭合基准为 4.1（0.6 标定）。`success_from_sensor` 用的是**当前标定值**，故其在 demo_v23 上只是启发式判据，**必须以人工抽检为准**（已写入 `label_scope`）。真正的修复属于"重新标定/统一量纲"，留待 2.9 多任务采集时统一。
54. **复核页 radio 的 `name` 必须逐集唯一（缺陷修复，设计未提）**：首版把 14 张卡的三组单选按钮写成共用 `name="intent"/"subtask"/"success"`，而 **HTML 中同名 radio 是全文档互斥的** → 选第 2 集的 intent 会自动清掉第 1 集的，表现为"整页只能填一张卡"。改为 `<episode_ref>:<字段>` 逐集唯一，JS 取值用结尾匹配 `input[name$=":intent"]`（不依赖 ref 的转义），并补 `test_radio_groups_are_per_episode` 回归。**流程教训**：首轮浏览器验证只覆盖"单卡选择 → 进度 +1"，未覆盖**跨卡**互不干扰；含多实例表单的交付物必须测"实例间隔离"。
55. **2.10 人工抽检结果（14 集）：κ_intent = 1.000（退化） / κ_success = 0.000（口径不一致，见下）**。人工分布：`intent` 全 `pick`；`subtask` lift 11 / grasp 2 / approach 1；`success` True 11 / False 3。per-subtask recall：`grasp` 1.00 / `lift` **0.00** / `approach` **0.00**；分歧 **14/14**。**结论：零样本 VLM 能识别任务级语义（`task_text` 逐集正确 → intent 全对），但不能判断阶段级进展（`subtask` 恒为 `grasp`，把"已抬起"也判成 grasp，而人工 11/14 认为已 `lift`）** → 阶段标注必须改为"按帧给切片"或对 VLM 微调。**本条的指标口径缺陷（§4.1 `LabelRecord.success` 定义未限定"夹住"还是"夹住并提起"）**：`success_from_sensor` 判的是 `is_holding`（**是否夹住**），而复核页问的是"是否成功夹住**并提起**"（更严）→ 两侧口径不同，`κ_success = 0` **至少部分是口径不一致的产物**，不能据此单独否定传感器判据。**修正方向**：把该字段的判定口径写进 schema（或在 `LabelRecord` 上加 `success_criterion` 字段），复核页措辞与判据必须一致。
56. **阶段标注改为「模型只出序列、代码取末段」（批次 23）**：§5.1 原设计由模型直接给单个 `subtask`（prompt 里写"取最靠后的主要阶段"），实测把"判断"与"取舍"都压给模型后它退化成整段恒答 `grasp`（偏差 55）。改为：prompt 要求输出按时间排序的 `stages` 序列，`subtask_from_payload` 取**末段**派生 → 模型只负责判断、取舍由代码负责，且非法阶段名**丢弃并记 note**（单个坏值不该毁掉一次外网调用）。**A/B 结果**：人工 `lift` recall **0.00 → 0.36**、分歧 **14 → 10**、分布 66:2 → 40:28，但 `grasp` recall 1.00 → 0.00（过度纠正）→ **prompt 有贡献但不是主要瓶颈**，残余误差属"看到了但与人工判断相反"（模型自述"已夹住但尚未抬起"），下一步应改抽帧方式 / 拆成独立二分类 / 微调，而非继续调措辞。
57. **复核页新增 `--prefill`（批次 23，设计未提）**：口径修正后需要重问 `success`，但让人重看 112 张图不合理。新增用**人自己之前填过的答案**预选（`ReviewItem.defaults` + `_radios(..., selected)`），只重答变化的那一栏即可。**硬约束：只允许预填人工答案，禁止预填模型答案**——后者会造成锚定、系统性抬高 κ，与"两条独立标注流"的前提冲突。
58. **`success` 问法必须与判据同口径（批次 23 收口偏差 55）**：复核页由"是否成功夹住**并提起**"改为「**夹爪是否夹住物体**」，并在 `success_from_sensor` docstring 写明该口径；`test_success_wording_matches_sensor_criterion` 固化"旧问法不得再出现"。**教训（一般化）**：任何"自动判据 vs 人工标注"的 κ，都必须先把**判据口径写进 schema/文档**再采集人工标签，否则 κ 衡量的是口径差异而不是标注质量。
59. **口径对齐后的复问结果：`κ_success` 仍为 0，且证明是判据本身无信号（批次 24）**。新口径下人工 `success` 为 True 6 / False 8（口径更严时曾为 11/3）→ `κ_success = 0.000`、分歧 13/14。**方法论要点（本条最重要）**：**当一侧（此处为传感器判据）方差为零时 κ 恒为 0，κ 无法区分"略优于随机"与"完全无关"**，故必须补**可分性指标**：以人工为真值，判据命中 6/14、错误率 **57%**；末 5 帧夹爪位置在"人工=夹住"（8.09–9.97）与"人工=没夹住"（6.90–9.97）两组间**完全重叠**。**结论**：§4.1 的 `success`（位置判据）在 demo_v23 上不可用是**信号问题而非口径问题**——这正面支持 P0「负载信号」的立项（`Present_Load` 才是"是否夹住"的可分辨信号，0.6 实测空载 55–69‰ vs 夹持 206‰）。**同时排除**了"位置只是指令回显"（`observation.state` 的 gripper 列与 `action` 列最大差 6.66–11.35，不相等）。
60. **采集链路缺 `Present_Load`，是 2.9 的前置阻塞（批次 24 新发现）**：`run_vla_collect.py` 仅为 SOP 打印器，实际执行 `lerobot-record` + `so101_follower`，其 features 由机器人 `observation` 推导，**只有六轴 `.pos` + 相机，不含负载通道** → 若照现模板采集 2.9 多任务集，新数据依旧做不了可靠 success 标注（偏差 59 的问题会原样复现）。**建议方案**：采集时**旁挂**按帧对齐的 `load.jsonl`（`RealEngine.read_load` 已具备），**不改 LeRobotDataset 的 fields**（与 §3 的"不改 fields，避免破坏既有 ACT/SmolVLA 训练脚本"原则一致）。待用户决策后再落地。
61. **采集期旁挂负载通道已落地（批次 25，收口偏差 60）**：新增三个模块 —— `src/load_sidecar.py`（旁挂格式：**按集缓冲、成集落盘** + `decode_loads` 复用 `decode_feetech_load` 并**逐关节容错** + `episode_loads` + `verify_alignment`）、`src/load_logging.py`（`LoadLoggingRobot` 包装 + `patch_record_pipeline`）、`src/run_vla_collect_record.py`（`lerobot-record` 的**透明包装 CLI**，只认 `--sidecar`/`--verify`、其余参数原样转交）。**四处关键决策**：① 不改数据集 fields —— features 是既有训练脚本的契约，包装层把 `observation_features`/`action_features` 原样透传、且不往观测里多加键；② 必须替换 `record_loop` —— 它同时用于"录某一集"（`dataset` 非空）与"集间复位"（`dataset` 为空），复位段也会调 `get_observation`/`send_action`，不区分就会把复位段负载混进帧序列；③ 负载在 `send_action` 里采而非 `get_observation` —— 录制循环中 `send_action` **恰好每存一帧调用一次**，而"无 policy/teleop"的跳过分支不调用它，天然与数据集帧一一对应；④ 按集缓冲 —— 坏集重录（←）是常规操作，逐帧写会留下被丢弃那一集的垃圾行（同集号再次开始 = 重录 → 丢弃旧缓冲）。
62. **`--verify` 是采集的硬门禁（设计未列）**：旁挂文件与数据集的对齐**无法自证**，故旁挂里额外存下与 `observation.state` **同源**的 `positions`，采集后强制跑一次逐帧比对（帧数 + 位置双判据）；不通过则该份旁挂数据不可用。这条门禁是为"帧数不符 / 顺序错位 / 重录残留 / 异常中断提交了未保存的集"这四类静默故障准备的 —— 旁挂数据一旦错位，后续的成功标注会全错且难以察觉。用合成数据集端到端验证：一致 → exit 0，少一帧 → exit 1。
63. **1.3 变体布局集中定义 + 缺键即报错（设计未细化）**：§5.1 只写了"特征构造函数"，未规定列序契约。实现在 `real_geometry` 里集中定义 `VARIANT_ORDER` / `VARIANT_EXTRA_FEATURES` / `variant_feature_names`（**累积展开**）/ `feature_values` / `build_feature_vector`。两条严格性是刻意的：① 累积顺序只在**一处**定义（训练与推理列序不一致时数值仍然"有限"，指标只是变差、事后无法归因）；② **不可用的量缺键报错、不填 0** —— 填 0 会把"没测到"伪装成"等于 0"，污染训练且无法事后区分；`feature_values` 还会校验 `DepthEstimate.method` 与来源匹配（把 stereo 当 known_size 塞进来是最隐蔽的一类错）。
64. **1.5 额外观测通道**复用 ENV 向量**而非新增 feature key（设计 §7.1 只给了 B0–B3 的输入内容，未规定落点）**：`VisualACTTrainingConfig.extra_features`（默认 `()`），列追加在 ENV 辅助向量尾部（13 → 13+n）。理由：不动 LeRobot 的 `FeatureType` 与 `normalization_mapping`（改 ENV 归一化会**连带改变 B0 基线**，让消融与历史结果不可比），老 checkpoint/老数据集零影响；默认 `()` 使 **B0 与历史行为逐位一致**。配套三处必须同时改：`make_visual_act_config` 的 `input_features` 维度、`task_metadata.json` 落盘 `extra_features`、`load_visual_act_checkpoints` **回读后重建**（不回读会用 13 维重建 → 与权重/归一化 tensor 形状不匹配而加载失败）。**已知代价（明写在 config 字段注释里）**：该通道**不做归一化**（与既有 ENV 通道现状一致），尺度标准化由上游数据准备负责。`VisualACTPolicy` 带通道但未给 `extra_provider` 时**直接报错**，不允许把通道当 0 用。
65. **P2 4.2–4.7 的落地偏差**：① `AdaptConfig.trainable_ratio` 语义定为"**实测**可训参数占比"（`zero_shot→0.0`、`full→1.0`、`head→None` 表示待 `apply_adaptation` 实测），与 §4 把它写成普通 float 略有出入；② `apply_adaptation` 的 `head` 策略按**参数名匹配**解冻，**匹配到 0 个参数即报错**（不许悄悄退化成"全训"或"全冻"）；③ `run_adapt_experiment` 的保留率分母为 0 时返回 `None` 而非 `0.0`（避免被读成"完全遗忘"），`peak_mem_mb` 拿不到记 `None`（lerobot CLI 不上报峰值显存）；④ 训练/评测均**依赖注入**（`train_fn`/`eval_fn`/`baseline_eval_fn`），默认实现走真实链路，但**明确声明"lerobot 0.3.4 无按模块冻结入口，默认训练路径不代表 head/full 冻结已生效"**（对应 §3.3 阻塞点 G），真实冻结须由训练侧拿到 policy 后调 `apply_adaptation`；⑤ `run_vla_adapt.py` 的 `MAX_ADAPT_UNITS = 6` 是**硬封顶且超限报错**（不静默截断）。**4.6–4.7 的落地偏差**：`GeneralizationSplit` 增 `warnings` 字段；`build_split` 首参取 **`combos`** 而非 §5.1 的 `dataset_root`（多任务数据集尚无、且其 schema 不携带"物体"身份，无法可靠推导组合）；仓库**没有同义改写模型**，`paraphrase` 协议实现为"由调用方提供同一物体的多条措辞、确定性留出一条"并附带显式告警（held-out 侧是**调用方给的变体**、非生成的同义句）；`proportion_ci` **自写 bootstrap**、不复用 `wm_ope._bootstrap_ci`（后者对 Spearman ρ 特化，不能用于比例）；默认 episode runner 基于状态输入、**忽略 instruction**（语言条件评测必须传自定义 `runner`），该边界写入报告 `scope_note`；新增 `HELDOUT_SUCCESS_TARGET = 0.40` 把验收 A7 落到报告字段 `heldout_meets_target`。
66. **待决策（阻塞 1.6 的实质意义）：B1/B3 的特征由谁产生？** 设计 §7.1 只写了"已知尺寸反解深度 / 双视角三角化"是**能力**，没规定仿真消融里这些数值从哪来。三条路各有代价：① **特权/真值（oracle）**——从 sim state 直接取物体位姿算距离，实现最省，但 B1/B2 会变成"特权信息消融"，A2 的 +15pt 不能解释为可部署能力；② **检测器/匹配**——B1 需目标像素 bbox、B3 需双视角对应点，仿真里同样要跑一套感知（且引入感知误差作为混淆变量）；③ **混合**——B1/B2 用 oracle 做**上界**、B3 用检测器做**真实**对照。**这是研究设计决策，不由实现方代定**：不定它，`run_depth_ablation.py` 即便写完，跑出来的 ΔGrasp 也无法归因。当前进度：1.5 的**通道接线已就绪**，等待该决策后补"特征生产"步骤（数据准备 + 推理侧 `extra_provider`）。
67. **方案③已落地（批次 27）+ 新发现的 B0 定义冲突**：用户选定方案③，新增 `src/ablation_features.py`（纯函数）：B1 = 物体→夹爪距离（oracle，**sigma 记 0** 以区别于检测路径）、B2 = `is_grasped` + 闭合度缩放的负载代理（oracle；**仿真无扭矩通道**，故是代理而非物理量）、B3 = 两视角像素中心 + 内参/相对位姿 → `real_geometry.triangulate`（**detector**）。**防循环论证**：B3 缺 detected 输入即报错——用特权位姿反推三角化 ≡ 直接给答案，"真做感知能拿多少"的对照会消失。**同时发现设计 §7.1 与实现的冲突（待澄清）**：§7.1 把 **B0 写成"仅 RGB"**，但 `visual_act` 现在的输入是 **RGB + robot_state(5) + env_state(13)**，而 env_state **已含** `obj_pos / goal_pos / (tcp−obj) / (obj−goal) / is_grasped`（见 `lerobot_act.ENV_STATE_INDICES`）——**特权几何量本来就在 B0 里**。故两种理解要做的事不同：**B0 = 现状** ⇒ B1/B2 的 oracle 列与 env_state 冗余、ΔGrasp 趋 0，消融无信息量；**B0 = 仅 RGB** ⇒ 需给 `visual_act` 增加"关闭 STATE/ENV 通道"的开关（B0 去掉、B1–B3 逐级加回），这才贴合 §7.1 与 week17"单目深度模糊"的诊断。**在澄清前不写 1.6**，否则变体的通道定义会返工。
68. **1.6 已落地（批次 28）：用户选「B0 = 现状」，于是本消融在现行口径下**不可能支持 A2**——这一点被写进代码而不是只写在文档里**。实现：`src/depth_ablation.py`（`build_plan` 规范顺序+必须含基线 / `read_env_state_dim`+`validate_dataset_dim` **开训前维度门禁** / `summarize_ablation` ΔGrasp 百分点+闭合高度误差+失败归因+两类告警 / `run_ablation` 注入式编排）与 `src/run_depth_ablation.py`（`--dry-run` 打印计划与口径、真跑前过门禁、报告落 `ablation.json`+`ablation.md`）。**关键判断（与偏差 67 同源，必须随结论报告）**：env_state 已含 `obj_pos/(tcp−obj)/is_grasped`，而 B1 的 `depth_known_size_m` = ‖tcp−obj‖、B2 = `is_grasped`+夹爪开合度，**都是它的确定性函数 → 信息增量恒为 0**；B3 的检测深度同样由物体真值位置决定，引入的是**感知误差**而非新信息。故 **B0/B1/B2 的差异原则上只应落在噪声内，A2（ΔGrasp ≥ 15pt）在本口径下不可能由 B1/B2 达成**。为防"只看数字不看前提"，这段结论被固化成代码常量 `REDUNDANCY_CAVEAT`，CLI 每次运行都会打印、报告里原样带上。**要回答 A2 必须改用 B0 = 仅 RGB**（关闭 STATE/ENV），届时只需给 `visual_act` 加通道开关 + 复用已就绪的 1.5 额外通道，不必重做特征生产。
69. **4.1 与 1.4 落地（批次 30）+ 三处与原设计的差异**：① **4.1 的成功判据改用 `holds_object`**（位置 OR 负载，第 38 周 0.6 实测标定），**不用** `outputs/infer_so101` 的 `GRASP_CLOSED = 10.0` —— 该值实测夹持位只有 6.69，用它会把"夹住"判成"没夹住"（真机复验时正是换判据才判对的）；② 主循环拆成"依赖注入的 `run_vla_loop`" + "只做装配的 `run_real`"，因此**真机代码可以在无机械臂/无模型的机器上被测试**（否则 4.1 永远只能靠真机试错）；③ 为此给 `RealEngine` 新增 `load_act_policy=False`（VLA 自带策略，不该顺带加载 ACT checkpoint 与 ACT 的 dataset_stats）与通用 `send_action`（未知名忽略、全被忽略则报错），并同步 `RobotEngine` 协议与 `MockEngine`。**1.4 的落地范围**：`handeye_calib.py` + `run_handeye_calib.py` 覆盖"绝对位姿 → 相对运动对 → 求解 → 逐对残差 + 留一法 + 门禁"，但**不含采集**（需 FK 或标记物检测，本仓库都没有，与设计把 1.4 标为"条件启用"一致）；`--self-test` 用合成真值 X 把整条链路测通。**新增一处原设计未提的诊断**：**留一法**（leave-one-out）——全量拟合的残差会被离群对自己拉小，留一法把它暴露出来，从而区分"数据有离群点"与"精度不够"；实测第 3 对 50 mm 离群时留一法为全量均值的 5.7× 并触发告警（门禁同时未通过）。**过程中自检抓到的一个自身缺陷**：CLI 初版 `--min-poses` 默认 3 → 只生成 3 个位姿（2 组相对运动）→ 留一法跑不起来、`--outlier-pair` 注入不到，等于自检漏掉最关键的诊断路径；改为 `--self-test-poses`（默认 10）+ 下限校验。**核验**：两个 P2 CLI 的 dry-run 实跑通过（泛化三协议无泄漏 / 适配 5 单元矩阵）。
71. **A 方案落地（批次 31）：ENV 通道口径开关 `env_state_mode`**。用户确认"先让 B0 真正无特权几何"后再跑消融。设计 §7.1 把 B0 写成"仅 RGB"，而实现里 B0 一直带着 13 维特权 `env_state`（含 `obj_pos/(tcp−obj)/is_grasped`），导致 B1/B2 的列是其确定性函数、ΔGrasp 必然是噪声。新增 `VisualACTTrainingConfig.env_state_mode`：`"full"`（默认，历史行为，**老 checkpoint/老数据集零影响**）/ `"extra_only"`（策略侧只留额外列；`extra_features=()` 时等于彻底无 ENV）。**关键取舍**：① 把"数据集列数"（`dataset_env_state_dim` = 13+n）与"策略侧维度"（`env_state_dim`）分开——**数据集不必重做**，同一份 15 列数据既能喂 `full` 也能喂 `extra_only`；② ENV 维为 0 时**整块省掉** feature/统计/批次键（三处守卫），否则会给 ACT 与归一化器喂 0 维张量；③ 元数据落盘 + checkpoint 回读 `env_state_mode`；④ `VisualACTPolicy.act` 同步 `extra_only`（训练/推理口径不一致时**形状不报错、结果全错**，是最难查的一类）。**未做**：`use_robot_state` 开关（保留本体感受）——设计说"仅 RGB"，但拿掉关节角会把"补几何信息"与"失去本体感受"混在一起，故本次只切 ENV 并如实记录该偏差。**云端验证**：B0 `policy_env_dim=0`/`act_env_feature=None`/批次无 ENV；B1=2、B2=4；`LeRobotDataset` 读出变体数据集正常（根因是缺 HF 缓存条目，非列追加）。
70. **1.6 真跑前置已落地（批次 29）+ 一个必须记住的硬链接陷阱**：新增 `src/ablation_dataset.py` + `src/run_prepare_ablation.py`，把基础数据集的 ENV 列从 13 扩到 `13+n`。**能离线做的原因**：oracle 需要的量数据集里本来就存了（`observation.state` = `tcp_pos+gripper_qpos`，`environment_state` 含 `obj_pos`/`is_grasped`），故额外列是对既有列的纯算术；夹爪开合范围用**数据极值**而非硬编码常数。B3 仍拒绝（单相机通道，两视角三角化无法在不循环论证的前提下得到）。**设计外但必须记的施工教训**：初版整树硬链接 → `meta/info.json` 与源数据集**共享 inode**，`Path.write_text` 是 `O_TRUNC` 写同一 inode，**穿透链接把源数据集的元数据也改了**，且三个变体共用一份元数据互相覆盖（实测三个 `info.json` 全变成 17、源也被改成 17）。**规则：硬链接只用于 `videos/`**（体积大的部分），`meta/**` 一律复制，`data/*.parquet` 跳过由我方 `pq.write_table` 重写（同理，截断写会毁源）。已修复源数据集并补两条回归测试（"准备两变体后源 `info.json`/`stats.json` 逐字节不变"、"两变体各拿自己的维度"）。**验证边界（如实记录）**：`LeRobotDataset(...)` 读取未跑通——lerobot 0.3.4 需先向 HuggingFace 解析 `repo_id` revision，本机访问不到 huggingface.co；**对原始 week15-cam 调同一 API 得到完全相同的 `ConnectError`**，故与列追加无关（既有环境条件），该步需在有网/有缓存环境补验。
72. **2.9 采集清单生成器落地（批次 32）+ 旁挂文件必须按 repo 分开**：新增 `src/vla_collect_plan.py`，把「物体 × 指令」固化成可执行清单，产出 `collect_tasks.json` / `collect_combos.json`（直接喂 P2 的 `run_vla_generalization --combos-json`）/ `collect_commands.ps1`（逐条录制命令 + 采后 `--verify`）。**设计外的关键修正（真实缺陷）**：初版让 9 个 repo **共用一份** `load_sidecar.jsonl`，但 `LoadSidecarWriter` 是**追加写**、`--verify` 又把**整份**旁挂对**单个**数据集根目录逐帧比对 → 第 2 个 repo 起校验必然把前一个 repo 的集号混进来而失败。改为 `sidecar_for(task)` **每 repo 一份**（`sidecars/<repo 名>.jsonl`），并加 `test_different_repos_use_different_sidecar` 回归。**其余取舍**：① repo 名用 **ASCII slug**（`demo_vla_01_pick_a`，序号区分物体、key 区分指令）——HF repo 名对非 ASCII 不可靠，中文物体名只出现在 `--dataset.single_task` 与清单注释里；② 三条指令 `pick_a`/`pick_b`（pick 的同义改写，供 paraphrase 协议）/`place_left`（提供**第二个意图**，否则意图只有一类、κ 退化）；③ `--print-only` 便于先核对再落盘；④ 上传口径显式化——默认 `push_to_hub=false`（只落本地，**先采 2 集试跑时别联网**），铺满后 `--push-to-hub` 走**私有 Hub**（与既有 `demo_v23` 一致），不静默改你既有工作流；⑤ **解释器必须是装了 lerobot 的那个**（`PYTHON` 常量 = `D:\miniconda\envs\lerobot\python.exe`）—— 首版命令写成裸 `python`，用户实跑命中 **`ModuleNotFoundError: No module named 'lerobot'`**（系统 `python` 上没有 lerobot），已改为 `& "<lerobot python>" ...` 并加回归测试；⑥ **清单里显式提示"本地数据集目录必须不存在"**——`LeRobotDataset.create` 用 `exist_ok=False`，中断重跑必然 **`FileExistsError`**（实测：舵机未上电那次失败后留下的空壳目录，只有 `meta/info.json`、零 episode，会挡住下一次录制）。脚本注释给出判据：**只有 `meta/info.json` = 空壳可删；已有 episode 则用 `--resume=true`（顶层参数，见 ⑩）续录、不要删**；⑦ **`--verify` 的 `--dataset-root` 不留占位符**——新增 `LEROBOT_CACHE_ROOT` + `dataset_root_for(task)`，按 lerobot 本地缓存约定 `<缓存根>\<repo_id>` **逐 repo 填好实际路径**（用户要求）；⑧ **旁挂提交点改为 `dataset.save_episode()`（真机实测暴露的丢数据 bug）**——原设计靠"下一集开始"（`begin_episode`）提交，收尾阶段崩溃时旁挂会比数据集**少最后一集**：实测录制 2 集后 `robot.disconnect()` 给夹爪写 `Torque_Enable=0` 连续 6 次 `Overload error!` 抛异常 → 调用方 `abort_episode()` 抹掉待提交缓冲，而**那一集其实已经 `save_episode()` 进数据集了**（`data/` 有 `episode_000000/000001.parquet`，旁挂只剩 ep0 的 693 帧）→ `--verify` 必失败、整份旁挂报废。修法：`patch_record_pipeline` 给数据集实例挂 `save_episode` 钩子，`LoadLoggingRobot.mark_episode_saved()` 在**数据集落盘同一事件**上提交旁挂；`abort_episode` 仍保留，但只对"尚未 save 的那一集"生效。**一般化教训**：**两个需要保持一致的落盘流，必须绑在同一个事件上**，靠"下一个事件来推进上一个"的隐式提交在异常路径上必然不一致。⑨ **采集命令默认不卸力矩（用户决策）**——`--robot.disable_torque_on_disconnect=false` 显式写进每条录制命令（lerobot 默认是 `true`）。理由就是 ⑧ 那次崩溃：退出时写 `Torque_Enable=0` 在夹爪过载时报错、中断收尾。**代价（已写入命令清单注释）**：脚本退出后舵机**持续通电**保持姿态（好处是不下垂）。要恢复旧行为用 `--torque-off-on-disconnect` 重新生成。⑩ **续录参数名修正**——清单注释初版写 `--dataset.resume=true`，但 `resume` 是 `RecordConfig` 的**顶层**字段（`lerobot/record.py:173`），正确写法是 **`--resume=true`**；已修正注释并加断言（`"--resume=true" in script and "--dataset.resume" not in script`）。**教训**：引用第三方 CLI 参数前必须回源码确认层级，别照"看起来合理"的命名习惯写。⑪ **`--instructions` 支持裁剪指令集（用户决策：砍掉 `pick_b`）**——用户指出 `pick_a` 与 `pick_b` **动作完全一样、没必要录两遍**，选择砍掉 `pick_b`（少录 1/3）。新增 `select_instructions(keys)`（按声明顺序过滤，空选择/未知 key 报错）+ `paraphrase_available()`（同义改写配对是否齐备），CLI 加 `--instructions pick_a,place_left`，生成时若配对不齐**打印告警**并写进 `collect_tasks.json` 的 `paraphrase_available`。**顺带修掉一个连带 bug**：`write_plan` 原先把 `instructions` 硬编码成 `DEFAULT_INSTRUCTIONS` —— 一旦裁剪，清单里就会列出**没录的指令**、与实际任务对不上；改为记录**实际使用**的那几条（新增回归 `test_records_actual_instructions_not_all_defaults`）。**取舍（已写进 docstring）**：砍 `pick_b` 后 **P2 的 `paraphrase` 协议没有留出措辞可用**（只剩 `combo` + `novel_object`），代价是失去"换说法还认不认"这一条文本泛化证据；`place_left` 不能砍（砍了意图只剩一类、A3 的 κ 退化）。**最终清单**：3 物体 × 2 指令（`pick_a`/`place_left`）× 20 集 = **6 repo / 120 集**。⑫ **相机取帧超时重试（真机实测新缺陷）**——录制中一路相机偶发 200ms 超时：`TimeoutError: Timed out waiting for frame from camera OpenCVCamera(0) after 200 ms. Read thread alive: True.`（`OpenCVCamera(0)` = handeye），**整场录制崩掉、当前集丢失**。取证：lerobot 的 `OpenCVCamera.async_read(timeout_ms=200)` 把等待**写死在签名默认值**里，`OpenCVCameraConfig` 只有 `color_mode/fps/height/index_or_path/rotation/warmup_s/width`、**没有可调项**；30fps 下 200ms = 连丢 ~6 帧，"read thread alive: True" 说明后台线程活着却拿不到帧 → 典型是**两路相机抢同一个 USB 控制器**。修法：在 `LoadLoggingRobot.get_observation` 里加**只对 `TimeoutError` 生效**的重试（3 次 × 200ms 间隔）——正常路径**完全不进入**该分支（零行为变化），`RuntimeError` 等真故障**立刻抛、不被掩盖**；同时累计 `camera_timeouts` 并在收尾汇总打印告警，**避免"靠重试掩盖相机退化"**。**数据结构上的连带确认**：该次崩溃发生在**复位段**（`Reset the environment` 之后），当时该集尚未 `save_episode()` → 数据集与旁挂**都没有**这一集、仍然一致（⑧ 的修复在真机第二次得到验证）。已加 4 项回归（`TestCameraTimeoutRetry`）。**教训**：生成的"可执行命令"里的解释器路径/数据集路径都不是装饰，必须与既有 SOP 一致并被测试钉住。**验证**：`src/tests/test_vla_collect_plan.py` 49 项、`src/tests/test_load_logging.py` 22 项；全量回归 **1169 passed / 7 skipped / 0 failed**（基线 1112 + 57）。**真机验证（3 集）**：`--verify` 报「集数 3，错位 0 集」，`episode_000000/000001/000002` 帧数与位置逐帧一致（最大偏差 ~9e-7）—— **⑧ 的修复在真机数据上得到确认**（修复前最后一集必被丢掉）。**注**：`src/run_vla_collect.py` 的 SOP 仍指向旧的单一 sidecar 模板，2.9 以本生成器的清单为准（后续可让 SOP 直接引用生成器产物）。
73. **P0 消融评测链落地 + A2 在本数据集不可评测的取证（批次 33）**：批次 32 的消融只训练不评估 → 拿不到 ΔGrasp（A2 的唯一判据）。新增 `src/ablation_eval.py`（`sim_oracle_inputs`/`make_oracle_provider`/`closure_range_from_dataset`/`success_rate`/`build_variant_result`/`evaluation_episodes`）+ `src/run_ablation_eval.py`（含 `--dry-run`）+ 29 项单测；特征**复用**训练侧公式（`oracle_features` + `variant_feature_names` 定列序），`evaluate_visual_act` 补 `extra_provider` 透传。**实跑暴露 3 个本地测试覆盖不到的真缺陷**：① 我把 `action_smoothing` 从签名删掉、调用处仍传 → `NameError`（→ 新增"打桩真正执行该函数"的装配测试）；② **口径回读读了错误的文件**——`env_state_mode`/`extra_features` 在 `task_metadata.json` 而不在 checkpoint 的 `config.json`（那是 `ACTPolicy` 的配置），于是退化成 `"full"`、给 B0 多注册 13 维 ENV 输入 → `encoder_1d_feature_pos_embed` `[3,512]` vs `[2,512]`（→ `resolve_checkpoint_env_config` + 真正把 `env_state_mode` 传进 config）；③ **对账 helper 只认 dict**，而内存里 `input_features` 是 `PolicyFeature` 对象 → 对账静默失效，把 ② 的可定位报错退化成晦涩形状不符（→ `env_feature_dim` 两形态都认）。**评测结果（每变体 2 任务 × 种子）**：B0 0.800（5 种子）/0.725（20 种子）；B1 0.300/0.350；B2 0.000/0.025。**两条决定性取证**：① **零 provider 对照**（恒返回 0 = 与训练分布一致）→ B1/B2 均回到 **0.850**，故 **额外 ENV token 无害**（批次 28/31 的"结构混淆"担忧不成立）、且那张 −37.5/−70.0 的**表作废**；② **分布并排比** → B1 的 depth 分布内（训练 0.063/0.055 vs 评测 0.034/0.050），而 B2 的 `load`（0/0 vs −68.1/46.3）与 `contact`（0/0 vs 0.69/0.46）**严重 OOD**。**根因在数据层**：**week15-cam 的 `is_grasped` 从未为 1**（该数据集没有抓取行为）→ 训练时接触/负载恒 0、评测时策略真会夹住 → OOD 不可避免，**B2 在这份数据上不可能被公平评测**。**B1 的下降仍未解释**（两列都分布内，却 0.85→0.35），**不能读成"深度无用"**。**处置（用户决策）**：暂停消融、A2 标为"本数据集不可评测"，等 2.9 真实抓取数据或 sim 侧重采带抓取的数据再做。**教训**：① **"能加载/能跑完"不等于"评测有效"**——评测链路必须自带**分布一致性检查**，否则会把"喂错值"读成"方法无效"；② 只测前置纯函数（`--dry-run`）**测不出函数体里的错**，装配路径要有"真正执行一遍"的测试。
74. **2.9 采集完成核验 + A3/A4 机读汇总 + S13 数据卡/模型卡（批次 34）**：6 repo / **120 集 / 62318 帧**，`--verify` **6/6 通过**（逐行对齐，位置最大偏差 9.26e-07；旁挂 `gripper.load` 无空值、`decode_errors` 全 0）；集数偏差 `02_pick_a` 19 / `02_place_left` 21。新增 `src/vla_label_summary.py` + `src/run_vla_label_summary.py` + 29 项单测，产物 `runs/week38/label_vla/summary/`。**A3（核心负结果）**：零样本 VLM 的 `intent` 预测 `{pick:119, none:1}`，按 repo 语义应为 `{pick:59, place:61}` → **`place` 被预测 0 次**、κ **0.0080**（退化）、置信度 mean 0.934 —— **高置信地全错**；**据此修正批次 26 的结论**：当时"VLM 能做任务级语义"是**单一意图数据造成的假象**，换成双意图即失效。**A4**：`success` 整体 `{true:58, false:62}`，且与 repo 语义高度同构（全 True 仅两个 `pick_a`、全 False 仅 `03_place_left`）→ **疑似"照抄任务类型"而非独立判断**；`subtask` `{grasp:56, lift:63, approach:1}`（并非恒为 grasp/lift）；**"口径对齐"未完成并如实标注**——仓库只有**逐帧**判据（`holds_object`/`detect_contact`/`holding_and_lifted`）、**无"逐集成功"函数**，而 `labels.jsonl` 的 `success` 本身就出自 `vlm_labeler.success_from_sensor`（**纯位置**），自比无意义 → 必须新增"逐集聚合"（把逐帧的"位置 OR 负载"判据聚成集级）才能对照，**不允许临时拍判据**。新增交付物 `docs/week38/dataset_card.md`（双相机索引已核图确认、旁挂通道性质、校验状态、已知局限）与 `docs/week38/model_card.md`（三变体真实 `validation_loss` 首末、两份评测报告数字、**为何不能作为 A2 证据**、明确"不可部署"）；演示视频仍未做。**教训**：**单类别数据会让"零样本判别能力"看起来成立** —— 评估集必须含**多类别/多意图**，否则 κ 与准确率都是假象。

### 2026-10-11 · 批次 37（S12 落地：数据升版 + merge + 训练入口修正）

75. **云端 lerobot 版本与采集环境错配（0.3.4 采集 vs 0.6.1 训练）**：本机采集环境 `D:\miniconda\envs\lerobot` 是 lerobot **0.3.4**（源码在 `C:\Users\Administrator\Downloads\lerobot`），产出的数据集是 **v2.1**；云端（121.43.136.28）装的是 **0.6.1**，其 `LeRobotDataset` 只接受 **v3.0** → 加载 v2.1 直接 `BackwardCompatibilityError: ... is in 2.1 format`。**处置**：逐 repo `lerobot.scripts.convert_dataset_v21_to_v30 --repo-id … --root <本地目录> --push-to-hub=false`（**原地转换**：旧版被移成 `<root>_old`、新版落回 `<root>`）。6/6 成功，**120 集 / 62318 帧逐 repo 无损**。**未做但需记住**：转换脚本只处理标准字段，本仓的**旁挂负载通道（`sidecars/*.jsonl`）不在数据集内**，故不受影响、也未被转换。

76. **"0.6.1 支持多数据集"是错的（我先前的推断被冒烟推翻）**：`DatasetConfig.repo_id` 注释确实写着 "You may provide a list of datasets here. train.py creates them all and concatenates them"，`datasets/multi_dataset.py` 也存在，但 `datasets/factory.make_dataset` 的实际分支是 **`else: raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")`** —— 紧跟其后的 `MultiLeRobotDataset(...)` 构造是**死代码**。**lerobot 0.3.4 亦如此**（同处 `raise`）。故**多 repo 列表无法直接训练**：第一次 `--dataset.repo_id='["a","b"]'` 被整串当成一个 repo id（`HFValidationError: Repo id must be in the form ...`）。**唯一支持路径是 merge 成一个 repo**：`lerobot-edit-dataset --new_repo_id <out> --operation.type merge --operation.repo_ids "['a','b']"`（底层 `datasets/aggregate.py::aggregate_datasets`，会**校验 fps/robot_type/features 一致**、统一 tasks、拼接视频与 parquet）。**教训**：**注释与"模块存在"都不是功能证据**，必须读实际执行分支。

77. **SmolVLA 输入相机名与本仓不一致，必须 `--rename_map`**：`lerobot/smolvla_base` 的 `config.json` 里 `input_features` 是 `camera1/camera2/camera3`（3 路），本仓数据集是 `fixed / handeye`（2 路）→ `policies/utils.validate_visual_features_consistency` 报 `Feature mismatch ... Missing: camera1/2/3; Extra: fixed/handeye`（它要求两侧**互为子集**）。**修法**：`--rename_map='{"observation.images.fixed":"observation.images.camera1","observation.images.handeye":"observation.images.camera2"}'`。**关键机制**：`policies/factory.make_policy` 是 `if not rename_map: validate_visual_features_consistency(...)` —— **只要给了 rename_map，该校验整段跳过**（官方留的"你负责命名对齐"入口）。缺的第 3 路相机由 `modeling_smolvla.prepare_images` 处理：它对 `config.image_features` 中**不在 batch 里**的键按 `empty_cameras`（base 默认 **0**）补零图——为 0 时即**只喂 2 路图**；10 步冒烟证明此路径可训练。**未做**：把 `empty_cameras` 设成 1 去凑满 3 路（base 按 3 路训，补空图理论上更贴分布，但与"只喂实际 2 路"是两种策略）——首轮取最小改动，记为后续可试项。

78. **训练入口与参数口径修正（`run_vla_train.py` 对 0.6.1 原本是坏的）**：① 入口 `lerobot.scripts.train` **不存在**（0.6.1 为 `lerobot.scripts.lerobot_train`，控制台脚本 `lerobot-train`）；② 微调必须 `--policy.path=lerobot/smolvla_base`（`--policy.type=smolvla` 是"从零初始化"）；③ 顶层**没有** `--device`（由 Accelerate 自动探测，`--policy.device` 仅在设为 `cpu` 时用于强制 CPU）与 `--eval_freq`（对应项是 `--eval_steps`，默认 0=关，且需 `dataset.eval_split>0` 才生效）。**处置**：`src/run_vla_train.py` 改入口 / 用 `--policy.path` / 删 `--device`+`--eval_freq` / 把 `--device` 映射到 `--policy.device` / 新增 `--rename-map`（默认本仓映射）/ `--dataset-repo` 默认改 `demo_vla_all`；`src/vla_adapt.py::_build_smolvla_train_command` 同步修，并**改为从 `run_vla_train` 取同一套常量**（`TRAIN_MODULE`/`DEFAULT_POLICY_PATH`/`DEFAULT_RENAME_MAP`）避免两处漂移。新增 `src/tests/test_run_vla_train.py`（12 项）把**已实测跑通**的口径钉死。**一般化教训**：**第三方 CLI 的"看起来合理"的参数命名必须回源码 / `--help` 核实** —— 本项目已在 `--dataset.resume` vs `--resume`（偏差 72⑩）与 `--policy.type` vs `--policy.path` 上各栽一次。

### 2026-10-11 · 批次 38（S12 收口：跨版本 checkpoint 迁移）

79. **0.6.1 训出的 SmolVLA checkpoint 可在机器人侧 0.3.4 跑起来，但需要一次"迁移"（不是配置问题，是格式搬家）**。三道门禁逐层取证：① **config**：0.3.4 的 `PreTrainedConfig.from_pretrained` 对该 config.json 抛 `DecodingError`，因为 0.6.1 保存时多打了 6 个字段（`use_peft`/`pretrained_path`/`pretrained_revision`/`rtc_config`/`compile_model`/`compile_mode`）——**注意入口必须是 ChoiceRegistry 基类**：先用具体类 `SmolVLAConfig.from_pretrained` 会因 `type` 不是 dataclass 字段而误报，且需先 `import` smolvla 模块完成 `register_subclass`，否则报 "Couldn't find a choice class for 'smolvla'"；剪掉这 6 个后解析出的 `input_features` 与训练时**逐项一致**。② **权重结构**：离线读 safetensors 头（500 张量）显示**无任何 `rtc`/`compile` 键** → 那 6 个字段纯属"保存时多打的默认值"，不涉及权重；模块名与形状（`action_in_proj (720,32)`、`action_time_mlp_in (720,1440)`、`lm_expert`）与 0.3.4 源码逐项对得上。③ **真正加载**：`SmolVLAPolicy.from_pretrained` 首次即成功，**唯一差异是 6 个归一化 buffer**。**迁移三处**：(a) 剪 config 的 6 个字段；(b) **相机键改回 `fixed`/`handeye`** —— 训练侧用 `--rename_map` 把 `fixed/handeye` → `camera1/camera2`，而推理侧 `vla_prepost.smolvla_prepare` 的键就是 `fixed/handeye`，不改会导致 `prepare_images` 找不到任何图键而报错；因 VISUAL 走 IDENTITY 归一化、**不涉及权重**，改名零代价（只需保持 `camera1→fixed` 的顺序），同时删掉从未使用的 `camera3`（训练时该路本就缺失、`empty_cameras=0` 不补零图 → 行为等价）；(c) **回填归一化统计** —— 0.6.1 把 mean/std 从 policy 挪进了 `*_processor.safetensors`，而 0.3.4 的 `Normalize` 仍把它当 buffer 存在 `state_dict` 里、缺省为 `inf` 且 `forward` 内有 `assert not isinf` → 必须从 processor 文件回填（`observation.state`/`action` 的 mean/std）。回填后缺键 **6 → 0**。**产物** `pretrained_model_lr034/`（自包含）；**验证**：重新加载 57.3 s + **真实前向出动作**（有限值、非全零）。**环境**：原 `lerobot` conda 环境是采集专用（CPU torch、**无 transformers**），且 0.3.4 要 `transformers<4.52`、0.6.1 要 `transformers 5.x` —— **互斥**，故**新建独立环境**（`smolvla_infer`）而不是动采集环境。**绕开的坑**：`huggingface_hub` 拉 hf-mirror 会失败（该镜像对 resolve 返回 307 重定向、hub 客户端不跟），但 **`curl -L` 正常** → 用 `curl` 只取 SmolVLM2-500M 的**非权重文件**（config/tokenizer 13 个），权重本就在 checkpoint 里（`load_vlm_weights=false`）。**遗留**：CPU 推理 **10.2 s/步**（450M + 2×512² 图），300 步一集 ≈ 50 min → 效率不实用，需在本机 GPU / 云端策略服务 / 降分辨率之间选。**另注**：云端 `121.43.136.28` 的 **SSH 主机指纹已变**（`REMOTE HOST IDENTIFICATION HAS CHANGED`），该 IP 很可能已被回收/重建。

### 2026-10-11 · 批次 39（P2 4.1 不上机预检）

80. **推理链路通；同时暴露"相机后端 MSMF 全挂"这一环境级阻塞**。预检做法：连 COM6 取一次观测 → 用迁移后的 checkpoint 跑 `predict_action_chunk`（**只推理、不下发**）→ 与训练统计量对照。**三个正面结论**：① 观测流正确（`fixed` = 桌面俯拍、`handeye` = 夹爪特写，**逐张核图**确认），`gripper.load = 0‰`、`holds_object = False`；② 预测 `(50, 6)` chunk 的 **300 个值 100% 落在训练范围内**（状态与动作**两条范围都过**）—— 这一项同时验证了**相机通道正确**与**迁移时回填的归一化统计正确**（统计量填错或相机错配几乎必然越界）；③ **chunk 复用下摊销 0.605 s/步**（30.3 s / 50 步）→ **CPU 完全可用**，此前记的"10.2 s/步"只在 `--reset-per-step`（每步重推）下成立。**一个必须记住的环境坑**：`run_camera_check.py` 报 **MSMF 索引 0–4 全部打不开**、**DSHOW 0/1/2 可用**，且 DSHOW 的枚举顺序是 `0 = 笔记本摄像头 / 1 = fixed / 2 = handeye`，**与项目约定的 MSMF 顺序 `fixed=2 / handeye=0` 不同**。**关键判断**：约定真正要固定的是"**同一台物理相机 → 同一通道**"，索引只是运行时细节 —— 故用 DSHOW + 核图后的 `1/2` **不违反口径**；但**不能不核图就换**（那正是"笔记本摄像头被喂进腕部通道"那类静默错配的复现路径，而 `--verify` 只比位置、查不出画面错）。**影响**：采集链路强制 MSMF → **当前无法采集**；恢复靠既有处置（换 USB 口重插 + 重启 Windows），否则每次都需核图后改用 DSHOW 映射。**边界（如实记录）**：预检只证明"机制通"（观测→策略→分布内动作），**不证明策略好用**；首轮 chunk 幅度偏小（与当前位姿最大偏差 1.31），既可能是"起点附近的小幅调整"，也可能是**当前位姿位于训练分布边缘**（`shoulder_lift −98` 贴近 min `−100`、`elbow 100` 恰为 max）或模型训练不足 —— 需真机闭环才能分辨。**补充（重启 + 清场后复查）**：① **MSMF 全挂的根因不是系统/硬件，而是新环境的 OpenCV 构建** —— 同一时刻**采集环境（`opencv-python`）MSMF 0/1/2 正常**、**新环境（`opencv-python-headless`）0–4 全挂**；把新环境换成 `opencv-python==5.0.0.93` 后立即恢复。**教训**：lerobot 默认依赖的是 `opencv-python-headless`，而它在本机**没有 MSMF 能力** —— "同版本号 + headless" 不等于 "同能力"，跨环境对比（而不是只在新环境里反复重插/重启）才是最快的定位方式。② **相机索引已漂移**：核图确认当前 `0 = 笔记本摄像头 / 1 = fixed / 2 = handeye`，与历史约定 `fixed=2 / handeye=0` **正好错位**；沿用旧常数会让 `fixed` 拿到腕部图、`handeye` 拿到笔记本摄像头而**不报错**。③ **处置**：`run_vla_infer.py` 新增 `--fixed-cam/--handeye-cam` 显式覆盖（默认保留历史 2/0，上机必须传核图值），**刻意不改** `real_executor.DEFAULT_CAM_IDX` 与 `run_camera_check.CAMERA_SLOTS` —— 那两个常量的护栏语义是"同一台物理相机 → 同一通道"，索引只是运行时细节，写死反而在下次枚举漂移时再次误导。

### 2026-10-17 · 批次 40（P2 4.1 真机闭环首跑）

81. **真机闭环链路全通，但策略未完成"抬起"——归因前先排除了两个非模型因素**。首跑 300 步 → `success=False`（末 `wrist_flex=63.57`）；复盘发现 **300 步比 `pick_a` 最短的一集（340 帧）还短** → **步数预算本身就不够**，这一条先修（不是模型问题）。二跑 600 步，并给 `run_vla_infer.py` 的 `run_vla_loop` 接上 `on_step` 每 50 步落轨迹（否则只有末状态、无法判断卡在"下探/闭合/抬起"哪一段）：`wrist_flex` 71.9 → **61.3(step50)** → 67.6(150) → 72.4(400) → **72.9 平台(450–600)**，`gripper` 1.74→1.39。**再自查判据**：`pick_a` 20 集里 **18/20** 的 `wrist_flex` 峰值 ≥75（均值 **83.6**，68.2–95.7）→ `LIFT_WRIST=75` **不是过严**。**结论**：链路（归位 2 s/误差 2.38 → 观测 → 推理 → chunk 复用 → `max_relative_target=8` 限幅 → 收尾复位）全部可用且可观测；策略**只学会"下探 + 闭合"，缺"抬起"**，最可能是**训练不足**（20k 步 = **2.57 epoch**）。**方法论（本条最重要）**：归因顺序必须是"**先确认步数预算 ≥ 一集长度 → 再确认判据与示范一致 → 最后才归因模型**"；否则很容易把"预算/口径"问题误判成"策略不行"（反过来也会把模型问题说成口径问题）。**刻意未做**：没有为了"让它成功"去放宽 `LIFT_WRIST` 或增大 `max_relative_target` —— 那会把训练问题与评测口径混在一起，事后无法归因。

### 2026-10-17 · 批次 41（"不去找物体"的归因）

82. **现象级归因：先证输入有效、再证模型能复现、最后锁定相机取景**。用户观察到"机械臂不朝桌面物体运动"，三项对照实验层层排除：① **反事实敏感性**（固定同一状态、只改输入）——灰 `fixed` Δmean **10.5**、灰 `handeye` **21.4**、两路全灰 **23.6**、交换两路 **6.3**、换 place 指令 **16.9**、空指令 **19.2** → **图与语言都在明显驱动输出**，排除"输入没接上/被忽略"（且腕部相机权重更大）；② **训练样本 BC 自检**——在 `demo_vla_01_pick_a` 的样本上让策略预测，idx 100/150/250 的误差 **mean 0.28 / 2.25 / 1.09** → **模型能复现示范**，排除归一化统计、相机通道、权重加载等实现问题；唯一不复现的是 **idx=0（每集起点）**（示范在该帧是 `−100/100` 的大跳变），而闭环恰好从该状态开始；③ **取景量化**（静态地标模板匹配，避开机械臂遮挡）——实时 `fixed` 相对训练帧 **水平偏移约 45 px**（电源排插 `dx=+43`、下半幅 `dx=+48`，相关 0.63–0.75），亮度基本一致（148→152）→ **物体在图像中的像素位置变了，像素→动作映射随之失效**，与"不去找物体"完全吻合。**方法论**：现象归因应遵循「**先证输入有影响 → 再证模型能复现 → 最后才谈域差/训练不足**」，每一步都用可量化的对照；直接跳到"重训"会掩盖真正的部署问题。**尚未排除**：物体摆放超出示范分布、BC 闭环漂移（单步好 ≠ 闭环稳）、chunk 间 30 s 停顿改变时间轮廓。**补充（相机是否真被移动的硬判定）**：用 `pick_a` **20 集各自的第 0 帧**做基线——模板（电源排插）匹配位移全部 **dx=dy=0、std=0.0**（匹配度 0.996–1.000），说明**采集期间固定相机纹丝不动**；同一模板在实时帧上却是 **dx=+43 / dy=−42、匹配度仅 0.752** → **相机确实被移动/转动过**（约 4° 量级），不是模板匹配噪声。**注意**：换 USB 插口**不能**修复取景（它只改枚举索引，改了还会重新洗牌），要调的是**相机本体的位置与朝向**。

### 2026-10-17 · 批次 44（P2 4.1 真机闭环首次成功）

83. **真机闭环成功；并得到"取景偏差 → 行为模式"的完整对照**。相机二次微调后（静态地标 `dx +7/dy −4`）跑 600 步：**`success=是`、退出码 0、348 步提前结束**，末状态 `wrist_flex=79.59 / gripper=3.77`，无缺图/异常。行为链为 **下探 → 闭合保持（50–250 步）→ 抬起（300→348 越过 75）**。**三次运行构成一条干净的剂量-反应曲线**：偏差 **43 px → 夹爪全程闭合（1.39）、不抬**；**20 px → 张开到 16.9（接近+张爪）、不抬**；**~7 px → 闭合抓取并抬起、成功**。**结论**：① P2 4.1 真机闭环打通；② **相机取景是决定性变量，"任务失败"不能直接归因模型能力**；③ 单次成功不足以定成功率，需重复。**方法论（可复用）**：先用一个**可量化的环境标定**（静态地标 + 基线离散度）把"环境误差"压到与训练一致，再评估策略；否则任何"模型不行"的结论都不可信。

### 2026-10-18 · 批次 46（40k 重训与跨版本部署复核）

84. **40k 重训收官，跨版本迁移流程可复用（本轮实测）**：40000 步 / **20.54 epoch** / 末 loss **0.032**（20k 为 0.086）。三件事值得固化：① **新机无外网**时必须 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`，否则进程会静默卡在 HF 连接（GPU 0%、无 dataloader worker）；② **大文件回传**用远端 `split -b 50m` + 逐块 scp（带重试/超时/字节校验）+ 本地组装 + **sha256 对账**（`scp -r` 两次中途卡死，不可依赖）；③ 迁移脚本除 7 个产物文件外，**必须先拿到两个 `*_processor.safetensors`** —— 0.6.1 的归一化统计只在这里，缺了会在"回填 buffer"一步直接失败（主下载清单曾漏这两个文件）。迁移后重载 45.8 s、真实前向输出 6 维有限非零动作。**边界**：这仍只证明"可加载、能出动作"，不证明策略能力。当前 fixed 取景（桌面静态地标）**dx +15 / dy −6 px、score 0.960**，按批次 44 的剂量-反应仍偏大，闭环前应再收敛。

### 2026-10-18 · 批次 47（40k 权重首次真机闭环）

85. **40k 权重首次闭环即"抓取并抬起"，且这次的位置判据有图像佐证**：在第 203 步提前成功（`wrist 76.07 / grip 7.74`），行为链是"闭合→张开对准→接触（负载 +272）→再闭合到 7.74 并保持→抬起"；`handeye` 150/200 帧可见橙色手柄停在两指之间，`fixed` 帧里桌面上已看不到螺丝刀。与批次 43/44 的对照说明：**判据更新（位置为主）后，"夹住"的读数特征是 grip 停在 5.5–10 之间且长时间不变，而空夹是贴到 1.60–3.9 并顶限位**；这次没有出现假阳性。**同时暴露两个收尾缺陷（不影响任务判定，但会污染退出码）**：① `park()` 45 s 超时后仍继续；② `engine.close()` 在 `Torque_Enable=0` 上连试 6 次报 `Overload error!` → 进程 exit 1 覆盖了"任务成功"的返回码。**一般化教训**：**任务结果与进程健康应分开报告**——把摘要写在退出码之前、并把收尾异常捕获为告警，否则外部脚本会把"夹爪舵机过载"误读成"任务失败"（反之这次是"任务成功但 exit 1"）。

### 2026-10-18 · 批次 48（收尾复位口径：成功默认保持位姿）

86. **"成功默认不 park" 是评测口径的一部分，不是偷懒**：批次 47 成功后 `park()` 立刻把手臂带回 home，现场无法核验"是否真夹住"。新增 `should_park(success, no_park, park_on_success)`：`--no-park` 一律不复位 / **成功默认保持位姿** / **失败必复位**（停在"下探贴桌"姿态有风险，安全默认不能取消）/ 显式 `--park-on-success` 才在成功时复位。第 2 次闭环（同一取景、同一物体位置）**第 402 步成功**（`wrist 75.01 / grip 9.76`），日志确认走了"保持位姿"分支，`step0400_handeye` 可见手柄夹在两指之间。**当前 2/2 成功**（203 步 / 402 步），但两次时序差异大（SmolVLA flow matching 随机性）→ 仍需更多样本才能谈成功率。**未修（已知）**：`engine.close()` 的 `Torque_Enable=0` 过载仍使 exit 1，且因卸力矩失败、力矩保持 → 手臂不松手（对"保持位姿核验"反而有利）。

### 2026-10-18 · 批次 49（夹持判据补上界）

87. **"位置判据"必须是一条带，不能只有下界**：第 3 次闭环 `grip 19.25 + wrist 75.45` 被判成功，人工确认"张开、没夹住、抬起了"。根因是 `is_holding` 只判 `pos > empty_baseline + margin`，而**张开位置（16.9–22.7）远高于该阈值**——"张开"与"夹住"在下界侧完全同号。修法：`GRIP_HOLD_MAX = 15.0`（实测夹住带 6.7–9.8，留厚件余量），`is_holding` 判定 `5.5 < pos < 15.0`，所有调用方（`holds_object` / `real_perception` / `vlm_labeler`）自动继承。**方法论（第三次踩同类坑）**：**单侧阈值只能排除一种失败模式**；"抬起来了"与"夹住了"是两个独立条件，而夹爪位置这个量的**两种非夹持状态（空闭、张开）分布在阈值两侧**，所以必须用区间。后续若换厚物体，应重新标定上界而不是放宽。**统计更正**：第 3 次记失败 → **2/3**。

### 2026-10-18 · 批次 50（过载保护与归位口径）

88. **夹爪过载的根因有两个，分别处理才不互相打架**：① **归位阶段**——数据集第 0 帧的夹爪目标是 **0.56（闭合值）**，每次任务前归位都会把夹爪往闭合方向硬顶；② **策略段**——策略持续输出闭合指令而实际夹空时，舵机长时间顶限位。处置：归位新增 `skip_joints`（默认跳过 `gripper.pos`，既不下发也不参与到达判定）；策略段新增 `GripperTorqueGuard`（`|load|>150‰` 持续 **3 s** → 把**夹爪那一维目标**改写为当前实际位置，其余关节继续跟随；负载回落 1 s 才解除）。**关键设计**：冻结 ≠ 卸力——每步仍下发"当前位姿"，所以**已夹住的物体不会掉**，只是不再往里挤；总线读不到负载时已冻结则保持冻结（不能把"读不到"当成"安全"）。**代价（如实记录）**：跳过夹爪归位后，起始夹爪状态不再等于示范起点（0.56），属分布偏移，需真机验证是否影响成功率。**一般化教训**：**"保护"要作用在指令层而不是通信层**——直接卸力矩会让物体掉落，改写目标位姿才能在保住夹持的同时停止加力。

### 2026-10-18 · 批次 53–54（P2 4.5 的工程收口）

89. **适配对照的两个"看起来能跑"的坑，都必须先在代码层堵住**：① **训练侧**——`lerobot_train` 的 CLI 没有按模块冻结入口，直接调它会让 `head` 与 `full` 训练成同一个东西；修法是把 seam 放进进程内（patch `make_policy` → `apply_adaptation`），官方 `optimizers.py` 会按 `requires_grad` 过滤。**验证方式不是"跑通"，而是看 `num_learnable_params`**：修正前 head 只有 46,832（0.01%），补上 `action_time_mlp_{in,out}` 与 `state_proj` 后为 **1,635,152（0.363%）**。② **评测侧**——`run_vla_adapt.py` 的默认 `eval_fn` 走的是 **ACT checkpoint 加载器**，对 SmolVLA checkpoint 不成立；因此离线评测单独实现（`run_adapt_eval_offline.py`），并在报告里强制携带 `CAVEAT`：**离线指标只反映示范拟合质量，A6 必须真机实测**。**方法论**：**对照实验的"自变量是否真的生效"必须有独立读数**（这里就是 `num_learnable_params`），否则"跑完了"会把 `head≡full` 当成结论。
90. **跨版本迁移从"一次性脚本"升级为可复用工具，并用"逐字节复现"验证**：`src/migrate_vla_checkpoint.py` 固化三步（剪 6 字段 / 相机键改回 / 回填归一化 buffer），对缺失 `*_processor.safetensors` 明确报错（远端 split 下载最易漏）。**验证口径**：用同一 40k checkpoint 重新迁移，`config.json` 与 `model.safetensors` 的 sha256 与手工迁移产物**完全一致**——这比"能加载"更强，因为它同时排除了"少回填一个 buffer 但恰好不报错"的静默偏差。
91. **"任务成功但进程 exit 1"是评测口径污染，不是小瑕疵**：收尾 `engine.close()` 的舵机过载异常会把 `run_real` 的返回值覆盖成 1（批次 47/48 各出现一次）。修法 `safe_close`：收尾异常只告警。**一般化教训**：**结论通道（summary/报告）与进程健康通道（退出码）必须分离**——任何在 `finally` 里可能抛异常的资源释放，都不能位于"决定退出码"的返回路径上。**同类教训的对称面**：批次 58→59 的重复评测也说明，**"看到差异"必须先看方差**，否则会把采样噪声写成结论（head vs full 的 0.1 差值小于 ±0.19 的标准差）。

### 2026-10-18 · 批次 64（A7 离线泛化的评测协议）

92. **A7 从"仿真成功率"改为"同一模型在 held-out / seen 两侧的离线指标 gap"**：设计 §5 的 A7 走 `vla_generalization.evaluate_generalization`（仿真成功率），但本仓物体（螺丝刀/黑色长条/指甲刀）**没有对应的 ManiSkill 环境**（runner 把物体名当 `env_id`）→ 真实路径不可达。本轮改为：按同一 split 排除 held-out 组合的 episode 训练 `head`，再用 `run_adapt_eval_offline.py` 在两侧评 action L1 / 夹住 F1，**用 40k（两侧都见过）测出集合难度带**作为对照。**边界**：这是训练侧代理，仍不能替代 A6。
93. **稀疏采样会让"夹住 F1"退化为 0，而且看起来像"泛化失败"**：首轮用 stride 40 + 每侧 2 集，F1 在一侧为 0.000、另一侧 0.636。实测 GT：`ep41` 447 帧仅 **6 帧**在夹住带内、`ep1` 451 帧 6 帧，而按 stride 40 只抽到 **1 帧与 0 帧**；120 集整集夹住占比中位数 **5.8%**（1%–60%）。**修法**：新增 `select_frame_ids(..., even=True)` + CLI `--even`，在所选 episode 全区间均匀取帧；默认仍保持 A5 的 stride 口径（不改历史可比性）。**方法论**：**分类指标必须同时报告阳性数/占比**，"F1=0"在阳性为 0 时不是模型结论。
94. **A7 的结论被训练预算限制，而不是被泛化能力限制**：2000 步 head 的 L1（2.78–3.42）比 40k 参考行（0.67–0.90）差 3–4 倍，属整体欠拟合；`combo`/`novel_object` 的 gap（−0.23 / +0.09）落在 40k 测出的集合难度带（−0.23 … +0.14）内，**没观察到留出退化**；`paraphrase` 的 +0.62 虽显著，但该 split 同时留出了**任务类型**（本仓每种物体只有 2 条指令变体），**不能归因为措辞泛化**。**要测 A7 必须先把训练预算提上去**，否则测到的是"没学会"。
95. **"多训一会儿"不是万能解：head 路线在 8k 步出现明显收益递减**。combo 协议 2k→8k 的 held-out L1 只从 2.860 降到 2.566（−10%），seen 从 3.647 降到 2.817（−23%），仍是 40k 参考行的 3–4 倍；gap 从 −0.79 收敛到 −0.25（始终 held-out 更易）。这说明 0.36% 参数的 head 在这批数据上**接近容量限制**——A7 若要可解释结论，应改用 full 微调（或更多数据），而不是继续加步数。**方法论**：加预算的实验必须设**饱和判据**（画出曲线看斜率），否则"再训 4 倍"会变成没有信息量的时间消耗。
96. **batch 混淆的复核：结论不变，但"显著性"必须按配对比较重算**。A5 原表 head 用 bs32、full 用 bs16；把 head 也改成 bs16 后（同子集/同步数/8 次重复）：30 集 head 比 full 低 **0.252（3.6σ）**、60 集高 0.149（1.8σ，不显著）；夹住 F1 差 −0.045/−0.086（<2σ）。**两条读数**：① 30 集上 head 的 L1 优势是真的；② 夹住 F1 与 60 集仍无差异，所以离线指标不足以选上机模型。**方法论**：混淆因子（batch）消除后，原本"落在误差带内"的差异可能变成显著——**不能只看"是否显著"这一条结论，要报告配对后的效应量与 σ 倍数**。

### 2026-10-19 · 批次 66（真机 place 判据）

97. **"松开夹爪"不等于"放到左边"**：40k 权重在 `把螺丝刀放到左边` 上会**抓起来后原地松手**（4 次尝试的 pan 位移 0.0–4.7°），
   若沿用 pick 判据（夹住+抬起）会在抓取瞬间就报成功；即使改成"搬运后释放"，没有横向位移闸门仍会把
   "抓一下又松开"判成完成。故 `--success-mode place` 的判据是**夹住抬起 → 横向 ≥15° → 张开释放**。
   **数据侧根因**：`place_left` 的 20 集同一集里多次开合（首释放 pan 分别为 +12.8/+16.5/−5.2，而最终放置
   在 −29.6/−35.8/−45.2），语言条件下的策略学到的是"先松手"。**修数据（切段/清洗）比调判据有效**。
98. **位置带内的假阳性：grip 12.27 也可能没夹住**。两段式的 pick 复跑在 grip 12.27 触发成功，但随后搬运时
   物体掉落——夹持带 `5.5 < pos < 15` 只能说明"手指没闭合到底"，不能证明物体在钳口内；
   **判成功时应同时看负载与后续搬运是否真的保持**（本轮 `hold_load` 在 12.27 时为 False，本可提前发现）。
99. **归位容差 3.0° 对"重力下垂位"不可达**：`place_left` 第 0 帧的 shoulder_lift 目标 −0.72，实测稳定在 2.93
   （差 3.65°），`home_to_initial` 因此循环到超时、策略永远不启动（现象：手臂抬到位后不动）。
   改用 `pick_a` 的初始位（已验证可达，0.5–1.0 s）即可。**一般化**：归位门禁要么按关节设置差异化容差，
   要么改用"接近且不再变化"作为收敛条件，而不是所有关节统一 3°。




