# 第 17 周设计文档 — ACT 泛化（随机背景/照明/位姿）

## 1. 概述

**本周目标：** 在第 16 周视觉 ACT（resnet18 + 128×128，`runs/maniskill_act/week16-visual/`）之上，验证视觉 ACT 的泛化能力：随机背景、照明和物体初始位姿；调节 chunk、归一化和动作平滑；报告分布内/外成功率和失败分布。

总计划原文：

> 第 17 周 | ACT 泛化 | 随机背景、照明和物体初始位姿；调节 chunk、归一化和动作平滑。 | 报告分布内/外成功率和失败分布。

### 1.1 前置依赖

- 第 16 周视觉 ACT checkpoint：`runs/maniskill_act/week16-visual/{PickCube,PushCube}-v1/best-validation/`（resnet18、128×128、chunk 8 / n_action_steps 4）。
- 第 16 周 `visual_act.py`：`VisualACTPolicy`、`evaluate_visual_act()`、`load_visual_act_checkpoints()`、`make_render_env()`。
- 第 15 周 v3 数据集：`runs/maniskill_lerobot/week15-cam/`（训练数据源，含 512×512 相机帧）。
- ManiSkill 场景可运行时改光照（已探针验证：`set_ambient_light()` + `update_render()` 可显著改变渲染图像）。
- 可选 NVIDIA RTX 4060（WSL2）用于多配置训练加速；本周文档同时覆盖 Mac 单机与 4060 协作两种执行路径。

### 1.2 本周关键决策

1. **"分布内" = 与训练数据同分布的评测场景**：即第 16 周固定 seed 评测的默认环境。
2. **"分布外" = 三项受控扰动叠加或单开**：
   - 背景：地面/桌面颜色或纹理变化；
   - 照明：`set_ambient_light()` 改变环境光颜色/强度 + 可选点光源；
   - 物体初始位姿：沿用第 14 周 seed 集外扩（更大位姿范围）并渲染到相机域。
   三者均可独立开关，以便归因。
3. **扰动在环境工厂层注入，不改 ManiSkill 任务**：新增 `make_perturbed_render_env()` 包装 `make_render_env()`，在 `reset` 后按扰动配置调用 `set_ambient_light()` / `update_render()`。
4. **超参调节只做受控单变量对比**：本周对 chunk_size（8 vs 4）做训练级对比；归一化与 action_smoothing 做**评测级**调节（不改训练数据统计量），避免重训组合爆炸。
5. **训练在 Mac 与 4060 两条路径均可执行**：默认继续用 Mac M4（慢但无需额外环境）；若 4060 (WSL2) 就绪，则用 `--device cuda` 训练多配置，产物回传 Mac 做统一评测。
6. **报告以分布内/外成功率和失败分布为核心**，不追求本周达到成功率门禁（视觉 ACT ≥60% 是第 22 周门禁）。

### 1.3 范围边界

本周包含：

- 随机背景/照明/位姿的评测环境工厂与扰动配置；
- 分布内/外固定 seed 评测；
- chunk 8 vs 4 训练对比（Mac 或 4060）；
- 归一化、action_smoothing 的评测级调节；
- 失败分布报告（复用 run_act_diagnose 思路）；
- 4060 (WSL2) 的 cuda 支持与环境清单。

本周不包含：

- 多相机、深度、语言条件、SmolVLA、世界模型；
- 真实硬件、ROS 2、Hub 上传；
- 修改第 15/16 周数据、checkpoint 或既有训练协议；
- 视觉 ACT 门禁判定（第 22 周）；
- 修改 ManiSkill 成功条件或动作空间。

## 2. 版本与设备设计

### 2.1 Mac M4（默认）

- 复用 `.venv-lerobot`（Python 3.12 + LeRobot 0.6.1 + ManiSkill 3.0.1 + `lerobot[dataset]`）。
- MPS 训练：128×128 + batch 8 约 0.68 s/step。多配置训练需控制 step 数（如 3–5k step/配置）。
- 评测与失败分布在本机完成。

### 2.2 NVIDIA RTX 4060（WSL2，可选加速）

- WSL2 Ubuntu 安装：CUDA 版 PyTorch（如 `pip install torch --index-url https://download.pytorch.org/whl/cu121`）、`mani_skill`、`lerobot[dataset]`、`h5py`、`pillow`。
- 训练用 `--device cuda`；仿真仍 `physx_cpu`（渲染与 rollout 不需 GPU）。
- 产物（checkpoint、training_summary）从 4060 回传 Mac `runs/maniskill_act/`，用 Mac 做统一评测。
- 需要在 Mac 记录每次训练的命令与配置，确保 4060 与 Mac 训练口径一致。

### 2.3 设备要求

- 记录 device、峰值 RSS/显存、训练总时长、单 step 推理时延。
- CUDA 显存用 `torch.cuda.max_memory_allocated`；Mac 用 RSS。

## 2.3 执行分工矩阵（Mac vs Windows 4060）

> 约定：代码仓库以 Mac 为源，`git` 同步到 4060。**所有代码改动在 Mac 完成并提交**；4060 只跑训练，产物回传 Mac 做统一评测。仿真始终 `physx_cpu`（Mac 与 4060 都不需要 GPU 仿真/渲染）。

| 环节 | 设备 | 说明 |
| --- | --- | --- |
| 写代码 / 改 bug / 测试 | Mac M4 | 唯一开发入口；所有 `src/`、`tests/` 改动在 Mac 完成并 `git` 提交 |
| 定向测试 / 完整回归 / compileall | Mac M4 | `.venv-lerobot` 全量验证 |
| 扰动探针（背景/照明/位姿） | Mac M4 | 验证 `set_ambient_light` 影响渲染图像 |
| 分布内/外**评测** | Mac M4 | 对已有 checkpoint 跑 `evaluate_visual_act()`，产出报告 |
| 失败分布分析 | Mac M4 | 复用 `run_act_diagnose` 思路 |
| chunk 8 vs 4 **训练**（多配置） | Windows 4060 (WSL2) | `--device cuda`；Mac 上太慢（M4 0.68 s/step） |
| 训练产物回传 | 4060 → Mac | checkpoint、`training_summary.json` 拷回 `runs/maniskill_act/` |
| 4060 环境搭建 | Windows 4060 (WSL2) | 按 `docs/week17/cuda_setup.md` |
| cuda 支持代码 | Mac M4 | 设备选择/参数校验/显存记录改动在此完成 |

> **回退路径**：4060 未就绪时，chunk 对比可在 Mac 上降低 step 数（如 3–5k）跑，仅作趋势结论，不追求收敛。

## 3. 架构设计

```text
visual_act.py（训练，Mac 或 4060）
   ├─ chunk_size=8  → week16-visual（已有）
   └─ chunk_size=4  → week17-c4
        │
        ▼
perturbation.py（新增，评测环境工厂）
   PerturbationConfig{background, lighting, pose}
   make_perturbed_render_env(config)  →  reset 后 set_ambient_light / update_render
        │
        ▼
visual_act.evaluate_visual_act(seed_set, perturbed_env_factory)
   ├─ 分布内（默认环境）
   └─ 分布外（随机背景/照明/位姿）
        │
        ▼
failure_report.py（复用 run_act_diagnose 思路）
   失败相位分类 → 分布内/外失败分布
        │
        ▼
runs/maniskill_act/week17-<tag>/
   ├─ generalization_report.json（分布内/外成功率）
   ├─ failure_distribution.json
   └─ hyperparameter_sweep.json（chunk/smoothing 对比）
```

### 3.1 模块职责

新增 `src/perturbation.py`：

- `PerturbationConfig`：背景、照明、位姿扰动参数与开关；
- `make_perturbed_render_env()`：包装 `make_render_env()`，在 reset 后应用扰动；
- 背景/照明应用与位姿扰动的确定性实现（可按 seed 派生扰动）。

新增 `src/run_generalization.py`：

- 生成分布内/外评测配置与 seed 集；
- 调用 `evaluate_visual_act()` 分别在分布内/外评测；
- 失败相位分析；
- 输出 `generalization_report.json`、`failure_distribution.json`。

新增 `src/tests/test_perturbation.py`：

- 扰动配置校验、环境工厂、图像差异、错误路径。

新增 `docs/week17/cuda_setup.md`（4060 WSL2 环境清单，可选）。

复用（不修改或仅最小修改）：

- `src/visual_act.py`：训练、`VisualACTPolicy`、`evaluate_visual_act()`；
- `src/lerobot_pipeline.py`：`make_render_env()`、`_image_hwc()`；
- `src/run_act_diagnose.py`：失败相位诊断。

## 4. 数据结构

### 4.1 扰动配置

```python
@dataclass(frozen=True)
class PerturbationConfig:
    env_id: str
    seed: int = 0
    background: bool = False          # 是否随机化背景
    lighting: bool = False            # 是否随机化照明
    pose: bool = False                # 是否随机化物体初始位姿
    max_steps: int = 50
```

背景/照明/位姿扰动都可通过 seed 派生确定性参数，保证可复现。

### 4.2 分布内/外评测配置

```python
@dataclass(frozen=True)
class GeneralizationEvalConfig:
    checkpoint_dir: Path              # 视觉 ACT checkpoint（如 week16-visual）
    output_dir: Path
    device: str = "mps"
    resolution: int = 128
    in_distribution_seeds: tuple[int, ...] = (500, ...)
    out_of_distribution_seeds: tuple[int, ...] = (600, ...)
    perturbation: PerturbationConfig | None = None
```

### 4.3 报告结构

```python
@dataclass(frozen=True)
class GeneralizationReport:
    schema_version: str
    in_distribution: dict[str, object]   # 每任务成功率/步数/异常
    out_of_distribution: dict[str, object]
    gap: dict[str, float]                # 分布内 - 分布外（按任务）
    failures: dict[str, object]          # 失败相位分布
    hyperparameters: dict[str, object]   # chunk/smoothing 对比
```

## 5. 接口设计

```python
def make_perturbed_render_env(
    config: ManiSkillTaskConfig,
    perturbation: PerturbationConfig,
) -> object:
    """创建带背景/照明/位姿扰动的渲染环境。"""


def evaluate_generalization(
    config: GeneralizationEvalConfig,
) -> GeneralizationReport:
    """分布内/外评测 + 失败分布 + 超参对比。"""


def analyze_failure_distribution(
    output_dir: Path,
    *,
    seeds: Sequence[int],
    device: str,
    resolution: int,
    env_factory: Callable,
) -> dict[str, object]:
    """复用 run_act_diagnose 对失败 episode 分类。"""


def run_hyperparameter_compare(
    config: GeneralizationEvalConfig,
    *,
    chunk_candidates: Sequence[int],
    smoothing_candidates: Sequence[float],
) -> dict[str, object]:
    """chunk/smoothing 的评测级对比。"""
```

## 6. 数据流

### 6.1 扰动应用

```text
make_perturbed_render_env(config, perturb)
  → make_render_env(config)
  → env.reset(seed)
  → perturb.background: 修改地面/桌面视觉属性
  → perturb.lighting: set_ambient_light(color) + update_render()
  → perturb.pose: 由 seed 派生并设置物体/goal 位姿
  → 返回 env
```

### 6.2 分布内/外评测

1. 分布内：`evaluate_visual_act(checkpoint, in_seeds, default_env_factory)`。
2. 分布外：`evaluate_visual_act(checkpoint, ood_seeds, perturbed_env_factory)`。
3. 聚合成功率、平均步数、异常、非有限动作，计算 gap。

### 6.3 失败分布

对分布外失败 episode 复用 `run_act_diagnose.diagnose_episode()`（或轻量复刻）逐帧分类，输出相位计数。

### 6.4 超参对比

- chunk 8 vs 4：训练级对比（4060 优先，Mac 可降 step）。
- normalization / smoothing：评测级对比（同一 checkpoint 换 processor/smoothing 参数）。

## 7. 失败处理与降级

| 失败 | 处理 |
| --- | --- |
| 背景/照明扰动不影响图像（diff≈0） | 拒绝评测，报告扰动未生效 |
| CUDA 不可用但配置要求 cuda | 明确错误并回退 mps/cpu |
| checkpoint 缺失 | 拒绝评测 |
| 扰动 seed 与分布内 seed 重叠 | 拒绝 |
| 4060 产物回传后 Mac 无法重载 | 记录 torch/lerobot 版本不一致并提示 |

## 8. 验收标准

1. 背景/照明/位姿扰动均可独立开关并确定性复现。
2. 扰动确实改变渲染图像（图像 diff 显著）。
3. 分布内/外固定 seed 评测完成并输出成功率、gap。
4. chunk/smoothing 对比完成（训练级或评测级）。
5. 失败相位分布报告输出。
6. cuda 支持与 4060 WSL2 环境清单文档化（可选路径）。
7. 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
8. 运行产物只写入已忽略的 `runs/`。

## 9. 参考资料

- 第 16 周 `docs/week16/design.md`（视觉 ACT 训练与 rollout）
- 第 14 周 `docs/week14/design.md`（位姿随机化与固定 seed 协议）
- 第 13 周 `docs/week13/design.md`（失败相位诊断）
- ManiSkill `ManiSkillScene.set_ambient_light` / `update_render`

## 10. 实际偏差

### 10.1 实现偏差（开发后回填）

| 设计项 | 设计 | 实际 |
| --- | --- | --- |
| `PerturbationConfig` 字段 | `env_id/seed/max_steps` | 移除这三个字段；seed 直接取自 `ManiSkillTaskConfig`，颜色由 episode seed 经固定 RNG 派生，新增 `background_colors`/`ambient_colors` 调色板 |
| 背景扰动实现 | 修改地面/桌面材质颜色 | ground 自带 grid texture 覆盖 `base_color`，且 `RenderShapeTriangleMesh.material` 无 setter；改为**新增彩色背景面板**（`add_box_visual` + `build_static`），持有 `RenderMaterial` 引用并在每次 reset 时 `set_base_color` 换色 |
| 位姿扰动实现 | 由 seed 派生并设置物体/goal 位姿 | 沿用第 14 周 seed 语义：分布外 seed 集（600–609）本身产生更广位姿；不额外修改场景物体位姿 |
| 分布内/外评测 | 复用 `evaluate_visual_act()` | 新增 `_evaluate_split()`（`run_generalization.py`）复用 `load_visual_act_checkpoints` + `run_visual_episode_in_env`，避免把 `evaluation.json` 写进 checkpoint 目录 |
| 失败分布 | 复用 `run_act_diagnose.diagnose_episode` | 该接口面向 state-only `LeRobotACTPolicy`；改为在 rollout 中捕获 `is_grasped/is_obj_placed/is_robot_static` 信号，复用 `act_stability.classify_failure_diagnosis` 分类规则 |
| 超参对比 | chunk 训练级 + smoothing 评测级 | smoothing 评测级已落地；chunk 对比待 4060 回传 week17-c4 checkpoint 后运行 `--chunk-compare` |

### 10.2 扰动探针结果（Mac M4，2026-08-11）

探针脚本：`runs/probes/week17/perturbation_probe.py`；结果：`runs/probes/week17/probe_result.json`。

| 扰动 | 方法 | 平均绝对像素差 | 结论 |
| --- | --- | --- | --- |
| 照明 | `set_ambient_light([0.9,0.3,0.3])` + `update_render()` | 10.598（红）/ 9.181（绿） | 有效 |
| 背景 | 新增彩色面板（红/蓝） | 21.268 / 20.360 | 有效 |
| 位姿 | seed 600 vs 601 物体位姿 | 距离 0.139 m | 有效 |

### 10.3 真实评测结果（Mac M4，week16-visual checkpoint，seed 集 500–509 / 600–609）

完整报告：`runs/maniskill_act/week17-eval/generalization_report.json`。

| 指标 | 分布内 | 分布外 | gap |
| --- | --- | --- | --- |
| 总体 | 0/20（0.0%） | 2/20（10.0%） | −10.0pp |
| PickCube-v1 | 0/10（0.0%） | 0/10（0.0%） | 0 |
| PushCube-v1 | 0/10（0.0%） | 2/10（20.0%） | −20.0pp |

- 失败分布（分布外）：PickCube 全为 `truncated_only`（9）+ `grasped_not_placed`（1）；PushCube 全为 `truncated_only`（8）。
- action_smoothing 评测级扫描（分布外）：1.0→10%、0.9→10%、0.75→5%、0.5→10%，波动在噪声范围内。
- 成功率为 0–10% 属预期：week16 视觉 ACT 仅训练 1500 step，本周验收为泛化评测管线，视觉 ACT ≥60% 门禁在第 22 周。
- chunk 对比待 4060（WSL2）训练 week17-c4 并回传后执行。

### 10.4 cuda 支持（Mac 改代码 / 4060 验证）

- `visual_act.py` / `lerobot_act.py` 的 `_select_device` 支持 `cuda`（不可用时明确报错，不再静默回退 cpu）；显存记录在 cuda 下改用 `torch.cuda.max_memory_allocated`。
- `VisualACTTrainingConfig`、`StabilityEvalConfig`、`run_act_training.py`、`run_act_diagnose.py` 的参数校验均接受 `cuda`。
- 4060 (WSL2) 环境已搭建并 smoke 通过（2026-08-11）：WSL2 Ubuntu 24.04 + 驱动 566.36（CUDA 12.7）+ torch 2.7.0+cu126 + lerobot 0.6.1 + mani_skill 3.0.1，`--device cuda` 训练 10 step/任务、checkpoint 保存重载、live rollout 全链路通过，峰值显存 960 MB。
- 环境要点（详见 `docs/week17/cuda_setup.md`）：lerobot 0.6.1 要求 `torch>=2.7`，需驱动 ≥ CUDA 12.6（原 536.67 仅 CUDA 12.2）；本机 NVIDIA 驱动缺 WSL Vulkan 集成文件，以 `sitecustomize_4060.py` 强制 ManiSkill `render_backend="cpu"`（lavapipe 软件 Vulkan）+ lerobot 视频解码走 `pyav`（torchcodec 0.7.0 在此组合 `_convert_to_tensor` op 无实现）。

### 10.5 20k step 训练结果与 WSL GPU 稳定性（2026-08-11）

**训练量提升**：应需求把 chunk 对比训练量从 1500 提到 20k step（week17-c4-20k / week17-c8-20k，commit `196d34c`，git LFS）。

| 运行 | chunk | 训练 | live rollout（固定 seed 500–509） |
| --- | --- | --- | --- |
| week17-c4（1500 step） | 4 | 206 s | 3/20（PickCube 0/10、PushCube 3/10） |
| week17-c4-20k | 4 | 1834 s | **15/20（PickCube 6/10、PushCube 9/10）** |
| week17-c8-20k | 8 | 分任务进程 | 训练完成（PickCube best 13750、PushCube best 13000），rollout 待 Mac `--chunk-compare` 统一评测 |

- 20k 训练显著提升：chunk 4 成功率 15% → 75%，验证训练量是关键瓶颈（第 22 周门禁已接近）。
- **WSL GPU 偶发挂死**：单进程连续 GPU 计算约 30–35 分钟后 `policy.forward` 卡死（GPU 利用率归零、主线程空转、独立 CUDA 进程健康），与锁频无关（锁频/全频均复现，C4-20k 第二次运行侥幸通过）。规避：**每任务独立进程**（各约 16 分钟）运行，未再复现。

> 开发中所有运行产物均写入已忽略的 `runs/`，未提交仓库。
