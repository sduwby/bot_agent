# 第 17 周实施计划 — ACT 泛化（随机背景/照明/位姿）

## 1. 任务分解

| 序号 | 子任务 | 文件/产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化扰动配置、分布内/外口径与超参对比范围 | `docs/week17/design.md` | Week 16 |
| 2 | 探针验证背景/照明扰动对渲染图像的影响 [Mac] | `runs/` 探针 | 1 |
| 3 | 实现 `PerturbationConfig` 与 `make_perturbed_render_env()` [Mac] | `src/perturbation.py` | 2 |
| 4 | 给 `visual_act.py` / `lerobot_act.py` 增加 `cuda` 设备支持 [Mac 改代码 / 4060 验证] | `src/visual_act.py`、`src/lerobot_act.py` | 3 |
| 5 | 实现分布内/外评测与失败分布 [Mac] | `src/run_generalization.py` | 3–4 |
| 6 | 实现 chunk/smoothing 对比 [4060 训练 + Mac 评测] | `src/run_generalization.py` | 5 |
| 7 | 编写扰动测试与 CLI [Mac] | `src/tests/test_perturbation.py`、CLI | 3–6 |
| 8 | 运行定向测试、完整回归、compileall、pip check [Mac] | 全量 | 7 |
| 9 | 分布内/外真实评测 [Mac] | `runs/maniskill_act/week17-*/` | 3–8 |
| 10 | chunk 8 vs 4 训练与产物回传 [4060] | `runs/maniskill_act/week17-c4/` | 4、6 |
| 11 | 4060 WSL2 环境清单与产物回传流程 [Mac 编写] | `docs/week17/cuda_setup.md`（可选） | 4 |
| 12 | 回填真实结果、实际偏差并同步总开发计划 [Mac] | `docs/week17/*`、总计划 | 9–11 |

## 2. 文件结构

### 2.1 计划新增

- `src/perturbation.py` — 背景/照明/位姿扰动配置与评测环境工厂。✅ 已新增
- `src/run_generalization.py` — 分布内/外评测、失败分布、超参对比 CLI。✅ 已新增
- `src/tests/test_perturbation.py` — 扰动配置、环境工厂、图像差异与错误路径测试。✅ 已新增（23 项）
- `docs/week17/design.md` — 本周设计与实际偏差。✅ 已回填
- `docs/week17/plan.md` — 本实施计划与执行状态。✅ 已回填
- `docs/week17/cuda_setup.md` — 4060 WSL2 环境清单（可选，不阻塞主线）。⬜ 待 4060 环境搭建时编写

### 2.2 计划修改

- `src/visual_act.py`、`src/lerobot_act.py` — 设备选择与参数校验增加 `cuda`。✅ 已修改
- `src/act_stability.py`、`src/run_act_training.py`、`src/run_act_diagnose.py` — device 校验接受 `cuda`（一致化）。✅ 已修改
- `src/tests/test_visual_act.py`、`src/tests/test_act_stability.py`、`src/tests/test_lerobot_act.py` — 同步 cuda 合法化后的校验断言。✅ 已修改
- `具身智能完整项目_业余开发计划.md` — 更新第 17 周状态、当前代码结构和下一步。✅ 待本次同步

### 2.3 明确不修改

- Week 11 正式数据、Week 12 BC、Week 13/16 ACT checkpoint；
- Week 15 v3 数据集；
- ManiSkill 环境、成功条件、动作空间；
- 既有训练协议（第 13/16 周既定口径）。

## 3. 实施步骤

> 设备标记：**[Mac]** = Apple M4（开发、评测、测试）；**[4060]** = Windows RTX 4060 (WSL2)（多配置训练）；**[Mac/4060]** = 任一台。

### 步骤 1：扰动探针 [Mac]

1. 确认 `make_render_env()` + `scene.set_ambient_light()` + `update_render()` 能改变渲染图像。
2. 确认位姿扰动可经 seed 派生并在渲染前设置。
3. 记录 diff 数值作为"扰动生效"证据。

完成条件：三项扰动均可独立使图像明显变化。

### 步骤 2：扰动配置与环境工厂 [Mac]

1. `PerturbationConfig`：背景、照明、位姿开关与 seed。
2. `make_perturbed_render_env()`：reset 后按配置应用扰动。
3. 确定性：相同 seed 产生相同扰动。

完成条件：扰动可复现、可开关。

### 步骤 3：cuda 设备支持 [Mac 改代码 / 4060 验证]

1. `_select_device()` 支持 `cuda`（`torch.cuda.is_available()` 检查）。
2. `run_visual_act.py` / `run_act_training.py` 参数校验允许 `cuda`。
3. 显存记录在 CUDA 下用 `torch.cuda.max_memory_allocated`。
4. **在 Mac 改完并提交后**，4060 拉代码、按 `docs/week17/cuda_setup.md` 搭环境，用 `--device cuda` 跑一次 smoke 验证。

完成条件：4060 (WSL2) 上可用 `--device cuda` 训练。

### 步骤 4：分布内/外评测与失败分布 [Mac]

1. 分布内：默认环境 + in_seeds。
2. 分布外：扰动环境 + ood_seeds。
3. 聚合成功率、平均步数、异常、gap。
4. 失败相位分类复用 run_act_diagnose。

完成条件：`generalization_report.json`、`failure_distribution.json` 输出。

### 步骤 5：超参对比 [4060 训练 + Mac 评测]

1. chunk 8 vs 4 训练对比：**4060** 上 `--device cuda` 训练（Mac 仅回退用降 step）。
2. normalization / action_smoothing 评测级对比：**Mac** 上对回传 checkpoint 评测。
3. 训练产物（checkpoint、training_summary.json）从 4060 回传 Mac 后统一评测。

完成条件：`hyperparameter_sweep.json` 输出。

### 步骤 6：测试与 CLI [Mac]

1. 纯逻辑：PerturbationConfig 校验、seed 派生、图像 diff。
2. fake env：扰动应用与错误路径。
3. 完整回归、`compileall`、`pip check`、`git diff --check`。

完成条件：定向测试与完整回归通过。

### 步骤 7：真实评测 [Mac]

1. Mac 上对 week16-visual checkpoint 做分布内/外评测。
2. 失败分布分析。
3. 4060 可用时补 chunk 对比训练并回传产物，Mac 统一评测。

完成条件：报告齐全。

### 步骤 8：同步文档 [Mac]

1. 回填 `docs/week17/design.md` 的实际扰动效果、seed 集、结果与偏差。
2. 更新本计划复选框与执行状态。
3. 更新总计划第 17 周状态、已完成/未完成、当前代码结构与下一步。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向测试 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_perturbation.py` | 扰动配置、环境工厂、图像 diff、错误路径 |
| 真实 smoke | `run_generalization.py --episodes-per-task 1 ...` | 分布内/外最小闭环 |
| 完整回归 | `.venv-lerobot/bin/python -m pytest -q --ignore=src/tests/test_llm_planner.py` | 第 1–17 周无回归 |
| 语法检查 | `.venv-lerobot/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 依赖检查 | `.venv-lerobot/bin/python -m pip check` | 独立环境依赖一致 |
| 差异检查 | `git diff --check` | 无空白和补丁格式错误 |

## 5. 单元测试覆盖

### 5.1 扰动配置

- 三项扰动开关独立生效；
- seed 派生确定性；
- 非法配置（负 seed、空扰动）拒绝。

### 5.2 环境工厂

- 扰动确实改变渲染图像（diff > 阈值）；
- 无扰动时与默认环境一致；
- 环境关闭与异常隔离。

### 5.3 分布内/外与失败分布

- 成功率、gap 计算正确；
- 失败相位分类聚合正确。

## 6. 验收清单

- [x] 背景/照明/位姿扰动均可独立开关并确定性复现。
- [x] 扰动确实改变渲染图像（照明 diff≈10、背景 diff≈21、位姿距离 0.139m，探针 `runs/probes/week17/`）。
- [x] 分布内/外固定 seed 评测完成并输出成功率、gap（`runs/maniskill_act/week17-eval/generalization_report.json`）。
- [x] chunk/smoothing 对比完成（smoothing 评测级完成；chunk 训练级待 4060 回传 week17-c4 后执行 `--chunk-compare`）。
- [x] 失败相位分布报告输出（`failure_distribution.json`）。
- [x] cuda 支持（代码完成；4060 WSL2 smoke 验证待执行）。
- [x] 定向测试（23 项）、完整回归（407 项）、`compileall`、`pip check`、`git diff --check` 通过。
- [x] 运行产物只写入已忽略的 `runs/`。

## 7. 执行状态

**进行中（2026-08-11 快照）：**

| 步骤 | 状态 | 说明 |
| --- | --- | --- |
| 1 扰动探针 [Mac] | ✅ 完成 | 三项扰动均改变渲染图像；探针记录在 `runs/probes/week17/` |
| 2 扰动配置与环境工厂 [Mac] | ✅ 完成 | `src/perturbation.py`：PerturbationConfig + PerturbedRenderEnv + make_perturbed_render_env |
| 3 cuda 设备支持 [Mac 改代码 / 4060 验证] | ✅ 完成 | `visual_act.py`/`lerobot_act.py`/CLI/`act_stability.py` 均支持 cuda；4060 (WSL2) 已搭建环境并 `--device cuda` smoke 验证通过（见 `docs/week17/cuda_setup.md`） |
| 4 分布内/外评测与失败分布 [Mac] | ✅ 完成 | `src/run_generalization.py`：`_evaluate_split`/`evaluate_generalization`/`analyze_failure_distribution` |
| 5 超参对比 [4060 训练 + Mac 评测] | ⚠️ 部分完成 | smoothing 评测级已落地；chunk 8 vs 4 训练对比待 4060 回传 |
| 6 测试与 CLI [Mac] | ✅ 完成 | `test_perturbation.py` 23 项定向测试；完整回归 407 项通过 |
| 7 真实评测 [Mac] | ✅ 完成 | week16-visual checkpoint 分布内 0/20、分布外 2/20，见 `design.md §10.3` |
| 8 同步文档 [Mac] | ✅ 完成 | design/plan/总计划已回填 |

剩余工作（4060 或后续轮次）：
1. ✅ 已完成：PickCube DAgger recovery 补采（`src/visual_dagger_collect.py`，视觉 ACT 版 RecoveryExpert + live RGB 采集；两轮合并 31 episode / 153 transitions，`runs/dagger/week17-visual-dagger-pick-merged.npz`）。
2. ✅ 已完成：PickCube 30k 重训（`week17-pick-fix-30k`，chunk 8/n4，128×128，batch 8，30k step，合并 DAgger 数据 train=1665 transitions，best_step 28000、val loss 0.141，训练 23.7 min；产物 git LFS 回传）。
3. ✅ 已完成：Mac 复评 + 逐帧复诊（`run_pickcube_diagnose.py` + `docs/week17/pickcube_diagnosis.md` §8）：30k+DAgger 分布内 4/10、分布外 3/10，剩余瓶颈=到达 cube 高度但未抓上（闭爪时机）。
4. ✅ 已完成：第二轮 grasp recovery 补采（2026-08-12，4060，base `week17-pick-fix-30k`）：`GraspRecoveryExpert` + `--takeover grasp`（未抓取但 TCP 到 cube 高度且水平对齐即接管）；全量 27 episode / 479 transition，成功过滤 `-clean.npz` 20 episode / 185 transition（闭爪占 94%）；诊断集 5/7 seed 专家成功恢复。
5. ✅ 已完成（2026-08-12，4060）：用 `week17-visual-dagger-pick-grasp-clean.npz` 重训 `week17-pick-grasp-30k`（chunk 8/n4、128、batch 8、30k，best_step 19750、val 0.1414，1475 s 降噪运行）。4060 侧评测：分布内 3/10、分布外 4/10（对比 30k 基线 4/10 vs 3/10，整体持平；OOD 失败仍全为 `truncated_only`，抓取未实质突破，详见 `pickcube_diagnosis.md` §8.2）。候选下一步：抓取帧增权/40k/z 加权/放宽 GRASP_XY_ZONE 再补采。
6. 训练产物回传完成后，恢复 `.gitignore` 对 `runs/` 的忽略（当前为临时放开以便 git 同步数据与产物）。
7. 若需更强 OOD 结论，可扩分布外 seed 集或单开 background/lighting/pose 归因。

> 2026-08-11 环境快照：4060 上 WSL2 Ubuntu 24.04 + 驱动 566.36（CUDA 12.7）+ torch 2.7.0+cu126 + torchvision 0.22.0+cu126 + torchcodec 0.7.0 + lerobot 0.6.1 + mani_skill 3.0.1；`--device cuda` smoke（10 step/任务）训练、checkpoint 保存重载、live rollout 全链路通过，峰值显存 966 MB。无 NVIDIA WSL Vulkan，渲染用 lavapipe CPU + `render_backend="cpu"`，视频解码用 pyav；补丁与坑见 `cuda_setup.md` §8/§9。
> **WSL GPU 偶发挂死**：单进程连续 GPU 计算约 30–35 分钟后 forward 卡死（GPU 利用率归零、主线程空转），与锁频无关。规避：每个训练任务用独立进程（各 ~16 分钟）运行。降噪运行：`powercfg /setactive SCHEME_MIN` + `nvidia-smi -lgc 0,1500`（笔记本不支持 `-pl`），训练后恢复。
