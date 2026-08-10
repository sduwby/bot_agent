# 第 16 周实施计划 — 视觉 ACT（单相机 RGB + proprioception）

## 1. 任务分解

| 序号 | 子任务 | 文件/产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化 VISUAL feature、分辨率、统计口径与 rollout 图像来源 | `docs/week16/design.md` | Week 13/15 |
| 2 | 确认 v3 数据集与 torchvision 依赖可用 | `pip check`、探针 | 1 |
| 3 | 实现 v3 读回（图像/状态/动作按任务分组） | `src/visual_act.py` | 2 |
| 4 | 实现含图像的 chunk 数据集与统计量 | `src/visual_act.py` | 3 |
| 5 | 实现 VISUAL+STATE+ENV ACTConfig 与 policy 创建 | `src/visual_act.py` | 4 |
| 6 | 实现训练、周期/best checkpoint、dev rollout 选择 | `src/visual_act.py` | 5 |
| 7 | 实现 VisualACTPolicy wrapper 与 live RGB rollout | `src/visual_act.py` | 5–6 |
| 8 | 实现显存/时长记录与固定评测 | `src/visual_act.py` | 7 |
| 9 | 实现 CLI | `src/run_visual_act.py` | 4–8 |
| 10 | 编写纯逻辑、fake env、真实 v3 smoke 与错误路径测试 | `src/tests/test_visual_act.py` | 3–9 |
| 11 | 真实训练（小分辨率 smoke + 正式）与 rollout | `runs/maniskill_act/week16-*/` | 2–10 |
| 12 | 回填真实结果、实际偏差并同步总开发计划 | `docs/week16/*`、总计划 | 11 |

## 2. 文件结构

### 2.1 计划新增

- `src/visual_act.py` — v3 读取、含图像 chunk 数据集、VISUAL ACT 训练、checkpoint、rollout、显存/时长报告。
- `src/run_visual_act.py` — 第 16 周 CLI。
- `src/tests/test_visual_act.py` — 第 16 周定向测试。
- `docs/week16/design.md` — 本周设计与实际偏差。
- `docs/week16/plan.md` — 本实施计划与执行状态。

### 2.2 计划修改

- `requirements-lerobot.txt` — 如新增依赖（torchvision 已随 lerobot 存在，预计无需修改）。
- `具身智能完整项目_业余开发计划.md` — 更新第 16 周状态、当前代码结构和下一步。

### 2.3 明确不修改

- Week 11 正式 HDF5/JSON 数据、Week 12 BC checkpoint、Week 13 ACT checkpoint；
- Week 15 `runs/maniskill_lerobot/week15-*/` v3 数据集；
- `lerobot_pipeline.py`、`lerobot_act.py`、`maniskill_intro.py` 的既有契约；
- ManiSkill 环境、成功条件或动作空间。

## 3. 实施步骤

### 步骤 1：确认前置

1. `runs/maniskill_lerobot/week15-cam/` 可被 `LeRobotDataset` 只读重开（200 episode / 3257 帧）。
2. torchvision resnet18 ImageNet 权重可加载。
3. `.venv-lerobot/bin/python -m pip check` 通过。

完成条件：v3 数据集与视觉依赖可用。

### 步骤 2：v3 读回与分组

1. `load_visual_dataset()` 读回 images/state/env_state/action/reward/success/done，按 `episode_index` 分组并按任务映射 env_id。
2. 图像缩放到 `resolution`（float[0,1]，CHW）。
3. 校验每帧 feature shape/dtype；缺失即异常。

完成条件：两任务各 100 episode 可读回。

### 步骤 3：chunk 数据集与统计

1. `build_visual_chunk_dataset()`：episode 内 8 步 chunk + pad mask，图像 (3,H,W)。
2. `compute_visual_act_stats()`：只从 train episode 计算 VISUAL（每通道 mean/std）、STATE/ENV/ACTION 统计。
3. 逆向校验 5/13 state 与原始 18D 一致。

完成条件：样本数与 train/validation transition 数一致；统计只来自 train。

### 步骤 4：配置与 policy

1. `make_visual_act_config()`：VISUAL(3,H,W) + STATE(5) + ENV(13) → ACTION(4)。
2. `make_visual_act_policy()`：`ACTPolicy` + `make_act_pre_post_processors`。
3. MPS forward/backward 探针（loss/梯度有限）。

完成条件：视觉 ACT 可训练。

### 步骤 5：训练与 checkpoint

1. 每任务独立 seed；周期 checkpoint + best-validation + dev rollout 选择。
2. metadata 记录架构（vision_backbone、resolution）、数据源、图像统计、loss、显存/时长。
3. 从磁盘重载 smoke rollout 后标记完成。

完成条件：两任务都有完整可重载 checkpoint。

### 步骤 6：rollout wrapper 与评测

1. `VisualACTPolicy.reset()` 清空 action queue。
2. `act(observation, image)`：live state → 5/13 + 图像 → select_action → post。
3. 固定 seed 评测：`make_render_env` → 逐帧 `set_state_dict → get_obs → state` + `render → image`。
4. 禁 scripted fallback；记录 success/steps/inference_ms/非有限/越界。

完成条件：真实 seed rollout 完成并落盘。

### 步骤 7：显存/时长报告

1. 训练：`resource.getrusage().ru_maxrss`（MPS/CPU）或 `torch.cuda.max_memory_allocated`。
2. 训练前后 RSS 差值、峰值、总时长、单 step 平均耗时。
3. 评测：每 episode 平均推理时延、总时长。
4. 写 `resource_report.json`。

完成条件：显存与时长记录齐全。

### 步骤 8：CLI

1. `run_visual_act.py` 提供训练、smoke、固定评测参数。
2. 提前解析绝对路径，避免 ManiSkill 改变 CWD。
3. 以退出码表达训练/评测是否通过。

完成条件：CLI 覆盖全流程。

### 步骤 9：测试

1. 纯逻辑：v3 读取、chunk、统计、配置、错误路径。
2. fake dataset/fake env：rollout、资源报告。
3. 真实 v3 smoke：小分辨率（64）、小 dim_model、每任务 1–2 episode。
4. 完整回归、`compileall`、`pip check`、`git diff --check`。

完成条件：定向测试通过；完整回归无回退。

### 步骤 10：真实训练与 rollout

1. 先 64×64 smoke 验证链路。
2. 正式训练 resnet18 + 224×224（若 M4 显存不足则降分辨率或 batch）。
3. 固定 seed rollout 与显存/时长报告。

完成条件：checkpoint、evaluation.json、resource_report.json 齐全。

### 步骤 11：同步文档

1. 回填 `docs/week16/design.md` 的真实版本、结果与偏差。
2. 更新本计划复选框与执行状态。
3. 更新总计划第 16 周状态、已完成/未完成、当前代码结构与下一步。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向测试 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_visual_act.py` | v3 读取、chunk、统计、配置、rollout、资源报告 |
| 真实 smoke | `.venv-lerobot/bin/python src/run_visual_act.py --resolution 64 --training-steps 5 --episodes-per-task 1 ...` | 真实 v3、训练、重载、真实环境最小闭环 |
| 完整回归 | `.venv-lerobot/bin/python -m pytest -q --ignore=src/tests/test_llm_planner.py` | 第 1–16 周无回归 |
| 语法检查 | `.venv-lerobot/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 依赖检查 | `.venv-lerobot/bin/python -m pip check` | 独立环境依赖一致 |
| 差异检查 | `git diff --check` | 无空白和补丁格式错误 |

## 5. 单元测试覆盖

### 5.1 v3 读取与分组

- 按任务、episode 分组；每帧 feature 齐全；
- 图像缩放 shape/dtype；缺失字段拒绝。

### 5.2 chunk 与统计

- 8 步 chunk 与 pad mask；不跨 episode；
- 统计只来自 train；VISUAL 每通道 shape (3,)；
- 5/13 逆向拼接一致。

### 5.3 配置与训练

- VISUAL/STATE/ENV/ACTION PolicyFeature 精确 shape；
- MPS 或 CPU forward/backward loss 有限；
- 周期/best checkpoint 保存；缺文件时加载失败。

### 5.4 rollout 与资源报告

- 每 episode reset 清空 queue；
- act(observation, image) 输出 shape/finite/clip；
- live 渲染 rollout 的异常隔离与环境关闭；
- RSS/显存与时长记录正确。

## 6. 验收清单

- [ ] 从 v3 读回图像/状态/动作并构造含图像 chunk 数据集。
- [ ] VISUAL/STATE/ENV/ACTION 统计只来自 train，图像 (3,H,W)。
- [ ] resnet18 + ACT 在 MPS 完成真实 forward/backward。
- [ ] 每任务完整 checkpoint 与 processors 可本地重载。
- [ ] 固定 seed live RGB 渲染 rollout 完成，无 scripted fallback。
- [ ] 显存、训练时长、单步推理时延全部落盘。
- [ ] 分辨率/batch 为受控参数并有记录。
- [ ] 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
- [ ] 运行产物只写入已忽略的 `runs/`。

## 7. 执行状态

### 已完成

- **步骤 1–2（前置与依赖）**：v3 数据集 `runs/maniskill_lerobot/week15-cam/` 只读重开正常；torchvision resnet18 ImageNet 权重可加载；`pip check` 通过。
- **步骤 3–4（v3 读回与 chunk）**：`load_visual_dataset()` 直接读回 `robot_state[5]` / `env_state[13]` / `images (3,H,W)` / `action[4]`；`build_visual_chunk_dataset()` 修复了 episode 边界越界切片并正确处理 pad；`compute_visual_act_stats()` 只从 train 统计 VISUAL 每通道 mean/std。
- **步骤 5–6（配置与训练）**：`make_visual_act_config()` / `make_visual_act_policy()`（VISUAL+STATE+ENV→ACTION）完成；MPS forward/backward 有限；训练、周期/best checkpoint、dev rollout 选择、metadata 完成。
- **步骤 7–8（rollout 与资源）**：`VisualACTPolicy` wrapper + `run_visual_episode_in_env()`（live RGB 渲染）完成；`write_resource_report()` 输出显存/时长（macOS `ru_maxrss` 字节→MB 修正）。
- **步骤 9（CLI 与测试）**：`run_visual_act.py` 提供完整 CLI（含架构参数）；`src/tests/test_visual_act.py` 11 项定向测试全部通过；完整回归 384 项通过；`compileall`、`pip check`、`git diff --check` 通过。
- **步骤 10（真实训练与 rollout）**：`runs/maniskill_act/week16-visual/` resnet18 + 128×128 + batch 8，1500 step/任务训练完成，checkpoint 可重载，20 episode live RGB rollout 完成，`resource_report.json`（峰值 RSS 2043.8 MB、训练 337.5 s、推理 150.2 ms/step）落盘。
- **步骤 11（文档同步）**：`docs/week16/design.md` 已回填实现偏差与结果；总计划已同步。

### 验收清单状态

- [x] 从 v3 读回图像/状态/动作并构造含图像 chunk 数据集。
- [x] VISUAL/STATE/ENV/ACTION 统计只来自 train，图像 (3,H,W)。
- [x] resnet18 + ACT 在 MPS 完成真实 forward/backward。
- [x] 每任务完整 checkpoint 与 processors 可本地重载。
- [x] 固定 seed live RGB 渲染 rollout 完成，无 scripted fallback。
- [x] 显存、训练时长、单步推理时延全部落盘。
- [x] 分辨率/batch 为受控参数并有记录。
- [x] 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
- [x] 运行产物只写入已忽略的 `runs/`。
