# 第 13 周实施计划 — 基于 LeRobot 的低维 ACT 主策略

## 1. 任务分解

| 序号 | 子任务 | 文件/产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化 LeRobot 版本、18 维 feature 映射、chunk 和评测协议 | `docs/week13/design.md` | Week 11–12 |
| 2 | 建立 Python 3.12 独立环境并执行兼容性探针 | `requirements-lerobot.txt`、忽略的 `.venv-lerobot/` | 1 |
| 3 | 对 Week 12 数据加载/状态接口做最小公开化，不改变 BC 行为 | `src/behavior_cloning.py` | 1–2 |
| 4 | 实现状态拆分、chunk dataset、padding mask 和训练集统计量 | `src/lerobot_act.py` | 3 |
| 5 | 接入 LeRobot `ACTConfig`、`ACTPolicy` 和 processors | `src/lerobot_act.py` | 2、4 |
| 6 | 实现训练、validation、best checkpoint 和本地安全重载 | `src/lerobot_act.py` | 5 |
| 7 | 实现 ManiSkill ACT wrapper、action queue reset 和真实 rollout | `src/lerobot_act.py` | 5–6 |
| 8 | 实现训练/评测/三策略比较 CLI | `src/run_act_training.py` | 6–7 |
| 9 | 编写纯逻辑、fake env、LeRobot smoke 和错误路径测试 | `src/tests/test_lerobot_act.py` | 3–8 |
| 10 | 正式训练并执行 ACT/BC/scripted 的固定 seed 比较 | `runs/maniskill_act/week13/` | 2–9 |
| 11 | 回填真实结果、实际偏差并同步总开发计划 | `docs/week13/*`、总计划 | 10 |

## 2. 文件结构

### 2.1 计划新增

- `requirements-lerobot.txt` — Python 3.12 独立环境中的 ManiSkill、LeRobot 和 macOS Pinocchio 固定依赖。
- `src/lerobot_act.py` — 数据适配、LeRobot ACT、训练、checkpoint、policy wrapper 和评测。
- `src/run_act_training.py` — 第 13 周训练、固定评测和公平比较 CLI。
- `src/tests/test_lerobot_act.py` — 状态、chunk、ACT 集成、checkpoint、rollout 和错误路径测试。
- `docs/week13/design.md` — 本周设计与实际偏差。
- `docs/week13/plan.md` — 本实施计划与执行状态。

### 2.2 计划修改

- `src/behavior_cloning.py` — 仅公开 Week 12 已验证的数据加载、episode 切分和 18 维状态接口，供 ACT 复用。
- `具身智能完整项目_业余开发计划.md` — 正式验收后更新第 13 周状态、当前代码结构和下一步。
- `docs/week13/design.md`、`docs/week13/plan.md` — 根据真实版本、设备、训练和评测结果回填。

### 2.3 明确不修改

- Week 10 scripted policy 的状态机、阈值和控制模式；
- Week 11 正式 HDF5/JSON 数据；
- Week 12 BC 网络、checkpoint schema 和历史验收产物；
- RGB、LeRobotDataset、真实硬件、ROS 2、Hub 上传或 SmolVLA 相关代码。

## 3. 实施步骤

### 步骤 1：依赖与设备兼容性探针

1. 新建 Python 3.12 `.venv-lerobot`，不改动 `.venv-maniskill`。
2. 安装 `lerobot==0.6.1`、`mani_skill==3.0.1` 和 macOS `pin==4.1.0`。
3. 保存 Python、LeRobot、Torch、TorchVision、NumPy、Gymnasium、ManiSkill、SAPIEN 和 Pinocchio 版本。
4. 验证 MoltenVK 自动定位以及两个任务的 reset/step/close。
5. 构造 state-only `ACTConfig`，在 MPS 上执行单 batch forward/backward、保存和本地重载。
6. MPS 不可用时复现错误后转 CPU；至少一种 device 通过才能继续。
7. 执行 `.venv-lerobot/bin/python -m pip check`。

完成条件：依赖组合、device、feature shape 和 checkpoint API 均有真实运行证据，不仅是 import 成功。

### 步骤 2：公开 Week 12 稳定接口

1. 为数据加载、episode 切分和 `observation_to_state()` 建立明确的公共名字或稳定 wrapper。
2. 保留 18 维顺序、HDF5 强校验、env state 恢复路径和异常类型。
3. 不复制 Week 12 的 HDF5 解析和 observation 重建代码。
4. 先运行 `src/tests/test_behavior_cloning.py`，证明 BC 结果路径无行为变化。

完成条件：ACT 能复用同一 `TaskTransitions` 和 episode ID；BC 定向测试保持通过。

### 步骤 3：状态 feature 与 chunk dataset

1. 用命名索引把 18 维状态拆为 `observation.state[5]` 与 `observation.environment_state[13]`。
2. 增加逆向拼接校验，证明信息无丢失、无重复。
3. 为每个 transition 构造 `action[8,4]` 和 `action_is_pad[8]`。
4. 禁止 chunk 跨 episode，验证每个 episode 首尾样本。
5. 只从 80 个训练 episode 计算三类 feature 的 mean/std。
6. validation 只消费既有统计量，不回写 normalization。

完成条件：每任务样本数等于原 train/validation transition 数，所有 mask 和 episode 边界测试通过。

### 步骤 4：LeRobot ACT 接入

1. 配置 LeRobot `PolicyFeature`：STATE `(5,)`、ENV `(13,)`、ACTION `(4,)`。
2. 创建 `ACTConfig`：`n_obs_steps=1`、`chunk_size=8`、`n_action_steps=4`、VAE 开启、temporal ensemble 关闭。
3. 使用 `make_act_pre_post_processors()` 管理 normalization、batch 和 device。
4. 使用官方 `ACTPolicy.forward()` 计算 L1 + KL loss，不复制或改写模型内部实现。
5. 使用官方 optimizer preset；固定随机种子并记录 device。

完成条件：真实训练 batch 的 loss、梯度和 action chunk 全部有限，视觉输入为空且没有伪图像。

### 步骤 5：训练与 checkpoint

1. 每任务独立训练 20,000 step，batch size 64。
2. 每 250 step 计算完整 validation loss。
3. 只保存 validation 总 loss 最低的 checkpoint。
4. policy、preprocessor、postprocessor 和 metadata 写入同一任务的 `best/` 目录。
5. 从磁盘本地重载 policy 和 processors；内存中对象不能直接进入正式评测。
6. 空输出目录保护、临时文件和原子替换沿用 Week 12 原则。

完成条件：两个任务都有完整标准 checkpoint，缺任一文件时加载明确失败。

### 步骤 6：ACT policy wrapper 与 rollout

1. `LeRobotACTPolicy.reset()` 清空 LeRobot 内部 action queue。
2. `act()` 执行 live observation → 18D → 5/13D → processor → `select_action()` → postprocessor。
3. 每个输出硬校验 `(4,)`、finite、float32，并裁剪到 action space。
4. 使用原 `run_episode_in_env()` 的 success、terminated、truncated 和最大步数契约。
5. episode 异常隔离，始终关闭环境并落盘错误类型。

完成条件：fake env 验证 queue reset；真实两个任务各完成至少一个未见 seed rollout。

### 步骤 7：公平比较与门禁

1. 固定 seed `100–149`，每任务 50 次。
2. 在同一 checkout 和依赖环境内分别运行 ACT、BC、scripted policy。
3. 三者使用相同环境工厂、任务配置、最大步数和 success 判据。
4. ACT 禁止 scripted fallback；失败不能由 expert 接管。
5. 写出 `evaluation.json` 和 `comparison.json`，包括成功率、步数、时延、异常与失败 episode。
6. 门禁为 `ACT >= max(75%, 同轮 BC)`；若 BC 复现 92%，ACT 必须至少 92/100。

完成条件：100 次 ACT rollout 全部产生终态记录，三策略结果可逐 seed 对齐。

### 步骤 8：失败后的唯一诊断迭代

如果正式 ACT 未达到 BC：

1. 先按失败 episode 检查 action queue、动作陈旧和 chunk 后半段误差；
2. 保持数据、chunk size、模型和 optimizer 不变；
3. 只将 `n_action_steps` 从 4 改为 1，重新执行固定评测；
4. 在设计“实际偏差”中记录假设、结果和是否接受；
5. 若仍不通过，Week 13 标记部分完成，保留 BC/scripted 主线，不继续无界调参。

### 步骤 9：同步文档

1. 回填 `docs/week13/design.md` 的实际版本、device、参数、结果与偏差。
2. 更新本计划的复选框和执行状态。
3. 更新总计划第 13 周状态、已完成/未完成、当前代码结构和下一步。
4. 若 ACT 未达门禁，不得把第 14 周写成可直接开始。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| LeRobot 环境探针 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_lerobot_act.py -k compatibility` | import、MPS/CPU、forward/backward、save/load |
| Week 13 定向 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_lerobot_act.py` | feature、chunk、processor、训练、checkpoint、rollout、CLI |
| BC 回归 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_behavior_cloning.py` | 公共接口调整不改变 BC 行为 |
| 正式数据 smoke | `.venv-lerobot/bin/python src/run_act_training.py --training-steps 10 --eval-episodes-per-task 1 ...` | 正式 HDF5、训练、重载和真实环境最小闭环 |
| 正式训练与评测 | `.venv-lerobot/bin/python src/run_act_training.py --output-dir runs/maniskill_act/week13 ...` | 双任务 best checkpoint、100 次 ACT rollout、三策略比较 |
| 基础环境完整回归 | `.venv/bin/python -m pytest -q` | 第 1–12 周原功能无回归 |
| LeRobot 环境完整回归 | `.venv-lerobot/bin/python -m pytest -q --ignore=src/tests/test_llm_planner.py` | ManiSkill、HDF5、Torch、LeRobot 路径 |
| LLM 专属回归 | `.venv/bin/python -m pytest -q src/tests/test_llm_planner.py` | 可选 OpenAI 依赖路径保持通过 |
| 语法检查 | `.venv/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 依赖检查 | `.venv-maniskill/bin/python -m pip check` 与 `.venv-lerobot/bin/python -m pip check` | 两套环境各自依赖一致 |
| 差异检查 | `git diff --check` | 无空白和补丁格式错误 |

## 5. 单元测试覆盖

### 5.1 状态和数据

- 18 维到 5/13 维拆分及逆向无损校验；
- dtype、shape、NaN/Inf 和越界索引拒绝；
- train/validation episode ID 与 Week 12 一致；
- normalization 只使用训练 episode。

### 5.2 action chunk

- 完整 8 步 chunk；
- episode 尾部 1–7 步的 padding/mask；
- 单步 episode；
- chunk 不跨相邻 episode；
- 非有限 action 和非法 chunk/action-step 配置拒绝。

### 5.3 LeRobot 集成

- STATE/ENV/ACTION `PolicyFeature` 精确 shape；
- 无 image feature 时可 forward/backward；
- loss 和梯度有限；
- preprocessor normalization 与 postprocessor 反归一化；
- `save_pretrained()` / `from_pretrained()` 只从本地目录完成；
- 缺 config、权重或 processor state 时明确失败。

### 5.4 rollout 和比较

- 每 episode 调用 policy reset；
- action queue 消费 4 步后重新 query；
- action shape、finite、clip 和 dtype；
- 异常隔离、环境关闭及失败记录；
- seed 集合去重、非负校验和逐策略对齐；
- 门禁使用 `max(75%, 同轮 BC)`，不是固定 75%。

## 6. 验收清单

- [x] Python 3.12 独立环境通过 LeRobot + ManiSkill 兼容性探针。
- [x] MPS 或 CPU 至少一种 device 完成真实 forward/backward、保存和重载。
- [x] 18 维信息无损拆为 STATE 5 维和 ENV 13 维。
- [x] 8 步 action chunk、尾部 padding 和 mask 通过边界测试。
- [x] 复用 Week 12 的 80/20 episode 切分，无 transition 泄漏。
- [x] 每任务产生完整 LeRobot best checkpoint 和 processors。
- [x] checkpoint 从磁盘重载后完成 100 次 ACT 固定评测。
- [x] BC 与 scripted policy 在同轮、同 seed、同环境中完成比较。
- [ ] ACT 总体成功率 ≥同轮 BC 且 ≥75%（正式结果 78% < 92%，未达标）。
- [ ] 非有限/越界 action 为 0（非有限为 0；越界原始输出 1623/2302 被裁剪，主实验未达 0）。
- [x] 训练时长、峰值内存、推理时延和依赖版本落盘。
- [x] 定向测试、完整回归、`compileall`、`pip check` 和 `git diff --check` 通过。
- [x] 总计划、实际偏差和执行状态同步完成。

## 7. 执行状态

截至 2026-08-07 晚，功能代码与测试已完成，正式训练已启动。

### 已完成

- **步骤 1–3（设计、环境、接口）**：`requirements-lerobot.txt` 已创建；Homebrew 安装 Python 3.12.13 并创建 `.venv-lerobot/`（LeRobot 0.6.1、ManiSkill 3.0.1、Torch 2.11.0、NumPy 2.2.6、Gymnasium 1.3.0、SAPIEN 3.0.3、Pinocchio 4.1.0）；MPS 兼容性探针通过。`behavior_cloning.py` 以文档方式固化 `TaskTransitions`、`load_bc_dataset()`、`split_task_transitions()`、`observation_to_state()` 公共契约，BC 定向测试 10/10 保持通过。
- **步骤 4–8（状态/chunk/ACT/训练/wrapper/CLI）**：`src/lerobot_act.py` 实现 18→5/13 无损拆分、episode 内 action chunk 与 padding mask、官方 `ACTConfig`/`ACTPolicy`/pre/post processor、双任务训练与 best checkpoint、`LeRobotACTPolicy` rollout wrapper、`evaluate_lerobot_act()`/`compare_act_bc_scripted()`；`src/run_act_training.py` 实现训练、smoke、固定评测与三策略比较 CLI，以退出码表达门禁。
- **步骤 9（测试）**：`src/tests/test_lerobot_act.py` 共 32 项定向测试全部通过；LeRobot 环境相关回归 189 项通过；基础 `.venv` 全量回归 293 通过；`.venv-maniskill` 基线 157 通过；`compileall` 通过。
- **真实数据冒烟**：`--training-steps 10` 与 500 步稳定性训练均跑通（训练→smoke→三策略比较→门禁退出码），验证正式 HDF5、训练、重载和真实环境闭环；500 步后 PickCube validation loss 0.449、PushCube 0.270。

### 进行中 → 已结束

- **步骤 10（正式训练）**：20,000 步/任务、batch 64 的正式训练完成（`runs/maniskill_act/week13/`），PickCube best validation loss 0.130、PushCube 0.054。seed 100–149 固定评测：ACT 78/100、BC 92/100、scripted 99/100。
- **步骤 11（文档回填）**：`docs/week13/design.md` 已记录实际版本、设备、实现偏差与正式结果；总开发计划已同步为部分完成。

### 诊断迭代结论（2026-08-07，经 fix_plan 复核修正）

- ACT 未达门禁（`max(75%, 92%) = 92%`），正式结果 78%；失败全在 PickCube 的 50 步截断。
- 已诊断样本显示末端问题：cube 能进入目标区（0.025m）但到达时 arm qvel 高达 0.72–1.52（需 <0.2），静止时 cube 已离开目标区。
- 实验口径缺陷已确认：random seed 未在模型创建前生效、action normalization 随 chunk size 改变、checkpoint 由 validation loss 选择（40k 更差即由此而来，非"训练更久有害"）、诊断未覆盖 qvel/placed/static/chunk offset。
- 修正后的结论：**validation action loss 不能可靠选择 rollout checkpoint**；`chunk_size=4` 只删从不执行的 offset 4–7，不消除 offset 1–3 陈旧。
- 按 `docs/week13/fix_plan.md` 分阶段执行：先修口径（seed/normalization/单任务/周期 checkpoint/dev rollout 选择/诊断），再 corrected 8/4 control，再 4/4 消融，必要时 qvel 与 use_vae=False。详细协议见 design.md 10.8–10.14。

### 执行状态（fix_plan 各阶段）

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 阶段 0 | 保护现状（git/产物只读） | ✅ 完成 |
| 阶段 1 | 更新 design.md/plan.md（10.8–10.14） | ✅ 完成 |
| 阶段 2 | 修复 seed、normalization、单任务、周期 checkpoint、dev 选择、诊断字段 | ✅ 完成 |
| 阶段 3 | 单元测试（40 项）+ BC 回归 + 完整回归（197 项）+ 静态检查 | ✅ 完成 |
| 阶段 4 | corrected 8/4 control（仅 PickCube，dev 150–169） | ✅ 完成（dev **0/20**，best-validation loss 0.1207；同 seed 历史模型 13/20、BC 19/20） |
| 阶段 5 | chunk 4/4 消融（只改 chunk_size） | ✅ 完成（dev **0/20**，与 8/4 无差异 → 否定"未执行尾部是主要瓶颈"） |
| 阶段 6 前置 | 隔离 normalization vs seeding | ✅ 完成（`week13-exp-norm-chunk` 用 chunk 统计仍 0/20 → **0/20 归因 deterministic seeding init 敏感性**，seed 20260807 是坏 basin） |
| 阶段 6 | seed 选择协议（design.md 10.18）：VAE 5 seed 全坏 basin，no-VAE@20260807=6/20 → no-VAE 快速 seed 扫描 | ✅ 完成（no-VAE 最优 6/20） |
| 阶段 6a | no-VAE 热启动（design.md 10.21） | ✅ 完成（dev 12/20，首个可复现接近历史） |
| 阶段 6b | qvel + no-VAE + 热启动（design.md 10.22，fix_plan 阶段 6） | ✅ 完成（dev 13/20 持平历史；seed 100–149 22/50 < 历史 28/50） |
| 阶段 7 | use_vae=False 深入（按需） | 已并入 no-VAE 系列；判定无效 |
| 阶段 8 | final seeds 200–249 门禁 | ⛔ 未达 dev 门禁，按协议不执行；Week 13 维持 ⚠️ 部分完成 |

阶段 2 已实现的修复点：
- 7.1 `random.seed/np.random.seed/torch.manual_seed` 在模型创建前生效，每任务独立（`task_seed = random_seed + task_index`）。
- 7.2 新增 `compute_act_dataset_stats(data, train_indices)`，STATE/ENV/ACTION 均按唯一训练 transition 统计，chunk 4/8 统计一致（测试已验证旧接口随 chunk size 变化）。
- 7.3 `--task PickCube-v1/PushCube-v1/all`（默认 all），单任务只创建该任务目录。
- 7.4 `--checkpoint-interval` 周期保存 `step-NNNNN`，训练结束保存 `best-validation`。
- 7.5 `select_best_rollout_checkpoint` 在 dev seeds 上评估周期 checkpoint，按（成功数→成功平均步数→早 step）选出 `best-rollout-dev`，写 `rollout_selection.json`。
- 7.6 `run_act_diagnose.py` 逐帧新增 `is_obj_placed`、`is_robot_static`、`arm_qvel[7]`、`max_abs_arm_qvel`、`chunk_query_id`、`chunk_offset`、queue 长度；逐 episode 新增 `ever_placed/ever_static/first_placed_step/placed_steps/static_steps/simultaneously_valid_steps/max_qvel_while_placed`。
- checkpoint 加载支持 `tasks` 与 `checkpoint_subdir`，默认解析 `best-rollout-dev → best → best-validation`。

### 待验证

- 修正口径后的 ACT dev（150–169）成功率能否 ≥同轮 BC。
- corrected 8/4 vs 4/4 的 dev 对比；qvel 是否提升。
- final seeds 200–249 门禁：PickCube ACT ≥同轮 BC、总体 ≥max(75%, 同轮 BC)、PushCube 无回退。
