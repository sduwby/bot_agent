# 第 13 周 ACT 修复与受控实验完整计划

> 日期：2026-08-07  
> 状态：待执行  
> 适用范围：`/Users/fittenwby/bot_agent` 当前工作树  
> 核心原则：先修实验口径，再做单变量实验；以固定 rollout 成功率选择 checkpoint，不以 validation loss 代替控制效果。

## 1. 目标与边界

### 1.1 最终目标

修复当前 state-only ACT 的实验与控制问题，使其在新的固定未见 seed 上达到：

- PickCube ACT 成功率不低于同轮 BC；
- PushCube 保持稳定，不因 PickCube 修复而回退；
- 总体 ACT 成功率满足 `ACT >= max(75%, 同轮 BC)`；
- 禁止 scripted fallback；
- 所有 rollout 都产生完整终态、动作和失败分类记录；
- 不通过降低成功阈值、放宽静止判据或修改环境来达标。

### 1.2 本轮包含

- 修复训练随机种子和 checkpoint 选择；
- 解除 action normalization 与 chunk size 的耦合；
- 完善 PickCube 位置/静止/qvel/chunk-offset 诊断；
- 建立 corrected `chunk_size=8, n_action_steps=4` 控制组；
- 执行 `chunk_size=4, n_action_steps=4` 受控消融；
- 必要时为 ACT 增加 Panda arm qvel；
- 必要时测试 deterministic ACT、末端样本重加权或执行位置加权 loss；
- 同步 Week 13 设计、实施计划和总开发计划。

### 1.3 本轮不包含

- RGB、视觉 ACT、SmolVLA、语言条件或多任务 ACT；
- 修改 ManiSkill 成功条件；
- scripted assist、末端死区或规则接管；
- 修改 Week 11 原始数据或覆盖 Week 12/13 checkpoint；
- 无界超参数搜索、不断更换随机 seed 或盲目延长训练。

## 2. 当前基线与已有实验

当前正式 ACT 配置：

```text
chunk_size=8
n_action_steps=4
training_steps=20,000
batch_size=64
use_vae=True
latent_dim=32
kl_weight=10
device=mps
random_seed=20260807
```

seed `100–149` 的正式结果：

| 策略 | PickCube | PushCube | 总体 |
| --- | ---: | ---: | ---: |
| ACT | 28/50 | 50/50 | 78/100 |
| BC | 42/50 | 50/50 | 92/100 |
| Scripted | 49/50 | 50/50 | 99/100 |

已完成但未解决问题的尝试：

| 尝试 | 结果 | 当前判断 |
| --- | ---: | --- |
| `n_action_steps=1` | 总体 59% | 丢失连续抬升所需的 chunk 动量 |
| `n_action_steps=2` | 只改善少量失败 seed | 未解决末端高速穿越 |
| EMA 0.5 | 失败 | 接近阶段变慢，部分 episode 无法抓取 |
| EMA 0.8 | 只改善少量 seed | 不能稳定刹车 |
| 40k 重训 | PickCube 9/50 | validation loss 选出了 rollout 更差的 checkpoint |
| 直接延长训练 | 无效 | 当前 loss 不能可靠选择控制效果 |

## 3. 问题分析

### 3.1 控制目标与训练目标错配

PickCube 成功要求：

```text
||cube - goal|| <= 0.025
且
Panda arm qvel < 0.2
```

当前训练只最小化 action chunk 的归一化 L1 和 KL：

```text
loss = L1(predicted_action_chunk, expert_action_chunk) + KL
```

该目标没有直接表达：

- cube 是否停留在目标区；
- 机器人是否静止；
- 是否在同一仿真 step 同时满足位置与静止；
- 越过目标后的恢复能力。

20k checkpoint 的 PickCube `best_step=18250`、validation loss `0.130`、rollout `28/50`；40k 运行选择 `best_step=34250`、validation loss `0.123`，rollout 却降到 `9/50`。

20k 与 40k 运行的前 20k validation 曲线逐项一致。因此准确结论是：

> validation action loss 不能可靠选择 rollout checkpoint；现有证据不能简单推广成“训练更久必然有害”。

### 3.2 状态缺少成功判据需要的速度

当前 18 维状态包含：

```text
tcp_pos[3]
obj_pos[3]
goal_pos[3]
tcp_minus_obj[3]
obj_minus_goal[3]
gripper_qpos[2]
is_grasped[1]
```

它没有包含 `agent.qvel`。这造成部分可观测性：两个位置完全相同的状态，机器人可能一个正在高速上升，另一个已经静止；正确控制动作不同，但 ACT 会把它们视为同一个输入。

BC 每一步重新观察并重新输出动作，能够用高频闭环部分弥补；ACT 一次开环消费 4 个动作，速度不可观测的影响更大。

复用当前 checkpoint 对 5 个失败 seed 只读复核：

| seed | 进入目标区 | 目标区内最大 arm qvel | 同时满足位置和静止 |
| --- | ---: | ---: | ---: |
| 100 | 未进入，最小距离 0.02524 | — | 0 步 |
| 102 | 3 步 | 0.719 | 0 步 |
| 104 | 3 步 | 1.370 | 0 步 |
| 111 | 2 步 | 1.515 | 0 步 |
| 116 | 1 步 | 1.084 | 0 步 |

这些失败中，cube 能进入目标区，但到达时 qvel 远高于 0.2；机器人静止时 cube 已离开目标区。qvel 缺失是比单纯调 loss 更直接的结构问题。

### 3.3 chunk 开环动作陈旧

LeRobot `select_action()` 的实际逻辑是：action queue 为空时预测完整 chunk，只把前 `n_action_steps` 放入队列，后续环境 step 直接消费队列，队列耗尽后才重新读取 observation。

当前 `8/4` 配置：

```text
offset 0：使用当前 observation，立即执行
offset 1：执行时环境已前进 1 步
offset 2：执行时环境已前进 2 步
offset 3：执行时环境已前进 3 步
offset 4–7：参与训练，但从不执行
```

所以 `chunk_size=4, n_action_steps=4` 不会消除 offset 1–3 的陈旧，只会删掉从不执行的 offset 4–7。它仍不是 BC 式逐步闭环控制。

### 3.4 未执行的 chunk 尾部占用大量监督

PickCube 的 chunk 8 有效监督 token 中：

```text
前 4 位占 56.4%
后 4 位占 43.6%
```

当前 20k checkpoint 的 PickCube validation MAE：

```text
offset 0: 0.162
offset 1: 0.090
offset 2: 0.105
offset 3: 0.115
offset 4: 0.127
offset 5: 0.135
offset 6: 0.135
offset 7: 0.178
```

聚合后前 4 位 MAE 为 `0.119`，后 4 位为 `0.142`。删除尾部可能让模型集中拟合真正执行的位置，但 offset 0 本身仍是误差最大的位置，因此尾部不能预先认定为唯一根因。

### 3.5 normalization 随 chunk size 改变

当前 action normalization 从所有重叠 chunk 的有效 action 计算。同一条原始 action 会因为位于多个 chunk 中而被重复统计，且重复次数取决于 chunk size。

PickCube 训练集统计对比：

| 统计方式 | z mean | gripper mean |
| --- | ---: | ---: |
| 每个原始 transition 统计一次 | 0.0039 | 0.0053 |
| chunk size 4 重叠统计 | 0.0400 | -0.0805 |
| chunk size 8 重叠统计 | 0.0959 | -0.2204 |

直接从 chunk 8 改成 chunk 4 会同时改变 decoder/VAE 长度、loss 位置、padding 分布和 action mean/std，不是严格的单变量实验。

修复原则：

> STATE、ENV、ACTION normalization 都从训练集的唯一 transition 计算，每个 transition 只统计一次，不能从展开后的重叠 chunk 计算。

修复 normalization 后必须先重训 corrected `8/4` control，再与 corrected `4/4` 比较。

### 3.6 random seed 没有控制模型初始化

当前训练顺序是先创建 ACTPolicy 和 optimizer，再调用 `torch.manual_seed(task_seed)`。因此 `random_seed` 没有严格控制模型初始化，不同入口是否先运行 probe 也可能影响初始 RNG 状态。

必须改为：

```text
确定 task_seed
→ 设置 Python/NumPy/Torch seed
→ 创建模型
→ 创建 optimizer
→ 创建 batch generator
→ 开始训练
```

每个任务必须独立设置 seed，不能让 PickCube 的训练长度影响 PushCube 初始化。

### 3.7 末端样本稀少，但“多数为零动作”不准确

PickCube 共 1904 个 transition：

- `obj_goal <= 0.018`：151 帧，占 7.9%；
- 其中位置动作近似零：约 33%；
- 全数据中的零位置动作：约 2.6%。

准确描述是：近目标样本少，真正的静止零动作样本更少，但近目标样本并非多数都是零动作。

更重要的是，episode 达到 success 后立即结束，数据几乎没有“高速进入目标区后越过”、长时间振荡或重新抬升的 recovery state。过采样只能改变已有样本权重，不能创造缺失的恢复数据。

### 3.8 现有诊断覆盖不足

正式结果有 22 个 PickCube 失败，但现有逐帧诊断只覆盖少量 seed，而且没有直接记录：

- `info.is_obj_placed`；
- `info.is_robot_static`；
- `agent.qvel`；
- 当前动作的 chunk offset；
- action query 与动作实际执行时的 observation 差异。

所以“所有失败完全同因”仍需全量证据。

### 3.9 VAE 不是当前首要调参方向

当前 checkpoint 实测 KLD：

```text
PickCube ≈ 2.3e-5
PushCube ≈ 1.9e-5
```

乘以 `kl_weight=10` 后仍只有约 `2e-4`，远小于 L1，说明 latent 已接近 posterior collapse。提高 KL 不会合理地产生“更平滑保守”的控制，更可能进一步压制 latent。

如果后续要处理 VAE，应优先对比 `use_vae=False`，而不是继续增大 `kl_weight`。

## 4. 总体执行顺序

```text
保护现状并更新文档
  ↓
修复 seed、normalization、单任务训练、周期 checkpoint 和诊断
  ↓
建立 corrected chunk8/n4 控制组
  ↓
执行 chunk4/n4 单变量消融
  ↓
为 ACT 增加 arm qvel
  ↓
必要时测试 use_vae=False
  ↓
必要时执行一次末端重加权或执行位置加权
  ↓
锁定配置后才运行 final seeds
```

## 5. 阶段 0：保护现状

开始前执行：

```bash
cd /Users/fittenwby/bot_agent
git status --short
git diff --check
```

当前 Week 13 位于未提交工作树中，不得执行：

```bash
git reset --hard
git checkout -- .
rm -rf runs/maniskill_act
```

已有产物只读保留：

```text
runs/maniskill_act/week13/
runs/maniskill_act/week13-retrain40k/
runs/maniskill_act/week13-smoke/
runs/maniskill_act/week13-stability/
```

新实验使用新目录：

```text
runs/maniskill_act/week13-fix-control-c8/
runs/maniskill_act/week13-exp-c4/
runs/maniskill_act/week13-exp-qvel/
runs/maniskill_act/week13-exp-no-vae/
runs/maniskill_act/week13-final/
```

## 6. 阶段 1：先更新设计与实施计划

代码修改前，先更新：

```text
docs/week13/design.md
docs/week13/plan.md
```

设计文档至少新增：

```text
10.8 本轮复核后的根因修正
10.9 实验基础修复设计
10.10 corrected 8/4 control
10.11 chunk 4/4 消融
10.12 qvel 可观测性设计
10.13 checkpoint rollout 选择协议
10.14 新 dev/final seed 协议
```

同时纠正三处过强结论：

1. “延长训练有害”改为“validation loss checkpoint selection 不可靠”；
2. “末端样本多数是零动作”改为实际比例；
3. “所有失败相同”改为“已诊断样本表现一致，尚需全量验证”。

## 7. 阶段 2：修复实验基础

### 7.1 修复 deterministic seeding

修改 `src/lerobot_act.py`，保证 task seed 在模型实例化前生效：

```python
task_seed = config.random_seed + task_index

random.seed(task_seed)
np.random.seed(task_seed)
torch.manual_seed(task_seed)

act_config, policy, pre, post = make_lerobot_act_policy(...)
optimizer = ...
generator = torch.Generator().manual_seed(task_seed)
```

要求：

- 每个任务独立设置 seed；
- metadata 记录实际 task seed；
- 相同 seed 的初始模型权重 hash 相同；
- 不同 seed 的初始模型权重 hash 不同；
- PickCube 训练步数变化不影响 PushCube 初始权重。

MPS 不一定保证所有算子 bitwise deterministic，但模型初始化、batch 顺序和首轮指标必须可复核。

### 7.2 修复 normalization

新增或替换为按唯一 transition 统计的接口，例如：

```python
def compute_act_dataset_stats(
    data: TaskTransitions,
    train_indices: Sequence[int],
) -> dict[str, dict[str, np.ndarray]]:
    ...
```

要求：

- robot/env state 每个训练 transition 统计一次；
- action 每个训练 transition 的 expert action 统计一次；
- 同一 train split 下，chunk size 4 与 8 的 mean/std 完全相同；
- validation 不参与统计。

### 7.3 支持单任务训练

为 `src/run_act_training.py` 增加最小参数：

```text
--task PickCube-v1
--task PushCube-v1
--task all
```

默认保持 `all`。实验阶段只训练 PickCube，避免重复训练已经稳定的 PushCube。

### 7.4 周期保存 checkpoint

新增：

```text
--checkpoint-interval 2500
```

20k 训练保存：

```text
step-02500/
step-05000/
...
step-20000/
best-validation/
best-rollout-dev/
```

禁止只保留 validation loss 最低的模型。

### 7.5 rollout-dev checkpoint 选择

固定开发 seed：

```text
PickCube seeds 150–169，共 20 个 episode
```

每个周期 checkpoint 都在这 20 个 seed 上评估。选择规则固定为：

1. 成功数最高；
2. 成功数相同则成功 episode 平均步数更少；
3. 仍相同则选择训练 step 更早的 checkpoint。

validation loss 继续记录，但不参与最终 checkpoint 排名。禁止使用 final seeds 选 checkpoint。

### 7.6 完善诊断字段

修改 `src/run_act_diagnose.py`，每一步新增：

```text
is_obj_placed
is_robot_static
arm_qvel[7]
max_abs_arm_qvel
chunk_query_id
chunk_offset
queue_length_before_action
queue_length_after_action
raw_action
applied_action
obj_goal_distance
```

每个 episode 汇总：

```text
first_placed_step
placed_steps
static_steps
simultaneously_valid_steps
max_qvel_while_placed
ever_placed
ever_static
```

重新诊断全部 22 个历史 PickCube 失败，不只抽样 5 个。

## 8. 阶段 3：验证实验基础修复

定向测试：

```bash
PYTHONPATH=src .venv-lerobot/bin/python -m pytest -q \
  src/tests/test_lerobot_act.py
```

BC 回归：

```bash
PYTHONPATH=src .venv-lerobot/bin/python -m pytest -q \
  src/tests/test_behavior_cloning.py
```

完整回归与静态检查：

```bash
PYTHONPATH=src .venv-lerobot/bin/python -m pytest -q
.venv-lerobot/bin/python -m compileall -q src
.venv-lerobot/bin/python -m pip check
git diff --check
```

至少覆盖以下测试：

- seed 在模型创建前生效；
- 单任务训练不创建另一任务目录；
- normalization 不受 chunk size 影响；
- 周期 checkpoint 保存；
- rollout-dev checkpoint 排名；
- queue offset 和 qvel 诊断；
- 旧 checkpoint 加载兼容性；
- 非空输出目录拒绝覆盖。

## 9. 阶段 4：corrected `8/4` 控制组

### 9.1 配置

只训练 PickCube：

```text
chunk_size=8
n_action_steps=4
training_steps=20,000
batch_size=64
validation_interval=250
checkpoint_interval=2500
random_seed=20260807
use_vae=True
kl_weight=10
normalization=unique train transitions
```

输出目录：

```text
runs/maniskill_act/week13-fix-control-c8/
```

预期命令形式：

```bash
PYTHONPATH=src .venv-lerobot/bin/python src/run_act_training.py \
  --task PickCube-v1 \
  --chunk-size 8 \
  --n-action-steps 4 \
  --training-steps 20000 \
  --batch-size 64 \
  --validation-interval 250 \
  --checkpoint-interval 2500 \
  --random-seed 20260807 \
  --device mps \
  --dev-seed-start 150 \
  --dev-episodes 20 \
  --output-dir runs/maniskill_act/week13-fix-control-c8
```

实际 CLI 参数可在实现时按现有风格命名，但必须表达相同契约。

### 9.2 目的与决策

该运行是修复实验口径后的控制组，用于分离 normalization、seed 和 checkpoint 选择的贡献。

- corrected `8/4` 在 dev seeds 上不低于同轮 BC：锁定配置，跳过 chunk 4，进入 final gate；
- 未达到：继续 `4/4` 消融。

## 10. 阶段 5：`chunk_size=4, n_action_steps=4` 消融

相对 corrected `8/4` control，只允许修改：

```text
chunk_size: 8 → 4
```

必须保持不变：

```text
n_action_steps
train/validation episode split
random seed
normalization 统计方式
模型宽度和深度
optimizer 和 learning rate
batch size
training steps
checkpoint interval
VAE 配置
dev seeds
```

输出目录：

```text
runs/maniskill_act/week13-exp-c4/
```

比较指标：

```text
validation total loss
每个 chunk offset 的 L1
前 4 位聚合 L1
训练时长与峰值内存
rollout 成功数
进入目标区次数
进入目标区时 max arm qvel
同时满足 placed+static 的次数
平均步数
chunk query 数
动作裁剪率
```

dev 判定：

- 成功率达到或超过同轮 BC：锁定 `4/4`；
- 比 corrected `8/4` 至少多成功 `2/20`、但仍低于 BC：保留为当前最佳 chunk 配置，进入 qvel 实验；
- 提升少于 `2/20` 或更差：否定“尾部监督是主要瓶颈”，qvel 实验使用 corrected `8/4`。

不得因为 validation loss 更低就判定实验成功。

## 11. 阶段 6：加入 Panda arm qvel

### 11.1 输入设计

Panda `is_static(0.2)` 检查 arm joints 并排除最后两个 gripper joints，因此增加：

```text
arm_qvel = agent.qvel[:7]
```

新的 ACT robot state：

```text
tcp_pos[3]
gripper_qpos[2]
arm_qvel[7]
总计 12 维
```

environment state 保持 13 维：

```text
obj_pos[3]
goal_pos[3]
tcp_minus_obj[3]
obj_minus_goal[3]
is_grasped[1]
```

ACT 总状态为 25 维。不增加全部 qpos、gripper qvel、图像、object angular velocity 或 task ID。

### 11.2 数据兼容

Week 11 HDF5 已包含 Panda qvel，不需要补采数据。BC 继续使用原 18 维状态和原 checkpoint。

建议在 `TaskTransitions` 中增加可选字段：

```python
arm_qvel: np.ndarray | None = None
```

shape 为 `[T, 7]`。`load_bc_dataset()` 在重建 observation 时同时保存 arm qvel，BC 训练忽略它，ACT v2 才消费它。

### 11.3 checkpoint 契约

25 维 checkpoint 升级 schema，例如：

```text
maniskill_state_act_v2_qvel
```

加载时严格校验 schema、robot/env feature shape、qvel normalization 和 config。旧 18 维 checkpoint 不得静默加载到 25 维模型。

### 11.4 实验与决策

在 corrected `8/4` 和 `4/4` 中选择 dev 成功率更高者作为 chunk 配置，只增加 qvel。

输出：

```text
runs/maniskill_act/week13-exp-qvel/
```

- dev 成功率不低于 BC：锁定并进入 final gate；
- 提升至少 `2/20` 但仍未达 BC：进入 deterministic ACT；
- 几乎无提升：先核对 qvel 数据和 live observation 是否逐帧对齐，不做随机 seed 彩票。

## 12. 阶段 7：deterministic ACT

仅当 qvel 后仍未达到 dev 门禁时执行。

相对最佳 qvel 配置只修改：

```text
use_vae=True → False
```

输出：

```text
runs/maniskill_act/week13-exp-no-vae/
```

判定：

- dev 成功率提高至少 `2/20`：保留；
- 无提升：恢复 `use_vae=True`；
- 不测试更大的 `kl_weight`。

## 13. 阶段 8：末端样本或 loss 实验

只有前述方案仍未通过时执行，且下面方案二选一，不同时修改。

### 13.1 优先：近目标源状态过采样

定义：

```text
near_goal = ||obj-goal|| <= 0.05
```

对 near-goal source transition 使用固定 4 倍采样权重。

限制：

- 只改变 sampler；
- 不复制数据文件；
- 不改变 loss、chunk、VAE 或 qvel；
- 记录实际采样比例；
- dev 成功率没有至少 `2/20` 提升则撤销。

### 13.2 备选：执行位置加权 loss

仅适用于最终选择 chunk 8 的情况：

```text
offset 0–3: 1.0
offset 4–7: 0.25
padding: 0
```

使用 `policy.model()` 获取预测，在仓库训练循环中计算 weighted L1；KL 按官方公式保留，不修改 LeRobot site-packages。若最终使用 chunk 4，则不需要该实验。

### 13.3 终端静止正则

本轮不作为首选。只有 qvel、deterministic ACT、近目标过采样均无效后才考虑。

如果实施，应只在 `obj_goal <= 0.025` 时惩罚预测位置动作幅度，避免在目标区外压制必要的残差修正。

## 14. 最终评测协议

### 14.1 dev/final 分离

开发和 checkpoint 选择：

```text
PickCube seeds 150–169
```

最终未见测试集：

```text
PickCube/PushCube seeds 200–249
```

final seeds 在配置锁定前禁止运行。原 `100–149` 已用于多轮人工调参，只作为历史比较，不再视为无偏最终测试集。

### 14.2 最终同轮比较

锁定配置后，在 `200–249` 上执行 ACT、BC、Scripted，每任务 50 次。

如果 PickCube 使用新 checkpoint、PushCube 复用历史 checkpoint，必须记录两个任务各自的 checkpoint path、config 和 schema；PushCube 仍需在新 final seeds 上重新评估，不能直接沿用历史 `50/50`。

### 14.3 最终门禁

必须同时满足：

1. PickCube ACT ≥同轮 PickCube BC；
2. 总体 ACT ≥同轮总体 BC；
3. 总体 ACT ≥75%；
4. PushCube 不发生明显回退；
5. 100 个 ACT episode 全部有终态记录；
6. exceptions=0；
7. nonfinite action=0；
8. 禁止 scripted fallback；
9. checkpoint 由 dev rollout 选择，不使用 final；
10. 相同 checkpoint 对固定 5 个 seed 重跑时，成功/失败和终止步数一致，或能明确解释非确定性来源。

## 15. 停止条件

最多执行以下四类模型实验：

1. corrected `8/4` control；
2. `4/4`；
3. qvel；
4. no-VAE 或一次末端重加权。

禁止：

- **无界**换 random seed 碰运气；但有界、假设驱动、用 dev seeds 选择 seed 的协议允许（见 design.md 10.18，依据：deterministic seed 20260807 是坏 basin，dev 0/20；历史任意初始化是好 basin，dev 13/20）；
- 延长到 60k/80k；
- 提高 KL；
- 连续尝试多组 EMA；
- 加死区或 scripted assist；
- 使用 final seeds 反复挑配置；
- 同时修改 chunk、qvel、VAE、loss 和 sampler。

如果四类实验后仍低于 BC：

- Week 13 保持“⚠️ 部分完成”；
- ACT 不成为主策略；
- BC 继续作为 learned baseline；
- Scripted 继续作为专家和安全 fallback；
- 下一阶段转为补采 recovery/terminal trajectory 或 DAgger，不继续在当前 3257 个 transition 上无界调参。

## 16. 预计文件修改

必须修改：

```text
docs/week13/design.md
docs/week13/plan.md
src/lerobot_act.py
src/run_act_training.py
src/run_act_diagnose.py
src/tests/test_lerobot_act.py
具身智能完整项目_业余开发计划.md
```

qvel 阶段才修改：

```text
src/behavior_cloning.py
src/tests/test_behavior_cloning.py
```

明确不修改：

```text
runs/maniskill_expert/week11-200/*
runs/maniskill_bc/week12/*
src/maniskill_intro.py 中的 scripted policy
ManiSkill site-packages
环境 success threshold
```

## 17. 文档和证据同步

每完成一个实验，立即在 `docs/week13/design.md` 记录：

```text
实验假设
唯一修改变量
运行命令
Git commit/hash 或 dirty-worktree 说明
数据 summary/hash
checkpoint 路径
train/validation episode IDs
dev seeds
best-validation checkpoint
best-rollout checkpoint
per-offset loss
rollout 结果
placed/static/qvel 诊断
是否接受假设
下一步
```

只有最终 gate 通过后，才能把总计划第 13 周改成“✅ 完成”；否则保持“⚠️ 部分完成”。

## 18. 新对话启动指令

要求：

1. 开发前先把新的问题分析、实验协议和执行顺序补充到 docs/week13/design.md 与 docs/week13/plan.md。
2. 第一阶段只修实验基础：
   - random seed 必须在模型创建前设置；
   - action normalization 改为每个训练 transition 只统计一次，且不随 chunk size 改变；
   - 支持只训练 PickCube；
   - 每 2500 step 保存 checkpoint；
   - 用 seeds 150–169 的 rollout 成功率选择 checkpoint；
   - 诊断增加 is_obj_placed、is_robot_static、arm qvel 和 chunk offset。
3. 完成单元测试、BC 回归、完整 pytest、compileall、pip check 和 git diff --check。
4. 然后只训练 PickCube corrected chunk8/n4 control。
5. control 未达同轮 BC 时，再训练 chunk4/n4；除 chunk_size 外保持所有变量不变。
6. 如果 chunk4 仍未达标，为 ACT 增加 Panda arm_qvel[:7]，BC 继续保持原18维和原 checkpoint 兼容。
7. 所有模型只在 seeds 150–169 上开发和选 checkpoint；seeds 200–249 在最终配置锁定前禁止运行。
8. 不提高 KL、不做 seed 彩票、不加 scripted fallback、不覆盖历史 runs。
9. 每一步都提供代码路径、命令、checkpoint、指标和失败证据；偏离计划前先更新设计文档。
```
