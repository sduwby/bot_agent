# 第 12 周设计文档 — 低维 Behavior Cloning 训练与独立评测

## 1. 概述

**本周目标：** 使用第 11 周已验收的 `PickCube-v1`、`PushCube-v1` 成功轨迹，完成 episode 级数据切分、低维状态归一化、小型 MLP Behavior Cloning、checkpoint 保存/加载和未见 seed 独立 rollout，验证从数据到策略评测的最小闭环。

总计划原文：

> 第 12 周 | 低维 BC | 用 state + action 训练小型 Behavior Cloning，验证数据、归一化、checkpoint 和 rollout 全链路。 | 独立评测成功率 ≥70%。

### 前置依赖

- 第 11 周正式数据位于已忽略的 `runs/maniskill_expert/week11-200/`，两个任务各 100 条成功轨迹、合计 3257 个 transition，200/200 静态检查和 50/50 action replay 已通过。
- 复用 `.venv-maniskill` 的 Python 3.11、ManiSkill 3.0.1、PyTorch 2.13.0、HDF5 和 macOS CPU 仿真环境。
- 复用第 10 周的 `ManiSkillTaskConfig`、环境工厂和 Gymnasium 契约，不修改 scripted expert。

### 数据现状与处理边界

实查第 11 周 HDF5 后发现：`actions` 和 `env_states` 完整，但 `RecordEpisode` 包在 `CPUGymWrapper` 外层时把 `obs` 的每个向量叶子保存成了每帧一个标量。第 11 周的长度检查只能证明 `T+1` 对齐，不能证明 observation 的叶子宽度完整。

第 12 周不伪造缺失 observation，也不重新采集或覆盖第 11 周正式数据。加载器先从完整 `env_states` 提取 28 维基础物理状态，交叉校验已保存 observation 的首元素与 Panda/cube/goal state 布局一致；随后逐帧恢复 env state 并调用 ManiSkill `get_obs()` 重建完整 live observation。最终训练特征只保留与 scripted expert 决策直接一致的 18 维任务状态：

- `tcp_pos[3]`、`obj_pos[3]`、`goal_pos[3]`；
- `tcp_pos - obj_pos[3]`、`obj_pos - goal_pos[3]`；
- 两个 gripper qpos；
- `is_grasped`（PushCube 缺少该字段时固定为 false）。

HDF5 中 Panda articulation state 的 `[13:22]`、`[22:31]` 分别映射到 qpos、qvel；cube actor 的前 7 维映射到 obj pose；`goal_site`（PickCube）或 `goal_region`（PushCube）的前 3 维映射到 goal position。加载器对这些 slice、T/T+1、重建后的单环境 batch 维和最终 18 维 shape 做硬校验，防止静默使用错误布局。

### 范围边界

- 每个任务独立训练一个同结构 MLP，避免把任务 ID 或多任务路由提前带入第 13 周。
- 按 episode 划分 80 条训练、20 条验证，不按 transition 随机切分，避免相邻状态泄漏。
- 独立评测使用训练数据之外的 seed `100–149`，每任务 50 次、总计 100 次。
- checkpoint 和评测 JSON 写入已忽略的 `runs/maniskill_bc/`，不提交模型二进制。
- 不做 RGB、ACT、action chunk、DAgger、数据补采或超参数搜索框架。

## 2. 架构设计

```text
run_bc_training.py
    │
    ├── BCTrainingConfig             数据切分、网络、优化与复现参数
    ├── load_bc_dataset()            env state 重建 → 18 维任务 state + action
    ├── train_behavior_cloning()     每任务归一化、MLP 训练、验证和 checkpoint
    ├── load_bc_checkpoint()         安全加载模型、统计量与数据契约
    └── evaluate_bc_checkpoint()     未见 seed 的真实 CPU rollout
            ├── BCPolicy             live state_dict → action
            └── evaluation.json      每 episode 与任务/总体成功率
```

新增 `src/behavior_cloning.py` 承担纯逻辑、训练和 rollout 编排；新增 `src/run_bc_training.py` 作为正式训练及评测入口。运行产物与源码分离。

## 3. 数据结构

### 3.1 训练配置

```python
@dataclass(frozen=True)
class BCTrainingConfig:
    dataset_summary: Path
    output_dir: Path
    train_episodes_per_task: int = 80
    validation_episodes_per_task: int = 20
    hidden_dims: tuple[int, ...] = (128, 128)
    epochs: int = 400
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    random_seed: int = 20260807
```

输出目录必须为空；训练/验证 episode 总数不能超过每个任务可用数量；所有数值和 shape 均在训练前校验。

### 3.2 任务数据

```python
@dataclass(frozen=True)
class TaskTransitions:
    env_id: str
    episode_ids: tuple[int, ...]
    episode_lengths: tuple[int, ...]
    states: np.ndarray       # [N, 18]
    actions: np.ndarray      # [N, 4]
```

数据按 metadata episode 顺序确定性读取。切分发生在 episode ID 层；转换为 transition 后才进入 `DataLoader`。

### 3.3 checkpoint

`checkpoint.pt` 只保存基础 Python 容器和 tensor：

- schema/version、创建时间与训练配置；
- 数据集 summary 的解析后绝对路径及数据契约；
- 每任务网络 `state_dict`；
- 每任务 state mean/std 与 action mean/std；
- train/validation episode IDs、transition 数量、最终/最佳 loss；
- 输入维度、动作维度和隐藏层结构。

加载使用 `torch.load(..., weights_only=True)`，并重新校验 schema、任务集合和 tensor shape。

### 3.4 评测摘要

`evaluation.json` 包含 checkpoint、seed 范围、每条 episode 的 success/步数/reward/终止状态，以及每任务和总体成功率。门禁以总体 100 次独立 rollout 的成功率计算，同时报告两个任务各自结果，避免总体平均掩盖单任务失效。

## 4. 接口设计

```python
def load_bc_dataset(
    summary_path: Path,
    *,
    observation_env_factory: Callable = make_maniskill_env,
) -> dict[str, TaskTransitions]: ...

def train_behavior_cloning(
    config: BCTrainingConfig,
    *,
    observation_env_factory: Callable = make_maniskill_env,
) -> BCTrainingSummary: ...

def load_bc_checkpoint(
    checkpoint_path: Path,
) -> dict[str, BCPolicy]: ...

def evaluate_bc_checkpoint(
    checkpoint_path: Path,
    *,
    seeds: Sequence[int],
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
) -> BCEvaluationSummary: ...
```

CLI：

```bash
.venv-maniskill/bin/python src/run_bc_training.py \
  --dataset-summary runs/maniskill_expert/week11-200/summary.json \
  --output-dir runs/maniskill_bc/week12 \
  --eval-seed-start 100 \
  --eval-episodes-per-task 50
```

正式验收不得跳过评测。训练或评测任一任务异常、checkpoint 不完整或总体成功率低于 70% 时返回非零。

## 5. 数据流

1. 读取并验证第 11 周 summary、metadata 和 HDF5 文件。
2. 按任务确定性选择前 80 个 episode 训练、后 20 个验证。
3. 从每个 episode 的完整 env state 重建 observation，生成 18 维任务 state，并与前 T 帧、T 个 action 对齐。
4. 仅用训练集计算 mean/std；近常量维使用 epsilon 防止除零。
5. 归一化 state/action，分别训练两个 MLP；按验证 MSE 保留最佳权重。
6. 原子保存 checkpoint 与机器可读训练摘要。
7. 重新从 checkpoint 构造 `BCPolicy`，不复用内存中的训练模型。
8. 对 seed 100–149 创建真实 CPU 环境并 rollout，动作反归一化后裁剪到 action space。
9. 写入 `evaluation.json`，计算分任务与总体成功率并应用 70% 门禁。

## 6. 异常与安全边界

| 场景 | 行为 |
| --- | --- |
| 第 11 周 summary/文件缺失 | 拒绝训练并指出路径 |
| env state 布局、actor、重建 observation 或维度不符 | 拒绝训练，禁止静默 reshape |
| episode/action/T+1 错位 | 拒绝训练 |
| train/validation episode 重叠 | 拒绝训练 |
| 标准差为零或极小 | 使用固定 epsilon，不产生 NaN |
| 输出目录非空 | 拒绝覆盖 checkpoint 和历史评测 |
| checkpoint schema/shape 错误 | 拒绝加载 |
| policy action 非有限或越界 | rollout 记失败；有限动作仅裁剪到合法边界 |
| 单 episode 环境异常 | 记录失败原因并关闭环境，继续其余固定评测 |

## 7. 测试与验收

### 自动测试

- 配置边界、非空目录保护和 episode 级无泄漏切分。
- HDF5 env state → 完整 observation → 18 维 live state 的字段映射、shape、有限数和 T/T+1 对齐。
- 归一化只使用训练集且可逆；常量维不产生非有限数。
- checkpoint 保存后重新加载，网络输出与保存前一致。
- fake environment rollout 的 reset seed、action shape、资源关闭和成功率汇总。
- CLI 退出码和低于 70% 门禁。

### 正式验收

1. 两个任务均使用 80 train / 20 validation episode，episode ID 无重叠。
2. checkpoint 由重新加载路径完成真实 rollout。
3. seed 100–149 每任务 50 次，总计 100 次独立评测。
4. 总体成功率至少 70%，并报告两个任务各自成功率。
5. checkpoint、训练摘要和评测摘要只写入 `runs/`。
6. 定向测试、完整回归、`compileall`、`pip check` 和差异检查通过。

## 8. 实际偏差

- 初版仅直接使用 env state 中可映射的 qpos、qvel、物体 pose 和目标位置，共 28 维。150 epoch、seed 100–109 的 smoke 中 PushCube 为 10/10，PickCube 仅 2/10；闭环诊断显示 PickCube 能抓住物体，但缺少精确 TCP/相对位置后在搬运途中停在目标之外，总体仅 60%。该结果未作为验收产物。
- 加入完整 TCP 和 `is_grasped` 后，保留全部 qpos/qvel/pose 的 36 维中间方案在相同 smoke 上仍只有 PickCube 1/10。闭环动作表明问题不在“能否抓住”，而在抓取后相对目标误差没有被小数据 MLP 稳定利用。
- 据此将数据路径固定为逐帧 `set_state_dict()` + `get_obs()`：第 11 周 env state 是权威恢复源，重建得到完整 TCP 和 `is_grasped`，不覆盖原 HDF5。最终使用包含显式相对位置的 18 维任务相关特征，而不是让 MLP 从绝对关节状态隐式学习 FK 和相对位置。
- 第二次 150 epoch smoke 在 seed 100–109 上两个任务均为 10/10。正式训练使用 400 epoch、每任务 80 train / 20 validation episode；PickCube 为 1512/392 transitions、最佳 epoch 303、validation MSE `0.146417`，PushCube 为 1085/268 transitions、最佳 epoch 360、validation MSE `0.019004`。
- 正式 checkpoint 从磁盘用 `weights_only=True` 重新加载后，在未见 seed 100–149 上完成 100 次真实 CPU rollout：PickCube 42/50（84%），PushCube 50/50（100%），总体 92/100（92%），超过 70% 门禁；100 次均无运行异常。正式产物位于 `runs/maniskill_bc/week12/`，checkpoint 约 160 KB，训练摘要约 4 KB，评测摘要约 24 KB。
- 正式训练耗时 6.40 秒，训练加 100 次评测总耗时 26.89 秒，最大常驻内存约 654 MB。独立 ManiSkill 环境第 12 周定向测试 10 项通过；该环境排除缺少可选 `openai` 的 LLM 专属测试后 303 项通过，基础环境完整回归 293 项通过、4 项可选 HDF5/PyTorch 测试跳过，基础环境的 3 项 LLM 测试单独通过；`compileall`、`pip check` 和 `git diff --check` 通过。
