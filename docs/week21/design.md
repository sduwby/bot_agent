# 第 21 周设计文档 — 世界模型支线（小型 dynamics + CEM）

## 1. 概述

**本周目标：** 完成一个可复现的世界模型实验：用第 15 周 v3 数据训练一个小的状态动力学模型（dynamics），配合 CEM 规划在 PushCube-v1 上做有限演示。这是可选研究支线——理解预测与 MPC，不作为阶段 C、硬件采购或最终交付的阻塞条件。

总计划原文：

> 第 21 周 | 世界模型支线（可选） | 有余力时复用 TD-MPC2 或训练小型 dynamics + CEM；只做 PushCube 有限演示。 | 完成一个可复现实验；未完成不阻塞主线。

### 1.1 前置依赖

- 第 15 周 v3 数据集 `runs/maniskill_lerobot/week15-cam/`（`open_lerobot_dataset` 读回 `observation.state[5]`/`observation.environment_state[13]`/`action[4]`）。
- 第 10 周 `maniskill_intro.py`：`ScriptedPolicy`、`ManiSkillTaskConfig`、`make_maniskill_env`。
- 第 12 周 `behavior_cloning.py`：`STATE_DIM=18`、`ACTION_DIM=4`、`observation_to_state`（live 状态提取）。
- PyTorch（`.venv-lerobot` 已有）。

### 1.2 本周关键决策

1. **不引入 TD-MPC2**：TD-MPC2 是为大规模 RL 设计的多目标模型，接入成本高且与"支线演示"定位不符。改为训练 22→128→128→18 的小型 MLP dynamics（输入 state+action，输出下一帧状态增量），训练在 CPU/MPS 上分钟级完成。
2. **只做 PushCube**：PickCube 有抓取/抬升不连续切换，dynamics 误差会放大；PushCube 是纯推进任务，连续且受控，适合首个世界模型演示。
3. **状态用 18 维拼接**：`observation.state[5]`（proprioception）+ `observation.environment_state[13]`，与第 12 周 BC 口径一致，直接复用 `split_lerobot_act_state` 的读回。
4. **CEM 规划**：每步采样 N=128 条长度为 H=8 的动作序列，用 dynamics 前向预测，得分 = 预测终点 `obj-goal` 距离（越近越好），取前 10% 精英重拟合高斯，迭代 4 轮，执行第一步（MPC 滚动优化）。动作钳制 [-1,1]，夹爪维度固定 -1（PushCube 无需夹爪）。
5. **评测协议**：固定 seed 500–509（10 次，与第 14 周稳定性评测同 seed 可对比），记录成功率、步数、与 scripted 基线的对照。**不参与门禁**。
6. **纯 NumPy 实现 CEM 与数据管线**：dynamics 前向用 NumPy 手写（单层参数矩阵从 PyTorch checkpoint 导出），使 CEM/rollout 逻辑可脱离 torch 测试；torch 只用于训练。

### 1.3 范围边界

本周包含：

- 从 v3 读回 PushCube transitions，构造 (state, action) → Δstate 训练集；
- 训练小型 MLP dynamics 并保存 checkpoint（权重 + 归一化 + 元数据）；
- NumPy CEM 规划器与单 episode 执行器；
- 10 次固定 seed 演示评测与报告落盘。

本周不包含：

- PickCube、视觉输入、TD-MPC2、多步 latent 预测；
- 在线训练、RL、与 ACT/scripted 的门禁比较（第 22 周做）；
- 真实硬件相关改动。

## 2. 架构设计

```text
week15-cam v3（PushCube transitions）
        │
        ▼
src/world_model.py
   ├─ load_dynamics_data()：state/action/next_state 读回与切分
   ├─ train_dynamics()：PyTorch MLP 训练 + checkpoint（weights.npz + metadata）
   ├─ DynamicsModel：NumPy 前向（Δstate 预测），从 checkpoint 加载
   ├─ CEMPlanner：采样→预测→精英重拟合→首步动作
   └─ run_world_model_episode()：live env + CEM MPC rollout
        │
        ▼
src/run_world_model.py（训练 + 演示评测 CLI）
        │
        ▼
runs/maniskill_world_model/week21-pushcube/
   ├─ dynamics_checkpoint/（weights.npz、normalization.json、metadata.json）
   └─ demo_report.json（成功率、步数、对比）
```

## 3. 数据结构

```python
@dataclass(frozen=True)
class DynamicsDataset:
    states: np.ndarray        # [N, 18]
    actions: np.ndarray       # [N, 4]
    next_states: np.ndarray   # [N, 18]
    episode_ids: np.ndarray   # [N]

@dataclass(frozen=True)
class WorldModelConfig:
    dataset_dir: Path
    repo_id: str
    output_dir: Path
    hidden_dims: tuple[int, ...] = (128, 128)
    train_episodes: int = 80
    validation_episodes: int = 20
    training_steps: int = 4000
    batch_size: int = 128
    learning_rate: float = 1e-3
    cem_population: int = 128
    cem_horizon: int = 8
    cem_elite_ratio: float = 0.1
    cem_iterations: int = 4
    seeds: tuple[int, ...] = tuple(range(500, 510))
```

## 4. 接口设计

```python
def load_dynamics_data(
    dataset_dir: Path, repo_id: str, *, task: str = "PushCube-v1"
) -> DynamicsDataset:
    """从 v3 读回 PushCube 的 (state, action, next_state) transitions。"""


def train_dynamics(
    config: WorldModelConfig,
    data: DynamicsDataset,
) -> dict[str, object]:
    """训练 MLP dynamics，导出 NumPy 权重并返回训练摘要。"""


class DynamicsModel:
    def predict_delta(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray: ...
    def predict_next(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray: ...


class CEMPlanner:
    def plan(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """CEM 优化返回首步 4D 动作。"""


def run_world_model_episode(
    config: ManiSkillTaskConfig, env: object, planner: CEMPlanner
) -> dict[str, object]:
    """live 环境 + MPC rollout，返回成功/步数。"""
```

## 5. 失败处理与降级

| 失败 | 处理 |
| --- | --- |
| 数据集缺失/读回异常 | 明确报错并提示先跑第 15 周管线 |
| 训练 loss 非有限 | 报错终止，不产出 checkpoint |
| CEM 预测发散（非有限） | 该采样序列计最低分，不崩溃 |
| 演示成功率极低 | 不影响主线：报告记录"世界模型在 PushCube 不优于 scripted 基线"结论即可 |

## 6. 验收标准

1. dynamics 训练完成、checkpoint 可重载（含归一化与元数据）。
2. CEM 规划器纯 NumPy 可测试。
3. PushCube 10 次固定 seed 演示完成并落盘报告。
4. 报告含与 scripted 基线的对照结论。
5. 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
6. 运行产物只写入已忽略的 `runs/`。

## 7. 实际偏差

### 7.1 实现偏差（2026-08-13）

1. **v3 状态读回用 `combine_lerobot_act_state` 还原 18D**：设计初稿直接拼接 robot[5]+env[13]，但 v3 的 robot/env 是从 18D 按索引抽取的（`ROBOT_STATE_INDICES`/`ENV_STATE_INDICES`），直接拼接会打乱字段顺序；实现改为还原为 `observation_to_state` 一致的 18D 布局（tcp/obj/goal/tcp-obj/obj-goal/gripper/grasped），保证训练数据与 live 状态口径一致。
2. **评测结果（4060 CPU 训练 + 真实 env，2026-08-13）**：PushCube 1253 transitions 训练 4000 step（best_step=600、val_loss 0.103）；CEM MPC 固定 seed 500–509 **0/10（0%）**，scripted 基线 10/10（100%，mean_steps 13.8）。诊断：CEM 在 dynamics 模型上优化的是"预测距离"，但模型在分布外状态（多步前向）预测发散，规划器找到了利用模型误差的"幻觉最优"动作（单步动作 norm 1.5–1.8、预测 obj/goal 漂移到数十米量级），真实环境不收敛。
3. **结论：小型 dynamics + CEM 在 PushCube 不优于 scripted 基线**。与设计预期一致（少量数据 + 短视 MPC 的已知局限）。**该支线不阻塞主线**：按 §1.3 与总计划，第 22 周阶段评测与选型不使用世界模型；本实验作为"预测与 MPC 理解"的一次可复现记录归档。
4. **`train` 子命令 `--hidden-dims` 解析修复**：初版把 `"128,128"` 直接 `split(",")` 得到字符串元组，配置校验失败；实现改为逐项 `int()` 转换。
