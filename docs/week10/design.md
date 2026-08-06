# 第 10 周设计文档 — ManiSkill 入门与脚本策略验收

## 1. 概述

**本周目标：** 在独立 Python 3.11 环境中安装并运行 ManiSkill 3，使用标准 Gymnasium 接口打开 `PickCube-v1` 和 `PushCube-v1`，明确 observation、action、reward、episode 的运行时契约，并用可复现的状态输入 scripted policy 完成两项任务。

**规划原文引用：**

> 第 10 周 | ManiSkill 入门 | 运行 PickCube / PushCube；理解 observation、action、reward、episode。 | 手动或 scripted policy 完成任务。

**前置依赖：**

- 第 7 周坐标系：理解 TCP、物体与目标位置。
- 第 8 周运动学：理解末端目标与关节控制之间的转换。
- 第 9 周控制接口：理解末端增量、夹爪动作和动作限幅。
- Python 3.11：ManiSkill 使用独立环境，不污染现有 AI2-THOR `.venv`。

## 2. 设计边界与原则

- 固定使用 `mani_skill==3.0.1`、`physx_cpu`、单环境和 Panda 机械臂，保证 Apple Silicon macOS 可运行。
- observation 使用 `state_dict`，显式读取 `extra.tcp_pose`、`extra.obj_pose`、`extra.goal_pos`，不提前进入第 16 周 RGB 策略。
- action 使用 `pd_ee_delta_pos`：前三维为归一化 TCP 位置增量，最后一维为夹爪控制；所有输出限制在环境 action space 内。
- reward 使用 `normalized_dense` 做过程观察；任务完成以 `info["success"]` 为准，不以累计 reward 代替成功判定。
- episode 严格遵循 Gymnasium 的 `reset → step → terminated/truncated → close` 生命周期，并区分成功、环境终止和时间截断。
- scripted policy 只完成本周入门验收，不保存专家轨迹、不训练策略、不修改 ManiSkill 内部状态。
- 官方 motion-planning 示例依赖 `mplib`；其 PyPI wheel 当前不支持本机 macOS，因此不作为本周本机验收路径。
- ManiSkill 为可选重依赖。纯逻辑测试通过延迟导入与 fake environment 运行，完整仓库测试不强制启动物理仿真。

## 3. 架构设计

```text
requirements-maniskill.txt
    └── 独立的 ManiSkill 3 运行依赖

src/maniskill_intro.py
    ├── ManiSkillTaskConfig       任务与环境参数
    ├── ObservationField          observation 字段契约
    ├── EpisodeResult             episode 验收结果
    ├── summarize_observation()   递归提取字段路径/shape/dtype
    ├── ScriptedPolicy            PickCube/PushCube 状态机策略
    ├── make_maniskill_env()      延迟导入并创建 CPU Gym 环境
    └── run_episode()             Gymnasium 生命周期与指标聚合

src/run_maniskill_demo.py
    └── CLI → 两任务/单任务 → inspection + scripted episode → JSON/文本摘要

src/tests/test_maniskill_intro.py
    └── 数据契约、观察摘要、策略阶段、episode 语义和失败路径
```

### 3.1 与第 9 周的连接

- 第 9 周的 `EndEffectorDeltaCommand` 对应本周 action 前三维的末端位置增量语义。
- 第 9 周的 `GripperPositionCommand` 是绝对位置；ManiSkill Panda 的最后一维是归一化夹爪控制命令，两者必须明确区分，不复用错误的数据类型。
- ManiSkill 自己负责 Panda 的低层控制、关节限位和物理推进；本周策略只产生标准化 action，不复制 IK 或动力学。

## 4. 数据结构

```python
@dataclass(frozen=True)
class ManiSkillTaskConfig:
    env_id: str
    seed: int = 0
    obs_mode: str = "state_dict"
    control_mode: str = "pd_ee_delta_pos"
    reward_mode: str = "normalized_dense"
    sim_backend: str = "physx_cpu"
    max_steps: int = 50


@dataclass(frozen=True)
class ObservationField:
    path: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class EpisodeResult:
    env_id: str
    seed: int
    observation_fields: tuple[ObservationField, ...]
    action_shape: tuple[int, ...]
    steps: int
    total_reward: float
    success: bool
    terminated: bool
    truncated: bool
    final_info: dict[str, object]
    phase_history: tuple[str, ...]
```

仅允许 `PickCube-v1` 与 `PushCube-v1`。seed 非负，`max_steps` 为正数。运行结果转换为纯 Python/NumPy 可序列化值，避免 Torch tensor 泄漏到 CLI JSON。

## 5. 接口设计

```python
def summarize_observation(observation: object) -> tuple[ObservationField, ...]: ...


class ScriptedPolicy:
    def __init__(self, env_id: str) -> None: ...
    def reset(self) -> None: ...
    def act(self, observation: Mapping[str, object]) -> np.ndarray: ...


def make_maniskill_env(config: ManiSkillTaskConfig) -> object: ...


def run_episode(
    config: ManiSkillTaskConfig,
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    policy_factory: Callable[[str], ScriptedPolicy] = ScriptedPolicy,
) -> EpisodeResult: ...
```

`make_maniskill_env()` 在调用时才导入 `gymnasium`、`mani_skill.envs` 和 `CPUGymWrapper`。缺依赖时抛出带安装命令的 `ManiSkillUnavailableError`，使普通仓库模块仍可导入。

## 6. Scripted policy

### 6.1 通用位置控制

1. 从 `state_dict["extra"]` 读取 TCP、物体和目标位置。
2. 计算 `target_tcp - current_tcp`。
3. 按控制尺度转为归一化动作并逐维裁剪到 `[-1, 1]`。
4. 位置误差低于阈值或阶段步数达到上限时进入下一阶段。
5. 夹爪动作始终显式给出：`+1` 张开，`-1` 闭合。

### 6.2 PickCube 状态机

```text
above_object → descend → close → lift_to_goal → settle
```

- 先移动到方块正上方，保持张开。
- 下探至抓取高度并连续闭合夹爪。
- 将 TCP 移向 goal；目标位置来自每个 episode 的 observation。
- 到达后保持若干步，等待机器人静止；最终以 `info.success` 判断。

### 6.3 PushCube 状态机

```text
above_behind → behind_object → push_through_goal → settle
```

- 根据 `goal_xy - object_xy` 计算单位推动方向。
- 移动到方块背后并降到推动高度，夹爪保持闭合。
- 沿目标方向推动到 goal 稍远位置，保证方块中心进入目标圆。
- 保持动作等待稳定；最终以 `info.success` 判断。

## 7. Observation、action、reward 与 episode 契约

| 概念 | 本周运行时证据 | 验收重点 |
| --- | --- | --- |
| observation | reset 后递归记录字段路径、shape、dtype | 必须包含 TCP、物体和目标状态 |
| action | `env.action_space` 的 shape/bounds + 每步 contains 校验 | scripted action 全部合法且有限 |
| reward | 每步标准化 dense reward，累计为 `total_reward` | 可观察过程，但不冒充成功标记 |
| episode | seed reset、逐步 step、terminated/truncated、close | 最多 50 步且资源总能关闭 |
| success | `info["success"]` | PickCube、PushCube 都必须为 true |

## 8. 错误处理与资源安全

| 场景 | 行为 |
| --- | --- |
| ManiSkill/Gymnasium 未安装 | 抛出 `ManiSkillUnavailableError`，提示独立环境安装命令 |
| 不支持的 env id | 构造配置或策略时 `ValueError` |
| observation 缺失关键字段 | `ObservationContractError`，列出缺失路径 |
| action 维度错误、非有限或越界 | episode 立即失败并报告契约错误 |
| step 返回非法 Gymnasium 五元组 | `EpisodeContractError` |
| episode 未在 `max_steps` 内成功 | 返回 `success=False`，保留 terminated/truncated 事实 |
| 任意 reset/step/policy 异常 | `finally` 中调用 `env.close()` |

## 9. 测试与验收策略

- 纯逻辑测试覆盖配置校验、递归 observation 摘要和 tensor/NumPy 标量归一化。
- fake environment 验证 reset/step 五元组、reward 累计、terminated/truncated 区分和无条件 close。
- 固定 observation 验证 PickCube/PushCube 每个状态机阶段和 action 边界。
- 非法 observation、非法 action、缺依赖和未知任务覆盖失败路径。
- 真实 ManiSkill CPU 环境分别运行两个 seed=0 scripted episode，记录字段、action shape、步数、累计 reward 和 success。
- 运行定向测试、完整仓库测试和 `compileall`。

## 10. 验收标准

1. 独立 Python 3.11 环境可导入 ManiSkill 3、SAPIEN、Torch 和 Gymnasium。
2. `PickCube-v1` 与 `PushCube-v1` 均能 reset/step/close，并输出 observation/action 契约。
3. scripted policy 的每个 action 都有限、维度正确且落在 action space 内。
4. 两项任务在固定 seed 下均于 episode 步数上限内返回 `info.success=True`。
5. 输出明确区分 reward、success、terminated 与 truncated。
6. 定向测试、完整回归和语法检查通过。
7. 总计划、第十周实施状态和本节“实际偏差”按真实结果同步。

## 11. 实际偏差

- 真实安装确认 `mani_skill==3.0.1` 会在 macOS 自动安装 `sapien==3.0.3`，但不会自动安装末端位姿控制所需的 Pinocchio Python binding；因此在 `requirements-maniskill.txt` 增加仅 macOS 生效的 `pin==4.1.0`。这是运行时依赖补充，不改变设计接口。
- 本机没有系统 Vulkan 驱动。按 ManiSkill macOS 前置要求通过 Homebrew 安装 `molten-vk==1.4.2`，并由 `make_maniskill_env()` 在未显式配置时查找稳定的 Homebrew ICD 路径；没有修改用户 `.zshrc`。SAPIEN 使用自带 Vulkan loader，MoltenVK 将设备映射到 Apple M4。
- `PickCube-v1` 的真实 observation 字段为 `agent.qpos/qvel` 以及 `extra.goal_pos/is_grasped/obj_pose/obj_to_goal_pos/tcp_pose/tcp_to_obj_pos`；scripted policy 在 seed 0 下 20 步成功，累计 normalized dense reward 为 `7.052832297980785`，`terminated=True`、`truncated=False`。
- `PushCube-v1` 的真实 observation 字段为 `agent.qpos/qvel` 以及 `extra.goal_pos/obj_pose/tcp_pose`；scripted policy 在 seed 0 下 14 步成功，累计 normalized dense reward 为 `3.9484396874904633`，`terminated=True`、`truncated=False`。
- 两个任务的真实 action shape 均为 `(4,)`；所有 scripted action 均通过 Gymnasium action space 检查。组合 CLI 验收通过，没有保存视频、轨迹或 `runs/` 产物。
- `src/tests/test_maniskill_intro.py` 定向测试 19 项通过；完整仓库回归 285 项通过，`compileall` 和独立环境 `pip check` 通过；额外组合运行 seed 1 时两个任务也成功。
