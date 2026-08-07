# 第 11 周设计文档 — ManiSkill 专家轨迹采集与回放

## 1. 概述

**本周目标：** 复用第 10 周 `PickCube-v1`、`PushCube-v1` 的 scripted expert，采集可供第 12 周低维 Behavior Cloning 使用的轻量成功轨迹，并用真实 CPU 仿真回放验证数据可重现。

总计划原文：

> 第 11 周 | 专家轨迹 | 用 scripted expert 采集可重放的 env state、action、seed 和 success；先保存轻量原始轨迹，再按需 replay RGB。 | 至少 200 条成功状态轨迹；抽检/回放 50 条。

### 前置依赖

- 第 10 周 `ManiSkillTaskConfig`、`ScriptedPolicy` 和 CPU Gym 环境已完成真实验收。
- 使用独立 `.venv-maniskill`，依赖 `mani_skill==3.0.1` 自带的 `RecordEpisode` 与 HDF5 支持。
- 第 11 周 HDF5 定向测试通过 `requirements-maniskill-dev.txt` 在同一 Python 3.11 环境补充 `pytest`。
- macOS 使用 `physx_cpu`；MoltenVK 仅满足 SAPIEN 渲染设备初始化，不启用 GPU 并行仿真。
- 轨迹写入已忽略的 `runs/`，不提交 HDF5、JSON、视频或回放产物。

### 范围边界

- 采集两个已有任务，每个任务至少 100 条成功轨迹，总计至少 200 条。
- 保存 state-dict observation、完整 env state、action、reward、terminated、truncated、seed 与 success。
- 仅保留成功 episode；失败尝试单独计数，不写入监督数据集。
- 抽检 50 条轨迹，默认两个任务各 25 条；从初始 env state 重放原 action，并验证最终 success。
- 不保存 RGB、depth、segmentation 或视频，不提前进入第 12 周训练和第 16 周视觉策略。
- 不修改第 10 周任务、控制模式和 scripted policy 行为。

## 2. 架构设计

```text
run_expert_collection.py
    │
    ├── TrajectoryCollectionConfig      数量、seed、重试和输出约束
    ├── collect_expert_dataset()         两任务采集编排与汇总
    │       ├── make_recording_env()     CPU Gym + RecordEpisode
    │       └── run_episode_in_env()     复用单环境执行 scripted policy
    │
    ├── inspect_trajectory_dataset()     HDF5/JSON 静态完整性检查
    └── replay_trajectory_dataset()      初始 env state + action 真实回放
            └── replay_summary.json      50 条抽检结果
```

新增 `src/expert_trajectory.py` 负责数据采集、结构检查和回放。第 10 周 `src/maniskill_intro.py` 只增加可复用的“在已有环境中运行一次 episode”入口，原 `run_episode()` 的创建、关闭和结果语义保持不变。

## 3. 数据结构

### 3.1 采集配置

```python
@dataclass(frozen=True)
class TrajectoryCollectionConfig:
    output_dir: Path
    episodes_per_task: int = 100
    seed_start: int = 0
    max_attempts_per_task: int = 300
    replay_count: int = 50
```

约束：数量均为非布尔正整数；`seed_start >= 0`；回放数不能超过目标成功轨迹总数；输出目录可以新建，但已有非空目录默认拒绝，防止覆盖历史数据。

### 3.2 任务采集结果

```python
@dataclass(frozen=True)
class TaskCollectionResult:
    env_id: str
    target_successes: int
    attempts: int
    successes: int
    failures: int
    total_steps: int
    trajectory_path: str
    metadata_path: str
```

### 3.3 数据集汇总

`summary.json` 包含：

- `schema_version`：固定为 `maniskill_expert_state_v1`；
- `created_at`：UTC ISO 8601 时间；
- 环境、obs/control/reward/sim backend 契约；
- 每个任务的目标数、尝试数、成功数、失败数、总步数和文件名；
- 总成功数、总失败数、总步数与采集完成标记；
- 回放目标数和 `replay_summary.json` 路径。

每个任务生成一组 ManiSkill 标准文件：

```text
runs/maniskill_expert/<timestamp>/
├── summary.json
├── replay_summary.json
├── pick_cube.h5
├── pick_cube.json
├── push_cube.h5
└── push_cube.json
```

HDF5 每个 `traj_<episode_id>` 包含：

- `actions[T, 4]`；
- `env_states[T+1, ...]`（嵌套 state dict）；
- `obs[T+1, ...]`（低维 `state_dict`）；
- `rewards[T]`；
- `terminated[T]`、`truncated[T]`、`success[T]`。

伴随 JSON 保存 `episode_seed`、`reset_kwargs`、`control_mode`、`elapsed_steps` 和最终 success。

## 4. 接口设计

### `maniskill_intro.py`

```python
def run_episode_in_env(
    config: ManiSkillTaskConfig,
    env: object,
    policy: Policy,
) -> EpisodeResult: ...
```

在调用方持有的已创建环境中执行一次 episode，不关闭环境；仍验证 observation、action space、Gymnasium 五元组、有限 reward 和终止语义。

### `expert_trajectory.py`

```python
def collect_expert_dataset(
    config: TrajectoryCollectionConfig,
    *,
    env_factory: Callable[..., object] = make_recording_env,
    policy_factory: Callable[[str], Policy] = ScriptedPolicy,
) -> CollectionSummary: ...

def inspect_trajectory_dataset(
    summary_path: Path,
) -> DatasetInspection: ...

def replay_trajectory_dataset(
    summary_path: Path,
    *,
    replay_count: int,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
) -> ReplaySummary: ...
```

### CLI

```bash
.venv-maniskill/bin/python src/run_expert_collection.py \
  --episodes-per-task 100 \
  --replay-count 50
```

可选 `--output-dir`、`--seed-start`、`--max-attempts-per-task`、`--skip-replay`。默认输出到 `runs/maniskill_expert/<timestamp>/`；成功打印机器可读摘要路径，未达到数量或回放失败返回非零。

## 5. 数据流

### 5.1 采集

1. 校验配置并创建全新的输出目录。
2. 针对每个任务创建一个 `RecordEpisode` 环境，在全部尝试间复用。
3. 每次用递增 seed reset，重置 scripted policy 并执行 episode。
4. episode 成功时 flush 到 HDF5；失败时清空内存 buffer 但不写入数据集。
5. 达到每任务 100 条成功轨迹后关闭文件，生成任务 JSON 元数据。
6. 静态检查轨迹组数、步数、关键字段和 `T/T+1` 对齐关系。
7. 原子写入 `summary.json`。

### 5.2 回放

1. 按任务均衡、确定性地选取 50 条 episode。
2. 从伴随 JSON 取得 seed 和 reset 参数，创建并复用普通 CPU 环境。
3. reset 后加载轨迹第 0 帧 env state。
4. 逐步执行原始 action，不注入后续 state 修正。
5. 记录最终 success、terminated、truncated、步数和失败原因。
6. 全部 50 条成功才通过验收，并原子写入 `replay_summary.json`。

## 6. 异常与安全边界

| 场景 | 行为 |
| --- | --- |
| 输出目录已包含文件 | 拒绝启动，避免覆盖数据 |
| episode 失败 | 计入失败尝试，不写入成功数据集 |
| 达到最大尝试仍不足目标数 | 关闭文件，写入失败汇总并返回非零 |
| HDF5 与 JSON 数量不一致 | 静态检查失败，禁止宣称采集完成 |
| action/env_state 长度错位 | 静态检查失败并指出 episode |
| 回放未成功 | 记录具体轨迹 ID，整体验收失败 |
| 中断或异常 | `finally` 关闭环境/HDF5；已落盘文件保留在忽略目录供诊断 |

## 7. 测试与验收

### 纯逻辑测试

- 配置边界和输出目录保护。
- 采集成功/失败过滤、递增 seed、重试上限和环境关闭。
- summary 的数量与字段聚合。
- HDF5/JSON 结构检查、缺组、长度错位和 success 校验。
- 回放任务分配、状态恢复、action 执行和失败汇总。
- 原 `run_episode()` 行为保持不变，新增已有环境入口不关闭调用方资源。

### 真实验收

1. 两任务合计至少 200 条成功轨迹，每个任务至少 100 条。
2. 所有轨迹均含 env state、state observation、action、seed 和 success。
3. 失败尝试不进入监督数据集。
4. 静态完整性检查 200/200 通过。
5. 初始 env state + action 真实回放 50/50 成功。
6. 不生成 RGB、视频或版本控制内产物。
7. 定向测试、完整回归、`compileall` 和独立环境 `pip check` 通过。

## 8. 实际偏差

- 实现直接复用 ManiSkill 3.0.1 的 `RecordEpisode` 标准 HDF5/JSON 格式；没有引入自定义二进制格式，也没有新增运行时依赖。为在独立 Python 3.11 环境复现 HDF5 测试，新增仅开发使用的 `requirements-maniskill-dev.txt`，在原依赖上补充 `pytest`。每个任务复用一个 CPU 环境，失败 attempt 通过 `flush_trajectory(save=False)` 丢弃，符合设计。
- macOS 上 SAPIEN 启动后进程相对路径会解析到 Python.app 的 Resources 目录。小样本首次运行发现该偏差后，将 `TrajectoryCollectionConfig.output_dir` 改为在启动仿真前 `expanduser().resolve()`；随后清理了误落到 Python.app 资源目录的 smoke 产物。正式数据全部位于仓库已忽略的 `runs/`。
- 小样本真实验证采集 4/4 条成功轨迹，静态检查 4/4，通过初始 env state + 原 action 回放 2/2；修复绝对路径后再次验证结果一致。
- 正式验收路径为 `runs/maniskill_expert/week11-200/`：PickCube 100/100 成功、1904 步，PushCube 100/100 成功、1353 步，两个任务均无失败 attempt；200/200 条静态完整性检查通过，50/50 条 action replay 成功。
- 正式 HDF5/JSON/summary/replay 产物合计约 3.1 MB，未保存 RGB、depth、segmentation 或视频；完整流程耗时 11.63 秒，最大常驻内存约 583 MB。
- 独立 ManiSkill 环境定向测试 30 项通过；基础环境完整回归 293 项通过、3 项因基础环境不含可选 `h5py` 而跳过，这 3 项 HDF5/回放测试已在独立环境通过。`compileall` 与独立环境 `pip check` 通过。
