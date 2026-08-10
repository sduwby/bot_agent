# 第 15 周设计文档 — LeRobot 数据管线

## 1. 概述

**本周目标：** 在第 14 周阶段 B 门禁通过（ACT 99/100）之后，建设第 15 周的 LeRobot 数据管线：把第 11 周 ManiSkill 成功轨迹转换为 LeRobotDataset v3 格式，包含 episode、timestamp、camera、state、action；实现 replay、检查和可视化，使任意轨迹可浏览并能发现异常帧，且字段缺失为 0。

总计划原文：

> 第 15 周 | LeRobot 数据管线 | 转换 episode、timestamp、camera、state、action；实现 replay、检查和可视化。 | 可浏览任意轨迹并发现异常帧，字段缺失为 0。

### 1.1 前置依赖

- 第 11 周正式数据位于已忽略的 `runs/maniskill_expert/week11-200/`：`pick_cube.h5`、`pick_cube.json`、`push_cube.h5`、`push_cube.json`、`summary.json`（schema `maniskill_expert_state_v1`，两任务各 100 条成功轨迹，共 3257 transition）。
- 第 12–14 周已验证 `env_states → set_state_dict() → get_obs() → 18 维状态` 是权威恢复路径；不读取已知宽度损失的 HDF5 `obs` 作为完整 observation。
- `.venv-lerobot`（Python 3.12 + LeRobot 0.6.1 + ManiSkill 3.0.1 + h5py）。第 15 周补充安装 `lerobot[dataset]` extras（`datasets`、`pyarrow`、`dill`、`av` 等），使 `LeRobotDataset` v3 可用；`pip check` 在开发开始前必须通过。
- ManiSkill 环境以 `render_mode="rgb_array"` 可输出 512×512 RGB 帧（已探针验证，~0.07 s/帧），用于 camera 字段。

### 1.2 本周关键决策

1. **数据事实源仍是第 11 周 HDF5 + env_states**。LeRobotDataset v3 是转换产物/消费格式，不修改原始数据；转换可重复、可校验。
2. **每帧使用 env_states 重建 live observation 后再写 LeRobotDataset**，与第 12–14 周 ACT/BC 的 observation 契约保持一致，避免把已损坏的 `obs` 标量叶子当作完整 observation。
3. **camera 字段在转换时按需渲染 RGB**：转换进程内以 `render_mode="rgb_array"` 创建环境，对每个 env_state 调用 `set_state_dict` 后 `render()`。camera 是可选 feature；开启时字段缺失必须为 0。
4. **state 拆分为 robot state 与 environment state**，沿用第 13 周 feature 语义：
   - `observation.state[5]`：`tcp_pos[3] + gripper_qpos[2]`；
   - `observation.environment_state[13]`：`obj_pos[3] + goal_pos[3] + (tcp-obj)[3] + (obj-goal)[3] + is_grasped[1]`。
5. **timestamp 由 LeRobotDataset 依据 fps 自动生成**（`frame_index / fps`），不额外造假时间。
6. **replay 不重跑物理仿真**：LeRobotDataset v3 本身可读回任意 episode 的完整状态/动作序列；本周的 replay 指"从 v3 读回 episode 并逐帧校验与源数据一致"。真实 action replay 已由第 11 周 `replay_trajectory_dataset()` 覆盖，不重复。
7. **检查=字段完整性与异常帧检测**：所有已声明 feature 在每帧都必须存在（missing=0），并检测 NaN/Inf、shape 错误、非有限值、越界 action。
8. **可视化=可浏览任意轨迹并发现异常帧**：提供逐 episode 的帧级摘要（状态统计、动作、异常标记）与可选相机帧缩略图。

### 1.3 范围边界

本周包含：

- 第 11 周 HDF5 → LeRobotDataset v3 的转换（episode、timestamp、camera、state、action、reward、success、next.done、task）；
- 18 维状态到 5/13 feature 的无损拆分与逆向校验；
- 可选 RGB 相机渲染（env_state → render）；
- LeRobotDataset v3 本地创建、写入、`save_episode`、`finalize` 与只读重开；
- episode 级读回（replay）、字段完整性检查、异常帧检测；
- 逐 episode 帧级浏览与可视化报告。

本周不包含：

- 视觉 ACT 训练或推理（第 16 周）；
- RGB → 相机标定、深度图或真实相机；
- 修改第 11 周原始数据或 checkpoint；
- 真实机械臂、ROS 2、CUDA、云 GPU 或 Hub 上传；
- SmolVLA、语言条件或多任务数据。

## 2. 版本与设备设计

### 2.1 运行环境

- 复用 `.venv-lerobot`，补充 `lerobot[dataset]==0.6.1`（安装 `datasets`、`pyarrow`、`dill`、`av`、`torchcodec` 等）。
- 转换进程创建 ManiSkill 环境时使用 `render_mode="rgb_array"`；macOS 继续由入口自动定位 MoltenVK ICD。
- 记录 Python、LeRobot、ManiSkill、SAPIEN、Pinocchio、h5py 版本与 `pip check` 结果。

### 2.2 设备要求

- Apple M4、16 GB 内存足够完成 200 条轨迹转换与渲染（实测 ~0.07 s/帧，200 条约 3257 帧 → 数分钟内）。
- 转换与检查必须记录耗时与峰值内存；不能只报告成功数量。

## 3. 架构设计

```text
runs/maniskill_expert/week11-200/  (pick_cube.h5 / push_cube.h5 / summary.json)
        │
        ▼
lerobot_pipeline.load_maniskill_summary()
        │  校验 schema、episode 元数据
        ▼
leobot_pipeline.LerobotConversionConfig
        │  features、camera 开关、fps、输出目录
        ▼
lerobot_pipeline.convert_maniskill_to_lerobot()
        │  对每个 episode：
        │    env_states → set_state_dict → get_obs → 18D → 5/13 split
        │    env_states → set_state_dict → render() → RGB (camera 开启时)
        │    actions / reward / success / terminated / truncated
        │  LeRobotDataset.add_frame × T
        │  LeRobotDataset.save_episode
        ▼
LeRobotDataset v3 (runs/maniskill_lerobot/week15-<tag>/)
        ▼
lerobot_pipeline.inspect_lerobot_dataset()
        │  字段完整性（missing=0）+ 异常帧检测（NaN/Inf/shape/越界）
        ▼
lerobot_pipeline.browse_lerobot_episode()
lerobot_pipeline.visualize_lerobot_episode()
        │  episode 帧级摘要 + 相机缩略图
        ▼
inspection.json / episode_<id>.json / 缩略图 PNG
```

### 3.1 模块职责

新增 `src/lerobot_pipeline.py`：

- 第 11 周 summary/元数据加载与校验；
- LeRobotDataset v3 feature schema 构造与 18D→5/13 拆分；
- 转换主流程（读 HDF5 → 重建 observation → 写入 v3）；
- 可选 camera 渲染；
- v3 只读重开与 episode 读回；
- 字段完整性检查、异常帧检测、逐 episode 浏览与可视化报告。

新增 `src/run_lerobot_pipeline.py`：

- 参数校验与默认输出目录；
- 转换、检查、浏览、可视化子命令；
- 以退出码表达转换/检查是否通过。

新增 `src/tests/test_lerobot_pipeline.py`：

- 不依赖 Unity 的 feature schema、18D 拆分/逆向、读回、检查、异常帧检测、浏览与错误路径测试；
- fake env / fake dataset 的转换 smoke；
- LeRobotDataset v3 可用时的真实本地创建/写入/重开 smoke。

复用（不修改或仅最小修改）：

- `src/expert_trajectory.py`：`TASK_FILE_STEMS`、`SCHEMA_VERSION`、HDF5 布局读取；
- `src/behavior_cloning.py`：`observation_to_state()`、`observation_to_arm_qvel()`、`STATE_DIM`、`ACTION_DIM`；
- `src/lerobot_act.py`：`ROBOT_STATE_DIM`、`ENV_STATE_DIM`、`split_lerobot_act_state()`；
- `src/maniskill_intro.py`：`make_maniskill_env()` 与 `ManiSkillTaskConfig`。

## 4. 数据结构

### 4.1 转换配置

```python
@dataclass(frozen=True)
class LerobotConversionConfig:
    source_summary: Path           # week11 summary.json
    output_dir: Path               # v3 dataset 根目录（必须不存在或为空）
    repo_id: str = "bot-agent/maniskill-week11"
    fps: int = 20                  # 用于 timestamp
    include_camera: bool = False
    tasks: tuple[str, ...] = SUPPORTED_TASKS
    robot_type: str = "panda"
```

`fps` 默认 20：第 11 周轨迹平均约 16–20 步，语义上以 1 步≈1 决策周期，真实采集帧率未知，故用可配置值并在 metadata 中记录，不伪造真实时间。

### 4.2 LeRobotDataset v3 feature schema

```python
features = {
    "observation.state": {"dtype": "float32", "shape": (5,), "names": None},
    "observation.environment_state": {"dtype": "float32", "shape": (13,), "names": None},
    "action": {"dtype": "float32", "shape": (4,), "names": None},
    "reward": {"dtype": "float32", "shape": (1,), "names": None},
    "success": {"dtype": "bool", "shape": (1,), "names": None},
    "next.done": {"dtype": "bool", "shape": (1,), "names": None},
}
# include_camera=True 时追加：
features["observation.images.top"] = {
    "dtype": "video", "shape": (512, 512, 3), "names": None,
}
```

- 标量类 feature 使用 shape `(1,)`（LeRobot v3 校验要求，非 `()`）。
- `use_videos=True` 且 camera 开启；否则 `use_videos=False`。
- 每个 episode 写 `T` 帧（transition 数），最后一帧 `next.done=True` 表示终止。

### 4.3 帧内容

对 episode 中第 `t` 个 transition：

```text
input:
  env_states[t] → set_state_dict → get_obs → observation
  observation → 18D state → split 5/13
  camera 开启时：env_states[t] → set_state_dict → render() → RGB (H,W,3)

frame:
  observation.state[5]
  observation.environment_state[13]
  action[t][4]
  reward[t]（标量 shape (1,)）
  success[t]（bool shape (1,)）
  next.done = terminated[t] or truncated[t]
  task = env_id 对应的任务文本（如 "PickCube-v1"）
```

- 每帧所有 feature 必须存在且 shape/dtype 合法；缺失即视为异常帧。
- `action` 必须有限；越界按 `[-1, 1]` 检查并标记。
- camera 开启时每帧必须有 RGB；全黑或 shape 错误标记为异常。

### 4.4 运行产物

```text
runs/maniskill_lerobot/week15-<tag>/
├── meta/
│   └── info.json
├── episodes/.../episodes_*.parquet
├── data/.../data_*.parquet
├── videos/.../observation.images.top/...mp4   # camera 开启时
├── conversion_summary.json
├── inspection.json
└── episode_browser/episode_<id>.json          # 浏览/可视化产物
```

## 5. 接口设计

```python
def load_maniskill_summary(summary_path: Path) -> dict[str, object]:
    """加载并校验第 11 周 summary。"""


def build_lerobot_features(include_camera: bool) -> dict[str, dict[str, object]]:
    """构造 LeRobotDataset v3 feature schema。"""


def maniskill_state_to_lerobot_features(
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """18D → (robot_state[5], env_state[13])，并做逆向校验。"""


def convert_maniskill_to_lerobot(
    config: LerobotConversionConfig,
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_render_env,
) -> ConversionSummary:
    """把第 11 周 HDF5 轨迹转换为 LeRobotDataset v3。"""


def open_lerobot_dataset(output_dir: Path, repo_id: str) -> object:
    """只读重开 v3 dataset。"""


def read_episode_frames(
    dataset: object, episode_index: int
) -> EpisodeFrames:
    """读回单个 episode 的所有帧（state/action/reward/success/done/camera）。"""


def inspect_lerobot_dataset(output_dir: Path, repo_id: str) -> InspectionSummary:
    """字段完整性（missing=0）+ 异常帧检测。"""


def browse_lerobot_episode(
    output_dir: Path, repo_id: str, episode_index: int
) -> dict[str, object]:
    """逐帧摘要（状态统计、动作、异常标记、camera 缩略图）。"""


def visualize_lerobot_episode(
    output_dir: Path, repo_id: str, episode_index: int, image_dir: Path
) -> dict[str, object]:
    """生成相机帧缩略图网格与帧级 JSON 报告。"""
```

## 6. 数据流

### 6.1 转换

1. 读取第 11 周 summary，校验 schema 与两个任务各 100 条成功轨迹。
2. 读取每任务 `*.json` 的 episode 元数据（episode_id、episode_seed、elapsed_steps）。
3. 打开对应 HDF5，对每个 episode：
   a. 逐帧 `env_states[t] → set_state_dict → get_obs → observation_to_state` 得到 18D state；
   b. 拆分为 5/13 feature 并逆向校验；
   c. camera 开启时逐帧 `set_state_dict → render()` 得到 RGB；
   d. 组装 action/reward/success/next.done/task 后 `add_frame`。
4. `save_episode` 每 episode 一次；全部完成后 `finalize`。
5. 写 `conversion_summary.json`，含每任务 episode 数、总帧数、feature schema、版本、耗时。

### 6.2 检查

1. 只读重开 v3 dataset。
2. 遍历所有 episode 所有帧：
   - 字段完整性：声明的 feature 在每帧都存在（missing=0）；
   - 值有效性：state/action/reward 有限；
   - action 越界标记；
   - camera 帧存在且非全黑（camera 开启时）。
3. 聚合每任务与总体，输出 `inspection.json`；missing 总数必须为 0。

### 6.3 浏览与可视化

1. `browse_lerobot_episode()`：读回单 episode，输出逐帧摘要（状态 norm、动作、异常标记）。
2. `visualize_lerobot_episode()`：保存相机帧缩略图网格与帧级 JSON 报告。
3. 任意 episode 都可独立浏览；异常帧会被明确标记。

## 7. 失败处理与降级

| 失败 | 处理 |
| --- | --- |
| `lerobot[dataset]` 未安装 | 明确错误并给出安装指引，不降级为手写 parquet |
| 输出目录已存在且非空 | 拒绝覆盖 |
| HDF5 布局或 schema 不符 | 拒绝转换，不猜测 slice |
| 重建 observation 失败或 18D shape 错误 | 该 episode 记为失败，不静默跳过 |
| 18D 拆分/逆向不一致 | 拒绝写入该帧 |
| feature 缺失或 shape/dtype 错误 | 该帧标记异常，汇总在 inspection 报告 |
| camera 帧缺失或全黑 | 该帧标记异常 |
| v3 写读不一致（frame 数/字段不符） | 检查阶段报错 |

## 8. 验收标准

1. `lerobot[dataset]==0.6.1` 安装且 `pip check` 通过。
2. 第 11 周 200 条轨迹可转换为 LeRobotDataset v3（两任务各 100 episode）。
3. 18D state 无损拆分为 5/13，逆向校验一致。
4. 转换产物可只读重开，任意 episode 可读回完整 state/action/reward/success/done（camera 可选）。
5. 字段完整性：所有已声明 feature 每帧都存在，missing=0。
6. 异常帧检测能识别 NaN/Inf、shape 错误、越界 action、camera 缺失/全黑。
7. 逐 episode 浏览与可视化报告可生成，任意轨迹可独立浏览。
8. 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
9. 运行产物只写入已忽略的 `runs/`。

## 9. 参考资料

- [LeRobotDataset v3 文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [LeRobot v0.6.1 源码](https://github.com/huggingface/lerobot/tree/v0.6.1/src/lerobot/datasets)
- 第 11 周 `docs/week11/design.md`、`docs/week11/plan.md`
- 第 13 周 `docs/week13/design.md`（5/13 feature 语义）

## 10. 实际偏差

### 10.1 实现偏差（2026-08-10）

1. **`.venv-lerobot` 安装 `lerobot[dataset]==0.6.1`**：新增 `datasets`、`pyarrow`、`dill`、`av`、`torchcodec` 等 extras；`pip check` 通过。`requirements-lerobot.txt` 已更新。
2. **标量 feature 使用 shape `(1,)`**：LeRobot v3 `validate_frame` 拒绝 `()`，所有 reward/success/next.done 以单元素数组写入。
3. **LeRobotDataset.create 要求 root 目录不存在**：`_prepare_output_directory()` 只校验"缺失或为空"，不预创建目录，交由 `create` 创建。
4. **`dataset.meta.episodes` 是逐 episode 的行，不是逐帧**：`_episode_frame_indices()` 改为按 `dataset[index]["episode_index"]` 全量扫描帧。
5. **ManiSkill wrapper 的 `render()` 需要先 `reset()`**：否则抛 `ResetNeeded`。转换每个任务时先 `env.reset(seed=0)` 再进入逐帧 `set_state_dict → get_obs / render`。
6. **视频 feature 读回默认 CHW `(3,512,512)`**：新增 `_image_hwc()` 把读回数组归一化为 HWC，供缩略图与全黑检测使用。
7. **转换环境创建失败按 episode 计入 errors**：`env_factory` 抛错时该任务所有 episode 记为失败，`episode_errors = len(episodes)`。

### 10.2 真实转换与检查结果（2026-08-10）

**无相机转换** `runs/maniskill_lerobot/week15-full/`：

| 任务 | episodes | frames | errors |
| --- | ---: | ---: | ---: |
| PickCube-v1 | 100 | 1904 | 0 |
| PushCube-v1 | 100 | 1353 | 0 |
| 合计 | 200 | 3257 | 0 |

- `conversion_summary.json`：schema `maniskill_lerobot_pipeline_v1`、fps=20、`complete=true`，耗时 6.4 s。
- `inspection.json`：**missing=0、abnormal_frames=0**，`complete=true`。

**带相机转换** `runs/maniskill_lerobot/week15-cam/`：

- 同样 200 episode / 3257 帧，errors=0；视频 feature `observation.images.top`（512×512×3）写入 `videos/`。
- `inspection.json`：**missing=0、abnormal_frames=0**。

**浏览/可视化**：`browse` 对任意 episode 输出逐帧状态范数、action、success/done；`visualize` 生成相机帧 PNG 缩略图与 `episode_<id>.json`（smoke 已验证 20 帧缩略图）。

**结论**：LeRobot 数据管线转换、replay（v3 读回）、检查（字段完整性 missing=0 + 异常帧检测）与可视化全部达标。
