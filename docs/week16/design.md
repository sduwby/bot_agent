# 第 16 周设计文档 — 视觉 ACT（单相机 RGB + proprioception）

## 1. 概述

**本周目标：** 在第 15 周 LeRobot v3 数据管线（含 512×512 相机帧）之上，把第 13 周 state-only ACT 扩展为视觉 ACT：输入单相机 RGB + proprioception（robot state），完成端到端训练与 rollout，控制分辨率、batch 和显存，并输出显存/时长记录。

总计划原文：

> 第 16 周 | 视觉 ACT | 在 ACT 中加入单相机 RGB + proprioception；控制分辨率、batch 和显存。 | 完成端到端训练和 rollout，输出显存/时长记录。

### 1.1 前置依赖

- 第 15 周 camera 数据集：`runs/maniskill_lerobot/week15-cam/`（LeRobotDataset v3，200 episode / 3257 帧，`observation.images.top` 512×512 RGB、`observation.state[5]`、`observation.environment_state[13]`、`action[4]`、`reward/success/next.done/task`，fps=20，missing=0）。
- 第 15 周无相机数据集 `runs/maniskill_lerobot/week15-full/` 作为对照（state-only，不用于本周训练）。
- 第 13 周 `lerobot_act.py` 的训练/checkpoint/rollout 框架；第 15 周 `lerobot_pipeline.py` 的 `make_render_env()` 与 `_image_hwc()`。
- `.venv-lerobot`（Python 3.12 + LeRobot 0.6.1 + ManiSkill 3.0.1 + `lerobot[dataset]`）。`pip check` 通过。
- torchvision ResNet18 ImageNet 权重可下载（已探针验证 44.7 MB，本机缓存可用）。

### 1.2 本周关键决策

1. **直接消费 LeRobotDataset v3，不重写中间格式**。`lerobot_pipeline.py` 已把 week11 转成 v3（含视频），本周从 v3 读回图像/状态/动作进行训练，避免第二份数据事实源。
2. **图像 feature 采用 `FeatureType.VISUAL`，shape `(3, H, W)`**。LeRobot 官方 ACT 的 VISUAL 统计按通道 mean/std（已确认 `compute_stats` 对 RGB 除以 255 后按 `(C,1,1)` 保留）；preprocessor 负责缩放。读取时 `return_uint8=True` 拿原始 uint8，训练内部转 `[0,1]` float32。
3. **分辨率可配置，默认 224**。ResNet18 ImageNet 预训练以 224×224 为基准；v3 里是 512×512，训练前缩放到配置分辨率（`--resolution 224`）。显存/时长随分辨率上升，作为受控参数记录。
4. **proprioception 复用第 13 周 5 维 robot state**（`tcp_pos[3] + gripper_qpos[2]`）+ 13 维 environment state；本周不引入 qvel（保持与 v3 数据集 feature 一致，后续如需再加）。
5. **训练沿用第 13 周协议**：每任务独立 seed、唯一 transition normalization、周期 checkpoint + dev rollout 选择、`weights_only` 重载、禁 scripted fallback 评测。
6. **rollout 用 live ManiSkill 环境渲染 RGB**（第 15 周已验证 `make_render_env` → `set_state_dict → render` 512×512）。评测时每步先 `env.reset(seed)` 再按帧 `set_state_dict → render` 得到与训练同域的图像。
7. **显存/时长记录**：训练记录 device、峰值显存（MPS/CPU 用 RSS 或 `torch.mps` 不可得则记录常驻内存）、总时长、单 step 推理时延；rollout 记录每 episode 步数与平均推理时延。
8. **先做小分辨率/小模型 smoke，再做正式训练**：正式训练用 resnet18 + 224×224 + chunk 8/n4；smoke 用 64×64 + 小 dim_model 验证链路。

### 1.3 范围边界

本周包含：

- 从 v3 数据集读回图像/状态/动作并构造 action chunk 数据集；
- VISUAL feature 的 ResNet18 backbone + ACT transformer 训练；
- 分辨率、batch、dim_model 等显存/时长受控参数；
- checkpoint 保存/重载（含 image processor 统计量）；
- live ManiSkill RGB 渲染 rollout 与固定 seed 评测；
- 显存、时长、推理时延记录。

本周不包含：

- 多相机、深度、语言条件、SmolVLA 或世界模型；
- 真实硬件、ROS 2、CUDA、云 GPU 或 Hub 上传；
- 修改第 11/13/15 周数据、checkpoint 或 pipeline；
- 修改 ManiSkill 环境、成功条件或动作空间；
- 视觉 ACT 门禁判定（第 22 周阶段门禁 C）。

## 2. 版本与设备设计

### 2.1 运行环境

- 复用 `.venv-lerobot`；`lerobot[dataset]==0.6.1`、torchvision 可用、`pip check` 通过。
- MPS 为主设备；若 MPS 不支持某算子则回退 CPU 并记录。
- 记录 Python、LeRobot、Torch、TorchVision、ManiSkill、SAPIEN、Pinocchio 版本。

### 2.2 设备要求

- Apple M4、16 GB 内存：224×224、batch 16、resnet18 可训练（预计数 GB 峰值）。
- 必须记录 device、峰值内存/显存、训练总时长、单 step 推理时延；不能只报告成功率。
- 显存测量：CUDA 用 `torch.cuda.max_memory_allocated`；MPS/CPU 记录 `resource.getrusage` 峰值 RSS（MB）。

## 3. 架构设计

```text
runs/maniskill_lerobot/week15-cam/  (LeRobotDataset v3, 512×512)
        │
        ▼
visual_act.load_visual_dataset()
        │  读回 images/state/env_state/action，按任务分组
        ▼
VisualACTChunkDataset
        │  image (T,3,H,W) uint8→float[0,1]
        │  robot_state (T,5), env_state (T,13)
        │  action_chunk (T,8,4), action_is_pad (T,8)
        ▼
compute_visual_act_stats()  （只从训练 episode 统计）
        │  VISUAL: 每通道 mean/std (3,1,1)
        │  STATE/ENV/ACTION: 每维度 mean/std
        ▼
make_visual_act_config() / make_visual_act_policy()
        │  ACTConfig(VISUAL+STATE+ENV→ACTION) + pre/post
        ▼
train_visual_act()
        │  周期 checkpoint + best-validation + dev rollout 选择
        ▼
runs/maniskill_act/week16-<tag>/
        ├── <task>/best-rollout-dev/ + step-*/ + task_metadata.json
        ├── training_summary.json
        ├── evaluation.json
        └── resource_report.json
        ▼
VisualACTPolicy wrapper + make_render_env
        ▼
evaluate_visual_act()（固定 seed，live RGB 渲染，无 fallback）
```

### 3.1 模块职责

新增 `src/visual_act.py`：

- v3 数据集读取（按任务、按 episode 分组）；
- `VisualACTChunkDataset`（图像 + 5/13 state + action chunk + pad mask）；
- 训练集统计量（VISUAL 每通道 mean/std、STATE/ENV/ACTION mean/std）；
- `ACTConfig`（VISUAL + STATE + ENV）与 pre/post processors；
- 双任务训练、周期/best-validation checkpoint、dev rollout 选择；
- `VisualACTPolicy` rollout wrapper（live observation + 渲染图 → 5/13 + 图像 → select_action）；
- 固定 seed 评测与显存/时长报告。

新增 `src/run_visual_act.py`：

- 参数校验与默认输出目录；
- 训练、smoke、固定评测 CLI；
- 以退出码表达训练/评测是否通过。

新增 `src/tests/test_visual_act.py`：

- 不依赖 Unity 的 v3 读取、chunk、统计、配置、checkpoint 错误路径测试；
- fake dataset/fake env 的 rollout 与资源报告测试；
- 真实 v3 数据集的小分辨率 smoke 训练与 rollout（每任务 1–2 episode）。

复用（不修改或仅最小修改）：

- `src/lerobot_pipeline.py`：`make_render_env()`、`_image_hwc()`、`open_lerobot_dataset()`；
- `src/lerobot_act.py`：`ROBOT_STATE_DIM`、`ENV_STATE_DIM`、`split_lerobot_act_state()`、`ACTTrainingConfig` 相关原子校验函数；
- `src/maniskill_intro.py`：`run_episode_in_env()`、`ManiSkillTaskConfig`。

## 4. 数据结构

### 4.1 训练配置

```python
@dataclass(frozen=True)
class VisualACTTrainingConfig:
    dataset_dir: Path          # week15-cam v3 目录
    repo_id: str
    output_dir: Path
    tasks: tuple[str, ...] = SUPPORTED_TASKS
    resolution: int = 224      # 正方形边长
    chunk_size: int = 8
    n_action_steps: int = 4
    training_steps: int = 20_000
    batch_size: int = 16
    validation_interval: int = 250
    checkpoint_interval: int | None = 2500
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    n_vae_encoder_layers: int = 4
    latent_dim: int = 32
    use_vae: bool = True
    kl_weight: float = 10.0
    dropout: float = 0.1
    random_seed: int = 20260816
    device: str = "mps"
```

### 4.2 v3 数据集读取

- `LeRobotDataset(repo_id, root=dataset_dir, return_uint8=True)`。
- 按 `dataset[index]["episode_index"]` 分组；`task` 文本映射到 env_id（"PickCube-v1"/"PushCube-v1"）。
- 每帧：`observation.images.top`（uint8 CHW）、`observation.state`（5）、`observation.environment_state`（13）、`action`（4）、`reward`、`success`、`next.done`。
- 图像缩放：`uint8 CHW → float[0,1] → (3, H, W)`（双线性插值到 `resolution`）。

### 4.3 action chunk

- 与第 13 周一致：episode 内 `action[t:t+chunk]` + `action_is_pad`；不跨 episode；尾部零填充并标记 pad。

### 4.4 feature schema

```python
input_features = {
    "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, resolution, resolution)),
    "observation.state": PolicyFeature(FeatureType.STATE, (5,)),
    "observation.environment_state": PolicyFeature(FeatureType.ENV, (13,)),
}
output_features = {"action": PolicyFeature(FeatureType.ACTION, (4,))}
```

### 4.5 checkpoint 与运行产物

```text
runs/maniskill_act/week16-<tag>/
├── PickCube-v1/
│   ├── best-rollout-dev/  + step-*/ + best-validation/
│   ├── task_metadata.json
│   └── metrics.jsonl
├── PushCube-v1/
│   └── ...
├── training_summary.json
├── evaluation.json
├── resource_report.json
└── comparison_visual.json
```

metadata 记录：架构（含 vision_backbone、resolution）、数据源（v3 目录 + repo_id）、每通道图像统计量、train/validation episode ID、best step、训练/验证 loss、显存与时长。

## 5. 接口设计

```python
def load_visual_dataset(
    config: VisualACTTrainingConfig,
) -> dict[str, VisualTaskData]:
    """从 v3 读回图像/状态/动作并按任务、episode 分组。"""


def build_visual_chunk_dataset(
    data: VisualTaskData,
    indices: Sequence[int],
    chunk_size: int,
    resolution: int,
) -> VisualACTChunkDataset:
    """构造含图像的 action chunk 数据集。"""


def compute_visual_act_stats(
    data: VisualTaskData,
    train_indices: Sequence[int],
    resolution: int,
) -> dict[str, dict[str, np.ndarray]]:
    """只从训练 episode 计算 VISUAL/STATE/ENV/ACTION 统计。"""


def make_visual_act_config(
    config: VisualACTTrainingConfig,
) -> object:
    """创建 VISUAL+STATE+ENV 的 ACTConfig。"""


def make_visual_act_policy(
    config: VisualACTTrainingConfig,
    dataset_stats: Mapping[str, Mapping[str, object]],
) -> tuple[object, object, object, object]:
    """创建 ACTPolicy 与 pre/post processors。"""


def train_visual_act(
    config: VisualACTTrainingConfig,
) -> VisualACTTrainingSummary:
    """对两个固定任务训练并保存最佳 LeRobot 视觉 ACT checkpoint。"""


def load_visual_act_checkpoints(
    output_dir: Path,
    *,
    device: str,
    resolution: int,
    tasks: Sequence[str] | None = None,
    checkpoint_subdir: str | None = None,
) -> dict[str, "VisualACTPolicy"]:
    """只从本地目录加载视觉 ACT policy 与 processors。"""


class VisualACTPolicy:
    """live observation + 渲染图 → 视觉 ACT rollout wrapper。"""
    def reset(self) -> None: ...
    def act(self, observation: Mapping[str, object], image: np.ndarray) -> np.ndarray: ...
```

`VisualACTPolicy.act()` 的处理顺序固定为：

```text
live observation → 18D state → 5/13 拆分
渲染图 → float[0,1] → (3,H,W)
→ preprocessor → ACTPolicy.select_action() → postprocessor
→ shape/finite/action-space 校验 → float32[4]
```

`reset()` 必须清空 LeRobot 内部 action queue，避免跨 episode 泄漏。

## 6. 训练与评测数据流

### 6.1 训练

1. 读取 v3 数据集（`return_uint8=True`），按任务与 episode 分组。
2. 固定前 80 episode 为 train、后 20 为 validation（与第 13 周一致）。
3. 每任务独立 seed（`random_seed + task_index`），在模型创建前生效。
4. 只用 train episode 计算 VISUAL/STATE/ENV/ACTION 的 mean/std。
5. 每个 transition 构造不跨 episode 的 8 步 action chunk；图像缩放到 `resolution`。
6. batch 经 preprocessor 后传入 `ACTPolicy.forward()`。
7. 每 250 step 评估完整 validation set；只按 validation 总 loss 保存 best-validation。
8. 每 `checkpoint_interval` 保存周期 checkpoint；dev seeds（150–169）rollout 选择 `best-rollout-dev`。
9. 从磁盘重载后完成 smoke rollout 才标记训练完成。

### 6.2 固定评测

- 固定 seed 集（如 500–549 每任务 50 个，与历史不重叠）。
- 每个 episode：`make_render_env` → `reset(seed)` → 逐帧 `set_state_dict → get_obs → state` + `render → image` → `VisualACTPolicy.act(observation, image)`。
- 记录 success、steps、inference_ms、非有限/越界 action、异常。
- 禁 scripted fallback。

### 6.3 显存/时长记录

- 训练：MPS/CPU 用 `resource.getrusage().ru_maxrss`（MB）；CUDA 用 `torch.cuda.max_memory_allocated`。
- 训练前/后 RSS 差值与峰值；总训练时长；单 step 平均耗时。
- 评测：每 episode 平均推理时延、总评测时长。
- 写入 `resource_report.json`。

## 7. 失败处理与降级

| 失败 | 处理 |
| --- | --- |
| v3 数据集缺失或字段缺失 | 拒绝训练，不静默跳过 |
| 分辨率非法或非正方形 | 拒绝配置 |
| VISUAL 统计计算中图像含 NaN/Inf | 拒绝训练 |
| MPS forward/backward 不支持或 NaN | 回退 CPU，记录 device 与性能差异 |
| 图像 shape 或 dtype 不符 | 该帧拒绝，不 reshape 猜测 |
| checkpoint/processor 缺失 | 拒绝评测，不使用内存中模型替代 |
| action 非有限或 shape 不为 `(4,)` | 当前 episode 失败并记录错误 |
| 峰值显存超出阈值 | 记录并提示降低分辨率/batch |

## 8. 验收标准

1. 从 v3 数据集正确读回图像/状态/动作，构造含图像的 action chunk。
2. VISUAL/STATE/ENV/ACTION 统计量只来自训练 episode，图像按 `(3,H,W)` 存储。
3. resnet18 + ACT 在 MPS 上完成真实 forward/backward（loss 有限、梯度有限）。
4. 每任务产生完整 checkpoint 与 processors，可本地重载。
5. 固定 seed 上完成 live RGB 渲染 rollout，无 scripted fallback。
6. 显存（RSS/`max_memory_allocated`）、训练时长、单步推理时延全部落盘。
7. 分辨率、batch 为受控参数，显存/时长随其变化有记录。
8. 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
9. 运行产物只写入已忽略的 `runs/`。
10. 不提交模型、轨迹、密钥或 v3 数据集。

## 9. 参考资料

- [LeRobot ACT 文档](https://huggingface.co/docs/lerobot/main/act)
- [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- 第 13 周 `docs/week13/design.md`（训练协议、5/13 feature、checkpoint）
- 第 15 周 `docs/week15/design.md`（v3 管线、make_render_env、_image_hwc）

## 10. 实际偏差

### 10.1 实现偏差（2026-08-10）

1. **`VisualTaskData` 不重建 18D**：v3 数据集直接提供 `observation.state[5]` 与 `observation.environment_state[13]`，`load_visual_dataset()` 直接保存 `robot_state` / `env_state` 两个字段，不再经 18D 拆分。`build_visual_chunk_dataset()` / `compute_visual_act_stats()` 直接消费。
2. **chunk 越界切片 bug**：`data.actions[t:t+chunk]` 在 episode 边界处会越过本 episode 取到下一 episode 的动作。改为用 episode offsets 计算 `available = end - t`，不足时显式 pad，`action_is_pad` 语义与第 13 周一致。
3. **`ACTION_IS_PAD_KEY` 常量补入 `visual_act.py`**（来自 LeRobot batch 契约）。
4. **macOS `ru_maxrss` 为字节**：`_peak_rss_mb()` 按 Darwin 除以 `1024²`、Linux 除以 `1024`；`run_visual_act.py` 的 `_peak_rss()` 同样修正。
5. **CLI 补充架构参数**：`--n-encoder-layers`、`--n-decoder-layers`、`--n-vae-encoder-layers`、`--latent-dim` 已加入 `run_visual_act.py`。
6. **`load_visual_act_checkpoints()` 从 `config.json` 重建 `VisualACTTrainingConfig`**：读取 `vision_backbone`、`dim_model` 等完整架构参数；`pretrained_backbone_weights=None` 避免重载时联网。

### 10.2 评测与训练结果（2026-08-10）

**真实训练**（`runs/maniskill_act/week16-visual/`，resnet18 + 128×128 + batch 8 + chunk 8/n4 + MPS，1500 step/任务）：

| 任务 | train | validation | best_step | best validation loss | 峰值 RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| PickCube-v1 | 1512 | 392 | 1400 | 0.3456 | 2043.8 MB |
| PushCube-v1 | 1085 | 268 | 1200 | 0.1450 | 2043.8 MB |

- 训练总时长 337.5 s；训练后 `best-validation` 与 `step-*` 周期 checkpoint 均存在，可本地重载。
- `resource_report.json`：峰值 RSS 2043.8 MB、训练 337.5 s、评测 21.9 s、平均推理 150.2 ms/step、20 episode。
- 固定 seed 500–509 rollout：20/20 截断（0%），nonfinite=0。**说明**：本周验收为"完成端到端训练与 rollout 并输出显存/时长记录"，不要求成功率达标（视觉 ACT 门禁在第 22 周）；本运行只训练 1500 step 以控制会话时长，rollout 0% 是训练量不足的预期结果，非管线缺陷。

**smoke**：64×64 小模型（dim_model 64）5 step 训练 + 重载 + 1 episode rollout 全链路通过。

**结论**：视觉 ACT 训练、checkpoint、live RGB 渲染 rollout、显存/时长记录全部完成。
