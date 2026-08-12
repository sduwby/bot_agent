# 第 18 周设计文档 — SmolVLA 适配探针

## 1. 概述

**本周目标：** 在 PickCube 视觉 ACT 收尾（第 17 周确认结构性瓶颈为单目深度模糊导致的闭合高度错误）之后，对 SmolVLA（Hugging Face 轻量 VLA，0.5B）做适配探针：映射相机、state、action 和任务文本，先测试已有 checkpoint/推理服务、不从零训练，完成固定零样本/少样本评测并作出 go/no-go 决策。

总计划原文：

> 第 18 周 | SmolVLA 适配探针 | 映射相机、state、action 和任务文本；先测试已有 checkpoint/推理服务，不从零训练。 | 完成 20 次固定零样本/少样本评测并作出 go/no-go 决策。

### 1.1 前置依赖

- 第 15 周 camera 数据集：`runs/maniskill_lerobot/week15-cam/`（LeRobotDataset v3，PickCube/PushCube 各 100 episode，512×512 相机帧 + state/action）。
- 第 16/17 周 `visual_act.py`（视觉 ACT 训练/评测）、`lerobot_pipeline.py`（`make_render_env`）、`perturbation.py`（扰动环境）、`run_generalization.py`（固定 seed 评测）。
- SmolVLA base 模型 `lerobot/smolvla_base`（0.5B，输入多相机+state+语言，输出连续 action chunk，flow matching 目标）。
- 腾讯云 GPU（本周用推理 + 可选小规模微调）：需 Linux + CUDA + Python 3.12 + LeRobot（含 `[smolvla]` extra）。

### 1.2 本周关键决策

1. **先从零样本探针开始，不从零训练**：用 `lerobot/smolvla_base` 预训练权重直接对 ManiSkill PickCube/PushCube 做推理，验证"视觉→动作"映射是否可行，输出 go/no-go。
2. **探针是 go/no-go 决策，不是追求成功率**：零样本大概率失败，重点看**是否有合理动作趋势**（接近/抓取方向正确、夹爪时序合理），以及失败模式是否区别于视觉 ACT 的"闭合高度错误"。
3. **输入映射沿用第 15 周 v3 数据契约**：`observation.images.top`（512×512→SmolVLA 要求尺寸）、`observation.state[5]` + `observation.environment_state[13]`（或直接 18D）、`action[4]`、任务文本（"Pick up the cube and place it on the target." / "Push the cube to the target."）。
4. **任务文本固定、多视角可选**：SmolVLA 支持多相机；本周先用单视角（与视觉 ACT 对齐），若时间允许再加第二视角对比。
5. **20 次固定评测 seed**：沿用第 17 周 30-seed 协议中的固定 seed（分布内 500–529 / 分布外 600–629），或至少 20 次/任务，保证与视觉 ACT 可对比。
6. **硬件：腾讯云租用**：SmolVLA base 0.5B，单 GPU（如 L20/A10/RTX 4090 等价）推理足够；20k step 微调 A100 约 4h，L20 稍慢，作为可选扩展。
7. **go/no-go 判据**：零样本评测中，若 PickCube 或 PushCube 出现 ≥1 次完整成功，或失败模式不同于视觉 ACT 的"闭合高度错误"（例如能正确闭合但方向偏），则继续小规模微调；若零样本完全无意义（动作发散/夹爪全开全闭）则 no-go。

### 1.3 范围边界

本周包含：

- SmolVLA base 模型在 ManiSkill 环境上的推理适配；
- 相机/state/action/任务文本映射；
- 固定 20 次零样本评测（PickCube + PushCube）；
- go/no-go 决策与文档；
- 可选：基于第 15 周 v3 数据的小规模微调（仅当零样本有正向信号）。

本周不包含：

- 从零训练 VLA 或大模型；
- 多相机/深度/语言条件之外的新输入；
- 真实硬件、SO-101、ROS 2；
- 修改第 15/16/17 周数据、checkpoint 或既有评测协议；
- 超过 20k step 的微调或复杂超参搜索。

## 2. 版本与设备设计

### 2.1 运行环境

- **Mac M4（开发 + 评测）**：复用 `.venv-lerobot`；SmolVLA 推理若 MPS 支持则本地跑 smoke，否则推理放云。
- **腾讯云 GPU（推理 + 可选微调）**：Linux + CUDA + Python 3.12 + `lerobot[smolvla]` + `mani_skill`（或仅推理 ManiSkill 环境用云上 CPU 渲染）。
- 记录 Python、LeRobot、Torch、SmolVLA 版本与 GPU 型号/显存。

### 2.2 设备要求

- 零样本推理：0.5B 模型，4–8GB 显存足够，L20（48GB）/ A10（24GB）/ 4090（24GB）均可。
- 若微调：batch 4 + gradient checkpointing，20k step 约 4–8h（L20 级别）。
- 必须记录推理时延、显存、成功/失败模式。

## 3. 架构设计

```text
lerobot/smolvla_base（预训练，0.5B）
        │
        ▼
smolvla_probe.py
   ├─ SmolVLAInputAdapter：从 ManiSkill observation 构造 SmolVLA 输入
   │    ├─ observation.images.top（resize 到 SmolVLA 要求）
   │    ├─ observation.state[5] + observation.environment_state[13]（或 18D）
   │    ├─ 任务文本（PickCube / PushCube 固定指令）
   │    └─ action 输出 → ManiSkill action[4]（连续）
   ├─ make_pre_post_processors（LeRobot 官方）
   └─ run_episode_in_env（ManiSkill rollout）
        │
        ▼
run_smolvla_probe.py（固定 20 次评测 CLI）
        │
        ▼
runs/smolvla/week18/
   ├─ probe_report.json（每任务成功率、失败模式、动作趋势）
   └─ go_no_go.json（决策与依据）
```

### 3.1 模块职责

新增 `src/smolvla_probe.py`：

- SmolVLA policy 加载（`SmolVLAPolicy.from_pretrained`）；
- 输入映射：observation → SmolVLA feature dict；
- 输出映射：SmolVLA action → ManiSkill action（连续 4D）；
- 单 episode 评测（复用 `run_episode_in_env` 或自写）；
- 20 次固定 seed 评测与报告。

新增 `src/run_smolvla_probe.py`：

- 参数校验与输出目录；
- 加载模型、执行评测、输出 go/no-go 决策。

复用（不修改或仅最小修改）：

- `src/lerobot_pipeline.py`：`make_render_env()`、`_image_hwc()`；
- `src/maniskill_intro.py`：`ManiSkillTaskConfig`、`run_episode_in_env()`；
- `src/perturbation.py`：扰动环境工厂；
- `src/run_generalization.py`：固定 seed 协议与失败分布逻辑。

## 4. 数据结构

### 4.1 SmolVLA 输入

```python
input_features = {
    "observation.images.top": FeatureType.VISUAL,   # 单视角（可选加 side）
    "observation.state": FeatureType.STATE,          # 5D proprioception
    "observation.environment_state": FeatureType.ENV, # 13D env state
}
output_features = {"action": FeatureType.ACTION, "action_is_pad": ...}
```

任务文本映射：

```text
PickCube-v1: "Pick up the cube and place it at the target."
PushCube-v1: "Push the cube to the target."
```

### 4.2 评测配置

```python
@dataclass(frozen=True)
class SmolVLAProbeConfig:
    model_id: str = "lerobot/smolvla_base"
    output_dir: Path
    device: str = "cuda"
    task: str = "all"              # PickCube-v1 / PushCube-v1 / all
    seeds: tuple[int, ...] = (500, ...)   # 固定 20 次
    perturbed: bool = False
    image_size: int = 224          # SmolVLA 要求的输入尺寸
```

### 4.3 报告结构

```python
@dataclass(frozen=True)
class ProbeEpisode:
    env_id: str
    seed: int
    steps: int
    success: bool
    truncated: bool
    error: str | None
    action_trend: dict[str, object]   # 接近方向、夹爪时序、z 趋势
    failure_mode: str | None

@dataclass(frozen=True)
class ProbeReport:
    model_id: str
    per_task: dict[str, object]      # 成功率、失败模式
    go_no_go: dict[str, object]      # 决策 + 依据
```

## 5. 接口设计

```python
def make_smolvla_policy(
    model_id: str,
    device: str,
    preprocessor_overrides: Mapping[str, object] | None = None,
) -> tuple[object, object, object]:
    """加载 SmolVLAPolicy + pre/post processors。"""


def observation_to_smolvla_input(
    observation: Mapping[str, object],
    image: np.ndarray,
    task_text: str,
    image_size: int,
) -> dict[str, object]:
    """ManiSkill observation → SmolVLA feature dict。"""


def run_smolvla_episode(
    config: ManiSkillTaskConfig,
    env: object,
    policy: object,
    pre: object,
    post: object,
    task_text: str,
    image_size: int,
) -> ProbeEpisode:
    """单 episode 评测，记录动作趋势与失败模式。"""


def evaluate_smolvla_probe(
    config: SmolVLAProbeConfig,
) -> ProbeReport:
    """固定 seed 评测 + go/no-go 决策并落盘。"""
```

## 6. 数据流

### 6.1 零样本推理

1. 加载 `lerobot/smolvla_base` + LeRobot pre/post processors。
2. 对每个 seed：创建 ManiSkill 环境 → reset → 逐帧构造 SmolVLA 输入（图像 resize + state + 任务文本）→ select_action → postprocess → ManiSkill action。
3. 记录每帧动作趋势（接近方向、夹爪开合、z 变化）。
4. 结束条件 success/terminated/truncated 或 max_steps。

### 6.2 go/no-go 决策

```text
go（继续微调）<=> 存在任一成功 episode
  或 失败模式 ≠ 视觉 ACT 的"闭合高度错误"（如能正确闭合但方向偏/时序差）
no-go（停止）<=> 全部失败且失败模式与视觉 ACT 相同，或动作发散
```

### 6.3 可选微调

仅当 go 时：用第 15 周 v3 数据 + `lerobot-train`（`--policy.path=lerobot/smolvla_base`，20k step，batch 4，gradient checkpointing），产物回传 Mac 复用第 16/17 周评测协议。

## 7. 失败处理与降级

| 失败 | 处理 |
| --- | --- |
| `lerobot[smolvla]` 未安装 | 明确错误并给安装指引 |
| 腾讯云 GPU 不可用 | 回退 Mac MPS 推理（0.5B 小模型可能可跑） |
| SmolVLA 输入尺寸/特征不符 | 记录并拒绝，不猜测 |
| 输出 action 非有限或 shape 不符 | 该 episode 失败并记录 |
| 零样本全失败且无意义 | no-go，回退视觉 ACT + 状态 ACT 分层方案 |
| 云上 ManiSkill 渲染不可用 | 用 CPU 渲染（第 13/17 周已验证） |

## 8. 验收标准

1. `lerobot[smolvla]` 安装且模型可加载。
2. 相机/state/action/任务文本映射完成。
3. 固定 20 次零样本评测（PickCube + PushCube）完成并落盘。
4. 每 episode 记录动作趋势与失败模式。
5. 输出 go/no-go 决策及依据。
6. 若 go：小规模微调（可选）或记录扩展计划。
7. 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
8. 运行产物只写入已忽略的 `runs/`。

## 9. 参考资料

- [SmolVLA 文档](https://huggingface.co/docs/lerobot/smolvla)
- [SmolVLA 论文](https://arxiv.org/abs/2506.01844)
- [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
- 第 17 周 `docs/week17/pickcube_diagnosis.md` §8.7（视觉 ACT 瓶颈结论）

## 10. 实际偏差

### 10.1 实现偏差（2026-08-12）

1. **SmolVLA base 预训练契约与 ManiSkill 维度不同**：`lerobot/smolvla_base` 输入为 `observation.state[6]` + 3 相机 `observation.images.camera{1,2,3}`（256×256）+ `task` 文本，输出 `action[6]`（SO-101 双臂）。与 ManiSkill（5/13 state、单相机、4D action）不同。`observation_to_smolvla_input()` 构造 6D state（tcp[3] + gripper[2] + grasped[1]）、单相机 `camera1`、4D action（截取前 4 维）。**零样本仅验证加载与动作信号；正式评测/微调需覆盖 input/output features 到第 15 周 v3 契约。**
2. **Mac 无法下载 0.5B 模型**：本地网络访问 HuggingFace 受限（20 分钟仅下载 config.json），Mac smoke 无法加载真实模型。改为**纯逻辑测试覆盖映射/决策/聚合**（8 项定向测试），真实模型加载/评测/微调全部在腾讯云执行（`docs/week18/cloud_runbook.md`）。
3. **go/no-go 判据补充**：在"存在成功"之外，增加"无成功但动作有方向意义（xy 尺度>0.01、饱和<0.9、夹爪 0.05–0.95 变化）"→ GO；全零/全饱和/夹爪无变化 → NO-GO。
4. **依赖更新**：`requirements-lerobot.txt` 改为 `lerobot[dataset,smolvla]==0.6.1`；本地补装 `timm`（SmolVLA ViT 需要）、`transformers`、`tokenizers`。

### 10.2 当前状态（2026-08-12）

- 代码：`smolvla_probe.py`、`run_smolvla_probe.py`、`tests/test_smolvla_probe.py` 已实现，8 项定向测试 + 完整回归 434 项通过。
- 云清单：`docs/week18/cloud_runbook.md` 已含腾讯云 80G/48G 安装、零样本 20 次、go/no-go、微调 20k step 命令。
- 待云上执行：零样本评测 → go/no-go → 可选微调。
