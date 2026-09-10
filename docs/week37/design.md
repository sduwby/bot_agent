# 第 37 周 设计文档：SmolVLA 框架（阶段 F S12 框架先行）

## 1. 概述

- **目标**：在数据/训练就绪前，先把"VLA 自然语言直驱"整条管线以**可运行框架**落地：录制协议扩展 → SmolVLA 微调训练入口 → 真机推理 runner → 评测复用；其中推理/训练用 dry-run + 现成单任务数据集（demo_v23）跑通代码链路，**不承诺泛化效果**。
- **规划原文**：§10.3 端到端任务、week18"统一模型训练/微调推迟到真机阶段（第 32 周前后）"、S12（VLA 落地，需多任务数据 + SmolVLA 微调后实现）。
- **前置**：week35 技能框架（real_*）、week36 视觉输入/路由/评测（S1–S11）、本地 lerobot 0.3.4 含 SmolVLA 策略代码。

## 2. 调研结论（约束清单）

| 项 | 结论 |
|---|---|
| SmolVLA 代码 | lerobot 0.3.4 存在：`SmolVLAPolicy`（`select_action(batch, noise=None)`，无 select_actions）、`SmolVLAConfig`（字段均有默认） |
| 处理器模块 | `smolvla_probe.py` 引用的 `processor_smolvla.py` **不存在**（week18 遗留 bug）；归一化/预处理内联在 `_prepare_batch/prepare_*` |
| 训练入口 | `--policy.type=smolvla` 可经 `policies/factory.py` 注册；`dataset_to_policy_features` 从数据集 features 推导 input/output；文本靠 `tasks.jsonl` 注入 `task` 键，**不需数据集显式文本 feature** |
| demo_v23 | 2 路相机（fixed/handeye）、6D 关节 state/action、2 条中文任务文本、68 集 —— 可直接用于"训练链路验证"（单任务效果预期差） |
| 相机契约 | 代码不硬编码相机名；由 checkpoint 的 `config.image_features` 决定槽位与顺序；**缺失路只在恰为布局末尾时可用 `empty_cameras` 补齐** |
| smolvla_base config | 本机**未缓存**、当前**无法联网核对**（代理/镜像不可达）→ 其相机布局/动作空间未知，真机接入前必须核对（见 §6 遗留） |
| state/action 语义 | smolvla_base 预训练为**末端位姿增量**空间；demo_v23 为**关节量**——微调而非零样本是正解（框架按微调设计） |

## 3. 架构设计（框架先行，微调后配）

```text
run_vla_train.py   ──► lerobot.scripts.train (--policy.type=smolvla)
   │                   数据集 demo_v23 或未来多任务集（tasks.jsonl 提供文本）
   ▼
outputs/act_.../smolvla_<job>/checkpoints/<step>/pretrained_model
   ▼ 拉回本地
run_vla_infer.py  --model-dir <smolvla checkpoint>   [新增，替换 VLARunner 占位]
   │  真机: RealEngine 取 fixed/handeye + state + task 文本
   │  每步: policy.reset()? 错——SmolVLA 用 action queue(n_action_steps)
   ▼
   policy.select_action(prepared_batch) → send_action
   │
real_evaluation.py（复用）：真实 NL 任务集批量评测 → report.md
```

### 关键设计决策

1. **推理不依赖 `processor_smolvla`**：自建薄封装 `vla_prepost.py`——
   - `prepare(batch, policy)`：把 `{"observation.state":(6,), "observation.images.fixed":(H,W,3)uint8, "observation.images.handeye":..., "task": str}` 组 batch，补 batch 维与图像 resize+pad（`config.resize_imgs_with_padding`）、uint8→[0,1]→[-1,1]（与 `prepare_images` 一致）；
   - 复用 `policy._prepare_batch` 的归一化（state/action mean/std 随 checkpoint 加载）；
   - `post(action)`：`unnormalize` + 截断到 `config.action_feature.shape[0]` → numpy。
   - **注意**：SmolVLA 自带 action queue（`deque(maxlen=n_action_steps)`），每步 `select_action` 会复用整段 chunk；与 ACT 一样需评估"每步 reset()"与"整段执行"的取舍——**框架先保留 queue 语义（v2.1 的 reset 位置待真机调）**，推理参数暴露 `--reset-per-step`。
2. **相机键与槽位是训练时确定的一等契约**：微调数据集用什么键，checkpoint 的 image_features 就是什么键 → 推理端按同一数据集键构造。为避免臆测 smolvla_base 布局，**框架默认"先微调再推理"**：图像键 = `observation.images.fixed/handeye`。
3. **动作空间以微调数据集为准**：6D 关节动作（demo_v23 口径），非 smolvla_base 末端位姿；训练/推理间同一数据集 stats。
4. **训练入口参数化**：`run_vla_train.py` 组装 `--policy.type=smolvla --dataset.repo_id=... --steps/batch_size/...`，云上执行（GPU），与 week29 ACT 云训同构。
5. **数据扩展**：录制已支持多任务文本（lerobot-record）；新增 `run_vla_collect.py` 文档化多任务采集协议（不同任务文本 + 每任务多条 demo + 位置随机化）。

## 4. 数据结构

- 推理 batch（喂 `prepare`）：
  `{"observation.state": np.ndarray(6,), "observation.images.fixed": uint8 (H,W,3), "observation.images.handeye": uint8, "task": str}`
- `VLARunnerResult`：`success/message/steps/action_trace`（复用 AgentOutcome 风格）
- 训练/推理用同一 LeRobotDataset stats（来自数据集 meta）

## 5. 接口设计

```bash
# 训练（云上 A10；demo_v23 仅验证链路，未来换多任务集）
python src/run_vla_train.py --dataset-repo wangbangyan/demo_v23 \
    --output-dir <远程输出> --job-name smolvla_demo --steps 2000 --device cuda

# 推理（真机就绪后；--dry-run 用 Mock 帧验证链路）
python src/run_vla_infer.py --mode dry-run --model-dir <checkpoint> \
    --instruction "夹起桌面上的螺丝刀" --dataset-root <demo_v23>
python src/run_vla_infer.py --model-dir <checkpoint> --instruction "..." \
    --dataset-root <demo_v23> --reset-per-step
```

`vla_prepost.py` 核心函数签名：
- `smolvla_prepare(state: np.ndarray, images: dict[str, np.ndarray], task: str, config) -> dict`
- `smolvla_post(action_tensor, config) -> np.ndarray`

## 6. 遗留（真机/数据前置，非框架可完成）

1. **联网核对 `lerobot/smolvla_base/config.json`**：确认其 image_features 相机键/顺序/数量与动作空间；决定零样本试跑或直接微调。当前代理/镜像不可达、本地无缓存。
2. **多任务数据采集**：框架默认微调路线需要 2+ 任务、每任务多条中文指令 demo（可复用 demo_v23 作链路验证）。
3. **真机闭环调参**：SmolVLA action queue 与 reset-per-step 的实际表现需真机实测（类比 ACT week29 的 queue 问题）。
4. `processor_smolvla` 缺失历史遗留：框架自实现 vla_prepost 后，`smolvla_probe.py` 的 ManiSkill 探针不再维护（冻结）。

## 7. 实际偏差（框架实现后回填，2026-09-06）

1. `run_vla_infer.py` 真实模式暂为**清晰 stub**（校验参数 + NotImplementedError + 完整链路注释），
   不加载真实模型——因本机无 smolvla checkpoint 且无法联网（代理/镜像当前不可达）。框架验证走 `--dry-run`。
2. `vla_prepost` 实测确认：输入 batch 图键可直接为数据集键（observation.images.fixed/handeye），
   prepare_images 按 config.image_features 顺序自行收集；prepost 不重复实现 resize 之外的归一化。
3. SmolVLA `select_action` 输出已 unnormalize（内部含反归一化+截断到真实 action_dim），
   `smolvla_post` 仅做 tensor→numpy + 展平，多余截断逻辑为防御性保留。
4. FakeVLAPolicy dry-run 从 0 状态起算输出全 0，属预期（验证链路而非收敛），已在输出文案注明。
5. 训练命令实测可解析（--print-only 通过）；采集 SOP 输出采集纪律与逐任务模板。
