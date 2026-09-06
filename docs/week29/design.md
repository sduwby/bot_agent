# 第 29 周设计文档 — ACT 训练迁移与闭环推理

## 1. 概述

**本周目标：** 在阿里云 A10 上搭建训练环境，用 demo_v23 训练 ACT 策略，拉回 checkpoint 并在本地 SO-101 上闭环推理。

**实际完成情况：** 训练环境就绪（torch 2.7.0+cu126）；30000 步训练约 80 分钟（0.16s/步）；checkpoint 经 scp 拉回本地 `outputs\act_demo_*_pretrained`；闭环推理脚本 `outputs\infer_so101.py` 可用（每步 `policy.reset()` 强制重推、自动归位、末尾保持 300 步）。验收"可部署 ACT checkpoint"达成；scripted fallback 未实现，标记为遗留。

### 1.1 前置依赖

- 第 28 周 `wangbangyan/demo_v23`（68 集 / 41915 帧）。
- 第 27 周任务与安全 SOP（归位、终止、急停、产物审计）。

## 2. 架构设计

```text
阿里云 A10 23GB（101.37.232.151）
   ├─ /opt/lerobot_env_v21（torch 2.7.0+cu126）
   ├─ HF_ENDPOINT=https://hf-mirror.com
   ├─ HF_HOME=/opt/hf_cache_v21
   └─ lerobot.scripts.train（policy=act）
        │ 30000 步，0.16s/步 ≈ 80 分钟
        ▼ scp 拉回
本地 outputs\act_demo_*_pretrained
        ▼
outputs\infer_so101.py（闭环推理）
   ├─ 每步 policy.reset() 清空 ACT action queue，强制逐步重推
   ├─ 归位到数据集首帧姿态
   └─ 固定步数策略 + 末尾 300 步保持
```

**关键点：** ACT 默认维护 action queue，约 100 步才看一次图像；闭环必须每步 `policy.reset()` 清空队列，否则视觉反馈形同虚设。

## 3. 数据结构

- 训练数据：`wangbangyan/demo_v23`（LeRobotDataset v2.1，68 集 / 41915 帧）。
- checkpoint：`outputs\act_demo_*_pretrained`（scp 自云端拉回）。
- 推理产物：`outputs\infer_run_*`（`state_action_log.csv` + `frames\`，继承第 27 周）。

## 4. 接口设计

```bash
# 云端训练
python -m lerobot.scripts.train \
  --policy.type=act \
  --dataset.repo_id=wangbangyan/demo_v23 \
  --steps=30000 --batch_size=8 --num_workers=4 --save_freq=2000
```

```python
# infer_so101.py 闭环核心（每步）
policy.reset()          # 清空 ACT action queue，强制本步重看图像
action = policy.select_action(obs)
```

## 5. 数据流

**输入：** demo_v23、A10 训练环境。
**处理：** 镜像下载数据集 → ACT 训练 30000 步（每 2000 步存 checkpoint）→ scp 拉回 → 本地闭环推理（归位 → 逐步 reset+推理 → 末尾保持 300 步）。
**输出：** 可部署 checkpoint；`outputs\infer_run_*` 审计产物。

**数据可见性操作：** 私有数据集训练时临时设为 public 供镜像下载，完成后改回 private。

## 6. 验收

- 得到可部署 ACT checkpoint 和 scripted fallback。

## 7. 实际偏差

1. **scripted fallback 未实现**：真机上不存在 scripted 策略（仿真阶段的 scripted 无法直接迁移），验收的 fallback 部分未完成，标记为遗留项。
2. 镜像下载需 `HF_ENDPOINT=https://hf-mirror.com`；私有仓库镜像不可见，临时转 public、训完改回 private，属计划外操作流程。
3. ACT action queue 的 100 步问题为调试中实测发现，以每步 `policy.reset()` 收口。
