# 第 18 周腾讯云环境清单与执行步骤（SmolVLA 探针 + 微调）

> 日期：2026-08-12
> 用途：SmolVLA 适配探针的云上执行说明。代码已在 `src/smolvla_probe.py` / `src/run_smolvla_probe.py`（Mac 开发完成），零样本评测与微调在云上执行。

## 1. 云 GPU 规格

- 目标：**80G 单卡 或 48G 双卡**（L20/A10/4090 等价均可）。
- SmolVLA base = 0.5B：零样本推理 4–8GB 显存足够；20k step 微调 A100 约 4h、L20 约 6–8h。

## 2. 云环境安装

推荐 Python 3.12 + CUDA 版 PyTorch + LeRobot（含 smolvla extra）。

```bash
# 1) 系统依赖（Ubuntu）
sudo apt-get update && sudo apt-get install -y git python3.12-venv ffmpeg libgl1

# 2) 虚拟环境
python3.12 -m venv .venv && source .venv/bin/activate

# 3) 先装 CUDA 版 PyTorch（版本需匹配 lerobot 0.6.1 要求 torch>=2.7,<2.12）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4) 拉代码 + 装依赖
git clone https://github.com/sduwby/bot_agent.git
cd bot_agent
git lfs pull
pip install -e "lerobot[dataset,smolvla]==0.6.1"
pip install mani_skill==3.0.1 pin==4.1.0 h5py pillow
pip check

# 5) 登录 HF（下载模型/数据集）
huggingface-cli login    # 或设置 HF_TOKEN
```

> 若网络访问 HuggingFace 受限，设置 `HF_ENDPOINT=https://hf-mirror.com`（镜像站）。

## 3. 数据准备

第 15 周 v3 数据集已在仓库 `runs/maniskill_lerobot/week15-cam/`（含相机帧）。确认可用：

```bash
PYTHONPATH=src python -c "
from lerobot_pipeline import open_lerobot_dataset
ds = open_lerobot_dataset('runs/maniskill_lerobot/week15-cam', 'bot-agent/maniskill-week11')
print('episodes:', ds.num_episodes, 'frames:', ds.num_frames)
"
```

若云上无此数据（`.gitignore` 忽略 runs/），从 Mac 同步：`rsync -av runs/maniskill_lerobot/ user@云:.../runs/maniskill_lerobot/`。

## 4. 零样本探针（先执行）

```bash
# 单 seed smoke（验证模型加载 + 环境闭环）
PYTHONPATH=src python src/run_smolvla_probe.py \
  --model-id lerobot/smolvla_base \
  --output-dir runs/smolvla/week18-smoke \
  --device cuda --seeds 500 --task PickCube-v1

# 正式零样本：分布内 20 次（seed 500–519），PickCube + PushCube
PYTHONPATH=src python src/run_smolvla_probe.py \
  --model-id lerobot/smolvla_base \
  --output-dir runs/smolvla/week18-zeroshot \
  --device cuda --seeds 500-519 --task all

# 可选：分布外扰动 20 次
PYTHONPATH=src python src/run_smolvla_probe.py \
  --model-id lerobot/smolvla_base \
  --output-dir runs/smolvla/week18-zeroshot-ood \
  --device cuda --seeds 600-619 --task all --perturbed
```

输出 `probe_report.json` 含：每任务成功率、xy/z 动作尺度、夹爪闭合比例、动作饱和率、go/no-go 决策。

**预期**：SmolVLA base 预训练于 SO-101（6D state/3 相机/6D action），与 ManiSkill 维度不同，零样本大概率失败。探针重点看**动作是否有方向意义**（非全零、非全饱和、夹爪有变化），据此 go/no-go。

## 5. go/no-go 判据

| 情况 | 决策 |
| --- | --- |
| 任一成功 episode | **GO**（继续微调） |
| 无成功，但动作有信号（xy 尺度>0.01、饱和<0.9、夹爪 0.05–0.95 变化） | **GO**（值得微调验证） |
| 动作全零/全饱和/夹爪无变化 | **NO-GO**（停止，回退视觉 ACT + 状态 ACT 分层方案） |

## 6. 微调（仅 go 时执行）

用第 15 周 v3 数据微调。**关键**：用 LeRobotDataset 直接做输入（图像 + state + task），features 由训练脚本根据数据集推断，可覆盖掉 SmolVLA base 的 SO-101 维度。

```bash
# 需要把 v3 数据集推上 Hub 或从本地读取。
# 方式 A：本地数据集路径（LeRobotDataset 支持 root 参数）
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.root=runs/maniskill_lerobot/week15-cam \
  --dataset.repo_id=bot-agent/maniskill-week11 \
  --batch_size=4 \
  --steps=20000 \
  --output_dir=runs/smolvla/week18-finetune \
  --job_name=smolvla_maniskill \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --wandb.enable=false
```

> 若 LeRobot 的 `lerobot-train` 不接受本地 v3 数据集 root，则需先推 Hub：
> `python -m lerobot.scripts.push_dataset_to_hub --repo_id 你的账号/maniskill-week11 --root runs/maniskill_lerobot/week15-cam`

微调完成后用 `run_smolvla_probe.py` 或复用 `visual_act` 评测协议对比 PickCube/PushCube。

## 7. 产物回传

- 零样本报告：`runs/smolvla/week18-*/probe_report.json`（小文件，可 git 提交）。
- 微调产物：`runs/smolvla/week18-finetune/`（含大权重，用 git LFS 回传或 rsync）。

## 8. 注意事项

- 云上 ManiSkill 渲染：用 CPU 渲染（`physx_cpu`），不需要 NVIDIA Vulkan。
- 腾讯云镜像若缺 CUDA 版 torch，先装 torch cu126 再装 lerobot，避免 pip 覆盖成 cpu 版。
- SmolVLA base 依赖 SmolVLM2-500M-Video-Instruct 的 VLM 权重，首次加载会下载 ~1GB，网络需稳定。
