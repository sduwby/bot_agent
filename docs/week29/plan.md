# 第 29 周实施计划 — ACT 训练迁移与闭环推理

## 1. 任务分解

| 序号 | 子任务 | 产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 云端环境搭建（A10 + lerobot_env_v21） | 可用训练环境 | 第 28 周 |
| 2 | 镜像与缓存配置 | HF_ENDPOINT / HF_HOME | 1 |
| 3 | 数据集可见性处理与下载 | demo_v23 上云 | 2 |
| 4 | ACT 训练 30000 步 | 云端 checkpoint | 3 |
| 5 | checkpoint 拉回本地 | `outputs\act_demo_*_pretrained` | 4 |
| 6 | 闭环推理脚本与真机验证 | `outputs\infer_so101.py` | 5 |

## 2. 文件结构

### 2.1 实际新增

- `outputs\infer_so101.py` — 本地闭环推理脚本（每步 `policy.reset()`、归位、末尾 300 步保持）。
- `outputs\act_demo_*_pretrained` — scp 拉回的 ACT checkpoint。
- `docs/week29/design.md` / `docs/week29/plan.md`（本文档，回溯补写）。

### 2.2 云端环境（101.37.232.151，A10 23GB）

- `/opt/lerobot_env_v21`（torch 2.7.0+cu126）。
- `HF_ENDPOINT=https://hf-mirror.com`、`HF_HOME=/opt/hf_cache_v21`。

## 3. 实施步骤

1. 在 A10 上创建 `/opt/lerobot_env_v21`，安装 torch 2.7.0+cu126 与 lerobot。
2. 配置 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HOME=/opt/hf_cache_v21`。
3. 将私有数据集 `wangbangyan/demo_v23` 临时设为 public，镜像下载到云端，完成后改回 private。
4. 训练：`python -m lerobot.scripts.train --policy.type=act --dataset.repo_id=... --steps=30000 --batch_size=8 --num_workers=4 --save_freq=2000`（0.16s/步，约 80 分钟）。
5. scp 拉回 checkpoint 至本地 `outputs\act_demo_*_pretrained`。
6. 编写 `outputs\infer_so101.py`：每步 `policy.reset()` 清空 ACT action queue（否则 100 步才看一次图像）；归位到数据集首帧姿态；固定步数 + 末尾 300 步保持；产物落 `outputs\infer_run_*`。
7. 真机闭环试运行并按 SOP 审计产物。

## 4. 测试计划

| 测试类型 | 方式 | 验证目标 |
| --- | --- | --- |
| 环境 | 云端 import torch / lerobot | torch 2.7.0+cu126 可用 |
| 训练 | 日志监控 loss 与速度 | ≈0.16s/步、checkpoint 每 2000 步落盘 |
| 拉回 | 本地加载 checkpoint | 权重完整可读 |
| 闭环 | 真机试运行 | 每步重推、归位正确、末尾保持 300 步 |
| 审计 | 检查 `outputs\infer_run_*` | csv + frames 齐全 |

## 5. 验收标准

- [x] 得到可部署 ACT checkpoint。
- [ ] scripted fallback（未实现，见实际偏差）。

## 6. 实际偏差

1. **scripted fallback 未实现**：真机无 scripted 策略可迁移，该验收项标记为遗留，闭环安全暂由 SOP（急停 Ctrl+C、断开松夹爪约定）兜底。
2. 私有数据集镜像不可见，采用"临时 public → 下载 → 改回 private"的计划外操作。
3. 训练超参按计划一次通过（30000 步 / batch 8 / 0.16s/步 / 约 80 分钟），无返工。
