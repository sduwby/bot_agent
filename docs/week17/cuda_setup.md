# 4060 (WSL2) CUDA 训练环境清单

> 适用范围：Windows + RTX 4060 系列（8 GB 显存）通过 WSL2 Ubuntu 训练视觉 ACT。
> 建立日期：2026-08-11。仓库以 Mac 为源，本机只负责训练与产物回传。

## 1. 目标

- 在 WSL2 Ubuntu 内用 `--device cuda` 训练第 17 周 chunk 4 视觉 ACT（`week17-c4`）。
- 仿真/渲染仍走 `physx_cpu`，不需要 GPU 仿真；GPU 仅用于训练与推理。
- 训练产物（checkpoint、`training_summary.json`）回传 Mac 统一评测。

## 2. 硬件与前置条件

| 项目 | 本机实测 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8 GB 显存 |
| 驱动 | **566.36**（Windows 侧 `nvidia-smi`，CUDA 12.7） |
| WSL | WSL 2.7.3.0，内核 6.6.114.1-microsoft-standard-WSL2 |
| 发行版 | Ubuntu 24.04 LTS（Python 3.12.3） |
| 磁盘 | WSL 镜像放在 `D:\downloads\WSL\Ubuntu-24.04` |

WSL 内 `nvidia-smi` 显示 `Driver 566.36 / CUDA 12.7`，`torch.cuda.is_available()=True`，算力 `(8, 9)`。

> **驱动必须 ≥ 550.54（CUDA 12.4）**：lerobot 0.6.1 要求 `torch>=2.7`，而 cu121/cu124 索引最高只发布到 torch 2.5.1/2.6.0；torch 2.7+ 只有 cu126+ 构建，需要驱动支持 CUDA≥12.6。原始 536.67（CUDA 12.2）驱动只能装 torch 2.5.1+cu121，无法满足 lerobot 0.6.1。

## 3. WSL2 Ubuntu 安装

官方 `wsl --install -d Ubuntu` 需要管理员权限且会触发 UAC；在非管理员会话会静默挂起。
免管理员方案：下载 Ubuntu 官方 WSL rootfs 后 `wsl --import`。

```powershell
# 1. 下载 rootfs（340 MB）
curl --ssl-no-revoke -L -o $env:TEMP\ubuntu-noble-wsl.rootfs.tar.gz `
  https://cloud-images.ubuntu.com/wsl/releases/24.04/current/ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz

# 2. 导入（目录可用 D 盘任意可写路径）
New-Item -ItemType Directory -Force -Path "D:\downloads\WSL\Ubuntu-24.04"
wsl --import Ubuntu-24.04 "D:\downloads\WSL\Ubuntu-24.04" "$env:TEMP\ubuntu-noble-wsl.rootfs.tar.gz" --version 2
wsl --list --verbose
```

> 注意：D 盘根目录无写权限，镜像目录要放到可写目录（如 `D:\downloads\WSL`）。

## 4. 系统依赖

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-venv python3-pip python3-dev git cmake build-essential pkg-config curl unzip
# ManiSkill/SAPIEN 渲染运行库
apt-get install -y libgl1 libegl1 libglib2.0-0 libgomp1 libxrender1 libxcursor1 libxdamage1 libxfixes3 libxrandr2 libxss1 libxtst6 libxi6
```

## 5. Python 环境

仓库克隆/复制到 WSL 文件系统（避免直接跑在 `/mnt/d` 慢速访问）：

```bash
cd /root
git clone https://github.com/sduwby/bot_agent.git   # 或 cp -r /mnt/d/downloads/bot_agent /root/
cd bot_agent

python3 -m venv .venv-lerobot
.venv-lerobot/bin/pip install --upgrade pip

# CUDA 版 PyTorch（cu126：torch 2.7 官方仅 cu126+ 有轮子，驱动 566.36 支持）
.venv-lerobot/bin/pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126

# 其余依赖（ManiSkill + LeRobot v3 数据管线）；torchcodec 锁 0.7.x 配对 torch 2.7
.venv-lerobot/bin/pip install -r requirements-lerobot.txt
.venv-lerobot/bin/pip install "torchcodec>=0.7,<0.8" pillow h5py pytest
.venv-lerobot/bin/pip check

# ManiSkill/SAPIEN 系统运行库 + FFmpeg（torchcodec 0.7 依赖系统 ffmpeg）
apt-get install -y libgl1 libegl1 libglib2.0-0 libgomp1 libxrender1 libxcursor1 \
  libxdamage1 libxfixes3 libxrandr2 libxss1 libxtst6 libxi6 ffmpeg
```

实测版本：

| 包 | 版本 |
| --- | --- |
| torch | 2.7.0+cu126 |
| torchvision | 0.22.0+cu126 |
| torchcodec | 0.7.0（见 §9 坑：本机 decode 崩溃，已改用 pyav） |
| mani_skill | 3.0.1 |
| lerobot | 0.6.1 |
| Python | 3.12.3 |

> 网络提示：若 PyPI/镜像较慢，可后台 `nohup pip install ... > install.log 2>&1 &` 并轮询日志；
> 下载中途被 kill 后重跑 `pip install` 会续传/跳过已装包。

## 6. 训练数据与产物同步

- **数据源**：`runs/maniskill_lerobot/week15-cam/`（含 512×512 相机帧的 LeRobotDataset v3）默认被 `.gitignore` 忽略。
- **临时放开**：Mac 端临时在 `.gitignore` 允许 `runs/` 后提交数据 commit，本机 `git pull` 拉取；数据到位后**必须恢复 `.gitignore`**，后续不再提交 `runs/`。
- **产物回传**：训练完成后把 `runs/maniskill_act/week17-c4/` 拷回 Mac（共享盘/网盘/临时 git 均可）。

## 7. 训练命令（week17-c4，chunk 4）

```bash
cd /root/bot_agent
.venv-lerobot/bin/python src/run_visual_act.py \
  --dataset-dir runs/maniskill_lerobot/week15-cam \
  --output-dir runs/maniskill_act/week17-c4 \
  --device cuda \
  --resolution 128 \
  --chunk-size 4 --n-action-steps 4 \
  --training-steps 1500 \
  --batch-size 8
```

参数口径与 week16-visual 一致（resnet18 + 128×128 + batch 8 + 1500 step），仅 chunk_size 从 8 改为 4，保证 chunk 8 vs 4 单变量对比。

smoke 已验证（2026-08-11，`runs/maniskill_act/week17-smoke/`）：CUDA 训练 10 step/任务、
checkpoint 保存重载、live rollout 全链路通过，峰值显存 960 MB（64×64、batch 2）。

### 8.1 降噪运行（风扇静音优先）

4060 为笔记本 GPU，无法改功耗上限（`nvidia-smi -pl` 不支持）；用以下方式降噪，代价是训练更慢：

```powershell
# Windows 管理员：省电计划 + GPU 锁频 1500MHz（降热→风扇慢）
powercfg /setactive SCHEME_MIN
nvidia-smi -lgc 0,1500
# 训练结束后恢复
powercfg /setactive SCHEME_MAX   # 或恢复原计划 GUID
nvidia-smi -rgc
```

## 8. 本机运行补丁（不修改仓库源码）

驱动无 NVIDIA Vulkan WSL 集成文件，且 torchcodec 0.7.0 在本机 decode 崩溃，因此在 venv
site-packages 放置 `sitecustomize_4060.py`，并在系统 `/usr/lib/python3.12/sitecustomize.py`
末尾追加 `import sitecustomize_4060`。补丁做三件事：

1. `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`（Mesa 软件 Vulkan，CPU 渲染）；
2. 对所有带 `sim_backend` 的 `gym.make` 注入 `render_backend="cpu"`（ManiSkill 默认
   `"gpu"` 会因无 CUDA Vulkan 物理设备而失败）；
3. 强制 lerobot 视频解码走 `pyav` 后端（torchcodec 0.7.0 的 `_convert_to_tensor` op
   在本组合下无实现）。

> 补丁文件在仓库外（`sitecustomize_4060.py` 源文件留存于 Windows 临时目录），重装 venv 后需重新放置。

## 9. 常见坑

| 现象 | 处理 |
| --- | --- |
| `wsl --install` 无输出挂起 | 非管理员会话触发 UAC；改用 §3 的 rootfs 导入 |
| git clone 报 `GnuTLS recv error` | 网络/代理问题；先 `git config --global http.version HTTP/1.1`，或从 Windows 已克隆仓库 `cp -r` 复制 |
| D 盘根建目录被拒 | 镜像/仓库放在可写子目录（`D:\downloads\...`） |
| lerobot 0.6.1 与 torch 版本冲突 | 0.6.1 要求 torch>=2.7，且 cu126 轮子要求驱动≥CUDA 12.6；驱动必须升到 ≥550.54 |
| torch 安装后 `nvidia_*` 包残留 | `pip uninstall torch torchvision torchcodec` 不清理依赖，重装时会被新版本覆盖 |
| torchcodec decode 崩溃（`_convert_to_tensor` 无实现） | 版本配对：torch 2.7 用 torchcodec 0.7.x；仍崩溃则用 pyav 后端（§8 补丁第 3 条） |
| ManiSkill 建环境 `ErrorIncompatibleDriver` | 驱动缺 WSL Vulkan ICD；用 §8 补丁强制 `render_backend="cpu"` + lavapipe |
| 训练显存不足 | 8 GB 下降低 `--resolution`/`--batch-size`（显存以 `torch.cuda.max_memory_allocated` 记录） |
| pip 下载被超时 kill | `nohup ... > log &` 后台执行后轮询，重跑会续传 |
