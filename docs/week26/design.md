# 第 26 周设计文档 — 摄像头与录制

## 1. 概述

**本周目标：** 确定三摄像头枚举与用途分工，解决 Windows 下多相机同时打开的后端问题，建立 LeRobotDataset v2.1 录制与回放校验流程。

**实际完成情况：** 三摄枚举实测完成（0=笔记本摄像头勿用、1=handeye、2=fixed）；修改 lerobot 相机后端为 MSMF 后多相机可同时打开；录制格式 640x480@30fps；回放校验用自写脚本导出帧对比。验收"录制 20 条可复现 episode"达成。

### 1.1 前置依赖

- 第 25 周：`lerobot-record` teleop 链路可用。

## 2. 架构设计

```text
三路摄像头（USB）
   ├─ index 0：笔记本摄像头（人脸）——勿用
   ├─ index 1：handeye，手臂特写
   └─ index 2：fixed，工作台俯拍
        │ OpenCV 捕获（Windows 必须 CAP_MSMF）
        ▼
lerobot-record ──→ LeRobotDataset v2.1（parquet + mp4，640x480@30fps）
        │
        ▼
自写回放校验脚本：导出帧对比，核验图像与数据一致性
```

**后端问题与修复：** Windows 下 OpenCV DSHOW 后端只能打开 index 0，多相机必须用 MSMF。已修改 `C:\Users\Administrator\Downloads\lerobot\src\lerobot\cameras\utils.py` 的 `get_cv2_backend` 返回 `CAP_MSMF`；editable 安装，修改直接生效。

## 3. 数据结构

- **LeRobotDataset v2.1**：每 episode 一条 parquet（状态/动作序列）+ 每路相机一段 mp4；分辨率 640x480、帧率 30fps。
- **相机配置**：`handeye` → index 1，`fixed` → index 2（键名即数据集图像列名）。

## 4. 接口设计

```python
# lerobot/src/lerobot/cameras/utils.py（已修改）
def get_cv2_backend() -> int:
    return cv2.CAP_MSMF  # Windows 多相机必须；原 DSHOW 只能打开 index 0
```

```bash
# 录制（双相机）
lerobot-record ... --robot.cameras="{ handeye: {type: opencv, index_or_path: 1, ...}, fixed: {type: opencv, index_or_path: 2, ...} }"
```

回放校验：自写脚本读取 LeRobotDataset，逐 episode 导出帧并与源视频对比。

## 5. 数据流

**输入：** 双相机视频流 + 遥操作关节数据。
**处理：** MSMF 后端同步捕获 640x480@30fps → lerobot-record 对齐时间戳 → 写 parquet/mp4。
**输出：** LeRobotDataset v2.1 数据集；回放校验脚本导出帧对比确认可复现。

## 6. 验收

- 录制 20 条可复现 episode（回放帧与记录数据一致）。

## 7. 实际偏差

1. 早期把 fixed 相机误配为 index 0，采到人脸画面；经自写 `cam_probe.py` 逐帧目视确认各路实际画面对应关系后，修正为 fixed=2、handeye=1。
2. DSHOW 多相机限制为计划外发现，以修改 lerobot 源码（editable 安装）方式解决，未向上游提交。
