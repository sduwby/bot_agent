# 第 26 周实施计划 — 摄像头与录制

## 1. 任务分解

| 序号 | 子任务 | 产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 三摄枚举实测与用途分工 | 相机 index 对照表 | 第 25 周 |
| 2 | Windows 多相机后端修复 | `get_cv2_backend` 返回 MSMF | 1 |
| 3 | `lerobot-record` 双相机录制 | LeRobotDataset v2.1 数据集 | 2 |
| 4 | 回放校验脚本 | 导出帧对比通过 | 3 |
| 5 | 20 条可复现 episode 验收 | 验收记录 | 3、4 |

## 2. 文件结构

### 2.1 实际新增

- `cam_probe.py` — 自写相机探针，逐帧目视确认 index 与画面对应关系。
- 回放校验脚本 — 读取 LeRobotDataset 导出帧对比。
- `docs/week26/design.md` / `docs/week26/plan.md`（本文档，回溯补写）。

### 2.2 实际修改

- `C:\Users\Administrator\Downloads\lerobot\src\lerobot\cameras\utils.py` — `get_cv2_backend` 返回 `CAP_MSMF`（editable 安装直接生效）。

### 2.3 实际使用

- `lerobot-record`；相机：index 1=handeye（手臂特写）、index 2=fixed（工作台俯拍）；index 0 为笔记本摄像头（人脸），勿用。

## 3. 实施步骤

1. 用 `cam_probe.py` 枚举 0/1/2 三路并逐帧目视确认画面内容，建立 index 对照。
2. 发现 DSHOW 只能打开 index 0，修改 `get_cv2_backend` 为 `CAP_MSMF`，验证双相机同开。
3. 配置 `lerobot-record` 双相机（handeye=1、fixed=2），640x480@30fps 录制。
4. 编写回放校验脚本，导出帧与源视频对比。
5. 连续录制 20 条 episode 并逐条回放校验。

## 4. 测试计划

| 测试类型 | 方式 | 验证目标 |
| --- | --- | --- |
| 枚举正确性 | `cam_probe.py` 逐帧目视 | index 与画面对应正确 |
| 多相机同开 | MSMF 下同时打开 1、2 | 无抢占、无黑帧 |
| 录制格式 | 检查数据集目录 | parquet+mp4、640x480@30fps |
| 可复现性 | 回放脚本导出帧对比 | 20 条 episode 全部一致 |

## 5. 验收标准

- [x] 录制 20 条可复现 episode。

## 6. 实际偏差

1. 早期 fixed 误配为 index 0 采到人脸，经 `cam_probe.py` 逐帧目视确认后修正为 fixed=2、handeye=1；教训：相机 index 必须先探针确认再进配置。
2. 后端修复采用直接改 lerobot 源码（editable），未做成配置项；升级 lerobot 时需重新应用该改动。
