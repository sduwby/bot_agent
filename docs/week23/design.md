# 第 23 周设计文档 — SO-101 到货核验与装配

## 1. 概述

**本周目标：** 完成 SO-101 leader + follower 双臂的到货核验、机械装配与电机 ID 配置，使全部关节可被软件单独识别，为后续标定与遥操作打底。

**实际完成情况：** 双臂装配完成；`lerobot-calibrate` 完成电机 ID 配置与关节识别；校准文件落盘；follower 固定于 COM6、leader 固定于 COM5。验收项"所有关节可单独识别"达成。

### 1.1 前置依赖

- 门禁 C 已通过（第 22 周），决策门 2 允许启动真实硬件阶段。
- SO-101 套件到货（含舵机、支架、驱动板、USB 转串口线）。
- 本地已安装 lerobot（editable 安装，源码位于 `C:\Users\Administrator\Downloads\lerobot`）。

## 2. 架构设计

```text
SO-101 硬件
   ├─ follower 臂（执行端，COM6）
   └─ leader 臂（遥操作输入端，COM5）
        │ USB 串口
        ▼
lerobot-calibrate（官方 CLI）
   ├─ 逐个舵机配置电机 ID
   ├─ 逐关节识别与校准
   └─ 校准文件落盘 .cache\huggingface\lerobot\calibration\
        │
        ▼
lerobot Robot / Teleoperator 对象（后续周复用）
```

- follower 与 leader 各自独立的校准文件，互不混用。
- 串口号（COM5/COM6）作为连接配置固定，后续所有脚本按此连线。

## 3. 数据结构

校准文件（JSON，路径固定）：

```text
C:\Users\Administrator\.cache\huggingface\lerobot\calibration\
   ├─ robots\so101_follower\my_awesome_follower_arm.json
   └─ teleoperators\so101_leader\my_awesome_leader_arm.json
```

文件内容包含各关节电机 ID、零位偏移与方向等校准参数，由 `lerobot-calibrate` 生成与回读，不手工编辑。

## 4. 接口设计

```bash
# follower 关节识别与校准（COM6）
lerobot-calibrate --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm

# leader 关节识别与校准（COM5）
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm
```

## 5. 数据流

**输入：** 装配完成的双臂、串口连接、lerobot 环境。
**处理：** 逐个舵机上电识别 → 配置电机 ID → 逐关节校准 → 写入校准 JSON。
**输出：** 两份校准文件；双臂进入"可标定/可遥操作"状态。

## 6. 验收

- 所有关节可单独识别（`lerobot-calibrate` 逐关节响应正常，校准文件生成）。

## 7. 实际偏差

无。装配、ID 配置与校准按 lerobot 官方流程一次完成。
