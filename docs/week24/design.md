# 第 24 周设计文档 — 标定与限位

## 1. 概述

**本周目标：** 完成 follower / leader 双臂的零位、方向与量程标定，实测关键关节的安全运动区间，建立硬件限位与异常处置约定。

**实际完成情况：** `lerobot-calibrate` 标定完成；由 demo 轨迹实测出 gripper 与 wrist_flex 的常用区间；发现 `disable_torque_on_disconnect=True` 的丢物风险；经历一次夹爪舵机（id_=6）Overload 过载保护并固化了断电重启 SOP。验收"无越界、抖动和异常发热"达成。

### 1.1 前置依赖

- 第 23 周：双臂装配完成、电机 ID 与校准文件就绪（follower=COM6、leader=COM5）。

## 2. 架构设计

```text
lerobot-calibrate
   ├─ 零位标定（各关节机械零位对齐）
   ├─ 方向标定（正/负向与 JSON 校准参数一致）
   └─ 量程标定（0–100 归一化区间映射到实际角度）
        │
        ▼
demo 轨迹实测 ──→ 关节安全区间表（本文档第 3 节）
        │
        ▼
异常处置约定
   ├─ disable_torque_on_disconnect 风险说明
   └─ 夹爪过载（Overload）断电重启 SOP
```

## 3. 数据结构

关节实测区间（单位为度，内部 0–100 归一化，由 demo 轨迹统计得出）：

| 关节 | 状态 | 实测值 |
| --- | --- | --- |
| gripper | 闭合 | ≈ 0–10 |
| gripper | 张开 | ≈ 13–23 |
| wrist_flex | 初始（桌面姿态） | ≈ 49 |
| wrist_flex | 抬起后 | ≈ 85–94 |

关键配置与异常事实：

- `disable_torque_on_disconnect=True`：程序断开连接时舵机卸扭矩，**夹爪会松开导致丢物**。
- 夹爪舵机（`id_=6`）Overload 过载保护：长时间满扭矩夹持触发；**SOP = 断电重启清除**。

## 4. 接口设计

```bash
# 标定（同第 23 周命令，校准文件复用）
lerobot-calibrate --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm
lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm
```

限位约定以文档形式固化（本周不改代码），后续采集/推理脚本按第 3 节区间解释关节读数。

## 5. 数据流

**输入：** 已识别的双臂、校准文件。
**处理：** 零位/方向/量程标定 → demo 轨迹回放统计关节读数 → 归纳安全区间与风险点。
**输出：** 关节安全区间表、`disable_torque_on_disconnect` 风险约定、过载处置 SOP。

## 6. 验收

- 标定后无越界、无抖动、无异常发热。

## 7. 实际偏差

1. 标定过程中发生过一次夹爪舵机（id_=6）Overload 过载保护（长时间满扭矩夹持所致），按断电重启清除；该事件固化为 SOP，不计入验收失败。
2. `disable_torque_on_disconnect=True` 的丢物风险为实测发现，原计划未预见；结论写入安全约定，后续周次（第 27 周 SOP）继承。
