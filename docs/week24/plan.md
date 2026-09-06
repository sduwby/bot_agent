# 第 24 周实施计划 — 标定与限位

## 1. 任务分解

| 序号 | 子任务 | 产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | follower 零位/方向/量程标定 | 更新的校准 JSON | 第 23 周 |
| 2 | leader 零位/方向/量程标定 | 更新的校准 JSON | 第 23 周 |
| 3 | demo 轨迹实测关节区间 | 关节安全区间表 | 1 |
| 4 | 断开行为与过载风险排查 | 风险约定与 SOP | 1 |
| 5 | 验收（无越界、抖动、异常发热） | 验收记录 | 1–4 |

## 2. 文件结构

### 2.1 实际新增/更新

- `C:\Users\Administrator\.cache\huggingface\lerobot\calibration\robots\so101_follower\my_awesome_follower_arm.json` — 标定后更新。
- `C:\Users\Administrator\.cache\huggingface\lerobot\calibration\teleoperators\so101_leader\my_awesome_leader_arm.json` — 标定后更新。
- `docs/week24/design.md` / `docs/week24/plan.md`（本文档，回溯补写）。

### 2.2 实际使用

- `lerobot-calibrate`（标定）。
- lerobot demo 回放/采集链路（关节读数实测来源）。

## 3. 实施步骤

1. 对 follower 运行 `lerobot-calibrate`，完成零位、方向、量程标定。
2. 对 leader 重复步骤 1。
3. 运行 demo 轨迹，记录 gripper 与 wrist_flex 读数：gripper 闭合 ≈0–10、张开 ≈13–23；wrist_flex 初始 ≈49、抬起后 ≈85–94。
4. 验证断开连接行为，确认 `disable_torque_on_disconnect=True` 导致夹爪松开丢物。
5. 夹爪过载（id_=6 Overload）事件处置：断电重启清除，写入 SOP。
6. 连续运行观察，确认无越界、抖动与异常发热。

## 4. 测试计划

| 测试类型 | 方式 | 验证目标 |
| --- | --- | --- |
| 标定有效性 | 标定后逐关节运动到已知姿态 | 读数与实测区间一致 |
| 量程边界 | 慢速逼近各关节两端 | 无越界、无抖动 |
| 热观察 | 连续运行后触摸舵机 | 无异常发热 |
| 断开行为 | 正常退出采集程序 | 复现并记录丢物风险 |
| 过载恢复 | 断电重启 | Overload 状态清除 |

## 5. 验收标准

- [x] 标定后无越界、无抖动、无异常发热。

## 6. 实际偏差

1. 计划外事件：夹爪舵机（id_=6）因长时间满扭矩夹持触发 Overload 过载保护一次，按断电重启清除；形成固定 SOP 并传递至第 27 周安全协议。
2. `disable_torque_on_disconnect=True` 的丢物风险为实测新发现，未在原计划中；以文档约定形式收口，未改 lerobot 源码默认值。
