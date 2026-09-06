# 第 25 周设计文档 — 遥操作

## 1. 概述

**本周目标：** 打通 leader→follower 的遥操作链路，验证位置同步质量与长时间稳定性，掌握录制会话的键盘控制流程。

**实际完成情况：** `lerobot-teleoperate` 与 `lerobot-record` 的 teleop 链路可用；leader(COM5)→follower(COM6) 30fps 位置同步；遥操作采集会话累计超过 30 分钟稳定；键盘控制流程确认（→/Enter 结束本阶段、← 重录、Esc 终止；Enter 映射为第 28 周补充，见该周偏差）。验收"连续遥操作 30 分钟稳定"达成。

### 1.1 前置依赖

- 第 23/24 周：双臂装配、标定完成，校准文件与串口（COM5/COM6）固定。

## 2. 架构设计

```text
leader 臂（COM5，teleoperator）
   │ 操作者手动摆动
   ▼
lerobot-teleoperate / lerobot-record（teleop 模式）
   │ 30fps 位置同步（关节角映射）
   ▼
follower 臂（COM6，robot）
   │ 同步复现姿态
   ▼
键盘控制（会话节奏）
   ├─ → / Enter：结束本阶段
   ├─ ←：重录本段
   └─ Esc：终止会话
```

- 遥操作为纯位置映射，无力反馈；同步频率 30fps。
- `lerobot-record` 复用同一 teleop 链路叠加数据采集，供第 26/28 周录制使用。

## 3. 数据结构

- 关节状态：6 关节位置（0–100 归一化），leader 读出 → follower 写入。
- 键盘事件：`→`/`Enter`（next）、`←`（rerecord）、`Esc`（stop），由 lerobot 的键盘监听线程分发到录制状态机。

## 4. 接口设计

```bash
# 纯遥操作
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm

# 遥操作 + 录制（同一链路，增加数据集参数）
lerobot-record \
  --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm \
  --dataset.repo_id=<repo> --dataset.fps=30 ...
```

## 5. 数据流

**输入：** 操作者摆动 leader 臂。
**处理：** 30fps 读取 leader 关节位置 → 映射 → 写入 follower；键盘事件控制会话阶段切换。
**输出：** follower 同步复现动作；录制模式下同步写入 episode 数据（第 26/28 周细化）。

## 6. 验收

- 连续遥操作 30 分钟稳定（无断连、无同步漂移积累、无舵机异常）。

## 7. 实际偏差

1. Enter 键映射（结束本阶段）本周尚未加入，初版仅 `→` 可用；第 28 周在 `control_utils.py` 中补充 Enter 映射以改善手动节奏，详见该周偏差记录。
