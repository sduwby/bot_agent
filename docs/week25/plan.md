# 第 25 周实施计划 — 遥操作

## 1. 任务分解

| 序号 | 子任务 | 产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | `lerobot-teleoperate` 链路打通 | 可用的纯遥操作 | 第 24 周 |
| 2 | 30fps 位置同步质量验证 | 同步质量结论 | 1 |
| 3 | `lerobot-record` teleop 模式验证 | 录制链路可用 | 1 |
| 4 | 键盘控制流程熟悉与确认 | 键位约定 | 3 |
| 5 | 30 分钟连续稳定性测试 | 验收记录 | 1–4 |

## 2. 文件结构

### 2.1 实际使用

- lerobot 官方 CLI：`lerobot-teleoperate`、`lerobot-record`（teleop 链路）。
- follower：COM6 / `my_awesome_follower_arm`；leader：COM5 / `my_awesome_leader_arm`。

### 2.2 实际新增

- `docs/week25/design.md` / `docs/week25/plan.md`（本文档，回溯补写）。

## 3. 实施步骤

1. 启动 `lerobot-teleoperate`，确认 leader 摆动时 follower 同步复现。
2. 观察同步频率（30fps）与延迟，逐关节检查方向一致性。
3. 切换 `lerobot-record` teleop 模式，确认录制链路叠加后遥操作不受影响。
4. 演练键盘控制：→ 结束本阶段、← 重录、Esc 终止（Enter 映射第 28 周补充）。
5. 进行累计超过 30 分钟的遥操作采集会话，记录稳定性。

## 4. 测试计划

| 测试类型 | 方式 | 验证目标 |
| --- | --- | --- |
| 同步正确性 | leader 逐关节摆动，目视 follower | 方向一致、无明显延迟 |
| 频率 | 日志观察循环频率 | ≈30fps |
| 录制叠加 | teleop 模式下录制短 episode | 遥操作与录制互不干扰 |
| 键盘控制 | 演练 next/rerecord/stop | 状态机切换正确 |
| 长稳 | 连续遥操作 ≥30 分钟 | 无断连、无漂移积累 |

## 5. 验收标准

- [x] 连续遥操作 30 分钟稳定。

## 6. 实际偏差

1. Enter 键（结束本阶段）映射本周未实现，初版只靠 `→`；第 28 周修改 `control_utils.py` 补充 Enter 映射，见该周文档。
2. 其余按计划执行，遥操作链路一次打通，无返工。
