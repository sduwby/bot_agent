# 第 23 周实施计划 — SO-101 到货核验与装配

## 1. 任务分解

| 序号 | 子任务 | 产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 到货核验（舵机、支架、驱动板、线材清点） | 硬件清单核对 | 门禁 C 通过 |
| 2 | follower / leader 双臂机械装配 | 装配完成的双臂 | 1 |
| 3 | lerobot 环境确认（editable 安装可用） | 可运行的 lerobot CLI | — |
| 4 | 串口识别（follower=COM6、leader=COM5） | 固定串口配置 | 2 |
| 5 | 电机 ID 配置与逐关节校准 | 两份校准 JSON | 3、4 |
| 6 | 关节单独识别验收 | 验收记录 | 5 |

## 2. 文件结构

### 2.1 实际新增

- `C:\Users\Administrator\.cache\huggingface\lerobot\calibration\robots\so101_follower\my_awesome_follower_arm.json` — follower 校准文件。
- `C:\Users\Administrator\.cache\huggingface\lerobot\calibration\teleoperators\so101_leader\my_awesome_leader_arm.json` — leader 校准文件。
- `docs/week23/design.md` / `docs/week23/plan.md`（本文档，回溯补写）。

### 2.2 实际使用

- lerobot 官方 CLI：`lerobot-calibrate`（editable 安装于 `C:\Users\Administrator\Downloads\lerobot`）。

## 3. 实施步骤

1. 清点套件并核对硬件清单。
2. 按官方文档完成 follower 与 leader 机械装配、驱动板接线。
3. 确认本机 lerobot editable 安装可用。
4. 分别上电识别串口：follower 固定 COM6、leader 固定 COM5。
5. 对 follower 运行 `lerobot-calibrate`：逐个舵机配置电机 ID，逐关节校准。
6. 对 leader 重复步骤 5。
7. 逐关节手动摆动验证软件端可单独识别，记录验收结论。

## 4. 测试计划

| 测试类型 | 方式 | 验证目标 |
| --- | --- | --- |
| 关节识别 | `lerobot-calibrate` 逐关节提示下摆动对应关节 | 每个关节被单独识别 |
| 校准持久化 | 重新运行校准命令回读既有 JSON | 校准文件存在且被加载 |
| 连接稳定 | 双臂同时上电 | COM5/COM6 不冲突 |

## 5. 验收标准

- [x] 所有关节可单独识别。

## 6. 实际偏差

无。硬件装配与校准均按官方流程顺利完成，未出现缺件或识别失败。
