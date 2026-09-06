# 第 30 周 设计文档：首次部署与失败分布

## 概述

- **本周目标**：在低速、人工看护下用已训练的 ACT 模型（act_demo_v23）执行 20 次真实闭环抓取，系统记录每轮结果，形成第一版失败分布。
- **规划原文**："首次部署：低速、人工看护下执行 20 次；记录失败视频和状态。验收：形成第一版失败分布。"
- **前置依赖**：week23-26（硬件/标定/遥操作/录制）、week28（demo_v23 数据集）、week29（act_demo_v23 checkpoint 已拉回本地）。

## 背景与动机

此前闭环测试是零散的（4 次：1 成功、2 咬空、1 夹住未抬离），无法回答"成功率多少、主要失败类型是什么"。本周把评测协议化：

- 固定考核位姿集合，消除"摆放位置"这一变量对成功率的混淆；
- 低速执行（限制单步关节增量），降低看护风险；
- 每轮自动保存状态日志与关键帧，失败类型自动初判 + 人工确认。

## 架构设计

新增 `outputs/deploy_eval_so101.py`（部署评测 CLI），复用 `outputs/infer_so101.py` 的全部基础设施（import 复用，不复制代码）：

| 组件 | 职责 |
|---|---|
| `infer_so101.py`（已有） | 模型加载、相机后端补丁、归位、观测构建、帧保存 |
| `deploy_eval_so101.py`（新增） | 评测协议编排：位置表 × 次数、逐轮执行、结果判定、汇总报告 |
| SO101Follower(max_relative_target) | 低速执行：限制每次 send_action 的关节增量 |
| 操作员 | 按提示摆放螺丝刀到指定位置；对自动判定结果确认/修正 |

### 与现有模块的关系

```
deploy_eval_so101.py
  ├── import infer_so101: load_policy / home_to_initial / build_observation_from_robot
  │                        / force_msmf_backend / save_rgb / CAM_IDX / STATE_NAMES ...
  ├── lerobot SO101Follower (COM6, 相机 fixed=2/handeye=1, max_relative_target=低速)
  └── 输出: outputs/deploy_eval_<timestamp>/
            ├── trials.csv            # 每轮一行
            ├── report.md             # 失败分布汇总
            └── trial_p<P>_t<T>/      # 每轮: state_action_log.csv + frames/
```

## 数据结构

**评测协议**（常量）：5 个考核位置（桌面左/中/右/近/远，由操作员按文字描述摆放），每位置 4 轮，共 20 轮。

**trials.csv 列**：
`trial_id, position_id, steps_run, final_gripper, final_wrist, auto_verdict, final_verdict, note`

**失败类型枚举**（auto_verdict 初判，final_verdict 人工确认后为准）：

| 类型 | 自动判定依据 |
|---|---|
| `success` | 结束时夹爪被物体挡住（gripper > 空闭基准+3）且 wrist_flex ≥ 75 |
| `grasp_empty` 咬空 | 夹爪全闭但未被挡住，wrist 抬起 |
| `grasped_not_lifted` 夹住未抬离 | 夹爪被挡住但 wrist < 75 |
| `no_grasp_attempt` 未尝试闭合 | 全程夹爪未闭合 |
| `abort` 中途异常 | 异常中断（通信错误等） |

**空闭基准**：评测开始前自动空闭夹爪一次测得（同此前验证过的方法，基准 ≈1.5，判定余量 +3）。

## 接口设计

```bash
python deploy_eval_so101.py \
  --model-dir outputs/act_demo_v23_pretrained \
  --dataset-root <demo_v23 根目录> --dataset-repo wangbangyan/demo_v23 \
  --steps 300 --hold-steps 300 \
  --max-relative-target 8.0   # 低速：单步关节增量上限（度）
```

- `--steps`：策略段步数（默认 300，覆盖"接近→闭合→抬起"实测所需 ~250 步）
- `--yes`：自动判定后不要求人工确认（批量模式）
- 每轮之间暂停，提示操作员把螺丝刀摆到下一个位置后按 Enter

## 数据流

```
每轮: 操作员摆位 → Enter → 归位 → 策略段(≤300步, 每步 policy.reset() 重推, 低速限幅)
     → 保持段(300步, 保持最后姿态) → 读取末态 → 自动判定 → 人工确认
     → 写 trials.csv 行 + 保存关键帧
全部轮次结束 → 汇总 report.md（成功率、失败类型分布、按位置分解）
```

## 安全

- 低速：max_relative_target 限制单步增量，异常时伤害可控；
- 每轮开始前有人工确认点（Enter），操作员可随时 Ctrl+C；
- 夹爪过载（Overload）应急：断电重启，该轮记为 abort；
- 保持阶段结束后脚本断开连接（disable_torque_on_disconnect=True 会松夹爪，操作员应在断开前取下螺丝刀）。
