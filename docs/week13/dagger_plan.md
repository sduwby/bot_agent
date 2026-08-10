# 第 13 周 DAgger / recovery 补采计划

> 日期：2026-08-10
> 状态：✅ 完成（接管模式补采 + 多轮 DAgger，final 门禁通过，见 design.md 10.24）
> 前置：design.md 10.8–10.22 结论——ACT 对 PickCube 末端"送入目标区 + 静止"封顶 ~13/20（65%），BC 19/20（95%）。根因：专家数据缺少 recovery state（cube 在 goal 附近但未到位/未静止时如何修正），ACT 无法学会末端静止。

## 1. 目标与边界

### 1.1 目标

- 补采 recovery/terminal 轨迹，扩充训练集；
- 用 DAgger 迭代重训，目标 ACT dev（150–169）成功率 ≥ 同轮 BC（19/20）。

### 1.2 包含

- RecoveryExpert（基于 scripted 的 lift_to_goal + settle 规则，grasp-aware）；
- 用当前最佳 ACT（qvel+no-VAE+warm-start，dev 13/20）rollout，在末端失败态收集 (state, arm_qvel, expert_action)；
- 合并原始 train + 补采数据重训（qvel+no-VAE+warm-start）；
- dev 评测与基线对比。

### 1.3 不包含

- 修改原始 week11 数据；
- rollout 阶段 scripted fallback/assist（只用于训练数据标注）；
- 无界 DAgger 轮数（本轮最多 2 轮）；
- 修改 ManiSkill 成功条件或环境。

## 2. 根因回顾

- PickCube 成功 = `|cube-goal|<=0.025` 且 `arm qvel < 0.2`（静止）。
- ACT 失败集中在末端：cube 进入目标区时速度过高、越过、回落、振荡，无法同时满足"到位+静止"。
- 专家数据每 episode 在 success 后立即结束，几乎没有"接近 goal 但偏移 → 修正 → 静止"的 recovery 状态。
- 诊断证据见 design.md 10.4–10.22。

## 3. 数据补采设计

### 3.1 RecoveryExpert

```python
class RecoveryExpert:
    """cube 被抓取后，把 cube 送入 goal 并静止的恢复专家。"""
    # 未抓取：返回 None（不标注）
    # 抓取且 |obj-goal| >= 0.018：action = move(tcp -> goal, gripper close)
    # 抓取且 |obj-goal| <  0.018：action = hold（零增量, gripper close）
```

该规则即 scripted policy 的 `lift_to_goal` + `settle` 行为，与相位无关，直接用几何量驱动，能对 ACT 访问到的任意"已抓取近目标"状态给出正确修正动作。

### 3.2 收集流程（DAgger round 1）

对每个 seed（dev 分布与训练分布混合，先取 20 个）：

1. 用当前最佳 ACT policy 在真实环境 rollout；
2. 每步如果 `is_grasped` 且 `|obj-goal| <= 0.12`，记录 `(18 维 state, arm_qvel[7], RecoveryExpert 动作)`；
3. 继续 rollout（仍由 ACT 控制，状态来自 ACT 诱导分布）；
4. 把连续记录段切分为 episode，追加到训练集。

预期每个 seed 产生 ~10–30 个 recovery transition，20 个 seed 约 200–600 条。

### 3.3 数据格式

补采 episode 以追加形式合并进 `TaskTransitions`：

- 原始：100 episode（80 train + 20 val），episode_id 0–99；
- 补采：N 个 episode，episode_id 100+，追加到 states/actions/arm_qvel 末尾；
- 训练参数：`train_episodes_per_task = 80 + N`、`validation_episodes_per_task = 20`（val 仍是原始 0–79 之后… 实际由 split 逻辑保证前 80+N 为 train、后 20 为原始 val）。

## 4. 训练与评测

- 配置与 qvel+no-VAE+warm-start 一致（use_qvel、use_vae=False、chunk 归一化、seed 20260807、warm-start 历史 checkpoint）；
- 周期 checkpoint + dev 150–169 rollout 选择；
- 对比基线：qvel warm-start dev 13/20。

## 5. 验收

- 补采数据量、recovery 覆盖（抓取且近目标）记录在案；
- round 1 dev 成功率：≥ BC 19/20 则锁定；≥ 基线 13/20 且有提升则保留并可选 round 2；无提升则撤销并记录；
- 回归测试通过。

## 6. 预计文件

- 新增 `src/dagger_collect.py`（RecoveryExpert + 收集）；
- 修改 `src/lerobot_act.py`（补采数据合并、dagger 训练入口）；
- 新增/修改 `src/run_act_dagger.py`（收集→合并→重训→评测 CLI）；
- 更新 `docs/week13/design.md`、`docs/week13/plan.md`、总计划。
