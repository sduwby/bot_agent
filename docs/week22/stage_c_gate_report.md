# 第 22 周阶段门禁 C 评审报告

- 日期：2026-08-13
- 执行环境：4060 (WSL2)，`.venv-lerobot`，`--device cuda`
- 评测协议：固定 seed 分布内 500–529 / 分布外 600–629（与第 14/17 周同口径）

## 1. 门禁定义

> 阶段门禁 C　视觉 ACT 在随机初始位置测试中成功率至少达到 60%，且 scripted fallback、安全约束和固定评测可用。SmolVLA 只有在同测试集带来明确泛化收益时才进入真实主线；世界模型实验不再作为采购和真实硬件阶段的阻塞条件。

**达标口径**（第 17 周结论延续）：视觉 ACT 的达标任务为 **PushCube-v1**（+ 已稳定抬升子技能组合）；PickCube-v1 定为视觉 ACT 弱项（结构性瓶颈：单目深度模糊导致闭合高度错误，上限 ~45–53%），作附注指标，由 scripted/状态 ACT 兜底。

## 2. 本周固定 seed 评测结果

| 候选 | 分布内（500–529） | 分布外（600–629） | nonfinite | exceptions |
| --- | --- | --- | --- | --- |
| scripted | 100.0% | 100.0% | 0 | 0 |
| visual_act（week17-c8-20k） | 60.0%（PushCube 100%、PickCube 20%） | 51.7% | 0 | 0 |
| route visual_act→scripted | 78.3% | —（route 只做分布内） | 0 | 0 |

## 3. 历史结果（同协议，第 12–14 周）

| 策略 | PickCube-v1 | PushCube-v1 |
| --- | --- | --- |
| BC | 84% | 100% |
| 状态 ACT（week13-dagger-train-r3） | 96% | 100% |
| scripted | 100% | 100% |

## 4. 判定

- 视觉 ACT 达标任务成功率：**PushCube 100% ≥ 60%** ✅
- scripted fallback 可用（末位安全网 + 路由切换验证）：✅
- 安全约束：nonfinite=0、exceptions=0；第 9 周 `control.py` 动作限幅契约沿用：✅
- 固定评测可用（`run_stage_selection.py` + 模型卡落盘）：✅
- SmolVLA：探针推迟到真机阶段，不阻塞；世界模型支线已归档，不参与：✅

**结论：阶段门禁 C 通过。** 主策略冻结为状态输入 ACT + scripted 兜底；视觉 ACT 为真机阶段升级候选。按决策门 2，可启动 SO-101 采购流程（第 23 周）。

## 5. 产物

- `runs/maniskill_selection/week22/model_card.json`
- `runs/maniskill_selection/week22/selection_report.md`
- `src/stage_selection.py` / `src/run_stage_selection.py` / `src/tests/test_stage_selection.py`
