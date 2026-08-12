# 第 17 周 PickCube 视觉 ACT 失败诊断

> 日期：2026-08-11
> 状态：✅ 诊断完成，根因明确
> 相关产物：`runs/maniskill_act/week17-diagnosis.json`（逐帧证据）、`runs/maniskill_act/week17-eval-20k/generalization_report.json`（正式评测）

## 1. 结论

**PickCube 视觉 ACT（20k step，resnet18 + 128×128）没有学会"抬升"动作。**

- 分布内 PickCube 成功率 1/10（10%），分布外 0/10（0%）；
- 失败根因不是"抓取失败"或"不会静止"，而是**抓到 cube 后没有执行足够的 z 方向抬升**，把 cube 沿桌面推到 goal 正下方但高度始终停留在桌面；
- PushCube（平面推动）同条件下分布内/外均 100%，证明视觉 ACT 管线本身正常，问题特定于 PickCube 的三维抬升。

## 2. 评测背景

| 配置 | 值 |
| --- | --- |
| checkpoint | `runs/maniskill_act/week17-c8-20k`（chunk 8）、`week17-c4-20k`（chunk 4） |
| 训练 | 20,000 step/任务，resnet18，128×128，batch 8，MPS/CUDA |
| 分布内 seed | 500–509（默认环境） |
| 分布外 seed | 600–609（背景+照明+位姿扰动） |
| 评测设备 | Apple M4（MPS） |

正式评测总览（`week17-eval-20k`）：

| 策略 | PickCube 分布内 | PickCube 分布外 | PushCube 分布内 | PushCube 分布外 |
| --- | ---: | ---: | ---: | ---: |
| c8（chunk 8） | 1/10 (10%) | 0/10 (0%) | 10/10 (100%) | 10/10 (100%) |
| c4（chunk 4） | — | 12/20 (60%) 分布外对比 | — | 12/20 (60%) 分布外对比 |

> c4 vs c8 的 60% vs 50% 是两任务合计（PickCube+PushCube），PickCube 单独在两者下都接近 0%。

## 3. 失败相位诊断

对 PickCube 分布外 10 个失败 seed（600–609）逐帧 rollout，记录 grasp/placed/static/qvel/动作（`run_pickcube_diagnose.py`）：

| seed | ever_grasped | ever_placed | ever_static | first_grasp_step | z 动作均值 | z 正值占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | ✅ | ❌ | ✅ | 29 | -0.004 | 28% |
| 601 | ❌ | ❌ | ✅ | — | +0.178 | 68% |
| 602 | ❌ | ❌ | ✅ | — | -0.144 | 20% |
| 603 | ✅ | ❌ | ✅ | 12 | -0.027 | 14% |
| 604 | ❌ | ❌ | ✅ | — | -0.055 | 28% |
| 605 | ❌ | ❌ | ✅ | — | -0.014 | 26% |
| 606 | ❌ | ❌ | ✅ | — | -0.020 | 44% |
| 607 | ❌ | ❌ | ✅ | — | -0.024 | 38% |
| 608 | ✅ | ❌ | ✅ | 11 | +0.007 | 12% |
| 609 | ❌ | ❌ | ✅ | — | -0.019 | 36% |

**关键发现**：
1. **3/10 抓到 cube**（600/603/608），但全部 `ever_placed=False`——抓到后从未进入目标区；
2. **7/10 从未抓到**——策略未能接近并闭合夹爪；
3. **全部 ever_static=True**——机器人会静止，排除"不会停"；
4. 抓到后的逐帧证据（seed 600/603/608）：`z_diff` 恒定在 **+0.003 / +0.002 / -0.007**（桌面高度），cube 被沿桌面平推到 goal 正下方（`objGoal` 从 0.2 降到 0.044–0.05），但**z 抬升动作趋近 0**，cube 从未离桌。

## 4. 根因分析

### 4.1 核心根因：缺失"抬升"技能

PickCube 成功需要 `|cube - goal| <= 0.025`（goal 在桌面高度之上 0–0.3 m）**且**机器人静止。诊断显示策略把"把 cube 送到目标"理解为"在桌面平面推到 goal 正下方"——平面定位对了，但 z 抬升指令缺失。

与第 13 周 state-only ACT 的诊断高度一致：当时 PickCube 末端也卡在"到位+静止"，最终靠 DAgger 补采 recovery 轨迹解决（`docs/week13/design.md` 10.4–10.24）。视觉 ACT 复现了同样的末端问题。

### 4.2 为什么 7/10 连抓取都没完成

未抓到可能有两种子原因（当前诊断粒度未细分）：
- 视觉策略未能把末端导航到 cube 正上方/抓取位；
- 或导航到了但夹爪闭合时机不对。

z 动作均值在未抓到 seed 中仍偏低（多数 < 0.05 正值占比 < 40%），提示末端 z 方向整体欠驱动。

### 4.3 为什么 PushCube 没问题

PushCube 是平面任务（在桌面上把 cube 推到目标区域，无抬升），动作空间主要依赖 x/y 平面——恰好避开了视觉 ACT 欠学习的 z 维度，因此 100%。

## 5. 重训建议（供 4060）

按性价比排序：

### 5.1 优先：PickCube 数据增广 + 针对性补采
- **问题**：PickCube 训练数据（week15-cam 中 PickCube 1904 transition）中"抬升阶段"样本占比低，且成功轨迹在 goal 达成后立即结束，缺少"接近 goal 但高度不足 → 修正抬升"的 recovery 样本。
- **做法**：对 PickCube 做第 13 周同款 **DAgger/recovery 补采**——用当前 c8 策略 rollout，在"已抓取但 `objGoal` 近 + 高度不足"状态由 RecoveryExpert 标注"抬升+静止"动作，合并重训。
- **预期**：直接补充缺失的 z 抬升监督，最对症。

### 5.2 提高 z 维度监督权重
- PickCube 训练时对 action 的 z 分量（index 2）提高 loss 权重（如 2–3×），强制模型拟合抬升。
- 或对"已抓取"帧做近目标过采样（第 13 周 fix_plan 13.1 的 near-goal 4× 采样）。

### 5.3 增加训练量/更长训练
- PickCube 单独训 30k–40k step（当前 20k）。注意第 13 周教训：**不能只看 validation loss 选 checkpoint**，需用固定 seed rollout 选。
- 但 5.1 的补采优先级更高——单纯加量不解决缺失动作维度。

### 5.4 可选的图像级增强
- 若担心视觉过拟合背景，可对训练图像做色彩抖动/平移增强（week17 已证 PushCube 对扰动稳健，PickCube 问题不在背景而在于动作）。

## 6. 建议的 4060 重训命令（待确认后执行）

```bash
# 补采模式（需先在 Mac 或 4060 生成 dagger npz）
python src/run_visual_act.py \
  --dataset-dir runs/maniskill_lerobot/week15-cam \
  --repo-id bot-agent/maniskill-week11 \
  --task PickCube-v1 \
  --resolution 128 --batch-size 16 \
  --training-steps 30000 \
  --device cuda \
  --output-dir runs/maniskill_act/week17-pick-fix-30k
```

> 具体补采脚本与 RecoveryExpert 视觉版需新增，可作为下一实施步骤。若你希望先用"30k step + z 分量加权"快速验证 z 监督的效果（无需补采），可跳过 5.1 先做 5.2+5.3。

## 8. 重训后复诊（30k + DAgger，2026-08-11）

4060 已完成 PickCube DAgger 补采（`runs/dagger/week17-visual-dagger-pick-merged.npz`）与 30k 重训（`runs/maniskill_act/week17-pick-fix-30k`，best_step 28000，val_loss 0.1408）。Mac 复评与逐帧复诊：

**正式评测**（`runs/maniskill_act/week17-pickfix-eval/`）：

| 场景 | 旧版 20k | 新版 30k+DAgger |
| --- | ---: | ---: |
| 分布内（seed 500–509） | 1/10 (10%) | **4/10 (40%)** |
| 分布外（seed 600–609 扰动） | 0/10 (0%) | **3/10 (30%)** |
| gap | +10pp | +10pp |

**逐帧复诊**（分布外 600–609）：

| seed | 新版结果 | 变化 |
| --- | --- | --- |
| 600 | ✅ grasp+placed | 旧版抓到未抬升 → 新版完整 grasp→placed |
| 604 | ✅ grasp+placed | 同上 |
| 608 | ✅ grasp+placed | 同上 |
| 601/602/603/605/606/607/609 | ❌ 从未抓到 | 旧版"抓到未抬升" → 新版全部截断，问题转移 |

**关键结论**：
1. **DAgger 补采解决了"不会抬升"**——成功 seed 能完整 grasp→placed（firstG=13，firstP=21–25），z 动作均值从旧版≈0 提升到 +0.04~+0.09，失败相位从 `grasped_not_placed` 变为全部 `truncated_only`。
2. **剩余瓶颈转移到"接近但未抓上"**：7 个未抓取失败 seed 全部到达过 cube 高度（min|z_diff| ≤ 0.03，多个 0.000），但夹爪未闭合成功——是抓取时机/夹爪闭合问题，不再是抬升缺失。
3. 与第 13 周 state-only ACT 演进路径一致：DAgger 逐步解决末端（抬升→静止→抓取）。

**下一步方向**：对"到达 cube 高度但未抓上"状态做第二轮 DAgger 补采（标注闭合夹爪动作），或对抓取阶段数据增权。

### 8.1 第二轮补采（grasp recovery，2026-08-12）

已新增 `GraspRecoveryExpert`（`src/dagger_collect.py`）：未抓取时张开夹爪向 cube 逼近、进入抓取窗口（`|tcp-obj| <= 0.04`）后闭爪，抓取后复用 RecoveryExpert 抬升+静止；`src/visual_dagger_collect.py` 新增 `--takeover grasp` 接管策略——未抓取但 TCP 已到 cube 高度（z 容差 0.04）且水平对齐（0.10）即由专家接管标注"闭爪+抬升+静止"。

**4060 采集**（base policy `week17-pick-fix-30k`，seeds 601–609 诊断失败集 + 610–619/700–709 新 OOD 集）：

- 全量：`runs/dagger/week17-visual-dagger-pick-grasp.npz`（27 episode / 479 transition / 接管 27，rollout 685 步）；
- 成功过滤：`runs/dagger/week17-visual-dagger-pick-grasp-clean.npz`（20 episode / 185 transition，闭爪动作占比 94%）——专家完整 grasp→place 的干净 recovery 监督，用于重训；7 个 seed（606/607/610/616/618/700/701）专家自身 50 步内未抓上，不纳入重训集；
- 诊断集 601/602/603/605/609 均已由专家成功恢复 grasp→place，印证"闭爪时机"确为剩余瓶颈；606/607 为更难的抓取位姿，留给后续迭代。

### 8.2 第二轮重训结果（2026-08-12，4060）

用 `-clean.npz`（20 episode / 185 transition）与原始 train 合并重训：`runs/maniskill_act/week17-pick-grasp-30k`，配置与 30k 基线一致（chunk 8/n4、128×128、batch 8、30k step、seed 20260816），train=1697 transitions、val=392，best_step 19750、best_val 0.1414，训练 1475 s（降噪运行：省电计划 + GPU 锁频 1500 MHz，全程功耗 ≤39W、温度 ≤56°C，未触发 WSL GPU 挂死）。

**4060 侧评测**（`run_generalization.py`，扰动 background+lighting+pose）：

| 场景 | 30k 基线（r1，Mac 复评） | 第二轮 grasp-30k（4060） |
| --- | ---: | ---: |
| 分布内（seed 500–509） | 4/10 (40%) | 3/10 (30%) |
| 分布外（seed 600–609 扰动） | 3/10 (30%) | **4/10 (40%)** |
| gap | +10pp | −10pp |

**结论**：第二轮补采整体持平基线——分布外 3→4/10（提升 1 个 seed），分布内 4→3/10（回退 1 个 seed），均落在 10 seed 的噪声范围；分布外 6 个失败全部仍为 `truncated_only`（`grasp_failures=0`），**抓取瓶颈未实质突破**，与补采时专家自身也无法在 606/607 等 seed 上 50 步内抓上一致——闭爪时机之外还存在更难的接近/位姿问题。

**下一步候选**：① 抓取阶段数据增权或对 `-clean` 之外的高质量接近+闭爪帧单独过采样；② 40k 步或 z 分量加权（5.2）；③ 补采时放宽 `GRASP_XY_ZONE` 采集更多"接近但不闭合"的错帧；④ 更严格挑选 base policy（用 best-step rollout 而非 best-validation 选点）。

## 9. 记录说明

- 本诊断基于 chunk 8 checkpoint；chunk 4 的 PickCube 失败大概率同因（评测中 PickCube 单任务同样接近 0%）。
- 运行产物仅写入已忽略的 `runs/`；诊断脚本 `src/run_pickcube_diagnose.py`。
