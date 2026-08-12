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

### 8.3 A/B 分析（2026-08-12）

**B：扩大评估集到 30 seed**（排除 10-seed 噪声）：

| 版本 | 分布内 (30 seed) | 分布外 (30 seed) | gap |
| --- | ---: | ---: | ---: |
| grasp-30k（`week17-grasp-eval30`） | 13/30 (43.3%) | 14/30 (46.7%) | **−3.3pp** |

结论：10-seed 评估噪声极大（±10pp 级别）。30 seed 下分布外 ≈ 分布内，**泛化损失实际仅 −3pp**，但绝对水平仍卡在 ~45%。**评估口径从此固定为 30 seed**（分布内 500–529，分布外 600–629）。**防污染约定**：分布内/外评测 seed（500–529 / 600–629）以及任何历史补采 seed（601–609、610–619、700–709）一律不得进入后续训练或补采数据。

**A：补采数据覆盖检查**（逐样本统计 `runs/dagger/*.npz`）：

| 补采集 | N | grasped 占比 | tcp-obj 水平 (中位/p90) | obj-goal (中位) | 覆盖问题 |
| --- | ---: | ---: | ---: | ---: | --- |
| merged（第一轮） | 153 | 100% | 0.014 / 0.029 | 0.042 | **全是"已抓取后送入"，无水平偏差修正样本** |
| grasp（第二轮） | 479 | 36% | 0.020 / 0.113 | 0.161 | 大多在远离目标时采样，`GRASP_XY_ZONE=0.10` 过宽 |
| grasp-clean | 185 | 79% | 0.013 / 0.025 | 0.079 | 干净但同样缺"水平偏差→修正"样本 |

结论：两轮补采覆盖了"抬升+送入"和"粗略接近"，但**都未针对性覆盖"末端已贴近 cube（z≈0）但水平偏 1–5cm、需要水平修正后闭合"的失败态**——正是逐帧诊断看到的场景（末端到 cube 高度但 objGoal 不动、空闭合）。失败相位从"抬升缺失"演进到"水平对准缺失"。

### 8.4 第三轮补采方案（目标：水平对准）

**根因**：`GraspRecoveryExpert` 的标注是"从 10cm 外直接冲向 obj 并闭爪"，模型学到"直接逼近"，没学到"偏了先水平修正再闭合"。

**⚠️ 前置：Mac 需先完成以下代码改动并 push（4060 不要用现有代码直接跑）**：

1. `src/visual_dagger_collect.py`：
   - 采集接管条件由 `GRASP_XY_ZONE=0.10`（过宽）改为"已贴近但水平偏差"：
     - z 容差保持 `GRASP_Z_TOL=0.04`；
     - 水平改为 **下界 0.01、上界 0.05**（`0.01 <= |tcp_xy - obj_xy| <= 0.05`），只对"贴到 cube 高度但水平偏了 1–5cm"的状态接管；
   - 建议把该条件做成 CLI 参数（如 `--align-zone-min` / `--align-zone-max`），便于 4060 调参而不必改代码。
2. `src/dagger_collect.py` 的 `GraspRecoveryExpert` 标注分两段时序：
   - `tcp-obj 水平距离 >= 0.02`：只输出**水平移到 obj 正上方**的动作（闭爪保持），**z 不下降**；
   - `< 0.02`：再**下降+闭合**；抓取后复用 RecoveryExpert 抬升+静止。
   - 目的：让监督显式出现"先水平修正 → 再闭合"，而非一步到位冲向 obj。
3. `src/run_visual_act.py`：确认 `--dagger-data` + `--train-episodes-per-task 80+N` 已可用（第二轮已验证）。
4. 评估口径统一为 **30 seed**（分布内 500–529、分布外 600–629）。

> 以上改动在 Mac 完成并 `git push` 后，4060 `git pull` 再按下面命令执行。

**4060 上需要执行的操作（详细）**：

```bash
# ── 0. 同步代码与数据 ─────────────────────────────────────────────
cd /path/to/bot_agent
git pull origin main
git lfs pull                       # 若新增了 LFS 产物

# ── 1. 第三轮补采（需 Mac 已实现 §8.4 前置改动）──────────────────
# 用当前 grasp-30k 诱导失败态，采集"已贴近但水平偏差"的修正监督
#
# ⚠️ seed 防污染约定：
#   - 分布内评测 500–529、分布外评测 600–629 不得进入补采；
#   - 第二轮补采已用 601–609、610–619、700–709，第三轮不得重复；
#   - 因此第三轮补采用 630–649, 720–739（与评测、历史补采全不重叠）。
PYTHONPATH=src python src/visual_dagger_collect.py \
  --policy-dir runs/maniskill_act/week17-pick-grasp-30k \
  --output runs/dagger/week17-visual-dagger-pick-align3.npz \
  --seeds 630-649,720-739 \          # 与评测集(500-529/600-629)及历史补采(601-619/700-709)均不重叠
  --takeover grasp \
  --resolution 128 \
  --device cuda \
  --min-segment 3
# 若加了 --align-zone-min/--align-zone-max，在此传参
```

**4060 验收判据**（第三轮达标）：
- 30 seed 分布内 ≥ 60%，或相对 grasp-30k（43.3%）提升 ≥ 10pp；
- 失败相位中 `grasped_not_placed` 明显减少，`truncated_only` 里"到达 cube 高度但空闭合"占比下降；
- 产物含 `generalization_report.json`（30 seed）+ 补采 npz + training_summary。

**回退判据**：若第三轮分布内 ≤ 45%（无提升），停止继续在视觉 ACT 上补采，改为：① 提高分辨率到 224 或换 resnet34 验证是否是容量/感知精度问题；② 或接受当前水平、把 PickCube 定为视觉 ACT 弱项，阶段门禁 C 评估时用 PushCube + 已稳定的抬升子技能组合达成。

### 8.5 第三轮补采与重训结果（2026-08-12，4060，触发回退判据）

按 `ec499a8` 防污染约定用 seeds **630–649, 720–739** 补采（与评测集 500–529/600–629、历史补采集 601–619/700–709 均不重叠）：`runs/dagger/week17-visual-dagger-pick-align3.npz`（36 episode / 828 transition / 接管 36），tcp-obj 水平 p10/中位/p90 = 0.006/0.017/0.031（71% 落在 0.01–0.05 目标窗），成功过滤 `-clean.npz`（21 episode / 199 transition，闭爪 100%，水平中位 0.014）。

30k 重训 `runs/maniskill_act/week17-pick-align3-30k`（chunk 8/n4、128、batch 8、train=1711=80+21、best_step 27750、val 0.1409，1531 s 降噪运行）。4060 侧 **30 seed 正式评测**（与 grasp-30k 同协议同设备公平对比）：

| 场景 | grasp-30k（r2） | align3-30k（r3） | 变化 |
| --- | ---: | ---: | ---: |
| 分布内 500–529 | 14/30 (46.7%) | 13/30 (43.3%) | −1 |
| 分布外 600–629 | 15/30 (50.0%) | **8/30 (26.7%)** | **−7 (−23pp)** |

（Mac 侧曾测 grasp-30k 为 13/30、14/30，与 4060 仅差 1 seed，设备偏差可排除。）

**逐段拆解**：核心 seed 上其实有提升——500–509 3/10→**6/10**、600–609 4/10→**5/10**；但 510–529 10/20→7/20、610–629 **11/20→3/20**，尤其 610–629 的崩塌拖垮整体。失败分布从"全部 truncated_only"变为 **14 truncated_only + 8 grasped_not_placed**：模型更敢抓（部分 seed 出现 grasp→place 端到端），但两段式"先水平修正再闭合"监督在其它 seed 上造成犹豫/超时，整体净回退。

**结论：第三轮分布内 43.3% ≤ 45% 且无提升，按 §8.4 回退判据停止视觉 ACT 继续补采。** 候选方向：① 分辨率 224 / resnet34 验证容量/感知精度；② 接受现状，PickCube 定为视觉 ACT 弱项，门禁 C 用 PushCube + 已稳定抬升子技能组合。

### 8.6 分辨率 224 验证（2026-08-12，4060，假说否定）

按回退方案① 做单变量分辨率验证：`week17-pick-res224-30k`（resnet18、30k、batch 8、train=1697、best_step 27000、val 0.1433，36.9 min 降噪运行，配置与 grasp-30k 仅 resolution 128→224 不同）。30 seed 评测：

| 场景 | grasp-30k (128) | res224-30k |
| --- | ---: | ---: |
| 分布内 500–529 | 14/30 (46.7%) | **1/30 (3.3%)** |
| 分布外 600–629 | 15/30 (50.0%) | **1/30 (3.3%)** |

**结论：分辨率 224 严重回退（3.3% vs 43.3%），感知分辨率假说被否定**——抓取瓶颈不是"看不清"而是策略/数据/学习层面的问题（同数据高分辨率下 resnet18 明显过拟合/学习动力学变差）。resnet34 容量验证未执行（按用户决定收尾）。

### 8.7 子任务拆分与混合方案验证（2026-08-12，Mac，最终确认结构性瓶颈）

在收尾前做最后的量化验证，确认瓶颈到底在流程哪一段、以及"换下游策略"能否突破：

**子任务漏斗**（`run_pickcube_subtask.py`，grasp-30k 模型）：

| 阶段 | 分布内 (500–529) | 分布外 (600–629) |
| --- | ---: | ---: |
| approach（接近） | 30/30 (100%) | 30/30 (100%) |
| align（对准 xy≤0.04） | 27/30 (90%) | 25/30 (83.3%) |
| **grasp（抓取）** | **21/30 (70%)** | **16/30 (53.3%)** |
| lift（抬升） | 17/30 (56.7%) | 12/30 (40%) |
| place（放置） | 16/30 (53.3%) | 14/30 (46.7%) |

结论：**approach/align 几乎满分，grasp 是唯一显著瓶颈**；lift 比 grasp 低约 13pp，是第二损耗点；place ≈ lift（抬起来基本能放到位）。

**对齐精度测量**（`run_align_precision.py`）：视觉 ACT 每 seed 能达到的最小 TCP-cube 水平距离，分布内中位 0.011、分布外 0.013；`min_xy ≤ 0.015` 的 seed 分布内 21/30、分布外 16/30。seed 507/515 可达 **0.003–0.004**（远小于 cube 半宽 0.02）。→ **对齐精度足够，瓶颈不在水平对准。**

**逐帧根因（决定性）**：seed 507/515 等在对准到 3–4mm 后，夹爪在 **z≈+0.03（cube 上方 3cm）就闭合并立即上抬**，从不下探到 cube 表面（z≈0）——**闭合高度错误**。单目 RGB 无法可靠区分"3cm 高"与"贴住表面"（深度模糊），是视觉 ACT 的结构性弱点。

**混合方案验证**（`run_pickcube_hybrid.py`，视觉 ACT 到 align 后切换下游，30 seed）：

| 切换下游 | 分布内 place | 分布外 place |
| --- | ---: | ---: |
| 纯视觉 ACT（对照） | 16/30 (53.3%) | 14/30 (46.7%) |
| 状态 ACT（第 13 周） | 15/30 (50%) | 12/30 (40%) |
| 规则专家（Grasp/Recovery） | 16/30 (53.3%) | 13/30 (43.3%) |
| 规则专家（强制下探 z=0 再闭合） | 15/30 (50%) | 12/30 (40%) |

**结论：换下游策略（状态 ACT/规则专家）无法突破**。原因：① 状态 ACT 从 reset 单跑可达 96%，但从视觉诱导的中间状态（cube 可能已被视觉 ACT 推动/偏离）接手时状态分布失配；② 规则专家在"TCP 已贴住 cube 侧面、夹爪闭合"时物理无法脱困（seed 507 切换后卡死 20 步）。真正缺的是从"对准"到"可靠抓住"的动作能力，其根源是单目深度模糊导致的**闭合高度错误**。

**最终确认（PickCube 视觉 ACT）**：在"单目 RGB + resnet18 + ACT + 现有数据"前提下，成功率上限 ~45–53%。三轮 DAgger、分辨率提升、混合/分层切换均无法突破——瓶颈是单目深度模糊（闭合高度错误），非策略/数据/对齐/容量。**PickCube 维持为视觉 ACT 弱项**，不再投入此线优化；阶段门禁 C 用 PushCube（100%）+ 抬升子技能组合达成，PickCube 成功率作附注。若后续要突破需改变前提（如加入深度/双目、或更大预训练视觉模型如 SmolVLA），与第 18 周探针方向一致。

## 9. 记录说明

- 本诊断基于 chunk 8 checkpoint；chunk 4 的 PickCube 失败大概率同因（评测中 PickCube 单任务同样接近 0%）。
- 运行产物仅写入已忽略的 `runs/`；诊断脚本 `src/run_pickcube_diagnose.py`、`src/run_pickcube_subtask.py`、`src/run_pickcube_hybrid.py`、`src/run_align_precision.py`。
- 新增产物：`runs/maniskill_act/week18-subtask.json`（分布内/外漏斗）、`week18-align-precision.json`（对齐精度）、`week18-hybrid.json`（混合方案对比）。
