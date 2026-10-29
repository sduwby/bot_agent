# 模型卡 · week38 P0 深度/负载消融**冒烟 checkpoint**（PickCube / PushCube, 5k 步）

> 对象：云上产出的 P0 消融冒烟权重
> `runs/week38/ablation_smoke/<B0|B1|B2>/<PickCube-v1|PushCube-v1>/best-validation/`。
>
> **结论先行：这些 checkpoint 不可用于部署；本卡的评测数字是不可采信的**（见 §4/§5）。
> 它们仅作为"消融管线跑通"的冒烟产物与一条负结果证据保留。

## 1. 产物位置与文件

每个 `(变体, 任务)` 目录含：

```
runs/week38/ablation_smoke/<variant>/<task>/best-validation/
  config.json  model.safetensors
  policy_preprocessor.json  policy_preprocessor_step_3_normalizer_processor.safetensors
  policy_postprocessor.json policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

同目录另有 `metrics.jsonl`（逐 250 步的 `validation_loss`）与 `task_metadata.json`（训练配置与统计）。

## 2. 训练口径（三变体）

来源：各 `task_metadata.json` + `logs/ablation_smoke.log`。

| 变体 | `env_state_mode` | 额外列（`extra_features`） | `policy_env_dim`（策略实际吃到的 ENV 维） | 数据集 `dataset_env_dim` |
|---|---|---|---|---|
| **B0** | `extra_only` | `[]`（**0 个**） | **0**（完全无 ENV 输入） | 13 |
| **B1** | `extra_only` | `depth_known_size_m`, `depth_known_size_sigma_m`（**2 个** oracle 深度列） | 2 | 15 |
| **B2** | `extra_only` | 上述 2 个 + `load_gripper_permille`, `contact_flag`（**4 个**） | 4 | 17 |

- `policy_env_dim` / `dataset_env_dim` 取自 `logs/ablation_smoke.log`：
  `[B0] dataset_env_dim=13 policy_env_dim=0`、`[B1] …15…2`、`[B2] …17…4`。
- 共同训练配置（`task_metadata.json`）：**`training_steps = 5000`**、`batch_size = 64`、
  `resolution = 224`、`vision_backbone = resnet18`（ImageNet 预训练）、`chunk_size = 8`、
  `n_action_steps = 4`、ACT（`dim_model=512, n_heads=8, n_encoder_layers=4, n_decoder_layers=1,
  latent_dim=32, use_vae=true, kl_weight=10.0`）。
- 数据：`runs/week38/ablation_data/B{0,1,2}`（`total_episodes=200`、`total_frames=3257`）。
  划分按任务不同（`task_metadata.json`）：
  - **PickCube-v1**（`task_index=0`）：训练 episode 0–79（`train_transitions=1512`），验证 80–99（`validation_transitions=392`）；
  - **PushCube-v1**（`task_index=1`）：训练 episode 100–179（`train_transitions=1085`），验证 180–199（`validation_transitions=268`）。
- 环境：`device=cuda`、`lerobot=0.6.1`、`torch=2.9.1+cu126`、`mani_skill=3.0.1`、`sapien=3.0.3`；
  `git_commit=null`（未记录代码版本）。`random_seed`：PickCube `20260816`、PushCube `20260817`。

### validation_loss 首末值（真实读数，来自 `metrics.jsonl`）

| 变体 | 任务 | 首个点（step 250） | 末点（step 5000） | best（`task_metadata.best_validation_loss` @ step） |
|---|---|---|---|---|
| B0 | PickCube-v1 | 0.4055940375036123 | 0.2549841160676917 | 0.25212720218969853 @4750 |
| B1 | PickCube-v1 | 0.4158235180134676 | 0.26140294512923884 | 0.24446975029244714 @4750 |
| B2 | PickCube-v1 | 0.38738676479884554 | 0.25124190838969485 | 0.24627369763899823 @4750 |
| B0 | PushCube-v1 | 0.20396971924981075 | 0.06574544154885989 | 0.06146135559277748 @4750 |
| B1 | PushCube-v1 | 0.20719258576186736 | 0.07360385625220056 | 0.061638679744592355 @4000 |
| B2 | PushCube-v1 | 0.21842578920855452 | 0.08418235538610772 | 0.06762960152839546 @4250 |

> 三个变体的验证损失同量级、都收敛到 ~0.06（PushCube）/ ~0.25（PickCube）——
> **训练本身的损失并不能区分变体优劣**（好坏要看成功率，而成功率见 §4 的无效性）。

## 3. 评测协议（两次）

来源：`runs/week38/ablation_smoke/eval_report/`（5 种子）与 `eval_report_seed20/`（20 种子）；
日志 `logs/ablation_eval.log`、`logs/ablation_eval_seed20.log`。
任务 `['PickCube-v1','PushCube-v1']`，checkpoint 子目录 `best-validation`，数据集根 `runs/week38/ablation_data`。

## 4. 评测数字（真实读数）与**为什么不能作为 A2 的证据**

### 4.1 两次评测的真实成功率

| 变体 | 5 种子（每变体 10 集） | 20 种子（每变体 40 集） |
|---|---|---|
| B0（基线） | **0.800** | **0.725** |
| B1 | 0.300 | 0.350 |
| B2 | 0.000 | 0.025 |
| Δ(B1−B0) | −50.0 pp | −37.5 pp |
| Δ(B2−B0) | −80.0 pp | −70.0 pp |
| 判定（判据 ΔGrasp ≥ 15pt） | **fail** | **fail** |

（`eval_report.json` / `eval_report_seed20.json`；`delta_grasp_pp`、`verdict` 均为 `fail`。）

### 4.2 为什么这些数字**不能**当作 A2（"P0 相对 B0 提升 ≥ 15 个百分点"）的证据

**① 零 provider 对照直接逆转结论。** `logs/zero_probe.log`：把 B1/B2 的特权列
**全部替换成常量 0**（与训练分布一致）后重测——

```
[零 provider] B1（2 列全 0）: 成功率 0.850（17/20）
[零 provider] B2（4 列全 0）: 成功率 0.850（17/20）
```

即"喂 0"比"喂真值"高得多：原来 −50/−80pp 的落差**不是"多出的信息有害"**，
而是评测时喂了训练阶段没见过的取值造成的**分布外失配**。
（注：零 provider 用 20 集，与 5/20 种子协议不同，仅作对照，不能与上表直接相减。）

**② 训练/评测分布并排比对，B2 的两列**训练恒 0、评测严重 OOD**。** `logs/dist_probe.log`：

| 变体 | 列 | 训练 mean/std | 评测 mean/std | 评测 min/max |
|---|---|---|---|---|
| B1 | depth_known_size_m | 0.0630 / 0.0545 | 0.0336 / 0.0500 | 0.01 / 0.17 |
| B1 | depth_known_size_sigma_m | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.00 / 0.00 |
| B2 | depth_known_size_m | 0.0630 / 0.0545 | 0.0260 / 0.0384 | 0.00 / 0.17 |
| B2 | load_gripper_permille | **0.0000 / 0.0000** | **−68.1188 / 46.2533** | −111.71 / 0.00 |
| B2 | contact_flag | **0.0000 / 0.0000** | **0.6867 / 0.4638** | 0.00 / 1.00 |

**③ 根因：week15-cam 的 `is_grasped` 从未为 1。** `eval_report.json` 的 `caveat` 明写：
"week15-cam 的 `is_grasped` 从未为 1，故 B2 的 `load_gripper_permille`/`contact_flag` 两列
**恒为 0、不承载信息**"（即该数据集本来就没有抓取行为，特权列无从取值）。

**④ 口径本身混入了结构变化，差值不可归因。** 同一 `caveat` 指出：B0 = `extra_only` + 空额外列
⇒ 策略**完全无 ENV 输入**；B1/B2 在同一基线上追加 oracle 特权列，因此差值**同时含两件事**：
① 多出的信息；② ACT 因多了 ENV 通道而**多一个 1D 输入 token（结构变化）**——
`logs/ablation_smoke.log` 里 reload checkpoint 报的 `size mismatch` 正是这个结构差
（`model.encoder_1d_feature_pos_embed.weight` 2→3；`model.encoder_env_state_input_proj.weight`
512×2→512×13、512×4→512×13）。所以 Δ **不能**读成"几何/深度信号的价值"。

**综上**：评测数字既受分布外失配污染、又混入结构变化、且两条特权列本身不承载信息，
**不足以支撑 A2 的任何结论**（正向或负向皆不可）。

## 5. 不可用于部署 + 适用边界

**明确不可用于部署**，理由：

1. **oracle 特权列**：B1/B2 在评测时吃的是**真值**深度/负载特征；部署（真机/仿真闭环）拿不到
   这些 oracle 值，输入分布与训练/评测都不同。
2. **评测协议已被证明无效**：零 provider 对照把结论整个逆转（§4.2①），checkpoint 未经过任何
   **合格**评测；成功率数字不可引用。
3. **冒烟规模**：仅 `training_steps=5000`、单任务、`git_commit=null`（不可复现代码版本），
   远低于可交付水平。

**适用边界（仅此而已）**：

- 作为"`env_state_mode=extra_only` + 额外列"这条消融管线**是否跑通**的冒烟产物；
- 作为**负结果证据**：说明"直接追加 oracle 特权列并配上常量训练分布"这一做法在评测端会引入
  分布外失配，必须配零 provider 对照才能正确归因（这正是本卡要传递的方法论教训）。

## 6. 溯源（claim → 文件）

| 内容 | 来源 |
|---|---|
| 变体口径 / 训练超参 / 数据划分 | `ablation_smoke/<v>/<task>/task_metadata.json` |
| validation_loss 首末/best | `ablation_smoke/<v>/<task>/metrics.jsonl` |
| `policy_env_dim` / `dataset_env_dim` | `logs/ablation_smoke.log` |
| 5 种子评测 | `eval_report/eval_report.json`、`eval_report/eval_report.md`、`logs/ablation_eval.log` |
| 20 种子评测 | `eval_report_seed20/eval_report.json`、`eval_report_seed20/eval_report.md` |
| 零 provider 对照 | `logs/zero_probe.log` |
| 训练/评测分布并排 | `logs/dist_probe.log` |
| 结构变化 size mismatch | `logs/ablation_smoke.log` |

## 7. SmolVLA 40k 部署权重（S12 / P2 4.1，2026-10-18）

> 与本卡 §1–§6 的 P0 消融冒烟 checkpoint **不是同一批产物**；这一节记录机器人侧真正部署过的模型。

### 7.1 训练口径

| 项 | 值 |
|---|---|
| 基础权重 | `lerobot/smolvla_base`（微调，不是从零） |
| 数据 | `wangbangyan/demo_vla_all`（6 任务，120 集 / 62,318 帧，v3.0） |
| 步数 / epoch | **40,000 步 / 20.54 epoch** |
| batch / workers | 32 / 12 |
| 最终 loss | **0.032**（20k 时为 0.086） |
| 吞吐 / 显存 | 53 样本/s；7.88 GB（A10 24 GB） |
| 硬件 | A10，16 核 / 60 GB，`121.199.47.127` |
| 相机映射 | `--rename_map fixed→camera1 / handeye→camera2`（缺第 3 路，`empty_cameras=0`） |

### 7.2 部署产物与校验

- 0.6.1 原样：`runs/week38/smolvla_vla_40k/pretrained_model/`（7 文件，`model.safetensors` 906,712,520 B）
- 跨版本迁移：`pretrained_model_lr034/`（2 文件，自包含；0.3.4 可直接 `from_pretrained`）
- 回传校验：sha256 `c91ac849…846c` 与云端一致（远端 split 50 MB + 逐块下载 + 本地组装）
- 迁移工具：`src/migrate_vla_checkpoint.py`（与手工迁移产物**逐字节一致**，已实测）

### 7.3 真机评测（P2 4.1，不完整）

| 次数 | 步数 | 结果 | 末状态 |
|---|---|---|---|
| 1 | 203（提前结束） | **成功**（位置判据，图像佐证） | `wrist 76.07 / grip 7.74` |
| 2 | 402（提前结束） | **成功**（图像佐证） | `wrist 75.01 / grip 9.76` |
| 3 | 175 | **假阳性 → 记失败**（张开 19.25 未夹住） | `wrist 75.45 / grip 19.25` |
| 4 | 600（跑满） | 失败（闭合悬停） | `wrist 66.04 / grip 1.74` |
| 5 | 600（跑满） | 失败（闭合悬停） | `wrist 71.49 / grip 1.74` |

**当前统计 2 / 5**。已知变量：fixed 取景 15 px（成功两次为 ~7 px）；失败两次均为"从第 0 步起闭合、不张开"。
**不可当作成功率结论**：样本小、取景未固定、SmolVLA 采样有随机性。

### 7.4 这个模型能做什么 / 不能做什么

- **能**：在 0.3.4 环境加载、真实前向出动作、真机完成"接近→闭合→抬起"（已两次成功）。
- **不能**：给出稳定成功率（2/5）；在 15 px 取景偏差下稳定工作；保证进程退出码正确
  （收尾 `Torque_Enable=0` 过载会让 exit code 变成 1，即使任务成功）。

## 8. P2 4.5 适配对照（训练中，2026-10-18）

矩阵：`zero_shot`（不训练，锚点 = `smolvla_base`）+ `head{30,60}` + `full{30,60}`，每单元 2000 步。

- **训练入口**：`src/run_vla_adapt_smolvla.py`（进程内 patch `make_policy` 后 `apply_adaptation`，
  真正按策略冻结；`lerobot_train` 的 CLI 做不到这一点，直接跑会让 head≡full）。
- **数据量自变量**：`--dataset.episodes` 等距抽 30 / 60 集（不复制数据集）。
- **head 实测解冻比例**：`1,635,152 / 450,046,176 = 0.363%`
  （`action_{in,out}_proj` + `action_time_mlp_{in,out}` + `state_proj`）。
- **评测**：离线代理指标用 `src/run_adapt_eval_offline.py`（action L1 / 夹住带 recall·precision·F1）；
  **A6 的 ≥20pp 必须真机实测**，离线指标不能替代。

### 8.1 离线对照结果（8 次重复，均值 ± 样本标准差）

held-out 用**奇数索引 episode**（训练子集只取偶数索引，无泄漏），stride 40、每次 81 帧；
同一批帧跑 5 个模型，帧起点取 8 个 offset 以估计采样噪声（SmolVLA 流匹配有随机性）。

| 单元 | 可训练参数 | action L1 | gripper L1 | 夹住 F1 |
|---|---:|---:|---:|---:|
| head_ep30 | 0.36% | 2.680 ± 0.199 | 2.258 ± 0.512 | 0.645 ± 0.067 |
| head_ep60 | 0.36% | 2.849 ± 0.220 | 2.217 ± 0.393 | 0.633 ± 0.062 |
| full_ep60 | 100% | 2.860 ± 0.157 | 1.872 ± 0.295 | 0.710 ± 0.068 |
| full_ep30 | 100% | 3.062 ± 0.133 | 2.005 ± 0.288 | 0.690 ± 0.057 |
| zero_shot（base） | 0% | **32.688 ± 0.272** | 6.424 ± 0.169 | 0.035 ± 0.049 |

**唯一稳健的结论**：**微调本身有效**（zero_shot 32.69 vs 适配单元 2.68–3.06，差约 100× 标准差）。
**head vs full、30 vs 60 集的差异都落在重复间误差带内**（head_ep30 与 full_ep30 的 0.38 差约 1.6σ，
仍未达 95% 显著），不能据此下结论——单次 81 帧曾给出"full 过拟合"的相反印象，
8 次重复后只能算趋势，不足以定论。
**已知混淆**：head 用 bs=32、full 用 bs=16（22 GB 卡上 450M 全参数放不下），
策略轴与 batch 轴不独立；每个策略内部 30 vs 60 的 batch 一致。
**仍不可替代 A6**：离线 L1 只反映示范拟合质量，真机成功率（相对 zero_shot ≥20pp）必须实测。

### 8.2 去混淆复跑（head 也用 bs=16）

§8.1 的已知混淆是 head 用 bs=32、full 用 bs=16。把 **head 改成 bs=16** 重跑 30 / 60 集
（同 episode 子集、同 2000 步、同 8 个 offset）：

| 单元 | batch | action L1 | gripper L1 | 夹住 F1 |
|---|---:|---:|---:|---:|
| head_ep30 | 16 | **2.810 ± 0.146** | 2.176 ± 0.337 | 0.645 ± 0.059 |
| full_ep30 | 16 | 3.062 ± 0.133 | 2.005 ± 0.288 | 0.690 ± 0.057 |
| head_ep60 | 16 | 3.009 ± 0.172 | 2.491 ± 0.395 | 0.624 ± 0.088 |
| full_ep60 | 16 | **2.860 ± 0.157** | 1.872 ± 0.295 | 0.710 ± 0.068 |

- **30 集**：head 的 L1 比 full 低 **0.252（≈3.6σ）** → 去混淆后"head_ep30 L1 更优"**仍成立**。
- **60 集**：head 比 full 高 0.149（≈1.8σ）→ **不显著**。
- **夹住 F1**：30 / 60 的 head−full 差为 −0.045 / −0.086，均 <2σ → 策略轴在夹住时机上**没有可分辨差异**。
- head 自身 bs16 vs bs32：30 集 2.810 vs 2.680、60 集 3.009 vs 2.849，均 <2σ → batch 对 head 影响小。

**结论**：去混淆细化了 §8.1——"30 集时 head 的 L1 更好"是真的，但**夹住 F1 与 60 集上都没有差异**；
离线指标仍不足以选上机模型，A6 必须真机实测。

## 9. A7 离线泛化 + 40k 参考行（2026-10-18）

### 9.1 40k 参考行（**in-sample，不是 held-out**）

在 A5 的同一批 6 个 episode（`[1,21,41,61,81,101]`，stride 40、81 帧）上评 40k 部署权重：

| 模型 | action L1 | gripper L1 | 夹住 F1 |
|---|---:|---:|---:|
| 40k（40000 步 / 120 集） | **0.786** | 0.595 | **0.889** |
| head_ep30（A5） | 2.680 ± 0.199 | 2.258 ± 0.512 | 0.645 ± 0.067 |
| zero_shot（A5） | 32.688 ± 0.272 | 6.424 ± 0.169 | 0.035 ± 0.049 |

**重要边界**：这 6 集**全部参与过 40k 的训练**（40k 用了全量 120 集），所以它是"拟合上界/参考行"，
**不能**读成 40k 的泛化能力，也不能与 A5 的 held-out 数字直接做同口径比较。
它的用途是给出"训练充分时同批评测能达到什么水平"（L1 0.79 / F1 0.89）。

### 9.2 A7 三协议（held-out 组合 vs seen 组合）

**协议**：按 `run_vla_generalization.py --dry-run` 的 split 排除 held-out 组合的 episode，只训 `head`
（2000 步、bs 32、DONE 幂等）；评测用 `--even` 在**两侧全部 episode** 上均匀取 160 帧，同一模型
分别在 held-out / seen 上评，3 次重复（同批帧，只重跑推理）。

| 协议 | 侧 | action L1 | gripper L1 | 夹住 F1 | GT 夹住占比 |
|---|---|---:|---:|---:|---:|
| combo（螺丝刀pick + 黑色长条pick 留出） | held-out | 2.779 ± 0.083 | 1.795 ± 0.061 | 0.513 ± 0.064 | 13.2% |
| | seen | 3.010 ± 0.120 | 1.568 ± 0.091 | 0.558 ± 0.022 | 11.6% |
| novel_object（指甲刀全部留出） | held-out | 2.936 ± 0.127 | 1.965 ± 0.099 | 0.586 ± 0.030 | 8.8% |
| | seen | 2.849 ± 0.065 | 2.009 ± 0.085 | 0.523 ± 0.038 | 13.7% |
| paraphrase（每种物体留 1 条指令） | held-out | 3.418 ± 0.124 | 1.961 ± 0.117 | 0.499 ± 0.114 | 13.7% |
| | seen | 2.797 ± 0.061 | 1.786 ± 0.055 | 0.464 ± 0.063 | 10.3% |

**gap（held-out − seen）**：

| 协议 | head L1 gap | head F1 gap | 40k L1 gap（集合难度对照） | 40k F1 gap |
|---|---:|---:|---:|---:|
| combo | −0.232 ± 0.084 | −0.045 ± 0.048 | −0.233 | −0.066 |
| novel_object | +0.087 ± 0.082 | +0.063 ± 0.028 | +0.143 | −0.009 |
| paraphrase | **+0.621 ± 0.079** | +0.035 ± 0.075 | +0.112 | +0.203 |

40k 在两个侧都见过 → 它的"gap"只反映**集合难度差**（−0.23 … +0.14 L1）。
**读出**：
1. `combo` / `novel_object` 的 head gap 与集合难度同量级（−0.23 / +0.09），**没有可分辨的留出退化**。
2. `paraphrase` 的 L1 gap **+0.62（≈7.8σ）**，超过集合难度带；但本仓只有 2 条指令变体/物体，
   该 split 同时把 **任务类型（pick vs place）** 留出，**语言改写与任务类型混淆**，不能归因为"措辞泛化差"。
3. F1 的重复间方差（para held-out ±0.114）大于协议间差异，**F1 不作为 gap 判据**。

### 9.3 结论与边界

- **A7 在 2000 步预算下不可结论**：head 的绝对误差（L1 2.78–3.42）比 40k 参考行（0.67–0.90）
  差 3–4 倍，说明这些单元**整体欠拟合**；在欠拟合状态下测"泛化 gap"会被"哪里都没学会"掩盖。
- **可复现的次生结论**：`combo` / `novel_object` 未观察到留出退化；`paraphrase` 的 L1 退化存在，
  但被任务类型混淆，需**真正的同义改写数据**才能重测。
- **协议修正**：首轮用 stride 40 + 2 集时，夹住 F1 出现 0.000 的"假退化"——实测这 2 集的
  GT 夹住帧只有 **1 帧与 0 帧**（`ep41` 全 447 帧仅 6 帧在夹住带内），F1 无定义即可为 0。
  已新增 `select_frame_ids(..., even=True)` 与 `--even`，在整段 episode 上均匀取帧。
- **仍不可替代 A6**：离线指标只反映示范拟合质量；真机成功率必须实测。

### 9.4 combo 协议的饱和曲线（2000 → 8000 步）

同一训练集（81 集）续训到 8000 步、每 2k 存一次 checkpoint，各点单次评测（`--even` / 160 帧）：

| 步数 | epoch | held-out L1 | seen L1 | gap | held-out F1 | seen F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 1.4 | 2.860 | 3.647 | −0.787 | 0.576 | 0.526 |
| 4,000 | 2.8 | 2.786 | 3.137 | −0.351 | 0.545 | 0.514 |
| 6,000 | 4.3 | 2.728 | 3.019 | −0.291 | 0.542 | 0.706 |
| 8,000 | 5.7 | **2.566** | **2.817** | **−0.251** | 0.557 | 0.564 |

- **前沿未追平**：8k 比 2k 只降 10%–23%，仍是 40k 参考行（0.67–0.90）的 **3–4 倍**；
  收益明显递减 → head（0.36% 参数）在这批数据上接近**容量限制**，不是单纯"再多训一会儿"。
- **gap 不随训练变大**：从 −0.79 收敛到 −0.25，方向始终是 held-out 更易（未出现留出退化）。
- **注意**：本表的 2k 点是 8k 调度下的中间 checkpoint（余弦 LR 调度按总步数归一化），
  **不等于** §9.2 里独立 2k 训练的模型，不能混用。

**对 A7 的直接含义**：用 head 做 A7，测到的差异主要由"head 容量/预算"决定；
要得到可解释的泛化结论，应改用 full 微调（或更大数据量）后再测。

