# 第 38 周 实施计划：深度化主线（VLM 数据引擎 + VLA 适配 + 世界模型评估）

> 状态：**待开发**（计划先行）。
> 设计依据：`docs/week38/design.md`（v2，含前置阻塞点与解除状态）。
> 执行期间每完成一个子任务即回填本文件 §7，并在 `design.md` §13 回填实际偏差。

## 0. 类型标注说明

任务表中每条标注执行类型，用于区分**可离线完成的代码开发**与**需要外部资源的采集/运行**：

| 标注 | 含义 | 外部前置 |
|---|---|---|
| **【代码】** | 纯代码开发：模块、脚本、测试、文档；可完全离线完成 | 无 |
| **【数据】** | 需采集数据：遥操作数据采集、人工标注与人工复核 | 真机 + 场地 + 人工工时 |
| **【真机】** | 需真机硬件操作：负载探针、标定采图、真机验证与录制 | SO-101 通电就绪 |
| **【云端】** | 需云 GPU 训练/评测：SmolVLA 微调与对照评测 | 阿里云 A10 + 联网 |

**判断原则**：带 **【数据】** 的是**外部依赖最重、最可能拖慢整体进度**的项；带 **【真机】** 的需排队占机；带 **【云端】** 的有算力与排队成本。这三类应尽早启动、与【代码】任务并行推进，避免成为关键路径。

## 1. 实施顺序与依赖

**总顺序：阶段 0（前置决策与探针）→ P0（感知消融）→ P1（VLM 数据引擎）→ P3（WM 离线评估）→ P2（VLA 适配，裁剪版）→ 汇总交付。**

| 阶段 | 支柱 | GPU | 依赖 | 说明 |
|---|---|---|---|---|
| 0 | 前置 | 否 | — | 负载探针、2D 定位源选型、意图 taxonomy；决定后续可做范围 |
| 1 | P0 感知 | 否 | 阶段 0 | 零硬件成本，直接抬高成功率上限；走 R2 免 FK 路线 |
| 2 | P1 数据 | 否 | 阶段 0（taxonomy） | 纯逻辑可 Mock；多任务采集人工并行 |
| 3 | P3 评估 | 否 | P0/P1 无依赖，可提前 | 代码全在手，无论结论正负均有产出 |
| 4 | P2 适配 | 是（A10） | P1 数据 + 4.1 门槛 | 依赖最多，首轮裁剪范围 |
| 5 | 交付 | — | 全部 | 保底交付 P0 + P3 两份报告 |

> 并行度：阶段 1 与阶段 2 无相互依赖，可同时推进；阶段 3 可在阶段 2 完成前启动。

### 关键路径提示（按类型）

```text
【数据】2.9 多任务采集 ─────────────► 阻塞 4.5 适配对照 ──► 5.2 数据卡
【数据】2.10 人工复核抽检 ─────────► 阻塞 A3(κ) / A4
【真机】0.2 负载探针 ──► 决定 B2；【真机】0.3 2D 定位实测 ──► 决定 P0 范围
【云端】1.6 / 4.5 / 4.7 ───────────► 有算力排队成本，宜早预约
【代码】其余全部 ───────────────────► 可离线推进，不阻塞
```

## 2. 任务分解

### 阶段 0：前置决策与探针

| # | 文件 | 子任务 | 类型 |
|---|---|---|---|
| 0.1 | `src/real_observation.py`（改造） | 新增 `read_load()`（经注入引擎读 STS3215 `Present_Load`，存器 60）与纯逻辑 `detect_contact(load, baseline, margin)` | 【代码】 |
| 0.2 | `src/run_load_probe.py`（新增） | 空载/夹持负载量程探针 CLI → `LoadProbeResult`，判定 **B2 是否成立** | 【代码】+【真机】 |
| 0.3 | `src/real_perception.py`（改造） | **2D 定位源选型**：实测 VLM bbox 精度 vs 引入 GroundingDINO 类检测器；输出像素级 bbox/中心点 | 【代码】+【真机】 |
| 0.4 | `src/vlm_labeler.py`（新增，先落 taxonomy 部分） | `IntentTaxonomy` + `define_intent_taxonomy()` + `map_instruction_to_intent()`（使自由文本可计算 κ） | 【代码】 |
| 0.5 | 决策记录（`docs/week38/` 内回填） | SO-101 FK 建模可行性评估；**默认 B3 三角化为条件任务**，仅在决定保留时启动 1.2 | 【代码】 |
| 0.6 | 基准标定（`run_load_probe.py` / 新脚本 + `real_verifier` 改造评估） | **空闭基准标定**：用真机管线自身的闭合行为在空载下测 `empty_grip_baseline` 的**跨运行/跨姿态波动**，并与 `margin` **联合标定**（建议改为"位置 OR 负载"判据）。**仅测量与给建议，不擅自改动真机判定行为** | 【代码】+【真机】 |

> **0.6 门禁**：`empty_grip_baseline`（代码现值 1.5）与实测空载闭合位置（3.8，2026-10-10）不符，且位置差（3.0）恰好等于 `margin`（3.0）处于临界。该基准决定真机管线全部"夹住/没夹住"判定（成功码、恢复决策、评测成功率、agent 自判），**必须在任何要报真机成功率的评测（第 30 周 20 轮、P2 真机抽检）之前完成**。对当前 P0/B2 无影响，不阻塞主线。

### 阶段 1：P0 感知消融（R2 路线：几何/负载作为策略输入）

| # | 文件 | 子任务 | 类型 |
|---|---|---|---|
| 1.1 | `src/real_geometry.py`（新增） | `DepthEstimate` 数据结构 + `depth_from_known_size(bbox_px, real_size_m, fx)`（主路径，免 FK） | 【代码】 |
| 1.2 | `src/real_geometry.py` | `triangulate(...)` + `solve_handeye(...)`（条件路径，仅 B3 使用） | 【代码】 |
| 1.3 | `src/real_geometry.py` | 几何量 → 策略输入特征构造函数（供 R2 注入） | 【代码】 |
| 1.4 | `src/run_handeye_calib.py`（新增） | 标定数据采集 + 求解 CLI（条件，输出残差） | 【代码】+【真机】 |
| 1.5 | `src/visual_act.py`（改造） | 增加额外观测特征通道，支持 B0–B3 输入变体 | 【代码】 |
| 1.6 | `src/run_depth_ablation.py`（新增） | B0–B3 累积式消融训练与评测 CLI → 成功率 / 闭合高度误差 / `ΔGrasp` 归因表 | 【代码】+【云端】 |

### 阶段 2：P1 VLM 数据引擎

| # | 文件 | 子任务 | 类型 |
|---|---|---|---|
| 2.1 | `src/vlm_labeler.py`（新增） | `LabelRecord` 数据结构（含 `intent` 字段） | 【代码】 |
| 2.2 | `src/vlm_labeler.py` | `label_episode(backend, frames, sensor_state, taxonomy, task_hint)`：VLM 出文本/切片/置信，传感器出成功判定，映射 intent | 【代码】 |
| 2.3 | `src/vlm_labeler.py` | `label_dataset(...)`：批量 + 断点续跑 + jsonl 落盘 | 【代码】 |
| 2.4 | `src/vlm_labeler.py` | 标注产物 → LeRobotDataset 兼容元数据（task/intent/subtask），不改 fields | 【代码】 |
| 2.5 | `src/label_audit.py`（新增） | `cohen_kappa(a, b)`、`compare_labels(...)` → `LabelAuditReport`（含 `label_scope` 适用边界） | 【代码】 |
| 2.6 | `src/label_audit.py` | 混淆矩阵、per-subtask recall、分歧样本导出 | 【代码】 |
| 2.7 | `src/run_vlm_label.py`（新增） | 标注 CLI | 【代码】 |
| 2.8 | `src/run_label_audit.py`（新增） | 审计 CLI（κ + 分歧样本，供人工抽检 20%） | 【代码】 |
| 2.9 | 数据采集（人工执行，复用 `run_vla_collect.py`） | 多任务集：≥3 物体 × 2 指令 × 30 集，沿用 week28 采集与质检流程 | **【数据】**+【真机】 |
| 2.10 | 人工复核（人工执行） | 抽检 20% 标注结果，作为 `label_audit` 的人工参照（决定 A3/A4 是否成立） | **【数据】** |

### 阶段 3：P3 世界模型离线评估

| # | 文件 | 子任务 | 类型 |
|---|---|---|---|
| 3.1 | `src/wm_ope.py`（新增） | `OPEReport` 数据结构（含 `candidate_family` / `real_success_var` / `scope_note`） | 【代码】 |
| 3.2 | `src/wm_ope.py` | `rollout_policy_in_world_model(policy, wm, init_states, horizon)` | 【代码】 |
| 3.3 | `src/wm_ope.py` | `evaluate_ope(...)` → Spearman ρ + bootstrap 置信区间 + 候选方差 | 【代码】 |
| 3.4 | `src/wm_ope.py` | ρ 偏低的归因路径（WM 训练分布与候选访问分布的 shift 统计） | 【代码】 |
| 3.5 | `src/run_wm_ope.py`（新增） | OPE CLI：候选 ≥5 且**跨行为族**，口径限定**仿真 PushCube 策略族** | 【代码】 |
| 3.6 | `src/world_model.py`（复用） | 确认 NumPy `DynamicsModel` 可被 OPE 直接调用 | 【代码】 |

### 阶段 4：P2 VLA 适配与泛化（首轮裁剪版）

| # | 文件 | 子任务 | 类型 |
|---|---|---|---|
| 4.1 | `src/run_vla_infer.py`（改造，**门槛任务**） | 真机推理接入适配后 checkpoint（替换 stub）；若链路未通，则**首轮全部在仿真评测** | 【代码】+【真机】 |
| 4.2 | `src/vla_adapt.py`（新增） | `AdaptConfig` / `AdaptResult` 数据结构 | 【代码】 |
| 4.3 | `src/vla_adapt.py` | `build_adapt_config(...)`、`apply_adaptation(...)`（`zero_shot` / `head` / `full`；**LoRA 延后**） | 【代码】 |
| 4.4 | `src/vla_adapt.py` | `run_adapt_experiment(...)`：训练 + 评测 + 保留率（遗忘）计算 | 【代码】 |
| 4.5 | `src/run_vla_adapt.py`（新增） | 对照 CLI：3 策略 × 2 数据量 = **封顶 6 个训练单元** | 【代码】+【云端】 |
| 4.6 | `src/vla_generalization.py`（新增） | `build_split(...)` + `evaluate_generalization(...)`（含 95% CI） | 【代码】 |
| 4.7 | `src/run_vla_generalization.py`（新增） | 泛化评测 CLI（`combo` / `paraphrase` / `novel_object`） | 【代码】+【云端】 |

### 阶段 5：汇总与交付

| # | 产出 | 说明 | 类型 |
|---|---|---|---|
| 5.1 | `runs/week38/*/report.md` + `report.json` | P0 / P3 为保底交付；P1 / P2 随数据与 GPU 进度滚动 | 【代码】 |
| 5.2 | 数据卡 + 模型卡 | 多任务语言标注数据集 / 适配策略对照 | **【数据】** |
| 5.3 | 演示视频 | 3 分钟：问题 → 方案 → 结果（需真机录制） | **【数据】**+【真机】 |
| 5.4 | 根目录开发计划同步 | 按 AGENTS.md 更新第 38 周状态与代码结构 | 【代码】 |
| 5.5 | `design.md` §13 偏差回填 | 记录实际实现与设计的偏离 | 【代码】 |

## 3. 文件结构

**新增（src/）** — 全部为【代码】
```text
src/real_geometry.py            # P0 已知尺寸反解（主）+ 三角化/标定（条件）+ B0–B3 变体特征布局
src/ablation_features.py        # P0 B0–B3 特征生产（方案③：B1/B2 特权上界 + B3 检测对照）
src/run_load_probe.py           # P0 负载量程探针 CLI            【运行时需真机】
src/run_handeye_calib.py        # P0 标定 CLI（条件）            【运行时需真机】
src/run_depth_ablation.py       # P0 B0–B3 消融 CLI              【运行需云端 GPU】
src/vlm_labeler.py              # P1 意图 taxonomy + VLM 自动标注 + 传感器成功判定
src/label_audit.py              # P1 标注可靠性审计
src/lerobot_label_source.py     # P1 本地 LeRobot 数据集 → 标注输入源（PyAV 抽帧 + 状态）
src/review_sheet.py             # P1 人工抽检页生成（可复现抽样 + 导出 LabelRecord jsonl）
src/load_sidecar.py             # P1 采集期**旁挂负载通道**：格式/写盘/逐帧对齐校验
src/load_logging.py             # P1 录制流水线挂钩：包装机器人 + 替换两处函数
src/run_vla_collect_record.py   # P1 采集 CLI：`lerobot-record` 透明包装 + `--verify`
src/run_vlm_label.py            # P1 标注 CLI
src/run_label_audit.py          # P1 审计 CLI
src/wm_ope.py                   # P3 世界模型离线策略评估
src/run_wm_ope.py               # P3 OPE CLI（仿真口径）
src/vla_adapt.py                # P2 适配策略与实验
src/vla_generalization.py       # P2 组合泛化协议
src/run_vla_adapt.py            # P2 适配对照 CLI                【运行需云端 GPU】
src/run_vla_generalization.py   # P2 泛化评测 CLI                【运行需云端 GPU】
```

**新增（tests/）** — 全部为【代码】
```text
tests/test_load_signal.py          # 负载读取与接触判定
tests/test_real_geometry.py        # 已知尺寸反解 / 三角化 / 标定
tests/test_perception_locate.py    # 2D 定位解析、bbox 校验与越界处理
tests/test_vlm_labeler.py          # taxonomy、标注、传感器判定优先、断点续跑
tests/test_label_audit.py          # κ、混淆矩阵、分歧导出、label_scope 自动说明
tests/test_lerobot_label_source.py # 抽帧下标、meta 解析、帧清单合并
tests/test_review_sheet.py         # 抽检抽样可复现、HTML 转义/选项、导出 schema
tests/test_load_sidecar.py         # 方向位解码、逐关节容错、成集落盘/重录丢弃、对齐校验
tests/test_load_logging.py         # features 透传、每帧一行、复位段不计、patch 还原
tests/test_wm_ope.py               # ρ、CI、归因路径
tests/test_vla_adapt.py            # 配置、参数解冻统计、保留率
tests/test_vla_generalization.py   # split 无泄漏、CI、协议切换
tests/test_ablation_features.py    # P0 变体列序/累积前缀/缺键报错 + ENV 额外通道契约
tests/test_ablation_feature_production.py  # 方案③ 特征生产：上界/检测两条来源的边界与三角化回代
```

**修改（src/）** — 全部为【代码】
```text
src/real_observation.py    # 新增 read_load / detect_contact
src/real_perception.py     # 新增像素级 2D 定位输出；公开 extract_json_object 供标注链路复用
src/visual_act.py          # 新增策略额外输入特征通道（R2）
src/run_vla_infer.py       # 真机推理接入适配后 checkpoint（门槛）
```

**数据与运行产物（非代码，需采集 / 真机 / 云端生成）**
```text
data/multi_task/              # 多任务遥操作数据集（≥3 物体 × 2 指令 × 30 集）   【数据】+【真机】
runs/week38/collect/load_sidecar.jsonl  # 采集期旁挂负载（按 episode/frame 对齐）  【真机】
runs/week38/labels/<ds>/      # 标注产物：labels.jsonl + frames/ + episode_metadata.jsonl
runs/week38/labels/<ds>/review/review.html  # 人工抽检页（人工填完后导出 human.jsonl）  【数据】
runs/week38/labels/<ds>/audit/              # audit.json + disagreements.jsonl
runs/week38/load_probe.json   # 负载量程探针结果                               【真机】
runs/week38/geometry/         # 标定采图与几何实测数据                          【真机】
runs/week38/vla_adapt/        # 适配实验产物（6 个训练单元）                     【云端】
runs/week38/wm_ope/           # OPE 报告（仿真，可本机运行）                     【代码】
```

## 4. 测试计划

### 4.1 单元测试（纯逻辑，不依赖 Unity / 真机 / 网络）— 全部【代码】

| 文件 | 覆盖范围 |
|---|---|
| `test_load_signal.py` | `detect_contact` 阈值边界、空载/夹持/越界、注入引擎契约、异常降级 |
| `test_real_geometry.py` | 已知尺寸反解的数值正确性与边界（除零、退化 bbox）；三角化合成数据回代；标定合成数据复原已知 R/t |
| `test_perception_locate.py` | VLM JSON 解析（正常/缺字段/非数组/非数字）；bbox 校验（长度、顺序、非有限）；`frame_size` 越界丢弃并计数；`center_px`/`size_px` 换算；backend 异常降级为空结果 |
| `test_vlm_labeler.py` | taxonomy 映射；FakeBackend 注入；文本/切片解析；传感器判定优先于 VLM；jsonl 读写与断点续跑；VLM 失败降级；`VlmLabelingBackend` 的围栏 JSON 解析、非法 subtask 置空、置信度钳制、缺 `task_text` 必须报错（不许静默回退到数据集指令） |
| `test_label_audit.py` | κ 手算校验（完全一致=1、随机≈0、反向=-1）；混淆矩阵；分歧样本导出；`label_scope` 透传；`default_label_scope` 的退化维标注 |
| `test_lerobot_label_source.py` | `sample_indices` 首尾包含/单帧取末/不重复；`read_dataset_meta` 的任务拼接与 `gripper.pos` 下标定位；帧清单**合并式**写入 |
| `test_review_sheet.py` | 抽检抽样可复现（同种子同结果、保序）；HTML 转义（防任务文本注入）；intent/subtask/success 选项齐全；导出走 `source=human` + taxonomy 版本 |
| `test_load_sidecar.py` | `Present_Load` 方向位解码；单关节异常**逐关节容错不中断**；写入器成集落盘 / 重录丢弃旧缓冲 / 异常不落盘 / 跨运行追加；`episode_loads` 按帧排序；对齐校验的帧数、位置与容差判定 |
| `test_load_logging.py` | **features 与观测键原样不变**（数据集 fields 契约）；属性委托；每录一帧一行；**集间复位段不计入**；重录替换；返回裁剪后的动作；读负载失败不中断采集；无总线容错；`patch_record_pipeline` 的集号跟踪 / kwargs 转发 / 原函数还原 / 收尾提交 |
| `test_wm_ope.py` | 完美排序 ρ=1；完全反向 ρ=-1；并列值；候选数不足时拒绝；候选同族时告警；归因路径触发 |
| `test_vla_adapt.py` | 配置构造；参数解冻统计（`trainable_ratio` 校验）；结果聚合；保留率除零保护 |
| `test_vla_generalization.py` | split 无泄漏（训练/held-out 组合不相交）；CI 计算；三种协议切换 |
| `test_ablation_features.py` | 变体列序与**累积前缀**（B_k ⊇ B_{k-1}）；`feature_values` 的 method 校验与"不可用量缺键"；`build_feature_vector` 定序与缺键/非有限报错；visual_act 的 ENV 维度**默认 13（B0 与历史逐位一致）**与扩展；`_append_extra_features` 的顺序/维度/非有限校验；`VisualACTPolicy` 带通道无 `extra_provider` 即报错 |

### 4.2 集成与硬件测试

| 项 | 类型 | 前置 |
|---|---|---|
| 负载探针：COM6 空载/夹持各 ≥5 次采样 → `LoadProbeResult.resolvable` 判定 | 【真机】 | SO-101 通电 |
| 几何：工作区 30–60 cm 内已知尺寸目标 ≥15 组 → 反解误差统计（条件项：标定残差） | 【真机】 | SO-101 通电 + 标定板 |
| P0 消融：`run_depth_ablation.py` 四组同 seed → `ΔGrasp` 与闭合高度误差 | 【云端】 | 训练算力 |
| P3 OPE：候选 ≥5 且跨行为族，每候选 ≥10 rollout，含 scripted 锚点 | 【代码】 | 仿真环境 |
| P2 适配：先在仿真（week15-cam 数据）跑通矩阵，再上真机抽检 | 【云端】+【真机】 | 云 GPU 后接真机 |
| 多任务采集：≥3 物体 × 2 指令 × 30 集 + 质检 | 【数据】+【真机】 | 场地 + 工时 |
| 人工复核：抽检 20% 标注结果 | 【数据】 | 人工工时 |

### 4.3 回归与质量门 — 全部【代码】

```bash
python -m compileall -q src
python -m pytest -q
```

- 既有真机相关 60 项测试不得回归
- 新增测试全部通过后方可计入验收

## 5. 验收标准

| 编号 | 验收项 | 阈值 | 对应设计条目 |
|---|---|---|---|
| A1 | P0 几何链路定位误差（工作区 30–60 cm） | ≤ 1.0 cm（标定残差 5–8 mm 亦可进入消融，去留由消融结果决定） | design §9 A1 |
| A2 | P0 抓取成功率相对 B0 提升 | ≥ 15 个百分点 | design §9 A2 |
| A3 | P1 离散意图标注一致率 κ | ≥ 0.6 | design §9 A3 |
| A4 | VLM 标注训练相对人工标注下降 | ≤ 10 个百分点 | design §9 A4 |
| A5 | P2 适配对照表（首轮） | ≥ 3 策略 × 2 数据量（LoRA 为后续） | design §9 A5 |
| A6 | 最优策略相对 zero_shot 提升 | ≥ 20 个百分点（以 zero_shot 实测基线校准） | design §9 A6 |
| A7 | held-out 组合成功率 | ≥ 40%（或降级为仿真协议验证） | design §9 A7 |
| A8 | P3 Spearman ρ | ≥ 0.6；若 < 0.6 则提交归因报告（负结果亦算通过） | design §9 A8 |
| A9 | 前置产物 | 意图 taxonomy 定义完成；负载量程探针完成 | design §9 A9 |
| A10 | 全部实验可复现 | 固定 seed / 脚本 / 配置 / 报告齐备 | design §9 A10 |
| A11 | 单元测试 + 回归 | 新增测试全过，既有 60 项不回归 | §4.3 |

## 6. 检查点与降级

| 检查点 | 触发条件 | 降级动作 |
|---|---|---|
| CP0（阶段 0 后） | 负载不可分辨 | B2 降级为"位置差 + 负载"联合判据 —— **实测未触发**（Δ=151‰ > 50‰，2026-10-10） |
| CP0（阶段 0 后） | 2D 定位精度不足 | 退回区域级定位，P0 只做 B2 |
| CP1（P0 后） | A1/A2 未达标 | 记录归因；B3 三角化整组取消，不影响 B0–B2 |
| CP2（P1 后） | κ < 0.5 | 退回全人工标注，P1 降级为"标注成本对比"研究 |
| CP3（P3 后） | ρ < 0.3 | 转负结果 + 归因报告（仍计入交付） |
| CP4（P2 门槛） | 4.1 真机闭环未通 | 首轮全部转仿真评测，真机仅作最终抽检 |
| CP5（P2 仿真后） | 对照矩阵无区分度 | 检查数据量与协议设计，必要时缩小任务集 |

- 任一阶段阻塞不影响其余支柱；P0 / P1 / P3 无 GPU 依赖，优先保底。

## 7. 执行记录

### 2026-10-10 · 批次 1（阶段 0 部分 + 阶段 1 部分，全部【代码】）

| 子任务 | 文件 | 状态 | 说明 |
|---|---|---|---|
| 0.1 | `src/real_observation.py` | ✅ | 新增 `ContactSignal` / `LoadProbeResult` / `read_load` / `detect_contact` / `summarize_load_probe` / `load_state_names`；读不到负载一律返回空 dict（不把"无信号"当"无接触"） |
| 0.1b | `src/real_executor.py` | ✅ | `RobotEngine` 协议 + `RealEngine`（`bus.sync_read("Present_Load", normalize=False)`）+ `MockEngine`（合成负载，`gripper_load` 可改写）新增 `read_load` |
| 0.2 | `src/run_load_probe.py` | ✅ | 双阶段采集 CLI（empty/loaded）→ 合并出 `LoadProbeResult`；**探针自行下发夹爪闭合指令**（闭合 → 稳定 → 采样 → 退回安全位，参数 `--close-pos/--release-pos/--settle`，`--no-command` 可退回只读）；同时记录夹爪位置作为 CP0 联合判据证据；标定缺失时明确报错而非阻塞在交互式标定；`--dry-run` 全链路通过；可分辨退出码 0、不可分辨退出码 1 |
| 1.1 | `src/real_geometry.py` | ✅ | `DepthEstimate` + `depth_from_known_size`（取包围盒较大边 + 一阶误差传播） |
| 1.2 | `src/real_geometry.py` | ✅ | `triangulate`（中点法，射线间隙作不确定度）、`solve_handeye`（AX=XB 四元数零空间法，规避旋转符号歧义） |
| 测试 | `tests/test_load_signal.py`、`tests/test_real_geometry.py` | ✅ | 48 项：接触阈值与降级链、探针判据、针孔反解、三角化还原与退化、AX=XB 还原已知 X（含纯平移退化） |

**验证结果**

- `python -m compileall -q src` 通过
- 新增 48 项全过
- 全量回归（跳过 3 个依赖 `ai2thor` 的模块）：**483 passed / 7 skipped / 0 failed**
- `run_load_probe.py --dry-run`（empty → loaded）链路通过，结论 `resolvable=True`

**本轮未动用**：真机、模型 API（全部为纯逻辑代码）。

### 2026-10-10 · 批次 1 修正（真机实测反馈，`--phase empty` 首次运行）

真机跑出：`gripper.load = 1079.0`（原始寄存器值）、`gripper.pos = 3.7657`（指令 3.0）。据此修两件事：

| 问题 | 处置 | 文件 |
|---|---|---|
| `Present_Load` 原始读数含方向位（bit10）：`1079 = 0x437` → 方向位=1、幅值 55 → 实际 −55‰。原始值直接做差会在换向时虚增 1024 | 新增 `decode_feetech_load`；`RealEngine.read_load` 与探针均改为解码后输出；`LOAD_MARGIN` 3.0 → **50.0**（千分比） | `real_observation.py`、`real_executor.py`、`run_load_probe.py` |
| 指令 3.0 但停在 3.77，5 次采样完全一致 → 被机械限位挡住（**3.77 即闭合极限**，不是没执行到） | 探针结论段新增位置判定说明（空载是否被限位挡住 / 夹持是否真挡住钳口）；`close-pos` 明确不可再调小 | `run_load_probe.py` |
| `MockEngine` 合成负载量级与新单位不一致 | 改为解码后量级（gripper 空载 55 / 夹持 180） | `real_executor.py` |

**验证**：新增 9 项解码测试；本批 57 项全过；全量回归 **492 passed / 7 skipped / 0 failed**。
**影响**：已产出的 `runs/week38/load_probe.json` 中 `empty` 样本为**修正前的原始值**，需重跑 `--phase empty` 后再跑 `--phase loaded`，否则两侧单位不一致。

### 2026-10-10 · 0.2 真机负载量程实测完成（CP0 通过）

真机结果（`runs/week38/load_probe.json`）：

```text
空载 -55.00‰(n=5) / 夹持 -206.00‰(n=5) / Δ=-151.00 / 噪声=0.00 / 阈值=50.00
夹爪位置: 空载 3.8 / 夹持 6.8 / Δ=+3.0
可分辨(resolvable) = True  → B2（负载/接触特征）可执行
```

结论：

1. **CP0 通过**：负载信号可分辨，|Δ|=151‰ 是阈值 50‰ 的 **3 倍**，余量充足；5 次采样完全一致（同分辨率下噪声为 0）。
2. **位置差仅 3.0，恰好等于 `is_holding` 的默认 `margin=3.0`**，位置判据处于临界。
3. **实测空载闭合位置为 3.8，而代码中 `empty_grip_baseline` 默认 `1.5`**（`real_executor.py`、`run_real_agent.py`、`run_real_eval.py`）——基准已与实测不符。若按实测 3.8 校准，则判据 `pos > 3.8 + 3.0 = 6.8` 对夹持值 6.8 为**假**，即**位置判据会漏判**。这正面佐证了引入负载信号（B2）的必要性。
4. 待确认：本次所用物体（Δpos 仅 3.0，疑似薄件）。若确为螺丝刀，则"薄物体位置差处于临界、负载仍可分辨"是本轮最有说服力的证据。
5. 待评估（不在本批改动内）：`empty_grip_baseline` 是否需要由 1.5 重标为实测值，以及 `margin` 是否需同步调整——会改变真机夹持判定行为，需单独决策。

### 2026-10-10 · 批次 2（0.3 + 0.4，纯【代码】）

| 子任务 | 文件 | 状态 | 说明 |
|---|---|---|---|
| 0.3 | `src/real_perception.py`（改造） | ✅ 模块部分 | 新增像素级 2D 定位：`Detection`（bbox 校验 + `center_px` / `size_px`）、`LocalizationResult`（含 `skipped`）、`ObjectLocator` 协议、`VLMObjectLocator`、`FakeObjectLocator`、`locate_objects()`（严格 JSON 解析；非法/越界候选丢弃并计数；后端异常降级为 `source="none"`）。既有 `describe_workspace` / `local_holding_state` **保持不变** |
| 0.4 | `src/vlm_labeler.py`（新增） | ✅ | `IntentTaxonomy`（构造期校验 + `to_dict` / `from_dict`）、`define_intent_taxonomy()`、`normalize_instruction()`（全角转半角 / 空白归一 / 小写）、`map_instruction_to_intent()`（按 taxonomy 顺序做关键词子串匹配；未命中返回 `None`，作为独立类别参与 κ） |
| 测试 | `tests/test_perception_locate.py`（新增）、`tests/test_vlm_labeler.py`（新增） | ✅ | 57 项：定位 JSON 解析与越界丢弃、bbox/置信度校验、`FakeObjectLocator` 降级；taxonomy 校验与序列化、文本归一化、四类意图映射、优先级随 taxonomy 顺序变化、未映射与空关键词边界 |

**验证**：本批 57 项全过；全量回归 **549 passed / 7 skipped / 0 failed**。

**说明与遗留**

- **0.3 的实测选型未做**（VLM bbox 精度 vs GroundingDINO 类检测器）：需要模型 API + 真机图像。接口已按 `ObjectLocator` 协议留好插入点，届时实现同一 Protocol 即可同口径对比。
- **0.4 的映射刻意用确定性关键词规则而非 LLM**：κ 衡量的是"VLM 标注 vs 人工标注"在同一 taxonomy 上的分歧，若映射本身是 LLM 则分歧无法归因。已知局限（不处理否定、多动作只取最高优先级）写在 docstring 里，待真实指令语料验证后迭代。

### 2026-10-10 · 批次 3（0.3 补充：模型冒烟 + 两个真实缺陷修复）

用现有 `.env`（DeepSeek）与 Qwen 兼容端点（`https://maas.qianwenaiapi.com/compatible-mode/v1`，模型 `qwen3-vl-plus` / `gui-plus`）做像素 bbox 冒烟，暴露并修复两个缺陷：

| 缺陷 | 现象 | 修复 | 文件 |
|---|---|---|---|
| **坐标约定被硬编码为像素** | `qwen3-vl-plus` 返回 `[304,304,438,526]`，x2/y2 超出 480 → 被判"越界丢弃"，**误读成"模型不准"**。实为 Qwen-VL 官方 `[0,1000)` 归一化（Qwen-VL 论文 §2.2；官方 blog 示例 `bbox_2d:[0,54,473,999]`）。换算后中心误差仅 **2.7 px** | 新增 `coord_space`（`auto`/`pixel`/`normalized_1000`/`normalized_1`）+ 按轴判定 + 换算；`LocalizationResult.coord_space` 记录实际约定 | `real_perception.py` |
| **严格 `json.loads` 不接受围栏** | 视觉模型普遍用 ` ```json ` 包裹；`qwen3-vl-plus` 实测即返回围栏内容 → 有效结果被判解析失败 | 新增 `_extract_json_object`：直接解析失败 → 截取首个配对完整的 `{...}`（扫描时跳过字符串内花括号） | `real_perception.py` |

**冒烟结果**（640×480 合成图，真值 bbox=(200,150,280,250)，中心误差）：

| 定位源 | 坐标约定 | 中心误差 | 尺寸误差 |
|---|---|---|---|
| DeepSeek v4-flash-vision | pixel | 87.7 px | −24×−28 |
| `qwen3-vl-plus` | normalized_1000（`auto` 正确判定，与显式指定一致） | **2.7 px** | +6×+7 |
| `gui-plus` | pixel | **2.2 px** | +2×0 |

**重要限定**：均为**单样本、合成高对比图**，无光照/遮挡/背景干扰；`gui-plus` 为 GUI 定位专用模型、测试图形似 UI 元素，成绩可能不可外推。**真实相机帧（细长低对比螺丝刀 + 真实光照）精度未知，不构成选型结论。**

**验证**：新增 11 项测试（围栏 4 + 坐标约定 7），本批 34 项全过；全量回归 **560 passed / 7 skipped / 0 failed**。

### 2026-10-10 · 批次 4（0.3 真机抓帧 + 定位源首轮评测）

新增两个工具并跑通：

| 文件 | 作用 |
|---|---|
| `src/run_perception_capture.py` | 真机抓帧：`cv2.VideoCapture(idx, CAP_MSMF)`，**不连机械臂、不占 COM6**；落 PNG + `meta.json`（含索引映射、分辨率、场景描述） |
| `src/run_perception_locate_eval.py` | 多源评测：同一批帧跑多个定位源，出**叠加图**（便于人工核对）+ `report.json`；支持 `--gt` 算中心/尺寸误差、`--dry-run` 离线验链路 |

真机帧：`runs/week38/perception_frames/p0/`（fixed×3 + handeye×3，640×480）。帧体检 mean 139–169、std 51–71、全动态范围 → 不是黑屏。

**真实帧触发并修好的两个工程问题**

| 问题 | 现象 | 处置 |
|---|---|---|
| `auto` 坐标判定不可靠 | `gui-plus` 在不同帧上被分别判成 `normalized_1000` 与 `pixel`。它实际用像素，那几帧是**越界像素框触发误判**，垃圾框被"救"成看似合理的归一化框 → 静默错误 | 评测脚本对每个源**显式钉死约定**（`gui-plus=pixel`、`qwen3-vl-plus=normalized_1000`），不使用 `auto`；`auto` 仅作发现用途 |
| 越界框处理过严 | `gui-plus` 给出 x2=644（图宽 640）的**边缘 4 px 过冲**，原逻辑整条丢弃 → 低估模型 | 改为 `_clamp_to_frame`：轻微越界**裁剪保留**并计入 `LocalizationResult.clamped`；越界 >20% 才判非法 |

**首轮评测（6 帧 × 2 源）**

- **螺丝刀框跨模型高度一致**：`gui-plus` (313,256,439,333) vs `qwen3-vl-plus` (312,253,436,335) → 中心差 ~2 px。两个独立模型族收敛到同一框，是无真值时最可用的交叉验证。
- **固定相机疑似看不到目标**：两模型在 fixed 帧都只找到"镜头/金属钩（右上）"与"工作台/机械臂（下方）"，**均未检出螺丝刀**。→ 若属实，P0 的"固定相机 + 已知尺寸反解"前提需重新评估（**待人工核对**）。
- 行为差异：`gui-plus` 报 1–5 个显著物体；`qwen3-vl-plus` 报 11–13 个（连背景人手/鞋子都列），对"找目标"是噪声但目标确实找到。

**验证**：新增 3 项（裁剪/越界丢弃/退化丢弃），perception 测试 36 项全过。

### 2026-10-10 · 批次 5（0.3 真机帧复核 + 多源配置通用化）

用户反馈"换位后两路相机能否都看到螺丝刀"，并指出定位源的坐标逻辑不该写死。本轮改动：

| 项 | 内容 | 文件 |
|---|---|---|
| 主模型名 | `.env` 的 `LLM_MODEL` 由 `deepseek-v4-flash-vision-exp` 改为 **`deepseek-flash`**（原名称该端点直接 400；`deepseek-v4-pro` 不支持图片输入） | `.env` |
| 多源配置 | 新增 `VLM_SOURCES` 与按源 `VLM_<源名>_API_BASE/_API_KEY/_MODEL/_COORD_SPACE`；逐级回退 全局 `LLM_API_*` → 源名 → 内置默认表 → `auto`。**新增归一化输出模型只需改 `.env`** | `.env`、`.env.example`、`run_perception_locate_eval.py` |
| `auto` 容差 | `_resolve_coord_space` 加 `_AUTO_OVERFLOW_TOLERANCE=2%`：4 px 边缘过冲（+0.6%）不再被误判为 `normalized_1000`，而 `qwen3-vl-plus` 的 +9.6% 越界仍能识别 | `real_perception.py` |

**第二轮真机帧**（`runs/week38/perception_frames/p0_v2`，用户挪动螺丝刀后重抓，fixed×3 + handeye×3；补入 Qwen key 后三源全跑通）

- **两路相机都检出螺丝刀**，且**三源框高度一致**（无真值时最有力的交叉验证）：
  - fixed 视角螺丝刀中心：`gui-plus` (359,63.5) / (356.5,63.5) / (356.5,63.5)；`qwen3-vl-plus` (356.2,64.3) / (354.2,63.4) / (354.9,64.1)；`deepseek-flash` (359,59) / (363,53) / (353,66) → **跨源差 2–5 px**。
  - handeye 视角：`gui-plus` (302,273.5) / (301.5,274) / (301.5,260.5)；`qwen3-vl-plus` (300.2,275.5) / (300.2,274.1) / (298.9,274.1)；`deepseek-flash` (301.5,261.5) / (304,290.5) / **误判帧见下** → **跨源差 2–4 px**。
- **fixed 视角框贴画面顶边且极宽**：三源一致给出 `x∈[123,129]…[585,590]`、`y≈0…127`（占画面宽 ~460 px / 72%，y1=0 表示顶部出画）。三源收敛说明目标确实可见，但**框被上边缘裁切** → 对"已知尺寸反解深度"不利（尺寸被系统性裁小 → 深度偏差）。建议把目标挪到固定相机视野中部（或调俯仰）让框完整入画。
- **`auto` 残余风险实锤**：`deepseek-flash` 在 `handeye_02` 帧被判成 `normalized_1000`（其余 5 帧均为 `pixel`），该帧全部框被按 0.64 缩放 → 螺丝刀中心变成 (190,165)，与另两源的 (299,274) 相差 ~110 px。即 #30 的 2% 容差**仍不足以**兜住"像素模型给出 >2% 越界框"的情形。**处置：像 `gui-plus` 一样把该源钉死**——加 `VLM_DEEPSEEK_FLASH_COORD_SPACE=pixel`。
- 行为差异：`gui-plus` 报 2–3 个显著物体；`qwen3-vl-plus` 报 2–8 个（含背景椅子/鞋子）；`deepseek-flash` 报 3–6 个。

**验证**：新增 1 项（`auto` 容差保留像素 4 px 过冲），全量回归 **563 passed / 7 skipped / 0 failed**（`lerobot` 环境，排除 3 个需 `ai2thor` 的模块）。

### 2026-10-10 · 批次 6（第三轮抓帧 + "框贴边"根因取证）

**第三轮真机帧**（`runs/week38/perception_frames/p0_v3`，用户再调相机与螺丝刀后抓取）

- 三源在 6 帧全部检出螺丝刀，`deepseek-flash` 钉死 `pixel` 后 6/6 帧均为 `pixel`（批次 5 的 `handeye_02` 误判消失）。新增 `.env`：`VLM_DEEPSEEK_FLASH_*`（含 `_COORD_SPACE=pixel`）。
- **两路视角的框都已完整入画、且三源高度一致**：
  - fixed：`gui-plus` (292.5,182.5) `[31,129,554,236]`、`qwen3-vl-plus` (290.9,174.2) `[31,115,551,234]`、`deepseek-flash` (290.0,175.0) `[25,118,555,232]` → 中心跨源差 **≤3.5 px**，`y1=115–129`（**不再贴顶边**）。
  - handeye：三源中心 (278.5–283.5, 175–189.5) → 跨源差 **≤5 px**。

**"框贴边（尺寸被裁）"根因取证 —— 排除"API 压缩图片"假设**

| 假设 | 取证 | 结论 |
|---|---|---|
| API 内部缩放导致框贴边 | `image_to_data_url` 仅在 `max(img.size) > 640` 时缩放；帧正好 640×480 → **不缩放**，仅 JPEG q85 重编码（实测发出 640×480 / 29 KB，原 PNG 397 KB） | **否定** |
| 同上（几何论证） | 降采样/JPEG 不改变像素位置，只能降精度（几像素），无法把内容"推"到画面边缘 | **否定** |
| 同上（反证） | 早前合成图冒烟（真值 200,150–280,250）误差仅 2.2/2.7 px；若 API 缩小后再按缩小空间给坐标，会出现成比例（~20%）缩放误差，实测没有 | **否定** |
| 真实取景裁切 | 像素取证：`p0_v2/fixed_00` 第 0 行 `x≈348–397` 有 46 个前景像素 → 该帧顶部**确有内容出画** | **成立（主因）** |
| 模型框过度外扩（独立问题） | 同帧模型报 `x∈[123,590]`，但该带内真实内容只在 `x≈348–584`（左侧 ~225 px 是空背景） | **成立（次因）** |

→ 结论：`p0_v2` 那个"又宽又贴顶"的怪框 = **真实顶部裁切 + 模型外扩** 叠加，**与图像压缩无关**。批次 5 把它笼统称作"尺寸裁剪"不够准确，此处更正。取景主因已由用户本次调整解决（`y1: 0 → 115–129`）；模型外扩属 bbox 质量，正是 0.3 选型要量化的对象。

**验证**：本轮仅新增 `.env` 配置项（无代码改动），回归结果同上。

### 2026-10-10 · 批次 7（0.3 选型表前置：真值标注工具）

0.3 的最后一步"选型表"需要**真值**。计划未定义真值获取方式，本批补上，原则是**真值必须独立于被评模型**（否则误差被系统性低估=循环论证），故工具只把模型结果当"候选初值"，最终值以人工确认为准。

| 文件 | 作用 |
|---|---|
| `src/run_perception_gt_label.py`（新增） | 真值标注 CLI：`--report <report.json>` 从三源评测结果逐轴取**中位数**生成候选框；OpenCV 窗口内鼠标拖拽重画 / `Enter` 采用 / `r` 恢复候选 / `s` 跳过 / `q` 保存退出；产出 `gt.json`（与 `--gt` 消费格式一致）；`--dry-run` 无窗口自检 |
| `tests/test_perception_gt_label.py`（新增） | 27 项纯逻辑：bbox 解析与退化/非有限拒绝、中位数对离群框的抑制、中英关键词别名匹配、跨源候选生成（取面积最大的**命中目标**而非全帧最大框）、`gt.json` 读写与评测格式兼容、`eval/` 子目录叠加图不误当帧 |

**实测候选**（`p0_v3`，三源中位数）：`fixed_00 [31,118,554,234]` / `fixed_01 [32,115,550,234]` / `fixed_02 [32,118,552,234]` / `handeye_00 [241,127,322,249]` / `handeye_01 [241,127,323,248]` / `handeye_02 [241,127,326,248]`。

**验证**：新增 27 项测试；全量回归 **590 passed / 7 skipped / 0 failed**（`lerobot` 环境，排除 3 个需 `ai2thor` 的模块）。

### 2026-10-10 · 批次 8（0.3 收尾：真值确认 → 定位源选型表）

1. **补齐尺寸误差**：`run_perception_locate_eval.py` 的 docstring 承诺"另算中心/尺寸误差"，实现却只有中心误差。新增纯函数 `bbox_error(det, gt) -> (中心欧氏误差 px, [宽差, 高差] px)`（尺寸差 = 检测 − 真值，正数=框偏大），逐帧与汇总均输出。
2. **真值确认**：用户用 `run_perception_gt_label.py` 确认 6 帧候选（三源中位数）→ `runs/week38/perception_frames/p0_v3/gt.json`。

**0.3 定位源选型表**（`p0_v3`，6 帧 = fixed×3 + handeye×3，命中阈值 ≤10 px）

| 定位源 | 坐标约定 | 平均中心误差 | 平均尺寸误差(宽×高) | 命中率 |
|---|---|---|---|---|
| **`qwen3-vl-plus`** | normalized_1000 | **1.31 px** | **−1.46 × +0.08 px** | **6/6 = 100%** |
| `gui-plus` | pixel | 3.37 px | +2.05 × −1.95 px | 6/6 = 100% |
| `deepseek-flash` | pixel | 22.81 px | −20.12 × −7.12 px | 3/6 = 50% |

**结论**

- **0.3 选型定为 `qwen3-vl-plus`**：中心误差 1.3 px、尺寸近乎无偏（±1.5 px），是在 640×480 帧上唯一两端都稳的源。`gui-plus` 可作为**交叉校验/次选**（3.4 px，轻微外扩 ~2 px）。
- **`deepseek-flash` 不作为定位源**：跨运行不稳定（同帧同源两次运行 hit_rate 0.83 → 0.50，`fixed_02` 单帧中心误差 87 px / 宽度差 −142 px），且尺寸系统性偏小。
- **P0 几何链前提成立**：~1.3 px 中心误差配合已知尺寸反解，在 30–60 cm 工作区远优于 A1 的 1.0 cm 门槛（换算见后续消融）。
- **限定**：样本仅 6 帧、单目标、单场景、单次拍摄；`deepseek-flash` 的跨运行波动本身就是其不可用的证据，但 `qwen3-vl-plus` 的稳定性仍需在多场景/多目标上复测。

**验证**：新增 14 项（`bbox_error` 中心/尺寸/欧氏；源注册表 env 回退与覆盖、非法约定报错、配色确定性）；全量回归 **604 passed / 7 skipped / 0 failed**。

### 2026-10-10 · 批次 9（0.6 空闭基准标定 —— 门禁项，真机实测）

新增 `src/run_grip_baseline_calib.py`：R 次**独立连接** × P 个**姿态** × N 次采样，
每次「张开 → 闭合 → 稳定 → 采样 → 退回」，测 `empty_grip_baseline` 的跨运行/跨姿态波动
并与 `margin` 联合标定。**只测量、只给建议，未改动任何真机判定行为。**

**真机实测**（3 次连接 × 3 姿态（wrist_flex 60/75/90）× 5 采样 = 45 个样本）

| 指标 | 实测 |
|---|---|
| 空载闭合位置 | 均值 **3.921** / σ **0.044** / 范围 [3.905, 4.045] |
| 跨运行波动（极差） | **0.047**（组均值 3.905 / 3.952 / 3.905） |
| 跨姿态波动（极差） | **0.047**（60→3.905、75→3.905、90→3.952） |
| 单次内采样噪声 σ | **0.0**（9 次试验全部 `close_status=stalled`，确实顶到限位） |
| 空载负载 | −62 ~ −69‰（CP0 时为 −55‰ → 负载基线跨次漂 ~7‰） |

**关键发现**

1. **`empty_grip_baseline` 默认值 1.5 与实测 3.92 相差 2.4，配置与物理含义脱节。** 旧配置
   （1.5 + margin 3.0 = 阈值 4.5）看似能检出薄件（夹持位 6.8 > 4.5），但它对空载上限 4.045
   只剩 **0.455 的净空**——名义 margin=3.0，**实际只有 0.46**，是靠"基准写小了"换来的，不是真余量。
2. **建议改为 `empty_grip_baseline = 4.1`（实测上限向上取整）+ `margin = 1.4` → 阈值 5.5**：
   对空载上限净空 **1.455（≈33σ）**，同时仍低于薄件夹持位 6.8（余量 1.3）。
3. 空闭位置**跨运行/跨姿态都极稳**（极差均 0.047、采样噪声 0）→ 基准可靠，无需为大漂移预留余量；
   `close_status=stalled` 也证实"空载闭合停在 3.9 是顶到机械限位"，与 CP0 结论一致。
4. **判据建议：位置 OR 负载**。CP0 实测夹持负载 Δ≈144–151‰ vs 负载阈值 50‰（≈3 倍余量），
   比位置判据稳健得多；薄于该件的目标应靠负载兜底。

**实测踩到的工程坑（已修，且写入 docstring）**

`max_relative_target=8.0` 会把**单次** `send_action` 的目标裁到"当前位置 ±8"
（日志形如 `goal_pos 3.0 → safe 11.18`）。首轮只发一次指令 → 夹爪停在 **11.86** 而非 3.9，
测出一个完全失真的"空闭位置"。必须像 `RealEngine.move_to` 那样**反复下发直到「到位/卡住」**。
另需屏蔽该限幅告警（本脚本就是要一直顶到限位，否则每次下发一条、输出被几千行淹没）。

**验证**：新增 19 项（`stats` / `spread_by` 分组极差 / `suggest_threshold` 上限取整与余量、
"旧配置净空仅 0.46"的对照、夹持位低于基准时判不可行 / `parse_poses`）；全量回归
**623 passed / 7 skipped / 0 failed**。

### 2026-10-10 · 批次 10（落地 0.6 结论：标定值 + 「位置 OR 负载」判据）

用户决策"从实际出发"，采纳 0.6 建议，改动真机夹持判定链路：

| 文件 | 改动 |
|---|---|
| `real_observation.py` | 新增标定常量 `EMPTY_GRIP_BASELINE=4.1` / `GRIP_POS_MARGIN=1.4` / `EMPTY_LOAD_BASELINE=-69.0`；`is_holding` 默认值改标定值；**`detect_contact` 由带符号比较改为增量幅值** `|load−baseline| > margin`；新增 `holds_object()`（位置 OR 负载） |
| `real_verifier.py` | 默认值取标定值；`verify_grasp_lift(..., final_load=None)` / `verify_holding(..., load=None)` 接入负载 |
| `real_executor.py` | `RealSkillExecutor` 默认 baseline → 4.1；新增 `_read_gripper_load()`（读不到则退化纯位置判据）；`MockEngine` 负载与 `grasp_result` **同口径**（咬空 −62‰ / 夹持 −206‰） |
| `run_real_agent.py` / `run_real_eval.py` | 默认 `empty_grip_baseline` → 标定值；CLI `--empty-grip-baseline` 默认同步 |

**必改的一处语义 bug（0.6 数据暴露）**：`detect_contact` 原判据是 ``load > baseline + margin``。
本硬件上"空载顶住机械限位"与"夹住物体"是**同向**扭矩、只有幅值不同（实测空载 −62‰、夹持 **−206‰**），
带符号比较 ``−206 > −69 + 50`` **恒为假 → 负载判据必然漏判、完全失效**。改为增量幅值后与
`summarize_load_probe` 的 ``|Δ|`` 口径一致。既有 4 项 `detect_contact` 测试在新语义下**全部仍通过**。

**安全性质（改后逐一验证）**

- 空载闭合上限 4.045 < 新阈值 5.5 → **不误判**（净空 1.455 ≈ 33σ）
- 薄件夹持位 6.8 > 5.5 → 位置判据可检出；更薄的件由负载判据兜底（|Δ|≈137‰ vs 50‰，≈2.7 倍）
- **读不到负载 → 退化纯位置判据**，绝不因"没读到"而判成夹住（`_read_gripper_load` 返回 `None`）

**验证**：新增 24 项（标定默认值与净空对照、`detect_contact` 幅值语义与"带符号会漏判"对照、
`holds_object` 真值表与负载缺失退化、`RealVerifier` 负载路径、`MockEngine` 负载口径、
无负载引擎退化）；全量回归 **647 passed / 7 skipped / 0 failed**；
`run_real_agent.py --dry-run --backend fake` 端到端 OK。

**遗留**：改的是**真机判定行为**，建议真机跑一次 `grasp_screwdriver` 复验（位置与负载两路各一次），
确认咬空/夹持都能正确分类。

### 2026-10-10 · 批次 11（真机判定复验 + 真机链路 3 个问题）

**A. 真机判定复验（两个场景都判对）**

| 场景 | 实测 | 位置判据 | 负载判据 | `holds_object` |
|---|---|---|---|---|
| 咬空（钳口空） | pos **3.91** / load **−62.0‰** | 3.91 > 5.5 → 假 | \|−62+69\|=7 > 50 → 假 | **False** ✅ |
| 夹住（螺丝刀在钳口） | pos **6.69** / load **−100.0‰** | 6.69 > 5.5 → **真** | \|−100+69\|=31 > 50 → 假 | **True** ✅ |

两条附带结论：① 夹住位 **6.69** 与 0.6 标定的 6.8 一致，交叉印证；② **这次负载只有 −100‰**（CP0 时是 −206‰），
差值 31 < `LOAD_MARGIN`=50 → **光靠负载判据会漏判**，「位置 OR 负载」的设计正好兜住。
→ 待累积更多样本后再决定是否下调 `LOAD_MARGIN`（目前位置判据是主力）。

**B. 真机链路 3 个问题（端到端实抓暴露，均已修）**

| # | 问题 | 证据 | 处置 |
|---|---|---|---|
| 1 | `grasp_and_lift` **不从初始姿态起跑**（缺归位） | 实测起跑时 `shoulder_lift=−100` vs 演示初始 `44.9`，偏差 **145** 单位 | `_do_grasp_screwdriver` 先 `home()`（与 `infer_so101` 闭环协议一致）；归位失败 → `FAIL_HOME_TIMEOUT` 且**不进入策略段** |
| 2 | 策略段超时**写死 120 s** | 默认 300+300 步，CPU 上 ~0.3 s/步 → 必然超时（本轮实测失败码就是 `timeout`） | 新增 `GRASP_STEP_BUDGET_S=0.5` + `RealSkillExecutor.grasp_timeout()` 按步数缩放（默认 300 s），可显式覆盖 |
| 3 | 归位**永远不可能成功** | `infer_so101.HOME_TOL=0.03` 注释写"弧度"，但状态是 lerobot 整流量（±100）→ 0.03 低于舵机分辨率（1 计数≈0.05） | `HOME_TOL` → **3.0**（对齐 `RealEngine.move_to` 容限）；`HOME_TIMEOUT` 10 → **45 s**；`home_to_initial` 增加 `timeout_s` 参数 |

**真机阻塞（端到端实抓未完成）**：`shoulder_lift` 卡在**恰好 −100.00**，两个方向各下发 12 次都 **Δ=0.00**；
而同一总线上的 `wrist_flex`（Δ=+4.66）、`elbow_flex`（Δ=+2.57）**正常响应** → 定位为**该关节级硬件问题**
（舵机过载保护 / 机械卡阻 / 标定偏移），非软件超时。需现场处置（项目既有 SOP：断电重启）后才能跑端到端实抓。

**验证**：新增 5 项（超时按步数缩放与下限、显式覆盖、归位失败不进入策略段、归位先于策略）；
全量回归 **652 passed / 7 skipped / 0 failed**；`run_real_agent.py --dry-run --backend fake` 端到端 OK。

### 2026-10-10 · 批次 12（真机端到端实抓：链路打通，抓取本身失败）

关节恢复后（`shoulder_lift` 双向 Δ −4.59/+4.67 正常）跑 `run_real_agent.py --backend fake`（真机引擎 + 脚本化后端）：

| 环节 | 结果 |
|---|---|
| `shoulder_lift` 微动复测 | ✅ 恢复（−5→Δ−4.59、+5→Δ+4.67） |
| **归位** | ✅ **成功，用时 2.4 s**（修复 #3 直接验证：`HOME_TOL` 改 3.0 后容差可达；旧值 0.03 必然超时） |
| **策略段** | ✅ 跑完 **196 s**（22:31:20→22:34:36），未触发超时（预算已按步数改为 300 s；旧值 120 s 必然超时） |
| **夹持判定** | ✅ 正常执行，返回 **`grasp_empty`**（夹爪闭在空处） |
| 抓取本身 | ❌ **失败**——策略没能把末端送到螺丝刀并夹住（与 `outputs/SO101_闭环问题总结_20260905.md` 记录的已知失败模式一致） |

**顺带发现**：Agent 最终把这次运行报成"成功：任务完成"。原因是 `FakeBackend` 的脚本固定为「1 次 grasp + done」（`--mock-result success`），grasp 失败后它照样发 `done`，而 **`done` 不校验目标是否真的达成** → 失败被"报成功"。属测试脚手架的产物，但也说明「done 需要一道目标校验」是真实缺口（P0 之外，待评估）。

→ 「归位 → ACT 策略 → 判定」整链**首次在真机打通**；剩下的差距是 ACT 策略本身的抓取能力（正是 P0 消融 B0 基线的对照对象）。

### 2026-10-10 · 批次 13（收尾自动复位 + "位置残留"假设取证）

**用户诉求**："运行结束后应自动回到起始位"。现状确实没有这一步（脚本只有 `grasp → done`，跑完停在策略最后把手臂带到的位姿）。

**先排除三个候选成因（都用真机取证，全部否定）**

| 假设 | 实测 | 结论 |
|---|---|---|
| 连接后被**残留 `Goal_Position`** 自动带走 | 连接后静置 2 s，六轴漂移 **全 0.00** | ❌ |
| 位置读数 / 标定零点对不上 | 标定文件在（六轴 `homing_offset`/`range` 齐全），原始值→归一化一致 | ❌ |
| 归位没到位 | `home() -> True`，0.0–0.3 s，最大偏差 **2.57**（±100 量纲） | ❌ |
| 断开松力矩导致**下垂** | 通电保持 25 s 位姿稳定；断开 6 s 后重连偏差 **≤0.29**（减速比自锁） | ❌ |

→ 真实成因就是**缺少收尾复位**，与"残留坐标"无关。

**改动**

| 文件 | 改动 |
|---|---|
| `real_executor.py` | `RealSkillExecutor.park()`：收尾复位到采集初始姿态；失败只返回 `False` 不抛异常（收尾动作不该盖过主流程结论）。docstring 注明"若夹着物体会把物体一起拖走" |
| `run_real_agent.py` | `finally` 中调用 `park()` 并打印结果；新增 `--no-park` 逃生开关 |
| `run_real_eval.py` | 每轮任务 `finally` 中 `park()`（避免任务间位姿漂移累积），引擎关闭改到 `finally` |

**真机验收**：完整跑一次 `run_real_agent.py --backend fake`（真机），结束后读位姿 —

```
[收尾] 复位到采集初始姿态: 成功
运行结束后位姿 = [-7.88, 44.16, -29.94, 88.74, -4.76, 1.67]
采集初始姿态   = [-6.80, 44.92, -30.61, 89.44, -6.58, 1.46]
最大偏差 1.82（±100 量纲，容差 3.0）✅
```

**验证**：新增 3 项（park 调用 home、失败返回 False、引擎异常被吞）；全量回归 **655 passed / 7 skipped / 0 failed**；dry-run 端到端显示 `[收尾] 复位到采集初始姿态: 成功`。

### 2026-10-10 · 批次 14（下垂复测 + 抓取失败根因取证：相机视角 OOD）

**① "断开不下垂"结论被推翻（用户指正）**：上一轮测的是**手臂压在桌面上**的位姿，没有下垂空间。用户把手臂抬到离桌面有间隙处后复测：

```
通电保持 3s  : shoulder_lift = -44.50   Δ=0.00
断开静置 10s : shoulder_lift = -34.89   Δ=+9.61   ← 确有下垂
```

→ `park()` 只保证**进程存活期间**回到起始位；进程退出后松力矩仍会掉约 10 个单位。彻底解决需 `disable_torque_on_disconnect=False`（代价：舵机持续通电），**待决策**。

**② 抓取失败根因取证：相机视角与训练时已不一致（疑似 OOD）**

线索：用户观察"能走到螺丝刀位置，但夹爪不张开、后续没有动作"；而历史闭环日志（9/5，990 步）显示 `action_gripper.pos` 在 **1.35–20.27** 正常变化 → **夹爪动作链路本身是好的**，问题在策略输入。

取证：归位到**采集初始姿态**后抓帧，与历史同姿态帧（`outputs/infer_run_*/frames/homed_*.png`）对比：

| 相机 | MAE | 相关系数 |
|---|---|---|
| fixed | 69.1 | **−0.109** |
| handeye | 66.3 | 0.265 |

**手臂姿态完全相同、画面基本不相关** → 命中项目硬约束"相机位置在采集与推理期间必须不变"：为感知评测调过两次相机。ACT 是纯视觉条件化策略，输入 OOD 时动作失去意义，与"走到位置后不再动作"的现象一致。

**限定**：相隔一个月，桌面物品/光照变化也会贡献差异；构图级不一致（fixed r≈−0.11）更像相机被挪动，**最终判据是人工对比**（当前帧 `runs/week38/cam_check_now/homed_fixed.png` vs 历史 `outputs/infer_run_20260906_025602/frames/homed_fixed.png`）。

**待决策**：① 是否改 `disable_torque_on_disconnect`；② fixed 相机是"调回与历史帧一致"还是"固定后重新采集微调"。

**③ 根因锁定：相机索引映射漂移（已修）**

用户逐帧人工确认当前枚举：**index 0 = 夹爪特写、index 1 = 笔记本摄像头、index 2 = 工作台俯拍** → 原映射 `handeye=1` 时策略的**腕部通道拿到的是笔记本摄像头画面**（对着人），这才是"走到螺丝刀后不再动作"的直接原因；`fixed=2` 一直是对的。

处置（四处）：`outputs/infer_so101.py` 的 `CAM_IDX`、`src/real_executor.py` 的 `DEFAULT_CAM_IDX`、`src/run_perception_capture.py` 的 `DEFAULT_CAMERAS`、`src/run_vla_collect.py` 的录制模板 → 全部改为 **`{"fixed": 2, "handeye": 0}`**，并在注释/文档里写明**枚举会漂移、跑前必须抓帧核对**。

**方法学教训（已写入注释）**：用"像素相关性与数据集训练帧比对"自动判映射**不可靠**——腕部相机的画面随手臂姿态变化，拍摄时手臂不在初始位姿，导致 **index0（真腕部）得分 0.135 反而低于 index1（笔记本摄像头）的 0.555**，给出错误结论。**最终判据必须是人眼核对。**

**仍未闭环**：fixed 相机（index 2）与 `dataset.fixed` 仅 r=0.303（同相机的正常水平应 ≈0.8），说明它被调过位置，需物理调回；调好后重跑抓取验证。

### 2026-10-10 · 批次 15（真机硬件故障取证：shoulder_lift 力矩不足）

重跑实抓时失败码变为 **`home_timeout`**（归位 45 s 未到位），日志显示 `elbow_flex` 反复被限幅在同一值（位置不变）。逐项取证：

| 检查 | 结果 | 结论 |
|---|---|---|
| 总线枚举 | 首次连接**缺 `id 6`（夹爪）** → 用户检查排线后恢复（六轴齐全） | 排线确有接触问题，已排除 |
| 寄存器体检 | `Torque_Enable=1`、电压 12.2–12.4 V、温度 43–47℃、六轴**配置完全一致**（`Max_Torque_Limit`/`Torque_Limit`=1000、`Protection_Current`=3000、PID 16/32/0、`Overload_Torque`=80） | **排除"力矩上限被设低"** |
| 自由度（当前位置） | shoulder_lift 上下 ±4 均能动（−3.14 / +3.91） | 当前位置**未卡死** |
| 抬升测试 | −52.7 → **冻结在 −42.46 / −35.40**（多次运行位置不同），60 次下发零位移；负载恒定 **347‰**；**电压稳定 12.1 V** | **排除供电/排线压降** |
| **堵转电气量** | 堵转：负载 **347‰**、电流 **256–263**（`Protection_Current`=3000）、温度 49℃；手臂下垂时电流仅 **4**；健康关节 `wrist_roll` 电流 11 | **电流接近保护上限但只输出 ≈35% 额定力矩** → 效率衰减（电机/减速箱），非配置问题 |

**结论**：`shoulder_lift`（id 2）**力矩不足/效率衰减**，抬不动整条手臂 → 归位失败 → 实抓链路被硬件阻断。**已提示用户停止继续堵转驱动**（电流接近保护值、温度在升）。

**待用户确认**：断电/松力矩后**用手抬该关节的手感**——顺滑 ⇒ 舵机本体故障（换件，BOM 有备件，换后**必须重新 `lerobot-calibrate`**）；卡涩/异响 ⇒ 机械问题（关节过紧 / 排线拉扯）。

**复测（用户物理调整 + 重新接好电源后）**：结果与调整前**完全一致** —— 抬升仍冻结（本次 −66.34），**堵转负载恒 347‰、电流恒 ~262**、电压 12.1 V；手臂姿态不同导致冻结位置不同，但**能力天花板一模一样**。

→ 说明**不是外部机械卡阻**（那会随机械状态变化），而是**舵机自身可输出的力矩就到这个水平**。另外期间还出现过整条总线 `found={}`（用户接电源后恢复）、以及此前 `id 6` 掉线——排线接触偏弱，建议顺手复查。

**建议**：更换 `id 2`（shoulder_lift）舵机 —— BOM 留有备件；**换后必须 `lerobot-calibrate` 重新标定**（新舵机零点/量程不同，否则位置全错），再复跑本批次的三项测试确认。

**⚠️ 结论更正（用户复测后）—— 舵机正常，"舵机衰减"判断为误判**

用户**手动把其他关节角度调开、让手臂离开桌面**后，同一个舵机跑出完全健康的曲线：

```
shoulder_lift: -99.83 → 45.86（目标 44.9，偏差 1.0）✓  保持 5 s 不回弹
负载:  276 → 305 → 277 → 273 → 164 → 140 → 119 → 100 → -58‰
电流:  232 → 190 → 140 →    92 →  87 →  38 →   8 →   0 →  8
```

**电流在最低点（重力力矩最大）最高、越往上越小、到顶几乎为 0，负载随重力正常变化** —— 这是健康舵机的标准曲线。**无需换件，也没有重新标定。**

**真正的成因（用户指出，正确）**：之前手臂完全塌下、**夹爪压在桌面上**，上抬等于要把整条手臂从桌面撬起 → 外部阻力极大 → 表现为"电流拉满、位置不动"（那个恒定的 347‰ 其实是**桌面的反作用力**，不是舵机极限）。同理，本批次最早那次 `shoulder_lift` 卡在 −100.00 两个方向都不动，也属于同一类情形。

**教训**：① **运行/归位前必须确认手臂未压在桌面上或卡在其它障碍上**；② 诊断"关节不动"时，**先排除外部接触阻力**，再看电气量——只看"电流高、位置不动"很容易误判为舵机故障（本次即为误判，成本是一整轮排查）。

### 2026-10-10 · 批次 16（公平测试：相机映射修正后的首次实抓 → 确认策略能力不足）

相机映射修正（`fixed=2 / handeye=0`）+ 手臂不压桌面后，跑一次完整 `run_real_agent --backend fake`（真机）：

| 环节 | 结果 |
|---|---|
| 归位 | ✅ 成功（0.1 s，手臂已在初始位姿） |
| 相机 | ✅ 日志显示使用 index **2 + 0**（映射已生效） |
| 策略段 | ✅ 跑完 196 s，未超时 |
| 夹持判定 | ✅ 正常执行，返回 **`grasp_empty`** |
| 收尾复位 | ✅ 成功 |
| **抓取本身** | ❌ **仍未夹到** |

**结论**：在**输入正确**的条件下策略仍抓不到 → **ACT 抓取能力不足**成立（此前"相机映射错"是并行存在的另一个问题，已修）。与历史一致：`outputs/SO101_闭环问题总结_20260905.md` 记录的 9/5 现象同为"走到一半停下、夹爪闭在空处"，当时映射尚未出错。

**含义**：抓取要真正可用需"重新采集 + 微调"（即 P1/P2 的内容），**不是继续调环境能解决的**。因此**暂时跳过真机抓取**，改为：① 用「真机就绪度冒烟测试」（不含策略）把除策略外的环节固化；② 推进不依赖真机抓取的深度化支柱（P1 数据引擎 / P3 世界模型评估）。

### 2026-10-10 · 批次 17（真机就绪度冒烟测试 + 抓到一个判据假阳性）

**新增 `src/run_real_smoke.py`**：每次上机前的 30 秒就绪检查，**不跑 ACT 策略**，逐项 PASS/FAIL（退出码 0/1）：
`总线(六轴在线) → 归位 → 双相机出帧(存图，含映射核对) → 负载可读 → 夹持判定(空载应 False) → 收尾复位`；
`--dry-run` 用 Mock 验证脚本链路，`--with-object` 追加"放入物体 → 应判夹持"段。

**真机首跑 5/6，冒烟测试立刻抓到一个真 bug（我此前埋的）**：

```
[FAIL] 夹持判定  空载 pos=1.39 load=0.0 → holds_object=True（期望 False）
```

根因：夹爪**张开/空转**时负载是 **0‰**，而旧判据是"相对空载基准（−69‰）做差"→ `|0 − (−69)| = 69 > LOAD_MARGIN(50)` → **误判成夹持**。本质错误在于**空载负载不是一个常数**：它随夹爪状态在 0‰（张开/空转）~69‰（闭合顶限位）之间变化，单一基准做差必然出错。

**修复**：`holds_object` 的负载判据由"相对基准做差"改为**幅值判据** `|load| > GRIP_LOAD_THRESHOLD`。阈值取 **150‰（偏保守）**，实测依据：

| 状态 | |load| | 是否应判夹持 |
|---|---|---|
| 夹爪张开/空转 | ≈ 0‰ | 否 |
| 空载闭合（顶限位） | 55–69‰ | 否 |
| 抓住螺丝刀（薄件） | ~100‰ | 是（但只比空载高 1.4×，**太接近**） |
| CP0 夹持 | ~206‰ | 是 |

→ 阈值取 150‰：**只对"明确受力"的夹持生效，宁可不触发也不许误判**；薄件由**位置判据**兜（实测夹持位 6.69 > 阈值 5.5，分离清晰）。`real_verifier` 的 `load_threshold` 参数同步替换 `empty_load_baseline/load_margin`。

**验证**：新增 2 项回归测试（张开态 load=0 不得误判；阈值保守性 −100 不触发 / −206 触发）；全量回归 **671 passed / 7 skipped / 0 failed**；**真机冒烟测试复跑 6/6 通过（exit 0）**。

**就绪度小结**：除 **ACT 策略抓取能力**外，真机链路（总线/归位/双相机/负载/夹持判定/收尾复位）**全部就绪**。

### 2026-10-10 · 批次 18（P3 代码完成 + 云上环境搭建）

**P3 代码（子任务 3.1–3.5）已完成**，本机可验证（不依赖 ManiSkill）：

| 文件 | 内容 |
|---|---|
| `src/wm_ope.py`（新增） | `OPEReport`（对齐设计 §4.2，另加 `warnings`/`verdict`/`attribution`）、`spearman_rho`（**并列值平均秩**；某侧全同则无定义抛错）、`bootstrap_ci`（按候选 Bernoulli 重采样）、`evaluate_ope`（候选校验：≥5 / 每候选 ≥10 rollout / 数量一致 / 跨族告警 / 成功率∈[0,1]）、`rollout_policy_in_world_model`、`state_to_observation`、`distribution_shift`/`attribute_low_rho`、`write_ope_report` |
| `src/run_wm_ope.py`（新增） | 候选注册表 **6 候选 / 4 行为族**（`scripted` 锚点 + `scripted-noise-0.05/0.20` + `scripted-lag-3` + `random` + `zero`）；两种模式：`--eval-sim`（ManiSkill，云上）与 `--wm --sim-eval`（纯 NumPy） |
| `tests/test_wm_ope.py`（新增） | **45 项**：ρ 完美/反向/并列手算/无定义、候选数不足/rollout 不足/单族告警、归因、报告落盘、seed 解析、注册表约束、包装策略 |

**两个关键设计决策**（写入 docstring）：
1. **两侧必须用同一批初始状态** —— 初始状态由**仿真侧记录**、WM 侧复用，而不是各自独立采样；否则比的是两个不同初始分布，ρ 无意义。
2. **逆适配器必须可逆** —— 测试强校验 `state → observation → observation_to_state` **完全等于**原状态；WM 内成功判据与 `ScriptedPolicy` 的 push 终止判据同口径（obj-goal xy 距离 < 0.025）。

**验证**：全量回归 **716 passed / 7 skipped / 0 failed**。

**云上环境（A10 24G / Python 3.12）搭建踩坑记录**：
1. `download.pytorch.org` 从国内云**拉不动**（卡在 0 字节/3.3 kB，5 分钟无进展）→ 改 **阿里云 PyPI 镜像**；PyPI 的 Linux `torch` 默认即带 CUDA，无需专用 index。同时把 torch 钉在 `>=2.7,<2.12` 以对齐 lerobot 0.6.1，避免装完被降级二次下载。
2. **v1 的 pip 未被杀死**，与 v2 的 pip 同时写同一个 venv → 互抢锁、看似"卡死"；v3 脚本开头强制清理残留 pip + 删半成品 venv。
3. **`pkill -f setup` 会杀掉脚本自身**（脚本名 `setup3.sh` 含 `setup`）→ 脚本自杀；改用 `setsid` + 精确 pattern。
4. **本地 `lerobot` 是 0.3.4**（`Downloads/lerobot` 源码版），读不了 v3 数据集（无 `LeRobotDataset`）→ **P3 整条链路都放云上跑**（顺带省掉传大文件）。

**云上执行顺序（待依赖装完）**：同步 `src`(4MB) + `week15-cam`(80MB) ✓ 已完成 → 云上训世界模型（`run_world_model.py train`）→ `run_wm_ope.py --eval-sim` → `--wm --sim-eval` 出 `report.json`/`report.md` → 回传本地。

### 2026-10-10 · 批次 19（P3 完成：OPE ρ = 0.398 → 归因报告）

**环境（云 A10 24G / Python 3.12）**：`torch==2.9.1+cpu`（SJTU `pytorch-wheels/cpu` 索引）、`lerobot[dataset]==0.6.1`、`mani_skill==3.0.1` + `sapien==3.0.3`、`gymnasium 1.4.0`；`pip check` 无冲突。

**结果**（7 候选 / 5 行为族，每候选 10 rollout）：

| 候选 | 行为族 | 仿真实测 | WM 预测 |
|---|---|---|---|
| `scripted` | scripted | 1.000 | 0.100 |
| `scripted-noise-0.5` | scripted-degraded | 0.900 | 0.000 |
| `scripted-noise-1.0` | scripted-degraded | 0.500 | 0.100 |
| `scripted-lag-8` | scripted-degraded | 0.000 | 0.000 |
| `scripted-reverse` | scripted-adversarial | 0.000 | 0.000 |
| `random` | random | 0.000 | 0.100 |
| `zero` | degenerate | 0.000 | 0.000 |

**Spearman ρ = 0.398（95% bootstrap CI [0.322, 0.398]）→ 判定 marginal**（低于 A8 的 0.6）。按设计"ρ < 0.6 则提交归因报告（**负结果亦算通过**）"交付。

**归因（两条硬证据）**：

1. **单步很准、多步指数发散**：独立 seed 现采 scripted 转移 → WM 单步 MAE(全维) **0.00655**、obj xy 每步位移 真实 0.00934 vs 预测 0.00965（**比值 1.03**），单步动力学基本正确。但 **40 步 rollout 的 obj-goal xy 距离：0.20→0.21→0.38→0.96→2.42→5.62→13.6→44.5→133→433→1629**（≈每 4 步 ×2.6）；真实中**恒定不变**的 goal 在想象里漂到 437×4468。即：MLP 对"恒定维度"的微小输出噪声经反馈被逐步放大 = 经典 rollout error compounding。horizon 40→25、训练规模 978 转移/4000 步 → **2220 转移/30000 步（256-256）** 之后，WM 侧才从"恒为 0（ρ 无定义）"变成可取值 {0.1, 0.0}，但**仍无辨别力**。
2. **访问分布偏移随候选退化急剧放大**：WM rollout 实际访问状态相对训练分布的 |z| —— `scripted` 230.8 / `lag-8` 326.2 / `zero` 467.1 / `random` 735.6 / **`reverse` 1189.7**，OOD 占比 0.73 → 0.98。即 WM 的想象 rollout **很快离开训练分布**，其成功率预测不可信。

**结论**：该 WM（MLP 1-step 动力学 + **开环** rollout）**不足以充当"廉价筛选器"**。要让它可用，需要多步/rollout 训练目标（而非 1-step MSE）、对恒定维度的结构化处理、或闭环校正（用真实状态重置）——超出本周范围，作为负结果与后续方向记录。

**⚠️ 与设计的偏差（6 项）**：① WM 训练数据源由 week15 v3 数据集改为**实时仿真采集**（v3 的 PushCube 演示物体每集只移动 5–7 cm、目标在 20 cm 外，**不含任何成功轨迹**，学出的动力学只会预测"物体不动"）；② horizon 40→25（对齐任务平均步数）；③ 候选退化档强化（σ≤0.20 与 lag-3 在仿真里仍 100% 成功，真实成功率坍塌成 {1.0,0.0} 无梯度）；④ WM 内成功判据由 3D 改为 **xy** 距离（物体在桌面 z≈0.02、目标贴地 z≈0.001，z 恒有 ~0.019 偏移，3D 配 0.025 阈值等效要求 xy<0.016）；⑤ 归因改用候选**实际访问过的状态**（原用初始状态，而两侧初始状态按设计相同 → 无区分度）；⑥ 归因统计**跳过训练中近似恒定的维度**（std≈0 被兜成 1.0，z 分数无意义，原实现把所有候选判成 OOD=1.0）。

**产物**：`runs/week38/wm_ope/pushcube/{report.json,report.md}` + `wm_training_summary.json`（云上生成，已回传本地）。回归 **720 passed / 7 skipped / 0 failed**（P3 代码部分）。

**遗留与下一步**

### 2026-10-10 · 批次 20（P1 代码核心 2.1–2.6 完成）

**数据源确认**：真机多任务集（2.9）尚未采集（真机线搁置），故 P1 先用本地已有的
`wangbangyan/demo_v23`（真机 SO-101 录制，**lerobot 0.3.x 格式** → 本地 lerobot 0.3.4 可读）作标注对象：
**68 episodes / 41915 帧 / 2 条指令**（"抓起螺丝刀并抬起"、"张开夹爪靠近螺丝刀，夹住并抬起"），
两条都映射到 `pick`。

**已实现**：

| 子任务 | 文件 | 内容 |
|---|---|---|
| 2.1 | `src/vlm_labeler.py` | `LabelRecord`（设计 §4.1 字段 + `taxonomy_version`/`note`）含完整校验与 `to_dict`/`from_dict` |
| 2.2 | 同上 | `build_label_record`（**纯函数**：payload + 传感器成功判定 + taxonomy → 记录）、`success_from_sensor`、`label_episode`、`LabelingBackend` 协议 + `FakeLabelingBackend` |
| 2.3 | 同上 | `EpisodeSample` / `EpisodeSource` 协议 / `label_dataset`（**逐条 flush 落盘** + `resume` 断点续跑 + `limit`） |
| 2.4 | 同上 | `write_lerobot_metadata`：episode 级 sidecar（task/intent/subtask），**不改 LeRobotDataset fields** |
| 2.5–2.6 | `src/label_audit.py`（新增） | `cohen_kappa`（`None`→`UNMAPPED` 独立类、`pe>=1` 退化处理）、`compare_labels` → `LabelAuditReport`（κ_intent/κ_success、混淆矩阵、per-subtask recall、分歧样本**按 VLM 置信度升序**排、`label_scope` 透传、taxonomy 版本不一致告警）、`write_audit_report` |

**关键设计决策**：
1. **两条流必须独立**：VLM 流的 intent 由确定性关键词规则映射、人工流由人独立判断；若映射也用 LLM，
   分歧无法归因（会混入映射自身的不确定性），κ 失去意义。
2. **逐条 flush 而非最后一次性写盘**：VLM 调用是外网请求、随时可能断，逐条落盘让"跑到一半失败"
   不丢已完成标注（也是 `resume` 的前提）——已用"第 3 条抛异常时前 2 条必须已落盘"的测试固化。
3. **`success` 只用传感器、无数据时为 `None`（不猜）**；且必须写明局限：真机演示数据集**没有负载通道**，
   只能用位置判据（弱于 `holds_object` 的「位置 OR 负载」）。
4. **必须报告 `label_scope`**：demo_v23 只有 2 条指令且都映射到 `pick` → **intent 维度在该数据集上
   没有信息量**（κ_intent 必然=1.0 的退化情形），这**不代表标注质量高**。审计报告默认要求显式声明
   适用边界，否则指标会被误读。

**验证**：`tests/test_vlm_labeler.py`（扩展）+ `tests/test_label_audit.py`（新增）共 **87 项**；
全量回归 **778 passed / 7 skipped / 0 failed**。κ 做了**手算校验**（完全一致=1、完全反向=−1、
chance=0、手算例 0.8、4 对里 3 对一致 = −1/3）。

**待办（2.7–2.10）**：`run_vlm_label.py` / `run_label_audit.py` 两个 CLI、`demo_v23` 的 `EpisodeSource`
（含 mp4 抽帧）、用 **Qwen3-VL** 跑自动标注、以及人工抽检工具（2.10，需用户配合）。
→ **已全部完成，见批次 21**（此后原文改为历史记录）。

### 2026-10-11 · 批次 21（P1 CLI 与人工抽检工具：2.7/2.8/2.9 前置 + Qwen3-VL 全量标注）

**已实现**：

| 子任务 | 文件 | 内容 |
|---|---|---|
| 2.7 | `src/run_vlm_label.py`（新增） | 标注 CLI：源名解析（复用 0.3 的 `VLM_<源>_*` 注册表）→ `VlmLabelingBackend` → `label_dataset`（逐条落盘/断点续跑）→ 旁挂元数据 → 人工复核页；`--dry-run` 用桩 chat 后端（仍走真实解析/校验路径）；`load_local_env` 读 `.env` |
| 2.2b | `src/vlm_labeler.py` | 新增 `VlmLabelingBackend`（注入式 `chat` 后端 + `image_loader`）。System Prompt **不给**数据集指令——否则 VLM 流的 intent 只是"把已知答案抄一遍"，κ 失去审计意义 |
| 2.9 前置 | `src/lerobot_label_source.py`（新增） | `LerobotEpisodeSource`：读 `meta/{info,episodes}.jsonl` 与 parquet 取 `gripper.pos`；**PyAV 解 AV1** 抽帧（demo_v23 的 AV1 用 OpenCV 解不动）、`sample_indices` 等间隔抽帧（单帧取末帧）并**落盘缓存**；`write_manifest` **合并式**写帧清单（断点续跑不丢历史集） |
| 2.8 | `src/run_label_audit.py`（新增） | 审计 CLI：`compare_labels` → `write_audit_report`；打印 κ / 混淆矩阵 / per-subtask recall / 警告 / `label_scope` |
| 2.5b | `src/label_audit.py` | 新增 `default_label_scope`：自动写明"intent 维度退化"与"success 仅位置判据"两条适用边界 |
| 2.10 前置 | `src/review_sheet.py`（新增） | 人工抽检页：`select_review_refs`（种子可复现的 20% 抽样）+ 自包含 HTML（内联 CSS/JS，仅相对路径引图）；人工只填三栏，导出**直接是 `LabelRecord` schema 的 jsonl**；**页面不显示模型答案**（避免锚定抬高 κ） |
| — | `src/real_perception.py` | `extract_json_object`：把围栏容错 JSON 解析提升为公开入口（供标注链路复用，不再复制粘贴） |

**Qwen3-VL 全量标注结果**（`runs/week38/labels/demo_v23`：68/68 集、544 帧、14 集抽检页）：

| 维度 | 分布 | 说明 |
|---|---|---|
| `intent` | **`pick` 68/68** | 与批次 20 预判一致 → 该数据集 intent 维度**没有信息量** |
| `subtask` | `grasp` 66 / `lift` 2 | 同样几乎退化为单一类 |
| `success` | `True` 68/68 | **传感器位置判据**（非 VLM 判断） |
| `confidence` | 0.95×65、0.90×3 | **几乎恒为高置信** |
| `task_text` | 逐集不同（"用夹爪夹起桌上的黄色螺丝刀"…） | 说明模型确实在读画面，且规则映射稳定落到 `pick` |

**必须写进结论的局限（`default_label_scope` 已自动写入审计报告）**：

1. **intent 与 subtask 两个维度都退化**：demo_v23 只有 2 条指令（都→`pick`），VLM 的 subtask 又 66/68 给同一类 → κ 在该数据集上**不可解释为标注质量**；A3 的真正检验须等 2.9 的多任务集（≥3 物体 × 2 指令 × 30 集）。
2. **success 只用位置判据**，且用的是**当前真机标定值**（baseline 4.1 / margin 1.4）；demo_v23 的 `gripper.pos` 量纲与当前标定**不同一**（实测该数据集张开 ≈1.5–4.0、闭合夹持 ≈7–10；而当前真机空载闭合 3.92）→ 位置判据在此数据集上只是启发式，**必须以人工抽检为准**。
3. **confidence 恒高（0.90–0.95）无区分度**：不能当作"标注可靠性"的过滤条件。

**验证**：新增 **37 项**测试（`test_review_sheet` 14、`test_lerobot_label_source` 12、`test_vlm_labeler` 后端 8、`test_label_audit.default_label_scope` 3），全量回归 **815 passed / 7 skipped / 0 failed**（`test_evaluation`/`test_executor`/`test_task_skills` 因本机缺 `ai2thor` 无法收集，与本批无关）。另外：真数据集 `--dry-run` 端到端跑通、合成人工流跑通审计 CLI（κ/混淆矩阵/分歧导出）、复核页经**浏览器实测**（14 卡 × 8 图全部加载、三栏进度联动、伪页脚固定）。

**当日缺陷修复（用户实填时发现，已修 + 已回归）**：复核页**只能填一张卡**——三组 radio 在 14 张卡上共用了同一组 `name`（`intent`/`subtask`/`success`），而 HTML 中**同名 radio 是全文档互斥**的，选第 2 集的 intent 会自动清掉第 1 集的。修复：radio 的 `name` 改为逐集唯一（`<episode_ref>:<字段>`），JS 取值的 CSS 选择器相应改为"结尾匹配" `input[name$=":intent"]`（不依赖 episode_ref 的转义）。补回归测试 `test_radio_groups_are_per_episode`（断言逐集唯一且无裸 `name`）。**教训**：首次浏览器验证只测了"单卡选择 → 进度 +1"，没测**跨卡**选择，故漏掉；表单类交付物的验证必须覆盖"多实例之间互不干扰"。修复后已用浏览器复测（填 3 张卡 → 前两张选择保持 → 进度 3/14 → 未填满时导出被正确拦截），回归 **816 passed**。

### 2026-10-11 · 批次 22（P1 子任务 2.10 人工抽检完成 → κ 结果）

**人工抽检**：14 集（68 的 20%，种子 38），用户经复核页填写并导出 `human.jsonl`。人工分布：
`intent` 全 `pick`（14/14）；`subtask` **lift 11 / grasp 2 / approach 1**；`success` **True 11 / False 3**；无备注。

**审计结果**（`runs/week38/labels/demo_v23/audit/`）：

| 指标 | 值 | 判读 |
|---|---|---|
| `κ_intent` | **1.000** | A3（≥0.6）**判 pass，但维度退化（仅 `pick` 一类）→ 不可解释为标注质量** |
| `κ_success` | **0.000** | po=11/14=0.786、pe=0.786 → **与随机猜测无异** |
| per-subtask recall（以人工为参照） | `grasp` **1.00** / `lift` **0.00** / `approach` **0.00** | 阶段维度大面积错 |
| 分歧样本 | **14 / 14** | 每条抽检至少有一项与 VLM 不一致 |

分歧构成：`subtask` 12 条（VLM 全判 `grasp`，人工 11 条 `lift` + 1 条 `approach`）+ `success` 3 条（VLM 全 `True`，人工 `False`），其中 `episode_000053` 两项同时分歧 → 12+3−1=14。

**核心结论（研究性，非流水账）**：**零样本 VLM 能识别"任务级语义"，但不能判断"阶段级进展"**。
证据：VLM 生成的 `task_text` 措辞逐集不同且全部正确描述"用夹爪夹起/拾取螺丝刀"（→ `intent` 完全一致）；
但 `subtask` **恒定输出 `grasp`**——把"已经抬起"也判成 `grasp`，而人工 11/14 认为已完成 `lift`。
即"在做什么"答对、"做到哪一步"答错。**故阶段标注不能靠零样本 VLM 直接产出**，需改为"
按帧输出阶段切片"或对 VLM 做阶段标注微调。这正对应设计 §10 的降级路径："VLM 标注噪声大 →
只用于语言标注与粗筛，成功判定以传感器为准；人工抽检 20%"。

**⚠️ 本次 κ_success 有口径缺陷，不能单独归因给传感器判据（我的设计问题，已记录）**：
`success_from_sensor` 判的是「**是否夹住**」（`is_holding` = 夹爪位置 > 5.5），而复核页问的是
「是否成功夹住**并提起**」（更严）。两侧口径不同 → 人工那 3 例 `False` 的依据恰是"没抬起来"，
故 `κ_success = 0` **至少部分由口径不一致造成**，若要归因传感器判据本身需重问（把措辞统一为"是否夹住"）。

**产物**：`runs/week38/labels/demo_v23/{labels.jsonl, human.jsonl, episode_metadata.jsonl, frames_manifest.json, frames/, review/review.html, audit/audit.json, audit/disagreements.jsonl}`

### 2026-10-11 · 批次 23（P1 两项修正：阶段 prompt 迭代 + success 口径对齐）

**任务 1 · 阶段标注 prompt 迭代（要求模型先输出 `stages` 序列，`subtask` 由代码取末段）**

动机：批次 22 发现 `subtask` 恒定输出 `grasp`，且与模型自己写的 `task_text`（"夹起桌上的螺丝刀"）自相矛盾 →
判断"整段取最靠后阶段"这一指令把"判断"和"取舍"都压给了模型。改为模型只输出**按时间排序的阶段序列**
`stages`，末段由 `subtask_from_payload` 派生（判断与取舍解耦）。

A/B（同一批 14 集人工真值，旧结果保留在 `demo_v23/`，新结果在 `demo_v23_stages/`；抽帧缓存复用，无重复解码）：

| 版本 | `subtask` 分布（68 集） | recall `grasp` | recall `lift` | recall `approach` | 分歧 |
|---|---|---|---|---|---|
| 旧（单标签 prompt） | grasp 66 / lift 2 | **1.00** | **0.00** | 0.00 | 14 / 14 |
| 新（`stages` 序列） | grasp 40 / lift 28 | 0.00 | **0.36** | 0.00 | **10 / 14** |

**结论（含负结果）**：**prompt 确有一份贡献，但不是主要瓶颈。**
- 有效的一面：`lift` recall **0.00 → 0.36**、分歧 **14 → 10**、分布从 66:2 变为 40:28（模型开始区分阶段），
  且模型 `note` 里开始显式写出推理链（"前两帧…approach；第3–4帧…grasp；后四帧…lift"）。
- 失效的一面：`grasp` recall 从 1.00 掉到 0.00（**过度纠正**，2 例人工 grasp 被改判 lift），`lift` 仍只 0.36。
- 残余误差**不是措辞问题**：模型为自己给出的 grasp 写了理由"已完全夹住**但尚未抬起**"——它是**看到了、
  但判断与人工相反**。故下一步应改善**信息与问题定义**而非再调措辞：① 抽帧方式（现为每相机 4 帧均匀采样，
  可能漏掉抬起瞬间）；② 把"是否抬起"拆成独立二分类问题；③ 对 VLM 做阶段标注微调。

**任务 2 · `success` 口径对齐 + 复核页预填**

- 批次 22 的口径缺陷已修：复核页问法由"是否成功夹住**并提起**"改为「**夹爪是否夹住物体**」，
  与 `success_from_sensor`（`is_holding`）判据一致；`success_from_sensor` docstring 也写明该口径。
- 新增 `--prefill <jsonl>`：用**人自己之前填过的答案**预选复核页（`demo_v23/review/review.html` 已预填 14 集
  × 3 栏 = 42 项），因此重问 `success` 时无需重看 112 张图。**只允许预填人工答案，不允许预填模型答案**（锚定）。
- 旧人工答案已备份为 `human_v1.jsonl`（用于 A/B 与追溯）。

**验证**：新增 **16 项**测试（`TestSubtaskFromPayload` 10、`TestStagesSchema` 3、`test_review_sheet` 3：
预填命中/未预选不带 `checked`/`success` 问法已对齐且旧问法不再出现），全量回归 **832 passed / 7 skipped / 0 failed**；
预填产物经代码校验（14 卡 × 42 个 `checked`、逐集取值正确、新问法命中且旧问法无残留）。

**待办**：用户在预填页上只重答 `success` 一栏 → 导出 `human.jsonl` → 重跑审计得到**口径干净**的 `κ_success`。

### 2026-10-11 · 批次 24（2.10 复问收口：κ_success + 位置判据零可分性取证）

用户在预填页上只重答 `success`（新口径：夹爪是否夹住物体）。人工分布由 True 11 / False 3 变为
**True 6 / False 8**；审计结果：`κ_intent` 1.000（退化）、**`κ_success` 0.000**、分歧 **13/14**、
per-subtask recall `grasp` 1.00 / `lift` 0.00 / `approach` 0.00。

**关键：`κ_success = 0` 不能只看 κ。** VLM（传感器位置判据）14/14 全判 `True` —— **一侧方差为零时 κ 恒为 0**
（pe 恒等于 po），它无法区分"略好于随机"与"完全无关"。故补做**可分性取证**（末 5 帧夹爪位置）：

| 人工 `success` | n | 末 5 帧均值的范围 |
|---|---|---|
| 夹住 | 6 | **8.09 – 9.97** |
| 没夹住 | 8 | **6.90 – 9.97** |

两组**完全重叠**；14/14 的尾值都 > 判据阈值 5.5 → 判据必然全判 `True`。
以人工为真值：**位置判据在该数据集上对"是否真的夹住物体"零可分性**，错误率 **8/14 = 57%**。

**排除"位置只是指令回显"的假设**：`observation.state` 的 gripper 列与 `action` 列**不相等**
（整段最大差 6.66–11.35）→ state 是真实测量而非命令回显；真实限制更可能是**位置信号饱和**
（无论是否夹到物体都顶到 6.9–10.0 同一区间）+ 该数据集**没有负载通道**。

**结论**：`success_from_sensor`（位置判据）在 demo_v23 上不可用，且**这不是口径问题**（口径已对齐）——
与 P0「负载信号」的立项动机完全一致：**要产出可靠的成功标注，数据集必须带 `Present_Load`**。

**新发现的下游缺口（高优先，需用户决策）**：`run_vla_collect.py` 只是 SOP 打印器，实际走
`lerobot-record` + `so101_follower`，其 features 只有六轴 `.pos` + 两路相机，**不含 `Present_Load`**
→ 若照现模板采集 2.9 多任务集，新数据同样无法做可靠 success 标注。建议方案（不破坏 LeRobotDataset
fields，符合设计原则）：采集时**旁挂**一个按帧对齐的 `load.jsonl`（`RealEngine.read_load` 已具备）。

### 2026-10-11 · 批次 25（2.9 前置落地：采集期**旁挂负载通道**）

动机（承接批次 24）：位置判据在 demo_v23 上零可分性 → 2.9 采集必须带负载通道；而现采集模板走
`lerobot-record`（features 只有六轴 `.pos` + 相机），照它采出来的数据集会原样复现这个坑。

| 项 | 文件 | 内容 |
|---|---|---|
| 旁挂格式 | `src/load_sidecar.py`（新增） | `LoadSidecarWriter`（**按集缓冲、成集落盘**）、`decode_loads`（复用 `decode_feetech_load`，逐关节容错）、`episode_loads`、`verify_alignment` + `SidecarAlignmentReport` |
| 录制挂钩 | `src/load_logging.py`（新增） | `LoadLoggingRobot`（每录一帧读 `Present_Load`；`observation_features`/`action_features` **原样透传**）+ `patch_record_pipeline`（只替换两个函数，退出还原） |
| CLI | `src/run_vla_collect_record.py`（新增） | `lerobot-record` 的**透明包装**：本脚本只认 `--sidecar` / `--verify`，**其余参数原样转交**；`--verify` 用旁挂里的 `positions` 与数据集 `observation.state` 逐帧比对 |
| SOP | `src/run_vla_collect.py`（改） | 模板由裸 `lerobot-record` 改为包装脚本，并**强制加 `--verify` 步骤**；录制纪律加两条 |

**关键设计点**：

1. **为什么不改数据集 fields**：features 是既有 ACT/SmolVLA 训练脚本的契约，加一列会全线失效。
   包装层把 features 原样透传，且**不往观测里多加键**（`build_dataset_frame` 只按 `dataset.features`
   取键、多余键会被静默忽略，但不加更干净）——负载只进旁挂文件。
2. **为什么必须替换 `record_loop`**：它同时用于"录某一集"（`dataset` 非空）与"集间复位"
   （`dataset` 为空），而复位段也会调 `get_observation`/`send_action`。不区分就会把复位段的负载
   混进帧序列 → 必须由它告知 wrapper 当前集号与是否在录制段。
3. **负载在 `send_action` 里采、而非 `get_observation`**：录制循环里 `send_action` **恰好每存一帧
   调用一次**，而"无 policy/teleop"的跳过分支不调用它 → 天然与数据集帧一一对应。
4. **按集缓冲、成集落盘**：坏集重录（←）是常规操作，逐帧写会在旁挂里留下被丢弃那一集的垃圾行；
   同集号再次开始 = 重录 → 丢弃旧缓冲。
5. **`positions` 字段是安全网**：旁挂里存下与 `observation.state` **同源**的位置值，事后可逐帧校验
   对齐；`--verify` 不通过则该份旁挂数据不可用（帧数不符/顺序错位/重录残留一次抓出）。

**验证**：新增 **32 项**测试 —— `test_load_sidecar`（14 项：方向位解码、越界逐关节容错、成集落盘、
重录丢弃、异常不落盘、追加续采、`episode_loads` 排序、对齐校验的帧数/位置/容差判定）；
`test_load_logging`（18 项：features 与观测键**不变**、属性委托、每帧一行、复位段不计、重录替换、
返回裁剪后动作、读负载失败不中断采集、无总线容错、patch 的集号跟踪 / kwargs 转发 / 原函数还原 /
收尾提交）。全量回归 **864 passed / 7 skipped / 0 failed**；CLI `--verify` 用合成数据集端到端实测
（一致 → exit 0；少一帧 → exit 1）。

### 2026-10-11 · 批次 26（P2 代码全量落地 + P0 额外观测通道接线）

**P2 4.2–4.7 全部写成（纯代码，等云 GPU / 多任务集才能跑）**：

| 子任务 | 文件 | 内容 |
|---|---|---|
| 4.2–4.4 | `src/vla_adapt.py`（新增） | `AdaptConfig`/`AdaptResult`、`build_adapt_config`、`apply_adaptation`（`zero_shot`/`head`/`full`，按参数名匹配解冻）、`run_adapt_experiment`（基线前后评测 → 保留率） |
| 4.5 | `src/run_vla_adapt.py`（新增） | 3 策略 × 2 数据量对照 CLI：`MAX_ADAPT_UNITS = 6` **硬封顶且超限报错（不静默截断）**；每单元落盘 `config.json`/`metrics.jsonl`/`adapt_result.json` |
| 4.6–4.7 | `src/vla_generalization.py`、`src/run_vla_generalization.py`（新增） | `GeneralizationSplit`（**构造期即做零泄漏校验**）、`build_split`（`combo`/`paraphrase`/`novel_object`）、`proportion_ci`（自写 bootstrap，**不复用** `wm_ope._bootstrap_ci`——后者对 ρ 特化）、`evaluate_generalization`（逐组合成功率 + 95% CI + gap + `scope_note`）、报告落盘 |

关键取舍（详见 design §13 偏差 65）：`apply_adaptation` 的 `head` 策略**若一个参数都没匹配到就报错**
（不许悄悄退化成"全训"或"全冻"）；保留率分母为 0 时返回 `None`（不是 0.0，避免被读成"完全遗忘"）；
`peak_mem_mb` 拿不到就记 `None`（不假装是 0）；默认训练路径**明确声明"lerobot 0.3.4 无按模块冻结入口，
冻结未必已生效"**，真实冻结需训练侧拿到 policy 后调 `apply_adaptation`。

**P0 1.3（几何量 → 策略输入特征）**：`real_geometry.py` 新增变体布局 —— `VARIANT_ORDER`、
`VARIANT_EXTRA_FEATURES`、`variant_feature_names`（**累积展开**，B0→0 列、B1→2、B2→4、B3→6）、
`feature_values`（`DepthEstimate`/负载 → 特征字典）、`build_feature_vector`（**定序 + 缺键即报错**）。
两条刻意的严格性：① 累积顺序只在**一处**定义 —— 列序一旦两侧不一致，数值仍然"有限"，指标只是变差，
事后几乎无法归因；② 不可用量**缺键报错**而不是填 0 —— 填 0 会把"没测到"伪装成"等于 0"。

**P0 1.5（策略额外观测通道）**：`visual_act.py` 增加**可选**的额外观测通道，**追加在 ENV 辅助向量尾部**
（13 → 13+n）：`VisualACTTrainingConfig.extra_features`（默认 `()`）+ `env_state_dim` 属性；
`load_visual_dataset` 按实际维度校验；`make_visual_act_config` 用实际维度注册 `input_features`；
`task_metadata.json` 落盘 `extra_features` 且 `load_visual_act_checkpoints` **回读后重建**
（不回读会用 13 维重建 config → 与权重/归一化 tensor 形状不匹配而加载失败）；
`VisualACTPolicy` 新增 `extra_features` + `extra_provider`，**带通道但没给 provider 直接报错**；
`run_visual_act.py` 新增 `--extra-features`。

三个刻意选择：① **复用 ENV 通道而不新增 feature key** —— 不动 LeRobot 的 `FeatureType`/归一化映射，
老 checkpoint 与老数据集零影响；② 默认 `()` → 维度仍是 13 → **B0 与历史行为逐位一致**，
既有基线可比；③ 该通道**不做归一化**（与既有 ENV 通道现状一致），尺度标准化由上游数据准备负责 ——
这条已写进 config 字段注释，避免后人误以为它被归一化了。

**验证**：新增 **119 项**测试 —— `test_vla_adapt.py` 33 项、`test_vla_generalization.py` 61 项、
`test_ablation_features.py` 25 项（变体列序/累积前缀/缺键报错/method 校验；ENV 维度默认 13 与
扩展、重复与空名拒绝、追加助手的顺序与维度/非有限校验、`extra_provider` 缺失即报错）；
全量回归 **983 passed / 7 skipped / 0 failed**（`compileall -q src` 亦通过）。

**尚未写**（下一批）：1.4 `run_handeye_calib.py`（设计标注为**条件启用**，且采集半边依赖尚不存在的 FK）、
1.6 `run_depth_ablation.py`、4.1 `run_vla_infer.py` 真机闭环。**其中 1.6 卡在一个需要用户决策的研究设计问题**
（B1/B3 的特征由谁产生，见 design §13 偏差 66）——代码可写，但不先定这个，跑出来的数字没有意义。

### 2026-10-11 · 批次 27（P0 特征生产落地：方案③；并发现 B0 定义与实现不符）

**用户决策：方案③** —— B1/B2 走 **oracle 特权量作上界**、B3 走 **检测/几何作真实对照**。据此新增
`src/ablation_features.py`（**纯函数**，不依赖仿真）：

| 变体 | 自身新增列 | 来源 | 备注 |
|---|---|---|---|
| B1 | `depth_known_size_m` / `_sigma_m` | oracle | 物体→夹爪距离；**sigma 记 0**（这正是"上界"的含义，与检测路径的 sigma 可区分） |
| B2 | `load_gripper_permille` / `contact_flag` | oracle | `is_grasped` 是真信号；仿真**无扭矩通道**，负载列是闭合度缩放的**单调代理**（未夹住即 0‰） |
| B3 | `depth_stereo_m` / `_sigma_m` | **detector** | 两视角像素中心 + 内参/相对位姿 → `real_geometry.triangulate` |

**关键防循环论证检查**：`variant_features("B3", oracle=...)` **缺 detected 直接报错** ——
用特权位姿反推三角化结果 ≡ 直接给答案，那"真做感知能拿多少"的对照就不存在了；测试另固化
"改动特权位姿**不得**改变 B3 的检测列"。

**同时发现一个必须澄清的规格问题（决定 1.6 怎么建通道）**：设计 §7.1 把 **B0 写成"仅 RGB"**，
但 `visual_act` 现在的输入是 **RGB + robot_state(5) + env_state(13)**，而 env_state **本身就包含**
`obj_pos / goal_pos / (tcp−obj) / (obj−goal) / is_grasped` —— **即特权几何量已经在 B0 里了**。
两种理解的代价完全不同：
- 若 **B0 = 现状**：B1/B2 的 oracle 列与 env_state 冗余 → **ΔGrasp 会接近 0、消融没有信息量**；
- 若 **B0 = 仅 RGB**（贴合设计与 week17"单目深度模糊"的诊断）：需给 `visual_act` 加"**关闭 state/env 通道**"
  的能力，B0 去掉、B1–B3 逐级加回（我实现的 1.5 额外通道对两种理解都适用）。

已请用户确认（design §13 偏差 67）。

**验证**：新增 **19 项**测试（`test_ablation_feature_production.py`：归一化闭合度端点与截断、输入校验、
B1 距离与零 sigma、B2 接触/负载代理与"未夹住即 0"、检测路径与 `real_geometry` **逐位一致**、
**三角化在合成几何上回代出 0.5 m**、B3 缺检测输入即报错、B3 混合来源且检测列不受特权位姿影响）；
全量回归 **1002 passed / 7 skipped / 0 failed**（`compileall -q src` 通过）。

### 2026-10-11 · 批次 28（P0 1.6 落地：B0–B3 消融编排 + ΔGrasp 归因表）

**口径（用户决策）：B0 = 现状**（`RGB + robot_state(5) + env_state(13)`），B1–B3 在其上追加
`ablation_features` 的额外列。新增 `src/depth_ablation.py`（**纯逻辑**）+ `src/run_depth_ablation.py`（CLI）：

- `build_plan`：按**规范顺序**展开变体与 ENV 维度（13/15/17/19），且**必须含 B0**（否则 ΔGrasp 没分母）；
- `read_env_state_dim` + `validate_dataset_dim`：**开训前的维度门禁**（只读 `meta/info.json`）——
  列数不符会让策略读到**错位**向量，指标只会"变差"而不报错，属最难归因的一类失败；
- `summarize_ablation`：ΔGrasp 以**百分点**（成功率差 ×100，与 A2 同口径）+ 闭合高度误差 + 失败归因占比，
  并对"存在非有限动作""归因占比之和偏离 1"给出告警；
- `run_ablation`：训练/评测**依赖注入**（本机无 GPU/仿真），编排顺序与"B0 跳过训练"可离线测；
- CLI：`--dry-run` 打印计划 + 口径说明；真跑前逐变体过维度门禁；报告落 `ablation.json` + `ablation.md`。

**⚠️ 该口径的固有局限（已写成代码常量 `REDUNDANCY_CAVEAT`，随 CLI 输出与报告一起打印）**：
`env_state` 已含 `obj_pos/(tcp−obj)/is_grasped`，而 B1 的 `depth_known_size_m` = ‖tcp−obj‖、
B2 = `is_grasped` + 夹爪开合度，**都是它们的确定性函数 → 信息增量为 0**。故
**B0/B1/B2 之间的差异在原则上只应落在噪声内；A2（ΔGrasp ≥ 15pt）在本口径下不可能由 B1/B2 达成**；
B3 的检测深度同样由物体真值位置决定，引入的是**感知误差**而非新信息。
**结论：现行口径下本消融只能作冗余性对照 / 噪声敏感性探针，不能论证"几何或深度信号值多少分"**——
要回答 A2 需改用 **B0 = 仅 RGB**（关闭 STATE/ENV 通道）。CLI 与报告都会原样带出这段说明，防"只看数字不看前提"。

**验证**：新增 **33 项**测试（`test_depth_ablation.py`：计划维度与规范顺序、必须含基线、`info.json`
维度读取与缺字段报错、维度门禁报文含期望/实际、ΔGrasp 百分点换算与 pass/fail/n-a 判定、口径说明恒定附带、
非有限动作与归因占比告警、报告落盘、`run_ablation` 的顺序/基线跳过训练/缺数据集/变体名不一致、
CLI 的 `--dataset` 解析与计划渲染）；`--dry-run` 实跑确认四变体的列与维度打印正确。
全量回归 **1035 passed / 7 skipped / 0 failed**（`compileall -q src` 通过）。

### 2026-10-11 · 批次 29（1.6 真跑前置：变体数据集的列追加）

新增 `src/ablation_dataset.py`（纯逻辑 + 落盘）+ `src/run_prepare_ablation.py`（CLI）：把基础 v3 数据集的
`observation.environment_state`（`fixed_size_list<float>[13]`）扩展成 `13+n`，额外列取
`ablation_features`（方案③：B1/B2 走 oracle 特权量）。**关键点：oracle 需要的量数据集里本来就存了**
（`observation.state` = `tcp_pos+gripper_qpos`、`environment_state` 含 `obj_pos`/`is_grasped`），
所以额外列是**对既有列做纯算术**——本机可算、可测，**不需要仿真**。夹爪开合范围也从数据本身取极值
（`closure_range`），不写死常数。**B3 直接拒绝并说明理由**（立体深度要两视角像素 + 内参/相对位姿，
数据集只有单相机通道；用真值位姿反推三角化属循环论证）。

**实跑发现并修掉一个真实的静默损坏 bug（正是维度门禁该抓的那类）**：初版把整棵树都硬链接，
于是 `meta/info.json` 与源数据集**共享 inode**，`write_text`（`O_TRUNC` 写同一 inode）**穿透链接改到了源数据集**；
三个变体还共用同一份元数据互相覆盖 → 三个 `info.json` 全变成最后一个的 17 维，**源数据集的 info.json 也被改成 17**。
修法：**硬链接只用于 `videos/`**（唯一体积大的部分），`meta/**` 一律复制，`data/*.parquet` 跳过由我方重写。
已修复源数据集元数据（info.shape → 13、stats 逐维数组 → 13）并补两条回归测试：
"准备两个变体后源 `info.json`/`stats.json` 必须逐字节不变"、"两个变体各拿到自己的维度"。

**实测（week15-cam，3257 帧）**：B0/B1/B2 生成成功 → 消融 CLI 的维度门禁**三个变体全部通过**
（B0=13 / B1=15 / B2=17），B1 新增列 `depth_known_size_m` 均值 0.063 m、`sigma` 列全 0（特权上界语义正确）；
生成后源数据集 `info.json` shape 仍为 13、`stats.json` 逐维仍为 13。视频走硬链接（零拷贝）。

**一个如实的验证边界**：`LeRobotDataset(...)` 读取**没能跑通**，因为 lerobot 0.3.4 会先向 HuggingFace
解析 `repo_id` 的 revision，而本机访问不到 huggingface.co。**控制实验证明这与列追加无关**——
对**原始** `week15-cam` 调同一个 API 得到完全相同的 `ConnectError`（既有环境条件）。
故验证覆盖到 parquet schema/数值、`info.json`、`stats.json`、parquet 内嵌 HF 元数据、以及消融的维度门禁；
"能被 LeRobotDataset 读出来"这一步需在有网/有缓存的环境里补验。

**验证**：新增 **31 项**测试（`test_ablation_dataset.py`：oracle 输入的列映射与短向量拒绝、闭合范围取极值
与退化拒绝、变体列数值与 B3 拒绝、三处元数据补丁（含深拷贝不改原对象）、stats 扩展与形状不符时保留原值、
**data parquet 必须跳过**、**准备两变体后源元数据逐字节不变**、**两变体各拿自己的维度**、B1 端到端
改造 parquet/info/stats/内嵌元数据、B0 保维、B3 拒绝、已存在输出拒绝、dry-run 不落盘、非数据集拒绝、
基础维度不符拒绝）；全量回归 **1066 passed / 7 skipped / 0 failed**（`compileall -q src` 通过）。

### 2026-10-11 · 批次 30（4.1 SmolVLA 真机闭环 + 1.4 手眼标定 CLI）

**4.1 `run_vla_infer.py` 真机闭环落地（替换 stub）**，分两层，为的是"真机代码也能离线测"：

- `run_vla_loop`：主循环本体，**依赖全部注入**（策略/prepost/成功判据/休眠）→ 用假策略+假引擎即可覆盖
  步数、动作关节定序、提前结束、每步 reset、维度校验、缺图暴露；
- `run_real`：只负责装配真实对象（`RealEngine` + `SmolVLAPolicy.from_pretrained`）再调用它。

关键判断：① 成功判据改用 **`holds_object`（位置 OR 负载，0.6 实测标定）**，**不用** `GRASP_CLOSED = 10.0`
——后者是早期宽阈值，实测夹持位只有 6.69，用它会把"夹住"判成"没夹住"；② **缺图必须显式暴露**
（`missing_images` 进返回值 + 警告）——SmolVLA 是视觉策略，喂空图会让动作莫名变差却不报错；
③ **归位失败即拒绝开跑**（不在采集初始姿态 = 分布外观测，结果无法解释），要强跑得 `--no-home`；
④ 收尾 `park()`、退出码 0/1（真机评测常放脚本里跑）。

配套引擎改动（`real_executor.py`）：`RealEngine` 新增 `load_act_policy=False`（VLA 自带策略，
不该顺带加载 ACT checkpoint 与 ACT 的 dataset_stats）+ 通用 `send_action`（未知名忽略、**全被忽略则报错**，
否则"发了动作却一个关节没动"会表现成"策略不收敛"）；`MockEngine.send_action` 记录动作供测试断言；
`RobotEngine` 协议同步。

**1.4 手眼标定 CLI 落地**：新增 `src/handeye_calib.py` + `src/run_handeye_calib.py` —— 绝对位姿 →
相对运动对（AX=XB 的 A/B 是**相对运动**，绝对位姿不满足该方程，这步是现场数据到求解器的桥）→
`solve_handeye` → 逐对残差 + **留一法** + 门禁（旋转 ≤2°、平移 ≤5 mm）。留一法的作用是把
"**数据里有离群点**"与"标定精度不够"分开：全量拟合的残差会被离群对自己拉小，留一法能把它暴露出来。
**如实声明**：本 CLI **不含采集** —— 取"机器人末端绝对位姿 + 相机绝对位姿"需要 **FK 或标记物检测**，
本仓库都没有（这正是设计把 1.4 标为"条件启用"的原因），故 `--pairs` 必须由外部产出；
`--self-test` 用合成真值 X 把"求解+残差+留一法+门禁"整条链路测通，保证外部数据一到就能直接算。

**自检当场抓到我自己 CLI 的一个设计错误**：`--min-poses` 默认 3 → 只生成 3 个位姿（2 组相对运动）
→ **留一法跑不起来、`--outlier-pair` 也注入不到**，等于"自检"漏掉了最关键的诊断路径。
改为 `--self-test-poses`（默认 10）并加下限校验后：干净数据留一法 max 0.000 mm、门禁通过、exit 0；
塞入第 3 对 50 mm 离群后离群被正确抓出（留一法 46.1 mm = 全量均值 8.1 mm 的 **5.7×** → 告警
"疑似该对是离群点" + 门禁未通过 + **exit 1**）。

**另核验了两个 P2 CLI 的 dry-run 实跑**：`run_vla_generalization.py` 三协议 split 均无泄漏
（combo 4/2、paraphrase 4/2 带"无改写模型"告警、novel_object 3/3）；`run_vla_adapt.py` 打出
5 单元矩阵（zero_shot 锚点 data=0 + head/full × 30/60）并逐单元写出 `config.json`。

**验证**：新增 **46 项**测试（`test_vla_infer.py` 22 项：判据四种组合、负载判据救薄件、
负载读取失败降级、主循环步数/关节定序/提前结束/reset/维度校验/缺图上报、dry-run 链路、
Mock 负载键名与真机一致；`test_handeye_calib.py` 24 项：变换求逆与正交性校验、相对运动构造、
干净数据反解真值、**离群对必须被留一法抓出**、紧门禁判失败、位姿过少时留一法优雅降级、
npz 读写与三类畸形输入报错、报告落盘）；全量回归 **1112 passed / 7 skipped / 0 failed**
（`compileall -q src` 通过）。

**尚未写**：B3 数据集需"重渲染第二视角"（sim 侧工作，需云）；1.4 的**采集半边**需 FK/标记物检测（外部条件）。

### 2026-10-11 · 批次 31（A 方案：ENV 通道口径开关 + 云端落地验证）

**决策（用户选 A）**：先让 **B0 成为真正的"无特权几何"口径**，消融才有信息量。

**`visual_act` 新增 `env_state_mode`**（默认 `"full"` = 历史行为，**老 checkpoint/老数据集零影响**）：

| 取值 | 策略侧 ENV | 用途 |
|---|---|---|
| `"full"` | 13 维特权 env_state + 额外列 | 历史行为（旧 B0） |
| `"extra_only"` | **只留额外列**（`extra_features=()` 时 = 彻底无 ENV） | 新 B0–B3 口径 |

四处关键设计：① **`dataset_env_state_dim`（数据集列数 = 13+n）与 `env_state_dim`（策略侧 = n）分离** ——
数据集不用动，同一个 15 列数据集既能喂 `full` 也能喂 `extra_only`；② `_select_env_state` 取**尾部 n 列**
（额外列本就追加在 13 维之后）；③ ENV 维 0 时**整块省掉** feature / 统计项 / 批次键（三处守卫），
避免给 ACT 与归一化器喂 0 维张量；④ `task_metadata.json` 落盘 `env_state_mode` 并由
`load_visual_act_checkpoints` **回读后重建**（不回读则 B0 会按 13 维重建、权重形状不符）；
`VisualACTPolicy.act` **同步 `extra_only`** —— 训练/推理口径不一致时形状不报错、结果全错，是最难查的一类。

**云端落地验证（A10 服务器）**：
- **任务 1 关闭**：变体数据集能被 `LeRobotDataset` 正常读出（B1 `env=(15,)` / B2 `(17,)` / 基线 `(13,)`）。
  **根因是缺 HF 缓存条目**（我此前用了一个不存在的 repo_id），**不是列追加有问题** —— 之前记的
  "验证边界"就此关闭；云端已建 `~/.cache/huggingface/lerobot/bot-agent/maniskill-week11` 软链。
- **1.5 接线在真数据上端到端成立**：`load_visual_dataset` + `make_visual_act_config` 对 B0/B1/B2 给出
  `env_state_feature` = **13 / 15 / 17**，图片键不变。
- **新口径实测**：`extra_only` 下 **B0 `policy_env_dim=0`、`act_env_feature=None`、批次里没有
  `environment_state`**；B1 = 2 维、B2 = 4 维 ✓（这正是让 ΔGrasp 有意义的前提）。

**验证**：本地全量回归 **1112 passed / 7 skipped / 0 failed**（默认 `full` 保持原行为）。

**遗留（纯环境/依赖，非代码）**：云端 venv 是 **CPU 版 torch**（`2.9.1+cpu`），训练需换 CUDA 版。
实测可用国内源：`mirror.sjtu.edu.cn/pytorch-wheels/cu126`（**15.8 MB/s**，含 `torch-2.9.1+cu126`）
+ **`pypi.nvidia.cn`**（nvidia-* 包，即 SJTU 页面指向的英伟达国内 PyPI 镜像）；官方
`download.pytorch.org` 只有 **175 kB/s**（cudnn 单个 707MB 就要 ~67 min）。安装已在后台推进。

### 2026-10-11 · 批次 32（2.9 采集清单生成器：物体 × 指令 → 可执行清单）

**背景**：用户有 3 个可夹取物体（螺丝刀 / 黑色长条 / 指甲刀），需生成**符合要求**的数据采集任务。

**新增 `src/vla_collect_plan.py`**（纯逻辑，不驱动机械臂）：把「物体 × 指令」固化成清单，落盘三件：

- `collect_tasks.json`：任务清单（每个 repo 的物体 / 指令 / 集数）
- `collect_combos.json`：`[[object, instruction], ...]`，**直接喂** `run_vla_generalization --combos-json`
- `collect_commands.ps1`：逐条录制命令 + 每条采后的 `--verify` 命令

**三条指令的配法（决定下游实验能不能跑）**：`pick_a`（夹起桌面上的{obj}）/ `pick_b`（把{obj}拿起来）是 **pick 的同义改写**（供 P2 paraphrase 协议）；`place_left`（把{obj}放到左边）提供**第二个意图**，否则意图只有一类、κ 是退化值。

**用户决策：砍掉 `pick_b`**（`pick_a` 与 `pick_b` 动作完全一样、没必要录两遍）→ 新增 `--instructions`（`select_instructions` 按声明顺序过滤，空选/未知 key 报错）+ `paraphrase_available()`，生成时若不齐**打印告警**并写进 `collect_tasks.json`。**代价（如实记录）**：P2 的 `paraphrase` 协议**没有留出措辞可用**（只剩 `combo` + `novel_object`），失去"换说法还认不认"的文本泛化证据。**顺带修掉连带 bug**：`write_plan` 原先硬编码 `DEFAULT_INSTRUCTIONS`，裁剪后清单会列出没录的指令 → 改为记录实际使用的几条（回归 `test_records_actual_instructions_not_all_defaults`）。

**关键修正（真实缺陷，非设计所列）**：初版让所有 repo **共用一份** `load_sidecar.jsonl`。但 `LoadSidecarWriter` 是**追加写**、`run_vla_collect_record.py --verify` 又把**整份**旁挂对**单个**数据集根目录逐帧比对 → **第 2 个 repo 起校验必然把前一个 repo 的集号混进来而失败**。改为 `sidecar_for(task)` **每 repo 一份**（`runs\week38\collect\sidecars\<repo 名>.jsonl`），并加回归 `test_different_repos_use_different_sidecar`。

**repo 命名**：ASCII slug（`demo_vla_01_pick_a`，序号区分物体、key 区分指令）——HF repo 名对非 ASCII 不可靠；中文物体名只出现在 `--dataset.single_task` 与清单注释里。**物体序号与指令 key 都不受裁剪影响**，故砍 `pick_b` 后 `demo_vla_01_pick_a` 的 repo 名**不变**（已录的 3 集仍属于它）。

**最终产出（3 物体 × 2 指令 `pick_a`/`place_left`，20 集/组合）**：

| repo | 物体 | 指令 |
|---|---|---|
| `wangbangyan/demo_vla_01_pick_a` / `_place_left` | 螺丝刀 | 夹起桌面上的螺丝刀 / 把螺丝刀放到左边 |
| `wangbangyan/demo_vla_02_pick_a` / `_place_left` | 黑色长条 | 夹起桌面上的黑色长条 / 把黑色长条放到左边 |
| `wangbangyan/demo_vla_03_pick_a` / `_place_left` | 指甲刀 | 夹起桌面上的指甲刀 / 把指甲刀放到左边 |

合计 **6 个 repo / 120 集**（产物在 `runs/week38/collect/`）。

**验证**：`src/tests/test_vla_collect_plan.py` **49 项**、`src/tests/test_load_logging.py` **22 项**（含 4 项相机超时重试回归）、`src/tests/test_camera_check.py` **6 项**；全量回归 **1175 passed / 7 skipped / 0 failed**（基线 1112 + 63）。

**相机取帧超时重试（真机实测新缺陷）**：录制中一路相机偶发 `TimeoutError: Timed out waiting for frame from camera OpenCVCamera(0) after 200 ms. Read thread alive: True.`（`OpenCVCamera(0)` = handeye）→ **整场崩掉、当前集丢失**。lerobot 的 `async_read(timeout_ms=200)` 是**签名默认值、配置里没有可调项**（`OpenCVCameraConfig` 只有 `color_mode/fps/height/index_or_path/rotation/warmup_s/width`）；30fps 下 200ms = 连丢 ~6 帧，"read thread alive" 说明是**两路相机抢同一个 USB 控制器**。修法：`LoadLoggingRobot.get_observation` 加**只对 `TimeoutError` 生效**的重试（3 次 × 200ms）——正常路径零行为变化、真故障立刻抛；并累计 `camera_timeouts` 在收尾汇总告警（**不靠重试掩盖相机退化**）。**连带确认**：该次崩溃发生在**复位段**（该集尚未 `save_episode()`）→ 数据集与旁挂都没有它、仍一致（⑧ 的修复真机第二次验证）。

**相机索引体检工具（新增 `src/run_camera_check.py`，批次 32 追加）**：`handeye=0` 在 16:06 复现同一失败（`OpenCVCamera(0) read failed (status=False)`，MSMF 报 `-2147024865` = `ERROR_DEVICE_NOT_CONNECTED`）。取证：Windows 侧 3 台相机**全部 Status OK**（`USB2.0 VGA UVC WebCam`×1 + `icspring camera`×2）、无残留进程占用；MSMF 下 **索引 0 打开成功但一帧读不出**，索引 1/2 正常，且**索引 2 的抓帧与已录 `fixed` 通道同景** → 确认 `fixed=2` 无误、就是 0 号槽位坏了。**新增只读体检工具**（不连机械臂、不出力矩、不做归位）：探 MSMF 0..4（可选 DSHOW 对照）→ 抓帧存图 → 按 `CAMERA_SLOTS` 给出"槽位是否可用"的结论，退出码 1 表示被阻塞。**顺带固化硬约束**：`CAMERA_SLOTS == {"fixed": 2, "handeye": 0}` 被单测钉住（索引漂移曾导致策略腕部通道拿到笔记本摄像头画面，且 **`--verify` 查不出画面错**）。**已录 43 集的两通道经抽帧确认无误**（handeye = 腕上视角能看见夹爪白手指、fixed = 俯拍整桌），不必重录。

**采集进度（2026-10-11 晚）**：`demo_vla_01_pick_a` **20/20** ✅、`demo_vla_01_place_left` **20/20** ✅、`demo_vla_02_pick_a` **3/20**（三者数据集与旁挂**逐集一致**）。**相机阻塞已解除**：腕上相机在 MSMF 里消失（索引 0 打开失败），**换一个 USB 口 + 重启后恢复**；已**逐张核图**确认 `MSMF 0 = 腕上 handeye`、`MSMF 2 = fixed 俯拍`、`MSMF 1 = 笔记本摄像头` → **原口径 `fixed=2 / handeye=0` 不变**，与已录 43 集一致，无需改索引/切后端/换相机。`demo_vla_02_pick_a` 重跑报 `FileExistsError` 是因为**已有 3 集**（非空壳）→ 正确姿势是 `--resume=true --dataset.num_episodes=17`，**不要删**。

**本事件的教训（值得写进 SOP）**：
1. **"能出图"≠"是那台相机"** —— 索引漂移会让通道静默错配，而 `--verify` 只比对关节位置、**查不出画面错**。所以**每场录制前先跑 `run_camera_check.py` 并打开图核对身份**，别只看"✅ 可用"。
2. **"不要改索引"这句话要说得更准**：数据集存的是**通道名**（`observation.images.handeye/.fixed`），不是索引号；真正必须固定的是"**同一台物理相机 → 同一通道**"。因此**核实过身份**的换索引是允许的（推理侧同口径即可），危险的是**盲改**。
3. **故障是渐变暴露的**：先是间歇性 200ms 取帧超时（已加重试兜住），再是索引 0 彻底读不出 —— 这类"先闪断后掉线"是 USB 接触/供电劣化的典型曲线，换口或换线比反复重启更对症。

**上传口径**：生成器默认 `push_to_hub=false`（**先采 2 集试跑时别联网**）；铺满后加 `--push-to-hub` 走**私有 Hub**（与既有 `demo_v23` 一致）。不静默改你既有工作流。

**解释器（实跑踩到的坑）**：生成的命令用 `& "D:\miniconda\envs\lerobot\python.exe" ...`，**不是裸 `python`** —— 系统 `python` 上没有 lerobot，用户实跑命中 `ModuleNotFoundError: No module named 'lerobot'`。已加 `test_uses_lerobot_interpreter_not_bare_python` 回归钉住。

**数据集目录（实跑踩到的坑）**：`LeRobotDataset.create` 用 `exist_ok=False`，**中断重跑必报 `FileExistsError`**（实测：舵机未上电那次失败留下的空壳目录，只有 `meta/info.json`、零 episode）。清单注释已写明判据：**只有 `meta/info.json` = 空壳可删；已有 episode 用续录、不要删**。**续录参数名已修正**：`resume` 是 `RecordConfig` 的**顶层**字段（`lerobot/record.py:173`），正确写法 `--resume=true`（初版注释误写 `--dataset.resume=true`）。已加 `test_warns_about_existing_dataset_dir` 回归（含 `assert "--dataset.resume" not in script`）。

**真机验证：3 集校验通过（修复生效）**：`--verify` 输出「集数 3，错位 0 集」，`episode_000000/000001/000002` 帧数 373/451/487 均与数据集一致、位置最大偏差 ~9e-7 → **旁挂提交点修复在真机数据上确认**（修复前最后一集必被丢掉）。

**`--dataset-root` 不留占位符（用户要求）**：新增 `LEROBOT_CACHE_ROOT` = `C:\Users\Administrator\.cache\huggingface\lerobot` + `dataset_root_for(task)`，按 lerobot 本地缓存约定 `<缓存根>\<repo_id>` 逐 repo 填好实际路径（如 `...\wangbangyan\demo_vla_01_pick_a`），9 条 `--verify` 全部就位。已加 `test_dataset_root_is_filled_per_repo` 回归。

**旁挂提交点改为 `dataset.save_episode()`（真机实测暴露的丢数据 bug，重要）**：原设计靠"下一集开始"提交旁挂，**收尾阶段崩溃时会丢最后一集**。实测：录满 2 集后 `robot.disconnect()` 给夹爪写 `Torque_Enable=0` 连续 6 次 `Overload error!` → 抛异常 → 调用方 `abort_episode()` 抹掉待提交缓冲，但**那一集已经 `save_episode()` 进数据集了** → `data/` 有 2 个 parquet，旁挂只剩 ep0 的 693 帧 → `--verify` 必失败、整份旁挂报废。修法：`patch_record_pipeline` 给数据集实例挂 `save_episode` 钩子 → `mark_episode_saved()` 在**同一事件**上提交旁挂；`abort_episode` 只对"尚未 save 的那一集"生效。**一般化教训：两个要保持一致的落盘流必须绑在同一事件上，不能靠"下一个事件推进上一个"。** 新增 4 项回归（含 `test_teardown_failure_keeps_saved_episodes`）。

**断开力矩（用户决策）**：每条录制命令显式带 `--robot.disable_torque_on_disconnect=false`（lerobot 默认 `true`）—— 退出时不再写 `Torque_Enable=0`，避开夹爪过载时的 `Overload error!`（就是上面那次收尾崩溃的触发点）。代价：脚本退出后舵机持续通电保持姿态。要恢复旧行为用 `--torque-off-on-disconnect` 重新生成。已加 2 项回归。

**真机侧实跑记录（2026-10-11）**：① 参数解析通过；② 首次失败于 COM6 舵机全部无应答（`found motor list: {}`）→ **根因是 follower 舵机未上电**，上电后总线正常；③ 第二次失败于 `FileExistsError`（上一条留下的空壳目录）→ 已删空壳、清单已补提示；④ `--dataset-root` 占位符已按用户要求替换为实际路径；⑤ 录制 2 集成功，但收尾 `robot.disconnect()` 夹爪 `Overload error!` → 暴露并修掉旁挂丢集 bug（见上）；⑥ 试录的 2 集与旁挂已按用户决策**删除**，`demo_vla_01_pick_a` 从零重录；⑦ 采集命令已按用户决策加 `--robot.disable_torque_on_disconnect=false`。**上机前记得先给夹爪断电重启复位过载态。**

**下一步（【真机】，需用户操作）**：按 `collect_commands.ps1` 逐条录制；**先采 2 集试跑 + `--verify` 通过**再铺满 20 集；物体位置须**随机覆盖**左/中/右、近/远。

### 2026-10-11 · 批次 33（P0 消融**评测链**落地 + A2 在本数据集不可评测的取证）

**背景**：批次 32 的消融只训练不评估（`run_ablation.sh` 产 checkpoint 就结束），拿不到 ΔGrasp —— A2 的唯一判据缺失。

**新增评测链**：`src/ablation_eval.py`（`sim_oracle_inputs` / `make_oracle_provider` / `closure_range_from_dataset` / `success_rate` / `build_variant_result` / `evaluation_episodes`）+ `src/run_ablation_eval.py`（CLI，含 `--dry-run`，不加载策略/不建 env）+ `src/tests/test_ablation_eval.py`（29 项）。特征一律**复用**训练侧公式（`ablation_features.oracle_features` + `real_geometry.variant_feature_names` 定列序），不自造一套。`visual_act.evaluate_visual_act` 补 `extra_provider` 透传（`load_visual_act_checkpoints` 早已支持）。

**实跑暴露并修掉 3 个真缺陷（本地 1200+ 项测试全绿、`--dry-run` 也退出 0，都覆盖不到）**：

1. **`NameError`（我的疏漏）**：把 `action_smoothing` 从 `evaluate_visual_act` 签名删掉后，调用处仍传它 → 一跑就崩。→ 新增 `test_evaluate_visual_act_wiring.py`（打桩**真正执行**该函数，而不是只测前置纯函数）。
2. **口径回读读了错误的文件**：`env_state_mode`/`extra_features` **不在 checkpoint 的 `config.json`**（那是 `ACTPolicy` 的配置），而在同级 `task_metadata.json` → 回读退化成默认 `"full"`，给 B0（`extra_only`+空列）多注册一个 13 维 ENV 输入，加载时报 `encoder_1d_feature_pos_embed` `[3,512]` vs `[2,512]`。→ 新增 `resolve_checkpoint_env_config`（`task_metadata.json` 权威、缺失时按 `input_features` 反推、**带额外列却无元数据即报错**）+ `env_state_mode=env_state_mode` 真正传进 config。
3. **对账 helper 只认 dict**：内存里 `ACTConfig.input_features` 是 `PolicyFeature` 对象（有 `.shape`、非 Mapping）→ 维度对账**静默返回 None**，把 bug 2 的可定位报错退化成晦涩的形状不符。→ `env_feature_dim` 两种形态都认 + 维度不一致时明确报错。

**评测结果（每变体 2 任务 × 种子）**：

| 变体 | 真实 oracle provider（5 种子/10 集）| 20 种子/40 集 | **零 provider**（与训练分布一致，10 种子/20 集）|
|---|---|---|---|
| B0（无 ENV）| 0.800 | **0.725** | — |
| B1（+2 列）| 0.300 | 0.350 | **0.850** |
| B2（+4 列）| 0.000 | 0.025 | **0.850** |

**决定性取证（两条，缺一不可）**：

- **零 provider 对照**：把 provider 换成"恒返回 0"（与训练分布一致）后，B1/B2 都回到 **0.850**，甚至略高于 B0 → ① **额外 ENV token 本身无害**（批次 28/31 担心的"多一个 1D token 的结构混淆"**不成立**）；② **之前的 −37.5/−70.0 是评测时喂了训练分布外的值造成的，那张表作废**。
- **分布并排比**：B1 的 `depth_known_size_m` 训练 0.063/0.055 vs 评测 0.034/0.050 → **同量级、分布内**；B2 的 `load_gripper_permille` 训练 **0/0** vs 评测 **−68.1/46.3**、`contact_flag` 训练 **0/0** vs 评测 **0.69/0.46** → **严重 OOD**。

**根因（数据层面，不是代码能绕过的）**：**week15-cam 里 `is_grasped` 从未为 1**（`meta/stats.json` 第 13 维 mean/std 全 0）→ 该数据集**没有抓取行为** → 训练时 `contact`/`load` 恒 0，而评测时策略真会夹住物体 → 这两列必然 OOD。**"接触/负载信号"（B2）在这份数据上不可能被公平评测。**

**B1 仍有一个未解释的下降**：它的两列都在分布内，却也从 0.85（喂 0）掉到 0.35（喂真值）→ **不能读成"深度无用"**，需单独排查（跑步数-成功率曲线，或 5k 步欠训/变化输入通道的干扰）。

**结论与处置（用户决策：暂停消融、记录结论）**：A2 在本数据集**不可评测**；B2 的数字作废、B1 的下降待查。云上 GPU 已无作业（可停机）。等 2.9 真机数据（**真实抓取**）齐备后再做 B2，或改从 sim 侧重采"带抓取行为"的 PickCube 数据。

**验证**：全量回归 **1213 passed / 7 skipped / 0 failed**（新增 `test_ablation_eval.py` 29 项 + `test_evaluate_visual_act_wiring.py` 9 项）。

### 2026-10-11 · 批次 34（2.9 采集完成核验 + A3/A4 机读汇总 + S13 数据卡/模型卡）

**2.9 采集完成并核验**：6 repo / **120 集 / 62318 帧**，`--verify` **6/6 通过**（数据集↔旁挂逐行对齐，位置最大偏差 9.26e-07；旁挂 `gripper.load` 无空值、`decode_errors` 全 0）。集数偏差：`02_pick_a` 19、`02_place_left` 21，其余各 20。

**A3/A4 机读汇总**（`src/vla_label_summary.py` + `src/run_vla_label_summary.py`，产物 `runs/week38/label_vla/summary/`）：

- **A3 = 零样本 VLM 在双意图数据上判别力为零（本批次核心负结果）**：`intent` 预测 `{pick:119, none:1}`，按 repo 语义应为 `{pick:59, place:61}` → **`place` 被预测 0 次**，κ **0.0080**、退化。置信度整体 mean 0.934（min 0.80 / max 0.95）——**高置信地全错**。
  → **修正批次 26 的结论**：当时"VLM 能做任务级语义（task_text 逐集正确 ⇒ intent 全对）"是**单一意图数据造成的假象**（全填 `pick` 当然对）。**换成双意图，该能力即失效。**
- **A4 成功标注**：整体 `{true:58, false:62}`；全 True 仅 `01_pick_a`、`03_pick_a`，全 False 仅 `03_place_left` —— 与 repo 语义高度同构，**疑似"照抄任务类型"而非独立判断**。`subtask` 分布 `{grasp:56, lift:63, approach:1}`（并非恒为 grasp/lift）。
- **A4 的"口径对齐"未完成并已如实标注**：仓库内只有**逐帧**判据（`holds_object`/`detect_contact`/`holding_and_lifted`），**没有"逐集成功"函数**；且 `labels.jsonl` 的 `success` 本身来自 `vlm_labeler.success_from_sensor`（**纯位置**），自比无意义 → 需新增"逐集聚合"（把逐帧的"位置 OR 负载"判据聚成集级成功）才能与位置判据对照。**不得临时拍判据充数。**

**S13 交付物（部分）**：新增 [dataset_card.md](file:///d:/FittenProject/bot_agent/docs/week38/dataset_card.md)（双相机索引已核图确认、旁挂通道性质、校验状态、已知局限）与 [model_card.md](file:///d:/FittenProject/bot_agent/docs/week38/model_card.md)（三变体真实 `validation_loss` 首末、两份评测报告数字、**为什么这些数字不能作为 A2 证据**、明确"不可部署"）。演示视频仍未做。

**验证**：全量回归 **1242 passed / 7 skipped / 0 failed**（新增 `test_vla_label_summary.py` 29 项）。

### 2026-10-11 · 批次 35（A3/A4 人工抽检 + 审计闭环：两个结论都被双意图真机数据证伪）

**人工抽检**：6 repo 各 4 集（共 24 集）已由用户填完导出 `human.jsonl`；**编号与 repo 的对应用各文件里的 `task_text` 逐条核对**（不是靠顺序猜），1–6 与 `01_pick_a / 01_place_left / 02_pick_a / 02_place_left / 03_pick_a / 03_place_left` 完全一致。

**A3（意图）—— 零样本 VLM 在真机双意图数据上不可用**：

| repo | κ_intent | 说明 |
|---|---|---|
| 3 个 `pick_a` | 1.000 | **但退化**（`intent` 仅 `pick` 一类，κ 不可解释为标注质量）|
| 3 个 `place_left` | **0.000** | 混淆矩阵：VLM 把 **4/4** 的 `place` 全预测成 `pick`；**`place` 类召回 0.00（12/12 全错）**|

→ 上一轮"VLM 能做任务级语义"**在双意图数据上被彻底证伪**：`pick` 侧的高 κ 是**单类别退化**造成的假象。**A3 判定：fail**（三个 `place_left` repo 全部 `κ_intent=0.000`）。

**A4（成功）—— 位置判据在真机数据上同样不可靠**：仅 `01_pick_a`(1.000)、`03_pick_a`(1.000) 通过，其余 4 个 repo **κ_success=0.000**；三个 `place_left` repo **人工判 `True`、自动判 `False`**（4/4 分歧）→ `success_from_sensor` 是**纯位置判据**，在"放到左边"任务上系统性偏保守；**总体一致率 13/24 = 54.2%**（批次 36 复核后更正；此前记的 5/24 是错的——那是我从"分歧条数"倒推所致），与 demo_v23 的取证（位置判据零可分性、错误率 57%）**在真机数据上复现**。`per-subtask recall`：`lift` 0.75/0.00/0.50（不稳）、`place` **0.00**（三 repo 全 0）。**注意本次审计尚未用上旁挂负载通道**（`success` 仍出自纯位置的 `success_from_sensor`）→ 这正是批次 25 引入旁挂要解决的缺口。

**待补（我欠的一步）**：把逐帧"位置 OR 负载"判据**聚合成集级成功**，再与纯位置判据对照，才能给出 A4 的最终口径结论（仓库目前只有逐帧判据、无逐集函数）。

### 2026-10-11 · 批次 36（逐集成功聚合落地：A4 口径收口）

**新增** `src/episode_success.py`（`assess_episode` 三口径纯函数 + 任务感知聚合）+ `src/run_episode_success.py`（CLI，产物 `runs/week38/episode_success/`）+ `src/tests/test_episode_success.py`（43 项）；**全部复用既有原语**（`success_from_sensor`/`is_holding`/`holds_object`/`is_lifted`/`holding_and_lifted`/`read_load_sidecar`/`episode_loads`/`cohen_kappa`），**未新增任何阈值**。

**① 旁挂负载通道首次真被用上 —— 确认有信号**：逐帧 `|gripper.load|` 各 repo min 0 / **median 79–83** / max 495–500‰；超 150‰ 阈值的帧占 **12.7%–14.8%**；**每集** `max|load|` 均 >150‰。**但集末 `|load|` 仅 ~50–100‰（<150）** → 在"只看集末"的口径下，**负载永远改不了结论**。

**② 口径对照（vs 人工 24 集）**：

| 口径 | 一致率 | κ |
|---|---|---|
| `position_only`（现状）| **13/24 = 54.2%** | 0.000 |
| `position_or_load` | **13/24 = 54.2%**（与上**逐集完全相同**）| 0.000 |
| `task_aware`（任务感知）| **19/24 = 79.2%** | 0.000 |

- **`position_or_load ≡ position_only`（每集都相同）**：因为现状判据只看**集末 5 帧**，而集末早已松手 → 负载也低。**结论：负载的价值在"过程中"（判定夹持发生在哪一帧），不在"集末"** —— 这正是"位置 OR 负载"当初被误解的地方。
- **`task_aware` 把一致率从 54.2% 提到 79.2%**，逐 repo：`01_pick` 4/4→3/4、`01_place` 2/4→2/4、`02_pick` 3/4→3/4、**`02_place` 0/4→4/4**、`03_pick` 4/4→3/4、**`03_place` 0/4→4/4**。pick 侧略降是**task_aware 更严**（要求出现抬起），属预期。
- **κ 全 0 不可解释**：人工 24 集**全部 `success=True`** → `pe=1` → 退化。**这是抽检设计的缺陷**（未抽到失败集），A4 只能报一致率。已写入报告。

**③ 根因取证（我此前的假设成立、但表述需修正）**：`success_from_sensor` = `any(is_holding(v) for v in gripper_pos[-5:])` —— 纯位置、**看集末 5 帧**、判"是否夹住"，不看松手/抬起。place 的终态是**松手**，故被系统性判 False（三个 place repo 集末尾 5 帧 `is_holding` 分别 4/20、2/21、0/20）。**但并非"place 一律 False"**（`01_place_left` 有 2 集末仍闭合 → 自动判 True）；`02_pick_a` 的漏判则是**薄件位置阈值**（`gripper.pos` 5.16–5.37 恰低于阈值 5.5），与 place 无关。

**④ 一个必须记进数据卡的 schema 修正**：这批真机数据的 `observation.state` **不是 tcp(3)+gripper(2)，而是 6 维关节位置**（`info.json` 的 `names` 为 shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll/gripper`.pos`，6 repo 一致；旁挂 `positions` 同为 6 轴）。因此**按名字取通道**（`gripper.pos` 判夹住、`wrist_flex.pos` 判抬起）才对，写死下标 `[3:5]` 会取到 wrist_flex/wrist_roll。→ 已在实现中按名字取，并需回填数据卡。

**⑤ 仍不可判**：`left_verified=False` —— 数据**无物体位置跟踪**，"物体是否被放到左边"**无法自动判定**（任务感知口径判的是"拿起后松手"，不含"左"），**不得用夹爪位置冒充物体位置**。place 口径还有一处歧义：以**末帧**判"已松手"，`01_place_left` ep003/ep019 集末夹爪仍闭合（判 False 而人工 True），可能确未松手、也可能末帧取帧偏差，**需人工复核**。

**验证**：全量回归 **1285 passed / 7 skipped / 0 failed**（新增 `test_episode_success.py` 43 项）。

**遗留与下一步**

- ~~0.2 真机采集（**【真机】**）~~ ✅ **已完成**：CP0 通过，负载信号可分辨（Δ=151‰），B2 可执行
- ~~0.3 2D 定位源选型（**【代码】**+**【真机】**）~~ ✅ **已完成**：选定 **`qwen3-vl-plus`**（6 帧平均中心误差 **1.31 px**、尺寸误差 ±1.5 px、命中率 **6/6**）；`gui-plus` 次选（3.37 px / 6-6）；`deepseek-flash` 不稳定（22.81 px / 3-6）不采用。工具链：`run_perception_capture.py`（抓帧）+ `run_perception_locate_eval.py`（多源评测 + `--gt` 误差）+ `run_perception_gt_label.py`（真值标注）
- ~~0.4 意图 taxonomy（**【代码】**）~~ ✅ **已完成**（`vlm_labeler.py` taxonomy + 映射，待真实语料验证局限）
- ~~0.6 空闭基准标定（**【代码】**+**【真机】**）~~ ✅ **已完成（门禁通过）**：实测空载闭合 **3.92 ± 0.04**（跨运行/跨姿态极差均 0.047，采样噪声 0），建议 `empty_grip_baseline=4.1` + `margin=1.4`（阈值 5.5）、判据改「位置 OR 负载」。**仅测量与给建议，真机判定行为未改动**
- ~~待决策（由 0.6 引出）：是否把 `empty_grip_baseline` 由 1.5 改为实测 4.1、`margin` 改为 1.4~~ ✅ **已决策并落地（批次 10）**：baseline=4.1、margin=1.4、「位置 OR 负载」判据已写入 `real_observation` / `real_verifier` / `real_executor` / CLI 默认值
- ~~待真机复验（由批次 10 引出）~~ ✅ **已完成（批次 11 A）**：咬空/夹住两个场景均判对
- ~~**真机阻塞（批次 11 B）**：`shoulder_lift` 卡在 −100.00、两个方向均无响应~~ ✅ **已解除**：用户现场处置后复测恢复（Δ −4.59 / +4.67 双向正常），端到端实抓已跑通（见批次 12）
- ~~运行结束后自动回到起始位~~ ✅ **已完成（批次 13）**：新增 `RealSkillExecutor.park()` + 两个 CLI 的 `finally` 收尾复位；真机验收"运行结束后位姿 vs 采集初始姿态"最大偏差 **1.82**（容差 3.0）
- **待决策（批次 14）**：① **fixed 相机视角与训练时已不一致**（同姿态帧 fixed r=−0.109）——是"把相机调回与历史帧一致"还是"相机固定后重新采集一小批做微调"？② 是否把 `disable_torque_on_disconnect` 改为 `False`（断开后保持力矩、不下垂，代价是舵机持续通电）
- ~~**真机阻塞（批次 15）**：`shoulder_lift`（id 2）力矩不足/效率衰减~~ ✅ **已排除（误判已更正）**：真因是**手臂塌下时夹爪压在桌面上**，外部阻力伪装成"舵机能力上限"。用户把其他关节调开后，同一舵机 `-99.83 → 45.86` 全程正常、电流/负载曲线健康。**运行前须确认手臂未压在桌面/障碍上**
- **ACT 策略抓取能力不足（批次 16 公平测试确认）**：相机映射修正后仍 `grasp_empty`，历史（9/5）同现象 → 需"重新采集 + 微调"（P1/P2），**已按用户决策暂时跳过真机抓取**
- **待决策（批次 17）**：`GRIP_LOAD_THRESHOLD=150‰` 偏保守（薄件 ~100‰ 不触发，靠位置判据兜）——是否下调需等更多薄件样本，避免引入误判
- ~~**P1 待人工执行（2.10，卡住 A3 结论）**~~ ✅ **已完成（批次 22）**：用户填 14 集并导出 `human.jsonl` → `κ_intent` **1.000（A3 判 pass 但维度退化）**、`κ_success` **0.000**、per-subtask recall `lift`/`approach` **均 0.00**、分歧 **14/14**
- **P1 待真机（2.9）**：多任务集采集——**清单已由 `src/vla_collect_plan.py` 生成（批次 32）**：3 物体（螺丝刀 / 黑色长条 / 指甲刀）× 3 指令（`pick_a`/`pick_b`/`place_left`）× 20 集 = **9 repo / 180 集**，命令见 `runs/week38/collect/collect_commands.ps1`（**每 repo 一份旁挂** + 采后 `--verify`）。**A3/A4 的实质结论依赖它**，也是 P2 泛化实验的数据前提
- **P1 结论级缺陷（批次 22→24）**：
  ① ~~`success` 口径不统一~~ ✅ **已修（批次 23）且已复问（批次 24）**——口径对齐后人工 True 6 / False 8，`κ_success` 仍 **0.000**，但**根因已定性为"判据无信号"而非口径**：末 5 帧夹爪位置在人工"夹住/没夹住"两组**完全重叠**（8.09–9.97 vs 6.90–9.97），判据命中 6/14、错误率 **57%**；并排除"位置=指令回显"（`state` ≠ `action`，最大差 6.66–11.35）。注意**一侧方差为零时 κ 恒为 0**，故必须辅以可分性指标，不能只看 κ；
  ② **`subtask` 阶段判断仍不可用**——prompt 改为 `stages` 序列后人工 `lift` recall **0.00 → 0.36**、分歧 14→10，但 `grasp` recall 由 1.00 掉到 0.00（过度纠正）；模型自述"已夹住**但尚未抬起**"说明是**判断分歧**而非没看到 → 下一步改**抽帧方式**（每相机 4 帧可能漏掉抬起瞬间）、把"是否抬起"**拆成独立二分类**、或微调 VLM；**不再是 prompt 措辞问题**；
  ③ `confidence` 恒高（0.90–0.95）不可用于过滤
- ~~**新增前置阻塞（批次 24）：采集链路不含 `Present_Load`**~~ ✅ **已落地（批次 25）**：新增 `load_sidecar.py`（旁挂格式 + 逐帧对齐校验）、`load_logging.py`（录制挂钩）、`run_vla_collect_record.py`（`lerobot-record` 的透明包装 + `--verify`），并把 `run_vla_collect.py` 的模板改为包装脚本、强制加校验步骤。**不改 LeRobotDataset fields**（避免破坏既有 ACT/SmolVLA 脚本）
- ~~1.3 特征构造 + 1.5 策略额外输入通道~~ ✅ **已完成（批次 26）**：`real_geometry` 变体布局（累积列序 + 缺键即报错）+ `visual_act` 额外观测通道（**复用 ENV 向量 13→13+n**，默认 `()` 使 B0 与历史逐位一致，checkpoint 回读 `extra_features`）
- ~~**待用户决策：B1/B3 的特征由谁产生**~~ ✅ **已决策（方案③，批次 27）**：B1/B2 用 oracle 特权量作**上界**、B3 用检测/几何作**真实对照**；`ablation_features.py` 已落地（含"B3 缺检测输入即报错"的防循环论证检查）。**注意**：B1/B2 的 ΔGrasp 是**上界**、不可解释为可部署能力；B2 的负载列是代理（仿真无扭矩通道）
- **新的待澄清（阻塞 1.6 建通道，design §13 偏差 67）**：设计把 **B0 写成"仅 RGB"**，但 `visual_act` 现在的输入含 `env_state`（**已含 obj_pos/(tcp−obj)/is_grasped 等特权几何**）→ 若 B0 = 现状则 B1/B2 与它冗余、ΔGrasp 趋 0；若 B0 = 仅 RGB 则需给 `visual_act` 加"**关闭 state/env 通道**"的开关。**两种理解要做的事完全不同**
- **P0/P2 尚未写**：1.4 `run_handeye_calib.py`（设计标为**条件启用**，且采集半边依赖尚不存在的 FK）、1.6 `run_depth_ablation.py`（等上一条决策）、4.1 `run_vla_infer.py` 真机闭环（替换 stub；需 `RealEngine` 补通用 `send_action`）
- `ai2thor` 相关 3 个测试模块需在基础环境执行（当前 `lerobot` 环境未装 `ai2thor`）

### 2026-10-11 · 批次 37（S12 落地：数据 v2.1→v3.0 + 多任务 merge + 训练入口修正 + 20k 微调启动）

**目标**：把 2.9 采到的 120 集真机数据真正喂进 SmolVLA 微调（阶段 F S12）。

**冒烟先行的三个决定性发现（不靠推理假设）**：

1. **0.6.1 的"多数据集"是装饰性注释、不是功能**。`DatasetConfig.repo_id` 注释写 "You may
   provide a list of datasets here"，但 `datasets/factory.make_dataset` 对非 `str` **直接
   `raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")`**（0.3.4 同样如此）→
   **多 repo 无法直接训练**，必须先 merge 成一个 repo。**这推翻了我此前"0.6.1 支持多数据集"
   的推断**（当时依据是注释 + `multi_dataset.py` 存在，未读 `make_dataset` 分支）。
2. **数据版本错配**：本机采集环境是 **lerobot 0.3.4**（`C:\Users\Administrator\Downloads\lerobot`），
   产出 **v2.1** 格式；云端是 **0.6.1**（只吃 **v3.0**）→ 加载即
   `BackwardCompatibilityError`。需先 `lerobot.scripts.convert_dataset_v21_to_v30`。
3. **相机名不符**：smolvla_base 的 `input_features` 是 `camera1/2/3`，本仓是 `fixed/handeye` →
   `validate_visual_features_consistency` 报 Feature mismatch；必须用 `--rename_map`。

**处置（云端 121.43.136.28）**：

| 步骤 | 命令 | 结果 |
|---|---|---|
| ① 数据升版 | `convert_dataset_v21_to_v30 --repo-id wangbangyan/<r> --root <root> --push-to-hub=false` × 6 | 6/6 成功，**120 集 / 62318 帧逐 repo 无损**（旧版原地备份为 `<root>_old`，核对后删除） |
| ② 合并多任务 | `lerobot-edit-dataset --new_repo_id wangbangyan/demo_vla_all --operation.type merge --operation.repo_ids "['…01_pick_a', …]"` | 产出 **`demo_vla_all`：120 集 / 62318 帧 / 6 任务**（3 物体 × pick/place），`robot_type=so101_follower`，fps 30 |
| ③ 10 步冒烟 | `lerobot.scripts.lerobot_train --policy.path=lerobot/smolvla_base --dataset.repo_id=wangbangyan/demo_vla_all --rename_map=… --steps=10` | **exit 0**，`dataset.num_frames=62318`、`num_learnable_params≈1.00e8 / total 4.50e8`、checkpoint 落盘 |
| ④ 正式微调 | 同 ③ 但 `--steps=20000 --batch_size=8` | 启动（PID 8146，nohup），稳定 **~4.95 step/s**，ETA ≈ 67 min，显存 3.4/24 GB |

**`src/run_vla_train.py` 修正（对 0.6.1 原本是坏的）**：入口 `lerobot.scripts.train`（**不存在**）→
`lerobot.scripts.lerobot_train`；`--policy.type=smolvla` → **`--policy.path=lerobot/smolvla_base`**；
删顶层 `--device`（Accelerate 自动探测）与 `--eval_freq`（改用 `--eval_steps`，默认 0=关），
`--device` 改映射为 `--policy.device`；新增 `--rename-map`（默认 `fixed→camera1 / handeye→camera2`）；
`--dataset-repo` 默认改为 `wangbangyan/demo_vla_all`；明确注释"多任务需先 merge"。同步修
`src/vla_adapt.py::_build_smolvla_train_command`，并让它**从 `run_vla_train` 取同一套常量**（避免两处漂移）。

**验证**：新增 `src/tests/test_run_vla_train.py`（12 项：入口名、`--policy.path`、无遗留
`--device`/`--eval_freq`、默认 rename_map、空 rename_map 省略、单 repo 非列表、`--dataset.root`
可选、resume、默认值、vla_adapt 共用常量）；全量回归 **1296 passed / 7 skipped / 0 failed**
（`ai2thor` 3 模块因该环境未装而排除，与本次改动无关）。

**遗留与下一步**

- **20k 微调进行中**（ETA ≈ 67 min）→ 完成后按 P2 4.1 真机闭环评测
- **数据卡需回填两条**（批次 36 已记）：`observation.state` 是 6 维关节位置（非 tcp+gripper）、`left_verified=False`
- **部署侧版本待决策**：训练用 0.6.1，而机器人侧运行 0.3.4 → 微调 checkpoint 能否被 0.3.4 加载需实测（否则机器人侧也要升 0.6.1）
- 仓库根 `lerobot_v21_src.tar.gz` 疑似采集环境的 lerobot 源码快照，用途待确认

### 2026-10-11 · 批次 38（S12 收口：微调 checkpoint 在机器人侧 0.3.4 上跑通）

**目标**：验证 P2 4.1 真机闭环的前提——0.6.1 训出的 checkpoint 能否在机器人侧 lerobot **0.3.4** 上加载并出动作。

**三道门禁（逐层取证，不靠推测）**：

| 门禁 | 做法 | 结果 |
|---|---|---|
| ① config | 用 0.3.4 的 `PreTrainedConfig.from_pretrained` 解析 checkpoint 的 config.json | ❌ `DecodingError`：6 个 0.6.x 专用字段（`use_peft`/`pretrained_path`/`pretrained_revision`/`rtc_config`/`compile_model`/`compile_mode`）；**剪掉后 ✅ 解析成功**，`input_features` 与训练时逐项一致 |
| ② 权重结构 | 离线读 safetensors 头（500 张量：VLM 345 + lm_expert 145 + action/state 头） | ✅ **无任何 `rtc`/`compile` 键** → 那 6 个字段纯属"保存时多打的默认值"；模块名与形状与 0.3.4 源码（`normalize.py`/`modeling_smolvla.py`）逐项对得上 |
| ③ 真正加载 | 在新建环境里 `SmolVLAPolicy.from_pretrained(...)` | ✅ 首次即 `LOAD OK`；唯一差异是 **6 个归一化 buffer**（0.6.1 把统计量移到了 pre/postprocessor 的 safetensors） |

**迁移（三处差异，产出可自包含的 0.3.4 checkpoint）**：

1. **剪 config 的 6 个 0.6.x 字段**；
2. **相机键改回**：训练侧用 `--rename_map` 把 `fixed/handeye` → `camera1/camera2`，而推理侧 `vla_prepost.smolvla_prepare` 用的是 `fixed/handeye` → 把 config 特征键映射回来（VISUAL 走 IDENTITY 归一化，不涉及权重；顺序 `camera1→fixed` 保持不变），并**删掉从未使用的 `camera3`**（训练时该路本就缺失，`empty_cameras=0` 不补零图 → 行为等价）；
3. **回填归一化统计**：0.3.4 的 `Normalize` 把 mean/std 当 buffer 存在 policy `state_dict` 里（缺省 `inf`，`forward` 内有 `assert not isinf`）→ 从 0.6.1 的 `*_processor.safetensors` 取 `observation.state` / `action` 的 mean/std 回填。**回填后缺键 6 → 0**。

**环境**：本地原 `lerobot` conda 环境是**采集专用**（CPU torch、**无 transformers**），且 0.3.4 要 `transformers<4.52`、云端 0.6.1 要 `transformers 5.x`（互斥）→ **新建独立环境 `smolvla_infer`**（Python 3.10，`lerobot[smolvla,feetech]` 0.3.4 + `transformers 4.51.3` + `accelerate` + CPU torch），**采集环境零改动**。SmolVLM2-500M 的**非权重文件**（config/tokenizer，13 个）用 `curl -L` 从 hf-mirror 拉到 `runs/week38/models/SmolVLM2-500M-Video-Instruct/`（`huggingface_hub` 对 hf-mirror 的 `307` 重定向处理不兼容，`curl` 可以）；权重本就在 checkpoint 里（`load_vlm_weights=false`）。

**产物**：`runs/week38/smolvla_vla/checkpoints/020000/pretrained_model_lr034/`（自包含，0.3.4 直接 `from_pretrained` 即可）。

**验证**：从迁移目录重新加载 57.3 s，`image_features = [fixed, handeye]`、batch 键齐；**真实前向出动作**（有限值、非全零）。

**遗留与下一步**

- ⚠️ **CPU 推理 10.2 s/步**（450M 模型 + 2 路 512² 图像）：300 步一集 ≈ 50 分钟。安全上没问题（`max_relative_target` 限幅），但**效率上不实用** → 需决策：本机 GPU / 云端做策略服务（async inference）/ 降低分辨率或推理频率
- ⚠️ **云端主机指纹已变**（`REMOTE HOST IDENTIFICATION HAS CHANGED`）→ `121.43.136.28` 很可能已被回收/重建，如需再用需先确认身份
- 真机闭环尚需在 `smolvla_infer` 环境里跑 `src/run_vla_infer.py`（当前 env 与脚本都在，缺的只是"上机"决策）

### 2026-10-11 · 批次 39（P2 4.1 不上机预检：推理链路通，但相机后端 MSMF 挂了）

**做**：在 `smolvla_infer` 环境里跑**只读预检**——连 COM6 取一次观测（6 轴状态 + 两路帧）→ 用迁移后的 checkpoint 跑 `predict_action_chunk`（**只推理、不下发任何动作**）→ 与训练统计量对照。

**结果**：

| 项 | 结果 |
|---|---|
| 观测 | `fixed` = 桌面俯拍、`handeye` = 夹爪特写（**逐张核图确认**）；`gripper.load = 0‰`、`holds_object = False` |
| 推理 | chunk `(50, 6)`，耗时 **30.3 s** → 摊销 **0.605 s/步**（chunk 复用） |
| 分布内 | **300 个预测值 100%** 落在训练范围内（状态范围与动作范围**两条都过**） |
| 幅度 | 与当前位姿最大偏差仅 **1.31**（动作平滑、无跳变），`shoulder_lift −96.8→−92.5`、`gripper 0.65→1.10` |
| 安全 | **未发送动作**，机械臂未移动；退出时按 `disable_torque_on_disconnect=False` 不写力矩 |

**⚠️ 顺带发现一个环境问题（比预检本身更重要）**：`run_camera_check.py` 报 **MSMF 索引 0–4 全部打不开**，而 **DSHOW 可开 0/1/2**；逐张核图得到 **DSHOW 顺序 `0 = 笔记本摄像头 / 1 = fixed(俯拍) / 2 = handeye(腕部)`** —— 与项目约定的 `fixed=2 / handeye=0`（**MSMF 顺序**）**不同**。本次预检用 `CAM_BACKEND=dshow / CAM_FIXED=1 / CAM_HANDEYE=2` 跑通，**物理相机↔通道的对应关系仍然正确**（核图可证）。**影响**：采集链路（`run_vla_collect_record.py` → lerobot OpenCVCamera）强制 MSMF，**此刻采集会起不来**；按既有处置（换 USB 口重插 + 重启 Windows）应可恢复 MSMF，否则就得每次核图后用 DSHOW 映射。

**产物**：`runs/week38/vla_preflight/`（`fixed_rgb.png`/`handeye_rgb.png`/`predicted_chunk.npy`）、`runs/week38/camera_check/20261016_193905{,_193930}/`（体检存图）。预检脚本为临时脚本，未入库。

**遗留与下一步**

- **上机前先处置相机后端**（重插 + 重启恢复 MSMF，或明确改用 DSHOW 并每次核图）
- 预检只证明"机制通"（观测→策略→分布内动作），**不证明策略好用** —— 抓取好不好要真机闭环才知道
- 首轮 chunk 幅度偏小，待闭环观察是否"过于保守"（可能需更多步数/更高 LR，或当前位姿位于训练分布边缘：`shoulder_lift −98` 贴近 min `−100`、`elbow 100` 等于 max）

**批次 39 补充（重启 + 清场后复查，2026-10-17 14:49）**

- **MSMF 根因不是系统/硬件，而是新环境的 OpenCV 构建**：同一时刻，**采集环境（`lerobot`，`opencv-python`）MSMF 0/1/2 正常**，而**新环境（`smolvla_infer`，`opencv-python-headless`）MSMF 0–4 全挂**。把新环境换成 **`opencv-python==5.0.0.93`**（先卸 headless）后 **MSMF 立即恢复**（0/1/2 可用）。→ 此前"重插/重启"都没必要，是一次**依赖选型**问题（lerobot 默认依赖 headless，而 headless 在本机没有 MSMF 能力）。
- **相机索引已漂移**：核图确认当前 **`0 = 笔记本摄像头 / 1 = fixed(俯拍) / 2 = handeye(腕部)`**，与历史约定 `fixed=2 / handeye=0` **正好错位**。若沿用旧常数 → `fixed` 会拿到腕部图、`handeye` 会拿到笔记本摄像头，**不报错、只让策略瞎跑**（正是 week38 #40 那类事故）。
- **处置**：`src/run_vla_infer.py` 新增 `--fixed-cam/--handeye-cam`（默认保留历史 2/0，**上机必须显式传核图后的值**）；本次为 `--fixed-cam 1 --handeye-cam 2`。**刻意不改** `real_executor.DEFAULT_CAM_IDX` 与 `run_camera_check.CAMERA_SLOTS` —— 那两个常量的护栏语义是"跨 repo 同一台物理相机 → 同一通道"，索引只是运行时细节；改死常数会让下台机器/下次枚举又错。
- **场景已恢复**：键盘移出 `fixed` 视野、机械臂回到画面上方，`MSMF_1`（fixed）与训练帧取景基本一致。
- **只归位验证通过（用户选择先不动策略）**：目标 home（数据集第 0 帧）`[−10.85, 15.51, 22.81, 71.93, −11.24, 0.56]` → 实到 `[−8.87, 14.24, 25.19, 72.02, −10.65, 1.60]`，**max 误差 2.38**（`HOME_TOL=3.0`）、耗时 **2.2 s**；`max_relative_target=8` 正常限幅；退出保留力矩（不下垂）。归位后 `fixed` 取景与训练帧基本一致（`runs/week38/vla_preflight/after_home_fixed.png`）。
- 回归：**1296 passed / 7 skipped / 0 failed**。

### 2026-10-17 · 批次 40（P2 4.1 真机闭环首跑：链路全通，策略未完成"抬起"）

**首跑（300 步）**：`exit 1`、`success=False`、末状态 `wrist_flex=63.57 / gripper=1.74`。
复盘发现 **300 步比 `pick_a` 最短的一集（340 帧）还短** → 步数预算不足，策略来不及做完。

**二跑（600 步；并给 `run_vla_infer.py` 加了每 50 步的轨迹日志）**：`success=False`，末状态 `wrist_flex=72.9 / gripper=1.39`。

| step | 0 | 50 | 150 | 300 | 400 | 450–600 |
|---|---|---|---|---|---|---|
| `wrist_flex` | 71.9 | **61.3** | 67.6 | 67.6 | 72.4 | **72.9（平台期）** |
| `gripper` | 1.74 | 2.23 | 1.74 | 1.46 | 1.39 | 1.39 |

**判据合理性先自查**（避免把"判据太严"误读成"策略不行"）：`pick_a` 20 集里 **18/20** 的 `wrist_flex` 最大值 ≥ 75（均值 **83.6**，范围 68.2–95.7）→ **`LIFT_WRIST=75` 没问题**。

**结论**：
- ✅ **链路完全打通**：600 步无崩溃、动作有界、`max_relative_target=8` 正常限幅、每 50 步状态可观测；归位 2 s、误差 2.38。
- ❌ **策略只学会"下探 + 闭合"，没有复现示范里的"抬起"**（示范峰值均值 83.6 vs 策略 72.9，差 10.7）。夹爪终值 1.39（位置判据算"夹住"），但**未抬起** → 任务判失败。
- 最可能原因是**训练不足**（20k 步 = **2.57 epoch**；120 集小数据集上 SmolVLA 通常需要更多步数/epoch）。

**下一步**：需要 GPU 重训更长步数（建议 **50k–60k**，并按 5k 存 checkpoint 便于选峰）；当前云端已停、且 `121.43.136.28` 的 SSH 主机指纹已变，重训前需先确认或新开机器。重训前**不要**改判据或调 `max_relative_target` —— 那会把"训练不足"和"评测口径"混在一起。

### 2026-10-17 · 批次 41（"不去找物体"的归因：排除接线/归一化，锁定相机取景偏移）

用户观察"机械臂没有朝桌面物体运动"，据此做三项归因实验：

1. **反事实敏感性**（固定同一状态，只改输入）：灰 `fixed` Δmean **10.5**、灰 `handeye` **21.4**、两路全灰 **23.6**、交换两路 **6.3**、换 place 指令 **16.9**、空指令 **19.2** → **图和指令都在明显驱动输出**，不是"输入没接上"（腕部相机影响更大）。
2. **训练样本 BC 自检**（在 `demo_vla_01_pick_a` 的样本上让策略预测，与示范动作比）：idx 100/150/250 的误差 **mean 0.28 / 2.25 / 1.09**（max 1.06 / 6.38 / 2.63）→ **模型能复现示范**，说明归一化统计、相机通道、权重都对；唯一复现不了的是 **idx=0（每集起点）**——示范在该帧是一个 `−100/100` 的大跳变，策略给的是小幅动作，而闭环正从这个状态开始。
3. **取景量化**（静态地标模板匹配，避免机械臂干扰）：实时 `fixed` 相对训练帧 **水平偏移约 45 px**（电源排插 `dx=+43`、下半幅 `dx=+48`，相关 0.63–0.75），亮度基本一致（148→152）。

**结论**：优先怀疑**固定相机取景相对训练已偏 ~45 px** —— 物体在图像里换了位置，像素→动作映射随之失效，与"不去找物体"的现象吻合。**处置顺序**：① 用 `run_camera_align.py --index 1 --dataset-repo wangbangyan/demo_vla_01_pick_a --watch` 把取景调回训练视角；② 把物体摆回示范位置；③ 再跑 600 步闭环。三者都排除后才归因训练不足。

**次要可能（尚未排除）**：物体摆放超出示范分布；BC 闭环漂移（单步好 ≠ 闭环稳）；chunk 之间 30 s 停顿改变了时间轮廓。

### 2026-10-17 · 批次 42（相机索引收敛为单一事实来源）

**问题**：`fixed` / `handeye` 的索引原先被抄在 **5 处**（`real_executor` / `vla_collect_plan` / `run_camera_check` / `run_perception_capture` / `outputs/infer_so101`）。2026-10-17 重插 + 重启后枚举变为 **`0 = 笔记本摄像头 / 1 = 桌面俯拍 / 2 = 夹爪`**，5 处**同时失效且不报错** —— `fixed` 会拿到腕部图、`handeye` 会拿到笔记本摄像头（与 #40 同类事故）。

**处置**：新增 `src/camera_slots.py` 作为**唯一事实来源**（`CAMERA_SLOTS = {"fixed": 1, "handeye": 2}`，含"为什么单独一个模块 / 硬约束 / 历史"三段说明）；上述 5 处全部改为引用它；`run_camera_align.py` 的 `--index` 默认值与对齐参考数据集（改为 `demo_vla_01_pick_a`）也走同一来源；`run_vla_infer.py` 的 `--fixed-cam/--handeye-cam` 默认值改取常量（显式传参仍可覆盖）。

**验证**：`run_camera_check.py` 现在输出 **`fixed = 索引 1 ✅ / handeye = 索引 2 ✅`**；新增 `test_slot_mapping_is_single_source_of_truth`（断言运行期与本工具**共用同一对象**）+ 更新钉值测试（`1/2`，注明核图日期）；全量回归 **1297 passed / 7 skipped / 0 failed**。

**遗留**：`runs/week38/collect/collect_commands.ps1` 里仍是旧的 `2/0`（历史产物，不影响已采的 120 集）；**重新采集前必须用生成器重出清单**。

### 2026-10-17 · 批次 43（相机微调后的复测 + 第二次真机闭环）

**取景复测（同姿态、两法交叉）**：

| 指标 | 调整前 | 调整后 |
|---|---|---|
| 电源排插模板 | score 0.752 / dx +43 / dy −42 | **score 0.886 / dx +19 / dy −6** |
| 全帧 ECC（欧氏） | — | **cc 0.761 / dx +19.6 / dy −5.7 / rot +0.69°** |

→ 从 ~43 px 降到 **~20 px、0.69°**，方向正确但**仍未到 ±5 px / 0.2° 的目标**（训练集 20 集基线的位移 std = 0）。

**第二次真机闭环（600 步，取景已改善）**：`success=False`，但**行为发生了质变**——夹爪从 1.39 一路**张开到 16.9**（上两次全程闭合 1.39），`wrist_flex` 在 56–73 之间（`step50 56.7 → step150 64.9 → step550 71.3`，仍未达 75）。

**结论**：① **相机取景是这条链路的关键变量**（20 px 的改善就改变了策略行为模式）；② 策略现在会"接近 + 张爪"，但**仍缺少"闭合 + 抬起"**，且未达抬起阈值；③ 下一步优先把取景压到 ±5 px（或做等价软件对齐）后再判"策略是否还有问题"，避免把取景误差读成模型能力。

**产物**：`runs/week38/vla_preflight/after_align2_fixed.png`、`overlay_train_live.png`（训练=红、实时=绿，错边即偏差）、`vla_run600_realigned.log`。

### 2026-10-17 · 批次 44（P2 4.1 真机闭环**首次成功**）

相机二次微调后（静态地标 `dx +7 / dy −4`）跑第三次 600 步闭环：**`success = 是`**、退出码 **0**，**348 步提前结束**，末状态 `wrist_flex=79.59 / gripper=3.77`；无缺图/异常告警。

| step | 0 | 50 | 200 | 300 | **348（成功）** |
|---|---|---|---|---|---|
| `wrist_flex` | 71.4 | 59.0 | 61.3 | 68.9 | **79.59** |
| `gripper` | 1.6 | 2.02 | 2.02 | 1.74 | **3.77** |

行为链完整：**下探 → 闭合并保持（50–250 步在桌面高度）→ 抬起（300→348 越过 75）**。

**三次闭环的取景-行为对照（本批次最有价值的结论）**：

| 取景偏差 | 夹爪行为 | 抬起 | 结果 |
|---|---|---|---|
| **43 px** | 全程闭合 1.39 | ❌ | 失败 |
| **20 px** | 张开到 16.9（接近+张爪） | ❌ | 失败 |
| **~7 px** | 闭合抓取 → 3.77 | ✅ 79.6 | **成功** |

→ **相机取景是这条链路的决定性变量**，且"不成功"未必是模型能力问题。**下一步**：再跑 2 次确认可复现（单次成功不足以定成功率）；随后再做软件对齐（默认不启用，仅在物理取景无法再压时作为兜底）。

**产物**：`vla_run600_align3.log`、`after_align3_home_fixed.png`、`overlay_train_live3.png`。

**⚠️ 更正（用户人工复核，同日）**：该次 `success=是` 是**判据假阳性** —— 用户现场确认"机械臂动作正常，但**没有夹到螺丝刀**，只是做完了动作"。**撤回上表的"成功"结论**，改记为"动作链完整、**未夹到物体**"。

**假阳性归因（已定位到分支）**：`holds_object = 位置(pos > 5.5) OR 负载(|load| > 150)`；末态 `gripper = 3.77`（空闭基准 ≈ 3.92）**过不了位置分支** → 只能是**负载分支**被触发。而"夹爪被指令闭合、实际夹空"时舵机持续顶限位，`|load|` 会超过 150‰ —— 所以**负载分支在本硬件上不能区分"真夹住"与"空夹顶限位"**。
**处置（先量后改，不拍脑袋）**：给 `run_vla_infer.py` 加**逐帧遥测**（`wrist/grip/load/hold_pos/hold_load/lifted`，每步一行）+ 每 50 步存双相机图（`runs/week38/vla_runs/<时间戳>/`）+ 成功触发时打印分量（`[success-trigger]`），据此再决定判据改法（候选：位置为主 / 负载需连续多帧 / 位置落在"夹住带"）。
**一般化教训**：**"位置 OR 负载"里的负载分支必须在真机上验证过"空夹"场景**；本次证明空夹时它会给假阳性，而只报一个布尔 `success` 又没有分量日志 → 只能靠人眼发现。

### 2026-10-17 · 批次 45（假阳性收口 + "不接近物体"的取证）

**判据修正**：`run_vla_infer.success_checker` 改为**位置为主**（`is_holding(pos>5.5)` AND `lifted`），负载只用于诊断并打印分量；新增回归 `test_success_checker_empty_closed_with_high_load_is_not_success`（空闭 3.77 + load −206 → **False**）与"真夹住+抬起 → True"。全量回归 **1298 passed / 7 skipped**。

**逐帧遥测（400 步）**：`grip` **全程 1.60**（夹爪一次没动）、`hold_pos` 全 False、`|load|` 最大 **85**（未过 150）→ 本轮到不了成功，也无假阳性。对比上一轮同模型同场景却出现 `grip 1.6→16.9` 的张开行为 → **SmolVLA 的 flow matching 有随机性，每次轨迹不同**。

**"不接近物体"的取证（存图）**：`step0100_fixed` 显示夹爪停在画面上方、螺丝刀在画面中左下方；`step0100_handeye` 显示螺丝刀在腕部视野上半部、距夹爪明显很远 → **机械臂只做垂直下探并悬停，没有横向接近物体**。

**数据侧排除**：逐帧统计 `|action − observation.state|`（8787 帧）：**只有第 0 集前 5 帧有 115.5 的遥操作瞬态**，其余 19 集前 5 帧均 ~2.0；全集 `mean 5.09`、`>20 的帧仅 129（1.5%）` → **数据集本身没有系统性污染**，之前"idx=0 不复现"是抽到了那一集的特例。

**结论**：链路与数据都没问题，**卡在策略行为本身**——闭环里不产生足够的横向接近分量（BC 在起点附近进入"悬停"固定点），叠加采样随机性。最可能的解法仍是**更长训练**（50k–60k）；需 GPU（云端已停、IP 主机指纹已变）。

**产物**：`vla_run400_telemetry.log`、`runs/week38/vla_runs/20261017_153420/`（每 50 步双相机帧）。

**补充排查（同批次，用户要求"先查其他原因"）**——逐项排除"非训练"因素：

| 嫌疑 | 检查 | 结论 |
|---|---|---|
| 腕部相机被碰偏 | 用**刚性手指**模板（姿态无关）比训练帧 | ✅ 未偏（score 0.867 / dx +5 / dy −5） |
| 固定相机 | 电源排插模板 | ✅ ~7 px |
| 动作/状态通道顺序 | `info.json` 的 `action.names` vs `state.names` | ✅ 完全一致 |
| 图像/语言接线、归一化、单位 | 反事实 + 训练样本 BC 复现 | ✅ 已在前几批次排除 |
| 数据质量 | `|action−state|` 统计 | ✅ 仅第 0 集前 5 帧异常 |

**新发现（可能的关键非训练因素）**：在**同一 home 姿态**下把实时帧与训练第 0 帧比，**桌面地标对齐（+7/−4 px），但夹爪/前臂错位 −49/−23 px**（score 0.755）。关节读数误差只有 ≤3 单位（`HOME_TOL`），不足以解释 ~4–5 cm 的夹爪位移 → **更像机械臂底座（或工作台/相机之间的相对位置）被移动过**。若成立，则"同一组关节角"会把夹爪送到与示范不同的桌面位置——策略自然伸不到物体，且**重训也学不到**（数据与部署的工作区不一致）。
**下一步（先验证再谈训练）**：核对机械臂底座/桌子的位置是否被动过；用 fixed 画面把夹爪的 home 成像位置调回训练帧附近（参考 `cmp_fixed_train_vs_home.png`）。这条排除后才归因训练量。

### 2026-10-18 · 批次 46（40k 重训完成 + 权重回传迁移 + 当前取景复测）

**训练**：新 A10 云机（16 核 / 60 GB / `121.199.47.127`）上 `smolvla_vla_40k` 跑满 **40000 步 / 20.54 epoch**，最终 **loss 0.032**（20k 终点为 0.086），吞吐 53 样本/s、显存 7.88 GB、GPU 100%；checkpoint 每 2k 存一次。
**必须记住的坑**：新机**无外网** → 起训会卡在"创建策略"（进程 STAT 停在 TCP 连接）→ 必须 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`（模型/数据/token 都在镜像缓存里）。

**权重回传**：`scp -r` 两次中途卡死（405 MB / 578 MB）→ 远端 `split -b 50m` 成 18 块 + `sha256sum`，本地逐块 scp（5 次重试 + 180 s 超时 + 字节数校验）后组装：`model.safetensors` **906,712,520 B**、sha256 **`c91ac849…846c` 与云端一致**；另补齐两个 processor `.safetensors`（归一化统计来源，主脚本漏拉）。

**迁移（0.6.1 → 0.3.4）复用批次 38 的三步**：剪 config 的 6 个 0.6.x 字段 → 相机键 `camera1/2 → fixed/handeye`（删 `camera3`）→ 从 processor 回填 `observation.state` / `action` 的 mean/std buffer（回填前缺 6 键、回填后缺 0）。产物 `runs/week38/smolvla_vla_40k/pretrained_model_lr034/`（2 文件、自包含），**重载 45.8 s + 真实前向出 6 维动作（有限、非零）**。

**当前取景复测（只读、未动机械臂）**：`run_camera_align.py --index 1` 全帧相关 **r=0.636**（含手臂姿态差异，不能单独当取景指标）；**桌面静态地标"电源排插"模板 score 0.960 / dx +15 / dy −6 px**。参照批次 44 的剂量-反应（43 px 失败 / 20 px 失败 / ~7 px 成功），**15 px 仍偏大**，上机前建议再微调到 ~10 px 内。

**待办**：机械臂上电 → 先量夹爪/前臂（刚体模板）判断底座是否已与训练一致 → fixed 取景收敛到 ~10 px 内 → 用 40k 权重跑闭环，多次统计成功率。

### 2026-10-18 · 批次 47（40k 权重首次真机闭环：抓取并抬起）

**本次取景**：未再微调相机（用户决定直接上机），实测静态地标 `dx +15 / dy −6 px`（比批次 43 的 20 px 略好、比批次 44 成功的 7 px 差）。
**上机前核对**：`run_camera_check.py` MSMF 0/1/2 可用；核图确认 `1 = 俯拍整桌`、`2 = 腕部近景（可见夹爪手指与螺丝刀）` → 与 `camera_slots.py` 一致。

**运行**：`run_vla_infer.py --model-dir runs/week38/smolvla_vla_40k/pretrained_model_lr034 --fixed-cam 1 --handeye-cam 2 --steps 600`。
**结果**：**第 203 步提前判定成功**（`wrist_flex=76.07 ≥ 75` 且 `gripper=7.74 > 5.5`，**位置分支**，非负载兜底）。遥测行为链：`grip 1.60(闭合)` → 步 100 张开到 9.9 → 步 112–116 负载冲到 **+217~272**（接触/压到物体）→ 步 120 张到 22.66（对准）→ 步 200 收到 **7.74 并保持 50+ 步** → `wrist 68.1 → 76.07` 抬起。
**图像佐证**：`step0100_handeye` 夹爪张开、螺丝刀在正前方；`step0150/0200_handeye` **橙色手柄进入并停留在两指之间**；`step0200_fixed` 桌面上的橙色螺丝刀已不可见（初始帧可见）→ 与"夹住并抬起"一致。

**两个必须记住的收尾问题（不影响任务判定）**：① **`park()` 45 s 超时**（收尾复位没在限时内回到初始姿态，程序按既有逻辑继续）；② **`engine.close()` 报 `Overload error!`**（夹爪 `Torque_Enable=0` 连试 6 次失败，采集期已知问题）→ 进程退出码 1，但 **任务成功与否由 `run_vla_loop` 的 summary 决定，本次 summary 为成功**。下次可在收尾前先把夹爪张开再卸力矩，或保留力矩（`disable_torque_on_disconnect=false`）。

**结论**：40k 权重在 15 px 取景偏差下仍完成抓取-抬起；单次成功仍不足以定成功率，需重复 2–3 次（相机当前未精确对齐，重复时要保持同一取景与物体位置）。

### 2026-10-18 · 批次 48（成功默认不复位 + 第 2 次闭环成功）

**改动（用户决策）**：`run_vla_infer.py` 新增 `should_park()` 与 `--park-on-success`。口径：`--no-park` 一律不复位；**成功默认保持当前位姿**（便于现场核验"是否真夹住"与连续评测）；**失败仍必复位**（安全默认）；只有显式 `--park-on-success` 才在成功时复位。新增 4 项回归（`TestShouldPark`），`test_vla_infer.py` **27 passed**。
**为什么改**：批次 47 成功后默认 `park()` 把手臂带回初始姿态，现场核验被破坏；而失败时停在"下探贴桌"姿态有风险，所以不能一刀切取消。

**第 2 次闭环（40k，同一取景/同一物体位置）**：**第 402 步成功**（`wrist 75.01 / grip 9.76`）；日志确认 `任务成功：保持当前位姿不复位`；`step0400_handeye` 可见橙色手柄正夹在两指之间、金属杆朝上，`step0400_fixed` 桌上已无螺丝刀。
**当前统计**：2 次运行 **2 次成功**（203 步 / 402 步），但**两次轨迹长度与动作时序差异明显**（SmolVLA flow matching 采样随机性），单看成功率仍需更多样本。
**仍未修**：`engine.close()` 的 `Overload error!` 仍使进程 exit 1（力矩因此保持、手臂不松手）；这条只影响退出码，不影响任务判定。

### 2026-10-18 · 批次 49（第 3 次闭环：张开未夹住被误判成功 → 判据补上界）

**运行**：同一命令跑第 3 次（螺丝刀这次是**横放**，与前两次姿态不同）。
**误判**：第 175 步判成功，末态 `wrist 75.45 / gripper 19.25`；**人工确认"找到并张开夹爪、没夹住、但抬起了"** → **假阳性**。
**根因**：`is_holding()` 只有下界（`pos > 4.1+1.4 = 5.5`），**"张开"（实测 16.9–22.7）同样满足**；本次 `19.25 > 5.5` 直接通过。
**遥测特征**：步 140 `grip=1.74`（空闭）→ 随后张到 19.25 → 手腕抬到 75.45；负载末尾仅 62‰（张开/空转水平）。
**修复**：`real_observation` 新增 `GRIP_HOLD_MAX = 15.0`，`is_holding` 改为**夹住带** `5.5 < pos < 15.0`（实测夹住 6.7–9.8，留厚件余量；张开 16.9–22.7 全部排除）。`holds_object` / `real_perception` / `vlm_labeler` 等调用方自动继承。
**回归**：新增 `test_wide_open_is_not_holding`、`test_upper_bound_still_admits_thick_object`、`test_success_checker_wide_open_with_lift_is_not_success`；相关 4 个文件 **177 passed**，全量（排除 3 个需 ai2thor 的仿真文件）**1305 passed / 7 skipped**。
**统计口径更正**：第 3 次**记为失败**；当前有效统计 **2 次成功 / 3 次运行**（成功率 2/3，样本仍小）。

### 2026-10-18 · 批次 50（第 4 次闭环失败 + 过载保护 / 归位跳过夹爪）

**第 4 次（第 3 个有效样本）**：600 步跑满，**失败**——`grip` 全程 1.74（空闭）、`wrist` 60.2→66.0，典型的"闭合悬停"失败模式。累计 **2/3**。

**用户提出的两个优化及落地**：
1. **归位不再动夹爪**（用户选定）：`infer_so101.home_to_initial` 新增 `skip_joints`，跳过的关节**既不下发目标也不参与到达判定**；`RealEngine.home(..., *, include_gripper=True)` 透传；`RealSkillExecutor.park(include_gripper=True)` 同步。`run_vla_infer.py` 默认 `--home-gripper=False`（只复位 5 个臂关节），需要旧行为时显式加 `--home-gripper`。
   **动机**：数据集第 0 帧的夹爪目标是 **0.56（闭合值）**，每次归位都往闭合方向硬顶，是夹爪过载的直接来源之一。
2. **夹爪过载保护（3 s 持续）**：新增 `GripperTorqueGuard`，`|load| > 150‰` 连续 **3 s** → **冻结夹爪目标为当前实际位置**（每步继续下发 → 保持夹持、不松手），负载回落连续 **1 s** 后解除；总线读不到负载时已冻结则保持冻结（不能假设安全）。已接入 `run_vla_loop`（仅覆盖夹爪那一维，其余关节继续跟随策略）。

**回归**：`test_vla_infer.py` 新增 `TestGripperTorqueGuard`（5 项）+ 2 项主循环集成测试；`test_real_agent_core.py` 新增 `park(include_gripper=...)` 透传与 `home_to_initial(skip_joints=...)` 测试；全量（排除 3 个需 ai2thor 的仿真文件）**1314 passed / 7 skipped**。

**待验证**：跳过夹爪归位后，起始夹爪状态不再等于示范起点（0.56）→ 是否影响成功率需真机再跑确认。

### 2026-10-18 · 批次 51（新口径首跑：仍是"闭合悬停"，与旧口径同型）

**运行**：新默认（归位跳过夹爪 + 过载保护）跑 600 步 → **失败**：`grip` 全程 **1.74**、`wrist 72 → 63.7 → 66.4 → 69 → 71.49`（下探后缓慢回升、无横向接近），与**批次 50 的旧口径运行（`grip 1.74`、`wrist 60→66`）同型**。
**归位日志**：`正在归位到采集初始姿态（跳过 ['gripper.pos']）... 已到达初始姿态 0.0s`。
**关键对照（用于排除改动嫌疑）**：两种归位口径下**起始夹爪读数都是 1.74**（旧口径的下发目标 0.56 也到不了，舵机停在 ~1.74；|1.74−0.56|=1.18 < HOME_TOL 3.0）→ **本次失败与"归位是否带夹爪"无关**，是策略进入了已知的"闭合悬停"失败模式。**过载保护未触发**（负载 −41…−94‰，未过 150‰），故也不是保护逻辑造成。
**当前统计**：**2 成功 / 4 有效运行**（成功两次都表现为"前期张开、步 100 附近对准"；失败两次都是"从第 0 步起闭合、不再张开"）。
**下一步判断**：该二分模式与取景偏差（当前 15 px，成功时为 7 px）高度相关 → 优先把 fixed 取景调回 ~7 px 再复跑，或做软件对齐；代码侧不再叠加改动。

### 2026-10-18 · 批次 52（归位口径 A/B：旧口径同样失败，排除改动嫌疑）

**A/B 设计**：同场景、同权重、同指令，仅切换归位口径（`--home-gripper` = 旧完整归位）。
**B 组（旧口径）结果**：**一样失败** —— 600 步，`grip` 全程 **1.60**、`wrist` 72→67.09（闭合悬停）。
**A 组（新口径，跳过夹爪）**：`grip 1.74`、`wrist` 72→71.49（同型）。
**结论**：**失败与"归位是否带夹爪"无关**，两组进入的是同一个策略失败模式；过载保护也未触发。累计 **2 成功 / 5 有效运行**。
**剩余最强变量**：fixed 取景（15 px vs 成功时的 7 px，剂量-反应已三次复现）→ 下一步只动取景（硬件微调或软件 warp），不再改代码逻辑。

### 2026-10-18 · 批次 53（P2 4.5：补上"进程内冻结"训练入口并启动矩阵）

**先修掉一个会让 A5 失效的缺口**：`run_vla_adapt.py` 的默认训练路径走 `lerobot_train` **CLI**，而该 CLI 没有按模块冻结参数 → `head` 与 `full` 会训练成同一个东西。新增 `src/run_vla_adapt_smolvla.py`：patch `lerobot_train.make_policy`，拿到 policy 后立即 `apply_adaptation` 改 `requires_grad`，再交给官方训练循环（`optimizers.py` 会跳过冻结参数）。
**数据量自变量**用 `--dataset.episodes` 等距抽子集（30 / 60 集），不复制数据集。
**实测口径修正**：SmolVLA 的 head 由 `action_{in,out}_proj` + `action_time_mlp_{in,out}` + `state_proj` 组成；原 `DEFAULT_HEAD_PATTERNS` 漏了后两组 → head 只有 **46,832** 参数（0.01%）。补入后实测 **1,635,152 / 450,046,176 = 0.363%**。
**云端冒烟通过**：`num_learnable_params=1635152`、训练 5 步正常、checkpoint 落盘。
**矩阵启动**（A10，`/root/adapt_matrix.log`）：`head`/`full` × 30/60 集，各 2000 步、bs=32、8 workers，单单元约 20 min（实测 1.68 step/s，loss 0.320→0.248）。
**回归**：新增 `test_run_vla_adapt_smolvla.py` 7 项 + head 命名覆盖测试，相关 41 passed。
**评测口径（后续）**：A5/A6 的因变量是**真机成功率**（A6 要求相对 zero_shot ≥20pp），故训练完需把各 checkpoint 迁移到 0.3.4 后上机评测；zero_shot 锚点用 `lerobot/smolvla_base`（不训练）。

### 2026-10-18 · 批次 54（4.5 配套：离线评测器 + 迁移工具 + 模型卡）

**① 离线评测器**（训练一结束即可用）：`src/adapt_eval_offline.py`（纯逻辑：action L1、夹爪 L1、夹住带 `recall/precision/F1`，夹住带与真机判据同源 `5.5–15.0`）+ `src/run_adapt_eval_offline.py`（云端 CLI：按 stride 抽帧跑 `predict_action_chunk`，产物 `offline_metrics.json`）。**报告里强制带 `CAVEAT`**：离线指标≠真机成功率，A6 必须实测。新增 12 项测试。
**② 迁移工具**：`src/migrate_vla_checkpoint.py` 把批次 38 的三步（剪 6 个 0.6.x 字段 / `camera1,2→fixed,handeye` 且删 `camera3` / 从 processor 回填 mean·std）固化成可复用 CLI，并对"processor 文件缺失"给出明确报错。**实测与手工迁移产物逐字节一致**（config 与权重 sha256 均相同），新增 7 项测试。
**③ 模型卡**：`docs/week38/model_card.md` 新增 §7（40k 部署权重：训练口径 / 回传校验 / 5 次真机结果 2 成功）与 §8（4.5 矩阵与评测口径）。
**④ A7 协议预检**：用 `collect_combos.json` 的 6 个真实组合跑 `run_vla_generalization.py --dry-run`，三协议 train/held-out **全部无泄漏**（`combo` 4/2、`paraphrase` 3/3、`novel_object` 4/2）。

### 2026-10-18 · 批次 55（云端重启事故 + 离线评测器实跑）

**事故**：矩阵跑到 `head_ep30` 第 **807/2000** 步时，云机在 **02:35 关机 / 02:36 重启**（`last -x` 可证），进程被杀；因 `save_freq=2000` **一个 checkpoint 都没留下**。
**处置**：`run_vla_adapt_smolvla.py` 增加 ① `--save-freq`（默认 **500**，重启最多丢一个窗口）；② `--resume`（从 `train/checkpoints/last` 续训）；③ 非 resume 时自动清空残留的 `train/`（否则 `lerobot_train` 会因 output_dir 已存在而报错）。矩阵已重启（`/root/adapt_matrix2.log`）。
**离线评测器实跑**：在 40k checkpoint 上跑 12 帧（6 个 episode 等距抽）→ `action L1=0.4880`、`gripper L1=0.2357`、`hold F1=1.000`，数据集键名确认为 `observation.images.fixed/handeye`。这同时验证了"checkpoint 自带 processor（含归一化统计）"这条加载路径可用。
**边界**：该 smoke 只证明评测链路能跑，数值不参与任何结论（帧数少且非正式 held-out）。

### 2026-10-18 · 批次 56（A5 汇总脚本 + 采集清单重出）

**① A5 离线汇总**：新增 `src/run_adapt_summary.py` —— 扫描 `<results-dir>/<unit>/offline_metrics.json`，可选读取 `<train-stats-dir>/<unit>/adapt_stats.json` 的可训练参数占比，按 action L1 升序渲染 Markdown 表（缺值显示 `—`，`zero_shot` 占比记 0），并强制附带 `CAVEAT`。新增 5 项测试（含"空目录必须报错"，避免空表被误读为全失败）。
**② 采集清单重出**：`vla_collect_plan.py` 已改为引用 `camera_slots`，但历史清单 `runs/week38/collect/collect_commands.ps1` 仍是旧索引 **2/0**。本次重出到新目录 `runs/week38/collect/regen_20261018/`（**不动历史清单**，保留已采 120 集的溯源），核对相机为 **fixed=1 / handeye=2**；仍为 6 repo / 120 集、`pick_a`+`place_left`、每 repo 一份旁挂、断开不卸力矩。
**回归**：全量（排除 3 个需 ai2thor 的仿真文件）**1346 passed / 7 skipped**。

### 2026-10-18 · 批次 57（`full` 策略 OOM → 降 batch 重跑）

**结果**：`head` 两个单元正常完成（ep30、ep60 各 2000 步，末 loss ≈0.222）；`full` 两个单元在**第 0 步就 OOM**（`torch.OutOfMemoryError`，22.06 GiB 全占满）。
**根因（对比 40k 日志才看清）**：`smolvla_base` 的默认微调只训练 **99,880,992（100M）** 参数——VLM 主干默认被冻结；而 `full` 策略把 **450,046,176（450M）** 全部解冻 → 优化器状态 + 激活在 bs=32 下放不下。
**处置**：`full` 改用 **bs=16**（+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`）重跑；实测显存 **15.39 GB**、1.43 step/s、loss 0.310(200) → 0.244(400)，正常推进。
**必须记录的混淆变量**：`head` 用 bs=32、`full` 用 bs=16（卡容量所限）→ **策略轴（head vs full）与 batch 轴混淆**；数据量轴（30 vs 60）在各策略内部仍同 bs、可比。A5 表里必须标注这一点。

### 2026-10-18 · 批次 58（A5 离线对照完成）

**评测口径**：held-out 用 **奇数索引 episode** `[1,21,41,61,81,101]`（训练子集只取偶数索引：30 集=每 4 个、60 集=每 2 个 → 奇数从未参与训练），stride 40、最多 120 帧（实际 81 帧）。5 个模型同一批帧。

| 单元 | 可训练参数 | action L1 | gripper L1 | 夹住 recall | precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **head_ep30** | 0.36% | **2.684** | 2.242 | 0.684 | 0.722 | 0.703 |
| head_ep60 | 0.36% | 2.820 | 2.417 | 0.684 | 0.684 | 0.684 |
| full_ep30 | 100% | 3.023 | 2.507 | 0.526 | 0.667 | 0.588 |
| full_ep60 | 100% | 3.092 | **2.061** | **0.789** | **0.789** | **0.789** |
| zero_shot（base） | 0% | **32.332** | 6.282 | 0.105 | 0.667 | 0.182 |

**三条结论**：
1. **适配本身有效**：zero_shot 的 action L1 是 32.3，任何微调单元都在 2.7–3.1（差一个数量级）——base 的动作空间与本任务完全不同，必须微调。
2. **full 的训练 loss 更低（0.113/0.125 vs head 0.222），但 held-out action L1 更差（3.0 vs 2.7）→ 小数据上过拟合**；head 的泛化更好。
3. **数据量不单调**：head 30→60 略变差（2.68→2.82）、full 30→60 也略变差（L1）但夹住 F1 从 0.588 升到 0.789 → 60 集的收益体现在"夹爪时机"而不是整体动作误差。

**必须标注的局限**：① head(bs32) vs full(bs16) 混淆；② 每单元仅 81 帧、单次采样（SmolVLA 有流匹配随机性）；③ 这是**离线代理**，A6 的 ≥20pp 仍需真机实测。产物：`runs/week38/vla_adapt_eval/A5_offline.md` 与各单元 `offline_metrics.json`。

### 2026-10-18 · 批次 59（A5 稳健性重复 + A7 真实路径阻塞确认）

**A7 的真实评测路径确认不可用（不是没跑，是跑不了）**：`evaluate_generalization` 的 runner 约定把 combo 的第一个元素**直接当仿真 env_id**（`runner(env_id, seed, instruction)`），而我们的组合是「螺丝刀 / 黑色长条 / 指甲刀」——ManiSkill 没有对应环境。降级路径同样绕不开这一点（`_scripted_runner` 会拿中文物体名去 `ManiSkillTaskConfig(env_id=...)`）。**结论**：A7 要么等有对应 sim 任务，要么写真机 runner；本轮不产出假结果。

**A5 稳健性**：为避免"单次 81 帧"的结论噪声，给 `run_adapt_eval_offline.py` 加 `--offset`（换一批帧），`run_adapt_summary.py` 支持 `<unit>_r1..rN` 聚合（均值 ± 样本标准差，`trainable_ratio` 从基名目录读取）。新增 3 项测试（重复聚合、单次无误差带、重复目录读基名 stats）。
**实跑**：5 个单元 × offset `0/13/26`，同 episode 集（奇数索引）、stride 40，产物 `outputs/vla_adapt_eval_repeat/<unit>_r<k>/`。

**结果（3 次重复，均值 ± 样本标准差）**：

| 单元 | 可训练参数 | action L1 | gripper L1 | 夹住 F1 |
|---|---:|---:|---:|---:|
| head_ep30 | 0.36% | 2.864 ± 0.190 | 2.501 ± 0.782 | 0.635 ± 0.081 |
| head_ep60 | 0.36% | 2.876 ± 0.379 | 2.048 ± 0.482 | 0.658 ± 0.054 |
| full_ep60 | 100% | 2.932 ± 0.076 | 1.989 ± 0.351 | 0.709 ± 0.041 |
| full_ep30 | 100% | 3.054 ± 0.075 | 2.054 ± 0.319 | 0.697 ± 0.028 |
| zero_shot | 0% | 32.621 ± 0.252 | 6.456 ± 0.113 | 0.064 ± 0.056 |

**⚠️ 更正批次 58 的两条结论**：
1. **"full 过拟合、head 泛化更好"不成立**——head 与 full 的 action L1 差（2.86–2.88 vs 2.93–3.05）**小于重复间的标准差**（head_ep30 ±0.19、head_ep60 ±0.38），属采样噪声；单次 81 帧的表不能支撑该判断。
2. **"30→60 集有益于夹住时机"也不成立**——head/full 的 30 vs 60 差异同样在误差带内。
**唯一稳健的结论**：**微调本身有效**（zero_shot 32.62 ± 0.25 vs 适配单元 2.86–3.05，差距 ≈100× 标准差）。**方法论**：小样本对照必须先看重复间方差再下结论；单次结果只能当"链路可用"的证据。

### 2026-10-18 · 批次 60（退出码修复 + A5 第二轮重复）

**退出码掩盖（真机遗留债）**：`run_vla_infer` 的 `finally` 直接调 `engine.close()`；夹爪舵机过载时 `Torque_Enable=0` 抛异常，会把"任务成功"的返回码覆盖成 1（批次 47/48 都出现）。新增 `safe_close(engine)`：收尾异常**只告警不抛出**，任务结论由 `summary["success"]` 决定。新增 2 项测试（正常关闭 → True；Overload 异常 → False 且不冒泡）。
**A5 第二轮重复**：offset 增加 `5/10/15/20/25`（与已有 0/13/26 合计 **8 次/单元**），目标是把 head vs full、30 vs 60 的置信区间压到能分辨的量级。

### 2026-10-18 · 批次 61（防误用护栏）

**① `run_vla_adapt.py` 加硬护栏**：非 dry-run 且未显式 `--allow-unsafe-default` 时**直接拒绝执行**，报错信息指向新入口（`run_vla_adapt_smolvla.py` + `run_adapt_eval_offline.py`）。原因：该 CLI 的默认路径有两个已知缺陷（`lerobot_train` CLI 无冻结入口 → head≡full；默认 eval 走 ACT 加载器）。`unsafe_default_error()` 为纯函数，2 条断言覆盖 dry-run/显式放行/拒绝三态。
**② `run_vla_infer --steps` 默认 30 → 600**：演示集最短一集 **340 帧**，旧默认低于它 → 策略必然跑不完。新增回归 `DEFAULT_STEPS >= 340`。
**回归**：全量 **1356 passed / 7 skipped**。

### 2026-10-18 · 批次 62（逐集聚合 + 矩阵幂等 + A5 八次重复收口）

**① 逐集成功聚合（补 A4 缺口）**：`real_observation` 新增 `EpisodeOutcome` 与 `aggregate_episode_success(states, loads, *, lift_threshold, use_load)` —— 规则与真机闭环一致：**存在某帧同时"夹住 + 已抬起"**即判整集成功（不是看末帧）。支持 6 维向量或命名字典、负载可选（`use_load=False` 与纯位置口径对齐）。新增 9 项测试（含"张开 19.25+抬起不算成功""成功后松手不影响整集""负载兜底薄件""维度/长度校验"）。
**② 训练矩阵幂等**：`run_vla_adapt_smolvla` 新增 `DONE` 完成标记与 `--force`；`should_skip()` 只认标记——**中断留下的半成品目录必须重跑**（不能把没训完的 checkpoint 当结果）。dry-run 不受跳过影响。新增 3 项测试。
**③ A5 最终结果（8 次重复）**：

| 单元 | action L1 | gripper L1 | 夹住 F1 |
|---|---:|---:|---:|
| head_ep30 | 2.680 ± 0.199 | 2.258 ± 0.512 | 0.645 ± 0.067 |
| head_ep60 | 2.849 ± 0.220 | 2.217 ± 0.393 | 0.633 ± 0.062 |
| full_ep60 | 2.860 ± 0.157 | 1.872 ± 0.295 | 0.710 ± 0.068 |
| full_ep30 | 3.062 ± 0.133 | 2.005 ± 0.288 | 0.690 ± 0.057 |
| zero_shot | **32.688 ± 0.272** | 6.424 ± 0.169 | 0.035 ± 0.049 |

**结论不变**：只有"微调有效"稳健；head_ep30 与 full_ep30 的 0.38 差约 **1.6σ**，仍不显著。模型卡 §8.1 已同步。
**回归**：全量 **1366 passed / 7 skipped**。

### 2026-10-18 · 批次 63（A4 逐集复评：聚合补上了，但 κ 仍不可解释）

**工具**：新增 `src/run_episode_success_audit.py` —— 用**旁挂 positions + gripper.load**（采集时逐帧校验过对齐）跑 `aggregate_episode_success`，与 VLM 的逐集 `success` 算 κ/混淆/准确率；两种口径（位置 OR 负载 / 纯位置）。新增 8 项测试。

**结果（120 集，全量）**：

| 口径 | 集数 | κ | 准确率 | TP | TN | FP | FN | 传感器成功 | VLM 成功 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 位置 OR 负载 | 120 | **−0.252** | 0.358 | 43 | 0 | 62 | 15 | **105** | 58 |
| 纯位置 | 120 | −0.236 | 0.367 | 43 | 1 | 61 | 15 | 104 | 58 |

**只取 `pick_a`（语义同为"夹起"）**：59 集，TP 37 / FN 15 / FP 7 / TN 0，准确率 **0.627**；κ 仍为负（两侧绝大多数为 True，κ 退化）。

**根因（比"缺函数"更深）**：
1. **`place_left` 两边的问法不同**——传感器判"是否夹住并抬起"，而 VLM 的 `success` 是按**整条指令**（"把 X 放到左边"）作答 → place 的 41 集里 VLM 只判 6 集成功、传感器判 41 集成功，语义不可比。
2. **`pick_a` 两侧几乎全 True**（VLM 52/59、传感器 44/59）→ 患病率极端不平衡，κ 天然趋向 0/负，**准确率 0.63 也不能读成"判据有效"**。
3. **VLM 标签本身不是人工真值**（仅 14 集抽检），分歧 22/59 时无法判定谁对。

**结论**：**A4 不能靠"补逐集聚合"闭合**。下一步二选一：① 按**任务分别提问**重新标注（pick 问"是否夹住并抬起"、place 问"是否放到目标位"），再各自算 κ；② 抽一批（≥30 集）人工真值，先确定哪一侧可信。

### 2026-10-18 · 批次 64（A7 离线泛化 + 40k 参考行）

**① 40k 参考行（in-sample）**：在 A5 同一批 6 集 / stride 40 / 81 帧上评 40k 部署权重：
`action L1 0.786 / gripper L1 0.595 / 夹住 F1 0.889`（A5 head_ep30 为 2.680 / 2.258 / 0.645）。
**必须标注边界**：这 6 集参与过 40k 训练（40k 用全量 120 集），它是"训练充分时的上界参考"，
**不是** 40k 的 held-out 表现，也不与 A5 的 held-out 数字同口径。

**② A7 训练**：用 `run_vla_generalization.py --dry-run` 拿到三协议 split（6 组合 = 3 物体 × 2 指令；
episode 映射 `0-19 螺丝刀pick / 20-39 螺丝刀place / 40-58 黑色长条pick / 59-79 黑色长条place /
80-99 指甲刀pick / 100-119 指甲刀place`），按协议**排除 held-out 组合的 episode** 只训 `head`
（2000 步 / bs 32；新增 `--episodes`（显式列表）、`--job-tag`、`--save-freq`、`--resume`、
`--force` 与 `DONE` 幂等）。三个单元各约 22 min。

| 协议 | 训练集 | held-out 组合 |
|---|---|---|
| combo | 81 集（20-39, 59-79, 80-99, 100-119） | 螺丝刀pick + 黑色长条pick（39 集） |
| novel_object | 80 集（0-79） | 指甲刀全部（40 集） |
| paraphrase | 61 集（0-19, 59-79, 80-99） | 每种物体留 1 条指令（59 集） |

**③ 首轮评测（stride 40 / 每侧 2-3 集）暴露协议缺陷**：夹住 F1 在 `[1,41]` 上为 0.000，
另一侧 0.636——但实测这 2 集的 GT 夹住帧只有 **1 帧与 0 帧**（`ep41` 447 帧中仅 6 帧在夹住带内，
`ep1` 451 帧中 6 帧；120 集的整集夹住占比中位数仅 **5.8%**）。**稀疏采样下 F1 可因"抽不到阳性"恒为 0**，
与模型能力无关。→ 新增纯函数 `select_frame_ids(..., even=True)` 与 CLI `--even`：在所选 episode 的
**全区间**均匀取帧（含首尾），`--max-frames` 控制预算；5 项测试，回归 **1380 passed / 7 skipped**
（排除 3 个需 ai2thor 的文件）。

**④ 复评（`--even` / 160 帧 / 两侧全 episode / 3 次重复）**：

| 协议 | 侧 | action L1 | gripper L1 | 夹住 F1 |
|---|---|---:|---:|---:|
| combo | held-out | 2.779 ± 0.083 | 1.795 ± 0.061 | 0.513 ± 0.064 |
| combo | seen | 3.010 ± 0.120 | 1.568 ± 0.091 | 0.558 ± 0.022 |
| novel_object | held-out | 2.936 ± 0.127 | 1.965 ± 0.099 | 0.586 ± 0.030 |
| novel_object | seen | 2.849 ± 0.065 | 2.009 ± 0.085 | 0.523 ± 0.038 |
| paraphrase | held-out | 3.418 ± 0.124 | 1.961 ± 0.117 | 0.499 ± 0.114 |
| paraphrase | seen | 2.797 ± 0.061 | 1.786 ± 0.055 | 0.464 ± 0.063 |

**gap（held-out − seen）**：combo **−0.232 ± 0.084**；novel_object **+0.087 ± 0.082**；
paraphrase **+0.621 ± 0.079**（≈7.8σ）。**集合难度对照**用 40k（两侧都见过）：
combo −0.233 / novel_object +0.143 / paraphrase +0.112 → 集合难度带约 ±0.23。

**⑤ 结论**：① **A7 在 2000 步预算下不可结论**——head 的 L1（2.78–3.42）比 40k 参考行（0.67–0.90）
差 3–4 倍，三个单元整体**欠拟合**，此时"泛化 gap"会被"哪里都没学会"掩盖。② `combo` / `novel_object`
的 gap 与集合难度同量级，**没观察到留出退化**。③ `paraphrase` 的 L1 退化超过难度带，但该 split
同时留出了**任务类型（pick vs place）**（本仓每种物体只有 2 条指令变体），**语言改写与任务类型混淆**，
不能归因。④ 夹住 F1 的重复方差（para held-out ±0.114）大于协议间差异，**F1 不作 gap 判据**。
⑤ 下一步：A7 需把训练预算提到 40k 量级（或至少数千步的饱和验证）后再测；paraphrase 需**真正的
同义改写数据**（当前仓库没有）另开一轮。

**产物**：`runs/week38/vla_adapt_eval_a7/{stride40,even}/`（各侧 `offline_metrics.json`）；
训练权重在云端 `outputs/vla_adapt_a7/head_a7_{combo,novel_object,paraphrase}/`（未回传，可重训）。


### 2026-10-19 · 批次 66（真机单测：夹起螺丝刀放到左边）

**新增**：`run_vla_infer.py --success-mode {pick,place}`。`place` = "夹住抬起 → 横向搬运 ≥15° → 张开释放"，
避免 pick 判据在抓取瞬间提前结束；2026-10-19 实测出现"夹住 5 步后原地张开"被判成功，故加**横向位移闸门**
（无位移的原地松手不算完成）。落点仍无法由传感器证明，必须图像/人工复核。`test_vla_infer.py` 44 passed。

**实跑（40k 权重，fixed=1 / handeye=2，COM6）**：

| 阶段 | 结果 |
|---|---|
| 单次 place（从 pick_a 初始位） | 900 步失败：反复"夹住→原地松手"，pan 位移 0.0–4.7° |
| 两段式 ① pick | 270 步成功（wrist 80.99 / grip 7.04 / load 181，真夹住） |
| 两段式 ② place（--no-home） | 900 步失败：4 次"夹住→原地松手"，pan 位移 0.7–2.2° |
| 复跑 ① pick | 534 步"成功"（grip 12.27）但随后搬运时物体掉落 → **位置带内的假阳性** |

**根因（数据侧）**：`demo_vla_01_place_left` 20 集噪声大——同一集内多次开合
（ep1 有 106 帧张开，首次释放 pan +12.8，最终放置 pan -29.6；ep2 首次释放 pan +16.5、最终 -35.8；
ep19 首次释放 pan -5.2、最终 -45.2）。语言条件下的策略因此学到"握住后先松手"，
而不是"搬到左边再松手"。**结论**：place 需要重新切分/清洗示范（或单独训"搬运"与"释放"两段），
不能靠放宽判据或加步数解决。

**附带定位**：`place_left` 第 0 帧初始位不可达——肩关节受重力停在 2.93，目标 −0.72，差 **3.65° > HOME_TOL 3.0**，
归位必然超时（表现就是"手臂抬到位后不动"）。改用 `pick_a` 初始位后归位 0.5–1.0 s 通过。
**回归**：`test_vla_infer.py` 44 passed。

