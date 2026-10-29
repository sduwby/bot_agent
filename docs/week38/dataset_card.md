# 数据卡 · week38 多任务语言标注演示数据集（SO-101 真机）

> 面向"别人拿到这份数据能不能用"。**每一项都对应真实文件**（见表尾"数据溯源"），
> 读不出的字段一律写"未记录"，不做推测。
>
> 采集/核验时间：2026-10-11。编码规范见仓库 `AGENTS.md`。

## 1. 数据集清单与规模

数据集根目录：`C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\<repo>`
（`collect_tasks.json` 里的 `lerobot_cache_root` = `C:\Users\Administrator\.cache\huggingface\lerobot`）

| # | repo（`wangbangyan/` 下） | 指令文本（`dataset.single_task`） | 集数 | 总帧数 | 本地 parquet 集数 |
|---|---|---|---|---|---|
| 1 | `demo_vla_01_pick_a` | 夹起桌面上的螺丝刀 | 20 | 8787 | 20 |
| 2 | `demo_vla_01_place_left` | 把螺丝刀放到左边 | 20 | 12284 | 20 |
| 3 | `demo_vla_02_pick_a` | 夹起桌面上的黑色长条 | **19** | 8530 | 19 |
| 4 | `demo_vla_02_place_left` | 把黑色长条放到左边 | **21** | 11640 | 21 |
| 5 | `demo_vla_03_pick_a` | 夹起桌面上的指甲刀 | 20 | 10207 | 20 |
| 6 | `demo_vla_03_place_left` | 把指甲刀放到左边 | 20 | 10870 | 20 |
| | **合计** | | **120** | **62318** | **120** |

- "集数/总帧数"取自各 repo 的 `meta/info.json`（`total_episodes` / `total_frames`），
  并与 `meta/episodes.jsonl` 逐集 `length` 求和一致（已核对，逐 repo 完全相同）。
- 计划集数是**每个 repo 20 集**（`collect_tasks.json` 的 `tasks[].episodes` = 20）；
  实际 `02_pick_a` 少 1 集、`02_place_left` 多 1 集（见 §7 已知偏差）。

## 2. 任务与指令文本

每种物体有两种指令（两个"意图"），由 `collect_tasks.json` 的模板生成：

| 物体 | pick 指令 | place 指令 |
|---|---|---|
| 螺丝刀 | 夹起桌面上的螺丝刀 | 把螺丝刀放到左边 |
| 黑色长条 | 夹起桌面上的黑色长条 | 把黑色长条放到左边 |
| 指甲刀 | 夹起桌面上的指甲刀 | 把指甲刀放到左边 |

每条 episode 的 `tasks` 字段（`meta/episodes.jsonl`）即上表对应文本；`total_tasks` = 1（每 repo 单一任务）。

## 3. 双相机配置与索引（已逐张核图）

| 通道（特征名） | 视角 | 后端/索引 | 分辨率 | 编码 | fps |
|---|---|---|---|---|---|
| `observation.images.fixed` | 工作台俯拍（整桌） | MSMF 索引 **2** | 640x480 | AV1（yuv420p） | 30 |
| `observation.images.handeye` | 腕上夹爪近景 | MSMF 索引 **0** | 640x480 | AV1（yuv420p） | 30 |

- 采集命令（`collect_commands.ps1`）里写死：
  `"fixed": {"type":"opencv","index_or_path":2,...}, "handeye": {"type":"opencv","index_or_path":0,...}`。
- 索引身份**已逐张核图确认**：`runs/week38/camera_check/20261011_161859/camera_check.txt`
  记录"`fixed` = 索引 2 ✅、`handeye` = 索引 0 ✅"，并给出 MSMF 0/1/2 的存图供人眼核对；
  `runs/week38/cam_map_check_after_replug/` 另存 `ref_dataset_fixed.png` /
  `ref_dataset_handeye.png` 与 `live_idx0/1/2.png` 的对照。
- ⚠️ **`2/0` 只是采集当时的枚举，不是永久约定**：2026-10-17 重插 + 重启后核图变为
  `fixed=1 / handeye=2`（`0 = 笔记本摄像头`）。真正要固定的是"**同一台物理相机 → 同一通道**"；
  唯一事实来源已收敛到 `src/camera_slots.py`，重新采集/推理前**必须**先跑
  `python src/run_camera_check.py` 逐张核图。
- `video.is_depth_map` = false（两路均为 RGB，**非**深度图）。

## 4. 动作 / 状态与数据集结构

- 机器人类型：`so101_follower`；`action` 与 `observation.state` 均为 **6 维** float32，
  名字与顺序：`shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos,
  wrist_roll.pos, gripper.pos`。
- 其他字段：`timestamp` / `frame_index` / `episode_index` / `index` / `task_index`。
- 目录结构：`data/chunk-000/episode_XXXXXX.parquet`、`videos/chunk-000/<相机特征名>/episode_XXXXXX.mp4`、
  `meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl}`。
- 数据格式版本：`codebase_version = "v2.1"`，`chunks_size = 1000`。

## 5. 旁挂负载通道（**数据集本身不含负载**）

- 采集时由 `src/run_vla_collect_record.py` + `src/load_logging.py` **旁挂**记录 STS3215
  `Present_Load`，写成 `runs/week38/collect/sidecars/<repo>.jsonl`（每 repo 一份）。
- **关键事实：负载不在数据集里。** `meta/info.json` 的 `features` 只有 6 轴 `.pos` + 两路相机，
  **没有** `load` 通道；负载只在旁挂 jsonl 中，按 `(episode_index, frame_index)` 与数据集帧一一对应。
  因此既有 ACT/SmolVLA 训练脚本不受影响，但**要用负载必须自行 join 旁挂文件**。
- 旁挂行结构（见 `src/load_sidecar.py` 模块头）：
  `{"dataset_repo", "episode_index", "frame_index", "t", "load": {"<joint>.load": 千分比…},
  "load_raw": {…原始 2 字节…}, "positions": {"<joint>.pos": …}}`。
  `load` 为**解码后**（千分比，方向位已处理），`load_raw` 为原始读数；`positions` 与数据集
  `observation.state` 同源，用于事后对齐校验。
- 覆盖情况（逐 repo，实际读取统计）：每 repo 的旁挂行数 = 该 repo 总帧数，
  `gripper.load` **无空值**（`01_pick_a` 8787/8787、`01_place_left` 12284/12284、
  `02_pick_a` 8530/8530、`02_place_left` 11640/11640、`03_pick_a` 10207/10207、
  `03_place_left` 10870/10870），`decode_errors` 行数均为 0。
- 参考：负载量程与判据阈值见 `src/real_observation.py`
  （`GRIP_LOAD_THRESHOLD = 150.0‰`、`EMPTY_LOAD_BASELINE = -69.0`）。

## 6. 校验状态

- **旁挂 ↔ 数据集逐帧对齐：6/6 通过。** 用 `src/run_vla_collect_record.py --verify`
  的同一实现（`load_sidecar.verify_alignment`）复核：逐 repo 集数 20/20/19/21/20/20，
  位置最大偏差均为 **9.26e-07**（≤ 1e-3 容差，差异来自数据集 float32 存储）。
- 逐集 `length`（`meta/episodes.jsonl`）与 `meta/info.json` 的 `total_frames` 求和一致。

## 7. 已知偏差与采集纪律

**已知偏差**

- `demo_vla_02_pick_a` = **19 集**、`demo_vla_02_place_left` = **21 集**，偏离计划各 20 集。
  （`collect_tasks.json` 计划 20；实际以 `meta/info.json` 为准。）
- `runs/week38/label_vla/dry_01_pick_a/` 是 `run_vlm_label.py --dry-run` 的桩产物（`labels.jsonl`
  内容为桩输出），**不是**真实数据；六个真实 repo 的标注产物**齐全且对称**：
  `labels.jsonl` / `episode_metadata.jsonl` / `frames_manifest.json` / `review/review.html` 各一份。

**采集纪律**（见 `runs/week38/collect/collect_commands.ps1` 顶部说明）

- 物体位置要求**随机覆盖**左/中/右、近/远（"固定轨迹会让策略只会背路径"）。
- 录制是交互式的：每集按 →/Enter 结束，**坏集按 ← 重录**（`load_sidecar` 按集缓冲、
  成集落盘，重录会丢弃旧缓冲，不会在旁挂文件里留下被丢弃那一集的垃圾行）。
- `disable_torque_on_disconnect=false`（不卸力矩，避开夹爪过载时的 `Overload error!`）。
- `push_to_hub=false`（本批只落本地）。
- 每个 repo 采完**必须**跑一次 `--verify`，通过才进下一个 repo。

## 8. 已知局限（务必先读）

1. **无 `is_grasped` 等语义标注。** 数据集只含 6 轴位置 + 两路 RGB，**没有**任何"是否夹住"
   的语义列。任何成功/夹持判定都必须由使用方从位置或旁挂负载自行推导。
2. **`intent` 自动标注不可用（零判别力）。** `runs/week38/label_vla/<repo>/labels.jsonl` 的
   `intent` 由确定性关键词规则从 VLM 的 `task_text` 映射（`src/vlm_labeler.py`）；120 条里
   `intent` 预测分布为 `{pick: 119, none: 1}`，`place` **一次都没被预测**（而期望有 61 条）——
   零样本 VLM 在双意图数据上意图判别力为零。详见
   `runs/week38/label_vla/summary/summary.md`（A3 段）。
3. **`success` 是"位置判据"而非人工/VLM 视觉判定。** labels.jsonl 里的 `success` 来自
   `vlm_labeler.success_from_sensor`（episode 末 5 帧夹爪位置），**数据集不含负载**时该判据
   对薄件可能漏判；本批虽有旁挂负载，但标注链路当时未接入负载。
4. **旁挂负载不随数据集分发。** 只在本仓库 `runs/week38/collect/sidecars/`（`runs/` 已被 gitignore，
   不会提交）—— 重新分发数据集时需一并交接旁挂文件，否则负载通道丢失。
5. **无 parquet 级完整性哈希 / 无 `git_commit`**：数据集的 `meta/info.json` 未记录内容哈希，
   无法用哈希校验"未被改动"。
6. **逐集成功必须用"存在某帧夹住且抬起"，不能看末帧或单帧。** 推荐口径（与真机闭环判据同源）：
   `aggregate_episode_success(states, loads)` —— 只要存在一帧同时满足
   `5.5 < gripper.pos < 15.0`（夹住带）与 `wrist_flex ≥ 75`（已抬起）即判整集成功，
   之后松手/复位不影响结论。实测该口径在 120 集上判 **105 集成功**（位置 OR 负载；纯位置
   口径为 104 集）；按 `pick_a` 单看 59 集与 VLM 逐集标签交叉为 TP 37 / FN 15 / FP 7 / TN 0
   （准确率 0.627）。
   **两个易踩的坑**：① 夹住带在整集里只占 **1%–60%**（中位数 5.8%），按 stride 40 抽帧时
   某些集可能一帧阳性都抽不到，F1 会无定义地变成 0；② 数据集里的 `action` 是**目标**而不是
   实际位置，逐帧标注 `success` 与逐集结论不可直接互推。

## 9. 数据溯源（claim → 文件）

| 字段 | 取值 | 来源文件 |
|---|---|---|
| 集数 / 总帧数 | 见 §1 | `meta/info.json`（`total_episodes` / `total_frames`）、`meta/episodes.jsonl` |
| 指令文本 | 见 §2 | `meta/tasks.jsonl`、`meta/episodes.jsonl[].tasks`、`collect_tasks.json` |
| 相机索引/分辨率/fps | 见 §3 | `collect_commands.ps1`（相机 JSON）、`meta/info.json.features`、`camera_check.txt` |
| 状态/动作维度 | 6 维 | `meta/info.json.features.{action,observation.state}` |
| 旁挂负载 | `sidecars/<repo>.jsonl` | `src/load_sidecar.py`、`src/load_logging.py` |
| 对齐校验 | 6/6 通过 | `src/run_vla_collect_record.py --verify`（`load_sidecar.verify_alignment`） |

> 未记录/不确定的字段：数据集的采集起止**绝对时间**未写入 `meta/info.json`；
> 每台相机的硬件型号/序列号未记录。
