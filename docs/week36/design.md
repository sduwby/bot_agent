# 第 36 周 设计文档：阶段 F 分段补齐（S1 起）——视觉输入与多段 Backlog

## 1. 概述

- **目标**：按"差距矩阵 → 分段补充（每段：设计文档 → 实现 → 测试）"推进阶段 F，直至达到第 40 周交付标准。
- **S1（本周首段）**：给 LLM Agent 接入**图像观测输入**——当前 DeepSeek 模型为 `deepseek-v4-flash-vision-exp`（视觉），但 Agent 只喂文本摘要。S1 打通"fixed/handeye 相机帧 → base64 → OpenAI 兼容多模态 content → LLM 决策"链路。
- 规划原文：§10.1 Perception / State（维护当前可见对象）+ 第 36 周 LLM Planner、§10.3 端到端任务。

## 2. 差距矩阵（盘点结论，阶段 F 需补模块）

已完成盘点：对照开发计划 23–40 周承诺 vs 仓库现状。**已覆盖**：技能框架 6 模块（real_*）、真机闭环原语（infer_so101）、部署评测协议骨架（deploy_eval）、demo_v23 数据/act_demo_v23 模型、LLM 后端（llm_planner）。

**待补（按依赖顺序，即后续 Segments）**：

| Seg | 模块 | 对应规划 | 现状 | 备注 |
|---|---|---|---|---|
| S2 | 真机 Safety 限速接线（RealEngine/infer 启用 max_relative_target） | §10.1 Safety | 仅 deploy_eval 有参数 | 一行级，先做 |
| S3 | 真机 scripted 确定性 pick 原语 | 第 29 周遗留 | 无 | 借 DAGGER 思路新写点位序列 |
| S4 | 真机多策略路由 real_skill_router（ACT/scripted/停滞监测/fallback） | 第 33 周 | 仿真版 skill_router | 重写为真机 obs dict 契约 |
| S5 | LLM 计划层 real_planner（技能序列 schema + 计划验证） | 第 36 周 | 现为单步决策 | 改单步为计划 |
| S6 | 真机感知层 real_perception（可见物体/区域/状态） | 第 37 周 | 仅关节摘要 | fixed 相机+区域原语 |
| S7 | 真机视觉验证扩展（放置/区域/夹持 + VLM） | 第 37 周 | 阈值判定仅 grasp/lift | |
| S8 | Recovery Manager real_recovery + action_monitor 接线 | 第 38 周 | 内联重试 1 次 | 失败码→恢复动作表 |
| S9 | 真机记忆 real_memory（JSONL 历史/失败/末态） | 第 38 周 | 仿真版 memory_* | |
| S10 | 真机端到端评测 harness real_evaluation | 第 39 周 | 仿真版 run_evaluation | NL 任务集批量执行 |
| S11 | 阶段 E 20 轮评测补跑 + §9.3 五维失败分类增强 | 第 30/31 周挂起 | deploy_eval 已就绪 | 执行项 |
| S12 | VLA 模式落地（SmolVLA 真机微调） | 第 32 周推迟 | VLARunner 占位 | 依赖多任务数据 |
| S13 | 交付物：README/架构图/数据卡/模型卡/演示/复盘 | 第 40 周 | 无 | |

依赖顺序：S2 → S1/S3 → S4 → S5 → S6/S7 → S8/S9 → S10 → S11/S12 → S13。
（注：S1 视觉输入与 S5/S6 有关联，先落地通道，感知语义后续增强。）

## 3. S1 架构设计

### 现状
`llm_planner.OpenAIBackend.chat(system, user)`：纯文本，`response_format=json_object`，无图像。
`real_agent.RealRobotAgent` 只把关节文本摘要放入 prompt。

### 设计
```text
[RealEngine.read_images()]  →  dict{fixed: ndarray RGB, handeye: ndarray RGB}
        │  (真机: build_observation_from_robot 取两路帧；Mock: 合成小图或 None)
        ▼
[llm_planner.image_to_data_url()]   ndarray/PIL/bytes → data:image/jpeg;base64,...
        ▼
[OpenAIBackend.chat(system, user, images=...)]
        messages: user content = [ {text}, {image_url:data_url} x N ]
        有图时不强制 json_object（部分视觉模型不支持），改靠 system prompt+parse 容错
        ▼
[real_agent] 每轮决策把当前两路帧一并送审；engine 读图失败→降级纯文本
```

### 关键决策
- **协议兼容**：`LLMBackend.chat` 增可选参 `images: list | None`；`RuleFallback`/`FakeBackend` 同步签名，不影响既有单测。
- **图像来源解耦**：`RobotEngine` Protocol 增 `read_images()`；Real 实现走相机，Mock 可注入合成帧，纯逻辑测试无需真机。
- **视觉模型适配**：带图请求用 `content` parts 结构；保留文本路径（无图时不改变现有 JSON mode 行为），避免回归。
- **降级策略**：引擎读图异常 → `images=None` 纯文本继续，不阻断 Agent 主循环。

## 4. 数据结构

- `OpenAIBackend.chat(system: str, user: str, images: list[str] | None = None) -> str`（images 为 data URL 列表，也可接受 ndarray/PIL/bytes 由函数统一编码）
- `RobotEngine.read_images() -> dict[str, Any] | None`：`{"fixed": <HWC uint8 RGB>, "handeye": <HWC uint8 RGB>}`
- Agent 每轮收集：`observation_summary`（文本）+ `images`（两路帧 data URL）
- `.env` 新增可选 `LLM_VISION=1/0`（默认按需；0 禁用图像以省 token/费用）

## 5. 数据流

```
闭环每轮:
  read_state()/read_images() → summarize_state + images
  → build_user_prompt(文本) + images
  → OpenAIBackend.chat(multimodal) → JSON decision → execute/verify（不变）
```

## 6. 测试与验收（S1）

- 单元：`image_to_data_url`（ndarray/PIL/bytes/坏输入）、多模态 content 构造含 text+image_url、RuleFallback/FakeBackend 接受 images 参数不破坏
- Agent 层：Mock 引擎带合成帧 → 断言 backend 收到 images；引擎 read_images 抛异常 → 降级纯文本仍完成
- CLI：`--mode skills --dry-run` 走真实 DeepSeek Vision，指令触发 grasp_screwdriver；另加 `--no-vision` 对照
- 验收：Dry-run 两轮（开/关视觉）均正常决策；相关回归 pytest 全过

## 7. 实际偏差（S1 结束回填）

1. 有图请求统一**不使用** `response_format=json_object`（视觉模型对 JSON mode 支持不一），改为靠 system prompt 的 JSON 输出约束 + `parse_decision` 容错提取；实测 DeepSeek vision 两种路径都能返回 JSON，选容错路径更稳。
2. `LLMBackend`/`RuleFallback`/测试后端统一增加 `images` 形参（默认 None），不破坏既有纯文本调用。
3. dry-run 首跑返回空内容，为旧 `__pycache__` 字节码所致，`compileall` 后恢复正常；无代码问题。
4. 仅传文本时保留 `response_format=json_object`，与 S1 前行为一致，无回归。

## 8. 执行结果（S2–S11 回填，2026-09-06）

- 新增模块：`src/real_router.py`（S3/S4）、`real_planner.py`（S5）、`real_perception.py`（S6/S7）、
  `real_recovery.py`（S8）、`real_memory.py`（S9）、`real_evaluation.py` + `src/run_real_eval.py`（S10/S11）；
- 接线：`RealEngine` 增 `max_relative_target`（S2 安全限幅）与 `move_to` 点位原语；`real_agent` 内联恢复
  逻辑替换为 `RealRecoveryManager`+`detect_loop`（S8）；新增任务集 `evaluation/real_tasks.json`。
- 偏差：scripted（S3）当前为"安全复位回退"原语（release+home），不含"按点位夹取"的确定性序列——后者依赖
  真机目标定位/标定，标注为真机实跑阶段补充；`StagnationMonitor` 仿真版的"TCP 位移停滞"判定因真机无
  tcp_pose 外设暂不迁移，由循环检测（detect_loop）替代。
- 验证：真机相关 55 项测试通过；`run_real_eval.py --dry-run` 生成 trials.csv+report.md；`compileall src` 通过。
