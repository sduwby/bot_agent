# 第 36 周 实施计划：分段补齐（S1 视觉输入先行）— 已执行至 S11

> 状态：S1（视觉输入）与 S2–S11 全部实现并测试；S12（VLA 落地）与 S13（交付物）需
> 多任务数据/机械臂实测，标注为遗留。

## 执行记录（S1 ✅）

1. ✅ `llm_planner.py`：`image_to_data_url` / `build_multimodal_content` / `OpenAIBackend.chat(images=...)`；RuleFallback 兼容新签名
2. ✅ `real_executor.py`：RobotEngine 增 `read_images()`；RealEngine 实现取两路帧；MockEngine 合成帧
3. ✅ `real_agent.py`：每轮携带相机帧，读图异常降级纯文本
4. ✅ `run_real_agent.py`：`--no-vision` / `--mock-images`；.env 加载
5. ✅ 单元测试（图像编码/多模态 content/降级路径）+ 回归
6. ✅ 真实 DeepSeek Vision dry-run 开/关验证通过

## 执行记录（S2–S11 ✅）

| Seg | 模块 | 完成项 | 验证 |
|---|---|---|---|
| S2 | `real_executor.RealEngine` | `max_relative_target` 参数接线（低速限幅） | compile+CLI |
| S3/S4 | `real_router.py` | PolicySource / ScriptedPolicy / RealSkillRouter / default_real_route（ACT→scripted 回退） | 单测 5 项 |
| S5 | `real_planner.py` | parse_plan / validate_plan / format_plan / execute_plan（技能序列 schema+校验） | 单测 6 项 |
| S6/S7 | `real_perception.py` | describe_workspace（多模态结构化问答）/ local_holding_state | 单测 3 项 |
| S8 | `real_recovery.py` | RealRecoveryManager + detect_loop；接入 real_agent 主循环 | 单测 6 项 + agent 回归 |
| S9 | `real_memory.py` | RealMemoryStore（JSONL 记忆/检索/注入） | 单测 2 项 |
| S10/S11 | `real_evaluation.py` + `run_real_eval.py` | EvalTrial/RealEvalRunner（CSV+report.md）；§9.3 verdict→失败维度映射；任务集 `evaluation/real_tasks.json` | 单测 2 项 + CLI dry-run 5/5 |

**测试汇总**：`test_real_modules(22) + test_real_agent_core + test_real_skills + test_llm_planner = 55 passed`；`compileall src` 通过。

## 遗留（S12/S13 与真机实测）

- S12 VLA 自然语言直驱：仍为占位（`run_real_agent --mode vla`），需多任务数据+SmolVLA 微调后实现；
- 真机实跑：skills 闭环一次、`run_real_eval.py`（非 dry-run）20 轮、感知/视觉验证真机链路，需机械臂就绪后执行；
- S13 交付物：README/架构图/数据卡/模型卡/演示，留待真机结果后整理。

## 验收标准（已完成核对）

1. pytest 真机相关 55 项全过 ✅
2. `--mode skills --dry-run`（视觉开/关）真实 DeepSeek 决策正确 ✅
3. `read_images` 异常降级不影响主循环 ✅
4. E2E harness dry-run 生成 trials.csv + report.md ✅
