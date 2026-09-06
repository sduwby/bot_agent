# 第 35 周 实施计划：真机技能框架 + LLM Agent（阶段 F 起步）

## 任务分解

1. 真机技能注册表 `src/real_skills.py`：技能 schema、描述生成、参数校验
2. 真机观测与验证 `src/real_observation.py` / `src/real_verifier.py`
3. 真机技能执行器 `src/real_executor.py`（封装 infer_so101 原语）
4. LLM Agent 主循环 `src/real_agent.py`（迁移 executor 骨架，复用 llm_planner/replanner/action_monitor/failure_classifier）
5. CLI 与 VLA 占位 `src/run_real_agent.py`（`--mode skills|vla`）
6. 单元测试（纯逻辑：技能 schema 校验、失败分类映射、replanner 恢复、VLA 占位返回结构）

## 文件结构

**新增**（`src/` 与既有决策层同目录）：
- `src/real_skills.py`
- `src/real_observation.py`
- `src/real_verifier.py`
- `src/real_executor.py`
- `src/real_agent.py`
- `src/run_real_agent.py`
- `src/tests/test_real_skills.py`
- `src/tests/test_real_agent_core.py`（主循环决策分发纯逻辑，mock 技能执行器）
- `docs/week35/design.md` / `plan.md`

**修改**：无（仅 import 复用既有模块与 `outputs/infer_so101.py` 原语）

## 实施步骤（按优先级）

1. `real_skills.py`：定义 v1 技能清单（grasp_screwdriver / reset_pose / verify_holding / release / done）、`get_real_skills_description()`、`validate_real_parameters()`；仿 skill_registry 纯函数风格 + 单测
2. `real_observation.py`：从关节/夹爪/空闭基准生成中文摘要；空闭基准逻辑复刻自 deploy_eval_so101（不复制运动学）
3. `real_verifier.py`：夹持判定（空闭基准）、抬起判定（wrist_flex）、失败码映射，复用 failure_classifier 关键词机制
4. `real_executor.py`：每个技能调用 infer_so101 的闭环原语/保持/归位；技能级超时；不直接驱动时由测试 mock
5. `real_agent.py`：迁移 executor 主循环骨架：LLM chat → parse_decision → validate → execute → verify → classify/replan；用 mock 技能执行器跑决策循环单测
6. `run_real_agent.py`：`--mode`、`--instruction`、`--model-dir`、`--dataset-root`、`--dataset-repo`、`--dry-run`（不连接机械臂，走 mock 执行器验证链路）
7. 语法检查 `python -m compileall -q src`；`run_real_agent.py --dry-run` 跑通
8. （真机验证另行预约）在机械臂就绪后以 `--mode skills --instruction "夹起螺丝刀"` 实跑一次

## 测试计划

- 纯逻辑单测（不依赖机械臂/Unity）：技能 schema 合法性、参数校验、失败码→恢复动作映射、主循环对"成功/失败/超限"三条路径的决策分发、VLA 占位模式返回结构
- `pytest src/tests/test_real_skills.py src/tests/test_real_agent_core.py`
- CLI `--dry-run` 端到端（mock 技能执行器）验证 skills 模式链路可走通、vla 模式给出"未实现"提示

## 验收标准

1. `run_real_agent.py --mode skills --dry-run` 用 mock 执行器跑通完整 LLM→技能→验证→恢复循环
2. `--mode vla` 可启动并返回结构化的"未实现占位"（不驱动机械臂、不报错崩溃）
3. 技能注册表 v1 五项齐全，参数校验与描述生成有单测覆盖
4. 失败闭环：技能返回失败码后能正确归类并按 replanner 生成恢复动作（retry/视角/abort）
5. 真机验证另行安排（机械臂就绪后实跑一次 grasp_screwdriver）

## 风险与回退

- LLM API Key 不可用时回退 RuleFallback（阶段 A 已验证），保证无 Key 也能演示决策循环
- infer_so101 原语 import 路径为 `outputs/`，若迁移到 `src/` 依赖冲突，则把闭环原语复制为 `src/real_engine.py` 并标记 deprecated
- VLA 模式仅为占位，不承诺任何推理；若未来启用需新增数据采集协议（列入 week36+ 待办）
