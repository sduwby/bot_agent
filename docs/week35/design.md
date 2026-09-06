# 第 35 周 设计文档：LLM Agent + 真机技能框架（阶段 F 起步）

## 1. 概述

- **本周目标**：把阶段 A 已验证的"LLM Planner → 技能库 → 执行器"架构从 AI2-THOR 迁移到真机 SO-101，落地阶段 F 的技能接口框架；保留"自然语言直驱 VLA"作为可选模式入口（默认关闭）。
- **规划原文**："第 35 周 技能封装：把真实策略封装为有超时、有错误码、有前置条件的技能 API。验收：可由脚本稳定组合调用。"
- **前置依赖**：week29 `outputs/infer_so101.py`（真机闭环原语）、week30 `outputs/deploy_eval_so101.py`（含空闭基准与结果判定逻辑）、`src/` 阶段 A 决策层代码。

### 决策记录（跳过中间周）

- 用户决定**跳过 30-34 周评测与补采**，直接进入阶段 F 开发；
- 优先做**技能框架**（主模式 `skills`）；同时**保留 VLA 自然语言直驱**入口（模式 `vla`），本阶段只建占位与模式开关，不实现推理逻辑；
- 技能执行底层复用已验证的 ACT 闭环（每步 `policy.reset()` 重推 + 固定步数 + 末尾保持），不重新发明。

## 2. 现状资产复用分析

| 来源 | 模块 | 复用策略 |
|---|---|---|
| `src/llm_planner.py` | LLMBackend / OpenAIBackend / RuleFallback / create_backend / parse_decision / build_user_prompt | ✅ 直接复用（逻辑层，无仿真 import）；System Prompt 文本重写为真机技能词表 |
| `src/skill_registry.py` | _SKILLS 结构 + get_skills_description / validate_parameters | ⚠️ 代码结构复用；技能清单按真机重建 |
| `src/action_monitor.py` | ActionMonitor.detect_loop | ✅ 直接复用（纯逻辑） |
| `src/failure_classifier.py` | FailureClassifier.classify_failure | ✅ 直接复用；真机失败文案按关键词扩充 |
| `src/replanner.py` | RecoveryAction / Replanner | ✅ 逻辑复用；恢复动作表按真机技能重映射 |
| `src/executor.py` | AgentExecutor 主循环骨架 | ❌ 深度绑定 ai2thor metadata/动作 dict；仅保留"决策循环 + 失败恢复"编排思想，真机版重写为 `RealRobotAgent` |
| `src/task_skills.py` | SkillExecutor / SkillResult | ⚠️ SkillResult 契约保留；实现换成真机技能执行器 |
| `src/verifier.py` | VerificationResult / SubtaskVerifier | ⚠️ 结果契约保留；真机验证改用关节/相机可观测信号 |
| `src/observation.py` | Observation / summarize | ⚠️ 输出给 LLM 的摘要契约保留；真机观测改关节+相机 |
| `outputs/infer_so101.py` | load_policy / home_to_initial / 闭环跑策略 / 保持 | ✅ 作为真机技能执行层原语直接复用 |
| `outputs/deploy_eval_so101.py` | 空闭基准 / 结果 classify | ✅ 方法抽取复用为"夹持验证" |

## 3. 架构设计

```text
run_real_agent.py  (新 CLI，--mode skills|vla)
│
├── mode=skills（默认，本周主体）
│    指令(中文) ──► [LLM Planner] ──► 技能名+参数 JSON
│                    │                  │
│                    │  每步技能执行     ▼
│                    │        [RealSkillExecutor]  (真机技能库，超时/错误码/前置条件)
│                    │                  │ 底层调用
│                    │        ┌─────────┴───────────────┐
│                    │        ▼                         ▼
│                    │  infer_so101 闭环原语      真机状态读取(关节/夹爪/空闭基准)
│                    │        │
│                    └──◄── [验证 / 失败分类 / Replanner 恢复] 闭环
│
└── mode=vla（预留入口）
     指令(中文) ──► [VLARunner 占位] ──► 提示"未实现"，返回计划结构（不驱动机械臂）
```

### 模块划分（新增文件，放 `src/` 与既有决策层同目录，便于复用 import）

| 模块 | 职责 |
|---|---|
| `real_skills.py` | 真机技能注册表（schema）+ 技能描述生成 + 参数校验（结构仿 skill_registry） |
| `real_executor.py` | `RealSkillExecutor`：实现每个真机技能（内部调用 infer_so101 组件） |
| `real_observation.py` | 真机观测收集 → 中文摘要（关节姿态/夹爪状态/空闭基准/最近结果） |
| `real_verifier.py` | 技能结果验证：夹持验证（空闭基准）、抬起验证（wrist）、成功/失败归类 |
| `real_agent.py` | `RealRobotAgent` 主循环：LLM 决策 → 技能分发 → 验证 → 恢复（迁移 executor 骨架） |
| `run_real_agent.py` | CLI：`--mode skills|vla`、`--instruction`、`--model-dir`、`--dataset-root` 等 |

> 真机代码与 `outputs/infer_so101.py` 的衔接：`real_executor.py` 内 `sys.path` 或 import 该模块（同机运行），保持闭环原语单一来源，不复制运动学/闭环代码。

## 4. 数据结构

### 技能清单（v1，真机可执行集合）

| skill | 参数 | 说明 | 失败码 |
|---|---|---|---|
| `grasp_screwdriver` | 无 | 执行 ACT 闭环夹起并抬离桌面螺丝刀（固定步数+保持） | `grasp_empty` / `not_lifted` / `timeout` / `comm_error` |
| `reset_pose` | 无 | 回到采集初始姿态（home_to_initial） | `home_timeout` |
| `verify_holding` | 无 | 检查夹爪当前是否夹住物体（空闭基准+夹爪位置） | `not_holding` |
| `release` | 无 | 张开夹爪放下（安全指令，先落到安全高度由人工执行） | `comm_error` |
| `done` | 无 | 结束任务，输出总结 | — |

### 状态观测（喂给 LLM 的摘要，`real_observation.py`）

```jsonc
{
  "task_instruction": "把螺丝刀夹起来",
  "joint_state": {"wrist_flex": 89.4, "gripper": 1.5, ...},
  "holding": true|false,          // 由空闭基准判定
  "last_skill_result": "success", // 或失败码
  "recovery_hint": "可重试 grasp_screwdriver"  // 由 failure_classifier/replanner 生成
}
```

### 技能执行返回（与阶段 A `SkillResult` 同构）

```python
@dataclass
class SkillResult:
    success: bool
    data: dict           # 含 skill/耗时/最终关节状态/验证细节
    failure_reason: str | None
```

## 5. 接口设计

```bash
# 主模式：技能框架（本周交付）
python run_real_agent.py --mode skills \
    --instruction "夹起桌面上的螺丝刀" \
    --model-dir outputs/act_demo_v23_pretrained \
    --dataset-root <demo_v23 根目录> --dataset-repo wangbangyan/demo_v23

# VLA 预留模式（占位，仅提示未实现并返回规划结构，不驱动机械臂）
python run_real_agent.py --mode vla --instruction "把螺丝刀放到盒子里"
```

`real_agent.py` 核心循环（伪码）：

```text
while not done and step < max_plan_steps:
    obs_summary = collect_and_summarize(robot)
    decision = llm.chat(system_prompt, build_user_prompt(instruction, obs_summary))
    skill_name, params = parse_decision(decision)
    if skill_name == "done": break
    validate_parameters(...) or ask LLM fix
    result = skill_executor.execute(skill_name, params)   # 每技能带超时
    record(history)
    if not result.success:
        category = classify_failure(skill_name, result.failure_reason)
        recovery = replanner.plan_recovery(category)      # retry/scan视角/abort/llm
        apply_recovery(recovery) or stop
    else:
        replanner.record_success(skill_name)
```

## 6. 数据流

```
用户指令 → LLM Planner(技能序列) → RealSkillExecutor → infer_so101 闭环原语 → 关节动作
        ▲                                                       │
        └── obs_summary(关节+夹爪+验证) ◄── real_verifier ◄──────┘
        失败 → failure_classifier → replanner → 恢复动作重入循环
```

## 7. 模式切换与 VLA 预留

- CLI `--mode skills|vla`；未指定默认 `skills`。
- `mode=vla` 仅注册 `VLARunner` 占位类（返回 NotImplemented 的结构化说明 + 未来需要的数据契约），确保将来接入 SmolVLA（week18 探针已确认 6D state/3 相机契约匹配）时无需改 CLI。

## 8. 安全与约束

- 每个技能内置超时（如 grasp ≤ 60s），超时归为 `timeout` 失败并保持姿态等人工接管；
- 每步执行前检查急停（Ctrl+C 中断传播）；
- `release` 只在人工确认安全时使用；默认执行完 `grasp_screwdriver` 后进入保持不自动松开（断开时除外）；
- LLM 决策为只读建议，动作执行始终受技能层前置条件与超时约束。
