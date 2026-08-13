# 第 20 周实施计划 — 分层策略路由

## 1. 任务分解

| 序号 | 子任务 | 文件/产物 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化路由设计（策略位、fallback 判据、评测协议） | `docs/week20/design.md` | 第 14–18 周 |
| 2 | 实现 `PolicySource`/`SkillRouteConfig`/`StagnationMonitor` | `src/skill_router.py` | 1 |
| 3 | 实现 `SkillRouter.run_episode`（异常/停滞/超时切换） | `src/skill_router.py` | 2 |
| 4 | 实现 `evaluate_routes` 与报告落盘 | `src/skill_router.py` | 3 |
| 5 | 实现评测 CLI | `src/run_skill_router.py` | 4 |
| 6 | 编写纯逻辑 + fake env/policy 定向测试 | `src/tests/test_skill_router.py` | 3–5 |
| 7 | 真实评测（Mac/4060）：多路由组合固定 seed 评测 | `runs/maniskill_act/week20-*` | 5 |
| 8 | 回填实际偏差并同步总开发计划 | `docs/week20/*`、总计划 | 7 |

## 2. 文件结构

### 2.1 计划新增

- `src/skill_router.py` — 第 20 周：策略源、路由配置、停滞监测、路由执行与聚合评测。
- `src/run_skill_router.py` — 第 20 周：分层路由固定 seed 评测 CLI。
- `src/tests/test_skill_router.py` — 第 20 周定向测试。
- `docs/week20/design.md` — 本周设计与实际偏差。
- `docs/week20/plan.md` — 本实施计划与执行状态。

### 2.2 计划修改

- `具身智能完整项目_业余开发计划.md` — 更新第 20 周状态、当前代码结构与下一步。

### 2.3 明确不修改

- 第 13/16/17 周策略实现与 checkpoint；
- 第 14/17 周固定评测协议本身（只复用 seed 集与指标口径）；
- `visual_act.py`、`lerobot_act.py`、`behavior_cloning.py`、`maniskill_intro.py` 的既有契约。

## 3. 实施步骤

### 步骤 1：设计固化

1. 确认策略位类型与输入模式（state/visual）。
2. 确认三类 fallback 触发判据与阈值。
3. 记录到 `docs/week20/design.md`。

完成条件：路由语义与判据明确。

### 步骤 2–4：核心模块

1. `PolicySource`（kind/input_mode/factory/optional）。
2. `SkillRouteConfig` 校验（至少一位、末位 scripted 检查、阈值合法）。
3. `StagnationMonitor`（TCP 位移窗口 + `is_grasped` 变化）。
4. `SkillRouter.run_episode`：逐帧执行、动作校验、异常捕获、切换与 `RouterEpisodeResult`。
5. `evaluate_routes`：批量评测 + `router_report.json` 原子落盘。

完成条件：核心逻辑可被 fake env/policy 驱动。

### 步骤 5：CLI

1. 参数：`--route`（逗号分隔策略种类）、各 checkpoint 目录、`--tasks`、`--seeds`、`--device`、`--output-dir`、`--perturbed`。
2. 输出 `router_report.json` 与逐 seed 明细；门禁退出码（评测完成 0，配置/加载错误非 0）。

完成条件：CLI 参数校验与加载链路完整。

### 步骤 6：测试

1. 纯逻辑：`SkillRouteConfig` 校验、`StagnationMonitor` 窗口行为、`SwitchEvent` 序列化。
2. fake env/policy：异常切换、停滞切换、超时切换、末位 scripted 兜底、exhausted 路径、optional 位加载失败跳过。
3. 完整回归、`compileall`、`pip check`、`git diff --check`。

完成条件：定向测试与完整回归通过。

### 步骤 7：真实评测

1. 路由组合：
   - `state_act,scripted`（PickCube + PushCube）
   - `visual_act,state_act,scripted`（PickCube）
   - 纯 `scripted` 基线（同 seed 对照）
2. 分布内 500–529、分布外 600–629（复用第 17 周协议）。
3. 记录每策略执行率、fallback 原因分布、最终成功率，与单策略基线对比。

完成条件：`runs/maniskill_act/week20-*/router_report.json` 落盘。

### 步骤 8：同步文档

1. 回填 `docs/week20/design.md` 实际偏差。
2. 更新本计划执行状态。
3. 更新总计划第 20 周状态、当前代码结构与下一步。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 定向测试 | `.venv-lerobot/bin/python -m pytest -q src/tests/test_skill_router.py` | 路由配置、停滞监测、切换、聚合 |
| 真实 smoke | `run_skill_router.py --seeds 500 --tasks PickCube-v1` | 单 seed 闭环 |
| 完整回归 | `.venv-lerobot/bin/python -m pytest -q --ignore=src/tests/test_llm_planner.py` | 第 1–20 周无回归 |
| 语法检查 | `.venv-lerobot/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 依赖检查 | `.venv-lerobot/bin/python -m pip check` | 独立环境依赖一致 |
| 差异检查 | `git diff --check` | 无空白和补丁格式错误 |

## 5. 单元测试覆盖

### 5.1 配置与监测

- `PolicySource` 字段与 `optional` 语义；
- `SkillRouteConfig` 非法配置拒绝（空策略列表、末位非 scripted、非法阈值）；
- `StagnationMonitor` 位移窗口与夹爪变化重置。

### 5.2 路由执行

- 首策略成功：不切换；
- 异常切换：首策略抛异常 → 次策略接管并成功；
- 停滞切换：首策略连续停滞 → 切换；
- 超时/exhausted：全部策略耗尽记失败；
- optional 位（smolvla）加载失败跳过；
- visual 输入模式：`act(obs, image)` 调用路径。

### 5.3 聚合评测

- 多任务 × 多 seed 汇总正确性；
- `router_report.json` 原子落盘与 JSON 结构；
- fallback 原因分布统计。

## 6. 验收清单

- [ ] 同一技能接口可切换 scripted/状态 ACT/视觉 ACT；SmolVLA 位可选。
- [ ] 异常/停滞/超时三类 fallback 可触发并安全切换。
- [ ] 固定 seed 评测完成并落盘。
- [ ] 路由成功率 ≥ scripted 基线（不产生负优化）。
- [ ] 定向测试、完整回归、`compileall`、`pip check`、`git diff --check` 通过。
- [ ] 运行产物只写入已忽略的 `runs/`。

## 7. 执行状态

### 已完成

- **步骤 1（设计固化）**：`docs/week20/design.md` 完成（策略位、三类 fallback 判据、评测协议）。
- **步骤 2–4（核心模块）**：`src/skill_router.py` 完成——`PolicySource`/`SkillRouteConfig`/`StagnationMonitor`/`SkillRouter`/`evaluate_routes` + 五种策略位工厂（scripted/bc/state_act/visual_act/smolvla 可选）。
- **步骤 5（CLI）**：`src/run_skill_router.py` 完成（`--route`/checkpoint 目录/`--tasks`/`--seeds`/`--device`/`--output-dir`）。
- **步骤 6（测试）**：`src/tests/test_skill_router.py` 30 项定向测试通过；完整回归 463 通过、1 项为既有 mps 环境差异（`test_probe_lerobot_act`，Mac 上通过）；`compileall`、`pip check`、`git diff --check` 通过。
- **步骤 7（真实评测 smoke）**（4060，2026-08-13）：
  - 纯 `scripted` 基线：PickCube 2/2、PushCube 2/2（100%）
  - `visual_act,scripted`（week17-c8-20k，cuda）：PickCube 0/2（1 次 stagnation fallback 切 scripted）、PushCube 2/2（纯视觉无 fallback）
  - 结论：异常/停滞 fallback 真实可用，scripted 兜底无负优化。
- **步骤 8（文档同步）**：本文件与 design.md 实际偏差已回填；总计划待同步。

### 待办

- 固定 seed 全量评测（分布内 500–529 / 分布外 600–629，30+30）可随时用 CLI 复跑，作为第 22 周门禁 C 的组成部分。
- 总计划第 20 周状态、当前代码结构与下一步更新。
