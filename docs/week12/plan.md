# 第 12 周实施计划 — 低维 Behavior Cloning 训练与独立评测

## 1. 任务分解

| 序号 | 子任务 | 文件 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化状态映射、episode 切分、checkpoint 和独立评测协议 | `docs/week12/design.md` | 第 11 周数据 |
| 2 | 实现 HDF5 校验、完整 observation 重建、18 维特征提取和 episode 切分 | `src/behavior_cloning.py` | 1 |
| 3 | 实现归一化、任务独立 MLP、训练和最佳权重保存 | `src/behavior_cloning.py` | 2 |
| 4 | 实现 checkpoint 安全加载和 `BCPolicy` | `src/behavior_cloning.py` | 3 |
| 5 | 实现未见 seed 的真实 rollout 与评测摘要 | `src/behavior_cloning.py` | 4 |
| 6 | 实现训练与正式验收 CLI | `src/run_bc_training.py` | 3–5 |
| 7 | 编写数据、训练、checkpoint、rollout 与 CLI 测试 | `src/tests/test_behavior_cloning.py` | 2–6 |
| 8 | 小样本训练/rollout，修正数据或运行时偏差 | 忽略的 `runs/` | 1–7 |
| 9 | 正式训练并执行 100 次独立 rollout | 忽略的 `runs/maniskill_bc/week12/` | 8 |
| 10 | 同步执行状态、设计实际偏差和总开发计划 | `docs/week12/*`、总计划 | 9 |

## 2. 文件结构

**新增文件：**

- `src/behavior_cloning.py` — 数据、归一化、MLP、训练、checkpoint、策略和 rollout 评测。
- `src/run_bc_training.py` — 第 12 周训练与验收 CLI。
- `src/tests/test_behavior_cloning.py` — 第 12 周纯逻辑和 fake-environment 测试。
- `docs/week12/design.md` — 第 12 周设计文档。
- `docs/week12/plan.md` — 第 12 周实施计划。

**修改文件：**

- `具身智能完整项目_业余开发计划.md` — 验收后更新第 12 周状态、代码结构和下一步。
- `docs/week12/design.md`、`docs/week12/plan.md` — 回填实际偏差和执行状态。

**明确不修改：**

- 第 10 周 scripted policy、控制模式与任务阈值。
- 第 11 周正式 HDF5/JSON 产物和采集代码。
- RGB、ACT、action chunk、视觉依赖或真实硬件接口。

## 3. 实施步骤

### 步骤 1：数据契约与切分

- 对照 summary、metadata 和 HDF5 逐 episode 校验。
- 从完整 env state 重建 observation，再提取与 live policy 一致的 18 维任务状态。
- 固定 episode 级 80/20 切分；保留 ID 与 transition 统计。
- 明确检测第 11 周 observation 叶子宽度损失，不把标量叶子作为完整输入。

### 步骤 2：归一化与训练

- 每任务只用训练 transition 计算 state/action mean/std。
- 建立 18→128→128→4 的 MLP，隐藏层 ReLU，输出保持线性。
- 使用 AdamW + MSE、固定随机 seed 和 CPU 确定性训练。
- 每 epoch 计算 train/validation loss，保存最低 validation loss 的权重。

### 步骤 3：checkpoint 与策略

- 保存 tensor、配置、数据契约、切分证据和最佳 loss。
- 通过 `weights_only=True` 重新加载并验证 schema/shape。
- `BCPolicy` 复用训练归一化，输出反归一化并裁剪为四维有限 action。

### 步骤 4：真实 rollout

- 每个任务使用 seed 100–149，逐 episode 创建/关闭 CPU 环境。
- 复用 `run_episode_in_env()` 的 Gymnasium 契约和 success 判定。
- 汇总每任务 50 次及总体 100 次成功率，写入原子 JSON。
- 低于 70% 明确失败，不用 scripted fallback 掩盖 BC 指标。

### 步骤 5：测试与正式验收

- 先用合成 HDF5 和 fake env 覆盖纯逻辑。
- 再用正式数据做 loader 和短 epoch 训练 smoke。
- 正式训练后从磁盘重新加载 checkpoint，执行完整独立评测。
- 运行定向测试、完整 pytest、`compileall`、`pip check` 和 `git diff --check`。
- 按真实结果回填设计偏差、计划状态和总开发计划。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 第 12 周定向 | `.venv-maniskill/bin/python -m pytest -q src/tests/test_behavior_cloning.py` | 数据、训练、checkpoint、rollout、CLI |
| 正式数据 smoke | `.venv-maniskill/bin/python src/run_bc_training.py --epochs 5 --eval-episodes-per-task 2 ...` | 真实数据和环境全链路 |
| 正式训练与评测 | `.venv-maniskill/bin/python src/run_bc_training.py ...` | 80/20 切分、checkpoint、100 次独立 rollout、≥70% |
| 完整回归 | `.venv/bin/python -m pytest -q` | 第 1–11 周功能无回归 |
| 独立环境回归 | `.venv-maniskill/bin/python -m pytest -q --ignore=src/tests/test_llm_planner.py` | 含 HDF5/PyTorch 的非 LLM 全量测试；该环境未安装可选 `openai` |
| LLM 专属回归 | `.venv/bin/python -m pytest -q src/tests/test_llm_planner.py` | 在已安装可选 `openai` 的基础环境验证 3 项 LLM 测试 |
| 语法检查 | `.venv/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 依赖检查 | `.venv-maniskill/bin/python -m pip check` | 仿真/训练环境依赖一致 |

## 5. 验收标准

- [x] 每任务 80 个训练 episode、20 个验证 episode，ID 无重叠。
- [x] 训练特征是从完整 env state 重建、与 live policy 一致的 18 维任务状态。
- [x] 归一化仅来自训练集，checkpoint 可独立加载。
- [x] seed 100–149 的 100 次独立 rollout 全部完成并落盘。
- [x] 总体独立评测成功率 ≥70%，同时报告分任务成功率。
- [x] 运行产物只位于已忽略的 `runs/`。
- [x] 定向测试、完整回归、`compileall`、`pip check` 和差异检查通过。
- [x] 总计划、设计实际偏差和执行状态完成同步。

## 6. 执行状态

第 12 周实现与验收已完成：

- 新增 `src/behavior_cloning.py`，完成第 11 周 HDF5 布局校验、env state 逐帧 observation 重建、18 维任务状态、episode 级 80/20 切分、训练集归一化、双任务 MLP、最佳 validation checkpoint、`weights_only` 安全加载和真实 rollout。
- 新增 `src/run_bc_training.py`，默认执行 400 epoch 训练和 seed 100–149 的每任务 50 次独立评测；总体低于 70% 返回非零，BC 评测不使用 scripted fallback。
- 新增 `src/tests/test_behavior_cloning.py`，10 项测试覆盖配置、数据映射证据、episode 切分、live shape、checkpoint 重载、资源关闭、输出保护和 CLI 错误路径。
- 正式训练使用 PickCube 1512/392、PushCube 1085/268 个 train/validation transitions；最佳 validation MSE 分别为 `0.146417` 和 `0.019004`。
- 正式产物位于忽略目录 `runs/maniskill_bc/week12/`：checkpoint 从磁盘重载后，PickCube 42/50、PushCube 50/50，总体 92/100（92%），超过 70% 门禁，100 次 rollout 无运行异常。
- 基础环境完整回归 293 项通过、4 项可选 HDF5/PyTorch 测试跳过；独立 ManiSkill 环境排除缺少可选 `openai` 的 LLM 专属测试后 303 项通过，基础环境的 3 项 LLM 测试单独通过；`compileall`、`pip check` 和 `git diff --check` 通过。
