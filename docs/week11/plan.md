# 第 11 周实施计划 — ManiSkill 专家轨迹采集与回放

## 1. 任务分解

| 序号 | 子任务 | 文件 | 依赖 |
| --- | --- | --- | --- |
| 1 | 固化数据格式、成功过滤、回放和安全边界 | `docs/week11/design.md` | 第 10 周实现 |
| 2 | 增加已有环境 episode runner，保持原接口兼容 | `src/maniskill_intro.py`、原测试 | 1 |
| 3 | 实现配置、记录环境、采集循环和汇总 | `src/expert_trajectory.py` | 1、2 |
| 4 | 实现 HDF5/JSON 静态检查和真实 action replay | `src/expert_trajectory.py` | 3 |
| 5 | 实现采集与回放 CLI | `src/run_expert_collection.py` | 3、4 |
| 6 | 编写纯逻辑和 fake-environment 回归测试 | `src/tests/test_expert_trajectory.py`、原测试 | 2–5 |
| 7 | 小样本真实仿真验证并修正运行时偏差 | 忽略的 `runs/` | 1–6 |
| 8 | 采集 200 条并回放 50 条，完成全量验收 | 忽略的 `runs/` | 7 |
| 9 | 同步执行状态、实际偏差和总开发计划 | `docs/week11/*`、总计划 | 8 |

## 2. 文件结构

**新增文件：**

- `requirements-maniskill-dev.txt` — 复用独立 ManiSkill 依赖并补充测试工具。
- `src/expert_trajectory.py` — 轨迹配置、采集、静态检查、回放和汇总。
- `src/run_expert_collection.py` — 第 11 周 CLI 入口。
- `src/tests/test_expert_trajectory.py` — 采集与回放纯逻辑测试。
- `docs/week11/design.md` — 第 11 周设计文档。
- `docs/week11/plan.md` — 第 11 周实施计划。

**修改文件：**

- `src/maniskill_intro.py` — 提取不负责关闭环境的 episode 执行入口。
- `src/tests/test_maniskill_intro.py` — 验证新入口资源所有权和原接口兼容。
- `具身智能完整项目_业余开发计划.md` — 验收后更新第 11 周状态、代码结构和下一步。
- `docs/week11/design.md`、`docs/week11/plan.md` — 回填实际偏差和执行状态。

**明确不修改：**

- 第 10 周 scripted policy 阈值、动作和任务配置。
- 第 1–9 周稳定模块。
- 基础 `.venv` 的依赖集合。
- 视觉观测、训练代码、模型或 checkpoint。

## 3. 实施步骤

### 步骤 1：复用 episode 执行

- 从 `run_episode()` 提取 `run_episode_in_env()`。
- 调用方传入环境与策略；函数 reset 并返回相同 `EpisodeResult`。
- 保持 `run_episode()` 的创建、异常关闭和结果完全兼容。

### 步骤 2：建立记录环境

- 延迟导入 ManiSkill `RecordEpisode`，普通测试无需安装 HDF5。
- 基于现有 `make_maniskill_env()` 建立单 CPU 记录环境。
- 固定 `save_video=False`、`record_env_state=True`、`record_reward=True`。
- 使用任务稳定短名，分别生成 `pick_cube.h5/json` 和 `push_cube.h5/json`。

### 步骤 3：实现成功轨迹采集

- 每任务复用一个环境，以 `seed_start + attempt_index` 递增 seed。
- 每个 attempt 创建/重置 scripted policy 并执行。
- success 才 flush 保存；失败只计数并丢弃 buffer。
- 达到目标或最大尝试后关闭环境，聚合任务结果。
- 原子写入总 `summary.json`。

### 步骤 4：静态完整性检查

- 延迟导入 `h5py`。
- 对照任务 JSON metadata 与 HDF5 trajectory group。
- 校验数量、ID、control mode、success、action shape、有限数和 `T/T+1`。
- 将检查统计纳入总汇总。

### 步骤 5：真实回放

- 每任务均衡选择 episode，余数按任务顺序分配。
- reset 对应 seed，加载初始 env state，执行原 action。
- 严格记录最终 success 和终止语义，不用中间 state 掩盖漂移。
- 原子写入 `replay_summary.json`。

### 步骤 6：CLI 与测试

- CLI 默认执行 100+100 条采集和 50 条回放。
- 支持小样本调试参数和显式跳过回放。
- fake recorder 覆盖成功过滤、失败重试和关闭。
- 临时 HDF5 覆盖结构检查；真实 ManiSkill 不进入普通 pytest。

### 步骤 7：真实验收与同步

- 先以每任务 2 条、回放 2 条验证真实 API。
- 再运行默认 200 条采集、50 条回放。
- 运行定向测试、完整 pytest、`compileall` 和独立环境 `pip check`。
- 检查产物大小、字段、失败数和 Git 差异。
- 回填设计实际偏差、计划执行状态和总开发计划。

## 4. 测试计划

| 测试类型 | 命令/范围 | 验证目标 |
| --- | --- | --- |
| 第 10 周回归 | `.venv/bin/python -m pytest -q src/tests/test_maniskill_intro.py` | episode runner 兼容和资源所有权 |
| 独立测试依赖 | `.venv-maniskill/bin/python -m pip install -r requirements-maniskill-dev.txt` | ManiSkill/HDF5 与 pytest 共用 Python 3.11 环境 |
| 第 11 周定向 | `.venv-maniskill/bin/python -m pytest -q src/tests/test_expert_trajectory.py` | 配置、采集、HDF5 检查、回放 |
| 小样本真实验证 | `.venv-maniskill/bin/python src/run_expert_collection.py --episodes-per-task 2 --replay-count 2` | 真实记录与回放 API |
| 默认真实验收 | `.venv-maniskill/bin/python src/run_expert_collection.py` | 200 条成功轨迹、50 条成功回放 |
| 完整回归 | `.venv/bin/python -m pytest -q` | 第 1–10 周功能无回归 |
| 语法检查 | `.venv/bin/python -m compileall -q src` | 所有 Python 文件可编译 |
| 依赖检查 | `.venv-maniskill/bin/python -m pip check` | 独立仿真环境依赖一致 |

## 5. 验收标准

- [x] 两任务各至少 100 条、总计至少 200 条成功轨迹。
- [x] 每条轨迹含 env state、state observation、action、seed 和 success。
- [x] 失败尝试未混入监督数据集。
- [x] 200 条静态完整性检查全部通过。
- [x] 50 条初始 state + action 真实回放全部成功。
- [x] 未保存 RGB/视频，全部运行产物位于忽略目录。
- [x] 定向测试、完整回归、`compileall` 和 `pip check` 通过。
- [x] 总计划、设计偏差和执行状态完成同步。

## 6. 执行状态

第 11 周实现与验收已完成：

- 新增 `src/expert_trajectory.py`，复用 ManiSkill 标准 `RecordEpisode`，完成成功过滤、HDF5/JSON 静态检查、确定性均衡抽样和初始 env state + action replay。
- 新增 `requirements-maniskill-dev.txt`，确保 HDF5 定向测试可在独立 Python 3.11 仿真环境复现，不向运行时依赖混入测试工具。
- 新增 `src/run_expert_collection.py`，默认采集 PickCube/PushCube 各 100 条成功轨迹并回放 50 条；非空输出目录受保护，macOS 仿真启动前固定绝对输出路径。
- 第 10 周 `run_episode()` 外部行为不变；新增 `run_episode_in_env()` 允许第 11 周在同一环境内高效采集多条轨迹。
- 正式数据位于忽略目录 `runs/maniskill_expert/week11-200/`：200/200 条成功且静态检查通过，3257 个状态动作 transition，失败 attempt 为 0，50/50 条真实回放成功，产物约 3.1 MB。
- 独立 ManiSkill 环境定向测试 30 项通过；基础环境完整回归 293 项通过、3 项可选 HDF5 测试跳过且已在独立环境通过；`compileall`、`pip check` 和 `git diff --check` 通过，可以进入第 12 周低维 BC。
