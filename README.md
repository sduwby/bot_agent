# bot_agent：桌面物品整理具身 Agent

**从 LLM 软件 Agent 到真实机械臂闭环系统** —— 一个为期 40 周、文档驱动、门禁评审的具身智能完整项目。

系统接收中文自然语言指令，通过双相机观察真实桌面，由 LLM 进行高层任务规划，调用经模仿学习训练的策略（ACT / SmolVLA）与确定性脚本技能完成抓取放置，并能识别失败、重新规划、记录完整轨迹。开发路线遵循「**先仿真、后硬件；先结构化状态、后纯视觉；先模仿学习、后强化学习**」，完整规划见 [具身智能完整项目_业余开发计划.md](具身智能完整项目_业余开发计划.md)。

## 核心特性

- **全链路闭环**：AI2-THOR 语义 Agent → ManiSkill 策略训练 → SO-101 真机采集部署 → LLM/VLM 决策层 → 量化评测，每一环都有代码、测试与门禁结论
- **分层决策架构**：LLM 只做秒级任务决策，10–50Hz 低层控制交给策略与控制器，互不侵入
- **模型风险分层**：Scripted 技能始终作为专家与安全回退，ACT 为主学习策略，SmolVLA 提供语言/视觉泛化
- **研究形状的对照实验**：第 38 周起收敛为「一个变量 + 一张对照表 + 一个归因结论」——VLA 适配 zero_shot / head / full 受控对照、单目深度歧义归因、世界模型离线评估
- **真机数据资产入库**：6 个 LeRobot 格式真机 VLA 微调数据集（双相机视频 + parquet 轨迹）与消融训练数据已随仓库同步

## 技术架构

```text
┌────────────────────────────────────────────────────────────────┐
│ L4 语言与规划层                                                  │
│    LLM 任务规划 (llm_planner) · 指令解析 (task_parser)           │
│    重规划与恢复决策 (replanner) · 真机路由 (real_router)          │
├────────────────────────────────────────────────────────────────┤
│ L3 感知与记忆层                                                  │
│    双相机观测 fixed/handeye (observation, real_observation)      │
│    VLM 视觉定位与标注 (vlm_labeler, real_perception)             │
│    场景记忆 (memory_*) · 失败分类 (failure_classifier)           │
├────────────────────────────────────────────────────────────────┤
│ L2 策略与技能层                                                  │
│    脚本化技能 (task_skills, real_skills) · 技能注册与校验          │
│    ACT 视觉策略 (visual_act) · SmolVLA 微调 (vla_adapt)          │
│    DAgger 数据回流 (dagger_collect)                              │
├────────────────────────────────────────────────────────────────┤
│ L1 执行与验证层                                                  │
│    Agent 主循环 (executor, real_agent)                           │
│    子任务验证 (verifier, real_verifier) · 失败恢复 (real_recovery)│
│    动作监控与安全约束 (action_monitor)                            │
├────────────────────────────────────────────────────────────────┤
│ L0 平台与硬件层                                                  │
│    AI2-THOR（语义仿真）· ManiSkill 3（策略训练）                  │
│    SO-101 follower/leader（真机）· 双相机标定 (handeye_calib)     │
└────────────────────────────────────────────────────────────────┘
```

**设计原则**：

- LLM 不进入高频控制环——只做任务级决策，低层控制由策略与控制器完成
- 纯逻辑与硬件驱动分离：`real_agent` 等核心模块不直接驱动机械臂，技能执行器由外部注入（Real 或 Mock），保证无硬件环境可完整测试
- 相机索引等易漂移配置收敛到唯一事实来源（`camera_slots.py`），上机前强制核图

### 第 38 周研究闭环

```text
VLM 数据引擎（降低标注成本） → VLA 适配与泛化（提供泛化能力） → 世界模型离线评估（降低真机试错）
                     深度感知（必配子问题：抬高物理成功率上限）
```

- **VLM 数据引擎**：多源 VLM 标注（`vlm_labeler`）+ 标签审计（`label_audit`），降低人工标注成本
- **VLA 适配对照**：同一份预训练 SmolVLA，只改解冻范围（`zero_shot` / `head` / `full`），在固定评测协议下给出成功率与遗忘对照表
- **世界模型离线评估**（`wm_ope`）：用离线策略评估替代部分真机试错
- **深度消融**（`depth_ablation`）：针对 PickCube 单目深度歧义的归因实验

## 数据资产（已入库）

| 路径 | 内容 |
|---|---|
| `runs/week38/_upload/wangbangyan/demo_vla_*` | 6 个真机 VLA 微调数据集（LeRobot 格式：parquet 轨迹 + meta + 双相机视频） |
| `runs/week38/collect/` | 真机采集 sidecar 与采集配置 |
| `runs/week38/ablation_data/` | 消融实验训练数据 |
| `runs/maniskill_lerobot/week15-*` | ManiSkill LeRobot v3 仿真数据集 |
| `runs/dagger/` | DAgger 回流数据 |
| `demo/` | 真机实际运行演示视频（`success.mp4` / `failed.mp4`） |

## 目录结构

```
├── src/                  # 核心代码
│   ├── run_agent.py      # LLM Agent 循环入口（AI2-THOR 仿真）
│   ├── executor.py       # 规划与技能编排
│   ├── llm_planner.py    # LLM 任务规划器
│   ├── task_skills.py    # 仿真器动作技能封装
│   ├── visual_act.py     # ACT 视觉策略（ManiSkill）
│   ├── vla_adapt.py 等   # VLA 适配、消融与评测（第 38 周）
│   ├── real_*.py         # 真实机械臂相关模块（感知/执行/校验/恢复）
│   ├── run_*.py          # 各实验与工具的 CLI 入口
│   └── tests/            # pytest 测试
├── docs/week<N>/         # 每周设计文档与实施计划
├── demo/                 # 真机实际运行演示视频（success/failed）
├── runs/                 # 运行产物（数据集入库，其余默认忽略）
├── .env.example          # 环境变量模板
└── requirements*.txt     # 分场景依赖清单
```

## 环境准备

项目按场景拆分依赖，建议使用独立虚拟环境：

```bash
# 仿真主线（AI2-THOR）
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# 需要调用 LLM API 时
python -m pip install -r requirements-llm.txt

# 测试
python -m pip install -r requirements-dev.txt

# ManiSkill / LeRobot / SmolVLA（独立环境，Python 3.12）
python -m pip install -r requirements-lerobot.txt
```

## 配置

复制环境变量模板并填写本地配置（`.env` 已被 git 忽略，切勿提交密钥）：

```bash
cp .env.example .env
```

关键变量：`LLM_API_BASE`、`LLM_API_KEY`、`LLM_MODEL`，以及可选的多模态定位源 `VLM_*` 配置。

## 快速开始

```bash
# 脚本化任务（不依赖 LLM）
python src/run_task.py --object-type CreditCard --receptacle-type Drawer

# LLM Agent
python src/run_agent.py --instruction "把信用卡放进抽屉"

# 运行测试
python -m pytest -q

# 提交前快速语法检查
python -m compileall -q src
```

输出会写入 `runs/` 下的时间戳目录。

## 开发约定

- **文档先行**：每周开发前先在 `docs/week<N>/` 完成 `design.md` 与 `plan.md`；实现与设计有偏差时补充「实际偏差」记录
- **门禁评审**：每阶段设定量化验收指标，未达标不扩展范围、不采购下一批设备
- 行为变更必须附带针对性测试；测试命名遵循 `test_*.py` / `Test*` / `test_*`
- 进度同步更新根目录开发计划文档与对应周次设计文档
- 详细仓库指南见 [AGENTS.md](AGENTS.md)

## 安全

- 禁止提交 `.env`、API 密钥、模拟器缓存或运行轨迹
- 密钥一律通过环境变量配置
