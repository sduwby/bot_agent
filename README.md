# bot_agent：桌面物品整理具身 Agent

从 LLM 软件 Agent 到真实机械臂闭环系统的业余开发项目。系统采用分层架构：LLM 负责语言理解与高层任务规划，视觉与状态模块负责感知，模仿学习策略（ACT / SmolVLA）与确定性脚本技能负责低层操作，控制器安全地驱动机械臂。开发路线为「先仿真、后硬件」，完整规划见 [具身智能完整项目_业余开发计划.md](具身智能完整项目_业余开发计划.md)。

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
├── runs/                 # 运行产物（截图、轨迹、摘要，默认忽略不提交）
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

- 每周开发前先在 `docs/week<N>/` 完成 `design.md` 与 `plan.md`；实现与设计有偏差时补充「实际偏差」记录。
- 行为变更必须附带针对性测试；测试命名遵循 `test_*.py` / `Test*` / `test_*`。
- 进度同步更新根目录开发计划文档与对应周次设计文档。
- 详细仓库指南见 [AGENTS.md](AGENTS.md)。

## 安全

- 禁止提交 `.env`、API 密钥、模拟器缓存或运行轨迹。
- 密钥一律通过环境变量配置。
