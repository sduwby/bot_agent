# 仓库指南

## 项目结构与模块组织

本仓库是基于 AI2-THOR 的 Python 具身智能原型。核心代码位于 `src/`：`run_agent.py` 启动 LLM Agent 循环，`executor.py` 编排规划与技能，`task_skills.py` 封装模拟器动作。`run_task.py`、`pickup_nearest.py` 和 `pickup_and_place.py` 等脚本用于单项实验。测试位于 `src/tests/`，设计文档放在 `docs/`，根目录保存项目规划资料。运行截图、轨迹和摘要统一写入已忽略的 `runs/`，不要提交这些产物。

## 构建、测试与本地开发

项目没有构建步骤，也尚未提供依赖锁定文件。请创建独立环境并按任务安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ai2thor pillow pytest
python -m pytest -q
```

仅在使用 API 模型时安装 `openai`。运行脚本化任务：

```bash
python src/run_task.py --object-type CreditCard --receptacle-type Drawer
```

运行 LLM Agent：

```bash
python src/run_agent.py --instruction "把信用卡放进抽屉"
```

输出会写入 `runs/` 下的时间戳目录。提交前可运行 `python -m compileall -q src` 做快速语法检查。

## 编码风格与命名约定

遵循 PEP 8，使用四个空格缩进。函数、变量和模块采用 `snake_case`，类采用 `PascalCase`，常量采用全大写命名。公共函数应保留类型标注；模块、类及不直观的辅助函数应有简短文档字符串。延续现有中文用户提示与文档风格。优先编写小型、可测试的辅助函数，不要把模拟器业务逻辑堆入 CLI 入口。目前未配置格式化器或静态检查器，提交前需人工检查差异。

## 测试规范

测试框架为 `pytest`。测试文件命名为 `test_*.py`，测试类为 `Test*`，测试函数为 `test_*`。纯逻辑测试不应依赖 Unity；参照 `src/tests/test_task_skills.py` 模拟外部依赖、控制器事件和文件产物。参数校验、Planner 输出解析及失败路径发生变更时，应补充回归测试。项目暂无强制覆盖率门槛，但行为变更必须包含针对性测试。

## 提交与合并请求规范

Git 历史目前只有 `init`，尚未形成固定约定。提交标题应简短并使用祈使语气，例如 `Add planner response validation`；无关改动应拆分提交。合并请求需说明行为变化、列出测试命令与结果、关联相关 Issue，并注明模拟器或 API 前置条件。若 AI2-THOR 的可视行为发生变化，应提供截图或运行产物路径，但不要提交 `runs/` 内容。

## 安全与配置

将 `.env.example` 复制为 `.env` 后填写本地配置。禁止提交 `.env`、API 密钥、模拟器缓存或运行轨迹。`LLM_API_BASE`、`LLM_API_KEY` 和 `LLM_MODEL` 必须继续通过环境变量配置。
