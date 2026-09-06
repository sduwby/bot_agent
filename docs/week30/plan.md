# 第 30 周 实施计划：首次部署与失败分布

## 任务分解

1. 编写 `outputs/deploy_eval_so101.py`（评测编排 + 低速 + 判定 + 报告）
2. 单轮 smoke 验证（1 个位置 1 轮，确认流程与产物正确）
3. 正式 20 轮评测（5 位置 × 4 轮）
4. 汇总 report.md，分析失败分布，输出第 31 周补采靶点

## 文件结构

**新增**：
- `outputs/deploy_eval_so101.py` — 评测 CLI
- `docs/week30/design.md` / `plan.md` — 本周文档

**修改**：无（infer_so101.py 仅被 import，不改动）

**产物（不入库）**：`outputs/deploy_eval_<timestamp>/`

## 实施步骤（按优先级）

1. 实现 deploy_eval_so101.py：协议常量 → 单轮执行（复用 infer_so101 组件）→ 自动判定 → 人工确认 → CSV/报告
2. `py_compile` 语法检查 + `--help` 验证参数解析
3. smoke：位置"中"跑 1 轮，检查 trials.csv 行、帧保存、判定合理性
4. 正式 20 轮（低速、人工看护，操作员按提示换位）
5. 生成 report.md 并复盘

## 测试计划

- 单元层面：脚本无 Unity 依赖部分（判定逻辑）通过 smoke 实测验证；
- 集成层面：smoke 1 轮 + 正式 20 轮；每轮产物（state_action_log.csv、关键帧）抽查可读；
- 验收核对：trials.csv 行数=20、report.md 含成功率与失败分布表。

## 验收标准

- 20 轮全部完成（abort 轮计入并注明原因）；
- `report.md` 给出：总成功率、各失败类型计数与占比、按位置的成功率分解；
- 依据失败分布明确第 31 周补采方向（如闭合时机、特定位置、抬起后静止）。

## 风险与回退

- 若低速限幅导致轨迹异常（限幅过严使动作变形），放宽 `--max-relative-target`（8→15）重跑 smoke；
- 若自动判定与人工观察系统性不符，以人工 final_verdict 为准并修正判定阈值后记入偏差。
