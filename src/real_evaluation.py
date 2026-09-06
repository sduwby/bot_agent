"""真机端到端评测 harness（阶段 F，SO-101）——S10/S11。

在部署评测（deploy_eval）的"trials.csv + report.md"范式上，把失败类型映射到
§9.3 五维（感知/策略/控制/规划/验证），并支持批量跑真实 NL 任务集。

真实执行需要机械臂；纯逻辑测试用 Mock。verdict→维度映射可离线单测。
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# §9.3 失败维度
DIM_PERCEPTION = "perception"     # 感知失败
DIM_POLICY = "policy"             # 策略执行失败
DIM_CONTROL = "control"           # 控制/驱动失败
DIM_PLAN = "plan"                 # 规划/决策失败
DIM_VERIFICATION = "verification"  # 验证失败
DIM_OK = "ok"

# deploy_eval verdict → 五维映射
VERDICT_TO_DIM: dict[str, str] = {
    "success": DIM_OK,
    "grasp_empty": DIM_POLICY,            # 咬空 = 策略对中/闭合时机失败
    "grasped_not_lifted": DIM_POLICY,     # 夹住未抬离 = 策略抬升失败
    "no_grasp_attempt": DIM_POLICY,       # 未尝试闭合
    "abort": DIM_CONTROL,                 # 中断/通信 = 控制
}

# 真机失败码 → 五维（供 skills 链路失败归类）
FAILCODE_TO_DIM: dict[str, str] = {
    "grasp_empty": DIM_POLICY,
    "not_lifted": DIM_POLICY,
    "not_holding": DIM_VERIFICATION,
    "home_timeout": DIM_CONTROL,
    "timeout": DIM_CONTROL,
    "comm_error": DIM_CONTROL,
}


def map_verdict_to_dim(verdict: str) -> str:
    """deploy_eval verdict → 五维失败维度。"""
    return VERDICT_TO_DIM.get(verdict, DIM_PLAN)


def map_failure_code_to_dim(failure_code: str | None) -> str:
    """real_skills 失败码 → 五维失败维度。"""
    if failure_code is None:
        return DIM_OK
    return FAILCODE_TO_DIM.get(failure_code, DIM_PLAN)


@dataclass
class EvalTrial:
    """一次评测轮记录。"""

    task_id: str
    instruction: str
    position: str = ""
    verdict: str = "abort"
    steps_run: int = 0
    final_gripper: float = -1.0
    final_wrist: float = -1.0
    note: str = ""

    @property
    def failure_dim(self) -> str:
        return map_verdict_to_dim(self.verdict)


class RealEvalRunner:
    """评测编排：对任务集逐条跑 router/executor，产出 CSV 与报告。

    实际执行需注入 runnable（真机 RealSkillRouter 或测试 Mock）。
    """

    def __init__(self, output_root: str | Path) -> None:
        self.run_dir = Path(output_root) / f"real_eval_{datetime.now():%Y%m%d_%H%M%S}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trials: list[EvalTrial] = []

    def add_trial(self, trial: EvalTrial) -> None:
        self.trials.append(trial)

    def aggregate(self) -> dict[str, Any]:
        n = len(self.trials)
        ok = sum(1 for t in self.trials if t.verdict == "success")
        dim_counts: dict[str, int] = {}
        for t in self.trials:
            dim_counts[t.failure_dim] = dim_counts.get(t.failure_dim, 0) + 1
        return {
            "total_trials": n,
            "successful": ok,
            "success_rate": round(ok / n, 3) if n else 0.0,
            "failure_dims": dim_counts,
        }

    def write_csv(self) -> Path:
        path = self.run_dir / "trials.csv"
        if not self.trials:
            path.write_text("", encoding="utf-8")
            return path
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.trials[0].__dataclass_fields__.keys()))
            w.writeheader()
            for t in self.trials:
                w.writerow(t.__dict__)
        return path

    def render_report(self) -> str:
        agg = self.aggregate()
        lines = [
            f"# 真机端到端评测报告 {datetime.now():%Y-%m-%d %H:%M}",
            "",
            f"- 总轮数: {agg['total_trials']}，成功: {agg['successful']}，成功率: {agg['success_rate']:.0%}",
            "",
            "## 失败维度分布（§9.3）",
            "",
            "| 维度 | 数量 |",
            "|---|--:|",
        ]
        for d, c in sorted(agg["failure_dims"].items(), key=lambda x: -x[1]):
            lines.append(f"| {d} | {c} |")
        lines.append("")
        lines.append("## 逐轮明细")
        lines.append("")
        lines.append("| task | position | verdict | dim | steps |")
        lines.append("|---|---|---|---|--:|")
        for t in self.trials:
            lines.append(f"| {t.task_id} | {t.position} | {t.verdict} | {t.failure_dim} | {t.steps_run} |")
        return "\n".join(lines)

    def save(self) -> tuple[Path, Path]:
        csv_path = self.write_csv()
        report_path = self.run_dir / "report.md"
        report_path.write_text(self.render_report(), encoding="utf-8")
        return csv_path, report_path


def load_real_task_set(path: str | Path) -> list[dict[str, Any]]:
    """加载真机任务集 JSON（仿 evaluation 任务集但面向真机单技能）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("任务集缺少非空 tasks")
    return tasks
