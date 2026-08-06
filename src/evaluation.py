"""阶段 A 批量评测的任务加载、指标聚合与报告渲染。"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from failure_classifier import FailureClassifier
from task_parser import parse_instruction


@dataclass(frozen=True)
class EvaluationTask:
    """一条版本化、可复现的批量评测任务。"""

    task_id: str
    name: str
    scene: str
    instruction: str
    tags: tuple[str, ...]


def load_task_set(path: Path) -> list[EvaluationTask]:
    """加载并严格校验包含 10–20 条任务的 JSON 任务集。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"任务集不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"任务集不是合法 JSON: {exc}") from exc

    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("任务集 version 必须为 1")
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not 10 <= len(raw_tasks) <= 20:
        raise ValueError("任务集必须包含 10–20 条任务")

    tasks: list[EvaluationTask] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_tasks, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条任务必须是对象")
        values: dict[str, str] = {}
        for field in ("id", "name", "scene", "instruction"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"第 {index} 条任务的 {field} 必须是非空字符串")
            values[field] = value.strip()
        if values["id"] in seen_ids:
            raise ValueError(f"任务 ID 重复: {values['id']}")
        seen_ids.add(values["id"])

        tags = item.get("tags", [])
        if not isinstance(tags, list) or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            raise ValueError(f"任务 {values['id']} 的 tags 必须是非空字符串数组")
        try:
            parse_instruction(values["instruction"])
        except ValueError as exc:
            raise ValueError(f"任务 {values['id']} 的指令不可解析: {exc}") from exc

        tasks.append(
            EvaluationTask(
                task_id=values["id"],
                name=values["name"],
                scene=values["scene"],
                instruction=values["instruction"],
                tags=tuple(tag.strip() for tag in tags),
            )
        )
    return tasks


def _termination_category(reason: str) -> str | None:
    """把失败 episode 的终止文本归一化为稳定类别。"""
    lowered = reason.casefold()
    if reason.startswith("异常:"):
        return "exception"
    if "任务解析失败" in reason:
        return "task_parse_failed"
    if "决策解析失败" in reason:
        return "decision_parse_failed"
    if "搜索耗尽" in reason:
        return "search_exhausted"
    if "最大步骤数" in reason:
        return "max_steps"
    if "验证失败" in reason or "验证未通过" in reason:
        return "task_verification_failed"
    if "失败" in lowered or "error" in lowered:
        return "unknown_termination"
    return None


def classify_episode_failures(
    action_history: list[dict[str, Any]],
    termination_reason: str,
) -> dict[str, int]:
    """统计 episode 内失败动作，并附加失败终止类别。"""
    counts: Counter[str] = Counter()
    for action in action_history:
        if action.get("success") is not False:
            continue
        skill = str(action.get("skill", ""))
        params = action.get("parameters")
        if not isinstance(params, dict):
            params = {}
        reason = str(action.get("result_summary", "") or "未知原因")
        if skill == "done" and "验证失败" in reason:
            counts["task_verification_failed"] += 1
        elif "缺少必填参数" in reason or "不在允许范围内" in reason:
            counts["parameter_validation"] += 1
        elif skill:
            failure = FailureClassifier.classify_failure(skill, params, reason)
            counts[failure.category.value] += 1
        else:
            counts["unknown"] += 1

    terminal = _termination_category(termination_reason)
    if terminal is not None:
        counts[f"termination:{terminal}"] += 1
    return dict(sorted(counts.items()))


def aggregate_results(
    results: list[dict[str, Any]],
    *,
    minimum_tasks: int = 10,
    success_threshold: float = 0.70,
    expected_tasks: int | None = None,
) -> dict[str, Any]:
    """聚合 episode 结果并给出阶段 A 门禁结论。"""
    total = len(results)
    expected = total if expected_tasks is None else expected_tasks
    successes = sum(bool(result.get("task_success")) for result in results)
    success_rate = successes / total if total else 0.0
    steps = [int(result.get("steps_taken", 0)) for result in results]
    planner_calls = [int(result.get("planner_calls", 0)) for result in results]
    llm_calls = [int(result.get("llm_calls", 0)) for result in results]

    action_failures: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    for result in results:
        for category, count in result.get("failure_counts", {}).items():
            if category.startswith("termination:"):
                if not result.get("task_success"):
                    terminations[category.removeprefix("termination:")] += int(count)
            else:
                action_failures[category] += int(count)

    enough_tasks = total >= minimum_tasks
    batch_complete = total == expected
    rate_passed = success_rate >= success_threshold
    gate_passed = enough_tasks and batch_complete and rate_passed

    def distribution(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"total": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0}
        return {
            "total": sum(values),
            "mean": round(statistics.mean(values), 2),
            "median": round(float(statistics.median(values)), 2),
            "min": min(values),
            "max": max(values),
        }

    return {
        "total_tasks": total,
        "expected_tasks": expected,
        "successful_tasks": successes,
        "failed_tasks": total - successes,
        "success_rate": round(success_rate, 4),
        "steps": distribution(steps),
        "planner_calls": distribution(planner_calls),
        "llm_calls": distribution(llm_calls),
        "action_failure_counts": dict(sorted(action_failures.items())),
        "failed_episode_terminations": dict(sorted(terminations.items())),
        "gate": {
            "minimum_tasks": minimum_tasks,
            "success_threshold": success_threshold,
            "enough_tasks": enough_tasks,
            "batch_complete": batch_complete,
            "success_rate_passed": rate_passed,
            "passed": gate_passed,
        },
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """将含元数据、汇总和逐任务结果的报告渲染为 Markdown。"""
    summary = report["summary"]
    metadata = report.get("metadata", {})
    gate = summary["gate"]
    gate_text = "通过" if gate["passed"] else "未通过"
    lines = [
        "# 阶段 A 批量评测报告",
        "",
        f"- 运行时间（UTC）：{metadata.get('started_at_utc', '未知')}",
        f"- 代码提交：`{metadata.get('git_commit', '未知')}`",
        f"- 工作区含未提交改动：`{metadata.get('git_dirty', '未知')}`",
        f"- 任务集：`{metadata.get('task_set', '未知')}`",
        f"- 任务集 SHA-256：`{metadata.get('task_set_sha256', '未知')}`",
        f"- Planner 后端：`{metadata.get('backend', '未知')}`",
        f"- 模型：`{metadata.get('model') or '无'}`",
        "",
        "## 指标总览",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 任务成功 | {summary['successful_tasks']}/{summary['total_tasks']} |",
        f"| 成功率 | {summary['success_rate']:.1%} |",
        f"| 动作数（总计 / 均值 / 中位数） | {summary['steps']['total']} / {summary['steps']['mean']:.2f} / {summary['steps']['median']:.2f} |",
        f"| Planner 调用（总计 / 均值） | {summary['planner_calls']['total']} / {summary['planner_calls']['mean']:.2f} |",
        f"| 真实 LLM 调用（总计 / 均值） | {summary['llm_calls']['total']} / {summary['llm_calls']['mean']:.2f} |",
        "",
        "## 逐任务结果",
        "",
        "| ID | 场景 | 指令 | 成功 | 动作 | Planner | LLM | 终止原因 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in report.get("episodes", []):
        reason = str(result.get("termination_reason", "")).replace("|", "\\|")
        instruction = str(result.get("instruction", "")).replace("|", "\\|")
        lines.append(
            f"| {result.get('task_id', '')} | {result.get('scene', '')} | "
            f"{instruction} | {'✅' if result.get('task_success') else '❌'} | "
            f"{result.get('steps_taken', 0)} | {result.get('planner_calls', 0)} | "
            f"{result.get('llm_calls', 0)} | {reason} |"
        )

    lines.extend(["", "## 失败分析", ""])
    if summary["action_failure_counts"]:
        lines.extend(["### 失败动作", ""])
        for category, count in summary["action_failure_counts"].items():
            lines.append(f"- `{category}`：{count}")
    else:
        lines.append("- 未记录到失败动作。")
    lines.extend(["", "### 失败 episode 终止原因", ""])
    if summary["failed_episode_terminations"]:
        for category, count in summary["failed_episode_terminations"].items():
            lines.append(f"- `{category}`：{count}")
    else:
        lines.append("- 无失败 episode。")

    if summary["llm_calls"]["total"] > 0:
        boundary_note = (
            "> 本报告使用真实 LLM 后端；LLM 调用数只统计实际进入外部 "
            "Planner API 的调用，自动搜索与恢复不计入。"
        )
    else:
        boundary_note = (
            "> 规则后端用于确定性闭环基线，不代表真实 LLM 表现；"
            "需使用真实 LLM 后端的独立报告评价模型 Planner。"
        )

    lines.extend([
        "",
        "## 阶段 A 门禁评审",
        "",
        f"**结论：{gate_text}。**",
        "",
        f"- 任务数不少于 {gate['minimum_tasks']}：{'是' if gate['enough_tasks'] else '否'}。",
        f"- 批次完整：{'是' if gate['batch_complete'] else '否'}。",
        f"- 成功率不低于 {gate['success_threshold']:.0%}：{'是' if gate['success_rate_passed'] else '否'}。",
        "",
        boundary_note,
        "",
    ])
    return "\n".join(lines)
