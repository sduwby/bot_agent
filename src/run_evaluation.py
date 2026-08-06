#!/usr/bin/env python3
"""运行阶段 A 固定任务集并生成可追溯的批量评测报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai2thor.controller import Controller

from env_config import load_local_env
from evaluation import (
    EvaluationTask,
    aggregate_results,
    classify_episode_failures,
    load_task_set,
    render_markdown_report,
)
from executor import AgentExecutor
from first_trajectory import create_run_dir
from llm_planner import RuleFallback, create_backend


def parse_args() -> argparse.Namespace:
    """解析批量评测参数。"""
    parser = argparse.ArgumentParser(description="运行阶段 A 批量评测")
    parser.add_argument(
        "--task-set",
        type=Path,
        default=Path("evaluation/stage_a_tasks.json"),
        help="固定任务集 JSON",
    )
    parser.add_argument(
        "--backend",
        choices=("rule", "auto"),
        default="rule",
        help="rule 为确定性基线；auto 使用本地 LLM 配置或规则回退",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="只运行指定任务 ID，可重复传入",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/evaluation"),
        help="批次产物根目录",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-search-steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    return parser.parse_args()


def _git_commit() -> str:
    """读取当前提交；失败时返回 unknown，不阻止评测。"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool | None:
    """记录工作区是否含未提交改动；无法读取时返回 None。"""
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _make_backend(mode: str):
    """为单个 episode 创建独立 Planner 后端。"""
    if mode == "rule":
        return RuleFallback()
    return create_backend()


def _backend_metadata(mode: str) -> tuple[str, str | None]:
    """在不输出密钥的前提下描述实际后端。"""
    if mode == "rule":
        return "RuleFallback", None
    if not os.environ.get("LLM_API_KEY", ""):
        return "RuleFallback (auto fallback)", None
    return "OpenAIBackend", os.environ.get("LLM_MODEL", "gpt-4o-mini")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_task(
    task: EvaluationTask,
    batch_dir: Path,
    *,
    backend_mode: str,
    max_steps: int,
    max_search_steps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """用独立 Controller 运行一条任务，并把异常归一化为失败结果。"""
    run_dir = batch_dir / task.task_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trajectory_path = run_dir / "trajectory.jsonl"
    records: list[dict[str, Any]] = []
    controller = None
    agent: AgentExecutor | None = None
    started = time.monotonic()
    summary: dict[str, Any]

    try:
        controller = Controller(
            scene=task.scene,
            width=width,
            height=height,
            quality="Low",
            gridSize=0.25,
            rotateStepDegrees=90,
        )
        with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
            agent = AgentExecutor(
                controller=controller,
                run_dir=run_dir,
                trajectory_file=trajectory_file,
                records=records,
                backend=_make_backend(backend_mode),
                max_steps=max_steps,
                max_search_steps=max_search_steps,
            )
            summary = agent.run(task.instruction)
    except Exception as exc:
        action_history = agent.action_history if agent is not None else []
        summary = {
            "instruction": task.instruction,
            "task_success": False,
            "termination_reason": f"异常: {type(exc).__name__}: {exc}",
            "final_verification": None,
            "steps_taken": len(action_history),
            "action_history": action_history,
            "steps_recorded": len(records),
            "planner_calls": getattr(agent, "planner_calls", 0),
            "llm_calls": getattr(agent, "llm_calls", 0),
        }
    finally:
        if controller is not None:
            controller.stop()

    _write_json(run_dir / "summary.json", summary)
    action_history = summary.get("action_history", [])
    termination_reason = str(summary.get("termination_reason", ""))
    result = {
        "task_id": task.task_id,
        "name": task.name,
        "scene": task.scene,
        "instruction": task.instruction,
        "tags": list(task.tags),
        "task_success": bool(summary.get("task_success")),
        "termination_reason": termination_reason,
        "final_verification": summary.get("final_verification"),
        "steps_taken": int(summary.get("steps_taken", 0)),
        "planner_calls": int(summary.get("planner_calls", 0)),
        "llm_calls": int(summary.get("llm_calls", 0)),
        "failure_counts": classify_episode_failures(
            action_history, termination_reason
        ),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "evaluation_result.json", result)
    return result


def _select_tasks(
    tasks: list[EvaluationTask], selected_ids: list[str]
) -> list[EvaluationTask]:
    if not selected_ids:
        return tasks
    id_set = set(selected_ids)
    unknown = sorted(id_set - {task.task_id for task in tasks})
    if unknown:
        raise ValueError(f"未知任务 ID: {', '.join(unknown)}")
    return [task for task in tasks if task.task_id in id_set]


def main() -> int:
    """执行批次、生成报告，并用退出码表达阶段门禁结论。"""
    args = parse_args()
    load_local_env()
    try:
        all_tasks = load_task_set(args.task_set)
        tasks = _select_tasks(all_tasks, args.task_id)
    except ValueError as exc:
        print(f"错误：{exc}")
        return 2

    task_set_bytes = args.task_set.read_bytes()
    batch_dir = create_run_dir(args.output_root)
    backend_name, model = _backend_metadata(args.backend)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "task_set": str(args.task_set),
        "task_set_sha256": hashlib.sha256(task_set_bytes).hexdigest(),
        "backend": backend_name,
        "model": model,
        "selected_task_ids": [task.task_id for task in tasks],
        "max_steps": args.max_steps,
        "max_search_steps": args.max_search_steps,
        "width": args.width,
        "height": args.height,
    }
    _write_json(batch_dir / "config.json", metadata)

    episodes_path = batch_dir / "episodes.jsonl"
    results: list[dict[str, Any]] = []
    with episodes_path.open("a", encoding="utf-8") as episodes_file:
        for index, task in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {task.task_id} {task.scene} {task.instruction}")
            result = run_task(
                task,
                batch_dir,
                backend_mode=args.backend,
                max_steps=args.max_steps,
                max_search_steps=args.max_search_steps,
                width=args.width,
                height=args.height,
            )
            results.append(result)
            episodes_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            episodes_file.flush()
            status = "成功" if result["task_success"] else "失败"
            print(
                f"  {status}，动作 {result['steps_taken']}，"
                f"Planner {result['planner_calls']}，LLM {result['llm_calls']}"
            )

    summary = aggregate_results(results, expected_tasks=len(tasks))
    report = {"metadata": metadata, "summary": summary, "episodes": results}
    _write_json(batch_dir / "report.json", report)
    (batch_dir / "report.md").write_text(
        render_markdown_report(report), encoding="utf-8"
    )

    print(f"评测目录：{batch_dir}")
    print(
        f"任务成功：{summary['successful_tasks']}/{summary['total_tasks']} "
        f"({summary['success_rate']:.1%})"
    )
    print(f"阶段 A 门禁：{'通过' if summary['gate']['passed'] else '未通过'}")
    return 0 if summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
