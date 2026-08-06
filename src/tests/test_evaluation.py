"""阶段 A 批量评测纯逻辑测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import (
    aggregate_results,
    classify_episode_failures,
    load_task_set,
    render_markdown_report,
)
from run_evaluation import run_task


def _task_set(count: int = 10) -> dict:
    return {
        "version": 1,
        "tasks": [
            {
                "id": f"a{index:02d}",
                "name": f"任务 {index}",
                "scene": "FloorPlan1",
                "instruction": "把苹果放到台面",
                "tags": ["baseline"],
            }
            for index in range(1, count + 1)
        ],
    }


def _write_task_set(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _result(index: int, success: bool) -> dict:
    termination = "任务状态验证通过" if success else "达到最大步骤数"
    return {
        "task_id": f"a{index:02d}",
        "task_success": success,
        "steps_taken": index,
        "planner_calls": 3,
        "llm_calls": 0,
        "failure_counts": classify_episode_failures([], termination),
    }


def test_load_versioned_task_set(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    _write_task_set(path, _task_set(12))

    tasks = load_task_set(path)

    assert len(tasks) == 12
    assert tasks[0].task_id == "a01"
    assert tasks[0].tags == ("baseline",)


@pytest.mark.parametrize("count", [9, 21])
def test_load_rejects_task_count_outside_range(tmp_path: Path, count: int) -> None:
    path = tmp_path / "tasks.json"
    _write_task_set(path, _task_set(count))

    with pytest.raises(ValueError, match="10–20"):
        load_task_set(path)


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    data = _task_set()
    data["tasks"][1]["id"] = data["tasks"][0]["id"]
    path = tmp_path / "tasks.json"
    _write_task_set(path, data)

    with pytest.raises(ValueError, match="ID 重复"):
        load_task_set(path)


def test_load_rejects_unparseable_instruction(tmp_path: Path) -> None:
    data = _task_set()
    data["tasks"][0]["instruction"] = "请整理房间"
    path = tmp_path / "tasks.json"
    _write_task_set(path, data)

    with pytest.raises(ValueError, match="指令不可解析"):
        load_task_set(path)


def test_classify_action_and_termination_failures() -> None:
    history = [
        {
            "skill": "pick",
            "parameters": {"object_type": "Apple"},
            "success": False,
            "result_summary": "扫描后未发现可抓取的 Apple",
        },
        {
            "skill": "done",
            "parameters": {},
            "success": False,
            "result_summary": "任务验证失败: 未放入目标容器",
        },
    ]

    counts = classify_episode_failures(history, "达到最大步骤数")

    assert counts == {
        "object_not_found": 1,
        "task_verification_failed": 1,
        "termination:max_steps": 1,
    }


def test_aggregate_exactly_seventy_percent_passes_gate() -> None:
    results = [_result(index, index <= 7) for index in range(1, 11)]

    summary = aggregate_results(results, expected_tasks=10)

    assert summary["success_rate"] == 0.7
    assert summary["steps"] == {
        "total": 55,
        "mean": 5.5,
        "median": 5.5,
        "min": 1,
        "max": 10,
    }
    assert summary["planner_calls"]["total"] == 30
    assert summary["failed_episode_terminations"] == {"max_steps": 3}
    assert summary["gate"]["passed"] is True


def test_aggregate_rejects_small_or_incomplete_batch() -> None:
    small = aggregate_results(
        [_result(index, True) for index in range(1, 10)], expected_tasks=9
    )
    incomplete = aggregate_results(
        [_result(index, True) for index in range(1, 11)], expected_tasks=12
    )

    assert small["gate"]["enough_tasks"] is False
    assert small["gate"]["passed"] is False
    assert incomplete["gate"]["batch_complete"] is False
    assert incomplete["gate"]["passed"] is False


def test_render_report_contains_metrics_episodes_and_gate() -> None:
    results = [_result(index, True) for index in range(1, 11)]
    summary = aggregate_results(results)
    report = {
        "metadata": {
            "backend": "RuleFallback",
            "task_set": "evaluation/stage_a_tasks.json",
        },
        "summary": summary,
        "episodes": [
            {
                **results[0],
                "scene": "FloorPlan1",
                "instruction": "把苹果放到台面",
                "termination_reason": "任务状态验证通过",
            }
        ],
    }

    markdown = render_markdown_report(report)

    assert "Planner 后端：`RuleFallback`" in markdown
    assert "任务成功 | 10/10" in markdown
    assert "| a01 | FloorPlan1 | 把苹果放到台面 | ✅ |" in markdown
    assert "**结论：通过。**" in markdown


def test_render_llm_report_uses_llm_boundary_note() -> None:
    results = [_result(index, True) for index in range(1, 11)]
    for result in results:
        result["llm_calls"] = 3
    report = {
        "metadata": {"backend": "OpenAIBackend"},
        "summary": aggregate_results(results),
        "episodes": [],
    }

    markdown = render_markdown_report(report)

    assert "本报告使用真实 LLM 后端" in markdown
    assert "自动搜索与恢复不计入" in markdown


def test_run_task_converts_controller_exception_to_failed_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_path = tmp_path / "tasks.json"
    _write_task_set(task_path, _task_set())
    task = load_task_set(task_path)[0]

    def fail_controller(**kwargs):
        raise RuntimeError("Unity unavailable")

    monkeypatch.setattr("run_evaluation.Controller", fail_controller)

    result = run_task(
        task,
        tmp_path / "batch",
        backend_mode="rule",
        max_steps=20,
        max_search_steps=8,
        width=240,
        height=180,
    )

    assert result["task_success"] is False
    assert result["failure_counts"] == {"termination:exception": 1}
    assert (tmp_path / "batch" / task.task_id / "summary.json").exists()
