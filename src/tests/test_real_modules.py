"""real_router / real_recovery / real_planner / real_memory / real_evaluation /
real_perception 的单元测试 — 纯逻辑，无机械臂。"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_planner import LLMBackend
from real_evaluation import EvalTrial, RealEvalRunner, map_failure_code_to_dim, map_verdict_to_dim
from real_executor import MockEngine, RealSkillExecutor
from real_memory import RealMemoryStore
from real_perception import PerceptionResult, describe_workspace, local_holding_state
from real_planner import Plan, PlanError, PlanStep, execute_plan, parse_plan, validate_plan
from real_recovery import LoopSignal, RealRecoveryManager, detect_loop
from real_router import (
    PolicySource,
    RealSkillRouter,
    ROUTE_ACT,
    ROUTE_SCRIPTED,
    ScriptedPolicy,
    default_real_route,
)
from real_skills import FAIL_COMM, FAIL_GRASP_EMPTY


def make_executor(mock_result: str = "success"):
    engine = MockEngine(grasp_result=mock_result)
    return RealSkillExecutor(engine, empty_grip_baseline=1.5)


class TestRealRouter:
    def test_default_route_has_act_then_scripted(self) -> None:
        route = default_real_route()
        assert [s.name for s in route] == [ROUTE_ACT, ROUTE_SCRIPTED]

    def test_act_success_no_fallback(self) -> None:
        router = RealSkillRouter(make_executor("success"), default_real_route())
        r = router.run_skill("grasp_screwdriver")
        assert r.success and r.final_policy == ROUTE_ACT and not r.switch_events

    def test_act_fail_falls_back_to_scripted(self) -> None:
        # Mock 抓取失败 → 路由切到 scripted（本实现 scripted 无抓取能力，报无能力）
        router = RealSkillRouter(make_executor("grasp_empty"), default_real_route())
        r = router.run_skill("grasp_screwdriver")
        assert not r.success
        assert any(s.to_policy == ROUTE_SCRIPTED for s in r.switch_events)

    def test_invalid_source_kind(self) -> None:
        try:
            PolicySource(kind="bad", name="x")
            assert False
        except ValueError:
            pass


class TestRealRecovery:
    def test_grasp_fail_retry_then_abort(self) -> None:
        m = RealRecoveryManager(max_retries=1)
        a1 = m.plan_recovery(skill_name="grasp_screwdriver", failure_code=FAIL_GRASP_EMPTY)
        assert a1.action == "retry"
        a2 = m.plan_recovery(skill_name="grasp_screwdriver", failure_code=FAIL_GRASP_EMPTY)
        assert a2.action == "abort" and a2.is_final

    def test_comm_aborts(self) -> None:
        m = RealRecoveryManager()
        a = m.plan_recovery(skill_name="grasp_screwdriver", failure_code=FAIL_COMM)
        assert a.action == "abort" and a.is_final

    def test_loop_detected(self) -> None:
        m = RealRecoveryManager()
        loop = LoopSignal(True, "IDENTICAL_REPEAT", "重复 grasp")
        a = m.plan_recovery(skill_name="grasp_screwdriver", failure_code=FAIL_GRASP_EMPTY, loop=loop)
        assert a.action == "reset_and_retry"

    def test_detect_loop_identical(self) -> None:
        hist = [
            {"skill": "grasp_screwdriver", "params": {}, "success": False},
            {"skill": "grasp_screwdriver", "params": {}, "success": False},
            {"skill": "grasp_screwdriver", "params": {}, "success": False},
        ]
        assert detect_loop(hist, window=3).is_loop

    def test_detect_loop_not_when_progress(self) -> None:
        hist = [
            {"skill": "grasp_screwdriver", "params": {}, "success": False},
            {"skill": "grasp_screwdriver", "params": {}, "success": True},
        ]
        assert not detect_loop(hist, window=3).is_loop


class TestRealPlanner:
    def test_parse_valid_plan(self) -> None:
        p = parse_plan(json.dumps({"reasoning": "r", "plan": [
            {"skill": "grasp_screwdriver", "parameters": {}},
            {"skill": "done", "parameters": {}},
        ]}))
        assert isinstance(p, Plan) and len(p.steps) == 2

    def test_parse_invalid(self) -> None:
        assert isinstance(parse_plan("不是JSON"), PlanError)
        assert isinstance(parse_plan(json.dumps({"skill": "x"})), PlanError)

    def test_validate_unknown_skill(self) -> None:
        p = Plan("", [PlanStep("fly", {})])
        assert validate_plan(p) is not None

    def test_validate_ok(self) -> None:
        p = Plan("", [PlanStep("grasp_screwdriver", {})])
        assert validate_plan(p) is None

    def test_execute_plan_runs_until_done_or_failure(self) -> None:
        ex = make_executor("success")
        p = Plan("", [PlanStep("grasp_screwdriver", {}), PlanStep("done", {})])
        results = execute_plan(ex, p)
        assert len(results) == 2
        assert results[-1][0].skill == "done" and results[-1][1].success

    def test_execute_plan_stops_on_failure(self) -> None:
        ex = make_executor("grasp_empty")
        p = Plan("", [PlanStep("grasp_screwdriver", {}), PlanStep("done", {})])
        results = execute_plan(ex, p)
        assert len(results) == 1 and not results[0][1].success


class TestRealMemory:
    def test_add_search_inject(self, tmp_path) -> None:
        store = RealMemoryStore(tmp_path, "sess1")
        store.add("grasp_screwdriver 成功", type_="action_experience", priority=90)
        store.add_failure("grasp_screwdriver", "grasp_empty")
        recs = store.search("失败", limit=5)
        assert any("失败" in r.content for r in recs)
        prompt = store.inject_into_prompt("失败", limit=3)
        assert prompt and "场景记忆" in prompt
        assert store.path.exists()

    def test_reload_from_disk(self, tmp_path) -> None:
        s1 = RealMemoryStore(tmp_path, "sess2")
        s1.add("重试一次")
        s2 = RealMemoryStore(tmp_path, "sess2")
        assert len(s2._records) == 1


class TestRealEval:
    def test_verdict_dim_mapping(self) -> None:
        assert map_verdict_to_dim("success") == "ok"
        assert map_verdict_to_dim("grasp_empty") == "policy"
        assert map_verdict_to_dim("abort") == "control"
        assert map_failure_code_to_dim("comm_error") == "control"

    def test_runner_csv_report(self, tmp_path) -> None:
        runner = RealEvalRunner(tmp_path)
        runner.add_trial(EvalTrial(task_id="r01", instruction="夹起", verdict="success", steps_run=10))
        runner.add_trial(EvalTrial(task_id="r02", instruction="夹起", verdict="grasp_empty", steps_run=5))
        csv_path, report_path = runner.save()
        assert csv_path.exists() and report_path.exists()
        assert "50%" in report_path.read_text(encoding="utf-8")


class FakeVisionBackend(LLMBackend):
    def chat(self, system, user, images=None) -> str:
        return json.dumps({"objects": [{"name": "screwdriver", "region": "center", "visible": True}],
                           "note": "看到螺丝刀"})


class TestRealPerception:
    def test_describe_workspace_with_vision(self) -> None:
        r = describe_workspace(FakeVisionBackend(), ["data:image/jpeg;base64,A"])
        assert r.objects and r.objects[0]["name"] == "screwdriver"

    def test_describe_no_images(self) -> None:
        r = describe_workspace(FakeVisionBackend(), None)
        assert not r.objects and r.confidence == 0.0

    def test_local_holding(self) -> None:
        assert local_holding_state(9.5, 1.5)["holding"] is True
        assert local_holding_state(2.0, 1.5)["holding"] is False
