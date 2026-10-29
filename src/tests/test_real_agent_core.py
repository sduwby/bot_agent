"""real_agent 决策分发/失败恢复主循环的单元测试 — 使用 Mock 引擎与假 LLM，无机械臂。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_planner import LLMBackend
from real_agent import RealRobotAgent
from real_executor import MockEngine, RealSkillExecutor
from real_skills import FAIL_GRASP_EMPTY, FAIL_HOME_TIMEOUT, FAIL_NOT_HOLDING


class ScriptedBackend(LLMBackend):
    """按脚本依次返回技能；默认：grasp_screwdriver → done。记录收到的 images。"""

    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.i = 0
        self.seen_images: list[list[str] | None] = []

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        self.seen_images.append(images)
        skill = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return json.dumps({"thought": "t", "skill": skill, "parameters": {}})


class ParseFailBackend(LLMBackend):
    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        return "不是 JSON"


def make_agent(mock_result: str = "success", retries: int = 1, max_plan: int = 6):
    engine = MockEngine(grasp_result=mock_result)
    executor = RealSkillExecutor(engine, empty_grip_baseline=1.5)
    return RealRobotAgent(executor, max_plan_steps=max_plan, max_retries=retries)


class TestAgentMainLoop:
    def test_success_path_grasp_then_done(self) -> None:
        agent = make_agent()
        agent.backend = ScriptedBackend(["grasp_screwdriver", "done"])
        out = agent.run("夹起螺丝刀")
        assert out.success
        assert out.steps[0].skill == "grasp_screwdriver"
        assert out.steps[-1].skill == "done"

    def test_parse_failure_aborts(self) -> None:
        agent = make_agent()
        agent.backend = ParseFailBackend()
        out = agent.run("夹起螺丝刀")
        assert not out.success
        assert "无法解析" in out.message

    def test_grasp_empty_retries_then_done(self) -> None:
        # 咬空 → 重试（Mock 第二次仍咬空，但 agent 默认视为抓取失败继续）→ 需人工确认路径
        agent = make_agent(mock_result="grasp_empty", retries=1)
        agent.backend = ScriptedBackend(["grasp_screwdriver", "grasp_screwdriver", "done"])
        out = agent.run("夹起螺丝刀")
        # 重试后仍失败则终止（不进入 done）
        assert not out.success
        assert any("重试" in s.recovery_note for s in out.steps)

    def test_over_retry_limit_aborts(self) -> None:
        agent = make_agent(mock_result="grasp_empty", retries=0)
        agent.backend = ScriptedBackend(["grasp_screwdriver", "grasp_screwdriver"])
        out = agent.run("夹起螺丝刀")
        assert not out.success
        assert "超出重试次数" in out.message

    def test_comm_error_aborts_immediately(self) -> None:
        engine = MockEngine(grasp_result="success")

        class BadEngine(MockEngine):
            def read_state(self):
                raise RuntimeError("串口断开")

        agent = RealRobotAgent(
            RealSkillExecutor(BadEngine(), empty_grip_baseline=1.5),
            max_retries=1,
        )
        # 触发 comm：让 verify_holding 读取状态时抛异常
        agent.backend = ScriptedBackend(["verify_holding"])
        out = agent.run("检查是否夹住")
        assert not out.success

    def test_max_plan_steps_reached(self) -> None:
        agent = make_agent(max_plan=2)
        agent.backend = ScriptedBackend(["grasp_screwdriver", "grasp_screwdriver", "grasp_screwdriver"])
        out = agent.run("夹起螺丝刀")
        assert not out.success
        assert "最大规划步数" in out.message


class TestSkillExecutorMock:
    def test_unknown_skill_rejected(self) -> None:
        executor = RealSkillExecutor(MockEngine(), empty_grip_baseline=1.5)
        r = executor.execute("fly")
        assert not r.success
        assert "未知技能" in r.failure_reason

    def test_schema_extra_params_lenient(self) -> None:
        # 参数校验仅检查必填与 enum；多余参数不报错（与阶段 A skill_registry 行为一致）
        executor = RealSkillExecutor(MockEngine(), empty_grip_baseline=1.5)
        r = executor.execute("release", params={"bogus": 1})
        assert r.success

    def test_release_and_verify_on_mock(self) -> None:
        engine = MockEngine(grasp_result="success")
        executor = RealSkillExecutor(engine, empty_grip_baseline=1.5)
        assert executor.execute("verify_holding").success
        assert executor.execute("release").success


class TestExecutorLoadSignal:
    """0.6 之后执行器把负载读数接进夹持判定。"""

    def test_mock_load_follows_grasp_result(self) -> None:
        """Mock 的负载必须与抓取结果同口径，否则「位置 OR 负载」会把咬空判成夹持。"""
        assert MockEngine(grasp_result="grasp_empty").read_load()["gripper.load"] == pytest.approx(-62.0)
        assert MockEngine(grasp_result="success").read_load()["gripper.load"] == pytest.approx(-206.0)

    def test_mock_load_override_is_honored(self) -> None:
        assert MockEngine(grasp_result="success", gripper_load=55.0).read_load()["gripper.load"] == 55.0

    def test_load_rescues_ultra_thin_grasp(self) -> None:
        """位置只到 4.2（低于阈值 5.5），负载 −206‰ → 仍判夹住。"""
        engine = MockEngine(grasp_result="success")
        engine.state["gripper.pos"] = 4.2
        executor = RealSkillExecutor(engine)
        assert executor.execute("grasp_screwdriver").success

    def test_empty_position_with_empty_load_is_grasp_empty(self) -> None:
        engine = MockEngine(grasp_result="grasp_empty")
        executor = RealSkillExecutor(engine)
        r = executor.execute("grasp_screwdriver")
        assert not r.success and r.failure_reason == FAIL_GRASP_EMPTY

    def test_engine_without_load_degrades_to_position(self) -> None:
        """引擎不实现 read_load 时，不能因为读不到负载就判成夹住。"""

        class NoLoadEngine(MockEngine):
            def read_load(self):
                raise AttributeError("此引擎不支持负载读取")

        engine = NoLoadEngine(grasp_result="success")
        engine.state["gripper.pos"] = 3.9  # 空载量级位置
        executor = RealSkillExecutor(engine)
        assert executor.execute("verify_holding").failure_reason == FAIL_NOT_HOLDING


class TestGraspPipelineGuards:
    """grasp 技能的两条前置保障：先归位、超时预算按步数缩放。"""

    def test_timeout_scales_with_steps(self) -> None:
        """默认 300+300 步 × 0.5 s/步 = 300 s（旧实现写死 120 s 必然超时）。"""
        assert RealSkillExecutor(MockEngine()).grasp_timeout() == pytest.approx(300.0)

    def test_timeout_keeps_floor_for_small_runs(self) -> None:
        assert RealSkillExecutor(MockEngine(), max_grasp_steps=50,
                                 hold_steps=50).grasp_timeout() == pytest.approx(120.0)

    def test_explicit_timeout_wins(self) -> None:
        ex = RealSkillExecutor(MockEngine(), grasp_timeout_s=42.0)
        assert ex.grasp_timeout() == pytest.approx(42.0)

    def test_home_failure_aborts_before_policy(self) -> None:
        """归位失败必须直接失败，且**不进入策略段**——否则会从错误起跑位姿空跑 600 步。"""
        engine = MockEngine(grasp_result="success")
        engine.home = lambda timeout_s=None, **_: False
        result = RealSkillExecutor(engine).execute("grasp_screwdriver")
        assert not result.success and result.failure_reason == FAIL_HOME_TIMEOUT
        assert engine.grasp_calls == 0

    def test_home_runs_before_policy(self) -> None:
        calls: list[str] = []
        engine = MockEngine(grasp_result="success")
        engine.home = lambda timeout_s=None, **_: (calls.append("home"), True)[1]
        original = engine.grasp_and_lift
        engine.grasp_and_lift = lambda s, h: (calls.append("grasp"), original(s, h))[1]
        assert RealSkillExecutor(engine).execute("grasp_screwdriver").success
        assert calls == ["home", "grasp"]


class TestPark:
    """收尾复位：运行结束后把机械臂送回采集初始姿态。"""

    def test_park_homes_the_arm(self) -> None:
        calls: list[str] = []
        engine = MockEngine()
        engine.home = lambda timeout_s=None, **_: (calls.append("home"), True)[1]
        assert RealSkillExecutor(engine).park() is True
        assert calls == ["home"]

    def test_park_reports_failure_without_raising(self) -> None:
        """复位失败（如超时）只返回 False —— 收尾动作不该盖过主流程结论。"""
        engine = MockEngine()
        engine.home = lambda timeout_s=None, **_: False
        assert RealSkillExecutor(engine).park() is False

    def test_park_swallows_engine_errors(self) -> None:
        engine = MockEngine()

        def boom(timeout_s=None, **_):
            raise RuntimeError("串口断开")

        engine.home = boom
        assert RealSkillExecutor(engine).park() is False

    def test_park_can_skip_gripper(self) -> None:
        """归位可以只复位臂关节：`include_gripper=False` 必须透传到引擎。"""
        seen: list[bool] = []

        def fake_home(timeout_s=None, *, include_gripper=True):
            seen.append(include_gripper)
            return True

        engine = MockEngine()
        engine.home = fake_home
        assert RealSkillExecutor(engine).park(include_gripper=False) is True
        assert seen == [False]


class TestHomeSkipGripper:
    """归位跳过夹爪：既不下发目标，也不参与到达判定。"""

    def test_skip_gripper_sends_only_arm_joints(self, monkeypatch) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "outputs"))
        import infer_so101

        sent: list[dict[str, float]] = []

        class _Robot:
            def send_action(self, action):
                sent.append(dict(action))

        class _DS:
            pass

        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 20.0], dtype=np.float32)  # 夹爪不在初始位
        monkeypatch.setattr(
            infer_so101, "build_observation_from_robot",
            lambda robot, ds: {"observation.state": state.copy()},
        )
        monkeypatch.setattr(infer_so101, "busy_wait", lambda dt: None)

        initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.56], dtype=np.float32)
        ok = infer_so101.home_to_initial(
            _Robot(), _DS(), initial, timeout_s=0.5, skip_joints=("gripper.pos",)
        )

        assert ok is True  # 夹爪不参与判定，故 5 个臂关节到位即可
        assert sent and "gripper.pos" not in sent[0]
        assert set(sent[0]) == set(infer_so101.STATE_NAMES) - {"gripper.pos"}


class TestAgentVision:
    def test_images_passed_to_backend(self) -> None:
        # MockEngine 返回合成帧 → backend 应收到两条 data URL（fixed + handeye）
        engine = MockEngine(grasp_result="success", with_images=True)
        executor = RealSkillExecutor(engine, empty_grip_baseline=1.5)
        backend = ScriptedBackend(["grasp_screwdriver", "done"])
        agent = RealRobotAgent(executor, backend=backend, max_plan_steps=4, vision=True)
        out = agent.run("夹起螺丝刀")
        assert out.success
        assert backend.seen_images, "backend 应至少收到一次图像"
        for imgs in backend.seen_images:
            assert imgs is not None and len(imgs) == 2
            assert imgs[0].startswith("data:image/jpeg;base64,")

    def test_vision_disabled_sends_none(self) -> None:
        engine = MockEngine(grasp_result="success", with_images=True)
        executor = RealSkillExecutor(engine, empty_grip_baseline=1.5)
        backend = ScriptedBackend(["grasp_screwdriver", "done"])
        agent = RealRobotAgent(executor, backend=backend, max_plan_steps=4, vision=False)
        out = agent.run("夹起螺丝刀")
        assert out.success
        assert all(imgs is None for imgs in backend.seen_images)

    def test_engine_image_failure_degrades_to_text(self) -> None:
        # read_images 抛异常 → 降级纯文本，不阻断主循环
        engine = MockEngine(grasp_result="success", with_images=True)

        def boom():
            raise RuntimeError("相机断开")

        engine.read_images = boom
        executor = RealSkillExecutor(engine, empty_grip_baseline=1.5)
        backend = ScriptedBackend(["grasp_screwdriver", "done"])
        agent = RealRobotAgent(executor, backend=backend, max_plan_steps=4, vision=True)
        out = agent.run("夹起螺丝刀")
        assert out.success
        assert all(imgs is None for imgs in backend.seen_images)
