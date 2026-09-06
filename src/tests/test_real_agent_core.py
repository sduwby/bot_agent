"""real_agent 决策分发/失败恢复主循环的单元测试 — 使用 Mock 引擎与假 LLM，无机械臂。"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_planner import LLMBackend
from real_agent import RealRobotAgent
from real_executor import MockEngine, RealSkillExecutor


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
