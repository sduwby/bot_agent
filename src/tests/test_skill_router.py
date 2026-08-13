"""第 20 周：分层策略路由定向测试。

覆盖：路由配置校验、StagnationMonitor 窗口行为、异常/停滞/超时切换、
末位 scripted 兜底、exhausted 路径、optional 位跳过与聚合评测。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from skill_router import (
    PolicySource,
    SkillRouteConfig,
    SkillRouter,
    SkillRouterError,
    StagnationMonitor,
    evaluate_routes,
    scripted_source,
)
from maniskill_intro import ManiSkillTaskConfig


class _StepPolicy:
    """记录调用并返回可配置动作的假策略。"""

    phase = "fake"

    def __init__(self, *, name: str, action: np.ndarray, fail: bool = False) -> None:
        self.name = name
        self._action = np.asarray(action, dtype=np.float32)
        self.fail = fail
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, observation: object) -> np.ndarray:
        self.act_calls += 1
        if self.fail:
            raise RuntimeError("fake act failure")
        return self._action.copy()


class _StepPolicyFactory:
    """按实例顺序逐个提供策略的工厂。"""

    def __init__(self, policies: list[object]) -> None:
        self._policies = list(policies)
        self.calls = 0

    def __call__(self, env_id: str) -> object:
        if self.calls >= len(self._policies):
            raise SkillRouterError("fake factory 实例耗尽")
        policy = self._policies[self.calls]
        self.calls += 1
        return policy


class _FakeEnv:
    """单 episode 假环境：脚本化的 step 行为。"""

    def __init__(self, *, success_after: int | None = None) -> None:
        self.reset_calls = 0
        self.steps = 0
        self._success_after = success_after
        self._observation = {
            "extra": {
                "tcp_pose": np.zeros(3, dtype=np.float32),
                "obj_pose": np.zeros(3, dtype=np.float32),
                "is_grasped": np.asarray(False),
            }
        }
        self.action_space = type(
            "Space",
            (),
            {"shape": (4,), "contains": lambda self, action: True},
        )()
        self._image = np.zeros((8, 8, 3), dtype=np.uint8)

    def reset(self, seed: int | None = None):
        self.reset_calls += 1
        self.steps = 0
        return (self._observation, {})

    def step(self, action: object):
        self.steps += 1
        success = (
            self._success_after is not None
            and self.steps >= self._success_after
        )
        return (
            self._observation,
            0.0,
            success,
            False,
            {"success": success},
        )

    def render(self) -> np.ndarray:
        return self._image

    def close(self) -> None:
        pass


def _config(
    policies: list[object],
    *,
    max_steps: int = 50,
    stagnation_window: int = 8,
) -> SkillRouteConfig:
    sources = [
        PolicySource(
            name=f"p{i}",
            kind="scripted",
            input_mode="state",
            factory=_StepPolicyFactory([policy]),
        )
        for i, policy in enumerate(policies)
    ]
    return SkillRouteConfig(
        env_id="PickCube-v1",
        sources=tuple(sources),
        max_steps=max_steps,
        stagnation_window=stagnation_window,
    )


class TestPolicySource:
    def test_valid(self) -> None:
        source = PolicySource(
            name="scripted",
            kind="scripted",
            input_mode="state",
            factory=scripted_source().factory,
        )
        assert source.name == "scripted"
        assert source.optional is False

    def test_invalid_kind(self) -> None:
        with pytest.raises(ValueError):
            PolicySource(
                name="x", kind="unknown", input_mode="state", factory=lambda e: None
            )

    def test_invalid_input_mode(self) -> None:
        with pytest.raises(ValueError):
            PolicySource(
                name="x", kind="scripted", input_mode="audio", factory=lambda e: None
            )

    def test_invalid_factory(self) -> None:
        with pytest.raises(ValueError):
            PolicySource(
                name="x", kind="scripted", input_mode="state", factory="nope"
            )


class TestSkillRouteConfig:
    def test_valid(self) -> None:
        config = _config([_StepPolicy(name="a", action=np.zeros(4))])
        assert config.env_id == "PickCube-v1"

    def test_empty_sources_rejected(self) -> None:
        with pytest.raises(ValueError):
            SkillRouteConfig(env_id="PickCube-v1", sources=())

    def test_unsupported_task_rejected(self) -> None:
        with pytest.raises(ValueError):
            SkillRouteConfig(
                env_id="Grasp-v1",
                sources=(scripted_source(),),
            )

    def test_invalid_max_steps(self) -> None:
        with pytest.raises(ValueError):
            SkillRouteConfig(
                env_id="PickCube-v1",
                sources=(scripted_source(),),
                max_steps=0,
            )

    def test_invalid_stagnation_window(self) -> None:
        with pytest.raises(ValueError):
            SkillRouteConfig(
                env_id="PickCube-v1",
                sources=(scripted_source(),),
                stagnation_window=0,
            )

    def test_window_larger_than_max_steps_rejected(self) -> None:
        with pytest.raises(ValueError):
            SkillRouteConfig(
                env_id="PickCube-v1",
                sources=(scripted_source(),),
                max_steps=5,
                stagnation_window=10,
            )

    def test_invalid_move_threshold(self) -> None:
        with pytest.raises(ValueError):
            SkillRouteConfig(
                env_id="PickCube-v1",
                sources=(scripted_source(),),
                stagnation_move_threshold=2.0,
            )


class TestStagnationMonitor:
    def test_no_stagnation_before_window_full(self) -> None:
        monitor = StagnationMonitor(window=4, move_threshold=0.005)
        observation = {
            "extra": {"tcp_pose": np.zeros(3), "is_grasped": np.asarray(False)}
        }
        for _ in range(3):
            monitor.update(observation)
        assert monitor.is_stagnant is False

    def test_stagnant_when_tcp_still(self) -> None:
        monitor = StagnationMonitor(window=4, move_threshold=0.005)
        observation = {
            "extra": {"tcp_pose": np.zeros(3), "is_grasped": np.asarray(False)}
        }
        for _ in range(4):
            monitor.update(observation)
        assert monitor.is_stagnant is True

    def test_moving_tcp_not_stagnant(self) -> None:
        monitor = StagnationMonitor(window=4, move_threshold=0.005)
        for index in range(4):
            monitor.update(
                {
                    "extra": {
                        "tcp_pose": np.asarray(
                            [index * 0.02, 0.0, 0.0], dtype=float
                        ),
                        "is_grasped": np.asarray(False),
                    }
                }
            )
        assert monitor.is_stagnant is False

    def test_reset_clears_window(self) -> None:
        monitor = StagnationMonitor(window=4, move_threshold=0.005)
        observation = {
            "extra": {"tcp_pose": np.zeros(3), "is_grasped": np.asarray(False)}
        }
        for _ in range(4):
            monitor.update(observation)
        assert monitor.is_stagnant is True
        monitor.reset()
        assert monitor.is_stagnant is False
        assert monitor.samples == 0

    def test_missing_tcp_treated_as_still(self) -> None:
        monitor = StagnationMonitor(window=3, move_threshold=0.005)
        observation = {"extra": {}}
        for _ in range(3):
            monitor.update(observation)
        assert monitor.is_stagnant is True


class TestSkillRouterRunEpisode:
    def _router(self) -> tuple[SkillRouter, _FakeEnv]:
        env = _FakeEnv()
        router = SkillRouter(env_factory=lambda config: env)
        return router, env

    def test_first_policy_succeeds_no_switch(self) -> None:
        env = _FakeEnv(success_after=2)
        router = SkillRouter(env_factory=lambda config: env)
        policy = _StepPolicy(name="a", action=np.zeros(4))
        config = _config([policy])
        result = router.run_episode(config, seed=0)
        assert policy.act_calls >= 1
        assert result.switch_events == ()
        assert result.final_policy == "p0"
        assert result.exhausted is False
        assert result.success is True

    def test_exception_switches_to_next(self) -> None:
        router, _ = self._router()
        failing = _StepPolicy(name="a", action=np.zeros(4), fail=True)
        recovering = _StepPolicy(name="b", action=np.zeros(4))
        config = _config([failing, recovering])
        result = router.run_episode(config, seed=0)
        assert recovering.act_calls >= 1
        assert result.switch_events
        event = result.switch_events[0]
        assert event.from_policy == "p0"
        assert event.to_policy == "p1"
        assert event.reason == "exception"

    def test_exception_on_last_policy_is_exhausted(self) -> None:
        router, _ = self._router()
        failing = _StepPolicy(name="a", action=np.zeros(4), fail=True)
        config = _config([failing])
        result = router.run_episode(config, seed=0)
        assert result.success is False
        assert result.exhausted is True
        assert result.error is not None

    def test_stagnation_switches_to_next(self) -> None:
        router, _ = self._router()
        stalling = _StepPolicy(name="a", action=np.zeros(4))
        recovering = _StepPolicy(name="b", action=np.zeros(4))
        config = SkillRouteConfig(
            env_id="PickCube-v1",
            sources=(
                PolicySource(
                    name="p0",
                    kind="scripted",
                    input_mode="state",
                    factory=_StepPolicyFactory([stalling]),
                ),
                PolicySource(
                    name="p1",
                    kind="scripted",
                    input_mode="state",
                    factory=_StepPolicyFactory([recovering]),
                ),
            ),
            max_steps=50,
            stagnation_window=4,
        )
        result = router.run_episode(config, seed=0)
        assert recovering.act_calls >= 1
        assert result.switch_events
        assert result.switch_events[0].reason == "stagnation"
        assert result.switch_events[0].at_step >= 4

    def test_max_steps_exhausted_without_success(self) -> None:
        router, _ = self._router()
        policy = _StepPolicy(name="a", action=np.zeros(4))
        config = _config([policy], max_steps=10)
        result = router.run_episode(config, seed=0)
        assert result.steps == 10
        assert result.success is False
        # 末位策略始终停滞 → exhausted。
        assert result.exhausted is True

    def test_reset_called_for_all_policies_at_start(self) -> None:
        env = _FakeEnv(success_after=2)
        router = SkillRouter(env_factory=lambda config: env)
        first = _StepPolicy(name="a", action=np.zeros(4))
        second = _StepPolicy(name="b", action=np.zeros(4))
        config = _config([first, second])
        router.run_episode(config, seed=0)
        assert first.reset_calls == 1
        assert second.reset_calls == 1

    def test_visual_input_mode_uses_render(self) -> None:
        env = _FakeEnv()
        router = SkillRouter(env_factory=lambda config: env)

        class VisualPolicy(_StepPolicy):
            def __init__(self) -> None:
                super().__init__(name="v", action=np.zeros(4))
                self.images_seen = 0

            def act(self, observation: object, image: np.ndarray) -> np.ndarray:
                self.images_seen += 1
                assert image.shape == (8, 8, 3)
                return np.zeros(4, dtype=np.float32)

        policy = VisualPolicy()
        config = SkillRouteConfig(
            env_id="PickCube-v1",
            sources=(
                PolicySource(
                    name="visual",
                    kind="visual_act",
                    input_mode="visual",
                    factory=_StepPolicyFactory([policy]),
                ),
            ),
            max_steps=3,
            stagnation_window=3,
        )
        router.run_episode(config, seed=0)
        assert policy.images_seen >= 1


class TestEvaluateRoutes:
    def test_aggregation_and_report(self, tmp_path: Path) -> None:
        config = SkillRouteConfig(
            env_id="PickCube-v1",
            sources=(
                PolicySource(
                    name="p0",
                    kind="scripted",
                    input_mode="state",
                    factory=lambda env_id: _StepPolicy(
                        name="a", action=np.zeros(4)
                    ),
                ),
            ),
            max_steps=5,
            stagnation_window=5,
        )
        summary = evaluate_routes(
            {"PickCube-v1": config},
            [0, 1],
            env_factory=lambda task_config: _FakeEnv(),
            output_dir=tmp_path,
        )
        assert summary.tasks[0].env_id == "PickCube-v1"
        assert summary.tasks[0].episodes == 2
        assert summary.tasks[0].route == ("p0",)
        assert len(summary.episodes) == 2
        report_path = tmp_path / "router_report.json"
        assert report_path.is_file()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["schema_version"].startswith("maniskill_skill_router")
        assert len(payload["episodes"]) == 2

    def test_mismatched_route_key_rejected(self) -> None:
        config = SkillRouteConfig(
            env_id="PickCube-v1", sources=(scripted_source(),)
        )
        with pytest.raises(ValueError):
            evaluate_routes(
                {"PushCube-v1": config},
                [0],
                env_factory=lambda task_config: _FakeEnv(),
            )

    def test_empty_routes_rejected(self) -> None:
        with pytest.raises(ValueError):
            evaluate_routes(
                {}, [0], env_factory=lambda task_config: _FakeEnv()
            )

    def test_duplicate_seeds_rejected(self) -> None:
        config = SkillRouteConfig(
            env_id="PickCube-v1", sources=(scripted_source(),)
        )
        with pytest.raises(ValueError):
            evaluate_routes(
                {"PickCube-v1": config},
                [0, 0],
                env_factory=lambda task_config: _FakeEnv(),
            )


class TestOptionalSource:
    def test_optional_load_failure_skipped(self) -> None:
        env = _FakeEnv()
        router = SkillRouter(env_factory=lambda config: env)

        def failing_factory(env_id: str) -> object:
            raise SkillRouterError("模型不可用")

        optional = PolicySource(
            name="smolvla",
            kind="smolvla",
            input_mode="visual",
            factory=failing_factory,
            optional=True,
        )
        working = _StepPolicy(name="scripted", action=np.zeros(4))
        config = SkillRouteConfig(
            env_id="PickCube-v1",
            sources=(
                optional,
                PolicySource(
                    name="p1",
                    kind="scripted",
                    input_mode="state",
                    factory=_StepPolicyFactory([working]),
                ),
            ),
            max_steps=3,
            stagnation_window=3,
        )
        result = router.run_episode(config, seed=0)
        assert working.act_calls >= 1
        assert result.final_policy == "p1"
        assert result.switch_events == ()

    def test_required_load_failure_raises(self) -> None:
        env = _FakeEnv()
        router = SkillRouter(env_factory=lambda config: env)

        def failing_factory(env_id: str) -> object:
            raise SkillRouterError("checkpoint 缺失")

        required = PolicySource(
            name="state_act",
            kind="state_act",
            input_mode="state",
            factory=failing_factory,
            optional=False,
        )
        config = SkillRouteConfig(
            env_id="PickCube-v1",
            sources=(required,),
            max_steps=3,
            stagnation_window=3,
        )
        with pytest.raises(SkillRouterError):
            router.run_episode(config, seed=0)


def test_scripted_source_builds_scripted_policy() -> None:
    source = scripted_source()
    policy = source.factory("PickCube-v1")
    from maniskill_intro import ScriptedPolicy

    assert isinstance(policy, ScriptedPolicy)
