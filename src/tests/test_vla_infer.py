"""SmolVLA 真机推理 runner 测试（4.1）— 纯逻辑，不连机械臂、不加载模型。

主循环 `run_vla_loop` 的依赖全部注入，因此这里能用假策略/假引擎把闭环语义钉死：
步数、动作关节顺序、提前结束、每步 reset、维度校验、缺图暴露。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from real_observation import EMPTY_GRIP_BASELINE, GRIP_POS_MARGIN, LIFT_WRIST, STATE_NAMES  # noqa: E402
from run_vla_infer import (  # noqa: E402
    DEFAULT_STEPS,
    GRIPPER_LOAD_KEY,
    IMAGE_KEYS,
    GripperTorqueGuard,
    gripper_load,
    holding_and_lifted,
    place_success_checker,
    run_dry,
    run_vla_loop,
    safe_close,
    should_park,
    success_checker,
)

HELD_STATE = {"gripper.pos": 6.8, "wrist_flex.pos": LIFT_WRIST}


class _FakePolicy:
    def __init__(self, action: np.ndarray | None = None) -> None:
        self.calls = 0
        self.reset_calls = 0
        self.batches: list[Any] = []
        self._action = action

    def reset(self) -> None:
        self.reset_calls += 1

    def select_action(self, batch: Any) -> np.ndarray:
        self.calls += 1
        self.batches.append(batch)
        return self._action if self._action is not None else np.zeros(len(STATE_NAMES))


class _FakeEngine:
    """按预设状态序列返回状态的假引擎。"""

    def __init__(self, states: list[dict[str, float]] | None = None,
                 images: dict[str, Any] | None = None, load: float | None = -206.0,
                 fail_load: bool = False) -> None:
        self.states = states or [dict(HELD_STATE)]
        self.images = images
        self.load = load
        self.fail_load = fail_load
        self.sent: list[dict[str, float]] = []
        self.reads = 0

    def read_state(self) -> dict[str, float]:
        self.reads += 1
        return dict(self.states[min(self.reads - 1, len(self.states) - 1)])

    def read_images(self) -> dict[str, Any] | None:
        return self.images

    def read_load(self) -> dict[str, float]:
        if self.fail_load:
            raise RuntimeError("总线读不到负载")
        return {GRIPPER_LOAD_KEY: self.load} if self.load is not None else {}

    def send_action(self, action: dict[str, float]) -> None:
        self.sent.append(dict(action))


def _images() -> dict[str, np.ndarray]:
    return {key: np.zeros((8, 8, 3), dtype=np.uint8) for key in IMAGE_KEYS}


def _prepare(state, images, task):
    return {"observation.state": state, "task": task, "n_images": len(images)}


def _post(action, action_dim=6):
    return np.asarray(action, dtype=float).reshape(-1)[:action_dim]


class TestDefaultSteps:
    def test_covers_shortest_demo_episode(self) -> None:
        """演示集最短一集 340 帧；默认步数低于它会必然跑不完（旧默认 30）。"""
        assert DEFAULT_STEPS >= 340


class TestSafeClose:
    """收尾断开异常不得覆盖任务结论（2026-10-18 夹爪过载把 success 覆盖成 exit 1）。"""

    def test_normal_close_returns_true(self) -> None:
        class _Engine:
            closed = False

            def close(self) -> None:
                self.closed = True

        engine = _Engine()
        assert safe_close(engine) is True
        assert engine.closed is True

    def test_close_error_is_swallowed(self) -> None:
        class _Engine:
            def close(self) -> None:
                raise RuntimeError("Failed to write 'Torque_Enable' [RxPacketError] Overload error!")

        assert safe_close(_Engine()) is False


class TestShouldPark:
    """收尾复位口径：成功默认保持位姿；失败必复位；--no-park 一律不复位。"""

    def test_success_keeps_pose_by_default(self) -> None:
        assert should_park(True, no_park=False, park_on_success=False) is False

    def test_success_parks_when_explicit(self) -> None:
        assert should_park(True, no_park=False, park_on_success=True) is True

    def test_failure_always_parks(self) -> None:
        assert should_park(False, no_park=False, park_on_success=False) is True

    def test_no_park_wins_over_everything(self) -> None:
        assert should_park(False, no_park=True, park_on_success=True) is False


class TestHoldingAndLifted:
    def test_held_and_lifted(self) -> None:
        assert holding_and_lifted(HELD_STATE, load=-206.0) is True

    def test_held_but_not_lifted(self) -> None:
        """夹住但没抬起来 → 不算成功（与 infer_so101 的"抬起"口径一致）。"""
        state = {"gripper.pos": 6.8, "wrist_flex.pos": LIFT_WRIST - 1.0}
        assert holding_and_lifted(state, load=-206.0) is False

    def test_lifted_but_not_held(self) -> None:
        state = {"gripper.pos": EMPTY_GRIP_BASELINE, "wrist_flex.pos": LIFT_WRIST + 10.0}
        assert holding_and_lifted(state, load=0.0) is False

    def test_load_judge_rescues_thin_object(self) -> None:
        """薄件位置贴基线时，负载判据应能判出夹住（位置 OR 负载）。"""
        thin = {"gripper.pos": EMPTY_GRIP_BASELINE + GRIP_POS_MARGIN * 0.5,
                "wrist_flex.pos": LIFT_WRIST + 5.0}
        assert holding_and_lifted(thin, load=None) is False
        assert holding_and_lifted(thin, load=-206.0) is True


class TestGripperLoad:
    def test_reads_from_engine(self) -> None:
        assert gripper_load(_FakeEngine(load=-180.0)) == pytest.approx(-180.0)

    def test_missing_key_gives_none(self) -> None:
        assert gripper_load(_FakeEngine(load=None)) is None

    def test_bus_failure_gives_none(self) -> None:
        """负载读不到不能连累整轮：返回 None，由位置判据兜底。"""
        assert gripper_load(_FakeEngine(fail_load=True)) is None

    def test_success_checker_empty_closed_with_high_load_is_not_success(self) -> None:
        """回归：空夹顶限位会出大负载，但**不得**被判成功（2026-10-17 真机假阳性）。"""
        state = {"gripper.pos": 3.77, "wrist_flex.pos": LIFT_WRIST + 5}
        engine = _FakeEngine(states=[state], load=-206.0)
        assert success_checker(engine)(state) is False

    def test_success_checker_real_grasp_and_lift_is_success(self) -> None:
        state = {"gripper.pos": HELD_STATE["gripper.pos"], "wrist_flex.pos": LIFT_WRIST + 5}
        engine = _FakeEngine(states=[state], load=-100.0)
        assert success_checker(engine)(state) is True

    def test_success_checker_wide_open_with_lift_is_not_success(self) -> None:
        """回归：张开（19.25）后抬起**不得**判成功（2026-10-18 第 3 次真机假阳性）。"""
        state = {"gripper.pos": 19.25, "wrist_flex.pos": LIFT_WRIST + 0.45}
        engine = _FakeEngine(states=[state], load=62.0)
        assert success_checker(engine)(state) is False


class TestPlaceSuccessChecker:
    """搬运类任务：抓起来就判成功不行，原地松开也不行，必须"搬运后释放"。"""

    def test_hold_and_lift_alone_is_not_success(self) -> None:
        check = place_success_checker()
        state = {"gripper.pos": 7.0, "wrist_flex.pos": LIFT_WRIST + 5, "shoulder_pan.pos": -5.0}
        assert check(state) is False

    def test_release_after_carry_is_success(self) -> None:
        check = place_success_checker()
        grip = {"gripper.pos": 7.0, "wrist_flex.pos": LIFT_WRIST + 5}
        assert check({**grip, "shoulder_pan.pos": -5.0}) is False
        assert check({**grip, "shoulder_pan.pos": -25.0}) is False  # 搬运中
        assert check(
            {"gripper.pos": 20.0, "wrist_flex.pos": LIFT_WRIST + 3, "shoulder_pan.pos": -25.0}
        ) is True

    def test_release_in_place_is_not_success(self) -> None:
        """回归：2026-10-19 实测"夹住 5 步后原地张开"曾被判成功。"""
        check = place_success_checker()
        assert check(
            {"gripper.pos": 7.0, "wrist_flex.pos": LIFT_WRIST + 2, "shoulder_pan.pos": -5.0}
        ) is False
        assert check(
            {"gripper.pos": 16.1, "wrist_flex.pos": LIFT_WRIST + 2, "shoulder_pan.pos": -5.0}
        ) is False

    def test_release_without_carry_is_not_success(self) -> None:
        """一开始就张开（没夹住过）→ 不能判成功。"""
        check = place_success_checker()
        assert check(
            {"gripper.pos": 22.0, "wrist_flex.pos": LIFT_WRIST + 20, "shoulder_pan.pos": 10.0}
        ) is False

    def test_hold_without_lift_then_open_is_not_success(self) -> None:
        """夹住但没抬起 → 不算搬运过，后续张开也不判成功。"""
        check = place_success_checker()
        assert check(
            {"gripper.pos": 7.0, "wrist_flex.pos": LIFT_WRIST - 10, "shoulder_pan.pos": -5.0}
        ) is False
        assert check(
            {"gripper.pos": 20.0, "wrist_flex.pos": LIFT_WRIST + 5, "shoulder_pan.pos": -25.0}
        ) is False

    def test_drop_while_carrying_then_open_is_not_success(self) -> None:
        """搬运途中掉落（夹爪位置回落到张开带以下）→ 之后张开也不算完成。"""
        check = place_success_checker()
        assert check(
            {"gripper.pos": 7.0, "wrist_flex.pos": LIFT_WRIST + 5, "shoulder_pan.pos": -5.0}
        ) is False
        assert check(
            {"gripper.pos": 4.0, "wrist_flex.pos": LIFT_WRIST + 5, "shoulder_pan.pos": -15.0}
        ) is False
        assert check(
            {"gripper.pos": 20.0, "wrist_flex.pos": LIFT_WRIST + 5, "shoulder_pan.pos": -25.0}
        ) is False


class TestGripperTorqueGuard:
    """过载保护：连续超限 3 s 冻结在当前位置，回落 1 s 后解除。"""

    def test_not_frozen_before_hold_window(self) -> None:
        guard = GripperTorqueGuard()
        assert guard.update(200.0, 7.0, 0.0) is None
        assert guard.update(200.0, 7.0, 2.9) is None
        assert guard.frozen is False

    def test_frozen_after_hold_window(self) -> None:
        guard = GripperTorqueGuard()
        for t in (0.0, 1.5, 3.0):
            frozen = guard.update(200.0, 7.0, t)
        assert guard.frozen is True
        assert frozen == pytest.approx(7.0)

    def test_frozen_holds_current_position_while_load_drops(self) -> None:
        """冻结后即使负载回落（<1 s），也必须继续下发当前位姿而不是放开。"""
        guard = GripperTorqueGuard()
        guard.update(200.0, 7.0, 0.0)
        guard.update(200.0, 7.0, 3.0)
        assert guard.update(20.0, 7.1, 3.5) == pytest.approx(7.1)
        assert guard.frozen is True

    def test_released_after_clear_window(self) -> None:
        guard = GripperTorqueGuard()
        guard.update(200.0, 7.0, 0.0)
        guard.update(200.0, 7.0, 3.0)
        assert guard.update(20.0, 7.0, 3.2) == pytest.approx(7.0)
        assert guard.update(20.0, 7.0, 4.4) is None
        assert guard.frozen is False

    def test_unknown_load_keeps_frozen(self) -> None:
        """总线读不到负载时不能假设安全：已冻结就保持冻结。"""
        guard = GripperTorqueGuard()
        guard.update(200.0, 7.0, 0.0)
        guard.update(200.0, 7.0, 3.0)
        assert guard.update(None, 7.0, 5.0) == pytest.approx(7.0)
        assert guard.frozen is True


class TestRunVlaLoop:
    def _run(self, **kwargs):
        defaults = dict(
            policy=_FakePolicy(),
            engine=_FakeEngine(states=[dict(HELD_STATE)], load=-206.0),
            instruction="夹起螺丝刀",
            steps=3,
            prepare=_prepare,
            post=_post,
            success_fn=lambda state: False,
        )
        defaults.update(kwargs)
        return run_vla_loop(**defaults)

    def test_guard_freezes_gripper_target(self) -> None:
        """连续过载 3 s 后，夹爪目标必须被改写成当前实际位置。"""
        class _ZeroGripPolicy(_FakePolicy):
            def select_action(self, batch):
                self.calls += 1
                action = np.full(len(STATE_NAMES), 5.0)
                action[STATE_NAMES.index("gripper.pos")] = 0.0  # 策略要求全闭
                return action

        clock = iter([0.0, 1.0, 2.0, 3.0, 4.0])
        engine = _FakeEngine(states=[{"gripper.pos": 7.0}], load=200.0)
        summary = self._run(policy=_ZeroGripPolicy(), engine=engine, steps=5,
                            guard=GripperTorqueGuard(), now_fn=lambda: next(clock))
        assert summary["steps_sent"] == 5
        grips = [a["gripper.pos"] for a in engine.sent]
        assert grips[:3] == pytest.approx([0.0, 0.0, 0.0])   # 3 s 内仍听策略
        assert grips[3:] == pytest.approx([7.0, 7.0])        # 触发后保持当前位姿

    def test_guard_keeps_other_joints_following_policy(self) -> None:
        clock = iter([0.0, 1.0, 2.0, 3.0])
        engine = _FakeEngine(states=[{"gripper.pos": 7.0}], load=200.0)
        self._run(engine=engine, steps=4, guard=GripperTorqueGuard(),
                  now_fn=lambda: next(clock))
        assert all(a["wrist_flex.pos"] == pytest.approx(0.0) for a in engine.sent)

    def test_sends_one_action_per_step(self) -> None:
        engine = _FakeEngine(states=[dict(HELD_STATE)], load=-206.0)
        policy = _FakePolicy()
        summary = self._run(policy=policy, engine=engine)
        assert summary["steps_sent"] == 3
        assert len(engine.sent) == 3
        assert policy.calls == 3

    def test_action_keys_follow_state_names_order(self) -> None:
        """动作必须按 `STATE_NAMES` 定序发下去：错序会让关节各动各的，且不易察觉。"""
        engine = _FakeEngine(load=-206.0)
        action = np.arange(len(STATE_NAMES), dtype=float)
        self._run(policy=_FakePolicy(action=action), engine=engine)
        assert list(engine.sent[0]) == list(STATE_NAMES)
        assert list(engine.sent[0].values()) == pytest.approx(list(action))

    def test_stops_early_on_success(self) -> None:
        engine = _FakeEngine(load=-206.0)
        policy = _FakePolicy()
        summary = self._run(policy=policy, engine=engine,
                            success_fn=lambda state: True)
        assert summary["success"] is True
        assert summary["steps_sent"] == 1
        assert policy.calls == 1

    def test_reset_per_step_calls_reset_each_step(self) -> None:
        policy = _FakePolicy()
        self._run(policy=policy, steps=4, reset_per_step=True)
        assert policy.reset_calls == 4

    def test_no_reset_by_default(self) -> None:
        policy = _FakePolicy()
        self._run(policy=policy, steps=4)
        assert policy.reset_calls == 0

    def test_prepare_receives_six_dim_state(self) -> None:
        policy = _FakePolicy()
        self._run(policy=policy, steps=1)
        batch = policy.batches[0]
        assert np.asarray(batch["observation.state"]).shape == (len(STATE_NAMES),)

    def test_missing_images_are_reported(self) -> None:
        """视觉策略喂空图会让动作莫名变差，故缺图必须显式暴露给调用方。"""
        summary = self._run(engine=_FakeEngine(images=None, load=-206.0), steps=1)
        assert summary["missing_images"] == IMAGE_KEYS

    def test_wrong_action_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="维度"):
            self._run(policy=_FakePolicy(action=np.zeros(3)), steps=1)

    def test_zero_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="steps"):
            self._run(steps=0)

    def test_sleep_fn_invoked(self) -> None:
        calls: list[int] = []
        self._run(steps=2, sleep_fn=lambda: calls.append(1))
        assert len(calls) == 2


class TestRunDry:
    def test_dry_run_uses_mock_engine_and_sends_actions(self) -> None:
        args = argparse.Namespace(instruction="夹起螺丝刀", steps=3, reset_per_step=False)
        assert run_dry(args) == 0

    def test_dry_run_with_reset(self) -> None:
        args = argparse.Namespace(instruction="夹起螺丝刀", steps=2, reset_per_step=True)
        assert run_dry(args) == 0


class TestMockEngineSendAction:
    def test_records_actions(self) -> None:
        from real_executor import MockEngine

        engine = MockEngine(with_images=True)
        engine.send_action({"shoulder_pan.pos": 1.0, "gripper.pos": 2.0})
        assert engine.sent_actions == [{"shoulder_pan.pos": 1.0, "gripper.pos": 2.0}]

    def test_mock_load_has_gripper_key(self) -> None:
        """Mock 的负载键名必须与真机一致，否则判据在 dry-run 里测不到。"""
        from real_executor import MockEngine

        assert GRIPPER_LOAD_KEY in MockEngine(grasp_result="success").read_load()
