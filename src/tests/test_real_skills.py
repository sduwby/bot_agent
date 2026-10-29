"""real_skills / real_verifier / real_observation 的单元测试 — 纯逻辑，无需机械臂。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from real_observation import (
    EMPTY_GRIP_BASELINE,
    GRIP_POS_MARGIN,
    is_holding,
    is_lifted,
    summarize_state,
)
from real_skills import (
    FAIL_GRASP_EMPTY,
    FAIL_NOT_HOLDING,
    FAIL_NOT_LIFTED,
    get_real_skill,
    get_real_skills,
    get_real_skills_description,
    validate_real_parameters,
)
from real_verifier import RealVerifier, classify_failure_code


class TestRealSkills:
    def test_skill_count_and_names(self) -> None:
        names = {s["name"] for s in get_real_skills()}
        assert names == {"grasp_screwdriver", "reset_pose", "verify_holding", "release", "done"}

    def test_each_skill_has_schema(self) -> None:
        for s in get_real_skills():
            assert s["parameters"]["type"] == "object"

    def test_validate_unknown(self) -> None:
        assert validate_real_parameters("fly", {}) is not None

    def test_validate_known_ok(self) -> None:
        assert validate_real_parameters("grasp_screwdriver", {}) is None

    def test_description_contains_skills(self) -> None:
        desc = get_real_skills_description()
        assert "grasp_screwdriver" in desc and "verify_holding" in desc

    def test_get_real_skill(self) -> None:
        assert get_real_skill("release") is not None
        assert get_real_skill("nope") is None


class TestRealVerifier:
    def test_grasp_empty_detected(self) -> None:
        # 空闭基准 1.5 + margin 3 = 4.5；gripper 1.5 咬空但抬起
        v = RealVerifier(empty_grip_baseline=1.5)
        r = v.verify_grasp_lift(final_gripper=1.5, final_wrist=89.0)
        assert not r.success and r.failure_code == FAIL_GRASP_EMPTY

    def test_not_lifted_detected(self) -> None:
        v = RealVerifier(empty_grip_baseline=1.5)
        r = v.verify_grasp_lift(final_gripper=9.5, final_wrist=60.0)
        assert not r.success and r.failure_code == FAIL_NOT_LIFTED

    def test_success(self) -> None:
        v = RealVerifier(empty_grip_baseline=1.5)
        r = v.verify_grasp_lift(final_gripper=9.5, final_wrist=89.0)
        assert r.success

    def test_verify_holding(self) -> None:
        v = RealVerifier(empty_grip_baseline=1.5)
        assert v.verify_holding(9.5).success
        assert v.verify_holding(2.0).failure_code == FAIL_NOT_HOLDING

    def test_classify_failure_code(self) -> None:
        assert classify_failure_code(FAIL_GRASP_EMPTY) == "retry"
        assert classify_failure_code(FAIL_NOT_LIFTED) == "retry"
        assert classify_failure_code(FAIL_NOT_HOLDING) == "retry"
        assert classify_failure_code("comm_error") == "comm"
        assert classify_failure_code("other") == "abort"


class TestRealObservation:
    def test_summarize_state(self) -> None:
        s = {"wrist_flex.pos": 89.0, "gripper.pos": 1.5}
        txt = summarize_state(s)
        assert "抬起" in txt and "闭合" in txt

    def test_helpers(self) -> None:
        assert is_holding(9.5, 1.5, margin=3.0)
        assert not is_holding(2.0, 1.5)
        assert is_lifted(89.0)
        assert not is_lifted(50.0)


class TestRealVerifierLoadSignal:
    """0.6 之后验证器接入负载信号：「位置 OR 负载」。"""

    def test_defaults_come_from_calibration(self) -> None:
        v = RealVerifier()
        assert v.empty_grip_baseline == pytest.approx(EMPTY_GRIP_BASELINE)
        assert v.margin == pytest.approx(GRIP_POS_MARGIN)

    def test_load_rescues_ultra_thin_object(self) -> None:
        """位置只到 4.2（未超阈值 5.5），但负载 −206‰ → 判为夹住。"""
        v = RealVerifier()
        assert v.verify_holding(4.2, load=-206.0).success

    def test_empty_close_with_empty_load_is_not_holding(self) -> None:
        v = RealVerifier()
        assert v.verify_holding(3.9, load=-62.0).failure_code == FAIL_NOT_HOLDING

    def test_grasp_lift_uses_load(self) -> None:
        """咬空时位置与负载都是空载量级 → 抬起也算 FAIL_GRASP_EMPTY。"""
        v = RealVerifier()
        r = v.verify_grasp_lift(final_gripper=4.0, final_wrist=89.0, final_load=-62.0)
        assert not r.success and r.failure_code == FAIL_GRASP_EMPTY

    def test_grasp_lift_success_keyed_on_load(self) -> None:
        v = RealVerifier()
        r = v.verify_grasp_lift(final_gripper=4.2, final_wrist=89.0, final_load=-206.0)
        assert r.success

    def test_load_omitted_falls_back_to_position(self) -> None:
        v = RealVerifier()
        assert v.verify_holding(9.5).success
        assert v.verify_holding(2.0).failure_code == FAIL_NOT_HOLDING
