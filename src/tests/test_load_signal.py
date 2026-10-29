"""real_observation 负载/接触信号测试 — 纯逻辑，无机械臂。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from real_observation import (  # noqa: E402
    EMPTY_GRIP_BASELINE,
    EMPTY_LOAD_BASELINE,
    GRIP_POS_MARGIN,
    LOAD_DIRECTION_BIT,
    LOAD_MAGNITUDE_MASK,
    LOAD_MARGIN,
    STATE_NAMES,
    aggregate_episode_success,
    decode_feetech_load,
    detect_contact,
    holds_object,
    is_holding,
    load_state_names,
    read_load,
    summarize_load_probe,
)


class FakeLoadEngine:
    """按预设返回负载的假引擎。"""

    def __init__(self, loads):
        self.loads = loads
        self.calls = 0

    def read_load(self):
        self.calls += 1
        return self.loads


class ErrorLoadEngine:
    def read_load(self):
        raise OSError("串口读取失败")


class NoLoadEngine:
    """未实现 read_load 的引擎（旧实现）。"""


class BadTypeLoadEngine:
    def read_load(self):
        return [1.0, 2.0]


class TestDecodeFeetechLoad:
    """原始 2 字节读数含方向位（bit10），必须先解码再比较。"""

    def test_positive_direction_bit_clear(self) -> None:
        assert decode_feetech_load(55) == pytest.approx(55.0)

    def test_negative_direction_bit_set(self) -> None:
        """1079 = 0x437：方向位=1、幅值=55 → −55（实测空载闭合的真实读数）。"""
        assert decode_feetech_load(1079) == pytest.approx(-55.0)

    def test_zero_load(self) -> None:
        assert decode_feetech_load(0) == pytest.approx(0.0)

    def test_max_magnitude_positive(self) -> None:
        assert decode_feetech_load(LOAD_MAGNITUDE_MASK) == pytest.approx(1023.0)

    def test_max_magnitude_negative(self) -> None:
        assert decode_feetech_load(LOAD_DIRECTION_BIT | LOAD_MAGNITUDE_MASK) == pytest.approx(-1023.0)

    def test_direction_flip_would_fake_huge_delta(self) -> None:
        """核心动机：解码后 |Δ| 很小，但原始值直接做差会虚增 1024。"""
        raw_positive, raw_negative = 55, 1079
        decoded_delta = abs(decode_feetech_load(raw_negative) - decode_feetech_load(raw_positive))
        raw_delta = abs(raw_negative - raw_positive)
        assert decoded_delta == pytest.approx(110.0)
        assert raw_delta == pytest.approx(1024.0)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            decode_feetech_load((LOAD_DIRECTION_BIT | LOAD_MAGNITUDE_MASK) + 1)

    def test_negative_raw_raises(self) -> None:
        with pytest.raises(ValueError):
            decode_feetech_load(-1)

    def test_non_finite_raises(self) -> None:
        with pytest.raises(ValueError):
            decode_feetech_load(float("nan"))


class TestDetectContact:
    def test_delta_above_margin_is_contact(self) -> None:
        signal = detect_contact(load=12.0, baseline=6.0, margin=3.0)
        assert signal.is_contact
        assert signal.delta == pytest.approx(6.0)
        assert signal.source == "present_load"

    def test_delta_below_margin_is_not_contact(self) -> None:
        assert not detect_contact(load=7.0, baseline=6.0, margin=3.0).is_contact

    def test_exactly_at_margin_is_not_contact(self) -> None:
        """判据是严格大于，边界值不算接触。"""
        assert not detect_contact(load=9.0, baseline=6.0, margin=3.0).is_contact

    def test_zero_margin_allows_any_increase(self) -> None:
        assert detect_contact(load=6.001, baseline=6.0, margin=0.0).is_contact

    def test_default_margin_is_used(self) -> None:
        signal = detect_contact(load=6.0 + LOAD_MARGIN, baseline=6.0)
        assert signal.margin == pytest.approx(LOAD_MARGIN)

    def test_negative_margin_raises(self) -> None:
        with pytest.raises(ValueError):
            detect_contact(load=1.0, baseline=0.0, margin=-1.0)

    def test_non_finite_load_raises(self) -> None:
        with pytest.raises(ValueError):
            detect_contact(load=float("nan"), baseline=0.0)
        with pytest.raises(ValueError):
            detect_contact(load=1.0, baseline=float("inf"))


class TestReadLoad:
    def test_reads_values_from_engine(self) -> None:
        engine = FakeLoadEngine({"gripper.load": 12, "wrist_roll.load": 3})
        assert read_load(engine) == {"gripper.load": 12.0, "wrist_roll.load": 3.0}

    def test_engine_without_read_load_returns_empty(self) -> None:
        """旧引擎没有负载能力 → 空 dict，而不是报错。"""
        assert read_load(NoLoadEngine()) == {}

    def test_engine_error_is_swallowed(self) -> None:
        assert read_load(ErrorLoadEngine()) == {}

    def test_non_dict_return_returns_empty(self) -> None:
        assert read_load(BadTypeLoadEngine()) == {}

    def test_non_numeric_entries_are_skipped(self) -> None:
        engine = FakeLoadEngine({"gripper.load": "bad", "wrist_roll.load": 5})
        assert read_load(engine) == {"wrist_roll.load": 5.0}

    def test_load_names_match_state_names(self) -> None:
        assert load_state_names() == [
            name.rsplit(".", 1)[0] + ".load" for name in STATE_NAMES
        ]
        assert len(load_state_names()) == len(STATE_NAMES)


class TestSummarizeLoadProbe:
    def test_clear_separation_is_resolvable(self) -> None:
        """取值按解码后量级（千分比）：空载 ~55‰、夹持 ~180‰。"""
        result = summarize_load_probe([55.0, 55.1, 54.9], [180.0, 180.2, 179.8])
        assert result.resolvable
        assert result.delta == pytest.approx(125.0, abs=0.5)
        assert result.n_empty == 3 and result.n_loaded == 3

    def test_small_delta_is_not_resolvable(self) -> None:
        result = summarize_load_probe([55.0, 55.1], [60.0, 60.1])
        assert not result.resolvable

    def test_noise_raises_effective_threshold(self) -> None:
        """抖动大时，即使 Δ 超过 margin 也判为不可分辨。"""
        result = summarize_load_probe([5.0, 15.0], [12.0, 20.0], margin=3.0)
        assert result.noise > result.margin
        assert result.threshold == pytest.approx(2.0 * result.noise)
        assert not result.resolvable

    def test_threshold_falls_back_to_margin_when_noise_small(self) -> None:
        result = summarize_load_probe([5.0, 5.0], [20.0, 20.0], margin=3.0)
        assert result.noise == pytest.approx(0.0)
        assert result.threshold == pytest.approx(3.0)

    def test_single_sample_per_phase_is_allowed(self) -> None:
        result = summarize_load_probe([5.0], [20.0], margin=3.0)
        assert result.noise == pytest.approx(0.0)
        assert result.resolvable

    def test_empty_samples_raise(self) -> None:
        with pytest.raises(ValueError):
            summarize_load_probe([], [1.0])
        with pytest.raises(ValueError):
            summarize_load_probe([1.0], [])

    def test_non_finite_samples_raise(self) -> None:
        with pytest.raises(ValueError):
            summarize_load_probe([5.0], [float("nan")])

    def test_note_contains_measurements(self) -> None:
        result = summarize_load_probe([5.0], [20.0])
        assert "空载" in result.note and "夹持" in result.note


class TestCalibratedHoldingJudgement:
    """0.6 真机标定后的默认值（baseline 4.1 / margin 1.4）。"""

    def test_defaults_match_calibration(self) -> None:
        assert EMPTY_GRIP_BASELINE == pytest.approx(4.1)
        assert GRIP_POS_MARGIN == pytest.approx(1.4)
        assert EMPTY_LOAD_BASELINE < 0

    def test_empty_close_position_is_not_holding(self) -> None:
        """实测空载闭合 3.92、上限 4.045；阈值 5.5 → 必须判未夹持。"""
        assert not is_holding(4.045)
        assert not is_holding(3.92)

    def test_thin_object_close_position_is_holding(self) -> None:
        """实测薄件（螺丝刀）夹持位 6.8 > 阈值 5.5 → 位置判据可检出。"""
        assert is_holding(6.8)

    def test_wide_open_is_not_holding(self) -> None:
        """张开（实测 16.9–22.7）不能被判成夹住。

        2026-10-18 第 3 次真机闭环：`grip 19.25 + 抬起` 被旧判据判成功，
        人工确认"张开、没夹住" → 必须补上界。
        """
        assert not is_holding(19.25)
        assert not is_holding(16.9)
        assert not is_holding(22.7)

    def test_upper_bound_still_admits_thick_object(self) -> None:
        """15.0 上界要留给厚件余量（实测夹住带 6.7–9.8）。"""
        assert is_holding(13.0)
        assert not is_holding(15.1)

    def test_net_clearance_old_vs_new(self) -> None:
        """旧配置(1.5+3.0)名义 margin=3.0，对空载上限实际只有 0.455 净空；
        新配置留 1.455。这正是 0.6 要修的问题。"""
        assert EMPTY_GRIP_BASELINE + GRIP_POS_MARGIN - 4.045 == pytest.approx(1.455)
        assert 1.5 + 3.0 - 4.045 == pytest.approx(0.455)


class TestDetectContactMagnitude:
    """负载判据用增量**幅值**：空载顶住限位与夹住物体是同向扭矩，只有幅值不同。"""

    def test_signed_comparison_would_miss(self) -> None:
        """对照：按旧带符号判据 ``load > baseline + margin``，实测值 −206 > −69+50 为假
        → 必然漏判；改为幅值后成立。"""
        assert not (-206.0 > EMPTY_LOAD_BASELINE + LOAD_MARGIN)
        assert detect_contact(-206.0, EMPTY_LOAD_BASELINE, LOAD_MARGIN).is_contact

    def test_empty_load_is_not_contact(self) -> None:
        assert not detect_contact(-62.0, EMPTY_LOAD_BASELINE, LOAD_MARGIN).is_contact
        assert not detect_contact(-69.0, EMPTY_LOAD_BASELINE, LOAD_MARGIN).is_contact

    def test_direction_agnostic(self) -> None:
        """反向扭矩同样算接触：接触会改变扭矩，方向不该成为漏判理由。"""
        assert detect_contact(20.0, 0.0, 5.0).is_contact
        assert detect_contact(-20.0, 0.0, 5.0).is_contact

    def test_delta_stays_signed_for_diagnostics(self) -> None:
        assert detect_contact(-206.0, EMPTY_LOAD_BASELINE, LOAD_MARGIN).delta == pytest.approx(-137.0)


class TestHoldsObject:
    """「位置 OR 负载」判据。"""

    def test_position_alone_is_enough(self) -> None:
        assert holds_object(6.8, load=None) is True

    def test_load_alone_is_enough(self) -> None:
        """极薄件位置没超阈值，但负载够了 → 仍判夹持。"""
        assert holds_object(4.2, load=-206.0) is True

    def test_neither_means_not_holding(self) -> None:
        assert holds_object(3.9, load=-62.0) is False

    def test_missing_load_degrades_to_position_only(self) -> None:
        """引擎无读负载能力时，不能因为"没读到"就判成夹住。"""
        assert holds_object(3.9, load=None) is False

    def test_idle_open_gripper_load_is_not_contact(self) -> None:
        """回归：夹爪**张开/空转**时 load≈0‰。

        若用"相对空载基准做差"（基准 −69‰），``|0 − (−69)| = 69 > 50`` → **误判成夹持**
        （week38 真机冒烟测试抓到的假阳性）。改用幅值判据后必须判 False。
        """
        assert abs(0.0 - EMPTY_LOAD_BASELINE) > LOAD_MARGIN  # 旧差值口径确实会误判
        assert holds_object(1.39, load=0.0) is False

    def test_load_criterion_is_conservative(self) -> None:
        """阈值 150‰ 偏保守：薄件实测 −100‰ 不触发（由位置判据兜），只有明确受力才触发。"""
        assert holds_object(3.9, load=-100.0) is False
        assert holds_object(3.9, load=-206.0) is True

    def test_thresholds_are_overridable(self) -> None:
        assert holds_object(5.0, None, empty_grip_baseline=4.1, pos_margin=0.5) is True
        assert holds_object(4.4, None, empty_grip_baseline=4.1, pos_margin=0.5) is False


def _vector(grip: float, wrist: float) -> list[float]:
    """按 STATE_NAMES 顺序构造 6 维关节向量。"""
    frame = [0.0] * len(STATE_NAMES)
    frame[STATE_NAMES.index("gripper.pos")] = grip
    frame[STATE_NAMES.index("wrist_flex.pos")] = wrist
    return frame


class TestAggregateEpisodeSuccess:
    """逐帧判据 → 逐集结论（A4 的对照缺口）。"""

    def test_any_frame_with_hold_and_lift_is_success(self) -> None:
        states = [_vector(2.0, 72.0), _vector(7.0, 78.0), _vector(19.0, 80.0)]
        outcome = aggregate_episode_success(states)
        assert outcome.success is True
        assert outcome.holding_and_lifted_frames == 1
        assert outcome.to_dict()["success"] is True

    def test_holding_without_lift_is_not_success(self) -> None:
        outcome = aggregate_episode_success([_vector(7.0, 60.0), _vector(7.0, 70.0)])
        assert outcome.success is False
        assert outcome.held_frames == 2
        assert outcome.lifted_frames == 0

    def test_lifted_without_holding_is_not_success(self) -> None:
        """张开 19.25 + 抬起不再算成功（2026-10-18 假阳性）。"""
        outcome = aggregate_episode_success([_vector(19.25, 76.0)])
        assert outcome.success is False
        assert outcome.lifted_frames == 1

    def test_last_frame_not_required(self) -> None:
        """成功后松手/复位不影响整集判定。"""
        states = [_vector(7.0, 78.0), _vector(1.6, 72.0)]
        assert aggregate_episode_success(states).success is True

    def test_load_can_rescue_thin_grasp(self) -> None:
        states = [_vector(3.9, 78.0)]
        assert aggregate_episode_success(states, loads=[-206.0]).success is True
        assert aggregate_episode_success(states, loads=[-206.0], use_load=False).success is False

    def test_accepts_named_dicts(self) -> None:
        states = [{"gripper.pos": 7.0, "wrist_flex.pos": 80.0}]
        assert aggregate_episode_success(states).success is True

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            aggregate_episode_success([])
        with pytest.raises(ValueError):
            aggregate_episode_success([_vector(7.0, 80.0)], loads=[-100.0, -200.0])
        with pytest.raises(ValueError):
            aggregate_episode_success([[1.0, 2.0, 3.0]])
