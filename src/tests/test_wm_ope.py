"""`wm_ope` 的纯逻辑测试（不依赖 ManiSkill / PyTorch）。"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from behavior_cloning import STATE_DIM, observation_to_state  # noqa: E402
from run_wm_ope import (  # noqa: E402
    LaggedScriptedPolicy,
    NoisyScriptedPolicy,
    ReversePolicy,
    ZeroPolicy,
    build_candidate,
    parse_seeds,
    select_candidates,
)
from wm_ope import (  # noqa: E402
    ACTION_DIM,
    MIN_CANDIDATES,
    MIN_ROLLOUTS_PER_CANDIDATE,
    PUSHCUBE_SUCCESS_DIST,
    RHO_USABLE,
    WMOPEError,
    attribute_low_rho,
    distribution_shift,
    evaluate_ope,
    format_markdown,
    rollout_policy_in_world_model,
    rollout_with_trajectory,
    spearman_rho,
    state_to_observation,
    write_ope_report,
)


def _rate_sequence(rate: float, n: int = MIN_ROLLOUTS_PER_CANDIDATE) -> list[float]:
    """构造均值恰为 rate 的 0/1 序列。"""
    ones = round(rate * n)
    return [1.0] * ones + [0.0] * (n - ones)


def _candidates(rates: list[float]) -> dict[str, list[float]]:
    return {f"c{index}": _rate_sequence(rate) for index, rate in enumerate(rates)}


def _families(names: list[str]) -> dict[str, str]:
    """交替给两个行为族，满足"跨族"要求。"""
    return {name: ("scripted" if index % 2 == 0 else "random") for index, name in enumerate(names)}


class TestSpearmanRho:
    def test_perfect_agreement_is_one(self) -> None:
        assert spearman_rho([0.1, 0.2, 0.3, 0.4, 0.5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)

    def test_perfect_reversal_is_minus_one(self) -> None:
        assert spearman_rho([0.5, 0.4, 0.3, 0.2, 0.1], [1, 2, 3, 4, 5]) == pytest.approx(-1.0)

    def test_monotonic_nonlinear_still_one(self) -> None:
        """秩相关只看序，非线性单调变换不该影响。"""
        assert spearman_rho([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 4.0, 9.0, 16.0, 25.0]) == pytest.approx(1.0)

    def test_ties_use_average_ranks(self) -> None:
        # left 的中间三项并列 → 秩 1,3,3,3,5；right → 1,2,3,4,5
        # 手算：8 / (sqrt(8)·sqrt(10)) = 0.894427…
        rho = spearman_rho([1.0, 2.0, 2.0, 2.0, 3.0], [1, 2, 3, 4, 5])
        assert rho == pytest.approx(0.8944271909999159, rel=1e-9)

    def test_constant_input_is_undefined(self) -> None:
        with pytest.raises(WMOPEError, match="全相同"):
            spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="长度不一致"):
            spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_too_few_points_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="至少需要 3 个候选"):
            spearman_rho([1.0, 2.0], [1.0, 2.0])


class TestEvaluateOpeValidation:
    def test_too_few_candidates_rejected(self) -> None:
        names = [f"c{i}" for i in range(MIN_CANDIDATES - 1)]
        with pytest.raises(WMOPEError, match="候选数"):
            evaluate_ope(_candidates([0.1, 0.2, 0.3, 0.4]), _families(names), [0.1, 0.2, 0.3, 0.4])

    def test_insufficient_rollouts_rejected(self) -> None:
        candidates = {f"c{i}": [1.0, 0.0] for i in range(MIN_CANDIDATES)}
        names = list(candidates)
        with pytest.raises(WMOPEError, match="每候选至少"):
            evaluate_ope(candidates, _families(names), [0.5] * MIN_CANDIDATES)

    def test_uneven_rollout_counts_rejected(self) -> None:
        candidates = {f"c{i}": _rate_sequence(0.5) for i in range(MIN_CANDIDATES - 1)}
        candidates["c-long"] = _rate_sequence(0.5, MIN_ROLLOUTS_PER_CANDIDATE + 2)
        names = list(candidates)
        with pytest.raises(WMOPEError, match="rollout 数必须一致"):
            evaluate_ope(candidates, _families(names), [0.5] * MIN_CANDIDATES)

    def test_single_family_warns(self) -> None:
        names = [f"c{i}" for i in range(MIN_CANDIDATES)]
        report = evaluate_ope(
            _candidates([0.1, 0.2, 0.3, 0.4, 0.5]),
            {name: "scripted" for name in names},
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )
        assert any("行为族" in item for item in report.warnings)

    def test_real_success_out_of_range_rejected(self) -> None:
        names = [f"c{i}" for i in range(MIN_CANDIDATES)]
        with pytest.raises(WMOPEError, match="\\[0,1\\]"):
            evaluate_ope(_candidates([0.1] * MIN_CANDIDATES), _families(names), [1.5] * MIN_CANDIDATES)

    def test_length_mismatch_rejected(self) -> None:
        names = [f"c{i}" for i in range(MIN_CANDIDATES)]
        with pytest.raises(WMOPEError, match="长度"):
            evaluate_ope(_candidates([0.1] * MIN_CANDIDATES), _families(names), [0.1] * 3)


class TestEvaluateOpeResult:
    def test_perfect_ranking_is_usable(self) -> None:
        rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        names = [f"c{i}" for i in range(MIN_CANDIDATES)]
        report = evaluate_ope(_candidates(rates), _families(names), rates)
        assert report.spearman_rho == pytest.approx(1.0)
        assert report.verdict == "usable"
        assert report.ci_low <= 1.0 <= report.ci_high
        assert report.n_rollouts_per_candidate == MIN_ROLLOUTS_PER_CANDIDATE

    def test_reversed_ranking_is_negative_result(self) -> None:
        rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        names = [f"c{i}" for i in range(MIN_CANDIDATES)]
        report = evaluate_ope(_candidates(rates[::-1]), _families(names), rates)
        assert report.spearman_rho == pytest.approx(-1.0)
        assert report.verdict == "negative_result"
        assert any("负结果" in item for item in report.warnings)

    def test_middle_rho_is_marginal(self) -> None:
        # wm 的秩为 1,4,2,5,3 → 手算 ρ = 5/10 = 0.5，落在 [RHO_LOW, RHO_USABLE)
        real = [0.1, 0.2, 0.3, 0.4, 0.5]
        wm = [0.1, 0.4, 0.2, 0.5, 0.3]
        names = [f"c{i}" for i in range(MIN_CANDIDATES)]
        report = evaluate_ope(_candidates(wm), _families(names), real)
        assert report.spearman_rho == pytest.approx(0.5)
        assert report.verdict == "marginal"

    def test_variance_is_bernoulli(self) -> None:
        rates = [0.0, 0.2, 0.5, 0.8, 1.0]
        names = [f"c{i}" for i in range(MIN_CANDIDATES)]
        report = evaluate_ope(_candidates(rates), _families(names), rates)
        for index, rate in enumerate(rates):
            assert report.real_success_var[index] == pytest.approx(rate * (1.0 - rate))

    def test_warnings_and_attribution_are_serializable(self) -> None:
        rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        names = [f"c{i}" for i in range(MIN_CANDIDATES)]
        shift = distribution_shift(_FakeWM(), np.zeros((4, STATE_DIM)), candidate="c0")
        report = evaluate_ope(
            _candidates(rates), {name: "scripted" for name in names}, rates,
            attribution=[shift],
        )
        payload = json.loads(json.dumps(report.to_json(), ensure_ascii=False))
        assert payload["candidates"] == names
        assert payload["attribution"][0]["candidate"] == "c0"
        assert payload["scope_note"]


class _FakeWM:
    """最小动力学：state_mean=0 / state_std=1，predict_next = state + delta。"""

    state_mean = np.zeros(STATE_DIM)
    state_std = np.ones(STATE_DIM)

    def __init__(self, delta: np.ndarray | None = None) -> None:
        self.delta = np.zeros(STATE_DIM) if delta is None else np.asarray(delta, dtype=float)
        self.calls = 0

    def predict_next(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.asarray(states, dtype=float) + self.delta


class _FakePolicy:
    def __init__(self, action: list[float]) -> None:
        self.action = np.asarray(action, dtype=float)
        self.resets = 0
        self.acts = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, observation: dict) -> np.ndarray:
        assert "extra" in observation and "tcp_pose" in observation["extra"]
        self.acts += 1
        return self.action


def _push_state(obj: list[float], goal: list[float], grasped: float = 0.0) -> np.ndarray:
    """按 18 维切片约定拼一个状态。"""
    tcp = np.array([0.0, 0.0, 0.1])
    obj_array = np.asarray(obj, dtype=float)
    goal_array = np.asarray(goal, dtype=float)
    return np.concatenate(
        (tcp, obj_array, goal_array, tcp - obj_array, obj_array - goal_array,
         np.zeros(2), [grasped])
    ).astype(np.float32)


class TestStateToObservation:
    def test_round_trip_is_exact(self) -> None:
        """强校验：state → 合成 observation → observation_to_state 必须完全一致。"""
        state = _push_state([0.1, 0.2, 0.02], [0.3, 0.1, 0.02], grasped=1.0)
        assert np.allclose(observation_to_state(state_to_observation(state)), state)

    def test_grasped_flag_is_passed_through(self) -> None:
        assert state_to_observation(_push_state([0, 0, 0], [1, 1, 0], grasped=1.0))["extra"]["is_grasped"] is True
        assert state_to_observation(_push_state([0, 0, 0], [1, 1, 0], grasped=0.0))["extra"]["is_grasped"] is False

    def test_bad_shape_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="shape"):
            state_to_observation([1.0, 2.0])


class TestRolloutInWorldModel:
    def test_success_when_object_reaches_goal(self) -> None:
        # obj 沿 x 每步朝 goal 移 0.01 → 第 4 步距离 0.01 < 0.025
        delta = np.zeros(STATE_DIM)
        delta[3] = 0.01
        wm = _FakeWM(delta)
        policy = _FakePolicy([0.0, 0.0, 0.0, -1.0])
        starts = [_push_state([0.0, 0.0, 0.0], [0.05, 0.0, 0.0])]
        results = rollout_policy_in_world_model(policy, wm, starts, 10)
        assert results == [1.0]

    def test_failure_when_horizon_too_short(self) -> None:
        delta = np.zeros(STATE_DIM)
        delta[3] = 0.01
        wm = _FakeWM(delta)
        policy = _FakePolicy([0.0, 0.0, 0.0, -1.0])
        starts = [_push_state([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])]
        assert rollout_policy_in_world_model(policy, wm, starts, 5) == [0.0]

    def test_success_criterion_is_xy_only(self) -> None:
        """回归：物体在桌面 z≈0.02、目标标记贴地 z≈0.001，z 恒有 ~0.019 固定偏移。

        用 3D 距离配 0.025 阈值时等效要求 xy < ~0.016，几乎不可能达标——第 38 周实测
        曾因此使 WM 侧成功率恒为 0、ρ 无定义。判据必须只看 xy（与 scripted/ManiSkill 同口径）。
        """
        obj, goal = [0.0, 0.0, 0.02], [0.02, 0.0, 0.001]
        xy = float(np.linalg.norm(np.asarray(obj[:2]) - np.asarray(goal[:2])))
        xyz = float(np.linalg.norm(np.asarray(obj) - np.asarray(goal)))
        assert xy < PUSHCUBE_SUCCESS_DIST < xyz  # 恰好卡在两种口径的分界上
        results = rollout_policy_in_world_model(
            _FakePolicy([0.0, 0.0, 0.0, -1.0]), _FakeWM(), [_push_state(obj, goal)], 1
        )
        assert results == [1.0]

    def test_reset_called_once_per_rollout(self) -> None:
        wm = _FakeWM()
        policy = _FakePolicy([0.0, 0.0, 0.0, -1.0])
        starts = [_push_state([0, 0, 0], [1, 1, 0]) for _ in range(4)]
        rollout_policy_in_world_model(policy, wm, starts, 3)
        assert policy.resets == 4

    def test_trajectory_records_visited_states(self) -> None:
        """归因需要候选**实际走过**的状态（不是初始状态）。"""
        wm = _FakeWM()
        policy = _FakePolicy([0.0, 0.0, 0.0, -1.0])
        results, trajectories = rollout_with_trajectory(
            policy, wm, [_push_state([0, 0, 0], [1, 1, 0])], 3
        )
        assert results == [0.0]
        assert len(trajectories) == 1
        assert trajectories[0].shape == (4, STATE_DIM)  # 初始 + 3 步

    def test_trajectory_stops_at_success(self) -> None:
        delta = np.zeros(STATE_DIM)
        delta[3] = 0.01
        results, trajectories = rollout_with_trajectory(
            _FakePolicy([0.0, 0.0, 0.0, -1.0]),
            _FakeWM(delta),
            [_push_state([0.0, 0.0, 0.0], [0.05, 0.0, 0.0])],
            10,
        )
        assert results == [1.0]
        assert trajectories[0].shape[0] < 11  # 达标即提前结束

    def test_bad_horizon_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="horizon"):
            rollout_policy_in_world_model(_FakePolicy([0, 0, 0, -1]), _FakeWM(), [np.zeros(STATE_DIM)], 0)

    def test_empty_starts_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="不能为空"):
            rollout_policy_in_world_model(_FakePolicy([0, 0, 0, -1]), _FakeWM(), [], 5)

    def test_policy_without_act_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="act"):
            rollout_policy_in_world_model(object(), _FakeWM(), [np.zeros(STATE_DIM)], 3)


class TestDistributionShift:
    def test_zero_shift_for_training_like_states(self) -> None:
        shift = distribution_shift(_FakeWM(), np.zeros((10, STATE_DIM)), candidate="c")
        assert shift.mean_abs_z == pytest.approx(0.0)
        assert shift.ood_fraction == pytest.approx(0.0)

    def test_detects_out_of_distribution(self) -> None:
        visited = np.zeros((10, STATE_DIM))
        visited[0, 0] = 10.0        # |z| = 10 > 3
        shift = distribution_shift(_FakeWM(), visited, candidate="c")
        assert shift.ood_fraction == pytest.approx(0.1)
        assert shift.p95_abs_z >= 3.0

    def test_bad_shape_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="shape"):
            distribution_shift(_FakeWM(), np.zeros((3, 5)), candidate="c")

    def test_constant_dims_are_ignored(self) -> None:
        """回归：训练中近似恒定的维度（std≈0 被兜成 1.0）会让 z 分数失去意义、
        把所有样本判成 OOD（实测全候选 OOD=1.0）。统计偏移时必须跳过这些维度。"""
        wm = _FakeWM()
        wm.state_std = np.ones(STATE_DIM)
        wm.state_std[5] = 1e-9          # 造一个"恒定"维度
        visited = np.zeros((4, STATE_DIM))
        visited[:, 5] = 100.0           # 该维度偏离极大，但不该被计入
        shift = distribution_shift(wm, visited, candidate="c")
        assert shift.n_dims == STATE_DIM - 1
        assert shift.mean_abs_z == pytest.approx(0.0)

    def test_all_constant_dims_rejected(self) -> None:
        wm = _FakeWM()
        wm.state_std = np.full(STATE_DIM, 1e-9)
        with pytest.raises(WMOPEError, match="所有维度"):
            distribution_shift(wm, np.zeros((2, STATE_DIM)), candidate="c")

    def test_attribute_low_rho_covers_all_candidates(self) -> None:
        shifts = attribute_low_rho(
            _FakeWM(), {"a": np.zeros((3, STATE_DIM)), "b": np.ones((3, STATE_DIM))}
        )
        assert [item.candidate for item in shifts] == ["a", "b"]


class TestReportOutput:
    def test_markdown_contains_key_figures(self) -> None:
        rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        names = [f"c{i}" for i in range(MIN_CANDIDATES)]
        report = evaluate_ope(_candidates(rates), _families(names), rates)
        text = format_markdown(report)
        assert "Spearman ρ" in text
        assert "口径边界" in text
        assert f"ρ ≥ {RHO_USABLE}" in text

    def test_write_report_creates_both_files(self, tmp_path) -> None:
        rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        names = [f"c{i}" for i in range(MIN_CANDIDATES)]
        report = evaluate_ope(_candidates(rates), _families(names), rates)
        json_path, markdown_path = write_ope_report(report, tmp_path)
        assert json_path.is_file() and markdown_path.is_file()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["spearman_rho"] == pytest.approx(1.0)


class TestParseSeeds:
    def test_range_comma_and_mixed(self) -> None:
        assert parse_seeds("500-503") == [500, 501, 502, 503]
        assert parse_seeds("500,502") == [500, 502]
        assert parse_seeds("500-501,600") == [500, 501, 600]

    def test_reversed_range_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="区间非法"):
            parse_seeds("510-500")

    def test_duplicates_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="重复"):
            parse_seeds("500-501,501")

    def test_empty_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="不能为空"):
            parse_seeds(",")


class TestCandidateRegistry:
    def test_default_registry_meets_design_requirements(self) -> None:
        """设计 §7.5：候选 ≥5 且跨行为族。"""
        specs = select_candidates(None)
        assert len(specs) >= MIN_CANDIDATES
        assert len({spec.family for spec in specs}) >= 2
        assert any(spec.name == "scripted" for spec in specs), "必须含 scripted 锚点"

    def test_unknown_candidate_rejected(self) -> None:
        with pytest.raises(WMOPEError, match="未知候选"):
            select_candidates(["nope"])

    def test_subset_selection_is_faithful(self) -> None:
        names = [spec.name for spec in select_candidates(None)][:3]
        assert [spec.name for spec in select_candidates(names)] == names

    def test_random_and_zero_need_no_mani_skill(self) -> None:
        for spec in select_candidates(["random", "zero"]):
            policy = build_candidate(spec, 0)
            action = policy.act({"agent": {}, "extra": {}})
            assert action.shape == (ACTION_DIM,)


class TestPolicyWrappers:
    def test_lagged_policy_holds_action_for_k_steps(self) -> None:
        base = _CountingPolicy([0.5, 0.0, 0.0, -1.0])
        policy = LaggedScriptedPolicy(base, 3)
        policy.reset()
        outputs = [policy.act({}) for _ in range(7)]
        assert base.calls == 3          # 7 步只查询了 3 次基策略
        for index in range(0, 3):
            assert outputs[index][0] == pytest.approx(0.5)

    def test_lagged_policy_reset_clears_state(self) -> None:
        base = _CountingPolicy([0.5, 0.0, 0.0, -1.0])
        policy = LaggedScriptedPolicy(base, 3)
        policy.act({})
        policy.reset()
        assert base.calls == 0

    def test_noisy_policy_stays_in_bounds(self) -> None:
        base = _CountingPolicy([1.0, 1.0, 1.0, 1.0])
        policy = NoisyScriptedPolicy(base, 5.0, np.random.default_rng(0))
        for _ in range(20):
            assert np.all(policy.act({}) <= 1.0)
            assert np.all(policy.act({}) >= -1.0)

    def test_zero_policy_is_all_zeros(self) -> None:
        assert np.all(ZeroPolicy().act({}) == 0.0)

    def test_reverse_policy_flips_translation_only(self) -> None:
        """反向推：xyz 取负，夹爪保持（与 scripted 语义对照）。"""
        base = _CountingPolicy([0.5, -0.25, 0.75, -1.0])
        policy = ReversePolicy(base)
        action = policy.act({})
        assert action[0] == pytest.approx(-0.5)
        assert action[1] == pytest.approx(0.25)
        assert action[2] == pytest.approx(-0.75)
        assert action[3] == pytest.approx(-1.0)

    def test_reverse_policy_reset_delegates(self) -> None:
        base = _CountingPolicy([0.0, 0.0, 0.0, -1.0])
        policy = ReversePolicy(base)
        policy.act({})
        policy.reset()
        assert base.calls == 0


class TestRegistryGradient:
    """回归：候选必须能产生**真实的成功率梯度**，否则 ρ 无定义。"""

    def test_too_weak_degradations_are_removed(self) -> None:
        # 实测：σ≤0.20 与 lag-3 在仿真里仍是 100% 成功 → 成功率坍塌成 {1.0,0.0}
        names = {spec.name for spec in select_candidates(None)}
        assert "scripted-noise-0.05" not in names
        assert "scripted-noise-0.20" not in names
        assert "scripted-lag-3" not in names

    def test_registry_covers_expected_families_and_anchors(self) -> None:
        specs = select_candidates(None)
        families = {spec.family for spec in specs}
        assert {"scripted", "scripted-degraded", "scripted-adversarial"} <= families
        assert any(spec.name == "scripted" for spec in specs)
        assert any(spec.name == "scripted-reverse" for spec in specs)
        assert len(specs) >= MIN_CANDIDATES


class _CountingPolicy:
    """记录调用次数的假策略。"""

    def __init__(self, action: list[float]) -> None:
        self._action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def act(self, observation: dict) -> np.ndarray:
        self.calls += 1
        return self._action
