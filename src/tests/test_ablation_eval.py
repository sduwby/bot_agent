"""P0 感知消融**评测链路**测试 — 纯逻辑，不依赖 torch / lerobot / pyarrow / 仿真。

最要紧的一条：`make_oracle_provider` 的输出**列序**必须等于
`real_geometry.variant_feature_names(variant)` —— 列序一错，策略会收到错位向量，
指标"只是变差"而不报错，消融结论随之报废。故这里既核对列序，又核对与训练侧
`ablation_dataset.frame_oracle_inputs` 的数值一致性。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ablation_dataset import frame_oracle_inputs  # noqa: E402
from ablation_eval import (  # noqa: E402
    AblationEvalError,
    build_variant_result,
    evaluation_episodes,
    make_oracle_provider,
    sim_oracle_inputs,
    success_rate,
)
from ablation_features import oracle_features  # noqa: E402
from depth_ablation import AblationError, VariantResult  # noqa: E402
from real_geometry import GeometryError, variant_feature_names  # noqa: E402
from run_ablation_eval import main, render_eval_plan  # noqa: E402

OPEN_WIDTH = 0.08
CLOSED_WIDTH = 0.0


def _observation(
    *,
    tcp: tuple[float, ...] = (0.10, 0.00, 0.40, 0.0, 0.0, 0.0, 1.0),
    obj: tuple[float, ...] = (0.10, 0.00, 0.45, 0.0, 0.0, 0.0, 1.0),
    is_grasped: float = 0.0,
    qpos: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04),
) -> dict[str, object]:
    """一份最小仿真观测（tcp/obj 取 7 维 pose，qpos 末两维是夹爪两关节）。"""
    return {
        "extra": {
            "tcp_pose": list(tcp),
            "obj_pose": list(obj),
            "is_grasped": is_grasped,
        },
        "agent": {"qpos": list(qpos)},
    }


class TestSimOracleInputs:
    def test_parses_fields_and_sums_gripper(self) -> None:
        inputs = sim_oracle_inputs(_observation(), open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH)
        assert list(inputs.tcp_pos_m) == pytest.approx([0.10, 0.00, 0.40])
        assert list(inputs.obj_pos_m) == pytest.approx([0.10, 0.00, 0.45])
        assert inputs.gripper_width == pytest.approx(0.08)  # qpos 末两维之和
        assert inputs.is_grasped is False

    def test_is_grasped_uses_strict_threshold(self) -> None:
        """阈值是 `> 0.5`：恰好 0.5 判为未夹住，0.6 判为夹住（与训练侧同口径）。"""
        at_threshold = sim_oracle_inputs(
            _observation(is_grasped=0.5), open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH
        )
        above = sim_oracle_inputs(
            _observation(is_grasped=0.6), open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH
        )
        assert at_threshold.is_grasped is False
        assert above.is_grasped is True

    def test_missing_extra_rejected(self) -> None:
        observation = _observation()
        del observation["extra"]
        with pytest.raises(AblationEvalError, match="extra"):
            sim_oracle_inputs(observation, open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH)

    def test_missing_agent_rejected(self) -> None:
        observation = _observation()
        del observation["agent"]
        with pytest.raises(AblationEvalError, match="agent"):
            sim_oracle_inputs(observation, open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH)

    def test_missing_tcp_field_rejected(self) -> None:
        observation = _observation()
        del observation["extra"]["tcp_pose"]  # type: ignore[index]
        with pytest.raises(AblationEvalError, match="tcp_pose"):
            sim_oracle_inputs(observation, open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH)

    def test_short_tcp_rejected(self) -> None:
        observation = _observation(tcp=(0.1, 0.0))
        with pytest.raises(AblationEvalError, match="tcp_pose"):
            sim_oracle_inputs(observation, open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH)

    def test_short_qpos_rejected(self) -> None:
        observation = _observation(qpos=(0.04,))
        with pytest.raises(AblationEvalError, match="qpos"):
            sim_oracle_inputs(observation, open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH)

    def test_non_scalar_is_grasped_rejected(self) -> None:
        observation = _observation()
        observation["extra"]["is_grasped"] = [0.0, 1.0]  # type: ignore[index]
        with pytest.raises(AblationEvalError, match="标量"):
            sim_oracle_inputs(observation, open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH)


class TestOracleProvider:
    def test_output_order_matches_variant_feature_names(self) -> None:
        """**最重要的一条**：列序必须等于 `variant_feature_names`。"""
        for variant in ("B1", "B2"):
            provider = make_oracle_provider(
                variant, open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH
            )
            values = oracle_features(
                variant,
                sim_oracle_inputs(
                    _observation(), open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH
                ),
            )
            expected = [float(values[name]) for name in variant_feature_names(variant)]
            assert provider(_observation()) == pytest.approx(expected)
            assert len(expected) == len(variant_feature_names(variant))

    def test_b1_columns(self) -> None:
        provider = make_oracle_provider(
            "B1", open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH
        )
        assert variant_feature_names("B1") == (
            "depth_known_size_m",
            "depth_known_size_sigma_m",
        )
        out = provider(_observation())
        assert out == pytest.approx([0.05, 0.0])  # ‖tcp−obj‖ = 0.05，特权上界 sigma=0

    def test_b0_has_no_extra_columns(self) -> None:
        provider = make_oracle_provider(
            "B0", open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH
        )
        assert variant_feature_names("B0") == ()
        assert provider(_observation()) == []

    def test_unknown_variant_rejected(self) -> None:
        with pytest.raises(GeometryError):
            make_oracle_provider("B9", open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH)

    def test_b3_rejected_needs_detector(self) -> None:
        """B3 的立体深度列 oracle 路径产不出 —— 必须显式失败而不是静默缺列。"""
        provider = make_oracle_provider(
            "B3", open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH
        )
        with pytest.raises(AblationEvalError, match="B3"):
            provider(_observation())

    def test_matches_training_side_frame_oracle_inputs(self) -> None:
        """训练侧从 (state, env) 还原，评测侧从仿真观测还原 —— 两者必须数值一致。"""
        tcp = (0.10, 0.00, 0.44)
        obj = (0.10, 0.00, 0.45)
        # 训练侧：state = tcp_pos(3) + gripper_qpos(2)；env = 13 维，env[:3]=obj、env[12]=is_grasped
        state = [*tcp, 0.02, 0.02]
        env = [0.0] * 13
        env[0:3] = list(obj)
        env[12] = 1.0
        training = frame_oracle_inputs(
            state, env, open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH
        )
        # 评测侧：同一批量，从仿真观测字段给出（qpos 末两维 = 夹爪两关节）
        evaluation_observation = _observation(
            tcp=(*tcp, 0.0, 0.0, 0.0, 1.0),
            obj=(*obj, 0.0, 0.0, 0.0, 1.0),
            is_grasped=1.0,
            qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02),
        )
        evaluation = sim_oracle_inputs(
            evaluation_observation, open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH
        )
        assert list(evaluation.tcp_pos_m) == pytest.approx(list(training.tcp_pos_m))
        assert list(evaluation.obj_pos_m) == pytest.approx(list(training.obj_pos_m))
        assert evaluation.gripper_width == pytest.approx(training.gripper_width)
        assert evaluation.is_grasped is training.is_grasped is True
        # 同序 provider 的输出也一致（评测侧 provider == 训练侧 oracle_features 定序结果）
        provider = make_oracle_provider("B2", open_width=OPEN_WIDTH, closed_width=CLOSED_WIDTH)
        values = oracle_features("B2", training)
        assert provider(evaluation_observation) == pytest.approx(
            [float(values[name]) for name in variant_feature_names("B2")]
        )


@dataclass(frozen=True)
class _Episode:
    """最小 episode 替身：只需 `.success` 与 `.nonfinite_actions`。"""

    success: bool
    nonfinite_actions: int = 0


class TestSuccessRate:
    def test_all_success(self) -> None:
        assert success_rate([_Episode(True), _Episode(True)]) == pytest.approx(1.0)

    def test_all_failure(self) -> None:
        assert success_rate([_Episode(False), _Episode(False)]) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        assert success_rate([_Episode(True), _Episode(False), _Episode(False)]) == pytest.approx(
            1.0 / 3.0
        )

    def test_empty_rejected(self) -> None:
        with pytest.raises(AblationEvalError, match="为空"):
            success_rate([])


class TestBuildVariantResult:
    def test_fields(self) -> None:
        result = build_variant_result("B1", [_Episode(True), _Episode(False)])
        assert isinstance(result, VariantResult)
        assert result.variant == "B1"
        assert result.n_episodes == 2
        assert result.success_rate == pytest.approx(0.5)
        assert result.closing_height_error_mm is None
        assert result.nonfinite_actions == 0

    def test_nonfinite_actions_summed(self) -> None:
        result = build_variant_result(
            "B2", [_Episode(True, 1), _Episode(False, 2), _Episode(True, 0)]
        )
        assert result.nonfinite_actions == 3

    def test_train_seconds_passthrough(self) -> None:
        result = build_variant_result("B0", [_Episode(True)], train_seconds=12.5)
        assert result.train_seconds == pytest.approx(12.5)

    def test_unknown_variant_rejected(self) -> None:
        with pytest.raises(AblationError):
            build_variant_result("B9", [_Episode(True)])


class TestEvaluationEpisodes:
    class _Summary:
        def __init__(self, episodes: tuple[_Episode, ...]) -> None:
            self.episodes = episodes

    def test_extracts_episodes(self) -> None:
        summary = self._Summary((_Episode(True), _Episode(False)))
        assert evaluation_episodes(summary) == (_Episode(True), _Episode(False))

    def test_missing_field_rejected(self) -> None:
        with pytest.raises(AblationEvalError, match="episodes"):
            evaluation_episodes(object())


class TestCli:
    def test_render_eval_plan_shows_provider_and_columns(self) -> None:
        text = render_eval_plan(
            ["B0", "B1", "B2"],
            ["PickCube-v1"],
            [0, 1],
            root=Path("runs/week38/ablation_smoke"),
            dataset_root=Path("runs/week38/ablation_data"),
            checkpoint_subdir="best-validation",
        )
        assert "depth_known_size_m" in text
        assert "contact_flag" in text
        assert "None（无额外通道）" in text

    def test_dry_run_exits_zero_without_policy(self) -> None:
        """`--dry-run` 必须在无 GPU/无 ManiSkill/无 pyarrow 时也跑通（此处即本机）。"""
        assert main(["--dry-run"]) == 0

    def test_dry_run_accepts_custom_subset(self) -> None:
        assert main(["--dry-run", "--variants", "B0,B1", "--tasks", "PickCube-v1",
                     "--seeds", "0,1,2"]) == 0

    def test_dry_run_rejects_missing_baseline(self) -> None:
        with pytest.raises(SystemExit):
            main(["--dry-run", "--variants", "B1,B2"])

    def test_dry_run_rejects_duplicate_seeds(self) -> None:
        with pytest.raises(SystemExit):
            main(["--dry-run", "--seeds", "0,0"])
