"""第 14 周稳定性评测：配置、位姿采样、门禁、失败分类与平滑扫描测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from maniskill_intro import SUPPORTED_TASKS
from act_stability import (
    FailurePhaseSummary,
    StabilityEvalConfig,
    StabilityEvalError,
    StabilityComparisonSummary,
    analyze_failures,
    classify_failure_diagnosis,
    compute_stability_gate,
    default_stability_seeds,
    evaluate_stability,
    load_stability_checkpoints,
    record_initial_poses,
    run_smoothing_sweep,
    summarize_pose_coverage,
)
from lerobot_act import PolicyEvaluationSummary

pytest.importorskip("torch")


def _fake_observation(seed: int = 0) -> dict[str, object]:
    return {
        "agent": {
            "qpos": np.arange(9, dtype=np.float32),
            "qvel": np.arange(9, dtype=np.float32) * 0.1,
        },
        "extra": {
            "tcp_pose": np.arange(7, dtype=np.float32),
            "obj_pose": np.array([seed * 0.01, -0.1, 0.02, 1, 0, 0, 0], dtype=np.float32),
            "goal_pos": np.array([0.2, 0.1, 0.3], dtype=np.float32),
            "is_grasped": np.asarray(False),
        },
    }


class FakeActionSpace:
    shape = (4,)

    def contains(self, action: np.ndarray) -> bool:
        return action.shape == self.shape and bool(np.all(np.abs(action) <= 1))


class FakePoseEnv:
    action_space = FakeActionSpace()

    def __init__(self, seed: int, *, with_pose: bool = True) -> None:
        self.seed = seed
        self.closed = False
        self._with_pose = with_pose

    def reset(self, *, seed: int) -> tuple[dict[str, object], dict[str, object]]:
        assert seed == self.seed
        return _fake_observation(seed), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        return _fake_observation(self.seed), 1.0, True, False, {"success": True}

    def close(self) -> None:
        self.closed = True


class FakeDiagnosis:
    def __init__(
        self,
        *,
        success: bool,
        ever_grasped: bool,
        ever_placed: bool,
        simultaneously_valid_steps: int,
        truncated: bool = False,
    ) -> None:
        self.success = success
        self.ever_grasped = ever_grasped
        self.ever_placed = ever_placed
        self.simultaneously_valid_steps = simultaneously_valid_steps
        self.truncated = truncated


def _config(tmp_path: Path, seeds: tuple[int, ...] = (500, 501)) -> StabilityEvalConfig:
    return StabilityEvalConfig(
        act_pick_checkpoint=tmp_path / "pick",
        act_push_checkpoint=tmp_path / "push",
        bc_checkpoint=tmp_path / "bc.pt",
        output_dir=tmp_path / "out",
        seeds=seeds,
        device="cpu",
    )


def test_default_stability_seeds() -> None:
    seeds = default_stability_seeds()
    assert len(seeds) == 50
    assert seeds == tuple(range(500, 550))
    # 与历史 eval/dev/final/DAgger 集不重叠。
    historical = set(range(100, 150)) | set(range(150, 170)) | set(
        range(200, 250)
    ) | set(range(300, 360))
    assert not historical.intersection(seeds)


def test_config_rejects_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        StabilityEvalConfig(
            act_pick_checkpoint=tmp_path,
            act_push_checkpoint=tmp_path,
            bc_checkpoint=tmp_path,
            output_dir=tmp_path,
            seeds=(500,),
            device="tpu",
        )
    with pytest.raises(ValueError, match="smoothing"):
        StabilityEvalConfig(
            act_pick_checkpoint=tmp_path,
            act_push_checkpoint=tmp_path,
            bc_checkpoint=tmp_path,
            output_dir=tmp_path,
            seeds=(500,),
            smoothing=0.0,
        )
    with pytest.raises(ValueError, match="不能重复"):
        StabilityEvalConfig(
            act_pick_checkpoint=tmp_path,
            act_push_checkpoint=tmp_path,
            bc_checkpoint=tmp_path,
            output_dir=tmp_path,
            seeds=(500, 500),
        )
    with pytest.raises(ValueError, match="不支持"):
        StabilityEvalConfig(
            act_pick_checkpoint=tmp_path,
            act_push_checkpoint=tmp_path,
            bc_checkpoint=tmp_path,
            output_dir=tmp_path,
            seeds=(500,),
            tasks=("Unknown-v1",),
        )


def test_record_initial_poses_and_coverage(tmp_path: Path) -> None:
    config = _config(tmp_path, seeds=(500, 501, 502))
    environments: list[FakePoseEnv] = []

    def factory(cfg: object) -> FakePoseEnv:
        env = FakePoseEnv(cfg.seed)
        environments.append(env)
        return env

    samples = record_initial_poses(config, env_factory=factory)
    assert len(samples) == len(SUPPORTED_TASKS) * 3
    coverage = summarize_pose_coverage(samples)
    assert set(coverage) == set(SUPPORTED_TASKS)
    pick = coverage["PickCube-v1"]
    assert pick["samples"] == 3
    assert pick["obj_max"][0] >= pick["obj_min"][0]
    assert pick["goal_mean"][2] == pytest.approx(0.3)
    assert all(env.closed for env in environments)


def test_record_initial_poses_missing_field_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path, seeds=(500,))

    class MissingPoseEnv:
        def reset(self, *, seed: int) -> tuple[dict[str, object], dict[str, object]]:
            return {"extra": {}}, {}

        def close(self) -> None:
            pass

    with pytest.raises(StabilityEvalError, match="obj_pose"):
        record_initial_poses(
            config, env_factory=lambda cfg: MissingPoseEnv()
        )


def test_compute_stability_gate_logic() -> None:
    assert compute_stability_gate(
        act_rate=0.96, bc_rate=0.92, threshold=0.80
    )["passed"] is True
    gate = compute_stability_gate(
        act_rate=0.90, bc_rate=0.92, threshold=0.80
    )
    assert gate["effective_threshold"] == 0.92
    assert gate["passed"] is False
    gate = compute_stability_gate(
        act_rate=0.96, bc_rate=0.92, exceptions=1
    )
    assert gate["passed"] is False
    gate = compute_stability_gate(
        act_rate=0.96, bc_rate=0.92, nonfinite=2
    )
    assert gate["passed"] is False
    with pytest.raises(ValueError, match="threshold"):
        compute_stability_gate(act_rate=0.9, bc_rate=0.9, threshold=1.5)


def test_classify_failure_diagnosis() -> None:
    assert (
        classify_failure_diagnosis(
            FakeDiagnosis(
                success=False,
                ever_grasped=False,
                ever_placed=False,
                simultaneously_valid_steps=0,
                truncated=True,
            )
        )
        == "truncated_only"
    )
    assert (
        classify_failure_diagnosis(
            FakeDiagnosis(
                success=False,
                ever_grasped=False,
                ever_placed=False,
                simultaneously_valid_steps=0,
            )
        )
        == "grasp_failed"
    )
    assert (
        classify_failure_diagnosis(
            FakeDiagnosis(
                success=False,
                ever_grasped=True,
                ever_placed=False,
                simultaneously_valid_steps=0,
            )
        )
        == "grasped_not_placed"
    )
    assert (
        classify_failure_diagnosis(
            FakeDiagnosis(
                success=False,
                ever_grasped=True,
                ever_placed=True,
                simultaneously_valid_steps=0,
            )
        )
        == "placed_but_not_static"
    )
    assert (
        classify_failure_diagnosis(
            FakeDiagnosis(
                success=False,
                ever_grasped=True,
                ever_placed=True,
                simultaneously_valid_steps=1,
            )
        )
        == "other"
    )


def test_analyze_failures_aggregates(tmp_path: Path, monkeypatch: Any) -> None:
    config = _config(tmp_path, seeds=(500, 501, 502, 503))

    class SequentialDiagnosis:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, env_id: str, seed: int, policy: object, **kwargs: object) -> FakeDiagnosis:
            del env_id, policy, kwargs
            self.calls += 1
            # 依次：成功、抓取失败、抓到未放、放入未静止。
            index = self.calls % 4
            if index == 0:
                return FakeDiagnosis(
                    success=True,
                    ever_grasped=True,
                    ever_placed=True,
                    simultaneously_valid_steps=1,
                )
            if index == 1:
                return FakeDiagnosis(
                    success=False,
                    ever_grasped=False,
                    ever_placed=False,
                    simultaneously_valid_steps=0,
                )
            if index == 2:
                return FakeDiagnosis(
                    success=False,
                    ever_grasped=True,
                    ever_placed=False,
                    simultaneously_valid_steps=0,
                )
            return FakeDiagnosis(
                success=False,
                ever_grasped=True,
                ever_placed=True,
                simultaneously_valid_steps=0,
            )

    diagnosis = SequentialDiagnosis()
    monkeypatch.setattr(
        "run_act_diagnose.diagnose_episode", diagnosis
    )
    policies: dict[str, object] = {}
    for env_id in SUPPORTED_TASKS:
        policies[env_id] = object()
    monkeypatch.setattr(
        "act_stability.load_stability_checkpoints",
        lambda *args, **kwargs: policies,
    )
    summaries = analyze_failures(
        config, seeds=config.seeds
    )
    assert len(summaries) == len(SUPPORTED_TASKS)
    for item in summaries:
        assert isinstance(item, FailurePhaseSummary)
        assert item.total_failures == 3
        assert item.phase_counts.get("grasp_failed") == 1
        assert item.phase_counts.get("grasped_not_placed") == 1
        assert item.phase_counts.get("placed_but_not_static") == 1
    assert (config.output_dir / "failure_analysis.json").is_file()


def test_load_stability_checkpoints_missing_dir(tmp_path: Path) -> None:
    config = _config(tmp_path, seeds=(500,))
    with pytest.raises(Exception):
        load_stability_checkpoints(
            config.act_pick_checkpoint,
            config.act_push_checkpoint,
            device="cpu",
        )


def test_summary_exceptions_aggregation() -> None:
    from act_stability import _summary_exceptions

    class Task:
        def __init__(self, exceptions: int) -> None:
            self.exceptions = exceptions

    class Summary:
        def __init__(self, tasks: list[Task]) -> None:
            self.tasks = tasks

    summary = Summary([Task(1), Task(2)])
    assert _summary_exceptions(summary) == 3


def _dummy_summary(
    policy_name: str, *, rate: float, total: int, successes: int,
    nonfinite: int = 0,
) -> PolicyEvaluationSummary:
    return PolicyEvaluationSummary(
        policy_name=policy_name,
        schema_version="x",
        created_at="t",
        requested_per_task=total,
        total_episodes=total,
        total_successes=successes,
        total_failures=total - successes,
        success_rate=rate,
        mean_inference_ms=1.0,
        mean_steps=10.0,
        total_chunk_queries=0,
        total_action_queries=0,
        nonfinite_actions=nonfinite,
        out_of_bounds_actions=0,
        tasks=(),
        episodes=(),
    )


def test_evaluate_stability_writes_json(tmp_path: Path, monkeypatch: Any) -> None:
    config = _config(tmp_path, seeds=(500,))
    policy_name = ["act", "bc", "scripted"]
    rates = [0.96, 0.92, 1.0]

    def fake_eval(
        config_: StabilityEvalConfig,
        policies: dict[str, object],
        name: str,
        *,
        env_factory: object,
    ) -> PolicyEvaluationSummary:
        del config_, policies, env_factory
        index = policy_name.index(name)
        return _dummy_summary(name, rate=rates[index], total=2, successes=2)

    monkeypatch.setattr("act_stability._evaluate_policies", fake_eval)
    monkeypatch.setattr(
        "act_stability.load_bc_checkpoint",
        lambda path: {env_id: object() for env_id in SUPPORTED_TASKS},
    )
    monkeypatch.setattr(
        "act_stability.load_stability_checkpoints",
        lambda *args, **kwargs: {
            env_id: object() for env_id in SUPPORTED_TASKS
        },
    )
    monkeypatch.setattr(
        "act_stability.ScriptedPolicy",
        lambda env_id: object(),
    )
    result = evaluate_stability(config)
    assert isinstance(result, StabilityComparisonSummary)
    assert result.act_passed
    assert result.act_rate == pytest.approx(0.96)
    assert result.effective_threshold == pytest.approx(0.92)
    payload = json.loads(
        (config.output_dir / "stability_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["act_passed"] is True


def test_run_smoothing_sweep(tmp_path: Path, monkeypatch: Any) -> None:
    config = _config(tmp_path, seeds=(500,))
    rates = {0.5: 0.90, 1.0: 0.96}

    def fake_eval(
        config_: StabilityEvalConfig,
        policies: dict[str, object],
        name: str,
        *,
        env_factory: object,
    ) -> PolicyEvaluationSummary:
        del policies, name, env_factory
        rate = rates[config_.smoothing]
        return _dummy_summary("act", rate=rate, total=2, successes=2)

    monkeypatch.setattr("act_stability._evaluate_policies", fake_eval)
    monkeypatch.setattr(
        "act_stability.load_stability_checkpoints",
        lambda *args, **kwargs: {
            env_id: object() for env_id in SUPPORTED_TASKS
        },
    )
    rows = run_smoothing_sweep(config, smoothing_values=[0.5, 1.0])
    assert [row.smoothing for row in rows] == [0.5, 1.0]
    assert rows[1].success_rate == pytest.approx(0.96)
    payload = json.loads(
        (config.output_dir / "smoothing_sweep.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(payload) == 2
    with pytest.raises(ValueError, match="不能为空"):
        run_smoothing_sweep(config, smoothing_values=[])
