"""第 38 周 P2：vla_adapt 单元测试 —— 配置校验、参数解冻统计、保留率除零保护、矩阵封顶。

全部纯逻辑：`apply_adaptation` 用**鸭子类型**的假 policy（带 ``named_parameters()``
与 ``numel()``），不依赖 torch / GPU；`run_adapt_experiment` 通过注入假实现验证编排。
仅"与真实 nn.Module 互操作"一例用 ``pytest.importorskip("torch")``。
"""

from __future__ import annotations

import pytest

from run_vla_adapt import MAX_ADAPT_UNITS, build_adapt_jobs, unsafe_default_error
from vla_adapt import (
    AdaptConfig,
    AdaptError,
    AdaptResult,
    apply_adaptation,
    build_adapt_config,
    run_adapt_experiment,
)


# --------------------------------------------------------------------------- #
# 假 policy：与 torch.nn.Module.named_parameters() 同契约，但零依赖
# --------------------------------------------------------------------------- #

class _Param:
    """形参替身：只提供 `.numel()` 与可写的 `.requires_grad`。"""

    def __init__(self, numel: int, requires_grad: bool = True) -> None:
        self._numel = int(numel)
        self.requires_grad = bool(requires_grad)

    def numel(self) -> int:
        return self._numel


class _FakePolicy:
    """按 (name, param) 列表暴露 `named_parameters()` 的假策略。"""

    def __init__(self, tree: list[tuple[str, _Param]]) -> None:
        self._tree = list(tree)

    def named_parameters(self):
        return iter(self._tree)


def _smolvla_like_tree() -> list[tuple[str, _Param]]:
    """模仿 SmolVLA 的参数命名：视觉/语言主干 + action head 三件套。"""
    return [
        ("model.vlm_with_expert.vlm.vision_encoder.weight", _Param(1000)),
        ("model.vlm_with_expert.lm_expert.layers.0.weight", _Param(2000)),
        ("model.action_in_proj.weight", _Param(64)),
        ("model.action_out_proj.weight", _Param(64)),
        ("model.action_head.proj.weight", _Param(32)),
    ]


def _values(policy: _FakePolicy) -> dict[str, bool]:
    return {name: param.requires_grad for name, param in policy._tree}


# --------------------------------------------------------------------------- #
# 1. 配置校验
# --------------------------------------------------------------------------- #

def test_config_accepts_all_supported_strategies() -> None:
    for strategy in ("zero_shot", "head", "full"):
        config = AdaptConfig(strategy, None, 30, 100, 38)
        assert config.strategy == strategy


def test_config_rejects_lora_naming_it_as_deferred() -> None:
    with pytest.raises(AdaptError) as excinfo:
        AdaptConfig("lora", None, 30, 100, 38)
    message = str(excinfo.value)
    assert "lora" in message.lower()
    assert "延后" in message


def test_config_rejects_unknown_strategy() -> None:
    with pytest.raises(AdaptError):
        AdaptConfig("magic", None, 30, 100, 38)


def test_config_normalizes_strategy_case() -> None:
    assert AdaptConfig("HEAD", None, 30, 100, 38).strategy == "head"


@pytest.mark.parametrize(
    "field, value",
    [("n_episodes", 0), ("n_episodes", -1), ("steps", -1), ("seed", -1)],
)
def test_config_rejects_invalid_counts(field: str, value: int) -> None:
    kwargs = {"n_episodes": 30, "steps": 100, "seed": 38}
    kwargs[field] = value
    with pytest.raises(AdaptError):
        AdaptConfig("head", None, **kwargs)


def test_config_allows_zero_steps_for_zero_shot() -> None:
    # zero_shot 允许 steps=0（不训练），保留调用方取值。
    assert AdaptConfig("zero_shot", 0.0, 30, 0, 0).steps == 0


def test_config_rejects_out_of_range_trainable_ratio() -> None:
    for bad in (1.5, -0.1):
        with pytest.raises(AdaptError):
            AdaptConfig("head", bad, 30, 100, 38)


def test_config_rejects_bool_counts() -> None:
    with pytest.raises(AdaptError):
        AdaptConfig("head", None, True, 100, 38)


# --------------------------------------------------------------------------- #
# 2. build_adapt_config 的名义比率
# --------------------------------------------------------------------------- #

def test_build_adapt_config_nominal_ratios() -> None:
    assert build_adapt_config("zero_shot", 30, 0, 38).trainable_ratio == 0.0
    assert build_adapt_config("full", 30, 2000, 38).trainable_ratio == 1.0
    assert build_adapt_config("head", 30, 2000, 38).trainable_ratio is None


# --------------------------------------------------------------------------- #
# 3. apply_adaptation 三种策略
# --------------------------------------------------------------------------- #

def test_apply_adaptation_zero_shot_freezes_everything() -> None:
    policy = _FakePolicy(_smolvla_like_tree())
    summary = apply_adaptation(policy, build_adapt_config("zero_shot", 30, 0, 38))
    assert summary["trainable_params"] == 0
    assert summary["trainable_ratio"] == 0.0
    assert summary["trainable_modules"] == ()
    assert summary["frozen_params"] == summary["total_params"]
    assert all(not flag for flag in _values(policy).values())


def test_apply_adaptation_full_unfreezes_everything() -> None:
    policy = _FakePolicy(_smolvla_like_tree())
    summary = apply_adaptation(policy, build_adapt_config("full", 30, 2000, 38))
    assert summary["trainable_params"] == summary["total_params"]
    assert summary["trainable_ratio"] == 1.0
    assert summary["frozen_params"] == 0
    assert all(_values(policy).values())


def test_unsafe_default_guard() -> None:
    """非 dry-run 必须拒绝旧路径（head≡full + ACT 加载器），并指向新入口。"""
    assert unsafe_default_error(dry_run=True, allow_unsafe_default=False) is None
    assert unsafe_default_error(dry_run=False, allow_unsafe_default=True) is None
    message = unsafe_default_error(dry_run=False, allow_unsafe_default=False)
    assert message is not None
    assert "run_vla_adapt_smolvla" in message
    assert "run_adapt_eval_offline" in message


def test_apply_adaptation_head_unfreezes_only_action_modules() -> None:
    policy = _FakePolicy(_smolvla_like_tree())
    summary = apply_adaptation(policy, build_adapt_config("head", 30, 2000, 38))

    total = 1000 + 2000 + 64 + 64 + 32
    assert summary["total_params"] == total
    assert summary["trainable_params"] == 64 + 64 + 32  # in/out proj + action head
    assert summary["trainable_ratio"] == pytest.approx((64 + 64 + 32) / total)
    assert summary["trainable_modules"] == (
        "model.action_in_proj",
        "model.action_out_proj",
        "model.action_head.proj",
    )
    flags = _values(policy)
    assert flags["model.vlm_with_expert.vlm.vision_encoder.weight"] is False
    assert flags["model.action_in_proj.weight"] is True
    assert flags["model.action_head.proj.weight"] is True


def test_head_patterns_cover_time_mlp_and_state_proj() -> None:
    """SmolVLA 实测命名：漏掉这两组会让 head 只剩 47k 参数（0.01%）。"""
    tree = [
        ("model.action_time_mlp_in.weight", _Param(10)),
        ("model.action_time_mlp_out.weight", _Param(10)),
        ("model.state_proj.weight", _Param(10)),
        ("model.vlm_with_expert.vlm.vision_encoder.weight", _Param(1000)),
    ]
    policy = _FakePolicy(tree)
    summary = apply_adaptation(policy, build_adapt_config("head", 30, 2000, 38))
    assert summary["trainable_params"] == 30
    assert _values(policy)["model.vlm_with_expert.vlm.vision_encoder.weight"] is False


def test_apply_adaptation_head_matches_nothing_raises_and_does_not_mutate() -> None:
    """安全性质：head 命中为空必须报错，且不得改动任何 requires_grad。"""
    policy = _FakePolicy([("backbone.conv.weight", _Param(48))])
    before = _values(policy)
    with pytest.raises(AdaptError) as excinfo:
        apply_adaptation(policy, build_adapt_config("head", 30, 2000, 38))
    assert "head" in str(excinfo.value)
    assert _values(policy) == before  # 全冻结/全解冻的静默退化被禁止


def test_apply_adaptation_custom_head_patterns() -> None:
    policy = _FakePolicy(
        [
            ("backbone.vision.weight", _Param(100)),
            ("custom.head.proj.weight", _Param(10)),
        ]
    )
    summary = apply_adaptation(
        policy, build_adapt_config("head", 30, 2000, 38), head_patterns=("custom.head",)
    )
    assert summary["trainable_params"] == 10
    assert summary["trainable_modules"] == ("custom.head.proj",)


def test_apply_adaptation_requires_named_parameters() -> None:
    with pytest.raises(AdaptError):
        apply_adaptation(object(), build_adapt_config("full", 30, 2000, 38))


def test_apply_adaptation_interops_with_real_torch_module() -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.action_out_proj = nn.Linear(4, 2)

    net = _Net()
    summary = apply_adaptation(net, build_adapt_config("head", 30, 2000, 38))
    assert summary["trainable_params"] > 0
    assert net.action_out_proj.weight.requires_grad is True
    assert net.backbone.weight.requires_grad is False


# --------------------------------------------------------------------------- #
# 4. run_adapt_experiment（注入假实现）
# --------------------------------------------------------------------------- #

def _fakes(before: float, after: float, train_outcome: object = 12.5):
    calls = {"train": 0, "eval": 0, "baseline": 0}

    def train_fn(config, dataset_root, output_dir, device):
        calls["train"] += 1
        return train_outcome

    def baseline_eval_fn(config, dataset_root, seeds, device, task):
        calls["baseline"] += 1
        return before

    def eval_fn(config, dataset_root, seeds, device, task):
        calls["eval"] += 1
        return after

    return train_fn, eval_fn, baseline_eval_fn, calls


def test_run_adapt_experiment_head_aggregates_result() -> None:
    config = build_adapt_config("head", 30, 100, 38)
    train_fn, eval_fn, baseline_eval_fn, calls = _fakes(before=0.4, after=0.6)
    result = run_adapt_experiment(
        config,
        "data/multi_task",
        [0, 1, 2],
        baseline_seeds=[100, 101],
        baseline_task="Pick-Original",
        train_fn=train_fn,
        eval_fn=eval_fn,
        baseline_eval_fn=baseline_eval_fn,
        device="cpu",
    )
    assert isinstance(result, AdaptResult)
    assert result.success_rate == 0.6
    assert result.retention_rate == pytest.approx(0.6 / 0.4)
    assert result.train_seconds == 12.5
    assert result.peak_mem_mb is None  # 未上报 → None（不是 0）
    assert calls == {"train": 1, "eval": 1, "baseline": 1}

    payload = result.to_dict()
    assert payload["config"]["strategy"] == "head"
    assert payload["config"]["trainable_ratio"] is None
    assert payload["retention_rate"] == pytest.approx(1.5)


def test_run_adapt_experiment_zero_shot_skips_training() -> None:
    config = build_adapt_config("zero_shot", 30, 100, 38)

    def train_fn(*_args, **_kwargs):
        raise AssertionError("zero_shot 不应调用 train_fn")

    _, eval_fn, baseline_eval_fn, calls = _fakes(before=0.5, after=0.5)
    result = run_adapt_experiment(
        config,
        "data/multi_task",
        [0, 1],
        train_fn=train_fn,
        eval_fn=eval_fn,
        baseline_eval_fn=baseline_eval_fn,
        device="cpu",
    )
    assert result.train_seconds == 0.0
    assert result.success_rate == 0.5
    assert result.retention_rate == 1.0  # 未训练 → 无遗忘
    assert calls["train"] == 0


def test_run_adapt_experiment_retention_division_by_zero_is_none() -> None:
    config = build_adapt_config("full", 30, 100, 38)
    train_fn, eval_fn, baseline_eval_fn, _ = _fakes(before=0.0, after=0.7)
    result = run_adapt_experiment(
        config,
        "data/multi_task",
        [0, 1],
        train_fn=train_fn,
        eval_fn=eval_fn,
        baseline_eval_fn=baseline_eval_fn,
        device="cpu",
    )
    assert result.success_rate == 0.7
    assert result.retention_rate is None  # 除零保护：不可计算 → None


def test_run_adapt_experiment_accepts_train_outcome_mapping() -> None:
    config = build_adapt_config("full", 30, 100, 38)
    train_fn, eval_fn, baseline_eval_fn, _ = _fakes(
        before=0.3, after=0.5, train_outcome={"train_seconds": 3.0, "peak_mem_mb": 512.0}
    )
    result = run_adapt_experiment(
        config,
        "data/multi_task",
        [0, 1],
        train_fn=train_fn,
        eval_fn=eval_fn,
        baseline_eval_fn=baseline_eval_fn,
        device="cpu",
    )
    assert result.train_seconds == 3.0
    assert result.peak_mem_mb == 512.0


def test_run_adapt_experiment_validates_seeds() -> None:
    config = build_adapt_config("head", 30, 100, 38)
    train_fn, eval_fn, baseline_eval_fn, _ = _fakes(before=0.4, after=0.6)
    with pytest.raises(AdaptError):
        run_adapt_experiment(
            config, "d", [], train_fn=train_fn, eval_fn=eval_fn,
            baseline_eval_fn=baseline_eval_fn, device="cpu",
        )
    with pytest.raises(AdaptError):
        run_adapt_experiment(
            config, "d", [0, 0], train_fn=train_fn, eval_fn=eval_fn,
            baseline_eval_fn=baseline_eval_fn, device="cpu",
        )


# --------------------------------------------------------------------------- #
# 5. CLI 矩阵构造（纯函数）与封顶
# --------------------------------------------------------------------------- #

def test_build_adapt_jobs_default_matrix_is_five_units() -> None:
    jobs = build_adapt_jobs(["zero_shot", "head", "full"], [30, 60], 2000, 38)
    assert len(jobs) == 5  # 1 锚点 + 2 策略 × 2 数据量
    assert [job.job_name for job in jobs] == [
        "zero_shot",
        "head-ep30",
        "head-ep60",
        "full-ep30",
        "full-ep60",
    ]


def test_build_adapt_jobs_zero_shot_is_single_anchor() -> None:
    jobs = build_adapt_jobs(["zero_shot"], "30,60", 2000, 38)
    assert len(jobs) == 1
    anchor = jobs[0]
    assert anchor.data_size == 0
    assert anchor.config.strategy == "zero_shot"
    assert anchor.config.n_episodes > 0  # 占位值，仅为满足配置校验


def test_build_adapt_jobs_orders_strategies_canonically() -> None:
    jobs = build_adapt_jobs(["full", "head"], [60], 100, 7)
    assert [job.job_name for job in jobs] == ["head-ep60", "full-ep60"]


def test_build_adapt_jobs_dedupes_strategies() -> None:
    jobs = build_adapt_jobs(["head", "head"], [30], 100, 7)
    assert [job.job_name for job in jobs] == ["head-ep30"]


def test_build_adapt_jobs_rejects_duplicate_sizes() -> None:
    with pytest.raises(AdaptError):
        build_adapt_jobs(["head"], "30,30", 100, 7)


def test_build_adapt_jobs_allows_exactly_six_units() -> None:
    jobs = build_adapt_jobs(["head", "full"], [30, 60, 90], 100, 7)
    assert len(jobs) == MAX_ADAPT_UNITS == 6


def test_build_adapt_jobs_caps_over_six_units() -> None:
    with pytest.raises(AdaptError):
        # 1 锚点 + 2 策略 × 3 数据量 = 7 > 6
        build_adapt_jobs(["zero_shot", "head", "full"], [30, 60, 90], 100, 7)


def test_build_adapt_jobs_rejects_lora_and_unknown() -> None:
    with pytest.raises(AdaptError):
        build_adapt_jobs(["lora"], [30], 100, 7)
    with pytest.raises(AdaptError):
        build_adapt_jobs(["nope"], [30], 100, 7)


def test_build_adapt_jobs_rejects_non_positive_size() -> None:
    with pytest.raises(AdaptError):
        build_adapt_jobs(["head"], [0], 100, 7)
