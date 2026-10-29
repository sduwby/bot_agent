"""`visual_act.evaluate_visual_act` 的装配回归 —— 打桩跑通整条循环，不需要 torch / 仿真。

## 为什么需要它

这条路径默认要 torch + ManiSkill 才能跑到，本地/CI 覆盖不到。曾经在这里出过一个真错：
`action_smoothing` 从签名里删掉后，调用处还在传它 → 真机/云上一跑就 `NameError`，
而本地 1204 项测试全绿、`--dry-run` 也照样退出 0（它在加载策略前就返回了）。

**教训**：只测"计划渲染"这类前置纯函数，测不出"函数体里引用了不存在的名字"。
这里用打桩把 `evaluate_visual_act` **真正执行一遍**，并把 `extra_provider` 是否透传钉住
（B1/B2 的额外观测通道全靠它，漏传会让消融在推理侧直接报错）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import visual_act  # noqa: E402
from visual_act import (  # noqa: E402
    VisualACTError,
    env_feature_dim,
    resolve_checkpoint_env_config,
    saved_env_feature_dim,
)


class _FakePolicy:
    """满足 `run_visual_episode_in_env` 与 `ACTEvaluationEpisode` 所需的最小接口。"""

    def __init__(self) -> None:
        self.action_queries = 0
        self.chunk_queries = 0
        self.inference_ms = 0.0
        self.nonfinite_actions = 0
        self.out_of_bounds_actions = 0

    def reset(self) -> None:
        self.action_queries = 0
        self.chunk_queries = 0

    def act(self, observation: object, image: np.ndarray) -> np.ndarray:  # noqa: ARG002
        self.action_queries += 1
        return np.zeros(7, dtype=np.float32)


class _FakeEnv:
    """一帧就结束的假环境：只要 loop 能跑通、退出条件成立即可。"""

    def __init__(self) -> None:
        self.closed = False

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:  # noqa: ARG002
        return {"extra": {}, "agent": {}}, {}

    def step(self, action: object) -> tuple[dict, float, bool, bool, dict]:  # noqa: ARG002
        return {"extra": {}, "agent": {}}, 1.0, True, False, {"success": True}

    def render(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class TestEvaluateVisualActWiring:
    def test_runs_episodes_and_forwards_extra_provider(self, tmp_path, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake_load(output_dir, *, device, resolution, tasks=None,  # noqa: ARG001
                      checkpoint_subdir=None, extra_provider=None):  # noqa: ARG001
            seen["extra_provider"] = extra_provider
            seen["tasks"] = tuple(tasks or ())
            return {env_id: _FakePolicy() for env_id in (tasks or ())}

        monkeypatch.setattr(visual_act, "load_visual_act_checkpoints", fake_load)
        provider = lambda observation: [1.0, 2.0]  # noqa: E731

        summary = visual_act.evaluate_visual_act(
            tmp_path,
            seeds=[0, 1],
            device="cpu",
            resolution=64,
            tasks=("PickCube-v1",),
            extra_provider=provider,
            env_factory=lambda config: _FakeEnv(),  # noqa: ARG005
        )

        assert seen["extra_provider"] is provider, "extra_provider 必须原样透传（B1/B2 靠它）"
        assert seen["tasks"] == ("PickCube-v1",)
        assert summary.total_episodes == 2, "1 个任务 × 2 个种子"
        assert summary.total_successes == 2
        assert (tmp_path / "evaluation.json").is_file(), "评测结果必须落盘"

    def test_provider_none_is_allowed_for_baseline(self, tmp_path, monkeypatch) -> None:
        """B0 没有额外通道 —— 传 None 不能报错。"""
        monkeypatch.setattr(
            visual_act,
            "load_visual_act_checkpoints",
            lambda *a, **k: {"PickCube-v1": _FakePolicy()},
        )
        summary = visual_act.evaluate_visual_act(
            tmp_path,
            seeds=[0],
            device="cpu",
            resolution=64,
            tasks=("PickCube-v1",),
            env_factory=lambda config: _FakeEnv(),  # noqa: ARG005
        )
        assert summary.total_episodes == 1


def _features(*, env_dim: int | None) -> dict:
    """造一份 `input_features`（ENV 通道可选）。"""
    features: dict = {
        "observation.images.top": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [5]},
    }
    if env_dim is not None:
        features["observation.environment_state"] = {"type": "ENV", "shape": [env_dim]}
    return features


class TestEnvFeatureDim:
    def test_reads_dim_and_handles_absence(self) -> None:
        assert env_feature_dim(_features(env_dim=2)) == 2
        assert env_feature_dim(_features(env_dim=None)) is None
        assert env_feature_dim({}) is None
        assert saved_env_feature_dim({"input_features": _features(env_dim=13)}) == 13

    def test_reads_policy_feature_objects(self) -> None:
        """回归：内存里的 `ACTConfig.input_features` 是 `PolicyFeature` 对象，不是 dict。

        只认 dict 会让"重建维度对账"在真正出问题时静默返回 `None`（被当成 0），
        于是一个能被直接定位的报错退化成晦涩的形状不符 —— 这个坑真的踩过。
        """

        class _PolicyFeature:
            def __init__(self, shape: tuple[int, ...]) -> None:
                self.shape = shape

        with_env = {
            "observation.state": _PolicyFeature((5,)),
            "observation.environment_state": _PolicyFeature((13,)),
        }
        assert env_feature_dim(with_env) == 13
        assert env_feature_dim({"observation.state": _PolicyFeature((5,))}) is None


class TestResolveCheckpointEnvConfig:
    """回归：checkpoint 的 `config.json` **不含**口径字段，必须从 `task_metadata.json` 读。

    曾经只读 `config.json` → `env_state_mode` 退化成 `"full"`，给 B0 多注册一个 13 维 ENV
    输入，加载时在 `encoder_1d_feature_pos_embed` 上以形状不符炸掉（[3,512] vs [2,512]）。
    """

    def test_metadata_wins_over_missing_config_fields(self) -> None:
        saved = {"input_features": _features(env_dim=None)}
        metadata = {"env_state_mode": "extra_only", "extra_features": []}
        assert resolve_checkpoint_env_config(saved, metadata) == ((), "extra_only")

    def test_metadata_carries_extra_feature_names_in_order(self) -> None:
        saved = {"input_features": _features(env_dim=4)}
        metadata = {
            "env_state_mode": "extra_only",
            "extra_features": ["depth_known_size_m", "depth_known_size_sigma_m",
                               "load_gripper_permille", "contact_flag"],
        }
        extra, mode = resolve_checkpoint_env_config(saved, metadata)
        assert mode == "extra_only"
        assert extra[-1] == "contact_flag" and len(extra) == 4, "列名与顺序必须原样保留"

    def test_infers_extra_only_when_no_env_channel(self) -> None:
        """无 ENV 通道 → 只可能是 `extra_only` + 空列（`full` 必有 13 维）。"""
        assert resolve_checkpoint_env_config(
            {"input_features": _features(env_dim=None)}, {}
        ) == ((), "extra_only")

    def test_infers_full_when_env_is_13(self) -> None:
        assert resolve_checkpoint_env_config(
            {"input_features": _features(env_dim=13)}, {}
        ) == ((), "full")

    def test_raises_when_extra_columns_lost_metadata(self) -> None:
        """带额外列却没有元数据 —— 列名无从恢复，必须报错而不是猜。"""
        with pytest.raises(VisualACTError, match="无法还原额外列"):
            resolve_checkpoint_env_config({"input_features": _features(env_dim=2)}, {})
