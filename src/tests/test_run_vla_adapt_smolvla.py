"""P2 4.5 SmolVLA 适配训练侧测试 — 纯逻辑，不加载 lerobot/torch。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_vla_adapt_smolvla import (  # noqa: E402
    DONE_MARKER,
    build_train_argv,
    make_frozen_make_policy,
    select_episodes,
    should_skip,
)
from vla_adapt import build_adapt_config  # noqa: E402


class _Param:
    def __init__(self, numel: int) -> None:
        self._numel = numel
        self.requires_grad = True

    def numel(self) -> int:
        return self._numel


class _FakePolicy:
    def __init__(self) -> None:
        self._params = {
            "model.action_in_proj.weight": _Param(64),
            "model.action_out_proj.weight": _Param(64),
            "model.vlm_with_expert.vlm.vision_encoder.weight": _Param(1000),
        }

    def named_parameters(self):
        return list(self._params.items())


class TestSelectEpisodes:
    def test_equal_spacing_not_prefix(self) -> None:
        picked = select_episodes(120, 30)
        assert len(picked) == 30
        assert len(set(picked)) == 30
        assert picked[0] == 0
        assert max(picked) >= 100  # 不是"只取前 30 集"

    def test_sixty(self) -> None:
        picked = select_episodes(120, 60)
        assert len(picked) == 60 and len(set(picked)) == 60

    def test_all_when_n_ge_total(self) -> None:
        assert select_episodes(10, 10) == list(range(10))
        assert select_episodes(10, 99) == list(range(10))

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            select_episodes(120, 0)

    @pytest.mark.parametrize("n", range(1, 121))
    def test_never_duplicates_or_underfills(self, n: int) -> None:
        """全量扫 n：取整不能产生重复，否则 `if idx not in picked` 会静默少抽。"""
        picked = select_episodes(120, n)
        assert len(picked) == n
        assert len(set(picked)) == n
        assert picked == sorted(picked)
        assert all(0 <= index < 120 for index in picked)


class TestBuildTrainArgv:
    def _argv(self, strategy: str = "head", n: int = 30):
        return build_train_argv(
            config=build_adapt_config(strategy, n, 2000, 38),
            dataset_repo="wangbangyan/demo_vla_all",
            dataset_root="/data/demo_vla_all",
            output_dir="/out/head_ep30",
            device="cuda",
            episodes=[0, 4, 8],
            batch_size=32,
            num_workers=8,
        )

    def test_core_flags(self) -> None:
        argv = self._argv()
        joined = " ".join(argv)
        assert "--policy.path=lerobot/smolvla_base" in argv
        assert "--dataset.repo_id=wangbangyan/demo_vla_all" in argv
        assert "--dataset.root=/data/demo_vla_all" in argv
        assert "--dataset.episodes=[0,4,8]" in argv
        assert "--save_freq=500" in argv
        assert "--policy.push_to_hub=false" in argv
        assert "observation.images.camera1" in joined  # rename_map 生效
        assert "--steps=2000" in argv

    def test_job_name_override(self) -> None:
        argv = build_train_argv(
            config=build_adapt_config("head", 30, 2000, 38),
            dataset_repo="r",
            dataset_root="",
            output_dir="/o",
            device="cuda",
            episodes=[0, 1],
            batch_size=32,
            num_workers=8,
            job_name="vla_adapt_head_combo",
        )
        assert "--job_name=vla_adapt_head_combo" in argv

    def test_no_root_flag_when_empty(self) -> None:
        argv = build_train_argv(
            config=build_adapt_config("full", 60, 2000, 38),
            dataset_repo="r",
            dataset_root="",
            output_dir="/o",
            device="cuda",
            episodes=[0],
            batch_size=8,
            num_workers=2,
        )
        assert not any(item.startswith("--dataset.root=") for item in argv)


class TestShouldSkip:
    """矩阵幂等：只有带 DONE 标记的单元才跳过（半成品必须重跑）。"""

    def test_no_marker_is_not_skipped(self, tmp_path) -> None:
        assert should_skip(tmp_path, force=False) is False

    def test_marker_is_skipped(self, tmp_path) -> None:
        (tmp_path / DONE_MARKER).write_text("trained\n", encoding="utf-8")
        assert should_skip(tmp_path, force=False) is True

    def test_force_ignores_marker(self, tmp_path) -> None:
        (tmp_path / DONE_MARKER).write_text("trained\n", encoding="utf-8")
        assert should_skip(tmp_path, force=True) is False


class TestFrozenMakePolicy:
    def test_applies_adaptation_and_writes_stats(self, tmp_path) -> None:
        cfg = build_adapt_config("head", 30, 2000, 38)
        stats_path = tmp_path / "adapt_stats.json"
        original = lambda **_: _FakePolicy()  # noqa: E731
        wrapped = make_frozen_make_policy(original, cfg, stats_path)

        policy = wrapped()
        flags = {name: param.requires_grad for name, param in policy.named_parameters()}
        assert flags["model.action_in_proj.weight"] is True
        assert flags["model.action_out_proj.weight"] is True
        assert flags["model.vlm_with_expert.vlm.vision_encoder.weight"] is False

        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        assert payload["config"]["strategy"] == "head"
        assert 0.0 < payload["trainable_ratio"] < 1.0
        assert payload["frozen_params"] == 1000
