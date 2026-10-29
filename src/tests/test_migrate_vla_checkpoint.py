"""checkpoint 迁移工具测试 — 纯逻辑部分（不加载 0.6.x 权重）。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from migrate_vla_checkpoint import (  # noqa: E402
    CAMERA_RENAME,
    FIELDS_TO_DROP,
    load_stats,
    prepare_config,
)

RAW = {
    "type": "smolvla",
    "use_peft": False,
    "pretrained_path": "lerobot/smolvla_base",
    "pretrained_revision": "main",
    "rtc_config": None,
    "compile_model": False,
    "compile_mode": None,
    "device": "cuda",
    "load_vlm_weights": True,
    "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.camera1": {"type": "VISUAL", "shape": [3, 512, 512]},
        "observation.images.camera2": {"type": "VISUAL", "shape": [3, 512, 512]},
        "observation.images.camera3": {"type": "VISUAL", "shape": [3, 512, 512]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
}


class TestPrepareConfig:
    def test_drops_06_only_fields(self) -> None:
        cfg = prepare_config(RAW)
        for field in FIELDS_TO_DROP:
            assert field not in cfg
        assert cfg["type"] == "smolvla"

    def test_renames_cameras_and_drops_camera3(self) -> None:
        features = prepare_config(RAW)["input_features"]
        assert "observation.images.fixed" in features
        assert "observation.images.handeye" in features
        assert "observation.images.camera1" not in features
        assert "observation.images.camera2" not in features
        assert "observation.images.camera3" not in features
        assert CAMERA_RENAME["observation.images.camera1"] == "observation.images.fixed"

    def test_sets_offline_fields(self) -> None:
        cfg = prepare_config(RAW, vlm_model_name="/models/smolvlm")
        assert cfg["device"] == "cpu"
        assert cfg["load_vlm_weights"] is False
        assert cfg["vlm_model_name"] == "/models/smolvlm"

    def test_does_not_mutate_input(self) -> None:
        before = {key: dict(value) if isinstance(value, dict) else value for key, value in RAW.items()}
        prepare_config(RAW)
        assert RAW == before

    def test_idempotent(self) -> None:
        once = prepare_config(RAW)
        twice = prepare_config(once)
        assert once == twice


class TestLoadStats:
    def test_missing_processors_raise(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError) as excinfo:
            load_stats(tmp_path)
        assert "processor" in str(excinfo.value)

    def test_parses_stat_suffixes(self, tmp_path) -> None:
        torch = pytest.importorskip("torch")
        from safetensors.torch import save_file

        save_file(
            {
                "observation.state.mean": torch.zeros(6),
                "observation.state.std": torch.ones(6),
                "action.mean": torch.zeros(6),
            },
            str(tmp_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"),
        )
        stats = load_stats(tmp_path)
        assert sorted(stats["observation.state"]) == ["mean", "std"]
        assert sorted(stats["action"]) == ["mean"]
