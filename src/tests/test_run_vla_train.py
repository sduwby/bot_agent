"""SmolVLA 训练入口组装测试（lerobot 0.6.x 口径回归）。

背景：云端实测发现旧实现有三处与 lerobot 0.6.x 不符——入口名（`lerobot.scripts.train`
不存在）、微调方式（应用 `--policy.path` 而非 `--policy.type`）、以及被删掉的
`--device` / `--eval_freq` 顶层参数。这些测试把**已实测跑通**的口径钉住，防止回退。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_vla_train import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    DEFAULT_RENAME_MAP,
    TRAIN_MODULE,
    build_parser,
    build_train_command,
)


def _cmd(**overrides: object) -> list[str]:
    argv: list[str] = []
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        else:
            argv.extend([flag, str(value)])
    return build_train_command(build_parser().parse_args(argv))


def test_uses_lerobot_train_entrypoint_not_legacy_train() -> None:
    cmd = _cmd()
    assert TRAIN_MODULE == "lerobot.scripts.lerobot_train"
    assert "lerobot.scripts.lerobot_train" in cmd
    # 0.6.x 没有 lerobot.scripts.train；出现即回退。
    assert "lerobot.scripts.train" not in cmd


def test_finetune_uses_policy_path_not_policy_type() -> None:
    cmd = _cmd()
    assert f"--policy.path={DEFAULT_POLICY_PATH}" in cmd
    assert "--policy.type=smolvla" not in cmd
    assert not any(item.startswith("--policy.type=") for item in cmd)


def test_no_legacy_top_level_flags() -> None:
    """`--device` 与 `--eval_freq` 在 0.6.x 顶层不存在（实测报 unrecognized arguments）。"""
    cmd = _cmd()
    assert not any(item.startswith("--eval_freq") for item in cmd)
    assert "--device=cuda" not in cmd
    assert not any(item.startswith("--device=") for item in cmd)


def test_device_maps_to_policy_device() -> None:
    cmd = _cmd(device="cpu")
    assert "--policy.device=cpu" in cmd


def test_default_rename_map_maps_repo_cameras_to_smolvla_slots() -> None:
    """本仓相机名（fixed/handeye）≠ smolvla_base（camera1..3），默认必须重命名。"""
    cmd = _cmd()
    rename = [item for item in cmd if item.startswith("--rename_map=")]
    assert len(rename) == 1
    assert rename[0] == f"--rename_map={DEFAULT_RENAME_MAP}"
    assert "observation.images.fixed" in rename[0]
    assert "observation.images.camera1" in rename[0]
    assert "observation.images.handeye" in rename[0]
    assert "observation.images.camera2" in rename[0]


def test_empty_rename_map_is_omitted() -> None:
    cmd = _cmd(rename_map="")
    assert not any(item.startswith("--rename_map=") for item in cmd)


def test_repo_id_is_a_single_string_not_a_list() -> None:
    """0.6.x 的 MultiLeRobotDataset 未启用：多 repo 列表会直接报错，必须传单 repo。"""
    cmd = _cmd(dataset_repo="wangbangyan/demo_vla_all")
    repo = [item for item in cmd if item.startswith("--dataset.repo_id=")]
    assert repo == ["--dataset.repo_id=wangbangyan/demo_vla_all"]
    assert "[" not in repo[0] and "," not in repo[0]


def test_dataset_root_appended_only_when_given() -> None:
    assert not any(item.startswith("--dataset.root=") for item in _cmd())
    cmd = _cmd(dataset_root="/data/demo_vla_all")
    assert "--dataset.root=/data/demo_vla_all" in cmd


def test_resume_flag() -> None:
    assert "--resume=true" not in _cmd()
    assert "--resume=true" in _cmd(resume=True)


def test_defaults_match_cloud_smoke_run() -> None:
    args = build_parser().parse_args([])
    assert args.steps == 20000
    assert args.eval_steps == 0
    assert args.dataset_repo == "wangbangyan/demo_vla_all"


def test_vla_adapt_builder_reuses_the_same_constants() -> None:
    """vla_adapt 与 run_vla_train 共用同一套入口/权重/重命名，避免两处漂移。"""
    from vla_adapt import build_adapt_config, _build_smolvla_train_command

    config = build_adapt_config("full", 30, 2000, 38)
    cmd = _build_smolvla_train_command(config, "", None, "cuda")
    assert TRAIN_MODULE in cmd
    assert f"--policy.path={DEFAULT_POLICY_PATH}" in cmd
    assert "--policy.device=cuda" in cmd
    assert f"--rename_map={DEFAULT_RENAME_MAP}" in cmd
    assert not any(item.startswith("--policy.type=") for item in cmd)
    assert not any(item.startswith("--device=") for item in cmd)
