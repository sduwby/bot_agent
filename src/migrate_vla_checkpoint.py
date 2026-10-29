#!/usr/bin/env python
"""把 lerobot 0.6.x 训出的 SmolVLA checkpoint 迁移成 0.3.4 可直接加载的版本。

## 三处差异（第 38 周批次 38 逐层取证）

1. **config**：0.6.x 多写 6 个字段，0.3.4 的 draccus 会直接拒绝 → 剪掉；
2. **相机键**：训练侧用 `--rename_map` 把 `fixed/handeye` 改名成 `camera1/camera2`，
   而机器人侧推理用的是 `fixed/handeye` → 改回去（VISUAL 走 IDENTITY 归一化，不改权重）；
   顺带删掉从未喂过数据的 `camera3`；
3. **归一化统计**：0.6.x 把 mean/std 移进 `*_processor.safetensors`，0.3.4 的
   `Normalize` 把它当 buffer 存在 policy state_dict 里（缺省 inf，forward 会 assert）
   → 从 processor 文件回填。

**注意**：第 3 步必须能读到两个 `*_processor.safetensors`；云端 `split` 下载时容易漏拉。

用法::

    python src/migrate_vla_checkpoint.py \
        --src runs/week38/smolvla_vla_40k/pretrained_model \
        --out runs/week38/smolvla_vla_40k/pretrained_model_lr034 \
        --vlm-model-name runs/week38/models/SmolVLM2-500M-Video-Instruct
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: 0.6.x 保存时多打、0.3.4 不认识的字段（纯保存侧默认值，不涉及权重）。
FIELDS_TO_DROP: tuple[str, ...] = (
    "use_peft",
    "pretrained_path",
    "pretrained_revision",
    "rtc_config",
    "compile_model",
    "compile_mode",
)

#: 训练侧 rename_map 的逆映射（顺序保持 camera1→fixed，视觉特征走 IDENTITY，不涉及权重）。
CAMERA_RENAME: dict[str, str] = {
    "observation.images.camera1": "observation.images.fixed",
    "observation.images.camera2": "observation.images.handeye",
}

#: 0.6.x 三个相机，实际只喂了两路 → camera3 删除。
DROPPED_CAMERAS: tuple[str, ...] = ("observation.images.camera3",)

#: 归一化统计的字段后缀（processor safetensors 里的键形如 `observation.state.mean`）。
STAT_SUFFIXES: tuple[str, ...] = (
    "count", "min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99",
)

#: 需要从 processor 回填 buffer 的模块路径 → 特征名。
BUFFER_FILL: dict[str, str] = {
    "normalize_inputs.buffer_observation_state": "observation.state",
    "normalize_targets.buffer_action": "action",
    "unnormalize_outputs.buffer_action": "action",
}


def prepare_config(
    raw: Mapping[str, Any],
    *,
    vlm_model_name: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """把 0.6.x 的 config.json 剪成 0.3.4 可解析的形态（纯函数，不读文件）。

    - 丢弃 `FIELDS_TO_DROP`；
    - `camera1/2 → fixed/handeye`，删除 `camera3`；
    - `load_vlm_weights=False`（VLM 权重就在 checkpoint 里）与 `vlm_model_name`
      指向本地目录（离线环境必需）。
    """
    cfg: dict[str, Any] = {k: v for k, v in raw.items() if k not in FIELDS_TO_DROP}
    features = {
        CAMERA_RENAME.get(key, key): value
        for key, value in dict(cfg.get("input_features", {})).items()
    }
    for key in DROPPED_CAMERAS:
        features.pop(key, None)
    cfg["input_features"] = features
    cfg["device"] = device
    cfg["load_vlm_weights"] = False
    if vlm_model_name:
        cfg["vlm_model_name"] = vlm_model_name
    return cfg


def load_stats(src_dir: Path) -> dict[str, dict[str, Any]]:
    """从 `*_processor.safetensors` 抽出 `{特征: {统计量: tensor}}`。

    Raises:
        FileNotFoundError: 一个 processor 文件都没有（迁移无法完成回填）。
    """
    from safetensors.torch import load_file

    files = sorted(Path(src_dir).glob("*_processor.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"{src_dir} 下没有 *_processor.safetensors：归一化统计无处回填。"
            "（远端 split 下载时最容易漏掉这两个文件）"
        )
    stats: dict[str, dict[str, Any]] = {}
    for path in files:
        for name, tensor in load_file(str(path)).items():
            for suffix in STAT_SUFFIXES:
                if name.endswith("." + suffix):
                    stats.setdefault(name[: -(len(suffix) + 1)], {})[suffix] = tensor
                    break
    return stats


def migrate(src_dir: str | Path, out_dir: str | Path, *, vlm_model_name: str | None = None) -> Path:
    """执行迁移并写到 `out_dir`，返回产物目录。"""
    src = Path(src_dir)
    out = Path(out_dir)
    raw = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg = prepare_config(raw, vlm_model_name=vlm_model_name)
    stats = load_stats(src)

    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    with tempfile.TemporaryDirectory(prefix="smolvla_mig_") as tmp:
        pruned = Path(tmp) / "pruned"
        pruned.mkdir()
        (pruned / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        config = PreTrainedConfig.from_pretrained(str(pruned))
        policy = SmolVLAPolicy.from_pretrained(str(src), config=config)

        with torch.no_grad():
            for attr, feature in BUFFER_FILL.items():
                module = policy
                for part in attr.split("."):
                    module = getattr(module, part)
                for stat in ("mean", "std"):
                    if feature not in stats or stat not in stats[feature]:
                        raise SystemExit(f"缺统计量 {feature}.{stat}（processor 文件不完整？）")
                    module[stat].data = stats[feature][stat].clone().to(dtype=torch.float32)

        if out.exists():
            shutil.rmtree(out)
        policy.config.device = "cpu"
        policy.save_pretrained(str(out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="SmolVLA checkpoint 0.6.x → 0.3.4 迁移")
    parser.add_argument("--src", required=True, help="0.6.x checkpoint 目录（含 7 个文件）")
    parser.add_argument("--out", required=True, help="输出目录（自包含）")
    parser.add_argument("--vlm-model-name", default="", help="SmolVLM 本地目录（离线推理必需）")
    args = parser.parse_args()
    out = migrate(args.src, args.out, vlm_model_name=args.vlm_model_name or None)
    print(f"[migrate] 已写出 {out}：{sorted(p.name for p in out.iterdir())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
