#!/usr/bin/env python
"""P2 4.5：在**训练环境（lerobot 0.6.x）**里对适配 checkpoint 跑离线评测。

流程：加载 checkpoint（自带 pre/post processor，含归一化统计）→ 在固定 episode 子集上
逐帧 `predict_action_chunk` → 取 chunk 首个动作与示范动作比较 → 输出指标表。

**口径边界**：这是训练侧代理指标，不等于真机成功率；A6 必须真机实测
（见 `adapt_eval_offline.CAVEAT`）。

用法（云上 A10）::

    python src/run_adapt_eval_offline.py \
        --model-dir outputs/vla_adapt/head_ep30/train/checkpoints/last/pretrained_model \
        --dataset-repo wangbangyan/demo_vla_all \
        --dataset-root ~/.cache/huggingface/lerobot/wangbangyan/demo_vla_all \
        --episodes "[18,38,58,78,98,118]" --device cuda --stride 5 --max-frames 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from adapt_eval_offline import (  # noqa: E402
    CAVEAT,
    compute_offline_metrics,
    parse_episode_list,
    select_frame_ids,
)


def _to_vector(value) -> np.ndarray:
    """tensor / ndarray / list → 一维 float ndarray。"""
    arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return np.asarray(arr, dtype=float).reshape(-1)


def _rename_cameras(item: dict, mapping: dict[str, str]) -> dict:
    """按映射重命名观测键（用于评测 `smolvla_base`：它期望 camera1/2，数据集是 fixed/handeye）。"""
    if not mapping:
        return item
    return {mapping.get(key, key): value for key, value in item.items()}


#: `smolvla_base` 期望的相机键（本仓数据集用 fixed/handeye）。
RENAME_TO_BASE_CAMERAS = {
    "observation.images.fixed": "observation.images.camera1",
    "observation.images.handeye": "observation.images.camera2",
}


def run(args: argparse.Namespace) -> dict[str, object]:
    episodes = parse_episode_list(args.episodes)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        args.dataset_repo,
        root=args.dataset_root or None,
        episodes=episodes,
    )
    print(f"[eval] 数据集 {args.dataset_repo}：{len(ds)} 帧 / episodes={episodes}", flush=True)
    if args.dry_run:
        item = ds[0]
        print("[eval] dry-run：batch 键 =", sorted(item), flush=True)
        return {"dry_run": True, "n_frames": len(ds)}

    import torch

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(args.model_dir)
    policy.to(args.device)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.model_dir)
    print(f"[eval] 模型已加载：{args.model_dir}（device={args.device}）", flush=True)

    frame_ids = select_frame_ids(
        len(ds),
        stride=args.stride,
        offset=args.offset,
        max_frames=args.max_frames,
        even=args.even,
    )
    rename = RENAME_TO_BASE_CAMERAS if args.base_cameras else {}
    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    with torch.no_grad():
        for count, index in enumerate(frame_ids, start=1):
            item = _rename_cameras(ds[index], rename)
            batch = pre(item)
            chunk = policy.predict_action_chunk(batch)
            first = chunk[0, 0] if chunk.ndim == 3 else chunk[0]
            try:
                action = post(first)
            except Exception:  # 后处理器对单条动作的形态容忍度不一，失败时直接用原值
                action = first
            preds.append(_to_vector(action)[: len(_to_vector(item["action"]))])
            gts.append(_to_vector(item["action"]))
            if count % 20 == 0:
                print(f"[eval] {count}/{len(frame_ids)} 帧", flush=True)

    metrics = compute_offline_metrics(np.stack(preds), np.stack(gts))
    payload = {
        "model_dir": args.model_dir,
        "dataset_repo": args.dataset_repo,
        "episodes": episodes,
        "stride": args.stride,
        "even": bool(args.even),
        "metrics": metrics.to_dict(),
        "caveat": CAVEAT,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "offline_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[eval] n={metrics.n_frames} actionL1={metrics.action_l1:.4f} "
        f"gripL1={metrics.gripper_l1:.4f} holdF1={metrics.hold_f1:.3f} → {out_dir}",
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="P2 4.5 适配 checkpoint 离线评测")
    parser.add_argument("--model-dir", required=True, help="checkpoint 目录（含 config/processor）")
    parser.add_argument("--dataset-repo", default="wangbangyan/demo_vla_all")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--episodes", required=True, help='episode 列表，如 "[18,38,58]"')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=5, help="每 N 帧取一帧")
    parser.add_argument("--offset", type=int, default=0,
                        help="帧起点（多次重复评测时用它换一批帧，避免单次采样结论）")
    parser.add_argument("--max-frames", type=int, default=120, help="最多评测帧数")
    parser.add_argument("--even", action="store_true",
                        help="在所选 episode 全区间均匀取帧（A7 整段任务用；默认按 stride）")
    parser.add_argument("--out", default="runs/week38/vla_adapt_eval")
    parser.add_argument("--base-cameras", action="store_true",
                        help="把 fixed/handeye 重命名为 camera1/2（评测未微调的 smolvla_base 时必须加）")
    parser.add_argument("--dry-run", action="store_true", help="只验证数据集加载与键名")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
