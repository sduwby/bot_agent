#!/usr/bin/env python
"""固定相机取景对齐（第 38 周 P0）。

背景：ACT 是纯视觉条件化的策略，**相机取景必须与采集时一致**。相机被调过位置后
策略输入 OOD，表现是"能走过去但夹爪不张开/后续没动作"。本工具循环抓「固定相机」
画面并与**数据集训练帧**比对，实时打印 MAE / 相关系数 —— 边微调相机朝向边看数字
往 0.8 以上爬即可，不必反复跑实验。

**适用范围**：只适用于**固定相机**（它的画面不随手臂姿态变化）。
**不要**用它判断腕部相机映射——腕部画面随姿态变，本指标会给出错误结论
（week38 实测：真腕部 index0 得分 0.135，反而低于无关的笔记本摄像头 0.555）。
腕部/整体映射必须**人眼核对**（`python outputs/infer_so101.py --camera-check`）。

用法::

    # 单次测量
    python src/run_camera_align.py --index 2

    # 循环刷新（每 2 秒），调相机时用
    python src/run_camera_align.py --index 2 --watch

    # 只导出训练参考帧供肉眼对照
    python src/run_camera_align.py --dump-reference
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from camera_slots import CAMERA_SLOTS  # noqa: E402  （唯一事实来源）

#: 对齐参考默认取**当前策略的训练数据集**（VLA 微调用的那批）；换数据集时显式传参。
DEFAULT_DATASET_ROOT = (
    r"C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\demo_vla_01_pick_a"
)
DEFAULT_DATASET_REPO = "wangbangyan/demo_vla_01_pick_a"
DEFAULT_OUT = "runs/week38/cam_align"

# 相关系数分档（经验值：同一相机、同取景、仅物体位置不同时实测 ≈0.8+）
GOOD = 0.80
CLOSE = 0.50


def frame_similarity(a, b) -> tuple[float, float]:
    """返回 ``(MAE, 皮尔逊相关系数)``；两者都是灰度、同尺寸后计算。

    MAE 越小、相关系数越接近 1 表示越像。
    """
    import cv2
    import numpy as np

    gray_a = cv2.cvtColor(np.asarray(a), cv2.COLOR_RGB2GRAY).astype(np.int16)
    other = np.asarray(b)
    if other.shape[:2] != gray_a.shape[:2]:
        other = cv2.resize(other, (gray_a.shape[1], gray_a.shape[0]))
    gray_b = cv2.cvtColor(other, cv2.COLOR_RGB2GRAY).astype(np.int16)
    mae = float(np.abs(gray_a - gray_b).mean())
    corr = float(np.corrcoef(gray_a.ravel(), gray_b.ravel())[0, 1])
    return mae, corr


def verdict(corr: float) -> str:
    if corr >= GOOD:
        return "良好（取景与训练一致）"
    if corr >= CLOSE:
        return "接近（继续微调）"
    return "明显不一致（需调相机）"


def to_hwc_u8(x):
    import numpy as np

    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        peak = float(arr.max()) if arr.size else 0.0
        if peak <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_reference(dataset_root: str, dataset_repo: str) -> dict:
    """取数据集第 0 帧的两路训练画面（策略真正的输入参照）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
    import infer_so101 as base  # 复用其本地缓存/数据集加载逻辑

    _, ds = base.load_policy(base.MODEL_DIR, dataset_repo, dataset_root)
    frame = ds[0]
    return {key.split(".")[-1]: to_hwc_u8(frame[key])
            for key in ("observation.images.fixed", "observation.images.handeye")}


def open_camera(index: int):
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        return None
    for _ in range(15):  # 等曝光稳定
        cap.read()
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="固定相机取景对齐（与训练帧比对）")
    parser.add_argument("--index", type=int, default=CAMERA_SLOTS["fixed"],
                        help=f"固定相机索引（默认 {CAMERA_SLOTS['fixed']}，取自 camera_slots.py）")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--out", default=DEFAULT_OUT, help="导出图片的目录")
    parser.add_argument("--watch", action="store_true", help="循环刷新（每 2 秒），调相机时用")
    parser.add_argument("--dump-reference", action="store_true", help="只导出训练参考帧")
    args = parser.parse_args()

    import cv2
    import numpy as np

    print(">>> 加载数据集训练参考帧 ...", flush=True)
    reference = load_reference(args.dataset_root, args.dataset_repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in reference.items():
        cv2.imwrite(str(out_dir / f"reference_{name}.png"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    print(f"    参考帧已导出 → {out_dir}\\reference_fixed.png / reference_handeye.png", flush=True)
    if args.dump_reference:
        return

    cap = open_camera(args.index)
    if cap is None:
        raise SystemExit(f"打不开相机 index={args.index}")
    print(f">>> 已打开 index={args.index}；目标：与 reference_fixed.png 的相关系数 ≥ {GOOD}", flush=True)
    if args.watch:
        print("    循环模式：调整相机朝向，观察相关系数变化；Ctrl+C 结束。", flush=True)

    latest_path = out_dir / f"live_idx{args.index}.png"
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("  读取失败，重试 ...", flush=True)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mae_f, r_f = frame_similarity(rgb, reference["fixed"])
            mae_h, r_h = frame_similarity(rgb, reference["handeye"])
            cv2.imwrite(str(latest_path), frame)
            print(f"  vs 训练 fixed: MAE={mae_f:5.1f}  r={r_f:+.3f}  [{verdict(r_f)}]"
                  f"   |  (参照) vs 训练 handeye: r={r_h:+.3f}", flush=True)
            if not args.watch:
                break
            import time

            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n已停止。", flush=True)
    finally:
        cap.release()
    print(f"最新实拍帧 → {latest_path}（可与 {out_dir}\\reference_fixed.png 肉眼对照）", flush=True)


if __name__ == "__main__":
    main()
