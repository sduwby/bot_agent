#!/usr/bin/env python
"""真机相机抓帧（第 38 周 P0 / 子任务 0.3 前置）。

为「2D 定位源选型」采集真实帧：固定相机 + 手眼相机各 N 帧，落盘 PNG + meta.json。

设计要点：直接用 ``cv2.VideoCapture(idx, cv2.CAP_MSMF)``，
**不连接机械臂、不占用 COM6**（Windows 上 DSHOW 只能打开 index 0，
故必须走 MSMF，见 `outputs/infer_so101.force_msmf_backend`）。

相机索引约定（2026-10-10 人工确认；**Windows 枚举顺序会漂移，用前请抓帧核对**）：
0=夹爪特写(handeye)、1=笔记本摄像头(勿用)、2=工作台俯拍(fixed)。

用法::

    # 两路各抓 3 帧，并记录场景描述（供后续人工标注参照）
    python src/run_perception_capture.py --out runs/week38/perception_frames/p0 `
        --repeat 3 --label "螺丝刀放在左前方"

    # 只抓固定相机
    python src/run_perception_capture.py --cameras fixed=2 --out <dir> --repeat 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from camera_slots import CAMERA_SLOTS  # noqa: E402  （唯一事实来源）

DEFAULT_CAMERAS = CAMERA_SLOTS


def parse_cameras(text: str) -> dict[str, int]:
    """解析 ``name=index[,name=index]``。"""
    mapping: dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--cameras 格式应为 name=index，收到 {part!r}")
        name, _, index = part.partition("=")
        name = name.strip()
        if not name:
            raise SystemExit(f"相机名不能为空：{part!r}")
        try:
            mapping[name] = int(index)
        except ValueError as exc:
            raise SystemExit(f"相机索引必须是整数：{part!r}") from exc
    if not mapping:
        raise SystemExit("--cameras 不能为空")
    return mapping


def grab_frame(cap, warmup: int, tries: int = 20):
    """丢弃 warmup 帧后取一帧有效图像；失败返回 None。"""
    for _ in range(max(0, warmup)):
        cap.read()
    for _ in range(max(1, tries)):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return frame
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="真机相机抓帧（P0/0.3 前置）")
    parser.add_argument("--out", default="runs/week38/perception_frames/frames",
                        help="输出目录（PNG + meta.json）")
    parser.add_argument("--cameras",
                        default=",".join(f"{k}={v}" for k, v in DEFAULT_CAMERAS.items()),
                        help="相机映射，如 fixed=1,handeye=2（默认取 camera_slots.CAMERA_SLOTS）")
    parser.add_argument("--repeat", type=int, default=3, help="每路抓几帧")
    parser.add_argument("--interval", type=float, default=0.3, help="帧间隔秒数")
    parser.add_argument("--warmup", type=int, default=10, help="每路开头丢弃的帧数（等曝光稳定）")
    parser.add_argument("--label", default="", help="场景描述，随 meta 保存供人工标注参照")
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat 至少为 1")

    import cv2

    backend = getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)
    cameras = parse_cameras(args.cameras)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[抓帧] cv2 {cv2.__version__} | 输出 {out_dir} | 每路 {args.repeat} 帧")
    records: list[dict] = []
    for name, index in cameras.items():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            print(f"[skip] {name}(index={index}) 打不开")
            cap.release()
            continue
        try:
            for i in range(args.repeat):
                frame = grab_frame(cap, args.warmup)
                if frame is None:
                    print(f"[fail] {name} 第 {i + 1} 帧读取失败")
                    continue
                path = out_dir / f"{name}_{i:02d}.png"
                if not cv2.imwrite(str(path), frame):
                    print(f"[fail] {name} 第 {i + 1} 帧写盘失败")
                    continue
                height, width = frame.shape[:2]
                records.append({
                    "camera": name, "index": index, "file": path.name,
                    "width": width, "height": height,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                })
                print(f"[ok] {path.name} {width}x{height}")
                time.sleep(max(0.0, args.interval))
        finally:
            cap.release()

    if not records:
        raise SystemExit("未抓到任何帧：请确认相机已上电、索引正确"
                         "（当前枚举 0=夹爪特写/1=笔记本摄像头/2=工作台俯拍）")

    meta = {
        "label": args.label,
        "cameras": cameras,
        "repeat": args.repeat,
        "channel_order_note": "PNG 由 cv2.imwrite 写出（颜色正确）；cv2.imread 读回为 BGR，喂 VLM 前需转 RGB",
        "frames": records,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"\n共 {len(records)} 帧 → {out_dir}（meta.json 已写入）")


if __name__ == "__main__":
    main()
