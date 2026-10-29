#!/usr/bin/env python
"""相机索引体检 —— 采集前跑一遍，确认 **哪个索引是哪台相机、能不能出帧**。

## 为什么需要它

本项目已经因为"相机索引漂移"栽过两次：策略腕部通道拿到过笔记本摄像头的画面
（历史 `fixed=2 / handeye=0`），以及 MSMF 索引卡在幽灵槽位（`OpenCVCamera(0) read failed`）。
**2026-10-17 重插 + 重启后又漂移为 `fixed=1 / handeye=2`** —— 映射现在只在
`src/camera_slots.py` 一处定义，本工具与运行时共用它。
索引错了 **`--verify` 查不出来**（它只比对关节位置，不看画面内容），等训练时才发现就晚了。

本脚本**只读**：不连机械臂、不出力矩、不做归位，只开相机、抓一帧、存图，供肉眼确认身份。

用法::

    python src/run_camera_check.py                 # 探 MSMF 0..4（与 lerobot 同后端）
    python src/run_camera_check.py --backends both # 加上 DSHOW 对照
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCHEMA_VERSION = "week38_camera_check_v1"

#: 采集/推理**必须**保持一致的槽位 —— 唯一事实来源是 `camera_slots.CAMERA_SLOTS`。
#: 本工具只负责"按该映射核图"，**不在这里另存一份**（曾在 5 处各抄一份，漂移后同时失效）。
from camera_slots import CAMERA_SLOTS  # noqa: E402
#: 探测范围：0..4 足够覆盖本机（3 台相机曾分布在 0/1/2）。
PROBE_INDICES = (0, 1, 2, 3, 4)
DEFAULT_OUT = "runs/week38/camera_check"


def render_probe_row(index: int, backend: str, result: dict[str, Any]) -> str:
    """一行探测结果（纯函数，便于单测）。"""
    status = result.get("status")
    if status == "ok":
        return (f"  索引 {index} [{backend}] 可用 {result['width']}x{result['height']}"
                f"  存图 {result['path']}")
    return f"  索引 {index} [{backend}] {status}：{result.get('detail', '')}"


def render_slot_verdict(slots: dict[str, int], usable: Sequence[int]) -> list[str]:
    """按配置槽位给出结论（纯函数）——一眼看出"录制起不来"是不是相机的问题。"""
    lines = ["配置要求的槽位："]
    for name, index in slots.items():
        ok = index in set(usable)
        mark = "✅ 可用" if ok else "❌ 不可用 ← 录制的阻塞点"
        lines.append(f"  {name:<8}= 索引 {index}   {mark}")
    return lines


def probe(index: int, backend_name: str, backend: int, out_dir: Path) -> dict[str, Any]:
    """开一台相机、抓一帧、存图；任何失败都记成结果而不是抛异常。"""
    import cv2

    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            return {"status": "打开失败"}
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"status": "打开成功但读帧失败",
                    "detail": "设备未连接/被占用（MSMF 常见于异常退出后残留的坏槽位）"}
        path = out_dir / f"{backend_name}_{index}.png"
        cv2.imwrite(str(path), frame)
        height, width = frame.shape[:2]
        return {"status": "ok", "width": width, "height": height, "path": path,
                "brightness": float(frame.mean())}
    except Exception as exc:  # 探测工具，失败也要继续探下一个索引
        return {"status": "异常", "detail": str(exc)}
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="相机索引体检（只读，不动机械臂）")
    parser.add_argument("--backends", choices=("msmf", "both"), default="msmf",
                        help="msmf=只探 lerobot 用的后端；both=另加 DSHOW 对照")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--stamp", default="", help="输出子目录名（默认时间戳）")
    args = parser.parse_args()

    import cv2

    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out).expanduser() / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    backends = [("MSMF", cv2.CAP_MSMF)]
    if args.backends == "both":
        backends.append(("DSHOW", cv2.CAP_DSHOW))

    usable: list[int] = []
    lines = [f"# 相机索引体检（schema {SCHEMA_VERSION}）", ""]
    for backend_name, backend in backends:
        lines.append(f"== 后端 {backend_name} ==")
        for index in PROBE_INDICES:
            result = probe(index, backend_name, backend, out_dir)
            lines.append(render_probe_row(index, backend_name, result))
            if backend_name == "MSMF" and result.get("status") == "ok":
                usable.append(index)
        lines.append("")

    lines.extend(render_slot_verdict(CAMERA_SLOTS, usable))
    lines += [
        "",
        "请打开上面的图片核对身份（**索引对了但相机认错了，一样会毁数据**）：",
        "  handeye → 应能看到夹爪的白色手指指着桌面（腕上近景）",
        "  fixed   → 应是俯拍整桌",
        "若某个槽位不可用：先把那台相机的 USB 换一个口重插；仍不行就重启系统。",
        "**不要因为槽位不可用就改索引** —— 改索引会让新旧数据集的同一通道来自不同物理相机。",
        f"\n输出目录：{out_dir.resolve()}",
    ]
    report = "\n".join(lines)
    print(report)
    (out_dir / "camera_check.txt").write_text(report + "\n", encoding="utf-8")

    blocked = [name for name, index in CAMERA_SLOTS.items() if index not in set(usable)]
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
