#!/usr/bin/env python
"""真值标注（第 38 周 P0 / 子任务 0.3 选型表前置）。

为一批真机帧标注目标物体的**像素级包围盒**，产出 `gt.json`，
`run_perception_locate_eval.py --gt <gt.json>` 可直接消费。

设计要点：**真值必须独立于被评模型**，否则误差会被系统性低估（循环论证）。
因此本工具只做"人工确认/微调"——候选框（通常来自多源中位数）仅作初值，
鼠标拖拽可随时覆盖；文件里保存的是人工确认后的结果。

交互按键：
    鼠标拖拽  重画当前框
    Enter/空格 采用当前框（未拖拽则采用候选）
    r         恢复为候选框
    s         跳过本帧
    q         保存已完成部分并退出

用法::

    # 无 GUI 自检（只打印候选与进度，不弹窗、不写文件）
    python src/run_perception_gt_label.py --frames <dir> --report <dir>/eval/report.json --dry-run

    # 交互标注（候选自动从评测结果生成，逐轴取中位数）
    python src/run_perception_gt_label.py --frames <dir> --report <dir>/eval/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 目标关键词别名（中英通用，不区分大小写）
_TARGET_ALIASES = {"screwdriver": ("screwdriver", "螺丝刀"), "螺丝刀": ("screwdriver", "螺丝刀")}


def iter_frames(frames_dir: Path) -> list[Path]:
    """帧目录下的 PNG。

    只取顶层（``glob`` 不递归），因此 ``<frames>/eval`` 里的叠加图不会被当成帧。
    """
    return sorted(p for p in frames_dir.glob("*.png"))


def parse_bbox(raw: object) -> list[float] | None:
    """解析 bbox 字段；非 4 个有限数值、或退化（x2<=x1 / y2<=y1）时返回 None。"""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return values


def median_bbox(boxes: list[object]) -> list[float] | None:
    """逐轴取中位数——比均值更能抑制单个来源的离群框。"""
    valid = [b for b in (parse_bbox(raw) for raw in boxes) if b is not None]
    if not valid:
        return None
    return [statistics.median(b[i] for b in valid) for i in range(4)]


def match_target(name: str, target: str) -> bool:
    """目标名匹配：中英关键词任一命中即可（``screwdriver`` ↔ ``螺丝刀``）。"""
    aliases = _TARGET_ALIASES.get(target.strip().lower(), (target.strip().lower(),))
    lowered = str(name).lower()
    return any(alias and alias in lowered for alias in aliases)


def candidates_from_report(report_path: Path, target: str) -> dict[str, list[float]]:
    """从评测 `report.json` 生成候选真值。

    每帧取各源中命中 ``target`` 的**面积最大**的框（一帧可能有多个候选），再逐轴取中位数。
    该候选只是"初值"，最终真值以人工确认为准。
    """
    data = json.loads(report_path.read_text(encoding="utf-8"))
    per_frame: dict[str, list[list[float]]] = {}
    for source_data in data.get("sources", {}).values():
        for entry in source_data.get("frames", []):
            hits = [d for d in entry.get("detections", []) if match_target(d.get("name", ""), target)]
            if not hits:
                continue
            best = max(hits, key=lambda d: (d["bbox_px"][2] - d["bbox_px"][0])
                       * (d["bbox_px"][3] - d["bbox_px"][1]))
            per_frame.setdefault(entry["file"], []).append(best["bbox_px"])
    out: dict[str, list[float]] = {}
    for name, boxes in per_frame.items():
        box = median_bbox(boxes)
        if box is not None:
            out[name] = box
    return out


def load_boxes(path: Path | None) -> dict[str, list[float]]:
    """读取 ``{文件: {bbox_px: [...]}}`` 或 ``{文件: [...]}`` 两种写法。"""
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", data)
    if not isinstance(frames, dict):
        return {}
    out: dict[str, list[float]] = {}
    for name, value in frames.items():
        raw = value.get("bbox_px") if isinstance(value, dict) else value
        box = parse_bbox(raw)
        if box is not None:
            out[str(name)] = box
    return out


def save_ground_truth(path: Path, target: str, boxes: dict[str, list[float]]) -> None:
    """写 ``gt.json``（与 `run_perception_locate_eval.py --gt` 的格式一致）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target,
        "frames": {name: {"bbox_px": [round(v, 1) for v in box]}
                   for name, box in sorted(boxes.items())},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _label_one(cv2, path: Path, current: list[float] | None,
               candidate: list[float] | None) -> list[float] | str:
    """单帧交互标注；返回 bbox / ``"skip"`` / ``"quit"``。"""
    image = cv2.imread(str(path))
    if image is None:
        return "skip"
    height, width = image.shape[:2]
    state: dict = {"box": list(current) if current else None, "start": None, "dragging": False}

    def draw(canvas, box, color, label):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (max(0, x1), max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["dragging"] = True
            state["box"] = [float(x), float(y), float(x), float(y)]
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            sx, sy = state["start"]
            state["box"] = [float(min(sx, x)), float(min(sy, y)),
                            float(max(sx, x)), float(max(sy, y))]
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
            box = state["box"]
            # 只点一下没拖动 → 退化框，回退到候选
            if box and ((box[2] - box[0]) < 2 or (box[3] - box[1]) < 2):
                state["box"] = list(candidate) if candidate else None

    window = f"GT 标注 {path.name} ({width}x{height})"
    hint = "拖拽=重画 | Enter/空格=采用 | r=候选 | s=跳过 | q=保存退出"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            canvas = image.copy()
            if candidate:
                draw(canvas, candidate, (0, 200, 0), "候选")
            if state["box"]:
                draw(canvas, state["box"], (0, 0, 255), "当前")
            cv2.putText(canvas, hint, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):  # Enter / Space
                return list(state["box"]) if state["box"] else "skip"
            if key == ord("r"):
                state["box"] = list(candidate) if candidate else None
            elif key == ord("s"):
                return "skip"
            elif key == ord("q"):
                return "quit"
    finally:
        cv2.destroyWindow(window)


def main() -> None:
    parser = argparse.ArgumentParser(description="真值标注（0.3 选型表前置）")
    parser.add_argument("--frames", required=True, help="帧目录（含 *.png）")
    parser.add_argument("--gt", default="", help="输出 gt.json，默认 <frames>/gt.json")
    parser.add_argument("--candidates", default="", help="候选真值 JSON（可选）")
    parser.add_argument("--report", default="", help="从评测 report.json 自动生成候选（可选）")
    parser.add_argument("--target", default="screwdriver", help="目标关键词（中英均可）")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不弹窗、不写文件")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    frames = iter_frames(frames_dir)
    if not frames:
        raise SystemExit(f"{frames_dir} 下没有 PNG 帧")
    gt_path = Path(args.gt) if args.gt else frames_dir / "gt.json"

    candidates = load_boxes(Path(args.candidates) if args.candidates else None)
    if args.report:
        candidates.update(candidates_from_report(Path(args.report), args.target))
    existing = load_boxes(gt_path)

    print(f"[标注] 帧 {len(frames)} 张 | 目标 {args.target} | 输出 {gt_path}")
    print(f"[标注] 候选 {len(candidates)} 帧 | 已有真值 {len(existing)} 帧")
    for path in frames:
        box = existing.get(path.name) or candidates.get(path.name)
        origin = "已有" if path.name in existing else ("候选" if path.name in candidates else "无  ")
        coords = ",".join(f"{v:.0f}" for v in box) if box else "-"
        print(f"  {path.name:18s} {origin}  [{coords}]")

    if args.dry_run:
        print("[dry-run] 未弹窗、未写文件")
        return

    boxes = dict(existing)
    quit_early = False
    for path in frames:
        current = boxes.get(path.name) or candidates.get(path.name)
        outcome = _label_one(cv2_import(), path, current, candidates.get(path.name))
        if outcome == "quit":
            quit_early = True
            break
        if outcome == "skip":
            continue
        boxes[path.name] = outcome

    save_ground_truth(gt_path, args.target, boxes)
    suffix = "（中途退出，已保存已完成部分）" if quit_early else ""
    print(f"[标注] 已写 {len(boxes)} 帧真值 → {gt_path}{suffix}")


def cv2_import():
    """延迟导入 cv2：纯逻辑（候选生成/读写）不需要 OpenCV，测试环境可无 cv2。"""
    import cv2

    return cv2


if __name__ == "__main__":
    main()
