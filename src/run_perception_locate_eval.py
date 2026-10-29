#!/usr/bin/env python
"""2D 定位源评测（第 38 周 P0 / 子任务 0.3）。

对同一批真机帧跑多个定位源，产出：

- 每个 (源, 帧) 一张**叠加图**（bbox 画在帧上），便于人工肉眼核对；
- `report.json`：各源的检测结果；若提供 `--gt`，另算中心/尺寸误差。

真值格式（`--gt`，只需给"目标物体"的框）：

```json
{"target": "screwdriver",
 "frames": {"fixed_00.png": {"bbox_px": [200, 150, 280, 250]}}}
```

评测口径（名称无关，避免命名差异干扰）：对每帧取**与该帧所有检测中中心误差最小者**
作为该源的成绩，并记录是否命中（中心误差 ≤ `--hit-px`，默认 10 px）。

用法::

    # 离线验证链路（FakeObjectLocator，不调 API）
    python src/run_perception_locate_eval.py --frames <dir> --dry-run

    # 真跑（源名与端点/密钥/坐标约定取自 .env 或环境变量）
    python src/run_perception_locate_eval.py --frames <dir> --models gui-plus,qwen3-vl-plus

    # 有真值时算误差
    python src/run_perception_locate_eval.py --frames <dir> --gt <dir>/gt.json

每个源按 ``VLM_<源名大写非字母数字转下划线>_API_BASE/_API_KEY/_MODEL/_COORD_SPACE``
配置，未设置则回退全局 ``LLM_API_*``、源名本身、内置默认表，最后回退 ``auto``。
**新增"输出归一化坐标"的模型只需在 .env 加几行，无需改代码。**
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import hypot
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 定位源注册（env 驱动，避免把"模型名 → 坐标约定/端点"钉死在代码里）
#
# 源名即 ``--models`` 里的名字；其配置来自环境变量
#   VLM_<源名大写、非字母数字转下划线>_API_BASE / _API_KEY / _MODEL / _COORD_SPACE
# 未设置的项按序回退：全局 LLM_API_* → 源名本身 → 内置默认表 → ``auto``。
# 这样新增一个"输出归一化坐标"的模型只需在 .env 里加几行，无需改代码。
# ---------------------------------------------------------------------------

# 内置默认坐标约定（仅在未配置 VLM_<源>_COORD_SPACE 时生效；未知模型回退 auto）
DEFAULT_COORD_SPACE = {"gui-plus": "pixel", "qwen3-vl-plus": "normalized_1000"}
_VALID_COORD_SPACES = ("auto", "pixel", "normalized_1000", "normalized_1")

# 叠加图画框颜色（BGR）；未知源按名字稳定散列取色，保证"同源同色"
_KNOWN_BGR = {"gui-plus": (0, 180, 0), "qwen3-vl-plus": (220, 120, 0),
              "deepseek-flash": (0, 60, 220), "fake": (160, 160, 160)}
_PALETTE = [(0, 180, 0), (220, 120, 0), (0, 60, 220), (160, 160, 160),
            (0, 200, 200), (200, 0, 200), (60, 200, 60), (180, 80, 0)]

DEFAULT_HIT_PX = 10.0


def bbox_error(det_bbox, gt_bbox) -> tuple[float, list[float]]:
    """返回 ``(中心欧氏误差 px, [宽差, 高差] px)``。

    尺寸差 = 检测尺寸 − 真值尺寸（**正数表示框偏大**，即模型外扩）。
    """
    dx1, dy1, dx2, dy2 = (float(v) for v in det_bbox)
    gx1, gy1, gx2, gy2 = (float(v) for v in gt_bbox)
    center_error = hypot((dx1 + dx2) / 2 - (gx1 + gx2) / 2, (dy1 + dy2) / 2 - (gy1 + gy2) / 2)
    size_error = [(dx2 - dx1) - (gx2 - gx1), (dy2 - dy1) - (gy2 - gy1)]
    return center_error, size_error


def source_bgr(name: str) -> tuple[int, int, int]:
    """叠加图颜色：已知源用固定色，未知源按名字稳定取色。"""
    if name in _KNOWN_BGR:
        return _KNOWN_BGR[name]
    return _PALETTE[sum(name.encode("utf-8")) % len(_PALETTE)]


def source_env_prefix(name: str) -> str:
    """``gui-plus`` → ``VLM_GUI_PLUS``（非字母数字转下划线）。"""
    return "VLM_" + "".join(ch if ch.isalnum() else "_" for ch in name).upper()


def resolve_source(name: str) -> dict:
    """解析一个定位源的端点/密钥/模型名/坐标约定（env 优先，逐级回退）。"""
    prefix = source_env_prefix(name)
    space = os.environ.get(f"{prefix}_COORD_SPACE") or DEFAULT_COORD_SPACE.get(name, "auto")
    if space not in _VALID_COORD_SPACES:
        raise SystemExit(f"{prefix}_COORD_SPACE 非法：{space!r}，应为 {_VALID_COORD_SPACES}")
    return {
        "name": name,
        "api_base": os.environ.get(f"{prefix}_API_BASE") or os.environ.get("LLM_API_BASE", ""),
        "api_key": os.environ.get(f"{prefix}_API_KEY") or os.environ.get("LLM_API_KEY", ""),
        "model": os.environ.get(f"{prefix}_MODEL") or name,
        "coord_space": space,
    }


def default_models() -> list[str]:
    """``--models`` 缺省值：优先取 ``VLM_SOURCES``。"""
    models = [m.strip() for m in os.environ.get("VLM_SOURCES", "").split(",") if m.strip()]
    return models or ["gui-plus"]


def load_frames(frames_dir: Path) -> list[Path]:
    paths = sorted(p for p in frames_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"{frames_dir} 下没有 PNG 帧")
    return paths


def load_ground_truth(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"真值文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", data)
    if not isinstance(frames, dict):
        raise SystemExit("真值格式错误：frames 必须是 {文件名: {bbox_px: [...]}}")
    return frames


def build_backends(models: list[str], dry_run: bool) -> dict[str, tuple[object, str]]:
    """为每个源建定位后端，返回 ``{源名: (定位器或(backend, locate), 坐标约定)}``。"""
    if dry_run:
        from real_perception import Detection, FakeObjectLocator

        locator = FakeObjectLocator([Detection(name="fake", bbox_px=(100.0, 100.0, 200.0, 200.0))])
        return {"fake": (locator, "auto")}

    from llm_planner import OpenAIBackend
    from real_perception import locate_objects

    backends: dict[str, tuple[object, str]] = {}
    for name in models:
        cfg = resolve_source(name)
        if not cfg["api_base"] or not cfg["api_key"]:
            raise SystemExit(
                f"定位源 {name} 缺少端点/密钥：请设置 {source_env_prefix(name)}_API_KEY"
                f"（未配置则回退到全局 LLM_API_BASE / LLM_API_KEY）。"
                f"当前端点 {cfg['api_base'] or '(空)'}")
        backend = OpenAIBackend(api_base=cfg["api_base"], api_key=cfg["api_key"],
                                model=cfg["model"])
        backends[name] = ((backend, locate_objects), cfg["coord_space"])
    return backends


def main() -> None:
    parser = argparse.ArgumentParser(description="2D 定位源评测（P0/0.3）")
    parser.add_argument("--frames", required=True, help="帧目录（含 *.png）")
    parser.add_argument("--out", default="", help="输出目录，默认写到 <frames>/eval")
    parser.add_argument("--models", default=",".join(default_models()),
                        help="逗号分隔的定位源名（配置见 VLM_<源>_* 环境变量）")
    parser.add_argument("--gt", default="", help="真值 JSON 路径（可选）")
    parser.add_argument("--hit-px", type=float, default=DEFAULT_HIT_PX,
                        help="命中阈值：中心误差 ≤ 该值算命中")
    parser.add_argument("--dry-run", action="store_true", help="用 FakeObjectLocator 验证链路")
    args = parser.parse_args()

    import cv2

    frames_dir = Path(args.frames)
    out_dir = Path(args.out) if args.out else frames_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames(frames_dir)
    ground_truth = load_ground_truth(Path(args.gt) if args.gt else None)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    locators = build_backends(models, args.dry_run)

    print(f"[评测] 帧 {len(frames)} 张 | 源 {list(locators)} | 输出 {out_dir}")
    report: dict[str, dict] = {}

    for source, (locator, space) in locators.items():
        entries = []
        for path in frames:
            bgr = cv2.imread(str(path))
            if bgr is None:
                print(f"  [skip] {source} {path.name}: 读取失败")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]

            if args.dry_run:
                result = locator.locate([rgb])
            else:
                from llm_planner import image_to_data_url

                backend, locate = locator
                url = image_to_data_url(rgb)
                result = locate(backend, [url], frame_size=(width, height), coord_space=space)

            record = {"file": path.name, "source": result.source,
                      "coord_space": result.coord_space, "skipped": result.skipped,
                      "clamped": result.clamped, "note": result.note,
                      "detections": [{"name": d.name, "bbox_px": list(d.bbox_px),
                                      "center_px": list(d.center_px),
                                      "size_px": list(d.size_px),
                                      "confidence": d.confidence}
                                     for d in result.detections]}

            gt = ground_truth.get(path.name)
            if gt and result.detections:
                scored = [(bbox_error(d.bbox_px, gt["bbox_px"]), d) for d in result.detections]
                (best_error, size_error), best_det = min(scored, key=lambda item: item[0][0])
                record["gt_bbox_px"] = list(gt["bbox_px"])
                record["best_center_error_px"] = round(best_error, 2)
                record["best_size_error_px"] = [round(v, 2) for v in size_error]
                record["best_detection"] = best_det.name
                record["hit"] = best_error <= args.hit_px

            entries.append(record)

            overlay = bgr.copy()
            (x1, y1, x2, y2) = (int(v) for v in gt["bbox_px"]) if gt else (0, 0, 0, 0)
            if gt:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(overlay, "GT", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            color = source_bgr(source)
            for index, det in enumerate(result.detections):
                bx1, by1, bx2, by2 = (int(v) for v in det.bbox_px)
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(overlay, f"#{index} {det.confidence:.2f}", (bx1, max(12, by1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            safe_source = source.replace("/", "_")
            cv2.imwrite(str(out_dir / f"{path.stem}__{safe_source}.png"), overlay)

            status = ""
            if "best_center_error_px" in record:
                dw, dh = record["best_size_error_px"]
                status = f" | 中心误差 {record['best_center_error_px']}px " \
                         f"尺寸差 {dw:+.1f}x{dh:+.1f}px " \
                         f"{'命中' if record['hit'] else '未命中'}"
            print(f"  {source:28s} {path.name:18s} 检测 {len(result.detections)} "
                  f"({result.coord_space}){status}")

        report[source] = {"coord_space_default": space, "frames": entries}

    totals = {}
    for source, data in report.items():
        scored = [e for e in data["frames"] if "best_center_error_px" in e]
        if scored:
            mean_error = sum(e["best_center_error_px"] for e in scored) / len(scored)
            mean_dw = sum(e["best_size_error_px"][0] for e in scored) / len(scored)
            mean_dh = sum(e["best_size_error_px"][1] for e in scored) / len(scored)
            hits = sum(1 for e in scored if e["hit"])
            totals[source] = {"scored_frames": len(scored),
                              "mean_center_error_px": round(mean_error, 2),
                              "mean_size_error_px": [round(mean_dw, 2), round(mean_dh, 2)],
                              "hits": hits, "hit_rate": round(hits / len(scored), 3)}
        else:
            detected = sum(1 for e in data["frames"] if e["detections"])
            totals[source] = {"frames": len(data["frames"]), "frames_with_detection": detected}

    (out_dir / "report.json").write_text(
        json.dumps({"frames_dir": str(frames_dir), "summary": totals, "sources": report},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 汇总 ===")
    for source, summary in totals.items():
        print(f"  {source}: {summary}")
    print(f"叠加图与 report.json → {out_dir}")


if __name__ == "__main__":
    main()
