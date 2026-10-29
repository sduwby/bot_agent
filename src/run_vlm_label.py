#!/usr/bin/env python
"""VLM 自动标注 + 人工抽检页生成 CLI（第 38 周 P1 / 子任务 2.7 + 2.10 前置）。

流程：

    LeRobot 数据集 ──抽帧(缓存)──► VlmLabelingBackend(Qwen3-VL) ──► labels.jsonl
                                                                  ├─ episode_metadata.jsonl（旁挂元数据）
                                                                  └─ review/review.html（人工抽检页）

用法::

    # 离线验证整条链路（不调 API，用桩后端）
    python src/run_vlm_label.py --dataset-root <demo_v23> --out runs/week38/labels/demo_v23 --dry-run

    # 真跑（源名与端点/密钥/模型取自 .env，见 run_perception_locate_eval 的注册表）
    python src/run_vlm_label.py --dataset-root <demo_v23> --out runs/week38/labels/demo_v23

标注**逐条落盘**并支持断点续跑：中途失败重跑会自动跳过已标注的 episode。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from env_config import load_local_env  # noqa: E402
from lerobot_label_source import LerobotEpisodeSource, load_manifest  # noqa: E402
from review_sheet import ReviewItem, select_review_refs, write_review_sheet  # noqa: E402
from run_perception_locate_eval import resolve_source, source_env_prefix  # noqa: E402
from vlm_labeler import (  # noqa: E402
    IntentTaxonomy,
    LabelRecord,
    VlmLabelingBackend,
    define_intent_taxonomy,
    label_dataset,
    read_label_records,
    write_lerobot_metadata,
)

#: 默认标注源（第 38 周 P0/0.3 选型结论）。
DEFAULT_SOURCE = "qwen3-vl-plus"
#: 默认相机：俯拍看整场景 + 夹爪特写看是否真的夹住。
DEFAULT_CAMERAS = ("observation.images.fixed", "observation.images.handeye")
DEFAULT_N_FRAMES = 4

LABELS_NAME = "labels.jsonl"
MANIFEST_NAME = "frames_manifest.json"


class _StubChat:
    """dry-run 桩：不调 API，固定返回一条合法标注（仍走真实的解析/校验路径）。"""

    reply = json.dumps(
        {"task_text": "抓起螺丝刀并抬起", "subtask": "lift", "confidence": 0.5,
         "note": "dry-run 桩输出"},
        ensure_ascii=False,
    )

    def chat(self, system: str, user: str, images: list | None = None) -> str:
        return self.reply


def build_backend(name: str, *, dry_run: bool) -> VlmLabelingBackend:
    """按源名构建标注后端（dry_run 时用桩 chat 后端，不调 API）。"""
    if dry_run:
        return VlmLabelingBackend(_StubChat())

    cfg = resolve_source(name)
    if not cfg["api_base"] or not cfg["api_key"]:
        raise SystemExit(
            f"标注源 {name} 缺少端点/密钥：请设置 {source_env_prefix(name)}_API_KEY"
            "（未配置则回退到全局 LLM_API_BASE / LLM_API_KEY）"
        )
    from llm_planner import OpenAIBackend

    chat = OpenAIBackend(api_base=cfg["api_base"], api_key=cfg["api_key"], model=cfg["model"])
    print(f"[标注] 源 {name}  model={cfg['model']}  endpoint={cfg['api_base']}")
    return VlmLabelingBackend(chat)


def build_review_sheet(
    out_dir: Path,
    records: list[LabelRecord],
    manifest: dict[str, dict],
    taxonomy: IntentTaxonomy,
    *,
    ratio: float,
    count: int | None,
    seed: int,
    dataset_label: str,
    prefill: dict[str, LabelRecord] | None = None,
) -> Path:
    """按抽检比例生成人工复核页（不含模型答案）。

    `prefill` 给定时，对同一 episode 预选**人工自己之前填过**的答案（用于"只重问某一个
    字段"的场景，例如判据口径改了要重问 success）。**不能**用它预填模型答案——那属锚定。
    """
    refs = [record.episode_ref for record in records]
    picked = select_review_refs(refs, ratio=ratio, count=count, seed=seed)
    prior = prefill or {}
    items = []
    for ref in picked:
        frames = tuple(manifest.get(ref, {}).get("frames") or ())
        if not frames:
            continue
        previous = prior.get(ref)
        defaults: dict[str, str] = {}
        if previous is not None:
            if previous.intent:
                defaults["intent"] = previous.intent
            if previous.subtask:
                defaults["subtask"] = previous.subtask
            if previous.success is not None:
                defaults["success"] = "true" if previous.success else "false"
        items.append(
            ReviewItem(
                episode_ref=ref,
                frames=frames,
                task=str(manifest.get(ref, {}).get("task") or ""),
                defaults=defaults,
            )
        )
    if not items:
        raise SystemExit("没有带帧文件的抽检样本：请确认抽帧产物与清单存在")
    return write_review_sheet(
        out_dir / "review" / "review.html",
        items,
        taxonomy=taxonomy,
        title=f"P1 标注人工抽检 · {dataset_label}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM 自动标注 + 人工抽检页（P1）")
    parser.add_argument("--dataset-root", required=True, help="LeRobot 数据集根目录")
    parser.add_argument("--out", required=True, help="输出目录（labels.jsonl / frames / review）")
    parser.add_argument("--frames-dir", default="",
                        help="抽帧缓存目录，默认 <out>/frames；A/B 重跑可指向旧目录复用缓存")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help=f"标注源名（默认 {DEFAULT_SOURCE}，配置见 VLM_<源>_*）")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS),
                        help="逗号分隔的相机特征名（默认 fixed+handeye）")
    parser.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES,
                        help="每个相机抽帧数（默认 4）")
    parser.add_argument("--limit", type=int, default=None, help="本次最多新增标注多少集")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有 labels.jsonl，重头标")
    parser.add_argument("--dry-run", action="store_true", help="用桩后端验证链路，不调 API")
    parser.add_argument("--no-review", action="store_true", help="不生成人工抽检页")
    parser.add_argument("--review-ratio", type=float, default=0.2, help="人工抽检比例（默认 0.2）")
    parser.add_argument("--review-count", type=int, default=None, help="人工抽检条数（覆盖比例）")
    parser.add_argument("--seed", type=int, default=38, help="抽检随机种子")
    parser.add_argument("--prefill", default="",
                        help="用旧的 human.jsonl 预选复核页答案（用于只重问某个字段）")
    args = parser.parse_args()

    taxonomy = define_intent_taxonomy()
    load_local_env(Path(__file__).resolve().parent.parent / ".env")  # 不覆盖 shell 已有配置
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = tuple(item.strip() for item in args.cameras.split(",") if item.strip())
    frames_dir = Path(args.frames_dir).expanduser() if args.frames_dir else out_dir / "frames"

    source = LerobotEpisodeSource(
        args.dataset_root, frames_dir, cameras=cameras, n_frames=args.n_frames
    )
    print(f"[标注] 数据集 {source.meta.root.name}｜{len(source.meta.episodes)} 集｜"
          f"相机 {cameras}｜每相机 {args.n_frames} 帧")

    backend = build_backend(args.source, dry_run=args.dry_run)
    records = label_dataset(
        backend,
        source,
        taxonomy,
        out_dir / LABELS_NAME,
        limit=args.limit,
        resume=not args.no_resume,
    )
    manifest_path = source.write_manifest(out_dir / MANIFEST_NAME)
    manifest = load_manifest(manifest_path)
    write_lerobot_metadata(
        records,
        out_dir / "episode_metadata.jsonl",
        task_text_by_episode={ref: str(item.get("task") or "") for ref, item in manifest.items()},
    )

    resolved = sum(1 for record in records if record.intent)
    print(f"\n[标注] 共 {len(records)} 条（intent 命中 {resolved}）→ {out_dir / LABELS_NAME}")
    print(f"[标注] 旁挂元数据 → {out_dir / 'episode_metadata.jsonl'}")
    print(f"[标注] 抽帧 {sum(len(v.get('frames') or ()) for v in manifest.values())} 张 → {frames_dir}")

    if not args.no_review:
        prefill: dict[str, LabelRecord] = {}
        if args.prefill:
            prefill = {item.episode_ref: item for item in read_label_records(Path(args.prefill))}
            print(f"[复核] 从 {args.prefill} 预填 {len(prefill)} 条历史人工答案")
        sheet = build_review_sheet(
            out_dir, records, manifest, taxonomy,
            ratio=args.review_ratio, count=args.review_count, seed=args.seed,
            dataset_label=source.meta.root.name, prefill=prefill,
        )
        n_review = selected_count(records, args.review_ratio, args.review_count, args.seed)
        print(f"[复核] 抽检 {n_review} 集 → {sheet}")
        print("       请人工填写后导出 human.jsonl，再运行 run_label_audit.py")


def selected_count(
    records: list[LabelRecord], ratio: float, count: int | None, seed: int
) -> int:
    """抽检页里的样本数（与 `build_review_sheet` 的筛选保持一致）。"""
    return len(select_review_refs([r.episode_ref for r in records], ratio=ratio,
                                  count=count, seed=seed))


if __name__ == "__main__":
    main()
