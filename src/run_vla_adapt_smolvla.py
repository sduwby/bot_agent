#!/usr/bin/env python
"""第 38 周 P2 4.5：SmolVLA 适配对照的**训练侧**（真正按策略冻结参数）。

## 为什么需要这个入口

`run_vla_adapt.py` 的默认训练路径调用 `lerobot.scripts.lerobot_train` 的 **CLI**，
而该 CLI 没有"按模块冻结"参数 → `head` 与 `full` 会训练成同一个东西，
对照表（A5）随之失效（`vla_adapt._default_train_fn` 的 docstring 已写明这条边界）。

本入口把解冻 seam 放到**进程内**：patch `lerobot_train.make_policy`，
拿到 policy 对象后立刻 `apply_adaptation` 改写 `requires_grad`，
再交给官方训练循环（`optimizers.py` 会跳过 `requires_grad=False` 的参数）。
训练命令的其余口径与 `run_vla_train.py` 完全一致（同一套常量，避免漂移）。

## 数据量自变量

用 `--dataset.episodes` 选子集，而非复制数据集：按总集数**等距抽取**，
保证 6 个任务的集数大致均衡。30 / 60 两档分别由 `select_episodes` 生成。

## 用法

    # 只打印将执行的训练命令与解冻口径（不训练）
    python src/run_vla_adapt_smolvla.py --strategy head --n-episodes 30 --dry-run

    # 云上实跑（输出到 <output-dir>/<job_name>/）
    python src/run_vla_adapt_smolvla.py --strategy full --n-episodes 60 \
        --dataset-root ~/.cache/huggingface/lerobot/wangbangyan/demo_vla_all \
        --output-dir outputs/vla_adapt --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from adapt_eval_offline import parse_episode_list  # noqa: E402
from vla_adapt import AdaptConfig, apply_adaptation, build_adapt_config  # noqa: E402

DEFAULT_DATASET_REPO = "wangbangyan/demo_vla_all"
DEFAULT_STEPS = 2000
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 8
DEFAULT_SEED = 38
DEFAULT_TOTAL_EPISODES = 120
#: 单元完成标记（幂等重跑依据）：训练成功或被显式跳过时才写。
DONE_MARKER = "DONE"


def should_skip(job_dir: Path, force: bool) -> bool:
    """已完成单元默认跳过（`--force` 可强制重跑）。

    完成判定只看 `DONE` 标记，而不是看 `train/` 是否存在：中断留下的半成品目录
    **必须重跑**（否则会把没训完的 checkpoint 当成结果）。
    """
    return (Path(job_dir) / DONE_MARKER).is_file() and not force


def _mark_done(job_dir: Path, note: str) -> None:
    (Path(job_dir) / DONE_MARKER).write_text(note + "\n", encoding="utf-8")


def select_episodes(total: int, n: int) -> list[int]:
    """按等距抽取选 ``n`` 个 episode（确定性、跨任务大致均衡、无重复）。

    ``n >= total`` 时返回全部；``n <= 0`` 报错。等距而非前缀，是为了避免
    "只取前 N 集 = 只取了前几个任务"。
    """
    if n <= 0:
        raise ValueError(f"n 必须是正整数，实际 {n}")
    if n >= total:
        return list(range(total))
    picked: list[int] = []
    for i in range(n):
        idx = round(i * total / n)
        idx = min(max(idx, 0), total - 1)
        if idx not in picked:
            picked.append(idx)
    return picked


def build_train_argv(
    *,
    config: AdaptConfig,
    dataset_repo: str,
    dataset_root: str,
    output_dir: str,
    device: str,
    episodes: Sequence[int],
    batch_size: int,
    num_workers: int,
    save_freq: int = 500,
    job_name: str | None = None,
) -> list[str]:
    """组装 `lerobot_train` 的参数（与 `run_vla_train.py` 同一口径）。

    `save_freq` 默认 500 而不是总步数：云端曾发生**机器重启**把 2000 步的单元整个丢掉
    （2026-10-18 02:36），密存盘让"重启/断连"最多损失一个窗口。
    """
    from run_vla_train import DEFAULT_POLICY_PATH, DEFAULT_RENAME_MAP, TRAIN_MODULE

    job = job_name or f"vla_adapt_{config.strategy}_ep{config.n_episodes}"
    argv = [
        f"--policy.path={DEFAULT_POLICY_PATH}",
        f"--dataset.repo_id={dataset_repo}",
        f"--output_dir={output_dir}",
        f"--job_name={job}",
        f"--steps={config.steps}",
        f"--batch_size={batch_size}",
        f"--num_workers={num_workers}",
        f"--save_freq={save_freq}",
        "--eval_steps=0",
        f"--seed={config.seed}",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--rename_map={DEFAULT_RENAME_MAP}",
        f"--dataset.episodes={json.dumps(list(episodes), separators=(',', ':'))}",
    ]
    if dataset_root:
        argv.append(f"--dataset.root={dataset_root}")
    return argv


def make_frozen_make_policy(
    original: Callable[..., Any], config: AdaptConfig, stats_path: Path
) -> Callable[..., Any]:
    """包装 `lerobot_train.make_policy`：构造后立即按策略冻结，并落盘实测解冻统计。"""

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        policy = original(*args, **kwargs)
        stats = apply_adaptation(policy, config)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "total_params": stats["total_params"],
                    "trainable_params": stats["trainable_params"],
                    "trainable_ratio": stats["trainable_ratio"],
                    "trainable_modules": list(stats["trainable_modules"]),
                    "frozen_params": stats["frozen_params"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[adapt] 冻结落地：strategy={config.strategy} "
            f"trainable={stats['trainable_ratio']:.4%} "
            f"({stats['trainable_params']:,}/{stats['total_params']:,})",
            flush=True,
        )
        return policy

    return _wrapped


def run_unit(args: argparse.Namespace) -> Path:
    """跑一个适配单元；返回产物目录。``zero_shot`` 不训练，只写配置。"""
    config = build_adapt_config(args.strategy, args.n_episodes, args.steps, args.seed)
    suffix = args.job_tag or f"ep{config.n_episodes}"
    job_dir = Path(args.output_dir) / f"{config.strategy}_{suffix}"
    if not args.dry_run and should_skip(job_dir, args.force):
        print(f"[adapt] 已完成，跳过：{job_dir}（--force 可重跑）", flush=True)
        return job_dir
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "adapt_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if config.strategy == "zero_shot":
        print(f"[adapt] zero_shot：不训练（锚点）→ {job_dir}", flush=True)
        _mark_done(job_dir, "zero_shot: 不训练（锚点）")
        return job_dir

    episodes = (
        parse_episode_list(args.episodes)
        if args.episodes
        else select_episodes(args.total_episodes, config.n_episodes)
    )
    # lerobot_train 拒绝写入已存在的 output_dir（resume=False）。配置/统计放在 job_dir，
    # 训练产物单独放 job_dir/train，二者不冲突也便于重复 dry-run。
    train_dir = job_dir / "train"
    if train_dir.exists() and not args.resume:
        import shutil

        shutil.rmtree(train_dir)  # 上次中断残留、且没有可用 checkpoint 时才需要重建
    argv = build_train_argv(
        config=config,
        dataset_repo=args.dataset_repo,
        dataset_root=args.dataset_root,
        output_dir=str(train_dir),
        device=args.device,
        episodes=episodes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_freq=args.save_freq,
        job_name=f"vla_adapt_{config.strategy}_{suffix}",
    )
    if args.resume:
        argv.append("--resume=true")
    (job_dir / "train_argv.json").write_text(
        json.dumps({"argv": argv, "episodes": episodes}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print("[adapt] dry-run 训练命令：")
        print("  python -m lerobot.scripts.lerobot_train " + " ".join(argv))
        return job_dir

    import lerobot.scripts.lerobot_train as trainer

    original = trainer.make_policy
    trainer.make_policy = make_frozen_make_policy(original, config, job_dir / "adapt_stats.json")
    saved_argv = sys.argv
    try:
        sys.argv = ["lerobot_train", *argv]
        trainer.main()
    finally:
        trainer.make_policy = original
        sys.argv = saved_argv
    print(f"[adapt] 单元完成 → {job_dir}", flush=True)
    _mark_done(
        job_dir,
        f"trained: strategy={config.strategy} episodes={len(episodes)} "
        f"steps={config.steps} batch_size={args.batch_size}",
    )
    return job_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 4.5：SmolVLA 适配对照（可真正冻结）")
    parser.add_argument("--strategy", required=True, choices=("zero_shot", "head", "full"))
    parser.add_argument("--n-episodes", type=int, default=30, help="微调数据量（集数）")
    parser.add_argument("--episodes", default="",
                        help='显式 episode 列表（如 "[0,1,2]"）；给定时忽略 --n-episodes 的等距抽样')
    parser.add_argument("--job-tag", default="",
                        help="产物目录后缀（默认 ep<n>；A7 这类子集实验用 combo 等可读标签）")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-root", default="", help="本地数据集根目录（v3.0）")
    parser.add_argument("--total-episodes", type=int, default=DEFAULT_TOTAL_EPISODES,
                        help="数据集总集数（用于等距选子集）")
    parser.add_argument("--output-dir", default="runs/week38/vla_adapt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--save-freq", type=int, default=500,
                        help="checkpoint 间隔（默认 500；机器重启/断连最多丢一个窗口）")
    parser.add_argument("--resume", action="store_true",
                        help="从 job_dir/train 的最后一个 checkpoint 续训（默认重新开始并清空 train/）")
    parser.add_argument("--force", action="store_true",
                        help="忽略 DONE 标记强制重跑该单元（默认已完成即跳过，保证矩阵幂等）")
    parser.add_argument("--dry-run", action="store_true", help="只写配置/命令，不训练")
    args = parser.parse_args()
    run_unit(args)


if __name__ == "__main__":
    main()
