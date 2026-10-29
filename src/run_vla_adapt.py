#!/usr/bin/env python
"""第 38 周 P2：VLA 适配对照 CLI（首轮 **3 策略 × 2 数据量，封顶 6 个单元**）。

研究核心（design §7.3）：**同数据、同 seed、同评测协议，只改一个自变量。
自变量 = 适配策略（zero_shot / head / full）× 微调数据量（30 / 60 集）。
zero_shot 是"不改权重"的锚点，不占数据量档位，故默认矩阵为
``zero_shot`` + ``head{30,60}`` + ``full{30,60}`` = 5 个单元，硬封顶 6 个。

产物（design §4「状态管理」，沿用已忽略的 `runs/` 约定）::

    <output-dir>/<job>/config.json        # 每个单元的自变量（可审计）
    <output-dir>/<job>/metrics.jsonl      # 追加式指标行
    <output-dir>/<job>/adapt_result.json  # AdaptResult.to_dict()

用法::

    # 先看计划矩阵（写 config.json，不训练、不占 GPU）
    python src/run_vla_adapt.py --dataset-repo <multi_task> --dry-run
    # 云上 A10 实跑
    python src/run_vla_adapt.py --dataset-repo <multi_task> \\
        --dataset-root <本地或远程> --output-dir <远程> --device cuda

.. warning::
   **真实训练/评测请走新入口**：`src/run_vla_adapt_smolvla.py`（进程内按策略冻结）
   + `src/run_adapt_eval_offline.py`（SmolVLA 离线评测）。本 CLI 的默认路径存在两个
   已知缺陷（head≡full、eval 用 ACT 加载器），非 dry-run 会被直接拒绝；
   仅历史复现时才加 `--allow-unsafe-default`。

矩阵构造被拆成纯函数 `build_adapt_jobs`，可在不运行 CLI 的前提下单测封顶逻辑。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from vla_adapt import (  # noqa: E402
    ADAPT_SCHEMA_VERSION,
    SUPPORTED_STRATEGIES,
    AdaptError,
    build_adapt_config,
    run_adapt_experiment,
)

#: 首轮矩阵硬封顶：3 策略 × 2 数据量 = 6 个单元（design §7.3「封顶 6 单元」）。
MAX_ADAPT_UNITS = 6
DEFAULT_STRATEGIES = ",".join(SUPPORTED_STRATEGIES)
DEFAULT_N_EPISODES = "30,60"
DEFAULT_STEPS = 2000
DEFAULT_SEED = 38
DEFAULT_OUTPUT_DIR = "runs/week38/vla_adapt"
DEFAULT_EVAL_SEEDS = "0,1,2,3,4"
DEFAULT_DEVICE = "cuda"


@dataclass(frozen=True)
class AdaptJob:
    """矩阵中的一个实验单元：一个 `AdaptConfig` + 可读的落盘标识。"""

    job_name: str
    config: AdaptConfig
    data_size: int
    note: str


def _order_strategies(strategies: Sequence[str]) -> list[str]:
    """去重、规整大小写，并按规范顺序排列；拒绝未知策略（含延后的 lora）。"""
    seen: list[str] = []
    for raw in strategies:
        name = str(raw).strip().lower()
        if name and name not in seen:
            seen.append(name)
    if not seen:
        raise AdaptError("--strategy 不能为空")
    unknown = [name for name in seen if name not in SUPPORTED_STRATEGIES]
    if unknown:
        raise AdaptError(
            f"未知策略 {unknown}；支持 {SUPPORTED_STRATEGIES}（lora 延后，见 design §3.3 G）"
        )
    return [name for name in SUPPORTED_STRATEGIES if name in seen]


def _parse_sizes(raw: str | Sequence[int | str]) -> list[int]:
    """解析数据量：``"30,60"`` / 单个 / 列表；去空、必须 > 0、拒绝重复。"""
    items = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
    sizes: list[int] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise AdaptError(f"数据量必须为正整数，实际 {value}")
        if value in sizes:
            raise AdaptError(f"数据量重复：{value}")
        sizes.append(value)
    if not sizes:
        raise AdaptError("--n-episodes 不能为空")
    return sizes


def build_adapt_jobs(
    strategies: Sequence[str],
    n_episodes: str | Sequence[int | str],
    steps: int,
    seed: int,
) -> list[AdaptJob]:
    """构造适配对照矩阵（纯函数，可离线单测）。

    - 每个非 zero_shot 策略 × 每个数据量档位各一个单元；
    - zero_shot 只有 **一个** 锚点单元（不训练、数据量为 0）；其 ``n_episodes``
      取首个数据量作占位，仅为满足 `AdaptConfig` 的 ``n_episodes > 0`` 校验；
    - 单元总数超过 `MAX_ADAPT_UNITS` 时**报错**而非静默截断（避免少跑单元却不知情）。
    """
    ordered = _order_strategies(strategies)
    sizes = _parse_sizes(n_episodes)
    jobs: list[AdaptJob] = []
    for strategy in ordered:
        if strategy == "zero_shot":
            jobs.append(
                AdaptJob(
                    job_name="zero_shot",
                    config=build_adapt_config(strategy, sizes[0], steps, seed),
                    data_size=0,
                    note="零样本锚点：不训练（数据量 0，n_episodes 为占位值）",
                )
            )
            continue
        for size in sizes:
            label = "head 仅解冻 action head" if strategy == "head" else "full 全参数微调"
            jobs.append(
                AdaptJob(
                    job_name=f"{strategy}-ep{size}",
                    config=build_adapt_config(strategy, size, steps, seed),
                    data_size=size,
                    note=label,
                )
            )
    if len(jobs) > MAX_ADAPT_UNITS:
        raise AdaptError(
            f"适配矩阵 {len(jobs)} 个单元超过封顶 {MAX_ADAPT_UNITS}（design §7.3）："
            "请减少策略数或数据量档位。"
        )
    return jobs


def _parse_seeds(raw: str, name: str) -> list[int]:
    """解析逗号分隔的 seed 列表（非负整数、去空、去重）。"""
    seeds: list[int] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        value = int(text)
        if value < 0:
            raise AdaptError(f"{name} 必须是非负整数，实际 {value}")
        if value in seeds:
            raise AdaptError(f"{name} 重复：{value}")
        seeds.append(value)
    if not seeds:
        raise AdaptError(f"{name} 不能为空")
    return seeds


def _config_payload(job: AdaptJob) -> dict[str, object]:
    """config.json 内容：让计划矩阵在未训练时也可审计。"""
    return {
        "schema_version": ADAPT_SCHEMA_VERSION,
        "job_name": job.job_name,
        "data_size": job.data_size,
        "note": job.note,
        "config": asdict(job.config),
    }


def _metrics_payload(job: AdaptJob, result: object) -> dict[str, object]:
    """metrics.jsonl 的单行指标（与 adapt_result.json 同源）。"""
    return {
        "job_name": job.job_name,
        "strategy": job.config.strategy,
        "n_episodes": job.config.n_episodes,
        "data_size": job.data_size,
        "steps": job.config.steps,
        "seed": job.config.seed,
        "success_rate": float(result.success_rate),
        "retention_rate": result.retention_rate,
        "train_seconds": float(result.train_seconds),
        "peak_mem_mb": result.peak_mem_mb,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def unsafe_default_error(dry_run: bool, allow_unsafe_default: bool) -> str | None:
    """默认训练/评测路径对 SmolVLA **无效**，真跑前必须拦住。

    两个独立缺陷（`vla_adapt._default_train_fn` / `_default_eval_fn` 的 docstring 已写明）：

    1. `lerobot_train` 的 CLI 没有按模块冻结入口 → `head` 与 `full` 训练成同一个东西；
    2. 默认 `eval_fn` 走 `lerobot_act` 的 ACT checkpoint 加载器，不能评 SmolVLA。

    dry-run（只写矩阵）不受影响；确认要跑旧路径需显式 `--allow-unsafe-default`。
    """
    if dry_run or allow_unsafe_default:
        return None
    return (
        "该 CLI 的默认训练/评测路径对 SmolVLA 无效：head 会等于 full（CLI 无冻结入口），"
        "且评测走 ACT 加载器。请改用 src/run_vla_adapt_smolvla.py（训练）+ "
        "src/run_adapt_eval_offline.py（评测）；确认要跑旧路径请加 --allow-unsafe-default。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="第 38 周 P2：VLA 适配对照（3 策略 × 2 数据量，封顶 6 单元）"
    )
    parser.add_argument("--dataset-repo", required=True, help="LeRobot 数据集 repo_id")
    parser.add_argument("--dataset-root", default="", help="本地数据集根目录（可选，默认走 Hub）")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="产物根目录")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cuda（云上 A10）或 cpu")
    parser.add_argument(
        "--n-episodes", default=DEFAULT_N_EPISODES, help="微调数据量（逗号分隔或单个），首轮 30,60"
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="每个单元的训练步数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="固定随机种子")
    parser.add_argument(
        "--strategy", default=DEFAULT_STRATEGIES, help=f"逗号分隔的策略，默认 {DEFAULT_STRATEGIES}"
    )
    parser.add_argument("--eval-seeds", default=DEFAULT_EVAL_SEEDS, help="评测 seed（逗号分隔）")
    parser.add_argument("--baseline-seeds", default="", help="保留率基线 seed，缺省同 --eval-seeds")
    parser.add_argument("--baseline-task", default="", help="基线任务名（保留率评测口径）")
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印计划矩阵并写 config.json，不训练/评测"
    )
    parser.add_argument(
        "--allow-unsafe-default", action="store_true",
        help="允许跑旧的默认训练/评测路径（head≡full、评测走 ACT 加载器；仅供历史复现）",
    )
    args = parser.parse_args()

    error = unsafe_default_error(args.dry_run, args.allow_unsafe_default)
    if error:
        parser.error(error)

    strategies = [item for item in args.strategy.split(",") if item.strip()]
    jobs = build_adapt_jobs(strategies, args.n_episodes, args.steps, args.seed)
    out_root = Path(args.output_dir).expanduser()
    eval_seeds = _parse_seeds(args.eval_seeds, "eval-seeds")
    baseline_seeds = (
        _parse_seeds(args.baseline_seeds, "baseline-seeds")
        if args.baseline_seeds.strip()
        else None
    )
    baseline_task = args.baseline_task or None

    print(f"[适配] 计划矩阵 {len(jobs)} 个单元（封顶 {MAX_ADAPT_UNITS}）→ {out_root}")
    for job in jobs:
        print(
            f"  - {job.job_name:14s} strategy={job.config.strategy:9s} "
            f"data={job.data_size:3d} steps={job.config.steps} "
            f"seed={job.config.seed}  # {job.note}"
        )

    for job in jobs:
        job_dir = out_root / job.job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        _write_json(job_dir / "config.json", _config_payload(job))
        if args.dry_run:
            continue
        result = run_adapt_experiment(
            job.config,
            args.dataset_root,
            eval_seeds,
            baseline_seeds=baseline_seeds,
            baseline_task=baseline_task,
            output_dir=job_dir,
            device=args.device,
        )
        _write_json(job_dir / "adapt_result.json", result.to_dict())
        _append_jsonl(job_dir / "metrics.jsonl", _metrics_payload(job, result))
        print(
            f"[适配] {job.job_name}: success={result.success_rate:.3f} "
            f"retention={result.retention_rate} train_s={result.train_seconds:.1f}"
        )

    if args.dry_run:
        print("[适配] dry-run：已写出各单元 config.json，未执行训练/评测")
    else:
        print(f"[适配] 完成，产物 → {out_root}")


if __name__ == "__main__":
    main()
