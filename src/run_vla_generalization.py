#!/usr/bin/env python
"""第 38 周 P2：VLA 组合泛化评测 CLI（combo / paraphrase / novel_object）。

两种运行方式：

1. `--dry-run`：只做 **split 生成 + 无泄漏断言**，对每种协议打印 train/held-out 划分，
   不加载任何 checkpoint / policy / GPU（可在无环境依赖的机器上验证协议正确性）。
2. 真实评测：以 `--model-dir` 记录适配后 checkpoint；当前本仓库的 SmolVLA/真机推理链路
   仍是 stub（设计 §3.3 F），故真实模式走**状态输入 scripted 策略的仿真协议验证**
   （验收 A7 的降级路径），产物写入 `--out/<protocol>/`。

用法::

    # 只验证协议与无泄漏（无需 GPU）
    python src/run_vla_generalization.py --dry-run --objects A,B --instructions 1,2

    # 组合泛化评测（需 ManiSkill）
    python src/run_vla_generalization.py --model-dir <ckpt> --protocol combo --heldout-ratio 0.3

输入组合二选一：`--combos-json <file>`（JSON 数组 `[[object, instruction], ...]`）
或 `--objects a,b` + `--instructions x,y`（取笛卡尔积）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from vla_generalization import (  # noqa: E402
    PROTOCOLS,
    VLAGeneralizationError,
    build_split,
    evaluate_generalization,
    write_generalization_report,
)

DEFAULT_OUT = "runs/week38/vla_generalization"


def parse_protocols(text: str) -> list[str]:
    """解析逗号分隔的协议名（保序去重），未知协议报错。"""
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if not parts:
        raise VLAGeneralizationError("--protocol 不能为空")
    ordered: list[str] = []
    for name in parts:
        if name not in PROTOCOLS:
            raise VLAGeneralizationError(f"未知协议 {name!r}，仅支持 {PROTOCOLS}")
        if name not in ordered:
            ordered.append(name)
    return ordered


def _load_combos(args: argparse.Namespace) -> list[tuple[str, str]]:
    """按 CLI 约定加载组合：`--combos-json` 或 `--objects` × `--instructions`。"""
    if args.combos_json and (args.objects or args.instructions):
        raise VLAGeneralizationError(
            "--combos-json 与 --objects/--instructions 不能同时给出（二选一）"
        )
    if args.combos_json:
        payload = json.loads(
            Path(args.combos_json).read_text(encoding="utf-8")
        )
        if not isinstance(payload, list):
            raise VLAGeneralizationError("--combos-json 必须是 JSON 数组")
        combos: list[tuple[str, str]] = []
        for item in payload:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise VLAGeneralizationError(
                    f"--combos-json 的每个元素必须是 [object, instruction]，实际 {item!r}"
                )
            combos.append((str(item[0]), str(item[1])))
        return combos
    if args.objects and args.instructions:
        objects = [item.strip() for item in args.objects.split(",") if item.strip()]
        instructions = [
            item.strip() for item in args.instructions.split(",") if item.strip()
        ]
        if not objects or not instructions:
            raise VLAGeneralizationError("--objects/--instructions 不能为空")
        return [(obj, instruction) for obj in objects for instruction in instructions]
    raise VLAGeneralizationError(
        "必须给出 --combos-json，或同时给出 --objects 与 --instructions"
    )


def _print_split(split) -> None:
    """打印一份 split 与无泄漏断言（dry-run 的核心输出）。"""
    print(
        f"[{split.protocol}] train={len(split.train_combos)} "
        f"heldout={len(split.heldout_combos)}"
    )
    print(f"  train   : {list(split.train_combos)}")
    print(f"  heldout : {list(split.heldout_combos)}")
    overlap = set(split.train_combos) & set(split.heldout_combos)
    if overlap:
        raise VLAGeneralizationError(
            f"{split.protocol} 无泄漏断言失败：{sorted(overlap)[:5]}"
        )
    print("  无泄漏断言：train ∩ heldout = ∅ ✔；两侧均非空 ✔")
    for warning in split.warnings:
        print(f"  告警：{warning}")


def _scripted_runner(max_steps: int):
    """状态输入 scripted 策略 runner（仿真协议验证；忽略 instruction）。"""
    from maniskill_intro import (
        ManiSkillTaskConfig,
        ScriptedPolicy,
        make_maniskill_env,
        run_episode_in_env,
    )

    def runner(env_id: str, seed: int, instruction: str) -> bool:
        config = ManiSkillTaskConfig(env_id=env_id, seed=seed, max_steps=max_steps)
        env = make_maniskill_env(config)
        try:
            result = run_episode_in_env(config, env, ScriptedPolicy(env_id))
        finally:
            env.close()
        return bool(result.success)

    return runner


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第 38 周 P2：VLA 组合泛化评测（combo / paraphrase / novel_object）"
    )
    parser.add_argument(
        "--model-dir", default="",
        help="适配后 checkpoint 目录（dry-run 可留空；真实推理接入待 S12）",
    )
    parser.add_argument(
        "--protocol", default=",".join(PROTOCOLS),
        help="逗号分隔的协议，缺省三种全跑",
    )
    parser.add_argument("--heldout-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0, help="split 生成随机种子")
    parser.add_argument("--episodes", type=int, default=10, help="每个组合的 episode 数")
    parser.add_argument("--base-seed", type=int, default=500, help="episode seed 起点")
    parser.add_argument("--max-steps", type=int, default=50, help="仿真单 episode 步数")
    parser.add_argument("--out", default=DEFAULT_OUT, help="报告输出目录")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印 split 与无泄漏断言，不加载 checkpoint/policy",
    )
    parser.add_argument(
        "--combos-json", default="",
        help="组合 JSON 文件：数组 [[object, instruction], ...]",
    )
    parser.add_argument(
        "--objects", default="", help="物体列表（逗号分隔），与 --instructions 取笛卡尔积",
    )
    parser.add_argument("--instructions", default="", help="指令列表（逗号分隔）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        protocols = parse_protocols(args.protocol)
        combos = _load_combos(args)
        print(
            f"[输入] {len(combos)} 个 (object, instruction) 组合 | 协议 {protocols} | "
            f"heldout_ratio={args.heldout_ratio} seed={args.seed}"
        )
        splits = {}
        for protocol in protocols:
            split = build_split(combos, protocol, args.heldout_ratio, args.seed)
            splits[protocol] = split
            _print_split(split)

        if args.dry_run:
            print("[dry-run] 已完成全部协议的 split 与无泄漏断言，未加载任何 checkpoint/policy。")
            return 0

        if not args.model_dir:
            print("[提示] 未给 --model-dir：本次为状态输入 scripted 策略的**仿真协议验证**"
                  "（验收 A7 的降级路径，设计 §3.3 F）。")
        out_dir = Path(args.out)
        for protocol, split in splits.items():
            report = evaluate_generalization(
                None,
                split,
                args.episodes,
                runner=_scripted_runner(args.max_steps),
                base_seed=args.base_seed,
            )
            report["model_dir"] = args.model_dir
            json_path, markdown_path = write_generalization_report(
                report, out_dir / protocol
            )
            print(
                f"[{protocol}] train={report['train']['success_rate']:.3f} "
                f"heldout={report['heldout']['success_rate']:.3f} "
                f"gap={report['gap']:+.4f} | 报告 {json_path} / {markdown_path}"
            )
        return 0
    except (
        VLAGeneralizationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"第 38 周泛化评测失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
