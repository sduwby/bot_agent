#!/usr/bin/env python
"""SmolVLA 微调训练入口（阶段 F S12）—— 云上 A10 使用。

组装 lerobot.scripts.train（--policy.type=smolvla）参数。
说明：SmolVLA 训练不要求数据集带显式文本 feature；任务文本来自 tasks.jsonl
注入的 task 键。默认 demo_v23 仅用于验证链路（单任务，泛化效果不做承诺）。

用法（服务器，建议 nohup）：
  python src/run_vla_train.py --dataset-repo wangbangyan/demo_v23 \
      --output-dir /opt/train_run/smolvla_demo --steps 2000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLA 微调训练（S12）")
    parser.add_argument("--dataset-repo", default="wangbangyan/demo_v23")
    parser.add_argument("--dataset-root", default="", help="本地缓存数据集目录（可选，默认走 Hub）")
    parser.add_argument("--output-dir", default="outputs/smolvla_run")
    parser.add_argument("--job-name", default="smolvla_demo")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", help="cuda（云上）或 cpu（链路验证）")
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-only", action="store_true", help="只打印命令不执行（核对参数）")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "lerobot.scripts.train",
        "--policy.type=smolvla",
        f"--dataset.repo_id={args.dataset_repo}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--device={args.device}",
        f"--eval_freq={args.eval_freq}",
        f"--save_freq={args.save_freq}",
        "--policy.push_to_hub=false",
    ]
    if args.dataset_root:
        cmd.append(f"--dataset.root={args.dataset_root}")
    if args.resume:
        cmd.append("--resume=true")

    print(">>> 训练命令：\n" + " ".join(cmd) + "\n")
    if args.print_only:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
