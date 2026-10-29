#!/usr/bin/env python
"""SmolVLA 微调训练入口（阶段 F S12）—— 云上 A10 使用。

组装 lerobot（>=0.6）的 `lerobot.scripts.lerobot_train` 参数，从 `lerobot/smolvla_base`
基础权重微调。**云端实测确认的口径**（2026-10-11）：

- lerobot 0.6.x **没有** `lerobot.scripts.train`，入口是 `lerobot.scripts.lerobot_train`；
- 微调用 `--policy.path=<基础权重>`，不是 `--policy.type=smolvla`；
- 设备由 Accelerate 自动探测，`--policy.device` 只用于强制 CPU（链路验证）；
- 0.6.x 的 `MultiLeRobotDataset` 未启用（`make_dataset` 直接 `raise NotImplementedError`），
  **多数据集无法直接用多个 repo_id 训练**，必须先合并成一个 repo：
  `lerobot-edit-dataset --operation.type merge --operation.repo_ids "['a','b']"`；
- v2.1 采集的数据需先转 v3.0：`lerobot.scripts.convert_dataset_v21_to_v30`；
- 数据集相机名（本仓 `fixed`/`handeye`）与 smolvla_base 的 `camera1..3` 不一致，
  必须用 `--rename_map` 重命名（同时会跳过相机名一致性校验）。

用法（服务器，建议 nohup）：
  python src/run_vla_train.py --dataset-repo wangbangyan/demo_vla_all \
      --output-dir outputs/smolvla_vla --steps 20000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

#: 本仓采集相机 → smolvla_base 期望相机的默认重命名（2 台：fixed→camera1、handeye→camera2）。
DEFAULT_RENAME_MAP = json.dumps(
    {
        "observation.images.fixed": "observation.images.camera1",
        "observation.images.handeye": "observation.images.camera2",
    },
    separators=(",", ":"),
)

#: 0.6.x 的训练入口模块（**不是** lerobot.scripts.train）。
TRAIN_MODULE = "lerobot.scripts.lerobot_train"
#: 微调的默认基础权重。
DEFAULT_POLICY_PATH = "lerobot/smolvla_base"


def build_train_command(args: argparse.Namespace) -> list[str]:
    """组装 lerobot 0.6.x 的 SmolVLA 微调命令（纯函数，便于测试）。"""
    cmd = [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        f"--policy.path={args.policy_path}",
        f"--dataset.repo_id={args.dataset_repo}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--save_freq={args.save_freq}",
        f"--eval_steps={args.eval_steps}",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
    ]
    if args.rename_map:
        cmd.append(f"--rename_map={args.rename_map}")
    if args.dataset_root:
        cmd.append(f"--dataset.root={args.dataset_root}")
    if args.resume:
        cmd.append("--resume=true")
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmolVLA 微调训练（S12）")
    parser.add_argument(
        "--dataset-repo",
        default="wangbangyan/demo_vla_all",
        help="单个 repo id（多任务需先 merge 成一个 repo，见模块 docstring）",
    )
    parser.add_argument("--dataset-root", default="", help="本地缓存数据集目录（可选，默认走 Hub）")
    parser.add_argument(
        "--policy-path",
        default=DEFAULT_POLICY_PATH,
        help="基础权重（微调起点）；0.6.x 用 --policy.path 而非 --policy.type",
    )
    parser.add_argument(
        "--rename-map",
        default=DEFAULT_RENAME_MAP,
        help="相机键重命名 JSON；默认把 fixed/handeye 映射到 camera1/camera2（传 '' 可关闭）",
    )
    parser.add_argument("--output-dir", default="outputs/smolvla_run")
    parser.add_argument("--job-name", default="smolvla_vla")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda",
        help="映射到 --policy.device；实际设备由 Accelerate 探测（cpu 会强制走 CPU）",
    )
    parser.add_argument("--eval-steps", type=int, default=0, help="0=关闭（需 dataset.eval_split>0 才生效）")
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-only", action="store_true", help="只打印命令不执行（核对参数）")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cmd = build_train_command(args)
    print(">>> 训练命令：\n" + " ".join(cmd) + "\n")
    if args.print_only:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
