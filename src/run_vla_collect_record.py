#!/usr/bin/env python
"""多任务采集（带**旁挂负载记录**）—— `lerobot-record` 的透明包装（第 38 周 P1 前置）。

`lerobot-record` 采出来的数据集不含 `Present_Load`，而 2.10 的取证表明"没有负载通道就做不了
可靠的成功标注"（位置判据对人工真值零可分性、错误率 57%）。本脚本**不改数据集 fields**，
只在录制时旁挂写一份 `load_sidecar.jsonl`，并按 `(episode_index, frame_index)` 与数据集帧对齐。

用法（**其余参数与 `lerobot-record` 完全一致，原样转交**）::

    python src/run_vla_collect_record.py --sidecar runs/week38/collect/load_sidecar.jsonl `
      --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm `
      --robot.cameras='{"fixed": {...}, "handeye": {...}}' `
      --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm `
      --dataset.repo_id=wangbangyan/demo_vla_grasp `
      --dataset.single_task="夹起桌面上的螺丝刀" `
      --dataset.num_episodes=20 --dataset.episode_time_s=99999 --dataset.reset_time_s=99999 `
      --dataset.push_to_hub=true --dataset.private=true --display_data=false

采集完**必须**跑一次对齐校验（否则旁挂数据不可信）::

    python src/run_vla_collect_record.py --verify `
      --dataset-root <数据集根目录> --sidecar runs/week38/collect/load_sidecar.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from load_sidecar import (  # noqa: E402
    LoadSidecarWriter,
    read_load_sidecar,
    verify_alignment,
)

DEFAULT_SIDECAR = "runs/week38/collect/load_sidecar.jsonl"


def extract_repo_id(argv: list[str]) -> str:
    """从转交给 lerobot 的参数里取 `--dataset.repo_id`（仅用于给旁挂文件做自描述）。"""
    for index, item in enumerate(argv):
        if item == "--dataset.repo_id" and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith("--dataset.repo_id="):
            return item.split("=", 1)[1]
    return ""


def run_collect(rest: list[str], sidecar: Path) -> int:
    import lerobot.record as record_module

    from load_logging import patch_record_pipeline

    repo_id = extract_repo_id(rest)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    writer = LoadSidecarWriter(sidecar, dataset_repo=repo_id)
    print(f"[采集] 旁挂负载 → {sidecar}（repo={repo_id or '(未指定)'}）")
    print("[采集] 其余参数原样转交 lerobot-record；数据集 fields 不变，仅旁挂记录")

    sys.argv = [sys.argv[0], *rest]  # lerobot 的 parser 自己解析 argv
    with patch_record_pipeline(record_module, writer) as holder:
        try:
            record_module.record()
        except BaseException:
            writer.abort_episode()  # 异常中断：未完成的那一集不落盘
            raise
        robot = holder.get("robot")
    failures = getattr(robot, "read_failures", 0)
    camera_timeouts = getattr(robot, "camera_timeouts", 0)
    print(f"\n[采集] 旁挂文件 → {sidecar}（提交 {writer.committed_episodes} 集，"
          f"读取负载失败 {failures} 次，相机取帧超时 {camera_timeouts} 次）")
    if failures:
        print("[采集] 警告：存在读取失败，旁挂序列可能有空洞，请核对后再用")
    if camera_timeouts:
        print("[采集] 警告：本场出现相机取帧超时（已自动重试救回）。若次数偏多，"
              "把两路相机插到不同的 USB 控制器上再采，避免帧率抖动影响数据质量")
    print("[采集] 下一步：跑一次对齐校验")
    print(f"        python src/run_vla_collect_record.py --verify --dataset-root <root> "
          f"--sidecar {sidecar}")
    return 0


def run_verify(dataset_root: str, sidecar: Path, tol: float) -> int:
    from lerobot_label_source import read_dataset_meta, read_gripper_positions

    records = read_load_sidecar(sidecar)
    if not records:
        print(f"[校验] 旁挂文件为空或不存在：{sidecar}")
        return 1
    meta = read_dataset_meta(dataset_root)
    episodes = {row["episode_index"] for row in records}
    positions = {index: read_gripper_positions(meta, index) for index in sorted(episodes)}

    report = verify_alignment(records, positions, tol=tol)
    print(f"[校验] 数据集 {meta.root}｜旁挂 {sidecar}")
    for line in report.summary_lines():
        print(f"  {line}")
    if not report.ok:
        print("  → 未通过：帧数或位置不吻合，旁挂数据不可直接用于成功标注")
        return 1
    print("  → 通过：旁挂与数据集逐帧一致")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="带旁挂负载记录的多任务采集（lerobot-record 的透明包装）"
    )
    parser.add_argument("--sidecar", default=DEFAULT_SIDECAR,
                        help=f"旁挂负载文件路径（默认 {DEFAULT_SIDECAR}；追加写入）")
    parser.add_argument("--verify", action="store_true",
                        help="只做对齐校验（不采集）")
    parser.add_argument("--dataset-root", default="", help="--verify 用的数据集根目录")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="--verify 的位置容差（数据集按 float32 存）")
    args, rest = parser.parse_known_args()
    sidecar = Path(args.sidecar).expanduser()

    if args.verify:
        if not args.dataset_root:
            parser.error("--verify 需要 --dataset-root")
        return run_verify(args.dataset_root, sidecar, args.tol)

    if not rest:
        parser.error("缺少采集参数：请按 lerobot-record 的用法传参（--robot.* / --dataset.* 等）")
    return run_collect(rest, sidecar)


if __name__ == "__main__":
    raise SystemExit(main())
