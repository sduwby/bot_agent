#!/usr/bin/env python
"""多任务采集协议 CLI（阶段 F S12）—— 打印 SOP，不驱动机械臂。

VLA 微调需要：多个任务 × 每个任务多条不同文本的 demo。
本脚本给出录制命令模板与采集纪律（可先 --print-only 核对）。
"""

from __future__ import annotations

import argparse

# 任务文本示例（按需扩展；每任务建议 ≥20 条 demo、位置随机化）
EXAMPLE_TASKS = [
    "夹起桌面上的螺丝刀",
    "把螺丝刀拿起来",
    "张开夹爪拿起螺丝刀再抬起",
    "把螺丝刀放到桌面左侧",   # 需要 place/zone 技能数据，框架先行阶段可选
]

RECORD_TEMPLATE = """\
D:\\miniconda\\envs\\lerobot\\Scripts\\lerobot-record.exe `
  --robot.type=so101_follower --robot.port=COM6 --robot.id=my_awesome_follower_arm `
  --robot.cameras='{{"fixed": {{"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}, "handeye": {{"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}}}' `
  --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_awesome_leader_arm `
  --dataset.repo_id={repo} `
  --dataset.single_task="{task}" `
  --dataset.num_episodes={num_eps} --dataset.episode_time_s=99999 --dataset.reset_time_s=99999 `
  --dataset.push_to_hub=true --dataset.private=true --display_data=false
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLA 多任务采集 SOP（S12）")
    parser.add_argument("--repo-prefix", default="wangbangyan")
    parser.add_argument("--dataset-name", default="demo_vla")
    parser.add_argument("--num-episodes-per-task", type=int, default=20)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    repo = f"{args.repo_prefix}/{args.dataset_name}"
    lines = [
        "# SmolVLA 多任务采集 SOP（docs/week37/design.md §6.2）",
        "",
        f"数据集：{repo}（每任务一条指令；多任务建议分开 repo，如 demo_vla_grasp / demo_vla_place）",
        "",
        "## 任务文本集合（当前建议，按需增减）",
        "",
    ]
    lines += [f"- {t}" for t in EXAMPLE_TASKS]
    lines += [
        "",
        "## 录制纪律",
        "",
        "- 每个任务单独录一个 repo（single_task 每次一条文本），后续多任务合并或分别微调",
        "- 每任务每条文本 ≥20 条 demo；螺丝刀位置随机覆盖桌面左/中/右/近/远",
        "- 每集至少录 1 秒再按 →/Enter 结束；坏集用 ← 重录；保存期间不按键",
        "- VLA 数据强调**文本与行为的对应**：同一动作尽量配多条同义文本",
        "",
        "## 录制命令模板（逐任务执行）",
        "",
    ]
    for i, task in enumerate(EXAMPLE_TASKS[:2]):  # 先打前两条示例
        lines.append(RECORD_TEMPLATE.format(repo=f"{repo}_{i}", task=task,
                                            num_eps=args.num_episodes_per_task))
    text = "\n".join(lines)
    print(text)
    if not args.print_only:
        Path_out = None  # 仅打印；如需落盘留空
        print("\n（提示：采集完成后再以 --dataset-repo 指向对应 repo 运行 run_vla_train.py 微调）")


if __name__ == "__main__":
    main()
