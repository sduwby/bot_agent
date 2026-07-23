#!/usr/bin/env python3
"""从中文指令直接驱动脚本化 pick→place 任务。"""

from __future__ import annotations

import argparse
from pathlib import Path

from run_task import run_task_once
from task_parser import parse_instruction, supported_terms


def parse_args() -> argparse.Namespace:
    """解析中文指令和仿真参数。"""
    parser = argparse.ArgumentParser(
        description="解析一条中文指令并执行对应的 AI2-THOR pick→place 任务。"
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="中文任务描述，例如：把信用卡放进抽屉",
    )
    parser.add_argument("--scene", default="FloorPlan1", help="要加载的场景名称")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/instruction_task"),
        help="运行产物的根目录",
    )
    parser.add_argument("--width", type=int, default=480, help="输出图像宽度")
    parser.add_argument("--height", type=int, default=360, help="输出图像高度")
    return parser.parse_args()


def main() -> int:
    """把中文指令编译为 objectType，再调起固定技能链。"""
    args = parse_args()
    try:
        spec = parse_instruction(args.instruction)
    except ValueError as exc:
        print(f"解析指令失败：{exc}")
        supported = supported_terms()
        print(f"支持物体：{', '.join(supported['objects'])}")
        print(f"支持容器：{', '.join(supported['receptacles'])}")
        return 2

    summary = run_task_once(
        scene=args.scene,
        object_type=spec.object_type,
        receptacle_type=spec.receptacle_type,
        output_root=args.output_root,
        width=args.width,
        height=args.height,
    )
    run_dir = Path(summary["run_dir"])
    task_success = bool(summary["task_success"])
    print(f"原始指令：{spec.raw_instruction}")
    print(f"解析后参数：pick({summary['task']['object_type']}) -> "
          f"place({summary['task']['receptacle_type']})")
    print(f"运行目录：{run_dir}")
    print(f"任务结果：{'成功' if task_success else '失败'}")
    return 0 if task_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
