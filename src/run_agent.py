#!/usr/bin/env python3
"""通过 LLM 驱动的 Agent 循环执行中文指令。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai2thor.controller import Controller

from executor import AgentExecutor
from first_trajectory import create_run_dir


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 LLM Agent 执行具身任务。"
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
        default=Path("runs/llm_agent"),
        help="运行产物的根目录",
    )
    parser.add_argument("--max-steps", type=int, default=20, help="最大循环步数")
    parser.add_argument("--width", type=int, default=480, help="输出图像宽度")
    parser.add_argument("--height", type=int, default=360, help="输出图像高度")
    return parser.parse_args()


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    """将任务摘要保存为 JSON。"""
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """初始化环境、运行 Agent 循环、输出结果。"""
    args = parse_args()
    run_dir = create_run_dir(args.output_root)
    trajectory_path = run_dir / "trajectory.jsonl"
    records: list[dict] = []
    controller = None
    final_summary: dict[str, Any] | None = None

    try:
        controller = Controller(
            scene=args.scene,
            width=args.width,
            height=args.height,
            quality="Low",
            gridSize=0.25,
            rotateStepDegrees=90,
        )
        with trajectory_path.open("w", encoding="utf-8") as trajectory_file:
            agent = AgentExecutor(
                controller=controller,
                run_dir=run_dir,
                trajectory_file=trajectory_file,
                records=records,
                max_steps=args.max_steps,
            )
            final_summary = agent.run(args.instruction)
    except Exception as exc:
        final_summary = {
            "instruction": args.instruction,
            "task_success": False,
            "termination_reason": f"异常: {type(exc).__name__}: {exc}",
            "steps_taken": 0,
            "action_history": [],
            "final_held_object_ids": [],
            "steps_recorded": len(records),
        }
    finally:
        if controller is not None:
            controller.stop()

    if final_summary is None:
        print("错误：未获取到任务摘要")
        return 1

    write_summary(run_dir, final_summary)
    task_success = bool(final_summary["task_success"])

    print(f"运行目录：{run_dir}")
    print(f"指令：{args.instruction}")
    print(f"场景：{args.scene}")
    print(f"步骤数：{final_summary['steps_taken']}")
    print(f"终止原因：{final_summary['termination_reason']}")
    print(f"任务结果：{'成功' if task_success else '失败'}")
    return 0 if task_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
