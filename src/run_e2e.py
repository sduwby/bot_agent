#!/usr/bin/env python3
"""端到端测试：运行完整的 Agent 流程并验证记忆模块。

用法：
    # 场景 A：物体在视野内（直接抓取，验证记忆不干扰正常流程）
    python run_e2e.py --scene FloorPlan1 --instruction "把信用卡放进抽屉"

    # 场景 B：物体在视野外（触发搜索，验证记忆和搜索策略）
    python run_e2e.py --scene FloorPlan1 --instruction "把钥匙放进抽屉"

    # 运行所有默认场景
    python run_e2e.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ai2thor.controller import Controller

from executor import AgentExecutor
from first_trajectory import create_run_dir
from llm_planner import RuleFallback


# ── 默认测试场景 ──

DEFAULT_SCENES = [
    {
        "name": "物体在视野内_直接抓取",
        "scene": "FloorPlan1",
        "instruction": "把信用卡放进抽屉",
        "expect": "memory_created",  # 预期：记忆正确创建
    },
    {
        "name": "需要搜索_杯子放台面",
        "scene": "FloorPlan6",
        "instruction": "把杯子放到台面",
        "expect": "search_triggered",
    },
    {
        "name": "苹果放台面_回归场景",
        "scene": "FloorPlan1",
        "instruction": "把苹果放到台面",
        "expect": "memory_created",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端测试：运行完整 Agent 流程")
    parser.add_argument("--scene", default=None, help="指定场景（默认运行所有）")
    parser.add_argument("--instruction", default=None, help="指定指令")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/e2e_test"),
        help="运行产物的根目录",
    )
    parser.add_argument("--max-steps", type=int, default=20, help="最大循环步数")
    parser.add_argument("--max-search-steps", type=int, default=8, help="最大搜索步数")
    parser.add_argument("--width", type=int, default=480, help="输出图像宽度")
    parser.add_argument("--height", type=int, default=360, help="输出图像高度")
    return parser.parse_args()


def run_agent_scenario(
    name: str,
    scene: str,
    instruction: str,
    output_root: Path,
    max_steps: int,
    max_search_steps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """运行单个测试场景，返回结果摘要。"""
    run_dir = create_run_dir(output_root)
    trajectory_path = run_dir / "trajectory.jsonl"
    records: list[dict] = []
    controller = None
    t0 = time.time()

    try:
        controller = Controller(
            scene=scene,
            width=width,
            height=height,
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
                backend=RuleFallback(),
                max_steps=max_steps,
                max_search_steps=max_search_steps,
            )
            summary = agent.run(instruction)
    except Exception as exc:
        summary = {
            "instruction": instruction,
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

    elapsed = time.time() - t0
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 统计记忆文件 — MemoryStore 使用 run_dir 作为 base_dir
    conv_dir = run_dir / "conversations"
    mem_dir = run_dir / "memories"
    conv_files = list(conv_dir.glob("*.jsonl")) if conv_dir.exists() else []
    memory_files = list(mem_dir.glob("*.jsonl")) if mem_dir.exists() else []

    # 读取 L1 记忆统计
    l1_stats = {"total": 0, "object_location": 0, "room_state": 0, "failure_lesson": 0}
    for mf in memory_files:
        for line in mf.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            rec = json.loads(line)
            l1_stats["total"] += 1
            t = rec.get("type", "")
            if t in l1_stats:
                l1_stats[t] += 1

    # 读取 L0 对话统计
    l0_count = 0
    for cf in conv_files:
        for line in cf.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                l0_count += 1

    return {
        "scenario": name,
        "scene": scene,
        "instruction": instruction,
        "task_success": summary.get("task_success", False),
        "termination_reason": summary.get("termination_reason", ""),
        "final_verification": summary.get("final_verification"),
        "steps_taken": summary.get("steps_taken", 0),
        "action_history": summary.get("action_history", []),
        "elapsed_seconds": round(elapsed, 2),
        "l0_conversation_count": l0_count,
        "l1_memory_stats": l1_stats,
        "run_dir": str(run_dir),
    }


def print_scenario_report(result: dict[str, Any], idx: int) -> None:
    """打印单个场景的测试报告。"""
    total = result["l1_memory_stats"]["total"]
    loc = result["l1_memory_stats"]["object_location"]
    room = result["l1_memory_stats"]["room_state"]
    fail = result["l1_memory_stats"]["failure_lesson"]
    l0 = result["l0_conversation_count"]

    print(f"\n{'='*60}")
    print(f"场景 {idx}: {result['scenario']}")
    print(f"  场景: {result['scene']}")
    print(f"  指令: {result['instruction']}")
    print(f"  结果: {'✅ 成功' if result['task_success'] else '❌ 失败'}")
    print(f"  终止原因: {result['termination_reason']}")
    verification = result.get("final_verification") or {}
    if verification:
        print(f"  最终验证: {verification.get('details', '')}")
    print(f"  步数: {result['steps_taken']}")
    print(f"  耗时: {result['elapsed_seconds']}s")
    print(f"  L0 对话记录: {l0} 条")
    print(f"  L1 记忆: {total} 条 (物体位置:{loc}, 容器状态:{room}, 失败教训:{fail})")
    print(f"  运行目录: {result['run_dir']}")


def main() -> int:
    args = parse_args()

    # 确定要运行的场景
    if args.scene and args.instruction:
        scenarios = [
            {
                "name": args.instruction,
                "scene": args.scene,
                "instruction": args.instruction,
                "expect": "memory_created",
            }
        ]
    elif args.scene or args.instruction:
        print("错误：必须同时指定 --scene 和 --instruction")
        return 1
    else:
        scenarios = DEFAULT_SCENES

    results = []
    overall_success = True

    for i, sc in enumerate(scenarios, 1):
        print(f"\n运行场景 {i}/{len(scenarios)}: {sc['name']}")
        result = run_agent_scenario(
            name=sc["name"],
            scene=sc["scene"],
            instruction=sc["instruction"],
            output_root=args.output_root,
            max_steps=args.max_steps,
            max_search_steps=args.max_search_steps,
            width=args.width,
            height=args.height,
        )
        results.append(result)
        print_scenario_report(result, i)

        # 验证：任务状态与 L0/L1 产物都必须满足要求
        scenario_ok = bool(result["task_success"])
        verification = result.get("final_verification") or {}
        if not verification.get("success", False):
            print("  ❌ 最终任务状态未通过精确验证")
            scenario_ok = False
        if result["l0_conversation_count"] == 0:
            print("  ❌ 未创建 L0 对话记录")
            scenario_ok = False
        if result["l1_memory_stats"]["total"] == 0:
            print("  ❌ 未创建 L1 记忆记录")
            scenario_ok = False
        if sc.get("expect") == "search_triggered":
            search_triggered = any(
                entry.get("skill") == "move_to"
                for entry in result["action_history"]
            )
            if not search_triggered:
                print("  ❌ 预期触发搜索，但轨迹中没有 move_to")
                scenario_ok = False
        overall_success = overall_success and scenario_ok

    # 汇总
    total_steps = sum(r["steps_taken"] for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    success_count = sum(1 for r in results if r["task_success"])

    print(f"\n{'='*60}")
    print(f"端到端测试完成")
    print(f"  场景总数: {len(scenarios)}")
    print(f"  任务成功: {success_count}/{len(scenarios)}")
    print(f"  总步数: {total_steps}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  输出目录: {args.output_root}")

    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
