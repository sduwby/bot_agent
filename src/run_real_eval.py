#!/usr/bin/env python
"""真机端到端评测 CLI（阶段 F S10/S11）：按真实任务集批量执行并生成报告。

用法：
  # dry-run（Mock 引擎）验证评测链路，不连接机械臂
  python src/run_real_eval.py --dry-run

  # 真机评测（需机械臂就绪 + 螺丝刀摆放到对应位置）
  python src/run_real_eval.py --model-dir <path> --dataset-root <path> --dataset-repo wangbangyan/demo_v23
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from real_evaluation import EvalTrial, RealEvalRunner, load_real_task_set
from real_executor import MockEngine, RealEngine, RealSkillExecutor
from real_router import RealSkillRouter, default_real_route

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 真机端到端评测（阶段 F）")
    parser.add_argument("--task-set", default=r"d:\FittenProject\bot_agent\evaluation\real_tasks.json")
    parser.add_argument("--dry-run", action="store_true", help="用 Mock 引擎，不连接机械臂")
    parser.add_argument("--model-dir", default=r"d:\FittenProject\bot_agent\outputs\act_demo_v23_pretrained")
    parser.add_argument("--dataset-root",
                        default=r"C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\demo_v23")
    parser.add_argument("--dataset-repo", default="wangbangyan/demo_v23")
    parser.add_argument("--max-relative-target", type=float, default=8.0, help="低速限幅（S2）")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--hold-steps", type=int, default=300)
    parser.add_argument("--output-root", default=r"d:\FittenProject\bot_agent\outputs")
    args = parser.parse_args()

    tasks = load_real_task_set(args.task_set)
    runner = RealEvalRunner(args.output_root)

    for task in tasks:
        verdict = "abort"
        note = ""
        steps_run = 0
        try:
            if args.dry_run:
                engine = MockEngine(grasp_result="success")
            else:
                engine = RealEngine(
                    model_dir=args.model_dir, dataset_root=args.dataset_root,
                    dataset_repo=args.dataset_repo, max_relative_target=args.max_relative_target,
                )
            executor = RealSkillExecutor(engine, empty_grip_baseline=1.5,
                                         max_grasp_steps=args.steps, hold_steps=args.hold_steps)
            router = RealSkillRouter(executor, default_real_route())
            result = router.run_skill("grasp_screwdriver")
            steps_run = result.steps
            if result.success:
                verdict = "success"
            else:
                verdict = "abort" if result.exhausted else "grasp_empty"
                note = result.error or result.outcome_message
            if not args.dry_run:
                engine.close()
        except Exception as exc:
            note = f"异常: {exc}"
            logging.error("任务 %s 异常: %s", task["id"], exc)

        trial = EvalTrial(task_id=task["id"], instruction=task.get("instruction", ""),
                          position=task.get("position", ""), verdict=verdict,
                          steps_run=steps_run, note=note)
        runner.add_trial(trial)
        logging.info("[%s] %s -> %s", trial.task_id, trial.position, trial.verdict)

    csv_path, report_path = runner.save()
    print(f"\n评测完成。\nCSV: {csv_path}\n报告: {report_path}")
    print(runner.render_report())


if __name__ == "__main__":
    main()
