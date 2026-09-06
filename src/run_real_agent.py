#!/usr/bin/env python
"""阶段 F 真机 Agent CLI：模式切换 skills（技能框架）| vla（自然语言直驱，预留占位）。

用法（在仓库根目录运行）：
  # 技能模式（默认）——dry-run 用 Mock 引擎跑通链路，不连接机械臂
  python src/run_real_agent.py --mode skills --dry-run --instruction "夹起螺丝刀"
  # VLA 预留模式——返回结构化占位说明，不驱动机械臂
  python src/run_real_agent.py --mode vla --instruction "把螺丝刀放进盒子"

真机验证（需机械臂就绪、桌上有螺丝刀、配置 .env 的 LLM_API_KEY）：
  python src/run_real_agent.py --mode skills \
      --instruction "夹起桌面上的螺丝刀" \
      --model-dir outputs/act_demo_v23_pretrained \
      --dataset-root C:/Users/Administrator/.cache/huggingface/lerobot/wangbangyan/demo_v23 \
      --dataset-repo wangbangyan/demo_v23
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 允许以仓库根目录为 cwd 运行时导入 src/ 兄弟模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_planner import LLMBackend

# Windows 控制台默认 GBK，中文/emoji 打印前统一切 UTF-8，避免 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from env_config import load_local_env

load_local_env(".env")  # 把仓库根目录 .env 写入进程环境（不覆盖已有 shell 配置）

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FakeBackend(LLMBackend):
    """确定性假后端：演示 skills 链路。

    - mock_result=success：grasp_screwdriver → done（正常完成）
    - mock_result 失败：连续两次 grasp_screwdriver（模拟一次重试）后 done，
      由 Agent 的重试上限(max_retries=1)决定最终判定。
    """

    def __init__(self, instruction: str, mock_result: str = "success") -> None:
        self.instruction = instruction
        self.mock_result = mock_result
        self.turn = 0
        self._grasp_times = 1 if mock_result == "success" else 2

    def chat(self, system: str, user: str, images: list[str] | None = None) -> str:
        self.turn += 1
        if self.turn <= self._grasp_times:
            return json.dumps(
                {"thought": "dry-run 假决策：夹起螺丝刀", "skill": "grasp_screwdriver", "parameters": {}}
            )
        return json.dumps(
            {"thought": "已夹起，任务完成", "skill": "done", "parameters": {}}
        )


class VLARunner:
    """VLA 自然语言直驱占位（默认关闭，不驱动机械臂）。

    未来启用路径：week18 探针已确认 lerobot/smolvla_base 0.5B 与 SO-101 契约匹配
    （6D state / 3 相机 / 6D action）。届时在此实现 图像观测→策略→动作 推理。
    """

    def run(self, instruction: str) -> dict:
        return {
            "mode": "vla",
            "implemented": False,
            "message": "VLA 自然语言直驱模式当前为占位，未接入推理，未驱动机械臂。",
            "future_contract": {
                "policy_hint": "lerobot/smolvla_base 0.5B（week18 探针确认契约匹配）",
                "obs": ["fixed 相机", "handeye 相机", "6D state"],
                "action": "6D action（与 ACT 同空间）",
                "blocker": "需按指令采集多任务数据集并微调，阶段 A-E 未包含该数据",
            },
        }


def build_skill_agent(args, dry_run: bool):
    """构建技能模式 Agent：dry_run 用 Mock，真机模式用 RealEngine。"""
    from real_executor import MockEngine, RealEngine, RealSkillExecutor

    if dry_run:
        engine = MockEngine(grasp_result=args.mock_result, with_images=args.mock_images)
        empty_baseline = 1.5
    else:
        engine = RealEngine(
            model_dir=args.model_dir,
            dataset_root=args.dataset_root,
            dataset_repo=args.dataset_repo,
            port=args.port,
        )
        empty_baseline = args.empty_grip_baseline
    executor = RealSkillExecutor(engine, empty_grip_baseline=empty_baseline,
                                 max_grasp_steps=args.steps, hold_steps=args.hold_steps)
    from real_agent import RealRobotAgent

    if args.backend == "fake":
        backend = FakeBackend(args.instruction, mock_result=args.mock_result)
    else:
        backend = None  # create_backend() 按 .env 自动选择 OpenAI/DeepSeek（LLM_API_KEY 已配置）
    agent = RealRobotAgent(executor, backend=backend, max_plan_steps=args.max_plan_steps,
                           vision=not args.no_vision)
    return agent, executor


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 阶段 F 真机 Agent（skills|vla）")
    parser.add_argument("--mode", choices=["skills", "vla"], default="skills",
                        help="skills=技能框架(默认)；vla=自然语言直驱预留占位")
    parser.add_argument("--instruction", required=True, help="中文自然语言任务指令")
    parser.add_argument("--dry-run", action="store_true",
                        help="skills 模式：用 Mock 引擎+假 LLM 跑通链路，不连接机械臂")
    # 真机引擎参数
    parser.add_argument("--model-dir", default=r"d:\FittenProject\bot_agent\outputs\act_demo_v23_pretrained")
    parser.add_argument("--dataset-root",
                        default=r"C:\Users\Administrator\.cache\huggingface\lerobot\wangbangyan\demo_v23")
    parser.add_argument("--dataset-repo", default="wangbangyan/demo_v23")
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--steps", type=int, default=300, help="grasp 技能策略段步数")
    parser.add_argument("--hold-steps", type=int, default=300, help="grasp 保持段步数")
    parser.add_argument("--empty-grip-baseline", type=float, default=1.5)
    parser.add_argument("--max-plan-steps", type=int, default=6)
    parser.add_argument("--mock-result", choices=["success", "grasp_empty", "not_lifted"],
                        default="success", help="dry-run 时 Mock 引擎的抓取结果")
    parser.add_argument("--backend", choices=["auto", "fake"], default="auto",
                        help="LLM 后端：auto=按 .env 用真实模型（DeepSeek/OpenAI，无 key 则规则回退），"
                             "fake=确定性假后端（dry-run 演示用）")
    parser.add_argument("--no-vision", action="store_true",
                        help="关闭图像输入，仅用文本观测决策（省 token/费用）")
    parser.add_argument("--mock-images", action="store_true",
                        help="dry-run 时让 Mock 引擎返回合成相机帧，验证视觉链路")
    args = parser.parse_args()

    if args.mode == "vla":
        result = VLARunner().run(args.instruction)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # skills 模式
    agent, executor = build_skill_agent(args, dry_run=args.dry_run)
    try:
        outcome = agent.run(args.instruction)
    finally:
        executor.engine.close()

    print("\n==== 执行记录 ====")
    for s in outcome.steps:
        tag = "[OK]" if s.result.success else "[FAIL]"
        print(f"{tag} {s.skill} {s.params or ''} "
              f"-> {'success' if s.result.success else s.result.failure_reason}"
              + (f" [{s.recovery_note}]" if s.recovery_note else ""))
    print("\n==== 结果 ====")
    print(f"{'成功' if outcome.success else '失败'}: {outcome.message}（用 {outcome.plan_steps_used} 步）")


if __name__ == "__main__":
    main()
