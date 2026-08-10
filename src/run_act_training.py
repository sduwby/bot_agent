"""运行第 13 周 LeRobot 低维 ACT 训练、固定评测与三策略比较。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from behavior_cloning import BehaviorCloningError
from lerobot_act import (
    ACTTrainingConfig,
    LeRobotACTError,
    compare_act_bc_scripted,
    evaluate_lerobot_act,
    probe_lerobot_act,
    select_best_rollout_checkpoint,
    train_lerobot_act,
)
from maniskill_intro import SUPPORTED_TASKS, ManiSkillUnavailableError


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (Path("runs") / "maniskill_act" / timestamp).expanduser().resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析数据、训练、固定评测和比较参数。"""
    parser = argparse.ArgumentParser(
        description="训练并评测第 13 周 LeRobot 低维 ACT"
    )
    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=Path("runs/maniskill_expert/week11-200/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-episodes-per-task", type=int, default=80)
    parser.add_argument("--validation-episodes-per-task", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--training-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--normalization-mode", type=str, default="unique",
                        choices=("unique", "chunk"),
                        help="action 统计方式：unique=唯一 transition，chunk=原始 chunk 展开")
    parser.add_argument("--use-vae", type=lambda v: v.lower() in ("1", "true", "yes", "on"),
                        default=True,
                        help="是否启用 VAE（False 为 deterministic ACT）")
    parser.add_argument("--use-qvel", action="store_true",
                        help="加入 Panda arm_qvel[:7]，robot state 5→12 维（schema v2）")
    parser.add_argument("--warm-start", type=Path, default=None,
                        help="从已有 checkpoint 目录热启动（载入共享权重）")
    parser.add_argument("--dagger-data", type=Path, default=None,
                        help="DAgger recovery 补采数据（npz），训练时合并")
    parser.add_argument("--task", type=str, default="all",
                        choices=("all",) + SUPPORTED_TASKS,
                        help="只训练指定任务或 all")
    parser.add_argument("--random-seed", type=int, default=20260807)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--eval-seed-start", type=int, default=100)
    parser.add_argument("--eval-episodes-per-task", type=int, default=50)
    parser.add_argument("--dev-seed-start", type=int, default=150)
    parser.add_argument("--dev-episodes", type=int, default=20)
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=Path("runs/maniskill_bc/week12/checkpoint.pt"),
    )
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="复用已有 checkpoint，只执行探针、smoke 与固定评测",
    )
    parser.add_argument(
        "--eval-n-action-steps",
        type=int,
        default=None,
        help="诊断迭代：覆盖评测时 ACT 每次消费的动作数（如 1）",
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=1.0,
        help="rollout 位置增量的 EMA 系数（(0,1]），1.0 表示不平滑",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """训练后从磁盘重载并完成固定 seed 的 ACT/BC/scripted 比较。"""
    args = parse_args(argv)
    output_dir = args.output_dir or _default_output_dir()
    try:
        if args.eval_seed_start < 0:
            raise ValueError("eval_seed_start 不能为负数")
        if args.eval_episodes_per_task <= 0:
            raise ValueError("eval_episodes_per_task 必须是正整数")
        if args.dev_seed_start < 0:
            raise ValueError("dev_seed_start 不能为负数")
        if args.dev_episodes <= 0:
            raise ValueError("dev_episodes 必须是正整数")
        if args.checkpoint_interval is not None and args.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval 必须是正整数")
        if args.device not in ("mps", "cpu"):
            raise ValueError("device 只支持 mps 或 cpu")
        tasks = SUPPORTED_TASKS if args.task == "all" else (args.task,)
        config = ACTTrainingConfig(
            dataset_summary=args.dataset_summary,
            output_dir=output_dir,
            train_episodes_per_task=args.train_episodes_per_task,
            validation_episodes_per_task=args.validation_episodes_per_task,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            training_steps=args.training_steps,
            batch_size=args.batch_size,
            validation_interval=args.validation_interval,
            checkpoint_interval=args.checkpoint_interval,
            normalization_mode=args.normalization_mode,
            use_vae=args.use_vae,
            use_qvel=args.use_qvel,
            warm_start_dir=args.warm_start,
            dagger_data=args.dagger_data,
            tasks=tasks,
            random_seed=args.random_seed,
            device=args.device,
        )
        # 统一使用解析后的绝对路径，避免 ManiSkill 改变 CWD 后相对路径失效。
        output_dir = config.output_dir
        bc_checkpoint = args.bc_checkpoint.expanduser().resolve()
        dev_seeds = range(
            args.dev_seed_start, args.dev_seed_start + args.dev_episodes
        )

        print("运行 LeRobot ACT 兼容性探针 ...")
        probe = probe_lerobot_act(config.device)
        print(f"设备：{probe['device']}，forward/backward/save/load 通过")

        if args.skip_training:
            print(f"跳过训练，复用已有 checkpoint：{config.output_dir}")
        else:
            training = train_lerobot_act(config)
            print(f"训练目录：{config.output_dir}")
            for task in training.tasks:
                print(
                    f"{task.env_id}: train={task.train_transitions}, "
                    f"validation={task.validation_transitions}, "
                    f"best_step={task.best_step}, "
                    f"validation_loss={task.best_validation_loss:.6f}, "
                    f"device={task.device}"
                )

        if config.checkpoint_interval is not None:
            print("在 dev seeds 上选择 best-rollout-dev ...")
            for env_id in config.tasks:
                selected = select_best_rollout_checkpoint(
                    output_dir,
                    env_id=env_id,
                    dev_seeds=dev_seeds,
                    device=config.device,
                    n_action_steps=args.eval_n_action_steps,
                    action_smoothing=args.action_smoothing,
                )
                print(f"{env_id} 选中 checkpoint：{selected.name}")

        print("从磁盘重载并执行每任务 1 次 smoke rollout ...")
        smoke = evaluate_lerobot_act(
            output_dir,
            seeds=[0],
            device=config.device,
            n_action_steps=args.eval_n_action_steps,
            action_smoothing=args.action_smoothing,
            tasks=config.tasks,
        )
        print(f"smoke：{smoke.total_successes}/{smoke.total_episodes}")

        seeds = range(
            args.eval_seed_start,
            args.eval_seed_start + args.eval_episodes_per_task,
        )
        comparison = compare_act_bc_scripted(
            output_dir,
            bc_checkpoint_path=bc_checkpoint,
            seeds=seeds,
            device=config.device,
            threshold=args.success_threshold,
            n_action_steps=args.eval_n_action_steps,
            action_smoothing=args.action_smoothing,
            tasks=config.tasks,
        )
        for policy in comparison.policies:
            print(
                f"{policy.policy_name}: {policy.total_successes}/"
                f"{policy.total_episodes} "
                f"({policy.success_rate:.1%})，平均推理 "
                f"{policy.mean_inference_ms:.2f} ms/step"
            )
        print(
            f"门禁：ACT {comparison.act_rate:.1%} >= "
            f"max({comparison.threshold:.1%}, BC "
            f"{comparison.bc_rate:.1%}) = "
            f"{comparison.effective_threshold:.1%}"
        )
        if not comparison.act_passed:
            print(
                f"ACT 未达到门禁 {comparison.effective_threshold:.1%}。",
                file=sys.stderr,
            )
            return 1
    except (
        LeRobotACTError,
        BehaviorCloningError,
        ManiSkillUnavailableError,
        OSError,
        ValueError,
    ) as exc:
        print(f"第 13 周 ACT 训练/评测失败：{exc}", file=sys.stderr)
        return 1

    print("第 13 周 LeRobot 低维 ACT 训练与固定评测验收通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
