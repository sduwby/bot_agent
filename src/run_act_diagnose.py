"""第 13 周 ACT 失败相位诊断：对指定 seed 重跑并输出逐帧证据。

不改模型、不重训，只分析固定 checkpoint 在失败 episode 上卡在哪个阶段。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from behavior_cloning import BehaviorCloningError, observation_to_state
from lerobot_act import (
    ACTION_SPACE_BOUND,
    LeRobotACTError,
    LeRobotACTPolicy,
    load_lerobot_act_checkpoints,
)
from maniskill_intro import (
    ManiSkillTaskConfig,
    ManiSkillUnavailableError,
    make_maniskill_env,
)

DIAG_SCHEMA_VERSION = "maniskill_act_diagnosis_v1"


@dataclass(frozen=True)
class StepTelemetry:
    step: int
    tcp: list[float]
    obj: list[float]
    goal: list[float]
    dist_tcp_obj: float
    dist_xy: float
    z_diff: float
    dist_obj_goal: float
    dist_obj_goal_xy: float
    is_grasped: bool
    is_obj_placed: bool | None
    is_robot_static: bool | None
    arm_qvel: list[float]
    max_abs_arm_qvel: float
    chunk_query_id: int
    chunk_offset: int
    queue_length_before_action: int
    queue_length_after_action: int
    raw_action: list[float]
    applied_action: list[float]
    raw_norm: float
    saturated: bool
    phase: str


@dataclass(frozen=True)
class EpisodeDiagnosis:
    env_id: str
    seed: int
    steps: int
    success: bool
    terminated: bool
    truncated: bool
    ever_grasped: bool
    final_is_grasped: bool
    ever_placed: bool
    ever_static: bool
    first_placed_step: int | None
    placed_steps: int
    static_steps: int
    simultaneously_valid_steps: int
    max_qvel_while_placed: float | None
    min_dist_tcp_obj: float
    min_dist_obj_goal: float
    max_raw_norm: float
    saturation_ratio: float
    final_phase: str
    telemetry: tuple[StepTelemetry, ...]


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _vector(observation: Mapping[str, object], key: str, size: int) -> np.ndarray:
    extra = observation.get("extra")
    if not isinstance(extra, Mapping) or key not in extra:
        raise ValueError(f"observation 缺少 extra.{key}")
    array = np.asarray(extra[key], dtype=np.float32).reshape(-1)
    if array.size < size or not np.all(np.isfinite(array)):
        raise ValueError(f"extra.{key} 不可用")
    return array[:size].astype(float)


def _scalar_bool(observation: Mapping[str, object], key: str) -> bool:
    extra = observation.get("extra")
    if not isinstance(extra, Mapping) or key not in extra:
        return False
    array = np.asarray(extra[key]).reshape(-1)
    if array.size != 1:
        return False
    return bool(array[0])


def _classify_phase(
    tcp: np.ndarray, obj: np.ndarray, is_grasped: bool
) -> str:
    if is_grasped:
        return "carrying"
    delta = tcp - obj
    horizontal = float(np.linalg.norm(delta[:2]))
    z_diff = float(delta[2])
    if z_diff > 0.04:
        return "above"
    if horizontal > 0.06:
        return "approaching"
    if abs(z_diff) > 0.015:
        return "descending"
    return "close_contact"


def _info_flag(info: Mapping[str, object], key: str) -> bool | None:
    if key not in info:
        return None
    array = np.asarray(info[key]).reshape(-1)
    if array.size != 1:
        return None
    return bool(array[0])


def _agent_qvel(observation: Mapping[str, object], size: int = 7) -> np.ndarray:
    agent = observation.get("agent")
    if not isinstance(agent, Mapping) or "qvel" not in agent:
        return np.zeros(size, dtype=float)
    array = np.asarray(agent["qvel"], dtype=np.float32).reshape(-1)
    if array.size < size:
        return np.zeros(size, dtype=float)
    return array[:size].astype(float)


def _capture_step(
    observation: Mapping[str, object],
    policy: LeRobotACTPolicy,
    step: int,
    info: Mapping[str, object],
) -> StepTelemetry:
    tcp = _vector(observation, "tcp_pose", 3)
    obj = _vector(observation, "obj_pose", 3)
    goal = _vector(observation, "goal_pos", 3)
    is_grasped = _scalar_bool(observation, "is_grasped")
    arm_qvel = _agent_qvel(observation, 7)
    raw = (
        policy.last_raw_action
        if policy.last_raw_action is not None
        else np.zeros(4, dtype=np.float32)
    )
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    applied = np.clip(raw, -ACTION_SPACE_BOUND, ACTION_SPACE_BOUND)
    return StepTelemetry(
        step=step,
        tcp=tcp.tolist(),
        obj=obj.tolist(),
        goal=goal.tolist(),
        dist_tcp_obj=float(np.linalg.norm(tcp - obj)),
        dist_xy=float(np.linalg.norm(tcp[:2] - obj[:2])),
        z_diff=float(tcp[2] - obj[2]),
        dist_obj_goal=float(np.linalg.norm(obj - goal)),
        dist_obj_goal_xy=float(np.linalg.norm(obj[:2] - goal[:2])),
        is_grasped=is_grasped,
        is_obj_placed=_info_flag(info, "is_obj_placed"),
        is_robot_static=_info_flag(info, "is_robot_static"),
        arm_qvel=arm_qvel.tolist(),
        max_abs_arm_qvel=float(np.max(np.abs(arm_qvel))),
        chunk_query_id=policy.chunk_query_id,
        chunk_offset=policy.chunk_offset,
        queue_length_before_action=policy.queue_length_before_action,
        queue_length_after_action=policy.queue_length_after_action,
        raw_action=raw.tolist(),
        applied_action=applied.tolist(),
        raw_norm=float(np.linalg.norm(raw)),
        saturated=bool(np.any(np.abs(raw) >= ACTION_SPACE_BOUND - 1e-6)),
        phase=_classify_phase(tcp, obj, is_grasped),
    )


def diagnose_episode(
    env_id: str,
    seed: int,
    policy: LeRobotACTPolicy,
    *,
    max_steps: int,
    env_factory: Callable[[ManiSkillTaskConfig], object],
) -> EpisodeDiagnosis:
    config = ManiSkillTaskConfig(env_id=env_id, seed=seed, max_steps=max_steps)
    env: Any | None = None
    try:
        env = env_factory(config)
        observation, _ = env.reset(seed=config.seed)
        if not isinstance(observation, Mapping):
            raise ValueError("observation 必须是 mapping")
        policy.reset()
        steps = 0
        success = False
        terminated = False
        truncated = False
        last_info: dict[str, object] = {}
        telemetry: list[StepTelemetry] = []
        for step in range(1, config.max_steps + 1):
            telemetry.append(
                _capture_step(observation, policy, step, last_info)
            )
            action = policy.act(observation)
            observation, _, raw_terminated, raw_truncated, info = env.step(
                action
            )
            last_info = info if isinstance(info, Mapping) else {}
            steps = step
            terminated = bool(np.asarray(raw_terminated).reshape(-1)[0])
            truncated = bool(np.asarray(raw_truncated).reshape(-1)[0])
            success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
            if success or terminated or truncated:
                break
        ever_grasped = any(item.is_grasped for item in telemetry)
        placed_items = [item for item in telemetry if item.is_obj_placed]
        static_items = [item for item in telemetry if item.is_robot_static]
        simultaneous = [
            item
            for item in telemetry
            if item.is_obj_placed and item.is_robot_static
        ]
        min_dist_tcp_obj = min(item.dist_tcp_obj for item in telemetry)
        min_dist_obj_goal = min(item.dist_obj_goal for item in telemetry)
        max_raw_norm = max(item.raw_norm for item in telemetry)
        saturation_ratio = sum(item.saturated for item in telemetry) / len(
            telemetry
        )
        max_qvel_while_placed = (
            max(item.max_abs_arm_qvel for item in placed_items)
            if placed_items
            else None
        )
        return EpisodeDiagnosis(
            env_id=env_id,
            seed=seed,
            steps=steps,
            success=success,
            terminated=terminated,
            truncated=truncated,
            ever_grasped=ever_grasped,
            final_is_grasped=bool(telemetry and telemetry[-1].is_grasped),
            ever_placed=bool(placed_items),
            ever_static=bool(static_items),
            first_placed_step=(
                min(item.step for item in placed_items) if placed_items else None
            ),
            placed_steps=len(placed_items),
            static_steps=len(static_items),
            simultaneously_valid_steps=len(simultaneous),
            max_qvel_while_placed=(
                float(max_qvel_while_placed)
                if max_qvel_while_placed is not None
                else None
            ),
            min_dist_tcp_obj=float(min_dist_tcp_obj),
            min_dist_obj_goal=float(min_dist_obj_goal),
            max_raw_norm=float(max_raw_norm),
            saturation_ratio=float(saturation_ratio),
            final_phase=telemetry[-1].phase if telemetry else "none",
            telemetry=tuple(telemetry),
        )
    finally:
        if env is not None:
            env.close()


def _checked_seeds(values: Sequence[int]) -> list[int]:
    if not values:
        raise ValueError("seeds 不能为空")
    checked: list[int] = []
    for seed in values:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seeds 必须是非负整数")
        checked.append(seed)
    if len(set(checked)) != len(checked):
        raise ValueError("seeds 不能重复")
    return checked


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能序列化 {type(value).__name__}")


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第 13 周 ACT 失败相位诊断"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/maniskill_act/week13"),
    )
    parser.add_argument(
        "--env-id", type=str, default="PickCube-v1",
        choices=("PickCube-v1", "PushCube-v1"),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="100,102,104,111,115",
        help="逗号分隔的 seed 列表",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--n-action-steps", type=int, default=None,
        help="诊断迭代：覆盖 ACT 每次消费的动作数（默认沿用保存配置）",
    )
    parser.add_argument(
        "--action-smoothing", type=float, default=1.0,
        help="rollout 位置增量的 EMA 系数（(0,1]），1.0 表示不平滑",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON 报告输出路径（默认写入 output-dir/diagnosis/）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _require_positive_int(args.max_steps, "max_steps")
        if args.env_id not in ("PickCube-v1", "PushCube-v1"):
            raise ValueError("env_id 不受支持")
        seeds = _checked_seeds(
            [int(item) for item in args.seeds.split(",") if item.strip()]
        )
        if args.device not in ("mps", "cpu", "cuda"):
            raise ValueError("device 只支持 mps、cuda 或 cpu")
        # 提前解析绝对路径，避免 ManiSkill 改变 CWD 后相对路径失效。
        output_dir = Path(args.output_dir).expanduser().resolve()
        report_path = (
            Path(args.report).expanduser().resolve()
            if args.report is not None
            else output_dir / "diagnosis" / f"failure_phase_{args.env_id}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        policies = load_lerobot_act_checkpoints(
            output_dir,
            device=args.device,
            n_action_steps=args.n_action_steps,
            action_smoothing=args.action_smoothing,
            tasks=[args.env_id],
        )
        policy = policies[args.env_id]
        diagnoses = [
            diagnose_episode(
                args.env_id,
                seed,
                policy,
                max_steps=args.max_steps,
                env_factory=make_maniskill_env,
            )
            for seed in seeds
        ]
        payload = {
            "schema_version": DIAG_SCHEMA_VERSION,
            "output_dir": str(output_dir),
            "env_id": args.env_id,
            "n_action_steps": args.n_action_steps,
            "episodes": [asdict(item) for item in diagnoses],
        }
        _atomic_write_json(report_path, payload)
        for item in diagnoses:
            print(
                f"seed {item.seed}: success={item.success} "
                f"steps={item.steps}/{args.max_steps} "
                f"grasped={item.ever_grasped} "
                f"min_dist_tcp_obj={item.min_dist_tcp_obj:.3f} "
                f"min_dist_obj_goal={item.min_dist_obj_goal:.3f} "
                f"max_raw_norm={item.max_raw_norm:.2f} "
                f"saturation={item.saturation_ratio:.0%} "
                f"final_phase={item.final_phase}"
            )
        print(f"报告已写入：{report_path}")
    except (
        LeRobotACTError,
        BehaviorCloningError,
        ManiSkillUnavailableError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ACT 诊断失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
