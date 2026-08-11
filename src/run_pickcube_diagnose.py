"""第 17 周 PickCube 视觉 ACT 失败相位诊断。

对分布外失败 episode 逐帧记录 grasp/placed/static/qvel/动作证据，
判断 PickCube 视觉 ACT 失败卡在哪个环节，为 4060 重训提供方向。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from maniskill_intro import ManiSkillTaskConfig
from perturbation import PerturbationConfig, make_perturbed_render_env
from visual_act import load_visual_act_checkpoints

CHECKPOINT = Path("runs/maniskill_act/week17-c8-20k").expanduser().resolve()
RESOLUTION = 128
SEEDS = tuple(range(600, 610))
OUTPUT = Path("runs/maniskill_act/week17-diagnosis.json").expanduser().resolve()


def _vec(obs: dict[str, object], key: str, size: int) -> np.ndarray:
    extra = obs.get("extra")
    if not isinstance(extra, dict) or key not in extra:
        return np.zeros(size, dtype=float)
    return np.asarray(extra[key], dtype=np.float32).reshape(-1)[:size].astype(float)


def _scalar(obs: dict[str, object], key: str) -> bool:
    extra = obs.get("extra")
    if not isinstance(extra, dict) or key not in extra:
        return False
    array = np.asarray(extra[key]).reshape(-1)
    return bool(array[0]) if array.size else False


def _agent_qvel(obs: dict[str, object], size: int = 7) -> np.ndarray:
    agent = obs.get("agent")
    if not isinstance(agent, dict) or "qvel" not in agent:
        return np.zeros(size, dtype=float)
    return np.asarray(agent["qvel"], dtype=np.float32).reshape(-1)[:size].astype(float)


def diagnose(checkpoint: Path, seed: int) -> dict[str, object]:
    policy = load_visual_act_checkpoints(
        checkpoint, device="mps", resolution=RESOLUTION, tasks=("PickCube-v1",)
    )["PickCube-v1"]
    config = ManiSkillTaskConfig(env_id="PickCube-v1", seed=seed)
    env = make_perturbed_render_env(
        config, PerturbationConfig(background=True, lighting=True, pose=True)
    )
    rows: list[dict[str, object]] = []
    try:
        obs, _ = env.reset(seed=seed)
        policy.reset()
        ever_grasped = False
        ever_placed = False
        ever_static = False
        first_grasp_step: int | None = None
        first_placed_step: int | None = None
        placed_steps = 0
        static_steps = 0
        simultaneous = 0
        max_qvel_while_placed: float | None = None
        success = False
        truncated = False
        for step in range(1, config.max_steps + 1):
            tcp = _vec(obs, "tcp_pose", 3)
            obj = _vec(obs, "obj_pose", 3)
            goal = _vec(obs, "goal_pos", 3)
            is_grasped = _scalar(obs, "is_grasped")
            qvel = _agent_qvel(obs, 7)
            dist_obj_goal = float(np.linalg.norm(obj - goal))
            action = policy.act(obs, env.render())
            result = env.step(action)
            obs, _, _, _, info = result
            if not isinstance(info, dict):
                info = {}
            is_obj_placed = bool(np.asarray(info.get("is_obj_placed", False)).reshape(-1)[0])
            is_robot_static = bool(np.asarray(info.get("is_robot_static", False)).reshape(-1)[0])
            success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
            truncated = bool(np.asarray(info.get("elapsed_steps", 0)).reshape(-1)[0] >= config.max_steps) or bool(
                np.asarray(result[4].get("elapsed_steps", 0)).reshape(-1)[0] >= config.max_steps
            )
            if is_grasped and not ever_grasped:
                first_grasp_step = step
            ever_grasped = ever_grasped or is_grasped
            if is_obj_placed and not ever_placed:
                first_placed_step = step
            ever_placed = ever_placed or is_obj_placed
            ever_static = ever_static or is_robot_static
            if is_obj_placed:
                placed_steps += 1
                if is_robot_static:
                    simultaneous += 1
                m = float(np.max(np.abs(qvel)))
                max_qvel_while_placed = m if max_qvel_while_placed is None else max(m, max_qvel_while_placed)
            if is_robot_static:
                static_steps += 1
            rows.append(
                {
                    "step": step,
                    "is_grasped": is_grasped,
                    "is_obj_placed": is_obj_placed,
                    "is_robot_static": is_robot_static,
                    "dist_obj_goal": round(dist_obj_goal, 4),
                    "z_diff": round(float(tcp[2] - obj[2]), 4),
                    "max_abs_qvel": round(float(np.max(np.abs(qvel))), 4),
                    "action": [round(float(v), 3) for v in np.asarray(action, dtype=float).reshape(-1)],
                }
            )
            if success:
                break
            if not success and bool(np.asarray(info.get("is_obj_placed", False)).reshape(-1)[0]) and step >= config.max_steps - 1:
                pass
        # 简化 truncated 判定：跑满 50 步即视为截断失败
        truncated = not success and len(rows) >= config.max_steps
    finally:
        env.close()
    return {
        "seed": seed,
        "success": success,
        "truncated": truncated,
        "steps": len(rows),
        "ever_grasped": ever_grasped,
        "ever_placed": ever_placed,
        "ever_static": ever_static,
        "first_grasp_step": first_grasp_step,
        "first_placed_step": first_placed_step,
        "placed_steps": placed_steps,
        "static_steps": static_steps,
        "simultaneously_valid_steps": simultaneous,
        "max_qvel_while_placed": max_qvel_while_placed,
        "frames": rows,
    }


def main() -> int:
    results = [diagnose(CHECKPOINT, seed) for seed in SEEDS]
    summary = {
        "checkpoint": str(CHECKPOINT),
        "resolution": RESOLUTION,
        "seeds": list(SEEDS),
        "episodes": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"诊断写入：{OUTPUT}")
    print(f"{'seed':>5} {'success':>7} {'grasp':>5} {'placed':>6} {'static':>6} {'simult':>6} {'firstG':>7} {'firstP':>7} {'maxQvPlaced':>11}")
    for r in results:
        print(
            f"{r['seed']:>5} {str(r['success']):>7} {str(r['ever_grasped']):>5} "
            f"{str(r['ever_placed']):>6} {str(r['ever_static']):>6} "
            f"{r['simultaneously_valid_steps']:>6} {str(r['first_grasp_step']):>7} "
            f"{str(r['first_placed_step']):>7} {str(r['max_qvel_while_placed']):>11}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
