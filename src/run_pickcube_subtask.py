"""第 17/18 周 PickCube 视觉 ACT 子任务成功率拆分。

把 PickCube 流程拆为五段并统计每段通过率，量化瓶颈到底在哪一段：

- approach  接近：TCP 到达 cube 上方附近（3D 距离显著缩小）
- align     对准：TCP 与 cube 水平偏差进入抓取窗口（xy <= 0.04）
- grasp     抓取：is_grasped=True
- lift      抬升：抓取后 cube 抬离桌面（obj_z 高于初始 + 阈值）
- place     放置：is_obj_placed 且 is_robot_static（成功）

输出：每 seed 各段达成情况 + 汇总漏斗（approach → align → grasp → lift → place）。
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

CHECKPOINT = Path("runs/maniskill_act/week17-pick-grasp-30k").expanduser().resolve()
RESOLUTION = 128
# 分布内：500-529；分布外：600-629（各 30 seed）。用扰动场景更贴近失败分布。
SEEDS = tuple(range(600, 630))
PERTURBED = True
OUTPUT = Path("runs/maniskill_act/week18-subtask.json").expanduser().resolve()

APPROACH_DIST = 0.10     # TCP-cube 3D 距离低于此视为"接近成功"
ALIGN_XY = 0.04          # TCP-cube 水平偏差低于此视为"对准"
GRASP_Z_TOL = 0.04       # 对准阶段 z 容差（对齐到 cube 高度）
LIFT_Z = 0.05            # 抓取后 obj_z 高于初始值此阈值视为"抬升"


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


def _subtask_funnel(checkpoint: Path, seed: int) -> dict[str, object]:
    policy = load_visual_act_checkpoints(
        checkpoint, device="mps", resolution=RESOLUTION, tasks=("PickCube-v1",)
    )["PickCube-v1"]
    config = ManiSkillTaskConfig(env_id="PickCube-v1", seed=seed)
    if PERTURBED:
        from perturbation import PerturbationConfig, make_perturbed_render_env

        perturbation = PerturbationConfig(
            background=True, lighting=True, pose=True
        )
        env = make_perturbed_render_env(config, perturbation)
    else:
        from lerobot_pipeline import make_render_env

        env = make_render_env(config)
    try:
        obs, _ = env.reset(seed=seed)
        policy.reset()
        approach = False
        align = False
        grasp = False
        lift = False
        place = False
        obj_z0: float | None = None
        for step in range(1, config.max_steps + 1):
            tcp = _vec(obs, "tcp_pose", 3)
            obj = _vec(obs, "obj_pose", 3)
            if obj_z0 is None:
                obj_z0 = float(obj[2])
            dist = float(np.linalg.norm(tcp - obj))
            xy_dist = float(np.linalg.norm(tcp[:2] - obj[:2]))
            if dist <= APPROACH_DIST:
                approach = True
            if xy_dist <= ALIGN_XY and abs(tcp[2] - obj[2]) <= GRASP_Z_TOL:
                align = True
            is_grasped = _scalar(obs, "is_grasped")
            if is_grasped:
                grasp = True
            if grasp and float(obj[2]) - (obj_z0 or 0.0) >= LIFT_Z:
                lift = True
            action = policy.act(obs, env.render())
            result = env.step(action)
            obs, _, _, _, info = result
            if not isinstance(info, dict):
                info = {}
            is_placed = bool(np.asarray(info.get("is_obj_placed", False)).reshape(-1)[0])
            is_static = bool(np.asarray(info.get("is_robot_static", False)).reshape(-1)[0])
            if is_placed and is_static:
                place = True
                break
            if bool(np.asarray(info.get("success", False)).reshape(-1)[0]):
                place = True
                break
    finally:
        env.close()
    return {
        "seed": seed,
        "approach": approach,
        "align": align,
        "grasp": grasp,
        "lift": lift,
        "place": place,
    }


def main() -> int:
    results = [_subtask_funnel(CHECKPOINT, seed) for seed in SEEDS]
    summary = {
        "checkpoint": str(CHECKPOINT),
        "resolution": RESOLUTION,
        "perturbed": PERTURBED,
        "seeds": list(SEEDS),
        "n": len(results),
        "funnel": {
            stage: sum(1 for r in results if r[stage]) / len(results)
            for stage in ("approach", "align", "grasp", "lift", "place")
        },
        "episodes": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"checkpoint: {CHECKPOINT.name}（{'扰动' if PERTURBED else '分布内'}）")
    print(f"{'阶段':<10} {'通过数':>6} {'通过率':>8}")
    for stage in ("approach", "align", "grasp", "lift", "place"):
        n = sum(1 for r in results if r[stage])
        print(f"{stage:<10} {n:>6}/{len(results)} {n/len(results):>8.1%}")
    print(f"\n报告写入：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
