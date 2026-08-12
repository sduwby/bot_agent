"""测量视觉 ACT 对齐精度：每 seed 能达到的最小 TCP-cube 水平距离。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from maniskill_intro import ManiSkillTaskConfig
from perturbation import PerturbationConfig, make_perturbed_render_env
from visual_act import load_visual_act_checkpoints

CHECKPOINT = Path("runs/maniskill_act/week17-pick-grasp-30k").expanduser().resolve()
RESOLUTION = 128
OUTPUT = Path("runs/maniskill_act/week18-align-precision.json").expanduser().resolve()


def _vec(obs: dict[str, object], key: str, size: int) -> np.ndarray:
    return np.asarray(obs["extra"][key], np.float32).reshape(-1)[:size].astype(float)


def _is_grasped(obs: dict[str, object]) -> bool:
    value = obs["extra"]["is_grasped"]
    return bool(np.asarray(value).reshape(-1)[0])


def _min_align(seed: int, perturbed: bool) -> dict[str, object]:
    policy = load_visual_act_checkpoints(
        CHECKPOINT, device="mps", resolution=RESOLUTION, tasks=("PickCube-v1",)
    )["PickCube-v1"]
    config = ManiSkillTaskConfig(env_id="PickCube-v1", seed=seed)
    if perturbed:
        env = make_perturbed_render_env(
            config, PerturbationConfig(background=True, lighting=True, pose=True)
        )
    else:
        from lerobot_pipeline import make_render_env

        env = make_render_env(config)
    try:
        obs, _ = env.reset(seed=seed)
        policy.reset()
        min_xy = 1e9
        min_xy_z: float | None = None
        min_xy_step = 0
        grasped = False
        for step in range(1, 51):
            tcp = _vec(obs, "tcp_pose", 3)
            obj = _vec(obs, "obj_pose", 3)
            xy = float(np.linalg.norm(tcp[:2] - obj[:2]))
            if xy < min_xy:
                min_xy = xy
                min_xy_z = float(abs(tcp[2] - obj[2]))
                min_xy_step = step
            if _is_grasped(obs):
                grasped = True
                break
            obs, _, _, _, _ = env.step(policy.act(obs, env.render()))
    finally:
        env.close()
    return {
        "seed": seed,
        "min_xy": round(min_xy, 4),
        "min_xy_z": round(min_xy_z, 4) if min_xy_z is not None else None,
        "min_xy_step": min_xy_step,
        "grasped": grasped,
    }


def main() -> int:
    report: dict[str, object] = {"checkpoint": str(CHECKPOINT)}
    for split_name, seeds, perturbed in (
        ("in_distribution", range(500, 530), False),
        ("out_of_distribution", range(600, 630), True),
    ):
        rows = [_min_align(seed, perturbed) for seed in seeds]
        xys = [r["min_xy"] for r in rows]
        report[split_name] = {
            "n": len(rows),
            "min_xy": {
                "p10": round(float(np.percentile(xys, 10)), 4),
                "median": round(float(np.median(xys)), 4),
                "p90": round(float(np.percentile(xys, 90)), 4),
                "max": round(float(max(xys)), 4),
            },
            "le_0_02": sum(1 for r in rows if r["min_xy"] <= 0.02),
            "le_0_015": sum(1 for r in rows if r["min_xy"] <= 0.015),
            "le_0_01": sum(1 for r in rows if r["min_xy"] <= 0.01),
            "grasped": sum(1 for r in rows if r["grasped"]),
            "worst": sorted(rows, key=lambda r: r["min_xy"], reverse=True)[:8],
        }
        print(f"=== {split_name} ===")
        print(f"  min_xy p10/中位/p90/max = "
              f"{report[split_name]['min_xy']['p10']}/"
              f"{report[split_name]['min_xy']['median']}/"
              f"{report[split_name]['min_xy']['p90']}/"
              f"{report[split_name]['min_xy']['max']}")
        print(f"  <=0.02: {report[split_name]['le_0_02']}  "
              f"<=0.015: {report[split_name]['le_0_015']}  "
              f"<=0.01: {report[split_name]['le_0_01']}  抓取: {report[split_name]['grasped']}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告写入：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
