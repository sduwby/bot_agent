"""PickCube 视觉 ACT + 状态 ACT 分层混合 rollout 评测。

方案 C 验证：视觉 ACT 负责"接近+对准"（approach→align），当 TCP 进入
抓取窗口（水平偏差 <= ALIGN_XY 且 z 容差内）后切换状态输入 ACT
（第 13 周 week13-dagger-train-r3，已验 PickCube 96%）完成抓取+抬升+放置。

对比目标：grasp-30k 纯视觉 ACT 分布内 53.3%、分布外 46.7%。

输出：分布内/外各 30 seed 的成功率 + 各段通过率漏斗。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot_act import load_lerobot_act_checkpoints
from maniskill_intro import ManiSkillTaskConfig
from perturbation import PerturbationConfig, make_perturbed_render_env
from visual_act import load_visual_act_checkpoints

VISUAL_CHECKPOINT = Path(
    "runs/maniskill_act/week17-pick-grasp-30k"
).expanduser().resolve()
STATE_CHECKPOINT = Path(
    "runs/maniskill_act/week13-dagger-train-r3"
).expanduser().resolve()
RESOLUTION = 128
ALIGN_XY = 0.04          # TCP-cube 水平偏差进入此值即切换
GRASP_Z_TOL = 0.04       # z 容差
APPROACH_DIST = 0.10
LIFT_Z = 0.05
MAX_STEPS = 50
OUTPUT = Path("runs/maniskill_act/week18-hybrid.json").expanduser().resolve()


class ComboExpert:
    """切换后接管：强制下探到 cube 表面再闭合，抓取后抬升+静止。

    基于诊断：视觉 ACT 常在 z≈+0.03（cube 上方 3cm）就闭合并抬走，从不下探
    到能夹住的高度。此专家强制把 TCP 降到 obj 表面（z 容差 0.01）后再闭爪，
    抓到后由 RecoveryExpert 抬升+静止。几何驱动、不依赖训练分布。
    """

    DESCEND_Z_TOL = 0.012  # 降到 cube 表面此距离内才闭合

    def __init__(self) -> None:
        from dagger_collect import GraspRecoveryExpert, RecoveryExpert

        self._grasp = GraspRecoveryExpert()
        self._lift = RecoveryExpert()

    def reset(self) -> None:
        self._grasp.reset() if hasattr(self._grasp, "reset") else None

    def act(self, observation: dict[str, object]) -> np.ndarray:
        if self._scalar_bool(observation, "is_grasped"):
            return self._lift.act(observation)
        tcp = self._vec(observation, "tcp_pose", 3)
        obj = self._vec(observation, "obj_pose", 3)
        z_diff = float(tcp[2] - obj[2])
        xy = float(np.linalg.norm(tcp[:2] - obj[:2]))
        if z_diff > self.DESCEND_Z_TOL:
            # 还太高：只下降（水平保持，闭爪）。
            delta = np.clip((obj - tcp) / 0.10, -1.0, 1.0)
            delta[0] = delta[1] = 0.0
            return np.concatenate((delta, [-1.0])).astype(np.float32)
        # 已到表面：交给 GraspRecoveryExpert（会保持水平修正+闭爪）。
        return self._grasp.act(observation)

    @staticmethod
    def _vec(obs: dict[str, object], key: str, size: int) -> np.ndarray:
        extra = obs.get("extra")
        if not isinstance(extra, dict) or key not in extra:
            return np.zeros(size, dtype=float)
        return np.asarray(extra[key], np.float32).reshape(-1)[:size].astype(float)

    @staticmethod
    def _scalar_bool(obs: dict[str, object], key: str) -> bool:
        extra = obs.get("extra")
        if not isinstance(extra, dict) or key not in extra:
            return False
        return bool(np.asarray(extra[key]).reshape(-1)[0])


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


def _bool_flag(info: dict[str, object], key: str) -> bool:
    try:
        return bool(np.asarray(info.get(key, False)).reshape(-1)[0])
    except (ValueError, IndexError, TypeError):
        return False


def hybrid_episode(
    *,
    seed: int,
    perturbed: bool,
    visual_policy: object,
    switch_policy: object,
) -> dict[str, object]:
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
        visual_policy.reset()
        switch_policy.reset()
        switched = False
        approach = False
        align = False
        grasp = False
        lift = False
        place = False
        obj_z0: float | None = None
        for step in range(1, MAX_STEPS + 1):
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

            if not switched and align:
                # 视觉已完成对准，切换到状态 ACT。
                switched = True
                switch_policy.reset()
            if switched:
                action = switch_policy.act(obs)
            else:
                action = visual_policy.act(obs, env.render())

            result = env.step(action)
            obs, _, _, _, info = result
            if not isinstance(info, dict):
                info = {}
            if _bool_flag(info, "is_obj_placed") and _bool_flag(
                info, "is_robot_static"
            ):
                place = True
                break
            if _bool_flag(info, "success"):
                place = True
                break
    finally:
        env.close()
    return {
        "seed": seed,
        "switched": switched,
        "approach": approach,
        "align": align,
        "grasp": grasp,
        "lift": lift,
        "place": place,
    }


def main() -> int:
    visual_policies = load_visual_act_checkpoints(
        VISUAL_CHECKPOINT, device="mps", resolution=RESOLUTION, tasks=("PickCube-v1",)
    )
    state_policies = load_lerobot_act_checkpoints(
        STATE_CHECKPOINT, device="mps", tasks=("PickCube-v1",)
    )
    vp = visual_policies["PickCube-v1"]
    sp = state_policies["PickCube-v1"]
    expert = ComboExpert()

    report: dict[str, object] = {
        "visual_checkpoint": str(VISUAL_CHECKPOINT),
        "state_checkpoint": str(STATE_CHECKPOINT),
        "resolution": RESOLUTION,
        "align_xy": ALIGN_XY,
        "switch_modes": {
            "state_act": "切换后由第 13 周状态 ACT 接管",
            "combo_expert": "切换后由 GraspRecoveryExpert+RecoveryExpert 规则接管",
        },
        "splits": {},
    }
    for split_name, seeds, perturbed in (
        ("in_distribution", range(500, 530), False),
        ("out_of_distribution", range(600, 630), True),
    ):
        mode_results: dict[str, object] = {}
        for mode, switch in (("state_act", sp), ("combo_expert", expert)):
            results = [
                hybrid_episode(
                    seed=seed, perturbed=perturbed, visual_policy=vp,
                    switch_policy=switch,
                )
                for seed in seeds
            ]
            funnel = {
                stage: sum(1 for r in results if r[stage]) / len(results)
                for stage in ("approach", "align", "grasp", "lift", "place")
            }
            mode_results[mode] = {
                "n": len(results),
                "funnel": funnel,
                "episodes": results,
            }
            print(f"=== {split_name} / {mode}（{len(results)} seed）===")
            for stage in ("approach", "align", "grasp", "lift", "place"):
                n = sum(1 for r in results if r[stage])
                print(f"  {stage:<10} {n:>3}/{len(results)} ({n/len(results):.1%})")
            print(
                f"  switched: "
                f"{sum(1 for r in results if r['switched'])}/{len(results)}"
            )
        report["splits"][split_name] = mode_results

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告写入：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
