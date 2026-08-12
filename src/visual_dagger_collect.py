"""视觉 ACT 的 DAgger recovery 补采：用当前视觉 ACT 诱导末端失败态，
RecoveryExpert 标注"抬升+静止"动作并记录 live RGB 图像，供合并重训。

与 ``dagger_collect.py`` 的区别：策略为 ``VisualACTPolicy``（live RGB 渲染），
记录的监督信号为 128×128 图像 + robot_state[5] + env_state[13] + action[4]。
只用于训练数据标注；rollout/评测阶段不使用 scripted 接管。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from behavior_cloning import BehaviorCloningError, observation_to_state
from dagger_collect import (
    RECOVERY_SCALE,
    SETTLE_DIST,
    GraspRecoveryExpert,
    RecoveryExpert,
    _obj_goal_distance,
    _scalar_bool,
    _vector,
)
from lerobot_act import (
    ACTION_DIM,
    ENV_STATE_DIM,
    ROBOT_STATE_DIM,
    split_lerobot_act_state,
)
from maniskill_intro import ManiSkillTaskConfig, ManiSkillUnavailableError
from visual_act import (
    VisualACTError,
    VisualTaskData,
    _image_hwc,
    _resize_image,
    load_visual_act_checkpoints,
)

VISUAL_DAGGER_SCHEMA_VERSION = "maniskill_visual_dagger_recovery_v1"

# "grasp" 接管模式：未抓取但 TCP 已到 cube 高度且水平有偏差即视为抓取窗口。
# 水平窗口由 --align-zone-min/--align-zone-max 控制，不再用固定 GRASP_XY_ZONE。
GRASP_Z_TOL = 0.04


def _split_observation(
    observation: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """从 observation 拆分 robot_state[5] 与 env_state[13]。"""
    state = observation_to_state(observation)
    robot, env_state = split_lerobot_act_state(state)
    return (
        np.asarray(robot, dtype=np.float32).reshape(-1),
        np.asarray(env_state, dtype=np.float32).reshape(-1),
    )


def collect_visual_dagger_episodes(
    policy_dir: Path,
    *,
    seeds: Sequence[int],
    resolution: int = 128,
    env_factory: Callable[
        [ManiSkillTaskConfig], object
    ] | None = None,
    device: str = "cuda",
    max_steps: int = 50,
    takeover_zone: float = 0.10,
    min_segment: int = 2,
    env_id: str = "PickCube-v1",
    takeover: str = "lift",
    align_zone_min: float = 0.01,
    align_zone_max: float = 0.05,
) -> dict[str, object]:
    """DAgger 接管模式：视觉 ACT rollout 到达末端停滞区后，专家接管到成功。

    记录接管点到成功/终止的完整 (image, robot_state, env_state, expert_action)
    轨迹，使补采数据同时包含 move 与 settle 阶段。

    ``takeover`` 指定接管策略与恢复专家：
    - "lift"：已抓取且 |obj-goal| <= takeover_zone 后由 ``RecoveryExpert``
      标注抬升+静止（第一轮补采）；
    - "grasp"：未抓取但 TCP 已到 cube 高度且水平有偏差
      （``align_zone_min <= tcp_xy-obj_xy <= align_zone_max``）后由
      ``GraspRecoveryExpert`` 标注水平修正+闭爪+抬升+静止（第三轮补采）。
      ``align_zone_min``/``align_zone_max`` 限定"已贴近但水平偏了"的接管窗口，
      避免把已对准或离太远的样本算进来。
    """
    if not seeds:
        raise ValueError("seeds 不能为空")
    checked_seeds: list[int] = []
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seeds 必须是非负整数")
        checked_seeds.append(seed)
    if len(set(checked_seeds)) != len(checked_seeds):
        raise ValueError("seeds 不能重复")
    if takeover not in ("lift", "grasp"):
        raise ValueError("takeover 只支持 lift 或 grasp")
    if takeover_zone <= 0:
        raise ValueError("takeover_zone 必须为正数")
    if max_steps <= 0:
        raise ValueError("max_steps 必须为正数")
    if resolution <= 0:
        raise ValueError("resolution 必须是正整数")
    if not (0 <= align_zone_min < align_zone_max):
        raise ValueError(
            "align_zone_min 必须 >= 0 且小于 align_zone_max"
        )

    if env_factory is None:
        from lerobot_pipeline import make_render_env

        env_factory = make_render_env

    policies = load_visual_act_checkpoints(
        policy_dir, device=device, resolution=resolution, tasks=[env_id]
    )
    policy = policies[env_id]
    expert: RecoveryExpert | GraspRecoveryExpert = (
        GraspRecoveryExpert() if takeover == "grasp" else RecoveryExpert()
    )
    episode_images: list[np.ndarray] = []
    episode_robot: list[np.ndarray] = []
    episode_env: list[np.ndarray] = []
    episode_actions: list[np.ndarray] = []
    episode_lengths: list[int] = []
    total_labeled = 0
    total_rollout_steps = 0
    takeover_episodes = 0

    for seed in checked_seeds:
        config = ManiSkillTaskConfig(env_id=env_id, seed=seed, max_steps=max_steps)
        env: Any | None = None
        segment_images: list[np.ndarray] = []
        segment_robot: list[np.ndarray] = []
        segment_env: list[np.ndarray] = []
        segment_actions: list[np.ndarray] = []
        taken_over = False
        try:
            env = env_factory(config)
            observation, _ = env.reset(seed=config.seed)
            if not isinstance(observation, Mapping):
                raise ValueError("observation 必须是 mapping")
            policy.reset()
            for _ in range(1, config.max_steps + 1):
                total_rollout_steps += 1
                image_raw = _image_hwc(np.asarray(env.render()))
                image_float = _resize_image(image_raw, resolution)
                image = np.clip(
                    np.round(np.transpose(image_float, (1, 2, 0)) * 255.0),
                    0,
                    255,
                ).astype(np.uint8)
                grasped = _scalar_bool(observation, "is_grasped")
                distance = _obj_goal_distance(observation)
                if not taken_over:
                    if takeover == "grasp":
                        tcp = _vector(observation, "tcp_pose", 3)
                        obj = _vector(observation, "obj_pose", 3)
                        xy_dist = float(np.linalg.norm(tcp[:2] - obj[:2]))
                        aligned = (
                            abs(tcp[2] - obj[2]) <= GRASP_Z_TOL
                            and align_zone_min <= xy_dist <= align_zone_max
                        )
                        if not grasped and aligned:
                            taken_over = True
                            takeover_episodes += 1
                    elif grasped and distance <= takeover_zone:
                        taken_over = True
                        takeover_episodes += 1
                if taken_over:
                    expert_action = expert.act(observation)
                    if expert_action is None:
                        break
                    robot, env_state = _split_observation(observation)
                    segment_images.append(image)
                    segment_robot.append(robot)
                    segment_env.append(env_state)
                    segment_actions.append(expert_action)
                    action = expert_action
                else:
                    action = policy.act(observation, image)

                observation, _, terminated, truncated, info = env.step(action)
                terminated = bool(np.asarray(terminated).reshape(-1)[0])
                truncated = bool(np.asarray(truncated).reshape(-1)[0])
                success = bool(
                    np.asarray(info.get("success", False)).reshape(-1)[0]
                )
                if success or terminated or truncated:
                    break
            if len(segment_images) >= min_segment:
                episode_images.append(np.stack(segment_images))
                episode_robot.append(np.stack(segment_robot))
                episode_env.append(np.stack(segment_env))
                episode_actions.append(np.stack(segment_actions))
                episode_lengths.append(len(segment_images))
                total_labeled += len(segment_images)
        finally:
            if env is not None:
                env.close()

    return {
        "schema_version": VISUAL_DAGGER_SCHEMA_VERSION,
        "env_id": env_id,
        "policy_dir": str(Path(policy_dir).expanduser().resolve()),
        "resolution": resolution,
        "seeds": list(checked_seeds),
        "takeover_zone": takeover_zone,
        "align_zone_min": align_zone_min,
        "align_zone_max": align_zone_max,
        "min_segment": min_segment,
        "takeover": takeover,
        "takeover_episodes": takeover_episodes,
        "n_episodes": len(episode_lengths),
        "episode_lengths": episode_lengths,
        "total_labeled_transitions": total_labeled,
        "total_rollout_steps": total_rollout_steps,
        "images": (
            np.concatenate(episode_images, axis=0).astype(np.uint8, copy=False)
            if episode_images
            else np.empty((0, resolution, resolution, 3), dtype=np.uint8)
        ),
        "robot_state": (
            np.concatenate(episode_robot, axis=0)
            if episode_robot
            else np.empty((0, ROBOT_STATE_DIM), dtype=np.float32)
        ),
        "env_state": (
            np.concatenate(episode_env, axis=0)
            if episode_env
            else np.empty((0, ENV_STATE_DIM), dtype=np.float32)
        ),
        "actions": (
            np.concatenate(episode_actions, axis=0)
            if episode_actions
            else np.empty((0, ACTION_DIM), dtype=np.float32)
        ),
    }


def save_visual_dagger_episodes(payload: Mapping[str, object], path: Path) -> None:
    """写补采 npz（含图像）与同名 .json 元数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in ("images", "robot_state", "env_state", "actions")
    }
    json_path = path.with_suffix(".json")
    temporary = json_path.with_name(f".{json_path.name}.tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(json_path)
    np.savez_compressed(
        path,
        images=np.asarray(payload["images"], dtype=np.uint8),
        robot_state=np.asarray(payload["robot_state"], dtype=np.float32),
        env_state=np.asarray(payload["env_state"], dtype=np.float32),
        actions=np.asarray(payload["actions"], dtype=np.float32),
    )


def load_visual_dagger_episodes(path: Path) -> dict[str, object]:
    """读回视觉 DAgger 补采 npz 与元数据。"""
    path = Path(path)
    arrays = np.load(path)
    json_path = path.with_suffix(".json")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "schema_version": metadata["schema_version"],
        "env_id": metadata["env_id"],
        "resolution": int(metadata["resolution"]),
        "episode_lengths": list(metadata["episode_lengths"]),
        "images": arrays["images"],
        "robot_state": arrays["robot_state"],
        "env_state": arrays["env_state"],
        "actions": arrays["actions"],
        "n_episodes": int(metadata["n_episodes"]),
        "total_labeled_transitions": int(
            metadata["total_labeled_transitions"]
        ),
    }


def merge_visual_dagger_payloads(
    payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """合并多轮视觉补采 payload（episode 拼接），供 DAgger 多轮累积。"""
    if not payloads:
        raise ValueError("没有可合并的补采 payload")
    all_lengths: list[int] = []
    image_parts: list[np.ndarray] = []
    robot_parts: list[np.ndarray] = []
    env_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    for payload in payloads:
        lengths = list(payload["episode_lengths"])
        all_lengths.extend(lengths)
        image_parts.append(np.asarray(payload["images"], dtype=np.uint8))
        robot_parts.append(np.asarray(payload["robot_state"], dtype=np.float32))
        env_parts.append(np.asarray(payload["env_state"], dtype=np.float32))
        action_parts.append(np.asarray(payload["actions"], dtype=np.float32))
    return {
        "schema_version": VISUAL_DAGGER_SCHEMA_VERSION,
        "env_id": str(payloads[0]["env_id"]),
        "resolution": int(payloads[0]["resolution"]),
        "n_episodes": len(all_lengths),
        "episode_lengths": all_lengths,
        "total_labeled_transitions": sum(all_lengths),
        "images": np.concatenate(image_parts, axis=0),
        "robot_state": np.concatenate(robot_parts, axis=0),
        "env_state": np.concatenate(env_parts, axis=0),
        "actions": np.concatenate(action_parts, axis=0),
    }


def build_visual_dagger_transitions(
    base: VisualTaskData,
    dagger: Mapping[str, object],
    *,
    resolution: int,
    start_episode_id: int = 100,
    insert_after: int = 80,
) -> VisualTaskData:
    """把视觉补采 episode 插入原始 train 之后、原始 validation 之前。

    train = 原始前 insert_after 个 episode + 补采；validation = 原始剩余。
    补采图像按训练分辨率缩放并转 float[0,1]，与 ``load_visual_dataset`` 对齐。
    """
    from visual_act import VisualACTError

    if resolution <= 0:
        raise ValueError("resolution 必须是正整数")
    lengths = list(dagger["episode_lengths"])
    images = np.asarray(dagger["images"], dtype=np.uint8)
    robot_state = np.asarray(dagger["robot_state"], dtype=np.float32)
    env_state = np.asarray(dagger["env_state"], dtype=np.float32)
    actions = np.asarray(dagger["actions"], dtype=np.float32)
    if sum(lengths) != len(images):
        raise ValueError("dagger episode_lengths 与 images 不一致")
    if insert_after < 0 or insert_after > len(base.episode_ids):
        raise ValueError("insert_after 超出原始 episode 数")

    resized = np.stack(
        [_resize_image(image, resolution) for image in images],
        axis=0,
    ).astype(np.float32, copy=False)

    dagger_ids = range(start_episode_id, start_episode_id + len(lengths))
    orig_lengths = list(base.episode_lengths)
    orig_offsets = np.concatenate(([0], np.cumsum(orig_lengths, dtype=np.int64)))
    d_offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))

    new_ids: list[int] = []
    new_lengths: list[int] = []
    image_parts: list[np.ndarray] = []
    robot_parts: list[np.ndarray] = []
    env_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []

    def add_original(idx: int) -> None:
        new_ids.append(int(base.episode_ids[idx]))
        new_lengths.append(orig_lengths[idx])
        image_parts.append(base.images[orig_offsets[idx]: orig_offsets[idx + 1]])
        robot_parts.append(base.robot_state[orig_offsets[idx]: orig_offsets[idx + 1]])
        env_parts.append(base.env_state[orig_offsets[idx]: orig_offsets[idx + 1]])
        action_parts.append(base.actions[orig_offsets[idx]: orig_offsets[idx + 1]])

    def add_dagger(k: int) -> None:
        new_ids.append(dagger_ids[k])
        new_lengths.append(lengths[k])
        image_parts.append(resized[d_offsets[k]: d_offsets[k + 1]])
        robot_parts.append(robot_state[d_offsets[k]: d_offsets[k + 1]])
        env_parts.append(env_state[d_offsets[k]: d_offsets[k + 1]])
        action_parts.append(actions[d_offsets[k]: d_offsets[k + 1]])

    for idx in range(insert_after):
        add_original(idx)
    for k in range(len(lengths)):
        add_dagger(k)
    for idx in range(insert_after, len(base.episode_ids)):
        add_original(idx)

    return VisualTaskData(
        env_id=base.env_id,
        episode_ids=tuple(new_ids),
        episode_lengths=tuple(new_lengths),
        images=np.concatenate(image_parts, axis=0).astype(np.float32, copy=False),
        robot_state=np.concatenate(robot_parts, axis=0).astype(
            np.float32, copy=False
        ),
        env_state=np.concatenate(env_parts, axis=0).astype(np.float32, copy=False),
        actions=np.concatenate(action_parts, axis=0).astype(np.float32, copy=False),
    )


def parse_args(argv: list[str] | None = None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="视觉 ACT DAgger recovery 补采")
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="补采 npz 输出路径")
    parser.add_argument("--seeds", type=str,
                        default="600,601,602,603,604,605,606,607,608,609,"
                                "150,151,152,153,154,155,156,157,158,159")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--takeover-zone", type=float, default=0.10)
    parser.add_argument("--min-segment", type=int, default=2)
    parser.add_argument(
        "--align-zone-min", type=float, default=0.01,
        help="grasp 接管：已贴近 cube 的最小水平偏差（默认 0.01）",
    )
    parser.add_argument(
        "--align-zone-max", type=float, default=0.05,
        help="grasp 接管：已贴近 cube 的最大水平偏差（默认 0.05，"
             "替代原 GRASP_XY_ZONE=0.10）",
    )
    parser.add_argument("--takeover", type=str, default="lift",
                        choices=("lift", "grasp"),
                        help="接管策略：lift=已抓取补抬升，grasp=未抓上补闭爪")
    parser.add_argument("--task", type=str, default="PickCube-v1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = args.output.expanduser().resolve()
        policy_dir = args.policy_dir.expanduser().resolve()
        seeds = [
            int(item) for item in args.seeds.split(",") if item.strip()
        ]
        payload = collect_visual_dagger_episodes(
            policy_dir,
            seeds=seeds,
            resolution=args.resolution,
            device=args.device,
            max_steps=args.max_steps,
            takeover_zone=args.takeover_zone,
            min_segment=args.min_segment,
            env_id=args.task,
            takeover=args.takeover,
            align_zone_min=args.align_zone_min,
            align_zone_max=args.align_zone_max,
        )
        save_visual_dagger_episodes(payload, output)
        print(
            f"视觉 DAgger 补采完成（{payload['takeover']}）："
            f"{payload['n_episodes']} 个 episode，"
            f"{payload['total_labeled_transitions']} 条 recovery transition，"
            f"接管 episode {payload['takeover_episodes']}，"
            f"rollout {payload['total_rollout_steps']} 步"
        )
        print(f"写入：{output}（+ .json 元数据）")
    except (
        VisualACTError,
        BehaviorCloningError,
        ManiSkillUnavailableError,
        OSError,
        ValueError,
    ) as exc:
        print(f"视觉 DAgger 补采失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
