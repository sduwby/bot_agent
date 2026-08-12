"""DAgger recovery 补采：用当前 ACT 诱导末端失败态，RecoveryExpert 标注修正动作。

只用于训练数据标注；rollout/评测阶段不使用 scripted 接管。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from behavior_cloning import (
    BehaviorCloningError,
    observation_to_arm_qvel,
    observation_to_state,
)
from lerobot_act import (
    ACTION_DIM,
    LeRobotACTError,
    load_lerobot_act_checkpoints,
)
from maniskill_intro import (
    ManiSkillTaskConfig,
    ManiSkillUnavailableError,
    make_maniskill_env,
)

RECOVERY_SCALE = 0.10
SETTLE_DIST = 0.018
DAGGER_SCHEMA_VERSION = "maniskill_dagger_recovery_v1"


class RecoveryExpert:
    """cube 被抓取后，把 cube 送入 goal 并静止的恢复专家。

    规则即 scripted policy 的 lift_to_goal + settle，几何驱动、不依赖相位：
    - 未抓取：返回 None（不在 recovery 区，不标注）；
    - 抓取且 |obj-goal| >= SETTLE_DIST：move 到 goal（闭爪）；
    - 抓取且 |obj-goal| <  SETTLE_DIST：零增量保持（闭爪）。
    """

    def act(self, observation: Mapping[str, object]) -> np.ndarray | None:
        tcp = _vector(observation, "tcp_pose", 3)
        obj = _vector(observation, "obj_pose", 3)
        goal = _vector(observation, "goal_pos", 3)
        if not _scalar_bool(observation, "is_grasped"):
            return None
        if float(np.linalg.norm(obj - goal)) < SETTLE_DIST:
            return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        delta = np.clip((goal - tcp) / RECOVERY_SCALE, -1.0, 1.0)
        return np.concatenate((delta, [-1.0])).astype(np.float32)


class GraspRecoveryExpert:
    """从"到达 cube 高度但未抓上"状态恢复：水平修正 → 下降闭合 → 抬升 → 静止。

    未抓取时按两段式时序给出修正动作，避免"边下降边水平移动"导致又一次错过：

    - 水平偏差 ``xy_dist >= XY_ALIGN_TOL``：**只做水平修正**——水平分量把 TCP 移到
      obj 正上方（z 增量置 0，保持当前高度），夹爪保持闭合；
    - 水平偏差 ``xy_dist <  XY_ALIGN_TOL``：**下降+闭合**——朝 obj 移动（含 z），闭爪；
    - 已抓取：复用 ``RecoveryExpert``（抬升到 goal + 静止）。

    几何驱动、不依赖相位，始终返回动作。``GRASP_TOL`` 保留用于兼容调用方；
    本实现以水平/垂直分量分别驱动，不再用三维距离单一判断。
    """

    GRASP_TOL = 0.04
    XY_ALIGN_TOL = 0.02

    def __init__(self) -> None:
        self._lift_expert = RecoveryExpert()

    def act(self, observation: Mapping[str, object]) -> np.ndarray:
        if _scalar_bool(observation, "is_grasped"):
            return self._lift_expert.act(observation)
        tcp = _vector(observation, "tcp_pose", 3)
        obj = _vector(observation, "obj_pose", 3)
        xy_dist = float(np.linalg.norm(tcp[:2] - obj[:2]))
        delta = np.clip((obj - tcp) / RECOVERY_SCALE, -1.0, 1.0)
        if xy_dist >= self.XY_ALIGN_TOL:
            # 只做水平修正：z 增量置 0，夹爪保持闭合。
            delta[2] = 0.0
        return np.concatenate((delta, [-1.0])).astype(np.float32)


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


def _obj_goal_distance(observation: Mapping[str, object]) -> float:
    obj = _vector(observation, "obj_pose", 3)
    goal = _vector(observation, "goal_pos", 3)
    return float(np.linalg.norm(obj - goal))


def collect_dagger_episodes(
    policy_dir: Path,
    *,
    seeds: Sequence[int],
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    device: str = "mps",
    max_steps: int = 50,
    takeover_zone: float = 0.10,
    min_segment: int = 2,
    env_id: str = "PickCube-v1",
) -> dict[str, object]:
    """DAgger 接管模式：ACT rollout 到达末端停滞区后，专家接管执行到成功。

    记录接管点到成功/终止的完整 (state, arm_qvel, expert_action) 轨迹，
    使补采数据同时包含 move 与 settle 阶段。
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
    if takeover_zone <= 0:
        raise ValueError("takeover_zone 必须为正数")
    if max_steps <= 0:
        raise ValueError("max_steps 必须为正数")

    policies = load_lerobot_act_checkpoints(
        policy_dir, device=device, tasks=[env_id]
    )
    policy = policies[env_id]
    expert = RecoveryExpert()
    episode_states: list[np.ndarray] = []
    episode_qvels: list[np.ndarray] = []
    episode_actions: list[np.ndarray] = []
    episode_lengths: list[int] = []
    total_labeled = 0
    total_rollout_steps = 0
    takeover_episodes = 0

    for seed in checked_seeds:
        config = ManiSkillTaskConfig(env_id=env_id, seed=seed, max_steps=max_steps)
        env: Any | None = None
        segment_states: list[np.ndarray] = []
        segment_qvels: list[np.ndarray] = []
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
                grasped = _scalar_bool(observation, "is_grasped")
                distance = _obj_goal_distance(observation)
                if not taken_over and grasped and distance <= takeover_zone:
                    taken_over = True
                    takeover_episodes += 1
                if taken_over:
                    expert_action = expert.act(observation)
                    if expert_action is None:
                        break
                    segment_states.append(observation_to_state(observation))
                    segment_qvels.append(
                        observation_to_arm_qvel(observation)
                    )
                    segment_actions.append(expert_action)
                    action = expert_action
                else:
                    action = policy.act(observation)

                observation, _, terminated, truncated, info = env.step(action)
                terminated = bool(np.asarray(terminated).reshape(-1)[0])
                truncated = bool(np.asarray(truncated).reshape(-1)[0])
                success = bool(
                    np.asarray(info.get("success", False)).reshape(-1)[0]
                )
                if success or terminated or truncated:
                    break
            # 收尾未关闭的 segment
            if len(segment_states) >= min_segment:
                episode_states.append(np.stack(segment_states))
                episode_qvels.append(np.stack(segment_qvels))
                episode_actions.append(np.stack(segment_actions))
                episode_lengths.append(len(segment_states))
                total_labeled += len(segment_states)
        finally:
            if env is not None:
                env.close()

    return {
        "schema_version": DAGGER_SCHEMA_VERSION,
        "env_id": env_id,
        "policy_dir": str(Path(policy_dir).expanduser().resolve()),
        "seeds": list(checked_seeds),
        "takeover_zone": takeover_zone,
        "min_segment": min_segment,
        "takeover_episodes": takeover_episodes,
        "n_episodes": len(episode_lengths),
        "episode_lengths": episode_lengths,
        "total_labeled_transitions": total_labeled,
        "total_rollout_steps": total_rollout_steps,
        "states": (
            np.concatenate(episode_states, axis=0)
            if episode_states
            else np.empty((0, 18), dtype=np.float32)
        ),
        "arm_qvel": (
            np.concatenate(episode_qvels, axis=0)
            if episode_qvels
            else np.empty((0, 7), dtype=np.float32)
        ),
        "actions": (
            np.concatenate(episode_actions, axis=0)
            if episode_actions
            else np.empty((0, ACTION_DIM), dtype=np.float32)
        ),
    }


def save_dagger_episodes(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in ("states", "arm_qvel", "actions")
    }
    json_path = path.with_suffix(".json")
    temporary = json_path.with_name(f".{json_path.name}.tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(json_path)
    np.savez_compressed(
        path,
        states=np.asarray(payload["states"], dtype=np.float32),
        arm_qvel=np.asarray(payload["arm_qvel"], dtype=np.float32),
        actions=np.asarray(payload["actions"], dtype=np.float32),
    )


def merge_dagger_payloads(
    payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """合并多轮补采 payload（episode 拼接），供 DAgger 多轮累积。"""
    if not payloads:
        raise ValueError("没有可合并的补采 payload")
    all_lengths: list[int] = []
    state_parts: list[np.ndarray] = []
    qvel_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    for payload in payloads:
        lengths = list(payload["episode_lengths"])
        all_lengths.extend(lengths)
        state_parts.append(np.asarray(payload["states"], dtype=np.float32))
        qvel_parts.append(np.asarray(payload["arm_qvel"], dtype=np.float32))
        action_parts.append(np.asarray(payload["actions"], dtype=np.float32))
    return {
        "schema_version": DAGGER_SCHEMA_VERSION,
        "env_id": str(payloads[0]["env_id"]),
        "n_episodes": len(all_lengths),
        "episode_lengths": all_lengths,
        "total_labeled_transitions": sum(all_lengths),
        "states": np.concatenate(state_parts, axis=0),
        "arm_qvel": np.concatenate(qvel_parts, axis=0),
        "actions": np.concatenate(action_parts, axis=0),
    }


def load_dagger_episodes(path: Path) -> dict[str, object]:
    path = Path(path)
    arrays = np.load(path)
    json_path = path.with_suffix(".json")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "schema_version": metadata["schema_version"],
        "env_id": metadata["env_id"],
        "episode_lengths": list(metadata["episode_lengths"]),
        "states": arrays["states"],
        "arm_qvel": arrays["arm_qvel"],
        "actions": arrays["actions"],
        "n_episodes": int(metadata["n_episodes"]),
        "total_labeled_transitions": int(
            metadata["total_labeled_transitions"]
        ),
    }


def build_dagger_transitions(
    base: object,
    dagger: Mapping[str, object],
    *,
    start_episode_id: int = 100,
    insert_after: int = 80,
) -> object:
    """把补采 episode 插入到原始 train 之后、原始 validation 之前。

    保证 train/validation 与原始一致：train = 原始前 insert_after 个 episode
    + 补采 episode；validation = 原始剩余 episode（不包含补采数据）。
    """
    from behavior_cloning import TaskTransitions

    lengths = list(dagger["episode_lengths"])
    states = np.asarray(dagger["states"], dtype=np.float32)
    arm_qvel = np.asarray(dagger["arm_qvel"], dtype=np.float32)
    actions = np.asarray(dagger["actions"], dtype=np.float32)
    if sum(lengths) != len(states):
        raise ValueError("dagger episode_lengths 与 states 不一致")
    if insert_after < 0 or insert_after > len(base.episode_ids):
        raise ValueError("insert_after 超出原始 episode 数")

    dagger_ids = range(start_episode_id, start_episode_id + len(lengths))
    orig_lengths = list(base.episode_lengths)
    orig_offsets = np.concatenate(([0], np.cumsum(orig_lengths, dtype=np.int64)))
    d_offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))

    new_ids: list[int] = []
    new_lengths: list[int] = []
    state_parts: list[np.ndarray] = []
    qvel_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []

    def add_original(idx: int) -> None:
        new_ids.append(int(base.episode_ids[idx]))
        new_lengths.append(orig_lengths[idx])
        state_parts.append(base.states[orig_offsets[idx]: orig_offsets[idx + 1]])
        action_parts.append(base.actions[orig_offsets[idx]: orig_offsets[idx + 1]])
        base_arm = (
            np.asarray(base.arm_qvel, dtype=np.float32)
            if base.arm_qvel is not None
            else np.zeros((len(base.states), 7), dtype=np.float32)
        )
        qvel_parts.append(base_arm[orig_offsets[idx]: orig_offsets[idx + 1]])

    def add_dagger(k: int) -> None:
        new_ids.append(dagger_ids[k])
        new_lengths.append(lengths[k])
        state_parts.append(states[d_offsets[k]: d_offsets[k + 1]])
        qvel_parts.append(arm_qvel[d_offsets[k]: d_offsets[k + 1]])
        action_parts.append(actions[d_offsets[k]: d_offsets[k + 1]])

    for idx in range(insert_after):
        add_original(idx)
    for k in range(len(lengths)):
        add_dagger(k)
    for idx in range(insert_after, len(base.episode_ids)):
        add_original(idx)

    return TaskTransitions(
        env_id=base.env_id,
        episode_ids=tuple(new_ids),
        episode_lengths=tuple(new_lengths),
        states=np.concatenate(state_parts, axis=0).astype(np.float32, copy=False),
        actions=np.concatenate(action_parts, axis=0).astype(np.float32, copy=False),
        arm_qvel=np.concatenate(qvel_parts, axis=0).astype(np.float32, copy=False),
    )


def parse_args(argv: list[str] | None = None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="DAgger recovery 补采")
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="补采 npz 输出路径")
    parser.add_argument("--seeds", type=str, default="150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--takeover-zone", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # 提前解析绝对路径，避免 ManiSkill 改变 CWD 后相对路径失效。
        output = args.output.expanduser().resolve()
        policy_dir = args.policy_dir.expanduser().resolve()
        seeds = [
            int(item) for item in args.seeds.split(",") if item.strip()
        ]
        payload = collect_dagger_episodes(
            policy_dir,
            seeds=seeds,
            device=args.device,
            max_steps=args.max_steps,
            takeover_zone=args.takeover_zone,
        )
        save_dagger_episodes(payload, output)
        print(
            f"补采完成：{payload['n_episodes']} 个 episode，"
            f"{payload['total_labeled_transitions']} 条 recovery transition，"
            f"rollout {payload['total_rollout_steps']} 步"
        )
        print(f"写入：{output}（+ .json 元数据）")
    except (
        LeRobotACTError,
        BehaviorCloningError,
        ManiSkillUnavailableError,
        OSError,
        ValueError,
    ) as exc:
        print(f"DAgger 补采失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
