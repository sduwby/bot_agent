#!/usr/bin/env python
"""第 38 周 P3：世界模型 OPE 排名的 CLI。

两种模式（可分开在两台机器上跑）：

1. **仿真实测**（需要 ManiSkill，云上跑）：把候选策略逐个在 PushCube 环境里 rollout，
   记录**每个 episode 的初始 18 维状态**与成败 → `sim_eval.json`。
2. **WM 预测**（纯 NumPy，本机/云上均可）：读取同一份 `sim_eval.json`，让世界模型从
   **完全相同的初始状态** rollout 同一批候选 → 与仿真实测做 Spearman ρ → `report.json` + `report.md`。

关键方法论：**两侧必须用同一批初始状态**，否则比的是两个不同的初始分布，ρ 无意义。
因此初始状态由仿真侧记录、WM 侧复用（不是各自独立采样）。

候选要求（设计 §7.5）：≥5 个且**跨行为族**，含 scripted 锚点。

用法::

    # 云上（ManiSkill）
    python src/run_wm_ope.py --eval-sim --seeds 500-509 --out runs/week38/wm_ope/pushcube
    # 任意机器（纯 NumPy）
    python src/run_wm_ope.py --wm runs/week38/wm/pushcube \\
        --sim-eval runs/week38/wm_ope/pushcube/sim_eval.json \\
        --out runs/week38/wm_ope/pushcube
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from wm_ope import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    WMOPEError,
    OPEReport,
    attribute_low_rho,
    evaluate_ope,
    rollout_policy_in_world_model,
    rollout_with_trajectory,
    write_ope_report,
)

PUSH_TASK = "PushCube-v1"
SIM_EVAL_SCHEMA_VERSION = "week38_wm_ope_sim_eval_v1"
DEFAULT_OUT = "runs/week38/wm_ope/pushcube"
DEFAULT_SEEDS = "500-509"
#: 训练用 seed 段与评估段（500-509）**不重叠**，避免世界模型见过评估初始状态。
DEFAULT_TRAIN_SEEDS = "700-899"
DEFAULT_TRAIN_EPISODES = 160
DEFAULT_VALIDATION_EPISODES = 30
DEFAULT_TRAINING_STEPS = 30000
DEFAULT_HIDDEN_DIMS = "256,256"
#: 与任务平均步数同量级：scripted 在仿真里 13–25 步完成（horizon 太大只会放大
#: rollout error compounding，实测 40 步时 WM 内轨迹会发散到 1e3 量级）。
DEFAULT_HORIZON = 25


# --------------------------------------------------------------------------- #
# 候选策略（均为**状态输入**，可在 18 维世界模型内 rollout）
# --------------------------------------------------------------------------- #

class NoisyScriptedPolicy:
    """scripted + 动作噪声：制造"同一策略族内的性能梯度"。"""

    def __init__(self, base: object, sigma: float, rng: np.random.Generator) -> None:
        self._base = base
        self._sigma = float(sigma)
        self._rng = rng

    def reset(self) -> None:
        reset = getattr(self._base, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation: Mapping[str, object]) -> np.ndarray:
        action = np.asarray(self._base.act(observation), dtype=np.float32)
        noisy = action + self._rng.normal(0.0, self._sigma, size=ACTION_DIM).astype(np.float32)
        return np.clip(noisy, -1.0, 1.0).astype(np.float32)


class LaggedScriptedPolicy:
    """scripted 动作重复执行 k 次：制造"控制带宽下降"的退化候选。"""

    def __init__(self, base: object, repeat: int) -> None:
        if repeat < 1:
            raise WMOPEError("repeat 必须 >= 1")
        self._base = base
        self._repeat = int(repeat)
        self._held: np.ndarray | None = None
        self._left = 0

    def reset(self) -> None:
        reset = getattr(self._base, "reset", None)
        if callable(reset):
            reset()
        self._held = None
        self._left = 0

    def act(self, observation: Mapping[str, object]) -> np.ndarray:
        if self._left <= 0:
            self._held = np.asarray(self._base.act(observation), dtype=np.float32)
            self._left = self._repeat
        self._left -= 1
        assert self._held is not None
        return self._held


class RandomPolicy:
    """均匀随机动作：最弱锚点。"""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def reset(self) -> None:
        return None

    def act(self, observation: Mapping[str, object]) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)


class ZeroPolicy:
    """恒定零动作：退化锚点（连夹爪都不动）。"""

    def reset(self) -> None:
        return None

    def act(self, observation: Mapping[str, object]) -> np.ndarray:
        return np.zeros(ACTION_DIM, dtype=np.float32)


class ReversePolicy:
    """把 scripted 的动作**反向**（xyz 取负）：把物体推向远离目标的方向。

    作为"明确失败"的对照锚点——它与 scripted 共用同一套状态机，只是方向反了，
    因此二者的成功率差异只来自动作语义，不来自策略复杂度差异。
    """

    def __init__(self, base: object) -> None:
        self._base = base

    def reset(self) -> None:
        reset = getattr(self._base, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation: Mapping[str, object]) -> np.ndarray:
        action = np.asarray(self._base.act(observation), dtype=np.float32).copy()
        action[:3] = -action[:3]
        return np.clip(action, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    kind: str
    param: float = 0.0


#: 候选注册表：7 个候选 / 5 个行为族，含 scripted 锚点（满足"≥5 且跨族"）。
#: **退化档从 σ=0.5 起**——第 38 周实测 σ≤0.20 与 lag-3 在仿真里仍是 100% 成功，
#: 退化太弱会使真实成功率坍塌成 {1.0, 0.0} 两个值，排序失去信息量（ρ 无意义）。
CANDIDATE_REGISTRY: tuple[CandidateSpec, ...] = (
    CandidateSpec("scripted", "scripted", "scripted"),
    CandidateSpec("scripted-noise-0.5", "scripted-degraded", "noisy", 0.5),
    CandidateSpec("scripted-noise-1.0", "scripted-degraded", "noisy", 1.0),
    CandidateSpec("scripted-lag-8", "scripted-degraded", "lag", 8.0),
    CandidateSpec("scripted-reverse", "scripted-adversarial", "reverse"),
    CandidateSpec("random", "random", "random"),
    CandidateSpec("zero", "degenerate", "zero"),
)


def build_candidate(spec: CandidateSpec, seed: int) -> object:
    """按注册表构造候选策略。"""
    if spec.kind == "scripted":
        from maniskill_intro import ScriptedPolicy

        return ScriptedPolicy(PUSH_TASK)
    if spec.kind == "noisy":
        from maniskill_intro import ScriptedPolicy

        return NoisyScriptedPolicy(
            ScriptedPolicy(PUSH_TASK), spec.param, np.random.default_rng(seed)
        )
    if spec.kind == "lag":
        from maniskill_intro import ScriptedPolicy

        return LaggedScriptedPolicy(ScriptedPolicy(PUSH_TASK), int(spec.param))
    if spec.kind == "reverse":
        from maniskill_intro import ScriptedPolicy

        return ReversePolicy(ScriptedPolicy(PUSH_TASK))
    if spec.kind == "random":
        return RandomPolicy(np.random.default_rng(seed))
    if spec.kind == "zero":
        return ZeroPolicy()
    raise WMOPEError(f"未知候选类型 {spec.kind!r}")


def select_candidates(names: Sequence[str] | None) -> list[CandidateSpec]:
    if not names:
        return list(CANDIDATE_REGISTRY)
    registry = {spec.name: spec for spec in CANDIDATE_REGISTRY}
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise WMOPEError(f"未知候选 {unknown}；可用：{sorted(registry)}")
    return [registry[name] for name in names]


def parse_seeds(text: str) -> list[int]:
    """解析 `500-509` / `500,502,504` / 混合写法。"""
    seeds: list[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, _, end_text = chunk.partition("-")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise WMOPEError(f"seed 区间非法：{chunk}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(chunk))
    if len(set(seeds)) != len(seeds):
        raise WMOPEError(f"seeds 重复：{text}")
    if not seeds:
        raise WMOPEError("seeds 不能为空")
    return seeds


# --------------------------------------------------------------------------- #
# 模式 1：仿真实测（需要 ManiSkill）
# --------------------------------------------------------------------------- #

def evaluate_in_sim(
    specs: Sequence[CandidateSpec],
    seeds: Sequence[int],
    *,
    horizon: int,
) -> dict[str, Any]:
    """在 PushCube 仿真里逐个 rollout 候选，记录初始状态与成败。"""
    from behavior_cloning import observation_to_state
    from maniskill_intro import ManiSkillTaskConfig, make_maniskill_env
    from wm_ope import SLICE_GOAL, SLICE_OBJ

    results: dict[str, Any] = {}
    for spec in specs:
        init_states: list[list[float]] = []
        successes: list[float] = []
        steps_used: list[int] = []
        final_distance: list[float] = []
        for seed in seeds:
            config = ManiSkillTaskConfig(env_id=PUSH_TASK, seed=seed)
            env = make_maniskill_env(config)
            try:
                policy = build_candidate(spec, seed)
                reset = getattr(policy, "reset", None)
                if callable(reset):
                    reset()
                observation, _ = env.reset(seed=seed)
                init_states.append(
                    [float(value) for value in observation_to_state(observation)]
                )
                success = False
                step_index = 0
                for step_index in range(1, horizon + 1):
                    action = np.asarray(policy.act(observation), dtype=np.float32)
                    if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
                        raise WMOPEError(f"候选 {spec.name} 输出非法 action")
                    observation, _, terminated, truncated, info = env.step(action)
                    success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                    if success or bool(np.asarray(terminated).reshape(-1)[0]) or bool(
                        np.asarray(truncated).reshape(-1)[0]
                    ):
                        break
                final_state = observation_to_state(observation)
                final_distance.append(
                    float(np.linalg.norm(final_state[SLICE_OBJ] - final_state[SLICE_GOAL]))
                )
            finally:
                env.close()
            successes.append(1.0 if success else 0.0)
            steps_used.append(int(step_index))
            print(
                f"  [sim] {spec.name:22s} seed={seed} success={int(success)} steps={step_index}",
                flush=True,
            )
        results[spec.name] = {
            "family": spec.family,
            "init_states": init_states,
            "successes": successes,
            "steps": steps_used,
            "final_obj_goal_dist": final_distance,
        }
    return {
        "schema_version": SIM_EVAL_SCHEMA_VERSION,
        "task": PUSH_TASK,
        "seeds": list(seeds),
        "horizon": horizon,
        "candidates": results,
    }


def collect_dynamics_dataset(
    spec: CandidateSpec,
    seeds: Sequence[int],
    *,
    horizon: int,
) -> Any:
    """在**实时仿真**里 rollout 指定策略，采集 (s, a, s') 转移用于训练动力学。

    为什么不用 week15 的 v3 数据集：第 38 周实测其 PushCube 演示里物体每集只移动
    5–7 cm，而目标在 20 cm 外——数据中**不含任何成功轨迹**（<0.025 的帧占比为 0），
    据此学出的动力学只会预测"物体不动"，永远预测不出成功。故改为从实时仿真现采
    （脚本化专家能跑出成功轨迹），训练 seed 与评估 seed 必须不重叠（见 main 的校验）。
    """
    from behavior_cloning import observation_to_state
    from maniskill_intro import ManiSkillTaskConfig, make_maniskill_env
    from world_model import DynamicsDataset

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    episode_ids: list[int] = []
    for episode_index, seed in enumerate(seeds):
        config = ManiSkillTaskConfig(env_id=PUSH_TASK, seed=int(seed))
        env = make_maniskill_env(config)
        try:
            policy = build_candidate(spec, int(seed))
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
            observation, _ = env.reset(seed=int(seed))
            for _ in range(horizon):
                state = observation_to_state(observation)
                action = np.asarray(policy.act(observation), dtype=np.float32)
                if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
                    raise WMOPEError(f"候选 {spec.name} 输出非法 action")
                observation, _, terminated, truncated, info = env.step(action)
                states.append(np.asarray(state, dtype=np.float32))
                actions.append(action)
                next_states.append(
                    np.asarray(observation_to_state(observation), dtype=np.float32)
                )
                episode_ids.append(episode_index)
                success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                if (
                    success
                    or bool(np.asarray(terminated).reshape(-1)[0])
                    or bool(np.asarray(truncated).reshape(-1)[0])
                ):
                    break
        finally:
            env.close()
    if not states:
        raise WMOPEError("仿真未采集到任何转移")
    return DynamicsDataset(
        states=np.stack(states, axis=0),
        actions=np.stack(actions, axis=0),
        next_states=np.stack(next_states, axis=0),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )


def train_world_model_from_sim(
    out_dir: Path,
    *,
    train_seeds: Sequence[int],
    horizon: int,
    train_episodes: int,
    validation_episodes: int,
    training_steps: int,
    hidden_dims: Sequence[int] = (256, 256),
    device: str = "cpu",
) -> dict[str, object]:
    """从实时仿真采集 scripted 专家转移并训练 dynamics，落盘到 out_dir/dynamics_checkpoint。

    默认模型/步数比 `world_model` 的原始默认（128,128 / 4000 步）更大：实测
    978 条转移 + 4000 步时**单步很准但多步 rollout 指数发散**（xy 距离 0.2→1629），
    因为 MLP 对"恒定不变"的维度（目标点）输出微小噪声，反馈进输入后被逐步放大。
    """
    from world_model import WorldModelConfig, train_dynamics

    spec = select_candidates(["scripted"])[0]
    data = collect_dynamics_dataset(spec, train_seeds, horizon=horizon)
    print(
        f"[WM] 仿真采集 {len(train_seeds)} 条 episode / {len(data.states)} 条转移",
        flush=True,
    )
    config = WorldModelConfig(
        dataset_dir=out_dir / "_sim_source",
        repo_id="sim-generated",
        output_dir=out_dir,
        hidden_dims=tuple(int(item) for item in hidden_dims),
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
        training_steps=training_steps,
        device=device,
    )
    return train_dynamics(config, data)


def evaluate_in_world_model(
    specs: Sequence[CandidateSpec],
    sim_eval: Mapping[str, Any],
    wm_dir: Path,
    *,
    collect_attribution: bool = True,
) -> OPEReport:
    """从 sim_eval 记录的初始状态出发，在 WM 内 rollout 同一批候选。"""
    from world_model import DynamicsModel

    wm = DynamicsModel(Path(wm_dir) / "dynamics_checkpoint")
    payload = sim_eval.get("candidates")
    if not isinstance(payload, Mapping):
        raise WMOPEError("sim_eval.json 缺少 candidates")
    horizon = int(sim_eval.get("horizon", DEFAULT_HORIZON))

    per_rollout: dict[str, list[float]] = {}
    real_success: list[float] = []
    families: dict[str, str] = {}
    visited: dict[str, list[np.ndarray]] = {}

    for spec in specs:
        entry = payload.get(spec.name)
        if not isinstance(entry, Mapping):
            raise WMOPEError(f"sim_eval.json 缺少候选 {spec.name}")
        seeds = list(sim_eval.get("seeds", []))
        init_states = [np.asarray(item, dtype=float) for item in entry["init_states"]]
        # 每条 rollout 用独立候选实例（策略可能是带 RNG/状态机的）
        rollout_results: list[float] = []
        visited_states: list[np.ndarray] = []
        for index, state in enumerate(init_states):
            policy = build_candidate(spec, int(seeds[index]) if index < len(seeds) else index)
            results, trajectories = rollout_with_trajectory(policy, wm, [state], horizon)
            rollout_results.append(results[0])
            visited_states.extend(trajectories[0])
        per_rollout[spec.name] = rollout_results
        real_success.append(float(np.mean(np.asarray(entry["successes"], dtype=float))))
        families[spec.name] = spec.family
        if collect_attribution:
            # 归因用**实际访问过的状态**（不是初始状态——两侧初始状态按设计相同，无区分度）
            visited[spec.name] = visited_states
        print(
            f"  [wm ] {spec.name:22s} real={real_success[-1]:.3f} "
            f"wm={np.mean(rollout_results):.3f}",
            flush=True,
        )

    ordering = [spec.name for spec in specs]
    attribution = (
        attribute_low_rho(wm, visited) if collect_attribution and visited else ()
    )
    report = evaluate_ope(
        {name: per_rollout[name] for name in ordering},
        families,
        real_success,
        attribution=attribution,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="第 38 周 P3：世界模型 OPE")
    parser.add_argument("--wm", default="", help="世界模型输出目录（含 dynamics_checkpoint）")
    parser.add_argument("--sim-eval", default="", help="仿真实测 json（含初始状态与成败）")
    parser.add_argument("--eval-sim", action="store_true", help="在仿真里跑候选（需 ManiSkill）")
    parser.add_argument("--train-wm-from-sim", action="store_true",
                        help="从实时仿真采集 scripted 专家转移来训练世界模型")
    parser.add_argument("--train-seeds", default=DEFAULT_TRAIN_SEEDS,
                        help="训练用 seed（**必须与 --seeds 不重叠**，否则泄漏）")
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--validation-episodes", type=int, default=DEFAULT_VALIDATION_EPISODES)
    parser.add_argument("--training-steps", type=int, default=DEFAULT_TRAINING_STEPS)
    parser.add_argument("--hidden-dims", default=DEFAULT_HIDDEN_DIMS)
    parser.add_argument("--candidates", default="", help="逗号分隔的候选名，默认全部注册表")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="如 500-509")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    specs = select_candidates(
        [item.strip() for item in args.candidates.split(",") if item.strip()]
        or None
    )
    out_dir = Path(args.out)

    if args.train_wm_from_sim:
        train_seeds = parse_seeds(args.train_seeds)
        overlap = set(train_seeds) & set(parse_seeds(args.seeds))
        if overlap:
            raise WMOPEError(
                f"训练 seed 与评估 seed 重叠 {sorted(overlap)}：会造成数据泄漏，请换一段区间"
            )
        print(
            f"[WM] 从仿真采集 {len(train_seeds)} 条 episode（horizon {args.horizon}）"
            f"| train {args.train_episodes} / val {args.validation_episodes}"
            f"| steps {args.training_steps}"
        )
        summary = train_world_model_from_sim(
            out_dir,
            train_seeds=train_seeds,
            horizon=args.horizon,
            train_episodes=args.train_episodes,
            validation_episodes=args.validation_episodes,
            training_steps=args.training_steps,
            hidden_dims=[int(item) for item in args.hidden_dims.split(",") if item.strip()],
        )
        print(
            f"[WM] 训练完成: best_validation_loss={summary.get('best_validation_loss')} "
            f"train_loss={summary.get('final_train_loss')} "
            f"transitions={summary.get('train_transitions')}"
        )
        print(f"[WM] checkpoint → {out_dir / 'dynamics_checkpoint'}")
        return

    if args.eval_sim:
        seeds = parse_seeds(args.seeds)
        print(f"[模拟] 候选 {len(specs)} 个 × seed {len(seeds)} 个 × horizon {args.horizon}")
        payload = evaluate_in_sim(specs, seeds, horizon=args.horizon)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "sim_eval.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[模拟] 已写 {target}")
        return

    if not args.wm or not args.sim_eval:
        parser.error("非 --eval-sim 模式必须同时给 --wm 与 --sim-eval")
    sim_eval = json.loads(Path(args.sim_eval).read_text(encoding="utf-8"))
    report = evaluate_in_world_model(specs, sim_eval, Path(args.wm))
    json_path, markdown_path = write_ope_report(report, out_dir)
    print(
        f"\n=== OPE ===\nSpearman ρ = {report.spearman_rho:.3f} "
        f"(CI [{report.ci_low:.3f}, {report.ci_high:.3f}])  判定: {report.verdict}\n"
        f"报告: {json_path}\n      {markdown_path}"
    )


if __name__ == "__main__":
    main()
