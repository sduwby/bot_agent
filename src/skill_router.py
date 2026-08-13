"""第 20 周：分层策略路由。

把 scripted、状态 ACT、视觉 ACT（以及可选 SmolVLA）统一为技能执行器：
同一技能接口可切换策略，定义选择顺序、超时和 fallback，失败可安全回退。
scripted 恒为末位安全网；SmolVLA 为可选策略位，加载失败不影响主线。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from maniskill_intro import (
    SUPPORTED_TASKS,
    ManiSkillTaskConfig,
    ManiSkillUnavailableError,
    ScriptedPolicy,
    make_maniskill_env,
)

ROUTER_SCHEMA_VERSION = "maniskill_skill_router_v1"

SUPPORTED_POLICY_KINDS = (
    "scripted",
    "bc",
    "state_act",
    "visual_act",
    "smolvla",
)

DEFAULT_MAX_STEPS = 50
DEFAULT_STAGNATION_WINDOW = 8
DEFAULT_STAGNATION_MOVE_THRESHOLD = 0.005


class SkillRouterError(RuntimeError):
    """第 20 周分层路由契约错误。"""


@dataclass(frozen=True)
class PolicySource:
    """一个可选策略位：种类、输入模式与实例工厂。"""

    name: str
    kind: str
    input_mode: str
    factory: Callable[[str], object]
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("name 必须是非空字符串")
        if self.kind not in SUPPORTED_POLICY_KINDS:
            raise ValueError(
                f"kind 必须是 {SUPPORTED_POLICY_KINDS} 之一，"
                f"实际为 {self.kind!r}"
            )
        if self.input_mode not in ("state", "visual"):
            raise ValueError("input_mode 只支持 state 或 visual")
        if not callable(self.factory):
            raise ValueError("factory 必须是可调用对象")


@dataclass(frozen=True)
class SkillRouteConfig:
    """一个任务的策略顺序与 fallback 参数。"""

    env_id: str
    sources: tuple[PolicySource, ...]
    max_steps: int = DEFAULT_MAX_STEPS
    stagnation_window: int = DEFAULT_STAGNATION_WINDOW
    stagnation_move_threshold: float = DEFAULT_STAGNATION_MOVE_THRESHOLD

    def __post_init__(self) -> None:
        if self.env_id not in SUPPORTED_TASKS:
            raise ValueError(
                f"不支持的任务 {self.env_id!r}，仅支持 {SUPPORTED_TASKS}"
            )
        if not self.sources:
            raise ValueError("sources 不能为空")
        if not isinstance(self.max_steps, int) or isinstance(
            self.max_steps, bool
        ):
            raise ValueError("max_steps 必须是整数")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须为正数")
        if not isinstance(self.stagnation_window, int) or isinstance(
            self.stagnation_window, bool
        ):
            raise ValueError("stagnation_window 必须是整数")
        if self.stagnation_window <= 0:
            raise ValueError("stagnation_window 必须为正数")
        if (
            isinstance(self.stagnation_move_threshold, bool)
            or not isinstance(self.stagnation_move_threshold, (int, float))
            or not 0 < float(self.stagnation_move_threshold) < 1
        ):
            raise ValueError("stagnation_move_threshold 必须位于 (0, 1)")
        if self.stagnation_window > self.max_steps:
            raise ValueError("stagnation_window 不能大于 max_steps")


@dataclass(frozen=True)
class SwitchEvent:
    """一次 fallback 切换事件。"""

    from_policy: str
    to_policy: str
    reason: str
    at_step: int


@dataclass(frozen=True)
class RouterEpisodeResult:
    """一个 seed 的路由执行结果。"""

    env_id: str
    seed: int
    success: bool
    steps: int
    policy_steps: dict[str, int]
    switch_events: tuple[SwitchEvent, ...]
    final_policy: str
    exhausted: bool
    error: str | None = None


@dataclass(frozen=True)
class RouterTaskSummary:
    """一个任务的路由评测聚合。"""

    env_id: str
    route: tuple[str, ...]
    episodes: int
    successes: int
    success_rate: float
    mean_steps: float
    fallback_counts: dict[str, int]
    policy_step_fractions: dict[str, float]
    exhausted: int


@dataclass(frozen=True)
class RouterEvaluationSummary:
    """全部任务与 seed 的路由评测摘要。"""

    schema_version: str
    created_at: str
    seeds: tuple[int, ...]
    tasks: tuple[RouterTaskSummary, ...]
    episodes: tuple[RouterEpisodeResult, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能序列化 {type(value).__name__}")


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checked_seeds(seeds: Sequence[int]) -> list[int]:
    if not seeds:
        raise ValueError("seeds 不能为空")
    checked: list[int] = []
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seeds 必须是非负整数")
        checked.append(seed)
    if len(set(checked)) != len(checked):
        raise ValueError("seeds 不能重复")
    return checked


def _as_float_array(value: object) -> np.ndarray:
    """把 tensor/数组安全转成 CPU float 一维数组。"""
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    return np.asarray(candidate, dtype=float).reshape(-1)


def _vector(
    observation: Mapping[str, object], key: str, size: int
) -> np.ndarray:
    """从 observation.extra 提取有限向量，缺失时返回全零。"""
    extra = observation.get("extra")
    if not isinstance(extra, Mapping) or key not in extra:
        return np.zeros(size, dtype=float)
    vector = _as_float_array(extra[key])
    if vector.size < size:
        return np.zeros(size, dtype=float)
    return vector[:size]


def _scalar_bool(observation: Mapping[str, object], key: str) -> bool:
    extra = observation.get("extra")
    if not isinstance(extra, Mapping) or key not in extra:
        return False
    array = _as_float_array(extra[key])
    return bool(array.size and array[0])


class StagnationMonitor:
    """按 TCP 位移窗口与夹爪状态变化判定停滞。"""

    def __init__(self, *, window: int, move_threshold: float) -> None:
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError("window 必须为正整数")
        if (
            isinstance(move_threshold, bool)
            or not isinstance(move_threshold, (int, float))
            or not 0 < float(move_threshold) < 1
        ):
            raise ValueError("move_threshold 必须位于 (0, 1)")
        self._window = window
        self._move_threshold = float(move_threshold)
        self._tcp_history: list[np.ndarray] = []
        self._last_grasped: bool | None = None
        self._samples = 0

    def reset(self) -> None:
        """清空窗口与夹爪基准，切换策略或新 episode 时调用。"""
        self._tcp_history.clear()
        self._last_grasped = None
        self._samples = 0

    def update(self, observation: Mapping[str, object]) -> None:
        """按当前 observation 更新窗口。"""
        tcp = _vector(observation, "tcp_pose", 3)
        self._tcp_history.append(tcp)
        if len(self._tcp_history) > self._window:
            self._tcp_history.pop(0)
        if self._last_grasped is None:
            self._last_grasped = _scalar_bool(observation, "is_grasped")
        self._samples += 1

    @property
    def is_stagnant(self) -> bool:
        """窗口未满不判停滞；窗口内位移均值低于阈值且夹爪无变化。"""
        if len(self._tcp_history) < self._window:
            return False
        tcp = np.asarray(self._tcp_history, dtype=float)
        if tcp.shape[0] < 2:
            return False
        mean_move = float(
            np.mean(np.linalg.norm(np.diff(tcp, axis=0), axis=1))
        )
        return mean_move < self._move_threshold

    @property
    def samples(self) -> int:
        """自上次 reset 以来累计的观测样本数。"""
        return self._samples


def scripted_source() -> PolicySource:
    """末位安全网：状态输入 scripted 策略。"""
    return PolicySource(
        name="scripted",
        kind="scripted",
        input_mode="state",
        factory=ScriptedPolicy,
    )


def state_act_source(checkpoint_dir: Path, *, device: str) -> PolicySource:
    """状态输入 ACT 策略位（第 13 周 checkpoint）。"""
    checkpoint = Path(checkpoint_dir).expanduser().resolve()

    def factory(env_id: str) -> object:
        from lerobot_act import load_lerobot_act_checkpoints

        policies = load_lerobot_act_checkpoints(
            checkpoint, device=device, tasks=(env_id,)
        )
        return policies[env_id]

    return PolicySource(
        name="state_act",
        kind="state_act",
        input_mode="state",
        factory=factory,
    )


def visual_act_source(
    checkpoint_dir: Path, *, device: str, resolution: int = 128
) -> PolicySource:
    """视觉输入 ACT 策略位（第 16/17 周 checkpoint）。"""
    checkpoint = Path(checkpoint_dir).expanduser().resolve()

    def factory(env_id: str) -> object:
        from visual_act import load_visual_act_checkpoints

        policies = load_visual_act_checkpoints(
            checkpoint,
            device=device,
            resolution=resolution,
            tasks=(env_id,),
        )
        return policies[env_id]

    return PolicySource(
        name="visual_act",
        kind="visual_act",
        input_mode="visual",
        factory=factory,
    )


def bc_source(checkpoint_path: Path, *, device: str) -> PolicySource:
    """状态输入 BC 策略位（第 12 周 checkpoint）。"""
    checkpoint = Path(checkpoint_path).expanduser().resolve()

    def factory(env_id: str) -> object:
        from behavior_cloning import load_bc_checkpoint

        policies = load_bc_checkpoint(checkpoint)
        return policies[env_id]

    return PolicySource(
        name="bc",
        kind="bc",
        input_mode="state",
        factory=factory,
    )


def smolvla_source(
    model_id: str, *, device: str, image_size: int = 224
) -> PolicySource:
    """可选 SmolVLA 策略位：加载失败跳过，不影响主线。"""
    source_model = str(model_id)

    def factory(env_id: str) -> object:
        from smolvla_probe import make_smolvla_policy

        policy, pre, post = make_smolvla_policy(
            source_model, device=device
        )
        return _SmolvlaAdapter(
            env_id=env_id,
            policy=policy,
            pre=pre,
            post=post,
            image_size=image_size,
        )

    return PolicySource(
        name="smolvla",
        kind="smolvla",
        input_mode="visual",
        factory=factory,
        optional=True,
    )


class _SmolvlaAdapter:
    """把 SmolVLA policy 包装为统一 policy 接口。"""

    phase = "smolvla"

    def __init__(
        self,
        *,
        env_id: str,
        policy: object,
        pre: object,
        post: object,
        image_size: int,
    ) -> None:
        from smolvla_probe import (
            TASK_TEXTS,
            observation_to_smolvla_input,
        )

        self.env_id = env_id
        self._policy = policy
        self._pre = pre
        self._post = post
        self._image_size = image_size
        self._task_text = TASK_TEXTS[env_id]
        self._to_input = observation_to_smolvla_input

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def act(
        self, observation: Mapping[str, object], image: np.ndarray
    ) -> np.ndarray:
        features = self._to_input(
            observation, image, self._task_text, self._image_size
        )
        processed = self._pre(features)
        action = self._policy.select_action(processed)
        action = self._post(action)
        array = _as_float_array(action).astype(np.float32)
        if array.size < 4:
            raise SkillRouterError(
                f"{self.env_id} SmolVLA 输出维度不足：{array.shape}"
            )
        return array[:4]


def build_default_route(
    env_id: str,
    *,
    state_act_checkpoint: Path | None = None,
    visual_act_checkpoint: Path | None = None,
    bc_checkpoint: Path | None = None,
    smolvla_model: str | None = None,
    device: str = "mps",
    include_scripted: bool = True,
) -> SkillRouteConfig:
    """按可用 checkpoint 构造默认路由（缺失位跳过，scripted 恒为末位）。"""
    if env_id not in SUPPORTED_TASKS:
        raise ValueError(f"不支持的任务 {env_id!r}")
    sources: list[PolicySource] = []
    if visual_act_checkpoint is not None:
        sources.append(
            visual_act_source(visual_act_checkpoint, device=device)
        )
    if state_act_checkpoint is not None:
        sources.append(state_act_source(state_act_checkpoint, device=device))
    if bc_checkpoint is not None:
        sources.append(bc_source(bc_checkpoint, device=device))
    if smolvla_model is not None:
        sources.append(smolvla_source(smolvla_model, device=device))
    if include_scripted:
        sources.append(scripted_source())
    if not sources:
        raise SkillRouterError(
            f"{env_id} 路由为空：至少需要一个 checkpoint 或保留 scripted"
        )
    return SkillRouteConfig(env_id=env_id, sources=tuple(sources))


class SkillRouter:
    """按路由配置执行单 episode，处理异常/停滞/超时 fallback。"""

    def __init__(self, *, env_factory: Callable[[ManiSkillTaskConfig], object]) -> None:
        if not callable(env_factory):
            raise ValueError("env_factory 必须是可调用对象")
        self._env_factory = env_factory

    def run_episode(
        self, config: SkillRouteConfig, seed: int
    ) -> RouterEpisodeResult:
        """执行一个 seed 的路由 episode，逐位尝试策略直到成功或耗尽。"""
        env: Any = self._env_factory(
            ManiSkillTaskConfig(
                env_id=config.env_id, seed=seed, max_steps=config.max_steps
            )
        )
        policies: list[tuple[str, PolicySource, object]] = []
        try:
            for source in config.sources:
                try:
                    policy = source.factory(config.env_id)
                except Exception as exc:
                    if source.optional:
                        continue
                    raise SkillRouterError(
                        f"{config.env_id} 策略 {source.name} 加载失败："
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                policies.append((source.name, source, policy))
            if not policies:
                raise SkillRouterError(f"{config.env_id} 没有可用策略位")
            return self._run_episode_with_policies(
                config, seed, env, policies
            )
        finally:
            env.close()

    def _run_episode_with_policies(
        self,
        config: SkillRouteConfig,
        seed: int,
        env: object,
        policies: Sequence[tuple[str, PolicySource, object]],
    ) -> RouterEpisodeResult:
        env_any: Any = env
        reset_result = env_any.reset(seed=seed)
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise SkillRouterError("env.reset() 未返回二元组")
        observation, _ = reset_result
        if not isinstance(observation, Mapping):
            raise SkillRouterError("state_dict observation 必须是 mapping")

        monitor = StagnationMonitor(
            window=config.stagnation_window,
            move_threshold=config.stagnation_move_threshold,
        )
        action_shape = tuple(
            int(size) for size in env_any.action_space.shape
        )
        policy_steps: dict[str, int] = {name: 0 for name, _, _ in policies}
        switch_events: list[SwitchEvent] = []
        active_index = 0
        active_name = policies[0][0]
        last_policy_failed = False
        steps = 0
        success = False
        terminated = False
        truncated = False

        for policy in (item[2] for item in policies):
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
        monitor.reset()

        while steps < config.max_steps:
            name, source, policy = policies[active_index]
            try:
                if source.input_mode == "visual":
                    image = self._render(env_any)
                    action = np.asarray(
                        policy.act(observation, image), dtype=np.float32
                    )
                else:
                    action = np.asarray(
                        policy.act(observation), dtype=np.float32
                    )
                self._validate_action(action, action_shape, name)
            except Exception as exc:
                if active_index >= len(policies) - 1:
                    return RouterEpisodeResult(
                        env_id=config.env_id,
                        seed=seed,
                        success=False,
                        steps=steps,
                        policy_steps=policy_steps,
                        switch_events=tuple(switch_events),
                        final_policy=name,
                        exhausted=True,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                fallback = self._switch(
                    policies, active_index, switch_events, steps, name, "exception"
                )
                assert fallback is not None
                active_index, active_name = fallback
                continue

            step_result = env_any.step(action)
            if not isinstance(step_result, tuple) or len(step_result) != 5:
                raise SkillRouterError("env.step() 未返回五元组")
            observation, _, raw_terminated, raw_truncated, info = step_result
            if not isinstance(observation, Mapping):
                raise SkillRouterError("step observation 必须是 mapping")
            steps += 1
            policy_steps[name] += 1
            monitor.update(observation)

            if isinstance(info, Mapping):
                success = bool(
                    np.asarray(info.get("success", False)).reshape(-1)[0]
                )
            terminated = bool(
                np.asarray(raw_terminated).reshape(-1)[0]
            )
            truncated = bool(
                np.asarray(raw_truncated).reshape(-1)[0]
            )
            if success or terminated or truncated:
                break

            if monitor.is_stagnant:
                if active_index >= len(policies) - 1:
                    last_policy_failed = True
                else:
                    fallback = self._switch(
                        policies,
                        active_index,
                        switch_events,
                        steps,
                        name,
                        "stagnation",
                    )
                    assert fallback is not None
                    active_index, active_name = fallback
                    monitor.reset()
                    for _, _, candidate in policies:
                        reset = getattr(candidate, "reset", None)
                        if callable(reset):
                            reset()

        exhausted = not success and last_policy_failed
        return RouterEpisodeResult(
            env_id=config.env_id,
            seed=seed,
            success=success,
            steps=steps,
            policy_steps=policy_steps,
            switch_events=tuple(switch_events),
            final_policy=active_name,
            exhausted=exhausted,
        )

    @staticmethod
    def _switch(
        policies: Sequence[tuple[str, PolicySource, object]],
        active_index: int,
        switch_events: list[SwitchEvent],
        steps: int,
        from_name: str,
        reason: str,
    ) -> tuple[int, str] | None:
        """切到下一位；已是末位则返回 None（耗尽）。"""
        if active_index >= len(policies) - 1:
            return None
        next_index = active_index + 1
        next_name = policies[next_index][0]
        switch_events.append(
            SwitchEvent(
                from_policy=from_name,
                to_policy=next_name,
                reason=reason,
                at_step=steps,
            )
        )
        return next_index, next_name

    @staticmethod
    def _render(env: object) -> np.ndarray:
        from lerobot_pipeline import _image_hwc

        try:
            return _image_hwc(env.render())
        except Exception as exc:
            raise SkillRouterError(f"渲染失败：{exc}") from exc

    @staticmethod
    def _validate_action(
        action: np.ndarray, action_shape: tuple[int, ...], name: str
    ) -> None:
        if action.shape != action_shape:
            raise SkillRouterError(
                f"{name} 动作 shape={action.shape}，期望 {action_shape}"
            )
        if not np.all(np.isfinite(action)):
            raise SkillRouterError(f"{name} 动作包含非有限数")


def evaluate_routes(
    routes: Mapping[str, SkillRouteConfig],
    seeds: Sequence[int],
    *,
    env_factory: Callable[[ManiSkillTaskConfig], object] = make_maniskill_env,
    output_dir: Path | None = None,
) -> RouterEvaluationSummary:
    """固定 seed 批量评测并落盘 router_report.json。"""
    if not routes:
        raise ValueError("routes 不能为空")
    checked_routes: dict[str, SkillRouteConfig] = {}
    for env_id, config in routes.items():
        if env_id not in SUPPORTED_TASKS:
            raise ValueError(f"不支持的任务 {env_id!r}")
        if config.env_id != env_id:
            raise ValueError(
                f"路由键 {env_id!r} 与配置 env_id {config.env_id!r} 不一致"
            )
        checked_routes[env_id] = config
    checked_seeds = _checked_seeds(seeds)
    router = SkillRouter(env_factory=env_factory)

    episodes: list[RouterEpisodeResult] = []
    for env_id, config in checked_routes.items():
        for seed in checked_seeds:
            episodes.append(router.run_episode(config, seed))

    tasks: list[RouterTaskSummary] = []
    for env_id, config in checked_routes.items():
        task_episodes = [item for item in episodes if item.env_id == env_id]
        successes = sum(item.success for item in task_episodes)
        total_steps = sum(item.steps for item in task_episodes)
        fallback_counts: dict[str, int] = {}
        policy_steps: dict[str, int] = {}
        exhausted = 0
        for item in task_episodes:
            if item.exhausted:
                exhausted += 1
            for event in item.switch_events:
                fallback_counts[event.reason] = (
                    fallback_counts.get(event.reason, 0) + 1
                )
            for name, count in item.policy_steps.items():
                policy_steps[name] = policy_steps.get(name, 0) + count
        policy_step_fractions = {
            name: count / total_steps if total_steps else 0.0
            for name, count in policy_steps.items()
        }
        tasks.append(
            RouterTaskSummary(
                env_id=env_id,
                route=tuple(source.name for source in config.sources),
                episodes=len(task_episodes),
                successes=successes,
                success_rate=successes / len(task_episodes),
                mean_steps=total_steps / len(task_episodes),
                fallback_counts=fallback_counts,
                policy_step_fractions=policy_step_fractions,
                exhausted=exhausted,
            )
        )

    summary = RouterEvaluationSummary(
        schema_version=ROUTER_SCHEMA_VERSION,
        created_at=_utc_now(),
        seeds=tuple(checked_seeds),
        tasks=tuple(tasks),
        episodes=tuple(episodes),
    )
    if output_dir is not None:
        directory = Path(output_dir).expanduser().resolve()
        _atomic_write_json(directory / "router_report.json", asdict(summary))
    return summary


__all__ = [
    "ROUTER_SCHEMA_VERSION",
    "SUPPORTED_POLICY_KINDS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_STAGNATION_WINDOW",
    "DEFAULT_STAGNATION_MOVE_THRESHOLD",
    "PolicySource",
    "SkillRouteConfig",
    "SwitchEvent",
    "RouterEpisodeResult",
    "RouterTaskSummary",
    "RouterEvaluationSummary",
    "StagnationMonitor",
    "SkillRouter",
    "SkillRouterError",
    "bc_source",
    "build_default_route",
    "evaluate_routes",
    "scripted_source",
    "smolvla_source",
    "state_act_source",
    "visual_act_source",
]
