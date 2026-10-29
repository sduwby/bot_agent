#!/usr/bin/env python
"""第 38 周 P2：VLA 组合泛化协议——split 生成、held-out 评测与置信区间。

研究问题：在「物体 × 指令」多任务数据上，模型对**训练时未见过的组合**泛化到什么程度？

本模块提供三件事（均为纯逻辑，不依赖 ManiSkill / PyTorch / 真机）：

1. `GeneralizationSplit` + `build_split`：按 `combo` / `paraphrase` / `novel_object`
   三种协议切出 train / held-out 两侧，并在**构造期**保证零泄漏（两侧组合不相交）。
2. `proportion_ci`：Bernoulli 成功率的 bootstrap 百分位区间（自带实现，不复用
   `wm_ope._bootstrap_ci`——后者对 Spearman ρ 特化，见其 docstring）。
3. `evaluate_generalization` + `write_generalization_report`：把脚本化 runner 的成败
   聚合成 train / held-out 成功率 + 95% CI + 泛化 gap，并落盘 `generalization.json` / `.md`。

**口径边界（写入报告）**：默认 runner 是**状态输入**策略回放，**忽略语言指令**；
因此 held-out 只衡量组合/物体层面的划分是否泛化，不构成语言条件泛化的证据。
语言条件评测必须由调用方注入自定义 `runner(env_id, seed, instruction)`。
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "week38_vla_generalization_v1"

#: 三种组合泛化协议（设计 §7.4）。
PROTOCOLS = ("combo", "paraphrase", "novel_object")

CI_LEVEL = 0.95
BOOTSTRAP_SAMPLES = 2000

#: 验收 A7：held-out 组合成功率 ≥ 40%（否则降级为仿真协议验证）。
HELDOUT_SUCCESS_TARGET = 0.40

#: 默认 runner 使用的 ManiSkill episode 长度（与 ManiSkillTaskConfig 默认一致）。
DEFAULT_MAX_STEPS = 50

SCOPE_NOTE = (
    "口径边界：默认 runner 为**状态输入**策略回放，**忽略语言指令**——"
    "因此 held-out 只衡量组合/物体层面的划分是否泛化，不构成语言条件泛化的证据；"
    "语言条件评测必须由调用方注入自定义 runner(env_id, seed, instruction)。"
)

#: 单条「组合」= (object, instruction)；object 在默认 runner 里被当作 env_id 使用。
Combo = tuple[str, str]


class VLAGeneralizationError(RuntimeError):
    """泛化协议或评测契约错误。"""


def _normalize_pairs(values: Sequence[object], label: str) -> tuple[Combo, ...]:
    """校验并归一化一组 (object, instruction)：元素为二元非空字符串、且无重复。"""
    normalized: list[Combo] = []
    seen: set[Combo] = set()
    for item in values:
        if isinstance(item, str) or not isinstance(item, (tuple, list)) or len(item) != 2:
            raise VLAGeneralizationError(
                f"{label} 的每个元素必须是 (object, instruction) 二元组，实际 {item!r}"
            )
        obj, instruction = item
        if not isinstance(obj, str) or not isinstance(instruction, str):
            raise VLAGeneralizationError(f"{label} 含非字符串字段：{item!r}")
        obj, instruction = obj.strip(), instruction.strip()
        if not obj or not instruction:
            raise VLAGeneralizationError(f"{label} 含空字符串字段：{item!r}")
        pair = (obj, instruction)
        if pair in seen:
            raise VLAGeneralizationError(f"{label} 内存在重复组合：{pair!r}")
        seen.add(pair)
        normalized.append(pair)
    return tuple(normalized)


@dataclass(frozen=True)
class GeneralizationSplit:
    """一次泛化切分：train / held-out 两侧的组合与协议（设计 §4）。

    `warnings` 为本次实现补充（设计 §4 仅有前三字段）：`paraphrase` 协议因本仓库
    无内置同义改写模型，需要把「哪些输入无法改写」的证据写给调用方，故并入 split 一并返回。
    """

    protocol: str
    train_combos: tuple[Combo, ...]
    heldout_combos: tuple[Combo, ...]
    warnings: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.protocol not in PROTOCOLS:
            raise VLAGeneralizationError(
                f"未知协议 {self.protocol!r}，仅支持 {PROTOCOLS}"
            )
        train = _normalize_pairs(self.train_combos, "train_combos")
        heldout = _normalize_pairs(self.heldout_combos, "heldout_combos")
        object.__setattr__(self, "train_combos", train)
        object.__setattr__(self, "heldout_combos", heldout)
        if not train or not heldout:
            raise VLAGeneralizationError(
                "train 与 heldout 两侧都不能为空：空的一侧无法评测"
            )
        overlap = set(train) & set(heldout)
        if overlap:
            # 泄漏是泛化评测里最致命的静默错误：训练见过的组合若混进 held-out，
            # 报出的"泛化能力"其实是记忆能力，因此必须在构造期直接拒绝。
            raise VLAGeneralizationError(
                f"train/heldout 发生泄漏（{len(overlap)} 个组合同时出现在两侧）："
                f"{sorted(overlap)[:5]}"
            )

    @property
    def combo_counts(self) -> dict[str, int]:
        """两侧组合数，供报告直接落盘。"""
        return {"train": len(self.train_combos), "heldout": len(self.heldout_combos)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": self.protocol,
            "train_combos": [list(item) for item in self.train_combos],
            "heldout_combos": [list(item) for item in self.heldout_combos],
            "combo_counts": self.combo_counts,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneralizationSplit":
        train = tuple(tuple(str(value) for value in item) for item in payload["train_combos"])
        heldout = tuple(
            tuple(str(value) for value in item) for item in payload["heldout_combos"]
        )
        warnings = tuple(str(item) for item in payload.get("warnings", ()))
        return cls(
            protocol=str(payload["protocol"]),
            train_combos=train,
            heldout_combos=heldout,
            warnings=warnings,
        )


def _clamped_count(ratio: float, total: int) -> int:
    """把比例换算成 [1, total-1] 内的整数个数，保证两侧都非空。"""
    if total < 2:
        raise VLAGeneralizationError(f"可切分单元的个数 {total} < 2，无法切出非空的两侧")
    return max(1, min(int(round(ratio * total)), total - 1))


def _ratio_split(
    combos: Sequence[Combo], heldout_ratio: float, seed: int
) -> tuple[tuple[Combo, ...], tuple[Combo, ...]]:
    """按比例做确定性随机切分（先排序再洗牌，使结果与输入顺序无关）。"""
    ordered = sorted(combos)
    if len(ordered) < 2:
        raise VLAGeneralizationError("组合数 < 2，无法切出非空的 train/heldout 两侧")
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    count = _clamped_count(heldout_ratio, len(shuffled))
    heldout = tuple(sorted(shuffled[:count]))
    train = tuple(sorted(shuffled[count:]))
    return train, heldout


def _covers_both_sides(
    train: Sequence[Combo], heldout: Sequence[Combo]
) -> bool:
    """每个物体与每条指令是否都同时出现在两侧（combo 协议的"对称"要求）。"""
    def sides(pairs: Sequence[Combo]) -> tuple[set[str], set[str]]:
        return {obj for obj, _ in pairs}, {instruction for _, instruction in pairs}

    train_objects, train_instructions = sides(train)
    held_objects, held_instructions = sides(heldout)
    return (
        train_objects == held_objects == (train_objects | held_objects)
        and train_instructions == held_instructions == (train_instructions | held_instructions)
    )


def _combo_split(
    combos: Sequence[Combo], heldout_ratio: float, seed: int
) -> tuple[tuple[Combo, ...], tuple[Combo, ...]]:
    """`combo` 协议：留出组合。

    规则（文档化）：
    - 若输入恰是「所有物体 × 所有指令」的**完整笛卡尔积**，走交叉积留一（leave-diagonal-out）：
      每个物体按 seed 派生的旋转留出 `k` 条指令（`k` 由 heldout_ratio 换算、钳制到 [1, 指令数-1]），
      从而每个物体、每条指令都落在两侧（经典例：训练 (A,1),(B,2)，held-out (A,2),(B,1)）。
    - 若交叉积不完整，或该留一法无法让每个物体/指令都出现在两侧（"when possible"），
      退回确定性比例切分（先排序再洗牌，尊重 heldout_ratio）。
    """
    objects = sorted({obj for obj, _ in combos})
    instructions = sorted({instruction for _, instruction in combos})
    grid = {(obj, instruction) for obj in objects for instruction in instructions}
    if set(combos) == grid and len(objects) >= 2 and len(instructions) >= 2:
        count = _clamped_count(heldout_ratio, len(instructions))
        offset = seed % len(instructions)
        train: list[Combo] = []
        heldout: list[Combo] = []
        for index, obj in enumerate(objects):
            held_columns = {(offset + index + step) % len(instructions) for step in range(count)}
            for column, instruction in enumerate(instructions):
                pair = (obj, instruction)
                (heldout if column in held_columns else train).append(pair)
        if _covers_both_sides(train, heldout):
            return tuple(sorted(train)), tuple(sorted(heldout))
    return _ratio_split(combos, heldout_ratio, seed)


def _paraphrase_split(
    combos: Sequence[Combo], seed: int
) -> tuple[tuple[Combo, ...], tuple[Combo, ...], tuple[str, ...]]:
    """`paraphrase` 协议：留出**指令改写变体**。

    规则（文档化）：按物体分组，对拥有 ≥2 个指令变体的物体，用 seed 确定性地留出其中
    **一个**变体；只有 1 个变体的物体整体进入 held-out，并给出告警。

    重要局限：本仓库没有内置同义改写模型，因此 held-out 侧**不是生成的同义句**，而是调用方
    提供的指令变体（代理）。真正的语言改写鲁棒性需要调用方给出改写数据集；此处把该边界以
    warning 形式一并返回。`heldout_ratio` 对本协议不生效（每个物体恰好留 1 个变体）。
    """
    by_object: dict[str, set[str]] = {}
    for obj, instruction in combos:
        by_object.setdefault(obj, set()).add(instruction)

    rng = random.Random(seed)
    train: list[Combo] = []
    heldout: list[Combo] = []
    warnings: list[str] = [
        "本仓库无内置同义改写模型：held-out 侧直接沿用调用方提供的指令变体作为改写代理，"
        "并不生成新的同义句；语言改写鲁棒性需调用方提供改写数据集。",
    ]
    for obj in sorted(by_object):
        variants = sorted(by_object[obj])
        if len(variants) >= 2:
            chosen = variants[rng.randrange(len(variants))]
            for instruction in variants:
                (heldout if instruction == chosen else train).append((obj, instruction))
        else:
            heldout.append((obj, variants[0]))
            warnings.append(
                f"物体 {obj!r} 只有 1 个指令变体，paraphrase 协议在该物体上退化为 "
                "novel_object（整物体 held-out）。"
            )
    if not train:
        raise VLAGeneralizationError(
            "paraphrase 协议要求至少一个物体提供 ≥2 个指令变体，"
            "否则整批退化为 novel_object 且 train 为空"
        )
    return tuple(sorted(train)), tuple(sorted(heldout)), tuple(warnings)


def _novel_object_split(
    combos: Sequence[Combo], heldout_ratio: float, seed: int
) -> tuple[tuple[Combo, ...], tuple[Combo, ...]]:
    """`novel_object` 协议：整体留出若干**物体**（其全部指令一起进 held-out）。"""
    objects = sorted({obj for obj, _ in combos})
    if len(objects) < 2:
        raise VLAGeneralizationError(
            "novel_object 协议至少需要 2 个不同物体（一个留训练、一个留 held-out）"
        )
    shuffled = list(objects)
    random.Random(seed).shuffle(shuffled)
    count = _clamped_count(heldout_ratio, len(objects))
    heldout_objects = set(shuffled[:count])
    ordered = sorted(combos)
    train = tuple(pair for pair in ordered if pair[0] not in heldout_objects)
    heldout = tuple(pair for pair in ordered if pair[0] in heldout_objects)
    return train, heldout


def build_split(
    combos: Sequence[Combo], protocol: str, heldout_ratio: float, seed: int
) -> GeneralizationSplit:
    """按协议切分 train / held-out（设计 §5.1；`combos` 为 (object, instruction) 序列）。

    Args:
        combos: (object, instruction) 组合序列，需无重复、字段非空。
        protocol: `combo` / `paraphrase` / `novel_object`。
        heldout_ratio: held-out 比例，须严格落在 (0, 1)。
        seed: 确定性随机种子（`random.Random(seed)`）。

    Returns:
        `GeneralizationSplit`；保证两侧非空且零泄漏。
    """
    if protocol not in PROTOCOLS:
        raise VLAGeneralizationError(f"未知协议 {protocol!r}，仅支持 {PROTOCOLS}")
    if isinstance(heldout_ratio, bool) or not isinstance(heldout_ratio, (int, float)):
        raise VLAGeneralizationError("heldout_ratio 必须是数值")
    ratio = float(heldout_ratio)
    if not 0.0 < ratio < 1.0:
        raise VLAGeneralizationError(
            f"heldout_ratio 必须严格落在 (0, 1)，实际 {heldout_ratio!r}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise VLAGeneralizationError("seed 必须是整数")
    cleaned = _normalize_pairs(combos, "combos")
    if not cleaned:
        raise VLAGeneralizationError("combos 不能为空")

    if protocol == "combo":
        train, heldout = _combo_split(cleaned, ratio, seed)
        return GeneralizationSplit(protocol, train, heldout)
    if protocol == "paraphrase":
        train, heldout, warnings = _paraphrase_split(cleaned, seed)
        return GeneralizationSplit(protocol, train, heldout, warnings)
    train, heldout = _novel_object_split(cleaned, ratio, seed)
    return GeneralizationSplit(protocol, train, heldout)


def _checked_seeds(values: Sequence[int]) -> list[int]:
    """与 `run_generalization._checked_seeds` 同口径（此处复制以免引入 torch 依赖）。"""
    if not values:
        raise VLAGeneralizationError("seeds 不能为空")
    checked: list[int] = []
    for seed in values:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise VLAGeneralizationError("seeds 必须是非负整数")
        checked.append(seed)
    if len(set(checked)) != len(checked):
        raise VLAGeneralizationError("seeds 不能重复")
    return checked


def _assert_disjoint_seeds(train_seeds: Sequence[int], heldout_seeds: Sequence[int]) -> None:
    """train 段与 held-out 段 episode 的 seed 必须不相交（评测泄漏防线）。"""
    _checked_seeds(train_seeds)
    _checked_seeds(heldout_seeds)
    overlap = set(train_seeds) & set(heldout_seeds)
    if overlap:
        raise VLAGeneralizationError(
            f"train/heldout seed 重叠：{sorted(overlap)[:5]}——会造成评测泄漏，请调整 base_seed"
        )


def proportion_ci(
    successes: int,
    total: int,
    *,
    level: float = CI_LEVEL,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Bernoulli 成功率的 bootstrap 百分位区间（自带实现，非 `wm_ope._bootstrap_ci`）。

    为什么另写：`wm_ope._bootstrap_ci` 是对 Spearman ρ 特化的（输入各候选成功率、输出 ρ 的
    区间），无法估计单一比例；此处沿用其**风格**（`np.random.default_rng(seed)` +
    `np.percentile` 百分位）对观测到的 0/1 序列做有放回重采样。

    `total == 0` 无法估计，显式返回退化区间 `(0.0, 0.0)`；调用方应先检查 `total`。
    给定 seed 结果确定。
    """
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise VLAGeneralizationError("total 必须是非负整数")
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise VLAGeneralizationError("successes 必须是整数")
    if not 0 <= successes <= total:
        raise VLAGeneralizationError(f"successes={successes} 必须落在 [0, {total}]")
    if not 0.0 < level < 1.0:
        raise VLAGeneralizationError("level 必须严格落在 (0, 1)")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise VLAGeneralizationError("samples 必须是正整数")
    if total == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    trials = np.zeros(total, dtype=float)
    trials[:successes] = 1.0
    draws = rng.integers(0, total, size=(samples, total))
    means = trials[draws].mean(axis=1)
    alpha = (1.0 - level) / 2.0 * 100.0
    return (
        float(np.percentile(means, alpha)),
        float(np.percentile(means, 100.0 - alpha)),
    )


def _default_runner(
    policy: object,
    env_factory: Callable[..., object] | None,
    max_steps: int | None,
) -> Callable[[str, int, str], bool]:
    """构造默认 runner：状态输入策略在 ManiSkill 里回放。

    明确边界：**忽略 instruction**（只回放状态输入策略），故语言条件评测必须自备 runner。
    """
    if policy is None:
        raise VLAGeneralizationError("runner 为 None 时必须提供 policy（默认 runner 需要策略回放）")
    if not callable(getattr(policy, "act", None)):
        raise VLAGeneralizationError("policy 必须实现 act(observation)")
    if env_factory is None:
        from maniskill_intro import make_maniskill_env

        env_factory = make_maniskill_env
    steps = DEFAULT_MAX_STEPS if max_steps is None else max_steps

    def runner(env_id: str, seed: int, instruction: str) -> bool:
        # instruction 有意不使用：默认 runner 为状态输入口径（见 docstring / SCOPE_NOTE）。
        from maniskill_intro import ManiSkillTaskConfig, run_episode_in_env

        config = ManiSkillTaskConfig(env_id=env_id, seed=seed, max_steps=steps)
        env = env_factory(config)
        try:
            result = run_episode_in_env(config, env, policy)
        finally:
            env.close()
        return bool(result.success)

    return runner


def _run_block(
    combos: Sequence[Combo],
    seeds: Sequence[int],
    n_episodes: int,
    runner: Callable[[str, int, str], bool],
    *,
    level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    per_combo: list[dict[str, Any]] = []
    cursor = 0
    for obj, instruction in combos:
        combo_seeds = list(seeds[cursor : cursor + n_episodes])
        cursor += n_episodes
        successes = sum(bool(runner(obj, seed, instruction)) for seed in combo_seeds)
        per_combo.append(
            {
                "object": obj,
                "instruction": instruction,
                "episodes": n_episodes,
                "successes": successes,
                "success_rate": successes / n_episodes,
                "seeds": combo_seeds,
            }
        )
    total_episodes = n_episodes * len(combos)
    total_successes = sum(item["successes"] for item in per_combo)
    ci_low, ci_high = proportion_ci(
        total_successes,
        total_episodes,
        level=level,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "combos": per_combo,
    }


def evaluate_generalization(
    policy: object,
    split: GeneralizationSplit,
    n_episodes: int,
    *,
    runner: Callable[[str, int, str], bool] | None = None,
    env_factory: Callable[..., object] | None = None,
    base_seed: int = 500,
    level: float = CI_LEVEL,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = 0,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """在 `split` 上评测 `policy`，聚合 train / held-out 成功率、95% CI 与泛化 gap。

    Args:
        policy: 策略对象；仅当 `runner` 为 None 时使用（默认 runner 需其 `act`）。
        split: `build_split` 产出的切分。
        n_episodes: **每个组合**执行多少次 episode。
        runner: 自定义回放函数，签名 `runner(env_id, seed, instruction) -> bool`；
            第一个参数是 combo 的 `object`（默认 runner 直接把它当 `env_id`），
            返回该 episode 是否成功。语言条件评测**必须**自备 runner——默认 runner
            忽略 instruction。
        env_factory: 环境工厂，默认 `maniskill_intro.make_maniskill_env`。
        base_seed: episode seed 起点；train 段与 held-out 段取**互不相交**的连续区间。
        level / bootstrap_samples / bootstrap_seed: 置信区间参数。
        max_steps: 默认 runner 的 episode 长度，None 取 `DEFAULT_MAX_STEPS`。

    Returns:
        dict：`schema_version` / `protocol` / `n_episodes` / 逐组合结果 / `train` / `heldout`
        （各含 episodes、successes、success_rate、ci_low、ci_high）/ `gap`（4 位小数）/
        `scope_note` 等。
    """
    if isinstance(n_episodes, bool) or not isinstance(n_episodes, int) or n_episodes <= 0:
        raise VLAGeneralizationError("n_episodes 必须是正整数")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise VLAGeneralizationError("base_seed 必须是非负整数")
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0
    ):
        raise VLAGeneralizationError("max_steps 必须是正整数或 None")
    if runner is None:
        runner = _default_runner(policy, env_factory, max_steps)
    if not callable(runner):
        raise VLAGeneralizationError("runner 必须可调用")

    train_combos = list(split.train_combos)
    heldout_combos = list(split.heldout_combos)
    train_seeds = list(range(base_seed, base_seed + len(train_combos) * n_episodes))
    heldout_start = base_seed + len(train_combos) * n_episodes
    heldout_seeds = list(
        range(heldout_start, heldout_start + len(heldout_combos) * n_episodes)
    )
    # 无泄漏防线：train 段与 held-out 段的 episode seed 必须不相交（否则等于用同一批
    # episode 同时算两侧成功率，"泛化 gap"就失去意义）。
    _assert_disjoint_seeds(train_seeds, heldout_seeds)

    train_block = _run_block(
        train_combos,
        train_seeds,
        n_episodes,
        runner,
        level=level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    heldout_block = _run_block(
        heldout_combos,
        heldout_seeds,
        n_episodes,
        runner,
        level=level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    def block_payload(block: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: block[key]
            for key in ("episodes", "successes", "success_rate", "ci_low", "ci_high")
        }

    gap = round(train_block["success_rate"] - heldout_block["success_rate"], 4)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": split.protocol,
        "n_episodes": n_episodes,
        "base_seed": base_seed,
        "combos": {
            "train": train_block["combos"],
            "heldout": heldout_block["combos"],
        },
        "train": block_payload(train_block),
        "heldout": block_payload(heldout_block),
        "gap": gap,
        "scope_note": SCOPE_NOTE,
        "split_warnings": list(split.warnings),
        "heldout_target": HELDOUT_SUCCESS_TARGET,
        "heldout_meets_target": bool(
            heldout_block["success_rate"] >= HELDOUT_SUCCESS_TARGET
        ),
    }


def format_generalization_markdown(report: Mapping[str, Any]) -> str:
    """把评测报告渲染成 markdown（人读 + 交付物），与 `wm_ope.format_markdown` 同风格。"""
    train = report["train"]
    heldout = report["heldout"]
    lines = [
        "# 第 38 周 P2：组合泛化评测",
        "",
        f"- 协议：**{report['protocol']}**，每组合 episode 数：**{report['n_episodes']}**",
        f"- train 成功率 = **{train['success_rate']:.3f}**"
        f"（95% CI [{train['ci_low']:.3f}, {train['ci_high']:.3f}]，"
        f"{train['successes']}/{train['episodes']}）",
        f"- held-out 成功率 = **{heldout['success_rate']:.3f}**"
        f"（95% CI [{heldout['ci_low']:.3f}, {heldout['ci_high']:.3f}]，"
        f"{heldout['successes']}/{heldout['episodes']}）",
        f"- 泛化 gap = **{report['gap']:+.4f}**（train − held-out，4 位小数与 run_generalization 同口径）",
        f"- A7 门槛：held-out ≥ {report['heldout_target']:.0%} → "
        f"{'达标' if report['heldout_meets_target'] else '未达标（或降级为仿真协议验证）'}",
        "",
        "## 各组合明细",
        "",
        "| 分组 | 物体 | 指令 | 成功/总数 | 成功率 |",
        "|---|---|---|---|---|",
    ]
    for group in ("train", "heldout"):
        for item in report["combos"][group]:
            lines.append(
                f"| {group} | `{item['object']}` | {item['instruction']} | "
                f"{item['successes']}/{item['episodes']} | {item['success_rate']:.3f} |"
            )
    lines += ["", "## 口径边界", "", str(report["scope_note"])]
    if report.get("split_warnings"):
        lines += ["", "## split 告警", "", *[f"- {item}" for item in report["split_warnings"]]]
    return "\n".join(lines) + "\n"


def write_generalization_report(
    report: Mapping[str, Any], out_dir: Path
) -> tuple[Path, Path]:
    """落盘 `generalization.json` + `generalization.md`，返回两个路径。"""
    directory = Path(out_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "generalization.json"
    markdown_path = directory / "generalization.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(format_generalization_markdown(report), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "CI_LEVEL",
    "DEFAULT_MAX_STEPS",
    "HELDOUT_SUCCESS_TARGET",
    "PROTOCOLS",
    "SCHEMA_VERSION",
    "SCOPE_NOTE",
    "VLAGeneralizationError",
    "GeneralizationSplit",
    "build_split",
    "evaluate_generalization",
    "format_generalization_markdown",
    "proportion_ci",
    "write_generalization_report",
]
