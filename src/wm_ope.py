#!/usr/bin/env python
"""第 38 周 P3：世界模型离线策略评估（OPE）——WM 预测成功率 vs 仿真实测的排序相关性。

研究问题：能否用 `world_model.py` 的动力学模型在**不跑仿真**的情况下，正确排出
一组候选策略的优劣顺序？若排序相关性够高，WM 就能当"廉价筛选器"用。

**口径边界（必须写进报告）**：`world_model.py` 的状态空间只有 18 维，因此只能
rollout **状态输入**的 PushCube 策略族（scripted / BC / 退化变体 / 随机）。
图像输入策略（视觉 ACT、SmolVLA）**不在范围内**——它们的观测包含相机帧，
无法用 18 维状态前向。这是本工具的外推边界，不是缺陷。

模块职责（纯逻辑，不依赖 ManiSkill）：
- `OPEReport` / `DistributionShift` 数据结构
- `spearman_rho`（含并列值处理）、bootstrap 置信区间
- `evaluate_ope`：候选校验（数量/跨族/rollout 数）→ ρ → CI → 判定
- `rollout_policy_in_world_model`：WM 内 rollout 候选策略
- `distribution_shift` / `attribute_low_rho`：ρ 偏低时的归因路径
- `write_ope_report`：落盘 report.json + report.md

仿真实测侧由 `run_wm_ope.py --eval-sim` 在云上（ManiSkill）产出，与本模块解耦。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

OPE_SCHEMA_VERSION = "week38_wm_ope_v1"

STATE_DIM = 18
ACTION_DIM = 4

#: 候选数下限（设计 §7.5：候选 ≥5，否则排序相关性没有统计意义）
MIN_CANDIDATES = 5
#: 每候选 rollout 数下限（设计 §7.5：每候选 ≥10 次）
MIN_ROLLOUTS_PER_CANDIDATE = 10
#: A8：ρ ≥ 0.6 视为工具可用
RHO_USABLE = 0.6
#: 设计 §9.2：ρ < 0.3 视为负结果（仍需提交归因报告）
RHO_LOW = 0.3

BOOTSTRAP_SAMPLES = 2000
CI_LEVEL = 0.95

#: 训练分布里 std 低于该值的维度视为"近似恒定"：其 z 分数无意义，统计访问偏移时跳过。
CONSTANT_STD_EPS = 1e-3

#: PushCube 成功判据（obj 与 goal 的 **xy** 距离）。
#: 为什么只看 xy：物体在桌面上（z≈0.02）、目标标记贴地（z≈0.001），z 方向恒有约
#: 0.019 的固定偏移；若用 3D 距离配 0.025 阈值，等效要求 xy < ~0.016，几乎不可能达标
#: （第 38 周实测 WM 侧成功率因此恒为 0、ρ 无定义）。`ScriptedPolicy` 的 push 终止判据
#: 与 ManiSkill 的 PushCube 成功判据都是 xy 口径，此处对齐。
PUSHCUBE_SUCCESS_DIST = 0.025

#: 18 维状态切片（与 `behavior_cloning.observation_to_state` 一致）
SLICE_TCP = slice(0, 3)
SLICE_OBJ = slice(3, 6)
SLICE_GOAL = slice(6, 9)
SLICE_QPOS_LAST2 = slice(15, 17)
INDEX_GRASPED = 17

SCOPE_NOTE = (
    "口径限定：世界模型仅 18 维状态空间，本报告只覆盖**状态输入**的 PushCube 策略族"
    "（scripted / bc / 退化变体 / 随机）。图像输入策略（视觉 ACT、SmolVLA）不在范围内，"
    "不可据此外推到真机或视觉策略。"
)


class WMOPEError(RuntimeError):
    """OPE 契约错误。"""


@dataclass(frozen=True)
class DistributionShift:
    """候选访问分布相对 WM 训练分布的偏移（归因用）。"""

    candidate: str
    mean_abs_z: float
    p95_abs_z: float
    ood_fraction: float          # |z| > 3 的样本占比
    n_dims: int = STATE_DIM      # 参与统计的维度数（排除训练中近似恒定的维度）


@dataclass(frozen=True)
class OPEReport:
    """OPE 报告（对应设计 §4.2；末尾字段为本次实现补充，见 design §13）。"""

    candidates: list[str]
    candidate_family: dict[str, str]
    real_success: list[float]
    wm_pred_success: list[float]
    real_success_var: list[float]
    spearman_rho: float
    ci_low: float
    ci_high: float
    n_rollouts_per_candidate: int
    scope_note: str = SCOPE_NOTE
    warnings: tuple[str, ...] = ()
    verdict: str = ""
    attribution: tuple[DistributionShift, ...] = field(default=())

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": OPE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidates": list(self.candidates),
            "candidate_family": dict(self.candidate_family),
            "real_success": list(self.real_success),
            "wm_pred_success": list(self.wm_pred_success),
            "real_success_var": list(self.real_success_var),
            "spearman_rho": self.spearman_rho,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_rollouts_per_candidate": self.n_rollouts_per_candidate,
            "scope_note": self.scope_note,
            "warnings": list(self.warnings),
            "verdict": self.verdict,
            "attribution": [
                {
                    "candidate": item.candidate,
                    "mean_abs_z": item.mean_abs_z,
                    "p95_abs_z": item.p95_abs_z,
                    "ood_fraction": item.ood_fraction,
                    "n_dims": item.n_dims,
                }
                for item in self.attribution
            ],
        }


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """平均秩（并列值取平均），用于带并列的 Spearman。"""
    array = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    sorted_values = array[order]
    index = 0
    while index < sorted_values.size:
        end = index
        while end + 1 < sorted_values.size and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        ranks[order[index : end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    return ranks


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman 秩相关（并列取平均秩）。某侧取值全相同则无定义 → 抛错。"""
    left = np.asarray(a, dtype=float).reshape(-1)
    right = np.asarray(b, dtype=float).reshape(-1)
    if left.size != right.size:
        raise WMOPEError(f"两组长度不一致：{left.size} vs {right.size}")
    if left.size < 3:
        raise WMOPEError(f"至少需要 3 个候选才算排序相关性，实际 {left.size}")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise WMOPEError("Spearman 输入含非有限数")
    rank_left = _average_ranks(left)
    rank_right = _average_ranks(right)
    centered_left = rank_left - rank_left.mean()
    centered_right = rank_right - rank_right.mean()
    denominator = float(np.sqrt((centered_left @ centered_left) * (centered_right @ centered_right)))
    if denominator <= 0.0:
        raise WMOPEError("某侧取值全相同（秩方差为 0），Spearman 无定义")
    return float((centered_left @ centered_right) / denominator)


def _bootstrap_ci(
    real_rates: Sequence[float],
    wm_rates: Sequence[float],
    *,
    n_rollouts: int,
    rng: np.random.Generator,
    samples: int = BOOTSTRAP_SAMPLES,
    level: float = CI_LEVEL,
) -> tuple[float, float]:
    """对每候选的 Bernoulli 结果重采样 → 重算 ρ → 百分位区间。"""
    rho_samples: list[float] = []
    for _ in range(samples):
        resampled = [
            float(rng.binomial(n_rollouts, min(max(rate, 0.0), 1.0))) / n_rollouts
            for rate in real_rates
        ]
        try:
            rho_samples.append(spearman_rho(resampled, wm_rates))
        except WMOPEError:
            continue
    if not rho_samples:
        raise WMOPEError("bootstrap 未能得到任何有效 ρ 样本（真实成功率取值过于集中）")
    alpha = (1.0 - level) / 2.0 * 100.0
    return (
        float(np.percentile(rho_samples, alpha)),
        float(np.percentile(rho_samples, 100.0 - alpha)),
    )


def _validate_candidates(
    candidates: Mapping[str, Sequence[float]],
    candidate_family: Mapping[str, str],
    real_success: Sequence[float],
) -> tuple[list[str], int, list[str]]:
    names = list(candidates.keys())
    if len(names) < MIN_CANDIDATES:
        raise WMOPEError(
            f"候选数 {len(names)} < {MIN_CANDIDATES}：排序相关性没有统计意义（设计 §7.5）"
        )
    if len(real_success) != len(names):
        raise WMOPEError(
            f"real_success 长度 {len(real_success)} 与候选数 {len(names)} 不一致"
        )
    lengths = {name: len(list(candidates[name])) for name in names}
    short = {name: size for name, size in lengths.items() if size < MIN_ROLLOUTS_PER_CANDIDATE}
    if short:
        raise WMOPEError(
            f"每候选至少 {MIN_ROLLOUTS_PER_CANDIDATE} 次 rollout，实际不足：{short}"
        )
    distinct = set(lengths.values())
    if len(distinct) != 1:
        raise WMOPEError(f"各候选 rollout 数必须一致，实际 {lengths}")

    warnings: list[str] = []
    families = {candidate_family.get(name, "") for name in names}
    families.discard("")
    if len(families) < 2:
        warnings.append(
            f"候选只覆盖 {len(families)} 个行为族（{sorted(families)}）："
            "成功率差异易被噪声淹没，ρ 的统计意义不足（设计 §3 决策 4）"
        )
    missing = [name for name in names if name not in candidate_family]
    if missing:
        warnings.append(f"以下候选未声明行为族：{missing}")
    for rate in real_success:
        if not (0.0 <= float(rate) <= 1.0):
            raise WMOPEError(f"真实成功率必须在 [0,1]，实际 {rate}")
    return names, distinct.pop(), warnings


def evaluate_ope(
    candidates: Mapping[str, Sequence[float]],
    candidate_family: Mapping[str, str],
    real_success: Sequence[float],
    *,
    rng: np.random.Generator | None = None,
    attribution: Sequence[DistributionShift] = (),
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> OPEReport:
    """候选校验 → WM 预测成功率 → ρ + bootstrap CI + 候选方差 → 判定。

    Args:
        candidates: 候选名 → **WM 内每次 rollout 的成败**（0/1 序列，≥10 条）。
        candidate_family: 候选名 → 行为族（scripted / bc / random / …）。
        real_success: 与候选顺序一致的**仿真实测成功率**。
        rng: bootstrap 随机源（便于复现）。
        attribution: ρ 偏低时的归因（由 `attribute_low_rho` 产出）。
        bootstrap_samples: bootstrap 次数。

    Returns:
        `OPEReport`；`verdict` 取 usable / marginal / negative_result。
    """
    names, n_rollouts, warnings = _validate_candidates(candidates, candidate_family, real_success)
    wm_rates = [float(np.mean(np.asarray(candidates[name], dtype=float))) for name in names]
    real_rates = [float(rate) for rate in real_success]
    real_var = [rate * (1.0 - rate) for rate in real_rates]

    rho = spearman_rho(real_rates, wm_rates)
    generator = rng if rng is not None else np.random.default_rng(20261010)
    ci_low, ci_high = _bootstrap_ci(
        real_rates, wm_rates, n_rollouts=n_rollouts, rng=generator, samples=bootstrap_samples
    )

    if rho >= RHO_USABLE:
        verdict = "usable"
    elif rho < RHO_LOW:
        verdict = "negative_result"
        warnings.append(
            f"ρ={rho:.3f} < {RHO_LOW}：按设计 §9.2 作为**负结果**提交，"
            "并附归因报告；工具在状态口径下不可用作廉价筛选器"
        )
    else:
        verdict = "marginal"

    return OPEReport(
        candidates=names,
        candidate_family={name: candidate_family.get(name, "") for name in names},
        real_success=real_rates,
        wm_pred_success=wm_rates,
        real_success_var=real_var,
        spearman_rho=rho,
        ci_low=ci_low,
        ci_high=ci_high,
        n_rollouts_per_candidate=n_rollouts,
        warnings=tuple(warnings),
        verdict=verdict,
        attribution=tuple(attribution),
    )


def state_to_observation(state: Sequence[float]) -> dict[str, object]:
    """18 维状态 → 合成 observation，供 `act(observation)` 型策略在 WM 内使用。

    只填策略真正读取的字段（`extra.tcp_pose/obj_pose/goal_pos/is_grasped` 与
    `agent.qpos[-2:]`）；其余按 0 补齐，保持与 `observation_to_state` 可逆。
    """
    array = np.asarray(state, dtype=float).reshape(-1)
    if array.size != STATE_DIM or not np.all(np.isfinite(array)):
        raise WMOPEError(f"state shape={array.shape} 期望 ({STATE_DIM},) 且有限")
    tcp = array[SLICE_TCP]
    obj = array[SLICE_OBJ]
    goal = array[SLICE_GOAL]

    def pad7(vector: np.ndarray) -> np.ndarray:
        return np.concatenate((vector, np.zeros(4, dtype=float)))

    return {
        "agent": {
            "qpos": np.concatenate((np.zeros(7, dtype=float), array[SLICE_QPOS_LAST2])),
        },
        "extra": {
            "tcp_pose": pad7(tcp),
            "obj_pose": pad7(obj),
            "goal_pos": goal.astype(float),
            "is_grasped": bool(array[INDEX_GRASPED] > 0.5),
        },
    }


def rollout_with_trajectory(
    policy: object,
    wm: object,
    init_states: Sequence[Sequence[float]],
    horizon: int,
    *,
    success_dist: float = PUSHCUBE_SUCCESS_DIST,
) -> tuple[list[float], list[np.ndarray]]:
    """在动力学模型内 rollout 候选策略，返回（每次成败, 每次**访问过的状态**）。

    - `policy` 需实现 `act(observation) -> action[4]`；有 `reset()` 时每条 rollout 前调用
      （scripted 是状态机，不 reset 会把上一轮的阶段带进来）。
    - 成功判据：rollout 内 **任一步** obj 与 goal 的 **xy** 距离 < `success_dist`
      （与 `ScriptedPolicy` 的 push 终止判据、ManiSkill 的 PushCube 成功判据同口径；
      只看 xy 的原因见 `PUSHCUBE_SUCCESS_DIST` 注释）。
    - 访问状态供 ρ 偏低时的归因使用：必须用候选**实际走过**的状态而不是初始状态——
      两侧初始状态按设计本来相同，用它做归因没有任何区分度（第 38 周实测踩到）。
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise WMOPEError("horizon 必须是正整数")
    states = [np.asarray(item, dtype=float).reshape(-1) for item in init_states]
    if not states:
        raise WMOPEError("init_states 不能为空")
    for index, item in enumerate(states):
        if item.size != STATE_DIM or not np.all(np.isfinite(item)):
            raise WMOPEError(f"init_states[{index}] 形状或数值非法：{item.shape}")

    act = getattr(policy, "act", None)
    reset = getattr(policy, "reset", None)
    if not callable(act):
        raise WMOPEError("policy 必须实现 act(observation)")

    results: list[float] = []
    trajectories: list[np.ndarray] = []
    for state in states:
        if callable(reset):
            reset()
        current = state.reshape(1, STATE_DIM).copy()
        visited: list[np.ndarray] = [current[0].copy()]
        success = False
        for _ in range(horizon):
            action = np.asarray(act(state_to_observation(current[0])), dtype=float).reshape(-1)
            if action.size != ACTION_DIM or not np.all(np.isfinite(action)):
                raise WMOPEError(f"策略输出 action 非法：shape={action.shape}")
            current = np.asarray(
                wm.predict_next(current, np.clip(action, -1.0, 1.0).reshape(1, ACTION_DIM)),
                dtype=float,
            ).reshape(1, STATE_DIM)
            if not np.all(np.isfinite(current)):
                raise WMOPEError("动力学前向产生非有限状态")
            visited.append(current[0].copy())
            distance = float(
                np.linalg.norm(
                    current[0, SLICE_OBJ][:2] - current[0, SLICE_GOAL][:2]
                )
            )
            if distance < success_dist:
                success = True
                break
        results.append(1.0 if success else 0.0)
        trajectories.append(np.stack(visited, axis=0))
    return results, trajectories


def rollout_policy_in_world_model(
    policy: object,
    wm: object,
    init_states: Sequence[Sequence[float]],
    horizon: int,
    *,
    success_dist: float = PUSHCUBE_SUCCESS_DIST,
) -> list[float]:
    """同 `rollout_with_trajectory`，只返回每次 rollout 的成败（设计 §5.1 的签名）。"""
    results, _ = rollout_with_trajectory(
        policy, wm, init_states, horizon, success_dist=success_dist
    )
    return results


def distribution_shift(
    wm: object, visited: Sequence[Sequence[float]], *, candidate: str, ood_z: float = 3.0
) -> DistributionShift:
    """候选访问状态相对 WM **训练分布**的偏移（标准化空间 |z| 统计）。

    **只统计训练中有方差的维度**：训练里近似恒定的维度（如 PushCube 的夹爪关节、
    `is_grasped`）其 std≈0，`world_model._normalization_stats` 会把它们兜成 1.0，
    此时 z 分数没有意义、还会把所有样本都判成 OOD（第 38 周实测全候选 OOD=1.0）。
    """
    array = np.asarray(visited, dtype=float)
    if array.ndim != 2 or array.shape[1] != STATE_DIM:
        raise WMOPEError(f"visited shape={array.shape} 期望 [N,{STATE_DIM}]")
    if array.shape[0] == 0 or not np.all(np.isfinite(array)):
        raise WMOPEError("visited 不能为空或含非有限数")
    mean = np.asarray(wm.state_mean, dtype=float)
    std = np.asarray(wm.state_std, dtype=float)
    if mean.shape != (STATE_DIM,) or std.shape != (STATE_DIM,):
        raise WMOPEError("WM 归一化统计维度不符")
    if np.any(std <= 0):
        raise WMOPEError("WM state_std 含非正值")
    informative = std > CONSTANT_STD_EPS
    if not bool(np.any(informative)):
        raise WMOPEError("WM 训练分布在所有维度上都近似恒定，无法统计访问偏移")
    z = np.abs((array[:, informative] - mean[informative]) / std[informative])
    per_sample = z.max(axis=1)
    return DistributionShift(
        candidate=candidate,
        mean_abs_z=float(per_sample.mean()),
        p95_abs_z=float(np.percentile(per_sample, 95)),
        ood_fraction=float(np.mean(per_sample > ood_z)),
        n_dims=int(informative.sum()),
    )


def attribute_low_rho(
    wm: object, visited_by_candidate: Mapping[str, Sequence[Sequence[float]]]
) -> tuple[DistributionShift, ...]:
    """ρ 偏低的归因路径：各候选的访问分布相对 WM 训练分布偏移多大。"""
    return tuple(
        distribution_shift(wm, states, candidate=name)
        for name, states in visited_by_candidate.items()
    )


def format_markdown(report: OPEReport) -> str:
    """把报告渲染成 markdown（人读 + 交付物）。"""
    rows = [
        "| 候选 | 行为族 | 仿真实测 | WM 预测 | 实测方差 |",
        "|---|---|---|---|---|",
    ]
    for index, name in enumerate(report.candidates):
        rows.append(
            f"| `{name}` | {report.candidate_family.get(name, '')} | "
            f"{report.real_success[index]:.3f} | {report.wm_pred_success[index]:.3f} | "
            f"{report.real_success_var[index]:.4f} |"
        )
    lines = [
        "# 第 38 周 P3：世界模型 OPE 排序相关性",
        "",
        f"- 候选数：**{len(report.candidates)}**，每候选 rollout：**{report.n_rollouts_per_candidate}**",
        f"- Spearman ρ = **{report.spearman_rho:.3f}**"
        f"（95% bootstrap CI [{report.ci_low:.3f}, {report.ci_high:.3f}]）",
        f"- 判定：**{report.verdict}**（A8 门槛 ρ ≥ {RHO_USABLE}）",
        "",
        "## 候选对比",
        "",
        *rows,
        "",
        "## 口径边界",
        "",
        report.scope_note,
    ]
    if report.warnings:
        lines += ["", "## 告警", "", *[f"- {item}" for item in report.warnings]]
    if report.attribution:
        lines += [
            "",
            "## 归因：访问分布偏移（|z| 越大越偏离 WM 训练分布）",
            "",
            "| 候选 | 平均 |z| | P95 |z| | OOD 占比 |",
            "|---|---|---|---|",
            *[
                f"| `{item.candidate}` | {item.mean_abs_z:.2f} | {item.p95_abs_z:.2f} | "
                f"{item.ood_fraction:.3f} |"
                for item in report.attribution
            ],
        ]
    return "\n".join(lines) + "\n"


def write_ope_report(report: OPEReport, out_dir: Path) -> tuple[Path, Path]:
    """落盘 report.json + report.md，返回两个路径。"""
    directory = Path(out_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    markdown_path = directory / "report.md"
    json_path.write_text(
        json.dumps(report.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(format_markdown(report), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "OPE_SCHEMA_VERSION",
    "ACTION_DIM",
    "BOOTSTRAP_SAMPLES",
    "CI_LEVEL",
    "MIN_CANDIDATES",
    "MIN_ROLLOUTS_PER_CANDIDATE",
    "PUSHCUBE_SUCCESS_DIST",
    "RHO_LOW",
    "RHO_USABLE",
    "SCOPE_NOTE",
    "STATE_DIM",
    "DistributionShift",
    "OPEReport",
    "WMOPEError",
    "attribute_low_rho",
    "distribution_shift",
    "evaluate_ope",
    "format_markdown",
    "rollout_policy_in_world_model",
    "rollout_with_trajectory",
    "spearman_rho",
    "state_to_observation",
    "write_ope_report",
]
