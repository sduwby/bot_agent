"""B0–B3 感知消融的编排与聚合（第 38 周 P0 / 子任务 1.6）。

## 实验口径（用户决策 2026-10-11）

**B0 = 现状**：`RGB + robot_state(5) + env_state(13)`（即 `visual_act` 的历史行为），
B1–B3 在其上**追加** `ablation_features` 给出的额外列（列进 ENV 辅助向量尾部）。

## ⚠️ 这个口径下的固有局限（必须随结论一起报告，不能省）

`env_state` 本身就含 `obj_pos / goal_pos / (tcp−obj) / (obj−goal) / is_grasped`
（见 `lerobot_act.ENV_STATE_INDICES`），而本仓的 B1/B2 额外列是它们的**确定性函数**：

- B1 `depth_known_size_m` = ‖tcp−obj‖ —— 正好是 `env_state[6:9]` 的范数；
- B2 `contact_flag` = `is_grasped`（`env_state[12]`）、`load_gripper_permille` = 夹爪开合度（在 `robot_state` 里）。

也就是说 **B1/B2 的信息增量恒等于 0** → **B0/B1/B2 之间原则上不该出现任何可复现差异**，
A2（ΔGrasp ≥ 15pt）在"B0 = 现状"口径下**不可能由 B1/B2 达成**。
B3 的检测深度同样由物体真值位置决定，它引入的是**感知误差**而不是新信息。

**结论**：现行口径下这套消融只能当**冗余性对照 / 噪声敏感性探针**用，
**不能**用来论证"几何/深度信号值多少分"。要回答 A2 必须改用 **B0 = 仅 RGB**
（关闭 STATE/ENV 通道）——这一点已记入 design §13 偏差 67。

模块职责：变体计划 → 数据集维度校验 → 结果聚合（ΔGrasp / 闭合高度误差 / 失败归因）→ 报告落盘。
训练与评测通过**注入的 callable** 完成（本机无 GPU / 无仿真，故不在这里硬编码训练）。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from real_geometry import VARIANT_ORDER, variant_feature_names

ABLATION_SCHEMA_VERSION = "week38_depth_ablation_v1"

#: A2 的判据：相对基线提升的百分点。
TARGET_DELTA_GRASP_PP = 15.0

#: 必须随报告一起输出的口径说明（防"只看数字不看前提"）。
REDUNDANCY_CAVEAT = (
    "B0 = 现状（RGB + robot_state + env_state）口径下，env_state 已含 obj_pos/(tcp−obj)/is_grasped，"
    "而 B1/B2 的 oracle 列是它们的确定性函数 → 信息增量为 0，B0/B1/B2 之间的差异只应落在噪声内；"
    "故本表只能作冗余性对照/噪声探针，不能论证几何或深度信号的价值（要回答 A2 需改用 B0 = 仅 RGB）"
)


class AblationError(RuntimeError):
    """消融编排的契约错误。"""


@dataclass(frozen=True)
class AblationVariant:
    """一个变体：名字 + 它自己要追加的列 + 由此决定的 ENV 维度。"""

    name: str
    extra_features: tuple[str, ...]
    env_state_dim: int

    def __post_init__(self) -> None:
        if self.name not in VARIANT_ORDER:
            raise AblationError(f"未知变体 {self.name!r}，应为 {VARIANT_ORDER} 之一")
        if self.env_state_dim < 13:
            raise AblationError(f"{self.name} 的 ENV 维度必须 ≥ 13")


def build_plan(
    variants: Sequence[str] = VARIANT_ORDER,
    *,
    base_env_state_dim: int = 13,
    derive_features: bool = True,
) -> tuple[AblationVariant, ...]:
    """构造变体计划（按 `VARIANT_ORDER` 的规范顺序，去重）。

    ``derive_features=False`` 时表示"这些列不由 `real_geometry` 的变体布局产出"
    （例如自己实验性地加列）——此时**必须**显式给出列数，否则无法校验维度。
    为保持单一事实来源，默认走 `derive_features=True`。
    """
    if not variants:
        raise AblationError("variants 不能为空")
    if not derive_features:
        raise AblationError(
            "derive_features=False 需要显式列数，当前不支持；"
            "请用 real_geometry 的变体布局（VARIANT_ORDER）以保证列序唯一"
        )
    wanted = set(variants)
    unknown = sorted(wanted - set(VARIANT_ORDER))
    if unknown:
        raise AblationError(f"未知变体 {unknown}，应为 {VARIANT_ORDER} 之一")
    if "B0" not in wanted:
        raise AblationError("计划里必须有基线 B0（否则没有 ΔGrasp 的分母）")
    plan: list[AblationVariant] = []
    for name in VARIANT_ORDER:  # 规范顺序，保证跨运行可比
        if name not in wanted:
            continue
        extra = variant_feature_names(name)
        plan.append(
            AblationVariant(
                name=name,
                extra_features=extra,
                env_state_dim=base_env_state_dim + len(extra),
            )
        )
    return tuple(plan)


def validate_dataset_dim(variant: AblationVariant, actual_dim: int) -> None:
    """校验某变体的数据集 ENV 维度是否与计划一致。

    防的是最隐蔽的一类错：列数对不上却照样能开训 —— 策略会拿到**错位**的向量，
    指标"只是变差了"，事后无法归因。
    """
    if int(actual_dim) != variant.env_state_dim:
        raise AblationError(
            f"变体 {variant.name} 的数据集 observation.environment_state 维度为 {actual_dim}，"
            f"期望 {variant.env_state_dim}（= 13 + {len(variant.extra_features)} 列 "
            f"{list(variant.extra_features)}）。列数不符会让策略读到错位向量，"
            "指标只会变差而不报错，必须挡在开训之前"
        )


def read_env_state_dim(dataset_root: str | Path) -> int:
    """从 v3 数据集的 `meta/info.json` 读 `observation.environment_state` 的维度。

    只读 JSON、不碰 parquet/视频，因此可离线用于**开训前的维度门禁**。
    """
    path = Path(dataset_root).expanduser() / "meta" / "info.json"
    if not path.is_file():
        raise AblationError(f"{dataset_root} 不是 LeRobot 数据集（缺 meta/info.json）")
    info = json.loads(path.read_text(encoding="utf-8"))
    try:
        shape = info["features"]["observation.environment_state"]["shape"]
        return int(shape[0])
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise AblationError(
            f"{path} 缺少 observation.environment_state 的 shape：{exc}"
        ) from exc


@dataclass(frozen=True)
class VariantResult:
    """一个变体的训练+评测结果。"""

    variant: str
    success_rate: float
    n_episodes: int
    closing_height_error_mm: float | None = None
    failure_shares: Mapping[str, float] = field(default_factory=dict)
    train_seconds: float | None = None
    nonfinite_actions: int = 0

    def __post_init__(self) -> None:
        if self.variant not in VARIANT_ORDER:
            raise AblationError(f"未知变体 {self.variant!r}")
        if not 0.0 <= float(self.success_rate) <= 1.0:
            raise AblationError(f"success_rate 必须在 [0,1]，实际 {self.success_rate}")
        if int(self.n_episodes) <= 0:
            raise AblationError("n_episodes 必须是正整数")


@dataclass(frozen=True)
class AblationReport:
    """ΔGrasp 对照表 + 归因 + 口径说明。"""

    baseline: str
    results: tuple[VariantResult, ...]
    delta_grasp_pp: Mapping[str, float]
    target_pp: float
    verdict: str
    caveat: str = REDUNDANCY_CAVEAT
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ABLATION_SCHEMA_VERSION,
            "baseline": self.baseline,
            "target_pp": self.target_pp,
            "verdict": self.verdict,
            "delta_grasp_pp": dict(self.delta_grasp_pp),
            "variants": [
                {
                    "variant": item.variant,
                    "success_rate": item.success_rate,
                    "n_episodes": item.n_episodes,
                    "closing_height_error_mm": item.closing_height_error_mm,
                    "failure_shares": dict(item.failure_shares),
                    "train_seconds": item.train_seconds,
                    "nonfinite_actions": item.nonfinite_actions,
                }
                for item in self.results
            ],
            "caveat": self.caveat,
            "notes": list(self.notes),
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"基线 {self.baseline}｜判据 ΔGrasp ≥ {self.target_pp:.0f}pt｜判定：{self.verdict}",
            f"{'变体':<6}{'成功率':>10}{'Δ(pp)':>10}{'闭合高度误差(mm)':>18}{'集数':>6}",
        ]
        for item in self.results:
            delta = self.delta_grasp_pp.get(item.variant)
            delta_text = "—" if delta is None else f"{delta:+.1f}"
            error = item.closing_height_error_mm
            error_text = "—" if error is None else f"{error:.2f}"
            lines.append(
                f"{item.variant:<6}{item.success_rate:>10.3f}{delta_text:>10}"
                f"{error_text:>18}{item.n_episodes:>6}"
            )
        lines.append(f"口径：{self.caveat}")
        for note in self.notes:
            lines.append(f"[警告] {note}")
        return lines


def _validated_shares(shares: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in shares.values())
    if shares and not 0.0 < total <= 1.0 + 1e-6:
        raise AblationError(f"failure_shares 之和应在 (0,1]，实际 {total:.4f}")
    return {str(key): float(value) for key, value in shares.items()}


def summarize_ablation(
    results: Sequence[VariantResult],
    *,
    baseline: str = "B0",
    target_pp: float = TARGET_DELTA_GRASP_PP,
    caveat: str = REDUNDANCY_CAVEAT,
) -> AblationReport:
    """把各变体结果聚合成 ΔGrasp 对照表。

    ΔGrasp 以**百分点**为单位（成功率差 × 100），与设计 §7.1 / A2 同口径。

    `caveat` 可覆盖：**口径不同结论的可解释性也不同**——`REDUNDANCY_CAVEAT` 只适用于
    "B0 已含 env_state"的旧口径；B0 若是 `extra_only`（无 ENV），旧文案就是错的。
    """
    if not results:
        raise AblationError("results 不能为空")
    by_name = {item.variant: item for item in results}
    if len(by_name) != len(results):
        raise AblationError("同一变体出现多次")
    if baseline not in by_name:
        raise AblationError(f"结果里没有基线 {baseline}")
    base_rate = by_name[baseline].success_rate
    deltas = {
        item.variant: round((item.success_rate - base_rate) * 100.0, 4)
        for item in results
        if item.variant != baseline
    }
    non_baseline = [item for item in results if item.variant != baseline]
    if not non_baseline:
        verdict = "n/a"
    else:
        best = max(deltas.values())
        verdict = "pass" if best >= float(target_pp) else "fail"
    notes: list[str] = []
    if any(item.nonfinite_actions for item in results):
        notes.append("存在非有限动作：该变体的成功率不可比，需先修策略再重跑")
    if any(item.failure_shares and abs(sum(item.failure_shares.values()) - 1.0) > 0.05
           for item in results):
        notes.append("失败归因占比之和偏离 1 超过 5%：归因分类可能漏了失败类型")
    return AblationReport(
        baseline=baseline,
        results=tuple(results),
        delta_grasp_pp=deltas,
        target_pp=float(target_pp),
        verdict=verdict,
        caveat=caveat,
        notes=tuple(notes),
    )


def write_ablation_report(report: AblationReport, out_dir: Path) -> tuple[Path, Path]:
    """落盘 `ablation.json` + `ablation.md`。"""
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "ablation.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path = directory / "ablation.md"
    body = ["# P0 感知消融（B0–B3）", "", *[f"- {line}" for line in report.summary_lines()], ""]
    markdown_path.write_text("\n".join(body), encoding="utf-8")
    return json_path, markdown_path


def run_ablation(
    plan: Sequence[AblationVariant],
    datasets: Mapping[str, Path],
    *,
    eval_fn: Callable[[AblationVariant, Path], VariantResult],
    train_fn: Callable[[AblationVariant, Path], float | None] | None = None,
    baseline: str = "B0",
    target_pp: float = TARGET_DELTA_GRASP_PP,
) -> AblationReport:
    """按计划逐变体训练+评测并聚合。

    `train_fn` / `eval_fn` **注入**：本机没有 GPU/仿真，硬编码训练路径无法离线验证；
    注入后编排逻辑（顺序、维度校验、聚合）就能被纯逻辑测试覆盖。
    B0 通常不需要训练（它就是历史基线），`train_fn` 返回 `None` 表示跳过训练。
    """
    results: list[VariantResult] = []
    for variant in plan:
        dataset = datasets.get(variant.name)
        if dataset is None:
            raise AblationError(
                f"缺少变体 {variant.name} 的数据集目录（每个变体的 ENV 列数不同，必须各给一份）"
            )
        if train_fn is not None and variant.name != baseline:
            train_fn(variant, dataset)
        result = eval_fn(variant, dataset)
        if result.variant != variant.name:
            raise AblationError(
                f"eval_fn 返回的变体名 {result.variant!r} 与计划 {variant.name!r} 不一致"
            )
        results.append(
            VariantResult(
                variant=result.variant,
                success_rate=result.success_rate,
                n_episodes=result.n_episodes,
                closing_height_error_mm=result.closing_height_error_mm,
                failure_shares=_validated_shares(result.failure_shares),
                train_seconds=result.train_seconds,
                nonfinite_actions=result.nonfinite_actions,
            )
        )
    return summarize_ablation(results, baseline=baseline, target_pp=target_pp)
