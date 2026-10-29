"""P0 感知消融的**评测**链路（B0/B1/B2）：把仿真观测还原成策略的额外观测通道。

## 这个模块存在的理由

训练侧（`run_depth_ablation` / `visual_act`）已经把 B1/B2 的额外列**离线**追加进了数据集，
但**评测时策略在仿真里跑 real rollout**，那里没有数据集的那几列 —— 必须从**实时观测**
把同样的量还原出来，且要与训练侧**同序同口径**。差一个列序，策略会收到错位向量：
数值仍然"有限"，成功率只是变差，事后无法归因（这正是 `real_geometry` 反复强调的一类错）。

## 单点复用，绝不重写公式

特征公式只存在于 `ablation_features`（`oracle_features`）与 `real_geometry`
（`variant_feature_names` 定义列序）。本模块只做两件事：

1. 从仿真 `observation` 里**取数**，装配成 `OracleInputs`（`sim_oracle_inputs`）；
2. 把 `oracle_features` 的输出**按 `variant_feature_names` 的列序**排成向量（`make_oracle_provider`）。

**缺任一观测字段一律报错**，绝不补 0：补 0 会把"这一帧没拿到该量"伪装成一个合法特征值，
消融结论随之失真且无法事后区分 —— 与 `ablation_dataset.frame_oracle_inputs` 同一风格。

## 为什么 B0 也在这里

B0 的 `extra_provider=None`（策略本就不带额外通道），但计划渲染与列序校验仍要走同一入口，
以免"B0 无额外列"这个事实散落成多处判断。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from ablation_dataset import GRASPED_THRESHOLD, STATE_KEY, closure_range
from ablation_features import OracleInputs, oracle_features
from depth_ablation import VariantResult
from real_geometry import variant_feature_names


class AblationEvalError(RuntimeError):
    """消融评测链路的契约错误（缺观测字段、形状不符、无 episode 等）。"""


def _as_vector(value: object, name: str, min_length: int) -> np.ndarray:
    """把观测字段解析成一维数值向量，维度不足即报错。"""
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise AblationEvalError(f"{name} 无法解析为数值向量：{exc}") from exc
    if array.shape[0] < min_length:
        raise AblationEvalError(f"{name} 至少需要 {min_length} 维，实际 {array.shape[0]}")
    return array


def _as_scalar(value: object, name: str) -> float:
    """把观测字段解析成标量浮点（不接受向量）。"""
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise AblationEvalError(f"{name} 无法解析为数值：{exc}") from exc
    if array.size != 1:
        raise AblationEvalError(f"{name} 必须是标量，实际 {array.size} 个元素")
    return float(array[0])


def _require_mapping(container: object, name: str) -> Mapping[str, object]:
    if not isinstance(container, Mapping):
        raise AblationEvalError(f"{name} 必须是 mapping，实际 {type(container).__name__}")
    return container


def _require_field(container: Mapping[str, object], key: str, where: str) -> object:
    if key not in container:
        raise AblationEvalError(f"{where} 缺少字段 {key!r}")
    return container[key]


def sim_oracle_inputs(
    observation: Mapping[str, object],
    *,
    open_width: float,
    closed_width: float,
) -> OracleInputs:
    """把仿真观测还原成 B1/B2 的 oracle 输入（与训练侧同口径）。

    字段位置（已确认）：
    - ``observation["extra"]["tcp_pose"]`` 前 3 维 → tcp 位置；
    - ``observation["extra"]["obj_pose"]`` 前 3 维 → 物体位置；
    - ``observation["extra"]["is_grasped"]`` > `GRASPED_THRESHOLD` → 是否夹住；
    - ``observation["agent"]["qpos"]`` 最后 2 维之和 → 夹爪开合度（与
      `ablation_dataset.GRIPPER_STATE_SLICE` 的 ``np.sum(state[3:5])`` 同一语义）。

    取这四处而不是"数据集的 5 维/13 维向量"：仿真实时观测用的就是这些键，且此处**不做任何
    补 0**——缺字段即失败，避免把"没拿到"当成"等于 0"。
    """
    extra = _require_mapping(
        _require_field(_require_mapping(observation, "observation"), "extra", "observation"),
        'observation["extra"]',
    )
    agent = _require_mapping(
        _require_field(_require_mapping(observation, "observation"), "agent", "observation"),
        'observation["agent"]',
    )
    tcp = _as_vector(_require_field(extra, "tcp_pose", 'observation["extra"]'), "extra.tcp_pose", 3)
    obj = _as_vector(_require_field(extra, "obj_pose", 'observation["extra"]'), "extra.obj_pose", 3)
    grasped = _as_scalar(
        _require_field(extra, "is_grasped", 'observation["extra"]'), "extra.is_grasped"
    )
    qpos = _as_vector(_require_field(agent, "qpos", 'observation["agent"]'), "agent.qpos", 2)
    return OracleInputs(
        tcp_pos_m=tcp[:3].tolist(),
        obj_pos_m=obj[:3].tolist(),
        is_grasped=bool(grasped > float(GRASPED_THRESHOLD)),
        gripper_width=float(np.sum(qpos[-2:])),
        open_width=float(open_width),
        closed_width=float(closed_width),
    )


def make_oracle_provider(
    variant: str,
    *,
    open_width: float,
    closed_width: float,
) -> Callable[[Mapping[str, object]], list[float]]:
    """构造供 `visual_act.VisualACTPolicy` 使用的额外观测通道 provider。

    返回的函数对每个观测：``sim_oracle_inputs`` → ``oracle_features`` → 按
    `variant_feature_names(variant)` 的**列序**输出 ``list[float]``。

    **列序必须来自 `variant_feature_names`**：这是训练与推理共享的单一事实来源，
    自己排一遍就等于埋一颗"错位但不报错"的雷。B0 返回空列表（无额外列）。

    变体名在**构造时**即校验（fail-fast），而不是等第一次推理才报错。
    """
    # 列序只取一次：既是单一事实来源，也让非法变体在构造阶段就暴露。
    names = variant_feature_names(variant)

    def provider(observation: Mapping[str, object]) -> list[float]:
        inputs = sim_oracle_inputs(
            observation, open_width=open_width, closed_width=closed_width
        )
        values = oracle_features(variant, inputs)
        missing = [name for name in names if name not in values]
        if missing:
            # B3 需要检测路径（两视角像素）才能产出立体深度列；本评测只覆盖 B0/B1/B2。
            raise AblationEvalError(
                f"变体 {variant} 缺少特征 {missing}：oracle 路径无法产出这些列"
                "（B3 需检测路径输入，本评测链路不支持）"
            )
        return [float(values[name]) for name in names]

    return provider


def closure_range_from_dataset(dataset_dir: str | Path) -> tuple[float, float]:
    """从变体数据集的 parquet 里取夹爪实际开合范围（max=张开、min=闭合）。

    评测要还原训练时的归一化口径，因此 open/closed 必须与**训练用数据**同源，
    而不是硬编码常数。`pyarrow` **函数内惰性导入**（仓库既有风格），
    好让本模块的纯逻辑部分在无 pyarrow 的机器上也能被导入与测试。
    """
    import pyarrow.parquet as pq  # 重型可选依赖：惰性导入

    root = Path(dataset_dir).expanduser()
    data_dir = root / "data"
    if not data_dir.is_dir():
        raise AblationEvalError(f"{root} 不是变体数据集（缺 data 目录）")
    parquets = sorted(data_dir.rglob("*.parquet"))
    if not parquets:
        raise AblationEvalError(f"{data_dir} 下没有 parquet 数据文件")
    arrays = [
        np.asarray(pq.read_table(str(path)).column(STATE_KEY).to_pylist(), dtype=np.float64)
        for path in parquets
    ]
    return closure_range(np.concatenate(arrays, axis=0))


def success_rate(episodes: Sequence[object]) -> float:
    """``成功 episode 数 / 总数``；空列表直接报错（0/0 不是合法结论）。"""
    if not episodes:
        raise AblationEvalError("episodes 不能为空：成功率的分母不能是 0")
    return sum(bool(getattr(item, "success")) for item in episodes) / len(episodes)


def build_variant_result(
    variant: str,
    episodes: Sequence[object],
    *,
    train_seconds: float | None = None,
) -> VariantResult:
    """把某变体的评测 episodes 聚合成 `depth_ablation.VariantResult`。

    `closing_height_error_mm` 传 `None`：那是"抓取闭合高度"这类需要额外插桩的量，
    本评测链路拿不到 —— 不拿一个替代数冒充（`VariantResult` 允许其为 None）。
    """
    nonfinite = sum(int(getattr(item, "nonfinite_actions", 0) or 0) for item in episodes)
    return VariantResult(
        variant=variant,
        success_rate=success_rate(episodes),
        n_episodes=len(episodes),
        closing_height_error_mm=None,
        train_seconds=train_seconds,
        nonfinite_actions=nonfinite,
    )


def evaluation_episodes(summary: object) -> tuple[object, ...]:
    """从 `lerobot_act.PolicyEvaluationSummary` 取出 episode 元组。

    只依赖 `.episodes` 属性（鸭子类型），因此本模块不必导入 lerobot —— 让纯逻辑测试
    在没有 torch/lerobot 的环境也能覆盖聚合链路的其余部分。
    """
    episodes = getattr(summary, "episodes", None)
    if episodes is None:
        raise AblationEvalError(
            f"评测摘要缺少 episodes 字段（{type(summary).__name__}）"
        )
    return tuple(episodes)


__all__ = [
    "AblationEvalError",
    "build_variant_result",
    "closure_range_from_dataset",
    "evaluation_episodes",
    "make_oracle_provider",
    "sim_oracle_inputs",
    "success_rate",
]
