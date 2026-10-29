"""第 38 周 P2：VLA 适配策略与对照实验（首轮 zero_shot / head / full）。

研究问题：**同一份预训练 SmolVLA，只改"解冻哪些参数"与"用多少数据微调"，
成功率与"原任务保留率（遗忘）"如何变化？** 适配在这里是自变量，不是流程步骤
（design §3.2 决策 5）。

本模块只负责"配置 + 参数解冻 seam + 实验编排"，不复制 lerobot 的训练实现。
三个策略差异全部落在 `requires_grad` 上：

- ``zero_shot``：一个参数都不训练（锚点基线）；
- ``head``：只训练 action head（按参数名匹配，见 `DEFAULT_HEAD_PATTERNS`）；
- ``full``：全部参数训练。

**为什么 LoRA 延后**：lerobot 0.3.4 没有现成的 LoRA / 按模块冻结入口
（design §3.3 G / plan 4.3），首轮裁掉，构造期显式报错而不是静默降级。

**训练 seam 的现实约束**：`lerobot.scripts.lerobot_train`（0.6.x 入口）的 CLI 不接受
"按模块冻结"参数，故 `apply_adaptation` 只有在**进程内训练循环**（拿到 policy 对象）
时才能真正生效；默认训练路径（走 CLI）只保证链路可跑，不代表冻结已落地——这条边界
写进 `_default_train_fn` 的 docstring，避免把"跑通"误读为"对照有效"。

导入安全：本模块**不在导入期导入 torch / lerobot**，`apply_adaptation` 只依赖鸭子
类型（`policy.named_parameters()` + `param.numel()` + `param.requires_grad`），
因此可在无 GPU 的纯逻辑测试里跑通。
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ADAPT_SCHEMA_VERSION = "week38_vla_adapt_v1"

#: 首轮支持的策略（design §7.3）。顺序即 CLI 矩阵的规范顺序。
SUPPORTED_STRATEGIES: tuple[str, ...] = ("zero_shot", "head", "full")
#: 已知但**刻意延后**的策略：构造期显式报错，不静默降级（design §3.3 G）。
DEFERRED_STRATEGIES: tuple[str, ...] = ("lora",)

#: head 策略默认解冻的参数名前缀。lerobot 的 action head 命名较分散，
#: 故给一组"子串"而非单一键：命中其一即解冻（`action_head` 已覆盖 `action_head_*`）。
#: 2026-10-18 实测 SmolVLA 的 head 由 `action_{in,out}_proj` + `action_time_mlp_{in,out}`
#: + `state_proj` 组成（合计约 1.6M / 450M ≈ 0.36%）；漏掉后两者会让 head 只剩 47k 参数。
DEFAULT_HEAD_PATTERNS: tuple[str, ...] = (
    "action_head",
    "action_out_proj",
    "action_in_proj",
    "action_model",
    "action_head_",
    "action_time_mlp",
    "state_proj",
)

#: 各策略的**名义** `trainable_ratio`：zero_shot 全冻结=0、full 全解冻=1；
#: head 在构造期无法知道命中多少权重 → None（由 `apply_adaptation` 实测填写）。
_NOMINAL_TRAINABLE_RATIO: dict[str, float | None] = {
    "zero_shot": 0.0,
    "head": None,
    "full": 1.0,
}


class AdaptError(RuntimeError):
    """适配配置、参数解冻或实验编排的契约错误。"""


def _require_int(value: object, name: str, *, minimum: int) -> int:
    """校验整数下界（排除 bool —— Python 里 bool 是 int 的子类）。"""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise AdaptError(f"{name} 必须是 >= {minimum} 的整数，实际 {value!r}")
    return int(value)


@dataclass(frozen=True)
class AdaptConfig:
    """一次适配实验的自变量配置。

    字段语义：

    - ``strategy``：``"zero_shot"`` / ``"head"`` / ``"full"``（``"lora"`` 已延后）。
    - ``trainable_ratio``：**可训练参数占比**。``None`` 表示"构造期未知、由
      `apply_adaptation` 实测填写"（head 命中多少权重无法先验确定）；名义值为
      ``zero_shot=0.0``、``full=1.0``。该字段是"结果"而非"输入"，但随配置落盘以便审计。
    - ``n_episodes``：微调数据量（自变量，首轮 30 / 60）。**必须 > 0**——zero_shot
      作为锚点不训练，其 ``n_episodes`` 仅作占位（见 `run_vla_adapt.build_adapt_jobs`）。
    - ``steps``：训练步数。zero_shot 保留调用方取值，但 `run_adapt_experiment`
      会**完全跳过训练**（见其 docstring）。
    - ``seed``：固定随机种子。
    """

    strategy: str
    trainable_ratio: float | None
    n_episodes: int
    steps: int
    seed: int

    def __post_init__(self) -> None:
        strategy = str(self.strategy).strip().lower()
        if strategy in DEFERRED_STRATEGIES:
            raise AdaptError(
                f"策略 {self.strategy!r} 为**延后项**（design §3.3 G / plan 4.3）："
                "lerobot 0.3.4 无 LoRA / 按模块冻结入口，首轮只支持 "
                f"{SUPPORTED_STRATEGIES}。"
            )
        if strategy not in SUPPORTED_STRATEGIES:
            raise AdaptError(
                f"不支持的适配策略 {self.strategy!r}；支持 {SUPPORTED_STRATEGIES}"
                f"（{DEFERRED_STRATEGIES} 延后）。"
            )
        # 规整大小写，使 "HEAD" 与 "head" 等价（frozen 需绕过赋值保护）。
        object.__setattr__(self, "strategy", strategy)

        _require_int(self.n_episodes, "n_episodes", minimum=1)
        _require_int(self.steps, "steps", minimum=0)
        _require_int(self.seed, "seed", minimum=0)

        ratio = self.trainable_ratio
        if ratio is not None:
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or not 0.0 <= float(ratio) <= 1.0
            ):
                raise AdaptError(
                    "trainable_ratio 必须为 None（head 未知）或位于 [0, 1] 的数值，"
                    f"实际 {ratio!r}"
                )


@dataclass(frozen=True)
class AdaptResult:
    """一次适配实验的因变量结果（design §4）。

    ``peak_mem_mb``：训练侧未上报峰值显存时记 ``None``（**不是 0**）——
    0 会被误读为"显存占用为零"，故用 None 明确表达"不可得"。
    """

    config: AdaptConfig
    success_rate: float
    retention_rate: float | None
    train_seconds: float
    peak_mem_mb: float | None

    def to_dict(self) -> dict[str, object]:
        """展开为可 JSON 序列化的 dict（供 adapt_result.json 落盘）。"""
        return {
            "config": asdict(self.config),
            "success_rate": float(self.success_rate),
            "retention_rate": (
                None if self.retention_rate is None else float(self.retention_rate)
            ),
            "train_seconds": float(self.train_seconds),
            "peak_mem_mb": None if self.peak_mem_mb is None else float(self.peak_mem_mb),
        }


def build_adapt_config(
    strategy: str, n_episodes: int, steps: int, seed: int
) -> AdaptConfig:
    """构造 `AdaptConfig`，并填入该策略的**名义** `trainable_ratio`。

    纯函数：不做任何 IO，便于 CLI 与测试复用。注意 zero_shot **不强制** ``steps=0``
    （保留调用方取值以保持配置可比），它"不训练"这件事由 `run_adapt_experiment` 落实。
    """
    nominal = _NOMINAL_TRAINABLE_RATIO.get(str(strategy).strip().lower())
    return AdaptConfig(
        strategy=strategy,
        trainable_ratio=nominal,
        n_episodes=n_episodes,
        steps=steps,
        seed=seed,
    )


def _normalize_pattern(pattern: str) -> str:
    """去掉模式尾部通配符，得到纯子串（``action_head_*`` → ``action_head_``）。"""
    return str(pattern).strip().rstrip("*")


def _matches_head(name: str, needles: Sequence[str]) -> bool:
    return any(needle and needle in name for needle in needles)


def _module_prefix(name: str) -> str:
    """参数名 → 所属模块前缀（``a.b.weight`` → ``a.b``；无点则原样返回）。"""
    return name.rsplit(".", 1)[0] if "." in name else name


def apply_adaptation(
    policy: Any,
    config: AdaptConfig,
    *,
    head_patterns: Sequence[str] | None = None,
) -> dict[str, object]:
    """就地改写 ``policy`` 参数的 ``requires_grad``，返回解冻统计。

    Args:
        policy: 任何提供 ``named_parameters()`` 的策略对象（torch 模块或鸭子类型替身）。
        config: 决定解冻范围的配置。
        head_patterns: head 策略的匹配模式；缺省用 `DEFAULT_HEAD_PATTERNS`。

    Returns:
        dict: ``strategy`` / ``total_params`` / ``trainable_params`` /
        ``trainable_ratio``（实测）/ ``trainable_modules``（保持可训练的模块前缀元组）
        / ``frozen_params``。

    Raises:
        AdaptError: policy 无 ``named_parameters()``，或 head 策略**匹配到 0 个参数**。

    安全性质：head 命中为空时**必须报错**，绝不静默退化为"全训练"或"全冻结"——
    前者会悄悄变成 full、后者会悄悄变成 zero_shot，两者都会让对照表失去意义却看不出来。
    """
    named_fn = getattr(policy, "named_parameters", None)
    if not callable(named_fn):
        raise AdaptError("policy 缺少 named_parameters()，无法判定可训练参数")
    entries: list[tuple[str, Any]] = [(name, param) for name, param in named_fn()]

    if config.strategy == "zero_shot":
        trainable_names: set[str] = set()
    elif config.strategy == "full":
        trainable_names = {name for name, _ in entries}
    else:  # head
        patterns = tuple(head_patterns) if head_patterns is not None else DEFAULT_HEAD_PATTERNS
        needles = tuple(_normalize_pattern(item) for item in patterns)
        trainable_names = {
            name for name, _ in entries if _matches_head(name, needles)
        }
        if not trainable_names:
            raise AdaptError(
                f"head 策略未匹配到任何参数（patterns={patterns}）：请核对策略命名，"
                "或显式传入 head_patterns；拒绝静默退化为『全训练』或『全冻结』。"
            )

    total_params = 0
    trainable_params = 0
    trainable_modules: list[str] = []
    for name, param in entries:
        numel = int(param.numel())
        total_params += numel
        keep = name in trainable_names
        param.requires_grad = keep
        if keep:
            trainable_params += numel
            prefix = _module_prefix(name)
            if prefix not in trainable_modules:
                trainable_modules.append(prefix)

    ratio = trainable_params / total_params if total_params else 0.0
    return {
        "strategy": config.strategy,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": ratio,
        "trainable_modules": tuple(trainable_modules),
        "frozen_params": total_params - trainable_params,
    }


def _checked_seeds(seeds: Sequence[int], name: str) -> list[int]:
    """校验 seed 列表非空、非负、不重复（与 `lerobot_act._checked_seeds` 同口径）。

    本地实现而非复用私有函数，是为了让本模块保持"导入期不拉 torch/lerobot"的约束。
    """
    values = list(seeds)
    if not values:
        raise AdaptError(f"{name} 不能为空")
    checked: list[int] = []
    for seed in values:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise AdaptError(f"{name} 必须是非负整数，实际 {seed!r}")
        checked.append(seed)
    if len(set(checked)) != len(checked):
        raise AdaptError(f"{name} 不能重复：{values}")
    return checked


def _safe_ratio(after: float, before: float) -> float | None:
    """保留率 = after / before，带**除零保护**（plan §4.1 明确要求）。

    基线成功率为 0 时比率无定义（不是"无穷大保留率"），返回 ``None``；
    若强行返回 0.0 会被误读为"完全遗忘"，故选择 None 表达"不可计算"。
    """
    if before == 0.0:
        return None
    return float(after) / float(before)


def _evaluate_checkpoint(
    checkpoint_dir: Path | None, seeds: Sequence[int], device: str
) -> float:
    """默认评测实现：从 checkpoint 目录加载并跑真实 rollout，返回成功率。"""
    if checkpoint_dir is None:
        raise AdaptError(
            "默认评测实现需要 output_dir（checkpoint 目录）；"
            "真实对照实验请注入 eval_fn / baseline_eval_fn 以各自指定口径。"
        )
    from lerobot_act import evaluate_policy_seeds, load_lerobot_act_checkpoints

    policies = load_lerobot_act_checkpoints(checkpoint_dir, device=device)
    summary = evaluate_policy_seeds("adapted", policies, list(seeds))
    return float(summary.success_rate)


def _make_default_eval_fn(checkpoint_dir: Path | None) -> Callable[..., float]:
    """把 checkpoint 目录绑定进默认评测闭包（签名与注入的 eval_fn 一致）。"""

    def _evaluate(
        config: AdaptConfig,
        dataset_root: str,
        seeds: Sequence[int],
        device: str,
        task: str | None,
    ) -> float:
        return _evaluate_checkpoint(checkpoint_dir, seeds, device)

    return _evaluate


def _build_smolvla_train_command(
    config: AdaptConfig,
    dataset_root: str,
    output_dir: Path | None,
    device: str,
) -> list[str]:
    """复用 `run_vla_train.py` 的 SmolVLA 训练命令组装方式（lerobot 0.6.x 口径）。"""
    # 单点事实来源：入口名 / 基础权重 / 相机重命名都从 run_vla_train 取，避免两处漂移。
    from run_vla_train import DEFAULT_POLICY_PATH, DEFAULT_RENAME_MAP, TRAIN_MODULE

    repo = Path(dataset_root).name if dataset_root else "wangbangyan/demo_vla_all"
    target = output_dir or (Path("outputs") / f"vla_adapt_{config.strategy}")
    cmd = [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        f"--policy.path={DEFAULT_POLICY_PATH}",
        f"--dataset.repo_id={repo}",
        f"--output_dir={target}",
        f"--job_name=vla_adapt_{config.strategy}",
        f"--steps={config.steps}",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--rename_map={DEFAULT_RENAME_MAP}",
    ]
    if dataset_root:
        cmd.append(f"--dataset.root={dataset_root}")
    return cmd


def _default_train_fn(
    config: AdaptConfig,
    dataset_root: str,
    output_dir: Path | None,
    device: str,
) -> float:
    """默认训练实现：跑 SmolVLA 微调命令，返回训练耗时（秒）。

    **边界（务必读）**：`lerobot.scripts.lerobot_train` 的 CLI 没有"按模块冻结"入口
    （design §3.3 G），因此这里的 head/full 并不会自动应用 `apply_adaptation` 的
    解冻结果——它只是让链路可跑。真正的对照实验必须由训练侧拿到 policy 对象、
    调用 `apply_adaptation` 后再训练。此默认路径需要 GPU/联网，测试不覆盖。
    """
    if config.strategy == "zero_shot":
        return 0.0
    if config.steps <= 0:
        raise AdaptError("head/full 微调需要 steps > 0")
    import subprocess
    import time

    cmd = _build_smolvla_train_command(config, dataset_root, output_dir, device)
    started = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - started


def run_adapt_experiment(
    config: AdaptConfig,
    dataset_root: str,
    eval_seeds: Sequence[int],
    *,
    baseline_seeds: Sequence[int] | None = None,
    baseline_task: str | None = None,
    train_fn: Callable[..., object] | None = None,
    eval_fn: Callable[..., float] | None = None,
    baseline_eval_fn: Callable[..., float] | None = None,
    output_dir: str | Path | None = None,
    device: str = "cuda",
) -> AdaptResult:
    """编排一次适配实验：基线评测 → (可选)训练 → 评测 → 保留率。

    流程（design §6.3）：

    (a) 训练前用 ``baseline_eval_fn`` 评一次（保留率的参考点，即"微调前"）；
    (b) ``strategy != "zero_shot"`` 时调用 ``train_fn``（zero_shot 完全跳过）；
    (c) 用 ``eval_fn`` 评一次（"微调后"成功率）；
    (d) ``retention_rate = 微调后 / 微调前``，基线为 0 时记 ``None``（`_safe_ratio`）。

    注入契约（三者都是可调用对象，缺省时用懒加载的真实管线实现）：

    - ``train_fn(config, dataset_root, output_dir, device)`` → 训练耗时秒，
      或 ``{"train_seconds": ..., "peak_mem_mb": ...}``（可上报峰值显存）；
    - ``eval_fn(config, dataset_root, seeds, device, task)`` → 成功率；
    - ``baseline_eval_fn`` 与 ``eval_fn`` 同签名（``task`` 取 ``baseline_task``）。

    测试通过注入假实现运行，无需 GPU；默认实现需要 GPU/联网，仅保证导入安全。
    """
    checked_eval = _checked_seeds(eval_seeds, "eval_seeds")
    checked_baseline = (
        _checked_seeds(baseline_seeds, "baseline_seeds")
        if baseline_seeds is not None
        else checked_eval
    )
    out_path = Path(output_dir).expanduser() if output_dir is not None else None

    if eval_fn is None:
        eval_fn = _make_default_eval_fn(out_path)
    if baseline_eval_fn is None:
        baseline_eval_fn = _make_default_eval_fn(out_path)
    if train_fn is None:
        train_fn = _default_train_fn

    # (a) 微调前基线评测 —— 保留率的参考点
    before = float(
        baseline_eval_fn(config, dataset_root, checked_baseline, device, baseline_task)
    )

    # (b) 训练（zero_shot 是"不改权重"的锚点，完全跳过）
    train_seconds = 0.0
    peak_mem_mb: float | None = None
    if config.strategy != "zero_shot":
        outcome = train_fn(config, dataset_root, out_path, device)
        if isinstance(outcome, Mapping):
            train_seconds = float(outcome.get("train_seconds", 0.0))
            reported_mem = outcome.get("peak_mem_mb")
            peak_mem_mb = None if reported_mem is None else float(reported_mem)
        else:
            train_seconds = float(outcome)

    # (c) 微调后评测
    success_rate = float(eval_fn(config, dataset_root, checked_eval, device, None))

    # (d) 保留率（除零保护）
    retention_rate = _safe_ratio(success_rate, before)

    return AdaptResult(
        config=config,
        success_rate=success_rate,
        retention_rate=retention_rate,
        train_seconds=train_seconds,
        peak_mem_mb=peak_mem_mb,
    )
