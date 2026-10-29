"""把 B0–B3 的额外列**追加进数据集**（第 38 周 P0 / 1.6 的真跑前置）。

`depth_ablation.run_ablation` 要求每个变体有自己的数据集（ENV 列数不同），本模块负责生成它们：
把基础 v3 数据集的 `observation.environment_state`（`fixed_size_list<float>[13]`）扩展成 `[13+n]`，
额外列取自 `ablation_features`（**方案③**：B1/B2 走 oracle 特权量）。

## 为什么能离线做（不需要仿真）

v3 数据集**已经存了** oracle 需要的全部量：

- `observation.state` = `tcp_pos(3) + gripper_qpos(2)` → `tcp_pos`、夹爪开合度；
- `observation.environment_state` = `obj_pos(3) / goal_pos(3) / (tcp−obj)(3) / (obj−goal)(3) / is_grasped(1)`
  （见 `lerobot_act.ENV_STATE_INDICES`）→ `obj_pos`、`is_grasped`。

所以 B1/B2 的额外列是**对既有列做纯算术**，本机可算、可测。

## B3 为什么这里拒绝

B3 的立体深度需要**两视角像素 + 内参/相对位姿**，而本数据集只有一个相机通道
（`observation.images.top`）—— 必须重新渲染第二视角。不重做就等于**用物体真值位姿反推三角化**
（循环论证，见 `ablation_features` 的说明），所以这里直接报错而不是糊一个假列。

## 三个必须小心的落盘细节

1. **data 下的 parquet 绝不能硬链接**：硬链接共享 inode，`pq.write_table` 会截断写，
   **直接把源数据集写坏**。所以复制时跳过 `data/*.parquet`，改为从读到的表**写到新文件**。
2. **三处元数据要同步**：`meta/info.json` 的 `features[...].shape`、`meta/stats.json` 的
   每维统计、以及 parquet 内嵌的 HF 元数据 `{"feature":…, "length": 13}` 的 `length`。
   漏掉任何一处都会让"列数一致"变成假象。
3. **meta 下的文件不能硬链接**：`write_text` 是 `O_TRUNC` 写同一个 inode，会**穿透链接把源数据集
   的元数据也改掉**，多变体还会共用同一份元数据互相覆盖。故硬链接**只用于 `videos/`**
   （见 `copy_dataset_tree`）。这一点是实跑 week15-cam 时踩出来的：三个变体的 info.json 全变成了 17。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ablation_features import OracleInputs, variant_features
from real_geometry import VARIANT_ORDER, GeometryError, variant_feature_names

STATE_KEY = "observation.state"
ENV_KEY = "observation.environment_state"
#: 基础 ENV 维度（见 `lerobot_act.ENV_STATE_DIM`）。
BASE_ENV_DIM = 13
#: `observation.state` 里夹爪两个关节的下标（`tcp_pos(3) + gripper_qpos(2)`）。
GRIPPER_STATE_SLICE = (3, 5)
#: `is_grasped` 在 ENV 向量里的下标（见 `lerobot_act.ENV_STATE_INDICES`）。
IS_GRASPED_INDEX = 12
#: 判定"夹住"的阈值：仿真里 `is_grasped` 是 0/1，取 0.5 中点。
GRASPED_THRESHOLD = 0.5


class AblationDatasetError(RuntimeError):
    """数据集准备契约错误。"""


@dataclass(frozen=True)
class PrepareReport:
    """一次变体数据集准备的产物摘要。"""

    variant: str
    out_root: Path
    env_dim: int
    frames: int
    extra_features: tuple[str, ...]
    open_width: float
    closed_width: float
    copied_files: int
    video_hardlinked: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "out_root": str(self.out_root),
            "env_dim": self.env_dim,
            "frames": self.frames,
            "extra_features": list(self.extra_features),
            "open_width": self.open_width,
            "closed_width": self.closed_width,
            "copied_files": self.copied_files,
            "video_hardlinked": self.video_hardlinked,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# 纯函数：从既有列算 oracle 输入、拼额外列、改三处元数据
# --------------------------------------------------------------------------- #


def frame_oracle_inputs(
    state_row: Sequence[float],
    env_row: Sequence[float],
    *,
    open_width: float,
    closed_width: float,
) -> OracleInputs:
    """把一帧的 `observation.state` / `environment_state` 还原成 oracle 输入。"""
    state = np.asarray(state_row, dtype=np.float64).reshape(-1)
    env = np.asarray(env_row, dtype=np.float64).reshape(-1)
    if state.shape[0] < GRIPPER_STATE_SLICE[1]:
        raise AblationDatasetError(
            f"{STATE_KEY} 至少需要 {GRIPPER_STATE_SLICE[1]} 维（tcp_pos + 夹爪两关节），"
            f"实际 {state.shape[0]}"
        )
    if env.shape[0] < BASE_ENV_DIM:
        raise AblationDatasetError(
            f"{ENV_KEY} 至少需要 {BASE_ENV_DIM} 维（含 is_grasped），实际 {env.shape[0]}"
        )
    start, stop = GRIPPER_STATE_SLICE
    return OracleInputs(
        tcp_pos_m=state[:3].tolist(),
        obj_pos_m=env[:3].tolist(),
        is_grasped=bool(env[IS_GRASPED_INDEX] > GRASPED_THRESHOLD),
        gripper_width=float(np.sum(state[start:stop])),
        open_width=float(open_width),
        closed_width=float(closed_width),
    )


def closure_range(states: np.ndarray) -> tuple[float, float]:
    """从数据本身取夹爪开合的**实际范围**（max=张开、min=闭合）。

    不用硬编码常数：不同机器人/不同标定的量纲不同，写死会把"闭合度"算成一个
    与物理无关的数，而它在消融里是要被模型当特征用的。
    """
    array = np.asarray(states, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < GRIPPER_STATE_SLICE[1]:
        raise AblationDatasetError(f"states 必须是 [T,≥{GRIPPER_STATE_SLICE[1]}] 的二维数组")
    start, stop = GRIPPER_STATE_SLICE
    widths = array[:, start:stop].sum(axis=1)
    open_width = float(np.max(widths))
    closed_width = float(np.min(widths))
    if not open_width > closed_width:
        raise AblationDatasetError(
            f"夹爪开合范围退化（max={open_width} ≤ min={closed_width}）：无法定义闭合度"
        )
    return open_width, closed_width


def variant_block(
    variant: str,
    states: np.ndarray,
    envs: np.ndarray,
    *,
    open_width: float,
    closed_width: float,
) -> np.ndarray:
    """算出该变体要追加的 `[T, n]` 额外列（B0 → `[T,0]`）。"""
    if variant not in VARIANT_ORDER:
        raise AblationDatasetError(f"未知变体 {variant!r}，应为 {VARIANT_ORDER} 之一")
    names = variant_feature_names(variant)
    state_array = np.asarray(states, dtype=np.float64)
    env_array = np.asarray(envs, dtype=np.float64)
    if state_array.shape[0] != env_array.shape[0]:
        raise AblationDatasetError(
            f"state 帧数 {state_array.shape[0]} 与 env 帧数 {env_array.shape[0]} 不一致"
        )
    if not names:
        return np.zeros((state_array.shape[0], 0), dtype=np.float32)
    try:
        rows = [
            variant_features(
                variant,
                oracle=frame_oracle_inputs(
                    state_array[i],
                    env_array[i],
                    open_width=open_width,
                    closed_width=closed_width,
                ),
            )
            for i in range(state_array.shape[0])
        ]
    except GeometryError as exc:
        # 统一成本模块的错误类型，调用方只需捕一种（消息保留原样，B3 的理由不会被吞）
        raise AblationDatasetError(str(exc)) from exc
    return np.asarray(rows, dtype=np.float32)


def patch_info_shape(info: Mapping[str, Any], env_dim: int) -> dict[str, Any]:
    """把 `meta/info.json` 里 ENV 特征的 shape 改成新维度。"""
    patched = json.loads(json.dumps(info))  # 深拷贝，不动调用方对象
    try:
        feature = patched["features"][ENV_KEY]
    except (KeyError, TypeError) as exc:
        raise AblationDatasetError(f"info.json 缺少 {ENV_KEY} 的 features 定义：{exc}") from exc
    if feature.get("dtype") == "video":  # pragma: no cover - 明显用错键
        raise AblationDatasetError(f"{ENV_KEY} 不应是视频特征")
    feature["shape"] = [int(env_dim)]
    return patched


def patch_hf_metadata(metadata: Mapping[str, Any], env_dim: int) -> dict[str, Any]:
    """改 parquet 内嵌 HF 元数据里该特征的 `length`（漏掉会让"列数一致"变成假象）。"""
    patched = json.loads(json.dumps(metadata))
    try:
        entry = patched["info"]["features"][ENV_KEY]
    except (KeyError, TypeError):
        return patched  # 没有该键就原样返回，由调用方决定是否告警
    entry["length"] = int(env_dim)
    return patched


def extend_stats(
    stats_entry: Mapping[str, Any],
    block: np.ndarray,
    *,
    base_dim: int = BASE_ENV_DIM,
) -> dict[str, Any]:
    """把额外列的统计追加到 `meta/stats.json` 的该特征条目上。

    逐维的 `min/max/mean/std` 直接追加；`count` 是标量（整表帧数）故保持不变。
    与既有维度数不符时**保留原值并记 warning**，而不是猜一个形状塞进去。
    """
    result = dict(stats_entry)
    if block.shape[1] == 0:
        return result
    for key in ("min", "max", "mean", "std"):
        if key not in result:
            continue
        existing = result[key]
        if not isinstance(existing, list) or len(existing) != base_dim:
            continue  # 形状不符 → 不动它（调用方按 warning 处理）
        if key == "min":
            extra = np.min(block, axis=0)
        elif key == "max":
            extra = np.max(block, axis=0)
        else:
            extra = (np.mean(block, axis=0) if key == "mean" else np.std(block, axis=0))
        result[key] = [float(v) for v in existing] + [float(v) for v in extra]
    return result


def stats_extended_ok(entry: Mapping[str, Any], env_dim: int) -> bool:
    """判断 stats 是否已成功扩展到目标维度（供调用方决定是否告警）。"""
    for key in ("min", "max", "mean", "std"):
        value = entry.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) != env_dim:
            return False
    return True


# --------------------------------------------------------------------------- #
# 落盘：复制树 + 重写 parquet + 三处元数据
# --------------------------------------------------------------------------- #


def _link_or_copy(source: Path, target: Path) -> bool:
    """优先硬链接（同卷、零拷贝），失败退回复制。返回是否用了硬链接。"""
    try:
        os.link(source, target)
        return True
    except OSError:
        shutil.copy2(source, target)
        return False


def copy_dataset_tree(
    base_root: Path,
    out_root: Path,
    *,
    hardlink: bool = True,
    hardlink_prefix: str = "videos",
) -> tuple[int, bool]:
    """把基础数据集整树复制到 `out_root`（返回 (复制文件数, 视频是否全部走硬链接)）。

    两条**安全**规则，都是硬链接的 inode 语义逼出来的：

    1. **只有 `videos/` 下的文件硬链接**（唯一体积大的部分，零拷贝才有意义）。其余一律复制。
       硬链接共享 inode，任何"写回"都会**穿透链接改到源数据集**——`Path.write_text` 就是
       `O_TRUNC` 写同一个 inode，于是"补一下新数据集的 info.json"会顺手把源数据集的
       info.json 也改掉，而且多个变体还会共用同一份元数据互相覆盖（实测踩到）。
    2. **`data/` 下的 parquet 直接跳过**：它们由调用方重写（`pq.write_table` 会截断写同一 inode）。
    """
    copied = 0
    all_linked = True
    for source in sorted(base_root.rglob("*")):
        if source.is_dir():
            continue
        relative = source.relative_to(base_root)
        target = out_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.parts and relative.parts[0] == "data" and source.suffix == ".parquet":
            continue  # 由调用方重写
        if hardlink and relative.parts and relative.parts[0] == hardlink_prefix:
            all_linked &= _link_or_copy(source, target)
        else:
            shutil.copy2(source, target)
        copied += 1
    return copied, all_linked


def _require_base_layout(base_root: Path) -> tuple[dict[str, Any], list[Path]]:
    info_path = base_root / "meta" / "info.json"
    if not info_path.is_file():
        raise AblationDatasetError(f"{base_root} 不是 LeRobot 数据集（缺 meta/info.json）")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    parquets = sorted((base_root / "data").rglob("*.parquet"))
    if not parquets:
        raise AblationDatasetError(f"{base_root}/data 下没有 parquet 数据文件")
    return info, parquets


def prepare_variant_dataset(
    base_root: str | Path,
    out_root: str | Path,
    variant: str,
    *,
    hardlink: bool = True,
    dry_run: bool = False,
) -> PrepareReport:
    """生成某变体的数据集（B0 只复制；B1/B2 追加列；B3 拒绝，理由见模块 docstring）。"""
    import pyarrow as pa  # 重型可选依赖：惰性导入，纯函数部分仍可离线测
    import pyarrow.parquet as pq

    base = Path(base_root).expanduser()
    out = Path(out_root).expanduser()
    if variant not in VARIANT_ORDER:
        raise AblationDatasetError(f"未知变体 {variant!r}，应为 {VARIANT_ORDER} 之一")
    if variant == "B3":
        raise AblationDatasetError(
            "B3 的立体深度需要两视角像素 + 内参/相对位姿，而本数据集只有单个相机通道"
            "（观察值里没有第二个视角）→ 必须重新渲染第二视角；用物体真值位姿反推三角化属"
            "循环论证，故这里拒绝而不是补一列假数据（见 ablation_features 的说明）"
        )
    if out.exists():
        raise AblationDatasetError(f"输出目录已存在：{out}（不覆盖，避免把已产出的变体数据集写坏）")
    info, parquets = _require_base_layout(base)

    names = variant_feature_names(variant)
    env_dim = BASE_ENV_DIM + len(names)
    warnings: list[str] = []

    # 第一遍：读全部帧，算数据驱动的夹爪范围与额外列
    tables = [pq.read_table(path) for path in parquets]
    states = np.concatenate(
        [np.asarray(table.column(STATE_KEY).to_pylist(), dtype=np.float64) for table in tables],
        axis=0,
    )
    envs = np.concatenate(
        [np.asarray(table.column(ENV_KEY).to_pylist(), dtype=np.float64) for table in tables],
        axis=0,
    )
    if envs.shape[1] != BASE_ENV_DIM:
        raise AblationDatasetError(
            f"{ENV_KEY} 维度为 {envs.shape[1]}，期望基础 {BASE_ENV_DIM}（本工具只从基础数据集出发）"
        )
    open_width, closed_width = closure_range(states)
    block = variant_block(
        variant, states, envs, open_width=open_width, closed_width=closed_width
    )
    frames = int(block.shape[0])
    if dry_run:
        return PrepareReport(
            variant=variant,
            out_root=out,
            env_dim=env_dim,
            frames=frames,
            extra_features=names,
            open_width=open_width,
            closed_width=closed_width,
            copied_files=0,
            video_hardlinked=False,
            warnings=("dry-run：未落盘",),
        )

    # 落盘
    copied, all_linked = copy_dataset_tree(base, out, hardlink=hardlink)
    offset = 0
    for path, table in zip(parquets, tables):
        rows = table.num_rows
        piece = block[offset : offset + rows]
        offset += rows
        merged = np.concatenate(
            [np.asarray(table.column(ENV_KEY).to_pylist(), dtype=np.float32), piece], axis=1
        )
        new_column = pa.array(merged.tolist(), type=pa.list_(pa.float32(), env_dim))
        index = table.schema.get_field_index(ENV_KEY)
        if index < 0:
            raise AblationDatasetError(f"{path} 缺少列 {ENV_KEY}")
        table = table.set_column(index, ENV_KEY, new_column)
        metadata = dict(table.schema.metadata or {})
        raw = metadata.get(b"huggingface")
        if raw is None:
            warnings.append(f"{path.name} 没有内嵌 HF 元数据，未改 length（info.json 已改）")
        else:
            try:
                patched = patch_hf_metadata(json.loads(raw.decode("utf-8")), env_dim)
                metadata[b"huggingface"] = json.dumps(patched).encode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                warnings.append(f"{path.name} 的内嵌 HF 元数据无法解析（{exc}），保留原值")
        table = table.replace_schema_metadata(metadata)
        target = out / path.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(target), compression="snappy")

    (out / "meta" / "info.json").write_text(
        json.dumps(patch_info_shape(info, env_dim), ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    stats_path = out / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        entry = stats.get(ENV_KEY)
        if isinstance(entry, Mapping):
            extended = extend_stats(entry, block)
            stats[ENV_KEY] = extended
            if not stats_extended_ok(extended, env_dim):
                warnings.append(
                    f"meta/stats.json 的 {ENV_KEY} 未扩展到 {env_dim} 维"
                    "（形状与预期不符，保留原值）——本仓库的训练路径自算统计，故不影响训练，"
                    "但请知悉该文件未同步"
                )
        else:
            warnings.append(f"meta/stats.json 缺少 {ENV_KEY} 条目，未改")
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    else:
        warnings.append("缺少 meta/stats.json，未改（请确认该数据集确实没有）")

    video_linked = all_linked and any((out / "videos").rglob("*.mp4"))
    return PrepareReport(
        variant=variant,
        out_root=out,
        env_dim=env_dim,
        frames=frames,
        extra_features=names,
        open_width=open_width,
        closed_width=closed_width,
        copied_files=copied,
        video_hardlinked=bool(video_linked),
        warnings=tuple(warnings),
    )
