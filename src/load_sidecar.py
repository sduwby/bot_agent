"""采集时**旁挂**记录负载通道（STS3215 `Present_Load`）—— 不改 LeRobotDataset fields。

## 为什么需要（第 38 周 P1/2.10 的实测取证）

`demo_v23` 只录了关节位置，**没有负载通道**。拿 14 集人工真值一量化，问题就很硬：

    人工=夹住  n=6  末 5 帧夹爪位置 8.09 – 9.97
    人工=没夹住 n=8  末 5 帧夹爪位置 6.90 – 9.97        ← 两组完全重叠

即位置判据对"是否真的夹住了物体"**零可分性**（错误率 8/14 = 57%），κ_success = 0 不是口径问题
而是信号问题。而 0.6 的负载实测是可分辨的（空载闭合 55–69‰ vs 夹持 206‰）——
**所以要在采集阶段就把负载录下来**，否则 2.9 的多任务集会原样复现这个坑。

## 格式

JSONL，每个**被存进数据集**的帧一行：

```json
{"dataset_repo": "wangbangyan/demo_vla",
 "episode_index": 0, "frame_index": 0, "t": 12.34,
 "load": {"gripper.load": -206.0},
 "load_raw": {"gripper.load": 1234.0},
 "positions": {"gripper.pos": 6.8, "shoulder_pan.pos": 1.2}}
```

- `load` / `load_raw` 为解码后（千分比，与 `real_observation` 同口径）/ 原始 2 字节读数；
- `positions` 与数据集 `observation.state` **同源**，用于事后校验对齐（见 `verify_alignment`）。

按 `(episode_index, frame_index)` 与数据集帧一一对应，故**不需要改 LeRobotDataset 的 fields**，
既有 ACT/SmolVLA 训练脚本不受影响。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from real_observation import decode_feetech_load, load_state_names

SIDECAR_SUFFIX = ".jsonl"


class SidecarError(RuntimeError):
    """负载旁挂文件的契约错误。"""


def _load_keys() -> tuple[str, ...]:
    return tuple(load_state_names())


def decode_loads(raw: Mapping[str, float]) -> tuple[dict[str, float], list[str]]:
    """把 `Present_Load` 原始读数解码为千分比，返回 ``(decoded, 解码失败的键)``。

    单个关节读数异常（超出 2 字节范围）不应中断整场采集，故**逐关节容错**并记录失败键。
    """
    decoded: dict[str, float] = {}
    failed: list[str] = []
    for key, value in raw.items():
        name = str(key)
        if not name.endswith(".load"):
            name = f"{name}.load"
        try:
            decoded[name] = decode_feetech_load(float(value))
        except (TypeError, ValueError):
            failed.append(name)
    return decoded, failed


class LoadSidecarWriter:
    """按集缓冲、**成集落盘**的负载旁挂写入器。

    为什么按集缓冲而不是逐帧落盘：录制里"坏集用 ← 重录"是常规操作，逐帧写会往文件里
    留下被丢弃那一集的垃圾行。缓冲后：同一集号再次开始 = 重录 → **丢弃旧缓冲**；
    集号推进 = 上一集确认保留 → 落盘。异常中断时未提交的那一集不会写入。
    """

    def __init__(self, path: str | Path, *, dataset_repo: str = "") -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_repo = dataset_repo
        self._handle = self.path.open("a", encoding="utf-8")
        self._pending_index: int | None = None
        self._pending: list[dict[str, Any]] = []
        self.committed_episodes = 0

    # -- 收集 -------------------------------------------------------------- #

    def begin_episode(self, episode_index: int) -> None:
        """开始记录某一集；集号与待提交缓冲相同表示**重录** → 丢弃旧缓冲。"""
        if self._pending_index is None:
            self._pending_index = int(episode_index)
            return
        if self._pending_index == int(episode_index):
            self._pending.clear()          # 重录：丢掉上一遍
            return
        self.commit_episode()
        self._pending_index = int(episode_index)

    def abort_episode(self) -> None:
        """放弃当前缓冲（不落盘）。"""
        self._pending_index = None
        self._pending.clear()

    def add_frame(
        self,
        episode_index: int,
        frame_index: int,
        load_raw: Mapping[str, float],
        positions: Mapping[str, float] | None = None,
    ) -> None:
        if self._pending_index is None or int(episode_index) != self._pending_index:
            raise SidecarError(
                f"add_frame 的集号 {episode_index} 与当前缓冲 {self._pending_index} 不一致"
                "（必须先调用 begin_episode）"
            )
        decoded, failed = decode_loads(load_raw)
        row: dict[str, Any] = {
            "dataset_repo": self.dataset_repo,
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "t": round(time.monotonic(), 4),
            "load": decoded,
            "load_raw": {str(k): float(v) for k, v in load_raw.items()},
        }
        if positions:
            row["positions"] = {str(k): float(v) for k, v in positions.items()}
        if failed:
            row["decode_errors"] = failed
        self._pending.append(row)

    def commit_episode(self) -> int:
        """把当前缓冲追加落盘，返回写入行数。"""
        count = len(self._pending)
        for row in self._pending:
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if count:
            self._handle.flush()
            self.committed_episodes += 1
        self._pending_index = None
        self._pending.clear()
        return count

    def close(self) -> None:
        """收尾：提交未落盘的缓冲并关闭文件。"""
        if self._pending_index is not None:
            self.commit_episode()
        self._handle.close()

    def __enter__(self) -> "LoadSidecarWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_load_sidecar(path: str | Path) -> list[dict[str, Any]]:
    """读取旁挂文件（不存在则返回空表）。"""
    target = Path(path).expanduser()
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def episode_loads(
    records: Iterable[Mapping[str, Any]], *, joint: str = "gripper.load"
) -> dict[int, list[float | None]]:
    """按集整理某关节的负载序列（按 `frame_index` 排序），供成功判定使用。"""
    buckets: dict[int, list[tuple[int, float | None]]] = {}
    for row in records:
        load = row.get("load") or {}
        value = load.get(joint)
        buckets.setdefault(int(row["episode_index"]), []).append(
            (int(row["frame_index"]), None if value is None else float(value))
        )
    return {
        index: [value for _, value in sorted(items)] for index, items in buckets.items()
    }


@dataclass(frozen=True)
class SidecarAlignment:
    """单集的对齐检查结果。"""

    episode_index: int
    dataset_frames: int
    sidecar_frames: int
    max_abs_diff: float
    tol: float = 1e-3

    @property
    def ok(self) -> bool:
        """帧数一致**且**位置逐帧吻合（数据集按 float32 存储，故留 1e-3 容差）。"""
        return self.dataset_frames == self.sidecar_frames and self.max_abs_diff <= self.tol

    def line(self) -> str:
        flag = "OK " if self.ok else "错位"
        return (
            f"[{flag}] episode_{self.episode_index:06d} "
            f"数据集 {self.dataset_frames} 帧 / 旁挂 {self.sidecar_frames} 帧 / "
            f"位置最大偏差 {self.max_abs_diff:.2e}"
        )


@dataclass(frozen=True)
class SidecarAlignmentReport:
    """旁挂文件与数据集的对齐报告。"""

    episodes: tuple[SidecarAlignment, ...]
    tol: float

    @property
    def ok(self) -> bool:
        return bool(self.episodes) and all(item.ok for item in self.episodes)

    def summary_lines(self) -> list[str]:
        flagged = [item for item in self.episodes if not item.ok]
        return [
            f"集数 {len(self.episodes)}，错位 {len(flagged)} 集（判定：{'通过' if self.ok else '不通过'}）",
            *[item.line() for item in self.episodes],
        ]


def verify_alignment(
    records: Sequence[Mapping[str, Any]],
    positions_by_episode: Mapping[int, Sequence[float]],
    *,
    joint: str = "gripper.pos",
    tol: float = 1e-3,
) -> SidecarAlignmentReport:
    """用旁挂文件里的 `positions` 复核"与数据集逐帧对齐"。

    对齐正确时同集同帧的 `positions[joint]` 必须与数据集 `observation.state` 中该关节
    完全一致（同一路 `sync_read` 的结果）。因此这个检查能把"帧数对不上/顺序错位/
    重录残留"一次性抓出来——**采集完必须跑一次**，否则旁挂数据不可信。
    """
    sidecar: dict[int, list[float]] = {}
    for row in records:
        positions = row.get("positions") or {}
        if joint not in positions:
            continue
        sidecar.setdefault(int(row["episode_index"]), []).append(float(positions[joint]))

    results: list[SidecarAlignment] = []
    for index in sorted(set(sidecar) | set(positions_by_episode)):
        expected = [float(v) for v in positions_by_episode.get(index, ())]
        actual = sidecar.get(index, [])
        overlap = min(len(expected), len(actual))
        max_diff = max(
            (abs(expected[i] - actual[i]) for i in range(overlap)), default=0.0
        )
        results.append(
            SidecarAlignment(
                episode_index=index,
                dataset_frames=len(expected),
                sidecar_frames=len(actual),
                max_abs_diff=max_diff,
            )
        )
    return SidecarAlignmentReport(episodes=tuple(results), tol=tol)
