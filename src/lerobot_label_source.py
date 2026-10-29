"""本地 LeRobot 数据集 → P1 标注输入源（第 38 周 P1，前置 2.9）。

`vlm_labeler.label_dataset` 只依赖 `EpisodeSource` 协议，本模块给出真数据集实现：

- 读 `meta/{info,episodes,tasks}.jsonl` 拿帧率、字段名、每集长度与任务文本；
- 读 `data/chunk-*/episode_*.parquet` 的 `observation.state` 取 `gripper.pos`
  （成功判定只用位置通道——真机演示数据**不含负载**，该局限必须写进 `label_scope`）；
- 用 **PyAV** 解 `videos/.../*.mp4` 抽帧（`demo_v23` 是 **AV1** 编码，OpenCV 解不动）；
- 抽出的帧**落盘缓存**：VLM 自动标注与人工复核（2.10）都要看图，缓存一次两处复用。

`pyarrow` / `av` 均为重型可选依赖，故**在函数内部惰性导入**，
这样 `sample_indices` / `read_dataset_meta` 这类纯函数可离线测试。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from vlm_labeler import EpisodeSample


class EpisodeSourceError(RuntimeError):
    """数据源读取契约错误。"""


#: `observation.state` 中夹爪位置的名字（`success_from_sensor` 需要它）。
GRIPPER_STATE_NAME = "gripper.pos"


def sample_indices(length: int, count: int) -> list[int]:
    """在 ``[0, length)`` 内等间隔取 ``count`` 个帧下标（含首尾，去重）。

    - ``count == 1`` 取**末帧**：演示的终态最能体现"是否夹住/抬起"，
      比取中间帧更利于子任务与成功判定；
    - ``length <= count`` 时返回全部帧，不重复。
    """
    if length <= 0:
        return []
    if count <= 0:
        raise ValueError("count 必须是正整数")
    if length <= count:
        return list(range(length))
    if count == 1:
        return [length - 1]
    step = (length - 1) / (count - 1)
    indices: list[int] = []
    for position in range(count):
        index = int(round(position * step))
        if not indices or index != indices[-1]:
            indices.append(index)
    return indices


@dataclass(frozen=True)
class EpisodeMeta:
    """单集元信息。"""

    episode_index: int
    length: int
    task: str

    @property
    def ref(self) -> str:
        return f"episode_{self.episode_index:06d}"


@dataclass(frozen=True)
class DatasetMeta:
    """数据集级元信息（只含标注链路用得到的部分）。"""

    root: Path
    fps: int
    state_names: tuple[str, ...]
    data_path: str
    video_path: str
    chunks_size: int
    episodes: tuple[EpisodeMeta, ...]

    def gripper_index(self) -> int:
        """`gripper.pos` 在 `observation.state` 中的下标。"""
        try:
            return list(self.state_names).index(GRIPPER_STATE_NAME)
        except ValueError as exc:
            raise EpisodeSourceError(
                f"observation.state 缺少 {GRIPPER_STATE_NAME}（实际 {self.state_names}）"
            ) from exc


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise EpisodeSourceError(f"缺少元数据文件: {path}")
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_dataset_meta(dataset_root: str | Path) -> DatasetMeta:
    """读取 `meta/` 下的数据集元信息（**不**触碰视频，纯 JSON）。"""
    root = Path(dataset_root).expanduser()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise EpisodeSourceError(f"{root} 不是 LeRobot 数据集（缺少 meta/info.json）")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    try:
        state_names = tuple(info["features"]["observation.state"]["names"] or ())
        episodes = tuple(
            EpisodeMeta(
                episode_index=int(row["episode_index"]),
                length=int(row["length"]),
                task="；".join(str(item) for item in row.get("tasks") or ()),
            )
            for row in _read_jsonl(root / "meta" / "episodes.jsonl")
        )
        return DatasetMeta(
            root=root,
            fps=int(info["fps"]),
            state_names=state_names,
            data_path=str(info["data_path"]),
            video_path=str(info["video_path"]),
            chunks_size=int(info.get("chunks_size", 1000)),
            episodes=episodes,
        )
    except KeyError as exc:
        raise EpisodeSourceError(f"meta 缺少字段: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise EpisodeSourceError(f"meta 字段类型不正确: {exc}") from exc


def read_gripper_positions(meta: DatasetMeta, episode_index: int) -> list[float]:
    """读单集的 `gripper.pos` 序列（parquet，向量化，不逐帧解码视频）。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - 环境缺失时的显式提示
        raise EpisodeSourceError("需要 pyarrow 才能读取数据集状态（pip install pyarrow）") from exc
    chunk = episode_index // max(meta.chunks_size, 1)
    path = meta.root / meta.data_path.format(episode_chunk=chunk, episode_index=episode_index)
    if not path.is_file():
        raise EpisodeSourceError(f"缺少状态文件: {path}")
    table = pq.read_table(str(path))
    if "observation.state" not in table.column_names:
        raise EpisodeSourceError(f"{path.name} 缺少 observation.state 列")
    index = meta.gripper_index()
    values: list[float] = []
    for row in table.column("observation.state").to_pylist():
        if row is None or index >= len(row):
            raise EpisodeSourceError(f"{path.name} 的 observation.state 维度异常")
        values.append(float(row[index]))
    return values


def decode_frames(
    video_path: str | Path,
    indices: list[int],
) -> dict[int, "object"]:
    """用 PyAV 解码指定帧下标，返回 ``{下标: RGB ndarray(H,W,3)}``。

    `demo_v23` 是 **AV1**（libdav1d）；OpenCV 的默认构建不含 AV1 解码器，
    必须走 PyAV。解码到最大目标下标即停，避免整段解完。
    """
    target = Path(video_path)
    if not target.is_file():
        raise EpisodeSourceError(f"缺少视频文件: {target}")
    if not indices:
        return {}
    try:
        import av
    except ImportError as exc:  # pragma: no cover
        raise EpisodeSourceError("需要 av (PyAV) 才能解码视频（pip install av）") from exc
    wanted = sorted(set(indices))
    last = wanted[-1]
    frames: dict[int, object] = {}
    try:
        with av.open(str(target)) as container:
            stream = container.streams.video[0]
            for position, frame in enumerate(container.decode(stream)):
                if position in wanted:
                    frames[position] = frame.to_ndarray(format="rgb24")
                if position >= last:
                    break
    except Exception as exc:  # 解码器/文件损坏统一升级为数据源错误
        raise EpisodeSourceError(f"解码 {target.name} 失败: {exc}") from exc
    missing = [index for index in wanted if index not in frames]
    if missing:
        raise EpisodeSourceError(f"{target.name} 只解出 {sorted(frames)}，缺 {missing}")
    return frames


class LerobotEpisodeSource:
    """LeRobot 数据集 → `EpisodeSample` 的**惰性**迭代源。

    帧文件写入 ``frames_dir/<episode_ref>/<相机简称>_<下标:04d>.png``；
    已存在则复用（断点续跑时不重复解码）。迭代结束后
    ``frames_by_episode`` / ``task_hint_by_episode`` 可被复核页生成器直接使用。
    """

    def __init__(
        self,
        dataset_root: str | Path,
        frames_dir: str | Path,
        *,
        cameras: tuple[str, ...] = ("observation.images.fixed",),
        n_frames: int = 4,
    ) -> None:
        if n_frames <= 0:
            raise ValueError("n_frames 必须是正整数")
        if not cameras:
            raise ValueError("cameras 不能为空")
        self.meta = read_dataset_meta(dataset_root)
        self.frames_dir = Path(frames_dir).expanduser()
        self.cameras = tuple(cameras)
        self.n_frames = int(n_frames)
        self.frames_by_episode: dict[str, tuple[str, ...]] = {}
        self.task_hint_by_episode: dict[str, str] = {}

    # -- 内部：抽帧 -------------------------------------------------------- #

    def _camera_short(self, camera: str) -> str:
        return camera.rsplit(".", 1)[-1]

    def _extract_frames(self, meta: EpisodeMeta) -> list[str]:
        chunk = meta.episode_index // max(self.meta.chunks_size, 1)
        indices = sample_indices(meta.length, self.n_frames)
        out_dir = self.frames_dir / meta.ref
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for camera in self.cameras:
            short = self._camera_short(camera)
            targets = [out_dir / f"{short}_{index:04d}.png" for index in indices]
            missing = [index for index, path in zip(indices, targets) if not path.is_file()]
            if missing:
                video = self.meta.root / self.meta.video_path.format(
                    episode_chunk=chunk, video_key=camera, episode_index=meta.episode_index
                )
                frames = decode_frames(video, indices)
                self._write_pngs(frames, indices, targets)
            paths.extend(str(path) for path in targets)
        return paths

    @staticmethod
    def _write_pngs(frames: dict[int, object], indices: list[int], targets: list[Path]) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise EpisodeSourceError("需要 opencv 才能写帧（pip install opencv-python）") from exc
        for index, path in zip(indices, targets):
            if path.is_file():
                continue
            cv2.imwrite(str(path), cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR))

    # -- EpisodeSource 协议 ------------------------------------------------ #

    def iter_episodes(self) -> Iterator[EpisodeSample]:
        for meta in self.meta.episodes:
            frames = tuple(self._extract_frames(meta))
            positions = read_gripper_positions(self.meta, meta.episode_index)
            self.frames_by_episode[meta.ref] = frames
            self.task_hint_by_episode[meta.ref] = meta.task
            yield EpisodeSample(
                episode_ref=meta.ref,
                frames=frames,
                sensor_state={"gripper_pos": positions},
                task_hint=meta.task,
            )

    def write_manifest(self, path: str | Path) -> Path:
        """写 ``{episode_ref: {frames, task}}`` 清单，供复核页与审计复用。

        **与已有文件合并**：断点续跑时本次只迭代了部分集，合并才能让复核页
        看到历史已标集对应的帧路径。
        """
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = load_manifest(target)
        for ref, frames in self.frames_by_episode.items():
            payload[ref] = {
                "frames": list(frames),
                "task": self.task_hint_by_episode.get(ref, payload.get(ref, {}).get("task", "")),
            }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return target


def load_manifest(path: str | Path) -> dict[str, dict]:
    """读取 `write_manifest` 产物（不存在时返回空表）。"""
    target = Path(path).expanduser()
    if not target.is_file():
        return {}
    data = json.loads(target.read_text(encoding="utf-8") or "{}")
    if not isinstance(data, dict):
        raise EpisodeSourceError(f"{target} 清单格式错误：应为对象")
    return data
