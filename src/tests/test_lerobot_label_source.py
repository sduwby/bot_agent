"""LeRobot 数据源测试 — 纯逻辑（不 import lerobot/av/pyarrow，不碰真数据集）。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lerobot_label_source import (  # noqa: E402
    EpisodeSourceError,
    LerobotEpisodeSource,
    load_manifest,
    read_dataset_meta,
    sample_indices,
)


class TestSampleIndices:
    def test_includes_first_and_last(self) -> None:
        assert sample_indices(100, 4) == [0, 33, 66, 99]

    def test_single_frame_takes_last(self) -> None:
        """单帧取末帧：演示终态最能体现"是否夹住/抬起"。"""
        assert sample_indices(100, 1) == [99]

    def test_length_le_count_returns_all(self) -> None:
        assert sample_indices(3, 5) == [0, 1, 2]

    def test_no_duplicates_when_spacing_collapses(self) -> None:
        picked = sample_indices(5, 5)
        assert picked == [0, 1, 2, 3, 4] and len(set(picked)) == 5

    def test_zero_length_returns_empty(self) -> None:
        assert sample_indices(0, 4) == []

    def test_non_positive_count_raises(self) -> None:
        with pytest.raises(ValueError):
            sample_indices(10, 0)


def _make_dataset(root, *, names=("shoulder_pan.pos", "gripper.pos"), episodes=((0, 10, ["拿起来"]),)):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {"observation.state": {"names": list(names)}},
            }
        ),
        encoding="utf-8",
    )
    with (root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for index, length, tasks in episodes:
            handle.write(json.dumps({"episode_index": index, "length": length, "tasks": tasks}) + "\n")
    return root


class TestReadDatasetMeta:
    def test_parses_state_names_and_episodes(self, tmp_path) -> None:
        meta = read_dataset_meta(_make_dataset(tmp_path))
        assert meta.fps == 30 and meta.gripper_index() == 1
        assert meta.episodes[0].ref == "episode_000000"
        assert meta.episodes[0].task == "拿起来"

    def test_multiple_tasks_are_joined(self, tmp_path) -> None:
        root = _make_dataset(tmp_path, episodes=((0, 10, ["A", "B"]),))
        assert read_dataset_meta(root).episodes[0].task == "A；B"

    def test_missing_info_raises(self, tmp_path) -> None:
        with pytest.raises(EpisodeSourceError, match="LeRobot"):
            read_dataset_meta(tmp_path)

    def test_missing_gripper_name_raises(self, tmp_path) -> None:
        root = _make_dataset(tmp_path, names=("shoulder_pan.pos",))
        with pytest.raises(EpisodeSourceError, match="gripper.pos"):
            read_dataset_meta(root).gripper_index()


class TestManifest:
    def test_load_missing_returns_empty(self, tmp_path) -> None:
        assert load_manifest(tmp_path / "none.json") == {}

    def test_write_merges_with_existing(self, tmp_path) -> None:
        root = _make_dataset(tmp_path / "ds")
        source = LerobotEpisodeSource(root, tmp_path / "frames", n_frames=2)
        manifest_path = tmp_path / "manifest.json"
        source.frames_by_episode["episode_000000"] = ("a.png",)
        source.task_hint_by_episode["episode_000000"] = "拿起来"
        source.write_manifest(manifest_path)
        # 第二次只带了另一集 → 必须合并而不是覆盖
        source.frames_by_episode = {"episode_000001": ("b.png",)}
        source.task_hint_by_episode = {"episode_000001": "放下"}
        source.write_manifest(manifest_path)
        payload = load_manifest(manifest_path)
        assert set(payload) == {"episode_000000", "episode_000001"}
        assert payload["episode_000000"]["task"] == "拿起来"
