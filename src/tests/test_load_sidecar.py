"""负载旁挂文件（格式/写盘/对齐校验）测试 — 纯逻辑，无 lerobot / 无硬件。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_sidecar import (  # noqa: E402
    LoadSidecarWriter,
    SidecarError,
    decode_loads,
    episode_loads,
    read_load_sidecar,
    verify_alignment,
)

# grip.raw = 0x400 | 206 → 方向位=1、幅值 206 → 解码 −206‰
GRIP_RAW_HELD = 0x400 | 206
GRIP_RAW_EMPTY = 0x400 | 62


class TestDecodeLoads:
    def test_decodes_direction_bit(self) -> None:
        decoded, failed = decode_loads({"gripper.load": GRIP_RAW_HELD, "shoulder_pan.load": 12})
        assert decoded["gripper.load"] == pytest.approx(-206.0)
        assert decoded["shoulder_pan.load"] == pytest.approx(12.0)
        assert failed == []

    def test_bare_joint_name_is_normalized(self) -> None:
        decoded, _ = decode_loads({"gripper": GRIP_RAW_EMPTY})
        assert decoded["gripper.load"] == pytest.approx(-62.0)

    def test_out_of_range_is_reported_not_raised(self) -> None:
        """单个关节读数异常不能中断整场采集。"""
        decoded, failed = decode_loads({"gripper.load": 99999, "shoulder_pan.load": 5})
        assert failed == ["gripper.load"]
        assert decoded["shoulder_pan.load"] == pytest.approx(5.0)


class TestLoadSidecarWriter:
    def _raw(self, grip: int = GRIP_RAW_HELD) -> dict:
        return {"shoulder_pan.load": 0, "shoulder_lift.load": 0, "elbow_flex.load": 0,
                "wrist_flex.load": 0, "wrist_roll.load": 0, "gripper.load": grip}

    def test_writes_one_row_per_frame(self, tmp_path) -> None:
        path = tmp_path / "load.jsonl"
        with LoadSidecarWriter(path, dataset_repo="repo/x") as writer:
            writer.begin_episode(0)
            for frame in range(3):
                writer.add_frame(0, frame, self._raw(), {"gripper.pos": 6.8})
        rows = read_load_sidecar(path)
        assert [row["frame_index"] for row in rows] == [0, 1, 2]
        assert rows[0]["dataset_repo"] == "repo/x"
        assert rows[0]["load"]["gripper.load"] == pytest.approx(-206.0)
        assert rows[0]["load_raw"]["gripper.load"] == GRIP_RAW_HELD
        assert rows[0]["positions"] == {"gripper.pos": 6.8}

    def test_episode_advance_commits_previous(self, tmp_path) -> None:
        path = tmp_path / "load.jsonl"
        with LoadSidecarWriter(path) as writer:
            writer.begin_episode(0)
            writer.add_frame(0, 0, self._raw())
            writer.begin_episode(1)          # 推进集号 → 提交第 0 集
            writer.add_frame(1, 0, self._raw(GRIP_RAW_EMPTY))
        rows = read_load_sidecar(path)
        assert [row["episode_index"] for row in rows] == [0, 1]

    def test_rerecord_same_episode_discards_old_frames(self, tmp_path) -> None:
        """坏集重录：同集号再次开始必须丢弃上一遍，否则旁挂里会留下垃圾行。"""
        path = tmp_path / "load.jsonl"
        with LoadSidecarWriter(path) as writer:
            writer.begin_episode(0)
            writer.add_frame(0, 0, self._raw(GRIP_RAW_EMPTY))
            writer.add_frame(0, 1, self._raw(GRIP_RAW_EMPTY))
            writer.begin_episode(0)          # 重录
            writer.add_frame(0, 0, self._raw(GRIP_RAW_HELD))
        rows = read_load_sidecar(path)
        assert len(rows) == 1
        assert rows[0]["load"]["gripper.load"] == pytest.approx(-206.0)

    def test_abort_discards_pending(self, tmp_path) -> None:
        path = tmp_path / "load.jsonl"
        with LoadSidecarWriter(path) as writer:
            writer.begin_episode(0)
            writer.add_frame(0, 0, self._raw())
            writer.abort_episode()
        assert read_load_sidecar(path) == []

    def test_add_frame_without_begin_raises(self, tmp_path) -> None:
        with LoadSidecarWriter(tmp_path / "l.jsonl") as writer:
            with pytest.raises(SidecarError):
                writer.add_frame(0, 0, self._raw())

    def test_appends_across_runs(self, tmp_path) -> None:
        """`--resume` 续采时同一文件追加而不是覆盖。"""
        path = tmp_path / "load.jsonl"
        for episode in (0, 1):
            with LoadSidecarWriter(path) as writer:
                writer.begin_episode(episode)
                writer.add_frame(episode, 0, self._raw())
        assert len(read_load_sidecar(path)) == 2


class TestEpisodeLoads:
    def test_sorted_by_frame_index(self, tmp_path) -> None:
        path = tmp_path / "l.jsonl"
        with LoadSidecarWriter(path) as writer:
            writer.begin_episode(0)
            writer.add_frame(0, 0, {"gripper.load": GRIP_RAW_EMPTY})
            writer.add_frame(0, 1, {"gripper.load": GRIP_RAW_HELD})
        loads = episode_loads(read_load_sidecar(path))
        assert loads[0] == [pytest.approx(-62.0), pytest.approx(-206.0)]

    def test_missing_joint_is_none(self, tmp_path) -> None:
        path = tmp_path / "l.jsonl"
        with LoadSidecarWriter(path) as writer:
            writer.begin_episode(0)
            writer.add_frame(0, 0, {"shoulder_pan.load": 3})
        assert episode_loads(read_load_sidecar(path))[0] == [None]


class TestVerifyAlignment:
    def _rows(self, positions: list[float]) -> list[dict]:
        return [
            {"episode_index": 0, "frame_index": i, "positions": {"gripper.pos": value}}
            for i, value in enumerate(positions)
        ]

    def test_exact_match_passes(self) -> None:
        report = verify_alignment(self._rows([1.0, 2.0, 3.0]), {0: [1.0, 2.0, 3.0]})
        assert report.ok and report.episodes[0].max_abs_diff == 0.0

    def test_frame_count_mismatch_fails(self) -> None:
        report = verify_alignment(self._rows([1.0, 2.0]), {0: [1.0, 2.0, 3.0]})
        assert not report.ok
        assert report.episodes[0].sidecar_frames == 2

    def test_position_mismatch_fails(self) -> None:
        report = verify_alignment(self._rows([1.0, 9.0]), {0: [1.0, 2.0]})
        assert not report.ok and report.episodes[0].max_abs_diff == pytest.approx(7.0)

    def test_float32_rounding_within_tolerance(self) -> None:
        report = verify_alignment(self._rows([1.00001]), {0: [1.0]})
        assert report.ok

    def test_missing_episode_in_dataset_fails(self) -> None:
        report = verify_alignment(self._rows([1.0]), {})
        assert not report.ok

    def test_empty_records_fails(self) -> None:
        assert not verify_alignment([], {0: [1.0]}).ok

    def test_summary_mentions_misaligned_episode(self) -> None:
        report = verify_alignment(self._rows([1.0]), {0: [1.0, 2.0]})
        lines = report.summary_lines()
        assert any("错位" in line for line in lines)
        assert any("不通过" in line for line in lines)
