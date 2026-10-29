"""变体数据集准备测试（P0 / 1.6 真跑前置）— 用**合成数据集**，不依赖仿真。

重点覆盖两类最容易造成静默损坏的行为：

1. **不许写坏源数据集**：`data` 下的 parquet 绝不能硬链接（共享 inode + 截断写 = 源数据被毁）；
2. **三处元数据必须同步**：`info.json` 的 shape、`stats.json` 的逐维统计、parquet 内嵌 HF 元数据的
   `length`——漏一处，"列数一致"就只是假象。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ablation_dataset import (  # noqa: E402
    BASE_ENV_DIM,
    ENV_KEY,
    STATE_KEY,
    AblationDatasetError,
    closure_range,
    copy_dataset_tree,
    extend_stats,
    frame_oracle_inputs,
    patch_hf_metadata,
    patch_info_shape,
    prepare_variant_dataset,
    stats_extended_ok,
    variant_block,
)


def _states() -> np.ndarray:
    """4 帧：tcp 固定，夹爪开合度 0.08 → 0.0（含两个关节）。"""
    return np.array([
        [0.10, 0.00, 0.40, 0.04, 0.04],  # 张开
        [0.10, 0.00, 0.42, 0.02, 0.02],
        [0.10, 0.00, 0.44, 0.00, 0.00],  # 闭合
        [0.10, 0.00, 0.46, 0.03, 0.03],
    ])


def _envs() -> np.ndarray:
    """4 帧：obj 位置 + 其余基础列，最后一位 is_grasped。"""
    env = np.zeros((4, BASE_ENV_DIM))
    env[:, 0:3] = [0.10, 0.00, 0.45]
    env[:, 12] = [0.0, 1.0, 0.0, 1.0]
    return env


class TestFrameOracleInputs:
    def test_maps_state_and_env(self) -> None:
        inputs = frame_oracle_inputs(
            _states()[0], _envs()[0], open_width=0.08, closed_width=0.0
        )
        assert inputs.tcp_pos_m == pytest.approx([0.10, 0.00, 0.40])
        assert inputs.obj_pos_m == pytest.approx([0.10, 0.00, 0.45])
        assert inputs.gripper_width == pytest.approx(0.08)
        assert inputs.is_grasped is False

    def test_is_grasped_flag(self) -> None:
        inputs = frame_oracle_inputs(
            _states()[1], _envs()[1], open_width=0.08, closed_width=0.0
        )
        assert inputs.is_grasped is True

    def test_short_state_rejected(self) -> None:
        with pytest.raises(AblationDatasetError, match=STATE_KEY):
            frame_oracle_inputs([0.1, 0.0], _envs()[0], open_width=0.08, closed_width=0.0)

    def test_short_env_rejected(self) -> None:
        with pytest.raises(AblationDatasetError, match=ENV_KEY):
            frame_oracle_inputs(_states()[0], [0.0, 0.0], open_width=0.08, closed_width=0.0)


class TestClosureRange:
    def test_uses_data_extremes(self) -> None:
        assert closure_range(_states()) == pytest.approx((0.08, 0.0))

    def test_degenerate_range_rejected(self) -> None:
        with pytest.raises(AblationDatasetError, match="退化"):
            closure_range(np.zeros((3, 5)))

    def test_bad_ndim_rejected(self) -> None:
        with pytest.raises(AblationDatasetError, match="二维"):
            closure_range(np.zeros(5))


class TestVariantBlock:
    def _block(self, variant: str) -> np.ndarray:
        return variant_block(
            variant, _states(), _envs(), open_width=0.08, closed_width=0.0
        )

    def test_b0_is_empty(self) -> None:
        assert self._block("B0").shape == (4, 0)

    def test_b1_distance_and_zero_sigma(self) -> None:
        block = self._block("B1")
        assert block.shape == (4, 2)
        # 四帧 tcp_z = 0.40/0.42/0.44/0.46，obj_z = 0.45 → 逐帧距离
        assert block[:, 0].tolist() == pytest.approx([0.05, 0.03, 0.01, 0.01])
        assert block[:, 1] == pytest.approx(0.0)  # 特权上界 → sigma 恒 0

    def test_b2_contact_follows_is_grasped(self) -> None:
        block = self._block("B2")
        assert block.shape == (4, 4)
        assert block[:, 3].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])

    def test_b2_load_is_zero_when_not_grasped(self) -> None:
        block = self._block("B2")
        # 帧 0 张开未夹住 → 0‰；帧 1 半闭且夹住 → 负值
        assert block[0, 2] == pytest.approx(0.0)
        assert block[1, 2] < 0.0

    def test_b3_rejected_without_second_view(self) -> None:
        """B3 的立体深度不能由单相机数据集算出（拒绝而非补假列）。"""
        with pytest.raises(AblationDatasetError, match="检测路径输入|B3"):
            self._block("B3")

    def test_frame_count_mismatch_rejected(self) -> None:
        with pytest.raises(AblationDatasetError, match="帧数"):
            variant_block("B1", _states(), _envs()[:2], open_width=0.08, closed_width=0.0)


class TestMetadataPatches:
    def test_info_shape_patched_and_copied(self) -> None:
        info = {"features": {ENV_KEY: {"dtype": "float32", "shape": [13]}}}
        patched = patch_info_shape(info, 17)
        assert patched["features"][ENV_KEY]["shape"] == [17]
        assert info["features"][ENV_KEY]["shape"] == [13]  # 不改原对象

    def test_info_missing_feature_rejected(self) -> None:
        with pytest.raises(AblationDatasetError, match=ENV_KEY):
            patch_info_shape({"features": {}}, 17)

    def test_hf_metadata_length_patched(self) -> None:
        metadata = {"info": {"features": {ENV_KEY: {"feature": {}, "length": 13}}}}
        patched = patch_hf_metadata(metadata, 15)
        assert patched["info"]["features"][ENV_KEY]["length"] == 15
        assert metadata["info"]["features"][ENV_KEY]["length"] == 13

    def test_hf_metadata_without_key_is_passthrough(self) -> None:
        metadata = {"info": {"features": {}}}
        assert patch_hf_metadata(metadata, 15) == metadata


class TestExtendStats:
    def _entry(self) -> dict:
        return {
            "min": [float(-i) for i in range(13)],
            "max": [float(i) for i in range(13)],
            "mean": [0.0] * 13,
            "std": [1.0] * 13,
            "count": 4,
        }

    def test_extends_each_per_dimension_stat(self) -> None:
        block = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        extended = extend_stats(self._entry(), block)
        assert len(extended["min"]) == 15 and len(extended["std"]) == 15
        assert extended["min"][13] == pytest.approx(1.0)
        assert extended["max"][14] == pytest.approx(4.0)
        assert extended["mean"][13] == pytest.approx(2.0)
        assert extended["count"] == 4  # 标量（整表帧数）不动
        assert stats_extended_ok(extended, 15)

    def test_empty_block_is_noop(self) -> None:
        entry = self._entry()
        assert extend_stats(entry, np.zeros((4, 0), dtype=np.float32)) == entry

    def test_wrong_shape_is_left_alone(self) -> None:
        """形状不符时保留原值（不猜一个形状塞进去），并由 `stats_extended_ok` 暴露出来。"""
        entry = self._entry()
        entry["min"] = [0.0]  # 只有 1 维，与 base_dim 不符
        extended = extend_stats(entry, np.array([[1.0, 2.0]]), )
        assert extended["min"] == [0.0]
        assert not stats_extended_ok(extended, 14)


def _write_synthetic_dataset(root: Path, *, frames: int = 4) -> Path:
    """写一个最小的 v3 数据集（单 parquet + 单"视频"文件 + info/stats）。"""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.top" / "chunk-000").mkdir(parents=True)

    state = _states()[:frames]
    env = _envs()[:frames]
    table = pa.table({
        STATE_KEY: pa.array(state.tolist(), type=pa.list_(pa.float32(), 5)),
        ENV_KEY: pa.array(env.tolist(), type=pa.list_(pa.float32(), BASE_ENV_DIM)),
        "action": pa.array([[0.0] * 4] * frames, type=pa.list_(pa.float32(), 4)),
    })
    metadata = {
        b"huggingface": json.dumps({
            "info": {"features": {ENV_KEY: {"feature": {"dtype": "float32"}, "length": 13}}},
            "fingerprint": "synthetic",
        }).encode("utf-8")
    }
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, str(root / "data" / "chunk-000" / "file-000.parquet"))

    (root / "meta" / "info.json").write_text(
        json.dumps({
            "codebase_version": "v3.0",
            "features": {
                STATE_KEY: {"dtype": "float32", "shape": [5], "names": None},
                ENV_KEY: {"dtype": "float32", "shape": [BASE_ENV_DIM], "names": None},
            },
            "total_frames": frames,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        }, indent=4),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps({
            ENV_KEY: {
                "min": [0.0] * BASE_ENV_DIM,
                "max": [1.0] * BASE_ENV_DIM,
                "mean": [0.0] * BASE_ENV_DIM,
                "std": [1.0] * BASE_ENV_DIM,
                "count": frames,
            }
        }, indent=4),
        encoding="utf-8",
    )
    (root / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4").write_bytes(b"fake-mp4")
    return root


class TestCopyDatasetTree:
    def test_data_parquet_is_skipped(self, tmp_path) -> None:
        """**安全属性**：硬链接 + 截断写会毁掉源数据集，所以 data 下的 parquet 必须跳过。"""
        base = _write_synthetic_dataset(tmp_path / "base")
        out = tmp_path / "out"
        copied, _ = copy_dataset_tree(base, out)
        assert not (out / "data" / "chunk-000" / "file-000.parquet").exists()
        assert (out / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4").is_file()
        assert (out / "meta" / "info.json").is_file()
        assert copied >= 3

    def test_source_survives_prepare(self, tmp_path) -> None:
        """端到端安全属性：准备完 B1 后，源数据集的 ENV 维度必须仍是 13。"""
        import pyarrow.parquet as pq

        base = _write_synthetic_dataset(tmp_path / "base")
        prepare_variant_dataset(base, tmp_path / "out", "B1")
        source = pq.read_table(str(base / "data" / "chunk-000" / "file-000.parquet"))
        assert source.schema.field(ENV_KEY).type.list_size == BASE_ENV_DIM

    def test_base_metadata_survives_two_variants(self, tmp_path) -> None:
        """**meta 下的文件绝不能硬链接**：`write_text` 是 O_TRUNC 写同一 inode，
        会穿透链接改掉源数据集的 info.json/stats.json；多个变体还会共用同一份元数据互相覆盖
        （实测踩到：三个变体的 info.json 全变成 17）。"""
        base = _write_synthetic_dataset(tmp_path / "base")
        before_info = (base / "meta" / "info.json").read_text(encoding="utf-8")
        before_stats = (base / "meta" / "stats.json").read_text(encoding="utf-8")

        prepare_variant_dataset(base, tmp_path / "b1", "B1")
        prepare_variant_dataset(base, tmp_path / "b2", "B2")

        assert (base / "meta" / "info.json").read_text(encoding="utf-8") == before_info
        assert (base / "meta" / "stats.json").read_text(encoding="utf-8") == before_stats
        assert json.loads(before_info)["features"][ENV_KEY]["shape"] == [BASE_ENV_DIM]

    def test_two_variants_get_their_own_metadata(self, tmp_path) -> None:
        """每个变体的 info.json 必须是**自己的**维度（共享 inode 时这里会全变成最后一个）。"""
        base = _write_synthetic_dataset(tmp_path / "base")
        prepare_variant_dataset(base, tmp_path / "b1", "B1")
        prepare_variant_dataset(base, tmp_path / "b2", "B2")
        b1 = json.loads((tmp_path / "b1" / "meta" / "info.json").read_text(encoding="utf-8"))
        b2 = json.loads((tmp_path / "b2" / "meta" / "info.json").read_text(encoding="utf-8"))
        assert b1["features"][ENV_KEY]["shape"] == [15]
        assert b2["features"][ENV_KEY]["shape"] == [17]


class TestPrepareVariantDataset:
    def test_b1_augments_parquet_info_stats_and_metadata(self, tmp_path) -> None:
        import pyarrow.parquet as pq

        base = _write_synthetic_dataset(tmp_path / "base")
        out = tmp_path / "out"
        report = prepare_variant_dataset(base, out, "B1")

        assert report.env_dim == 15 and report.frames == 4
        assert report.extra_features == ("depth_known_size_m", "depth_known_size_sigma_m")

        table = pq.read_table(str(out / "data" / "chunk-000" / "file-000.parquet"))
        assert table.schema.field(ENV_KEY).type.list_size == 15
        env = np.asarray(table.column(ENV_KEY).to_pylist(), dtype=np.float64)
        assert env[:, 13].tolist() == pytest.approx([0.05, 0.03, 0.01, 0.01], abs=1e-6)
        assert env[:, 14] == pytest.approx(0.0)
        assert np.allclose(env[:, :13], _envs(), atol=1e-6)  # 原列不变（float32 存储）

        info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
        assert info["features"][ENV_KEY]["shape"] == [15]

        stats = json.loads((out / "meta" / "stats.json").read_text(encoding="utf-8"))
        assert len(stats[ENV_KEY]["min"]) == 15

        metadata = json.loads(table.schema.metadata[b"huggingface"].decode("utf-8"))
        assert metadata["info"]["features"][ENV_KEY]["length"] == 15

    def test_b0_keeps_dimension(self, tmp_path) -> None:
        base = _write_synthetic_dataset(tmp_path / "base")
        report = prepare_variant_dataset(base, tmp_path / "out", "B0")
        assert report.env_dim == BASE_ENV_DIM
        assert report.extra_features == ()

    def test_b3_refused_with_reason(self, tmp_path) -> None:
        base = _write_synthetic_dataset(tmp_path / "base")
        with pytest.raises(AblationDatasetError, match="两视角"):
            prepare_variant_dataset(base, tmp_path / "out", "B3")

    def test_existing_output_refused(self, tmp_path) -> None:
        base = _write_synthetic_dataset(tmp_path / "base")
        (tmp_path / "out").mkdir()
        with pytest.raises(AblationDatasetError, match="已存在"):
            prepare_variant_dataset(base, tmp_path / "out", "B1")

    def test_dry_run_writes_nothing(self, tmp_path) -> None:
        base = _write_synthetic_dataset(tmp_path / "base")
        out = tmp_path / "out"
        report = prepare_variant_dataset(base, out, "B2", dry_run=True)
        assert report.env_dim == 17
        assert not out.exists()

    def test_non_dataset_rejected(self, tmp_path) -> None:
        with pytest.raises(AblationDatasetError, match="info.json"):
            prepare_variant_dataset(tmp_path, tmp_path / "out", "B1")

    def test_base_with_wrong_env_dim_rejected(self, tmp_path) -> None:
        """门禁读的是 **parquet 的实际维度**（不是 info.json），所以要用 15 维 parquet 触发。"""
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        base = _write_synthetic_dataset(tmp_path / "base")
        path = base / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(str(path))
        widened = np.concatenate([_envs(), np.zeros((4, 2), dtype=np.float32)], axis=1)
        column = pa.array(widened.tolist(), type=pa.list_(pa.float32(), 15))
        index = table.schema.get_field_index(ENV_KEY)
        pq.write_table(table.set_column(index, ENV_KEY, column), str(path))
        with pytest.raises(AblationDatasetError, match="基础"):
            prepare_variant_dataset(base, tmp_path / "out", "B1")
