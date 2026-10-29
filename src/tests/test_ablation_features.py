"""P0 消融的额外观测通道测试（第 38 周 1.3 + 1.5）— 纯逻辑，不训练、不跑仿真。

覆盖两件最容易"静默错位"的事：

- **列序与维度契约**（`real_geometry` 的变体布局）：一旦顺序不一致，训练与推理
  仍会产出"有限"的数值，指标只是变差，事后几乎无法归因；
- **ENV 辅助向量的追加**（`visual_act._append_extra_features` 与 config 维度）：
  默认必须保持 13 维（B0 = 历史行为，老 checkpoint/老数据集不受影响）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from real_geometry import (  # noqa: E402
    VARIANT_EXTRA_FEATURES,
    VARIANT_ORDER,
    DepthEstimate,
    GeometryError,
    build_feature_vector,
    feature_values,
    variant_feature_names,
)

B1_NAMES = ("depth_known_size_m", "depth_known_size_sigma_m")
B2_NAMES = ("load_gripper_permille", "contact_flag")
B3_NAMES = ("depth_stereo_m", "depth_stereo_sigma_m")


class TestVariantFeatureLayout:
    def test_b0_has_no_extra_column(self) -> None:
        assert variant_feature_names("B0") == ()

    def test_columns_are_cumulative_in_documented_order(self) -> None:
        assert variant_feature_names("B1") == B1_NAMES
        assert variant_feature_names("B2") == B1_NAMES + B2_NAMES
        assert variant_feature_names("B3") == B1_NAMES + B2_NAMES + B3_NAMES

    def test_each_variant_is_a_prefix_of_the_next(self) -> None:
        """B_k ⊇ B_{k-1}：累积语义一旦被写成"各自独立"，消融就不再是累积消融。"""
        for previous, current in zip(VARIANT_ORDER, VARIANT_ORDER[1:]):
            names = variant_feature_names(previous)
            assert variant_feature_names(current)[: len(names)] == names

    def test_variant_order_matches_the_extra_table(self) -> None:
        assert set(VARIANT_ORDER) == set(VARIANT_EXTRA_FEATURES)

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(GeometryError, match="variant"):
            variant_feature_names("B4")


class TestFeatureValues:
    def test_maps_depth_estimate(self) -> None:
        depth = DepthEstimate(z_m=0.42, method="known_size", uncertainty_m=0.01)
        values = feature_values(depth=depth)
        assert values["depth_known_size_m"] == pytest.approx(0.42)
        assert values["depth_known_size_sigma_m"] == pytest.approx(0.01)

    def test_rejects_wrong_method(self) -> None:
        """把 stereo 估深当作 known_size 填进来是最隐蔽的一类错，必须在入口挡住。"""
        stereo = DepthEstimate(z_m=0.4, method="stereo", uncertainty_m=0.02)
        with pytest.raises(GeometryError, match="known_size"):
            feature_values(depth=stereo)
        known = DepthEstimate(z_m=0.4, method="known_size", uncertainty_m=0.02)
        with pytest.raises(GeometryError, match="stereo"):
            feature_values(stereo=known)

    def test_omits_unavailable_quantities(self) -> None:
        """没测到的量**不出现**在字典里（而不是填 0），由装配层统一报错。"""
        assert feature_values() == {}

    def test_load_and_contact(self) -> None:
        values = feature_values(load_permille=-206.0, contact=True)
        assert values == {"load_gripper_permille": -206.0, "contact_flag": 1.0}
        assert feature_values(contact=False)["contact_flag"] == 0.0

    def test_non_finite_load_raises(self) -> None:
        with pytest.raises(GeometryError):
            feature_values(load_permille=float("nan"))


class TestBuildFeatureVector:
    def _all_values(self) -> dict[str, float]:
        return feature_values(
            depth=DepthEstimate(z_m=0.42, method="known_size", uncertainty_m=0.01),
            stereo=DepthEstimate(z_m=0.44, method="stereo", uncertainty_m=0.03),
            load_permille=-206.0,
            contact=True,
        )

    def test_b0_is_empty_vector(self) -> None:
        vector = build_feature_vector("B0", {})
        assert vector.shape == (0,) and vector.dtype == np.float32

    def test_vector_order_follows_names(self) -> None:
        vector = build_feature_vector("B3", self._all_values())
        assert vector.shape == (6,)
        expected = [self._all_values()[name] for name in variant_feature_names("B3")]
        assert vector.tolist() == pytest.approx(expected)

    def test_missing_key_raises_and_lists_it(self) -> None:
        values = self._all_values()
        del values["load_gripper_permille"]
        with pytest.raises(GeometryError, match="load_gripper_permille"):
            build_feature_vector("B2", values)

    def test_b1_does_not_require_b2_columns(self) -> None:
        """变体只要自己那几列——否则 B1 会被 B2/B3 的列"顺带"强制要求。"""
        only_b1 = feature_values(
            depth=DepthEstimate(z_m=0.42, method="known_size", uncertainty_m=0.01)
        )
        assert build_feature_vector("B1", only_b1).shape == (2,)

    def test_non_finite_value_raises(self) -> None:
        values = self._all_values()
        values["contact_flag"] = float("inf")
        with pytest.raises(GeometryError, match="非有限"):
            build_feature_vector("B2", values)


class TestVisualActExtraChannel:
    def _config(self, **kwargs):
        visual_act = pytest.importorskip("visual_act")
        return visual_act, visual_act.VisualACTTrainingConfig(
            dataset_dir=kwargs.pop("dataset_dir", "."),
            repo_id="demo",
            output_dir=kwargs.pop("output_dir", "."),
            tasks=("PickCube-v1",),
            **kwargs,
        )

    def test_default_channel_is_base_13_dim(self) -> None:
        """B0 必须与历史行为**完全一致**（老 checkpoint、老数据集不受影响）。"""
        visual_act, config = self._config()
        assert config.extra_features == ()
        assert config.env_state_dim == visual_act.ENV_STATE_DIM == 13

    def test_dim_grows_with_extra_features(self) -> None:
        _, config = self._config(extra_features=variant_feature_names("B2"))
        assert config.env_state_dim == 13 + 4

    def test_blank_extra_feature_rejected(self) -> None:
        with pytest.raises(ValueError, match="非空字符串"):
            self._config(extra_features=("depth", "  "))

    def test_duplicate_extra_feature_rejected(self) -> None:
        with pytest.raises(ValueError, match="重复"):
            self._config(extra_features=("depth", "depth"))

    def test_append_helper_concatenates_in_order(self) -> None:
        visual_act = pytest.importorskip("visual_act")
        env_state = np.zeros((13,), dtype=np.float32)
        extras = np.array([1.0, 2.0], dtype=np.float32)
        merged = visual_act._append_extra_features(env_state, extras, 2, "PickCube-v1")
        assert merged.shape == (15,)
        assert merged[:13].tolist() == [0.0] * 13
        assert merged[13:].tolist() == [1.0, 2.0]

    def test_append_helper_rejects_wrong_dim(self) -> None:
        visual_act = pytest.importorskip("visual_act")
        with pytest.raises(visual_act.VisualACTError, match="维度不符"):
            visual_act._append_extra_features(
                np.zeros((13,), dtype=np.float32), np.zeros((3,), dtype=np.float32), 2, "PickCube-v1"
            )

    def test_append_helper_rejects_non_finite(self) -> None:
        visual_act = pytest.importorskip("visual_act")
        with pytest.raises(visual_act.VisualACTError, match="非有限"):
            visual_act._append_extra_features(
                np.zeros((13,), dtype=np.float32),
                np.array([float("nan"), 1.0], dtype=np.float32),
                2,
                "PickCube-v1",
            )


class _DummyPolicy:
    """只提供 `VisualACTPolicy.__init__` 会读到的字段。"""

    class config:  # noqa: N801 - 模拟 lerobot policy 的 config 属性
        n_action_steps = 1

    def reset(self) -> None:  # pragma: no cover - 本类测试不调用
        return None


class TestPolicyRequiresExtraProvider:
    def test_missing_provider_is_rejected(self) -> None:
        """带额外通道的 checkpoint 若无 provider，宁可报错也不能把通道当 0 用。"""
        visual_act = pytest.importorskip("visual_act")
        with pytest.raises(visual_act.VisualACTError, match="extra_provider"):
            visual_act.VisualACTPolicy(
                env_id="PickCube-v1",
                policy=_DummyPolicy(),
                preprocessor=lambda batch: batch,
                postprocessor=lambda action: action,
                resolution=224,
                extra_features=("depth_known_size_m",),
            )

    def test_provider_is_accepted(self) -> None:
        visual_act = pytest.importorskip("visual_act")
        wrapper = visual_act.VisualACTPolicy(
            env_id="PickCube-v1",
            policy=_DummyPolicy(),
            preprocessor=lambda batch: batch,
            postprocessor=lambda action: action,
            resolution=224,
            extra_features=("depth_known_size_m",),
            extra_provider=lambda observation: [0.4],
        )
        assert wrapper._extra_features == ("depth_known_size_m",)

    def test_b0_policy_needs_no_provider(self) -> None:
        visual_act = pytest.importorskip("visual_act")
        wrapper = visual_act.VisualACTPolicy(
            env_id="PickCube-v1",
            policy=_DummyPolicy(),
            preprocessor=lambda batch: batch,
            postprocessor=lambda action: action,
            resolution=224,
        )
        assert wrapper._extra_features == ()
