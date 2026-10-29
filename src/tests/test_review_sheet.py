"""人工复核页生成测试 — 纯逻辑，无模型 API、无真数据集。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from review_sheet import (  # noqa: E402
    ReviewItem,
    ReviewSheetError,
    build_review_html,
    select_review_refs,
    write_review_sheet,
)
from vlm_labeler import define_intent_taxonomy  # noqa: E402


class TestSelectReviewRefs:
    def test_ratio_rounds_to_nearest(self) -> None:
        refs = [f"ep{i}" for i in range(68)]
        assert len(select_review_refs(refs, ratio=0.2)) == 14

    def test_deterministic_for_same_seed(self) -> None:
        refs = [f"ep{i}" for i in range(20)]
        assert select_review_refs(refs, ratio=0.3, seed=7) == select_review_refs(
            refs, ratio=0.3, seed=7
        )

    def test_different_seed_changes_sample(self) -> None:
        refs = [f"ep{i}" for i in range(20)]
        assert select_review_refs(refs, ratio=0.3, seed=1) != select_review_refs(
            refs, ratio=0.3, seed=2
        )

    def test_preserves_input_order(self) -> None:
        refs = [f"ep{i}" for i in range(10)]
        picked = select_review_refs(refs, count=4, seed=5)
        assert picked == [ref for ref in refs if ref in set(picked)]

    def test_count_overrides_ratio(self) -> None:
        assert len(select_review_refs([f"ep{i}" for i in range(30)], count=3)) == 3

    def test_at_least_one_for_small_sets(self) -> None:
        assert len(select_review_refs(["ep0", "ep1"], ratio=0.2)) == 1

    def test_empty_input(self) -> None:
        assert select_review_refs([]) == []

    def test_bad_ratio_raises(self) -> None:
        with pytest.raises(ReviewSheetError):
            select_review_refs(["ep0"], ratio=0.0)


def _items() -> list[ReviewItem]:
    return [
        ReviewItem("episode_000000", (r"d:\out\frames\episode_000000\fixed_0000.png",), "抓起螺丝刀并抬起"),
        ReviewItem("episode_000001", (r"d:\out\frames\episode_000001\fixed_0000.png",), "<b>注入</b>"),
    ]


class TestBuildReviewHtml:
    def test_contains_every_episode_and_frame(self) -> None:
        html = build_review_html(
            _items(), taxonomy=define_intent_taxonomy(), base_dir=r"d:\out\review"
        )
        assert "episode_000000" in html and "episode_000001" in html
        # 相对路径 + 正斜杠（供 file:// 直接打开）
        assert "../frames/episode_000000/fixed_0000.png" in html

    def test_escapes_task_text(self) -> None:
        html = build_review_html(
            _items(), taxonomy=define_intent_taxonomy(), base_dir=r"d:\out\review"
        )
        assert "&lt;b&gt;注入&lt;/b&gt;" in html
        assert "<b>注入</b>" not in html

    def test_options_cover_taxonomy_and_subtasks(self) -> None:
        html = build_review_html(
            _items(), taxonomy=define_intent_taxonomy(), base_dir=r"d:\out\review"
        )
        for intent in define_intent_taxonomy().intents:
            assert f'name="episode_000000:intent" value="{intent}"' in html
        for subtask in ("approach", "align", "grasp", "lift", "place"):
            assert f'name="episode_000000:subtask" value="{subtask}"' in html
        assert 'name="episode_000000:success" value="true"' in html

    def test_radio_groups_are_per_episode(self) -> None:
        """回归：radio 的 name 必须**逐集唯一**。

        HTML 中同名 radio 是**全文档互斥**的；各集若共用 `intent` 这个 name，
        选第 2 集的 intent 会自动清掉第 1 集的 —— 表现为"整页只能填一张卡"。
        """
        html = build_review_html(
            _items(), taxonomy=define_intent_taxonomy(), base_dir=r"d:\out\review"
        )
        for field in ("intent", "subtask", "success"):
            assert f'name="episode_000000:{field}"' in html
            assert f'name="episode_000001:{field}"' in html
            # 不得存在未加命名空间的裸 name
            assert f'name="{field}"' not in html

    def test_export_emits_human_source_and_version(self) -> None:
        html = build_review_html(
            _items(), taxonomy=define_intent_taxonomy(), base_dir=r"d:\out\review"
        )
        assert "source: 'human'" in html
        assert 'const TAXONOMY_VERSION = "v1";' in html

    def test_success_wording_matches_sensor_criterion(self) -> None:
        """回归：success 的问法必须与判据一致（判据只判"是否夹住"，不问"是否抬起"）。

        原版问"是否成功夹住并提起"，而 `success_from_sensor` 只判夹住 → 两侧口径不同，
        κ_success 会被口径差异污染。
        """
        html = build_review_html(
            _items(), taxonomy=define_intent_taxonomy(), base_dir=r"d:\out\review"
        )
        assert "夹爪是否夹住物体" in html
        assert "夹住了" in html and "没夹住" in html
        assert "并提起" not in html

    def test_prefill_preselects_previous_answers(self) -> None:
        items = [
            ReviewItem(
                "episode_000000",
                (r"d:\out\frames\episode_000000\fixed_0000.png",),
                "抓起螺丝刀并抬起",
                defaults={"intent": "pick", "subtask": "lift", "success": "false"},
            )
        ]
        html = build_review_html(
            items, taxonomy=define_intent_taxonomy(), base_dir=r"d:\out\review"
        )
        assert 'name="episode_000000:intent" value="pick" checked' in html
        assert 'name="episode_000000:subtask" value="lift" checked' in html
        assert 'name="episode_000000:success" value="false" checked' in html
        # 未预选的项不能带 checked
        assert 'name="episode_000000:subtask" value="grasp" checked' not in html

    def test_without_defaults_nothing_is_checked(self) -> None:
        html = build_review_html(
            _items(), taxonomy=define_intent_taxonomy(), base_dir=r"d:\out\review"
        )
        # 注意不能断言 "checked" 整体——CSS 选择器里的 `:checked` 合法存在
        assert " checked>" not in html

    def test_requires_items(self) -> None:
        with pytest.raises(ReviewSheetError):
            build_review_html([], taxonomy=define_intent_taxonomy(), base_dir=".")


class TestWriteReviewSheet:
    def test_writes_html_file(self, tmp_path) -> None:
        target = tmp_path / "review" / "review.html"
        path = write_review_sheet(target, _items(), taxonomy=define_intent_taxonomy())
        assert path == target and target.is_file()
        assert "episode_000000" in target.read_text(encoding="utf-8")
