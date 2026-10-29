"""逐集成功判据（`episode_success` + `run_episode_success`）单测 — 纯逻辑。

不依赖 torch / 网络 / 真机。数据集读取用例在临时目录里**自造**一个最小 LeRobot
数据集（`meta/info.json` + `meta/episodes.jsonl` + 一集 parquet）与旁挂 jsonl。
`pyarrow` 缺失时相关用例自动跳过，保证纯逻辑部分在任何环境都可跑。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from episode_success import (  # noqa: E402
    CRITERIA,
    EpisodeChannels,
    EpisodeSuccess,
    EpisodeSuccessError,
    assess_episode,
    divergent_criteria,
    episode_held_and_lifted,
    episode_held_any,
    episode_lifted_any,
    episode_position_only,
    episode_position_or_load,
    episode_released_at_end,
    task_from_instruction,
)
from vlm_labeler import LabelRecord  # noqa: E402

from run_episode_success import (  # noqa: E402
    evaluate_criterion,
    load_channel_stats,
    load_human_by_task,
    run,
)

# 阈值口径（来自 real_observation 的实测标定）：夹住 > 4.1+1.4 = 5.5；抬起 >= 75。
HOLD = 8.0
OPEN = 2.0
LIFTED = 80.0
NOT_LIFTED = 60.0
HIGH_LOAD = -200.0
LOW_LOAD = -50.0


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

class TestEpisodeChannels:
    def test_empty_gripper_pos_rejected(self) -> None:
        with pytest.raises(EpisodeSuccessError):
            EpisodeChannels(gripper_pos=())

    def test_none_optional_channels_allowed(self) -> None:
        channels = EpisodeChannels(gripper_pos=(1.0, 2.0))
        assert channels.wrist_flex_pos is None and channels.load is None


# --------------------------------------------------------------------------- #
# 口径 1：position_only（复刻现状）
# --------------------------------------------------------------------------- #

class TestPositionOnly:
    def test_tail_closed_is_true(self) -> None:
        assert episode_position_only(EpisodeChannels(gripper_pos=(OPEN, OPEN, HOLD))) is True

    def test_tail_open_is_false(self) -> None:
        """夹住只发生在尾 5 帧之外 → 与 success_from_sensor 一致判 False。"""
        channels = EpisodeChannels(gripper_pos=(HOLD, OPEN, OPEN, OPEN, OPEN, OPEN))
        assert episode_position_only(channels) is False

    def test_single_frame_is_decidable(self) -> None:
        assert episode_position_only(EpisodeChannels(gripper_pos=(HOLD,))) is True
        assert episode_position_only(EpisodeChannels(gripper_pos=(OPEN,))) is False


# --------------------------------------------------------------------------- #
# 口径 2：position_or_load
# --------------------------------------------------------------------------- #

class TestPositionOrLoad:
    def test_load_rescues_position_miss(self) -> None:
        """尾帧位置未超阈值，但负载超阈值 → 判夹住。"""
        channels = EpisodeChannels(gripper_pos=(OPEN,), load=(HIGH_LOAD,))
        assert episode_position_or_load(channels) is True

    def test_open_and_low_load_is_false(self) -> None:
        channels = EpisodeChannels(gripper_pos=(OPEN, OPEN), load=(LOW_LOAD, LOW_LOAD))
        assert episode_position_or_load(channels) is False

    def test_missing_load_falls_back_to_position(self) -> None:
        assert episode_position_or_load(EpisodeChannels(gripper_pos=(HOLD,))) is True
        assert episode_position_or_load(EpisodeChannels(gripper_pos=(OPEN,))) is False

    def test_short_load_series_is_tolerated(self) -> None:
        """旁挂缺帧（load 比位置短）时越界按"缺"处理，不报错。"""
        channels = EpisodeChannels(gripper_pos=(OPEN, OPEN, HOLD), load=(LOW_LOAD,))
        assert episode_position_or_load(channels) is True


# --------------------------------------------------------------------------- #
# 口径 3 子判据：夹住且抬起 / 集末已释放
# --------------------------------------------------------------------------- #

class TestHeldAndLifted:
    def test_held_and_lifted_frame(self) -> None:
        channels = EpisodeChannels(gripper_pos=(OPEN, HOLD), wrist_flex_pos=(NOT_LIFTED, LIFTED))
        assert episode_held_and_lifted(channels) is True

    def test_held_but_not_lifted(self) -> None:
        channels = EpisodeChannels(gripper_pos=(HOLD, HOLD), wrist_flex_pos=(NOT_LIFTED, NOT_LIFTED))
        assert episode_held_and_lifted(channels) is False

    def test_lifted_but_not_held(self) -> None:
        channels = EpisodeChannels(gripper_pos=(OPEN, OPEN), wrist_flex_pos=(LIFTED, LIFTED))
        assert episode_held_and_lifted(channels) is False

    def test_load_can_supply_the_hold_side(self) -> None:
        channels = EpisodeChannels(
            gripper_pos=(OPEN,), wrist_flex_pos=(LIFTED,), load=(HIGH_LOAD,)
        )
        assert episode_held_and_lifted(channels) is True

    def test_missing_wrist_is_none(self) -> None:
        assert episode_held_and_lifted(EpisodeChannels(gripper_pos=(HOLD,))) is None


class TestReleasedAtEnd:
    def test_open_and_low_load_is_released(self) -> None:
        channels = EpisodeChannels(gripper_pos=(HOLD, OPEN), load=(HIGH_LOAD, LOW_LOAD))
        assert episode_released_at_end(channels) is True

    def test_end_still_closed_is_not_released(self) -> None:
        assert episode_released_at_end(EpisodeChannels(gripper_pos=(OPEN, HOLD))) is False

    def test_end_open_but_high_load_is_not_released(self) -> None:
        channels = EpisodeChannels(gripper_pos=(HOLD, OPEN), load=(LOW_LOAD, HIGH_LOAD))
        assert episode_released_at_end(channels) is False

    def test_single_frame(self) -> None:
        assert episode_released_at_end(EpisodeChannels(gripper_pos=(OPEN,))) is True
        assert episode_released_at_end(EpisodeChannels(gripper_pos=(HOLD,))) is False


class TestDiagnostics:
    def test_held_any_and_lifted_any(self) -> None:
        channels = EpisodeChannels(
            gripper_pos=(OPEN, HOLD, HOLD, OPEN),
            wrist_flex_pos=(NOT_LIFTED, NOT_LIFTED, LIFTED, LIFTED),
        )
        assert episode_held_any(channels) is True
        assert episode_lifted_any(channels) is True

    def test_held_any_false_when_never_held(self) -> None:
        assert episode_held_any(EpisodeChannels(gripper_pos=(OPEN, OPEN))) is False

    def test_lifted_any_none_without_wrist(self) -> None:
        assert episode_lifted_any(EpisodeChannels(gripper_pos=(HOLD,))) is None


# --------------------------------------------------------------------------- #
# 任务类型映射 + 任务感知聚合
# --------------------------------------------------------------------------- #

class TestTaskFromInstruction:
    def test_pick_and_place(self) -> None:
        assert task_from_instruction("夹起桌面上的螺丝刀") == "pick"
        assert task_from_instruction("把螺丝刀放到左边") == "place"

    def test_empty_is_none(self) -> None:
        assert task_from_instruction("") is None


class TestAssessEpisode:
    def _place_channels(self, *, release: bool) -> EpisodeChannels:
        end = OPEN if release else HOLD
        return EpisodeChannels(
            gripper_pos=(OPEN, HOLD, HOLD, OPEN, end),
            wrist_flex_pos=(NOT_LIFTED, LIFTED, LIFTED, LIFTED, NOT_LIFTED),
            load=(LOW_LOAD, HIGH_LOAD, HIGH_LOAD, LOW_LOAD, LOW_LOAD),
        )

    def test_pick_aware_true_when_held_and_lifted(self) -> None:
        result = assess_episode("ep0", "pick", self._place_channels(release=True))
        assert result.task_aware is True and result.pick_achieved is True

    def test_pick_aware_false_when_never_lifted(self) -> None:
        channels = EpisodeChannels(gripper_pos=(HOLD, HOLD), wrist_flex_pos=(NOT_LIFTED, NOT_LIFTED))
        result = assess_episode("ep0", "pick", channels)
        assert result.position_only is True and result.task_aware is False

    def test_place_aware_true_when_released(self) -> None:
        result = assess_episode("ep0", "place", self._place_channels(release=True))
        assert result.task_aware is True and result.released_at_end is True

    def test_place_aware_false_when_end_still_held(self) -> None:
        result = assess_episode("ep0", "place", self._place_channels(release=False))
        assert result.pick_achieved is True and result.released_at_end is False
        assert result.task_aware is False

    def test_place_aware_false_when_pick_not_achieved(self) -> None:
        channels = EpisodeChannels(gripper_pos=(OPEN, OPEN), wrist_flex_pos=(LIFTED, LIFTED))
        result = assess_episode("ep0", "place", channels)
        assert result.pick_achieved is False and result.task_aware is False

    def test_unknown_task_leaves_aware_none(self) -> None:
        result = assess_episode("ep0", "stack", self._place_channels(release=True))
        assert result.task_aware is None
        # 前两种口径仍照常给出
        assert result.position_or_load is not None

    def test_left_never_verified(self) -> None:
        result = assess_episode("ep0", "place", self._place_channels(release=True))
        assert result.left_verified is False

    def test_value_by_criterion(self) -> None:
        result = assess_episode("ep0", "pick", self._place_channels(release=True))
        for criterion in CRITERIA:
            assert result.value(criterion) == getattr(result, criterion)

    def test_value_rejects_unknown_criterion(self) -> None:
        result = assess_episode("ep0", "pick", EpisodeChannels(gripper_pos=(HOLD,)))
        with pytest.raises(EpisodeSuccessError):
            result.value("nope")

    def test_to_dict_roundtrip_keys(self) -> None:
        result = assess_episode("ep0", "pick", self._place_channels(release=True))
        payload = result.to_dict()
        assert payload["episode_ref"] == "ep0" and payload["left_verified"] is False


class TestDivergentCriteria:
    def _result(self, **kwargs) -> EpisodeSuccess:
        base = dict(
            episode_ref="ep0", task="pick", position_only=True, position_or_load=True,
            task_aware=True, pick_achieved=True, released_at_end=None,
            held_any=True, lifted_any=True,
        )
        base.update(kwargs)
        return EpisodeSuccess(**base)

    def test_all_equal_returns_empty(self) -> None:
        assert divergent_criteria(self._result()) == ()

    def test_difference_is_reported(self) -> None:
        assert divergent_criteria(self._result(task_aware=False)) == CRITERIA


# --------------------------------------------------------------------------- #
# 报告装配（口径对照 / 负载统计）
# --------------------------------------------------------------------------- #

def _success(ref: str, criterion: str, value) -> EpisodeSuccess:
    values = {name: None for name in CRITERIA}
    values[criterion] = value
    return EpisodeSuccess(
        episode_ref=ref, task="pick", position_only=values["position_only"],
        position_or_load=values["position_or_load"], task_aware=values["task_aware"],
        pick_achieved=None, released_at_end=None, held_any=None, lifted_any=None,
    )


def _human(ref: str, success: bool, task: str = "t") -> LabelRecord:
    return LabelRecord(episode_ref=ref, task_text=task, success=success, source="human")


class TestEvaluateCriterion:
    def test_confusion_agreement_and_kappa(self) -> None:
        auto = {"ep0": _success("ep0", "position_only", True),
                "ep1": _success("ep1", "position_only", False)}
        human = {"ep0": _human("ep0", True), "ep1": _human("ep1", True)}
        comp = evaluate_criterion(auto, human, "position_only")
        assert comp["confusion"] == [[0, 1], [0, 1]]
        assert comp["agreement"] == pytest.approx(0.5)
        assert comp["disagreements"] == ["ep1"]

    def test_unpaired_values_dropped(self) -> None:
        auto = {"ep0": _success("ep0", "position_only", None),
                "ep1": _success("ep1", "position_only", True)}
        human = {"ep0": _human("ep0", True), "ep1": _human("ep1", True)}
        comp = evaluate_criterion(auto, human, "position_only")
        assert comp["n_pairs"] == 1 and comp["agreement"] == pytest.approx(1.0)

    def test_no_pairs_gives_none(self) -> None:
        comp = evaluate_criterion({}, {}, "position_only")
        assert comp["n_pairs"] == 0 and comp["agreement"] is None and comp["kappa"] is None


class TestLoadChannelStats:
    def test_stats_and_threshold_counting(self) -> None:
        loads = {0: [0.0, -200.0, -50.0], 1: [-60.0, None]}
        stats = load_channel_stats(loads)
        assert stats["frames"] == 4
        assert stats["frames_over_threshold"] == 1
        assert stats["episodes_with_any_over_threshold"] == 1
        assert stats["load_abs_max"] == pytest.approx(200.0)
        assert stats["per_episode_max_abs_load"] == {"0": 200.0, "1": 60.0}


class TestLoadHumanByTask:
    def test_groups_by_task_text(self, tmp_path) -> None:
        path = tmp_path / "human (1).jsonl"
        path.write_text(
            "\n".join(
                json.dumps(record.to_dict(), ensure_ascii=False)
                for record in [_human("ep0", True), _human("ep1", False)]
            ),
            encoding="utf-8",
        )
        grouped = load_human_by_task(tmp_path)
        assert set(grouped) == {"t"} and len(grouped["t"]) == 2

    def test_missing_dir_returns_empty(self, tmp_path) -> None:
        assert load_human_by_task(tmp_path / "nope") == {}


# --------------------------------------------------------------------------- #
# 端到端（临时最小数据集；需要 pyarrow）
# --------------------------------------------------------------------------- #

def _write_min_dataset(
    root, instruction: str, episodes: list[list[tuple[float, ...]]]
) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": [
                    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
                ],
            }
        },
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        json.dumps(
            {"episode_index": i, "tasks": [instruction], "length": len(rows)},
            ensure_ascii=False,
        )
        for i, rows in enumerate(episodes)
    ]
    (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for i, rows in enumerate(episodes):
        table = pa.table({"observation.state": pa.array(rows, type=pa.list_(pa.float32()))})
        pq.write_table(
            table, root / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
        )


def _row(wrist: float, grip: float) -> tuple[float, ...]:
    return (0.0, 0.0, 0.0, wrist, 0.0, grip)


class TestRunPipeline:
    def _setup(self, tmp_path, instruction: str, episodes, loads, human):
        ds_root = tmp_path / "ds"
        repo = "demo_x"
        _write_min_dataset(ds_root / repo, instruction, episodes)
        sidecar_dir = tmp_path / "sidecars"
        sidecar_dir.mkdir()
        with (sidecar_dir / f"{repo}.jsonl").open("w", encoding="utf-8") as handle:
            for episode_index, values in enumerate(loads):
                for frame_index, value in enumerate(values):
                    handle.write(json.dumps({
                        "dataset_repo": repo,
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "load": {"gripper.load": value},
                    }, ensure_ascii=False) + "\n")
        human_dir = tmp_path / "human"
        human_dir.mkdir()
        (human_dir / "human (1).jsonl").write_text(
            "\n".join(json.dumps(record.to_dict(), ensure_ascii=False) for record in human)
            + "\n",
            encoding="utf-8",
        )
        return ds_root, sidecar_dir, human_dir, repo

    def test_place_episode_release_detected(self, tmp_path) -> None:
        # ep0：夹起并松手（位置尾帧已张开）→ position_only=False、task_aware=True
        instruction = "把螺丝刀放到左边"
        episodes = [
            [_row(NOT_LIFTED, 8.0), _row(LIFTED, 8.0), _row(LIFTED, 8.0),
             _row(LIFTED, 3.0), _row(NOT_LIFTED, 2.0), _row(NOT_LIFTED, 2.0),
             _row(NOT_LIFTED, 2.0), _row(NOT_LIFTED, 2.0)],
        ]
        loads = [[0.0, -200.0, -180.0, -80.0, -50.0, -50.0, -50.0, -50.0]]
        human = [_human("episode_000000", True, task=instruction)]
        ds_root, sidecar_dir, human_dir, _ = self._setup(
            tmp_path, instruction, episodes, loads, human
        )
        out = tmp_path / "out"
        report = run(ds_root, sidecar_dir, human_dir, out)

        block = report["per_repo"]["demo_x"]
        assert block["intent"] == "place"
        assert block["criteria_comparison"]["task_aware"]["n_pairs"] == 1
        ep = report["episode_results"]["demo_x"][0]
        assert ep["position_only"] is False
        assert ep["task_aware"] is True and ep["released_at_end"] is True
        assert ep["left_verified"] is False
        assert (out / "report.json").is_file() and (out / "report.md").is_file()

    def test_overall_comparison_present(self, tmp_path) -> None:
        instruction = "夹起桌面上的螺丝刀"
        episodes = [[_row(LIFTED, 8.0), _row(LIFTED, 8.0)],
                    [_row(NOT_LIFTED, 2.0), _row(NOT_LIFTED, 2.0)]]
        loads = [[-200.0, -200.0], [0.0, 0.0]]
        human = [
            _human("episode_000000", True, task=instruction),
            _human("episode_000001", True, task=instruction),
        ]
        ds_root, sidecar_dir, human_dir, _ = self._setup(
            tmp_path, instruction, episodes, loads, human
        )
        report = run(ds_root, sidecar_dir, human_dir, tmp_path / "out")
        pooled = report["criteria_comparison_pooled"]
        assert set(pooled) == set(CRITERIA)
        # ep0 夹住抬起→True，ep1 全开→False → 与"人工全 True"一致率 50%
        assert pooled["task_aware"]["agreement"] == pytest.approx(0.5)
        assert report["load_channel_summary"]["demo_x"]["frames_over_threshold"] == 2
