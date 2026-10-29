"""vlm_labeler 意图 taxonomy / 指令映射 / 标注链路测试 — 纯逻辑，无模型 API。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vlm_labeler import (  # noqa: E402
    INTENT_KEYWORDS,
    SUBTASK_NAMES,
    VLM_LABEL_SYSTEM_PROMPT,
    EpisodeSample,
    FakeLabelingBackend,
    IntentTaxonomy,
    LabelRecord,
    LabelingError,
    VlmLabelingBackend,
    build_label_record,
    define_intent_taxonomy,
    label_dataset,
    label_episode,
    map_instruction_to_intent,
    normalize_instruction,
    read_label_records,
    subtask_from_payload,
    success_from_sensor,
    write_lerobot_metadata,
)


class TestIntentTaxonomy:
    def test_default_taxonomy(self) -> None:
        taxonomy = define_intent_taxonomy()
        assert taxonomy.intents == ("stack", "sort", "place", "pick")
        assert taxonomy.slots == ("object", "target_zone")
        assert taxonomy.version == "v1"

    def test_empty_intents_raise(self) -> None:
        with pytest.raises(ValueError):
            IntentTaxonomy(intents=(), slots=(), version="v1")

    def test_blank_intent_raises(self) -> None:
        with pytest.raises(ValueError):
            IntentTaxonomy(intents=("pick", "  "), slots=(), version="v1")

    def test_duplicate_intent_raises(self) -> None:
        with pytest.raises(ValueError):
            IntentTaxonomy(intents=("pick", "pick"), slots=(), version="v1")

    def test_blank_version_raises(self) -> None:
        with pytest.raises(ValueError):
            IntentTaxonomy(intents=("pick",), slots=(), version="  ")

    def test_dict_roundtrip(self) -> None:
        taxonomy = define_intent_taxonomy()
        assert IntentTaxonomy.from_dict(taxonomy.to_dict()) == taxonomy

    def test_from_dict_missing_field_raises(self) -> None:
        with pytest.raises(ValueError):
            IntentTaxonomy.from_dict({"intents": ["pick"], "slots": []})

    def test_from_dict_non_mapping_raises(self) -> None:
        with pytest.raises(ValueError):
            IntentTaxonomy.from_dict(["pick"])  # type: ignore[arg-type]

    def test_keyword_table_covers_default_intents(self) -> None:
        """默认 taxonomy 的每个意图都必须在关键词表里有条目，否则永远匹配不到。"""
        for intent in define_intent_taxonomy().intents:
            assert INTENT_KEYWORDS.get(intent), f"{intent} 缺少关键词"


class TestNormalizeInstruction:
    def test_collapses_whitespace_and_lowercases(self) -> None:
        assert normalize_instruction("  Pick   UP  ") == "pick up"

    def test_fullwidth_alphanumerics_become_halfwidth(self) -> None:
        assert normalize_instruction("ＡＢＣ１２３") == "abc123"

    def test_ideographic_space_is_treated_as_space(self) -> None:
        assert normalize_instruction("拿起\u3000螺丝刀") == "拿起 螺丝刀"

    def test_non_str_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_instruction(None)  # type: ignore[arg-type]


class TestMapInstructionToIntent:
    @pytest.mark.parametrize("text", ["拿起螺丝刀", "夹起方块", "抓取杯子", "把螺丝刀拾起来"])
    def test_pick(self, text: str) -> None:
        assert map_instruction_to_intent(text, define_intent_taxonomy()) == "pick"

    @pytest.mark.parametrize("text", ["把方块放到托盘", "放进容器里", "把杯子放在左边"])
    def test_place(self, text: str) -> None:
        assert map_instruction_to_intent(text, define_intent_taxonomy()) == "place"

    @pytest.mark.parametrize("text", ["按颜色分类", "把方块归类", "sort by color"])
    def test_sort(self, text: str) -> None:
        assert map_instruction_to_intent(text, define_intent_taxonomy()) == "sort"

    @pytest.mark.parametrize("text", ["把方块叠起来", "堆叠两个杯子", "stack the cubes"])
    def test_stack(self, text: str) -> None:
        assert map_instruction_to_intent(text, define_intent_taxonomy()) == "stack"

    def test_composite_intent_wins_over_single_action(self) -> None:
        taxonomy = define_intent_taxonomy()
        # 同时含 pick 与 place 关键词 → 默认优先级里 place 高于 pick
        assert map_instruction_to_intent("拿起方块放到托盘", taxonomy) == "place"
        # 同时含 place 与 stack 关键词 → stack 优先级最高
        assert map_instruction_to_intent("把方块叠起来放到托盘", taxonomy) == "stack"

    def test_priority_follows_taxonomy_order(self) -> None:
        text = "拿起方块放到托盘"
        place_first = IntentTaxonomy(intents=("place", "pick"), slots=(), version="v1")
        pick_first = IntentTaxonomy(intents=("pick", "place"), slots=(), version="v1")
        assert map_instruction_to_intent(text, place_first) == "place"
        assert map_instruction_to_intent(text, pick_first) == "pick"

    def test_unmapped_returns_none(self) -> None:
        assert map_instruction_to_intent("打开抽屉", define_intent_taxonomy()) is None

    def test_blank_text_returns_none(self) -> None:
        assert map_instruction_to_intent("   ", define_intent_taxonomy()) is None
        assert map_instruction_to_intent("", define_intent_taxonomy()) is None

    def test_fullwidth_input_is_normalized_before_matching(self) -> None:
        assert map_instruction_to_intent("ＰＩＣＫ　ＵＰ", define_intent_taxonomy()) == "pick"

    def test_custom_keywords_override(self) -> None:
        taxonomy = IntentTaxonomy(intents=("poke",), slots=(), version="v1")
        assert map_instruction_to_intent("戳一下方块", taxonomy) is None
        custom = {"poke": ("戳",)}
        assert map_instruction_to_intent("戳一下方块", taxonomy, keywords=custom) == "poke"

    def test_intent_without_keywords_never_matches(self) -> None:
        taxonomy = IntentTaxonomy(intents=("unknown_intent", "pick"), slots=(), version="v1")
        # 无关键词的意图被跳过，继续匹配到 pick
        assert map_instruction_to_intent("拿起方块", taxonomy) == "pick"

    def test_blank_keyword_is_ignored(self) -> None:
        """空关键词会匹配一切，必须被跳过。"""
        taxonomy = IntentTaxonomy(intents=("bad", "pick"), slots=(), version="v1")
        custom = {"bad": ("  ",), "pick": ("拿",)}
        assert map_instruction_to_intent("拿起方块", taxonomy, keywords=custom) == "pick"


class TestLabelRecord:
    def test_minimal_record_is_valid(self) -> None:
        record = LabelRecord(episode_ref="ep0", task_text="拿起螺丝刀")
        assert record.intent is None and record.source == "vlm"

    def test_bad_subtask_raises(self) -> None:
        with pytest.raises(LabelingError):
            LabelRecord(episode_ref="ep0", task_text="t", subtask="fly")

    def test_all_known_subtasks_accepted(self) -> None:
        for name in SUBTASK_NAMES:
            assert LabelRecord(episode_ref="ep0", task_text="t", subtask=name).subtask == name

    def test_bad_confidence_raises(self) -> None:
        with pytest.raises(LabelingError):
            LabelRecord(episode_ref="ep0", task_text="t", confidence=1.5)

    def test_bad_frame_range_raises(self) -> None:
        with pytest.raises(LabelingError):
            LabelRecord(episode_ref="ep0", task_text="t", frame_range=(10, 3))

    def test_bad_source_raises(self) -> None:
        with pytest.raises(LabelingError):
            LabelRecord(episode_ref="ep0", task_text="t", source="robot")

    def test_blank_episode_ref_raises(self) -> None:
        with pytest.raises(LabelingError):
            LabelRecord(episode_ref="   ", task_text="t")

    def test_round_trip_is_stable(self) -> None:
        record = LabelRecord(
            episode_ref="ep3", task_text="拿起螺丝刀", frame_range=(0, 120), intent="pick",
            subtask="grasp", success=True, confidence=0.8, source="human",
            taxonomy_version="v1", note="抽检",
        )
        assert LabelRecord.from_dict(record.to_dict()) == record

    def test_from_dict_missing_ref_raises(self) -> None:
        with pytest.raises(LabelingError):
            LabelRecord.from_dict({"task_text": "t"})


class TestSuccessFromSensor:
    def test_missing_channel_is_none(self) -> None:
        assert success_from_sensor(None) is None
        assert success_from_sensor({}) is None
        assert success_from_sensor({"gripper_pos": []}) is None

    def test_closed_tail_is_holding(self) -> None:
        assert success_from_sensor({"gripper_pos": [1.0, 2.0, 6.8, 6.9]}) is True

    def test_any_holding_frame_in_tail_counts(self) -> None:
        """语义：**末尾窗口内只要有一帧**到达夹持位即算成功（末尾帧可能已松开）。"""
        assert success_from_sensor({"gripper_pos": [1.0, 1.2, 6.8, 1.4]}) is True

    def test_never_holding_is_false(self) -> None:
        assert success_from_sensor({"gripper_pos": [1.0, 1.2, 1.4] * 3}) is False

    def test_only_tail_is_considered(self) -> None:
        """窗口之外的高位置不算成功（夹持必须发生在 episode 后段）。"""
        assert success_from_sensor({"gripper_pos": [6.8] + [1.0] * 10}) is False

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(LabelingError):
            success_from_sensor({"gripper_pos": ["oops"]})


class TestBuildLabelRecord:
    def test_intent_from_task_text(self) -> None:
        record = build_label_record(
            "ep0", {"task_text": "抓起螺丝刀并抬起"}, define_intent_taxonomy()
        )
        assert record.intent == "pick"

    def test_task_falls_back_to_hint(self) -> None:
        record = build_label_record(
            "ep0", {"subtask": "lift"}, define_intent_taxonomy(), default_task="把方块放到托盘"
        )
        assert record.intent == "place" and record.subtask == "lift"

    def test_frame_range_parsed(self) -> None:
        record = build_label_record(
            "ep0", {"task_text": "t", "frame_range": [3, 9]}, define_intent_taxonomy()
        )
        assert record.frame_range == (3, 9)

    def test_taxonomy_version_recorded(self) -> None:
        assert build_label_record(
            "ep0", {"task_text": "t"}, define_intent_taxonomy()
        ).taxonomy_version == "v1"

    def test_unmapped_intent_stays_none(self) -> None:
        assert build_label_record(
            "ep0", {"task_text": "打开抽屉"}, define_intent_taxonomy()
        ).intent is None

    def test_non_mapping_payload_raises(self) -> None:
        with pytest.raises(LabelingError):
            build_label_record("ep0", ["x"], define_intent_taxonomy())  # type: ignore[arg-type]


class _FakeSource:
    """最小假数据源（只实现 iter_episodes，不继承任何基类）。"""

    def __init__(self, samples) -> None:
        self._samples = list(samples)

    def iter_episodes(self):
        return iter(self._samples)


def _samples(count: int) -> _FakeSource:
    return _FakeSource(
        [
            EpisodeSample(
                episode_ref=f"ep{index}",
                frames=("frame.png",),
                sensor_state={"gripper_pos": [6.8]},
                task_hint="抓起螺丝刀并抬起",
            )
            for index in range(count)
        ]
    )


class TestLabelDataset:
    def test_writes_jsonl_and_returns_records(self, tmp_path) -> None:
        backend = FakeLabelingBackend([{"task_text": "抓起螺丝刀并抬起", "confidence": 0.9}] * 3)
        out = tmp_path / "labels.jsonl"
        records = label_dataset(backend, _samples(3), define_intent_taxonomy(), out, echo=False)
        assert len(records) == 3 and len(read_label_records(out)) == 3
        assert records[0].intent == "pick"

    def test_resume_skips_already_labeled(self, tmp_path) -> None:
        out = tmp_path / "labels.jsonl"
        first = FakeLabelingBackend([{"task_text": "抓起螺丝刀并抬起"}] * 2)
        label_dataset(first, _samples(2), define_intent_taxonomy(), out, echo=False)
        assert first.calls == 2
        second = FakeLabelingBackend([{"task_text": "拿起方块"}])
        records = label_dataset(second, _samples(3), define_intent_taxonomy(), out, echo=False)
        assert second.calls == 1          # 已标的前 2 条被跳过，只调 1 次
        assert len(records) == 3

    def test_resume_false_restarts(self, tmp_path) -> None:
        out = tmp_path / "labels.jsonl"
        label_dataset(
            FakeLabelingBackend([{"task_text": "t"}] * 3), _samples(3),
            define_intent_taxonomy(), out, echo=False,
        )
        backend = FakeLabelingBackend([{"task_text": "t"}] * 3)
        label_dataset(backend, _samples(3), define_intent_taxonomy(), out, resume=False, echo=False)
        assert backend.calls == 3

    def test_limit_caps_new_writes(self, tmp_path) -> None:
        backend = FakeLabelingBackend([{"task_text": "t"}] * 5)
        records = label_dataset(
            backend, _samples(5), define_intent_taxonomy(), tmp_path / "l.jsonl", limit=2, echo=False
        )
        assert len(records) == 2 and backend.calls == 2

    def test_partial_failure_keeps_written_records(self, tmp_path) -> None:
        """逐条落盘：第 3 条失败时前 2 条必须已写盘（VLM 调用随时可能断）。"""
        out = tmp_path / "labels.jsonl"
        backend = FakeLabelingBackend(
            [{"task_text": "t"}, {"task_text": "t"}, LabelingError("网络断了")]
        )
        with pytest.raises(LabelingError):
            label_dataset(backend, _samples(3), define_intent_taxonomy(), out, echo=False)
        assert len(read_label_records(out)) == 2


class TestWriteLerobotMetadata:
    def test_sidecar_rows(self, tmp_path) -> None:
        records = [
            LabelRecord(
                episode_ref="ep0", task_text="拿起螺丝刀", intent="pick",
                subtask="grasp", success=True,
            )
        ]
        path = write_lerobot_metadata(records, tmp_path / "episodes_meta.jsonl")
        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert row["episode_ref"] == "ep0" and row["intent"] == "pick"

    def test_task_falls_back_to_hint_map(self, tmp_path) -> None:
        path = write_lerobot_metadata(
            [LabelRecord(episode_ref="ep0", task_text="")],
            tmp_path / "m.jsonl",
            task_text_by_episode={"ep0": "抓起螺丝刀"},
        )
        assert json.loads(path.read_text(encoding="utf-8").strip())["task"] == "抓起螺丝刀"


class _FakeChat:
    """假多模态后端：记录调用参数、返回预设回复。"""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, list | None]] = []

    def chat(self, system: str, user: str, images=None) -> str:
        self.calls.append((system, user, images))
        return self.reply


def _image_loader(path: str) -> str:
    return f"data:image/jpeg;base64,{path}"


class TestVlmLabelingBackend:
    def test_parses_fenced_json(self) -> None:
        chat = _FakeChat('```json\n{"task_text": "抓起螺丝刀并抬起", '
                         '"subtask": "lift", "confidence": 0.8, "note": "ok"}\n```')
        backend = VlmLabelingBackend(chat, image_loader=_image_loader)
        payload = backend.label(["frame.png"])
        assert payload["task_text"] == "抓起螺丝刀并抬起"
        assert payload["subtask"] == "lift"
        assert payload["confidence"] == pytest.approx(0.8)
        assert len(chat.calls[0][2]) == 1

    def test_intent_is_not_returned_by_backend(self) -> None:
        """intent 必须由规则从 task_text 派生（保证 κ 可归因），后端不得直接给。"""
        chat = _FakeChat('{"task_text": "拿起螺丝刀", "subtask": "grasp", "confidence": 0.5}')
        payload = VlmLabelingBackend(chat, image_loader=_image_loader).label(["f.png"])
        assert "intent" not in payload

    def test_invalid_subtask_is_nulled_with_note(self) -> None:
        chat = _FakeChat('{"task_text": "拿起螺丝刀", "subtask": "pick up", "confidence": 0.7}')
        payload = VlmLabelingBackend(chat, image_loader=_image_loader).label(["f.png"])
        assert payload["subtask"] is None
        assert "pick up" in payload["note"]

    def test_confidence_is_clamped(self) -> None:
        chat = _FakeChat('{"task_text": "拿起螺丝刀", "confidence": 3}')
        assert VlmLabelingBackend(chat, image_loader=_image_loader).label(["f.png"])[
            "confidence"] == 1.0
        chat = _FakeChat('{"task_text": "拿起螺丝刀", "confidence": "abc"}')
        assert VlmLabelingBackend(chat, image_loader=_image_loader).label(["f.png"])[
            "confidence"] == 0.0

    def test_missing_task_text_raises(self) -> None:
        """不允许静默回退到数据集指令，否则会把"没读出来"伪装成"读对了"。"""
        chat = _FakeChat('{"subtask": "lift", "confidence": 0.9}')
        with pytest.raises(LabelingError, match="task_text"):
            VlmLabelingBackend(chat, image_loader=_image_loader).label(["f.png"])

    def test_unparseable_reply_raises(self) -> None:
        chat = _FakeChat("对不起，我看不清")
        with pytest.raises(LabelingError, match="无法解析"):
            VlmLabelingBackend(chat, image_loader=_image_loader).label(["f.png"])

    def test_empty_frames_raises(self) -> None:
        with pytest.raises(LabelingError, match="frames"):
            VlmLabelingBackend(_FakeChat("{}"), image_loader=_image_loader).label([])

    def test_end_to_end_build_record_maps_intent(self) -> None:
        chat = _FakeChat('{"task_text": "抓起螺丝刀并抬起", "subtask": "lift", "confidence": 0.9}')
        backend = VlmLabelingBackend(chat, image_loader=_image_loader)
        record = label_episode(
            backend, ["f.png"], {"gripper_pos": [6.8]}, define_intent_taxonomy(),
            episode_ref="ep0",
        )
        assert record.intent == "pick" and record.subtask == "lift" and record.success is True


class TestSubtaskFromPayload:
    def test_takes_last_stage(self) -> None:
        subtask, note = subtask_from_payload({"stages": ["approach", "grasp", "lift"]})
        assert subtask == "lift" and note == ""

    def test_single_stage(self) -> None:
        assert subtask_from_payload({"stages": ["grasp"]})[0] == "grasp"

    def test_respects_temporal_order_not_sorting(self) -> None:
        """stages 是时间顺序，取**末段**即"最后走到哪"；不能排序后取最大。"""
        assert subtask_from_payload({"stages": ["lift", "grasp"]})[0] == "grasp"

    def test_invalid_entries_dropped_and_noted(self) -> None:
        subtask, note = subtask_from_payload({"stages": ["approach", "pick up", "lift"]})
        assert subtask == "lift" and "pick up" in note

    def test_case_and_whitespace_normalized(self) -> None:
        assert subtask_from_payload({"stages": [" Lift "]})[0] == "lift"

    def test_empty_stages_falls_back_to_subtask(self) -> None:
        subtask, note = subtask_from_payload({"stages": [], "subtask": "grasp"})
        assert subtask == "grasp" and "回退" in note

    def test_non_list_stages_ignored(self) -> None:
        subtask, note = subtask_from_payload({"stages": "lift"})
        assert subtask is None and "不是数组" in note

    def test_missing_stages_falls_back_with_note(self) -> None:
        subtask, note = subtask_from_payload({"subtask": "lift"})
        assert subtask == "lift" and "回退" in note

    def test_invalid_subtask_nulled(self) -> None:
        subtask, note = subtask_from_payload({"subtask": "pick up"})
        assert subtask is None and "pick up" in note

    def test_nothing_given(self) -> None:
        assert subtask_from_payload({}) == (None, "")


class TestStagesSchema:
    def test_prompt_asks_for_stage_sequence(self) -> None:
        """回归：prompt 必须要求阶段**序列**。

        原版只要"最靠后的主要阶段"，实测模型退化成整段恒答 `grasp`（人工 recall
        `lift`=0.00），且与它自己写的 task_text（"夹起桌上的螺丝刀"）自相矛盾。
        """
        assert '"stages"' in VLM_LABEL_SYSTEM_PROMPT
        assert "按时间顺序" in VLM_LABEL_SYSTEM_PROMPT
        assert SUBTASK_NAMES[0] in VLM_LABEL_SYSTEM_PROMPT

    def test_backend_derives_subtask_from_stages(self) -> None:
        chat = _FakeChat('{"task_text": "抓起螺丝刀并抬起", '
                         '"stages": ["approach", "grasp", "lift"], "confidence": 0.9}')
        payload = VlmLabelingBackend(chat, image_loader=_image_loader).label(["f.png"])
        assert payload["subtask"] == "lift"

    def test_backend_end_to_end_with_stages(self) -> None:
        chat = _FakeChat('{"task_text": "抓起螺丝刀并抬起", '
                         '"stages": ["grasp", "lift"], "confidence": 0.8}')
        backend = VlmLabelingBackend(chat, image_loader=_image_loader)
        record = label_episode(
            backend, ["f.png"], {"gripper_pos": [6.8]}, define_intent_taxonomy(),
            episode_ref="ep0",
        )
        assert record.subtask == "lift" and record.intent == "pick"
