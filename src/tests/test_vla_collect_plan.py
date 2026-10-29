"""多任务采集编排（物体 × 指令 → 可执行清单）测试 — 纯逻辑，无硬件。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vla_collect_plan import (  # noqa: E402
    CAMERA_INDICES,
    DEFAULT_INSTRUCTIONS,
    CollectPlanError,
    build_collect_plan,
    collect_combos,
    dataset_root_for,
    paraphrase_available,
    render_record_command,
    render_script,
    render_table,
    render_verify_command,
    select_instructions,
    sidecar_for,
    write_plan,
)


class TestBuildCollectPlan:
    def test_expands_object_first_then_instruction(self) -> None:
        tasks = build_collect_plan(["A", "B"], episodes=5)
        # 物体优先：A 的三条指令在前，B 的三条在后
        assert [(t.object_name, t.instruction_key) for t in tasks] == [
            ("A", "pick_a"),
            ("A", "pick_b"),
            ("A", "place_left"),
            ("B", "pick_a"),
            ("B", "pick_b"),
            ("B", "place_left"),
        ]

    def test_index_is_one_based_and_contiguous(self) -> None:
        tasks = build_collect_plan(["A", "B", "C"])
        assert [t.index for t in tasks] == list(range(1, len(tasks) + 1))

    def test_repo_id_is_ascii_slug_with_object_ordinal(self) -> None:
        tasks = build_collect_plan(["螺丝刀", "蓝色方块"], episodes=20)
        # 中文物体名不能进 repo 名（HF 对非 ASCII 不可靠），用序号区分
        assert tasks[0].repo_id == "wangbangyan/demo_vla_01_pick_a"
        assert tasks[3].repo_id == "wangbangyan/demo_vla_02_pick_a"
        assert all(t.repo_id.isascii() for t in tasks)

    def test_instruction_substitutes_object(self) -> None:
        tasks = build_collect_plan(["螺丝刀"])
        assert tasks[0].instruction == "夹起桌面上的螺丝刀"
        assert tasks[1].instruction == "把螺丝刀拿起来"
        assert tasks[2].instruction == "把螺丝刀放到左边"

    def test_repo_id_has_no_object_name_leak(self) -> None:
        """repo 名只有序号，不含物体名 —— 中文物体名不进 repo。"""
        tasks = build_collect_plan(["螺丝刀"])
        assert "螺丝刀" not in tasks[0].repo_id

    def test_rejects_duplicate_objects(self) -> None:
        with pytest.raises(CollectPlanError, match="不能重复"):
            build_collect_plan(["A", "A"])

    def test_rejects_duplicate_instruction_keys(self) -> None:
        with pytest.raises(CollectPlanError, match="key 不能重复"):
            build_collect_plan(["A"], [("k", "拿起{object}"), ("k", "放下{object}")])

    def test_rejects_template_without_object_placeholder(self) -> None:
        with pytest.raises(CollectPlanError, match="占位"):
            build_collect_plan(["A"], [("pick", "拿起它")])

    def test_rejects_empty_objects(self) -> None:
        with pytest.raises(CollectPlanError, match="至少需要一个物体"):
            build_collect_plan(["", "  "])

    def test_rejects_non_positive_episodes(self) -> None:
        with pytest.raises(CollectPlanError, match="episodes"):
            build_collect_plan(["A"], episodes=0)

    def test_strips_whitespace_in_object_names(self) -> None:
        tasks = build_collect_plan(["  螺丝刀 ", " 蓝色方块"])
        assert [t.object_name for t in tasks[:3]] == ["螺丝刀"] * 3
        assert tasks[3].object_name == "蓝色方块"

    def test_two_default_instructions_are_paraphrases_of_pick(self) -> None:
        """pick_a / pick_b 都是 pick 意图的同义改写 —— 供 paraphrase 协议。"""
        keys = [key for key, _ in DEFAULT_INSTRUCTIONS]
        assert keys[:2] == ["pick_a", "pick_b"]
        assert "拿起来" in DEFAULT_INSTRUCTIONS[1][1]
        # place_left 提供第二个意图，否则意图只有一类、κ 退化
        assert "放到左边" in DEFAULT_INSTRUCTIONS[2][1]


class TestSelectInstructions:
    def test_keeps_declaration_order_not_input_order(self) -> None:
        """采集按声明顺序走，传参顺序不该影响清单顺序。"""
        picked = select_instructions("place_left,pick_a")
        assert [key for key, _ in picked] == ["pick_a", "place_left"]

    def test_accepts_sequence(self) -> None:
        assert [key for key, _ in select_instructions(["place_left"])] == ["place_left"]

    def test_rejects_unknown_key(self) -> None:
        with pytest.raises(CollectPlanError, match="未知的指令 key"):
            select_instructions("pick_a,nope")

    def test_rejects_empty_selection(self) -> None:
        with pytest.raises(CollectPlanError, match="至少要选一条指令"):
            select_instructions(" , ")

    def test_subset_builds_fewer_tasks(self) -> None:
        """砍掉 pick_b → 每物体少一个 repo。"""
        tasks = build_collect_plan(["螺丝刀", "指甲刀"],
                                   select_instructions("pick_a,place_left"))
        assert [(t.object_name, t.instruction_key) for t in tasks] == [
            ("螺丝刀", "pick_a"), ("螺丝刀", "place_left"),
            ("指甲刀", "pick_a"), ("指甲刀", "place_left"),
        ]


class TestParaphraseAvailable:
    def test_true_when_pair_present(self) -> None:
        assert paraphrase_available(select_instructions("pick_a,pick_b,place_left")) is True

    def test_false_when_partner_dropped(self) -> None:
        """缺任一边 → P2 的 paraphrase 协议无留出措辞可用。"""
        assert paraphrase_available(select_instructions("pick_a,place_left")) is False

    def test_default_is_available(self) -> None:
        assert paraphrase_available(DEFAULT_INSTRUCTIONS) is True


class TestCollectCombos:
    def test_shape_and_order_matches_tasks(self) -> None:
        tasks = build_collect_plan(["A", "B"])
        combos = collect_combos(tasks)
        assert combos == [[t.object_name, t.instruction] for t in tasks]
        assert all(len(row) == 2 for row in combos)
        assert combos[0] == ["A", "夹起桌面上的A"]


class TestRenderRecordCommand:
    def test_uses_wrapper_runner_with_sidecar(self) -> None:
        """**必须**走 run_vla_collect_record.py + --sidecar（否则无法做成功标注）。"""
        task = build_collect_plan(["A"])[0]
        command = render_record_command(task)
        assert "run_vla_collect_record.py" in command
        assert "--sidecar" in command
        assert "lerobot-record" not in command

    def test_uses_lerobot_interpreter_not_bare_python(self) -> None:
        """回归：裸 `python` 解析到系统解释器、没有 lerobot → ModuleNotFoundError。"""
        command = render_record_command(build_collect_plan(["A"])[0])
        assert command.startswith('& "D:\\miniconda\\envs\\lerobot\\python.exe"')
        assert not command.startswith("python ")

    def test_python_is_overridable(self) -> None:
        command = render_record_command(build_collect_plan(["A"])[0], python=r"C:\py\python.exe")
        assert command.startswith('& "C:\\py\\python.exe"')

    def test_torque_is_not_disabled_on_disconnect_by_default(self) -> None:
        """回归（真机实测）：退出时卸力矩会在夹爪过载时报 Overload error，中断收尾。"""
        command = render_record_command(build_collect_plan(["A"])[0])
        assert "--robot.disable_torque_on_disconnect=false" in command

    def test_torque_off_can_be_requested(self) -> None:
        command = render_record_command(build_collect_plan(["A"])[0],
                                        disable_torque_on_disconnect=True)
        assert "--robot.disable_torque_on_disconnect=true" in command

    def test_contains_fixed_camera_indices(self) -> None:
        task = build_collect_plan(["A"])[0]
        command = render_record_command(task)
        assert f'"index_or_path": {CAMERA_INDICES["fixed"]}' in command
        assert f'"index_or_path": {CAMERA_INDICES["handeye"]}' in command

    def test_embeds_repo_and_quoted_task_text(self) -> None:
        task = build_collect_plan(["螺丝刀"])[0]
        command = render_record_command(task)
        assert f"--dataset.repo_id={task.repo_id}" in command
        assert f'--dataset.single_task="{task.instruction}"' in command

    def test_creates_dataset_without_push_to_hub(self) -> None:
        task = build_collect_plan(["A"], episodes=7)[0]
        command = render_record_command(task)
        assert "--dataset.num_episodes=7" in command
        assert "--dataset.push_to_hub=false" in command
        assert "--dataset.private" not in command

    def test_push_to_hub_adds_private_flag(self) -> None:
        """上传口径与既有 demo_v23 一致：私有 Hub。"""
        task = build_collect_plan(["A"])[0]
        command = render_record_command(task, push_to_hub=True)
        assert "--dataset.push_to_hub=true" in command
        assert "--dataset.private=true" in command


class TestSidecarFor:
    def test_each_repo_gets_its_own_file(self) -> None:
        tasks = build_collect_plan(["A", "B"])
        paths = [sidecar_for(t, r"runs\collect\sidecars") for t in tasks]
        assert len(set(paths)) == len(tasks)  # **不能共用一份**（否则 verify 必失败）

    def test_filename_derived_from_repo_basename(self) -> None:
        task = build_collect_plan(["A"])[0]
        assert sidecar_for(task, "sidecars").endswith("demo_vla_01_pick_a.jsonl")

    def test_uses_windows_separators(self) -> None:
        task = build_collect_plan(["A"])[0]
        assert sidecar_for(task, r"runs\week38\collect\sidecars") == (
            r"runs\week38\collect\sidecars\demo_vla_01_pick_a.jsonl"
        )


class TestDatasetRootFor:
    def test_repo_id_becomes_relative_path(self) -> None:
        task = build_collect_plan(["A"])[0]
        assert dataset_root_for(task, r"C:\cache\lerobot") == (
            r"C:\cache\lerobot\wangbangyan\demo_vla_01_pick_a"
        )

    def test_default_matches_lerobot_cache_layout(self) -> None:
        from vla_collect_plan import LEROBOT_CACHE_ROOT

        assert LEROBOT_CACHE_ROOT.endswith(r".cache\huggingface\lerobot")
        assert dataset_root_for(build_collect_plan(["A"])[0]).startswith(LEROBOT_CACHE_ROOT)


class TestRenderVerifyCommand:
    def test_verify_targets_dataset_root_and_sidecar(self) -> None:
        command = render_verify_command(dataset_root="runs/x", sidecar="s.jsonl")
        assert "--verify" in command
        assert "--dataset-root runs/x" in command
        assert "--sidecar s.jsonl" in command

    def test_placeholder_when_sidecar_omitted(self) -> None:
        assert "--sidecar <该 repo 的旁挂文件>" in render_verify_command()


class TestRenderScript:
    def test_every_task_gets_record_and_verify(self) -> None:
        tasks = build_collect_plan(["A", "B"])
        script = render_script(tasks, sidecar_dir="sidecars", runner="run.py")
        assert script.count("run.py --verify") == len(tasks)
        assert script.count("run.py") == len(tasks) * 2

    def test_different_repos_use_different_sidecar(self) -> None:
        """回归：共用一份旁挂会让第 2 个 repo 起的 --verify 失败。"""
        tasks = build_collect_plan(["A", "B"])
        script = render_script(tasks, sidecar_dir="sidecars", runner="run.py")
        assert "sidecars\\demo_vla_01_pick_a.jsonl" in script
        assert "sidecars\\demo_vla_02_pick_a.jsonl" in script

    def test_comments_state_object_and_instruction(self) -> None:
        tasks = build_collect_plan(["螺丝刀"])
        script = render_script(tasks, sidecar_dir="sidecars", runner="run.py")
        assert "物体「螺丝刀」 × pick_a" in script

    def test_push_to_hub_propagates_to_all_commands(self) -> None:
        tasks = build_collect_plan(["A", "B"])
        script = render_script(tasks, sidecar_dir="sidecars", runner="run.py", push_to_hub=True)
        assert script.count("--dataset.push_to_hub=true") == len(tasks)
        assert "--dataset.push_to_hub=false" not in script

    def test_warns_about_existing_dataset_dir(self) -> None:
        """回归：LeRobotDataset.create 用 exist_ok=False，中断重跑会 FileExistsError。"""
        script = render_script(build_collect_plan(["A"]), sidecar_dir="sidecars", runner="run.py")
        assert "FileExistsError" in script
        # resume 是 RecordConfig 的**顶层**字段，不是 dataset 子字段
        assert "--resume=true" in script
        assert "--dataset.resume" not in script

    def test_dataset_root_is_filled_per_repo(self) -> None:
        """--verify 的 --dataset-root 要填实际路径，不留占位符。"""
        tasks = build_collect_plan(["A", "B"])
        script = render_script(tasks, sidecar_dir="sidecars", runner="run.py",
                               cache_root=r"C:\cache\lerobot")
        assert "<该 repo 的数据集根目录>" not in script
        assert r"--dataset-root C:\cache\lerobot\wangbangyan\demo_vla_01_pick_a" in script
        assert r"--dataset-root C:\cache\lerobot\wangbangyan\demo_vla_02_pick_a" in script


class TestRenderTable:
    def test_reports_repo_count_and_episode_total(self) -> None:
        table = render_table(build_collect_plan(["A", "B"], episodes=10))
        assert "6 个 repo" in table
        assert "60 集" in table


class TestWritePlan:
    def test_writes_three_artifacts(self, tmp_path) -> None:
        tasks = build_collect_plan(["螺丝刀", "蓝色方块"], episodes=20)
        tasks_path, combos_path, script_path = write_plan(
            tasks, tmp_path / "collect", sidecar_dir="sidecars", runner="run.py"
        )
        assert tasks_path.name == "collect_tasks.json"
        assert combos_path.name == "collect_combos.json"
        assert script_path.name == "collect_commands.ps1"
        assert tasks_path.exists() and combos_path.exists() and script_path.exists()

    def test_tasks_json_roundtrips_object_and_instruction(self, tmp_path) -> None:
        tasks = build_collect_plan(["螺丝刀"], episodes=3)
        tasks_path, _, _ = write_plan(tasks, tmp_path, sidecar_dir="sidecars", runner="run.py")
        payload = json.loads(tasks_path.read_text(encoding="utf-8"))
        entry = payload["tasks"][0]
        assert entry["object"] == "螺丝刀"
        assert entry["instruction"] == "夹起桌面上的螺丝刀"
        assert entry["repo_id"] == "wangbangyan/demo_vla_01_pick_a"
        assert entry["episodes"] == 3
        assert payload["lerobot_cache_root"].endswith(r".cache\huggingface\lerobot")

    def test_combos_json_matches_generalization_input(self, tmp_path) -> None:
        """产物可直接喂 run_vla_generalization --combos-json。"""
        tasks = build_collect_plan(["A", "B"])
        _, combos_path, _ = write_plan(tasks, tmp_path, sidecar_dir="sidecars", runner="run.py")
        combos = json.loads(combos_path.read_text(encoding="utf-8"))
        assert combos[0] == ["A", "夹起桌面上的A"]
        assert len(combos) == len(tasks)

    def test_script_is_utf8_readable(self, tmp_path) -> None:
        tasks = build_collect_plan(["螺丝刀"])
        _, _, script_path = write_plan(tasks, tmp_path, sidecar_dir="sidecars", runner="run.py")
        assert "物体「螺丝刀」" in script_path.read_text(encoding="utf-8")

    def test_records_actual_instructions_not_all_defaults(self, tmp_path) -> None:
        """回归：清单里记的 instructions 必须是**实际使用**的那几条（曾硬编码成全部默认）。"""
        instructions = select_instructions("pick_a,place_left")
        tasks = build_collect_plan(["A"], instructions)
        tasks_path, _, _ = write_plan(tasks, tmp_path, sidecar_dir="sidecars", runner="run.py",
                                      instructions=instructions)
        payload = json.loads(tasks_path.read_text(encoding="utf-8"))
        assert [item["key"] for item in payload["instructions"]] == ["pick_a", "place_left"]
        assert payload["paraphrase_available"] is False
        assert [task["instruction_key"] for task in payload["tasks"]] == ["pick_a", "place_left"]
