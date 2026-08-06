"""AgentExecutor 主循环集成测试（使用轻量 fake controller）。"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from executor import AgentExecutor
from task_skills import SkillResult


def _metadata(
    *,
    inventory: list[dict[str, Any]] | None = None,
    objects: list[dict[str, Any]] | None = None,
    position: dict[str, float] | None = None,
    rotation_y: float = 0.0,
) -> dict[str, Any]:
    return {
        "sceneName": "FloorPlan1",
        "agent": {
            "position": position or {"x": 0.0, "y": 0.9, "z": 0.0},
            "rotation": {"y": rotation_y},
        },
        "inventoryObjects": inventory or [],
        "objects": objects or [],
    }


class FakeController:
    """只提供 AgentExecutor 构建 Observation 所需的 latest event。"""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.last_event = SimpleNamespace(metadata=metadata)


class SequenceBackend:
    """按顺序返回决策并保存实际收到的 prompt。"""

    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = iter(decisions)
        self.prompts: list[str] = []

    def chat(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return json.dumps(next(self.decisions), ensure_ascii=False)


def _decision(skill: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "thought": f"执行 {skill}",
        "skill": skill,
        "parameters": parameters or {},
    }


def _make_agent(
    tmp_path: Path,
    controller: FakeController,
    backend: SequenceBackend,
    *,
    max_steps: int = 12,
    max_search_steps: int = 6,
) -> AgentExecutor:
    agent = AgentExecutor(
        controller=controller,
        run_dir=tmp_path,
        trajectory_file=io.StringIO(),
        records=[],
        backend=backend,
        max_steps=max_steps,
        max_search_steps=max_search_steps,
    )
    agent.skills.record_initial_observation = lambda: None
    return agent


def test_done_without_target_state_is_failure_and_persists_l0(tmp_path: Path) -> None:
    controller = FakeController(_metadata())
    backend = SequenceBackend([_decision("done") for _ in range(3)])
    agent = _make_agent(tmp_path, controller, backend, max_steps=3)

    summary = agent.run("把信用卡放进抽屉")

    assert summary["task_success"] is False
    assert summary["final_verification"]["success"] is False
    assert agent.store.conversations_file.exists()
    assert len(agent.store.conversations_file.read_text().splitlines()) == 4


def test_place_uses_pre_action_object_id_for_verification(tmp_path: Path) -> None:
    card = {
        "objectId": "card_0",
        "objectType": "CreditCard",
        "visible": True,
        "pickupable": True,
        "distance": 1.0,
        "parentReceptacles": [],
    }
    drawer = {
        "objectId": "drawer_0",
        "objectType": "Drawer",
        "visible": True,
        "receptacle": True,
        "distance": 1.0,
    }
    controller = FakeController(
        _metadata(
            inventory=[{"objectId": "card_0", "objectType": "CreditCard"}],
            objects=[card, drawer],
        )
    )
    backend = SequenceBackend([
        _decision("place", {"receptacle_type": "Drawer"}),
        _decision("done"),
    ])
    agent = _make_agent(tmp_path, controller, backend)
    verify_calls: list[str] = []
    original_verify = agent.verifier.verify_place

    def execute(skill: str, params: dict[str, Any]) -> SkillResult:
        controller.last_event.metadata["inventoryObjects"] = []
        card["parentReceptacles"] = ["drawer_0"]
        return SkillResult(True, {})

    def verify(metadata: dict[str, Any], object_id: str, receptacle_type: str):
        verify_calls.append(object_id)
        return original_verify(metadata, object_id, receptacle_type)

    agent._execute_skill = execute
    agent.verifier.verify_place = verify

    summary = agent.run("把信用卡放进抽屉")

    assert summary["task_success"] is True
    assert verify_calls == ["card_0"]


def test_retry_refreshes_observation_before_next_planner_call(tmp_path: Path) -> None:
    apple = {
        "objectId": "apple_0",
        "objectType": "Apple",
        "visible": True,
        "pickupable": True,
        "distance": 1.0,
        "parentReceptacles": [],
    }
    drawer = {
        "objectId": "drawer_0",
        "objectType": "Drawer",
        "visible": True,
        "receptacle": True,
        "distance": 1.0,
    }
    controller = FakeController(_metadata(objects=[apple, drawer]))
    backend = SequenceBackend([
        _decision("pick", {"object_type": "Apple"}),
        _decision("place", {"receptacle_type": "Drawer"}),
        _decision("done"),
    ])
    agent = _make_agent(tmp_path, controller, backend)
    pick_calls = 0

    def execute(skill: str, params: dict[str, Any]) -> SkillResult:
        nonlocal pick_calls
        if skill == "pick":
            pick_calls += 1
            if pick_calls == 1:
                return SkillResult(False, {}, "抓取失败")
            controller.last_event.metadata["inventoryObjects"] = [
                {"objectId": "apple_0", "objectType": "Apple"}
            ]
            return SkillResult(True, {"target": {"object_id": "apple_0"}})
        controller.last_event.metadata["inventoryObjects"] = []
        apple["parentReceptacles"] = ["drawer_0"]
        return SkillResult(True, {})

    agent._execute_skill = execute

    summary = agent.run("把苹果放进抽屉")

    assert summary["task_success"] is True
    assert pick_calls == 2
    assert "当前持有: Apple" in backend.prompts[1]
    assert summary["planner_calls"] == 3
    assert summary["llm_calls"] == 0


def test_backend_marked_as_llm_increments_both_call_counters(tmp_path: Path) -> None:
    controller = FakeController(_metadata())
    backend = SequenceBackend([_decision("done")])
    backend.counts_as_llm = True
    agent = _make_agent(tmp_path, controller, backend, max_steps=1)

    summary = agent.run("把信用卡放进抽屉")

    assert summary["planner_calls"] == 1
    assert summary["llm_calls"] == 1


def test_repeated_scan_loop_recovers_with_movement(tmp_path: Path) -> None:
    controller = FakeController(_metadata())
    backend = SequenceBackend([
        _decision("scan"),
        _decision("scan"),
        _decision("scan"),
        _decision("done"),
    ])
    agent = _make_agent(tmp_path, controller, backend, max_steps=5)
    agent._execute_skill = lambda skill, params: SkillResult(True, {})

    summary = agent.run("把信用卡放进抽屉")

    assert summary["task_success"] is False
    assert any(
        entry["skill"] == "move_to" for entry in summary["action_history"]
    )


def test_search_moves_until_target_visible_then_completes(tmp_path: Path) -> None:
    apple = {
        "objectId": "apple_0",
        "objectType": "Apple",
        "visible": False,
        "pickupable": True,
        "distance": 2.0,
        "parentReceptacles": [],
    }
    drawer = {
        "objectId": "drawer_0",
        "objectType": "Drawer",
        "visible": True,
        "receptacle": True,
        "distance": 1.0,
    }
    controller = FakeController(_metadata(objects=[apple, drawer]))
    backend = SequenceBackend([
        _decision("pick", {"object_type": "Apple"}),
        _decision("pick", {"object_type": "Apple"}),
        _decision("place", {"receptacle_type": "Drawer"}),
        _decision("done"),
    ])
    agent = _make_agent(tmp_path, controller, backend)
    agent._inspect_visible_search_containers = lambda: None
    move_calls = 0

    def execute(skill: str, params: dict[str, Any]) -> SkillResult:
        nonlocal move_calls
        if skill == "move_to":
            move_calls += 1
            controller.last_event.metadata["agent"]["position"]["z"] += 0.25
            if move_calls == 2:
                apple["visible"] = True
                apple["distance"] = 1.0
            return SkillResult(True, {})
        if skill == "pick":
            if not apple["visible"]:
                return SkillResult(
                    False, {}, "扫描后未发现可抓取的 Apple"
                )
            controller.last_event.metadata["inventoryObjects"] = [
                {"objectId": "apple_0", "objectType": "Apple"}
            ]
            return SkillResult(True, {"target": {"object_id": "apple_0"}})
        if skill == "place":
            controller.last_event.metadata["inventoryObjects"] = []
            apple["parentReceptacles"] = ["drawer_0"]
            return SkillResult(True, {})
        return SkillResult(True, {})

    agent._execute_skill = execute

    summary = agent.run("把苹果放进抽屉")

    assert summary["task_success"] is True
    assert move_calls == 2
    assert agent.store.conversations_file.exists()
    assert agent.store.memories_file.exists()


def test_placement_failure_forces_move_even_when_receptacle_visible(
    tmp_path: Path,
) -> None:
    apple = {
        "objectId": "apple_0",
        "objectType": "Apple",
        "visible": True,
        "pickupable": True,
        "distance": 1.0,
        "parentReceptacles": [],
    }
    drawer = {
        "objectId": "drawer_0",
        "objectType": "Drawer",
        "visible": True,
        "receptacle": True,
        "distance": 1.0,
    }
    controller = FakeController(
        _metadata(
            inventory=[{"objectId": "apple_0", "objectType": "Apple"}],
            objects=[apple, drawer],
        )
    )
    backend = SequenceBackend([
        _decision("place", {"receptacle_type": "Drawer"}),
    ])
    agent = _make_agent(tmp_path, controller, backend, max_steps=2)
    agent.search.set_reachable_positions([
        {"x": 2.0, "y": 0.9, "z": 2.0},
    ])
    agent._reachable_positions_loaded = True

    def execute(skill: str, params: dict[str, Any]) -> SkillResult:
        if skill == "place":
            return SkillResult(False, {}, "所有指定类型容器都无法完成经过验证的放置")
        return SkillResult(True, {})

    agent._execute_skill = execute

    summary = agent.run("把苹果放进抽屉")

    assert summary["action_history"][0]["skill"] == "place"
    assert summary["action_history"][1]["parameters"] == {
        "direction": "reachable",
        "target_position": {"x": 2.0, "y": 0.9, "z": 2.0},
    }
    assert len(backend.prompts) == 1


def test_reachable_position_query_is_cached(tmp_path: Path) -> None:
    controller = FakeController(_metadata())
    controller.step = lambda **kwargs: None
    agent = _make_agent(
        tmp_path,
        controller,
        SequenceBackend([_decision("done")]),
    )
    calls: list[dict[str, Any]] = []

    def act(action: dict[str, Any]):
        calls.append(action)
        return SimpleNamespace(metadata={
            "lastActionSuccess": True,
            "actionReturn": [{"x": 1.0, "y": 0.9, "z": 2.0}],
        })

    agent.skills.act = act

    agent._ensure_reachable_positions()
    agent._ensure_reachable_positions()

    assert calls == [{"action": "GetReachablePositions"}]
    assert agent.search.is_reachable_position(
        {"x": 1.0, "y": 0.9, "z": 2.0}
    )


def test_reachable_move_rejects_untrusted_target(tmp_path: Path) -> None:
    controller = FakeController(_metadata())
    agent = _make_agent(
        tmp_path,
        controller,
        SequenceBackend([_decision("done")]),
    )
    agent.search.set_reachable_positions([
        {"x": 1.0, "y": 0.9, "z": 2.0},
    ])

    result = agent._do_reachable_move(
        {"x": 9.0, "y": 0.9, "z": 9.0}
    )

    assert result.success is False
    assert "可达点集合" in (result.failure_reason or "")


def test_object_search_opens_visible_closed_container(tmp_path: Path) -> None:
    cabinet = {
        "objectId": "cabinet_0",
        "objectType": "Cabinet",
        "visible": False,
        "receptacle": True,
        "openable": True,
        "isOpen": False,
        "distance": 1.0,
    }
    controller = FakeController(_metadata(objects=[cabinet]))
    controller.step = lambda **kwargs: None
    agent = _make_agent(
        tmp_path,
        controller,
        SequenceBackend([_decision("done")]),
    )
    agent._task_spec = SimpleNamespace(
        object_type="Cup", receptacle_type="CounterTop"
    )
    agent._search_target_type = "Cup"
    agent.skills.scan_receptacles = lambda excluded: [{
        "object_id": "cabinet_0",
        "object_type": "Cabinet",
        "distance": 1.0,
        "rotation_y": 90.0,
        "openable": True,
    }]
    agent._scan_search_target = lambda: True
    agent.skills.rotate_to = lambda rotation: cabinet.update(visible=True)
    actions: list[dict[str, Any]] = []

    def act(action: dict[str, Any]):
        actions.append(action)
        cabinet["isOpen"] = True
        return SimpleNamespace(metadata={
            **controller.last_event.metadata,
            "lastActionSuccess": True,
        })

    agent.skills.act = act

    agent._inspect_visible_search_containers()

    assert actions == [{"action": "OpenObject", "objectId": "cabinet_0"}]
    assert cabinet["isOpen"] is True


def test_object_search_closes_container_without_target(tmp_path: Path) -> None:
    cabinet = {
        "objectId": "cabinet_0",
        "objectType": "Cabinet",
        "visible": True,
        "receptacle": True,
        "openable": True,
        "isOpen": False,
        "distance": 1.0,
    }
    controller = FakeController(_metadata(objects=[cabinet]))
    controller.step = lambda **kwargs: None
    agent = _make_agent(
        tmp_path,
        controller,
        SequenceBackend([_decision("done")]),
    )
    agent._task_spec = SimpleNamespace(
        object_type="Apple", receptacle_type="CounterTop"
    )
    agent._search_target_type = "Apple"
    agent.skills.scan_receptacles = lambda excluded: [{
        "object_id": "cabinet_0",
        "object_type": "Cabinet",
        "distance": 1.0,
        "rotation_y": 90.0,
        "openable": True,
    }]
    agent._scan_search_target = lambda: False
    agent.skills.rotate_to = lambda rotation: None
    actions: list[str] = []

    def act(action: dict[str, Any]):
        actions.append(action["action"])
        cabinet["isOpen"] = action["action"] == "OpenObject"
        return SimpleNamespace(metadata={
            **controller.last_event.metadata,
            "lastActionSuccess": True,
        })

    agent.skills.act = act

    agent._inspect_visible_search_containers()
    agent._inspect_visible_search_containers()

    assert actions == ["OpenObject", "CloseObject"]
    assert cabinet["isOpen"] is False


def test_search_target_scan_looks_down_and_restores_camera(
    tmp_path: Path,
) -> None:
    controller = FakeController(_metadata())
    agent = _make_agent(
        tmp_path,
        controller,
        SequenceBackend([_decision("done")]),
    )
    agent._search_target_type = "Apple"
    scans = iter([
        [],
        [{"object_id": "apple_0", "object_type": "Apple"}],
    ])
    agent.skills.scan_pickupable = lambda: next(scans)
    actions: list[str] = []

    def act(action: dict[str, Any]):
        actions.append(action["action"])
        return SimpleNamespace(metadata={"lastActionSuccess": True})

    agent.skills.act = act

    assert agent._scan_search_target() is True
    assert actions == ["LookDown", "LookUp"]
    assert agent._search_target_horizon_delta == 30


def test_pick_reuses_discovered_horizon_and_restores_camera(
    tmp_path: Path,
) -> None:
    controller = FakeController(_metadata())
    agent = _make_agent(
        tmp_path,
        controller,
        SequenceBackend([_decision("done")]),
    )
    agent._search_target_horizon_delta = 30
    actions: list[str] = []

    def act(action: dict[str, Any]):
        actions.append(action["action"])
        return SimpleNamespace(metadata={"lastActionSuccess": True})

    agent.skills.act = act
    agent.skills.pick = lambda object_type: SkillResult(
        True, {"target": {"object_id": "apple_0"}}
    )

    result = agent._do_pick("Apple")

    assert result.success is True
    assert actions == ["LookDown", "LookUp"]
    assert agent._search_target_horizon_delta == 0


def test_receptacle_search_does_not_open_unrelated_containers(
    tmp_path: Path,
) -> None:
    controller = FakeController(_metadata())
    agent = _make_agent(
        tmp_path,
        controller,
        SequenceBackend([_decision("done")]),
    )
    agent._task_spec = SimpleNamespace(
        object_type="Cup", receptacle_type="Drawer"
    )
    agent._search_target_type = "Drawer"
    calls: list[set[str]] = []
    agent.skills.scan_receptacles = lambda excluded: calls.append(excluded) or []

    agent._inspect_visible_search_containers()

    assert calls == []


def test_receptacle_search_revisits_known_viewpoint(tmp_path: Path) -> None:
    controller = FakeController(_metadata(
        position={"x": 0.0, "y": 0.9, "z": 0.0}
    ))
    agent = _make_agent(
        tmp_path,
        controller,
        SequenceBackend([_decision("done")]),
    )
    known = {"x": 1.0, "y": 0.9, "z": 2.0}
    agent._search_target_type = "Microwave"
    agent._known_receptacle_viewpoints = {"microwave": [known]}

    selected = agent._next_known_receptacle_viewpoint(
        {"x": 0.0, "y": 0.9, "z": 0.0}
    )

    assert selected == known
    assert agent._next_known_receptacle_viewpoint(
        {"x": 0.0, "y": 0.9, "z": 0.0}
    ) is None
