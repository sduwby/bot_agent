"""skill_registry 的单元测试 — 纯逻辑，无需 AI2-THOR。"""

from __future__ import annotations

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skill_registry import (
    get_skills,
    get_skill,
    get_skills_description,
    validate_parameters,
)


class TestGetSkills:
    def test_returns_five_skills(self) -> None:
        skills = get_skills()
        assert len(skills) == 5

    def test_skill_names(self) -> None:
        names = [s["name"] for s in get_skills()]
        assert set(names) == {"scan", "pick", "place", "move_to", "done"}

    def test_each_skill_has_required_fields(self) -> None:
        for skill in get_skills():
            assert "name" in skill
            assert "description" in skill
            assert "parameters" in skill
            assert "type" in skill["parameters"]


class TestGetSkill:
    def test_find_existing_skill(self) -> None:
        skill = get_skill("pick")
        assert skill is not None
        assert skill["name"] == "pick"

    def test_unknown_skill_returns_none(self) -> None:
        assert get_skill("unknown") is None

    def test_each_skill_name_resolvable(self) -> None:
        for skill in get_skills():
            assert get_skill(skill["name"]) is not None


class TestGetSkillsDescription:
    def test_contains_all_skill_names(self) -> None:
        desc = get_skills_description()
        for skill in get_skills():
            assert skill["name"] in desc

    def test_contains_parameter_description(self) -> None:
        desc = get_skills_description()
        assert "object_type" in desc
        assert "receptacle_type" in desc
        assert "direction" in desc

    def test_is_readable_text(self) -> None:
        desc = get_skills_description()
        assert isinstance(desc, str)
        assert len(desc) > 100


class TestValidateParameters:
    """参数校验的完整覆盖。"""

    def test_scan_no_params_passes(self) -> None:
        assert validate_parameters("scan", {}) is None

    def test_scan_none_params_passes(self) -> None:
        assert validate_parameters("scan", None) is None

    def test_done_no_params_passes(self) -> None:
        assert validate_parameters("done", {}) is None

    def test_pick_valid_passes(self) -> None:
        assert validate_parameters("pick", {"object_type": "CreditCard"}) is None

    def test_pick_missing_object_type_fails(self) -> None:
        err = validate_parameters("pick", {})
        assert err is not None
        assert "object_type" in err

    def test_pick_none_params_fails(self) -> None:
        err = validate_parameters("pick", None)
        assert err is not None
        assert "object_type" in err

    def test_place_valid_passes(self) -> None:
        assert validate_parameters("place", {"receptacle_type": "Drawer"}) is None

    def test_place_missing_receptacle_type_fails(self) -> None:
        err = validate_parameters("place", {})
        assert err is not None
        assert "receptacle_type" in err

    def test_move_to_valid_direction_passes(self) -> None:
        for direction in ("forward", "backward", "left", "right"):
            assert validate_parameters("move_to", {"direction": direction}) is None

    def test_move_to_reachable_requires_numeric_target(self) -> None:
        valid = {
            "direction": "reachable",
            "target_position": {"x": 1.0, "y": 0.9, "z": 2.0},
        }
        assert validate_parameters("move_to", valid) is None

        err = validate_parameters(
            "move_to", {"direction": "reachable"}
        )
        assert err is not None
        assert "target_position" in err

    def test_move_to_invalid_enum_fails(self) -> None:
        err = validate_parameters("move_to", {"direction": "up"})
        assert err is not None
        assert "up" in err

    def test_move_to_missing_direction_fails(self) -> None:
        err = validate_parameters("move_to", {})
        assert err is not None
        assert "direction" in err

    def test_unknown_skill_fails(self) -> None:
        err = validate_parameters("unknown_skill", {})
        assert err is not None
        assert "unknown_skill" in err.lower() or "unknown" in err.lower()

    def test_extra_params_ignored(self) -> None:
        """额外的参数不应报错。"""
        assert validate_parameters("pick", {"object_type": "Apple", "extra": 1}) is None
