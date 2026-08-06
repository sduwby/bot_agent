"""测试 FailureClassifier 的失败分类功能。"""

import pytest
from failure_classifier import FailureCategory, FailureClassifier, FailureInfo


class TestClassifyFailure:
    """测试 classify_failure 方法。"""

    def test_pick_not_found_chinese(self):
        """pick 失败，原因包含"未发现"→ OBJECT_NOT_FOUND。"""
        result = FailureClassifier.classify_failure(
            skill_name="pick",
            params={"object_type": "CreditCard"},
            failure_reason="扫描后未发现可抓取的 CreditCard",
        )
        assert result.category == FailureCategory.OBJECT_NOT_FOUND
        assert result.skill_name == "pick"

    def test_pick_not_found_english(self):
        """pick 失败，原因包含"not found"→ OBJECT_NOT_FOUND。"""
        result = FailureClassifier.classify_failure(
            skill_name="pick",
            params={"object_type": "Key"},
            failure_reason="Object not found in scene",
        )
        assert result.category == FailureCategory.OBJECT_NOT_FOUND

    def test_pick_grasp_failed(self):
        """pick 失败但物体可见 → GRASP_FAILED。"""
        result = FailureClassifier.classify_failure(
            skill_name="pick",
            params={"object_type": "Apple"},
            failure_reason="抓取后未在 inventoryObjects 中找到目标",
        )
        assert result.category == FailureCategory.GRASP_FAILED

    def test_place_failed(self):
        """place 失败 → PLACEMENT_FAILED。"""
        result = FailureClassifier.classify_failure(
            skill_name="place",
            params={"receptacle_type": "Drawer"},
            failure_reason="所有指定类型容器都无法完成经过验证的放置",
        )
        assert result.category == FailureCategory.PLACEMENT_FAILED

    def test_place_failed_short(self):
        """place 失败（简短原因）→ PLACEMENT_FAILED。"""
        result = FailureClassifier.classify_failure(
            skill_name="place",
            params={"receptacle_type": "Cabinet"},
            failure_reason="放置失败",
        )
        assert result.category == FailureCategory.PLACEMENT_FAILED

    def test_place_receptacle_not_found(self):
        result = FailureClassifier.classify_failure(
            skill_name="place",
            params={"receptacle_type": "Drawer"},
            failure_reason="扫描后未发现容器 Drawer",
        )
        assert result.category == FailureCategory.OBJECT_NOT_FOUND

    def test_move_to_blocked(self):
        """move_to 失败 → NAVIGATION_BLOCKED。"""
        result = FailureClassifier.classify_failure(
            skill_name="move_to",
            params={"direction": "forward"},
            failure_reason="无法移动：前方有障碍物",
        )
        assert result.category == FailureCategory.NAVIGATION_BLOCKED

    def test_scan_failed_unknown(self):
        """scan 失败 → UNKNOWN。"""
        result = FailureClassifier.classify_failure(
            skill_name="scan",
            params={},
            failure_reason="未知错误",
        )
        assert result.category == FailureCategory.UNKNOWN

    def test_unknown_skill(self):
        """未知技能 → UNKNOWN。"""
        result = FailureClassifier.classify_failure(
            skill_name="teleport",
            params={"x": 1.0},
            failure_reason="不支持此技能",
        )
        assert result.category == FailureCategory.UNKNOWN

    def test_original_reason_preserved(self):
        """原始失败原因应完整保留。"""
        reason = "扫描后未发现可抓取的 CreditCard"
        result = FailureClassifier.classify_failure(
            skill_name="pick",
            params={"object_type": "CreditCard"},
            failure_reason=reason,
        )
        assert result.original_reason == reason

    def test_params_preserved(self):
        """参数应完整保留。"""
        params = {"object_type": "Book", "extra": True}
        result = FailureClassifier.classify_failure(
            skill_name="pick",
            params=params,
            failure_reason="扫描后未发现可抓取的 Book",
        )
        assert result.params == params


class TestFailureCategory:
    """测试 FailureCategory 枚举。"""

    def test_all_categories(self):
        """枚举应包含 5 个类别。"""
        categories = list(FailureCategory)
        assert len(categories) == 5
        category_names = {c.value for c in categories}
        assert "object_not_found" in category_names
        assert "grasp_failed" in category_names
        assert "placement_failed" in category_names
        assert "navigation_blocked" in category_names
        assert "unknown" in category_names

    def test_str_enum(self):
        """枚举值应为字符串。"""
        assert isinstance(FailureCategory.OBJECT_NOT_FOUND, str)
