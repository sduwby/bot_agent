"""测试 Replanner 的恢复策略生成。"""

import pytest
from action_monitor import LoopDetection
from failure_classifier import FailureCategory, FailureInfo
from replanner import Replanner, RecoveryAction


def _make_failure(category, skill="pick", params=None, reason="test failure"):
    return FailureInfo(
        category=category,
        skill_name=skill,
        params=params or {},
        original_reason=reason,
    )


def _make_loop(loop_type="IDENTICAL_REPEAT"):
    return LoopDetection(
        is_loop=True,
        loop_type=loop_type,
        window_size=3,
        details="test loop",
    )


class TestPlanRecovery:
    """测试 plan_recovery 方法。"""

    def test_object_not_found_triggers_search(self):
        """OBJECT_NOT_FOUND → search 动作。"""
        rp = Replanner()
        failure = _make_failure(FailureCategory.OBJECT_NOT_FOUND)
        result = rp.plan_recovery(failure_info=failure)
        assert result.action == "search"
        assert result.is_final is False

    def test_grasp_failed_triggers_retry(self):
        """GRASP_FAILED → retry 动作。"""
        rp = Replanner()
        failure = _make_failure(FailureCategory.GRASP_FAILED, params={"object_type": "Apple"})
        result = rp.plan_recovery(failure_info=failure)
        assert result.action == "retry"
        assert result.skill_name == "pick"
        assert result.params == {"object_type": "Apple"}

    def test_placement_failed_triggers_reposition_search(self):
        """PLACEMENT_FAILED → 换位搜索，而不是原地重试。"""
        rp = Replanner()
        failure = _make_failure(
            FailureCategory.PLACEMENT_FAILED,
            skill="place",
            params={"receptacle_type": "Drawer"},
        )
        result = rp.plan_recovery(failure_info=failure)
        assert result.action == "search"
        assert "新视点" in result.reason

    def test_navigation_blocked_changes_direction(self):
        """NAVIGATION_BLOCKED → 换方向。"""
        rp = Replanner()
        failure = _make_failure(
            FailureCategory.NAVIGATION_BLOCKED,
            skill="move_to",
            params={"direction": "forward"},
        )
        result = rp.plan_recovery(failure_info=failure)
        assert result.action == "retry"
        assert result.params == {"direction": "backward"}

    def test_loop_detected_triggers_scan(self):
        """LOOP_DETECTED → scan 动作。"""
        rp = Replanner()
        loop = _make_loop()
        result = rp.plan_recovery(loop_detection=loop)
        assert result.action == "scan"
        assert result.skill_name == "scan"

    def test_repeated_scan_loop_moves_to_new_view(self):
        rp = Replanner()
        loop = LoopDetection(
            is_loop=True,
            loop_type="IDENTICAL_REPEAT",
            window_size=3,
            details="连续 3 步重复执行 scan{}",
        )

        result = rp.plan_recovery(loop_detection=loop)

        assert result.action == "retry"
        assert result.skill_name == "move_to"

    def test_unknown_category_goes_to_llm(self):
        """UNKNOWN → llm_decision。"""
        rp = Replanner()
        failure = _make_failure(FailureCategory.UNKNOWN)
        result = rp.plan_recovery(failure_info=failure)
        assert result.action == "llm_decision"

    def test_no_failure_info_goes_to_llm(self):
        """无失败信息 → llm_decision。"""
        rp = Replanner()
        result = rp.plan_recovery()
        assert result.action == "llm_decision"


class TestRetryLimits:
    """测试重试次数限制。"""

    def test_max_retries_exceeded_aborts(self):
        """超过最大重试次数 → abort。"""
        rp = Replanner(max_retries=2)
        failure = _make_failure(FailureCategory.GRASP_FAILED, params={"object_type": "A"})

        # 第 1 次 → retry
        r1 = rp.plan_recovery(failure_info=failure)
        assert r1.action == "retry"

        # 第 2 次 → retry
        r2 = rp.plan_recovery(failure_info=failure)
        assert r2.action == "retry"

        # 第 3 次 → abort
        r3 = rp.plan_recovery(failure_info=failure)
        assert r3.action == "abort"
        assert r3.is_final is True

    def test_retry_count_increments(self):
        """重试计数应正确递增。"""
        rp = Replanner(max_retries=3)
        failure = _make_failure(FailureCategory.GRASP_FAILED, params={})

        rp.plan_recovery(failure_info=failure)
        assert rp.should_retry("pick") is True
        assert rp._retry_counts["pick"] == 1

    def test_reset_clears_counts(self):
        """reset 应清零计数。"""
        rp = Replanner()
        rp._retry_counts["pick"] = 5
        rp.reset()
        assert rp._retry_counts == {}

    def test_placement_search_does_not_consume_grasp_retry_count(self):
        """放置换位搜索不应影响抓取重试计数。"""
        rp = Replanner(max_retries=1)
        pick_failure = _make_failure(FailureCategory.GRASP_FAILED, params={})
        place_failure = _make_failure(FailureCategory.PLACEMENT_FAILED, skill="place", params={})

        r1 = rp.plan_recovery(failure_info=pick_failure)
        assert r1.action == "retry"

        r2 = rp.plan_recovery(failure_info=place_failure)
        assert r2.action == "search"
        assert rp._retry_counts == {"pick": 1}


class TestRecoveryAction:
    """测试 RecoveryAction 数据类。"""

    def test_fields(self):
        """数据类字段正确。"""
        action = RecoveryAction(
            action="retry",
            reason="test",
            skill_name="pick",
            params={"x": 1},
            is_final=False,
        )
        assert action.action == "retry"
        assert action.skill_name == "pick"
        assert action.params == {"x": 1}
        assert action.is_final is False
