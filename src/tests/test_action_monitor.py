"""测试 ActionMonitor 的循环检测功能。"""

import pytest
from action_monitor import ActionMonitor, LoopDetection


class TestDetectLoop:
    """测试 detect_loop 方法。"""

    def test_no_loop_different_actions(self):
        """不同动作不应检测为循环。"""
        history = [
            {"skill": "scan", "parameters": {}},
            {"skill": "pick", "parameters": {"object_type": "Apple"}},
            {"skill": "place", "parameters": {"receptacle_type": "Drawer"}},
        ]
        result = ActionMonitor.detect_loop(history)
        assert result.is_loop is False
        assert result.loop_type is None

    def test_identical_repeat_three_same(self):
        """连续 3 次相同动作应检测为 IDENTICAL_REPEAT。"""
        history = [
            {"skill": "pick", "parameters": {"object_type": "Apple"}},
            {"skill": "pick", "parameters": {"object_type": "Apple"}},
            {"skill": "pick", "parameters": {"object_type": "Apple"}},
        ]
        result = ActionMonitor.detect_loop(history)
        assert result.is_loop is True
        assert result.loop_type == "IDENTICAL_REPEAT"
        assert "连续 3 步" in result.details

    def test_identical_repeat_with_different_params(self):
        """相同技能但不同参数不应检测为循环。"""
        history = [
            {"skill": "pick", "parameters": {"object_type": "Apple"}},
            {"skill": "pick", "parameters": {"object_type": "Key"}},
            {"skill": "pick", "parameters": {"object_type": "Book"}},
        ]
        result = ActionMonitor.detect_loop(history)
        assert result.is_loop is False

    def test_oscillation_two_actions(self):
        """两个动作交替应检测为 OSCILLATION。"""
        history = [
            {"skill": "pick", "parameters": {"object_type": "Apple"}},
            {"skill": "place", "parameters": {"receptacle_type": "Drawer"}},
            {"skill": "pick", "parameters": {"object_type": "Apple"}},
            {"skill": "place", "parameters": {"receptacle_type": "Drawer"}},
        ]
        result = ActionMonitor.detect_loop(history, window=4)
        assert result.is_loop is True
        assert result.loop_type == "OSCILLATION"

    def test_oscillation_window_3(self):
        """window=3 时交替循环检测（A→B→A）。"""
        history = [
            {"skill": "pick", "parameters": {"object_type": "Apple"}},
            {"skill": "place", "parameters": {"receptacle_type": "Drawer"}},
            {"skill": "pick", "parameters": {"object_type": "Apple"}},
        ]
        result = ActionMonitor.detect_loop(history, window=3)
        assert result.is_loop is True
        assert result.loop_type == "OSCILLATION"

    def test_history_shorter_than_window(self):
        """历史少于窗口大小不应报错。"""
        history = [
            {"skill": "pick", "parameters": {"object_type": "Apple"}},
        ]
        result = ActionMonitor.detect_loop(history, window=3)
        assert result.is_loop is False

    def test_empty_history(self):
        """空历史不应报错。"""
        result = ActionMonitor.detect_loop([], window=3)
        assert result.is_loop is False

    def test_scan_repeat(self):
        """连续 scan 也应检测为循环。"""
        history = [
            {"skill": "scan", "parameters": {}},
            {"skill": "scan", "parameters": {}},
            {"skill": "scan", "parameters": {}},
        ]
        result = ActionMonitor.detect_loop(history)
        assert result.is_loop is True
        assert result.loop_type == "IDENTICAL_REPEAT"

    def test_custom_window_size(self):
        """自定义窗口大小应生效。"""
        history = [
            {"skill": "pick", "parameters": {"object_type": "A"}},
            {"skill": "pick", "parameters": {"object_type": "A"}},
            {"skill": "pick", "parameters": {"object_type": "A"}},
            {"skill": "pick", "parameters": {"object_type": "A"}},
        ]
        # window=4 应该检测到
        result = ActionMonitor.detect_loop(history, window=4)
        assert result.is_loop is True
        assert result.window_size == 4

    def test_not_alternating_three_unique(self):
        """三个不同动作不应误判为振荡。"""
        history = [
            {"skill": "scan", "parameters": {}},
            {"skill": "pick", "parameters": {"object_type": "A"}},
            {"skill": "place", "parameters": {"receptacle_type": "B"}},
        ]
        result = ActionMonitor.detect_loop(history)
        assert result.is_loop is False


class TestLoopDetection:
    """测试 LoopDetection 数据类。"""

    def test_dataclass_fields(self):
        """数据类应有正确的字段。"""
        detection = LoopDetection(
            is_loop=True,
            loop_type="IDENTICAL_REPEAT",
            window_size=3,
            details="test details",
        )
        assert detection.is_loop is True
        assert detection.loop_type == "IDENTICAL_REPEAT"
        assert detection.window_size == 3
        assert detection.details == "test details"

    def test_no_loop_default(self):
        """无循环时 loop_type 应为 None。"""
        detection = LoopDetection(
            is_loop=False,
            loop_type=None,
            window_size=3,
            details="no loop",
        )
        assert detection.loop_type is None