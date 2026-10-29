"""真机就绪度冒烟测试的纯逻辑测试。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_real_smoke import all_passed, render_checks  # noqa: E402


class TestRenderChecks:
    def test_marks_pass_and_fail(self) -> None:
        text = render_checks([("总线", True, "6/6"), ("归位", False, "超时")])
        assert "[PASS] 总线" in text
        assert "[FAIL] 归位" in text
        assert "1/2 项通过" in text

    def test_counts_all_passed(self) -> None:
        text = render_checks([("A", True, ""), ("B", True, "")])
        assert "2/2 项通过" in text


class TestAllPassed:
    def test_true_only_when_every_check_passes(self) -> None:
        assert all_passed([("a", True, ""), ("b", True, "")]) is True
        assert all_passed([("a", True, ""), ("b", False, "")]) is False

    def test_empty_is_not_success(self) -> None:
        """一条检查都没跑出来，不能算通过（避免"静默成功"）。"""
        assert all_passed([]) is False
