"""相机体检工具测试 — 只测纯逻辑（不碰真实相机）。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_camera_check import (  # noqa: E402
    CAMERA_SLOTS,
    PROBE_INDICES,
    render_probe_row,
    render_slot_verdict,
)


class TestSlotsArePinned:
    def test_slot_mapping_is_single_source_of_truth(self) -> None:
        """映射只在 `camera_slots.py` 定义；本工具必须与运行时共用同一个对象。"""
        from camera_slots import CAMERA_SLOTS as SOURCE

        assert CAMERA_SLOTS is SOURCE

    def test_slot_mapping_matches_verified_enumeration(self) -> None:
        """硬约束：**同一台物理相机 → 同一通道**。

        2026-10-17 逐张核图确认：`0 = 笔记本摄像头 / 1 = 桌面俯拍(fixed) / 2 = 夹爪(handeye)`。
        历史上是 `fixed=2 / handeye=0`（当时枚举不同）—— 改值必须重新核图并写清日期。
        """
        assert CAMERA_SLOTS == {"fixed": 1, "handeye": 2}

    def test_probe_range_covers_known_cameras(self) -> None:
        assert set(CAMERA_SLOTS.values()).issubset(set(PROBE_INDICES))


class TestRenderProbeRow:
    def test_ok_row_shows_backend_resolution_and_path(self) -> None:
        row = render_probe_row(2, "DSHOW", {
            "status": "ok", "width": 640, "height": 480, "path": "out/DSHOW_2.png",
        })
        assert "索引 2" in row and "[DSHOW]" in row
        assert "640x480" in row and "DSHOW_2.png" in row

    def test_failure_row_shows_reason(self) -> None:
        row = render_probe_row(0, "MSMF", {
            "status": "打开成功但读帧失败", "detail": "设备未连接",
        })
        assert "索引 0" in row and "[MSMF]" in row and "读帧失败" in row and "设备未连接" in row


class TestRenderSlotVerdict:
    def test_flags_unusable_slot_as_blocker(self) -> None:
        lines = "\n".join(render_slot_verdict({"fixed": 2, "handeye": 0}, usable=[1, 2]))
        assert "fixed" in lines and "✅ 可用" in lines
        assert "handeye" in lines and "❌ 不可用" in lines
        assert "阻塞点" in lines

    def test_all_usable_has_no_blocker(self) -> None:
        lines = "\n".join(render_slot_verdict({"fixed": 2, "handeye": 0}, usable=[0, 1, 2]))
        assert "❌" not in lines
