"""录制流水线旁挂挂钩测试 — 纯逻辑，用假机器人 / 假录制模块，无 lerobot、无硬件。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from load_logging import (  # noqa: E402
    CAMERA_READ_ATTEMPTS,
    LOAD_REGISTER,
    LoadLoggingRobot,
    patch_record_pipeline,
)
from load_sidecar import LoadSidecarWriter, read_load_sidecar  # noqa: E402

GRIP_RAW_HELD = 0x400 | 206


class _FakeBus:
    """假总线：默认返回可解码的负载读数，可设为抛异常以覆盖读失败路径。"""

    def __init__(self, *, fail: bool = False, grip_raw: int = GRIP_RAW_HELD) -> None:
        self.fail = fail
        self.grip_raw = grip_raw
        self.calls = 0

    def sync_read(self, register, normalize=False):  # noqa: ANN001
        self.calls += 1
        if self.fail:
            raise RuntimeError("串口读失败")
        return {
            "shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0,
            "wrist_flex": 0, "wrist_roll": 0, "gripper": self.grip_raw,
        }


class _FakeRobot:
    """最小假机器人（SO101Follower 的接口子集）。"""

    name = "so101_follower"
    robot_type = "so101_follower"

    def __init__(self, *, bus: _FakeBus | None = None) -> None:
        self.bus = bus or _FakeBus()
        self.observation_features = {"shoulder_pan.pos": float, "gripper.pos": float,
                                     "fixed": (480, 640, 3)}
        self.action_features = {"gripper.pos": float}
        self.sent: list[dict] = []

    def get_observation(self) -> dict:
        return {"shoulder_pan.pos": 1.5, "gripper.pos": 6.8, "fixed": "IMAGE"}

    def send_action(self, action: dict) -> dict:
        self.sent.append(dict(action))
        return {**action, "gripper.pos": 6.79}


class _FakeDataset:
    """假数据集：`save_episode()` 是 lerobot 的**提交点**，旁挂要挂在它上面。"""

    def __init__(self, num_episodes: int) -> None:
        self.num_episodes = num_episodes
        self.saved = 0

    def save_episode(self) -> None:
        self.saved += 1


class _TimeoutRobot(_FakeRobot):
    """前 N 次 `get_observation` 抛 `TimeoutError`，之后正常 —— 模拟一次 USB 卡顿。"""

    def __init__(self, fail_times: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fail_times = fail_times
        self.calls = 0

    def get_observation(self) -> dict:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("Timed out waiting for frame from camera OpenCVCamera(0) after 200 ms")
        return super().get_observation()


class _BrokenRobot(_FakeRobot):
    """非超时的真故障：必须立刻抛出，不能靠重试掩盖。"""

    def get_observation(self) -> dict:
        raise RuntimeError("机器人通信断了")


def _wrap(tmp_path, **kwargs):
    robot = _FakeRobot(**kwargs)
    writer = LoadSidecarWriter(tmp_path / "load.jsonl", dataset_repo="repo/x")
    return robot, LoadLoggingRobot(robot, writer), writer


class TestLoadLoggingRobot:
    def test_features_pass_through_unchanged(self, tmp_path) -> None:
        """关键：数据集 fields 由 features 推导，必须原样透传（否则破坏既有脚本）。"""
        robot, wrapped, writer = _wrap(tmp_path)
        assert wrapped.observation_features is robot.observation_features
        assert wrapped.action_features is robot.action_features
        with writer:
            pass

    def test_observation_keys_unchanged(self, tmp_path) -> None:
        """不得往观测里塞额外字段（避免数据集 schema 漂移）。"""
        robot, wrapped, writer = _wrap(tmp_path)
        assert wrapped.get_observation() == robot.get_observation()
        with writer:
            pass

    def test_delegates_unknown_attributes(self, tmp_path) -> None:
        _, wrapped, writer = _wrap(tmp_path)
        assert wrapped.name == "so101_follower"
        assert wrapped.robot_type == "so101_follower"
        with writer:
            pass

    def test_records_one_row_per_frame(self, tmp_path) -> None:
        _, wrapped, writer = _wrap(tmp_path)
        with writer:
            wrapped.begin_episode(0)
            for _ in range(3):
                wrapped.get_observation()
                wrapped.send_action({"gripper.pos": 6.8})
        rows = read_load_sidecar(writer.path)
        assert [row["frame_index"] for row in rows] == [0, 1, 2]
        assert rows[0]["load"]["gripper.load"] == -206.0
        assert rows[0]["positions"] == {"shoulder_pan.pos": 1.5, "gripper.pos": 6.8}

    def test_reset_segment_is_not_recorded(self, tmp_path) -> None:
        """集间复位段也会调 send_action，必须不计入任何一集。"""
        _, wrapped, writer = _wrap(tmp_path)
        with writer:
            wrapped.begin_episode(0)
            wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})
            wrapped.stop_recording()
            wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})   # 复位段：不记
            wrapped.begin_episode(1)
            wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})
        rows = read_load_sidecar(writer.path)
        assert [(row["episode_index"], row["frame_index"]) for row in rows] == [(0, 0), (1, 0)]

    def test_rerecord_replaces_previous_frames(self, tmp_path) -> None:
        _, wrapped, writer = _wrap(tmp_path)
        with writer:
            wrapped.begin_episode(0)
            for _ in range(2):
                wrapped.get_observation()
                wrapped.send_action({"gripper.pos": 6.8})
            wrapped.begin_episode(0)                    # 重录
            wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})
        rows = read_load_sidecar(writer.path)
        assert len(rows) == 1 and rows[0]["frame_index"] == 0

    def test_send_action_returns_wrapped_result(self, tmp_path) -> None:
        """必须返回**实际下发**的动作（可能被 max_relative_target 裁剪过）。"""
        _, wrapped, writer = _wrap(tmp_path)
        with writer:
            wrapped.begin_episode(0)
            assert wrapped.send_action({"gripper.pos": 6.8})["gripper.pos"] == 6.79

    def test_load_read_failure_does_not_break_collection(self, tmp_path) -> None:
        _, wrapped, writer = _wrap(tmp_path, bus=_FakeBus(fail=True))
        with writer:
            wrapped.begin_episode(0)
            wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})
        assert wrapped.read_failures == 1
        rows = read_load_sidecar(writer.path)
        assert len(rows) == 1 and rows[0]["load"] == {}

    def test_no_bus_is_tolerated(self, tmp_path) -> None:
        robot = _FakeRobot()
        robot.bus = None
        writer = LoadSidecarWriter(tmp_path / "l.jsonl")
        wrapped = LoadLoggingRobot(robot, writer)
        with writer:
            wrapped.begin_episode(0)
            wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})
        assert read_load_sidecar(writer.path)[0]["load"] == {}

    def test_load_read_uses_raw_register(self, tmp_path) -> None:
        robot, wrapped, writer = _wrap(tmp_path)
        with writer:
            wrapped.begin_episode(0)
            wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})
        assert robot.bus.calls == 1          # 每录一帧一次
        assert LOAD_REGISTER == "Present_Load"

    def test_mark_episode_saved_stops_further_frames(self, tmp_path) -> None:
        """save 之后到下一集开始之间不该再有帧混进来。"""
        _, wrapped, writer = _wrap(tmp_path)
        with writer:
            wrapped.begin_episode(0)
            wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})
            wrapped.mark_episode_saved()
            wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})   # 已提交：不记
        assert len(read_load_sidecar(writer.path)) == 1

    def test_abort_drops_an_unsaved_episode(self, tmp_path) -> None:
        """中断发生在"尚未 save_episode"的一集 → 该集不落旁挂（数据集里也没有它）。"""
        _, wrapped, writer = _wrap(tmp_path)
        with writer:
            wrapped.begin_episode(0)
            wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})
            writer.abort_episode()
        assert read_load_sidecar(writer.path) == []


class TestCameraTimeoutRetry:
    """回归（真机实测）：一路相机偶发 200ms 取帧超时，曾让整场录制崩掉、丢掉当前集。"""

    def test_recovers_from_transient_timeout(self, tmp_path) -> None:
        robot = _TimeoutRobot(fail_times=2)
        writer = LoadSidecarWriter(tmp_path / "l.jsonl")
        wrapped = LoadLoggingRobot(robot, writer)
        with writer:
            wrapped.begin_episode(0)
            observation = wrapped.get_observation()
            wrapped.send_action({"gripper.pos": 6.8})
        assert observation == {"shoulder_pan.pos": 1.5, "gripper.pos": 6.8, "fixed": "IMAGE"}
        assert wrapped.camera_timeouts == 2
        assert robot.calls == 3                                # 两次重试
        assert len(read_load_sidecar(writer.path)) == 1        # 帧照常记，流程没被打断

    def test_raises_after_all_attempts(self, tmp_path) -> None:
        """连续超时说明相机真有问题 —— 抛出去，别无限等。"""
        robot = _TimeoutRobot(fail_times=99)
        writer = LoadSidecarWriter(tmp_path / "l.jsonl")
        wrapped = LoadLoggingRobot(robot, writer)
        with writer, pytest.raises(TimeoutError, match="连续超时"):
            wrapped.get_observation()
        assert wrapped.camera_timeouts == CAMERA_READ_ATTEMPTS
        assert robot.calls == CAMERA_READ_ATTEMPTS

    def test_other_errors_are_not_retried(self, tmp_path) -> None:
        """非超时的故障立刻抛，重试不该掩盖真问题。"""
        writer = LoadSidecarWriter(tmp_path / "l.jsonl")
        wrapped = LoadLoggingRobot(_BrokenRobot(), writer)
        with writer, pytest.raises(RuntimeError, match="通信断了"):
            wrapped.get_observation()
        assert wrapped.camera_timeouts == 0

    def test_timeouts_counter_starts_at_zero(self, tmp_path) -> None:
        _, wrapped, writer = _wrap(tmp_path)
        with writer:
            wrapped.begin_episode(0)
            wrapped.get_observation()
        assert wrapped.camera_timeouts == 0


class _FakeRecordModule:
    """假 `lerobot.record` 模块：模拟"录制段 + 集间复位段"的调用序列。"""

    def __init__(self, robot: _FakeRobot) -> None:
        self._robot = robot
        self.calls: list[tuple[object, object, dict]] = []

        def make_robot_from_config(cfg):  # noqa: ANN001
            return self._robot

        def record_loop(robot=None, events=None, fps=None, dataset=None, **kwargs):  # noqa: ANN001
            self.calls.append((robot, dataset, kwargs))
            robot.get_observation()          # 录/复位段都会走这两句
            robot.send_action({"gripper.pos": 6.8})
            if dataset is not None:
                dataset.save_episode()       # 与 lerobot.record 一致：录制段结束即提交

        self.make_robot_from_config = make_robot_from_config
        self.record_loop = record_loop


class TestPatchRecordPipeline:
    def test_wraps_robot_and_tracks_episodes(self, tmp_path) -> None:
        module = _FakeRecordModule(_FakeRobot())
        writer = LoadSidecarWriter(tmp_path / "load.jsonl")
        with patch_record_pipeline(module, writer) as holder:
            robot = module.make_robot_from_config(object())
            assert isinstance(robot, LoadLoggingRobot)
            assert holder["robot"] is robot
            module.record_loop(robot=robot, events={}, fps=30, dataset=_FakeDataset(0),
                               teleop=None, control_time_s=1)
            module.record_loop(robot=robot, events={}, fps=30, dataset=None,
                               teleop=None, control_time_s=1)      # 复位段
            module.record_loop(robot=robot, events={}, fps=30, dataset=_FakeDataset(1),
                               teleop=None, control_time_s=1)
        rows = read_load_sidecar(writer.path)
        assert [(row["episode_index"], row["frame_index"]) for row in rows] == [(0, 0), (1, 0)]

    def test_forwards_kwargs_to_original_loop(self, tmp_path) -> None:
        module = _FakeRecordModule(_FakeRobot())
        writer = LoadSidecarWriter(tmp_path / "load.jsonl")
        with patch_record_pipeline(module, writer) as holder:
            robot = module.make_robot_from_config(object())
            module.record_loop(robot=robot, events={"stop_recording": False}, fps=30,
                               dataset=_FakeDataset(0), teleop="LEADER", control_time_s=99)
        _, _, kwargs = module.calls[0]
        assert kwargs == {"teleop": "LEADER", "control_time_s": 99}

    def test_restores_originals_after_exit(self, tmp_path) -> None:
        module = _FakeRecordModule(_FakeRobot())
        original_make, original_loop = module.make_robot_from_config, module.record_loop
        writer = LoadSidecarWriter(tmp_path / "load.jsonl")
        with patch_record_pipeline(module, writer):
            assert module.make_robot_from_config is not original_make
        assert module.make_robot_from_config is original_make
        assert module.record_loop is original_loop

    def test_close_commits_pending_episode(self, tmp_path) -> None:
        """`close()` 是兜底：提交仍未落盘的缓冲（正常路径已由 `save_episode` 钩子提交）。

        异常中断的"这一集没被数据集保存"由**调用方**负责（`run_vla_collect_record` 在异常时调
        `writer.abort_episode()`）；**已 `save_episode` 的集不会被 abort 抹掉**，见下一条回归。
        """
        module = _FakeRecordModule(_FakeRobot())
        writer = LoadSidecarWriter(tmp_path / "load.jsonl")
        try:
            with patch_record_pipeline(module, writer):
                robot = module.make_robot_from_config(object())
                module.record_loop(robot=robot, events={}, fps=30, dataset=_FakeDataset(0),
                                   teleop=None, control_time_s=1)
                raise RuntimeError("采集中断")
        except RuntimeError:
            pass
        assert len(read_load_sidecar(writer.path)) == 1
        assert writer._handle.closed

    def test_save_episode_commits_sidecar_immediately(self, tmp_path) -> None:
        """提交点是 `dataset.save_episode()`，不必等下一集开始。"""
        module = _FakeRecordModule(_FakeRobot())
        writer = LoadSidecarWriter(tmp_path / "load.jsonl")
        with patch_record_pipeline(module, writer):
            robot = module.make_robot_from_config(object())
            module.record_loop(robot=robot, events={}, fps=30, dataset=_FakeDataset(0),
                               teleop=None, control_time_s=1)
        # 若靠"下一集开始"提交，这一步会是空的
        assert [row["episode_index"] for row in read_load_sidecar(writer.path)] == [0]

    def test_teardown_failure_keeps_saved_episodes(self, tmp_path) -> None:
        """回归（真机实测）：`robot.disconnect()` 里夹爪 Overload 触发 abort，
        不得丢掉**已存进数据集**的集——否则旁挂比数据集少一集、`--verify` 必然失败。"""
        module = _FakeRecordModule(_FakeRobot())
        writer = LoadSidecarWriter(tmp_path / "load.jsonl")
        try:
            with patch_record_pipeline(module, writer):
                robot = module.make_robot_from_config(object())
                module.record_loop(robot=robot, events={}, fps=30, dataset=_FakeDataset(0),
                                   teleop=None, control_time_s=1)
                module.record_loop(robot=robot, events={}, fps=30, dataset=_FakeDataset(1),
                                   teleop=None, control_time_s=1)
                writer.abort_episode()      # 调用方在异常路径上的动作
                raise RuntimeError("Failed to write 'Torque_Enable' on id_=6 ... Overload error!")
        except RuntimeError:
            pass
        assert [row["episode_index"] for row in read_load_sidecar(writer.path)] == [0, 1]
