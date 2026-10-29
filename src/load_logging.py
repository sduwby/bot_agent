"""把 lerobot 录制流水线接到旁挂负载记录上（第 38 周 P1 前置，2.9 采集用）。

`lerobot-record` 的数据集 features 由机器人的 `observation_features` 推导，只有六轴 `.pos`
+ 两路相机，**不含 `Present_Load`** → 照它采集出来的数据集会和 `demo_v23` 一样，
做不了可靠的成功标注（位置判据零可分性，见 `load_sidecar` 的说明）。

本模块提供两件事：

1. `LoadLoggingRobot`：包一层机器人，每录一帧旁挂读一次 `Present_Load` 写进旁挂文件。
   `observation_features` / `action_features` **原样透传** →
   **LeRobotDataset 的 fields 完全不变**，既有 ACT/SmolVLA 训练脚本不受影响；
2. `patch_record_pipeline(...)`：临时替换录制流水线的两个函数，把 wrapper 与
   "现在录第几集 / 现在不是录制段"的上下文接上，退出时还原。

## 为什么必须替换这两个函数

- `make_robot_from_config`：`record()` 内部由它建机器人实例，只有在这里才能把实例包起来；
- `record_loop`：它同时用于"录某一集"（`dataset` 非空）与"集间复位"（`dataset` 为空）。
  复位段也会调 `get_observation`/`send_action`，不区分就会把复位段的负载混进帧序列 →
  **必须**由它告知 wrapper 当前的集号与是否在录制段。
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

from load_sidecar import LoadSidecarWriter

logger = logging.getLogger(__name__)

#: STS3215 负载寄存器名（与 `real_executor.RealEngine.read_load` 同源）。
LOAD_REGISTER = "Present_Load"

#: 相机取帧超时的重试次数与间隔。lerobot 的 `OpenCVCamera.async_read` 把等待写死成 200ms
#: （配置里没有可调项），30fps 下 200ms 等于连丢 ~6 帧；实测两路相机抢同一个 USB 控制器时
#: 会偶发一次，**一次卡顿就整场录制崩掉、丢掉当前集**。这里只重试几轮把它熬过去；
#: 正常路径**完全不进入**这段代码，零行为变化。
CAMERA_READ_ATTEMPTS = 3
CAMERA_RETRY_DELAY_S = 0.2


class LoadLoggingRobot:
    """机器人包装：每录一帧旁挂记录负载；features 与其余接口原样透传。"""

    def __init__(self, robot: Any, writer: LoadSidecarWriter) -> None:
        self._robot = robot
        self._writer = writer
        self._episode_index: int | None = None
        self._frame_index = 0
        self._recording = False
        self._positions: dict[str, float] = {}
        self.read_failures = 0
        #: 相机取帧超时累计次数（>0 说明这一场出现过卡顿，数据质量需留意）。
        self.camera_timeouts = 0

    # -- 透传（保证数据集 fields 与既有接口不变） --------------------------- #

    def __getattr__(self, item: str) -> Any:
        # 只在本类没定义时才走到这里；用 object.__getattribute__ 避免自递归
        return getattr(object.__getattribute__(self, "_robot"), item)

    @property
    def observation_features(self) -> Any:
        return self._robot.observation_features

    @property
    def action_features(self) -> Any:
        return self._robot.action_features

    # -- 录制上下文 -------------------------------------------------------- #

    def begin_episode(self, episode_index: int) -> None:
        """进入"录第 episode_index 集"；同集号再次进入 = 重录（由 writer 丢弃旧缓冲）。"""
        self._writer.begin_episode(int(episode_index))
        self._episode_index = int(episode_index)
        self._frame_index = 0
        self._recording = True

    def stop_recording(self) -> None:
        """进入集间复位段：该段的观测/动作都不算入任何一集。"""
        self._recording = False

    def mark_episode_saved(self) -> None:
        """数据集已把当前集落盘（`dataset.save_episode()`）→ 旁挂**同步提交**该集。

        为什么要在这里提交、而不是等"下一集开始"（`begin_episode`）才提交：
        收尾阶段的异常（实测：`robot.disconnect()` 里夹爪 `Overload error!`）会让调用方走
        `writer.abort_episode()`，而那时**这一集已经存进数据集了** —— 若旁挂还没提交，
        它就被当作"未完成的一集"丢掉，结果是**旁挂比数据集少一集、`--verify` 必然失败**。
        把提交点钉在 `save_episode` 上，旁挂与数据集就在同一个事件上落盘。
        """
        self._recording = False
        self._writer.commit_episode()

    def finish(self) -> None:
        """收尾：提交未落盘的缓冲并关闭旁挂文件。"""
        self._recording = False
        self._writer.close()

    # -- 观测 / 动作 ------------------------------------------------------- #

    def get_observation(self) -> dict[str, Any]:
        observation = self._get_observation_with_retry()
        # 记下与数据集 observation.state 同源的位置值，供事后校验逐帧对齐
        self._positions = {
            key: float(value)
            for key, value in observation.items()
            if key.endswith(".pos") and isinstance(value, (int, float))
        }
        return observation

    def _get_observation_with_retry(self) -> dict[str, Any]:
        """相机取帧超时重试（只看 `TimeoutError`，真故障仍照原样抛出）。

        记录超时次数而不是静默吞掉：采完汇总里会显示，**相机真退化时不会被重试掩盖**。
        """
        for attempt in range(1, CAMERA_READ_ATTEMPTS + 1):
            try:
                return self._robot.get_observation()
            except TimeoutError as exc:
                self.camera_timeouts += 1
                logger.warning(
                    "相机取帧超时（第 %d/%d 次尝试）：%s", attempt, CAMERA_READ_ATTEMPTS, exc
                )
                if attempt < CAMERA_READ_ATTEMPTS:
                    time.sleep(CAMERA_RETRY_DELAY_S)
        raise TimeoutError(
            f"相机取帧连续超时 {CAMERA_READ_ATTEMPTS} 次（累计 {self.camera_timeouts} 次）——"
            "多半是两路相机抢同一个 USB 控制器，换一个 USB 口/控制器再试"
        )

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # 在"下发动作之前"采样本帧负载（对应刚读到的那一帧状态），且只在录制段采。
        # 放在 send_action 而非 get_observation：录制循环里 send_action 恰好每存一帧调用一次，
        # 而 "无 policy/teleop" 的跳过分支不会调用它 —— 天然与数据集帧一一对应。
        raw = self._read_load() if self._recording else {}
        sent = self._robot.send_action(action)
        if self._recording and self._episode_index is not None:
            self._writer.add_frame(
                self._episode_index, self._frame_index, raw, self._positions
            )
            self._frame_index += 1
        return sent

    def _read_load(self) -> dict[str, float]:
        bus = getattr(self._robot, "bus", None)
        if bus is None:
            return {}
        try:
            raw = bus.sync_read(LOAD_REGISTER, normalize=False)
        except Exception as exc:  # 读负载失败不能中断整场采集
            self.read_failures += 1
            if self.read_failures <= 3 or self.read_failures % 100 == 0:
                logger.warning(
                    "读取 %s 失败（累计 %d 次）：%s", LOAD_REGISTER, self.read_failures, exc
                )
            return {}
        return {f"{motor}.load": float(value) for motor, value in raw.items()}


@contextlib.contextmanager
def patch_record_pipeline(record_module: Any, writer: LoadSidecarWriter):
    """临时把录制流水线接到旁挂记录上；退出时还原并收尾。

    `record_module` 为 `lerobot.record`（**注入**以便离线测试）。
    `yield` 出的 dict 在 `record()` 返回后含 `"robot"`（被包装的实例，可读 `read_failures`）。

    除了替换 `record_loop`，还会给数据集实例挂一个 `save_episode` 钩子：数据集的提交点
    （`dataset.save_episode()`）就是旁挂的提交点，见 `LoadLoggingRobot.mark_episode_saved`。
    """
    original_make = record_module.make_robot_from_config
    original_loop = record_module.record_loop
    holder: dict[str, Any] = {"robot": None}

    def make(cfg: Any) -> Any:
        robot = LoadLoggingRobot(original_make(cfg), writer)
        holder["robot"] = robot
        return robot

    def hook_save_episode(dataset: Any, robot: LoadLoggingRobot) -> None:
        """把"数据集落盘一集"与"旁挂提交一集"绑在同一事件上（只挂一次）。"""
        if dataset is None or getattr(dataset, "_sidecar_hooked", False):
            return
        original_save = dataset.save_episode

        def save_episode(*args: Any, **kwargs: Any) -> Any:
            result = original_save(*args, **kwargs)
            robot.mark_episode_saved()
            return result

        dataset.save_episode = save_episode
        dataset._sidecar_hooked = True

    def loop(
        robot: Any = None,
        events: Any = None,
        fps: Any = None,
        dataset: Any = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(robot, LoadLoggingRobot):
            if dataset is not None:
                robot.begin_episode(dataset.num_episodes)
                hook_save_episode(dataset, robot)
            else:
                robot.stop_recording()
        return original_loop(robot, events, fps, dataset=dataset, **kwargs)

    record_module.make_robot_from_config = make
    record_module.record_loop = loop
    try:
        yield holder
    finally:
        record_module.make_robot_from_config = original_make
        record_module.record_loop = original_loop
        robot = holder.get("robot")
        if isinstance(robot, LoadLoggingRobot):
            robot.finish()
