"""真机技能执行器（SO-101）：把底层闭环原语封装为带超时/错误码的技能。

设计：
- `RobotEngine`（协议）：机械臂原语的最小接口，便于测试用 Mock 替换。
- `RealEngine`：基于 `outputs/infer_so101.py` 的模块级函数（load_policy /
  get_initial_state / home_to_initial / build_observation_from_robot / predict_action /
  busy_wait）+ SO101Follower 驱动实现；闭环控制序列与 deploy_eval 保持一致。
  真机连接/时序仅在真机验证时执行，skills 模式的纯逻辑测试一律走 Mock。
- `RealSkillExecutor`：技能分发（grasp_screwdriver / reset_pose / verify_holding /
  release），每个技能返回 SkillResult（success/data/failure_reason）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from camera_slots import CAMERA_SLOTS
from real_observation import EMPTY_GRIP_BASELINE, STATE_NAMES, decode_feetech_load
from real_skills import (
    FAIL_COMM,
    FAIL_GRASP_EMPTY,
    FAIL_HOME_TIMEOUT,
    FAIL_NOT_HOLDING,
    FAIL_NOT_LIFTED,
    FAIL_TIMEOUT,
    validate_real_parameters,
)


@dataclass
class SkillResult:
    """技能执行结果（契约与 src/task_skills.SkillResult 同构）。"""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None


# 策略段每步的时间预算（秒）。CPU 上每步要读两路相机 + 推理，实测约 0.3 s/步；
# 旧实现把超时写死 120 s，用默认 300+300 步**必然超时**（见 docs/week38/plan.md 批次 11）。
GRASP_STEP_BUDGET_S = 0.5

# 相机索引：**唯一事实来源在 `camera_slots.py`**（Windows 枚举会因重启/重插漂移，
# 2026-10-10 曾是 fixed=2/handeye=0，2026-10-17 核图后为 fixed=1/handeye=2）。
# 映射错了策略等于瞎跑（曾把笔记本摄像头喂进腕部通道），故运行前务必
# `python src/run_camera_check.py` 逐张核图。
DEFAULT_CAM_IDX = CAMERA_SLOTS


class RobotEngine(Protocol):
    """机械臂原语接口（真机或 Mock 实现）。"""

    def read_state(self) -> dict[str, float]: ...
    def read_images(self) -> dict[str, Any] | None:
        """读取当前相机帧。返回 {"fixed": HWC uint8 RGB, "handeye": HWC uint8 RGB} 或 None。"""
        ...
    def read_load(self) -> dict[str, float]:
        """读取各关节负载（Present_Load）。返回 {<joint>.load: 读数}；不支持时返回空 dict。"""
        ...
    def home(self, timeout_s: float | None = None, *, include_gripper: bool = True) -> bool:
        """归位到采集初始姿态；``timeout_s=None`` 时用 infer_so101 的 HOME_TIMEOUT。"""
        ...
    def move_to(self, target: dict[str, float], duration_s: float = 1.0) -> bool:
        """点位移动：把关节逐步逼近 target（用于 scripted/恢复原语）。"""
        ...
    def send_action(self, action: dict[str, float]) -> None:
        """发送一组关节目标（单位与数据集一致）。VLA 闭环每步都靠它。"""
        ...
    def grasp_and_lift(self, steps: int, hold_steps: int) -> tuple[dict[str, float], dict[str, float]]:
        """跑 ACT 闭环夹起并抬起，返回 (最终状态 dict, 末段状态)。"""
        ...
    def release(self) -> None: ...
    def close(self) -> None: ...


class RealEngine:
    """真机引擎：延迟 import outputs/infer_so101，避免纯逻辑测试拉起重依赖。

    安全：启用 max_relative_target（单步关节增量限幅），见 docs/week36 S2。
    """

    def __init__(
        self,
        model_dir: str,
        dataset_root: str,
        dataset_repo: str,
        port: str = "COM6",
        robot_id: str = "my_awesome_follower_arm",
        cam_idx: dict[str, int] | None = None,
        max_relative_target: float = 8.0,
        load_act_policy: bool = True,
    ) -> None:
        self.model_dir = model_dir
        self.dataset_root = dataset_root
        self.dataset_repo = dataset_repo
        self.port = port
        self.robot_id = robot_id
        self.cam_idx = cam_idx or DEFAULT_CAM_IDX
        self.max_relative_target = max_relative_target
        #: VLA（SmolVLA）闭环时为 False：它自带策略，加载 ACT checkpoint 与
        #: ACT 的 dataset_stats 只会白白多读一份权重，且容易让人误以为在用 ACT。
        self.load_act_policy = load_act_policy
        self._base = None
        self._ds = None
        self._policy = None
        self._robot = None
        self._device = None

    # -- 延迟加载 ----------------------------------------------------------
    def _lazy_init(self) -> None:
        if self._base is not None:
            return
        import sys

        # 复用 outputs/infer_so101.py（与闭环脚本同机、单一来源）
        sys.path.insert(0, r"d:\FittenProject\bot_agent\outputs")
        import importlib

        base = importlib.import_module("infer_so101")
        base.force_msmf_backend()
        if self.load_act_policy:
            policy, ds = base.load_policy(self.model_dir, self.dataset_repo, self.dataset_root)
        else:
            # VLA 模式：只要"机器人 + 数据集"（归位要用数据集的初始姿态），
            # 不加载 ACT checkpoint —— 那是另一条路径的策略。
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            policy = None
            ds = LeRobotDataset(self.dataset_repo, root=self.dataset_root, download_videos=False)
            logging.info(
                "VLA 模式：已加载数据集（%s 集 / %s 帧），未加载 ACT 策略",
                ds.meta.total_episodes, ds.meta.total_frames,
            )

        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        cameras = {
            name: OpenCVCameraConfig(index_or_path=idx, fps=base.FPS, width=640, height=480)
            for name, idx in self.cam_idx.items()
        }
        robot = SO101Follower(
            SO101FollowerConfig(port=self.port, id=self.robot_id, cameras=cameras,
                                max_relative_target=self.max_relative_target)
        )
        robot.connect()
        self._base = base
        self._ds = ds
        self._policy = policy
        self._robot = robot
        import torch

        self._device = torch.device("cpu")

    def _state_from_obs(self, obs: dict[str, Any]) -> dict[str, float]:
        state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
        return {name: float(v) for name, v in zip(STATE_NAMES, state)}

    def read_state(self) -> dict[str, float]:
        self._lazy_init()
        obs = self._base.build_observation_from_robot(self._robot, self._ds)
        return self._state_from_obs(obs)

    def read_images(self) -> dict[str, Any] | None:
        self._lazy_init()
        obs = self._base.build_observation_from_robot(self._robot, self._ds)
        return {key: obs[key] for key in ("observation.images.fixed", "observation.images.handeye")
                if key in obs}

    def read_load(self) -> dict[str, float]:
        """读取各关节负载（Present_Load），并解码为带符号千分比。

        STS3215 的原始读数含方向位（bit10），直接比较会把换向误判为大负载，
        故统一用 `decode_feetech_load` 解码后再向上暴露。
        """
        self._lazy_init()
        raw = self._robot.bus.sync_read("Present_Load", normalize=False)
        return {f"{name}.load": decode_feetech_load(value) for name, value in raw.items()}

    def send_action(self, action: dict[str, float]) -> None:
        """发送一组关节目标（关节量，单位与数据集一致）。

        单步增量由 `max_relative_target` 在驱动器侧限幅（S2 安全），故这里不再自行平滑。
        未知名直接忽略（VLA 输出可能带多余维），但**若全部被忽略则报错** ——
        否则"发了动作却一个关节都没动"会表现成"策略不收敛"，极难排查。
        """
        self._lazy_init()
        payload = {name: float(value) for name, value in action.items() if name in STATE_NAMES}
        if not payload:
            raise ValueError(
                f"send_action 收到空动作（键 = {sorted(action)}），已知关节为 {list(STATE_NAMES)}"
            )
        self._robot.send_action(payload)

    def home(self, timeout_s: float | None = None, *, include_gripper: bool = True) -> bool:
        """归位到采集初始姿态。

        ``timeout_s=None`` 用 `infer_so101.HOME_TIMEOUT`（大行程归位需要几十秒；
        旧值 10 s + 0.03 的容差会必然超时，见 docs/week38/plan.md 批次 11）。
        ``include_gripper=False`` 时**不移动夹爪**（也不参与到达判定）——数据集第 0 帧
        的夹爪是闭合值，反复往闭合方向硬顶会把舵机顶进过载保护。
        """
        self._lazy_init()
        base = self._base
        initial = base.get_initial_state(self._ds)
        skip = () if include_gripper else ("gripper.pos",)
        if timeout_s is None:
            return base.home_to_initial(self._robot, self._ds, initial, skip_joints=skip)
        return base.home_to_initial(
            self._robot, self._ds, initial, timeout_s=timeout_s, skip_joints=skip
        )

    def move_to(self, target: dict[str, float], duration_s: float = 1.0) -> bool:
        """按关节逐步逼近目标位姿；由 SO101Follower 的 max_relative_target 限幅单步增量。"""
        self._lazy_init()
        import time

        deadline = time.perf_counter() + duration_s
        tol = 3.0  # 各关节逼近误差容限（度）
        while time.perf_counter() < deadline:
            state = self.read_state()
            done = True
            for name in STATE_NAMES:
                if name in target and abs(state.get(name, 0) - target[name]) > tol:
                    done = False
                    break
            if done:
                return True
            # 并发发送整组目标，由驱动器内部限幅逐步执行
            self._robot.send_action({name: float(v) for name, v in target.items()
                                     if name in STATE_NAMES})
            self._base.busy_wait(self._base.DT)
        return False

    def grasp_and_lift(self, steps: int, hold_steps: int) -> tuple[dict[str, float], dict[str, float]]:
        """复刻闭环序列：每步 policy.reset() 重推 + send_action，随后保持。

        Returns:
            (最终状态 dict, 保持开始前的末段状态 dict)。
            此方法仅在真机验证时调用；纯逻辑测试用 Mock。
        """
        self._lazy_init()
        base, policy, ds, robot = self._base, self._policy, self._ds, self._robot
        policy.reset()
        initial = base.get_initial_state(ds)
        last_action = {name: float(v) for name, v in zip(STATE_NAMES, initial)}
        final_state: dict[str, float] = {}
        prev_state: dict[str, float] = {}
        for step in range(steps + hold_steps):
            obs = base.build_observation_from_robot(robot, ds)
            state = self._state_from_obs(obs)
            if step < steps:
                policy.reset()
                action = base.predict_action(obs, policy, self._device, use_amp=False, task="")
                last_action = {name: float(v) for name, v in zip(STATE_NAMES, action.tolist())}
            robot.send_action(last_action)
            if step == steps - 1:
                prev_state = dict(state)
            final_state = dict(state)
            base.busy_wait(base.DT)
        return final_state, prev_state

    def release(self) -> None:
        self._lazy_init()
        self._robot.send_action({"gripper.pos": 20.0})
        time.sleep(1.0)

    def close(self) -> None:
        if self._robot is not None:
            try:
                self._robot.disconnect()
            finally:
                self._robot = None


class MockEngine:
    """测试/干跑用假引擎：不连接机械臂，按预设序列返回状态。"""

    def __init__(self, grasp_result: str = "success", with_images: bool = False,
                 gripper_load: float | None = None) -> None:
        self.grasp_result = grasp_result  # success / grasp_empty / not_lifted
        self.grasp_calls = 0
        self.with_images = with_images
        # 负载与 grasp_result 保持一致，口径取第 38 周 0.6 真机实测
        # （空载 ≈ −62‰、夹持 ≈ −206‰；负号 = 闭合方向，与 decode_feetech_load 一致）。
        # 若不这样配对，「位置 OR 负载」判据会把"咬空"误判成"夹持"
        # （Mock 原实现恒为 +180，符号与量级都不对）。
        if gripper_load is None:
            gripper_load = -62.0 if grasp_result == "grasp_empty" else -206.0
        self.gripper_load = float(gripper_load)  # 模拟夹爪负载（解码后千分比）
        self.state = {  # 模拟“夹住并抬起”状态
            "shoulder_pan.pos": -6.8, "shoulder_lift.pos": 44.9,
            "elbow_flex.pos": -30.6, "wrist_flex.pos": 89.4,
            "wrist_roll.pos": -6.6, "gripper.pos": 9.5,
        }

    def read_state(self) -> dict[str, float]:
        return dict(self.state)

    def read_images(self) -> dict[str, Any] | None:
        if not self.with_images:
            return None
        # 合成两路假帧（HWC uint8 RGB），仅用于链路验证
        import numpy as np

        return {
            "observation.images.fixed": np.full((64, 64, 3), 128, dtype=np.uint8),
            "observation.images.handeye": np.full((64, 64, 3), 200, dtype=np.uint8),
        }

    def send_action(self, action: dict[str, float]) -> None:
        """记录收到的关节目标（供测试断言 VLA 闭环确实在发动作）。"""
        if not hasattr(self, "sent_actions"):
            self.sent_actions: list[dict[str, float]] = []
        self.sent_actions.append({name: float(value) for name, value in action.items()})

    def read_load(self) -> dict[str, float]:
        """返回合成负载（解码后千分比）；``gripper_load`` 可直接改写以模拟空载/夹持。"""
        loads = {
            "shoulder_pan.load": 35.0, "shoulder_lift.load": 320.0,
            "elbow_flex.load": 180.0, "wrist_flex.load": 95.0,
            "wrist_roll.load": 28.0, "gripper.load": self.gripper_load,
        }
        return loads

    def home(self, timeout_s: float | None = None, *, include_gripper: bool = True) -> bool:
        return True

    def move_to(self, target: dict[str, float], duration_s: float = 1.0) -> bool:
        for name in STATE_NAMES:
            if name in target:
                self.state[name] = float(target[name])
        return True

    def grasp_and_lift(self, steps: int, hold_steps: int) -> tuple[dict[str, float], dict[str, float]]:
        self.grasp_calls += 1
        if self.grasp_result == "grasp_empty":
            self.state["gripper.pos"] = 1.5  # 咬空
        elif self.grasp_result == "not_lifted":
            self.state["wrist_flex.pos"] = 60.0
        return dict(self.state), dict(self.state)

    def release(self) -> None:
        self.state["gripper.pos"] = 20.0

    def close(self) -> None:
        pass


class RealSkillExecutor:
    """真机技能执行器：技能分发 + 验证，返回 SkillResult。"""

    def __init__(self, engine: RobotEngine, empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
                 max_grasp_steps: int = 300, hold_steps: int = 300,
                 grasp_timeout_s: float | None = None) -> None:
        self.engine = engine
        self.empty_grip_baseline = empty_grip_baseline
        self.max_grasp_steps = max_grasp_steps
        self.hold_steps = hold_steps
        self.grasp_timeout_s = grasp_timeout_s

    def grasp_timeout(self) -> float:
        """策略段超时预算：显式给了 ``grasp_timeout_s`` 就用它，否则按步数缩放。"""
        if self.grasp_timeout_s is not None:
            return float(self.grasp_timeout_s)
        return max(120.0, GRASP_STEP_BUDGET_S * (self.max_grasp_steps + self.hold_steps))

    def park(self, *, include_gripper: bool = True) -> bool:
        """收尾复位：把机械臂送回采集初始姿态，避免把它留在策略结束时的任意位姿。

        运行结束（含异常 / 中断）都应调用；失败不抛异常（收尾动作不该盖过主流程结论）。
        ``include_gripper=False`` 时只复位臂关节、不碰夹爪（避免归位把夹爪往闭合方向硬顶）。
        **注意**：若此刻夹着物体，复位会把物体一起拖走 —— 需要时请先 `release`
        或人工干预后再调用。
        """
        try:
            return bool(self.engine.home(include_gripper=include_gripper))
        except Exception:
            return False

    def _read_gripper_load(self) -> float | None:
        """读夹爪负载（‰）；引擎不具备该能力或读取失败时返回 None（退化为纯位置判据）。"""
        try:
            loads = self.engine.read_load()
        except Exception:
            return None
        if not isinstance(loads, dict):
            return None
        value = loads.get("gripper.load")
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    # -- 技能分发 ----------------------------------------------------------
    def execute(self, skill_name: str, params: dict[str, Any] | None = None) -> SkillResult:
        error = validate_real_parameters(skill_name, params)
        if error:
            return SkillResult(False, failure_reason=error)
        handler = getattr(self, f"_do_{skill_name}", None)
        if handler is None:
            return SkillResult(False, failure_reason=f"未实现的技能: {skill_name}")
        try:
            return handler()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # 通信/驱动层错误统一归类
            return SkillResult(False, data={"skill": skill_name},
                               failure_reason=f"{FAIL_COMM}: {exc}")

    # -- 各技能实现 --------------------------------------------------------
    def _do_grasp_screwdriver(self) -> SkillResult:
        import time

        from real_verifier import RealVerifier

        # 与闭环推理协议（outputs/infer_so101.py）一致：**先归位到采集初始姿态再跑策略**。
        # ACT 是学出来的轨迹，起跑位姿严重偏离初始姿态会让策略动作失去意义
        # （实测曾出现 shoulder_lift 偏差 145 单位，见 docs/week38/plan.md 批次 11）。
        if not self.engine.home():
            return SkillResult(False, data={"skill": "grasp_screwdriver"},
                               failure_reason=FAIL_HOME_TIMEOUT)

        timeout_s = self.grasp_timeout()
        start = time.perf_counter()
        final_state, _ = self.engine.grasp_and_lift(self.max_grasp_steps, self.hold_steps)
        if time.perf_counter() - start > timeout_s:
            return SkillResult(False, data={"skill": "grasp_screwdriver"},
                               failure_reason=FAIL_TIMEOUT)

        verifier = RealVerifier(self.empty_grip_baseline)
        result = verifier.verify_grasp_lift(
            float(final_state.get("gripper.pos", 0)),
            float(final_state.get("wrist_flex.pos", 0)),
            self._read_gripper_load(),
        )
        return SkillResult(
            result.success,
            data={"skill": "grasp_screwdriver", "final_state": final_state},
            failure_reason=result.failure_code if not result.success else None,
        )

    def _do_reset_pose(self) -> SkillResult:
        ok = self.engine.home()
        return SkillResult(
            ok,
            data={"skill": "reset_pose"},
            failure_reason=None if ok else FAIL_HOME_TIMEOUT,
        )

    def _do_verify_holding(self) -> SkillResult:
        from real_observation import holds_object

        state = self.engine.read_state()
        gripper = float(state.get("gripper.pos", 0))
        load = self._read_gripper_load()
        holding = holds_object(gripper, load, empty_grip_baseline=self.empty_grip_baseline)
        return SkillResult(
            holding,
            data={"skill": "verify_holding", "holding": holding,
                  "gripper": gripper, "gripper_load": load, "state": state},
            failure_reason=None if holding else FAIL_NOT_HOLDING,
        )

    def _do_release(self) -> SkillResult:
        self.engine.release()
        return SkillResult(True, data={"skill": "release"})

    def _do_done(self) -> SkillResult:
        return SkillResult(True, data={"skill": "done"})
