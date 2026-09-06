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

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from real_observation import STATE_NAMES
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


class RobotEngine(Protocol):
    """机械臂原语接口（真机或 Mock 实现）。"""

    def read_state(self) -> dict[str, float]: ...
    def read_images(self) -> dict[str, Any] | None:
        """读取当前相机帧。返回 {"fixed": HWC uint8 RGB, "handeye": HWC uint8 RGB} 或 None。"""
        ...
    def home(self, timeout_s: float = 10.0) -> bool: ...
    def move_to(self, target: dict[str, float], duration_s: float = 1.0) -> bool:
        """点位移动：把关节逐步逼近 target（用于 scripted/恢复原语）。"""
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
    ) -> None:
        self.model_dir = model_dir
        self.dataset_root = dataset_root
        self.dataset_repo = dataset_repo
        self.port = port
        self.robot_id = robot_id
        self.cam_idx = cam_idx or {"fixed": 2, "handeye": 1}
        self.max_relative_target = max_relative_target
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
        policy, ds = base.load_policy(self.model_dir, self.dataset_repo, self.dataset_root)

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

    def home(self, timeout_s: float = 10.0) -> bool:
        self._lazy_init()
        base = self._base
        initial = base.get_initial_state(self._ds)
        return base.home_to_initial(self._robot, self._ds, initial)

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

    def __init__(self, grasp_result: str = "success", with_images: bool = False) -> None:
        self.grasp_result = grasp_result  # success / grasp_empty / not_lifted
        self.grasp_calls = 0
        self.with_images = with_images
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

    def home(self, timeout_s: float = 10.0) -> bool:
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

    def __init__(self, engine: RobotEngine, empty_grip_baseline: float = 1.5,
                 max_grasp_steps: int = 300, hold_steps: int = 300) -> None:
        self.engine = engine
        self.empty_grip_baseline = empty_grip_baseline
        self.max_grasp_steps = max_grasp_steps
        self.hold_steps = hold_steps

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

        from real_observation import is_holding, is_lifted
        from real_verifier import RealVerifier

        start = time.perf_counter()
        final_state, _ = self.engine.grasp_and_lift(self.max_grasp_steps, self.hold_steps)
        if time.perf_counter() - start > 120:
            return SkillResult(False, data={"skill": "grasp_screwdriver"},
                               failure_reason=FAIL_TIMEOUT)

        verifier = RealVerifier(self.empty_grip_baseline)
        result = verifier.verify_grasp_lift(
            float(final_state.get("gripper.pos", 0)),
            float(final_state.get("wrist_flex.pos", 0)),
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
        from real_observation import is_holding

        state = self.engine.read_state()
        gripper = float(state.get("gripper.pos", 0))
        holding = is_holding(gripper, self.empty_grip_baseline)
        return SkillResult(
            holding,
            data={"skill": "verify_holding", "holding": holding,
                  "gripper": gripper, "state": state},
            failure_reason=None if holding else FAIL_NOT_HOLDING,
        )

    def _do_release(self) -> SkillResult:
        self.engine.release()
        return SkillResult(True, data={"skill": "release"})

    def _do_done(self) -> SkillResult:
        return SkillResult(True, data={"skill": "done"})
