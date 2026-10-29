#!/usr/bin/env python
"""SmolVLA 推理 runner（阶段 F S12）：替换 run_real_agent 的 VLARunner 占位。

- `--dry-run`：FakeVLAPolicy + 合成状态/帧，验证 prepost 与主循环链路（无需真实模型）
- 真实模式：`SmolVLAPolicy.from_pretrained(model_dir)` + `RealEngine` 观测，
  每步 `select_action` → `send_action`（关节量，dataset 口径）
- `--reset-per-step`：每步 `policy.reset()` 清空 action queue 强制重推
  （SmolVLA 默认 `n_action_steps>1` 复用整段 chunk，与 ACT 的 queue 问题同源，
  真机实测后决定默认值；代码先暴露开关）

**分层**：`run_vla_loop` 是主循环本体，**全部依赖注入**（策略/prepost/成功判据/休眠），
因此可以在没有机械臂、没有模型的机器上跑纯逻辑测试；`run_real` 只负责把真实的
`RealEngine` + SmolVLA 策略装起来再调用它。

用法::

    python src/run_vla_infer.py --dry-run --instruction "夹起螺丝刀"
    python src/run_vla_infer.py --model-dir <smolvla ckpt> --instruction "夹起螺丝刀" \
        --dataset-root <demo_v23> --dataset-repo wangbangyan/demo_v23 --port COM6
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from camera_slots import CAMERA_SLOTS  # noqa: E402  （唯一事实来源）
from real_observation import (  # noqa: E402
    EMPTY_GRIP_BASELINE,
    GRIP_HOLD_MAX,
    GRIP_LOAD_THRESHOLD,
    GRIP_POS_MARGIN,
    LIFT_WRIST,
    STATE_NAMES,
    holds_object,
    is_holding,
)

#: 两路相机键（与采集/数据集一致）。
IMAGE_KEYS = ("observation.images.fixed", "observation.images.handeye")
#: 夹爪负载在 `read_load()` 里的键（见 `RealEngine.read_load`）。
GRIPPER_LOAD_KEY = "gripper.load"
#: 闭环默认步数：演示集最短一集 340 帧，默认必须覆盖（旧默认 30 会让策略必然跑不完）。
DEFAULT_STEPS = 600


def build_obs(engine: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """从引擎取 state + 两路相机帧（缺图给空 dict；关节按 `STATE_NAMES` 定序）。"""
    state = engine.read_state()
    vector = np.asarray([float(state.get(name, 0.0)) for name in STATE_NAMES], dtype=np.float32)
    images: dict[str, np.ndarray] = {}
    try:
        frames = engine.read_images() or {}
        images = {key: value for key, value in frames.items() if value is not None}
    except (AttributeError, RuntimeError, OSError) as exc:  # 相机掉线不该让整轮崩掉
        logging.warning("读取相机帧失败（本轮将只带状态推理）：%s", exc)
    return vector, images


def gripper_load(engine: Any) -> float | None:
    """读夹爪负载（解码后千分比）。不支持/读取失败 → `None`，由位置判据单独兜底。"""
    try:
        value = engine.read_load().get(GRIPPER_LOAD_KEY)
    except (AttributeError, KeyError, TypeError, RuntimeError, OSError) as exc:
        logging.warning("读取夹爪负载失败（改用位置判据）：%s", exc)
        return None
    return None if value is None else float(value)


def holding_and_lifted(
    state: Mapping[str, float],
    load: float | None = None,
    *,
    empty_grip_baseline: float = EMPTY_GRIP_BASELINE,
    margin: float = GRIP_POS_MARGIN,
) -> bool:
    """成功判据：**夹住 且 抬到位**（与 `outputs/infer_so101` 的"抬起"口径一致）。

    夹住用 `holds_object`（位置 OR 负载，第 38 周 0.6 实测标定），**不用**
    `GRASP_CLOSED = 10.0`：那是早期的宽阈值，实测夹持位只有 6.69，
    用它会反过来把"夹住"判成"没夹住"（真机复验时正是靠新判据才判对的）。
    """
    held = holds_object(
        float(state.get("gripper.pos", 0.0)),
        load,
        empty_grip_baseline=empty_grip_baseline,
        pos_margin=margin,
    )
    lifted = float(state.get("wrist_flex.pos", 0.0)) >= LIFT_WRIST
    return bool(held and lifted)


def safe_close(engine: Any) -> bool:
    """安全断开引擎：**收尾异常不得覆盖任务结论**。

    真机实测（2026-10-18）：夹爪舵机过载时 `Torque_Enable=0` 会抛异常，若直接放在
    `finally` 里，会把"任务成功"的返回码覆盖成 1 → 外部脚本误读为失败。
    """
    try:
        engine.close()
        return True
    except Exception as exc:  # noqa: BLE001 - 收尾失败只告警
        logging.warning("收尾断开失败（不影响任务判定）：%s", exc)
        return False


def should_park(success: bool, *, no_park: bool, park_on_success: bool) -> bool:
    """收尾是否执行复位。

    - `--no-park`：一律不复位（任何结果都保持位姿）。
    - 成功：默认**不复位**（保留"夹住并抬起"的终态，便于现场核验与连续评测）；
      只有显式 `--park-on-success` 才复位。
    - 失败：**必复位**（安全默认，避免停在下探/贴近桌面的危险姿态）。
    """
    if no_park:
        return False
    return (not success) or park_on_success


def success_checker(engine: Any) -> Callable[[Mapping[str, float]], bool]:
    """闭环成功判据：**位置为主**（夹住且抬起），负载只用于诊断。

    **为什么不用 `holds_object` 的"位置 OR 负载"**：真机实测发现，策略把夹爪指令闭合
    但实际夹空时，舵机持续顶限位，`|load|` 会超过 150‰ → 负载分支给出**假阳性**
    （2026-10-17 一次 600 步运行被判成功，人工复核确认"没夹到物体"，末态
    `gripper=3.77` 恰是空闭基准 ≈3.92）。而真夹住螺丝刀时 `gripper.pos ≈ 6.7–9.8`
    → **位置判据对本题可靠**。薄件（pos 不升高）的漏判风险由人工/负载遥测兜底，
    不再让它直接决定 `success`。

    **位置也必须是"带"而不是"只比空闭高"**：2026-10-18 第 3 次运行里策略
    "张开到 19.25 再抬起"被判成功，人工确认没夹住 —— 故改用 `is_holding`
    的上下界（空闭基准之上、`GRIP_HOLD_MAX` 之下）。

    触发时打印分量，避免再次出现"只看到布尔 true"的情况。
    """

    def check(state: Mapping[str, float]) -> bool:
        load = gripper_load(engine)
        grip = float(state.get("gripper.pos", 0.0))
        held = is_holding(grip)
        lifted = float(state.get("wrist_flex.pos", 0.0)) >= LIFT_WRIST
        ok = bool(held and lifted)
        if ok:
            logging.info(
                "[success-trigger] wrist=%.2f grip=%.2f load=%s | hold_pos=%s hold_load=%s",
                float(state.get("wrist_flex.pos", 0.0)),
                grip,
                "NA" if load is None else f"{float(load):.0f}",
                held,
                None if load is None else abs(float(load)) > GRIP_LOAD_THRESHOLD,
            )
        return ok

    return check


def place_success_checker(
    lift_threshold: float = LIFT_WRIST,
    min_pan_delta: float = 15.0,
) -> Callable[[Mapping[str, float]], bool]:
    """放置任务的成功判据：**夹住抬起 → 横向搬运 ≥ min_pan_delta° → 张开释放**。

    为什么不能复用 `success_checker`：pick 判据在"夹住 + 抬起"那一刻就返回 True，
    循环会**在搬运之前提前结束**——对"把 X 放到左边"这类任务等于只完成了抓取。

    为什么要加横向位移：2026-10-19 实测出现"夹住 → 5 步后原地张开"被判成功
    （pan 几乎没动）。没有位移闸门时，"抓一下又松开"会冒充"搬运完成"。

    **边界**：夹爪位置只能观测到"释放"这个动作，**无法证明物体落在目标位**
    （数据与传感器都没有物体位置跟踪，见 `episode_success.py`）。因此这里的
    success 是"搬运后释放"的**代理**，落点必须由图像/人工复核。
    """

    carrying = False
    pan_at_grasp = 0.0
    max_delta = 0.0

    def check(state: Mapping[str, float]) -> bool:
        nonlocal carrying, pan_at_grasp, max_delta
        grip = float(state.get("gripper.pos", 0.0))
        wrist = float(state.get("wrist_flex.pos", 0.0))
        pan = float(state.get("shoulder_pan.pos", 0.0))
        held = is_holding(grip) and wrist >= lift_threshold
        if held:
            if not carrying:
                carrying = True
                pan_at_grasp = pan
                max_delta = 0.0
                logging.info(
                    "[place] 已夹住并抬起（pan=%.1f grip=%.2f wrist=%.1f）"
                    "→ 等待搬运 ≥%.0f° 后释放",
                    pan,
                    grip,
                    wrist,
                    min_pan_delta,
                )
            else:
                max_delta = max(max_delta, abs(pan - pan_at_grasp))
            return False
        if not carrying:
            return False
        # 夹持段结束：只有"已搬运足够远 + 张开"才算完成
        done = grip >= GRIP_HOLD_MAX and max_delta >= min_pan_delta
        if done:
            logging.info(
                "[place] 已横向搬运 %.1f° 后释放（grip=%.2f）→ 记为完成（落点需图像复核）",
                max_delta,
                grip,
            )
        elif grip >= GRIP_HOLD_MAX:
            logging.info(
                "[place] 夹爪张开但横向位移仅 %.1f°（<%.0f°）→ 不判成功（原地放下）",
                max_delta,
                min_pan_delta,
            )
        else:
            logging.info(
                "[place] 搬运中止（grip=%.2f，最大横向位移=%.1f°）", grip, max_delta
            )
        carrying = False
        max_delta = 0.0
        return done

    return check


#: 夹爪过载保护参数：连续超限多久冻结、负载回落后多久解除。
GRIP_OVERLOAD_LIMIT = GRIP_LOAD_THRESHOLD
GRIP_OVERLOAD_HOLD_S = 3.0
GRIP_OVERLOAD_RELEASE_S = 1.0


class GripperTorqueGuard:
    """夹爪过载保护：连续超限 > `hold_s` 后**冻结夹爪目标位置**（保持当前位姿）。

    设计口径（用户 2026-10-18 决策）：

    - **不是卸力**：冻结后每步仍下发"当前实际位置"，舵机持续保持 → 已夹住的物体不会掉；
    - **只是不再往里挤**：策略后续的闭合指令被忽略，避免持续顶限位把舵机顶进过载保护；
    - 负载回落持续 `release_s` 后自动解除冻结，恢复听策略指令。

    `load=None`（总线读不到）时：未冻结则不做判断；已冻结则**保持冻结**（无法证明已安全）。
    """

    def __init__(self, limit: float = GRIP_OVERLOAD_LIMIT,
                 hold_s: float = GRIP_OVERLOAD_HOLD_S,
                 release_s: float = GRIP_OVERLOAD_RELEASE_S) -> None:
        self.limit = float(limit)
        self.hold_s = float(hold_s)
        self.release_s = float(release_s)
        self.frozen = False
        self._over_since: float | None = None
        self._clear_since: float | None = None

    def update(self, load: float | None, grip_pos: float, now: float) -> float | None:
        """返回 `None` = 不干预；返回数值 = 本步应把夹爪目标冻结在该位置。"""
        if load is None:
            return float(grip_pos) if self.frozen else None

        over = abs(float(load)) > self.limit
        if not self.frozen:
            if over:
                if self._over_since is None:
                    self._over_since = now
                elif now - self._over_since >= self.hold_s:
                    self.frozen = True
                    logging.warning(
                        "夹爪过载持续 %.1fs（|load|=%.0f‰ > %.0f‰）→ 冻结夹爪在 %.2f，保持当前位姿",
                        now - self._over_since, abs(float(load)), self.limit, grip_pos,
                    )
            else:
                self._over_since = None
        else:
            if over:
                self._clear_since = None
            elif self._clear_since is None:
                self._clear_since = now
            elif now - self._clear_since >= self.release_s:
                self.frozen = False
                self._over_since = None
                self._clear_since = None
                logging.info("夹爪负载回落持续 %.1fs → 解除冻结", self.release_s)
        return float(grip_pos) if self.frozen else None


def run_vla_loop(
    *,
    policy: Any,
    engine: Any,
    instruction: str,
    steps: int,
    prepare: Callable[..., Mapping[str, Any]],
    post: Callable[..., np.ndarray],
    success_fn: Callable[[Mapping[str, float]], bool],
    reset_per_step: bool = False,
    state_keys: Sequence[str] = tuple(STATE_NAMES),
    sleep_fn: Callable[[], None] | None = None,
    on_step: Callable[[int, Mapping[str, float]], None] | None = None,
    guard: "GripperTorqueGuard | None" = None,
    now_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """SmolVLA 闭环主循环（**依赖全部注入**，故可离线测试）。

    每步：读 state + 两路帧 → `prepare` 组 batch →（可选 `policy.reset()`）→
    `select_action` → `post` 转关节量 → `engine.send_action` → 判成功即提前结束。

    **缺图必须显式暴露**：`SmolVLA` 是视觉策略，若两路帧都拿不到仍照常推理，
    等于喂了空图，动作会莫名其妙地差——故这里把缺失的键记进返回值，由调用方决定是否中止。
    """
    if steps <= 0:
        raise ValueError("steps 必须是正整数")
    state_keys = tuple(state_keys)
    grip_index = state_keys.index("gripper.pos") if "gripper.pos" in state_keys else None
    sent = 0
    success = False
    last_state: dict[str, float] = {}
    missing_images: tuple[str, ...] = ()
    for step in range(int(steps)):
        state = engine.read_state()
        images = engine.read_images() or {}
        images = {key: value for key, value in images.items() if value is not None}
        missing_images = tuple(key for key in IMAGE_KEYS if key not in images)
        vector = np.asarray(
            [float(state.get(name, 0.0)) for name in state_keys], dtype=np.float32
        )
        batch = prepare(vector, images, instruction)
        if reset_per_step:
            policy.reset()
        action = post(policy.select_action(batch), action_dim=len(state_keys))
        targets = np.asarray(action, dtype=float).reshape(-1)
        if targets.shape != (len(state_keys),):
            raise ValueError(
                f"策略输出维度 {targets.shape} 与关节数 {len(state_keys)} 不符"
            )
        if guard is not None and grip_index is not None:
            frozen = guard.update(
                gripper_load(engine), float(state.get("gripper.pos", 0.0)), now_fn()
            )
            if frozen is not None:  # 过载保护：夹爪保持当前位置，不跟随策略继续闭合
                targets[grip_index] = frozen
        engine.send_action(dict(zip(state_keys, targets.tolist())))
        sent += 1
        if sleep_fn is not None:
            sleep_fn()
        last_state = dict(engine.read_state())
        if success_fn(last_state):
            success = True
            break
        if on_step is not None:
            on_step(step, last_state)
    return {
        "steps_sent": sent,
        "success": success,
        "final_state": last_state,
        "missing_images": missing_images,
    }


def run_dry(args: argparse.Namespace) -> int:
    """FakeVLAPolicy + 合成状态/帧：验证 prepost 与主循环形态（不加载真实模型）。"""
    from real_executor import MockEngine
    from vla_prepost import FakeVLAPolicy, smolvla_post, smolvla_prepare

    policy = FakeVLAPolicy(action_dim=len(STATE_NAMES))
    engine = MockEngine(with_images=True)
    summary = run_vla_loop(
        policy=policy,
        engine=engine,
        instruction=args.instruction,
        steps=args.steps,
        reset_per_step=args.reset_per_step,
        prepare=lambda state, images, task: smolvla_prepare(state, images, task),
        post=smolvla_post,
        success_fn=lambda state: False,  # dry-run 只看链路，不判成功
    )
    print(f"[dry-run] 策略调用 {policy.calls} 次，reset {policy.reset_calls} 次，"
          f"发出动作 {summary['steps_sent']} 条，末状态={_brief(summary['final_state'])}")
    print("[dry-run] SmolVLA 推理链路 OK（未加载真实模型，仅框架验证）")
    return 0


def _brief(state: Mapping[str, float]) -> str:
    keys = ("wrist_flex.pos", "gripper.pos")
    return "{" + ", ".join(f"{key}={round(float(state.get(key, 0.0)), 2)}" for key in keys) + "}"


def run_real(args: argparse.Namespace) -> int:
    """真机闭环：归位 → 加载 SmolVLA → 主循环 → 记录结果 → 收尾复位。

    返回进程退出码（0 = 成功、1 = 未成功）——真机评测常放在脚本里跑，
    用退出码比让人读日志可靠。
    """
    if not args.model_dir or not args.dataset_root or not args.dataset_repo:
        raise SystemExit(
            "真机推理需要 --model-dir（SmolVLA checkpoint）、--dataset-root 与 --dataset-repo。"
            "SmolVLA checkpoint 需先微调（或用 lerobot/smolvla_base）。框架链路请用 --dry-run 验证。"
        )
    from real_executor import RealEngine, RealSkillExecutor
    from vla_prepost import smolvla_post, smolvla_prepare

    engine = RealEngine(
        args.model_dir,
        args.dataset_root,
        args.dataset_repo,
        port=args.port,
        # 显式传相机索引：Windows 枚举会漂移，且**索引错配不会报错、只会让策略瞎跑**
        cam_idx={"fixed": args.fixed_cam, "handeye": args.handeye_cam},
        max_relative_target=args.max_relative_target,
        # VLA 自带策略：不要顺手把 ACT checkpoint 也加载进来（那是另一条路径）
        load_act_policy=False,
    )
    executor = RealSkillExecutor(engine)
    summary: dict[str, Any] | None = None
    try:
        if args.no_home:
            logging.warning("跳过归位（--no-home）：请自行确认当前位姿在训练分布内")
        elif not engine.home(timeout_s=args.home_timeout, include_gripper=args.home_gripper):
            # 不在采集初始姿态就开跑，等于把分布外观测喂给策略 —— 结果无法解释
            logging.error("归位失败：当前位姿不在采集初始姿态，拒绝继续（可用 --no-home 强制）")
            return 1

        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的明确提示
            logging.error("加载 SmolVLA 需要 lerobot 的 smolvla 策略模块：%s", exc)
            return 1
        policy = SmolVLAPolicy.from_pretrained(args.model_dir)
        policy.to(args.device)
        policy.eval()

        run_dir = Path("runs/week38/vla_runs") / time.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        logging.info("本轮轨迹/存图目录：%s", run_dir)

        def _on_step(i: int, state: Mapping[str, float]) -> None:
            """逐帧遥测：位置/负载/两个分支/是否抬起 —— 用于分辨"真夹住"与"空夹顶限位"。"""
            load = engine.read_load().get(GRIPPER_LOAD_KEY)
            grip = float(state.get("gripper.pos", 0.0))
            logging.info(
                "[step %d] wrist=%.1f grip=%.2f load=%s hold_pos=%s hold_load=%s lifted=%s",
                i,
                float(state.get("wrist_flex.pos", 0.0)),
                grip,
                "NA" if load is None else f"{float(load):.0f}",
                is_holding(grip),
                None if load is None else abs(float(load)) > GRIP_LOAD_THRESHOLD,
                float(state.get("wrist_flex.pos", 0.0)) >= LIFT_WRIST,
            )
            if i % 50 == 0:
                import cv2

                for key, img in (engine.read_images() or {}).items():
                    if img is not None:
                        cv2.imwrite(
                            str(run_dir / f"step{i:04d}_{key.split('.')[-1]}.png"),
                            np.asarray(img)[:, :, ::-1],
                        )

        summary = run_vla_loop(
            policy=policy,
            engine=engine,
            instruction=args.instruction,
            steps=args.steps,
            reset_per_step=args.reset_per_step,
            prepare=lambda state, images, task: smolvla_prepare(
                state, images, task, device=args.device, image_keys=IMAGE_KEYS
            ),
            post=smolvla_post,
            success_fn=(
                success_checker(engine)
                if args.success_mode == "pick"
                else place_success_checker()
            ),
            sleep_fn=lambda: engine._base.busy_wait(engine._base.DT),  # noqa: SLF001 - 与采集同频
            # 逐帧遥测（原来只在成功时给末状态，无法分辨"空夹顶限位"造成的假阳性）
            on_step=_on_step,
            # 过载保护：夹爪连续超限 3 s 后冻结在当前位置（保持夹持、不再往里挤）
            guard=GripperTorqueGuard(),
        )
        if summary["missing_images"]:
            logging.warning("以下相机键缺失，本轮推理的视觉输入不完整：%s", summary["missing_images"])
        print(f"[vla] 发出动作 {summary['steps_sent']} 条｜"
              f"成功={'是' if summary['success'] else '否'}｜末状态={_brief(summary['final_state'])}")
        return 0 if summary["success"] else 1
    finally:
        success = bool(summary and summary.get("success"))
        if should_park(success, no_park=args.no_park, park_on_success=args.park_on_success):
            executor.park(include_gripper=args.home_gripper)
        elif args.no_park:
            logging.warning("跳过收尾复位（--no-park）")
        else:
            logging.info("任务成功：保持当前位姿不复位（--park-on-success 可强制复位）")
        safe_close(engine)


def main() -> int:
    parser = argparse.ArgumentParser(description="SmolVLA 推理（阶段 F S12）")
    parser.add_argument("--dry-run", action="store_true",
                        help="FakeVLAPolicy + Mock 状态，验证 prepost/循环链路")
    parser.add_argument("--model-dir", default="", help="SmolVLA checkpoint 目录")
    parser.add_argument("--instruction", default="夹起桌面上的螺丝刀")
    parser.add_argument(
        "--success-mode",
        choices=("pick", "place"),
        default="pick",
        help="成功判据：pick=夹住且抬起；place=夹住抬起后张开释放（搬运类任务，"
             "避免在抓取瞬间提前结束；落点仍需图像/人工复核）",
    )
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--dataset-repo", default="")
    parser.add_argument("--port", default="COM6", help="SO-101 follower 串口")
    parser.add_argument(
        "--fixed-cam",
        type=int,
        default=CAMERA_SLOTS["fixed"],
        help=f"fixed（俯拍）相机索引，默认 {CAMERA_SLOTS['fixed']}（取自 camera_slots.py）；"
        "**每次上机前用 run_camera_check.py 核图确认**，Windows 枚举会漂移",
    )
    parser.add_argument(
        "--handeye-cam",
        type=int,
        default=CAMERA_SLOTS["handeye"],
        help=f"handeye（腕部）相机索引，默认 {CAMERA_SLOTS['handeye']}（取自 camera_slots.py）",
    )
    parser.add_argument("--device", default="cpu", help="推理设备（真机默认 cpu）")
    parser.add_argument("--max-relative-target", type=float, default=8.0,
                        help="单步关节增量限幅（安全上限，见 S2）")
    parser.add_argument("--home-timeout", type=float, default=45.0, help="归位超时（秒）")
    parser.add_argument("--home-gripper", action="store_true",
                        help="归位时也把夹爪拉到数据集初始位置（默认否：只复位 5 个臂关节，"
                             "避免反复往闭合方向硬顶导致舵机过载）")
    parser.add_argument("--no-home", action="store_true", help="跳过归位（谨慎：易喂出分布外观测）")
    parser.add_argument("--no-park", action="store_true", help="跳过收尾复位")
    parser.add_argument("--park-on-success", action="store_true",
                        help="成功时也执行收尾复位（默认否：成功后保持位姿便于核验；失败时仍会复位）")
    parser.add_argument("--reset-per-step", action="store_true",
                        help="每步 policy.reset() 强制重推（默认 false 用 action queue）")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"闭环步数（默认 {DEFAULT_STEPS}；必须 ≥ 最短演示集 340 帧）")
    args = parser.parse_args()

    return run_dry(args) if args.dry_run else run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
