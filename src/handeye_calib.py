"""手眼标定的**工程化封装**（第 38 周 P0 / 子任务 1.4）。

`real_geometry.solve_handeye` 只做"给定 (A_i, B_i) 相对运动对 → 解 X"。本模块补齐它两侧：

- **输入侧**：从"机器人末端绝对位姿 + 相机绝对位姿"算出相对运动对（这才是现场能采集到的东西），
  以及 npz 读写；
- **输出侧**：诊断指标 —— 逐对残差、留一法（leave-one-out）残差、自检用的合成数据。

## ⚠️ 本模块**不含采集**（设计里的阻塞点，如实记录）

现场采集需要 **FK（正运动学）** 或标记物检测才能拿到"机器人末端绝对位姿 / 相机（标记物）绝对位姿"，
本仓库**都没有**（训练在仿真、真机侧只有关节读数，没有 URDF/FK）。所以：

- `--pairs` 的 npz 必须由**外部**产出（有 FK 的环境，或标定板/ArUco 检测）；
- `--self-test` 用合成数据把"求解 + 残差 + 留一法 + 门禁"整条链路测通，
  保证外部数据一到就能直接算，而不是那时才发现代码有问题。

这也是设计把 1.4 标为"条件启用"的原因：它只在 B3（双视角三角化）真正要做时才需要，
而 B3 目前连第二视角都还没有（见 `ablation_dataset` 的说明）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from real_geometry import GeometryError, HandEyeCalibration, solve_handeye

#: npz 里两把键（绝对位姿，4×4 齐次矩阵，形状 [N,4,4]）。
ROBOT_POSES_KEY = "robot_poses"
CAMERA_POSES_KEY = "camera_poses"
PAIRS_SCHEMA_VERSION = "week38_handeye_pairs_v1"

#: 自检/门禁默认阈值：旋转 2°、平移 5 mm（手眼标定在桌面级作业里的经验量级）。
DEFAULT_MAX_ROT_DEG = 2.0
DEFAULT_MAX_TRANS_MM = 5.0
#: 留一法残差相对"全量拟合残差"的容许倍数：超过说明**存在离群对**（数据有问题，不是模型不准）。
LOO_OUTLIER_FACTOR = 5.0
#: 最少位姿数：解 AX=XB 至少要 2 组相对运动 = 3 个位姿，且 3 个位姿时旋转方程仍可能退化。
MIN_POSES = 3


class CalibrationError(RuntimeError):
    """标定输入/流程契约错误。"""


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """4×4 刚体变换求逆（用 R^T 而不是通用求逆：更快也更稳）。"""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise CalibrationError("变换必须是 4×4 有限数值矩阵")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise CalibrationError("变换的旋转部分不是正交矩阵（R^T R ≠ I）")
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverse


def compose_transform(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """4×4 变换相乘 ``first @ second``。"""
    return np.asarray(first, dtype=float) @ np.asarray(second, dtype=float)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """旋转矩阵对应的转角（度）。数值截断到 [0, 180]，避免浮点越界导致 acos 出 nan。"""
    matrix = np.asarray(rotation, dtype=float)
    cosine = (np.trace(matrix[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def relative_motions(poses: Sequence[np.ndarray]) -> list[np.ndarray]:
    """绝对位姿序列 → 相邻**相对运动** ``inv(P_i) @ P_{i+1}``。

    AX=XB 的 A/B 都是相对运动（绝对位姿不满足该方程），这一步是现场数据到求解器的桥。
    """
    if len(poses) < 2:
        raise CalibrationError("至少需要 2 个位姿才能构造相对运动")
    matrices = [np.asarray(pose, dtype=float) for pose in poses]
    return [compose_transform(invert_transform(current), nxt)
            for current, nxt in zip(matrices, matrices[1:])]


def build_pairs(
    robot_poses: Sequence[np.ndarray],
    camera_poses: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """机器人/相机绝对位姿 → ``(A_i, B_i)`` 相对运动对。"""
    if len(robot_poses) != len(camera_poses):
        raise CalibrationError(
            f"机器人位姿数 {len(robot_poses)} 与相机位姿数 {len(camera_poses)} 不一致"
        )
    if len(robot_poses) < MIN_POSES:
        raise CalibrationError(
            f"至少需要 {MIN_POSES} 个位姿（{MIN_POSES - 1} 组相对运动），实际 {len(robot_poses)}"
        )
    return list(zip(relative_motions(robot_poses), relative_motions(camera_poses)))


def calibration_matrix(calibration: HandEyeCalibration) -> np.ndarray:
    """把 `HandEyeCalibration` 还原成 4×4 齐次矩阵。"""
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(calibration.R, dtype=float)
    matrix[:3, 3] = np.asarray(calibration.t, dtype=float).reshape(3)
    return matrix


def motion_residuals(
    calibration: HandEyeCalibration,
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """逐对残差：用解出的 X 预测 ``B̂ = X⁻¹ A X`` 并与实测 B 比。

    返回旋转（度）与平移（毫米）的平均/最大值 + 逐对明细。
    """
    matrix = calibration_matrix(calibration)
    inverse = invert_transform(matrix)
    rotations: list[float] = []
    translations: list[float] = []
    for index, (motion_a, motion_b) in enumerate(pairs):
        predicted = compose_transform(compose_transform(inverse, motion_a), matrix)
        measured = np.asarray(motion_b, dtype=float)
        rotations.append(rotation_angle_deg(np.asarray(measured[:3, :3]).T @ predicted[:3, :3]))
        translations.append(float(np.linalg.norm(measured[:3, 3] - predicted[:3, 3])) * 1000.0)
    if not rotations:
        raise CalibrationError("没有可评估的运动对")
    return {
        "rot_deg_mean": float(np.mean(rotations)),
        "rot_deg_max": float(np.max(rotations)),
        "trans_mm_mean": float(np.mean(translations)),
        "trans_mm_max": float(np.max(translations)),
        "per_pair": [
            {"pair": index, "rot_deg": round(rot, 6), "trans_mm": round(trans, 6)}
            for index, (rot, trans) in enumerate(zip(rotations, translations))
        ],
    }


def leave_one_out(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    method: str = "ax_xb",
) -> dict[str, Any]:
    """留一法：每次留出一对不求，用它检验其余对解出的 X。

    全量拟合的残差会被**离群对**拉小（它自己参与了拟合），留一法能把它暴露出来 ——
    这是"数据有问题"与"标定精度不够"的分界线。
    """
    if len(pairs) < MIN_POSES:
        raise CalibrationError(f"留一法至少需要 {MIN_POSES} 对相对运动，实际 {len(pairs)}")
    rotations: list[float] = []
    translations: list[float] = []
    for hold_out in range(len(pairs)):
        rest = [pair for index, pair in enumerate(pairs) if index != hold_out]
        try:
            calibration = solve_handeye(rest, method=method)
        except GeometryError as exc:
            raise CalibrationError(f"留一法在第 {hold_out} 对上求解失败：{exc}") from exc
        residuals = motion_residuals(calibration, [pairs[hold_out]])
        rotations.append(residuals["rot_deg_max"])
        translations.append(residuals["trans_mm_max"])
    worst = int(np.argmax(translations))
    return {
        "rot_deg_mean": float(np.mean(rotations)),
        "rot_deg_max": float(np.max(rotations)),
        "trans_mm_mean": float(np.mean(translations)),
        "trans_mm_max": float(np.max(translations)),
        "worst_pair": worst,
    }


@dataclass(frozen=True)
class CalibrationReport:
    """标定结论（含门禁判定与所有诊断量）。"""

    method: str
    n_poses: int
    n_pairs: int
    residual_m: float
    residuals: dict[str, Any]
    leave_one_out: dict[str, Any]
    max_rot_deg: float
    max_trans_mm: float
    passed: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PAIRS_SCHEMA_VERSION,
            "method": self.method,
            "n_poses": self.n_poses,
            "n_pairs": self.n_pairs,
            "residual_m": self.residual_m,
            "thresholds": {"max_rot_deg": self.max_rot_deg, "max_trans_mm": self.max_trans_mm},
            "residuals": self.residuals,
            "leave_one_out": self.leave_one_out,
            "passed": self.passed,
            "warnings": list(self.warnings),
        }


def calibrate(
    robot_poses: Sequence[np.ndarray],
    camera_poses: Sequence[np.ndarray],
    *,
    method: str = "ax_xb",
    max_rot_deg: float = DEFAULT_MAX_ROT_DEG,
    max_trans_mm: float = DEFAULT_MAX_TRANS_MM,
) -> CalibrationReport:
    """完整流程：绝对位姿 → 相对运动对 → 求解 → 残差/留一法 → 门禁。"""
    pairs = build_pairs(robot_poses, camera_poses)
    try:
        calibration = solve_handeye(pairs, method=method)
    except GeometryError as exc:
        raise CalibrationError(f"AX=XB 求解失败：{exc}") from exc
    residuals = motion_residuals(calibration, pairs)
    warnings: list[str] = []
    try:
        loo = leave_one_out(pairs, method=method)
    except CalibrationError as exc:
        loo = {"error": str(exc)}
        warnings.append(f"留一法未完成：{exc}")
    passed = (
        residuals["rot_deg_max"] <= max_rot_deg
        and residuals["trans_mm_max"] <= max_trans_mm
    )
    if "error" not in loo and residuals["trans_mm_mean"] > 0:
        ratio = loo["trans_mm_max"] / residuals["trans_mm_mean"]
        if ratio > LOO_OUTLIER_FACTOR:
            warnings.append(
                f"留一法最差对 {loo['worst_pair']} 的平移残差是全量均值的 {ratio:.1f} 倍"
                f"（>{LOO_OUTLIER_FACTOR:g}×）→ 疑似该对是离群点，先复查数据再谈精度"
            )
    return CalibrationReport(
        method=method,
        n_poses=len(robot_poses),
        n_pairs=len(pairs),
        residual_m=float(calibration.residual_m),
        residuals=residuals,
        leave_one_out=loo,
        max_rot_deg=float(max_rot_deg),
        max_trans_mm=float(max_trans_mm),
        passed=bool(passed),
        warnings=tuple(warnings),
    )


def synthetic_poses(
    *,
    n_poses: int = 8,
    seed: int = 0,
    noise_deg: float = 0.0,
    noise_mm: float = 0.0,
    outlier_pair: int | None = None,
    outlier_mm: float = 30.0,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """造一组**已知真值 X** 的合成位姿（`--self-test` 用）。

    构造方式：任取机器人绝对位姿 ``P_i``，令相对运动满足 ``B_i = X⁻¹ A_i X``，
    再沿链累乘出相机绝对位姿 ``Q_{i+1} = Q_i B_i`` —— 这样反解必然能拿到 X，
    从而把"求解 + 残差 + 留一法"整条链路测通（含可选噪声与离群对）。
    """
    rng = np.random.default_rng(seed)
    # 真值 X：绕 z 轴 20° + 平移 (0.05, -0.02, 0.03) m
    angle = np.radians(20.0)
    truth = np.eye(4)
    truth[:3, :3] = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    truth[:3, 3] = [0.05, -0.02, 0.03]

    robot_poses: list[np.ndarray] = []
    for _ in range(n_poses):
        pose = np.eye(4)
        axis_angle = rng.normal(scale=0.5, size=3)
        theta = float(np.linalg.norm(axis_angle))
        if theta < 1e-6:  # 退化为零旋转时给一个确定的极小转轴，避免除零
            axis_angle = np.array([0.0, 0.0, 1e-6])
            theta = 1e-6
        axis = axis_angle / theta  # **必须单位化**：否则 Rodrigues 公式不再给出正交矩阵
        skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                         [-axis[1], axis[0], 0.0]])
        pose[:3, :3] = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
        pose[:3, 3] = rng.normal(scale=0.1, size=3)
        robot_poses.append(pose)

    motions_a = relative_motions(robot_poses)
    camera_poses: list[np.ndarray] = [np.eye(4)]
    for motion_a in motions_a:
        measured = compose_transform(
            compose_transform(invert_transform(truth), motion_a), truth
        )
        if noise_deg or noise_mm:
            measured = _perturb(measured, rng, noise_deg, noise_mm)
        if outlier_pair is not None and len(camera_poses) - 1 == outlier_pair:
            measured = _perturb(measured, rng, 0.0, outlier_mm)
        camera_poses.append(compose_transform(camera_poses[-1], measured))
    return robot_poses, camera_poses, truth


def _perturb(transform: np.ndarray, rng: np.random.Generator,
             noise_deg: float, noise_mm: float) -> np.ndarray:
    rotation = np.asarray(transform[:3, :3], dtype=float)
    delta = rng.normal(scale=np.radians(noise_deg), size=3) if noise_deg else np.zeros(3)
    theta = float(np.linalg.norm(delta))
    if theta > 0:
        axis = delta / theta
        skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                         [-axis[1], axis[0], 0.0]])
        rotation = (np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)) @ rotation
    translation = np.asarray(transform[:3, 3], dtype=float)
    if noise_mm:
        translation = translation + rng.normal(scale=noise_mm / 1000.0, size=3)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def write_pairs(
    path: str | Path,
    robot_poses: Sequence[np.ndarray],
    camera_poses: Sequence[np.ndarray],
) -> Path:
    """把绝对位姿写成 npz（外部采集工具按这个格式交数据即可）。"""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        schema=np.asarray(PAIRS_SCHEMA_VERSION),
        **{
            ROBOT_POSES_KEY: np.asarray(robot_poses, dtype=float),
            CAMERA_POSES_KEY: np.asarray(camera_poses, dtype=float),
        },
    )
    return target


def read_pairs(path: str | Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """读 npz；形状/键名不对时给出**可操作**的报错（而不是 numpy 的原始异常）。"""
    source = Path(path).expanduser()
    if not source.is_file():
        raise CalibrationError(f"标定数据不存在：{source}")
    with np.load(source, allow_pickle=False) as payload:
        missing = [key for key in (ROBOT_POSES_KEY, CAMERA_POSES_KEY) if key not in payload]
        if missing:
            raise CalibrationError(
                f"{source} 缺少键 {missing}（需要 {ROBOT_POSES_KEY} / {CAMERA_POSES_KEY}，"
                "形状均为 [N,4,4] 的绝对位姿）"
            )
        robot = np.asarray(payload[ROBOT_POSES_KEY], dtype=float)
        camera = np.asarray(payload[CAMERA_POSES_KEY], dtype=float)
    for name, poses in ((ROBOT_POSES_KEY, robot), (CAMERA_POSES_KEY, camera)):
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise CalibrationError(f"{name} 形状应为 [N,4,4]，实际 {poses.shape}")
        if not np.all(np.isfinite(poses)):
            raise CalibrationError(f"{name} 含非有限数值")
    if len(robot) < MIN_POSES:
        raise CalibrationError(f"至少需要 {MIN_POSES} 个位姿，实际 {len(robot)}")
    return [pose for pose in robot], [pose for pose in camera]


def write_report(report: CalibrationReport, out_dir: str | Path) -> tuple[Path, Path]:
    """落盘 `handeye.json` + `handeye.md`。"""
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "handeye.json"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path = directory / "handeye.md"
    lines = [
        "# 手眼标定结果（AX=XB）",
        "",
        f"- 来源：{report.method}｜位姿 {report.n_poses} 个｜相对运动对 {report.n_pairs} 组",
        f"- 求解器报告的平均平移残差：{report.residual_m * 1000:.3f} mm",
        f"- 逐对残差：旋转 mean {report.residuals['rot_deg_mean']:.3f}° / "
        f"max {report.residuals['rot_deg_max']:.3f}°，"
        f"平移 mean {report.residuals['trans_mm_mean']:.3f} mm / "
        f"max {report.residuals['trans_mm_max']:.3f} mm",
        f"- 门禁：旋转 ≤ {report.max_rot_deg:g}° 且平移 ≤ {report.max_trans_mm:g} mm → "
        f"**{'通过' if report.passed else '未通过'}**",
    ]
    if "error" not in report.leave_one_out:
        lines.append(
            f"- 留一法：旋转 max {report.leave_one_out['rot_deg_max']:.3f}°，"
            f"平移 max {report.leave_one_out['trans_mm_max']:.3f} mm"
            f"（最差对 = {report.leave_one_out['worst_pair']}）"
        )
    for warning in report.warnings:
        lines.append(f"- [警告] {warning}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
