#!/usr/bin/env python
"""手眼标定 CLI（第 38 周 P0 / 1.4）：求解 AX=XB 并输出残差与门禁结论。

用法::

    # 用外部采集的位姿数据（npz：robot_poses / camera_poses，各 [N,4,4] 绝对位姿）
    python src/run_handeye_calib.py --pairs runs/week38/handeye/pairs.npz `
        --out runs/week38/handeye

    # 自检：合成一组已知真值 X 的位姿，把"求解+残差+留一法+门禁"整条链路测通
    python src/run_handeye_calib.py --self-test --out runs/week38/handeye/selftest

    # 自检 + 人为噪声/离群对，确认诊断指标**真的会报警**（而不是永远说"通过"）
    python src/run_handeye_calib.py --self-test --noise-deg 0.3 --noise-mm 1.5 `
        --outlier-pair 3 --out runs/week38/handeye/selftest_noisy

**⚠️ 本 CLI 不含采集**：现场取"机器人末端绝对位姿 + 相机绝对位姿"需要 **FK** 或标记物检测，
本仓库都没有（详见 `handeye_calib` 模块开头的说明）。所以 `--pairs` 必须由外部产出；
`--self-test` 是为了保证外部数据一到就能直接算。退出码 0 = 门禁通过、1 = 未通过。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from handeye_calib import (  # noqa: E402
    DEFAULT_MAX_ROT_DEG,
    DEFAULT_MAX_TRANS_MM,
    MIN_POSES,
    CalibrationError,
    CalibrationReport,
    calibrate,
    read_pairs,
    synthetic_poses,
    write_pairs,
    write_report,
)


def render_report(report: CalibrationReport) -> str:
    lines = [
        f"来源 {report.method}｜位姿 {report.n_poses} 个｜相对运动对 {report.n_pairs} 组",
        f"求解器平均平移残差 {report.residual_m * 1000:.3f} mm",
        f"逐对残差：旋转 mean {report.residuals['rot_deg_mean']:.3f}° / "
        f"max {report.residuals['rot_deg_max']:.3f}°｜"
        f"平移 mean {report.residuals['trans_mm_mean']:.3f} mm / "
        f"max {report.residuals['trans_mm_max']:.3f} mm",
    ]
    if "error" not in report.leave_one_out:
        lines.append(
            f"留一法：旋转 max {report.leave_one_out['rot_deg_max']:.3f}°｜"
            f"平移 max {report.leave_one_out['trans_mm_max']:.3f} mm"
            f"（最差对 {report.leave_one_out['worst_pair']}）"
        )
    lines.append(
        f"门禁（旋转 ≤ {report.max_rot_deg:g}° 且平移 ≤ {report.max_trans_mm:g} mm）→ "
        f"{'通过' if report.passed else '未通过'}"
    )
    return "\n".join(f"[标定] {line}" for line in lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="手眼标定 AX=XB（含残差与留一法诊断）")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", help="标定位姿 npz（robot_poses / camera_poses，各 [N,4,4]）")
    source.add_argument("--self-test", action="store_true", help="用合成位姿自检整条链路")
    parser.add_argument("--out", default="runs/week38/handeye", help="报告输出目录")
    parser.add_argument("--method", default="ax_xb", choices=("ax_xb", "board", "aruco"),
                        help="数据来源标记（求解方式一致，仅作记录）")
    parser.add_argument("--max-rot-deg", type=float, default=DEFAULT_MAX_ROT_DEG)
    parser.add_argument("--max-trans-mm", type=float, default=DEFAULT_MAX_TRANS_MM)
    parser.add_argument("--self-test-poses", type=int, default=10,
                        help="自检生成的位姿数（留一法至少需要 4 个位姿 = 3 组相对运动才跑得起来）")
    parser.add_argument("--noise-deg", type=float, default=0.0, help="自检噪声：旋转（度）")
    parser.add_argument("--noise-mm", type=float, default=0.0, help="自检噪声：平移（毫米）")
    parser.add_argument("--outlier-pair", type=int, default=None,
                        help="自检：把第 i 对相对运动人为破坏（验证留一法会报警）")
    parser.add_argument("--save-pairs", action="store_true",
                        help="自检时把合成位姿也写成 npz（便于复现/喂给别的工具）")
    args = parser.parse_args()

    try:
        if args.self_test:
            # 生成数必须够留一法用（>=4 个位姿），否则"自检"会漏掉最关键的诊断路径
            if args.self_test_poses < MIN_POSES + 1:
                print(f"[标定] --self-test-poses 至少为 {MIN_POSES + 1}（留一法需要 ≥3 组相对运动）")
                return 1
            robot, camera, truth = synthetic_poses(
                n_poses=args.self_test_poses,
                noise_deg=args.noise_deg,
                noise_mm=args.noise_mm,
                outlier_pair=args.outlier_pair,
            )
            if args.save_pairs:
                path = write_pairs(Path(args.out) / "synthetic_pairs.npz", robot, camera)
                print(f"[标定] 合成位姿 → {path}")
            print("[标定] 真值 X（自检用，非求解结果）："
                  f"{json.dumps(truth.tolist(), ensure_ascii=False)}")
        else:
            robot, camera = read_pairs(args.pairs)
        report = calibrate(
            robot, camera, method=args.method,
            max_rot_deg=args.max_rot_deg, max_trans_mm=args.max_trans_mm,
        )
    except CalibrationError as exc:
        print(f"[标定] 失败：{exc}")
        return 1

    print(render_report(report))
    for warning in report.warnings:
        print(f"[标定][警告] {warning}")
    json_path, markdown_path = write_report(report, args.out)
    print(f"[标定] 报告 → {json_path}")
    print(f"[标定]        → {markdown_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
