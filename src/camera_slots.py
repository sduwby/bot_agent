"""相机索引的**唯一事实来源**（Windows 枚举会漂移，改这里一处即可）。

## 为什么单独一个模块

`fixed` / `handeye` 的索引曾被抄在 5 个地方（`real_executor`、`vla_collect_plan`、
`run_camera_check`、`run_perception_capture`、`outputs/infer_so101`）。2026-10-17
重插 + 重启后枚举变成 `0=笔记本摄像头 / 1=fixed(俯拍) / 2=handeye(腕部)`，
5 处**同时失效且不报错**：`fixed` 会拿到腕部图、`handeye` 会拿到笔记本摄像头。
故收敛到此一处。

## 硬约束（不是"改了就行"）

真正要固定的是「**同一台物理相机 → 同一通道**」。索引只是运行时的枚举细节：

1. **每次上机前**跑 `python src/run_camera_check.py`，**逐张看图**（看画面内容，不看索引号）；
2. 确认下面每个通道确实指向那台物理相机；
3. 不对就改这里的值 —— 但**不要**为了"让某个索引可用"而把通道随便对调。

## 历史

- 2026-10-10：`fixed=2 / handeye=0`（当时 `0=handeye`、`1=笔记本`、`2=fixed`）
- 2026-10-17：核图后改为 `fixed=1 / handeye=2`（`0=笔记本`、`1=俯拍`、`2=腕部`）
"""

from __future__ import annotations

#: 通道名 → OpenCV/MSMF 相机索引（**只在这里定义**）。
CAMERA_SLOTS: dict[str, int] = {"fixed": 1, "handeye": 2}

__all__ = ["CAMERA_SLOTS"]
