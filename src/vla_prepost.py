"""SmolVLA 推理预处理/后处理薄封装（阶段 F S12，SO-101）。

背景：lerobot 0.3.4 的 `processor_smolvla.py` 不存在（week18 探针遗留 bug），
`SmolVLAPolicy._prepare_batch / prepare_images / prepare_state / prepare_language`
已内联归一化与 resize。本模块只负责"真机 numpy/uint8 观测 → 策略可吃的 tensor batch"
与"策略输出 → numpy 动作"，不重复实现归一化，避免随上游版本漂移。

契约（依据本机 modeling_smolvla.py 实测）：
- select_action(batch) 输入键：observation.state(float32)、observation.images.*
  (float [0,1], 可为 4D (B,C,H,W) 或 5D (B,T,C,H,W))、task(str)
- 输出：已 unnormalize 的 (B, action_dim) float32 tensor
- 图像长宽任意：prepare_images 内部按 config.resize_imgs_with_padding 统一处理
"""

from __future__ import annotations

from typing import Any

import numpy as np

from real_observation import STATE_NAMES


def _to_tensor(x: Any) -> Any:
    import torch

    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def smolvla_prepare(
    state: np.ndarray | dict[str, float],
    images: dict[str, np.ndarray],
    task: str,
    *,
    device: str = "cpu",
    image_keys: tuple[str, ...] = ("observation.images.fixed", "observation.images.handeye"),
) -> dict[str, Any]:
    """把真机观测组装成 SmolVLAPolicy.select_action 的 batch。

    Args:
        state: 6D 关节状态（ndarray 或 {name: value} dict）。
        images: {"observation.images.fixed": uint8 HWC RGB, ...}，缺键则跳过。
        task: 中文/英文任务指令文本。

    Returns:
        dict: 含 observation.state (1,6)、存在的 observation.images.* (1,3,H,W)、task。
    """
    import torch

    if isinstance(state, dict):
        state = np.array([state.get(n, 0.0) for n in STATE_NAMES], dtype=np.float32)

    batch: dict[str, Any] = {
        "observation.state": torch.as_tensor(state, dtype=torch.float32, device=device)[None],
        "task": task,
    }
    for key in image_keys:
        img = images.get(key)
        if img is None:
            continue
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        # HWC uint8 -> CHW float [0,1]
        t = torch.as_tensor(arr, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
        batch[key] = t[None]
    return batch


def smolvla_post(action_tensor: Any, action_dim: int = 6) -> np.ndarray:
    """把策略输出 (B, action_dim) tensor 转成单条 numpy 动作向量。"""
    arr = _to_tensor(action_tensor).detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[0]
    return np.asarray(arr[:action_dim], dtype=np.float32).reshape(-1)


class FakeVLAPolicy:
    """dry-run 用假策略：验证 prepost/主循环链路，不加载真实模型。

    行为可编程：record 每步观察计数；返回与 state 同维、接近 home 的确定性动作。
    """

    def __init__(self, action_dim: int = 6) -> None:
        self.action_dim = action_dim
        self.calls = 0
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def select_action(self, batch: dict[str, Any]):
        import torch

        self.calls += 1
        out = torch.zeros(1, self.action_dim, dtype=torch.float32)
        if "observation.state" in batch:
            st = batch["observation.state"]
            if st.ndim >= 2:
                out = out + st[:, -1:] if st.ndim > 2 else st[:, :self.action_dim].clone()
        return out
