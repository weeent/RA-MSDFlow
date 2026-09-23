"""RA-MSDFlow 的 flow_path 模块。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .feature_preprocess import tensor_statistics


def _as_batch_time(time: Tensor | float, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """执行 `_as_batch_time` 所需的处理。"""

    if not isinstance(time, Tensor):
        time = torch.tensor(time, device=device, dtype=dtype)
    time = time.to(device=device, dtype=dtype)
    if time.ndim == 0:
        time = time.expand(batch_size)
    elif time.ndim == 1 and time.numel() == 1:
        time = time.expand(batch_size)
    elif time.ndim != 1 or time.numel() != batch_size:
        raise ValueError(f"time must be scalar or [B={batch_size}], got {tuple(time.shape)}")
    if not torch.isfinite(time).all() or time.amin().item() < 0.0 or time.amax().item() > 1.0:
        raise ValueError("Flow Matching time must be finite and lie in [0, 1]")
    return time


@dataclass(slots=True)
class ReversedFlowBatch:
    """`ReversedFlowBatch` 组件。"""

    x0: Tensor
    noise: Tensor
    time: Tensor
    xt: Tensor
    target_velocity: Tensor

    def summary(self) -> dict[str, object]:
        return {
            "time": {
                "min": float(self.time.min().item()),
                "mean": float(self.time.mean().item()),
                "max": float(self.time.max().item()),
            },
            "x0": tensor_statistics(self.x0),
            "noise": tensor_statistics(self.noise),
            "xt": tensor_statistics(self.xt),
            "target_velocity": tensor_statistics(self.target_velocity),
        }


class ReversedFlowMatching:
    """`ReversedFlowMatching` 组件。"""

    def sample(
        self, x0: Tensor, time: Tensor | float, noise: Tensor | None = None
    ) -> ReversedFlowBatch:
        if x0.ndim != 4:
            raise ValueError(f"x0 must be [B,C,H,W], got {tuple(x0.shape)}")
        if not torch.is_floating_point(x0):
            raise TypeError("x0 must be floating point")
        # 步骤 1：按当前协议处理。
        noise = torch.randn_like(x0) if noise is None else noise.to(device=x0.device, dtype=x0.dtype)
        if noise.shape != x0.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} does not match x0 {tuple(x0.shape)}")
        # 步骤 2：按当前协议处理。
        time_vector = _as_batch_time(time, x0.shape[0], device=x0.device, dtype=x0.dtype)
        time_view = time_vector[:, None, None, None]
        # 步骤 3：按当前协议处理。
        xt = (1.0 - time_view) * x0 + time_view * noise
        target_velocity = noise - x0
        return ReversedFlowBatch(x0=x0, noise=noise, time=time_vector, xt=xt, target_velocity=target_velocity)

    @staticmethod
    def mse_loss(predicted_velocity: Tensor, target_velocity: Tensor) -> Tensor:
        """执行 `mse_loss` 所需的处理。"""

        if predicted_velocity.shape != target_velocity.shape:
            raise ValueError(
                f"velocity shape {tuple(predicted_velocity.shape)} does not match target {tuple(target_velocity.shape)}"
            )
        return F.mse_loss(predicted_velocity, target_velocity)

    @staticmethod
    def euler(xt: Tensor, velocity: Tensor, dt: float | Tensor) -> Tensor:
        """执行 `euler` 所需的处理。"""

        if xt.shape != velocity.shape:
            raise ValueError("Euler state and velocity must have identical shapes")
        return xt + velocity * dt

    @staticmethod
    def endpoint_errors(batch: ReversedFlowBatch) -> dict[str, float]:
        """执行 `endpoint_errors` 所需的处理。"""

        zeros = batch.time == 0
        ones = batch.time == 1
        errors: dict[str, float] = {}
        if zeros.any():
            errors["t0_max_abs_error"] = float((batch.xt[zeros] - batch.x0[zeros]).abs().max().item())
        if ones.any():
            errors["t1_max_abs_error"] = float((batch.xt[ones] - batch.noise[ones]).abs().max().item())
        return errors
