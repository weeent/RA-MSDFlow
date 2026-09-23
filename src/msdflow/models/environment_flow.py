"""正常环境视图之间的配对条件 Flow Matching。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .smooth_velocity import SmoothEnvironmentVelocity, SmoothVelocityOutput


def _batch_time(time: Tensor | float | None, reference: Tensor) -> Tensor:
    batch = reference.shape[0]
    if time is None:
        return torch.rand(batch, device=reference.device, dtype=reference.dtype)
    value = torch.as_tensor(time, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0 or (value.ndim == 1 and value.numel() == 1):
        value = value.reshape(1).expand(batch)
    if value.shape != (batch,) or value.min().item() < 0 or value.max().item() > 1:
        raise ValueError(f"time must be scalar or [B={batch}] in [0,1]")
    return value


@dataclass(slots=True)
class PairedFlowPath:
    feature_from: Tensor
    feature_to: Tensor
    time: Tensor
    feature_t: Tensor
    target_velocity: Tensor


@dataclass(slots=True)
class EnvironmentFlowOutput:
    loss: Tensor
    path: PairedFlowPath
    prediction: SmoothVelocityOutput

    def summary(self) -> dict[str, object]:
        return {
            "loss": float(self.loss.detach().item()),
            "time_min": float(self.path.time.min().item()),
            "time_mean": float(self.path.time.mean().item()),
            "time_max": float(self.path.time.max().item()),
            "target_velocity_mean_abs": float(self.path.target_velocity.detach().abs().mean().item()),
            "prediction": self.prediction.summary(),
        }


class PairedEnvironmentFlow(nn.Module):
    """学习同一正常物体在两个环境条件之间的特征运输。"""

    def __init__(self, velocity: SmoothEnvironmentVelocity) -> None:
        super().__init__()
        self.velocity = velocity

    def sample_path(self, feature_from: Tensor, feature_to: Tensor, time: Tensor | float | None = None) -> PairedFlowPath:
        if feature_from.shape != feature_to.shape or feature_from.ndim != 4:
            raise ValueError("paired feature endpoints must have identical [B,C,H,W] shapes")
        time_vector = _batch_time(time, feature_from)
        view = time_vector[:, None, None, None]
        # 直线路径的解析目标速度恒为 z_B-z_A；无需数值求导。
        feature_t = (1.0 - view) * feature_from + view * feature_to
        return PairedFlowPath(feature_from, feature_to, time_vector, feature_t, feature_to - feature_from)

    def forward(
        self,
        feature_from: Tensor,
        feature_to: Tensor,
        environment_from: Tensor,
        environment_to: Tensor,
        *,
        time: Tensor | float | None = None,
    ) -> EnvironmentFlowOutput:
        path = self.sample_path(feature_from, feature_to, time)
        prediction = self.velocity.forward_with_details(
            path.feature_t, path.time, environment_from, environment_to
        )
        loss = F.mse_loss(prediction.velocity, path.target_velocity)
        return EnvironmentFlowOutput(loss=loss, path=path, prediction=prediction)

    def transport(
        self,
        feature: Tensor,
        environment_from: Tensor,
        environment_to: Tensor,
        *,
        steps: int = 4,
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """用固定步长 Euler 积分把特征运输到目标正常环境中心。"""

        if steps <= 0:
            raise ValueError("steps must be positive")
        state = feature
        trajectory = [state] if return_trajectory else []
        dt = 1.0 / steps
        for step in range(steps):
            # 中点时间比左端点 Euler 对平滑速度场更稳定，但更新仍保持一阶显式形式。
            time = torch.full(
                (feature.shape[0],), (step + 0.5) * dt, device=feature.device, dtype=feature.dtype
            )
            state = state + dt * self.velocity(state, time, environment_from, environment_to)
            if return_trajectory:
                trajectory.append(state)
        return (state, trajectory) if return_trajectory else state
