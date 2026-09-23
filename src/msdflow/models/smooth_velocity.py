"""受低频约束的环境速度场。

速度被拆为通道仿射项 ``a⊙z+b`` 与低分辨率空间残差 ``Up(r)``。
该结构允许全局曝光/色偏和缓慢阴影变化，同时限制模型直接重绘局部高频缺陷。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sinusoidal_time_embedding(time: Tensor, dimension: int) -> Tensor:
    """将连续时间 [B] 编码为稳定的正余弦向量。"""

    if time.ndim != 1 or dimension <= 0 or dimension % 2:
        raise ValueError("time must be [B] and embedding dimension must be positive/even")
    half = dimension // 2
    frequencies = torch.exp(
        torch.arange(half, device=time.device, dtype=time.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = time[:, None] * frequencies[None, :] * 1000.0
    return torch.cat((angles.sin(), angles.cos()), dim=1)


@dataclass(slots=True)
class SmoothVelocityOutput:
    velocity: Tensor
    channel_scale: Tensor
    channel_shift: Tensor
    spatial_low_resolution: Tensor
    spatial_upsampled: Tensor

    def summary(self) -> dict[str, object]:
        def stats(value: Tensor) -> dict[str, object]:
            value = value.detach().float()
            return {
                "shape": list(value.shape),
                "mean_abs": float(value.abs().mean().item()),
                "max_abs": float(value.abs().max().item()),
                "finite": bool(torch.isfinite(value).all().item()),
            }

        return {
            "velocity": stats(self.velocity),
            "channel_scale": stats(self.channel_scale),
            "channel_shift": stats(self.channel_shift),
            "spatial_low_resolution": stats(self.spatial_low_resolution),
        }


class SmoothEnvironmentVelocity(nn.Module):
    """预测从一个环境条件运输到另一个环境条件所需的平滑特征速度。"""

    def __init__(
        self,
        in_channels: int,
        condition_dim: int,
        *,
        hidden_dim: int = 128,
        spatial_hidden_channels: int = 64,
        lowres_size: int = 4,
        max_scale_rate: float = 0.25,
    ) -> None:
        super().__init__()
        if min(in_channels, condition_dim, hidden_dim, spatial_hidden_channels, lowres_size) <= 0:
            raise ValueError("all dimensions and lowres_size must be positive")
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for sinusoidal time embedding")
        self.in_channels = int(in_channels)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.lowres_size = int(lowres_size)
        self.max_scale_rate = float(max_scale_rate)

        # 全局上下文由源/目标环境、环境差、时间和当前特征通道均值共同构成。
        self.feature_projection = nn.Linear(in_channels, hidden_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.context = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.scale_head = nn.Linear(hidden_dim, in_channels)
        self.shift_head = nn.Linear(hidden_dim, in_channels)

        # 空间分支只能在小网格上生成残差，然后双线性上采样，形成结构性低通约束。
        self.spatial_in = nn.Conv2d(in_channels, spatial_hidden_channels, 1)
        self.spatial_condition = nn.Linear(hidden_dim, spatial_hidden_channels)
        self.spatial_body = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(spatial_hidden_channels, spatial_hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(spatial_hidden_channels, in_channels, 3, padding=1),
        )
        # 零初始化使新模型初始为恒等 ODE，便于检查运输确实来自学习而非随机扰动。
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)
        nn.init.zeros_(self.spatial_body[-1].weight)
        nn.init.zeros_(self.spatial_body[-1].bias)

    def _validate(self, zt: Tensor, time: Tensor, environment_from: Tensor, environment_to: Tensor) -> None:
        if zt.ndim != 4 or zt.shape[1] != self.in_channels:
            raise ValueError(f"expected zt [B,{self.in_channels},H,W], got {tuple(zt.shape)}")
        batch = zt.shape[0]
        if time.shape != (batch,):
            raise ValueError(f"time must be [B={batch}]")
        expected = (batch, self.condition_dim)
        if environment_from.shape != expected or environment_to.shape != expected:
            raise ValueError(f"environment endpoints must both be {expected}")
        if any(value.device != zt.device for value in (time, environment_from, environment_to)):
            raise ValueError("state, time and environment tensors must share one device")

    def forward_with_details(
        self, zt: Tensor, time: Tensor, environment_from: Tensor, environment_to: Tensor
    ) -> SmoothVelocityOutput:
        self._validate(zt, time, environment_from, environment_to)
        dtype = zt.dtype
        # 步骤 1：将起点、终点及其差值编码，显式告诉网络运输方向。
        condition = torch.cat(
            (environment_from.to(dtype), environment_to.to(dtype), (environment_to - environment_from).to(dtype)), dim=1
        )
        condition_context = self.condition_projection(condition)
        time_context = sinusoidal_time_embedding(time.to(dtype), self.hidden_dim)
        feature_context = self.feature_projection(zt.mean(dim=(2, 3)))
        context = self.context(torch.cat((condition_context, time_context, feature_context), dim=1))

        # 步骤 2：通道仿射速度描述曝光/白平衡等全局变化。
        channel_scale = self.max_scale_rate * torch.tanh(self.scale_head(context))
        channel_shift = self.shift_head(context)
        affine_velocity = channel_scale[:, :, None, None] * zt + channel_shift[:, :, None, None]

        # 步骤 3：低分辨率残差描述缓慢阴影；禁止直接在原始 patch 网格自由卷积。
        target_h = min(self.lowres_size, zt.shape[-2])
        target_w = min(self.lowres_size, zt.shape[-1])
        pooled = F.adaptive_avg_pool2d(zt, (target_h, target_w))
        spatial = self.spatial_in(pooled) + self.spatial_condition(context)[:, :, None, None]
        spatial_low = self.spatial_body(spatial)
        spatial_up = F.interpolate(spatial_low, size=zt.shape[-2:], mode="bilinear", align_corners=False)
        velocity = affine_velocity + spatial_up
        if not torch.isfinite(velocity).all():
            raise RuntimeError("environment velocity contains NaN or infinity")
        return SmoothVelocityOutput(velocity, channel_scale, channel_shift, spatial_low, spatial_up)

    def forward(self, zt: Tensor, time: Tensor, environment_from: Tensor, environment_to: Tensor) -> Tensor:
        return self.forward_with_details(zt, time, environment_from, environment_to).velocity

    def summary(self) -> dict[str, object]:
        parameters = list(self.parameters())
        return {
            "model": "smooth_environment_velocity",
            "in_channels": self.in_channels,
            "condition_dim": self.condition_dim,
            "lowres_size": self.lowres_size,
            "max_scale_rate": self.max_scale_rate,
            "parameter_count": sum(item.numel() for item in parameters),
            "trainable_parameter_count": sum(item.numel() for item in parameters if item.requires_grad),
            "decomposition": "channel_affine_plus_low_frequency_spatial_residual",
        }
