"""可解释光度统计与可学习低频编码组成的混合环境描述器。

输入：未做 ImageNet 归一化、范围为 [0,1] 的 RGB 图像 ``[B,3,H,W]``。
输出：8 维固定光度统计、可选的 ``learned_dim`` 维低频编码及拼接向量。
中间产物：32x32 低频图，可用于检查编码器是否错误关注了局部缺陷。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from envfm.data.photometric_descriptor import PhotometricDescriptor


def _tensor_summary(value: Tensor) -> dict[str, object]:
    detached = value.detach().float()
    return {
        "shape": list(value.shape),
        "min": float(detached.amin().item()),
        "mean": float(detached.mean().item()),
        "max": float(detached.amax().item()),
        "std": float(detached.std(unbiased=False).item()),
        "finite": bool(torch.isfinite(detached).all().item()),
    }


@dataclass(slots=True)
class EnvironmentDescriptorOutput:
    """保留环境描述器的最终张量和关键中间结果。"""

    photo8: Tensor
    learned: Tensor
    combined: Tensor
    low_resolution: Tensor

    def summary(self) -> dict[str, object]:
        return {
            "photo8": _tensor_summary(self.photo8),
            "learned": _tensor_summary(self.learned) if self.learned.numel() else {"shape": list(self.learned.shape)},
            "combined": _tensor_summary(self.combined),
            "low_resolution": _tensor_summary(self.low_resolution),
        }


class LowFrequencyEnvironmentEncoder(nn.Module):
    """从低分辨率 RGB 中提取环境编码，尽量抑制局部缺陷泄漏。

    卷积只在 32x32 低频图上运行，随后全局平均池化。它能补充八维统计无法表示的
    阴影形状、非线性色偏和背景低频结构，但仍有意舍弃高频裂纹与划痕。
    """

    def __init__(self, output_dim: int = 16, hidden_channels: int = 24, lowres_size: int = 32) -> None:
        super().__init__()
        if output_dim <= 0 or hidden_channels <= 0 or lowres_size < 8:
            raise ValueError("output_dim/hidden_channels must be positive and lowres_size >= 8")
        self.output_dim = int(output_dim)
        self.lowres_size = int(lowres_size)
        self.network = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 5, stride=2, padding=2),
            nn.GroupNorm(4 if hidden_channels % 4 == 0 else 1, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(4 if (hidden_channels * 2) % 4 == 0 else 1, hidden_channels * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, output_dim),
        )

    def low_pass(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected RGB [B,3,H,W], got {tuple(image.shape)}")
        if not torch.is_floating_point(image) or not torch.isfinite(image).all():
            raise ValueError("environment encoder requires finite floating-point input")
        if image.amin().item() < -1e-5 or image.amax().item() > 1.0 + 1e-5:
            raise ValueError("environment encoder expects RGB values in [0,1]")
        # area 插值执行局部平均，避免简单缩放把高频缺陷混叠到低频图中。
        return F.interpolate(image, size=(self.lowres_size, self.lowres_size), mode="area")

    def forward_with_lowres(self, image: Tensor) -> tuple[Tensor, Tensor]:
        lowres = self.low_pass(image)
        return self.network(lowres), lowres

    def forward(self, image: Tensor) -> Tensor:
        return self.forward_with_lowres(image)[0]


class HybridEnvironmentDescriptor(nn.Module):
    """拼接固定 8 维统计和可选的可学习低频编码。"""

    def __init__(
        self,
        *,
        learned_dim: int = 16,
        lowres_size: int = 32,
        learned_encoder: LowFrequencyEnvironmentEncoder | None = None,
    ) -> None:
        super().__init__()
        if learned_dim < 0:
            raise ValueError("learned_dim must be non-negative")
        self.learned_dim = int(learned_dim)
        self.photo = PhotometricDescriptor(lowres_size=lowres_size)
        self.learned_encoder: LowFrequencyEnvironmentEncoder | None
        if learned_dim == 0:
            if learned_encoder is not None:
                raise ValueError("learned_encoder must be None when learned_dim=0")
            self.learned_encoder = None
        else:
            self.learned_encoder = learned_encoder or LowFrequencyEnvironmentEncoder(
                output_dim=learned_dim, lowres_size=lowres_size
            )
            if self.learned_encoder.output_dim != learned_dim:
                raise ValueError("learned encoder output dimension does not match learned_dim")

    @property
    def output_dim(self) -> int:
        return 8 + self.learned_dim

    def forward(self, image: Tensor, *, detach_learned: bool = False) -> EnvironmentDescriptorOutput:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        photo8 = self.photo(image)
        if self.learned_encoder is None:
            lowres = F.interpolate(image, size=(self.photo.lowres_size, self.photo.lowres_size), mode="area")
            learned = image.new_empty((image.shape[0], 0))
        else:
            learned, lowres = self.learned_encoder.forward_with_lowres(image)
            if detach_learned:
                learned = learned.detach()
        combined = torch.cat((photo8, learned), dim=1)
        if not torch.isfinite(combined).all():
            raise RuntimeError("hybrid environment descriptor contains NaN or infinity")
        return EnvironmentDescriptorOutput(photo8=photo8, learned=learned, combined=combined, low_resolution=lowres)


class EnvironmentStandardizer(nn.Module):
    """任意维环境向量的 train-only 均值/标准差标准化器。"""

    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("std", torch.ones(dimension))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def fitted(self) -> bool:
        return int(self.count.item()) > 0

    @torch.no_grad()
    def fit(self, values: Tensor) -> "EnvironmentStandardizer":
        values = torch.as_tensor(values, dtype=torch.float32)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != self.dimension:
            raise ValueError(f"expected non-empty [N,{self.dimension}], got {tuple(values.shape)}")
        if not torch.isfinite(values).all():
            raise ValueError("environment values contain NaN or infinity")
        self.mean.copy_(values.mean(0).to(self.mean.device))
        self.std.copy_(values.std(0, unbiased=False).clamp_min(self.eps).to(self.std.device))
        self.count.fill_(values.shape[0])
        return self

    def forward(self, values: Tensor) -> Tensor:
        if not self.fitted:
            raise RuntimeError("EnvironmentStandardizer must be fitted on normal train data")
        if values.shape[-1] != self.dimension:
            raise ValueError(f"expected last dimension {self.dimension}")
        return (values - self.mean.to(values)) / self.std.to(values).clamp_min(self.eps)

    def inverse(self, values: Tensor) -> Tensor:
        if not self.fitted:
            raise RuntimeError("EnvironmentStandardizer must be fitted first")
        return values * self.std.to(values) + self.mean.to(values)

    def summary(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "count": int(self.count.item()),
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
        }
