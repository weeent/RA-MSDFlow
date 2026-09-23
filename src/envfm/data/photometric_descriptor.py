"""RA-MSDFlow 的 photometric_descriptor 模块。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .photometric_augment import srgb_to_linear


class PhotometricDescriptor(nn.Module):
    """`PhotometricDescriptor` 组件。"""

    def __init__(self, lowres_size: int = 32, blur_kernel_size: int = 5, blur_sigma: float = 1.0, eps: float = 1e-6):
        super().__init__()
        if lowres_size < 4:
            raise ValueError("lowres_size must be at least 4")
        if blur_kernel_size < 1 or blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be a positive odd integer")
        self.lowres_size = int(lowres_size)
        self.blur_kernel_size = int(blur_kernel_size)
        self.blur_sigma = float(blur_sigma)
        self.eps = float(eps)
        kernel = self._make_gaussian_kernel(blur_kernel_size, blur_sigma)
        self.register_buffer("gaussian_kernel", kernel.view(1, 1, blur_kernel_size, blur_kernel_size), persistent=False)
        coordinate = torch.linspace(-1.0, 1.0, lowres_size)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        denominator_x = (xx.square()).sum().clamp_min(eps)
        denominator_y = (yy.square()).sum().clamp_min(eps)
        self.register_buffer("grid_x", xx.view(1, 1, lowres_size, lowres_size), persistent=False)
        self.register_buffer("grid_y", yy.view(1, 1, lowres_size, lowres_size), persistent=False)
        self.register_buffer("denominator_x", denominator_x, persistent=False)
        self.register_buffer("denominator_y", denominator_y, persistent=False)

    @staticmethod
    def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        coordinate = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        one_dimensional = torch.exp(-0.5 * (coordinate / sigma).square())
        one_dimensional /= one_dimensional.sum()
        return torch.outer(one_dimensional, one_dimensional)

    @staticmethod
    def _as_batch(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not torch.is_floating_point(image):
            raise TypeError("PhotometricDescriptor expects a floating-point tensor")
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        elif image.ndim == 4:
            squeeze = False
        else:
            raise ValueError(f"expected [3,H,W] or [B,3,H,W], got {tuple(image.shape)}")
        if image.shape[1] != 3:
            raise ValueError(f"expected RGB channel dimension 3, got {image.shape[1]}")
        if image.amin().item() < -1e-5 or image.amax().item() > 1.0 + 1e-5:
            raise ValueError("PhotometricDescriptor expects values in [0, 1]")
        return image, squeeze

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, _ = self._as_batch(image)
        # 步骤 1：按当前协议处理。
        lowres = F.interpolate(batch, size=(self.lowres_size, self.lowres_size), mode="bilinear", align_corners=False)
        padding = self.blur_kernel_size // 2
        blurred = F.conv2d(
            lowres,
            self.gaussian_kernel.expand(3, 1, -1, -1).to(dtype=lowres.dtype),
            padding=padding,
            groups=3,
        )
        # 步骤 2：按当前协议处理。
        linear = srgb_to_linear(blurred)
        red, green, blue = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        flat_luminance = luminance.flatten(start_dim=1)
        # 步骤 3：按当前协议处理。
        mean_luminance = flat_luminance.mean(dim=1)
        std_luminance = flat_luminance.std(dim=1, unbiased=False)
        quantiles = torch.quantile(flat_luminance, torch.tensor([0.1, 0.9], device=flat_luminance.device), dim=1)
        # 步骤 4：按当前协议处理。
        log_red_green = torch.log((red + self.eps) / (green + self.eps)).mean(dim=(1, 2, 3))
        log_blue_green = torch.log((blue + self.eps) / (green + self.eps)).mean(dim=(1, 2, 3))
        # 步骤 5：拟合当前统计量。
        gradient_x = (luminance * self.grid_x.to(dtype=luminance.dtype)).sum(dim=(1, 2, 3)) / self.denominator_x.to(dtype=luminance.dtype)
        gradient_y = (luminance * self.grid_y.to(dtype=luminance.dtype)).sum(dim=(1, 2, 3)) / self.denominator_y.to(dtype=luminance.dtype)
        descriptor = torch.stack(
            (
                mean_luminance,
                std_luminance,
                quantiles[0],
                quantiles[1],
                log_red_green,
                log_blue_green,
                gradient_x,
                gradient_y,
            ),
            dim=1,
        )
        # 步骤 6：计算环境条件。
        if not torch.isfinite(descriptor).all():
            raise RuntimeError("photometric descriptor contains non-finite values")
        return descriptor


class ConditionStandardizer(nn.Module):
    """`ConditionStandardizer` 组件。"""

    def __init__(self, mean: torch.Tensor | None = None, std: torch.Tensor | None = None, count: int = 0, eps: float = 1e-6):
        super().__init__()
        if mean is None:
            mean = torch.zeros(8, dtype=torch.float32)
        if std is None:
            std = torch.ones(8, dtype=torch.float32)
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(std, dtype=torch.float32).flatten()
        if mean.numel() != 8 or std.numel() != 8:
            raise ValueError("condition mean and std must have exactly eight values")
        if (std < 0).any():
            raise ValueError("condition standard deviations must be non-negative")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("count", torch.tensor(int(count), dtype=torch.long))
        self.eps = float(eps)

    @property
    def fitted(self) -> bool:
        return int(self.count.item()) > 0

    @torch.no_grad()
    def fit(self, conditions: torch.Tensor | Iterable[torch.Tensor]) -> "ConditionStandardizer":
        """执行 `fit` 所需的处理。"""

        if isinstance(conditions, torch.Tensor):
            values = conditions
        else:
            chunks = [torch.as_tensor(chunk) for chunk in conditions]
            if not chunks:
                raise ValueError("cannot fit condition statistics from no samples")
            values = torch.cat(chunks, dim=0)
        values = torch.as_tensor(values, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] != 8 or values.shape[0] == 0:
            raise ValueError(f"expected non-empty [N,8] conditions, got {tuple(values.shape)}")
        if not torch.isfinite(values).all():
            raise ValueError("conditions contain NaN or infinity")
        self.mean.copy_(values.mean(dim=0).to(self.mean.device))
        self.std.copy_(values.std(dim=0, unbiased=False).to(self.std.device))
        self.count.fill_(values.shape[0])
        return self

    def transform(self, conditions: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("ConditionStandardizer must be fitted on normal training conditions first")
        values = torch.as_tensor(conditions)
        if values.shape[-1] != 8:
            raise ValueError(f"expected last dimension 8, got {tuple(values.shape)}")
        return (values - self.mean.to(dtype=values.dtype)) / (self.std.to(dtype=values.dtype) + self.eps)

    def inverse_transform(self, conditions: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(conditions)
        if values.shape[-1] != 8:
            raise ValueError(f"expected last dimension 8, got {tuple(values.shape)}")
        return values * (self.std.to(dtype=values.dtype) + self.eps) + self.mean.to(dtype=values.dtype)

    def to_dict(self, *, manifest_digest: str | None = None) -> dict[str, object]:
        output: dict[str, object] = {
            "dimension": 8,
            "count": int(self.count.item()),
            "eps": self.eps,
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
        }
        if manifest_digest is not None:
            output["manifest_digest"] = manifest_digest
        return output

    def save(self, output_path: str | Path, *, manifest_digest: str | None = None, overwrite: bool = False) -> Path:
        target = Path(output_path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"condition statistics already exist: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(manifest_digest=manifest_digest), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, input_path: str | Path) -> "ConditionStandardizer":
        source = Path(input_path)
        with source.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if int(values.get("dimension", 0)) != 8:
            raise ValueError(f"unsupported condition dimension in {source}")
        return cls(mean=torch.tensor(values["mean"]), std=torch.tensor(values["std"]), count=int(values["count"]), eps=float(values.get("eps", 1e-6)))
