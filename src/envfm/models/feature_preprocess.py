"""RA-MSDFlow 的 feature_preprocess 模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def tensor_statistics(tensor: Tensor) -> dict[str, object]:
    """执行 `tensor_statistics` 所需的处理。"""

    detached = tensor.detach().float()
    return {
        "shape": list(detached.shape),
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "min": float(detached.amin().item()),
        "max": float(detached.amax().item()),
        "mean_l2_per_sample": float(detached.flatten(start_dim=1).norm(dim=1).mean().item()),
        "finite": bool(torch.isfinite(detached).all().item()),
    }


@dataclass(slots=True)
class FeatureProcessingOutput:
    """`FeatureProcessingOutput` 组件。"""

    selected: Tensor
    pooled: Tensor
    normalized: Tensor

    def summary(self) -> dict[str, object]:
        return {
            "selected": tensor_statistics(self.selected),
            "pooled": tensor_statistics(self.pooled),
            "normalized": tensor_statistics(self.normalized),
        }


class WTFeaturePreprocessor(nn.Module):
    """`WTFeaturePreprocessor` 组件。"""

    def __init__(self, branch_index: int = 2, pool_type: str = "avg", layer_norm: bool = True) -> None:
        super().__init__()
        if branch_index not in {0, 1, 2}:
            raise ValueError("branch_index must select layer1=0, layer2=1, or layer3=2")
        if pool_type not in {"avg", "max", "identity"}:
            raise ValueError("pool_type must be 'avg', 'max', or 'identity'")
        self.branch_index = int(branch_index)
        self.pool_type = pool_type
        self.layer_norm = bool(layer_norm)
        if pool_type == "avg":
            self.pool: nn.Module = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.Identity()

    def _select(self, features: Sequence[Tensor] | Tensor) -> Tensor:
        if isinstance(features, Tensor):
            selected = features
        else:
            if len(features) != 3:
                raise ValueError(f"expected three backbone features, received {len(features)}")
            selected = features[self.branch_index]
        if selected.ndim != 4:
            raise ValueError(f"selected feature must be [B,C,H,W], got {tuple(selected.shape)}")
        return selected

    def forward_with_details(self, features: Sequence[Tensor] | Tensor) -> FeatureProcessingOutput:
        # 步骤 1：按当前协议处理。
        selected = self._select(features)
        # 步骤 2：按当前协议处理。
        pooled = self.pool(selected)
        # 步骤 3：按当前协议处理。
        normalized = F.layer_norm(pooled, pooled.shape[1:]) if self.layer_norm else pooled
        return FeatureProcessingOutput(selected=selected, pooled=pooled, normalized=normalized)

    def forward(self, features: Sequence[Tensor] | Tensor) -> Tensor:
        return self.forward_with_details(features).normalized

    def summary(self) -> dict[str, object]:
        return {
            "branch_index": self.branch_index,
            "pool_type": self.pool_type,
            "layer_norm": self.layer_norm,
            "learnable_parameter_count": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }
