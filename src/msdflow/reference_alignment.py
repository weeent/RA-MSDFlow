"""RA-MSDFlow 的 reference_alignment 模块。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def _as_float64(values: Tensor) -> Tensor:
    return values.detach().to(dtype=torch.float64, device="cpu")


def _validate_images(images: Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"RGB alignment expects [B,3,H,W], got {tuple(images.shape)}")
    if not torch.is_floating_point(images):
        raise TypeError("RGB alignment expects a floating-point tensor")


def _validate_features(features: Tensor) -> None:
    if features.ndim != 4:
        raise ValueError(f"feature alignment expects [B,C,H,W], got {tuple(features.shape)}")
    if not torch.is_floating_point(features):
        raise TypeError("feature alignment expects a floating-point tensor")


@dataclass(slots=True)
class RgbMomentAlignment:
    """`RgbMomentAlignment` 组件。"""

    mean_source: Tensor | None = None
    std_source: Tensor | None = None
    mean_target: Tensor | None = None
    std_target: Tensor | None = None
    eps: float = 1e-6

    @property
    def fitted(self) -> bool:
        return self.mean_target is not None

    @staticmethod
    def _statistics(images: Tensor) -> tuple[Tensor, Tensor]:
        _validate_images(images)
        flat = _as_float64(images).permute(1, 0, 2, 3).reshape(images.shape[1], -1)
        return flat.mean(1), flat.std(1, unbiased=False)

    def fit(self, source_images: Tensor, target_images: Tensor) -> "RgbMomentAlignment":
        mean_s, std_s = self._statistics(source_images)
        mean_t, std_t = self._statistics(target_images)
        self.mean_source, self.std_source = mean_s, std_s.clamp_min(self.eps)
        self.mean_target, self.std_target = mean_t, std_t.clamp_min(self.eps)
        return self

    def apply(self, images: Tensor) -> Tensor:
        if not self.fitted:
            raise RuntimeError("RgbMomentAlignment must be fitted before apply")
        _validate_images(images)
        view = lambda value: value.to(images).view(3, 1, 1)  # noqa: E731
        gain = view(self.std_source) / view(self.std_target)
        aligned = view(self.mean_source) + gain * (images - view(self.mean_target))
        return aligned.clamp(0.0, 1.0)

    def summary(self) -> dict[str, object]:
        if not self.fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "mean_source": [float(v) for v in self.mean_source],
            "std_source": [float(v) for v in self.std_source],
            "mean_target": [float(v) for v in self.mean_target],
            "std_target": [float(v) for v in self.std_target],
        }


@dataclass(slots=True)
class FeatureMomentAlignment:
    """`FeatureMomentAlignment` 组件。"""

    mean_source: Tensor | None = None
    std_source: Tensor | None = None
    mean_target: Tensor | None = None
    std_target: Tensor | None = None
    eps: float = 1e-6

    @property
    def fitted(self) -> bool:
        return self.mean_target is not None

    @staticmethod
    def _statistics(features: Tensor) -> tuple[Tensor, Tensor]:
        _validate_features(features)
        flat = _as_float64(features).permute(1, 0, 2, 3).reshape(features.shape[1], -1)
        return flat.mean(1), flat.std(1, unbiased=False)

    def fit(self, source_features: Tensor, target_features: Tensor) -> "FeatureMomentAlignment":
        mean_s, std_s = self._statistics(source_features)
        mean_t, std_t = self._statistics(target_features)
        self.mean_source, self.std_source = mean_s, std_s.clamp_min(self.eps)
        self.mean_target, self.std_target = mean_t, std_t.clamp_min(self.eps)
        return self

    def apply(self, features: Tensor) -> Tensor:
        if not self.fitted:
            raise RuntimeError("FeatureMomentAlignment must be fitted before apply")
        _validate_features(features)
        channels = features.shape[1]
        view = lambda value: value.to(features).view(1, channels, 1, 1)  # noqa: E731
        gain = view(self.std_source) / view(self.std_target)
        return view(self.mean_source) + gain * (features - view(self.mean_target))

    def summary(self) -> dict[str, object]:
        if not self.fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "channels": int(self.mean_source.numel()),
            "mean_gap_l2": float((self.mean_source - self.mean_target).norm()),
            "std_gap_l2": float((self.std_source - self.std_target).norm()),
        }


@dataclass(slots=True)
class ShrinkageCoral:
    """`ShrinkageCoral` 组件。"""

    shrinkage: float = 0.1
    max_patches: int = 20000
    patch_seed: int = 9826
    eigenvalue_floor: float = 1e-4
    mean_source: Tensor | None = None
    mean_target: Tensor | None = None
    transform: Tensor | None = None
    covariance_eigenvalues: dict[str, list[float]] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return self.transform is not None

    def _patch_matrix(self, features: Tensor) -> Tensor:
        _validate_features(features)
        matrix = _as_float64(features).permute(1, 0, 2, 3).reshape(features.shape[1], -1).transpose(0, 1)
        if matrix.shape[0] > self.max_patches:
            generator = torch.Generator().manual_seed(self.patch_seed)
            index = torch.randperm(matrix.shape[0], generator=generator)[: self.max_patches]
            matrix = matrix[index]
        return matrix

    def _shrunk_covariance(self, matrix: Tensor) -> Tensor:
        centered = matrix - matrix.mean(0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / max(matrix.shape[0] - 1, 1)
        dimension = covariance.shape[0]
        identity = torch.eye(dimension, dtype=covariance.dtype)
        trace_scale = torch.diagonal(covariance).sum() / dimension
        return (1.0 - self.shrinkage) * covariance + self.shrinkage * trace_scale * identity

    def _matrix_power(self, covariance: Tensor, exponent: float) -> Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        floored = eigenvalues.clamp_min(self.eigenvalue_floor)
        powered = floored.pow(exponent)
        return (eigenvectors * powered.unsqueeze(0)) @ eigenvectors.transpose(0, 1)

    def fit(self, source_features: Tensor, target_features: Tensor) -> "ShrinkageCoral":
        source = self._patch_matrix(source_features)
        target = self._patch_matrix(target_features)
        self.mean_source = source.mean(0)
        self.mean_target = target.mean(0)
        source_covariance = self._shrunk_covariance(source)
        target_covariance = self._shrunk_covariance(target)
        self.covariance_eigenvalues = {
            "source": [float(v) for v in torch.linalg.eigvalsh(source_covariance)],
            "target": [float(v) for v in torch.linalg.eigvalsh(target_covariance)],
        }
        whitening = self._matrix_power(target_covariance, -0.5)
        colouring = self._matrix_power(source_covariance, 0.5)
        self.transform = whitening @ colouring
        return self

    def apply(self, features: Tensor) -> Tensor:
        if not self.fitted:
            raise RuntimeError("ShrinkageCoral must be fitted before apply")
        _validate_features(features)
        channels = features.shape[1]
        if self.transform.shape[0] != channels:
            raise ValueError(
                f"CORAL transform is {self.transform.shape[0]}D but features have {channels} channels"
            )
        matrix = _as_float64(features).permute(1, 0, 2, 3).reshape(channels, -1).transpose(0, 1)
        centered = matrix - self.mean_target
        # 按 CORAL 公式变换行向量特征。
        aligned = centered @ self.transform + self.mean_source
        restored = aligned.transpose(0, 1).reshape(channels, features.shape[0], *features.shape[2:])
        return restored.permute(1, 0, 2, 3).to(features.dtype).to(features.device)

    def summary(self) -> dict[str, object]:
        if not self.fitted:
            return {"fitted": False}
        summary: dict[str, object] = {
            "fitted": True,
            "shrinkage": float(self.shrinkage),
            "eigenvalue_floor": float(self.eigenvalue_floor),
            "transform_finite": bool(torch.isfinite(self.transform).all()),
        }
        for name, values in self.covariance_eigenvalues.items():
            tensor = torch.tensor(values, dtype=torch.float64)
            summary[f"{name}_eigenvalue_min"] = float(tensor.min())
            summary[f"{name}_eigenvalue_max"] = float(tensor.max())
        return summary
