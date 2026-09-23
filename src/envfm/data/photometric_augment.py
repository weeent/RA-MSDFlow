"""RA-MSDFlow 的 photometric_augment 模块。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import blake2b
import math
import random
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - enables manifest-only commands before PyTorch setup.
    torch = None  # type: ignore[assignment]


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for applying photometric transforms. "
            "Install requirements-data.txt into the EnvFM training environment."
        )
    return torch


@dataclass(frozen=True, slots=True)
class PhotometricParams:
    """`PhotometricParams` 组件。"""

    variant: str = "clean"
    exposure_ev: float = 0.0
    red_gain: float = 1.0
    green_gain: float = 1.0
    blue_gain: float = 1.0
    gradient_x: float = 0.0
    gradient_y: float = 0.0

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "PhotometricParams":
        if not values:
            return cls()
        allowed = {
            "variant",
            "exposure_ev",
            "red_gain",
            "green_gain",
            "blue_gain",
            "gradient_x",
            "gradient_y",
        }
        unknown = set(values).difference(allowed)
        if unknown:
            raise ValueError(f"unknown photometric parameters: {sorted(unknown)}")
        return cls(
            variant=str(values.get("variant", "clean")),
            exposure_ev=float(values.get("exposure_ev", 0.0)),
            red_gain=float(values.get("red_gain", 1.0)),
            green_gain=float(values.get("green_gain", 1.0)),
            blue_gain=float(values.get("blue_gain", 1.0)),
            gradient_x=float(values.get("gradient_x", 0.0)),
            gradient_y=float(values.get("gradient_y", 0.0)),
        )


def stable_seed(base_seed: int, *parts: str) -> int:
    """执行 `stable_seed` 所需的处理。"""

    value = "\u241f".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(blake2b(value, digest_size=8).digest(), byteorder="big", signed=False)


def _log_uniform(rng: random.Random, max_abs_log: float) -> float:
    return math.exp(rng.uniform(-max_abs_log, max_abs_log))


def sample_photometric_params(
    variant: str, seed: int, *, exposure_ev_range: float = 0.75, color_log_range: float = 0.18,
    gradient_range: float = 0.35,
) -> PhotometricParams:
    """执行 `sample_photometric_params` 所需的处理。"""

    if variant not in {"clean", "exposure", "white_balance", "gradient", "compound"}:
        raise ValueError(f"unsupported photometric variant: {variant}")
    rng = random.Random(int(seed))
    if variant == "clean":
        return PhotometricParams(variant="clean")

    exposure_ev = rng.uniform(-exposure_ev_range, exposure_ev_range) if variant in {"exposure", "compound"} else 0.0
    if variant in {"white_balance", "compound"}:
        red_gain = _log_uniform(rng, color_log_range)
        blue_gain = _log_uniform(rng, color_log_range)
    else:
        red_gain = 1.0
        blue_gain = 1.0
    if variant in {"gradient", "compound"}:
        gradient_x = rng.uniform(-gradient_range, gradient_range)
        gradient_y = rng.uniform(-gradient_range, gradient_range)
    else:
        gradient_x = 0.0
        gradient_y = 0.0
    return PhotometricParams(
        variant=variant,
        exposure_ev=exposure_ev,
        red_gain=red_gain,
        green_gain=1.0,
        blue_gain=blue_gain,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
    )


def _validate_image(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    torch_module = _require_torch()
    if not torch_module.is_floating_point(image):
        raise TypeError("photometric transforms require a floating-point RGB tensor in [0, 1]")
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True
    elif image.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"expected [3,H,W] or [B,3,H,W], got {tuple(image.shape)}")
    if image.shape[1] != 3:
        raise ValueError(f"expected RGB channel dimension 3, got {image.shape[1]}")
    if not torch.isfinite(image).all():
        raise ValueError("image contains NaN or infinity")
    if image.amin().item() < -1e-5 or image.amax().item() > 1.0 + 1e-5:
        raise ValueError("image values must lie in [0, 1]")
    return image, squeeze


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    """执行 `srgb_to_linear` 所需的处理。"""

    # `srgb_to_linear` 的实现说明。
    _require_torch()
    return torch.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    """执行 `linear_to_srgb` 所需的处理。"""

    # `linear_to_srgb` 的实现说明。
    _require_torch()
    image = image.clamp_min(0.0)
    result = torch.where(image <= 0.0031308, image * 12.92, 1.055 * image.pow(1.0 / 2.4) - 0.055)
    return result.clamp(0.0, 1.0)


def illumination_plane(
    height: int, width: int, gradient_x: float, gradient_y: float, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """执行 `illumination_plane` 所需的处理。"""

    torch_module = _require_torch()
    horizontal = torch_module.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    vertical = torch_module.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    yy, xx = torch_module.meshgrid(vertical, horizontal, indexing="ij")
    plane = 1.0 + float(gradient_x) * xx + float(gradient_y) * yy
    return plane.clamp_min(0.05).unsqueeze(0).unsqueeze(0)


def apply_photometric_transform(
    image: torch.Tensor, params: PhotometricParams | Mapping[str, Any] | None
) -> torch.Tensor:
    """执行 `apply_photometric_transform` 所需的处理。"""

    batch, squeeze = _validate_image(image)
    specification = params if isinstance(params, PhotometricParams) else PhotometricParams.from_mapping(params)
    # 步骤 1：解码图像。
    linear = srgb_to_linear(batch)
    # 步骤 2：构建当前模块。
    torch_module = _require_torch()
    gains = torch_module.tensor(
        [specification.red_gain, specification.green_gain, specification.blue_gain],
        dtype=linear.dtype,
        device=linear.device,
    ).view(1, 3, 1, 1)
    # 步骤 3：转换特征表示。
    exposure = 2.0 ** specification.exposure_ev
    # 步骤 4：按当前协议处理。
    plane = illumination_plane(
        linear.shape[-2],
        linear.shape[-1],
        specification.gradient_x,
        specification.gradient_y,
        device=linear.device,
        dtype=linear.dtype,
    )
    # 步骤 5：应用当前变换。
    transformed = linear * exposure * gains * plane
    result = linear_to_srgb(transformed)
    return result.squeeze(0) if squeeze else result
