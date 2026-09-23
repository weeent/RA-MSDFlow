"""RA-MSDFlow 的 backbone 模块。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn


def project_root() -> Path:
    """执行 `project_root` 所需的处理。"""

    return Path(__file__).resolve().parents[3]


def default_reference_resnet_path() -> Path:
    return project_root() / "third_party" / "wt_flow_fmad" / "models" / "resnet" / "resnet.py"


def default_wrn_weight_path() -> Path:
    return project_root() / "weights" / "torchvision" / "wide_resnet50_2-95faca4d.pth"


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """执行 `file_sha256` 所需的处理。"""

    digest = sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference_module(source_path: Path) -> ModuleType:
    """执行 `_load_reference_module` 所需的处理。"""

    if not source_path.is_file():
        raise FileNotFoundError(f"WT-Flow reference ResNet source is missing: {source_path}")
    module_name = "envfm_wtflow_reference_resnet"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    specification = importlib.util.spec_from_file_location(module_name, source_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load WT-Flow reference source: {source_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _load_state_dict(weight_path: Path) -> dict[str, Tensor]:
    """执行 `_load_state_dict` 所需的处理。"""

    if not weight_path.is_file():
        raise FileNotFoundError(f"WideResNet-50-2 checkpoint is missing: {weight_path}")
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:  # `_load_state_dict` 的实现说明。
        state = torch.load(weight_path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"unexpected checkpoint type at {weight_path}: {type(state)!r}")
    nested = state.get("state_dict")
    if isinstance(nested, dict):
        state = nested
    if not all(isinstance(key, str) for key in state):
        raise TypeError(f"checkpoint keys at {weight_path} are not strings")
    return state  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class BackboneProvenance:
    """`BackboneProvenance` 组件。"""

    reference_source: str
    reference_sha256: str
    weights: str
    weights_sha256: str
    output_channels: tuple[int, int, int]
    unexpected_checkpoint_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_source": self.reference_source,
            "reference_sha256": self.reference_sha256,
            "weights": self.weights,
            "weights_sha256": self.weights_sha256,
            "output_channels": list(self.output_channels),
            "unexpected_checkpoint_keys": list(self.unexpected_checkpoint_keys),
        }


class FrozenWideResNet50(nn.Module):
    """`FrozenWideResNet50` 组件。"""

    output_channels = (256, 512, 1024)

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        reference_resnet_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        source_path = Path(reference_resnet_path or default_reference_resnet_path()).resolve()
        checkpoint_path = Path(weights_path or default_wrn_weight_path()).resolve()
        reference_module = _load_reference_module(source_path)

        # 步骤 1：加载当前输入。
        self.encoder = reference_module.wide_resnet50_2(pretrained=False, progress=False)
        # 步骤 2：加载当前输入。
        state_dict = _load_state_dict(checkpoint_path)
        incompatibility = self.encoder.load_state_dict(state_dict, strict=False)
        tolerated = tuple(sorted(key for key in incompatibility.unexpected_keys if key.startswith(("layer4.", "fc."))))
        unexpected = tuple(sorted(set(incompatibility.unexpected_keys).difference(tolerated)))
        if incompatibility.missing_keys or unexpected:
            raise RuntimeError(
                "local WRN checkpoint does not match the candidate extractor; "
                f"missing={incompatibility.missing_keys}, unexpected={unexpected}"
            )
        # 步骤 3：按当前协议处理。
        self._freeze_encoder()
        self.provenance = BackboneProvenance(
            reference_source=str(source_path),
            reference_sha256=file_sha256(source_path),
            weights=str(checkpoint_path),
            weights_sha256=file_sha256(checkpoint_path),
            output_channels=self.output_channels,
            unexpected_checkpoint_keys=tolerated,
        )

    def _freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "FrozenWideResNet50":
        """执行 `train` 所需的处理。"""

        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected normalized [B,3,H,W] image, got {tuple(image.shape)}")
        # 步骤 4：构建当前模块。
        self.encoder.eval()
        features = self.encoder(image)
        if not isinstance(features, (list, tuple)) or len(features) != 3:
            raise RuntimeError("candidate WRN extractor did not return layer1/layer2/layer3 features")
        return tuple(features)  # type: ignore[return-value]

    def summary(self) -> dict[str, object]:
        """执行 `summary` 所需的处理。"""

        parameters = list(self.encoder.parameters())
        return {
            **self.provenance.to_dict(),
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            "encoder_training": self.encoder.training,
        }
