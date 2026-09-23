"""把完成训练的 MSD-Flow 导出为一个可直接推理的只读文件。

输入：完整 ``MSDFlowModel``（条件描述器、模式库、环境流和正常性流）。
输出：单个 ``inference_bundle.pt``；推理端不再拼装四个相互依赖的文件。
中间产物：模型结构常数和完整 state dict，均由本模块自动提取。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch

from envfm.training.checkpoint import atomic_torch_save, file_digest
from msdflow.conditions import DiagonalGaussianModeBank, EnvironmentStandardizer, HybridEnvironmentDescriptor
from msdflow.models import (
    MSDFlowModel,
    ModeConditionedNormalityFlow,
    PairedEnvironmentFlow,
    SmoothEnvironmentVelocity,
)


INFERENCE_BUNDLE_VERSION = 1


@dataclass(frozen=True, slots=True)
class FrozenMSDFlowInfo:
    """记录实际加载文件的身份，供结果摘要追溯。"""

    path: str
    sha256: str
    format_version: int
    architecture: Mapping[str, object]
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _architecture(model: MSDFlowModel) -> dict[str, object]:
    """从真实模块读取构造参数，避免另写一份易漂移的 manifest。"""

    environment_velocity = model.environment_flow.velocity
    normality_velocity = model.normality_flow.velocity
    conditioner = normality_velocity.photo_conditioner
    if environment_velocity.in_channels != normality_velocity.in_channels:
        raise ValueError("environment and normality flows use different feature channels")
    if environment_velocity.condition_dim != model.mode_bank.dimension:
        raise ValueError("environment flow and mode bank use different condition dimensions")
    return {
        "feature_channels": environment_velocity.in_channels,
        "learned_environment_dim": model.descriptor.learned_dim,
        "descriptor_lowres_size": model.descriptor.photo.lowres_size,
        "environment_dimension": model.mode_bank.dimension,
        "n_modes": model.mode_bank.n_components,
        "gmm_min_variance": model.mode_bank.min_variance,
        "gmm_support_quantile": model.mode_bank.support_quantile,
        "environment_hidden_dim": environment_velocity.hidden_dim,
        "spatial_hidden_channels": environment_velocity.spatial_in.out_channels,
        "environment_lowres_size": environment_velocity.lowres_size,
        "max_scale_rate": environment_velocity.max_scale_rate,
        "normality_base_channels": normality_velocity.base_channels,
        "normality_condition_hidden_dim": conditioner.hidden_dim,
        "normality_condition_dim": model.normality_flow.condition_dim,
    }


def export_inference_bundle(
    output_path: str | Path,
    model: MSDFlowModel,
    *,
    provenance: Mapping[str, object] | None = None,
) -> Path:
    """原子写出唯一的正式推理模型文件。"""

    if not model.standardizer.fitted or not model.mode_bank.fitted:
        raise RuntimeError("cannot export an unfitted standardizer or mode bank")
    payload = {
        "format_version": INFERENCE_BUNDLE_VERSION,
        "artifact_role": "msdflow_inference_bundle",
        "architecture": _architecture(model),
        # 保存时统一转 CPU，减少设备相关性；加载后再一次性搬到目标设备。
        "model_state_dict": {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        },
        "provenance": dict(provenance or {}),
    }
    return atomic_torch_save(payload, output_path)


def load_inference_bundle(
    input_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MSDFlowModel, FrozenMSDFlowInfo]:
    """严格重建、加载并冻结完整模型；任何结构或键不匹配都立即报错。"""

    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"inference bundle does not exist: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict) or int(payload.get("format_version", 0)) != INFERENCE_BUNDLE_VERSION:
        raise ValueError("unsupported MSD-Flow inference bundle")
    if payload.get("artifact_role") != "msdflow_inference_bundle":
        raise ValueError("input file is not an MSD-Flow inference bundle")
    if "architecture" not in payload or "model_state_dict" not in payload:
        raise ValueError("inference bundle is incomplete")

    architecture = dict(payload["architecture"])
    descriptor = HybridEnvironmentDescriptor(
        learned_dim=int(architecture["learned_environment_dim"]),
        lowres_size=int(architecture["descriptor_lowres_size"]),
    )
    standardizer = EnvironmentStandardizer(int(architecture["environment_dimension"]))
    mode_bank = DiagonalGaussianModeBank(
        int(architecture["environment_dimension"]),
        int(architecture["n_modes"]),
        min_variance=float(architecture["gmm_min_variance"]),
        support_quantile=float(architecture["gmm_support_quantile"]),
    )
    environment_flow = PairedEnvironmentFlow(
        SmoothEnvironmentVelocity(
            int(architecture["feature_channels"]),
            int(architecture["environment_dimension"]),
            hidden_dim=int(architecture["environment_hidden_dim"]),
            spatial_hidden_channels=int(architecture["spatial_hidden_channels"]),
            lowres_size=int(architecture["environment_lowres_size"]),
            max_scale_rate=float(architecture["max_scale_rate"]),
        )
    )
    normality_flow = ModeConditionedNormalityFlow(
        int(architecture["feature_channels"]),
        int(architecture["n_modes"]),
        base_channels=int(architecture["normality_base_channels"]),
        condition_hidden_dim=int(architecture["normality_condition_hidden_dim"]),
        # Version-1 历史 bundle 没有该键；默认回退到 n_modes，保持严格兼容。
        condition_dim=int(architecture.get("normality_condition_dim", architecture["n_modes"])),
    )
    model = MSDFlowModel(descriptor, standardizer, mode_bank, environment_flow, normality_flow)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(torch.device(device)).eval().requires_grad_(False)
    if not model.standardizer.fitted or not model.mode_bank.fitted:
        raise RuntimeError("inference bundle contains unfitted condition statistics")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("failed to freeze MSD-Flow for inference")

    info = FrozenMSDFlowInfo(
        path=str(source),
        sha256=file_digest(source),
        format_version=INFERENCE_BUNDLE_VERSION,
        architecture=architecture,
        provenance=dict(payload.get("provenance", {})),
    )
    return model, info
