"""环境描述标准化器、模式库和描述器权重的统一持久化格式。

输入：已经在正常训练描述上拟合的 standardizer/mode bank，以及已训练或固定的描述器。
输出：单个 ``conditions.pt``，推理端据此恢复完全相同的环境坐标系和 GMM。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from envfm.training.checkpoint import atomic_torch_save, file_digest

from .descriptor import EnvironmentStandardizer, HybridEnvironmentDescriptor
from .mode_bank import DiagonalGaussianModeBank, ModeBankFitResult


CONDITION_BUNDLE_VERSION = 1


def save_condition_bundle(
    output_path: str | Path,
    *,
    descriptor: HybridEnvironmentDescriptor,
    standardizer: EnvironmentStandardizer,
    mode_bank: DiagonalGaussianModeBank,
    fit_result: ModeBankFitResult,
    provenance: Mapping[str, object] | None = None,
) -> Path:
    """保存 train-only 条件产物，并拒绝未拟合或维度不一致的对象。"""

    if not standardizer.fitted or not mode_bank.fitted:
        raise RuntimeError("standardizer and mode bank must be fitted before saving")
    if descriptor.output_dim != standardizer.dimension or descriptor.output_dim != mode_bank.dimension:
        raise ValueError("condition bundle components have inconsistent dimensions")
    state = {
        "format_version": CONDITION_BUNDLE_VERSION,
        "artifact_role": "msdflow_train_only_environment_conditions",
        "descriptor_config": {
            "learned_dim": descriptor.learned_dim,
            "lowres_size": descriptor.photo.lowres_size,
        },
        "mode_bank_config": {
            "dimension": mode_bank.dimension,
            "n_components": mode_bank.n_components,
            "min_variance": mode_bank.min_variance,
            "support_quantile": mode_bank.support_quantile,
        },
        "descriptor_state_dict": {
            name: value.detach().cpu().clone() for name, value in descriptor.state_dict().items()
        },
        "standardizer_state_dict": {
            name: value.detach().cpu().clone() for name, value in standardizer.state_dict().items()
        },
        "mode_bank_state_dict": {
            name: value.detach().cpu().clone() for name, value in mode_bank.state_dict().items()
        },
        "mode_bank_fit": asdict(fit_result),
        "provenance": dict(provenance or {}),
    }
    return atomic_torch_save(state, output_path)


def read_condition_bundle(input_path: str | Path) -> dict[str, Any]:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"condition bundle does not exist: {source}")
    try:
        state = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(source, map_location="cpu")
    if not isinstance(state, dict) or int(state.get("format_version", 0)) != CONDITION_BUNDLE_VERSION:
        raise ValueError("unsupported MSD-Flow condition bundle")
    if state.get("artifact_role") != "msdflow_train_only_environment_conditions":
        raise ValueError("input file is not an MSD-Flow condition bundle")
    required = {
        "descriptor_config",
        "mode_bank_config",
        "descriptor_state_dict",
        "standardizer_state_dict",
        "mode_bank_state_dict",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"condition bundle is missing fields: {sorted(missing)}")
    return state


def load_condition_bundle(
    input_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[HybridEnvironmentDescriptor, EnvironmentStandardizer, DiagonalGaussianModeBank, dict[str, object]]:
    """严格重建三个条件组件，并返回文件 hash 与拟合摘要。"""

    source = Path(input_path).resolve()
    state = read_condition_bundle(source)
    descriptor_config = state["descriptor_config"]
    bank_config = state["mode_bank_config"]
    descriptor = HybridEnvironmentDescriptor(
        learned_dim=int(descriptor_config["learned_dim"]),
        lowres_size=int(descriptor_config["lowres_size"]),
    )
    standardizer = EnvironmentStandardizer(int(bank_config["dimension"]))
    mode_bank = DiagonalGaussianModeBank(
        int(bank_config["dimension"]),
        int(bank_config["n_components"]),
        min_variance=float(bank_config["min_variance"]),
        support_quantile=float(bank_config["support_quantile"]),
    )
    descriptor.load_state_dict(state["descriptor_state_dict"], strict=True)
    standardizer.load_state_dict(state["standardizer_state_dict"], strict=True)
    mode_bank.load_state_dict(state["mode_bank_state_dict"], strict=True)
    run_device = torch.device(device)
    for module in (descriptor, standardizer, mode_bank):
        module.to(run_device).eval().requires_grad_(False)
    if not standardizer.fitted or not mode_bank.fitted:
        raise RuntimeError("loaded condition bundle is not fitted")
    info = {
        "path": str(source),
        "sha256": file_digest(source),
        "descriptor_config": dict(descriptor_config),
        "mode_bank_config": dict(bank_config),
        "mode_bank_fit": state.get("mode_bank_fit", {}),
        "provenance": state.get("provenance", {}),
    }
    return descriptor, standardizer, mode_bank, info
