"""低频编码预训练、环境运输和正常性流训练 CLI。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from envfm.training.checkpoint import file_digest
from msdflow.conditions import LowFrequencyEnvironmentEncoder, load_condition_bundle
from msdflow.data import CachedPairedFeatureDataset, PairedEnvironmentDataset, build_environment_pairs
from msdflow.inference.checkpoint import export_inference_bundle
from msdflow.models import MSDFlowModel, ModeConditionedNormalityFlow, PairedEnvironmentFlow, SmoothEnvironmentVelocity
from msdflow.training import (
    EnvironmentCodeRegressor,
    EnvironmentCodeTrainer,
    EnvironmentTransportTrainer,
    NormalityFlowTrainer,
)

from .common import load_json_config, make_loader, required, select_records, training_config


def _device(config: Mapping[str, Any]) -> torch.device:
    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _cached_loaders(config: Mapping[str, Any], device: torch.device):
    train = CachedPairedFeatureDataset(required(config, "train_cache_index"))
    validation = CachedPairedFeatureDataset(required(config, "validation_cache_index"))
    batch_size = int(config.get("batch_size", 8))
    workers = int(config.get("num_workers", 0))
    seed = int(config.get("seed", 9826))
    return train, validation, make_loader(
        train, batch_size=batch_size, shuffle=True, num_workers=workers, seed=seed, pin_memory=device.type == "cuda"
    ), make_loader(
        validation, batch_size=batch_size, shuffle=False, num_workers=workers, seed=seed + 1, pin_memory=device.type == "cuda"
    )


def run_train_descriptor(config: Mapping[str, Any]) -> Path:
    device = _device(config)
    manifest = required(config, "manifest")
    category = str(required(config, "category"))
    variants = tuple(config.get("target_variants", ["exposure", "white_balance", "gradient", "compound"]))
    datasets = []
    for split in (str(config.get("train_split", "train")), str(config.get("validation_split", "val"))):
        records = select_records(
            manifest,
            category=category,
            split=split,
            dataset=None if config.get("dataset") is None else str(config["dataset"]),
            normal_only=True,
        )
        datasets.append(
            PairedEnvironmentDataset(
                build_environment_pairs(records, target_variants=variants, base_seed=int(config.get("data_seed", 9826))),
                image_size=config.get("image_size", 256),
            )
        )
    train_loader = make_loader(
        datasets[0], batch_size=int(config.get("batch_size", 16)), shuffle=True,
        num_workers=int(config.get("num_workers", 0)), seed=int(config.get("seed", 9826)), pin_memory=device.type == "cuda"
    )
    val_loader = make_loader(
        datasets[1], batch_size=int(config.get("batch_size", 16)), shuffle=False,
        num_workers=int(config.get("num_workers", 0)), seed=int(config.get("seed", 9826)) + 1, pin_memory=device.type == "cuda"
    )
    model = EnvironmentCodeRegressor(
        LowFrequencyEnvironmentEncoder(
            output_dim=int(config.get("learned_dim", 16)),
            # 固定小编码器宽度，避免训练与推理出现额外结构配置。
            hidden_channels=24,
            lowres_size=int(config.get("descriptor_lowres_size", 32)),
        )
    )
    result = EnvironmentCodeTrainer(
        model,
        run_directory=required(config, "output_directory"),
        config=training_config(config.get("training")),
        device=device,
    ).fit(train_loader, val_loader)
    return result.summary_path


def run_train_transport(config: Mapping[str, Any]) -> Path:
    device = _device(config)
    train, validation, train_loader, val_loader = _cached_loaders(config, device)
    _, standardizer, mode_bank, _ = load_condition_bundle(required(config, "condition_bundle"), device=device)
    channels = int(torch.as_tensor(train[0]["feature_a"]).shape[0])
    if int(torch.as_tensor(validation[0]["feature_a"]).shape[0]) != channels:
        raise ValueError("train/validation feature channels differ")
    model = PairedEnvironmentFlow(
        SmoothEnvironmentVelocity(
            channels,
            mode_bank.dimension,
            hidden_dim=int(config.get("environment_hidden_dim", 128)),
            spatial_hidden_channels=int(config.get("spatial_hidden_channels", 64)),
            lowres_size=int(config.get("environment_lowres_size", 4)),
            max_scale_rate=float(config.get("max_scale_rate", 0.25)),
        )
    )
    result = EnvironmentTransportTrainer(
        model,
        environment_standardizer=standardizer,
        run_directory=required(config, "output_directory"),
        config=training_config(config.get("training")),
        device=device,
    ).fit(train_loader, val_loader)
    return result.summary_path


def _load_stage_best(model: torch.nn.Module, path: str | Path, expected_stage: str) -> None:
    """从 CPU 严格加载某一训练阶段的 best.pt，避免恢复时夹带优化器状态。"""

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or state.get("stage") != expected_stage:
        raise ValueError(f"checkpoint stage mismatch: expected {expected_stage!r}")
    if "model_state_dict" not in state:
        raise ValueError("checkpoint has no model_state_dict")
    model.load_state_dict(state["model_state_dict"], strict=True)


def run_train_normality(config: Mapping[str, Any]) -> Path:
    device = _device(config)
    train, validation, train_loader, val_loader = _cached_loaders(config, device)
    condition_path = Path(required(config, "condition_bundle")).resolve()
    descriptor, standardizer, mode_bank, _ = load_condition_bundle(condition_path, device=device)
    channels = int(torch.as_tensor(train[0]["feature_a"]).shape[0])
    environment_flow = PairedEnvironmentFlow(
        SmoothEnvironmentVelocity(
            channels,
            mode_bank.dimension,
            hidden_dim=int(config.get("environment_hidden_dim", 128)),
            spatial_hidden_channels=int(config.get("spatial_hidden_channels", 64)),
            lowres_size=int(config.get("environment_lowres_size", 4)),
            max_scale_rate=float(config.get("max_scale_rate", 0.25)),
        )
    )
    environment_checkpoint = Path(required(config, "environment_checkpoint")).resolve()
    _load_stage_best(environment_flow, environment_checkpoint, "paired_environment_flow")
    normality = ModeConditionedNormalityFlow(
        channels,
        mode_bank.n_components,
        base_channels=int(config.get("normality_base_channels", 128)),
        condition_hidden_dim=int(config.get("normality_condition_hidden_dim", 64)),
        # K=1 消融设为4可保持条件 MLP 与 K=4 主模型同参数量。
        condition_dim=int(config.get("normality_condition_dim", mode_bank.n_components)),
    )
    output = Path(required(config, "output_directory")).resolve()
    result = NormalityFlowTrainer(
        normality,
        mode_bank=mode_bank,
        environment_flow=environment_flow,
        environment_standardizer=standardizer,
        canonicalize_to_mode=bool(config.get("canonicalize_to_mode", True)),
        transport_steps=int(config.get("environment_steps", 4)),
        endpoint_policy=str(config.get("endpoint_policy", "target")),  # type: ignore[arg-type]
        run_directory=output,
        config=training_config(config.get("training")),
        device=device,
    ).fit(train_loader, val_loader)

    # 训练结束后重新载入验证集最优权重；最终推理包不能误用最后一个 epoch。
    _load_stage_best(normality, result.best_checkpoint, "mode_conditioned_normality_flow")
    full_model = MSDFlowModel(descriptor, standardizer, mode_bank, environment_flow, normality)
    export_inference_bundle(
        output / "inference_bundle.pt",
        full_model,
        provenance={
            "condition_bundle": str(condition_path),
            "condition_bundle_sha256": file_digest(condition_path),
            "environment_checkpoint": str(environment_checkpoint),
            "environment_checkpoint_sha256": file_digest(environment_checkpoint),
            "normality_checkpoint": str(result.best_checkpoint),
            "normality_checkpoint_sha256": file_digest(result.best_checkpoint),
            "normality_best_epoch": result.best_epoch,
        },
    )
    return result.summary_path


def _main(runner, description: str, argv: Sequence[str] | None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = runner(load_json_config(args.config))
    print(json.dumps({"summary": str(path)}, ensure_ascii=False, indent=2))


def main_descriptor(argv: Sequence[str] | None = None) -> None:
    _main(run_train_descriptor, "Pretrain MSD-Flow environment code", argv)


def main_transport(argv: Sequence[str] | None = None) -> None:
    _main(run_train_transport, "Train MSD-Flow paired environment flow", argv)


def main_normality(argv: Sequence[str] | None = None) -> None:
    _main(run_train_normality, "Train MSD-Flow mode-conditioned normality flow", argv)
