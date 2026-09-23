"""配对特征缓存和 train-only 环境模式产物的 CLI。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from envfm.models.backbone import FrozenWideResNet50
from envfm.models.feature_preprocess import WTFeaturePreprocessor
from envfm.training.checkpoint import atomic_write_json, file_digest
from msdflow.conditions import (
    DiagonalGaussianModeBank,
    EnvironmentStandardizer,
    save_condition_bundle,
)
from msdflow.data import (
    CachedPairedFeatureDataset,
    PairedEnvironmentDataset,
    build_environment_pairs,
    build_paired_feature_cache,
)

from .common import load_descriptor, load_json_config, make_loader, required, select_records


def run_cache(config: Mapping[str, Any]) -> Path:
    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest_path = Path(required(config, "manifest")).resolve()
    records = select_records(
        manifest_path,
        category=str(required(config, "category")),
        split=str(required(config, "split")),
        dataset=None if config.get("dataset") is None else str(config["dataset"]),
        normal_only=True,
    )
    variants = tuple(config.get("target_variants", ["exposure", "white_balance", "gradient", "compound"]))
    pairs = build_environment_pairs(
        records,
        target_variants=variants,
        base_seed=int(config.get("data_seed", 9826)),
        include_reverse=bool(config.get("include_reverse", False)),
    )
    descriptor, descriptor_info = load_descriptor(
        learned_dim=int(config.get("learned_dim", 0)),
        lowres_size=int(config.get("descriptor_lowres_size", 32)),
        checkpoint_path=config.get("descriptor_checkpoint"),
    )
    dataset = PairedEnvironmentDataset(pairs, image_size=config.get("image_size", 256))
    loader = make_loader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        seed=int(config.get("data_seed", 9826)),
        pin_memory=device.type == "cuda",
    )
    backbone = FrozenWideResNet50(
        weights_path=config.get("backbone_weights"),
        reference_resnet_path=config.get("reference_resnet_source"),
    )
    preprocessor = WTFeaturePreprocessor(branch_index=int(config.get("branch_index", 2)))
    output = Path(required(config, "output_directory")).resolve()
    return build_paired_feature_cache(
        loader,
        backbone,
        descriptor,
        output,
        feature_preprocessor=preprocessor,
        device=device,
        metadata={
            "manifest": str(manifest_path),
            "manifest_sha256": file_digest(manifest_path),
            "category": str(config["category"]),
            "split": str(config["split"]),
            "pair_summary": dataset.summary(),
            "descriptor": descriptor_info,
            "backbone": backbone.summary(),
            "environment_values": "raw_unstandardized",
        },
    )


def _unique_environment_values(dataset: CachedPairedFeatureDataset) -> torch.Tensor:
    seen: set[tuple[str, str]] = set()
    values: list[torch.Tensor] = []
    for item in dataset:
        for suffix in ("a", "b"):
            key = (str(item["base_id"]), str(item[f"variant_{suffix}"]))
            if key in seen:
                continue
            seen.add(key)
            values.append(torch.as_tensor(item[f"environment_{suffix}"]).float().unsqueeze(0))
    if not values:
        raise ValueError("feature cache contains no environment values")
    return torch.cat(values, dim=0)


def run_fit_conditions(config: Mapping[str, Any]) -> Path:
    cache = CachedPairedFeatureDataset(required(config, "train_cache_index"))
    values = _unique_environment_values(cache)
    learned_dim = int(config.get("learned_dim", values.shape[1] - 8))
    descriptor, descriptor_info = load_descriptor(
        learned_dim=learned_dim,
        lowres_size=int(config.get("descriptor_lowres_size", 32)),
        checkpoint_path=config.get("descriptor_checkpoint"),
    )
    if descriptor.output_dim != values.shape[1]:
        raise ValueError("cached environment dimension does not match descriptor")
    standardizer = EnvironmentStandardizer(values.shape[1]).fit(values)
    normalized = standardizer(values)
    mode_bank = DiagonalGaussianModeBank(
        values.shape[1],
        int(config.get("n_modes", 4)),
        min_variance=float(config.get("min_variance", 1e-3)),
        support_quantile=float(config.get("support_quantile", 0.99)),
    )
    fit_result = mode_bank.fit(
        normalized,
        max_iterations=int(config.get("max_em_iterations", 100)),
        tolerance=float(config.get("em_tolerance", 1e-5)),
        seed=int(config.get("seed", 9826)),
    )
    output = Path(required(config, "output_path")).resolve()
    save_condition_bundle(
        output,
        descriptor=descriptor,
        standardizer=standardizer,
        mode_bank=mode_bank,
        fit_result=fit_result,
        provenance={
            "train_cache_index": str(cache.index_path),
            "unique_normal_environment_samples": values.shape[0],
            "descriptor": descriptor_info,
        },
    )
    summary = {
        "condition_bundle": str(output),
        "condition_bundle_sha256": file_digest(output),
        "raw_environment_samples": values.shape[0],
        "raw_environment_dimension": values.shape[1],
        "standardizer": standardizer.summary(),
        "mode_bank": mode_bank.summary(),
        "fit_result": {
            "iterations": fit_result.iterations,
            "converged": fit_result.converged,
            "final_mean_log_likelihood": fit_result.final_mean_log_likelihood,
            "support_threshold": fit_result.support_threshold,
            "component_counts": list(fit_result.component_counts),
        },
    }
    return atomic_write_json(summary, output.with_suffix(".summary.json"))


def main_cache(argv: Sequence[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build MSD-Flow paired feature cache")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = run_cache(load_json_config(args.config))
    print(json.dumps({"cache_index": str(path)}, ensure_ascii=False, indent=2))


def main_conditions(argv: Sequence[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fit MSD-Flow train-only environment modes")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = run_fit_conditions(load_json_config(args.config))
    print(json.dumps({"condition_summary": str(path)}, ensure_ascii=False, indent=2))
