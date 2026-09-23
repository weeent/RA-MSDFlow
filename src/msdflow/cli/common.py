"""CLI 共用的配置、记录选择、描述器加载和 DataLoader 工具。"""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from envfm.data.records import SampleRecord, read_records_jsonl
from envfm.training.checkpoint import file_digest
from msdflow.conditions import HybridEnvironmentDescriptor
from msdflow.training import TrainingConfig


def load_json_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("configuration must contain one JSON object")
    value["_config_path"] = str(source)
    value["_config_sha256"] = file_digest(source)
    return value


def required(config: Mapping[str, Any], key: str) -> Any:
    if key not in config or config[key] is None or config[key] == "":
        raise ValueError(f"configuration is missing required field {key!r}")
    return config[key]


def select_records(
    manifest_path: str | Path,
    *,
    category: str,
    split: str,
    dataset: str | None = None,
    normal_only: bool = False,
) -> list[SampleRecord]:
    records = [
        record
        for record in read_records_jsonl(manifest_path)
        if record.category == category
        and record.split == split
        and (dataset is None or record.dataset == dataset)
        and (not normal_only or record.label == 0)
    ]
    if not records:
        raise ValueError(f"no records for dataset={dataset}, category={category}, split={split}")
    if normal_only and any(record.label != 0 for record in records):
        raise ValueError("normal-only selection contains invalid labels")
    return records


def training_config(value: Mapping[str, Any] | None) -> TrainingConfig:
    source = dict(value or {})
    allowed = {field.name for field in fields(TrainingConfig)}
    unknown = set(source).difference(allowed)
    if unknown:
        raise ValueError(f"unknown training settings: {sorted(unknown)}")
    return TrainingConfig(**source)


def make_loader(
    dataset: object,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def load_descriptor(
    *,
    learned_dim: int,
    lowres_size: int = 32,
    checkpoint_path: str | Path | None = None,
) -> tuple[HybridEnvironmentDescriptor, dict[str, object]]:
    """加载环境编码预训练 best；learned_dim=0 时不需要 checkpoint。"""

    descriptor = HybridEnvironmentDescriptor(learned_dim=learned_dim, lowres_size=lowres_size)
    info: dict[str, object] = {"learned_dim": learned_dim, "checkpoint": None}
    if checkpoint_path is None:
        if learned_dim > 0:
            raise ValueError("learned environment descriptor requires descriptor_checkpoint")
        return descriptor, info
    source = Path(checkpoint_path).resolve()
    try:
        state = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(source, map_location="cpu")
    if state.get("stage") != "environment_code_pretraining":
        raise ValueError("descriptor checkpoint has the wrong training stage")
    model_state = state["model_state_dict"]
    encoder_state = {
        name.removeprefix("encoder."): value
        for name, value in model_state.items()
        if name.startswith("encoder.")
    }
    if descriptor.learned_encoder is None or not encoder_state:
        raise ValueError("descriptor checkpoint does not contain a learned encoder")
    descriptor.learned_encoder.load_state_dict(encoder_state, strict=True)
    info["checkpoint"] = {"path": str(source), "sha256": file_digest(source), "epoch": int(state["epoch"])}
    return descriptor, info
