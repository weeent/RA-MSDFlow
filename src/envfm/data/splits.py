"""RA-MSDFlow 的 splits 模块。"""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Sequence

from .photometric_augment import sample_photometric_params, stable_seed
from .records import SampleRecord


def _eligible_normal_base_ids(records: Sequence[SampleRecord], eligible_splits: set[str]) -> list[str]:
    """执行 `_eligible_normal_base_ids` 所需的处理。"""

    return sorted(
        {
            record.base_id
            for record in records
            if record.label == 0 and record.split in eligible_splits
        }
    )


def stable_normal_split(
    records: Sequence[SampleRecord], *, val_ratio: float = 0.2, seed: int = 9826,
    eligible_splits: Iterable[str] = ("train",),
) -> list[SampleRecord]:
    """执行 `stable_normal_split` 所需的处理。"""

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    eligible = set(eligible_splits)
    base_ids = _eligible_normal_base_ids(records, eligible)
    if len(base_ids) < 2 and val_ratio > 0.0:
        raise ValueError("at least two normal base images are needed for a non-empty validation split")

    # `stable_normal_split` 的实现说明。
    # `stable_normal_split` 的实现说明。
    group_by_base_id: dict[str, tuple[str, str]] = {}
    for record in records:
        if record.base_id in base_ids:
            group_by_base_id[record.base_id] = (record.dataset, record.category)
    grouped: dict[tuple[str, str], list[str]] = {}
    for base_id, group in group_by_base_id.items():
        grouped.setdefault(group, []).append(base_id)
    val_base_ids: set[str] = set()
    for group, group_base_ids in sorted(grouped.items()):
        if len(group_base_ids) < 2 and val_ratio > 0.0:
            raise ValueError(f"category {group[0]}/{group[1]} has fewer than two normal training images")
        # `stable_normal_split` 的实现说明。
        ranked = sorted(group_base_ids, key=lambda base_id: (stable_seed(seed, "split", base_id), base_id))
        requested = int(round(len(ranked) * val_ratio))
        val_count = min(max(requested, 1 if val_ratio > 0.0 else 0), max(len(ranked) - 1, 0))
        val_base_ids.update(ranked[:val_count])

    assigned: list[SampleRecord] = []
    for record in records:
        # 步骤 1：构建数据划分。
        # `stable_normal_split` 的实现说明。
        # `stable_normal_split` 的实现说明。
        if record.split in eligible and record.label != 0:
            continue
        if record.split in eligible:
            split = "val" if record.base_id in val_base_ids else "train"
            assigned.append(record.with_updates(split=split))
        else:
            # 步骤 2：按当前协议处理。
            # 按评估协议处理。
            assigned.append(record)
    assert_no_train_val_overlap(assigned)
    return assigned


def assert_no_train_val_overlap(records: Sequence[SampleRecord]) -> None:
    """执行 `assert_no_train_val_overlap` 所需的处理。"""

    train_ids = {record.base_id for record in records if record.split == "train"}
    val_ids = {record.base_id for record in records if record.split == "val"}
    overlap = sorted(train_ids.intersection(val_ids))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(f"base_id leakage between train and val ({len(overlap)}): {preview}")


def expand_training_photometric_variants(
    records: Sequence[SampleRecord], *, variants: Sequence[str] = ("clean", "exposure", "white_balance", "gradient"),
    seed: int = 9826,
) -> list[SampleRecord]:
    """执行 `expand_training_photometric_variants` 所需的处理。"""

    if not variants:
        raise ValueError("at least one photometric variant is required")
    if len(set(variants)) != len(variants):
        raise ValueError("photometric variants must be unique")
    expanded: list[SampleRecord] = []
    for record in records:
        if record.split == "train" and record.label == 0:
            if record.variant_id != "clean" or record.augmentation:
                raise ValueError(
                    "training manifest already contains photometric variants; expand only an unexpanded split manifest"
                )
            for variant in variants:
                # 固定可复现的随机状态。
                augmentation_seed = stable_seed(seed, "photometric", record.base_id, variant)
                params = sample_photometric_params(variant, augmentation_seed)
                expanded.append(
                    record.with_updates(
                        variant_id=variant,
                        augmentation_seed=augmentation_seed,
                        augmentation=params.to_dict(),
                    )
                )
        else:
            expanded.append(record)
    assert_no_train_val_overlap(expanded)
    return expanded


def expand_evaluation_photometric_variants(
    records: Sequence[SampleRecord],
    *,
    variants: Sequence[str] = ("clean", "exposure", "white_balance", "gradient", "compound"),
    seed: int = 9826,
    eligible_splits: Iterable[str] = ("test",),
) -> list[SampleRecord]:
    """执行 `expand_evaluation_photometric_variants` 所需的处理。"""

    if not variants or len(set(variants)) != len(variants):
        raise ValueError("evaluation variants must be non-empty and unique")
    eligible = set(eligible_splits)
    expanded: list[SampleRecord] = []
    for record in records:
        if record.split not in eligible:
            expanded.append(record)
            continue
        if record.variant_id != "clean" or record.augmentation:
            raise ValueError("evaluation expansion expects unmodified source records")
        for variant in variants:
            # 步骤 1：固定随机种子。
            augmentation_seed = stable_seed(seed, "evaluation_photometric", record.base_id, variant)
            params = sample_photometric_params(variant, augmentation_seed)
            # 步骤 2：处理像素掩码。
            expanded.append(
                record.with_updates(
                    domain="regular" if variant == "clean" else f"synthetic/{variant}",
                    variant_id=variant,
                    augmentation_seed=augmentation_seed,
                    augmentation=params.to_dict(),
                )
            )
    assert_no_train_val_overlap(expanded)
    return expanded


def split_summary(records: Sequence[SampleRecord]) -> dict[str, object]:
    """执行 `split_summary` 所需的处理。"""

    by_split = Counter(record.split for record in records)
    by_variant = Counter(record.variant_id for record in records)
    train_base_ids = {record.base_id for record in records if record.split == "train"}
    val_base_ids = {record.base_id for record in records if record.split == "val"}
    return {
        "records": len(records),
        "records_by_split": dict(sorted(by_split.items())),
        "records_by_variant": dict(sorted(by_variant.items())),
        "unique_train_base_ids": len(train_base_ids),
        "unique_val_base_ids": len(val_base_ids),
        "train_val_base_overlap": len(train_base_ids.intersection(val_base_ids)),
    }
