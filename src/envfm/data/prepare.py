"""RA-MSDFlow 的 prepare 模块。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import IndustrialAnomalyDataset, fit_condition_standardizer
from .manifest_builders import audit_records, build_dataset_manifest
from .records import SampleRecord, read_records_jsonl, records_digest, write_records_jsonl
from .splits import (
    assert_no_train_val_overlap,
    expand_evaluation_photometric_variants,
    expand_training_photometric_variants,
    split_summary,
    stable_normal_split,
)


DEFAULT_TRAIN_VARIANTS = ("clean", "exposure", "white_balance", "gradient")
DEFAULT_MVTEC_EVAL_VARIANTS = ("clean", "exposure", "white_balance", "gradient", "compound")


def audit_prepared_protocol(
    records: Sequence[SampleRecord],
    *,
    train_variants: Sequence[str],
    controlled_test_domains: bool,
    test_variants: Sequence[str],
) -> dict[str, object]:
    """执行 `audit_prepared_protocol` 所需的处理。"""

    categories = sorted({record.category for record in records})
    train_val_non_normal = [
        record.base_id
        for record in records
        if record.split in {"train", "val"} and record.label != 0
    ]
    train_ids = {record.base_id for record in records if record.split == "train"}
    val_ids = {record.base_id for record in records if record.split == "val"}
    train_variants_by_base: dict[str, set[str]] = {}
    test_variants_by_base: dict[str, set[str]] = {}
    for record in records:
        if record.split == "train":
            train_variants_by_base.setdefault(record.base_id, set()).add(record.variant_id)
        elif record.split == "test":
            test_variants_by_base.setdefault(record.base_id, set()).add(record.variant_id)
    expected_train = set(train_variants)
    expected_test = set(test_variants) if controlled_test_domains else {"clean"}
    bad_train_variants = sorted(
        base_id
        for base_id, variants in train_variants_by_base.items()
        if variants != expected_train
    )
    bad_test_variants = sorted(
        base_id
        for base_id, variants in test_variants_by_base.items()
        if variants != expected_test
    )
    non_clean_validation = sorted(
        record.base_id
        for record in records
        if record.split == "val" and (record.variant_id != "clean" or bool(record.augmentation))
    )
    category_counts: dict[str, dict[str, int]] = {}
    categories_missing_required_split: list[str] = []
    for category in categories:
        counts = {
            split: sum(
                record.category == category and record.split == split and record.label == 0
                for record in records
            )
            for split in ("train", "val", "test")
        }
        counts["test_anomaly"] = sum(
            record.category == category and record.split == "test" and record.label == 1
            for record in records
        )
        category_counts[category] = counts
        if (
            counts["train"] == 0
            or counts["val"] == 0
            or counts["test"] == 0
            or counts["test_anomaly"] == 0
        ):
            categories_missing_required_split.append(category)
    overlap = sorted(train_ids.intersection(val_ids))
    valid = not (
        train_val_non_normal
        or overlap
        or bad_train_variants
        or bad_test_variants
        or non_clean_validation
        or categories_missing_required_split
    )
    return {
        "valid": valid,
        "train_val_non_normal_count": len(train_val_non_normal),
        "train_val_non_normal_preview": train_val_non_normal[:20],
        "train_val_base_overlap_count": len(overlap),
        "train_val_base_overlap_preview": overlap[:20],
        "bad_train_variant_base_count": len(bad_train_variants),
        "bad_train_variant_base_preview": bad_train_variants[:20],
        "bad_test_variant_base_count": len(bad_test_variants),
        "bad_test_variant_base_preview": bad_test_variants[:20],
        "non_clean_validation_count": len(non_clean_validation),
        "non_clean_validation_preview": non_clean_validation[:20],
        "categories_missing_required_split": categories_missing_required_split,
        "category_counts": category_counts,
        "expected_train_variants": sorted(expected_train),
        "expected_test_variants": sorted(expected_test),
    }


@dataclass(frozen=True, slots=True)
class DatasetPreparationSpec:
    """`DatasetPreparationSpec` 组件。"""

    name: str
    root: str
    builder: str
    aebad_variant: str = "S"
    use_official_validation: bool = False
    controlled_test_domains: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetPreparationSpec":
        required = {"name", "root", "builder"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"dataset preparation entry is missing: {sorted(missing)}")
        return cls(
            name=str(value["name"]),
            root=str(value["root"]),
            builder=str(value["builder"]),
            aebad_variant=str(value.get("aebad_variant", "S")),
            use_official_validation=bool(value.get("use_official_validation", False)),
            controlled_test_domains=bool(value.get("controlled_test_domains", False)),
        )


def _adopt_official_validation(records: Sequence[SampleRecord]) -> list[SampleRecord]:
    """执行 `_adopt_official_validation` 所需的处理。"""

    adopted = [
        record.with_updates(split="val")
        if record.split == "validation" and record.label == 0
        else record
        for record in records
    ]
    if not any(record.split == "val" for record in adopted):
        raise ValueError("use_official_validation was requested, but no normal validation records exist")
    if not any(record.split == "train" and record.label == 0 for record in adopted):
        raise ValueError("dataset has no normal training records")
    assert_no_train_val_overlap(adopted)
    return adopted


def prepare_records_for_training(
    records: Sequence[SampleRecord],
    *,
    use_official_validation: bool,
    val_ratio: float,
    seed: int,
    train_variants: Sequence[str] = DEFAULT_TRAIN_VARIANTS,
    controlled_test_domains: bool = False,
    test_variants: Sequence[str] = DEFAULT_MVTEC_EVAL_VARIANTS,
) -> list[SampleRecord]:
    """执行 `prepare_records_for_training` 所需的处理。"""

    if not records:
        raise ValueError("cannot prepare an empty raw manifest")
    # 步骤 1：构建数据划分。
    split_records = (
        _adopt_official_validation(records)
        if use_official_validation
        else stable_normal_split(records, val_ratio=val_ratio, seed=seed, eligible_splits=("train",))
    )
    # 步骤 2：按当前协议处理。
    prepared = expand_training_photometric_variants(
        split_records,
        variants=tuple(train_variants),
        seed=seed,
    )
    # 步骤 3：按当前协议处理。
    if controlled_test_domains:
        prepared = expand_evaluation_photometric_variants(
            prepared,
            variants=tuple(test_variants),
            seed=seed,
            eligible_splits=("test",),
        )
    assert_no_train_val_overlap(prepared)
    return prepared


def _write_json(value: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
    return path


def prepare_dataset(
    spec: DatasetPreparationSpec,
    *,
    output_root: str | Path,
    seed: int = 9826,
    val_ratio: float = 0.2,
    image_size: int = 256,
    train_variants: Sequence[str] = DEFAULT_TRAIN_VARIANTS,
    test_variants: Sequence[str] = DEFAULT_MVTEC_EVAL_VARIANTS,
    reuse_raw_manifest: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    """执行 `prepare_dataset` 所需的处理。"""

    dataset_directory = Path(output_root).resolve() / spec.name
    dataset_directory.mkdir(parents=True, exist_ok=True)
    raw_path = dataset_directory / "raw.jsonl"
    prepared_path = dataset_directory / f"prepared_seed{seed}.jsonl"
    summary_path = dataset_directory / f"prepared_seed{seed}.summary.json"
    if (prepared_path.exists() or summary_path.exists()) and not overwrite:
        raise FileExistsError(f"prepared outputs already exist for {spec.name}; enable overwrite explicitly")

    # 步骤 4：按当前协议处理。
    if reuse_raw_manifest and raw_path.is_file():
        raw_records = read_records_jsonl(raw_path)
        raw_source = "reused_manifest"
    else:
        raw_records = build_dataset_manifest(spec.builder, spec.root, aebad_variant=spec.aebad_variant)
        raw_audit = audit_records(raw_records, verify_images=False)
        if not raw_audit.valid:
            raise ValueError(f"raw manifest audit failed for {spec.name}")
        write_records_jsonl(raw_records, raw_path, overwrite=overwrite)
        raw_source = "scanned_dataset_root"

    # 按清单约束处理样本。
    # 保持掩码与图像几何一致。
    raw_audit = audit_records(raw_records, verify_images=False)
    if not raw_audit.valid:
        raise ValueError(f"raw manifest audit failed for {spec.name}")

    prepared = prepare_records_for_training(
        raw_records,
        use_official_validation=spec.use_official_validation,
        val_ratio=val_ratio,
        seed=seed,
        train_variants=train_variants,
        controlled_test_domains=spec.controlled_test_domains,
        test_variants=test_variants,
    )
    prepared_audit = audit_records(prepared, verify_images=False)
    if not prepared_audit.valid:
        raise ValueError(f"prepared manifest audit failed for {spec.name}")
    protocol_audit = audit_prepared_protocol(
        prepared,
        train_variants=train_variants,
        controlled_test_domains=spec.controlled_test_domains,
        test_variants=test_variants,
    )
    if not bool(protocol_audit["valid"]):
        raise ValueError(f"prepared training protocol audit failed for {spec.name}: {protocol_audit}")
    write_records_jsonl(prepared, prepared_path, overwrite=overwrite)

    # 步骤 5：计算环境条件。
    category_statistics: dict[str, object] = {}
    for category in sorted({record.category for record in prepared}):
        category_records = [record for record in prepared if record.category == category]
        dataset = IndustrialAnomalyDataset(
            category_records,
            image_size=image_size,
            apply_augmentation=True,
            normalize=False,
        )
        standardizer, condition_summary = fit_condition_standardizer(dataset, only_normal_train=True)
        condition_path = dataset_directory / "conditions" / f"{category}_seed{seed}.json"
        standardizer.save(
            condition_path,
            manifest_digest=str(condition_summary["manifest_digest"]),
            overwrite=overwrite,
        )
        category_statistics[category] = {
            "path": str(condition_path),
            **condition_summary,
        }

    # 步骤 6：写入结果。
    summary: dict[str, object] = {
        "phase": "all_dataset_preparation",
        "dataset_spec": asdict(spec),
        "seed": seed,
        "val_ratio": val_ratio,
        "image_size": image_size,
        "train_variants": list(train_variants),
        "test_variants": list(test_variants) if spec.controlled_test_domains else [],
        "raw_source": raw_source,
        "raw_manifest": str(raw_path),
        "raw_records": len(raw_records),
        "raw_manifest_sha256": records_digest(raw_records),
        "raw_audit": raw_audit.to_dict(),
        "eligible_train_non_normal_records": sum(
            record.split == "train" and record.label != 0 for record in raw_records
        ),
        "prepared_manifest": str(prepared_path),
        "prepared_manifest_sha256": records_digest(prepared),
        "prepared_audit": prepared_audit.to_dict(),
        "prepared_protocol_audit": protocol_audit,
        "split_summary": split_summary(prepared),
        "categories": dict(sorted(Counter(record.category for record in prepared).items())),
        "domains": dict(sorted(Counter(record.domain for record in prepared).items())),
        "condition_statistics": category_statistics,
    }
    _write_json(summary, summary_path)
    return {**summary, "summary_path": str(summary_path)}
