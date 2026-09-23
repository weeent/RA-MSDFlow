"""RA-MSDFlow 的 manifest_builders 模块。"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .records import SampleRecord, is_image_file, iter_image_files, make_base_id


# 按评估协议处理。
# `manifest_builders` 的实现说明。
VISA_CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
MPDD_CATEGORIES = (
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
)


class ManifestError(RuntimeError):
    """`ManifestError` 组件。"""


@dataclass(slots=True)
class AuditReport:
    """`AuditReport` 组件。"""

    total_records: int
    counts_by_dataset: dict[str, int]
    counts_by_category: dict[str, int]
    counts_by_split: dict[str, int]
    counts_by_domain: dict[str, int]
    counts_by_label: dict[str, int]
    missing_images: list[str]
    missing_masks: list[str]
    duplicate_image_paths: list[str]
    unreadable_images: list[str]
    base_id_cross_split: dict[str, list[str]]

    @property
    def valid(self) -> bool:
        return not (
            self.missing_images
            or self.missing_masks
            or self.duplicate_image_paths
            or self.unreadable_images
            or self.base_id_cross_split
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "total_records": self.total_records,
            "counts_by_dataset": self.counts_by_dataset,
            "counts_by_category": self.counts_by_category,
            "counts_by_split": self.counts_by_split,
            "counts_by_domain": self.counts_by_domain,
            "counts_by_label": self.counts_by_label,
            "missing_images": self.missing_images,
            "missing_masks": self.missing_masks,
            "duplicate_image_paths": self.duplicate_image_paths,
            "unreadable_images": self.unreadable_images,
            "base_id_cross_split": self.base_id_cross_split,
        }


def _require_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise ManifestError(f"{description} does not exist or is not a directory: {path}")
    return path


def _normalise_domain(parts: Sequence[str]) -> str:
    """执行 `_normalise_domain` 所需的处理。"""

    cleaned = [part for part in parts if part and part != "."]
    return "/".join(cleaned) if cleaned else "regular"


def _mask_candidates(directory: Path, image: Path) -> list[Path]:
    """执行 `_mask_candidates` 所需的处理。"""

    stem = image.stem
    return [
        directory / f"{stem}_mask.png",
        directory / f"{stem}_mask.jpg",
        directory / f"{stem}.png",
        directory / f"{stem}.jpg",
    ]


def _find_mask(directory: Path, image: Path) -> str | None:
    for candidate in _mask_candidates(directory, image):
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _new_record(
    *, dataset: str, category: str, root: Path, image: Path, label: int, defect_type: str,
    split: str, official_split: str, domain: str, mask_path: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SampleRecord:
    # 校验并规范化数据路径。
    relative_path = image.relative_to(root).as_posix()
    return SampleRecord(
        dataset=dataset,
        category=category,
        image_path=str(image.resolve()),
        label=label,
        defect_type=defect_type,
        split=split,
        official_split=official_split,
        domain=domain,
        base_id=make_base_id(dataset, category, official_split, relative_path),
        mask_path=mask_path,
        metadata=dict(metadata or {}),
    )


def build_mvtec_ad_manifest(root: str | Path) -> list[SampleRecord]:
    """执行 `build_mvtec_ad_manifest` 所需的处理。"""

    dataset_root = _require_directory(Path(root), "MVTec AD root")
    records: list[SampleRecord] = []
    for category_dir in sorted((path for path in dataset_root.iterdir() if path.is_dir()), key=lambda path: path.name):
        category = category_dir.name
        train_good = category_dir / "train" / "good"
        test_dir = category_dir / "test"
        if not train_good.is_dir() or not test_dir.is_dir():
            # 校验并规范化数据路径。
            continue

        # 保持掩码与图像几何一致。
        for image in iter_image_files(train_good):
            records.append(
                _new_record(
                    dataset="mvtec_ad", category=category, root=dataset_root, image=image,
                    label=0, defect_type="good", split="train", official_split="train", domain="regular",
                )
            )

        # 保持掩码与图像几何一致。
        for defect_dir in sorted((path for path in test_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image in iter_image_files(defect_dir):
                mask_path = None if label == 0 else _find_mask(category_dir / "ground_truth" / defect_type, image)
                records.append(
                    _new_record(
                        dataset="mvtec_ad", category=category, root=dataset_root, image=image,
                        label=label, defect_type=defect_type, split="test", official_split="test", domain="regular",
                        mask_path=mask_path, metadata={"mask_expected": label == 1},
                    )
                )
    if not records:
        raise ManifestError(f"no standard MVTec AD records found under {dataset_root}")
    return records


def _strict_category_directories(
    root: Path,
    *,
    expected_categories: Sequence[str],
    evaluation_names: Sequence[str],
    dataset_name: str,
) -> list[Path]:
    """执行 `_strict_category_directories` 所需的处理。"""

    expected = set(expected_categories)
    detected: list[Path] = []
    for child in sorted(
        (path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name
    ):
        has_train = (child / "train" / "good").is_dir()
        has_evaluation = any((child / name).is_dir() for name in evaluation_names)
        if has_train and has_evaluation:
            detected.append(child)
    names = {path.name for path in detected}
    if names != expected:
        missing = sorted(expected.difference(names))
        unexpected = sorted(names.difference(expected))
        raise ManifestError(
            f"{dataset_name} category extraction is incomplete or unexpected under {root}; "
            f"missing={missing}, unexpected={unexpected}, detected={sorted(names)}"
        )
    return detected


def _assert_normal_only_training_directory(category_dir: Path, *, dataset_name: str) -> None:
    """执行 `_assert_normal_only_training_directory` 所需的处理。"""

    train_root = category_dir / "train"
    unexpected = [
        image
        for image in iter_image_files(train_root)
        if image.relative_to(train_root).parts[0] != "good"
    ]
    if unexpected:
        preview = ", ".join(
            path.relative_to(category_dir).as_posix() for path in unexpected[:5]
        )
        raise ManifestError(
            f"{dataset_name}/{category_dir.name} contains images outside train/good: {preview}"
        )


def _resolve_visa_1cls_root(root: Path) -> Path:
    """执行 `_resolve_visa_1cls_root` 所需的处理。"""

    dataset_root = _require_directory(root, "VisA root")
    for candidate in (dataset_root, dataset_root / "1cls"):
        if not candidate.is_dir():
            continue
        detected = {
            child.name
            for child in candidate.iterdir()
            if child.is_dir()
            and (child / "train" / "good").is_dir()
            and (child / "test").is_dir()
        }
        if detected == set(VISA_CATEGORIES):
            return candidate
    raw_layout_detected = any(
        (child / "Data" / "Images").is_dir()
        for child in dataset_root.iterdir()
        if child.is_dir()
    )
    suffix = (
        " The raw Data/Images layout was detected; first run VisA's official "
        "utils/prepare_data.py with split_csv/1cls.csv so the published split "
        "and binary masks are preserved."
        if raw_layout_detected
        else ""
    )
    raise ManifestError(
        f"no complete VisA 1cls MVTec-style layout found under {dataset_root}.{suffix}"
    )


def _build_visa_official_csv_manifest(dataset_root: Path, split_file: Path) -> list[SampleRecord]:
    """执行 `_build_visa_official_csv_manifest` 所需的处理。"""

    records: list[SampleRecord] = []
    with split_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"object", "split", "label", "image", "mask"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ManifestError(
                f"VisA split CSV {split_file} must contain columns {sorted(required)}"
            )
        for line_number, row in enumerate(reader, start=2):
            category = str(row["object"]).strip()
            official_split = str(row["split"]).strip().lower()
            label_name = str(row["label"]).strip().lower()
            if category not in VISA_CATEGORIES:
                raise ManifestError(
                    f"unknown VisA category {category!r} at {split_file}:{line_number}"
                )
            if official_split not in {"train", "test"}:
                raise ManifestError(
                    f"unknown VisA split {official_split!r} at {split_file}:{line_number}"
                )
            if label_name not in {"normal", "anomaly"}:
                raise ManifestError(
                    f"unknown VisA label {label_name!r} at {split_file}:{line_number}"
                )
            label = 0 if label_name == "normal" else 1
            if official_split == "train" and label != 0:
                raise ManifestError(
                    f"VisA official train contains an anomalous row at {split_file}:{line_number}"
                )
            image_value = str(row["image"]).strip()
            if not image_value:
                raise ManifestError(f"empty VisA image path at {split_file}:{line_number}")
            image = (dataset_root / Path(*image_value.split("/"))).resolve()
            # 校验并规范化数据路径。
            # 校验并规范化数据路径。
            try:
                image.relative_to(dataset_root.resolve())
            except ValueError as error:
                raise ManifestError(
                    f"VisA image path escapes dataset root at {split_file}:{line_number}"
                ) from error
            mask_value = str(row.get("mask", "")).strip()
            mask_path: str | None = None
            if label == 1:
                if not mask_value:
                    raise ManifestError(
                        f"VisA anomaly has no mask path at {split_file}:{line_number}"
                    )
                mask = (dataset_root / Path(*mask_value.split("/"))).resolve()
                try:
                    mask.relative_to(dataset_root.resolve())
                except ValueError as error:
                    raise ManifestError(
                        f"VisA mask path escapes dataset root at {split_file}:{line_number}"
                    ) from error
                mask_path = str(mask)
            elif mask_value:
                raise ManifestError(
                    f"VisA normal row unexpectedly references a mask at {split_file}:{line_number}"
                )
            records.append(
                _new_record(
                    dataset="visa",
                    category=category,
                    root=dataset_root,
                    image=image,
                    label=label,
                    defect_type="good" if label == 0 else "anomaly",
                    split=official_split,
                    official_split=official_split,
                    domain="regular",
                    mask_path=mask_path,
                    metadata={
                        "mask_expected": label == 1,
                        "source_protocol": "official_1cls_csv",
                        "split_csv_line": line_number,
                    },
                )
            )
    categories = {record.category for record in records}
    if categories != set(VISA_CATEGORIES):
        raise ManifestError(
            f"VisA 1cls CSV category set mismatch; missing={sorted(set(VISA_CATEGORIES) - categories)}, "
            f"unexpected={sorted(categories - set(VISA_CATEGORIES))}"
        )
    return records


def build_visa_manifest(root: str | Path) -> list[SampleRecord]:
    """执行 `build_visa_manifest` 所需的处理。"""

    requested_root = _require_directory(Path(root), "VisA root")
    split_file = requested_root / "split_csv" / "1cls.csv"
    if split_file.is_file():
        records = _build_visa_official_csv_manifest(requested_root, split_file)
        if not records:
            raise ManifestError(f"no VisA records found in {split_file}")
        return records

    dataset_root = _resolve_visa_1cls_root(requested_root)
    categories = _strict_category_directories(
        dataset_root,
        expected_categories=VISA_CATEGORIES,
        evaluation_names=("test",),
        dataset_name="VisA",
    )
    records: list[SampleRecord] = []
    for category_dir in categories:
        category = category_dir.name
        _assert_normal_only_training_directory(category_dir, dataset_name="VisA")
        for image in iter_image_files(category_dir / "train" / "good"):
            records.append(
                _new_record(
                    dataset="visa",
                    category=category,
                    root=dataset_root,
                    image=image,
                    label=0,
                    defect_type="good",
                    split="train",
                    official_split="train",
                    domain="regular",
                    metadata={"source_protocol": "official_1cls"},
                )
            )
        for defect_dir in sorted(
            (path for path in (category_dir / "test").iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image in iter_image_files(defect_dir):
                mask_path = (
                    None
                    if label == 0
                    else _find_mask(category_dir / "ground_truth" / defect_type, image)
                )
                records.append(
                    _new_record(
                        dataset="visa",
                        category=category,
                        root=dataset_root,
                        image=image,
                        label=label,
                        defect_type=defect_type,
                        split="test",
                        official_split="test",
                        domain="regular",
                        mask_path=mask_path,
                        metadata={
                            "mask_expected": label == 1,
                            "source_protocol": "official_1cls",
                        },
                    )
                )
    if not records:
        raise ManifestError(f"no VisA records found under {dataset_root}")
    return records


def _resolve_mpdd_root(root: Path) -> Path:
    """执行 `_resolve_mpdd_root` 所需的处理。"""

    dataset_root = _require_directory(root, "MPDD root")
    for candidate in (dataset_root, dataset_root / "MPDD"):
        if not candidate.is_dir():
            continue
        detected = {
            child.name
            for child in candidate.iterdir()
            if child.is_dir()
            and (child / "train" / "good").is_dir()
            and any((child / name).is_dir() for name in ("test", "validation", "val"))
        }
        if detected == set(MPDD_CATEGORIES):
            return candidate
    raise ManifestError(f"no complete six-category MPDD layout found under {dataset_root}")


def build_mpdd_manifest(root: str | Path) -> list[SampleRecord]:
    """执行 `build_mpdd_manifest` 所需的处理。"""

    dataset_root = _resolve_mpdd_root(Path(root))
    categories = _strict_category_directories(
        dataset_root,
        expected_categories=MPDD_CATEGORIES,
        evaluation_names=("test", "validation", "val"),
        dataset_name="MPDD",
    )
    records: list[SampleRecord] = []
    for category_dir in categories:
        category = category_dir.name
        _assert_normal_only_training_directory(category_dir, dataset_name="MPDD")
        for image in iter_image_files(category_dir / "train" / "good"):
            records.append(
                _new_record(
                    dataset="mpdd",
                    category=category,
                    root=dataset_root,
                    image=image,
                    label=0,
                    defect_type="good",
                    split="train",
                    official_split="train",
                    domain="regular",
                )
            )
        evaluation_directories = [
            category_dir / name
            for name in ("test", "validation", "val")
            if (category_dir / name).is_dir()
        ]
        if len(evaluation_directories) != 1:
            raise ManifestError(
                f"MPDD/{category} must contain exactly one of test, validation, or val; "
                f"found={[path.name for path in evaluation_directories]}"
            )
        evaluation_dir = evaluation_directories[0]
        mask_root = next(
            (
                candidate
                for candidate in (category_dir / "ground_truth", category_dir / "masks")
                if candidate.is_dir()
            ),
            category_dir / "ground_truth",
        )
        for defect_dir in sorted(
            (path for path in evaluation_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image in iter_image_files(defect_dir):
                mask_path = None if label == 0 else _find_mask(mask_root / defect_type, image)
                records.append(
                    _new_record(
                        dataset="mpdd",
                        category=category,
                        root=dataset_root,
                        image=image,
                        label=label,
                        defect_type=defect_type,
                        split="test",
                        official_split=evaluation_dir.name,
                        domain="regular",
                        mask_path=mask_path,
                        metadata={"mask_expected": label == 1},
                    )
                )
    if not records:
        raise ManifestError(f"no MPDD records found under {dataset_root}")
    return records


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestError(f"invalid metadata JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ManifestError(f"metadata row at {path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _robustad_mask_path(root: Path, value: object | None) -> str | None:
    if not value:
        return None
    candidate = Path(str(value))
    # 校验并规范化数据路径。
    if not candidate.is_absolute():
        candidate = root / candidate
    return str(candidate.resolve())


def build_robustad_manifest(root: str | Path) -> list[SampleRecord]:
    """执行 `build_robustad_manifest` 所需的处理。"""

    dataset_root = _require_directory(Path(root), "RobustAD root")
    records: list[SampleRecord] = []
    for category_dir in sorted((path for path in dataset_root.iterdir() if path.is_dir()), key=lambda path: path.name):
        category = category_dir.name
        partitions = sorted((path for path in category_dir.iterdir() if path.is_dir() and "_data_dir_" in path.name), key=lambda path: path.name)
        for partition in partitions:
            suffix = partition.name.rsplit("_data_dir_", maxsplit=1)[-1]
            is_train = suffix == "train"
            split = "train" if is_train else "test"
            official_split = "train" if is_train else suffix
            # 按评估协议处理。
            # `build_robustad_manifest` 的实现说明。
            domain = "train" if is_train else suffix
            metadata_path = partition / "metadata.jsonl"
            if metadata_path.is_file():
                rows = _read_jsonl(metadata_path)
                for row in rows:
                    relative_name = row.get("file_name")
                    if relative_name is None:
                        raise ManifestError(f"missing file_name in {metadata_path}")
                    image = partition / str(relative_name)
                    label = int(row.get("label", -1))
                    if label not in {0, 1}:
                        raise ManifestError(f"unsupported RobustAD label {label} in {metadata_path}")
                    raw_mask = row.get("mask")
                    mask_path = _robustad_mask_path(dataset_root, raw_mask) if label == 1 else None
                    records.append(
                        _new_record(
                            dataset="robustad", category=category, root=dataset_root, image=image,
                            label=label, defect_type="good" if label == 0 else "anomaly", split=split,
                            official_split=official_split, domain=domain, mask_path=mask_path,
                            # 保持掩码与图像几何一致。
                            metadata={"mask_expected": label == 1 and raw_mask is not None, "partition": partition.name},
                        )
                    )
                continue

            # `build_robustad_manifest` 的实现说明。
            for label_name, label in (("normal", 0), ("anomaly", 1)):
                image_dir = partition / label_name
                for image in iter_image_files(image_dir):
                    mask_path = _find_mask(partition / "masks", image) if label == 1 else None
                    records.append(
                        _new_record(
                            dataset="robustad", category=category, root=dataset_root, image=image,
                            label=label, defect_type="good" if label == 0 else "anomaly", split=split,
                            official_split=official_split, domain=domain, mask_path=mask_path,
                            metadata={"mask_expected": label == 1 and mask_path is not None, "partition": partition.name, "metadata_fallback": True},
                        )
                    )
    if not records:
        raise ManifestError(f"no RobustAD records found under {dataset_root}")
    return records


def _resolve_aebad_root(root: Path, variant: str) -> Path:
    candidate = root / f"AeBAD_{variant}"
    return candidate if candidate.is_dir() else root


def _aebad_s_mask_path(source: Path, defect_type: str, relative_under_defect: Path) -> str | None:
    exact = source / "ground_truth" / defect_type / relative_under_defect
    if exact.is_file():
        return str(exact.resolve())
    return _find_mask(exact.parent, relative_under_defect)


def _build_aebad_s_manifest(root: Path) -> list[SampleRecord]:
    source = _resolve_aebad_root(root, "S")
    _require_directory(source / "train" / "good", "AeBAD-S train/good")
    _require_directory(source / "test", "AeBAD-S test")
    records: list[SampleRecord] = []

    # 校验并规范化数据路径。
    for image in iter_image_files(source / "train" / "good"):
        relative = image.relative_to(source / "train" / "good")
        records.append(
            _new_record(
                dataset="aebad_s", category="aebad_s", root=source, image=image,
                label=0, defect_type="good", split="train", official_split="train",
                domain=_normalise_domain(relative.parts[:-1]),
            )
        )
    for defect_dir in sorted((path for path in (source / "test").iterdir() if path.is_dir()), key=lambda path: path.name):
        defect_type = defect_dir.name
        label = 0 if defect_type == "good" else 1
        for image in iter_image_files(defect_dir):
            relative = image.relative_to(defect_dir)
            mask_path = None if label == 0 else _aebad_s_mask_path(source, defect_type, relative)
            records.append(
                _new_record(
                    dataset="aebad_s", category="aebad_s", root=source, image=image,
                    label=label, defect_type=defect_type, split="test", official_split="test",
                    domain=_normalise_domain(relative.parts[:-1]), mask_path=mask_path,
                    metadata={"mask_expected": label == 1},
                )
            )
    return records


def _build_aebad_v_manifest(root: Path) -> list[SampleRecord]:
    source = _resolve_aebad_root(root, "V")
    _require_directory(source / "train" / "good", "AeBAD-V train/good")
    _require_directory(source / "test", "AeBAD-V test")
    records: list[SampleRecord] = []
    for image in iter_image_files(source / "train" / "good"):
        relative = image.relative_to(source / "train" / "good")
        records.append(
            _new_record(
                dataset="aebad_v", category="aebad_v", root=source, image=image,
                label=0, defect_type="good", split="train", official_split="train",
                domain=_normalise_domain(relative.parts[:-1]),
            )
        )
    for video_dir in sorted((path for path in (source / "test").iterdir() if path.is_dir()), key=lambda path: path.name):
        for defect_dir in sorted((path for path in video_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image in iter_image_files(defect_dir):
                records.append(
                    _new_record(
                        dataset="aebad_v", category="aebad_v", root=source, image=image,
                        label=label, defect_type=defect_type, split="test", official_split="test",
                        domain=video_dir.name, metadata={"mask_expected": False},
                    )
                )
    return records


def build_aebad_manifest(root: str | Path, variant: str = "S") -> list[SampleRecord]:
    """执行 `build_aebad_manifest` 所需的处理。"""

    dataset_root = _require_directory(Path(root), "AeBAD root")
    normalized = variant.lower()
    if normalized == "s":
        records = _build_aebad_s_manifest(dataset_root)
    elif normalized == "v":
        records = _build_aebad_v_manifest(dataset_root)
    elif normalized == "both":
        records = _build_aebad_s_manifest(dataset_root) + _build_aebad_v_manifest(dataset_root)
    else:
        raise ValueError("AeBAD variant must be 'S', 'V', or 'both'")
    if not records:
        raise ManifestError(f"no AeBAD-{variant} records found under {dataset_root}")
    return records


def _mvtec_ad2_domain(image: Path) -> str:
    """执行 `_mvtec_ad2_domain` 所需的处理。"""

    parts = image.stem.split("_", maxsplit=1)
    return parts[1] if len(parts) == 2 else "regular"


def build_mvtec_ad2_manifest(root: str | Path) -> list[SampleRecord]:
    """执行 `build_mvtec_ad2_manifest` 所需的处理。"""

    dataset_root = _require_directory(Path(root), "MVTec AD 2 root")
    records: list[SampleRecord] = []
    split_order = ("train", "validation", "test_public", "test_private", "test_private_mixed")
    for category_dir in sorted((path for path in dataset_root.iterdir() if path.is_dir()), key=lambda path: path.name):
        category = category_dir.name
        for official_split in split_order:
            split_dir = category_dir / official_split
            if not split_dir.is_dir():
                continue
            if official_split in {"train", "validation"}:
                for image in iter_image_files(split_dir / "good"):
                    records.append(
                        _new_record(
                            dataset="mvtec_ad2", category=category, root=dataset_root, image=image,
                            label=0, defect_type="good", split=official_split, official_split=official_split,
                            domain=_mvtec_ad2_domain(image),
                        )
                    )
                continue
            if official_split == "test_public":
                for type_dir in sorted((path for path in split_dir.iterdir() if path.is_dir() and path.name != "ground_truth"), key=lambda path: path.name):
                    label = 0 if type_dir.name == "good" else 1
                    for image in iter_image_files(type_dir):
                        mask_path = None
                        if label == 1:
                            mask_path = _find_mask(split_dir / "ground_truth" / "bad", image)
                        records.append(
                            _new_record(
                                dataset="mvtec_ad2", category=category, root=dataset_root, image=image,
                                label=label, defect_type=type_dir.name, split="test", official_split=official_split,
                                domain=_mvtec_ad2_domain(image), mask_path=mask_path,
                                metadata={"mask_expected": label == 1, "labels_available": True},
                            )
                        )
                continue
            # 按评估协议处理。
            # 按评估协议处理。
            for image in iter_image_files(split_dir):
                records.append(
                    _new_record(
                        dataset="mvtec_ad2", category=category, root=dataset_root, image=image,
                        label=-1, defect_type="unknown", split=official_split, official_split=official_split,
                        domain=_mvtec_ad2_domain(image), metadata={"mask_expected": False, "labels_available": False},
                    )
                )
    if not records:
        raise ManifestError(f"no MVTec AD 2 records found under {dataset_root}")
    return records


def build_dataset_manifest(dataset: str, root: str | Path, *, aebad_variant: str = "S") -> list[SampleRecord]:
    """执行 `build_dataset_manifest` 所需的处理。"""

    builders: dict[str, Callable[[str | Path], list[SampleRecord]]] = {
        "mvtec_ad": build_mvtec_ad_manifest,
        "robustad": build_robustad_manifest,
        "mvtec_ad2": build_mvtec_ad2_manifest,
        "visa": build_visa_manifest,
        "mpdd": build_mpdd_manifest,
    }
    normalized = dataset.lower().replace("-", "_")
    if normalized == "aebad":
        return build_aebad_manifest(root, variant=aebad_variant)
    if normalized not in builders:
        raise ValueError(
            f"unsupported dataset '{dataset}'; choose mvtec_ad, robustad, aebad, "
            "mvtec_ad2, visa, or mpdd"
        )
    return builders[normalized](root)


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def audit_records(records: Sequence[SampleRecord], *, verify_images: bool = False) -> AuditReport:
    """执行 `audit_records` 所需的处理。"""

    # `audit_records` 的实现说明。
    # `audit_records` 的实现说明。
    path_counts: Counter[tuple[str, str, str]] = Counter(
        (record.image_path, record.split, record.variant_id) for record in records
    )
    duplicate_paths = sorted(
        f"{path} [split={split}, variant={variant}]"
        for (path, split, variant), count in path_counts.items()
        if count > 1
    )
    missing_images = sorted(record.image_path for record in records if not Path(record.image_path).is_file())
    missing_masks = sorted(
        record.mask_path or "<missing mask path>"
        for record in records
        if record.is_anomalous and bool(record.metadata.get("mask_expected", False)) and (record.mask_path is None or not Path(record.mask_path).is_file())
    )
    base_to_splits: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.split in {"train", "val"}:
            base_to_splits[record.base_id].add(record.split)
    base_id_cross_split = {base_id: sorted(splits) for base_id, splits in base_to_splits.items() if len(splits) > 1}
    unreadable: list[str] = []
    if verify_images:
        from PIL import Image

        for image_path in sorted(set(record.image_path for record in records if Path(record.image_path).is_file())):
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError):
                unreadable.append(image_path)
    return AuditReport(
        total_records=len(records),
        counts_by_dataset=_counter_dict(record.dataset for record in records),
        counts_by_category=_counter_dict(f"{record.dataset}/{record.category}" for record in records),
        counts_by_split=_counter_dict(record.split for record in records),
        counts_by_domain=_counter_dict(f"{record.dataset}/{record.category}/{record.domain}" for record in records),
        counts_by_label=_counter_dict(str(record.label) for record in records),
        missing_images=missing_images,
        missing_masks=missing_masks,
        duplicate_image_paths=duplicate_paths,
        unreadable_images=unreadable,
        base_id_cross_split=base_id_cross_split,
    )
