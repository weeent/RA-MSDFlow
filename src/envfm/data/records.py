"""RA-MSDFlow 的 records 模块。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def is_image_file(path: Path) -> bool:
    """执行 `is_image_file` 所需的处理。"""

    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def iter_image_files(root: Path) -> list[Path]:
    """执行 `iter_image_files` 所需的处理。"""

    if not root.is_dir():
        return []
    return sorted((path for path in root.rglob("*") if is_image_file(path)), key=lambda p: p.as_posix())


def make_base_id(dataset: str, category: str, official_split: str, relative_path: str) -> str:
    """执行 `make_base_id` 所需的处理。"""

    normalized = relative_path.replace("\\", "/").lstrip("/")
    return f"{dataset}/{category}/{official_split}/{normalized}"


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """`SampleRecord` 组件。"""

    dataset: str
    category: str
    image_path: str
    label: int
    defect_type: str
    split: str
    official_split: str
    domain: str
    base_id: str
    mask_path: str | None = None
    variant_id: str = "clean"
    augmentation_seed: int | None = None
    augmentation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in {-1, 0, 1}:
            raise ValueError(f"label must be -1, 0, or 1; received {self.label}")
        if not self.dataset or not self.category or not self.image_path:
            raise ValueError("dataset, category, and image_path must be non-empty")
        if not self.base_id:
            raise ValueError("base_id must be non-empty")
        if self.label == 0 and self.mask_path is not None:
            raise ValueError("normal records must not have a pixel mask")

    @property
    def is_normal(self) -> bool:
        return self.label == 0

    @property
    def is_anomalous(self) -> bool:
        return self.label == 1

    def with_updates(self, **changes: Any) -> "SampleRecord":
        """执行 `with_updates` 所需的处理。"""

        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """执行 `to_dict` 所需的处理。"""

        result = asdict(self)
        result["augmentation"] = dict(self.augmentation)
        result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SampleRecord":
        """执行 `from_dict` 所需的处理。"""

        known = {
            "dataset",
            "category",
            "image_path",
            "label",
            "defect_type",
            "split",
            "official_split",
            "domain",
            "base_id",
            "mask_path",
            "variant_id",
            "augmentation_seed",
            "augmentation",
            "metadata",
        }
        unknown = set(value).difference(known)
        if unknown:
            raise ValueError(f"unknown SampleRecord fields: {sorted(unknown)}")
        return cls(
            dataset=str(value["dataset"]),
            category=str(value["category"]),
            image_path=str(value["image_path"]),
            label=int(value["label"]),
            defect_type=str(value["defect_type"]),
            split=str(value["split"]),
            official_split=str(value["official_split"]),
            domain=str(value["domain"]),
            base_id=str(value["base_id"]),
            mask_path=None if value.get("mask_path") is None else str(value["mask_path"]),
            variant_id=str(value.get("variant_id", "clean")),
            augmentation_seed=(
                None if value.get("augmentation_seed") is None else int(value["augmentation_seed"])
            ),
            augmentation=dict(value.get("augmentation", {})),
            metadata=dict(value.get("metadata", {})),
        )


def canonical_record_json(record: SampleRecord) -> str:
    """执行 `canonical_record_json` 所需的处理。"""

    return json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def records_digest(records: Sequence[SampleRecord]) -> str:
    """执行 `records_digest` 所需的处理。"""

    digest = sha256()
    for record in records:
        digest.update(canonical_record_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_records_jsonl(
    records: Iterable[SampleRecord], output_path: str | Path, *, overwrite: bool = False
) -> Path:
    """执行 `write_records_jsonl` 所需的处理。"""

    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {target}; pass overwrite=True to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in rows:
            handle.write(canonical_record_json(record))
            handle.write("\n")
    temporary.replace(target)
    return target


def read_records_jsonl(input_path: str | Path) -> list[SampleRecord]:
    """执行 `read_records_jsonl` 所需的处理。"""

    source = Path(input_path)
    records: list[SampleRecord] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(SampleRecord.from_dict(json.loads(stripped)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid record at {source}:{line_number}: {error}") from error
    return records
