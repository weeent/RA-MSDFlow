"""正常验证集专用的缺陷阈值与环境分数经验校准。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from msdflow.models import EmpiricalScoreCalibrator


CALIBRATION_VERSION = 1


def _finite_vector(values: Sequence[float] | Tensor, name: str) -> Tensor:
    vector = torch.as_tensor(values, dtype=torch.float64).flatten().cpu()
    if vector.numel() == 0 or not torch.isfinite(vector).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return vector


def _digest(values: Tensor) -> str:
    encoded = json.dumps(values.tolist(), separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class MSDCategoryCalibration:
    category: str
    normal_count: int
    defect_quantile: float
    defect_threshold: float
    support_reference: tuple[float, ...]
    transport_reference: tuple[float, ...]
    defect_scores_sha256: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["support_reference"] = list(self.support_reference)
        value["transport_reference"] = list(self.transport_reference)
        return value


@dataclass(frozen=True, slots=True)
class MSDCalibrationTable:
    categories: Mapping[str, MSDCategoryCalibration]
    format_version: int = CALIBRATION_VERSION
    source_split: str = "normal_validation_only"
    decision_rule: str = "defect_score_strictly_greater_than_threshold"

    def record(self, category: str) -> MSDCategoryCalibration:
        if category not in self.categories:
            raise KeyError(f"calibration has no category {category!r}")
        return self.categories[category]

    def predict_defect(self, scores: Tensor, category: str) -> Tensor:
        return scores > self.record(category).defect_threshold

    def environment_calibrator(self, category: str, *, device: str | torch.device = "cpu") -> EmpiricalScoreCalibrator:
        record = self.record(category)
        calibrator = EmpiricalScoreCalibrator().to(device)
        calibrator.fit(
            torch.tensor(record.support_reference, device=device),
            torch.tensor(record.transport_reference, device=device),
        )
        return calibrator

    def environment_score(self, support: Tensor, transport: Tensor, category: str) -> Tensor:
        record = self.record(category)
        calibrator = self.environment_calibrator(category, device=support.device)
        return calibrator(support, transport)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "source_split": self.source_split,
            "decision_rule": self.decision_rule,
            "categories": {name: self.categories[name].to_dict() for name in sorted(self.categories)},
        }

    def save(self, output_path: str | Path, *, overwrite: bool = False) -> Path:
        target = Path(output_path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"calibration file already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, input_path: str | Path) -> "MSDCalibrationTable":
        with Path(input_path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if int(value.get("format_version", 0)) != CALIBRATION_VERSION:
            raise ValueError("unsupported MSD-Flow calibration version")
        if value.get("source_split") != "normal_validation_only":
            raise ValueError("calibration must originate from normal validation only")
        categories = {
            name: MSDCategoryCalibration(
                category=str(record["category"]),
                normal_count=int(record["normal_count"]),
                defect_quantile=float(record["defect_quantile"]),
                defect_threshold=float(record["defect_threshold"]),
                support_reference=tuple(float(item) for item in record["support_reference"]),
                transport_reference=tuple(float(item) for item in record["transport_reference"]),
                defect_scores_sha256=str(record["defect_scores_sha256"]),
            )
            for name, record in value["categories"].items()
        }
        return cls(categories=categories)


def fit_msd_calibration(
    defect_scores: Mapping[str, Sequence[float] | Tensor],
    support_scores: Mapping[str, Sequence[float] | Tensor],
    transport_scores: Mapping[str, Sequence[float] | Tensor],
    *,
    labels: Mapping[str, Sequence[int] | Tensor] | None = None,
    defect_quantile: float = 0.95,
) -> MSDCalibrationTable:
    """同时拟合缺陷阈值和两个环境经验 CDF，拒绝任何非正常验证标签。"""

    if not 0.0 < defect_quantile < 1.0:
        raise ValueError("defect_quantile must lie in (0,1)")
    if not defect_scores or set(defect_scores) != set(support_scores) or set(defect_scores) != set(transport_scores):
        raise ValueError("three score mappings must contain the same non-empty categories")
    records: dict[str, MSDCategoryCalibration] = {}
    for category in sorted(defect_scores):
        defect = _finite_vector(defect_scores[category], "defect scores")
        support = _finite_vector(support_scores[category], "support scores")
        transport = _finite_vector(transport_scores[category], "transport scores")
        if not (defect.shape == support.shape == transport.shape):
            raise ValueError(f"calibration score counts differ for {category!r}")
        if labels is not None:
            category_labels = torch.as_tensor(labels[category]).flatten().cpu()
            if category_labels.numel() != defect.numel() or torch.any(category_labels != 0):
                raise ValueError(f"calibration category {category!r} is not normal-only")
        threshold = float(torch.quantile(defect, defect_quantile, interpolation="linear").item())
        records[category] = MSDCategoryCalibration(
            category=category,
            normal_count=defect.numel(),
            defect_quantile=defect_quantile,
            defect_threshold=threshold,
            support_reference=tuple(float(item) for item in support.sort().values.tolist()),
            transport_reference=tuple(float(item) for item in transport.sort().values.tolist()),
            defect_scores_sha256=_digest(defect),
        )
    return MSDCalibrationTable(records)
