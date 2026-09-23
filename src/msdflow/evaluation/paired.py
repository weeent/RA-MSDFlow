"""在相同测试样本上比较 MSD-Flow 与主 baseline 的正常误报。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from envfm.evaluation.io import read_prediction_rows


@dataclass(frozen=True, slots=True)
class PairedFPRComparison:
    category: str
    domain: str
    samples: int
    reference_fpr: float
    candidate_fpr: float
    candidate_minus_reference: float
    ci_low: float
    ci_high: float
    corrected_by_candidate: int
    worsened_by_candidate: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["base_id"]), str(row.get("variant_id", "clean"))


def _index_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    index: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = _key(row)
        if key in index:
            raise ValueError(f"prediction directory contains duplicate sample {key}")
        index[key] = row
    return index


def compare_prediction_directories(
    reference_directory: str | Path,
    candidate_directory: str | Path,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 9826,
) -> list[PairedFPRComparison]:
    """按 category/domain 对相同正常样本的二元误报差做 paired bootstrap。"""

    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    reference_rows = _index_rows(read_prediction_rows(reference_directory))
    candidate_rows = _index_rows(read_prediction_rows(candidate_directory))
    if set(reference_rows) != set(candidate_rows):
        missing_reference = len(set(candidate_rows).difference(reference_rows))
        missing_candidate = len(set(reference_rows).difference(candidate_rows))
        raise ValueError(
            f"paired comparison requires identical samples; missing reference={missing_reference}, candidate={missing_candidate}"
        )
    groups: dict[tuple[str, str], list[tuple[bool, bool]]] = {}
    for key in sorted(reference_rows):
        reference = reference_rows[key]
        candidate = candidate_rows[key]
        if int(reference["label"]) != int(candidate["label"]):
            raise ValueError(f"label mismatch for {key}")
        for field in ("dataset", "category", "domain"):
            if str(reference[field]) != str(candidate[field]):
                raise ValueError(f"{field} mismatch for {key}")
        if int(reference["label"]) != 0:
            continue
        if reference.get("threshold") is None or candidate.get("threshold") is None:
            raise ValueError("paired FPR comparison requires clean-validation thresholds")
        group = (str(reference["category"]), str(reference["domain"]))
        reference_error = float(reference["image_score"]) > float(reference["threshold"])
        candidate_error = float(candidate["image_score"]) > float(candidate["threshold"])
        groups.setdefault(group, []).append((reference_error, candidate_error))

    rng = np.random.default_rng(seed)
    output: list[PairedFPRComparison] = []
    for (category, domain), values in sorted(groups.items()):
        errors = np.asarray(values, dtype=np.float64)
        count = errors.shape[0]
        difference = errors[:, 1] - errors[:, 0]
        # 按样本对重采样，保留两个方法在同一图像上的相关性。
        indices = rng.integers(0, count, size=(bootstrap_samples, count))
        bootstrap = difference[indices].mean(1)
        output.append(
            PairedFPRComparison(
                category=category,
                domain=domain,
                samples=count,
                reference_fpr=float(errors[:, 0].mean()),
                candidate_fpr=float(errors[:, 1].mean()),
                candidate_minus_reference=float(difference.mean()),
                ci_low=float(np.quantile(bootstrap, 0.025)),
                ci_high=float(np.quantile(bootstrap, 0.975)),
                corrected_by_candidate=int(np.logical_and(errors[:, 0] == 1, errors[:, 1] == 0).sum()),
                worsened_by_candidate=int(np.logical_and(errors[:, 0] == 0, errors[:, 1] == 1).sum()),
            )
        )
    return output
