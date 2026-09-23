"""RA-MSDFlow 的 runner 模块。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from envfm.data.records import SampleRecord, read_records_jsonl

from .io import iter_prediction_maps, read_prediction_rows
from .metrics import aupro, binary_auroc, binary_rates_at_threshold, pixel_auroc
from .protocols import assign_domain


_RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(slots=True)
class EvaluationRunResult:
    metric_rows: list[dict[str, object]]
    audit: dict[str, object]


def _manifest_index(records: list[SampleRecord]) -> dict[tuple[str, str], SampleRecord]:
    index: dict[tuple[str, str], SampleRecord] = {}
    for record in records:
        key = (record.base_id, record.variant_id)
        if key in index:
            raise ValueError(f"manifest contains duplicate base_id/variant_id: {key}")
        index[key] = record
    return index


def _load_binary_mask(record: SampleRecord, output_shape: tuple[int, int]) -> np.ndarray | None:
    if record.label == 0:
        return np.zeros(output_shape, dtype=np.uint8)
    if record.label != 1 or record.mask_path is None:
        return None
    with Image.open(record.mask_path) as image:
        mask = image.convert("L").resize((output_shape[1], output_shape[0]), resample=_RESAMPLING.NEAREST)
        return (np.asarray(mask) > 0).astype(np.uint8)


def evaluate_prediction_directory(
    output_directory: str | Path,
    manifest_path: str | Path,
    *,
    method: str,
    seed: int,
    aupro_max_fpr: float = 0.05,
) -> EvaluationRunResult:
    """执行 `evaluate_prediction_directory` 所需的处理。"""

    predictions = read_prediction_rows(output_directory)
    manifest = _manifest_index(read_records_jsonl(manifest_path))
    maps_add = list(iter_prediction_maps(output_directory, predictions, preferred_key="anomaly_map_add"))
    maps_mul = list(iter_prediction_maps(output_directory, predictions, preferred_key="anomaly_map_mul"))
    groups: dict[tuple[str, str, str, bool], list[int]] = defaultdict(list)
    records: list[SampleRecord] = []

    # 步骤 1：按当前协议处理。
    for index, row in enumerate(predictions):
        key = (str(row["base_id"]), str(row.get("variant_id", "clean")))
        record = manifest.get(key)
        if record is None:
            raise KeyError(f"prediction sample is absent from prepared manifest: {key}")
        if int(row["label"]) != record.label:
            raise ValueError(f"prediction/manifest label mismatch for {key}")
        assignment = assign_domain(record.dataset, record.category, record.domain, record.variant_id)
        groups[(record.category, assignment.environment, assignment.family, assignment.is_primary)].append(index)
        records.append(record)

    metric_rows: list[dict[str, object]] = []
    omitted: Counter[str] = Counter()
    for (category, environment, family, is_primary), indices in sorted(groups.items()):
        labels = np.asarray([int(predictions[index]["label"]) for index in indices], dtype=np.int64)
        scores = np.asarray([float(predictions[index]["image_score"]) for index in indices], dtype=np.float64)
        known = np.isin(labels, [0, 1])
        labels = labels[known]
        scores = scores[known]
        dataset = records[indices[0]].dataset

        def add(metric: str, value: float, samples: int) -> None:
            metric_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "category": category,
                    "environment": environment,
                    "environment_family": family,
                    "is_primary": is_primary,
                    "seed": int(seed),
                    "metric": metric,
                    "value": float(value),
                    "samples": int(samples),
                }
            )

        # 步骤 2：读取当前输入。
        if (labels == 0).any() and (labels == 1).any():
            add("image_auroc", binary_auroc(labels, scores), labels.size)
        else:
            omitted["image_auroc_missing_class"] += 1
        thresholds = {
            float(predictions[index]["threshold"])
            for index in indices
            if predictions[index].get("threshold") is not None and int(predictions[index]["label"]) in {0, 1}
        }
        if len(thresholds) > 1:
            raise ValueError(f"group {category}/{environment} contains multiple applied thresholds")
        if thresholds and labels.size:
            rates = binary_rates_at_threshold(labels, scores, next(iter(thresholds)))
            if (labels == 0).any():
                add("normal_fpr_at_clean_tau", rates.fpr, int((labels == 0).sum()))
            if (labels == 1).any():
                add("anomaly_tpr_at_clean_tau", rates.tpr, int((labels == 1).sum()))
        else:
            omitted["threshold_metrics_missing_threshold"] += 1

        # 步骤 3：处理像素掩码。
        valid_maps: list[np.ndarray] = []
        valid_maps_mul: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for prediction_index in indices:
            add_map = maps_add[prediction_index]
            mul_map = maps_mul[prediction_index]
            if add_map is None or mul_map is None or records[prediction_index].label not in {0, 1}:
                continue
            mask = _load_binary_mask(records[prediction_index], tuple(add_map.shape))
            if mask is None:
                continue
            if mul_map.shape != add_map.shape:
                raise ValueError("additive and multiplicative map shapes differ")
            valid_maps.append(add_map)
            valid_maps_mul.append(mul_map)
            masks.append(mask)
        if masks and any(mask.any() for mask in masks):
            mask_array = np.stack(masks)
            add_array = np.stack(valid_maps)
            mul_array = np.stack(valid_maps_mul)
            add("pixel_auroc", pixel_auroc(mask_array, add_array), len(masks))
            pro = aupro(mask_array, mul_array, max_fpr=aupro_max_fpr)
            add(f"aupro_{aupro_max_fpr:g}", pro.score, pro.regions)
        else:
            omitted["pixel_metrics_missing_masks"] += 1

    audit = {
        "prediction_rows": len(predictions),
        "manifest_matches": len(records),
        "groups": len(groups),
        "metric_rows": len(metric_rows),
        "omitted": dict(sorted(omitted.items())),
        "labels": dict(sorted(Counter(str(row["label"]) for row in predictions).items())),
        "all_metric_values_finite": all(np.isfinite(float(row["value"])) for row in metric_rows),
    }
    return EvaluationRunResult(metric_rows, audit)
