"""在 EnvFM 固定异常指标上增加 MSD-Flow 环境机制指标。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from envfm.data.records import read_records_jsonl
from envfm.evaluation.io import read_prediction_rows
from envfm.evaluation.protocols import assign_domain
from envfm.evaluation.runner import evaluate_prediction_directory


@dataclass(slots=True)
class MSDEvaluationRunResult:
    metric_rows: list[dict[str, object]]
    audit: dict[str, object]


def evaluate_msd_prediction_directory(
    output_directory: str | Path,
    manifest_path: str | Path,
    *,
    method: str,
    seed: int,
    aupro_max_fpr: float = 0.05,
) -> MSDEvaluationRunResult:
    """计算标准 AD 指标，并按类别/环境报告双分数机制统计。"""

    base = evaluate_prediction_directory(
        output_directory,
        manifest_path,
        method=method,
        seed=seed,
        aupro_max_fpr=aupro_max_fpr,
    )
    rows = read_prediction_rows(output_directory)
    dual_fields = (
        "defect_score",
        "environment_support_score",
        "environment_transport_score",
        "in_environment_support",
        "top_mode_ids",
        "top_mode_weights",
    )
    dual_complete = all(all(field in row for field in dual_fields) for row in rows)
    # WT-Flow/PatchCore 等 baseline 只有统一异常字段；保留标准指标并明确跳过机制指标。
    if not dual_complete:
        return MSDEvaluationRunResult(
            list(base.metric_rows),
            {**base.audit, "mechanism_metric_rows": 0, "dual_score_fields_complete": False},
        )
    manifest = {(record.base_id, record.variant_id): record for record in read_records_jsonl(manifest_path)}
    groups: dict[tuple[str, str, str, bool], list[int]] = defaultdict(list)
    records = []
    for index, row in enumerate(rows):
        key = (str(row["base_id"]), str(row.get("variant_id", "clean")))
        if key not in manifest:
            raise KeyError(f"prediction sample is missing from manifest: {key}")
        record = manifest[key]
        assignment = assign_domain(record.dataset, record.category, record.domain, record.variant_id)
        groups[(record.category, assignment.environment, assignment.family, assignment.is_primary)].append(index)
        records.append(record)

    metrics = list(base.metric_rows)
    mechanism_rows = 0
    for (category, environment, family, is_primary), indices in sorted(groups.items()):
        normal_indices = [index for index in indices if int(rows[index]["label"]) == 0]
        if not normal_indices:
            continue
        dataset = records[indices[0]].dataset

        def add(name: str, value: float, samples: int) -> None:
            nonlocal mechanism_rows
            if np.isfinite(value):
                metrics.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "category": category,
                        "environment": environment,
                        "environment_family": family,
                        "is_primary": is_primary,
                        "seed": int(seed),
                        "metric": name,
                        "value": float(value),
                        "samples": int(samples),
                    }
                )
                mechanism_rows += 1

        add(
            "environment_ood_rate_normal",
            float(np.mean([not bool(rows[index]["in_environment_support"]) for index in normal_indices])),
            len(normal_indices),
        )
        for field, metric in (
            ("environment_support_score", "support_score_mean_normal"),
            ("environment_transport_score", "transport_score_mean_normal"),
            ("mode_entropy", "mode_entropy_mean_normal"),
            ("defect_score", "defect_score_mean_normal"),
        ):
            values = np.asarray([float(rows[index][field]) for index in normal_indices], dtype=np.float64)
            add(metric, float(values.mean()), values.size)
        calibrated = [rows[index].get("environment_score") for index in normal_indices]
        if all(value is not None for value in calibrated):
            values = np.asarray(calibrated, dtype=np.float64)
            add("environment_score_mean_normal", float(values.mean()), values.size)

        # 相关性只作诊断：若缺陷分数仍随环境分数同步上升，说明分解不充分。
        all_calibrated = [rows[index].get("environment_score") for index in indices]
        if len(indices) >= 3 and all(value is not None for value in all_calibrated):
            defect = np.asarray([float(rows[index]["image_score"]) for index in indices])
            environment_values = np.asarray(all_calibrated, dtype=np.float64)
            if defect.std() > 0 and environment_values.std() > 0:
                add("defect_environment_pearson", float(np.corrcoef(defect, environment_values)[0, 1]), len(indices))

    audit = {
        **base.audit,
        "mechanism_metric_rows": mechanism_rows,
        "dual_score_fields_complete": dual_complete,
    }
    return MSDEvaluationRunResult(metrics, audit)
