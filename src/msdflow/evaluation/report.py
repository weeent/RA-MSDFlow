"""生成逐 seed、跨 seed、机制指标和配对误报比较表。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from envfm.evaluation.aggregate import aggregate_across_seeds, write_csv
from envfm.training.checkpoint import atomic_write_json, file_digest

from .paired import compare_prediction_directories
from .runner import evaluate_msd_prediction_directory


def build_evaluation_report(
    run_specs: Sequence[Mapping[str, object]],
    output_directory: str | Path,
    *,
    comparisons: Sequence[Mapping[str, object]] = (),
    aupro_max_fpr: float = 0.05,
) -> Path:
    """读取不可变预测目录并写出论文长表；绝不重新拟合阈值。"""

    if not run_specs:
        raise ValueError("at least one prediction run is required")
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for spec in run_specs:
        result = evaluate_msd_prediction_directory(
            spec["prediction_directory"],
            spec["manifest"],
            method=str(spec["method"]),
            seed=int(spec["seed"]),
            aupro_max_fpr=aupro_max_fpr,
        )
        per_seed.extend(result.metric_rows)
        audits.append({"run": dict(spec), "audit": result.audit})
    aggregate = aggregate_across_seeds(per_seed)
    per_seed_path = write_csv(per_seed, output / "per_seed_metrics.csv")
    aggregate_path = write_csv(aggregate, output / "aggregate_metrics.csv")
    main = [row for row in aggregate if bool(row.get("is_primary")) and row["metric"] in {
        "image_auroc", "normal_fpr_at_clean_tau", "anomaly_tpr_at_clean_tau", "pixel_auroc", "aupro_0.05"
    }]
    mechanism = [row for row in aggregate if row["metric"] in {
        "environment_ood_rate_normal", "support_score_mean_normal", "transport_score_mean_normal",
        "environment_score_mean_normal", "mode_entropy_mean_normal", "defect_environment_pearson"
    }]
    main_path = write_csv(main, output / "main_results.csv") if main else None
    mechanism_path = write_csv(mechanism, output / "mechanism_results.csv") if mechanism else None

    comparison_rows: list[dict[str, object]] = []
    for spec in comparisons:
        rows = compare_prediction_directories(
            spec["reference_directory"],
            spec["candidate_directory"],
            bootstrap_samples=int(spec.get("bootstrap_samples", 2000)),
            seed=int(spec.get("seed", 9826)),
        )
        comparison_rows.extend(
            {
                "reference_method": str(spec["reference_method"]),
                "candidate_method": str(spec["candidate_method"]),
                **row.to_dict(),
            }
            for row in rows
        )
    comparison_path = write_csv(comparison_rows, output / "paired_fpr_comparisons.csv") if comparison_rows else None

    artifacts = {}
    for name, path in (
        ("per_seed_metrics", per_seed_path),
        ("aggregate_metrics", aggregate_path),
        ("main_results", main_path),
        ("mechanism_results", mechanism_path),
        ("paired_fpr_comparisons", comparison_path),
    ):
        artifacts[name] = None if path is None else {"path": str(path), "sha256": file_digest(path)}
    return atomic_write_json(
        {
            "phase": "msdflow_inference_evaluation_cli",
            "runs": len(run_specs),
            "metric_rows": len(per_seed),
            "aggregate_rows": len(aggregate),
            "comparison_rows": len(comparison_rows),
            "audits": audits,
            "artifacts": artifacts,
        },
        output / "evaluation_summary.json",
    )
