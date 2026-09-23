"""RA-MSDFlow 的 reference_pilot 模块。"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from envfm.data.dataset import IndustrialAnomalyDataset, normalize_imagenet
from envfm.data.records import SampleRecord
from envfm.evaluation.aggregate import write_csv
from envfm.evaluation.metrics import binary_auroc
from envfm.evaluation.protocols import assign_domain
from envfm.models.backbone import FrozenWideResNet50
from envfm.models.feature_preprocess import WTFeaturePreprocessor
from envfm.training.checkpoint import atomic_write_json, file_digest
from msdflow.inference import RoutingDiagnosticPipeline, load_inference_bundle
from msdflow.reference_alignment import (
    FeatureMomentAlignment,
    RgbMomentAlignment,
    ShrinkageCoral,
)

from .common import load_json_config, make_loader, required, select_records

ALIGNMENTS = ("unaligned", "rgb_moment", "feature_moment", "shrinkage_coral")


def _branch_for_channels(channels: int) -> int:
    mapping = {256: 0, 512: 1, 1024: 2}
    if channels not in mapping:
        raise ValueError("feature channels must match WRN layer1/2/3")
    return mapping[channels]


def _write_jsonl(rows: Sequence[Mapping[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear"))


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def _score_digest(values: Sequence[float]) -> str:
    digest = sha256()
    for value in values:
        digest.update(f"{float(value):.10g}".encode("ascii"))
    return digest.hexdigest()


# D1.1 诊断阶段。


def build_reference_split(
    records: Sequence[SampleRecord],
    *,
    target_domains: Mapping[str, str],
    reference_shots: int,
    reference_seed: int,
) -> tuple[list[SampleRecord], list[dict[str, object]], dict[str, object]]:
    """执行 `build_reference_split` 所需的处理。"""

    if reference_shots <= 0:
        raise ValueError("reference_shots must be positive")
    manifest = [record for record in records]
    evaluation: list[SampleRecord] = []
    audit: list[dict[str, object]] = []
    per_environment: dict[str, object] = {}
    used_reference_ids: set[str] = set()

    # 步骤 1：按当前协议处理。
    for environment, domain in target_domains.items():
        candidates = [
            record
            for record in manifest
            if record.domain == domain and record.split == "test" and record.label == 0
        ]
        base_ids = sorted({record.base_id for record in candidates})
        if len(base_ids) < reference_shots:
            raise ValueError(
                f"environment {environment!r} has only {len(base_ids)} normal base ids "
                f"for {reference_shots} reference shots"
            )
        # 步骤 2：按当前协议处理。
        generator = torch.Generator().manual_seed(reference_seed)
        permutation = torch.randperm(len(base_ids), generator=generator).tolist()
        reference_ids = {base_ids[index] for index in permutation[:reference_shots]}
        used_reference_ids.update(reference_ids)
        for record in sorted(
            (r for r in candidates if r.base_id in reference_ids), key=lambda r: (r.base_id, r.variant_id)
        ):
            audit.append(
                {
                    "environment": environment,
                    "domain": domain,
                    "base_id": record.base_id,
                    "image_path": record.image_path,
                    "label": record.label,
                    "variant_id": record.variant_id,
                    "role": "reference",
                }
            )
        evaluation_normal = [r for r in candidates if r.base_id not in reference_ids]
        anomalies = [
            record
            for record in manifest
            if record.domain == domain and record.split == "test" and record.label != 0
        ]
        per_environment[environment] = {
            "domain": domain,
            "reference_base_ids": sorted(reference_ids),
            "reference_count": len(reference_ids),
            "evaluation_normal": len(evaluation_normal),
            "evaluation_anomaly": len(anomalies),
        }

    for record in manifest:
        if record.split != "test":
            continue
        if record.label == 0 and record.base_id in used_reference_ids:
            continue
        evaluation.append(record)
    return evaluation, audit, per_environment


# D1.2 诊断阶段。


@torch.inference_mode()
def _encode_split(
    records: Sequence[SampleRecord],
    *,
    backbone: torch.nn.Module,
    preprocessor: torch.nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor, list[dict[str, object]]]:
    """执行 `_encode_split` 所需的处理。"""

    if not records:
        return torch.empty(0), torch.empty(0), []
    dataset = IndustrialAnomalyDataset(records, image_size=int(config.get("image_size", 256)))
    loader = make_loader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        seed=int(config.get("reference_seed", 9826)),
        pin_memory=device.type == "cuda",
    )
    features: list[Tensor] = []
    raws: list[Tensor] = []
    rows: list[dict[str, object]] = []
    for batch in loader:
        image = torch.as_tensor(batch["image"]).to(device, non_blocking=device.type == "cuda")
        raw = torch.as_tensor(batch["image_raw"]).to(device, non_blocking=device.type == "cuda")
        feature = preprocessor(backbone(image))
        if not isinstance(feature, Tensor):
            raise TypeError("feature preprocessor must return one tensor")
        features.append(feature.detach().float().cpu())
        raws.append(raw.detach().float().cpu())
        for index in range(image.shape[0]):
            rows.append(
                {
                    "base_id": str(batch["base_id"][index]),
                    "variant_id": str(batch["variant_id"][index]),
                    "domain": str(batch["domain"][index]),
                    "label": int(torch.as_tensor(batch["label"]).flatten()[index].item()),
                    "image_path": str(batch["path"][index]),
                }
            )
    return torch.cat(features), torch.cat(raws), rows


# D1.4 诊断阶段。


@torch.inference_mode()
def _mode0_scores(
    pipeline: RoutingDiagnosticPipeline,
    features: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    """执行 `_mode0_scores` 所需的处理。"""

    scores: list[float] = []
    for start in range(0, features.shape[0], batch_size):
        chunk = features[start : start + batch_size].to(device)
        # 步骤 1：按当前协议处理。
        # `_mode0_scores` 的实现说明。
        expanded = chunk.unsqueeze(1).expand(-1, pipeline.model.mode_bank.n_components, -1, -1, -1)
        maps = pipeline._normality_maps(expanded)  # noqa: SLF001 - frozen v3 scoring path
        scores.extend(float(v) for v in pipeline._image_scores(maps)[:, 0].cpu())  # noqa: SLF001
    return scores


@torch.inference_mode()
def _rgb_aligned_features(
    raw: Tensor,
    alignment: RgbMomentAlignment,
    *,
    backbone: torch.nn.Module,
    preprocessor: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    outputs: list[Tensor] = []
    for start in range(0, raw.shape[0], batch_size):
        chunk = raw[start : start + batch_size].to(device)
        aligned = alignment.apply(chunk)
        feature = preprocessor(backbone(normalize_imagenet(aligned)))
        if not isinstance(feature, Tensor):
            raise TypeError("feature preprocessor must return one tensor")
        outputs.append(feature.detach().float().cpu())
    return torch.cat(outputs)


# D1.6 诊断阶段。


def _metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, float]:
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    normal = label_array == 0
    anomalous = ~normal
    metrics = {
        "normal_fpr": float((score_array[normal] > threshold).mean()) if normal.any() else float("nan"),
        "anomaly_tpr": float((score_array[anomalous] > threshold).mean()) if anomalous.any() else float("nan"),
        "image_auroc": float(binary_auroc(label_array.tolist(), score_array.tolist()))
        if normal.any() and anomalous.any()
        else float("nan"),
        "normal_samples": int(normal.sum()),
        "anomaly_samples": int(anomalous.sum()),
        "normal_score_median": _median(score_array[normal].tolist()),
        "anomaly_score_median": _median(score_array[anomalous].tolist()),
    }
    return metrics


def _covariance_gap(source: Tensor, target: Tensor) -> float:
    def covariance(values: Tensor) -> Tensor:
        flat = values.double().permute(1, 0, 2, 3).reshape(values.shape[1], -1)
        centered = flat - flat.mean(1, keepdim=True)
        return centered @ centered.transpose(0, 1) / max(flat.shape[1] - 1, 1)

    return float((covariance(source) - covariance(target)).norm())


def run_reference_pilot(config: Mapping[str, Any]) -> Path:
    """执行 `run_reference_pilot` 所需的处理。"""

    device = torch.device(str(config.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset_name = str(required(config, "dataset"))
    category = str(required(config, "category"))
    manifest_path = Path(required(config, "manifest")).resolve()
    target_domains = {str(k): str(v) for k, v in required(config, "target_domains").items()}
    reference_shots = int(config.get("reference_shots", 20))
    reference_seed = int(config.get("reference_seed", 9826))
    output_directory = Path(required(config, "output_directory")).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    # 步骤 1：加载当前输入。
    bundle_path = Path(required(config, "inference_bundle")).resolve()
    model, frozen_info = load_inference_bundle(bundle_path, device=device)
    channels = int(model.environment_flow.velocity.in_channels)  # type: ignore[attr-defined]
    backbone = FrozenWideResNet50(
        weights_path=config.get("backbone_weights"),
        reference_resnet_path=config.get("reference_resnet_source"),
    ).to(device).eval()
    preprocessor = WTFeaturePreprocessor(branch_index=_branch_for_channels(channels)).to(device).eval()
    pipeline = RoutingDiagnosticPipeline(
        model,
        clean_anchor_mode=int(config.get("clean_anchor_mode", 0)),
        environment_steps=int(config.get("environment_steps", 4)),
        normality_steps=int(config.get("normality_steps", 20)),
        softmin_temperature=float(config.get("softmin_temperature", 0.1)),
        output_size=int(config.get("image_size", 256)),
        top_fraction=float(config.get("top_fraction", 0.03)),
    )
    for module in (model, backbone, preprocessor):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    # 步骤 2：构建数据划分。
    test_records = select_records(
        manifest_path, category=category, split=str(config.get("test_split", "test")), dataset=dataset_name
    )
    evaluation_records, reference_audit, per_environment = build_reference_split(
        test_records,
        target_domains=target_domains,
        reference_shots=reference_shots,
        reference_seed=reference_seed,
    )
    reference_split_path = output_directory / "reference_split.json"
    atomic_write_json(
        {
            "dataset": dataset_name,
            "category": category,
            "manifest": str(manifest_path),
            "manifest_sha256": file_digest(manifest_path),
            "reference_shots": reference_shots,
            "reference_seed": reference_seed,
            "target_domains": target_domains,
            "environments": per_environment,
            "reference_records": reference_audit,
        },
        reference_split_path,
    )

    # 步骤 3：构建数据划分。
    train_records = [
        record
        for record in select_records(
            manifest_path,
            category=category,
            split=str(config.get("train_split", "train")),
            dataset=dataset_name,
        )
        if record.label == 0 and record.variant_id == "clean"
    ]
    validation_records = select_records(
        manifest_path,
        category=category,
        split=str(config.get("validation_split", "val")),
        dataset=dataset_name,
        normal_only=True,
    )
    reference_by_environment: dict[str, list[SampleRecord]] = {}
    for environment, domain in target_domains.items():
        wanted = set(per_environment[environment]["reference_base_ids"])
        reference_by_environment[environment] = [
            record
            for record in test_records
            if record.domain == domain and record.label == 0 and record.base_id in wanted
        ]
    source_features, source_raw, _ = _encode_split(
        train_records, backbone=backbone, preprocessor=preprocessor, config=config, device=device
    )
    validation_features, _, validation_rows = _encode_split(
        validation_records, backbone=backbone, preprocessor=preprocessor, config=config, device=device
    )
    reference_features: dict[str, Tensor] = {}
    reference_raw: dict[str, Tensor] = {}
    for environment in target_domains:
        features, raw, _ = _encode_split(
            reference_by_environment[environment],
            backbone=backbone,
            preprocessor=preprocessor,
            config=config,
            device=device,
        )
        reference_features[environment] = features
        reference_raw[environment] = raw

    # 步骤 4：拟合当前统计量。
    alignments: dict[str, dict[str, object]] = {}
    for environment in target_domains:
        rgb = RgbMomentAlignment().fit(source_raw, reference_raw[environment])
        moment = FeatureMomentAlignment().fit(source_features, reference_features[environment])
        coral = ShrinkageCoral(
            shrinkage=float(config.get("coral_shrinkage", 0.1)),
            max_patches=int(config.get("coral_max_patches", 20000)),
            patch_seed=reference_seed,
        ).fit(source_features, reference_features[environment])
        alignments[environment] = {"rgb_moment": rgb, "feature_moment": moment, "shrinkage_coral": coral}
    torch.save(
        {
            environment: {
                "rgb_moment": {
                    "mean_source": operators["rgb_moment"].mean_source,
                    "std_source": operators["rgb_moment"].std_source,
                    "mean_target": operators["rgb_moment"].mean_target,
                    "std_target": operators["rgb_moment"].std_target,
                },
                "feature_moment": {
                    "mean_source": operators["feature_moment"].mean_source,
                    "std_source": operators["feature_moment"].std_source,
                    "mean_target": operators["feature_moment"].mean_target,
                    "std_target": operators["feature_moment"].std_target,
                },
                "shrinkage_coral": {
                    "mean_source": operators["shrinkage_coral"].mean_source,
                    "mean_target": operators["shrinkage_coral"].mean_target,
                    "transform": operators["shrinkage_coral"].transform,
                    "shrinkage": operators["shrinkage_coral"].shrinkage,
                    "eigenvalues": operators["shrinkage_coral"].covariance_eigenvalues,
                },
            }
            for environment, operators in alignments.items()
        },
        output_directory / "alignment_stats.pt",
    )
    atomic_write_json(
        {
            environment: {
                name: operator.summary() for name, operator in operators.items()
            }
            for environment, operators in alignments.items()
        },
        output_directory / "alignment_stats.json",
    )

    # 步骤 5：按当前协议处理。
    batch_size = int(config.get("batch_size", 8))
    validation_scores = _mode0_scores(
        pipeline, validation_features, device=device, batch_size=batch_size
    )
    if any(int(row["label"]) != 0 for row in validation_rows):
        raise ValueError("clean validation split contains anomalous records")
    source_threshold = _quantile(validation_scores, float(config.get("calibration_quantile", 0.95)))

    # 步骤 6：计算异常分数。
    prediction_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    for environment, domain in target_domains.items():
        environment_records = [
            record for record in evaluation_records if record.domain == domain
        ]
        features, raw, rows = _encode_split(
            environment_records,
            backbone=backbone,
            preprocessor=preprocessor,
            config=config,
            device=device,
        )
        operators = alignments[environment]
        # 步骤 6：按当前协议处理。
        # `run_reference_pilot` 的实现说明。
        feature_spaces: dict[str, Tensor] = {
            "unaligned": features,
            "rgb_moment": _rgb_aligned_features(
                raw,
                operators["rgb_moment"],  # type: ignore[arg-type]
                backbone=backbone,
                preprocessor=preprocessor,
                device=device,
                batch_size=batch_size,
            ),
            "feature_moment": operators["feature_moment"].apply(features),  # type: ignore[union-attr]
            "shrinkage_coral": operators["shrinkage_coral"].apply(features),  # type: ignore[union-attr]
        }
        scores: dict[str, list[float]] = {
            name: _mode0_scores(pipeline, values, device=device, batch_size=batch_size)
            for name, values in feature_spaces.items()
        }
        reference_scores = _mode0_scores(
            pipeline, reference_features[environment], device=device, batch_size=batch_size
        )
        target_threshold = _quantile(reference_scores, float(config.get("calibration_quantile", 0.95)))

        for index, row in enumerate(rows):
            prediction_rows.append(
                {
                    "environment": environment,
                    "domain": domain,
                    "base_id": row["base_id"],
                    "image_path": row["image_path"],
                    "variant_id": row["variant_id"],
                    "label": row["label"],
                    "is_reference": False,
                    "source_threshold": source_threshold,
                    "target_threshold": target_threshold,
                    **{f"{name}_score": scores[name][index] for name in ALIGNMENTS},
                    "target_threshold_only_score": scores["unaligned"][index],
                }
            )
        # 步骤 7：按当前协议处理。
        # `run_reference_pilot` 的实现说明。
        source_mean = source_features.mean((0, 2, 3))
        source_std = source_features.std((0, 2, 3))
        normal_mask = np.asarray([row["label"] == 0 for row in rows])
        for name, values in feature_spaces.items():
            scores_array = np.asarray(scores[name], dtype=np.float64)
            gap_rows.append(
                {
                    "environment": environment,
                    "domain": domain,
                    "alignment": name,
                    "channel_mean_gap_l2": float((source_mean - values.mean((0, 2, 3))).norm()),
                    "channel_std_gap_l2": float((source_std - values.std((0, 2, 3))).norm()),
                    "covariance_frobenius_gap": _covariance_gap(source_features, values),
                    "reference_normal_score_median": _median(reference_scores),
                    "evaluation_normal_score_median": _median(scores_array[normal_mask].tolist()),
                    "evaluation_anomaly_score_median": _median(scores_array[~normal_mask].tolist()),
                }
            )

    # 步骤 8：按当前协议处理。
    v3_path = config.get("v3_predictions")
    control_rows: list[dict[str, object]] = []
    if v3_path:
        v3_rows = [
            json.loads(line)
            for line in Path(str(v3_path)).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        reference_ids = {entry["base_id"] for entry in reference_audit}
        for row in v3_rows:
            if str(row["domain"]) not in set(target_domains.values()):
                continue
            if int(row["label"]) == 0 and str(row["base_id"]) in reference_ids:
                continue
            control_rows.append(
                {
                    "environment": next(
                        name for name, domain in target_domains.items() if domain == str(row["domain"])
                    ),
                    "base_id": row["base_id"],
                    "label": int(row["label"]),
                    "m0a_score": float(row["posterior_top2_score"]),
                }
            )

    # 步骤 9：按当前协议处理。
    metric_rows: list[dict[str, object]] = []
    macro_rows: list[dict[str, object]] = []
    method_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for environment in target_domains:
        entries = [row for row in prediction_rows if row["environment"] == environment]
        labels = [int(row["label"]) for row in entries]
        for name in ALIGNMENTS:
            values = [float(row[f"{name}_score"]) for row in entries]
            metrics = _metrics(labels, values, source_threshold)
            baseline_median = _median([float(r["unaligned_score"]) for r in entries if int(r["label"]) == 0])
            normal_median = _median(
                [values[index] for index, label in enumerate(labels) if label == 0]
            )
            metrics.update(
                {
                    "method": name,
                    "environment": environment,
                    "threshold": source_threshold,
                    "normal_ratio_drop": 1.0 - normal_median / baseline_median
                    if baseline_median
                    else float("nan"),
                }
            )
            metric_rows.append(metrics)
            method_scores[name][environment] = metrics
        # 按参考样本协议处理。
        # `run_reference_pilot` 的实现说明。
        m1_threshold = _median([float(row["target_threshold"]) for row in entries])
        m1 = _metrics(
            labels, [float(row["target_threshold_only_score"]) for row in entries], m1_threshold
        )
        metric_rows.append(
            {
                "method": "target_threshold_only",
                "environment": environment,
                "threshold": m1_threshold,
                **m1,
            }
        )
        m1["normal_ratio_drop"] = float("nan")
        method_scores["target_threshold_only"][environment] = m1
    if control_rows:
        for environment in target_domains:
            entries = [row for row in control_rows if row["environment"] == environment]
            metrics = _metrics(
                [int(row["label"]) for row in entries],
                [float(row["m0a_score"]) for row in entries],
                source_threshold,
            )
            metric_rows.append({"method": "v3_posterior_top2", "environment": environment,
                                "threshold": source_threshold, **metrics})
            method_scores["v3_posterior_top2"][environment] = metrics

    for method in ("v3_posterior_top2", "unaligned", "target_threshold_only") + ALIGNMENTS[1:]:
        available = method_scores.get(method)
        if not available:
            continue
        macro_rows.append(
            {
                "method": method,
                "macro_normal_fpr": float(np.nanmean([v["normal_fpr"] for v in available.values()])),
                "macro_anomaly_tpr": float(np.nanmean([v["anomaly_tpr"] for v in available.values()])),
                "macro_image_auroc": float(np.nanmean([v["image_auroc"] for v in available.values()])),
                "environments": ",".join(sorted(available)),
            }
        )

    write_csv(metric_rows, output_directory / "d1_metrics.csv")
    write_csv(gap_rows, output_directory / "distribution_gap.csv")
    write_csv(macro_rows, output_directory / "d1_macro_metrics.csv")
    _write_jsonl(prediction_rows, output_directory / "predictions.jsonl")
    atomic_write_json(
        {
            "dataset": dataset_name,
            "category": category,
            "source_threshold": source_threshold,
            "source_validation_samples": len(validation_scores),
            "source_validation_scores_sha256": _score_digest(validation_scores),
            "inference_bundle_sha256": file_digest(bundle_path),
            "per_environment_target_threshold": {
                environment: _median(
                    [
                        float(row["target_threshold"])
                        for row in prediction_rows
                        if row["environment"] == environment
                    ]
                )
                for environment in target_domains
            },
            "decision_rule": "anomalous_if_score_strictly_greater_than_threshold",
        },
        output_directory / "calibration.json",
    )
    summary = {
        "phase": "reference_guided_d1_feasibility",
        "dataset": dataset_name,
        "category": category,
        "feature_channels": channels,
        "branch_index": _branch_for_channels(channels),
        "frozen_bundle": frozen_info.to_dict() if hasattr(frozen_info, "to_dict") else str(frozen_info),
        "environments": per_environment,
        "reference_records": len(reference_audit),
        "evaluation_rows": len(prediction_rows),
        "source_train_rows": len(train_records),
        "validation_rows": len(validation_rows),
        "alignments": {
            environment: {name: operator.summary() for name, operator in operators.items()}
            for environment, operators in alignments.items()
        },
        "artifacts": {
            name: {"path": str(output_directory / name), "sha256": file_digest(output_directory / name)}
            for name in (
                "reference_split.json",
                "alignment_stats.json",
                "calibration.json",
                "predictions.jsonl",
                "d1_metrics.csv",
                "distribution_gap.csv",
                "d1_macro_metrics.csv",
            )
            if (output_directory / name).is_file()
        },
        "finite": bool(
            all(
                math.isfinite(float(row[f"{name}_score"]))
                for row in prediction_rows
                for name in ALIGNMENTS
            )
        ),
        "frozen_check": {
            "all_modules_in_eval": bool(
                not model.training and not backbone.training and not preprocessor.training
            ),
            "all_parameters_requires_grad_false": bool(
                all(
                    not parameter.requires_grad
                    for module in (model, backbone, preprocessor)
                    for parameter in module.parameters()
                )
            ),
            "trainable_parameter_count": int(
                sum(
                    parameter.numel()
                    for module in (model, backbone, preprocessor)
                    for parameter in module.parameters()
                    if parameter.requires_grad
                )
            ),
        },
        "statistics_provenance": {
            "source_rows": len(train_records),
            "source_split": str(config.get("train_split", "train")),
            "source_variant": "clean",
            "validation_rows": len(validation_records),
            "reference_rows": {environment: len(records) for environment, records in reference_by_environment.items()},
            "evaluation_rows": {
                environment: sum(1 for row in prediction_rows if row["environment"] == environment)
                for environment in target_domains
            },
        },
    }
    atomic_write_json(summary, output_directory / "d1_summary.json")
    print(json.dumps({"output_directory": str(output_directory)}, indent=2))
    return output_directory


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MSD-Flow D1 reference-guided feasibility pilot")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_reference_pilot(load_json_config(args.config))
