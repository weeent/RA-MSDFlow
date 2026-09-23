"""MSD-Flow Pilot v3 的冻结模式路由与环境表示诊断 CLI。

本入口一次完成：
1. 从 normal-train 配对缓存确定 clean anchor，并统计 mode×variant；
2. 在同一冻结 inference bundle 上计算五种路由策略；
3. 每种策略只用 clean normal validation 拟合自己的 95% 阈值；
4. 导出逐样本候选分数、Photo/learned 距离分解和逐域指标。

程序不创建优化器，也不调用 ``backward``，因此不会改变模型权重。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from envfm.data.dataset import IndustrialAnomalyDataset
from envfm.data.records import SampleRecord, records_digest
from envfm.evaluation.aggregate import write_csv
from envfm.evaluation.metrics import binary_auroc
from envfm.evaluation.protocols import assign_domain
from envfm.models.backbone import FrozenWideResNet50
from envfm.models.feature_preprocess import WTFeaturePreprocessor
from envfm.training.checkpoint import atomic_write_json, file_digest
from msdflow.data import CachedPairedFeatureDataset
from msdflow.inference import (
    ROUTING_POLICIES,
    RoutingDiagnosticPipeline,
    decompose_environment_distance,
    load_inference_bundle,
)

from .common import load_json_config, make_loader, required, select_records


def _branch_for_channels(channels: int) -> int:
    mapping = {256: 0, 512: 1, 1024: 2}
    if channels not in mapping:
        raise ValueError("feature_channels must match WRN layer1/2/3")
    return mapping[channels]


def _prediction_file(path: str | Path) -> Path:
    source = Path(path).resolve()
    if source.is_dir():
        source = source / "predictions.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"prediction JSONL does not exist: {source}")
    return source


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = _prediction_file(path)
    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"prediction row must be an object at {source}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"prediction JSONL is empty: {source}")
    return rows


def _write_jsonl(rows: Iterable[Mapping[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("cannot write an empty routing JSONL")
    temporary.replace(path)
    return path


def _score_digest(values: Sequence[float]) -> str:
    payload = json.dumps(list(values), separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(payload).hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    vector = torch.as_tensor(values, dtype=torch.float64)
    if vector.numel() == 0 or not torch.isfinite(vector).all():
        raise ValueError("quantile values must be non-empty and finite")
    return float(torch.quantile(vector, probability, interpolation="linear").item())


def _distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be non-empty and finite")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _normalized_mutual_information(counts: np.ndarray) -> float:
    """从 mode×variant 计数计算 sqrt-normalized mutual information。"""

    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if counts.ndim != 2 or total <= 0:
        raise ValueError("contingency matrix must be non-empty and two-dimensional")
    joint = counts / total
    row = joint.sum(1, keepdims=True)
    column = joint.sum(0, keepdims=True)
    expected = row @ column
    mask = joint > 0
    mutual_information = float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))
    row_values = row.reshape(-1)
    column_values = column.reshape(-1)
    row_entropy = -float(np.sum(row_values[row_values > 0] * np.log(row_values[row_values > 0])))
    column_entropy = -float(
        np.sum(column_values[column_values > 0] * np.log(column_values[column_values > 0]))
    )
    denominator = math.sqrt(row_entropy * column_entropy)
    return 0.0 if denominator == 0 else mutual_information / denominator


def _environment_distances(model: torch.nn.Module, standardized: Tensor) -> tuple[Tensor, Tensor]:
    # 独立小函数便于对 cache 和图像推理共用同一距离定义。
    return decompose_environment_distance(model, standardized)  # type: ignore[arg-type]


def _collect_train_environment_audit(
    cache_path: str | Path,
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> tuple[int, list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """读取 forward-train cache，返回 clean anchor、列联表和逐环境距离。"""

    cache = CachedPairedFeatureDataset(cache_path)
    unique: dict[tuple[str, str], Tensor] = {}
    for item in cache:
        labels = (int(torch.as_tensor(item["label_a"]).item()), int(torch.as_tensor(item["label_b"]).item()))
        if labels != (0, 0) or str(item["split"]) != "train":
            raise ValueError("forward_train_cache must contain normal train pairs only")
        for suffix in ("a", "b"):
            key = (str(item["base_id"]), str(item[f"variant_{suffix}"]))
            value = torch.as_tensor(item[f"environment_{suffix}"]).float().cpu()
            if key in unique and not torch.allclose(unique[key], value, atol=1e-6, rtol=0.0):
                raise ValueError(f"duplicate cached environment differs for {key}")
            unique[key] = value
    if not unique:
        raise ValueError("forward_train_cache contains no environments")

    ordered = sorted(unique)
    raw = torch.stack([unique[key] for key in ordered]).to(device)
    with torch.inference_mode():
        standardized = model.standardizer(raw)  # type: ignore[attr-defined]
        assignment = model.mode_bank.assign(  # type: ignore[attr-defined]
            standardized, top_m=model.mode_bank.n_components  # type: ignore[attr-defined]
        )
        top1 = assignment.posterior.argmax(1).cpu().tolist()
        photo_distance, learned_distance = _environment_distances(model, standardized)
    modes = int(model.mode_bank.n_components)  # type: ignore[attr-defined]

    by_variant: dict[str, Counter[int]] = defaultdict(Counter)
    base_modes: dict[str, dict[str, int]] = defaultdict(dict)
    distance_rows: list[dict[str, object]] = []
    for index, ((base_id, variant), mode_id) in enumerate(zip(ordered, top1)):
        by_variant[variant][int(mode_id)] += 1
        base_modes[base_id][variant] = int(mode_id)
        photo = float(photo_distance[index].item())
        learned = float(learned_distance[index].item())
        distance_rows.append(
            {
                "base_id": base_id,
                "split": "train",
                "group": "train_synthetic_normal",
                "environment": variant,
                "variant_id": variant,
                "label": 0,
                "top1_mode": int(mode_id),
                "photo_distance": photo,
                "learned_distance": learned,
                "learned_share": learned / max(photo + learned, 1e-12),
            }
        )

    if "clean" not in by_variant:
        raise ValueError("forward_train_cache has no clean environment entries")
    clean_counts = by_variant["clean"]
    clean_anchor = min(
        (mode_id for mode_id, count in clean_counts.items() if count == max(clean_counts.values())),
        default=-1,
    )
    if clean_anchor < 0:
        raise RuntimeError("failed to determine clean anchor mode")

    variants = sorted(by_variant)
    contingency_rows: list[dict[str, object]] = []
    matrix = np.zeros((len(variants), modes), dtype=np.int64)
    variant_entropies: dict[str, float] = {}
    for row_index, variant in enumerate(variants):
        total = sum(by_variant[variant].values())
        row: dict[str, object] = {"variant": variant, "total": total}
        probabilities = []
        for mode_id in range(modes):
            count = by_variant[variant][mode_id]
            matrix[row_index, mode_id] = count
            row[f"mode_{mode_id}_count"] = count
            row[f"mode_{mode_id}_fraction"] = count / total
            probabilities.append(count / total)
        variant_entropies[variant] = -sum(value * math.log(value) for value in probabilities if value > 0)
        contingency_rows.append(row)

    changed: dict[str, list[bool]] = defaultdict(list)
    for variants_by_base in base_modes.values():
        if "clean" not in variants_by_base:
            continue
        for variant, mode_id in variants_by_base.items():
            if variant != "clean":
                changed[variant].append(mode_id != variants_by_base["clean"])
    audit = {
        "cache": str(Path(cache_path).resolve()),
        "cache_samples": len(cache),
        "unique_environment_samples": len(ordered),
        "clean_anchor_mode": clean_anchor,
        "clean_mode_counts": {str(key): value for key, value in sorted(clean_counts.items())},
        "nmi_mode_variant": _normalized_mutual_information(matrix),
        "variant_mode_entropy": variant_entropies,
        "mode_change_rate_from_clean": {
            variant: float(np.mean(values)) for variant, values in sorted(changed.items()) if values
        },
        # 从列联表的另一方向检查每个模式由哪些合成环境构成。
        "mode_variant_composition": {
            str(mode_id): {
                variant: {
                    "count": int(matrix[row_index, mode_id]),
                    "fraction_within_mode": float(
                        matrix[row_index, mode_id] / max(matrix[:, mode_id].sum(), 1)
                    ),
                }
                for row_index, variant in enumerate(variants)
            }
            for mode_id in range(modes)
        },
    }
    return clean_anchor, contingency_rows, distance_rows, audit


def _run_image_split(
    records: Sequence[SampleRecord],
    *,
    split: str,
    pipeline: RoutingDiagnosticPipeline,
    model: torch.nn.Module,
    backbone: torch.nn.Module,
    preprocessor: torch.nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """对一个 manifest split 运行冻结候选模式推理。"""

    dataset = IndustrialAnomalyDataset(records, image_size=config.get("image_size", 256))
    loader = make_loader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        seed=int(config.get("seed", 9826)),
        pin_memory=device.type == "cuda",
    )
    output_rows: list[dict[str, object]] = []
    first_batch_summary: dict[str, object] | None = None
    with torch.inference_mode():
        for batch in loader:
            image = torch.as_tensor(batch["image"]).to(device, non_blocking=device.type == "cuda")
            raw = torch.as_tensor(batch["image_raw"]).to(device, non_blocking=device.type == "cuda")
            encoded = backbone(image)
            feature = preprocessor(encoded)
            if not isinstance(feature, Tensor):
                raise TypeError("feature preprocessor must return one tensor")
            _, standardized = model.encode_environment(raw, detach_learned=True)  # type: ignore[attr-defined]
            result = pipeline.predict_features(feature, standardized)
            if first_batch_summary is None:
                first_batch_summary = result.summary()

            labels = torch.as_tensor(batch["label"]).flatten().cpu()
            full_ids = result.assignment.top_indices.detach().cpu()
            full_weights = result.assignment.top_weights.detach().float().cpu()
            candidate = result.candidate_scores.detach().float().cpu()
            raw_candidate = result.raw_candidate_scores.detach().float().cpu()
            transport = result.per_mode_transport_magnitude.detach().float().cpu()
            support = result.assignment.support_score.detach().float().cpu()
            in_support = result.assignment.in_support.detach().cpu()
            photo_distance = result.photo_distance.detach().float().cpu()
            learned_distance = result.learned_distance.detach().float().cpu()
            policy_cpu = {name: value.detach().float().cpu() for name, value in result.policy_scores.items()}

            batch_size = labels.numel()
            for index in range(batch_size):
                dataset_name = str(batch["dataset"][index])  # type: ignore[index]
                category = str(batch["category"][index])  # type: ignore[index]
                domain = str(batch["domain"][index])  # type: ignore[index]
                variant = str(batch["variant_id"][index])  # type: ignore[index]
                assignment = assign_domain(dataset_name, category, domain, variant)
                photo = float(photo_distance[index].item())
                learned_value = float(learned_distance[index].item())
                row: dict[str, object] = {
                    "sample_index": len(output_rows),
                    "split": split,
                    "image_path": str(batch["path"][index]),  # type: ignore[index]
                    "base_id": str(batch["base_id"][index]),  # type: ignore[index]
                    "dataset": dataset_name,
                    "category": category,
                    "domain": domain,
                    "environment": assignment.environment,
                    "environment_family": assignment.family,
                    "is_primary": assignment.is_primary,
                    "variant_id": variant,
                    "label": int(labels[index].item()),
                    "environment_support_score": float(support[index].item()),
                    "in_environment_support": bool(in_support[index].item()),
                    "top_mode_ids": [int(value) for value in full_ids[index].tolist()],
                    "top_mode_weights": [float(value) for value in full_weights[index].tolist()],
                    "candidate_scores": [float(value) for value in candidate[index].tolist()],
                    "raw_candidate_scores": [float(value) for value in raw_candidate[index].tolist()],
                    "per_mode_transport_magnitude": [float(value) for value in transport[index].tolist()],
                    "photo_distance": photo,
                    "learned_distance": learned_value,
                    "learned_share": learned_value / max(photo + learned_value, 1e-12),
                    "image_min_mode": int(candidate[index].argmin().item()),
                    "raw_min_mode": int(raw_candidate[index].argmin().item()),
                    "clean_anchor_mode": pipeline.clean_anchor_mode,
                }
                row.update({f"{name}_score": float(values[index].item()) for name, values in policy_cpu.items()})
                output_rows.append(row)
    if first_batch_summary is None:
        raise RuntimeError(f"split {split!r} produced no batches")
    return output_rows, first_batch_summary


def _fit_policy_calibration(
    validation_rows: Sequence[Mapping[str, object]], probability: float
) -> tuple[dict[str, float], dict[str, object]]:
    if any(int(row["label"]) != 0 or str(row["split"]) != "val" for row in validation_rows):
        raise ValueError("policy calibration requires normal validation rows only")
    thresholds: dict[str, float] = {}
    records: dict[str, object] = {}
    for policy in ROUTING_POLICIES:
        values = [float(row[f"{policy}_score"]) for row in validation_rows]
        threshold = _quantile(values, probability)
        thresholds[policy] = threshold
        records[policy] = {
            "normal_count": len(values),
            "quantile": probability,
            "threshold": threshold,
            "score_sha256": _score_digest(values),
            "decision_rule": "score_strictly_greater_than_threshold",
        }
    return thresholds, records


def _policy_metrics(
    test_rows: Sequence[Mapping[str, object]], thresholds: Mapping[str, float]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, bool], list[Mapping[str, object]]] = defaultdict(list)
    for row in test_rows:
        grouped[
            (
                str(row["dataset"]),
                str(row["category"]),
                str(row["environment"]),
                bool(row["is_primary"]),
            )
        ].append(row)
    metrics: list[dict[str, object]] = []
    for (dataset, category, environment, is_primary), rows in sorted(grouped.items()):
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        for policy in ROUTING_POLICIES:
            scores = np.asarray([float(row[f"{policy}_score"]) for row in rows], dtype=np.float64)
            normal = scores[labels == 0]
            anomaly = scores[labels == 1]
            threshold = float(thresholds[policy])
            metric: dict[str, object] = {
                "dataset": dataset,
                "category": category,
                "environment": environment,
                "is_primary": is_primary,
                "policy": policy,
                "threshold": threshold,
                "samples": int(labels.size),
                "normal_samples": int(normal.size),
                "anomaly_samples": int(anomaly.size),
                "normal_fpr": None if normal.size == 0 else float(np.mean(normal > threshold)),
                "anomaly_tpr": None if anomaly.size == 0 else float(np.mean(anomaly > threshold)),
                "image_auroc": None,
                "normal_score_median": None if normal.size == 0 else float(np.median(normal)),
                "normal_threshold_ratio_median": (
                    None if normal.size == 0 else float(np.median(normal / threshold))
                ),
                "anomaly_score_median": None if anomaly.size == 0 else float(np.median(anomaly)),
            }
            if normal.size and anomaly.size:
                metric["image_auroc"] = binary_auroc(labels, scores)
            metrics.append(metric)

    # 相对 P0 的归一化 normal 中位数下降在同一环境内计算，避免 checkpoint 尺度差异。
    index = {(str(row["environment"]), str(row["policy"])): row for row in metrics}
    for row in metrics:
        p0 = index.get((str(row["environment"]), "posterior_top2"))
        current = row.get("normal_threshold_ratio_median")
        reference = None if p0 is None else p0.get("normal_threshold_ratio_median")
        row["normal_ratio_drop_vs_p0"] = (
            None
            if current is None or reference is None or float(reference) == 0
            else 1.0 - float(current) / float(reference)
        )
    return metrics


def _verify_reference(
    test_rows: Sequence[Mapping[str, object]], reference_path: str | Path, tolerance: float
) -> dict[str, object]:
    """确认 P0 与 v2 正式预测逐样本一致，防止诊断路径悄然改变模型。"""

    reference_rows = _read_jsonl(reference_path)
    reference = {
        (str(row["base_id"]), str(row.get("variant_id", "clean"))): float(row["image_score"])
        for row in reference_rows
    }
    current = {
        (str(row["base_id"]), str(row.get("variant_id", "clean"))): float(row["posterior_top2_score"])
        for row in test_rows
    }
    if len(reference) != len(reference_rows) or len(current) != len(test_rows):
        raise ValueError("reference/current predictions contain duplicate sample identities")
    if set(reference) != set(current):
        raise ValueError("v3 P0 and v2 reference sample sets differ")
    errors = [abs(current[key] - reference[key]) for key in sorted(current)]
    maximum = max(errors, default=0.0)
    if maximum > tolerance:
        raise ValueError(f"posterior_top2 differs from v2: max_abs_error={maximum:.9g}")
    return {
        "path": str(_prediction_file(reference_path)),
        "sha256": file_digest(_prediction_file(reference_path)),
        "samples": len(current),
        "tolerance": tolerance,
        "max_abs_error": maximum,
        "passed": True,
    }


def _descriptor_summary(rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    summaries: list[dict[str, object]] = []
    for group, values in sorted(grouped.items()):
        photo = [float(row["photo_distance"]) for row in values]
        learned = [float(row["learned_distance"]) for row in values]
        share = [float(row["learned_share"]) for row in values]
        summaries.append(
            {
                "group": group,
                "photo": _distribution_summary(photo),
                "learned": _distribution_summary(learned),
                "learned_share": _distribution_summary(share),
            }
        )

    # AUROC 只衡量描述距离中的内容/缺陷泄漏，不参与策略继续门。
    auc_rows: list[dict[str, object]] = []
    by_environment: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            by_environment[str(row["environment"])].append(row)
    for environment, values in sorted(by_environment.items()):
        labels = np.asarray([int(row["label"]) for row in values], dtype=np.int64)
        if not (np.any(labels == 0) and np.any(labels == 1)):
            continue
        auc_rows.append(
            {
                "environment": environment,
                "photo_distance_anomaly_auroc": binary_auroc(
                    labels, [float(row["photo_distance"]) for row in values]
                ),
                "learned_distance_anomaly_auroc": binary_auroc(
                    labels, [float(row["learned_distance"]) for row in values]
                ),
                "interpretation": "content_or_defect_leakage_diagnostic_only",
            }
        )
    return summaries, auc_rows


def run_diagnosis(config: Mapping[str, Any]) -> Path:
    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    probability = float(config.get("calibration_quantile", 0.95))
    if not 0.0 < probability < 1.0:
        raise ValueError("calibration_quantile must lie in (0,1)")

    output = Path(required(config, "output_directory")).resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = Path(required(config, "manifest")).resolve()
    dataset_name = None if config.get("dataset") is None else str(config["dataset"])
    category = str(required(config, "category"))
    validation = select_records(
        manifest,
        category=category,
        split=str(config.get("validation_split", "val")),
        dataset=dataset_name,
        normal_only=False,
    )
    test = select_records(
        manifest,
        category=category,
        split=str(config.get("test_split", "test")),
        dataset=dataset_name,
        normal_only=False,
    )
    if any(record.label != 0 for record in validation):
        raise ValueError("v3 calibration split must contain normal samples only")
    # 步骤 1：推理包已包含条件统计、模式库和两条流，直接加载并冻结即可。
    bundle_path = Path(required(config, "inference_bundle")).resolve()
    model, frozen_info = load_inference_bundle(bundle_path, device=device)
    clean_anchor, contingency_rows, train_distances, mode_audit = _collect_train_environment_audit(
        required(config, "forward_train_cache"), model, device=device
    )
    pipeline = RoutingDiagnosticPipeline(
        model,  # type: ignore[arg-type]
        clean_anchor_mode=clean_anchor,
        environment_steps=int(config.get("environment_steps", 4)),
        normality_steps=int(config.get("normality_steps", 20)),
        softmin_temperature=float(config.get("softmin_temperature", 0.1)),
        output_size=config.get("image_size", 256),
        top_fraction=float(config.get("top_fraction", 0.03)),
    )
    backbone = FrozenWideResNet50(
        weights_path=config.get("backbone_weights"),
        reference_resnet_path=config.get("reference_resnet_source"),
    ).to(device).eval().requires_grad_(False)
    preprocessor = WTFeaturePreprocessor(
        branch_index=_branch_for_channels(model.environment_flow.velocity.in_channels)  # type: ignore[attr-defined]
    ).to(device).eval().requires_grad_(False)

    # 步骤 2：先跑 clean val 拟合五个独立阈值，再跑 test；二者不能混合校准。
    validation_rows, validation_first = _run_image_split(
        validation,
        split="val",
        pipeline=pipeline,
        model=model,
        backbone=backbone,
        preprocessor=preprocessor,
        config=config,
        device=device,
    )
    thresholds, calibration_records = _fit_policy_calibration(validation_rows, probability)
    test_rows, test_first = _run_image_split(
        test,
        split="test",
        pipeline=pipeline,
        model=model,
        backbone=backbone,
        preprocessor=preprocessor,
        config=config,
        device=device,
    )
    # 可选地核对 P0 与已有 v2 逐样本结果；未回传旧 JSONL 时不阻断诊断。
    reference_audit = None
    if config.get("v2_predictions") is not None:
        reference_audit = _verify_reference(
            test_rows,
            str(config["v2_predictions"]),
            tolerance=float(config.get("p0_reproduction_tolerance", 1e-6)),
        )
    policy_metrics = _policy_metrics(test_rows, thresholds)

    # 步骤 3：形成逐样本环境距离表和组级摘要；异常标签只用于诊断，不参与校准。
    descriptor_rows = list(train_distances)
    for row in validation_rows + test_rows:
        label_name = "normal" if int(row["label"]) == 0 else "anomaly"
        group = (
            "clean_validation_normal"
            if row["split"] == "val"
            else f"{row['environment']}_test_{label_name}"
        )
        descriptor_rows.append(
            {
                "base_id": row["base_id"],
                "split": row["split"],
                "group": group,
                "environment": row["environment"],
                "variant_id": row["variant_id"],
                "label": row["label"],
                "top1_mode": row["top_mode_ids"][0],  # type: ignore[index]
                "photo_distance": row["photo_distance"],
                "learned_distance": row["learned_distance"],
                "learned_share": row["learned_share"],
            }
        )
    descriptor_summaries, descriptor_auc = _descriptor_summary(descriptor_rows)

    predictions_path = _write_jsonl(validation_rows + test_rows, output / "routing_predictions.jsonl")
    metrics_path = write_csv(policy_metrics, output / "policy_metrics.csv")
    contingency_path = write_csv(contingency_rows, output / "mode_variant_contingency.csv")
    distance_path = write_csv(descriptor_rows, output / "descriptor_distance_decomposition.csv")
    calibration_path = atomic_write_json(
        {
            "source_split": "clean_normal_validation_only",
            "category": category,
            "records_digest": records_digest(validation),
            "inference_bundle_sha256": frozen_info.sha256,
            "policies": calibration_records,
        },
        output / "policy_calibration.json",
    )
    summary = {
        "phase": "msdflow_pilot_v3_routing_diagnosis",
        "config": {
            "path": config.get("_config_path"),
            "sha256": config.get("_config_sha256"),
        },
        "dataset": dataset_name,
        "category": category,
        "device": str(device),
        "counts": {"validation": len(validation_rows), "test": len(test_rows)},
        "manifest": {
            "path": str(manifest),
            "sha256": file_digest(manifest),
            "validation_records_sha256": records_digest(validation),
            "test_records_sha256": records_digest(test),
        },
        "frozen_model": frozen_info.to_dict(),
        "backbone": backbone.summary(),
        "model_frozen": not model.training and not any(parameter.requires_grad for parameter in model.parameters()),
        "settings": {
            "policies": list(ROUTING_POLICIES),
            "clean_anchor_mode": clean_anchor,
            "environment_steps": pipeline.environment_steps,
            "normality_steps": pipeline.normality_solver.steps,
            "softmin_temperature": pipeline.softmin_temperature,
            "top_fraction": pipeline.top_fraction,
            "top_k": pipeline.top_k,
            "output_size": list(pipeline.output_size),
        },
        "mode_variant_audit": mode_audit,
        "descriptor_group_summaries": descriptor_summaries,
        "descriptor_anomaly_auroc": descriptor_auc,
        "v2_reproduction_audit": reference_audit,
        "diagnostic_questions": {
            "wrong_routing": "compare P1/P2/P3 with posterior_top2 in policy_metrics.csv",
            "transport_harm": "compare image_min_all4 with raw_no_transport_min4",
            "missing_environment_support": "inspect shifted-normal support scores and all four candidate scores",
            "descriptor_content_leakage": "inspect Photo/learned distance groups and clean anomaly AUROC",
        },
        "first_batch_intermediates": {"validation": validation_first, "test": test_first},
        "artifacts": {
            name: {"path": str(path), "sha256": file_digest(path)}
            for name, path in (
                ("routing_predictions", predictions_path),
                ("policy_calibration", calibration_path),
                ("policy_metrics", metrics_path),
                ("mode_variant_contingency", contingency_path),
                ("descriptor_distance_decomposition", distance_path),
            )
        },
    }
    if not summary["model_frozen"]:
        raise RuntimeError("diagnostic inference changed the frozen model state")
    return atomic_write_json(summary, output / "routing_diagnostic_summary.json")


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run frozen MSD-Flow routing diagnostics")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = run_diagnosis(load_json_config(args.config))
    print(json.dumps({"routing_diagnostic_summary": str(path)}, ensure_ascii=False, indent=2))
