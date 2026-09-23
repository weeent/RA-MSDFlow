"""训练并评估 Reference-Guided Multi-Center OT-CFM。

一个配置对应一个数据集类别，可包含多个目标环境。源特征和冻结评分器只加载一次；
每个目标环境独立选择正常参考、拟合 CORAL、训练 OT-CFM、校准并评价。
异常样本直到最终 ``evaluate`` 步骤才会被编码和评分。
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from envfm.data.records import SampleRecord, records_digest
from envfm.evaluation.metrics import aupro, binary_auroc, binary_rates_at_threshold, pixel_auroc
from envfm.models.backbone import FrozenWideResNet50
from envfm.models.feature_preprocess import WTFeaturePreprocessor
from envfm.training.checkpoint import atomic_write_json, file_digest
from msdflow.inference import RoutingDiagnosticPipeline, load_inference_bundle
from msdflow.reference_alignment import ShrinkageCoral
from msdflow.reference_otcfm import (
    BalancedSourcePatchBank,
    OTCFMTrainingConfig,
    ReferenceOTCFM,
    build_balanced_source_bank,
    feature_maps_to_patches,
    train_reference_otcfm,
)
from msdflow.score_calibration import EmpiricalTailCalibrator, crossfit_folds

from .common import load_json_config, required, select_records
from .reference_pilot import _encode_split


_RESAMPLING = getattr(Image, "Resampling", Image)


def _branch_for_channels(channels: int) -> int:
    mapping = {256: 0, 512: 1, 1024: 2}
    if channels not in mapping:
        raise ValueError("feature channels must match WRN layer1/2/3")
    return mapping[channels]


def _reference_split(
    records: Sequence[SampleRecord],
    *,
    domain: str,
    shots: int,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """按唯一 base_id 固定抽 normal reference，并从目标评价集删除它们。"""

    normal = [record for record in records if record.split == "test" and record.domain == domain and record.label == 0]
    base_ids = sorted({record.base_id for record in normal})
    if shots <= 0 or len(base_ids) <= shots:
        raise ValueError(
            f"domain {domain!r} needs more than {shots} target normal base ids; found {len(base_ids)}"
        )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(base_ids), generator=generator).tolist()
    reference_ids = {base_ids[index] for index in order[:shots]}
    reference = [record for record in normal if record.base_id in reference_ids]
    evaluation = [
        record
        for record in records
        if record.split == "test"
        and record.domain == domain
        and not (record.label == 0 and record.base_id in reference_ids)
        and record.label in {0, 1}
    ]
    if not any(record.label == 0 for record in evaluation) or not any(record.label == 1 for record in evaluation):
        raise ValueError("evaluation needs at least one remaining normal and one anomaly")
    return reference, evaluation


@torch.inference_mode()
def _score_features(
    pipeline: RoutingDiagnosticPipeline,
    features: Tensor,
    *,
    device: torch.device,
    batch_size: int,
    clean_anchor_mode: int,
) -> tuple[list[float], np.ndarray]:
    """使用同一冻结 normality head 返回图像分数和像素异常图。"""

    scores: list[float] = []
    maps: list[np.ndarray] = []
    mode_count = pipeline.model.mode_bank.n_components
    for start in range(0, features.shape[0], batch_size):
        chunk = features[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
        expanded = chunk.unsqueeze(1).expand(-1, mode_count, -1, -1, -1)
        candidates = pipeline._normality_maps(expanded)  # noqa: SLF001 - 复用冻结评分器的候选图
        chosen = candidates[:, clean_anchor_mode]
        score = pipeline._single_image_scores(chosen)  # noqa: SLF001
        scores.extend(float(value) for value in score.cpu())
        maps.extend(chosen.float().cpu().numpy())
    return scores, np.asarray(maps, dtype=np.float32)


def _load_masks(records: Sequence[SampleRecord], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """只读取 mask；``valid`` 标记该数据集是否提供真实像素标注。"""

    masks: list[np.ndarray] = []
    valid: list[bool] = []
    for record in records:
        if record.mask_path is None:
            masks.append(np.zeros((image_size, image_size), dtype=np.uint8))
            valid.append(record.label == 0)
            continue
        with Image.open(record.mask_path) as image:
            resized = image.convert("L").resize((image_size, image_size), resample=_RESAMPLING.NEAREST)
            masks.append((np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8))
            valid.append(True)
    return np.asarray(masks), np.asarray(valid, dtype=bool)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("calibration scores are empty")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear"))


def _metric_row(
    *,
    method: str,
    environment: str,
    records: Sequence[SampleRecord],
    scores: Sequence[float],
    decision_scores: Sequence[float],
    maps: np.ndarray,
    threshold: float,
    score_scale: str,
    calibration_samples: int | None,
    target_fpr_supported: bool | None,
    image_size: int,
    compute_pixel_metrics: bool,
) -> dict[str, object]:
    labels = [record.label for record in records]
    # 连续原始分数保留方法本身的排序信息，供 AUROC 使用；FPR/TPR 使用校准后的
    # 决策分数。这样不会因 20-shot 经验 p 值只有少量离散等级而损伤 AUROC。
    rates = binary_rates_at_threshold(labels, decision_scores, threshold)
    row: dict[str, object] = {
        "method": method,
        "environment": environment,
        "threshold": threshold,
        "decision_score_scale": score_scale,
        "calibration_samples": "" if calibration_samples is None else calibration_samples,
        "target_fpr_supported": "" if target_fpr_supported is None else target_fpr_supported,
        "normal_fpr": rates.fpr,
        "anomaly_tpr": rates.tpr,
        "image_auroc": binary_auroc(labels, scores),
        "normal_samples": sum(label == 0 for label in labels),
        "anomaly_samples": sum(label == 1 for label in labels),
        "pixel_auroc": "",
        "aupro_005": "",
    }
    if compute_pixel_metrics:
        masks, valid = _load_masks(records, image_size)
        # 未提供 anomaly mask 的数据集不能伪造 pixel 指标；只在全部评价行有效时计算。
        if bool(valid.all()) and masks.any():
            row["pixel_auroc"] = pixel_auroc(masks, maps)
            row["aupro_005"] = aupro(masks, maps, max_fpr=0.05).score
    return row


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _training_config(value: Mapping[str, Any] | None) -> OTCFMTrainingConfig:
    source = dict(value or {})
    allowed = set(OTCFMTrainingConfig.__dataclass_fields__)
    unknown = set(source).difference(allowed)
    if unknown:
        raise ValueError(f"unknown OT-CFM training settings: {sorted(unknown)}")
    return OTCFMTrainingConfig(**source)


def _fit_reference_adapter(
    *,
    channels: int,
    source_features: Tensor,
    reference_features: Tensor,
    source_bank: BalancedSourcePatchBank,
    source_std: Tensor,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    output_directory: Path,
    resume: bool,
) -> tuple[ShrinkageCoral, ReferenceOTCFM, list[dict[str, float]], Path]:
    """仅用给定正常 reference 拟合 CORAL 与 OT-CFM，并返回冻结适配器。

    输入不含异常标签或评价特征。该函数同时用于最终全量 reference 适配器和
    交叉拟合折，确保两条路径的模型、损失和超参数完全相同。
    """

    coral = ShrinkageCoral(
        shrinkage=float(config.get("coral_shrinkage", 0.1)),
        max_patches=int(config.get("coral_max_patches", 20000)),
        # patch 子采样沿用固定 reference seed；训练 seed 只控制 OT-CFM 优化。
        patch_seed=int(config.get("reference_seed", 9826)),
    ).fit(source_features, reference_features)
    model = ReferenceOTCFM(
        channels,
        hidden_dim=int(config.get("hidden_dim", 256)),
        time_dim=int(config.get("time_dim", 64)),
        depth=int(config.get("depth", 3)),
    )
    model.set_statistics(
        source_mean=coral.mean_source.float(),  # type: ignore[union-attr]
        source_std=source_std,
        target_mean=coral.mean_target.float(),  # type: ignore[union-attr]
        coral_transform=coral.transform.float(),  # type: ignore[union-attr]
    )
    history, adapter_path = train_reference_otcfm(
        model,
        source_bank,
        feature_maps_to_patches(reference_features),
        config=_training_config(config.get("training")),
        device=device,
        seed=seed,
        output_directory=output_directory,
        resume=resume,
    )
    model.eval().requires_grad_(False)
    return coral, model, history, adapter_path


def _cross_fitted_reference_scores(
    *,
    channels: int,
    source_features: Tensor,
    reference_features: Tensor,
    reference_records: Sequence[SampleRecord],
    source_bank: BalancedSourcePatchBank,
    source_std: Tensor,
    pipeline: RoutingDiagnosticPipeline,
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    clean_anchor_mode: int,
    seed: int,
    output_directory: Path,
) -> tuple[list[float], list[float], list[dict[str, object]]]:
    """生成 CORAL/OT-CFM 未参与拟合的 reference 正常分数。

    每一折只用其余 reference 拟合适配器，再给 held-out reference 评分。输出顺序
    与 ``reference_records`` 完全一致，可直接用于经验尾校准。中间折仅承担校准，
    不读取正式评价 normal 或 anomaly。
    """

    if reference_features.shape[0] != len(reference_records):
        raise ValueError("reference feature count does not match records")
    folds = crossfit_folds(
        len(reference_records),
        int(config.get("calibration_folds", 5)),
        seed=seed,
    )
    coral_oof = np.full(len(reference_records), np.nan, dtype=np.float64)
    otcfm_oof = np.full(len(reference_records), np.nan, dtype=np.float64)
    fold_summaries: list[dict[str, object]] = []
    for fold_index, (fit_indices, held_out_indices) in enumerate(folds):
        fit_tensor = reference_features[fit_indices]
        held_out_tensor = reference_features[held_out_indices]
        fold_output = output_directory / f"fold_{fold_index:02d}"
        fold_seed = seed + 1000 + fold_index
        coral, model, history, adapter_path = _fit_reference_adapter(
            channels=channels,
            source_features=source_features,
            reference_features=fit_tensor,
            source_bank=source_bank,
            source_std=source_std,
            config=config,
            device=device,
            seed=fold_seed,
            output_directory=fold_output,
            resume=bool(config.get("resume", True)),
        )

        # held-out 图没有参与本折 CORAL 统计或 OT-CFM 训练，消除训练内分数偏低。
        coral_held_out = coral.apply(held_out_tensor)
        coral_scores, _ = _score_features(
            pipeline,
            coral_held_out,
            device=device,
            batch_size=batch_size,
            clean_anchor_mode=clean_anchor_mode,
        )
        otcfm_held_out = model.transport_features(
            held_out_tensor.to(device),
            steps=int(config.get("flow_steps", 8)),
            chunk_size=int(config.get("patch_chunk_size", 4096)),
        ).cpu()
        otcfm_scores, _ = _score_features(
            pipeline,
            otcfm_held_out,
            device=device,
            batch_size=batch_size,
            clean_anchor_mode=clean_anchor_mode,
        )
        coral_oof[held_out_indices] = np.asarray(coral_scores, dtype=np.float64)
        otcfm_oof[held_out_indices] = np.asarray(otcfm_scores, dtype=np.float64)
        fold_summaries.append(
            {
                "fold": fold_index,
                "fit_count": len(fit_indices),
                "held_out_count": len(held_out_indices),
                "fit_base_ids": [reference_records[index].base_id for index in fit_indices],
                "held_out_base_ids": [reference_records[index].base_id for index in held_out_indices],
                "adapter": {"path": str(adapter_path), "sha256": file_digest(adapter_path)},
                "training_last": history[-1] if history else None,
            }
        )
    if not np.isfinite(coral_oof).all() or not np.isfinite(otcfm_oof).all():
        raise RuntimeError("cross-fitting did not score every reference sample exactly once")
    return coral_oof.tolist(), otcfm_oof.tolist(), fold_summaries


def run_reference_otcfm(config: Mapping[str, Any]) -> Path:
    """完成某类别全部目标环境的 source cache、训练、校准与主表评价。"""

    device = torch.device(str(config.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset_name = str(required(config, "dataset"))
    category = str(required(config, "category"))
    manifest_path = Path(required(config, "manifest")).resolve()
    target_domains = {str(name): str(domain) for name, domain in required(config, "target_domains").items()}
    output_root = Path(required(config, "output_directory")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    image_size = int(config.get("image_size", 256))
    batch_size = int(config.get("batch_size", 8))
    reference_shots = int(config.get("reference_shots", 20))
    reference_seed = int(config.get("reference_seed", 9826))
    train_seed = int(config.get("train_seed", 9827))
    clean_anchor_mode = int(config.get("clean_anchor_mode", 0))
    calibration_quantile = float(config.get("calibration_quantile", 0.95))
    if not 0.0 < calibration_quantile < 1.0:
        raise ValueError("calibration_quantile must be between zero and one")
    target_fpr = 1.0 - calibration_quantile

    # 步骤 1：冻结源域异常评分器、WRN 编码器和特征预处理器。
    bundle_path = Path(required(config, "inference_bundle")).resolve()
    scorer, scorer_info = load_inference_bundle(bundle_path, device=device)
    channels = int(scorer.environment_flow.velocity.in_channels)  # type: ignore[attr-defined]
    backbone = FrozenWideResNet50(
        weights_path=config.get("backbone_weights"),
        reference_resnet_path=config.get("reference_resnet_source"),
    ).to(device).eval().requires_grad_(False)
    preprocessor = WTFeaturePreprocessor(branch_index=_branch_for_channels(channels)).to(device).eval().requires_grad_(False)
    pipeline = RoutingDiagnosticPipeline(
        scorer,
        clean_anchor_mode=clean_anchor_mode,
        environment_steps=int(config.get("environment_steps", 4)),
        normality_steps=int(config.get("normality_steps", 20)),
        output_size=image_size,
        top_fraction=float(config.get("top_fraction", 0.03)),
    )

    # 步骤 2：source normal train/val 只编码一次，供所有目标环境复用。
    source_records = [
        record
        for record in select_records(manifest_path, dataset=dataset_name, category=category, split=str(config.get("train_split", "train")))
        if record.label == 0 and record.variant_id == "clean"
    ]
    validation_records = select_records(
        manifest_path,
        dataset=dataset_name,
        category=category,
        split=str(config.get("validation_split", "val")),
        normal_only=True,
    )
    source_features, _, _ = _encode_split(
        source_records, backbone=backbone, preprocessor=preprocessor, config=config, device=device
    )
    validation_features, _, _ = _encode_split(
        validation_records, backbone=backbone, preprocessor=preprocessor, config=config, device=device
    )
    source_bank = build_balanced_source_bank(
        source_features,
        n_modes=int(config.get("source_modes", 4)),
        max_patches=int(config.get("max_source_patches", 40000)),
        seed=train_seed,
    )
    source_scores, _ = _score_features(
        pipeline,
        validation_features,
        device=device,
        batch_size=batch_size,
        clean_anchor_mode=clean_anchor_mode,
    )
    source_threshold = _quantile(source_scores, calibration_quantile)
    test_records = select_records(manifest_path, dataset=dataset_name, category=category, split="test")

    metric_rows: list[dict[str, object]] = []
    environment_summaries: dict[str, object] = {}
    for environment_index, (environment, domain) in enumerate(target_domains.items()):
        environment_output = output_root / environment
        environment_output.mkdir(parents=True, exist_ok=True)
        # 步骤 3：每个环境独立抽取 normal reference；异常图此时尚未被编码。
        references, evaluation = _reference_split(
            test_records, domain=domain, shots=reference_shots, seed=reference_seed
        )
        reference_features, _, _ = _encode_split(
            references, backbone=backbone, preprocessor=preprocessor, config=config, device=device
        )

        # 步骤 4：全部 reference 训练最终适配器；它只用于正式测试推理。
        source_std = source_bank.patches.std(0, unbiased=False).clamp_min(1e-6)
        coral, model, history, adapter_path = _fit_reference_adapter(
            channels=channels,
            source_features=source_features,
            reference_features=reference_features,
            source_bank=source_bank,
            source_std=source_std,
            config=config,
            device=device,
            seed=train_seed + environment_index,
            output_directory=environment_output,
            resume=bool(config.get("resume", True)),
        )

        # 步骤 5：先保留训练内 reference 分数用于审计，再生成交叉拟合分数。
        # 旧实现直接用下列训练内分数定阈值，导致 reference 被拟合得过好、阈值过低。
        raw_reference_scores, _ = _score_features(
            pipeline, reference_features, device=device, batch_size=batch_size, clean_anchor_mode=clean_anchor_mode
        )
        coral_reference = coral.apply(reference_features)
        coral_reference_scores, _ = _score_features(
            pipeline, coral_reference, device=device, batch_size=batch_size, clean_anchor_mode=clean_anchor_mode
        )
        transported_reference = model.transport_features(
            reference_features.to(device),
            steps=int(config.get("flow_steps", 8)),
            chunk_size=int(config.get("patch_chunk_size", 4096)),
        ).cpu()
        otcfm_reference_scores, _ = _score_features(
            pipeline, transported_reference, device=device, batch_size=batch_size, clean_anchor_mode=clean_anchor_mode
        )
        coral_oof_scores, otcfm_oof_scores, fold_summaries = _cross_fitted_reference_scores(
            channels=channels,
            source_features=source_features,
            reference_features=reference_features,
            reference_records=references,
            source_bank=source_bank,
            source_std=source_std,
            pipeline=pipeline,
            config=config,
            device=device,
            batch_size=batch_size,
            clean_anchor_mode=clean_anchor_mode,
            seed=train_seed + environment_index,
            output_directory=environment_output / "score_calibration",
        )
        # threshold-only 没有用 reference 拟合表示，其原始 reference 分数天然是 out-of-fit；
        # CORAL/OT-CFM 则必须使用上面的 held-out 分数。
        calibrators = {
            "target_threshold_only": EmpiricalTailCalibrator.fit(raw_reference_scores, alpha=target_fpr),
            "coral": EmpiricalTailCalibrator.fit(coral_oof_scores, alpha=target_fpr),
            "reference_otcfm": EmpiricalTailCalibrator.fit(otcfm_oof_scores, alpha=target_fpr),
        }
        target_thresholds = {name: calibrator.threshold for name, calibrator in calibrators.items()}

        # 步骤 6：模型和阈值冻结后才读取目标域剩余 normal 与全部 anomaly。
        evaluation_features, _, evaluation_rows = _encode_split(
            evaluation, backbone=backbone, preprocessor=preprocessor, config=config, device=device
        )
        if [row["base_id"] for row in evaluation_rows] != [record.base_id for record in evaluation]:
            raise RuntimeError("feature encoding changed evaluation record order")
        raw_scores, raw_maps = _score_features(
            pipeline, evaluation_features, device=device, batch_size=batch_size, clean_anchor_mode=clean_anchor_mode
        )
        coral_features = coral.apply(evaluation_features)
        coral_scores, coral_maps = _score_features(
            pipeline, coral_features, device=device, batch_size=batch_size, clean_anchor_mode=clean_anchor_mode
        )
        transported = model.transport_features(
            evaluation_features.to(device),
            steps=int(config.get("flow_steps", 8)),
            chunk_size=int(config.get("patch_chunk_size", 4096)),
        ).cpu()
        otcfm_scores, otcfm_maps = _score_features(
            pipeline, transported, device=device, batch_size=batch_size, clean_anchor_mode=clean_anchor_mode
        )
        # 每个 target-calibrated 方法保留原始连续分数，同时生成统一的 -log(p) 决策分数。
        method_outputs: dict[str, dict[str, object]] = {
            "frozen_source": {
                "raw_scores": raw_scores,
                "decision_scores": raw_scores,
                "maps": raw_maps,
                "threshold": source_threshold,
                "score_scale": "raw_source",
                "calibration_samples": None,
                "target_fpr_supported": None,
            },
            "target_threshold_only": {
                "raw_scores": raw_scores,
                "decision_scores": calibrators["target_threshold_only"].transform(raw_scores).tolist(),
                "maps": raw_maps,
                "threshold": target_thresholds["target_threshold_only"],
                "score_scale": "negative_log_normal_tail_p",
                "calibration_samples": calibrators["target_threshold_only"].sample_count,
                "target_fpr_supported": calibrators["target_threshold_only"].supports_target_fpr,
            },
            "coral": {
                "raw_scores": coral_scores,
                "decision_scores": calibrators["coral"].transform(coral_scores).tolist(),
                "maps": coral_maps,
                "threshold": target_thresholds["coral"],
                "score_scale": "negative_log_normal_tail_p",
                "calibration_samples": calibrators["coral"].sample_count,
                "target_fpr_supported": calibrators["coral"].supports_target_fpr,
            },
            "reference_otcfm": {
                "raw_scores": otcfm_scores,
                "decision_scores": calibrators["reference_otcfm"].transform(otcfm_scores).tolist(),
                "maps": otcfm_maps,
                "threshold": target_thresholds["reference_otcfm"],
                "score_scale": "negative_log_normal_tail_p",
                "calibration_samples": calibrators["reference_otcfm"].sample_count,
                "target_fpr_supported": calibrators["reference_otcfm"].supports_target_fpr,
            },
        }
        for method, values in method_outputs.items():
            metric_rows.append(
                _metric_row(
                    method=method,
                    environment=environment,
                    records=evaluation,
                    scores=values["raw_scores"],  # type: ignore[arg-type]
                    decision_scores=values["decision_scores"],  # type: ignore[arg-type]
                    maps=values["maps"],  # type: ignore[arg-type]
                    threshold=float(values["threshold"]),
                    score_scale=str(values["score_scale"]),
                    calibration_samples=(
                        None if values["calibration_samples"] is None else int(values["calibration_samples"])
                    ),
                    target_fpr_supported=(
                        None
                        if values["target_fpr_supported"] is None
                        else bool(values["target_fpr_supported"])
                    ),
                    image_size=image_size,
                    compute_pixel_metrics=bool(config.get("compute_pixel_metrics", True)),
                )
            )

        prediction_path = environment_output / "predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
            for index, record in enumerate(evaluation):
                score_columns: dict[str, float] = {}
                for name, values in method_outputs.items():
                    raw_values = values["raw_scores"]
                    decision_values = values["decision_scores"]
                    # ``*_score`` 保留原始连续分数以兼容已有汇总；新增 calibrated
                    # 字段承担跨环境固定阈值决策。
                    score_columns[f"{name}_score"] = float(raw_values[index])  # type: ignore[index]
                    score_columns[f"{name}_score_calibrated"] = float(decision_values[index])  # type: ignore[index]
                handle.write(
                    json.dumps(
                        {
                            "base_id": record.base_id,
                            "image_path": record.image_path,
                            "label": record.label,
                            "defect_type": record.defect_type,
                            "dataset": dataset_name,
                            "category": category,
                            "environment": environment,
                            "domain": domain,
                            **score_columns,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        atomic_write_json(
            {
                "environment": environment,
                "domain": domain,
                "reference_shots": reference_shots,
                "reference_base_ids": sorted({record.base_id for record in references}),
                "evaluation_normal": sum(record.label == 0 for record in evaluation),
                "evaluation_anomaly": sum(record.label == 1 for record in evaluation),
                "source_threshold": source_threshold,
                "target_thresholds": target_thresholds,
                "score_calibration": {
                    "policy": "cross_fitted_empirical_tail",
                    "fold_count": len(fold_summaries),
                    "folds": fold_summaries,
                    "calibrators": {name: calibrator.summary() for name, calibrator in calibrators.items()},
                    "in_sample_reference_scores": {
                        "coral": {
                            "median": float(np.median(coral_reference_scores)),
                            "maximum": float(np.max(coral_reference_scores)),
                        },
                        "reference_otcfm": {
                            "median": float(np.median(otcfm_reference_scores)),
                            "maximum": float(np.max(otcfm_reference_scores)),
                        },
                    },
                    "out_of_fold_reference_scores": {
                        "coral": {
                            "median": float(np.median(coral_oof_scores)),
                            "maximum": float(np.max(coral_oof_scores)),
                        },
                        "reference_otcfm": {
                            "median": float(np.median(otcfm_oof_scores)),
                            "maximum": float(np.max(otcfm_oof_scores)),
                        },
                    },
                },
                "adapter": {"path": str(adapter_path), "sha256": file_digest(adapter_path)},
                "coral": coral.summary(),
                "training_last": history[-1] if history else None,
                "training_points": len(history),
            },
            environment_output / "summary.json",
        )
        environment_summaries[environment] = {
            "domain": domain,
            "reference_records_sha256": records_digest(references),
            "evaluation_records_sha256": records_digest(evaluation),
            "output": str(environment_output),
        }

    # 步骤 7：每类别直接生成可被矩阵汇总的主表分片与审计摘要。
    _write_csv(metric_rows, output_root / "metrics.csv")
    summary_path = output_root / "otcfm_summary.json"
    atomic_write_json(
        {
            "phase": "reference_guided_multicenter_otcfm",
            "dataset": dataset_name,
            "category": category,
            "manifest": {"path": str(manifest_path), "sha256": file_digest(manifest_path)},
            "source_train": {"records": len(source_records), "sha256": records_digest(source_records)},
            "source_validation": {"records": len(validation_records), "threshold": source_threshold},
            "source_patch_bank": source_bank.summary(),
            "reference_shots": reference_shots,
            "reference_seed": reference_seed,
            "train_seed": train_seed,
            "calibration_quantile": calibration_quantile,
            "target_fpr": target_fpr,
            "calibration_folds": int(config.get("calibration_folds", 5)),
            "feature_channels": channels,
            "branch_index": _branch_for_channels(channels),
            "frozen_scorer": scorer_info.to_dict(),
            "environments": environment_summaries,
            "metrics": metric_rows,
            "training_config": asdict(_training_config(config.get("training"))),
        },
        summary_path,
    )
    return summary_path


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train/evaluate reference-guided multi-center OT-CFM")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result = run_reference_otcfm(load_json_config(args.config))
    print(json.dumps({"otcfm_summary": str(result)}, ensure_ascii=False, indent=2))
