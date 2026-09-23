"""从 prepared manifest 运行 MSD-Flow 正常校准或测试预测。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from envfm.data.dataset import IndustrialAnomalyDataset
from envfm.data.records import records_digest
from envfm.models.backbone import FrozenWideResNet50
from envfm.models.feature_preprocess import WTFeaturePreprocessor
from envfm.training.checkpoint import file_digest
from msdflow.inference import (
    MSDCalibrationTable,
    MSDFlowInferencePipeline,
    MSDPredictionArtifactWriter,
    fit_msd_calibration,
    load_inference_bundle,
)

from .common import load_json_config, make_loader, required, select_records


def _branch_for_channels(channels: int) -> int:
    mapping = {256: 0, 512: 1, 1024: 2}
    if channels not in mapping:
        raise ValueError("feature_channels must match WRN layer1/2/3")
    return mapping[channels]


def run_inference(config: Mapping[str, Any]) -> Path:
    mode = str(config.get("mode", "predict"))
    if mode not in {"calibrate", "predict"}:
        raise ValueError("mode must be calibrate or predict")
    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    category = str(required(config, "category"))
    split = str(config.get("split", "val" if mode == "calibrate" else "test"))
    manifest_path = Path(required(config, "manifest")).resolve()
    selected = select_records(
        manifest_path,
        category=category,
        split=split,
        dataset=None if config.get("dataset") is None else str(config["dataset"]),
        normal_only=False,
    )
    if mode == "calibrate" and any(record.label != 0 for record in selected):
        raise ValueError("calibration is restricted to normal validation samples")

    # 一个推理包包含条件统计、模式库和两条流，避免人工拼错多份文件。
    model, frozen_info = load_inference_bundle(required(config, "inference_bundle"), device=device)
    calibrator = None
    defect_threshold = None
    if mode == "predict":
        # 正式测试必须使用正常 validation 的固定阈值和环境经验 CDF。
        calibration = MSDCalibrationTable.load(required(config, "calibration"))
        record = calibration.record(category)
        calibrator = calibration.environment_calibrator(category, device=device)
        defect_threshold = record.defect_threshold

    pipeline = MSDFlowInferencePipeline(
        model,
        top_m=int(config.get("top_m", 2)),
        environment_steps=int(config.get("environment_steps", 4)),
        normality_steps=int(config.get("normality_steps", 20)),
        softmin_temperature=float(config.get("softmin_temperature", 0.1)),
        output_size=config.get("image_size", 256),
        top_fraction=float(config.get("top_fraction", 0.03)),
        environment_calibrator=calibrator,
    )
    backbone = FrozenWideResNet50(
        weights_path=config.get("backbone_weights"),
        reference_resnet_path=config.get("reference_resnet_source"),
    ).to(device)
    branch_index = _branch_for_channels(model.environment_flow.velocity.in_channels)
    preprocessor = WTFeaturePreprocessor(branch_index=branch_index).to(device)
    dataset = IndustrialAnomalyDataset(selected, image_size=config.get("image_size", 256))
    loader = make_loader(
        dataset,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        seed=int(config.get("seed", 9826)),
        pin_memory=device.type == "cuda",
    )
    output_directory = Path(required(config, "output_directory")).resolve()
    writer = MSDPredictionArtifactWriter(output_directory)
    collected_defect: list[float] = []
    collected_support: list[float] = []
    collected_transport: list[float] = []
    collected_labels: list[int] = []
    first_batch_summary: dict[str, object] | None = None
    generated_calibration = None
    try:
        for batch in loader:
            image = torch.as_tensor(batch["image"]).to(device, non_blocking=device.type == "cuda")
            raw = torch.as_tensor(batch["image_raw"]).to(device, non_blocking=device.type == "cuda")
            prediction = pipeline.predict_images(
                image, raw, backbone=backbone, feature_preprocessor=preprocessor
            )
            writer.write_batch(batch, prediction, defect_threshold=defect_threshold)
            collected_defect.extend(prediction.defect_score.detach().double().cpu().tolist())
            collected_support.extend(prediction.environment_support_score.detach().double().cpu().tolist())
            collected_transport.extend(prediction.environment_transport_score.detach().double().cpu().tolist())
            collected_labels.extend(torch.as_tensor(batch["label"]).flatten().cpu().tolist())
            # 逐样本统计由 writer 在线聚合；只保留首批详细轨迹，避免 summary 随数据集膨胀。
            if first_batch_summary is None:
                first_batch_summary = prediction.summary()

        if mode == "calibrate":
            calibration = fit_msd_calibration(
                {category: collected_defect},
                {category: collected_support},
                {category: collected_transport},
                labels={category: collected_labels},
                defect_quantile=float(config.get("calibration_quantile", 0.95)),
            )
            calibration_path = calibration.save(output_directory / "calibration.json")
            generated_calibration = {
                "path": str(calibration_path),
                "sha256": file_digest(calibration_path),
                "record": calibration.record(category).to_dict(),
            }
        summary_path = writer.finalize(
            {
                "phase": "msdflow_inference",
                "mode": mode,
                "dataset": config.get("dataset"),
                "category": category,
                "split": split,
                "device": str(device),
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": file_digest(manifest_path),
                    "selected_records": len(selected),
                    "selected_records_sha256": records_digest(selected),
                },
                "frozen_model": frozen_info.to_dict(),
                "backbone": backbone.summary(),
                "settings": {
                    "top_m": pipeline.top_m,
                    "environment_steps": pipeline.environment_steps,
                    "normality_steps": pipeline.normality_solver.steps,
                    "softmin_temperature": pipeline.softmin_temperature,
                    "top_fraction": pipeline.top_fraction,
                    "top_k": pipeline.top_k,
                    "output_size": list(pipeline.output_size),
                },
                "applied_defect_threshold": defect_threshold,
                "generated_calibration": generated_calibration,
                "first_batch_intermediates": first_batch_summary,
            }
        )
    except Exception:
        writer.abort()
        raise
    return summary_path


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate or predict with frozen MSD-Flow")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = run_inference(load_json_config(args.config))
    print(json.dumps({"inference_summary": str(path)}, ensure_ascii=False, indent=2))
