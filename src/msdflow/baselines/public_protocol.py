"""公开 baseline 的统一数据与分数协议。

输入是阶段一生成的 JSONL manifest。``prepare_public_baseline`` 把一个
dataset/category/environment/seed 作业映射成 MVTec 目录，并写出逐样本索引；
作者代码只需把每个测试图像的原始分数写入 native JSONL。
``finalize_public_baseline`` 再用同环境正常 reference 校准固定阈值并导出主表行。

本模块没有任何模型代码，因此不会悄悄改变第三方方法。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from envfm.data.records import SampleRecord, read_records_jsonl, records_digest
from envfm.evaluation.metrics import aupro, binary_auroc, binary_rates_at_threshold, pixel_auroc
from envfm.training.checkpoint import atomic_write_json, file_digest
from msdflow.score_calibration import EmpiricalTailCalibrator

from .public_registry import PUBLIC_BASELINES


_RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True, slots=True)
class BaselineProtocolConfig:
    """一个公开 baseline 作业的最小、可审计配置。"""

    method: str
    manifest: str
    dataset: str
    category: str
    environment: str
    target_domain: str
    reference_shots: int
    reference_seed: int
    output_directory: str
    link_mode: str = "hardlink"
    target_fpr: float = 0.05

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BaselineProtocolConfig":
        required = (
            "method",
            "manifest",
            "dataset",
            "category",
            "environment",
            "target_domain",
            "reference_shots",
            "reference_seed",
            "output_directory",
        )
        missing = [key for key in required if key not in value or value[key] in {None, ""}]
        if missing:
            raise ValueError(f"baseline protocol config is missing {missing}")
        config = cls(
            method=str(value["method"]),
            manifest=str(value["manifest"]),
            dataset=str(value["dataset"]),
            category=str(value["category"]),
            environment=str(value["environment"]),
            target_domain=str(value["target_domain"]),
            reference_shots=int(value["reference_shots"]),
            reference_seed=int(value["reference_seed"]),
            output_directory=str(value["output_directory"]),
            link_mode=str(value.get("link_mode", "hardlink")),
            target_fpr=float(value.get("target_fpr", 0.05)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.method not in PUBLIC_BASELINES or not PUBLIC_BASELINES[self.method].main_table:
            raise ValueError(f"method must be a main-table public baseline: {sorted(PUBLIC_BASELINES)}")
        if self.reference_shots < 1:
            raise ValueError("reference_shots must be positive")
        if self.link_mode not in {"hardlink", "symlink", "copy"}:
            raise ValueError("link_mode must be hardlink, symlink, or copy")
        if not 0.0 < self.target_fpr < 1.0:
            raise ValueError("target_fpr must be between zero and one")


def _jsonl_write(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _jsonl_read(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} must be an object")
            rows.append(value)
    return rows


def _safe_name(value: str) -> str:
    """生成第三方 ImageFolder/DataLoader 都能接受的短文件名。"""

    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned.strip("_") or "sample"


def _record_filename(record: SampleRecord) -> str:
    digest = sha256(record.base_id.encode("utf-8")).hexdigest()[:12]
    # RD++/GNL 的作者 loader 只 glob ``*.png``。PIL/OpenCV 会按文件头解码，
    # 因此链接后的统一扩展名既不改变像素，也能让五个作者 loader 读取同一清单。
    return f"{digest}_{_safe_name(Path(record.image_path).stem)}.png"


def _materialize(source: Path, target: Path, mode: str) -> str:
    """链接或复制一个文件；hardlink 跨盘失败时安全退化为 copy。"""

    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
        return "symlink"
    if mode == "hardlink":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            shutil.copy2(source, target)
            return "copy_fallback"
    shutil.copy2(source, target)
    return "copy"


def _materialize_rgb_image(source: Path, target: Path, mode: str) -> str:
    """暂存三通道图像，RGB 原图仍使用零拷贝链接。

    MVTec AD 2 的部分类别只有测试图是灰度图，而五个作者实现都使用
    ImageNet 三通道归一化。仅对非 RGB 输入做确定性的 ``convert('RGB')``
    并保存为 PNG；这等价于把灰度值复制到三个通道，不引入新语义。
    """

    if not source.is_file():
        raise FileNotFoundError(source)
    with Image.open(source) as image:
        if image.mode == "RGB":
            return _materialize(source, target, mode)
        converted = image.convert("RGB")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        converted.save(target, format="PNG")
    return "rgb_png_conversion"


def _write_loader_compat_zero_mask(image_path: Path, mask_path: Path) -> None:
    """为作者 loader 写一张与原图同尺寸的全零掩码。

    RobustAD/PiledBags 官方只提供图像级标签。五个第三方 loader 却把所有
    anomaly 都当作分割样本，并无条件打开 ground_truth/*_mask.png。
    这张全零图只满足 loader 的文件契约；统一 finalizer 仍根据
    original_mask_path=None 留空像素指标，不会把它当作真实标注。
    """

    with Image.open(image_path) as image:
        size = image.size
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.exists() or mask_path.is_symlink():
        mask_path.unlink()
    Image.new("L", size, color=0).save(mask_path, format="PNG")


def _select_records(
    records: Sequence[SampleRecord], config: BaselineProtocolConfig
) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord]]:
    """返回 source normal train、target normal reference、剩余 target test。"""

    category_rows = [
        record
        for record in records
        if record.dataset == config.dataset and record.category == config.category
    ]
    source_train = sorted(
        (
            record
            for record in category_rows
            if record.split == "train" and record.label == 0 and record.variant_id == "clean"
        ),
        key=lambda record: (record.base_id, record.image_path),
    )
    # 部分真实域移数据把环境写在 domain，部分旧 manifest 写在 variant_id；
    # target_domain 显式给出后只做相等匹配，不用异常标签猜域。
    target = [
        record
        for record in category_rows
        if record.split == "test"
        and record.label in {0, 1}
        and (record.domain == config.target_domain or record.variant_id == config.target_domain)
    ]
    target_normals = sorted(
        (record for record in target if record.label == 0),
        key=lambda record: (record.base_id, record.image_path),
    )
    if not source_train:
        raise ValueError("no clean source normal training records were found")
    if len(target_normals) < config.reference_shots:
        raise ValueError(
            f"target domain has {len(target_normals)} normal samples, fewer than "
            f"reference_shots={config.reference_shots}"
        )
    rng = np.random.default_rng(config.reference_seed)
    chosen = sorted(rng.choice(len(target_normals), size=config.reference_shots, replace=False).tolist())
    references = [target_normals[index] for index in chosen]
    reference_ids = {record.base_id for record in references}
    evaluation = sorted(
        (record for record in target if record.base_id not in reference_ids),
        key=lambda record: (record.label, record.defect_type, record.base_id, record.image_path),
    )
    if not any(record.label == 0 for record in evaluation):
        raise ValueError("reference selection left no target normal sample for FPR evaluation")
    if not any(record.label == 1 for record in evaluation):
        raise ValueError("target environment contains no anomaly sample for AUROC/TPR")
    return source_train, references, evaluation


def _stage_row(
    record: SampleRecord,
    *,
    role: str,
    category_root: Path,
    link_mode: str,
) -> dict[str, Any]:
    if role == "source_train":
        relative = Path("train") / "good" / _record_filename(record)
    elif record.label == 0:
        relative = Path("test") / "good" / _record_filename(record)
    else:
        defect = _safe_name(record.defect_type or "anomaly")
        relative = Path("test") / defect / _record_filename(record)
    staged_image = category_root / relative
    materialized_as = _materialize_rgb_image(Path(record.image_path), staged_image, link_mode)

    staged_mask: Path | None = None
    staged_mask_kind: str | None = None
    if record.label == 1:
        defect = relative.parent.name
        staged_mask = category_root / "ground_truth" / defect / f"{staged_image.stem}_mask.png"
        if record.mask_path is not None:
            _materialize(Path(record.mask_path), staged_mask, link_mode)
            staged_mask_kind = "original"
        else:
            # PiledBags 没有官方像素掩码；补零图只防止作者 loader 崩溃。
            _write_loader_compat_zero_mask(Path(record.image_path), staged_mask)
            staged_mask_kind = "synthetic_zero_loader_compat"

    return {
        "base_id": record.base_id,
        "dataset": record.dataset,
        "category": record.category,
        "environment": record.domain,
        "variant_id": record.variant_id,
        "role": role,
        "label": record.label,
        "defect_type": record.defect_type,
        "original_image_path": str(Path(record.image_path).resolve()),
        "original_mask_path": None if record.mask_path is None else str(Path(record.mask_path).resolve()),
        "staged_image_path": str(staged_image.resolve()),
        "staged_relative_path": str(Path(config_path_part(category_root)) / relative).replace("\\", "/"),
        "staged_mask_path": None if staged_mask is None else str(staged_mask.resolve()),
        "staged_mask_kind": staged_mask_kind,
        "materialized_as": materialized_as,
    }


def config_path_part(category_root: Path) -> str:
    """返回索引中稳定的 ``category/...`` 相对前缀。"""

    return category_root.name


def prepare_public_baseline(config: BaselineProtocolConfig) -> Path:
    """准备一个方法/类别/环境/seed 的 MVTec 兼容数据目录。"""

    records = read_records_jsonl(config.manifest)
    source_train, references, evaluation = _select_records(records, config)
    output = Path(config.output_directory).resolve()
    stage_root = output / "mvtec_stage"
    category_root = stage_root / config.category
    if category_root.exists():
        shutil.rmtree(category_root)
    rows: list[dict[str, Any]] = []
    for record in source_train:
        rows.append(_stage_row(record, role="source_train", category_root=category_root, link_mode=config.link_mode))
    for record in references:
        rows.append(_stage_row(record, role="reference", category_root=category_root, link_mode=config.link_mode))
    for record in evaluation:
        rows.append(_stage_row(record, role="evaluation", category_root=category_root, link_mode=config.link_mode))

    index_path = output / "protocol_index.jsonl"
    _jsonl_write(rows, index_path)
    spec = PUBLIC_BASELINES[config.method]
    summary = {
        "phase": "public_baseline_prepare",
        "protocol_version": 1,
        "config": asdict(config),
        "baseline": spec.to_dict(),
        "manifest": {"path": str(Path(config.manifest).resolve()), "sha256": file_digest(config.manifest)},
        "stage_root": str(stage_root),
        "category_root": str(category_root),
        "protocol_index": str(index_path),
        "counts": {
            "source_train_normal": len(source_train),
            "target_reference_normal": len(references),
            "target_evaluation_normal": sum(record.label == 0 for record in evaluation),
            "target_evaluation_anomaly": sum(record.label == 1 for record in evaluation),
            "loader_compat_zero_masks": sum(
                row.get("staged_mask_kind") == "synthetic_zero_loader_compat" for row in rows
            ),
            "rgb_converted_images": sum(
                row.get("materialized_as") == "rgb_png_conversion" for row in rows
            ),
        },
        "digests": {
            "source_train": records_digest(source_train),
            "reference": records_digest(references),
            "evaluation": records_digest(evaluation),
        },
        "reference_base_ids": [record.base_id for record in references],
        "native_prediction_contract": {
            "path": str(output / "native_predictions.jsonl"),
            "required_fields": ["base_id", "image_score"],
            "optional_fields": ["anomaly_map_path"],
            "required_roles": ["reference", "evaluation"],
        },
    }
    atomic_write_json(summary, output / "prepare_summary.json")
    return output / "prepare_summary.json"


def _load_native_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in _jsonl_read(path):
        base_id = str(row.get("base_id", ""))
        if not base_id:
            raise ValueError("native prediction row is missing base_id")
        if base_id in predictions:
            raise ValueError(f"duplicate native prediction for {base_id}")
        score = float(row.get("image_score", math.nan))
        if not math.isfinite(score):
            raise ValueError(f"non-finite image_score for {base_id}")
        predictions[base_id] = {**row, "base_id": base_id, "image_score": score}
    return predictions


def _pixel_metrics(
    evaluation_rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]]
) -> tuple[float | str, float | str]:
    """在所有 map/mask 可用时计算统一像素指标，否则明确留空。"""

    map_paths = [predictions[str(row["base_id"])].get("anomaly_map_path") for row in evaluation_rows]
    if not map_paths or any(path in {None, ""} for path in map_paths):
        return "", ""
    maps = [np.asarray(np.load(str(path)), dtype=np.float32).squeeze() for path in map_paths]
    if any(array.ndim != 2 for array in maps) or len({array.shape for array in maps}) != 1:
        raise ValueError("all anomaly maps must be two-dimensional and have one common shape")
    height, width = maps[0].shape
    masks: list[np.ndarray] = []
    for row in evaluation_rows:
        if int(row["label"]) == 0:
            masks.append(np.zeros((height, width), dtype=np.uint8))
            continue
        mask_path = row.get("original_mask_path")
        if mask_path in {None, ""} or not Path(str(mask_path)).is_file():
            return "", ""
        with Image.open(str(mask_path)) as image:
            resized = image.convert("L").resize((width, height), resample=_RESAMPLING.NEAREST)
            masks.append((np.asarray(resized) > 0).astype(np.uint8))
    mask_array = np.stack(masks)
    map_array = np.stack(maps)
    return float(pixel_auroc(mask_array, map_array)), float(aupro(mask_array, map_array, max_fpr=0.05).score)


def finalize_public_baseline(
    *,
    prepare_summary_path: str | Path,
    native_predictions_path: str | Path,
    output_directory: str | Path | None = None,
    reported_method: str | None = None,
) -> Path:
    """校准作者方法的原始分数并生成统一主表分片。

    ``reported_method`` 只给内部消融重命名结果；数据准备与第三方来源仍由
    ``config.method`` 审计，公开 baseline 的默认行为完全不变。
    """

    with Path(prepare_summary_path).open("r", encoding="utf-8") as handle:
        prepared = json.load(handle)
    config = BaselineProtocolConfig.from_mapping(prepared["config"])
    output_method = config.method if reported_method is None else str(reported_method).strip()
    if not output_method or not all(
        char.isalnum() or char in {"_", "-"} for char in output_method
    ):
        raise ValueError("reported_method must contain only letters, digits, '_' or '-'")
    index_rows = _jsonl_read(prepared["protocol_index"])
    predictions = _load_native_predictions(native_predictions_path)
    scored_rows = [row for row in index_rows if row["role"] in {"reference", "evaluation"}]
    expected_ids = {str(row["base_id"]) for row in scored_rows}
    missing = expected_ids.difference(predictions)
    extras = set(predictions).difference(expected_ids)
    if missing or extras:
        raise ValueError(
            f"native prediction IDs mismatch: missing={len(missing)}, extras={len(extras)}"
        )

    reference_rows = [row for row in scored_rows if row["role"] == "reference"]
    evaluation_rows = [row for row in scored_rows if row["role"] == "evaluation"]
    if any(int(row["label"]) != 0 for row in reference_rows):
        raise ValueError("reference calibration contains an anomaly")
    reference_scores = [predictions[str(row["base_id"])]["image_score"] for row in reference_rows]
    raw_scores = [predictions[str(row["base_id"])]["image_score"] for row in evaluation_rows]
    labels = [int(row["label"]) for row in evaluation_rows]
    calibrator = EmpiricalTailCalibrator.fit(reference_scores, alpha=config.target_fpr)
    decision_scores = calibrator.transform(raw_scores)
    rates = binary_rates_at_threshold(labels, decision_scores, calibrator.threshold)
    pixel_auc, pixel_pro = _pixel_metrics(evaluation_rows, predictions)

    destination = Path(output_directory or Path(prepare_summary_path).parent).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    standard_rows: list[dict[str, Any]] = []
    for index, row in enumerate(evaluation_rows):
        native = predictions[str(row["base_id"])]
        standard_rows.append(
            {
                "method": output_method,
                "base_id": row["base_id"],
                "image_path": row["original_image_path"],
                "mask_path": row["original_mask_path"],
                "label": int(row["label"]),
                "defect_type": row["defect_type"],
                "dataset": config.dataset,
                "category": config.category,
                "environment": config.environment,
                "reference_seed": config.reference_seed,
                "image_score": float(raw_scores[index]),
                "image_score_calibrated": float(decision_scores[index]),
                "decision_threshold": calibrator.threshold,
                "anomaly_map_path": native.get("anomaly_map_path"),
            }
        )
    predictions_path = destination / "predictions.jsonl"
    _jsonl_write(standard_rows, predictions_path)

    metric_row = {
        "method": output_method,
        "dataset": config.dataset,
        "category": config.category,
        "environment": config.environment,
        "reference_seed": config.reference_seed,
        "threshold": calibrator.threshold,
        "decision_score_scale": "negative_log_normal_tail_p",
        "calibration_samples": len(reference_rows),
        "target_fpr_supported": calibrator.supports_target_fpr,
        "normal_fpr": float(rates.fpr),
        "anomaly_tpr": float(rates.tpr),
        "image_auroc": float(binary_auroc(labels, raw_scores)),
        "normal_samples": labels.count(0),
        "anomaly_samples": labels.count(1),
        "pixel_auroc": pixel_auc,
        "aupro_005": pixel_pro,
        "auroc_score_scale": "native_raw",
    }
    metrics_path = destination / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_row))
        writer.writeheader()
        writer.writerow(metric_row)
    atomic_write_json(
        {
            "phase": "public_baseline_finalize",
            "protocol_version": 1,
            "baseline": PUBLIC_BASELINES[config.method].to_dict(),
            "reported_method": output_method,
            "prepare_summary": {
                "path": str(Path(prepare_summary_path).resolve()),
                "sha256": file_digest(prepare_summary_path),
            },
            "native_predictions": {
                "path": str(Path(native_predictions_path).resolve()),
                "sha256": file_digest(native_predictions_path),
            },
            "calibration": calibrator.summary(),
            "metrics": metric_row,
            "outputs": {"predictions": str(predictions_path), "metrics": str(metrics_path)},
        },
        destination / "baseline_summary.json",
    )
    return destination / "baseline_summary.json"
