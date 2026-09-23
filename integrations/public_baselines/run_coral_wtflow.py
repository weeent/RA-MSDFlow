"""运行 CORAL alignment + 冻结 WT-Flow scorer 消融。

该入口读取公共协议已准备的数据和已有 WT-Flow ``final.pt``，不重新训练模型。
Reference 用 K 折交叉拟合得到 OOF 分数；最终 CORAL 只由 source normals 和全部
reference normals 拟合，异常与剩余测试正常图仅在最后评分。
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import (  # noqa: E402
    author_repository,
    read_index,
    read_json,
    rows_by_staged_path,
    save_map,
    seed_all,
    write_native_predictions,
)
from msdflow.baselines.coral_wtflow import FrozenWTFlowFeatureScorer  # noqa: E402
from msdflow.reference_alignment import ShrinkageCoral  # noqa: E402
from msdflow.score_calibration import crossfit_folds  # noqa: E402


def _digest(path: str | Path) -> str:
    hasher = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@torch.inference_mode()
def _encode(loader, extractor: nn.Module, pool: nn.Module, device: torch.device) -> torch.Tensor:
    """冻结提取 WT-Flow branch-2 特征，输出 CPU ``[N,1024,H,W]``。"""

    batches: list[torch.Tensor] = []
    extractor.eval()
    for image, _label, _mask in loader:
        image = image.to(device, non_blocking=device.type == "cuda")
        feature = pool(extractor(image)[2])
        feature = nn.functional.layer_norm(feature, feature.shape[1:])
        batches.append(feature.float().cpu())
    if not batches:
        raise ValueError("feature loader is empty")
    return torch.cat(batches, dim=0)


@torch.inference_mode()
def _score_in_batches(
    scorer: FrozenWTFlowFeatureScorer,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[float], np.ndarray]:
    """分批搬运特征到 GPU，避免整套高分辨率特征常驻显存。"""

    scores: list[float] = []
    maps: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        output = scorer.score(features[start : start + batch_size].to(device))
        scores.extend(float(value) for value in output.image_scores.cpu())
        maps.extend(output.anomaly_maps.float().cpu().numpy())
    return scores, np.asarray(maps, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="CORAL-aligned frozen WT-Flow ablation runner")
    parser.add_argument("--prepare-summary", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--coral-shrinkage", type=float, default=0.1)
    parser.add_argument("--coral-max-patches", type=int, default=20000)
    parser.add_argument("--top-fraction", type=float, default=0.03)
    args = parser.parse_args()

    prepared = read_json(args.prepare_summary)
    index_rows = read_index(prepared)
    category = str(prepared["config"]["category"])
    reference_seed = int(prepared["config"]["reference_seed"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed_all(reference_seed)

    with author_repository(args.repository):
        import train_config as c
        import datasets.datasets as author_datasets
        from datasets import MVTecDataset
        from model import MiniUnet
        from models.extractors import build_extractor

        if category not in author_datasets.MVTEC_CLASS_NAMES:
            author_datasets.MVTEC_CLASS_NAMES.append(category)
        c.device = device
        c.data_path = str(prepared["stage_root"])
        c.dataset = "mvtec"
        c.class_name = category
        c.input_size = (args.image_size, args.image_size)
        c.batch_size = args.batch_size
        c.workers = args.workers
        c.BN_enable = False
        c.LN_enable = True
        c.PE_enable = False
        c.single_branch_enable = True
        c.id_branch = 2
        c.top_k = args.top_fraction

        train_data = MVTecDataset(c, is_train=True)
        test_data = MVTecDataset(c, is_train=False)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=device.type == "cuda",
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=device.type == "cuda",
        )
        extractor, _ = build_extractor(c)
        extractor = extractor.to(device).eval().requires_grad_(False)
        pool = nn.AvgPool2d(3, 2, 1) if c.pool_type == "avg" else nn.Identity()
        source_features = _encode(train_loader, extractor, pool, device)
        target_features = _encode(test_loader, extractor, pool, device)

        model = MiniUnet(1024, c.base_channels).to(device)
        try:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        scorer = FrozenWTFlowFeatureScorer(
            model, steps=args.steps, output_size=args.image_size, top_fraction=args.top_fraction
        )

    # 将作者 dataset 的稳定路径顺序映射回公共协议 base_id。
    index_by_path = rows_by_staged_path(index_rows)
    feature_by_base_id: dict[str, torch.Tensor] = {}
    for path, feature in zip(test_data.x, target_features, strict=True):
        row = index_by_path[str(Path(path).resolve())]
        base_id = str(row["base_id"])
        if base_id in feature_by_base_id:
            raise ValueError(f"duplicate target base_id {base_id!r}")
        feature_by_base_id[base_id] = feature
    reference_rows = [row for row in index_rows if row["role"] == "reference"]
    evaluation_rows = [row for row in index_rows if row["role"] == "evaluation"]
    reference_features = torch.stack([feature_by_base_id[str(row["base_id"])] for row in reference_rows])
    evaluation_features = torch.stack([feature_by_base_id[str(row["base_id"])] for row in evaluation_rows])

    # OOF reference 分数：held-out 样本从未参与相应 CORAL 统计估计。
    oof_scores = np.full(len(reference_rows), np.nan, dtype=np.float64)
    fold_audit: list[dict[str, object]] = []
    for fold_index, (fit_indices, held_indices) in enumerate(
        crossfit_folds(len(reference_rows), args.calibration_folds, seed=reference_seed)
    ):
        coral = ShrinkageCoral(
            shrinkage=args.coral_shrinkage,
            max_patches=args.coral_max_patches,
            patch_seed=reference_seed,
        ).fit(source_features, reference_features[fit_indices])
        held_aligned = coral.apply(reference_features[held_indices])
        held_scores, _ = _score_in_batches(
            scorer, held_aligned, device=device, batch_size=args.batch_size
        )
        oof_scores[held_indices] = np.asarray(held_scores, dtype=np.float64)
        fold_audit.append(
            {
                "fold": fold_index,
                "fit_base_ids": [reference_rows[i]["base_id"] for i in fit_indices],
                "held_out_base_ids": [reference_rows[i]["base_id"] for i in held_indices],
                "coral": coral.summary(),
            }
        )
    if not np.isfinite(oof_scores).all():
        raise RuntimeError("cross-fitting did not score every reference exactly once")

    final_coral = ShrinkageCoral(
        shrinkage=args.coral_shrinkage,
        max_patches=args.coral_max_patches,
        patch_seed=reference_seed,
    ).fit(source_features, reference_features)
    aligned_evaluation = final_coral.apply(evaluation_features)
    evaluation_scores, evaluation_maps = _score_in_batches(
        scorer, aligned_evaluation, device=device, batch_size=args.batch_size
    )

    native: list[dict[str, object]] = [
        {"base_id": row["base_id"], "image_score": float(oof_scores[index]), "score_role": "oof_reference"}
        for index, row in enumerate(reference_rows)
    ]
    for row, score, anomaly_map in zip(evaluation_rows, evaluation_scores, evaluation_maps, strict=True):
        native.append(
            {
                "base_id": row["base_id"],
                "image_score": float(score),
                "score_role": "final_aligned_evaluation",
                "anomaly_map_path": save_map(anomaly_map, output, str(row["base_id"])),
            }
        )
    native_path = write_native_predictions(native, output / "native_predictions.jsonl")
    summary = {
        "phase": "coral_wtflow",
        "method": "coral_wtflow",
        "prepare_summary": str(Path(args.prepare_summary).resolve()),
        "checkpoint": {"path": str(Path(args.checkpoint).resolve()), "sha256": _digest(args.checkpoint)},
        "feature_branch": 2,
        "feature_shape": list(source_features.shape[1:]),
        "reference_count": len(reference_rows),
        "evaluation_count": len(evaluation_rows),
        "reference_base_ids": [row["base_id"] for row in reference_rows],
        "folds": fold_audit,
        "final_coral": final_coral.summary(),
        "native_predictions": str(native_path),
        "scores_finite": bool(np.isfinite(oof_scores).all() and np.isfinite(evaluation_scores).all()),
    }
    (output / "coral_wtflow_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
