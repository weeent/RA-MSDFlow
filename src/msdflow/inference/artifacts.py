"""流式写出单一缺陷分数、单一异常图及必要的环境诊断量。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from envfm.training.checkpoint import atomic_write_json, file_digest

from .pipeline import MSDFlowInferenceOutput


class RunningTensorStatistics:
    """逐批累计有限张量的标量统计，不缓存完整异常图。"""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    @torch.no_grad()
    def update(self, value: Tensor) -> None:
        flat = value.detach().double().flatten().cpu()
        if flat.numel() == 0 or not torch.isfinite(flat).all():
            raise ValueError("prediction statistics require non-empty finite tensors")
        self.count += flat.numel()
        self.total += float(flat.sum().item())
        self.total_square += float(flat.square().sum().item())
        self.minimum = min(self.minimum, float(flat.min().item()))
        self.maximum = max(self.maximum, float(flat.max().item()))

    def summary(self) -> dict[str, object]:
        if self.count == 0:
            return {"count": 0}
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
            "finite": True,
        }


class MSDPredictionArtifactWriter:
    """每批写一个压缩异常图分片，逐图标量写入 JSONL。"""

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory).resolve()
        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise FileExistsError(f"prediction directory is not empty: {self.output_directory}")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.maps_directory = self.output_directory / "maps"
        self.maps_directory.mkdir()
        self.predictions_path = self.output_directory / "predictions.jsonl"
        self.temporary_path = self.predictions_path.with_suffix(".jsonl.tmp")
        self.handle = self.temporary_path.open("w", encoding="utf-8", newline="\n")
        self.batch_index = 0
        self.sample_count = 0
        self.closed = False
        self.statistics = {
            name: RunningTensorStatistics()
            for name in ("defect", "support", "transport", "environment", "anomaly_map", "entropy")
        }

    def write_batch(
        self,
        batch: Mapping[str, object],
        output: MSDFlowInferenceOutput,
        *,
        defect_threshold: float | None = None,
    ) -> None:
        """校验一批元数据，并以原子替换方式写入异常图分片。"""

        if self.closed:
            raise RuntimeError("prediction writer is closed")
        scores = output.defect_score.detach().float().cpu()
        maps = output.anomaly_map.detach().float().cpu()
        batch_size = scores.numel()
        if maps.shape[0] != batch_size:
            raise ValueError("anomaly-map batch size does not match scores")

        shard_name = f"batch_{self.batch_index:06d}.npz"
        shard_path = self.maps_directory / shard_name
        temporary = shard_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, anomaly_map=maps.numpy())
        temporary.replace(shard_path)

        required_text = ("path", "base_id", "category", "dataset", "domain", "variant_id")
        for name in required_text:
            value = batch.get(name)
            if not isinstance(value, Sequence) or len(value) != batch_size:
                raise ValueError(f"batch metadata {name!r} does not match score count")
        labels = torch.as_tensor(batch["label"]).flatten().cpu()
        if labels.numel() != batch_size:
            raise ValueError("batch labels do not match score count")

        support = output.environment_support_score.detach().float().cpu()
        transport = output.environment_transport_score.detach().float().cpu()
        environment = None if output.environment_score is None else output.environment_score.detach().float().cpu()
        entropy = output.mode_entropy.detach().float().cpu()
        mode_ids = output.canonicalization.assignment.top_indices.detach().cpu()
        mode_weights = output.canonicalization.assignment.top_weights.detach().float().cpu()
        in_support = output.canonicalization.assignment.in_support.detach().cpu()
        decisions = None if defect_threshold is None else scores > defect_threshold

        # image_score 是统一评估器读取的字段；它与 defect_score 完全相同，不是第二种评分。
        for row_index in range(batch_size):
            row = {
                "sample_index": self.sample_count + row_index,
                "image_path": str(batch["path"][row_index]),  # type: ignore[index]
                "base_id": str(batch["base_id"][row_index]),  # type: ignore[index]
                "category": str(batch["category"][row_index]),  # type: ignore[index]
                "dataset": str(batch["dataset"][row_index]),  # type: ignore[index]
                "domain": str(batch["domain"][row_index]),  # type: ignore[index]
                "variant_id": str(batch["variant_id"][row_index]),  # type: ignore[index]
                "label": int(labels[row_index].item()),
                "image_score": float(scores[row_index].item()),
                "defect_score": float(scores[row_index].item()),
                "environment_support_score": float(support[row_index].item()),
                "environment_transport_score": float(transport[row_index].item()),
                "environment_score": None if environment is None else float(environment[row_index].item()),
                "in_environment_support": bool(in_support[row_index].item()),
                "mode_entropy": float(entropy[row_index].item()),
                "top_mode_ids": [int(item) for item in mode_ids[row_index].tolist()],
                "top_mode_weights": [float(item) for item in mode_weights[row_index].tolist()],
                "threshold": defect_threshold,
                "predicted_anomaly": None if decisions is None else bool(decisions[row_index].item()),
                "map_file": f"maps/{shard_name}",
                "map_index": row_index,
            }
            self.handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

        self.statistics["defect"].update(scores)
        self.statistics["support"].update(support)
        self.statistics["transport"].update(transport)
        if environment is not None:
            self.statistics["environment"].update(environment)
        self.statistics["anomaly_map"].update(maps)
        self.statistics["entropy"].update(entropy)
        self.sample_count += batch_size
        self.batch_index += 1

    def finalize(self, run_summary: Mapping[str, object]) -> Path:
        """关闭 JSONL 并写入包含中间统计摘要的 ``inference_summary.json``。"""

        if self.closed or self.sample_count == 0:
            raise RuntimeError("cannot finalize a closed or empty prediction writer")
        self.handle.flush()
        self.handle.close()
        self.temporary_path.replace(self.predictions_path)
        self.closed = True
        summary = {
            **dict(run_summary),
            "artifacts": {
                "predictions_jsonl": str(self.predictions_path),
                "predictions_sha256": file_digest(self.predictions_path),
                "maps_directory": str(self.maps_directory),
                "map_shards": self.batch_index,
                "map_key": "anomaly_map",
            },
            "statistics": {name: accumulator.summary() for name, accumulator in self.statistics.items()},
            "samples": self.sample_count,
        }
        return atomic_write_json(summary, self.output_directory / "inference_summary.json")

    def abort(self) -> None:
        """异常时关闭临时句柄；保留现场供定位，不伪造完整结果。"""

        if not self.closed:
            self.handle.close()
            self.closed = True
