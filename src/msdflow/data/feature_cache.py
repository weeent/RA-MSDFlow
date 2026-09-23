"""将冻结 backbone 特征写成小型分片，避免重复编码和单文件内存峰值。

输入：``PairedEnvironmentDataset`` 的 batch、冻结特征提取器和混合环境描述器。
输出：若干 ``shard_XXXXXX.pt`` 与一个 ``index.json``。训练时通过
``CachedPairedFeatureDataset`` 按样本读取，原始图像无需再次进入 GPU。
"""

from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from envfm.training.checkpoint import atomic_torch_save, atomic_write_json


TENSOR_FIELDS = (
    "feature_a",
    "feature_b",
    "environment_a",
    "environment_b",
    "condition8_a",
    "condition8_b",
    "augmentation_a",
    "augmentation_b",
    "label_a",
    "label_b",
)
TEXT_FIELDS = ("base_id", "pair_id", "variant_a", "variant_b", "dataset", "category", "split")


def _cpu_tensor(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu").contiguous()


class PairedFeatureCacheWriter:
    """逐 batch 写分片，并在结束时发布可审计索引。"""

    def __init__(self, output_directory: str | Path, *, metadata: Mapping[str, object] | None = None) -> None:
        self.output_directory = Path(output_directory)
        self.index_path = self.output_directory / "index.json"
        if self.index_path.exists():
            raise FileExistsError(f"feature cache already exists: {self.index_path}")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        self.shards: list[dict[str, object]] = []
        self.total_samples = 0

    def write(self, batch: Mapping[str, object]) -> Path:
        """校验字段并原子写入一个分片；所有张量在磁盘上保持 CPU 格式。"""

        missing = set(TENSOR_FIELDS + TEXT_FIELDS).difference(batch)
        if missing:
            raise KeyError(f"cache batch is missing fields: {sorted(missing)}")
        tensors = {name: _cpu_tensor(batch[name]) for name in TENSOR_FIELDS}  # type: ignore[arg-type]
        batch_size = tensors["feature_a"].shape[0]
        if batch_size <= 0 or any(value.shape[0] != batch_size for value in tensors.values()):
            raise ValueError("all tensor cache fields must share a non-empty batch dimension")
        texts: dict[str, list[str]] = {}
        for name in TEXT_FIELDS:
            values = [str(item) for item in batch[name]]  # type: ignore[union-attr]
            if len(values) != batch_size:
                raise ValueError(f"text field {name!r} does not match batch size")
            texts[name] = values
        shard_id = len(self.shards)
        path = self.output_directory / f"shard_{shard_id:06d}.pt"
        atomic_torch_save({"format_version": 1, "tensors": tensors, "texts": texts}, path)
        self.shards.append({"file": path.name, "samples": batch_size, "start": self.total_samples})
        self.total_samples += batch_size
        return path

    def finalize(self) -> Path:
        if not self.shards:
            raise RuntimeError("cannot finalize an empty feature cache")
        index = {
            "format_version": 1,
            "samples": self.total_samples,
            "shards": self.shards,
            "tensor_fields": list(TENSOR_FIELDS),
            "text_fields": list(TEXT_FIELDS),
            "metadata": self.metadata,
        }
        return atomic_write_json(index, self.index_path)


@torch.no_grad()
def build_paired_feature_cache(
    loader: Iterable[Mapping[str, object]],
    feature_extractor: nn.Module | Callable[[Tensor], Tensor],
    descriptor: nn.Module,
    output_directory: str | Path,
    *,
    feature_preprocessor: nn.Module | Callable[[object], Tensor] | None = None,
    device: str | torch.device = "cpu",
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """运行冻结编码器并建立配对特征缓存。

    ``feature_extractor`` 可返回单个特征张量，也可返回由现有
    ``WTFeaturePreprocessor`` 接收的多层特征。descriptor 的返回值可为张量，
    或含 ``combined`` 属性的 ``EnvironmentDescriptorOutput``。
    """

    run_device = torch.device(device)
    if isinstance(feature_extractor, nn.Module):
        feature_extractor.eval()
        feature_extractor.requires_grad_(False)
        feature_extractor.to(run_device)
    if isinstance(descriptor, nn.Module):
        descriptor.eval().to(run_device)
    if isinstance(feature_preprocessor, nn.Module):
        feature_preprocessor.eval().to(run_device)
    writer = PairedFeatureCacheWriter(output_directory, metadata=metadata)

    for raw_batch in loader:
        image_a = raw_batch["image_a"].to(run_device)  # type: ignore[union-attr]
        image_b = raw_batch["image_b"].to(run_device)  # type: ignore[union-attr]
        raw_a = raw_batch["image_raw_a"].to(run_device)  # type: ignore[union-attr]
        raw_b = raw_batch["image_raw_b"].to(run_device)  # type: ignore[union-attr]

        # 步骤 1：冻结 backbone 编码 A/B；同一个模型用于两端，避免引入伪域差异。
        encoded_a = feature_extractor(image_a)
        encoded_b = feature_extractor(image_b)
        feature_a = feature_preprocessor(encoded_a) if feature_preprocessor is not None else encoded_a
        feature_b = feature_preprocessor(encoded_b) if feature_preprocessor is not None else encoded_b
        if not isinstance(feature_a, Tensor) or not isinstance(feature_b, Tensor):
            raise TypeError("processed backbone output must be a tensor")

        # 步骤 2：环境描述来自未做 ImageNet 归一化的 sRGB，防止颜色语义被破坏。
        descriptor_a = descriptor(raw_a)
        descriptor_b = descriptor(raw_b)
        environment_a = getattr(descriptor_a, "combined", descriptor_a)
        environment_b = getattr(descriptor_b, "combined", descriptor_b)
        writer.write(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "environment_a": environment_a,
                "environment_b": environment_b,
                "condition8_a": raw_batch["condition8_a"],
                "condition8_b": raw_batch["condition8_b"],
                "augmentation_a": raw_batch["augmentation_a"],
                "augmentation_b": raw_batch["augmentation_b"],
                "label_a": raw_batch["label_a"],
                "label_b": raw_batch["label_b"],
                **{name: raw_batch[name] for name in TEXT_FIELDS},
            }
        )
    return writer.finalize()


class CachedPairedFeatureDataset(Dataset[dict[str, object]]):
    """读取分片缓存；每个 worker 只保留最近访问的一个分片。"""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        with self.index_path.open("r", encoding="utf-8") as handle:
            self.index = json.load(handle)
        if int(self.index.get("format_version", 0)) != 1:
            raise ValueError("unsupported MSD-Flow feature cache version")
        self.shards = list(self.index["shards"])
        self.starts = [int(shard["start"]) for shard in self.shards]
        self.samples = int(self.index["samples"])
        self._cached_shard_id = -1
        self._cached_payload: dict[str, object] | None = None

    def __len__(self) -> int:
        return self.samples

    def _load_shard(self, shard_id: int) -> dict[str, object]:
        if shard_id != self._cached_shard_id:
            path = self.index_path.parent / str(self.shards[shard_id]["file"])
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict) or payload.get("format_version") != 1:
                raise ValueError(f"invalid feature shard: {path}")
            self._cached_payload = payload
            self._cached_shard_id = shard_id
        assert self._cached_payload is not None
        return self._cached_payload

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += self.samples
        if index < 0 or index >= self.samples:
            raise IndexError(index)
        shard_id = bisect_right(self.starts, index) - 1
        local_index = index - self.starts[shard_id]
        payload = self._load_shard(shard_id)
        tensors = payload["tensors"]
        texts = payload["texts"]
        return {
            **{name: tensors[name][local_index] for name in TENSOR_FIELDS},  # type: ignore[index]
            **{name: texts[name][local_index] for name in TEXT_FIELDS},  # type: ignore[index]
        }

    def summary(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "shard_count": len(self.shards),
            "metadata": self.index.get("metadata", {}),
            "tensor_fields": self.index.get("tensor_fields", []),
        }
