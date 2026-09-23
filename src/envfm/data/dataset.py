"""RA-MSDFlow 的 dataset 模块。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .photometric_augment import apply_photometric_transform
from .photometric_descriptor import ConditionStandardizer, PhotometricDescriptor
from .records import SampleRecord, records_digest


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_RESAMPLING = getattr(Image, "Resampling", Image)


def _parse_image_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        return image_size, image_size
    if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size must be a positive int or (height, width)")
    return int(image_size[0]), int(image_size[1])


def _pil_to_rgb_tensor(path: str | Path, size: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    """执行 `_pil_to_rgb_tensor` 所需的处理。"""

    source = Path(path)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        original_size = (rgb.height, rgb.width)
        # 与 WT-Flow 参考实现保持一致。
        resized = rgb.resize((size[1], size[0]), resample=_RESAMPLING.BILINEAR)
        array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1), original_size


def _pil_to_mask_tensor(path: str | Path | None, size: tuple[int, int]) -> torch.Tensor:
    """执行 `_pil_to_mask_tensor` 所需的处理。"""

    if path is None:
        return torch.zeros((1, size[0], size[1]), dtype=torch.float32)
    with Image.open(path) as image:
        gray = image.convert("L")
        # `_pil_to_mask_tensor` 的实现说明。
        resized = gray.resize((size[1], size[0]), resample=_RESAMPLING.NEAREST)
        array = np.asarray(resized, dtype=np.uint8).copy()
    return torch.from_numpy((array > 0).astype(np.float32)).unsqueeze(0)


def normalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    """执行 `normalize_imagenet` 所需的处理。"""

    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image - mean) / std


class IndustrialAnomalyDataset(Dataset[dict[str, object]]):
    """`IndustrialAnomalyDataset` 组件。"""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        *,
        image_size: int | tuple[int, int] = 256,
        descriptor: PhotometricDescriptor | None = None,
        condition_standardizer: ConditionStandardizer | None = None,
        apply_augmentation: bool = True,
        normalize: bool = True,
    ) -> None:
        if not records:
            raise ValueError("IndustrialAnomalyDataset requires at least one record")
        self.records = list(records)
        self.image_size = _parse_image_size(image_size)
        self.descriptor = descriptor if descriptor is not None else PhotometricDescriptor()
        self.descriptor.eval()
        self.condition_standardizer = condition_standardizer
        self.apply_augmentation = bool(apply_augmentation)
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        # 按评估协议处理。
        image_raw, original_size = _pil_to_rgb_tensor(record.image_path, self.image_size)
        # 保持掩码与图像几何一致。
        if self.apply_augmentation:
            image_raw = apply_photometric_transform(image_raw, record.augmentation)
        # `__getitem__` 的实现说明。
        with torch.no_grad():
            condition_raw = self.descriptor(image_raw).squeeze(0).to(dtype=torch.float32)
        condition = (
            self.condition_standardizer.transform(condition_raw)
            if self.condition_standardizer is not None
            else condition_raw.clone()
        )
        # `__getitem__` 的实现说明。
        image = normalize_imagenet(image_raw) if self.normalize else image_raw.clone()
        # 保持掩码与图像几何一致。
        mask = _pil_to_mask_tensor(record.mask_path, self.image_size)
        return {
            "image": image,
            "image_raw": image_raw,
            "photo_condition": condition.to(dtype=torch.float32),
            "photo_condition_raw": condition_raw,
            "mask": mask,
            "has_mask": torch.tensor(record.mask_path is not None, dtype=torch.bool),
            "label": torch.tensor(record.label, dtype=torch.long),
            "path": record.image_path,
            "base_id": record.base_id,
            "category": record.category,
            "dataset": record.dataset,
            "domain": record.domain,
            "variant_id": record.variant_id,
            "original_size": torch.tensor(original_size, dtype=torch.long),
        }


@torch.no_grad()
def fit_condition_standardizer(
    dataset: IndustrialAnomalyDataset, *, only_normal_train: bool = True
) -> tuple[ConditionStandardizer, dict[str, object]]:
    """执行 `fit_condition_standardizer` 所需的处理。"""

    selected_indices = [
        index
        for index, record in enumerate(dataset.records)
        if (not only_normal_train) or (record.split == "train" and record.label == 0)
    ]
    if not selected_indices:
        raise ValueError("no normal training records were available to fit condition statistics")
    conditions: list[torch.Tensor] = []
    for index in selected_indices:
        # `fit_condition_standardizer` 的实现说明。
        conditions.append(dataset[index]["photo_condition_raw"].unsqueeze(0))  # type: ignore[index,union-attr]
    values = torch.cat(conditions, dim=0)
    standardizer = ConditionStandardizer().fit(values)
    summary = {
        "selected_records": len(selected_indices),
        "manifest_digest": records_digest([dataset.records[index] for index in selected_indices]),
        "raw_condition_mean": values.mean(dim=0).tolist(),
        "raw_condition_std": values.std(dim=0, unbiased=False).tolist(),
        "raw_condition_min": values.amin(dim=0).tolist(),
        "raw_condition_max": values.amax(dim=0).tolist(),
        "records_by_variant": dict(sorted(Counter(dataset.records[index].variant_id for index in selected_indices).items())),
    }
    return standardizer, summary
