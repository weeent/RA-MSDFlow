"""为同一正常原图生成空间严格对齐的环境视图对。

输入：EnvFM ``SampleRecord`` 清单中的正常 train/val 记录。
输出：``PairedEnvironmentDataset``，每项包含 A/B 两个环境视图、八维光度描述、
已知增强参数和可追溯的 ``base_id``。两端共享同一原图和几何尺寸，只改变光度。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from envfm.data.dataset import IndustrialAnomalyDataset
from envfm.data.photometric_augment import (
    PhotometricParams,
    sample_photometric_params,
    stable_seed,
)
from envfm.data.photometric_descriptor import PhotometricDescriptor
from envfm.data.records import SampleRecord


SUPPORTED_VARIANTS = ("clean", "exposure", "white_balance", "gradient", "compound")


def photometric_parameter_vector(params: PhotometricParams | Mapping[str, object]) -> Tensor:
    """把增强参数转换成适合监督低频环境编码器的五维目标。

    五维依次为曝光 EV、log(R/G)、log(B/G)、横向光照梯度和纵向光照梯度。
    使用对数颜色增益可使乘性白平衡变化转为近似线性变化。
    """

    value = params if isinstance(params, PhotometricParams) else PhotometricParams.from_mapping(params)
    if min(value.red_gain, value.green_gain, value.blue_gain) <= 0:
        raise ValueError("RGB gains must be positive")
    return torch.tensor(
        [
            value.exposure_ev,
            math.log(value.red_gain / value.green_gain),
            math.log(value.blue_gain / value.green_gain),
            value.gradient_x,
            value.gradient_y,
        ],
        dtype=torch.float32,
    )


@dataclass(frozen=True, slots=True)
class EnvironmentPairRecord:
    """一对共享原图的环境视图及其可审计标识。"""

    source: SampleRecord
    target: SampleRecord
    pair_id: str

    def __post_init__(self) -> None:
        if self.source.base_id != self.target.base_id:
            raise ValueError("paired views must originate from the same base_id")
        if self.source.label != 0 or self.target.label != 0:
            raise ValueError("environment transport may only use normal samples")
        if self.source.split != self.target.split:
            raise ValueError("paired views must stay in the same experiment split")


def _variant_record(record: SampleRecord, variant: str, seed: int) -> SampleRecord:
    """创建可重放的光度视图，不改写输入记录。"""

    params = sample_photometric_params(variant, seed)
    return record.with_updates(
        variant_id=variant,
        augmentation_seed=seed,
        augmentation=params.to_dict(),
        metadata={**record.metadata, "msdflow_pair_variant": variant},
    )


def build_environment_pairs(
    records: Sequence[SampleRecord],
    *,
    target_variants: Sequence[str] = ("exposure", "white_balance", "gradient", "compound"),
    base_seed: int = 9826,
    include_reverse: bool = False,
) -> list[EnvironmentPairRecord]:
    """从正常记录构造 ``clean -> shifted`` 配对清单。

    异常记录会直接拒绝，而不是静默过滤，避免训练协议被上游清单错误污染。
    对每个 ``base_id`` 和目标环境使用稳定哈希种子，因此跨平台、跨进程可精确重放。
    """

    if not records:
        raise ValueError("at least one base record is required")
    invalid = [record.base_id for record in records if record.label != 0]
    if invalid:
        raise ValueError(f"environment pairs require normal-only records; found {len(invalid)} invalid records")
    unknown = set(target_variants).difference(SUPPORTED_VARIANTS)
    if unknown:
        raise ValueError(f"unsupported photometric variants: {sorted(unknown)}")
    if not target_variants:
        raise ValueError("target_variants must not be empty")

    pairs: list[EnvironmentPairRecord] = []
    for record in records:
        clean_seed = stable_seed(base_seed, record.base_id, "clean")
        clean = _variant_record(record, "clean", clean_seed)
        for variant in target_variants:
            target_seed = stable_seed(base_seed, record.base_id, variant)
            target = _variant_record(record, variant, target_seed)
            pair_id = f"{record.base_id}::clean->{variant}::{target_seed}"
            pairs.append(EnvironmentPairRecord(clean, target, pair_id))
            if include_reverse:
                pairs.append(EnvironmentPairRecord(target, clean, pair_id + "::reverse"))
    return pairs


class PairedEnvironmentDataset(Dataset[dict[str, object]]):
    """加载环境视图对，并保留 backbone 输入和原始 sRGB 两套张量。

    ``image_a/image_b`` 已按 ImageNet 归一化，可送入冻结 backbone；
    ``image_raw_a/image_raw_b`` 仍在 [0,1]，用于环境描述器；
    ``condition8_*`` 是现有 EnvFM 八维光度统计；``augmentation_*`` 是五维已知参数。
    """

    def __init__(
        self,
        pairs: Sequence[EnvironmentPairRecord],
        *,
        image_size: int | tuple[int, int] = 256,
        descriptor: PhotometricDescriptor | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("PairedEnvironmentDataset requires at least one pair")
        self.pairs = list(pairs)
        descriptor = descriptor if descriptor is not None else PhotometricDescriptor()
        # 分别建立两个 EnvFM 数据集，确保图像读取、增强和 ImageNet 归一化与旧实验一致。
        self._source = IndustrialAnomalyDataset(
            [pair.source for pair in pairs], image_size=image_size, descriptor=descriptor
        )
        self._target = IndustrialAnomalyDataset(
            [pair.target for pair in pairs], image_size=image_size, descriptor=descriptor
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        source = self._source[index]
        target = self._target[index]
        # 只复制模型需要的字段，避免 DataLoader 拼接不必要的 mask 和路径结构。
        return {
            "image_a": source["image"],
            "image_b": target["image"],
            "image_raw_a": source["image_raw"],
            "image_raw_b": target["image_raw"],
            "condition8_a": source["photo_condition_raw"],
            "condition8_b": target["photo_condition_raw"],
            "augmentation_a": photometric_parameter_vector(pair.source.augmentation),
            "augmentation_b": photometric_parameter_vector(pair.target.augmentation),
            "label_a": torch.tensor(0, dtype=torch.long),
            "label_b": torch.tensor(0, dtype=torch.long),
            "base_id": pair.source.base_id,
            "pair_id": pair.pair_id,
            "variant_a": pair.source.variant_id,
            "variant_b": pair.target.variant_id,
            "dataset": pair.source.dataset,
            "category": pair.source.category,
            "split": pair.source.split,
        }

    def summary(self) -> dict[str, object]:
        """提供轻量统计摘要，便于检查生成数据。"""

        by_target: dict[str, int] = {}
        for pair in self.pairs:
            by_target[pair.target.variant_id] = by_target.get(pair.target.variant_id, 0) + 1
        return {
            "pair_count": len(self.pairs),
            "unique_base_ids": len({pair.source.base_id for pair in self.pairs}),
            "target_variants": dict(sorted(by_target.items())),
            "all_normal": all(pair.source.label == pair.target.label == 0 for pair in self.pairs),
            "all_spatially_paired": all(pair.source.image_path == pair.target.image_path for pair in self.pairs),
        }
