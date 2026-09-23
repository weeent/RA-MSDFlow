"""用已知光度增强参数预训练可学习低频环境编码器。"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from msdflow.conditions import LowFrequencyEnvironmentEncoder

from .engine import StageTrainerBase


class EnvironmentCodeRegressor(nn.Module):
    """低频编码器及五维增强参数预测头；训练后只保留 encoder 用于混合描述。"""

    def __init__(self, encoder: LowFrequencyEnvironmentEncoder, target_dim: int = 5) -> None:
        super().__init__()
        self.encoder = encoder
        self.regression_head = nn.Sequential(
            nn.LayerNorm(encoder.output_dim),
            nn.Linear(encoder.output_dim, max(encoder.output_dim, target_dim * 2)),
            nn.SiLU(),
            nn.Linear(max(encoder.output_dim, target_dim * 2), target_dim),
        )
        # 五个增强参数的自然范围不同；按采样上界归一化，避免曝光一维支配 MSE。
        if target_dim != 5:
            raise ValueError("current photometric supervision is fixed to five parameters")
        self.register_buffer("target_scale", torch.tensor([0.75, 0.18, 0.18, 0.35, 0.35]))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        code = self.encoder(image)
        return self.regression_head(code), code

    def normalize_target(self, target: Tensor) -> Tensor:
        return target / self.target_scale.to(target).clamp_min(1e-6)

    def denormalize_prediction(self, prediction: Tensor) -> Tensor:
        return prediction * self.target_scale.to(prediction)


class EnvironmentCodeTrainer(StageTrainerBase):
    """让低频编码对曝光、白平衡和光照梯度具有可辨识性。"""

    stage_name = "environment_code_pretraining"

    def compute_loss(self, batch: Mapping[str, object], *, training: bool) -> tuple[Tensor, Mapping[str, float]]:
        labels_a = batch["label_a"].to(self.device)  # type: ignore[union-attr]
        labels_b = batch["label_b"].to(self.device)  # type: ignore[union-attr]
        self._assert_normal(labels_a, "label_a")
        self._assert_normal(labels_b, "label_b")
        images = torch.cat(
            (batch["image_raw_a"].to(self.device), batch["image_raw_b"].to(self.device)), dim=0  # type: ignore[union-attr]
        )
        targets = torch.cat(
            (batch["augmentation_a"].to(self.device), batch["augmentation_b"].to(self.device)), dim=0  # type: ignore[union-attr]
        )
        prediction, code = self.model(images)
        normalized_target = self.model.normalize_target(targets)
        loss = F.mse_loss(prediction, normalized_target)
        raw_prediction = self.model.denormalize_prediction(prediction)
        return loss, {
            "parameter_mae": float((raw_prediction.detach() - targets).abs().mean().item()),
            "code_std": float(code.detach().float().std(unbiased=False).item()),
        }
