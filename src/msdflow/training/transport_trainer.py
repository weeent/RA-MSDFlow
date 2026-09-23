"""配对环境运输 FM 的训练适配器。"""

from __future__ import annotations

from typing import Mapping

from torch import Tensor

from msdflow.conditions import EnvironmentStandardizer
from msdflow.models import PairedEnvironmentFlow

from .engine import StageTrainerBase


class EnvironmentTransportTrainer(StageTrainerBase):
    """消费配对特征缓存，学习 ``z_A,e_A -> z_B,e_B`` 的平滑速度。"""

    stage_name = "paired_environment_flow"

    model: PairedEnvironmentFlow

    def __init__(
        self,
        model: PairedEnvironmentFlow,
        *,
        environment_standardizer: EnvironmentStandardizer | None = None,
        **kwargs: object,
    ) -> None:
        self.environment_standardizer = environment_standardizer
        super().__init__(model, **kwargs)
        if self.environment_standardizer is not None:
            if not self.environment_standardizer.fitted:
                raise RuntimeError("environment standardizer must be fitted on normal train data")
            self.environment_standardizer.to(self.device).eval().requires_grad_(False)

    def compute_loss(self, batch: Mapping[str, object], *, training: bool) -> tuple[Tensor, Mapping[str, float]]:
        labels_a = batch["label_a"].to(self.device)  # type: ignore[union-attr]
        labels_b = batch["label_b"].to(self.device)  # type: ignore[union-attr]
        self._assert_normal(labels_a, "label_a")
        self._assert_normal(labels_b, "label_b")
        environment_a = batch["environment_a"].to(self.device)  # type: ignore[union-attr]
        environment_b = batch["environment_b"].to(self.device)  # type: ignore[union-attr]
        # 缓存可保存原始描述或已标准化描述；传入 standardizer 时在训练边界统一转换。
        if self.environment_standardizer is not None:
            environment_a = self.environment_standardizer(environment_a)
            environment_b = self.environment_standardizer(environment_b)
        output = self.model(
            batch["feature_a"].to(self.device),  # type: ignore[union-attr]
            batch["feature_b"].to(self.device),  # type: ignore[union-attr]
            environment_a,
            environment_b,
        )
        return output.loss, {
            "target_velocity_mean_abs": float(output.path.target_velocity.detach().abs().mean().item()),
            "predicted_velocity_mean_abs": float(output.prediction.velocity.detach().abs().mean().item()),
            "spatial_residual_mean_abs": float(output.prediction.spatial_upsampled.detach().abs().mean().item()),
        }
