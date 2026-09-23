"""按环境模式建模正常特征分布的 WT-Flow 兼容反向 FM。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from envfm.models.conditional_unet import ConditionalUNetOutput, EnvFMConditionalMiniUNet
from envfm.models.flow_path import ReversedFlowBatch, ReversedFlowMatching


@dataclass(slots=True)
class NormalityFlowOutput:
    loss: Tensor
    path: ReversedFlowBatch
    prediction: ConditionalUNetOutput

    def summary(self) -> dict[str, object]:
        return {
            "loss": float(self.loss.detach().item()),
            "path": self.path.summary(),
            "prediction": self.prediction.summary(),
        }


class ModeConditionedNormalityFlow(nn.Module):
    """以环境模式后验/one-hot 为条件学习 ``normal feature -> noise`` 速度。"""

    def __init__(
        self,
        in_channels: int,
        n_modes: int,
        *,
        base_channels: int = 128,
        condition_hidden_dim: int = 64,
        condition_dim: int | None = None,
    ) -> None:
        super().__init__()
        if n_modes <= 0:
            raise ValueError("n_modes must be positive")
        self.n_modes = int(n_modes)
        self.condition_dim = self.n_modes if condition_dim is None else int(condition_dim)
        if self.condition_dim < self.n_modes:
            raise ValueError("condition_dim cannot be smaller than n_modes")
        self.velocity = EnvFMConditionalMiniUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            condition_dim=self.condition_dim,
            condition_hidden_dim=condition_hidden_dim,
        )
        self.path = ReversedFlowMatching()

    def _validate_modes(self, mode_weights: Tensor, batch_size: int) -> None:
        if mode_weights.shape != (batch_size, self.n_modes):
            raise ValueError(f"mode_weights must be [B,{self.n_modes}]")
        if not torch.isfinite(mode_weights).all() or (mode_weights < 0).any():
            raise ValueError("mode weights must be finite and non-negative")
        if not torch.allclose(mode_weights.sum(1), torch.ones(batch_size, device=mode_weights.device), atol=1e-4):
            raise ValueError("each mode-weight row must sum to one")

    def _condition_input(self, mode_weights: Tensor, batch_size: int) -> Tensor:
        """校验 K 维模式权重，并按需右侧补零到固定条件宽度。

        K=1 消融可设置 ``condition_dim=4``，使条件 MLP 与 K=4 主模型具有
        完全相同的参数量；唯一有效模式仍由第一维 one-hot 表示。
        """

        self._validate_modes(mode_weights, batch_size)
        if self.condition_dim == self.n_modes:
            return mode_weights
        padding = mode_weights.new_zeros(batch_size, self.condition_dim - self.n_modes)
        return torch.cat((mode_weights, padding), dim=1)

    def forward(
        self,
        feature: Tensor,
        mode_weights: Tensor,
        *,
        time: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> NormalityFlowOutput:
        condition = self._condition_input(mode_weights, feature.shape[0])
        if time is None:
            time = torch.rand(feature.shape[0], device=feature.device, dtype=feature.dtype)
        path = self.path.sample(feature, time, noise=noise)
        prediction = self.velocity.forward_with_details(path.xt, path.time, condition)
        loss = self.path.mse_loss(prediction.velocity, path.target_velocity)
        return NormalityFlowOutput(loss=loss, path=path, prediction=prediction)

    def predict_velocity(self, state: Tensor, time: Tensor, mode_weights: Tensor) -> Tensor:
        condition = self._condition_input(mode_weights, state.shape[0])
        return self.velocity(state, time, condition)

    def summary(self) -> dict[str, object]:
        return {
            "model": "mode_conditioned_normality_flow",
            "n_modes": self.n_modes,
            "condition_dim": self.condition_dim,
            **self.velocity.summary(),
        }
