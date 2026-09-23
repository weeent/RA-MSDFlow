"""模式条件正常性 FM 的训练适配器。"""

from __future__ import annotations

from typing import Literal, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from msdflow.conditions import DiagonalGaussianModeBank, EnvironmentStandardizer
from msdflow.models import ModeConditionedNormalityFlow, PairedEnvironmentFlow

from .engine import StageTrainerBase


class NormalityFlowTrainer(StageTrainerBase):
    """在固定环境模式库上训练正常性流，不反向更新 GMM。"""

    stage_name = "mode_conditioned_normality_flow"

    model: ModeConditionedNormalityFlow

    def __init__(
        self,
        model: ModeConditionedNormalityFlow,
        *,
        mode_bank: DiagonalGaussianModeBank,
        environment_flow: PairedEnvironmentFlow | None = None,
        environment_standardizer: EnvironmentStandardizer | None = None,
        canonicalize_to_mode: bool = True,
        transport_steps: int = 4,
        endpoint_policy: Literal["target", "both"] = "target",
        **kwargs: object,
    ) -> None:
        if not mode_bank.fitted:
            raise RuntimeError("normality training requires a fitted train-only mode bank")
        if endpoint_policy not in {"target", "both"}:
            raise ValueError("endpoint_policy must be 'target' or 'both'")
        if canonicalize_to_mode and environment_flow is None:
            raise ValueError("canonicalize_to_mode=True requires a trained environment_flow")
        if transport_steps <= 0:
            raise ValueError("transport_steps must be positive")
        self.mode_bank = mode_bank
        self.environment_flow = environment_flow
        self.environment_standardizer = environment_standardizer
        self.canonicalize_to_mode = bool(canonicalize_to_mode)
        self.transport_steps = int(transport_steps)
        self.endpoint_policy = endpoint_policy
        super().__init__(model, **kwargs)
        self.mode_bank.to(self.device).eval().requires_grad_(False)
        if self.environment_flow is not None:
            self.environment_flow.to(self.device).eval().requires_grad_(False)
        if self.environment_standardizer is not None:
            if not self.environment_standardizer.fitted:
                raise RuntimeError("environment standardizer must be fitted on normal train data")
            self.environment_standardizer.to(self.device).eval().requires_grad_(False)

    def compute_loss(self, batch: Mapping[str, object], *, training: bool) -> tuple[Tensor, Mapping[str, float]]:
        label_a = batch["label_a"].to(self.device)  # type: ignore[union-attr]
        label_b = batch["label_b"].to(self.device)  # type: ignore[union-attr]
        self._assert_normal(label_a, "label_a")
        self._assert_normal(label_b, "label_b")
        feature_b = batch["feature_b"].to(self.device)  # type: ignore[union-attr]
        environment_b = batch["environment_b"].to(self.device)  # type: ignore[union-attr]
        if self.endpoint_policy == "both":
            feature = torch.cat((batch["feature_a"].to(self.device), feature_b), dim=0)  # type: ignore[union-attr]
            environment = torch.cat((batch["environment_a"].to(self.device), environment_b), dim=0)  # type: ignore[union-attr]
        else:
            # 默认只用 target，避免一个 base_id 的 clean 端因多个增强 variant 被重复计权。
            feature, environment = feature_b, environment_b
        if self.environment_standardizer is not None:
            environment = self.environment_standardizer(environment)
        with torch.no_grad():
            if self.canonicalize_to_mode:
                # 先分配最近模式，再用已冻结环境流把正常特征运输到该模式中心。
                assignment = self.mode_bank.assign(environment, top_m=1)
                mode_index = assignment.top_indices[:, 0]
                target_environment = self.mode_bank.centers[mode_index].to(feature)
                assert self.environment_flow is not None
                transported = self.environment_flow.transport(
                    feature,
                    environment,
                    target_environment,
                    steps=self.transport_steps,
                )
                assert isinstance(transported, Tensor)
                feature = transported
                mode_weights = F.one_hot(mode_index, num_classes=self.mode_bank.n_components).to(feature)
            else:
                # 此分支保留作“仅模式条件、不做显式运输”的关键自消融。
                mode_weights = self.mode_bank.posterior(environment)
        output = self.model(feature, mode_weights)
        entropy = -(mode_weights * mode_weights.clamp_min(1e-12).log()).sum(1).mean()
        return output.loss, {
            "mode_entropy": float(entropy.item()),
            "predicted_velocity_mean_abs": float(output.prediction.velocity.detach().abs().mean().item()),
            "target_velocity_mean_abs": float(output.path.target_velocity.detach().abs().mean().item()),
        }
