"""组合环境描述、模式发现、环境运输与正常性建模的统一外壳。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from msdflow.conditions import (
    DiagonalGaussianModeBank,
    EnvironmentDescriptorOutput,
    EnvironmentModeAssignment,
    EnvironmentStandardizer,
    HybridEnvironmentDescriptor,
)

from .environment_flow import PairedEnvironmentFlow
from .normality_flow import ModeConditionedNormalityFlow


@dataclass(slots=True)
class CanonicalTransportOutput:
    """测试特征向 top-M 正常环境中心运输后的完整结果。"""

    canonical_features: Tensor  # `CanonicalTransportOutput` 的实现说明。
    target_environments: Tensor  # `CanonicalTransportOutput` 的实现说明。
    assignment: EnvironmentModeAssignment
    transport_magnitude: Tensor  # `CanonicalTransportOutput` 的实现说明。

    def summary(self) -> dict[str, object]:
        return {
            "canonical_feature_shape": list(self.canonical_features.shape),
            "target_environment_shape": list(self.target_environments.shape),
            "transport_magnitude_mean": float(self.transport_magnitude.mean().item()),
            "assignment": self.assignment.summary(),
        }


class MSDFlowModel(nn.Module):
    """MSD-Flow 研究模型；backbone 保持在数据缓存阶段，不属于该对象。"""

    def __init__(
        self,
        descriptor: HybridEnvironmentDescriptor,
        standardizer: EnvironmentStandardizer,
        mode_bank: DiagonalGaussianModeBank,
        environment_flow: PairedEnvironmentFlow,
        normality_flow: ModeConditionedNormalityFlow,
    ) -> None:
        super().__init__()
        if descriptor.output_dim != standardizer.dimension or descriptor.output_dim != mode_bank.dimension:
            raise ValueError("descriptor, standardizer and mode-bank dimensions must agree")
        if mode_bank.n_components != normality_flow.n_modes:
            raise ValueError("mode-bank component count must equal normality-flow mode count")
        self.descriptor = descriptor
        self.standardizer = standardizer
        self.mode_bank = mode_bank
        self.environment_flow = environment_flow
        self.normality_flow = normality_flow

    def encode_environment(self, raw_rgb: Tensor, *, detach_learned: bool = True) -> tuple[EnvironmentDescriptorOutput, Tensor]:
        output = self.descriptor(raw_rgb, detach_learned=detach_learned)
        return output, self.standardizer(output.combined)

    def canonicalize(
        self,
        feature: Tensor,
        standardized_environment: Tensor,
        *,
        top_m: int = 2,
        steps: int = 4,
    ) -> CanonicalTransportOutput:
        """把每个样本分别运输到 posterior 最大的 top-M 个正常环境中心。"""

        assignment = self.mode_bank.assign(standardized_environment, top_m=top_m)
        batch, channels, height, width = feature.shape
        target = self.mode_bank.centers[assignment.top_indices].to(feature)
        expanded_feature = feature[:, None].expand(-1, top_m, -1, -1, -1).reshape(
            batch * top_m, channels, height, width
        )
        expanded_source = standardized_environment[:, None].expand(-1, top_m, -1).reshape(batch * top_m, -1)
        expanded_target = target.reshape(batch * top_m, -1)
        transported = self.environment_flow.transport(
            expanded_feature, expanded_source, expanded_target, steps=steps
        )
        assert isinstance(transported, Tensor)
        canonical = transported.reshape(batch, top_m, channels, height, width)
        magnitude = (canonical - feature[:, None]).square().mean(dim=(2, 3, 4)).sqrt()
        return CanonicalTransportOutput(canonical, target, assignment, magnitude)

    @staticmethod
    def softmin_defect_map(anomaly_maps: Tensor, mode_weights: Tensor, *, temperature: float = 0.1) -> Tensor:
        """按归一化模式后验对多个候选异常图做加权 soft-min。

        输入 ``anomaly_maps=[B,M,H,W]``、``mode_weights=[B,M]``；输出 ``[B,H,W]``。
        温度越小越接近逐像素最小值，温度较大时会保留其他可信模式的信息。
        """

        if anomaly_maps.ndim != 4 or mode_weights.shape != anomaly_maps.shape[:2]:
            raise ValueError("anomaly_maps and mode_weights have incompatible shapes")
        if temperature <= 0 or (mode_weights < 0).any():
            raise ValueError("temperature must be positive and weights non-negative")
        weights = mode_weights / mode_weights.sum(1, keepdim=True).clamp_min(1e-12)
        log_weight = torch.log(weights.clamp_min(1e-12))[:, :, None, None]
        return -temperature * torch.logsumexp(log_weight - anomaly_maps / temperature, dim=1)

    def summary(self) -> dict[str, object]:
        return {
            "model": "mode_aware_shift_defect_flow",
            "environment_dimension": self.descriptor.output_dim,
            "mode_count": self.mode_bank.n_components,
            "descriptor": {
                "fixed_dimension": 8,
                "learned_dimension": self.descriptor.learned_dim,
            },
            "standardizer": self.standardizer.summary(),
            "mode_bank": self.mode_bank.summary(),
            "environment_flow": self.environment_flow.velocity.summary(),
            "normality_flow": self.normality_flow.summary(),
        }
