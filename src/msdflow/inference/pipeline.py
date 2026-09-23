"""MSD-Flow 的最小推理闭环：候选环境运输、正常性积分和单一缺陷分数。

输入：冻结特征 ``[B,C,H,W]`` 和标准化环境向量 ``[B,D]``。
输出：一个图像级缺陷分数、一个像素异常图，以及环境机制诊断量。
中间产物：top-M 规范化特征和各模式候选异常图；只保存在内存中，不落盘。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from msdflow.conditions import EnvironmentDescriptorOutput
from msdflow.models import CanonicalTransportOutput, EmpiricalScoreCalibrator, MSDFlowModel

from .solver import ModeEulerSolveOutput, ModeNormalityEulerSolver


def _tensor_stats(value: Tensor) -> dict[str, object]:
    detached = value.detach().float()
    return {
        "shape": list(detached.shape),
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "finite": bool(torch.isfinite(detached).all().item()),
    }


@dataclass(slots=True)
class MSDFlowInferenceOutput:
    """最终论文输出和定位故障所需的最少中间张量。"""

    defect_score: Tensor
    anomaly_map: Tensor
    candidate_anomaly_maps: Tensor
    environment_support_score: Tensor
    environment_transport_score: Tensor
    environment_score: Tensor | None
    mode_entropy: Tensor
    environment_descriptor: Tensor
    descriptor_details: EnvironmentDescriptorOutput | None
    canonicalization: CanonicalTransportOutput
    normality_solution: ModeEulerSolveOutput

    def summary(self) -> dict[str, object]:
        return {
            "defect_score": _tensor_stats(self.defect_score),
            "anomaly_map": _tensor_stats(self.anomaly_map),
            "environment_support_score": _tensor_stats(self.environment_support_score),
            "environment_transport_score": _tensor_stats(self.environment_transport_score),
            "environment_score": None if self.environment_score is None else _tensor_stats(self.environment_score),
            "mode_entropy": _tensor_stats(self.mode_entropy),
            "environment_descriptor": _tensor_stats(self.environment_descriptor),
            "canonicalization": self.canonicalization.summary(),
            "normality_solver": self.normality_solution.summary(),
        }


class MSDFlowInferencePipeline:
    """运行冻结 MSD-Flow；评分公式不依赖 WT-Flow 的 NumPy 实现。"""

    def __init__(
        self,
        model: MSDFlowModel,
        *,
        top_m: int = 2,
        environment_steps: int = 4,
        normality_steps: int = 20,
        softmin_temperature: float = 0.1,
        output_size: int | tuple[int, int] = 256,
        top_fraction: float = 0.03,
        environment_calibrator: EmpiricalScoreCalibrator | None = None,
    ) -> None:
        if top_m <= 0 or top_m > model.mode_bank.n_components:
            raise ValueError("top_m exceeds available environment modes")
        if environment_steps <= 0 or normality_steps <= 0 or softmin_temperature <= 0:
            raise ValueError("flow steps and softmin temperature must be positive")
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if len(output_size) != 2 or min(output_size) <= 0:
            raise ValueError("output_size must be a positive integer or (height,width)")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must lie in (0,1]")
        self.model = model
        self.top_m = int(top_m)
        self.environment_steps = int(environment_steps)
        self.softmin_temperature = float(softmin_temperature)
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.top_fraction = float(top_fraction)
        self.top_k = max(1, round(self.output_size[0] * self.output_size[1] * self.top_fraction))
        self.normality_solver = ModeNormalityEulerSolver(normality_steps)
        self.environment_calibrator = environment_calibrator

    def _image_score(self, anomaly_map: Tensor) -> Tensor:
        """取异常图最高 ``top_fraction`` 像素均值，完全在当前设备上计算。"""

        values = anomaly_map.flatten(1).topk(self.top_k, dim=1).values
        return values.mean(1)

    @torch.inference_mode()
    def predict_features(
        self,
        feature: Tensor,
        standardized_environment: Tensor,
        *,
        descriptor_details: EnvironmentDescriptorOutput | None = None,
    ) -> MSDFlowInferenceOutput:
        """从缓存特征执行主推理路径。"""

        self.model.eval()
        # 步骤 1：把每个测试特征分别运输到后验最大的 top-M 正常环境中心。
        canonical = self.model.canonicalize(
            feature, standardized_environment, top_m=self.top_m, steps=self.environment_steps
        )
        batch, modes, channels, height, width = canonical.canonical_features.shape
        flat_feature = canonical.canonical_features.reshape(batch * modes, channels, height, width)
        flat_mode_ids = canonical.assignment.top_indices.reshape(-1)
        flat_mode_weights = F.one_hot(flat_mode_ids, num_classes=self.model.mode_bank.n_components).to(flat_feature)

        # 步骤 2：每个候选经模式条件正常性流映射到高斯端点。
        solution = self.normality_solver.solve(self.model.normality_flow, flat_feature, flat_mode_weights)
        # 高斯端点的逐 patch 能量越大，说明该位置越难由正常分布解释。
        candidate_energy = 0.5 * solution.final_state.square().mean(1, keepdim=True)
        candidate_energy = F.interpolate(
            candidate_energy, size=self.output_size, mode="bilinear", align_corners=False
        ).squeeze(1)
        candidate_maps = candidate_energy.reshape(batch, modes, *self.output_size)

        # 步骤 3：按 top-M 后验做 soft-min，只输出一个最终异常图和一个图像分数。
        weights = canonical.assignment.top_weights.to(candidate_maps)
        anomaly_map = self.model.softmin_defect_map(
            candidate_maps, weights, temperature=self.softmin_temperature
        )
        defect_score = self._image_score(anomaly_map)

        # 步骤 4：环境分数只解释域移强度，不参与缺陷排序。
        transport_score = (weights * canonical.transport_magnitude).sum(1)
        support_score = canonical.assignment.support_score
        environment_score = None
        if self.environment_calibrator is not None:
            environment_score = self.environment_calibrator(support_score, transport_score)
        posterior = canonical.assignment.posterior.clamp_min(1e-12)
        mode_entropy = -(posterior * posterior.log()).sum(1)
        return MSDFlowInferenceOutput(
            defect_score=defect_score,
            anomaly_map=anomaly_map,
            candidate_anomaly_maps=candidate_maps,
            environment_support_score=support_score,
            environment_transport_score=transport_score,
            environment_score=environment_score,
            mode_entropy=mode_entropy,
            environment_descriptor=standardized_environment,
            descriptor_details=descriptor_details,
            canonicalization=canonical,
            normality_solution=solution,
        )

    @torch.inference_mode()
    def predict_images(
        self,
        normalized_image: Tensor,
        raw_rgb: Tensor,
        *,
        backbone: nn.Module,
        feature_preprocessor: nn.Module,
    ) -> MSDFlowInferenceOutput:
        """从数据层的 ImageNet 图像和原始 RGB 图像直接执行推理。"""

        backbone.eval().requires_grad_(False)
        feature_preprocessor.eval().requires_grad_(False)
        feature = feature_preprocessor(backbone(normalized_image))
        if not isinstance(feature, Tensor):
            raise TypeError("feature preprocessor must return one tensor")
        details, environment = self.model.encode_environment(raw_rgb, detach_learned=True)
        return self.predict_features(feature, environment, descriptor_details=details)
