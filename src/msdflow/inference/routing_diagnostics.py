"""MSD-Flow 的冻结模式路由诊断。

输入：冻结 WRN 特征 ``[B,C,H,W]``、标准化环境描述 ``[B,D]`` 和一个由
normal-train 确定的 clean anchor 模式。
输出：全部 K 个运输/不运输候选分数、五种冻结策略分数，以及 Photo-8 与
learned descriptor 的 Mahalanobis 距离分解。

本模块只做推理，不包含优化器、反向传播或参数更新。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from msdflow.conditions import EnvironmentModeAssignment
from msdflow.models import MSDFlowModel

from .pipeline import MSDFlowInferencePipeline
from .solver import ModeNormalityEulerSolver


ROUTING_POLICIES = (
    "posterior_top2",
    "clean_anchor",
    "uniform_softmin_all4",
    "image_min_all4",
    "raw_no_transport_min4",
)


def decompose_environment_distance(model: MSDFlowModel, environment: Tensor) -> tuple[Tensor, Tensor]:
    """在完整描述的 top-1 GMM 分量下拆分 Photo-8 和 learned 维距离。

    ``environment`` 必须已经使用 inference bundle 内的 train-only 统计标准化。
    返回的两项都是逐样本平方 Mahalanobis 项之和，不包含混合权重或 log-det。
    """

    if environment.ndim != 2 or environment.shape[1] != model.mode_bank.dimension:
        raise ValueError("standardized environment has the wrong shape")
    posterior = model.mode_bank.posterior(environment)
    mode_ids = posterior.argmax(1)
    centers = model.mode_bank.centers[mode_ids].to(environment)
    variances = model.mode_bank.variances[mode_ids].to(environment).clamp_min(
        model.mode_bank.min_variance
    )
    terms = (environment - centers).square() / variances
    photo_dimensions = min(8, terms.shape[1])
    return terms[:, :photo_dimensions].sum(1), terms[:, photo_dimensions:].sum(1)


def _stats(value: Tensor) -> dict[str, object]:
    detached = value.detach().float()
    return {
        "shape": list(detached.shape),
        "mean": float(detached.mean().item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "finite": bool(torch.isfinite(detached).all().item()),
    }


@dataclass(slots=True)
class RoutingDiagnosticOutput:
    """一批样本的完整诊断结果；候选图只在内存中保留。"""

    candidate_anomaly_maps: Tensor  # [B,K,H,W]，模式顺序固定为 0..K-1
    raw_candidate_anomaly_maps: Tensor  # `RoutingDiagnosticOutput` 的实现说明。
    candidate_scores: Tensor  # `RoutingDiagnosticOutput` 的实现说明。
    raw_candidate_scores: Tensor  # `RoutingDiagnosticOutput` 的实现说明。
    per_mode_transport_magnitude: Tensor  # `RoutingDiagnosticOutput` 的实现说明。
    policy_scores: Mapping[str, Tensor]  # 每项 [B]
    policy_anomaly_maps: Mapping[str, Tensor]  # 每项 [B,H,W]
    assignment: EnvironmentModeAssignment
    photo_distance: Tensor  # `RoutingDiagnosticOutput` 的实现说明。
    learned_distance: Tensor  # `RoutingDiagnosticOutput` 的实现说明。

    def summary(self) -> dict[str, object]:
        return {
            "candidate_scores": _stats(self.candidate_scores),
            "raw_candidate_scores": _stats(self.raw_candidate_scores),
            "per_mode_transport_magnitude": _stats(self.per_mode_transport_magnitude),
            "policy_scores": {name: _stats(value) for name, value in self.policy_scores.items()},
            "photo_distance": _stats(self.photo_distance),
            "learned_distance": _stats(self.learned_distance),
            "assignment": self.assignment.summary(),
        }


class RoutingDiagnosticPipeline:
    """用同一冻结 checkpoint 比较 v3 规定的五种模式路由策略。"""

    def __init__(
        self,
        model: MSDFlowModel,
        *,
        clean_anchor_mode: int,
        environment_steps: int = 4,
        normality_steps: int = 20,
        softmin_temperature: float = 0.1,
        output_size: int | tuple[int, int] = 256,
        top_fraction: float = 0.03,
    ) -> None:
        mode_count = model.mode_bank.n_components
        # 历史 v3 报告的三个策略名保留 ``all4``，以免破坏既有产物；
        # 实际计算始终枚举 bundle 内的全部 K 个模式。正式 K=1 消融只复用
        # ``_normality_maps``/``_single_image_scores``，因此不再人为限制 K=4。
        if not 0 <= clean_anchor_mode < mode_count:
            raise ValueError("clean_anchor_mode is outside the fitted mode bank")
        if environment_steps <= 0 or normality_steps <= 0:
            raise ValueError("Euler step counts must be positive")
        if softmin_temperature <= 0 or not 0.0 < top_fraction <= 1.0:
            raise ValueError("temperature and top_fraction must be positive")
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if len(output_size) != 2 or min(output_size) <= 0:
            raise ValueError("output_size must be a positive integer or (height,width)")

        self.model = model
        self.clean_anchor_mode = int(clean_anchor_mode)
        self.environment_steps = int(environment_steps)
        self.normality_solver = ModeNormalityEulerSolver(int(normality_steps))
        self.softmin_temperature = float(softmin_temperature)
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.top_fraction = float(top_fraction)
        self.top_k = max(1, round(self.output_size[0] * self.output_size[1] * self.top_fraction))

        # P0 直接调用正式 v2 pipeline，保证诊断自对照复用完全相同的实现路径。
        self.posterior_top2_pipeline = MSDFlowInferencePipeline(
            model,
            # K=1 时安全退化为 top-1；K>=2 保持已经验收的 v3 路径。
            top_m=min(2, mode_count),
            environment_steps=self.environment_steps,
            normality_steps=self.normality_solver.steps,
            softmin_temperature=self.softmin_temperature,
            output_size=self.output_size,
            top_fraction=self.top_fraction,
        )

    def _image_scores(self, maps: Tensor) -> Tensor:
        """对 ``[B,K,H,W]`` 候选图逐模式计算最高像素 top-fraction 均值。"""

        if maps.ndim != 4:
            raise ValueError("candidate anomaly maps must be [B,K,H,W]")
        return maps.flatten(2).topk(self.top_k, dim=2).values.mean(2)

    def _single_image_scores(self, maps: Tensor) -> Tensor:
        if maps.ndim != 3:
            raise ValueError("policy anomaly maps must be [B,H,W]")
        return maps.flatten(1).topk(self.top_k, dim=1).values.mean(1)

    def _normality_maps(self, features: Tensor) -> Tensor:
        """按固定模式 ID 顺序将 ``[B,K,C,H,W]`` 映射为候选异常图。"""

        if features.ndim != 5:
            raise ValueError("mode-expanded features must be [B,K,C,H,W]")
        batch, modes, channels, height, width = features.shape
        if modes != self.model.mode_bank.n_components:
            raise ValueError("expanded feature mode count does not match mode bank")
        flat = features.reshape(batch * modes, channels, height, width)
        mode_ids = torch.arange(modes, device=flat.device).repeat(batch)
        one_hot = F.one_hot(mode_ids, num_classes=modes).to(flat)
        solution = self.normality_solver.solve(self.model.normality_flow, flat, one_hot)
        energy = 0.5 * solution.final_state.square().mean(1, keepdim=True)
        energy = F.interpolate(
            energy, size=self.output_size, mode="bilinear", align_corners=False
        ).squeeze(1)
        return energy.reshape(batch, modes, *self.output_size)

    def _distance_decomposition(self, environment: Tensor) -> tuple[Tensor, Tensor]:
        """在完整 24D top-1 模式下分解 Photo-8/learned 维平方距离。"""

        return decompose_environment_distance(self.model, environment)

    @torch.inference_mode()
    def predict_features(self, feature: Tensor, standardized_environment: Tensor) -> RoutingDiagnosticOutput:
        """执行五种冻结策略；输入与模型必须在同一设备。"""

        self.model.eval()
        if feature.ndim != 4 or standardized_environment.ndim != 2:
            raise ValueError("feature/environment must be [B,C,H,W] and [B,D]")
        if feature.shape[0] != standardized_environment.shape[0]:
            raise ValueError("feature and environment batch sizes differ")
        if feature.device != standardized_environment.device:
            raise ValueError("feature and environment must share a device")

        batch, channels, height, width = feature.shape
        mode_count = self.model.mode_bank.n_components

        # 步骤 1：复算正式 v2 P0 分数，后续由 CLI 与原预测逐样本核对。
        posterior_top2 = self.posterior_top2_pipeline.predict_features(
            feature, standardized_environment
        )

        # 步骤 2：不依据 posterior 截断，按 ID 0..K-1 枚举全部正常环境中心。
        assignment = self.model.mode_bank.assign(standardized_environment, top_m=mode_count)
        expanded_feature = feature[:, None].expand(-1, mode_count, -1, -1, -1).reshape(
            batch * mode_count, channels, height, width
        )
        expanded_source = standardized_environment[:, None].expand(-1, mode_count, -1).reshape(
            batch * mode_count, -1
        )
        targets = self.model.mode_bank.centers.to(feature)[None].expand(batch, -1, -1)
        transported = self.model.environment_flow.transport(
            expanded_feature,
            expanded_source,
            targets.reshape(batch * mode_count, -1),
            steps=self.environment_steps,
        )
        assert isinstance(transported, Tensor)
        transported = transported.reshape(batch, mode_count, channels, height, width)
        candidate_maps = self._normality_maps(transported)
        candidate_scores = self._image_scores(candidate_maps)

        # 步骤 3：P4 使用完全相同的模式正常性流，只删除环境运输这一因素。
        raw_expanded = feature[:, None].expand(-1, mode_count, -1, -1, -1)
        raw_maps = self._normality_maps(raw_expanded)
        raw_scores = self._image_scores(raw_maps)

        # 步骤 4：构造 P1--P4；P3/P4 选择整张图的一个模式，禁止逐像素拼接。
        anchor_map = candidate_maps[:, self.clean_anchor_mode]
        uniform_weights = feature.new_full((batch, mode_count), 1.0 / mode_count)
        uniform_map = self.model.softmin_defect_map(
            candidate_maps, uniform_weights, temperature=self.softmin_temperature
        )
        min_score, min_mode = candidate_scores.min(1)
        raw_min_score, raw_min_mode = raw_scores.min(1)
        row_ids = torch.arange(batch, device=feature.device)
        image_min_map = candidate_maps[row_ids, min_mode]
        raw_min_map = raw_maps[row_ids, raw_min_mode]
        policy_maps = {
            "posterior_top2": posterior_top2.anomaly_map,
            "clean_anchor": anchor_map,
            "uniform_softmin_all4": uniform_map,
            "image_min_all4": image_min_map,
            "raw_no_transport_min4": raw_min_map,
        }
        policy_scores = {
            "posterior_top2": posterior_top2.defect_score,
            "clean_anchor": candidate_scores[:, self.clean_anchor_mode],
            "uniform_softmin_all4": self._single_image_scores(uniform_map),
            "image_min_all4": min_score,
            "raw_no_transport_min4": raw_min_score,
        }

        # 步骤 5：记录逐模式运输量与描述距离，定位路由和表示层面的失败。
        transport_magnitude = (transported - feature[:, None]).square().mean((2, 3, 4)).sqrt()
        photo_distance, learned_distance = self._distance_decomposition(standardized_environment)
        diagnostic_tensors = {
            "candidate_maps": candidate_maps,
            "raw_candidate_maps": raw_maps,
            "candidate_scores": candidate_scores,
            "raw_candidate_scores": raw_scores,
            "transport_magnitude": transport_magnitude,
            "photo_distance": photo_distance,
            "learned_distance": learned_distance,
        }
        for name, value in diagnostic_tensors.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"diagnostic tensor {name!r} contains NaN or infinity")
        for name, score in policy_scores.items():
            if score.shape != (batch,) or not torch.isfinite(score).all():
                raise FloatingPointError(f"policy {name!r} produced invalid scores")
        return RoutingDiagnosticOutput(
            candidate_anomaly_maps=candidate_maps,
            raw_candidate_anomaly_maps=raw_maps,
            candidate_scores=candidate_scores,
            raw_candidate_scores=raw_scores,
            per_mode_transport_magnitude=transport_magnitude,
            policy_scores=policy_scores,
            policy_anomaly_maps=policy_maps,
            assignment=assignment,
            photo_distance=photo_distance,
            learned_distance=learned_distance,
        )
