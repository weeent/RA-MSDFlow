"""对已编码特征运行冻结 WT-Flow，用于 reference-alignment 消融。

输入是 WT-Flow 已完成平均池化和无参数 LayerNorm 的单分支特征 ``[B,C,H,W]``；
输出是连续图像异常分数与上采样异常图。模型参数在本模块中不会更新。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class WTFlowFeatureScoreOutput:
    """冻结 WT-Flow 特征评分的最小输出。"""

    image_scores: Tensor  # `WTFlowFeatureScoreOutput` 的实现说明。
    anomaly_maps: Tensor  # `WTFlowFeatureScoreOutput` 的实现说明。


class FrozenWTFlowFeatureScorer:
    """复现作者 ``generate_pdf`` 与 ``post_process_single`` 的特征后半段。"""

    def __init__(
        self,
        model: nn.Module,
        *,
        steps: int = 1,
        output_size: int | tuple[int, int] = 256,
        top_fraction: float = 0.03,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if len(output_size) != 2 or min(output_size) <= 0:
            raise ValueError("output_size must be positive")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must lie in (0,1]")
        self.model = model.eval().requires_grad_(False)
        self.steps = int(steps)
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.top_fraction = float(top_fraction)
        self.top_k = max(1, int(self.output_size[0] * self.output_size[1] * self.top_fraction))

    @torch.inference_mode()
    def score(self, features: Tensor) -> WTFlowFeatureScoreOutput:
        """从已对齐特征积分到高斯端点，并复现作者的乘法异常图。"""

        if features.ndim != 4 or not torch.is_floating_point(features):
            raise ValueError("features must be a floating-point [B,C,H,W] tensor")
        if not torch.isfinite(features).all():
            raise ValueError("features contain NaN or infinity")
        self.model.eval()
        state = features
        dt = 1.0 / self.steps
        for step_index in range(self.steps):
            time = torch.full(
                (state.shape[0],), step_index * dt, device=state.device, dtype=state.dtype
            )
            velocity = self.model(x=state, t=time, y=None)
            state = state + velocity * dt

        # 与作者实现一致：logp=-0.5*mean_c(y^2)，再使用 max(exp(logp))-exp(logp)。
        log_probability = -0.5 * state.square().mean(1)
        logp_map = F.interpolate(
            log_probability.unsqueeze(1),
            size=self.output_size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)
        probability = torch.exp(logp_map)
        anomaly_map = probability.amax(dim=(1, 2), keepdim=True) - probability
        image_scores = anomaly_map.flatten(1).topk(self.top_k, dim=1).values.mean(1)
        if not torch.isfinite(image_scores).all() or not torch.isfinite(anomaly_map).all():
            raise FloatingPointError("WT-Flow produced non-finite scores")
        return WTFlowFeatureScoreOutput(image_scores=image_scores, anomaly_maps=anomaly_map)
