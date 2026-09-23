"""把量纲不同的环境支持分数和运输幅度映射到经验百分位。"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EmpiricalScoreCalibrator(nn.Module):
    """用正常 validation 样本拟合经验 CDF，再输出 [0,1] 环境分数。

    该模块不训练参数，也不接触异常标签。组合前分别做百分位变换，可避免
    ``-log p(e)`` 与特征运输幅度因量纲差异而由某一项永久支配。
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("support_reference", torch.empty(0))
        self.register_buffer("transport_reference", torch.empty(0))

    @property
    def fitted(self) -> bool:
        return self.support_reference.numel() > 0

    @torch.no_grad()
    def fit(self, support_scores: Tensor, transport_scores: Tensor) -> "EmpiricalScoreCalibrator":
        support_scores = support_scores.detach().flatten().float()
        transport_scores = transport_scores.detach().flatten().float()
        if support_scores.numel() == 0 or support_scores.shape != transport_scores.shape:
            raise ValueError("normal-validation score vectors must be non-empty and aligned")
        if not torch.isfinite(support_scores).all() or not torch.isfinite(transport_scores).all():
            raise ValueError("calibration scores must be finite")
        self.support_reference = support_scores.sort().values.to(self.support_reference.device)
        self.transport_reference = transport_scores.sort().values.to(self.transport_reference.device)
        return self

    @staticmethod
    def _percentile(values: Tensor, reference: Tensor) -> Tensor:
        # searchsorted 给出小于等于当前值的正常验证样本比例，即经验 CDF。
        rank = torch.searchsorted(reference.to(values), values.contiguous(), right=True)
        return rank.to(values.dtype) / max(reference.numel(), 1)

    def forward(
        self,
        support_scores: Tensor,
        transport_scores: Tensor,
    ) -> Tensor:
        if not self.fitted:
            raise RuntimeError("calibrator must be fitted on normal validation scores")
        if support_scores.shape != transport_scores.shape:
            raise ValueError("score shapes must match")
        support_percentile = self._percentile(support_scores, self.support_reference)
        transport_percentile = self._percentile(transport_scores, self.transport_reference)
        # 描述超出支持域或运输幅度异常，任一项高即判为环境偏离。
        return torch.maximum(support_percentile, transport_percentile)

    def summary(self) -> dict[str, object]:
        return {
            "fitted": self.fitted,
            "normal_validation_samples": int(self.support_reference.numel()),
            "combination_space": "empirical_percentile",
            "default_combination": "max",
        }
