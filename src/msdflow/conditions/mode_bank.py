"""纯 PyTorch 对角高斯混合环境模式库。

模式库只在正常训练环境描述上拟合。每个分量中心 ``c_k`` 表示一个正常环境中心，
对角方差 ``sigma_k^2`` 表示该中心各描述维度允许的变化范围。测试环境若在训练分布
低密度区，会被 ``in_support=False`` 标记，而不会被强制解释成某个已见环境。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ModeBankFitResult:
    iterations: int
    converged: bool
    final_mean_log_likelihood: float
    support_threshold: float
    component_counts: tuple[int, ...]


@dataclass(slots=True)
class EnvironmentModeAssignment:
    log_probability: Tensor
    posterior: Tensor
    top_indices: Tensor
    top_weights: Tensor
    support_score: Tensor
    in_support: Tensor

    def summary(self) -> dict[str, object]:
        return {
            "batch_size": int(self.posterior.shape[0]),
            "mode_count": int(self.posterior.shape[1]),
            "in_support_rate": float(self.in_support.float().mean().item()),
            "support_score_mean": float(self.support_score.mean().item()),
            "posterior_row_sum_max_error": float((self.posterior.sum(1) - 1).abs().max().item()),
            "top_weight_row_sum_max_error": float((self.top_weights.sum(1) - 1).abs().max().item()),
        }


class DiagonalGaussianModeBank(nn.Module):
    """用 EM 拟合多中心正常环境，并输出后验责任度。"""

    def __init__(
        self,
        dimension: int,
        n_components: int = 4,
        *,
        min_variance: float = 1e-3,
        support_quantile: float = 0.99,
    ) -> None:
        super().__init__()
        if dimension <= 0 or n_components <= 0:
            raise ValueError("dimension and n_components must be positive")
        if not 0.5 < support_quantile < 1.0:
            raise ValueError("support_quantile must lie in (0.5,1)")
        self.dimension = int(dimension)
        self.n_components = int(n_components)
        self.min_variance = float(min_variance)
        self.support_quantile = float(support_quantile)
        self.register_buffer("centers", torch.zeros(n_components, dimension))
        self.register_buffer("variances", torch.ones(n_components, dimension))
        self.register_buffer("mixture_weights", torch.full((n_components,), 1.0 / n_components))
        self.register_buffer("support_threshold", torch.tensor(float("inf")))
        self.register_buffer("sample_count", torch.tensor(0, dtype=torch.long))

    @property
    def fitted(self) -> bool:
        return int(self.sample_count.item()) > 0

    def _component_log_prob(self, values: Tensor) -> Tensor:
        diff = values[:, None, :] - self.centers[None, :, :].to(values)
        variance = self.variances.to(values).clamp_min(self.min_variance)
        log_det = torch.log(variance).sum(-1)
        mahalanobis = (diff.square() / variance[None, :, :]).sum(-1)
        normalizer = self.dimension * math.log(2.0 * math.pi)
        return -0.5 * (normalizer + log_det[None, :] + mahalanobis)

    def log_prob(self, values: Tensor) -> Tensor:
        self._validate_values(values)
        if not self.fitted:
            raise RuntimeError("mode bank must be fitted on normal training descriptors")
        weighted = self._component_log_prob(values) + torch.log(
            self.mixture_weights.to(values).clamp_min(1e-12)
        )[None, :]
        return torch.logsumexp(weighted, dim=1)

    def posterior(self, values: Tensor) -> Tensor:
        self._validate_values(values)
        if not self.fitted:
            raise RuntimeError("mode bank must be fitted on normal training descriptors")
        weighted = self._component_log_prob(values) + torch.log(
            self.mixture_weights.to(values).clamp_min(1e-12)
        )[None, :]
        return torch.softmax(weighted, dim=1)

    def _validate_values(self, values: Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"expected [N,{self.dimension}], got {tuple(values.shape)}")
        if values.shape[0] == 0 or not torch.isfinite(values).all():
            raise ValueError("environment descriptors must be non-empty and finite")

    @torch.no_grad()
    def _initialize_centers(self, values: Tensor, generator: torch.Generator) -> Tensor:
        """使用 k-means++ 初始化，降低 EM 落入重复中心的概率。"""

        first = int(torch.randint(values.shape[0], (1,), generator=generator).item())
        chosen = [values[first]]
        minimum_distance = (values - chosen[0]).square().sum(1)
        for _ in range(1, self.n_components):
            probability = minimum_distance.clamp_min(1e-12)
            probability = probability / probability.sum()
            next_index = int(torch.multinomial(probability, 1, generator=generator).item())
            chosen.append(values[next_index])
            distance = (values - chosen[-1]).square().sum(1)
            minimum_distance = torch.minimum(minimum_distance, distance)
        return torch.stack(chosen)

    @torch.no_grad()
    def fit(
        self,
        values: Tensor,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        seed: int = 9826,
    ) -> ModeBankFitResult:
        """在 CPU/训练设备上执行 EM；输入必须已用 train-only 统计标准化。"""

        values = torch.as_tensor(values, dtype=torch.float32, device=self.centers.device)
        self._validate_values(values)
        if values.shape[0] < self.n_components:
            raise ValueError("number of descriptors must be at least n_components")
        generator = torch.Generator(device=values.device).manual_seed(seed)
        self.centers.copy_(self._initialize_centers(values, generator))
        global_variance = values.var(0, unbiased=False).clamp_min(self.min_variance)
        self.variances.copy_(global_variance.expand_as(self.variances))
        self.mixture_weights.fill_(1.0 / self.n_components)

        previous = -float("inf")
        converged = False
        iteration = 0
        for iteration in range(1, max_iterations + 1):
            # E 步：计算每个样本由各环境中心解释的后验责任度。
            component = self._component_log_prob(values)
            weighted = component + torch.log(self.mixture_weights.clamp_min(1e-12))[None, :]
            responsibility = torch.softmax(weighted, dim=1)
            counts = responsibility.sum(0).clamp_min(1e-6)
            # M 步：按责任度更新先验、中心和对角方差。
            centers = responsibility.transpose(0, 1) @ values / counts[:, None]
            difference = values[:, None, :] - centers[None, :, :]
            variances = (responsibility[:, :, None] * difference.square()).sum(0) / counts[:, None]
            self.centers.copy_(centers)
            self.variances.copy_(variances.clamp_min(self.min_variance))
            self.mixture_weights.copy_((counts / counts.sum()).clamp_min(1e-8))
            self.mixture_weights.div_(self.mixture_weights.sum())
            mean_ll = float(torch.logsumexp(weighted, 1).mean().item())
            if math.isfinite(previous) and abs(mean_ll - previous) <= tolerance * (1.0 + abs(previous)):
                converged = True
                break
            previous = mean_ll

        self.sample_count.fill_(values.shape[0])
        train_score = -self.log_prob(values)
        threshold = torch.quantile(train_score, self.support_quantile)
        self.support_threshold.copy_(threshold)
        hard_counts = torch.bincount(self.posterior(values).argmax(1), minlength=self.n_components)
        return ModeBankFitResult(
            iterations=iteration,
            converged=converged,
            final_mean_log_likelihood=float(self.log_prob(values).mean().item()),
            support_threshold=float(threshold.item()),
            component_counts=tuple(int(item) for item in hard_counts.tolist()),
        )

    def assign(self, values: Tensor, *, top_m: int = 2) -> EnvironmentModeAssignment:
        """输出完整后验与归一化 top-M 权重，供多模式缺陷分数 soft-min 使用。"""

        if not 1 <= top_m <= self.n_components:
            raise ValueError("top_m must lie in [1,n_components]")
        log_probability = self.log_prob(values)
        posterior = self.posterior(values)
        top_weights, top_indices = torch.topk(posterior, k=top_m, dim=1)
        # top-k 截断后必须重新归一化；否则 soft-min 会受丢弃概率质量影响。
        top_weights = top_weights / top_weights.sum(1, keepdim=True).clamp_min(1e-12)
        support_score = -log_probability
        return EnvironmentModeAssignment(
            log_probability=log_probability,
            posterior=posterior,
            top_indices=top_indices,
            top_weights=top_weights,
            support_score=support_score,
            in_support=support_score <= self.support_threshold.to(support_score),
        )

    def summary(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "n_components": self.n_components,
            "sample_count": int(self.sample_count.item()),
            "support_quantile": self.support_quantile,
            "support_threshold": float(self.support_threshold.item()),
            "mixture_weights": self.mixture_weights.detach().cpu().tolist(),
            "center_norms": self.centers.detach().cpu().norm(dim=1).tolist(),
            "variance_min": float(self.variances.min().item()),
            "variance_max": float(self.variances.max().item()),
        }
