"""少样本目标环境的异常分数校准。

本模块只处理一维图像异常分数，不读取图像、特征或异常标签。输入是未参与相应
适配器拟合的正常 reference 分数；输出是跨环境可比较的经验尾分数以及统一阈值。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def crossfit_folds(sample_count: int, fold_count: int, *, seed: int) -> list[tuple[list[int], list[int]]]:
    """构造确定性的交叉拟合索引。

    输入：``sample_count`` 个正常 reference、期望折数和随机种子。
    输出：若干 ``(fit_indices, held_out_indices)``。每个样本恰好作为 held-out
    样本一次，且任何一折的拟合集与留出集都不相交。
    """

    if sample_count < 2:
        raise ValueError("cross-fitting needs at least two reference samples")
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    actual_folds = min(int(fold_count), int(sample_count))
    order = np.random.default_rng(seed).permutation(sample_count).tolist()
    held_out_groups = [order[index::actual_folds] for index in range(actual_folds)]
    folds: list[tuple[list[int], list[int]]] = []
    all_indices = set(range(sample_count))
    for held_out in held_out_groups:
        held_out_set = set(held_out)
        fit = sorted(all_indices.difference(held_out_set))
        folds.append((fit, sorted(held_out)))
    return folds


@dataclass(frozen=True, slots=True)
class EmpiricalTailCalibrator:
    """把不同环境的原始异常分数映射到统一的正常尾概率尺度。

    对待测原始分数 ``s``，右尾 p 值定义为

    ``p(s) = (1 + #{a_i >= s}) / (n + 1)``，

    其中 ``a_i`` 是正常 reference 的交叉拟合分数。最终异常分数为
    ``-log(p)``，因此越大越异常。加一修正避免有限样本下得到零概率。
    """

    calibration_scores: tuple[float, ...]
    alpha: float = 0.05

    @classmethod
    def fit(cls, scores: Sequence[float], *, alpha: float = 0.05) -> "EmpiricalTailCalibrator":
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("calibration scores must be a non-empty one-dimensional sequence")
        if not np.isfinite(values).all():
            raise ValueError("calibration scores must all be finite")
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be between zero and one")
        return cls(tuple(float(value) for value in values), float(alpha))

    @property
    def sample_count(self) -> int:
        return len(self.calibration_scores)

    @property
    def minimum_p_value(self) -> float:
        """有限 reference 数量能够分辨的最小尾概率。"""

        return 1.0 / (self.sample_count + 1)

    @property
    def supports_target_fpr(self) -> bool:
        """reference 数量是否足以在经验尺度上触达目标 FPR。"""

        return self.minimum_p_value <= self.alpha

    @property
    def threshold(self) -> float:
        """与 ``p <= alpha`` 等价、适配严格 ``score > threshold`` 的统一阈值。"""

        # 既有评估器采用严格大于。向负无穷移动一个浮点单位，使 p==alpha
        # 时也被判为异常，与标准 conformal 拒绝规则保持一致。
        return float(np.nextafter(-math.log(self.alpha), -math.inf))

    def p_values(self, scores: Sequence[float]) -> np.ndarray:
        """输入任意原始图像分数，输出相同形状的正常右尾经验 p 值。"""

        query = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(query).all():
            raise ValueError("query scores must all be finite")
        reference = np.sort(np.asarray(self.calibration_scores, dtype=np.float64))
        # searchsorted(left) 给出严格小于 query 的数量，剩余项即 >= query。
        greater_or_equal = reference.size - np.searchsorted(reference, query, side="left")
        return (greater_or_equal.astype(np.float64) + 1.0) / (reference.size + 1.0)

    def transform(self, scores: Sequence[float]) -> np.ndarray:
        """输入原始分数，输出跨环境统一的 ``-log(p)`` 异常分数。"""

        return -np.log(self.p_values(scores))

    def summary(self) -> dict[str, object]:
        """返回写入运行报告的关键统计，不保存任何异常样本信息。"""

        values = np.asarray(self.calibration_scores, dtype=np.float64)
        return {
            "method": "cross_fitted_empirical_tail",
            "sample_count": self.sample_count,
            "target_fpr": self.alpha,
            "minimum_resolvable_p": self.minimum_p_value,
            "supports_target_fpr": self.supports_target_fpr,
            "common_score": "negative_log_normal_tail_p",
            "common_threshold": self.threshold,
            "raw_score_min": float(values.min()),
            "raw_score_median": float(np.median(values)),
            "raw_score_max": float(values.max()),
        }
