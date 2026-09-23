"""RA-MSDFlow 的 metrics 模块。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def _binary_inputs(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.size != s.size:
        raise ValueError("labels and scores must be non-empty and have the same length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("binary metrics require labels in {0,1}")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinity")
    return y, s


def binary_auroc(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """执行 `binary_auroc` 所需的处理。"""

    y, s = _binary_inputs(labels, scores)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires at least one positive and one negative")
    # 步骤 1：计算异常分数。
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        stop = start + 1
        while stop < y.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    # 步骤 2：转换特征表示。
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


@dataclass(frozen=True, slots=True)
class BinaryRates:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    tpr: float
    fpr: float

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "tpr": self.tpr,
            "fpr": self.fpr,
        }


def binary_rates_at_threshold(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
) -> BinaryRates:
    """执行 `binary_rates_at_threshold` 所需的处理。"""

    y, s = _binary_inputs(labels, scores)
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    predicted = s > float(threshold)
    tp = int(np.logical_and(predicted, y == 1).sum())
    fp = int(np.logical_and(predicted, y == 0).sum())
    tn = int(np.logical_and(~predicted, y == 0).sum())
    fn = int(np.logical_and(~predicted, y == 1).sum())
    return BinaryRates(
        threshold=float(threshold),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        tpr=float(tp / (tp + fn)) if tp + fn else float("nan"),
        fpr=float(fp / (fp + tn)) if fp + tn else float("nan"),
    )


def pixel_auroc(masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    """执行 `pixel_auroc` 所需的处理。"""

    ground_truth = np.asarray(masks)
    scores = np.asarray(anomaly_maps, dtype=np.float64)
    if ground_truth.shape != scores.shape or ground_truth.ndim != 3:
        raise ValueError("masks and anomaly_maps must have identical [N,H,W] shapes")
    return binary_auroc((ground_truth > 0).reshape(-1), scores.reshape(-1))


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """执行 `_connected_components` 所需的处理。"""

    foreground = np.asarray(mask, dtype=bool)
    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    components: list[np.ndarray] = []
    for row in range(height):
        for column in range(width):
            if not foreground[row, column] or visited[row, column]:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            pixels: list[int] = []
            while stack:
                current_row, current_column = stack.pop()
                pixels.append(current_row * width + current_column)
                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and foreground[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
            components.append(np.asarray(pixels, dtype=np.int64))
    return components


@dataclass(frozen=True, slots=True)
class AUPROResult:
    score: float
    max_fpr: float
    regions: int
    curve_points: int

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "max_fpr": self.max_fpr,
            "regions": self.regions,
            "curve_points": self.curve_points,
        }


def aupro(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    *,
    max_fpr: float = 0.05,
    max_thresholds: int = 200,
) -> AUPROResult:
    """执行 `aupro` 所需的处理。"""

    ground_truth = np.asarray(masks) > 0
    scores = np.asarray(anomaly_maps, dtype=np.float64)
    if ground_truth.shape != scores.shape or ground_truth.ndim != 3:
        raise ValueError("masks and anomaly_maps must have identical [N,H,W] shapes")
    if not np.isfinite(scores).all():
        raise ValueError("anomaly_maps contain NaN or infinity")
    if not 0.0 < max_fpr <= 1.0 or max_thresholds < 2:
        raise ValueError("max_fpr must be in (0,1] and max_thresholds at least two")
    components: list[tuple[int, np.ndarray]] = []
    for sample_index, mask in enumerate(ground_truth):
        components.extend((sample_index, indices) for indices in _connected_components(mask))
    if not components:
        raise ValueError("AUPRO requires at least one anomalous connected region")
    negative = ~ground_truth
    if not negative.any():
        raise ValueError("AUPRO requires background pixels")

    # 步骤 3：计算异常分数。
    unique = np.unique(scores)
    if unique.size <= max_thresholds:
        finite_thresholds = unique[::-1]
    else:
        quantiles = np.linspace(1.0, 0.0, max_thresholds)
        finite_thresholds = np.unique(np.quantile(scores, quantiles))[::-1]
    thresholds = np.concatenate(([np.inf], finite_thresholds, [-np.inf]))
    fprs: list[float] = []
    pros: list[float] = []
    flat_scores = scores.reshape(scores.shape[0], -1)
    for threshold in thresholds:
        predicted = scores > threshold
        fprs.append(float(predicted[negative].mean()))
        flat_prediction = predicted.reshape(predicted.shape[0], -1)
        overlaps = [float(flat_prediction[index, pixels].mean()) for index, pixels in components]
        pros.append(float(np.mean(overlaps)))

    # 步骤 4：按当前协议处理。
    best_by_fpr: dict[float, float] = {}
    for fpr, pro in zip(fprs, pros):
        best_by_fpr[fpr] = max(pro, best_by_fpr.get(fpr, 0.0))
    curve_x = np.asarray(sorted(best_by_fpr), dtype=np.float64)
    curve_y = np.asarray([best_by_fpr[value] for value in curve_x], dtype=np.float64)
    below = curve_x < max_fpr
    clipped_x = curve_x[below]
    clipped_y = curve_y[below]
    cutoff_y = float(np.interp(max_fpr, curve_x, curve_y))
    clipped_x = np.concatenate((clipped_x, [max_fpr]))
    clipped_y = np.concatenate((clipped_y, [cutoff_y]))
    trapezoid = getattr(np, "trapezoid", np.trapz)
    area = float(trapezoid(clipped_y, clipped_x) / max_fpr)
    return AUPROResult(score=area, max_fpr=float(max_fpr), regions=len(components), curve_points=clipped_x.size)
