"""冻结 WT-Flow 特征评分后半段的 CPU 测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msdflow.baselines.coral_wtflow import FrozenWTFlowFeatureScorer


class _ZeroVelocity(nn.Module):
    def forward(self, *, x: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        self.last_time_shape = tuple(t.shape)
        return torch.zeros_like(x)


class CoralWTFlowScorerTests(unittest.TestCase):
    def test_feature_scorer_matches_frozen_euler_contract(self) -> None:
        model = _ZeroVelocity()
        scorer = FrozenWTFlowFeatureScorer(
            model, steps=2, output_size=(8, 8), top_fraction=0.25
        )
        features = torch.randn(3, 4, 4, 4)
        output = scorer.score(features)
        self.assertEqual(tuple(output.image_scores.shape), (3,))
        self.assertEqual(tuple(output.anomaly_maps.shape), (3, 8, 8))
        self.assertTrue(torch.isfinite(output.image_scores).all())
        self.assertTrue(torch.isfinite(output.anomaly_maps).all())
        self.assertEqual(model.last_time_shape, (3,))
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
