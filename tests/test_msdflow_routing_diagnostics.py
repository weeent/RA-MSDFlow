"""路由诊断的 CPU 单元测试；不依赖真实图像或 CUDA。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msdflow.cli.diagnose import (
    _fit_policy_calibration,
    _normalized_mutual_information,
    _verify_reference,
)
from msdflow.conditions import (
    DiagonalGaussianModeBank,
    EnvironmentStandardizer,
    HybridEnvironmentDescriptor,
)
from msdflow.inference import ROUTING_POLICIES, RoutingDiagnosticPipeline
from msdflow.models import (
    MSDFlowModel,
    ModeConditionedNormalityFlow,
    PairedEnvironmentFlow,
    SmoothEnvironmentVelocity,
)


def _four_mode_model() -> MSDFlowModel:
    """建立尺寸很小但接口完整的冻结四模式模型。"""

    torch.manual_seed(310)
    descriptor = HybridEnvironmentDescriptor(learned_dim=2, lowres_size=8)
    clusters = []
    for mode_id in range(4):
        center = torch.full((10,), float(mode_id) * 2.0 - 3.0)
        clusters.append(center + 0.15 * torch.randn(12, 10))
    raw_environment = torch.cat(clusters)
    standardizer = EnvironmentStandardizer(10).fit(raw_environment)
    mode_bank = DiagonalGaussianModeBank(10, 4)
    mode_bank.fit(standardizer(raw_environment), seed=311)
    environment_flow = PairedEnvironmentFlow(
        SmoothEnvironmentVelocity(
            4, 10, hidden_dim=16, spatial_hidden_channels=8, lowres_size=2
        )
    )
    normality_flow = ModeConditionedNormalityFlow(
        4, 4, base_channels=8, condition_hidden_dim=8
    )
    return (
        MSDFlowModel(
            descriptor, standardizer, mode_bank, environment_flow, normality_flow
        )
        .eval()
        .requires_grad_(False)
    )


class RoutingPipelineTests(unittest.TestCase):
    def test_all_five_policies_have_fixed_shapes_and_finite_values(self) -> None:
        model = _four_mode_model()
        pipeline = RoutingDiagnosticPipeline(
            model,
            clean_anchor_mode=1,
            environment_steps=1,
            normality_steps=1,
            output_size=8,
            top_fraction=0.25,
        )
        feature = torch.randn(2, 4, 8, 8)
        environment = model.standardizer(torch.randn(2, 10))
        output = pipeline.predict_features(feature, environment)

        self.assertEqual(tuple(output.candidate_anomaly_maps.shape), (2, 4, 8, 8))
        self.assertEqual(tuple(output.raw_candidate_anomaly_maps.shape), (2, 4, 8, 8))
        self.assertEqual(tuple(output.candidate_scores.shape), (2, 4))
        self.assertEqual(set(output.policy_scores), set(ROUTING_POLICIES))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.policy_scores.values()))
        # P3/P4 必须选择一张完整候选图，不能逐像素拼接。
        self.assertTrue(
            torch.equal(output.policy_scores["image_min_all4"], output.candidate_scores.min(1).values)
        )
        self.assertTrue(
            torch.equal(
                output.policy_scores["raw_no_transport_min4"],
                output.raw_candidate_scores.min(1).values,
            )
        )
        self.assertEqual(tuple(output.photo_distance.shape), (2,))
        self.assertEqual(tuple(output.learned_distance.shape), (2,))
        self.assertFalse(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertFalse(model.training)

    def test_single_mode_model_uses_top1_and_fixed_condition_width(self) -> None:
        """K=1 可评分，且 condition_dim=4 与 K=4 主模型保持参数量一致。"""

        torch.manual_seed(312)
        descriptor = HybridEnvironmentDescriptor(learned_dim=2, lowres_size=8)
        raw_environment = 0.15 * torch.randn(24, 10)
        standardizer = EnvironmentStandardizer(10).fit(raw_environment)
        mode_bank = DiagonalGaussianModeBank(10, 1)
        mode_bank.fit(standardizer(raw_environment), seed=313)
        environment_flow = PairedEnvironmentFlow(
            SmoothEnvironmentVelocity(4, 10, hidden_dim=16, spatial_hidden_channels=8, lowres_size=2)
        )
        single = ModeConditionedNormalityFlow(
            4, 1, base_channels=8, condition_hidden_dim=8, condition_dim=4
        )
        four = ModeConditionedNormalityFlow(
            4, 4, base_channels=8, condition_hidden_dim=8, condition_dim=4
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in single.parameters()),
            sum(parameter.numel() for parameter in four.parameters()),
        )
        model = MSDFlowModel(
            descriptor, standardizer, mode_bank, environment_flow, single
        ).eval().requires_grad_(False)
        pipeline = RoutingDiagnosticPipeline(
            model, clean_anchor_mode=0, environment_steps=1, normality_steps=1, output_size=8
        )
        output = pipeline.predict_features(
            torch.randn(2, 4, 8, 8), model.standardizer(torch.randn(2, 10))
        )
        self.assertEqual(pipeline.posterior_top2_pipeline.top_m, 1)
        self.assertEqual(tuple(output.candidate_scores.shape), (2, 1))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.policy_scores.values()))


class RoutingProtocolTests(unittest.TestCase):
    def test_policy_thresholds_are_independent_and_normal_val_only(self) -> None:
        rows = []
        for sample_index in range(5):
            row: dict[str, object] = {"label": 0, "split": "val"}
            for policy_index, policy in enumerate(ROUTING_POLICIES):
                row[f"{policy}_score"] = float(sample_index + policy_index * 10)
            rows.append(row)
        thresholds, audit = _fit_policy_calibration(rows, 0.95)
        self.assertEqual(set(thresholds), set(ROUTING_POLICIES))
        self.assertEqual(len({value for value in thresholds.values()}), len(ROUTING_POLICIES))
        self.assertTrue(all(audit[name]["normal_count"] == 5 for name in ROUTING_POLICIES))
        bad = [dict(rows[0], label=1)]
        with self.assertRaisesRegex(ValueError, "normal validation"):
            _fit_policy_calibration(bad, 0.95)

    def test_p0_reference_is_matched_by_sample_identity(self) -> None:
        current = [
            {"base_id": "b", "variant_id": "clean", "posterior_top2_score": 0.2},
            {"base_id": "a", "variant_id": "clean", "posterior_top2_score": 0.1},
        ]
        reference = [
            {"base_id": "a", "variant_id": "clean", "image_score": 0.1},
            {"base_id": "b", "variant_id": "clean", "image_score": 0.2},
        ]
        with tempfile.TemporaryDirectory(prefix="msd_route_ref_") as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in reference), encoding="utf-8"
            )
            audit = _verify_reference(current, path, tolerance=1e-7)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["max_abs_error"], 0.0)
            current[0]["posterior_top2_score"] = 0.21
            with self.assertRaisesRegex(ValueError, "differs from v2"):
                _verify_reference(current, path, tolerance=1e-7)

    def test_mode_variant_nmi_has_expected_extremes(self) -> None:
        perfectly_aligned = np.eye(4, dtype=np.int64) * 10
        independent = np.ones((4, 4), dtype=np.int64) * 10
        self.assertAlmostEqual(_normalized_mutual_information(perfectly_aligned), 1.0)
        self.assertAlmostEqual(_normalized_mutual_information(independent), 0.0)


if __name__ == "__main__":
    unittest.main()
