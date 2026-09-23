"""Reference OT-CFM 的 CPU 机制测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import numpy as np

from msdflow.cli.reference_otcfm_matrix import expand_matrix
from msdflow.reference_otcfm import (
    OTCFMTrainingConfig,
    ReferenceOTCFM,
    build_balanced_source_bank,
    feature_maps_to_patches,
    load_reference_otcfm,
    patches_to_feature_maps,
    sinkhorn_coupling,
    train_reference_otcfm,
)
from msdflow.score_calibration import EmpiricalTailCalibrator, crossfit_folds


class ReferenceOTCFMTests(unittest.TestCase):
    def test_crossfit_folds_are_deterministic_disjoint_and_complete(self) -> None:
        folds = crossfit_folds(20, 5, seed=9827)
        repeated = crossfit_folds(20, 5, seed=9827)
        self.assertEqual(folds, repeated)
        held_out = []
        for fit_indices, held_out_indices in folds:
            self.assertFalse(set(fit_indices).intersection(held_out_indices))
            self.assertEqual(len(fit_indices), 16)
            self.assertEqual(len(held_out_indices), 4)
            held_out.extend(held_out_indices)
        self.assertEqual(sorted(held_out), list(range(20)))

    def test_empirical_tail_calibration_has_common_scale(self) -> None:
        calibrator = EmpiricalTailCalibrator.fit(list(range(20)), alpha=0.05)
        transformed = calibrator.transform([-1.0, 19.0, 20.0])
        # 低分正常点尾概率接近 1；超过全部 reference 的点达到最小尾概率 1/21。
        self.assertTrue(np.allclose(calibrator.p_values([-1.0, 19.0, 20.0]), [1.0, 2 / 21, 1 / 21]))
        self.assertGreater(float(transformed[-1]), calibrator.threshold)
        self.assertLess(float(transformed[1]), calibrator.threshold)
        self.assertTrue(calibrator.supports_target_fpr)
        self.assertAlmostEqual(calibrator.minimum_p_value, 1 / 21)

    def test_two_shot_cannot_resolve_five_percent_tail(self) -> None:
        calibrator = EmpiricalTailCalibrator.fit([0.1, 0.2], alpha=0.05)
        self.assertFalse(calibrator.supports_target_fpr)
        self.assertEqual(calibrator.minimum_p_value, 1 / 3)

    def test_patch_roundtrip_and_balanced_bank(self) -> None:
        generator = torch.Generator().manual_seed(1)
        features = torch.randn(8, 6, 4, 4, generator=generator)
        patches = feature_maps_to_patches(features)
        restored = patches_to_feature_maps(patches, tuple(features.shape))
        self.assertTrue(torch.equal(features, restored))
        bank = build_balanced_source_bank(features, n_modes=4, max_patches=32, seed=2)
        self.assertEqual(bank.patches.shape, (32, 6))
        counts = torch.bincount(bank.mode_ids, minlength=4)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_sinkhorn_has_uniform_marginals(self) -> None:
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        coupling, cost = sinkhorn_coupling(target, source, epsilon=0.03, iterations=80)
        self.assertTrue(torch.allclose(coupling.sum(1), torch.full((2,), 0.5), atol=1e-5))
        self.assertTrue(torch.allclose(coupling.sum(0), torch.full((2,), 0.5), atol=1e-5))
        self.assertGreater(float(coupling.diagonal().sum()), 0.99)
        self.assertLess(float((coupling * cost).sum()), 1e-4)

    def test_zero_velocity_starts_from_coral_and_preserves_shape(self) -> None:
        model = ReferenceOTCFM(3, hidden_dim=8, time_dim=4, depth=1)
        model.set_statistics(
            source_mean=torch.tensor([1.0, 2.0, 3.0]),
            source_std=torch.ones(3),
            target_mean=torch.tensor([0.5, 0.5, 0.5]),
            coral_transform=torch.eye(3),
        )
        features = torch.randn(2, 3, 2, 2)
        transported = model.transport_features(features, steps=2, chunk_size=3)
        expected = features - 0.5 + torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)
        self.assertEqual(transported.shape, features.shape)
        self.assertTrue(torch.allclose(transported, expected, atol=1e-6))

    def test_short_training_and_compact_checkpoint_roundtrip(self) -> None:
        source = torch.randn(6, 4, 2, 2)
        target = source[:2] + 0.2
        bank = build_balanced_source_bank(source, n_modes=2, max_patches=24, seed=3)
        model = ReferenceOTCFM(4, hidden_dim=8, time_dim=4, depth=1)
        model.set_statistics(
            source_mean=bank.patches.mean(0),
            source_std=bank.patches.std(0, unbiased=False).clamp_min(1e-3),
            target_mean=feature_maps_to_patches(target).mean(0),
            coral_transform=torch.eye(4),
        )
        config = OTCFMTrainingConfig(
            steps=2,
            patch_batch_size=8,
            sinkhorn_iterations=5,
            amp=False,
            log_interval=1,
            checkpoint_interval=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            history, path = train_reference_otcfm(
                model,
                bank,
                feature_maps_to_patches(target),
                config=config,
                device=torch.device("cpu"),
                seed=4,
                output_directory=directory,
            )
            loaded, provenance = load_reference_otcfm(path)
            self.assertEqual(len(history), 2)
            self.assertEqual(provenance["seed"], 4)
            probe = torch.randn(1, 4, 2, 2)
            self.assertTrue(torch.allclose(model.eval().transport_features(probe), loaded.transport_features(probe)))
            self.assertTrue((Path(directory) / "last.pt").is_file())
            # 断点恢复必须从 CPU ByteTensor RNG 状态继续，而不能把状态误搬到计算设备。
            resumed = ReferenceOTCFM(4, hidden_dim=8, time_dim=4, depth=1)
            resumed_history, _ = train_reference_otcfm(
                resumed,
                bank,
                feature_maps_to_patches(target),
                config=OTCFMTrainingConfig(
                    steps=3,
                    patch_batch_size=8,
                    sinkhorn_iterations=5,
                    amp=False,
                    log_interval=1,
                    checkpoint_interval=1,
                ),
                device=torch.device("cpu"),
                seed=999,
                output_directory=directory,
                resume=True,
            )
            self.assertEqual(int(resumed_history[-1]["step"]), 3)

    def test_main_matrix_expands_repeated_seeds(self) -> None:
        matrix = {
            "base": {"device": "cuda:0"},
            "jobs": [
                {
                    "dataset": "robustad",
                    "category": "PCB",
                    "manifest": "m.jsonl",
                    "inference_bundle": "b.pt",
                    "target_domains": {"lighting": "test1"},
                    "seeds": [11, 12, 13],
                    "output_directory": "out/seed{train_seed}",
                }
            ],
        }
        jobs = expand_matrix(matrix)
        self.assertEqual([job["train_seed"] for job in jobs], [11, 12, 13])
        self.assertEqual([job["reference_seed"] for job in jobs], [11, 12, 13])
        self.assertEqual(len({job["output_directory"] for job in jobs}), 3)


if __name__ == "__main__":
    unittest.main()
