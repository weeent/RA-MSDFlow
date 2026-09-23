"""MSD-Flow 四模块的 CPU 关键单元测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from envfm.data.records import SampleRecord
from msdflow.conditions import (
    DiagonalGaussianModeBank,
    EnvironmentStandardizer,
    HybridEnvironmentDescriptor,
    LowFrequencyEnvironmentEncoder,
)
from msdflow.data import (
    CachedPairedFeatureDataset,
    PairedEnvironmentDataset,
    build_environment_pairs,
    build_paired_feature_cache,
)
from msdflow.models import (
    MSDFlowModel,
    ModeConditionedNormalityFlow,
    PairedEnvironmentFlow,
    SmoothEnvironmentVelocity,
)
from msdflow.training import EnvironmentTransportTrainer, NormalityFlowTrainer, TrainingConfig


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(20, 220, 32, dtype=np.uint8)
    image = np.stack(np.meshgrid(x, x), axis=-1)
    rgb = np.concatenate((image, image[..., :1]), axis=-1)
    Image.fromarray(rgb, mode="RGB").save(path)


def _normal_record(path: Path, base_id: str = "toy/object/train/000.png") -> SampleRecord:
    return SampleRecord(
        dataset="toy",
        category="object",
        image_path=str(path),
        label=0,
        defect_type="good",
        split="train",
        official_split="train",
        domain="clean",
        base_id=base_id,
    )


class _ToyPairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: int = 6) -> None:
        generator = torch.Generator().manual_seed(77)
        self.a = torch.randn(samples, 4, 8, 8, generator=generator)
        self.b = self.a + 0.15
        self.ea = torch.randn(samples, 2, generator=generator) * 0.1 - 1.0
        self.eb = torch.randn(samples, 2, generator=generator) * 0.1 + 1.0

    def __len__(self) -> int:
        return self.a.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "feature_a": self.a[index],
            "feature_b": self.b[index],
            "environment_a": self.ea[index],
            "environment_b": self.eb[index],
            "label_a": torch.tensor(0),
            "label_b": torch.tensor(0),
        }


class MSDFlowDataTests(unittest.TestCase):
    def test_pair_replay_and_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="msdflow_data_") as temporary:
            image_path = Path(temporary) / "000.png"
            _write_rgb(image_path)
            record = _normal_record(image_path)
            pairs = build_environment_pairs([record], target_variants=("exposure", "gradient"), base_seed=11)
            replay = build_environment_pairs([record], target_variants=("exposure", "gradient"), base_seed=11)
            self.assertEqual(pairs, replay)
            dataset = PairedEnvironmentDataset(pairs, image_size=32)
            first, repeated = dataset[0], dataset[0]
            self.assertTrue(torch.equal(first["image_raw_b"], repeated["image_raw_b"]))
            self.assertEqual(dataset.pairs[0].source.base_id, dataset.pairs[0].target.base_id)
            self.assertTrue(dataset.summary()["all_normal"])

            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            index = build_paired_feature_cache(
                loader,
                nn.AvgPool2d(4),
                HybridEnvironmentDescriptor(learned_dim=0),
                Path(temporary) / "cache",
                metadata={"purpose": "unit-test"},
            )
            cached = CachedPairedFeatureDataset(index)
            self.assertEqual(len(cached), 2)
            self.assertEqual(tuple(cached[0]["feature_a"].shape), (3, 8, 8))
            self.assertEqual(tuple(cached[0]["environment_a"].shape), (8,))
            self.assertEqual(cached.summary()["shard_count"], 1)

    def test_pair_builder_rejects_anomaly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.png"
            _write_rgb(path)
            bad = _normal_record(path).with_updates(label=1, defect_type="bad")
            with self.assertRaisesRegex(ValueError, "normal-only"):
                build_environment_pairs([bad])


class MSDFlowConditionTests(unittest.TestCase):
    def test_hybrid_descriptor_shapes_and_finiteness(self) -> None:
        image = torch.rand(3, 3, 48, 40)
        descriptor = HybridEnvironmentDescriptor(learned_dim=6)
        output = descriptor(image)
        self.assertEqual(tuple(output.photo8.shape), (3, 8))
        self.assertEqual(tuple(output.learned.shape), (3, 6))
        self.assertEqual(tuple(output.combined.shape), (3, 14))
        self.assertEqual(tuple(output.low_resolution.shape), (3, 3, 32, 32))
        self.assertTrue(torch.isfinite(output.combined).all())

    def test_mode_bank_finds_two_centers_and_marks_ood(self) -> None:
        generator = torch.Generator().manual_seed(12)
        values = torch.cat(
            (torch.randn(50, 2, generator=generator) * 0.08 - 2, torch.randn(50, 2, generator=generator) * 0.08 + 2)
        )
        standardizer = EnvironmentStandardizer(2).fit(values)
        normalized = standardizer(values)
        bank = DiagonalGaussianModeBank(2, 2, support_quantile=0.95)
        result = bank.fit(normalized, seed=3)
        assignment = bank.assign(normalized[:4], top_m=2)
        self.assertTrue(result.final_mean_log_likelihood == result.final_mean_log_likelihood)
        self.assertLess(float((assignment.top_weights.sum(1) - 1).abs().max()), 1e-6)
        remote = bank.assign(torch.tensor([[20.0, 20.0]]), top_m=1)
        self.assertFalse(bool(remote.in_support.item()))


class MSDFlowModelTests(unittest.TestCase):
    def test_environment_flow_path_and_zero_initialized_identity(self) -> None:
        velocity = SmoothEnvironmentVelocity(4, 2, hidden_dim=16, spatial_hidden_channels=8, lowres_size=2)
        flow = PairedEnvironmentFlow(velocity)
        feature_a = torch.randn(2, 4, 8, 8)
        feature_b = feature_a + 0.2
        environment_a = torch.zeros(2, 2)
        environment_b = torch.ones(2, 2)
        path = flow.sample_path(feature_a, feature_b, torch.tensor([0.0, 1.0]))
        self.assertTrue(torch.equal(path.feature_t[0], feature_a[0]))
        self.assertTrue(torch.equal(path.feature_t[1], feature_b[1]))
        output = flow(feature_a, feature_b, environment_a, environment_b, time=torch.full((2,), 0.5))
        self.assertEqual(float(output.prediction.velocity.abs().max()), 0.0)
        transported = flow.transport(feature_a, environment_a, environment_b, steps=2)
        self.assertTrue(torch.equal(transported, feature_a))

    def test_normality_flow_and_full_wrapper(self) -> None:
        normality = ModeConditionedNormalityFlow(4, 2, base_channels=8, condition_hidden_dim=8)
        feature = torch.randn(2, 4, 8, 8)
        modes = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
        output = normality(feature, modes, time=torch.tensor([0.25, 0.75]), noise=torch.zeros_like(feature))
        self.assertEqual(output.loss.ndim, 0)
        output.loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in normality.parameters()))

        descriptor = HybridEnvironmentDescriptor(learned_dim=0)
        standardizer = EnvironmentStandardizer(8).fit(torch.randn(20, 8))
        mode_bank = DiagonalGaussianModeBank(8, 2)
        train_environment = standardizer(torch.randn(30, 8))
        mode_bank.fit(train_environment)
        env_flow = PairedEnvironmentFlow(
            SmoothEnvironmentVelocity(4, 8, hidden_dim=16, spatial_hidden_channels=8, lowres_size=2)
        )
        model = MSDFlowModel(descriptor, standardizer, mode_bank, env_flow, normality)
        canonical = model.canonicalize(feature.detach(), train_environment[:2], top_m=2, steps=2)
        self.assertEqual(tuple(canonical.canonical_features.shape), (2, 2, 4, 8, 8))
        anomaly = torch.rand(2, 2, 8, 8)
        combined = model.softmin_defect_map(anomaly, canonical.assignment.top_weights)
        self.assertEqual(tuple(combined.shape), (2, 8, 8))


class MSDFlowTrainingTests(unittest.TestCase):
    def test_transport_and_normality_trainers_write_artifacts(self) -> None:
        train_loader = DataLoader(_ToyPairDataset(6), batch_size=3, shuffle=False)
        val_loader = DataLoader(_ToyPairDataset(4), batch_size=2, shuffle=False)
        with tempfile.TemporaryDirectory(prefix="msdflow_training_") as temporary:
            config = TrainingConfig(
                epochs=1, learning_rate=1e-3, weight_decay=0.0, checkpoint_interval_epochs=5, amp=False
            )
            env_flow = PairedEnvironmentFlow(
                SmoothEnvironmentVelocity(4, 2, hidden_dim=16, spatial_hidden_channels=8, lowres_size=2)
            )
            result = EnvironmentTransportTrainer(
                env_flow, run_directory=Path(temporary) / "transport", config=config, device="cpu"
            ).fit(train_loader, val_loader)
            self.assertTrue(result.last_checkpoint.is_file())
            self.assertTrue(result.best_checkpoint.is_file())

            mode_bank = DiagonalGaussianModeBank(2, 2)
            mode_bank.fit(torch.cat((torch.randn(20, 2) - 1, torch.randn(20, 2) + 1)))
            normality = ModeConditionedNormalityFlow(4, 2, base_channels=8, condition_hidden_dim=8)
            normal_result = NormalityFlowTrainer(
                normality,
                mode_bank=mode_bank,
                environment_flow=env_flow,
                run_directory=Path(temporary) / "normality",
                config=config,
                device="cpu",
            ).fit(train_loader, val_loader)
            self.assertTrue(normal_result.summary_path.is_file())


if __name__ == "__main__":
    unittest.main()
