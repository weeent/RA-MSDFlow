"""MSD-Flow 推理、评估和 CLI 数据契约的 CPU 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from envfm.data.records import SampleRecord, write_records_jsonl
from msdflow.conditions import (
    DiagonalGaussianModeBank,
    EnvironmentStandardizer,
    HybridEnvironmentDescriptor,
)
from msdflow.evaluation import compare_prediction_directories, evaluate_msd_prediction_directory
from msdflow.inference import (
    MSDCalibrationTable,
    MSDFlowInferencePipeline,
    MSDPredictionArtifactWriter,
    export_inference_bundle,
    fit_msd_calibration,
    load_inference_bundle,
)
from msdflow.models import MSDFlowModel, ModeConditionedNormalityFlow, PairedEnvironmentFlow, SmoothEnvironmentVelocity


def _components():
    torch.manual_seed(90)
    descriptor = HybridEnvironmentDescriptor(learned_dim=0)
    raw_environment = torch.cat((torch.randn(20, 8) - 1, torch.randn(20, 8) + 1))
    standardizer = EnvironmentStandardizer(8).fit(raw_environment)
    mode_bank = DiagonalGaussianModeBank(8, 2)
    fit_result = mode_bank.fit(standardizer(raw_environment), seed=91)
    environment_flow = PairedEnvironmentFlow(
        SmoothEnvironmentVelocity(4, 8, hidden_dim=16, spatial_hidden_channels=8, lowres_size=2)
    )
    normality_flow = ModeConditionedNormalityFlow(4, 2, base_channels=8, condition_hidden_dim=8)
    model = MSDFlowModel(descriptor, standardizer, mode_bank, environment_flow, normality_flow)
    return model, fit_result


def _record(path: Path, base_id: str, label: int, mask: Path | None = None) -> SampleRecord:
    return SampleRecord(
        dataset="mvtec_ad",
        category="toy",
        image_path=str(path),
        label=label,
        defect_type="good" if label == 0 else "scratch",
        split="test",
        official_split="test",
        domain="regular",
        base_id=base_id,
        mask_path=None if mask is None else str(mask),
    )


class MSDInferenceTests(unittest.TestCase):
    def test_calibration_roundtrip_and_anomaly_rejection(self) -> None:
        table = fit_msd_calibration(
            {"toy": [0.1, 0.2, 0.3, 0.4]},
            {"toy": [1.0, 2.0, 3.0, 4.0]},
            {"toy": [0.2, 0.1, 0.4, 0.3]},
            labels={"toy": [0, 0, 0, 0]},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = table.save(Path(temporary) / "calibration.json")
            loaded = MSDCalibrationTable.load(path)
            self.assertEqual(loaded.record("toy").defect_threshold, table.record("toy").defect_threshold)
            score = loaded.environment_score(torch.tensor([2.5]), torch.tensor([0.25]), "toy")
            self.assertTrue(0 <= float(score.item()) <= 1)
        with self.assertRaisesRegex(ValueError, "normal-only"):
            fit_msd_calibration(
                {"toy": [0.1, 0.2]}, {"toy": [1.0, 2.0]}, {"toy": [0.1, 0.2]}, labels={"toy": [0, 1]}
            )

    def test_pipeline_outputs_top_m_maps_and_dual_scores(self) -> None:
        model, _ = _components()
        calibration = fit_msd_calibration(
            {"toy": [0.1, 0.2, 0.3]}, {"toy": [5.0, 6.0, 7.0]}, {"toy": [0.0, 0.1, 0.2]}
        )
        pipeline = MSDFlowInferencePipeline(
            model,
            top_m=2,
            environment_steps=2,
            normality_steps=2,
            output_size=16,
            environment_calibrator=calibration.environment_calibrator("toy"),
        )
        feature = torch.randn(2, 4, 8, 8)
        environment = model.standardizer(torch.randn(2, 8))
        output = pipeline.predict_features(feature, environment)
        self.assertEqual(tuple(output.candidate_anomaly_maps.shape), (2, 2, 16, 16))
        self.assertEqual(tuple(output.anomaly_map.shape), (2, 16, 16))
        self.assertEqual(tuple(output.defect_score.shape), (2,))
        self.assertIsNotNone(output.environment_score)
        self.assertTrue(torch.isfinite(output.defect_score).all())

    def test_checkpoint_bundle_strict_reconstruction(self) -> None:
        model, _ = _components()
        with tempfile.TemporaryDirectory(prefix="msd_ckpt_") as temporary:
            root = Path(temporary)
            bundle = export_inference_bundle(root / "inference_bundle.pt", model, provenance={"seed": 90})
            loaded, info = load_inference_bundle(bundle)
            self.assertFalse(loaded.training)
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
            self.assertEqual(info.provenance["seed"], 90)
            original = model.state_dict()
            restored = loaded.state_dict()
            self.assertEqual(set(original), set(restored))
            self.assertTrue(all(torch.equal(original[key].cpu(), restored[key].cpu()) for key in original))


class MSDEvaluationTests(unittest.TestCase):
    def test_artifact_writer_is_readable_by_unified_evaluator(self) -> None:
        model, _ = _components()
        pipeline = MSDFlowInferencePipeline(model, top_m=2, environment_steps=1, normality_steps=1, output_size=16)
        prediction = pipeline.predict_features(torch.randn(2, 4, 8, 8), model.standardizer(torch.randn(2, 8)))
        with tempfile.TemporaryDirectory(prefix="msd_eval_") as temporary:
            root = Path(temporary)
            image = root / "image.png"
            mask = root / "mask.png"
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image)
            mask_array = np.zeros((16, 16), dtype=np.uint8)
            mask_array[4:8, 4:8] = 255
            Image.fromarray(mask_array).save(mask)
            records = [_record(image, "normal", 0), _record(image, "anomaly", 1, mask)]
            manifest = write_records_jsonl(records, root / "manifest.jsonl")
            output_dir = root / "predictions"
            writer = MSDPredictionArtifactWriter(output_dir)
            writer.write_batch(
                {
                    "path": [str(image), str(image)], "base_id": ["normal", "anomaly"],
                    "category": ["toy", "toy"], "dataset": ["mvtec_ad", "mvtec_ad"],
                    "domain": ["regular", "regular"], "variant_id": ["clean", "clean"],
                    "label": torch.tensor([0, 1]),
                },
                prediction,
                defect_threshold=float(prediction.defect_score.mean().item()),
            )
            writer.finalize({"test": True})
            result = evaluate_msd_prediction_directory(output_dir, manifest, method="msdflow", seed=1)
            self.assertTrue(result.audit["dual_score_fields_complete"])
            self.assertGreater(len(result.metric_rows), 0)

    def test_paired_fpr_comparison_detects_corrected_errors(self) -> None:
        with tempfile.TemporaryDirectory(prefix="msd_pair_") as temporary:
            root = Path(temporary)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            base_rows = []
            for index in range(6):
                base_rows.append(
                    {"base_id": f"id{index}", "variant_id": "clean", "category": "toy", "dataset": "robustad",
                     "domain": "test1", "label": 0, "threshold": 0.5, "image_score": 0.8 if index < 4 else 0.2}
                )
            with (reference / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for row in base_rows:
                    handle.write(json.dumps(row) + "\n")
            with (candidate / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for index, row in enumerate(base_rows):
                    changed = dict(row)
                    changed["image_score"] = 0.2 if index < 3 else row["image_score"]
                    handle.write(json.dumps(changed) + "\n")
            comparison = compare_prediction_directories(reference, candidate, bootstrap_samples=200)[0]
            self.assertLess(comparison.candidate_minus_reference, 0)
            self.assertEqual(comparison.corrected_by_candidate, 3)


if __name__ == "__main__":
    unittest.main()
