"""RA-MSDFlow 的 test_data_layer 模块。"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
import csv

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from envfm.data.dataset import IndustrialAnomalyDataset, fit_condition_standardizer
from envfm.data.manifest_builders import (
    MPDD_CATEGORIES,
    VISA_CATEGORIES,
    ManifestError,
    audit_records,
    build_mpdd_manifest,
    build_mvtec_ad_manifest,
    build_visa_manifest,
)
from envfm.data.photometric_augment import PhotometricParams, apply_photometric_transform
from envfm.data.photometric_descriptor import PhotometricDescriptor
from envfm.data.records import SampleRecord
from envfm.data.splits import expand_training_photometric_variants, stable_normal_split


def _write_rgb(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 16, 3), value, dtype=np.uint8), mode="RGB").save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((12, 16), dtype=np.uint8)
    array[2:5, 3:7] = 255
    Image.fromarray(array, mode="L").save(path)


class DataLayerTests(unittest.TestCase):
    def test_mvtec_builder_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mvtec"
            _write_rgb(root / "bottle" / "train" / "good" / "000.png")
            _write_rgb(root / "bottle" / "test" / "good" / "001.png")
            _write_rgb(root / "bottle" / "test" / "crack" / "002.png")
            _write_mask(root / "bottle" / "ground_truth" / "crack" / "002_mask.png")
            records = build_mvtec_ad_manifest(root)
            report = audit_records(records, verify_images=True)
            self.assertTrue(report.valid)
            self.assertEqual(report.total_records, 3)
            self.assertEqual(report.counts_by_label, {"0": 2, "1": 1})

    def test_visa_builder_reads_complete_official_one_class_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "VisA_pytorch" / "1cls"
            for category in VISA_CATEGORIES:
                _write_rgb(root / category / "train" / "good" / "train.png")
                _write_rgb(root / category / "test" / "good" / "normal.png")
                _write_rgb(root / category / "test" / "bad" / "anomaly.png")
                _write_mask(root / category / "ground_truth" / "bad" / "anomaly.png")
            records = build_visa_manifest(root.parent)
            report = audit_records(records, verify_images=True)
            self.assertTrue(report.valid)
            self.assertEqual(len(records), len(VISA_CATEGORIES) * 3)
            self.assertEqual({record.dataset for record in records}, {"visa"})
            self.assertEqual({record.category for record in records}, set(VISA_CATEGORIES))
            self.assertTrue(all(record.label == 0 for record in records if record.split == "train"))

    def test_visa_builder_reads_raw_images_through_official_split_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "VisA_20220922"
            rows: list[dict[str, str]] = []
            for category in VISA_CATEGORIES:
                train = f"{category}/Data/Images/Normal/train.JPG"
                normal = f"{category}/Data/Images/Normal/test.JPG"
                anomaly = f"{category}/Data/Images/Anomaly/anomaly.JPG"
                mask = f"{category}/Data/Masks/Anomaly/anomaly.png"
                _write_rgb(root / Path(*train.split("/")))
                _write_rgb(root / Path(*normal.split("/")))
                _write_rgb(root / Path(*anomaly.split("/")))
                _write_mask(root / Path(*mask.split("/")))
                rows.extend(
                    [
                        {"object": category, "split": "train", "label": "normal", "image": train, "mask": ""},
                        {"object": category, "split": "test", "label": "normal", "image": normal, "mask": ""},
                        {"object": category, "split": "test", "label": "anomaly", "image": anomaly, "mask": mask},
                    ]
                )
            split_file = root / "split_csv" / "1cls.csv"
            split_file.parent.mkdir(parents=True)
            with split_file.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("object", "split", "label", "image", "mask"))
                writer.writeheader()
                writer.writerows(rows)
            records = build_visa_manifest(root)
            report = audit_records(records, verify_images=True)
            self.assertTrue(report.valid)
            self.assertEqual(len(records), len(VISA_CATEGORIES) * 3)
            self.assertEqual(
                {record.metadata["source_protocol"] for record in records},
                {"official_1cls_csv"},
            )

    def test_mpdd_builder_maps_mixed_validation_to_test_and_rejects_train_defects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "MPDD"
            for category in MPDD_CATEGORIES:
                _write_rgb(root / category / "train" / "good" / "train.png")
                _write_rgb(root / category / "validation" / "good" / "normal.png")
                _write_rgb(root / category / "validation" / "scratch" / "anomaly.png")
                _write_mask(root / category / "ground_truth" / "scratch" / "anomaly_mask.png")
            records = build_mpdd_manifest(root)
            report = audit_records(records, verify_images=True)
            self.assertTrue(report.valid)
            self.assertEqual(len(records), len(MPDD_CATEGORIES) * 3)
            held_out = [record for record in records if record.official_split == "validation"]
            self.assertTrue(held_out)
            self.assertTrue(all(record.split == "test" for record in held_out))
            _write_rgb(root / MPDD_CATEGORIES[0] / "train" / "scratch" / "contamination.png")
            with self.assertRaisesRegex(ManifestError, "outside train/good"):
                build_mpdd_manifest(root)

    def test_split_and_expansion_never_leak_base_ids(self) -> None:
        records = [
            SampleRecord(
                dataset="toy", category="object", image_path=f"unused/{index}.png", label=0,
                defect_type="good", split="train", official_split="train", domain="regular",
                base_id=f"toy/object/train/{index}.png",
            )
            for index in range(10)
        ]
        split_once = stable_normal_split(records, val_ratio=0.2, seed=13)
        split_twice = stable_normal_split(records, val_ratio=0.2, seed=13)
        self.assertEqual(split_once, split_twice)
        expanded = expand_training_photometric_variants(split_once, seed=13)
        train_ids = {record.base_id for record in expanded if record.split == "train"}
        val_ids = {record.base_id for record in expanded if record.split == "val"}
        self.assertFalse(train_ids.intersection(val_ids))
        self.assertEqual(sum(record.split == "train" for record in expanded), 8 * 4)
        self.assertEqual(sum(record.split == "val" for record in expanded), 2)

    def test_split_drops_non_normal_rows_from_eligible_train_only(self) -> None:
        # `test_split_drops_non_normal_rows_from_eligible_train_only` 的实现说明。
        # `test_split_drops_non_normal_rows_from_eligible_train_only` 的实现说明。
        # 按评估协议处理。
        normal_train = [
            SampleRecord(
                dataset="toy", category="object", image_path=f"unused/good_{index}.png", label=0,
                defect_type="good", split="train", official_split="train", domain="regular",
                base_id=f"toy/object/train/good_{index}.png",
            )
            for index in range(4)
        ]
        anomalous_train = SampleRecord(
            dataset="toy", category="object", image_path="unused/train_defect.png", label=1,
            defect_type="defect", split="train", official_split="train", domain="regular",
            base_id="toy/object/train/train_defect.png",
        )
        anomalous_test = SampleRecord(
            dataset="toy", category="object", image_path="unused/test_defect.png", label=1,
            defect_type="defect", split="test", official_split="test", domain="lighting",
            base_id="toy/object/test/test_defect.png",
        )

        split_records = stable_normal_split(
            [*normal_train, anomalous_train, anomalous_test],
            val_ratio=0.25,
            seed=13,
        )

        self.assertNotIn(anomalous_train.base_id, {record.base_id for record in split_records})
        self.assertIn(anomalous_test, split_records)
        self.assertTrue(all(record.label == 0 for record in split_records if record.split in {"train", "val"}))

    def test_photometric_descriptor_reacts_to_named_environment_changes(self) -> None:
        image = torch.full((3, 64, 64), 0.45, dtype=torch.float32)
        descriptor = PhotometricDescriptor()
        baseline = descriptor(image)[0]
        exposure = descriptor(apply_photometric_transform(image, PhotometricParams(variant="exposure", exposure_ev=0.5)))[0]
        white_balance = descriptor(
            apply_photometric_transform(image, PhotometricParams(variant="white_balance", red_gain=1.2, blue_gain=0.9))
        )[0]
        gradient = descriptor(
            apply_photometric_transform(image, PhotometricParams(variant="gradient", gradient_x=0.3, gradient_y=-0.2))
        )[0]
        self.assertGreater(exposure[0].item(), baseline[0].item())
        self.assertGreater(white_balance[4].item(), baseline[4].item())
        self.assertGreater(abs(gradient[6].item()), abs(baseline[6].item()) + 1e-4)
        self.assertTrue(torch.isfinite(torch.stack((baseline, exposure, white_balance, gradient))).all())

    def test_dataset_keeps_mask_geometry_and_separates_raw_from_normalized_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            _write_rgb(image_path)
            _write_mask(mask_path)
            record = SampleRecord(
                dataset="toy", category="object", image_path=str(image_path), label=1,
                defect_type="crack", split="test", official_split="test", domain="lighting",
                base_id="toy/object/test/image.png", mask_path=str(mask_path),
                augmentation={"variant": "exposure", "exposure_ev": 0.5},
                metadata={"mask_expected": True},
            )
            dataset = IndustrialAnomalyDataset([record], image_size=32)
            sample = dataset[0]
            clean_sample = IndustrialAnomalyDataset([record.with_updates(augmentation={})], image_size=32)[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["photo_condition"].shape), (8,))
            self.assertTrue(sample["has_mask"].item())
            self.assertTrue(torch.equal(sample["mask"], clean_sample["mask"]))
            self.assertFalse(torch.equal(sample["image"], sample["image_raw"]))

    def test_condition_statistics_only_use_normal_train_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.png"
            test_path = root / "test.png"
            _write_rgb(train_path, 80)
            _write_rgb(test_path, 220)
            records = [
                SampleRecord("toy", "object", str(train_path), 0, "good", "train", "train", "regular", "toy/train.png"),
                SampleRecord("toy", "object", str(test_path), 0, "test", "test", "test", "lighting", "toy/test.png"),
            ]
            dataset = IndustrialAnomalyDataset(records, image_size=32, normalize=False)
            standardizer, summary = fit_condition_standardizer(dataset)
            self.assertEqual(summary["selected_records"], 1)
            self.assertEqual(int(standardizer.count.item()), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
