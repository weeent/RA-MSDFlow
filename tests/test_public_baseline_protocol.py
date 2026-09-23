"""公开 baseline 薄适配层的 CPU 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from envfm.data.records import SampleRecord, write_records_jsonl
from msdflow.baselines.public_protocol import (
    BaselineProtocolConfig,
    finalize_public_baseline,
    prepare_public_baseline,
)


class PublicBaselineProtocolTests(unittest.TestCase):
    def _image(self, root: Path, name: str, value: int) -> Path:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(value, value, value)).save(path)
        return path

    def _manifest(
        self,
        root: Path,
        *,
        category: str = "PCB",
        include_anomaly_masks: bool = True,
    ) -> Path:
        rows: list[SampleRecord] = []
        for index in range(4):
            image = self._image(root, f"raw/source_{index}.jpg", 20 + index)
            rows.append(
                SampleRecord(
                    dataset="RobustAD", category=category, image_path=str(image), label=0,
                    defect_type="good", split="train", official_split="train", domain="clean",
                    base_id=f"source-{index}", variant_id="clean",
                )
            )
        for index in range(5):
            image = self._image(root, f"raw/normal_{index}.jpg", 40 + index)
            rows.append(
                SampleRecord(
                    dataset="RobustAD", category=category, image_path=str(image), label=0,
                    defect_type="good", split="test", official_split="test", domain="lighting",
                    base_id=f"normal-{index}", variant_id="lighting",
                )
            )
        for index in range(2):
            image = self._image(root, f"raw/anomaly_{index}.jpg", 220)
            mask = (
                self._image(root, f"raw/anomaly_{index}_mask.png", 255)
                if include_anomaly_masks
                else None
            )
            rows.append(
                SampleRecord(
                    dataset="RobustAD", category=category, image_path=str(image), label=1,
                    defect_type="scratch", split="test", official_split="test", domain="lighting",
                    base_id=f"anomaly-{index}",
                    mask_path=None if mask is None else str(mask),
                    variant_id="lighting",
                )
            )
        return write_records_jsonl(rows, root / "manifest.jsonl")

    def test_prepare_is_deterministic_and_stages_only_normal_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            config = BaselineProtocolConfig(
                method="wtflow", manifest=str(manifest), dataset="RobustAD", category="PCB",
                environment="lighting", target_domain="lighting", reference_shots=2,
                reference_seed=9827, output_directory=str(root / "job"), link_mode="copy",
            )
            first = prepare_public_baseline(config)
            first_summary = json.loads(first.read_text(encoding="utf-8"))
            second = prepare_public_baseline(config)
            second_summary = json.loads(second.read_text(encoding="utf-8"))
            self.assertEqual(first_summary["reference_base_ids"], second_summary["reference_base_ids"])
            self.assertEqual(first_summary["counts"]["source_train_normal"], 4)
            self.assertEqual(first_summary["counts"]["target_evaluation_normal"], 3)
            self.assertEqual(first_summary["counts"]["target_evaluation_anomaly"], 2)
            train_files = list((root / "job" / "mvtec_stage" / "PCB" / "train" / "good").glob("*.png"))
            self.assertEqual(len(train_files), 4)

    def test_prepare_writes_zero_masks_only_for_loader_compatibility(self) -> None:
        """PiledBags 无真值 mask 时，作者 loader 可读，统一指标仍知道标注缺失。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(
                root,
                category="PiledBags",
                include_anomaly_masks=False,
            )
            config = BaselineProtocolConfig(
                method="wtflow",
                manifest=str(manifest),
                dataset="RobustAD",
                category="PiledBags",
                environment="lighting",
                target_domain="lighting",
                reference_shots=2,
                reference_seed=9827,
                output_directory=str(root / "job"),
                link_mode="copy",
            )
            prepare_summary = prepare_public_baseline(config)
            prepared = json.loads(prepare_summary.read_text(encoding="utf-8"))
            index_rows = [
                json.loads(line)
                for line in Path(prepared["protocol_index"]).read_text(encoding="utf-8").splitlines()
            ]
            anomaly_rows = [row for row in index_rows if row["label"] == 1]

            self.assertEqual(prepared["counts"]["loader_compat_zero_masks"], 2)
            self.assertEqual(len(anomaly_rows), 2)
            for row in anomaly_rows:
                self.assertIsNone(row["original_mask_path"])
                self.assertEqual(row["staged_mask_kind"], "synthetic_zero_loader_compat")
                staged_mask = Path(row["staged_mask_path"])
                self.assertTrue(staged_mask.is_file())
                with Image.open(staged_mask) as image:
                    self.assertEqual(image.size, (8, 8))
                    self.assertIsNone(image.getbbox())

    def test_prepare_converts_grayscale_inputs_to_rgb(self) -> None:
        """AD2 同类别混合灰度/RGB 时，所有作者 loader 都收到三通道输入。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            Image.new("L", (8, 8), color=80).save(root / "raw" / "source_0.jpg")
            config = BaselineProtocolConfig(
                method="wtflow", manifest=str(manifest), dataset="RobustAD", category="PCB",
                environment="lighting", target_domain="lighting", reference_shots=2,
                reference_seed=9827, output_directory=str(root / "job"), link_mode="copy",
            )
            prepare_summary = prepare_public_baseline(config)
            prepared = json.loads(prepare_summary.read_text(encoding="utf-8"))
            index_rows = [
                json.loads(line)
                for line in Path(prepared["protocol_index"]).read_text(encoding="utf-8").splitlines()
            ]
            converted = [row for row in index_rows if row["materialized_as"] == "rgb_png_conversion"]
            self.assertEqual(prepared["counts"]["rgb_converted_images"], 1)
            self.assertEqual(len(converted), 1)
            with Image.open(converted[0]["staged_image_path"]) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (8, 8))

    def test_finalize_uses_reference_only_and_exports_raw_auroc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            config = BaselineProtocolConfig(
                method="msflow", manifest=str(manifest), dataset="RobustAD", category="PCB",
                environment="lighting", target_domain="lighting", reference_shots=2,
                reference_seed=9827, output_directory=str(root / "job"), link_mode="copy",
            )
            prepare_summary = prepare_public_baseline(config)
            prepared = json.loads(prepare_summary.read_text(encoding="utf-8"))
            index_rows = [json.loads(line) for line in Path(prepared["protocol_index"]).read_text(encoding="utf-8").splitlines()]
            native = root / "native.jsonl"
            with native.open("w", encoding="utf-8") as handle:
                for row in index_rows:
                    if row["role"] == "source_train":
                        continue
                    score = 10.0 if row["label"] == 1 else 0.1
                    handle.write(json.dumps({"base_id": row["base_id"], "image_score": score}) + "\n")
            summary = finalize_public_baseline(
                prepare_summary_path=prepare_summary,
                native_predictions_path=native,
                output_directory=root / "final",
            )
            result = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(result["metrics"]["image_auroc"], 1.0)
            self.assertEqual(result["calibration"]["sample_count"], 2)
            prediction_rows = (root / "final" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(prediction_rows), 5)


    def test_finalize_accepts_safe_internal_method_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            config = BaselineProtocolConfig(
                method="wtflow", manifest=str(manifest), dataset="RobustAD", category="PCB",
                environment="lighting", target_domain="lighting", reference_shots=2,
                reference_seed=9827, output_directory=str(root / "job"), link_mode="copy",
            )
            prepare_summary = prepare_public_baseline(config)
            prepared = json.loads(prepare_summary.read_text(encoding="utf-8"))
            index_rows = [json.loads(line) for line in Path(prepared["protocol_index"]).read_text(encoding="utf-8").splitlines()]
            native = root / "native.jsonl"
            with native.open("w", encoding="utf-8") as handle:
                for row in index_rows:
                    if row["role"] != "source_train":
                        handle.write(json.dumps({"base_id": row["base_id"], "image_score": float(row["label"])}) + "\n")
            summary = finalize_public_baseline(
                prepare_summary_path=prepare_summary,
                native_predictions_path=native,
                output_directory=root / "final",
                reported_method="coral_wtflow",
            )
            result = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(result["reported_method"], "coral_wtflow")
            self.assertEqual(result["metrics"]["method"], "coral_wtflow")
            self.assertEqual(result["baseline"]["key"], "wtflow")


if __name__ == "__main__":
    unittest.main()
