"""类别平衡公开 baseline 汇总器的 CPU 测试。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msdflow.cli.summarize_public_baselines import summarize


class CategoryBalancedSummaryTests(unittest.TestCase):
    def test_two_environment_category_does_not_receive_double_weight(self) -> None:
        """PCB 两环境均为0、MetalParts一环境为1时，类别宏平均必须为0.5。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (9827, 9828):
                for category, environment, value in (
                    ("PCB", "lighting", 0.0),
                    ("PCB", "white_balance", 0.0),
                    ("MetalParts", "lighting", 1.0),
                ):
                    final = root / f"{category}_{environment}_{seed}" / "final"
                    final.mkdir(parents=True)
                    row = {
                        "method": "wtflow", "dataset": "robustad", "category": category,
                        "environment": environment, "reference_seed": seed,
                        "image_auroc": value, "normal_fpr": value, "anomaly_tpr": value,
                        "pixel_auroc": value, "aupro_005": value,
                    }
                    with (final / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader(); writer.writerow(row)
                    (final / "baseline_summary.json").write_text(json.dumps({}), encoding="utf-8")
            _long, macro = summarize(root, root / "tables")
            with macro.open("r", encoding="utf-8", newline="") as handle:
                result = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(result["image_auroc_mean"]), 0.5)
            self.assertAlmostEqual(float(result["image_auroc_std"]), 0.0)
            self.assertEqual(int(result["image_auroc_n"]), 2)
            self.assertEqual(
                result["aggregation"],
                "environment_then_category_within_seed_then_seed_mean",
            )


if __name__ == "__main__":
    unittest.main()
