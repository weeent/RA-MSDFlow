"""MSD-Flow 多运行评估汇总 CLI。"""

from __future__ import annotations

import json
from typing import Sequence

from msdflow.evaluation import build_evaluation_report

from .common import load_json_config, required


def run(config_path: str) -> object:
    config = load_json_config(config_path)
    runs = required(config, "runs")
    if not isinstance(runs, list):
        raise TypeError("runs must be a list")
    comparisons = config.get("comparisons", [])
    if not isinstance(comparisons, list):
        raise TypeError("comparisons must be a list")
    return build_evaluation_report(
        runs,
        required(config, "output_directory"),
        comparisons=comparisons,
        aupro_max_fpr=float(config.get("aupro_max_fpr", 0.05)),
    )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Summarize MSD-Flow predictions")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = run(args.config)
    print(json.dumps({"evaluation_summary": str(path)}, ensure_ascii=False, indent=2))
