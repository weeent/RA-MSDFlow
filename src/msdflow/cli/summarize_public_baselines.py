"""汇总公开 baseline 逐作业指标，生成类别平衡的论文主表输入。

正式口径为：同一 seed 内先平均类别的环境，再对类别等权平均；最后在
reference seeds 上报告均值和样本标准差。这样 PCB 的两个环境不会让该类别
获得双倍权重，类别差异也不会被误写成随机种子方差。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


METRICS = ("image_auroc", "normal_fpr", "anomaly_tpr", "pixel_auroc", "aupro_005")


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("no completed public baseline metrics were found")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number(row: Mapping[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {None, ""}:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _category_balanced_seed_values(
    rows: Sequence[Mapping[str, str]], metric: str
) -> list[float]:
    """返回每个 seed 的类别平衡指标。

    输入逐作业行。每行对应一个 method/dataset/category/environment/seed。
    先在 ``(seed, category, environment)`` 内平均可能的重复值，再平均环境，
    随后对同一 seed 的类别等权平均。缺失像素指标不会以零填充，例如没有真实
    mask 的 PiledBags 会自然退出 P-AUROC/AUPRO 汇总。
    """

    by_seed_category_environment: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row, metric)
        if value is None:
            continue
        key = (row["reference_seed"], row["category"], row["environment"])
        by_seed_category_environment[key].append(value)

    by_seed_category: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (seed, category, _environment), values in by_seed_category_environment.items():
        by_seed_category[(seed, category)].append(float(np.mean(values)))

    by_seed: dict[str, list[float]] = defaultdict(list)
    for (seed, _category), environment_values in by_seed_category.items():
        by_seed[seed].append(float(np.mean(environment_values)))

    def _seed_key(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return [float(np.mean(by_seed[seed])) for seed in sorted(by_seed, key=_seed_key)]


def summarize(root: str | Path, output_directory: str | Path) -> tuple[Path, Path]:
    """输入 baseline 输出根目录，生成 long CSV 和类别平衡 dataset-macro CSV。"""

    root_path = Path(root).resolve()
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for path in sorted(root_path.rglob("metrics.csv")):
        # 只接受端到端入口的 final/metrics.csv，避开作者仓库自己的临时表。
        if path.parent.name != "final" or not (path.parent / "baseline_summary.json").is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row.get("method", ""), row.get("dataset", ""), row.get("category", ""),
                    row.get("environment", ""), row.get("reference_seed", ""),
                )
                if key in seen:
                    raise ValueError(f"duplicate completed baseline job: {key}")
                seen.add(key)
                rows.append({**row, "metrics_path": str(path)})
    rows.sort(
        key=lambda row: (
            row.get("method", ""), row.get("dataset", ""), row.get("category", ""),
            row.get("environment", ""), int(row.get("reference_seed", "0")),
        )
    )
    output = Path(output_directory).resolve()
    long_path = output / "public_baselines_long.csv"
    _write_csv(rows, long_path)

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["dataset"])].append(row)
    macro_rows: list[dict[str, object]] = []
    for (method, dataset), group in sorted(groups.items()):
        summary: dict[str, object] = {
            "method": method,
            "dataset": dataset,
            "jobs": len(group),
            "categories": len({row["category"] for row in group}),
            "environments": len({row["environment"] for row in group}),
            "seeds": len({row["reference_seed"] for row in group}),
            "aggregation": "environment_then_category_within_seed_then_seed_mean",
        }
        for metric in METRICS:
            seed_values = _category_balanced_seed_values(group, metric)
            summary[f"{metric}_mean"] = "" if not seed_values else float(np.mean(seed_values))
            summary[f"{metric}_std"] = (
                "" if len(seed_values) < 2 else float(np.std(seed_values, ddof=1))
            )
            # n 现在明确表示参与最终均值/标准差的 seed 数，不再表示作业数。
            summary[f"{metric}_n"] = len(seed_values)
        macro_rows.append(summary)
    macro_path = output / "public_baselines_dataset_macro.csv"
    _write_csv(macro_rows, macro_path)
    (output / "public_baselines_summary.json").write_text(
        json.dumps(
            {
                "root": str(root_path),
                "completed_jobs": len(rows),
                "methods": sorted({row["method"] for row in rows}),
                "datasets": sorted({row["dataset"] for row in rows}),
                "aggregation": "environment_then_category_within_seed_then_seed_mean",
                "long_table": str(long_path),
                "dataset_macro": str(macro_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return long_path, macro_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize completed public baseline jobs")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)
    long_path, macro_path = summarize(args.root, args.output_directory)
    print(json.dumps({"long_table": str(long_path), "dataset_macro": str(macro_path)}, indent=2))


if __name__ == "__main__":
    main()
