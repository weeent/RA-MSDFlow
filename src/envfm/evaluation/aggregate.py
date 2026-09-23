"""RA-MSDFlow 的 aggregate 模块。"""

from __future__ import annotations

import csv
from collections import defaultdict
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


GROUP_KEYS = (
    "dataset",
    "method",
    "category",
    "environment",
    "environment_family",
    "is_primary",
    "metric",
)


def aggregate_across_seeds(
    rows: Iterable[Mapping[str, object]],
    *,
    group_keys: Sequence[str] = GROUP_KEYS,
) -> list[dict[str, object]]:
    """执行 `aggregate_across_seeds` 所需的处理。"""

    grouped: dict[tuple[object, ...], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        value = float(row["value"])
        if not math.isfinite(value):
            continue
        grouped[tuple(row[key] for key in group_keys)].append((int(row["seed"]), value))
    output: list[dict[str, object]] = []
    # 步骤 1：固定随机种子。
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        seeds = [seed for seed, _ in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed rows for aggregation group {key}")
        measurements = np.asarray([value for _, value in values], dtype=np.float64)
        output.append(
            {
                **dict(zip(group_keys, key)),
                "mean": float(measurements.mean()),
                "std": float(measurements.std(ddof=1)) if measurements.size > 1 else 0.0,
                "seeds": ";".join(str(seed) for seed in sorted(seeds)),
                "seed_count": measurements.size,
            }
        )
    return output


def write_csv(rows: Sequence[Mapping[str, object]], output_path: str | Path) -> Path:
    """执行 `write_csv` 所需的处理。"""

    if not rows:
        raise ValueError("cannot write an empty CSV")
    preferred = list(GROUP_KEYS) + ["seed", "value", "mean", "std", "seed_count", "seeds"]
    all_keys = {key for row in rows for key in row}
    columns = [key for key in preferred if key in all_keys] + sorted(all_keys.difference(preferred))
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    # 步骤 2：按当前协议处理。
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return target
