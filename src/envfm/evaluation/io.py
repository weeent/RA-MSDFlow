"""RA-MSDFlow 的 io 模块。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np


def read_prediction_rows(output_directory: str | Path) -> list[dict[str, object]]:
    source = Path(output_directory).resolve() / "predictions.jsonl"
    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"prediction row {line_number} is not an object")
            required = {"base_id", "category", "dataset", "domain", "variant_id", "label", "image_score"}
            missing = required.difference(value)
            if missing:
                raise ValueError(f"prediction row {line_number} is missing {sorted(missing)}")
            if not np.isfinite(float(value["image_score"])):
                raise ValueError(f"prediction row {line_number} has a non-finite image score")
            rows.append(value)
    if not rows:
        raise ValueError(f"prediction file is empty: {source}")
    return rows


def iter_prediction_maps(
    output_directory: str | Path,
    rows: list[Mapping[str, object]],
    *,
    preferred_key: str,
) -> Iterator[np.ndarray | None]:
    """执行 `iter_prediction_maps` 所需的处理。"""

    root = Path(output_directory).resolve()
    cached_path: Path | None = None
    cached_arrays: dict[str, np.ndarray] = {}
    for row in rows:
        if row.get("map_file") is None:
            yield None
            continue
        path = root / str(row["map_file"])
        if path != cached_path:
            # 步骤 1：按当前协议处理。
            with np.load(path) as archive:
                cached_arrays = {key: archive[key] for key in archive.files}
            cached_path = path
        keys = [preferred_key, "anomaly_map", "anomaly_map_add", "anomaly_map_mul"]
        selected = next((key for key in keys if key in cached_arrays), None)
        if selected is None:
            raise ValueError(f"map shard {path} contains none of the supported arrays")
        index = int(row.get("map_index", 0))
        yield np.asarray(cached_arrays[selected][index], dtype=np.float32)
