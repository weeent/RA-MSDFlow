"""RA-MSDFlow 的 prepare_all 模块。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from envfm.data.prepare import DatasetPreparationSpec, prepare_dataset
from envfm.training.checkpoint import atomic_write_json


def run(config_path: str | Path, *, overwrite: bool = False) -> Path:
    source = Path(config_path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), list):
        raise ValueError("preparation config must contain a datasets list")
    output_root = Path(config.get("output_root", "manifests")).resolve()
    summaries: list[dict[str, object]] = []
    # 步骤 1：固定随机种子。
    for value in config["datasets"]:
        if not isinstance(value, dict):
            raise TypeError("each dataset preparation entry must be an object")
        summaries.append(
            prepare_dataset(
                DatasetPreparationSpec.from_mapping(value),
                output_root=output_root,
                seed=int(config.get("seed", 9826)),
                val_ratio=float(config.get("val_ratio", 0.2)),
                image_size=int(config.get("image_size", 256)),
                train_variants=tuple(config.get("train_variants", ["clean", "exposure", "white_balance", "gradient"])),
                test_variants=tuple(config.get("test_variants", ["clean", "exposure", "white_balance", "gradient", "compound"])),
                reuse_raw_manifest=bool(config.get("reuse_raw_manifest", True)),
                overwrite=overwrite,
            )
        )
    # 步骤 2：按当前协议处理。
    output = Path(config.get("summary", output_root / f"preparation_seed{config.get('seed', 9826)}.json"))
    atomic_write_json(
        {
            "phase": "prepare_all_datasets",
            "config": str(source),
            "datasets": summaries,
            "all_audits_valid": all(bool(item["prepared_audit"]["valid"]) for item in summaries),  # type: ignore[index]
            "all_protocol_audits_valid": all(
                bool(item["prepared_protocol_audit"]["valid"]) for item in summaries  # type: ignore[index]
            ),
        },
        output,
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare all EnvFM datasets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = run(args.config, overwrite=args.overwrite)
    print(json.dumps({"preparation_summary": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
