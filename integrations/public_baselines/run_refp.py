"""运行作者 ReFP-AD，并按统一协议导出逐图原始分数。

训练只读取 ``mvtec_stage/<category>/train/good``；reference 和评价图只在
训练完成后推理，因此不会使用目标异常选 checkpoint。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    author_repository,
    prediction_rows,
    read_index,
    read_json,
    save_map,
    seed_all,
    write_native_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official ReFP-AD protocol runner")
    parser.add_argument("--prepare-summary", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", help="已有 final.pt；提供后跳过 source-only 训练")
    parser.add_argument("--flow-epochs", type=int)
    parser.add_argument("--ebm-epochs", type=int)
    args = parser.parse_args()

    prepared = read_json(args.prepare_summary)
    index_rows = read_index(prepared)
    config = prepared["config"]
    category = str(config["category"])
    seed = int(config["reference_seed"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_all(seed)

    with author_repository(args.repository):
        from config import CFG, Mode
        from pipeline import build_pipeline

        cfg = CFG(
            dataset="MVTec",
            data_path=str(prepared["stage_root"]),
            out_dir=str(output / "author_run"),
            cache_dir=output / "token_cache",
            mode=Mode.PER_CATEGORY,
            seed=seed,
            device=args.device,
        )
        # CFG 默认放入 MVTec 15 类；本协议每个作业只训练当前类别。
        cfg.categories = [category]
        if args.flow_epochs is not None:
            cfg.flow_epochs = args.flow_epochs
        if args.ebm_epochs is not None:
            cfg.epochs = args.ebm_epochs
        pipe = build_pipeline(cfg)
        if args.checkpoint:
            pipe.load_checkpoint(args.checkpoint)
        else:
            pipe.train()
            pipe.save_checkpoint(output / "author_run" / "final.pt")

        native: list[dict[str, object]] = []
        for row in prediction_rows(index_rows):
            image_score, anomaly_map, _ = pipe.inference(Path(row["staged_image_path"]), category=category)
            native.append(
                {
                    "base_id": row["base_id"],
                    "image_score": float(image_score),
                    "anomaly_map_path": save_map(anomaly_map, output, str(row["base_id"])),
                }
            )
    write_native_predictions(native, output / "native_predictions.jsonl")


if __name__ == "__main__":
    main()
