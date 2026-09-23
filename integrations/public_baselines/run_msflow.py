"""运行作者 MSFlow，禁用逐 epoch 测试并导出最终模型原始分数。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import (
    author_repository,
    read_index,
    read_json,
    rows_by_staged_path,
    save_map,
    seed_all,
    write_native_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official MSFlow protocol runner")
    parser.add_argument("--prepare-summary", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--meta-epochs", type=int, default=25)
    parser.add_argument("--sub-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint", help="已有 final.pt；提供后跳过 source-only 训练")
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
        import default as c
        import datasets as author_datasets
        from datasets import MVTecDataset
        from models.extractors import build_extractor
        from models.flow_models import build_msflow_model
        from post_process import post_process
        from train import inference_meta_epoch, train_meta_epoch
        from utils import load_weights, save_weights

        if category not in author_datasets.MVTEC_CLASS_NAMES:
            author_datasets.MVTEC_CLASS_NAMES.append(category)
        c.device = torch.device(args.device)
        c.data_path = str(prepared["stage_root"])
        c.class_name = category
        c.input_size = (args.image_size, args.image_size)
        c.meta_epochs = args.meta_epochs
        c.sub_epochs = args.sub_epochs
        c.batch_size = args.batch_size
        c.workers = args.workers
        c.amp_enable = args.amp
        c.pro_eval = False
        c.ckpt_dir = str(output / "author_run")

        train_dataset = MVTecDataset(c, is_train=True)
        test_dataset = MVTecDataset(c, is_train=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.workers, pin_memory=True
        )
        extractor, output_channels = build_extractor(c)
        extractor = extractor.to(c.device).eval()
        parallel_flows, fusion_flow = build_msflow_model(c, output_channels)
        parallel_flows = [flow.to(c.device) for flow in parallel_flows]
        fusion_flow = fusion_flow.to(c.device)
        parameters = list(fusion_flow.parameters())
        for flow in parallel_flows:
            parameters.extend(flow.parameters())
        optimizer = torch.optim.Adam(parameters, lr=c.lr)
        scaler = torch.cuda.amp.GradScaler() if c.amp_enable else None
        warmup = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=c.lr_warmup_from, end_factor=1.0,
                total_iters=c.lr_warmup_epochs * c.sub_epochs,
            )
            if c.lr_warmup else None
        )
        milestones = [value for value in c.lr_decay_milestones if value < c.meta_epochs * c.sub_epochs]
        decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, c.lr_decay_gamma) if milestones else None

        if args.checkpoint:
            load_weights(parallel_flows, fusion_flow, args.checkpoint)
        else:
            # 作者代码原本每个 epoch 读取 test 并据 AUROC 保存 best。本 runner 只保存固定末轮。
            for epoch in range(c.meta_epochs):
                train_meta_epoch(
                    c, epoch, train_loader, extractor, parallel_flows, fusion_flow,
                    parameters, optimizer, warmup, decay, scaler,
                )
            save_weights(c.meta_epochs - 1, parallel_flows, fusion_flow, "final", c.ckpt_dir)

        _, _, outputs_list, size_list = inference_meta_epoch(
            c, c.meta_epochs, test_loader, extractor, parallel_flows, fusion_flow
        )
        image_scores, _, anomaly_maps = post_process(c, size_list, outputs_list)
        index_by_path = rows_by_staged_path(index_rows)
        native: list[dict[str, object]] = []
        for path, score, anomaly_map in zip(test_dataset.x, image_scores, anomaly_maps):
            row = index_by_path[str(Path(path).resolve())]
            native.append(
                {
                    "base_id": row["base_id"],
                    "image_score": float(score),
                    "anomaly_map_path": save_map(np.asarray(anomaly_map), output, str(row["base_id"])),
                }
            )
    write_native_predictions(native, output / "native_predictions.jsonl")


if __name__ == "__main__":
    main()
