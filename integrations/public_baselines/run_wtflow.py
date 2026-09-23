"""运行作者 WT-Flow，训练期间不访问目标测试集。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
    parser = argparse.ArgumentParser(description="Official WT-Flow protocol runner")
    parser.add_argument("--prepare-summary", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--checkpoint", help="已有 final.pt；提供后跳过 source-only 训练")
    args = parser.parse_args()

    prepared = read_json(args.prepare_summary)
    index_rows = read_index(prepared)
    category = str(prepared["config"]["category"])
    seed = int(prepared["config"]["reference_seed"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_all(seed)

    with author_repository(args.repository):
        import train_config as c
        import datasets.datasets as author_datasets
        from datasets import MVTecDataset
        from infer import generate_pdf
        from model import MiniUnet
        from models.extractors import build_extractor
        from post_process import post_process_single
        from reversed_flow import ReversedFlow

        if category not in author_datasets.MVTEC_CLASS_NAMES:
            author_datasets.MVTEC_CLASS_NAMES.append(category)
        c.device = torch.device(args.device)
        c.data_path = str(prepared["stage_root"])
        c.dataset = "mvtec"
        c.class_name = category
        c.input_size = (args.image_size, args.image_size)
        c.batch_size = args.batch_size
        c.workers = args.workers
        c.epochs = args.epochs
        c.step = args.steps
        c.BN_enable = False
        c.LN_enable = True
        c.PE_enable = False
        c.single_branch_enable = True
        c.id_branch = 2
        c.top_k = 0.03

        train_data = MVTecDataset(c, is_train=True)
        test_data = MVTecDataset(c, is_train=False)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=c.batch_size, shuffle=True, num_workers=c.workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=c.workers, pin_memory=True
        )
        extractor, _ = build_extractor(c)
        extractor = extractor.to(c.device).eval()
        for parameter in extractor.parameters():
            parameter.requires_grad_(False)
        channels = [256, 512, 1024][c.id_branch]
        model = MiniUnet(channels, c.base_channels).to(c.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=c.lr_adjust_epoch, gamma=c.gamma)
        flow = ReversedFlow()
        pool = nn.AvgPool2d(3, 2, 1) if c.pool_type == "avg" else nn.Identity()

        checkpoint = output / "author_run" / "final.pt"
        if args.checkpoint:
            saved = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(saved["model"])
        else:
            # 只训练固定 epoch 后的 final；不计算目标 test AUROC，也不据其选权重。
            for _ in range(c.epochs):
                model.train()
                for image, _, _ in train_loader:
                    image = image.to(c.device)
                    with torch.no_grad():
                        x0 = pool(extractor(image)[c.id_branch])
                        if c.LN_enable:
                            x0 = nn.functional.layer_norm(x0, x0.shape[1:])
                    time = torch.rand(x0.size(0), device=c.device)
                    xt, noise = flow.create_flow_rev(x0, time)
                    velocity = model(x=xt, t=time, y=None)
                    loss = flow.mse_loss_rev(velocity, x0, noise)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": c.epochs - 1}, checkpoint)

        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for batch_index, (image, _, _) in enumerate(test_loader):
                log_probability, _ = generate_pdf(
                    c, model, extractor, c.step, image, batch_index, None
                )
                outputs.append(log_probability)
        image_scores, _, anomaly_maps = post_process_single(c, outputs)
        index_by_path = rows_by_staged_path(index_rows)
        native: list[dict[str, object]] = []
        for path, score, anomaly_map in zip(test_data.x, image_scores, anomaly_maps):
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
