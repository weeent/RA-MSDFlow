"""运行 ADShift 作者公开的 GNL/DINL，并导出统一原始分数。

训练使用作者 AugMix/DINL；推理使用作者 EFDM test-time feature transfer，参考锚点
固定为 source train 的第一张正常图，不使用目标 reference 更新模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

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
    parser = argparse.ArgumentParser(description="Official GNL/ADShift protocol runner")
    parser.add_argument("--prepare-summary", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lambda-efdm", type=float, default=0.5)
    parser.add_argument("--checkpoint", help="已有 final.pt；提供后跳过 source-only 训练")
    args = parser.parse_args()

    prepared = read_json(args.prepare_summary)
    index_rows = read_index(prepared)
    seed = int(prepared["config"]["reference_seed"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_all(seed)

    with author_repository(args.repository):
        from dataset import AugMixDatasetMVTec
        from de_resnet import de_wide_resnet50_2
        from resnet import wide_resnet50_2 as train_wrn
        from resnet_TTA import wide_resnet50_2 as test_wrn
        from test import cal_anomaly_map
        from train_mvtec_DINL import loss_fucntion, loss_fucntion_last

        device = torch.device(args.device)
        if device.type == "cuda":
            # 作者 EFDM 实现内部使用 ``.to('cuda')``，需令当前卡与作业设备一致。
            torch.cuda.set_device(device)
        category_root = Path(prepared["category_root"])
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        resize_only = transforms.Compose([transforms.Resize((256, 256))])
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        train_data = ImageFolder(root=str(category_root / "train"), transform=resize_only)
        train_data = AugMixDatasetMVTec(train_data, preprocess)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

        encoder, bottleneck = train_wrn(pretrained=True)
        encoder = encoder.to(device).eval()
        bottleneck = bottleneck.to(device)
        decoder = de_wide_resnet50_2(pretrained=False).to(device)
        optimizer = torch.optim.Adam(
            list(decoder.parameters()) + list(bottleneck.parameters()), lr=0.005, betas=(0.5, 0.999)
        )
        checkpoint = output / "author_run" / "final.pt"
        if args.checkpoint:
            saved = torch.load(args.checkpoint, map_location="cpu")
            bottleneck.load_state_dict(saved["bn"])
            decoder.load_state_dict(saved["decoder"])
        else:
            for _ in range(args.epochs):
                bottleneck.train()
                decoder.train()
                for normal, augmix_image, gray_image in train_loader:
                    normal = normal.to(device)
                    augmix_image = augmix_image.to(device)
                    gray_image = gray_image.to(device)
                    with torch.no_grad():
                        normal_features = encoder(normal)
                        augmix_features = encoder(augmix_image)
                        gray_features = encoder(gray_image)
                    bn_normal = bottleneck(normal_features)
                    bn_augmix = bottleneck(augmix_features)
                    bn_gray = bottleneck(gray_features)
                    out_normal = decoder(bn_normal)
                    out_augmix = decoder(bn_augmix)
                    out_gray = decoder(bn_gray)
                    loss_bn = loss_fucntion([bn_normal], [bn_augmix]) + loss_fucntion([bn_normal], [bn_gray])
                    loss_last = loss_fucntion_last(out_normal, out_augmix) + loss_fucntion_last(out_normal, out_gray)
                    loss = 0.9 * loss_fucntion(normal_features, out_normal) + 0.05 * loss_bn + 0.05 * loss_last
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"bn": bottleneck.state_dict(), "decoder": decoder.state_dict()}, checkpoint)

        # 作者把训练与 TTA encoder 分成两个类；二者共享 ImageNet 主干和 checkpoint 中的 bn/decoder。
        tta_encoder, _ = test_wrn(pretrained=True)
        tta_encoder = tta_encoder.to(device).eval()
        bottleneck.eval()
        decoder.eval()
        source_rows = [row for row in index_rows if row["role"] == "source_train"]
        if not source_rows:
            raise ValueError("GNL needs one source-normal anchor")
        anchor = preprocess(Image.open(source_rows[0]["staged_image_path"]).convert("RGB").resize((256, 256)))
        anchor = anchor.unsqueeze(0).to(device)
        native: list[dict[str, object]] = []
        with torch.no_grad():
            for row in prediction_rows(index_rows):
                image = preprocess(Image.open(row["staged_image_path"]).convert("RGB").resize((256, 256)))
                image = image.unsqueeze(0).to(device)
                features = tta_encoder(image, anchor, "EFDM_test", lamda=args.lambda_efdm)
                reconstructed = decoder(bottleneck(features))
                anomaly_map, _ = cal_anomaly_map(features, reconstructed, 256, amap_mode="a")
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                native.append(
                    {
                        "base_id": row["base_id"],
                        "image_score": float(np.max(anomaly_map)),
                        "anomaly_map_path": save_map(anomaly_map, output, str(row["base_id"])),
                    }
                )
    write_native_predictions(native, output / "native_predictions.jsonl")


if __name__ == "__main__":
    main()
