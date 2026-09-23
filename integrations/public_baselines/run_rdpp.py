"""运行作者 RD++，用固定末轮模型替代 test-AUROC 选 best。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
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


def _init_rdpp_worker(worker_id: int) -> None:
    """执行 `_init_rdpp_worker` 所需的处理。"""

    del worker_id
    import cv2
    import numba

    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    numba.set_num_threads(1)
    # 设置 DataLoader 并行行为。
    seed = int(torch.initial_seed() % (2 ** 32))
    np.random.seed(seed)


def _loader_options(workers: int, *, batch_size: int, shuffle: bool) -> dict[str, object]:
    """执行 `_loader_options` 所需的处理。"""

    options: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        options.update({
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "prefetch_factor": 2,
            "worker_init_fn": _init_rdpp_worker,
        })
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description="Official RD++ protocol runner")
    parser.add_argument("--prepare-summary", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-cpu-threads", type=int, default=2)
    parser.add_argument("--checkpoint", help="已有 final.pt；提供后跳过 source-only 训练")
    args = parser.parse_args()

    prepared = read_json(args.prepare_summary)
    index_rows = read_index(prepared)
    category = str(prepared["config"]["category"])
    seed = int(prepared["config"]["reference_seed"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    # 处理 CUDA 设备兼容性。
    # 设置 DataLoader 并行行为。
    torch.set_num_threads(max(1, args.torch_cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # `main` 的实现说明。
        pass

    with author_repository(args.repository):
        import cv2
        import numba
        from dataset.dataset import MVTecDataset_test, MVTecDataset_train, get_data_transforms
        from model.de_resnet import de_wide_resnet50_2
        from model.resnet import wide_resnet50_2
        from utils.utils_test import cal_anomaly_map
        from utils.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion

        # 也覆盖 workers=0 的安全回退路径。
        cv2.setNumThreads(0)
        numba.set_num_threads(1)

        device = torch.device(args.device)
        if device.type == "cuda":
            # 作者损失内部使用 ``.to('cuda')``；先设当前卡以兼容多卡调度。
            torch.cuda.set_device(device)
        category_root = Path(prepared["category_root"])
        transform, gt_transform = get_data_transforms(256, 256)
        encoder, bottleneck = wide_resnet50_2(pretrained=True)
        encoder = encoder.to(device).eval()
        bottleneck = bottleneck.to(device)
        decoder = de_wide_resnet50_2(pretrained=False).to(device)
        projection = MultiProjectionLayer(base=64).to(device)
        projection_loss = Revisit_RDLoss()
        optimizer_projection = torch.optim.Adam(projection.parameters(), lr=0.001, betas=(0.5, 0.999))
        optimizer_distill = torch.optim.Adam(
            list(decoder.parameters()) + list(bottleneck.parameters()), lr=0.005, betas=(0.5, 0.999)
        )

        checkpoint = output / "author_run" / "final.pt"
        if args.checkpoint:
            saved = torch.load(args.checkpoint, map_location="cpu")
            projection.load_state_dict(saved["proj"])
            decoder.load_state_dict(saved["decoder"])
            bottleneck.load_state_dict(saved["bn"])
        else:
            train_data = MVTecDataset_train(root=str(category_root / "train"), transform=transform)
            # 先在父进程完成一次 Numba JIT，避免多个 spawn worker 同时编译并
            # 竞争同一个 cache 文件。小尺寸调用只用于预热，不进入训练样本。
            train_data.simplexNoise.rand_3d_octaves((3, 10, 10), 1, 0.6)
            train_loader = torch.utils.data.DataLoader(
                train_data,
                **_loader_options(args.workers, batch_size=args.batch_size, shuffle=True),
            )
            # 完整复用作者两个损失；唯一协议改动是训练阶段不构造 test loader 的前向。
            for _ in range(args.epochs):
                bottleneck.train()
                projection.train()
                decoder.train()
                optimizer_projection.zero_grad(set_to_none=True)
                optimizer_distill.zero_grad(set_to_none=True)
                for batch_index, (image, noisy_image, _) in enumerate(train_loader):
                    image = image.to(device)
                    noisy_image = noisy_image.to(device)
                    with torch.no_grad():
                        features = encoder(image)
                        noisy_features = encoder(noisy_image)
                    projected_noisy, projected_normal = projection(features, features_noise=noisy_features)
                    loss_projection = projection_loss(noisy_features, projected_noisy, projected_normal)
                    reconstructed = decoder(bottleneck(projected_normal))
                    loss_distill = loss_fucntion(features, reconstructed)
                    loss = loss_distill + 0.2 * loss_projection
                    loss.backward()
                    if (batch_index + 1) % 2 == 0 or batch_index + 1 == len(train_loader):
                        optimizer_projection.step()
                        optimizer_distill.step()
                        optimizer_projection.zero_grad(set_to_none=True)
                        optimizer_distill.zero_grad(set_to_none=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"proj": projection.state_dict(), "decoder": decoder.state_dict(), "bn": bottleneck.state_dict()},
                checkpoint,
            )
            # persistent worker 只服务训练；显式关闭后再建立测试 loader。
            iterator = getattr(train_loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
                train_loader._iterator = None

        encoder.eval()
        projection.eval()
        bottleneck.eval()
        decoder.eval()
        test_data = MVTecDataset_test(root=str(category_root), transform=transform, gt_transform=gt_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            **_loader_options(args.workers, batch_size=1, shuffle=False),
        )
        index_by_path = rows_by_staged_path(index_rows)
        native: list[dict[str, object]] = []
        with torch.no_grad():
            for loader_index, (image, _, _, _, _) in enumerate(test_loader):
                image = image.to(device)
                features = encoder(image)
                projected = projection(features)
                reconstructed = decoder(bottleneck(projected))
                anomaly_map, _ = cal_anomaly_map(features, reconstructed, image.shape[-1], amap_mode="a")
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                row = index_by_path[str(Path(test_data.img_paths[loader_index]).resolve())]
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
