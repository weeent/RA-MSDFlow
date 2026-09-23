"""少样本正常参考引导的多中心 OT-CFM 特征运输。

本模块只解决一件事：利用某个目标拍摄环境的少量已确认正常图，把目标域 patch
特征运输到源域正常特征分布。它不读取异常样本，也不替换已有冻结异常评分器。

输入
----
* 源域正常特征图 ``[N_s,C,H,W]``；
* 目标环境正常参考特征图 ``[N_t,C,H,W]``；
* 推理时待检测特征图 ``[B,C,H,W]``。

输出
----
* 一个小型 patch 速度场 checkpoint；
* 运输后的特征图，形状与输入完全一致；
* 训练过程的 OT 成本、CFM 损失和恒等约束统计。

中间产物
--------
* 由源图像级特征聚类得到的多中心、均衡源 patch bank；
* reference CORAL 粗对齐统计；
* 每步 minibatch Sinkhorn 耦合与由它采样的 CFM 端点对。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from envfm.training.checkpoint import atomic_torch_save


FORMAT_VERSION = 1


def feature_maps_to_patches(features: Tensor) -> Tensor:
    """把 ``[B,C,H,W]`` 变成逐空间位置的 ``[B*H*W,C]`` patch 矩阵。"""

    if features.ndim != 4 or not torch.is_floating_point(features):
        raise ValueError("features must be a floating-point [B,C,H,W] tensor")
    return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])


def patches_to_feature_maps(patches: Tensor, shape: tuple[int, int, int, int]) -> Tensor:
    """恢复 patch 矩阵；``shape`` 必须是原始 ``(B,C,H,W)``。"""

    batch, channels, height, width = shape
    if patches.shape != (batch * height * width, channels):
        raise ValueError("patch matrix and requested feature-map shape are inconsistent")
    return patches.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


@dataclass(slots=True)
class BalancedSourcePatchBank:
    """按源域正常模式均衡保存的 patch bank；张量常驻 CPU，避免长期占显存。"""

    patches: Tensor
    mode_ids: Tensor
    image_mode_ids: Tensor
    centers: Tensor

    @property
    def n_modes(self) -> int:
        return int(self.centers.shape[0])

    def sample(self, count: int, *, generator: torch.Generator) -> Tensor:
        """从每个模式取近似相同数量的 patch，防止大模式淹没小模式。"""

        if count <= 0:
            raise ValueError("sample count must be positive")
        per_mode = math.ceil(count / self.n_modes)
        pieces: list[Tensor] = []
        for mode in range(self.n_modes):
            available = torch.nonzero(self.mode_ids == mode, as_tuple=False).flatten()
            if available.numel() == 0:
                continue
            draw = torch.randint(available.numel(), (per_mode,), generator=generator)
            pieces.append(self.patches[available[draw]])
        if not pieces:
            raise RuntimeError("source patch bank contains no usable mode")
        return torch.cat(pieces, dim=0)[:count]

    def summary(self) -> dict[str, object]:
        counts = torch.bincount(self.mode_ids, minlength=self.n_modes)
        return {
            "patches": int(self.patches.shape[0]),
            "channels": int(self.patches.shape[1]),
            "modes": self.n_modes,
            "images_per_mode": torch.bincount(self.image_mode_ids, minlength=self.n_modes).tolist(),
            "patches_per_mode": counts.tolist(),
            "finite": bool(torch.isfinite(self.patches).all()),
        }


def _kmeans(features: Tensor, n_modes: int, *, seed: int, iterations: int = 25) -> tuple[Tensor, Tensor]:
    """在少量图像级向量上执行确定性余弦 k-means，不引入 sklearn 依赖。"""

    if features.ndim != 2 or features.shape[0] < n_modes:
        raise ValueError("k-means needs [N,D] with N >= n_modes")
    values = F.normalize(features.float().cpu(), dim=1)
    generator = torch.Generator().manual_seed(seed)
    centers = values[torch.randperm(values.shape[0], generator=generator)[:n_modes]].clone()
    assignment = torch.zeros(values.shape[0], dtype=torch.long)
    for _ in range(iterations):
        updated = (1.0 - values @ centers.transpose(0, 1)).argmin(1)
        if torch.equal(updated, assignment) and _ > 0:
            break
        assignment = updated
        for mode in range(n_modes):
            members = values[assignment == mode]
            if members.numel():
                centers[mode] = F.normalize(members.mean(0), dim=0)
    return assignment, centers


def build_balanced_source_bank(
    source_features: Tensor,
    *,
    n_modes: int = 4,
    max_patches: int = 40000,
    seed: int = 9826,
) -> BalancedSourcePatchBank:
    """先按图像内容发现正常中心，再为每个中心保留等量空间 patch。"""

    if source_features.ndim != 4 or source_features.shape[0] < 1:
        raise ValueError("source_features must be non-empty [N,C,H,W]")
    n_modes = min(int(n_modes), int(source_features.shape[0]))
    image_embedding = source_features.detach().float().mean(dim=(2, 3)).cpu()
    image_mode_ids, centers = _kmeans(image_embedding, n_modes, seed=seed)
    generator = torch.Generator().manual_seed(seed + 17)
    per_mode_limit = max(1, int(max_patches) // n_modes)
    patch_pieces: list[Tensor] = []
    mode_pieces: list[Tensor] = []
    for mode in range(n_modes):
        mode_features = source_features[image_mode_ids == mode]
        patches = feature_maps_to_patches(mode_features.detach().float().cpu())
        if patches.shape[0] > per_mode_limit:
            index = torch.randperm(patches.shape[0], generator=generator)[:per_mode_limit]
            patches = patches[index]
        patch_pieces.append(patches)
        mode_pieces.append(torch.full((patches.shape[0],), mode, dtype=torch.long))
    return BalancedSourcePatchBank(
        patches=torch.cat(patch_pieces),
        mode_ids=torch.cat(mode_pieces),
        image_mode_ids=image_mode_ids,
        centers=centers,
    )


def sinkhorn_coupling(
    target: Tensor,
    source: Tensor,
    *,
    epsilon: float = 0.07,
    iterations: int = 30,
) -> tuple[Tensor, Tensor]:
    """计算均匀边缘的熵正则 minibatch OT 耦合。

    成本采用归一化特征的余弦距离。返回 ``(coupling, cost)``，其中 coupling
    的行列和分别接近 ``1/N_t`` 与 ``1/N_s``。
    """

    if target.ndim != 2 or source.ndim != 2 or target.shape[1] != source.shape[1]:
        raise ValueError("target/source must be [N,C] matrices with the same C")
    if epsilon <= 0 or iterations <= 0:
        raise ValueError("epsilon and iterations must be positive")
    x = F.normalize(target.float(), dim=1)
    y = F.normalize(source.float(), dim=1)
    cost = (1.0 - x @ y.transpose(0, 1)).clamp_min(0.0)
    log_kernel = -cost / float(epsilon)
    log_a = torch.full((target.shape[0],), -math.log(target.shape[0]), device=target.device)
    log_b = torch.full((source.shape[0],), -math.log(source.shape[0]), device=source.device)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    coupling = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])
    return coupling, cost


def sample_ot_pairs(
    target: Tensor,
    source: Tensor,
    *,
    epsilon: float,
    iterations: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """每个目标 patch 按 Sinkhorn 行条件分布抽一个源端点。"""

    with torch.no_grad():
        coupling, cost = sinkhorn_coupling(target, source, epsilon=epsilon, iterations=iterations)
        probabilities = coupling / coupling.sum(1, keepdim=True).clamp_min(1e-12)
        # torch.multinomial 使用当前设备 RNG；显式 generator 使训练可重放。
        source_index = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        endpoint = source[source_index]
        selected_cost = cost.gather(1, source_index[:, None]).mean()
        entropy = -(coupling.clamp_min(1e-12) * coupling.clamp_min(1e-12).log()).sum()
    return target, endpoint, {
        "ot_selected_cost": float(selected_cost.item()),
        "ot_plan_entropy": float(entropy.item()),
        "ot_row_error": float((coupling.sum(1) - 1.0 / target.shape[0]).abs().max().item()),
        "ot_col_error": float((coupling.sum(0) - 1.0 / source.shape[0]).abs().max().item()),
    }


def sinusoidal_time_embedding(time: Tensor, dimension: int) -> Tensor:
    """把 CFM 连续时间 ``[B]`` 编码为正余弦向量。"""

    if time.ndim != 1 or dimension <= 0 or dimension % 2:
        raise ValueError("time must be [B] and dimension must be positive/even")
    half = dimension // 2
    frequencies = torch.exp(
        torch.arange(half, device=time.device, dtype=time.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = time[:, None] * frequencies[None, :] * 1000.0
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class PatchVelocityField(nn.Module):
    """对每个空间 patch 共享参数的小型残差速度场。"""

    def __init__(self, feature_dim: int, *, hidden_dim: int = 256, time_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        if min(feature_dim, hidden_dim, time_dim, depth) <= 0 or time_dim % 2:
            raise ValueError("dimensions/depth must be positive and time_dim even")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.depth = int(depth)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.time_projection = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(depth)
        )
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        # 零初始化保证训练开始时是恒等运输，避免随机网络立即破坏冻结特征。
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, patches: Tensor, time: Tensor) -> Tensor:
        if patches.ndim != 2 or patches.shape[1] != self.feature_dim:
            raise ValueError("patches must be [N,feature_dim]")
        if time.shape != (patches.shape[0],):
            raise ValueError("time must be [N]")
        context = self.time_projection(sinusoidal_time_embedding(time, self.time_dim))
        hidden = self.input_projection(patches) + context
        for block in self.blocks:
            hidden = hidden + block(hidden + context)
        return self.output_projection(F.silu(hidden))


class ReferenceOTCFM(nn.Module):
    """CORAL 粗对齐 + OT-CFM 残差运输器。

    CORAL 只利用 source normal 与 target reference 的固定统计量，负责均值/协方差级
    归位；速度场学习剩余的非线性、多中心分布差异。
    """

    def __init__(self, feature_dim: int, *, hidden_dim: int = 256, time_dim: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.velocity = PatchVelocityField(feature_dim, hidden_dim=hidden_dim, time_dim=time_dim, depth=depth)
        self.register_buffer("source_mean", torch.zeros(feature_dim))
        self.register_buffer("source_std", torch.ones(feature_dim))
        self.register_buffer("target_mean", torch.zeros(feature_dim))
        self.register_buffer("coral_transform", torch.eye(feature_dim))
        self.register_buffer("statistics_fitted", torch.tensor(False, dtype=torch.bool))

    @property
    def feature_dim(self) -> int:
        return self.velocity.feature_dim

    @torch.no_grad()
    def set_statistics(
        self,
        *,
        source_mean: Tensor,
        source_std: Tensor,
        target_mean: Tensor,
        coral_transform: Tensor,
    ) -> None:
        """写入 reference-only 粗对齐统计；之后它们随 checkpoint 保存。"""

        vector_shape = (self.feature_dim,)
        matrix_shape = (self.feature_dim, self.feature_dim)
        if source_mean.shape != vector_shape or source_std.shape != vector_shape or target_mean.shape != vector_shape:
            raise ValueError("alignment means/std have the wrong feature dimension")
        if coral_transform.shape != matrix_shape:
            raise ValueError("CORAL transform has the wrong shape")
        self.source_mean.copy_(source_mean.to(self.source_mean))
        self.source_std.copy_(source_std.to(self.source_std).clamp_min(1e-6))
        self.target_mean.copy_(target_mean.to(self.target_mean))
        self.coral_transform.copy_(coral_transform.to(self.coral_transform))
        self.statistics_fitted.fill_(True)

    def coarse_align(self, target_patches: Tensor) -> Tensor:
        if not bool(self.statistics_fitted.item()):
            raise RuntimeError("reference alignment statistics are not fitted")
        return (target_patches - self.target_mean) @ self.coral_transform + self.source_mean

    def normalize_source(self, source_patches: Tensor) -> Tensor:
        return (source_patches - self.source_mean) / self.source_std

    def denormalize_source(self, patches: Tensor) -> Tensor:
        return patches * self.source_std + self.source_mean

    def prepare_target(self, target_patches: Tensor) -> Tensor:
        return self.normalize_source(self.coarse_align(target_patches))

    def cfm_loss(self, target_start: Tensor, source_endpoint: Tensor, time: Tensor) -> Tensor:
        """线性条件路径 ``z_t=(1-t)x_t+t x_s`` 的速度回归损失。"""

        path = (1.0 - time[:, None]) * target_start + time[:, None] * source_endpoint
        desired_velocity = source_endpoint - target_start
        return F.mse_loss(self.velocity(path, time), desired_velocity)

    @torch.inference_mode()
    def transport_patches(self, target_patches: Tensor, *, steps: int = 8, chunk_size: int = 4096) -> Tensor:
        """Euler 积分目标 patch；分块只节省显存，不改变逐 patch 结果。"""

        if steps <= 0 or chunk_size <= 0:
            raise ValueError("steps and chunk_size must be positive")
        self.eval()
        prepared = self.prepare_target(target_patches)
        outputs: list[Tensor] = []
        dt = 1.0 / steps
        for start in range(0, prepared.shape[0], chunk_size):
            state = prepared[start : start + chunk_size]
            for step in range(steps):
                time = torch.full((state.shape[0],), step / steps, device=state.device, dtype=state.dtype)
                state = state + dt * self.velocity(state, time)
            outputs.append(self.denormalize_source(state))
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def transport_features(self, features: Tensor, *, steps: int = 8, chunk_size: int = 4096) -> Tensor:
        shape = tuple(int(v) for v in features.shape)
        patches = feature_maps_to_patches(features)
        transported = self.transport_patches(patches, steps=steps, chunk_size=chunk_size)
        return patches_to_feature_maps(transported, shape)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class OTCFMTrainingConfig:
    steps: int = 2000
    patch_batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    sinkhorn_epsilon: float = 0.07
    sinkhorn_iterations: int = 30
    identity_weight: float = 0.01
    gradient_clip_norm: float = 1.0
    amp: bool = True
    log_interval: int = 50
    checkpoint_interval: int = 250


def train_reference_otcfm(
    model: ReferenceOTCFM,
    source_bank: BalancedSourcePatchBank,
    target_reference_patches: Tensor,
    *,
    config: OTCFMTrainingConfig,
    device: torch.device,
    seed: int,
    output_directory: str | Path,
    resume: bool = True,
) -> tuple[list[dict[str, float]], Path]:
    """训练一个目标环境的 OT-CFM；只读取正常 source/reference patch。"""

    if target_reference_patches.ndim != 2 or target_reference_patches.shape[1] != model.feature_dim:
        raise ValueError("target reference patches have the wrong shape")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.velocity.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    cpu_generator = torch.Generator().manual_seed(seed)
    device_generator = torch.Generator(device=device).manual_seed(seed + 1)
    history: list[dict[str, float]] = []
    first_step = 1
    last_path = output / "last.pt"
    if resume and last_path.is_file():
        # checkpoint 必须在 CPU 加载，RNG ByteTensor 不能被 map 到 CUDA。
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        cpu_generator.set_state(payload["cpu_generator_state"])
        # CUDA Generator 的 state 仍是 CPU ByteTensor；搬到 GPU 会触发与旧训练器相同的恢复错误。
        device_generator.set_state(payload["device_generator_state"])
        history = list(payload.get("history", []))
        first_step = int(payload["step"]) + 1

    # CORAL 是固定的 1024x1024 线性变换；只预计算一次，避免在 2000 个优化步重复大矩阵乘法。
    prepared_pieces: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, target_reference_patches.shape[0], 4096):
            raw = target_reference_patches[start : start + 4096].to(
                device, non_blocking=device.type == "cuda"
            )
            prepared_pieces.append(model.prepare_target(raw).float().cpu())
    target_cpu = torch.cat(prepared_pieces)
    model.train()
    for step in range(first_step, config.steps + 1):
        target_index = torch.randint(target_cpu.shape[0], (config.patch_batch_size,), generator=cpu_generator)
        target_start = target_cpu[target_index].to(device, non_blocking=device.type == "cuda")
        source_raw = source_bank.sample(config.patch_batch_size, generator=cpu_generator).to(
            device, non_blocking=device.type == "cuda"
        )
        source_endpoint = model.normalize_source(source_raw)
        paired_target, paired_source, ot_stats = sample_ot_pairs(
            target_start.detach(),
            source_endpoint.detach(),
            epsilon=config.sinkhorn_epsilon,
            iterations=config.sinkhorn_iterations,
            generator=device_generator,
        )
        time = torch.rand(config.patch_batch_size, generator=device_generator, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=config.amp and device.type == "cuda"):
            cfm = model.cfm_loss(paired_target, paired_source, time)
            identity_time = torch.rand(config.patch_batch_size, generator=device_generator, device=device)
            identity = model.velocity(source_endpoint, identity_time).square().mean()
            loss = cfm + config.identity_weight * identity
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.velocity.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        record = {
            "step": float(step),
            "loss": float(loss.detach().item()),
            "cfm_loss": float(cfm.detach().item()),
            "identity_loss": float(identity.detach().item()),
            "gradient_norm": float(torch.as_tensor(gradient_norm).item()),
            **ot_stats,
        }
        if step == 1 or step % config.log_interval == 0 or step == config.steps:
            history.append(record)
        if step % config.checkpoint_interval == 0 or step == config.steps:
            atomic_torch_save(
                {
                    "format_version": FORMAT_VERSION,
                    "artifact_role": "reference_otcfm_training_checkpoint",
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cpu_generator_state": cpu_generator.get_state(),
                    "device_generator_state": device_generator.get_state().cpu(),
                    "history": history,
                    "training_config": asdict(config),
                },
                last_path,
            )

    final_path = output / "adapter.pt"
    export_reference_otcfm(
        final_path,
        model,
        provenance={"seed": seed, "training_config": asdict(config), "history": history},
    )
    return history, final_path


def export_reference_otcfm(
    path: str | Path,
    model: ReferenceOTCFM,
    *,
    provenance: Mapping[str, object] | None = None,
) -> Path:
    """导出仅含推理所需参数的小 checkpoint。"""

    architecture = {
        "feature_dim": model.feature_dim,
        "hidden_dim": model.velocity.hidden_dim,
        "time_dim": model.velocity.time_dim,
        "depth": model.velocity.depth,
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "artifact_role": "reference_otcfm_adapter",
        "architecture": architecture,
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "provenance": dict(provenance or {}),
    }
    return atomic_torch_save(payload, path)


def load_reference_otcfm(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[ReferenceOTCFM, dict[str, object]]:
    """重建、严格加载并冻结一个 OT-CFM adapter。"""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("artifact_role") != "reference_otcfm_adapter" or int(payload.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError("unsupported reference OT-CFM checkpoint")
    architecture = payload["architecture"]
    model = ReferenceOTCFM(
        int(architecture["feature_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        time_dim=int(architecture["time_dim"]),
        depth=int(architecture["depth"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, dict(payload.get("provenance", {}))
