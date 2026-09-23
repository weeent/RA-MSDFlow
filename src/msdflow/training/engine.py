"""供三个 MSD-Flow 训练阶段共享的轻量训练引擎。

该引擎只接收 train/normal-val loader，不提供 test loader 参数，从接口层阻止测试泄漏。
每个 epoch 写小型 JSON 指标；大型 checkpoint 按固定间隔和最后一轮写入。
支持 CUDA/AMP 训练和断点续训。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from envfm.training.checkpoint import (
    OPTIMIZER_STATE_STORAGE_DTYPES,
    atomic_torch_save,
    atomic_write_json,
    checkpoint_write_lock,
    pack_optimizer_state_dict,
    restore_optimizer_state_dict,
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    checkpoint_interval_epochs: int = 5
    optimizer_state_storage_dtype: str = "float16"
    checkpoint_io_lock_path: str | None = None
    amp: bool = True
    seed: int = 9826
    resume: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0 or self.checkpoint_interval_epochs <= 0:
            raise ValueError("epochs, learning_rate and checkpoint interval must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm < 0:
            raise ValueError("weight_decay and gradient_clip_norm must be non-negative")
        if self.optimizer_state_storage_dtype not in OPTIMIZER_STATE_STORAGE_DTYPES:
            raise ValueError(f"unsupported optimizer storage dtype: {self.optimizer_state_storage_dtype}")


@dataclass(frozen=True, slots=True)
class StageFitResult:
    stage: str
    run_directory: Path
    last_checkpoint: Path
    best_checkpoint: Path
    metrics_path: Path
    summary_path: Path
    best_epoch: int
    best_validation_loss: float


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


class StageTrainerBase:
    """定义统一循环；子类只实现 ``compute_loss`` 与 batch 字段约定。"""

    stage_name = "abstract"

    def __init__(
        self,
        model: nn.Module,
        *,
        run_directory: str | Path,
        config: TrainingConfig | None = None,
        device: str | torch.device = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.run_directory = Path(run_directory)
        self.config = config or TrainingConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable on this machine")
        self.model.to(self.device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("trainer requires at least one trainable parameter")
        self.optimizer = optimizer or torch.optim.AdamW(
            trainable, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        self.last_path = self.run_directory / "checkpoints" / "last.pt"
        self.best_path = self.run_directory / "checkpoints" / "best.pt"
        self.metrics_path = self.run_directory / "logs" / "epoch_metrics.jsonl"
        self.summary_path = self.run_directory / "logs" / "training_summary.json"
        existing = [path for path in (self.last_path, self.metrics_path) if path.exists()]
        if existing and not self.config.resume:
            raise FileExistsError(f"training artifacts already exist: {existing}")
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._set_seed(self.config.seed)
        # 只在 CUDA 上启用 fp16 autocast/scaler；CPU 路径保持纯 FP32，减少平台差异。
        self.amp_enabled = bool(self.config.amp and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _autocast(self):
        return (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if self.amp_enabled
            else nullcontext()
        )

    def compute_loss(self, batch: Mapping[str, object], *, training: bool) -> tuple[Tensor, Mapping[str, float]]:
        raise NotImplementedError

    @staticmethod
    def _assert_normal(labels: Tensor, field: str) -> None:
        if labels.numel() == 0 or not torch.all(labels == 0):
            raise ValueError(f"{field} contains non-normal labels; MSD-Flow training is normal-only")

    def _run_epoch(self, loader: Iterable[Mapping[str, object]], *, training: bool) -> dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        batches = 0
        samples = 0
        auxiliary_totals: dict[str, float] = {}
        context = nullcontext() if training else torch.no_grad()
        with context:
            for batch in loader:
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    loss, auxiliary = self.compute_loss(batch, training=training)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise RuntimeError(f"{self.stage_name} produced a non-finite scalar loss")
                batch_samples = self.batch_size(batch)
                if training:
                    self.scaler.scale(loss).backward()
                    if self.config.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                total_loss += float(loss.detach().item()) * batch_samples
                samples += batch_samples
                batches += 1
                for name, value in auxiliary.items():
                    auxiliary_totals[name] = auxiliary_totals.get(name, 0.0) + float(value) * batch_samples
        if batches == 0 or samples == 0:
            raise ValueError(f"{self.stage_name} loader is empty")
        return {
            "loss": total_loss / samples,
            "batches": float(batches),
            "samples": float(samples),
            **{name: value / samples for name, value in auxiliary_totals.items()},
        }

    def batch_size(self, batch: Mapping[str, object]) -> int:
        """默认从首个张量读取 batch size；子类可覆盖。"""

        for value in batch.values():
            if isinstance(value, Tensor) and value.ndim > 0:
                return int(value.shape[0])
        raise ValueError("batch contains no tensor with a batch dimension")

    def _checkpoint_state(self, epoch: int, best_epoch: int, best_loss: float) -> dict[str, object]:
        return {
            "format_version": 1,
            "stage": self.stage_name,
            "epoch": epoch,
            # 权重和优化器状态先复制到 CPU，避免 torch.save 长时间占用 CUDA stream。
            "model_state_dict": _cpu_state_dict(self.model),
            "optimizer_state_dict": pack_optimizer_state_dict(
                self.optimizer.state_dict(), self.config.optimizer_state_storage_dtype
            ),
            "optimizer_state_storage_dtype": self.config.optimizer_state_storage_dtype,
            "scaler_state_dict": self.scaler.state_dict(),
            "config": asdict(self.config),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _resume(self) -> tuple[int, int, float]:
        if not self.config.resume:
            return 0, -1, float("inf")
        if not self.last_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {self.last_path}")
        # RNG 状态必须保留在 CPU；模型和优化器随后会按参数设备自动恢复。
        try:
            state = torch.load(self.last_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(self.last_path, map_location="cpu")
        if state.get("stage") != self.stage_name:
            raise ValueError("checkpoint stage does not match trainer")
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(restore_optimizer_state_dict(state["optimizer_state_dict"]))
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to(self.device)
        self.scaler.load_state_dict(state.get("scaler_state_dict", {}))
        torch.set_rng_state(state["cpu_rng_state"])
        if self.device.type == "cuda" and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        return int(state["epoch"]) + 1, int(state["best_epoch"]), float(state["best_validation_loss"])

    def fit(
        self,
        train_loader: Iterable[Mapping[str, object]],
        normal_validation_loader: Iterable[Mapping[str, object]],
    ) -> StageFitResult:
        """训练并用正常验证 FM loss 选模；接口故意不接受测试集。"""

        start_epoch, best_epoch, best_loss = self._resume()
        pending_best: dict[str, Tensor] | None = None
        epoch_rows: list[dict[str, object]] = []
        for epoch in range(start_epoch, self.config.epochs):
            train_metrics = self._run_epoch(train_loader, training=True)
            # 每个 epoch 使用相同验证随机流，避免 t/noise 采样噪声掩盖模型变化。
            validation_devices = (
                [self.device.index if self.device.index is not None else torch.cuda.current_device()]
                if self.device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=validation_devices):
                torch.manual_seed(self.config.seed + 100_000)
                validation_metrics = self._run_epoch(normal_validation_loader, training=False)
            if validation_metrics["loss"] < best_loss:
                best_loss = validation_metrics["loss"]
                best_epoch = epoch
                pending_best = _cpu_state_dict(self.model)
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "normal_validation": validation_metrics,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
            }
            epoch_rows.append(row)
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

            persist = (epoch + 1) % self.config.checkpoint_interval_epochs == 0 or epoch == self.config.epochs - 1
            if persist:
                # 多卡独立任务可共享一个锁文件，防止机械盘同时承受多个大 checkpoint 写入。
                with checkpoint_write_lock(self.config.checkpoint_io_lock_path):
                    if pending_best is not None:
                        atomic_torch_save(
                            {
                                "format_version": 1,
                                "stage": self.stage_name,
                                "epoch": best_epoch,
                                "selection_metric": "normal_validation_loss",
                                "model_state_dict": pending_best,
                            },
                            self.best_path,
                        )
                        pending_best = None
                    atomic_torch_save(self._checkpoint_state(epoch, best_epoch, best_loss), self.last_path)

        if best_epoch < 0 or not math.isfinite(best_loss):
            raise RuntimeError("training finished without a finite validation result")
        summary = {
            "stage": self.stage_name,
            "device": str(self.device),
            "amp_enabled": self.amp_enabled,
            "epochs_requested": self.config.epochs,
            "epochs_executed_this_call": len(epoch_rows),
            "best_epoch": best_epoch,
            "best_normal_validation_loss": best_loss,
            "checkpoint_interval_epochs": self.config.checkpoint_interval_epochs,
            "optimizer_state_storage_dtype": self.config.optimizer_state_storage_dtype,
            "checkpoint_io_lock_path": self.config.checkpoint_io_lock_path,
            "last_checkpoint": str(self.last_path),
            "best_checkpoint": str(self.best_path),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "all_losses_finite": all(
                math.isfinite(float(row[split]["loss"]))
                for row in epoch_rows
                for split in ("train", "normal_validation")
            ),
        }
        atomic_write_json(summary, self.summary_path)
        return StageFitResult(
            stage=self.stage_name,
            run_directory=self.run_directory,
            last_checkpoint=self.last_path,
            best_checkpoint=self.best_path,
            metrics_path=self.metrics_path,
            summary_path=self.summary_path,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
        )
