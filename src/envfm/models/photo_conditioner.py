"""RA-MSDFlow 的 photo_conditioner 模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .feature_preprocess import tensor_statistics


ConditionMode = Literal["none", "correct", "zero", "shuffle_train"]
VALID_CONDITION_MODES: tuple[ConditionMode, ...] = ("none", "correct", "zero", "shuffle_train")


@dataclass(slots=True)
class ConditionApplication:
    """`ConditionApplication` 组件。"""

    requested_mode: str
    applied_mode: str
    condition_was_provided: bool
    input_condition: Tensor
    effective_condition: Tensor
    embedding: Tensor
    permutation: Tensor

    def summary(self) -> dict[str, object]:
        """执行 `summary` 所需的处理。"""

        identity = torch.arange(self.permutation.numel(), device=self.permutation.device)
        fixed_points = int((self.permutation == identity).sum().item())
        row_distance = (self.input_condition.detach().float() - self.effective_condition.detach().float()).norm(dim=1)
        return {
            "requested_mode": self.requested_mode,
            "applied_mode": self.applied_mode,
            "condition_was_provided": self.condition_was_provided,
            "input_condition": tensor_statistics(self.input_condition),
            "effective_condition": tensor_statistics(self.effective_condition),
            "embedding": tensor_statistics(self.embedding),
            "embedding_abs_max": float(self.embedding.detach().abs().max().item()),
            "permutation": self.permutation.detach().cpu().tolist(),
            "permutation_fixed_points": fixed_points,
            "input_to_effective_l2_mean": float(row_distance.mean().item()),
            "input_to_effective_changed_rows": int((row_distance > 0).sum().item()),
        }


class PhotoConditioner(nn.Module):
    """`PhotoConditioner` 组件。"""

    def __init__(self, condition_dim: int = 8, hidden_dim: int = 128, output_dim: int = 768) -> None:
        super().__init__()
        if condition_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("condition_dim, hidden_dim, and output_dim must be positive")
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        # 步骤 1：按当前协议处理。
        self.input_projection = nn.Linear(self.condition_dim, self.hidden_dim)
        self.activation = nn.SiLU()
        # 步骤 2：按当前协议处理。
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        # 步骤 3：按当前协议处理。
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _validate_permutation(permutation: Tensor, batch_size: int, device: torch.device) -> Tensor:
        """执行 `_validate_permutation` 所需的处理。"""

        candidate = torch.as_tensor(permutation, device=device, dtype=torch.long)
        if candidate.ndim != 1 or candidate.numel() != batch_size:
            raise ValueError(f"condition permutation must have shape [{batch_size}], got {tuple(candidate.shape)}")
        if not torch.equal(torch.sort(candidate).values, torch.arange(batch_size, device=device)):
            raise ValueError("condition permutation must contain every batch index exactly once")
        if torch.any(candidate == torch.arange(batch_size, device=device)):
            raise ValueError("shuffle_train permutation must have no fixed points")
        return candidate

    @staticmethod
    def _cyclic_derangement(
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> Tensor:
        """执行 `_cyclic_derangement` 所需的处理。"""

        if batch_size < 2:
            raise ValueError("shuffle_train needs batch_size >= 2; use a condition queue for batch size one")
        random_device = torch.device(generator.device) if generator is not None else device
        shift = int(torch.randint(1, batch_size, (1,), device=random_device, generator=generator).item())
        return (torch.arange(batch_size, device=device) + shift) % batch_size

    def _prepare_condition(
        self,
        condition: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, bool]:
        """执行 `_prepare_condition` 所需的处理。"""

        # 步骤 4：按当前协议处理。
        # 处理 CUDA 设备兼容性。
        # 处理 CUDA 设备兼容性。
        # `_prepare_condition` 的实现说明。
        expected_device = torch.device(device)
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())
        if condition is None:
            return torch.zeros(batch_size, self.condition_dim, device=expected_device, dtype=dtype), False
        if not torch.is_floating_point(condition):
            raise TypeError("photo_condition must be a floating-point tensor")
        if condition.ndim != 2 or tuple(condition.shape) != (batch_size, self.condition_dim):
            raise ValueError(
                f"photo_condition must have shape [{batch_size},{self.condition_dim}], got {tuple(condition.shape)}"
            )
        if condition.device != expected_device:
            raise ValueError(
                f"photo_condition is on {condition.device}, but feature tensors are on {expected_device}"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("photo_condition contains NaN or infinity")
        return condition.to(dtype=dtype), True

    def forward_with_details(
        self,
        condition: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        mode: ConditionMode = "correct",
        training: bool,
        permutation: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ConditionApplication:
        """执行 `forward_with_details` 所需的处理。"""

        if mode not in VALID_CONDITION_MODES:
            raise ValueError(f"unknown condition mode {mode!r}; choose one of {VALID_CONDITION_MODES}")
        # 步骤 4：校验输入约束。
        input_condition, was_provided = self._prepare_condition(
            condition,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        identity = torch.arange(batch_size, device=device)

        # 步骤 5：按当前协议处理。
        if mode == "none":
            if permutation is not None:
                raise ValueError("a permutation is only valid for training-time shuffle_train")
            effective_condition = torch.zeros_like(input_condition)
            # 与 WT-Flow 参考实现保持一致。
            embedding = torch.zeros(batch_size, self.output_dim, device=device, dtype=dtype)
            applied_mode = "none"
            applied_permutation = identity
        elif mode == "zero":
            if permutation is not None:
                raise ValueError("a permutation is only valid for training-time shuffle_train")
            effective_condition = torch.zeros_like(input_condition)
            embedding = self.output_projection(self.activation(self.input_projection(effective_condition)))
            applied_mode = "zero"
            applied_permutation = identity
        elif mode == "shuffle_train" and training:
            if not was_provided:
                raise ValueError("shuffle_train requires a paired photo_condition tensor")
            applied_permutation = (
                self._validate_permutation(permutation, batch_size, device)
                if permutation is not None
                else self._cyclic_derangement(batch_size, device, generator)
            )
            effective_condition = input_condition[applied_permutation]
            embedding = self.output_projection(self.activation(self.input_projection(effective_condition)))
            applied_mode = "shuffled"
        else:
            if not was_provided:
                raise ValueError(f"condition mode {mode!r} requires a paired photo_condition tensor")
            if permutation is not None:
                raise ValueError("a permutation is only valid for training-time shuffle_train")
            effective_condition = input_condition
            embedding = self.output_projection(self.activation(self.input_projection(effective_condition)))
            applied_mode = "correct" if mode == "correct" else "correct_eval"
            applied_permutation = identity

        # 步骤 6：按当前协议处理。
        if not torch.isfinite(embedding).all():
            raise RuntimeError("photometric condition embedding contains NaN or infinity")
        return ConditionApplication(
            requested_mode=mode,
            applied_mode=applied_mode,
            condition_was_provided=was_provided,
            input_condition=input_condition,
            effective_condition=effective_condition,
            embedding=embedding,
            permutation=applied_permutation,
        )

    def forward(self, condition: Tensor) -> Tensor:
        """执行 `forward` 所需的处理。"""

        if condition.ndim != 2:
            raise ValueError(f"photo_condition must have shape [B,{self.condition_dim}], got {tuple(condition.shape)}")
        application = self.forward_with_details(
            condition,
            batch_size=condition.shape[0],
            device=condition.device,
            dtype=condition.dtype,
            mode="correct",
            training=self.training,
        )
        return application.embedding

    def summary(self) -> dict[str, object]:
        """执行 `summary` 所需的处理。"""

        parameters = list(self.parameters())
        final_weight = self.output_projection.weight.detach().float()
        final_bias = self.output_projection.bias.detach().float()
        return {
            "condition_dim": self.condition_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            "output_projection_weight_l2": float(final_weight.norm().item()),
            "output_projection_bias_l2": float(final_bias.norm().item()),
            "output_projection_is_exact_zero": bool(torch.count_nonzero(final_weight).item() == 0 and torch.count_nonzero(final_bias).item() == 0),
        }
