"""RA-MSDFlow 的 conditional_unet 模块。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .feature_preprocess import tensor_statistics


class _WTResidualTimeBlock(nn.Module):
    """`_WTResidualTimeBlock` 组件。"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.fc = nn.Linear(time_emb_dim, in_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        # 步骤 1：按当前协议处理。
        time_bias = self.fc(temb)[:, :, None, None]
        # 步骤 2：按当前协议处理。
        # `forward` 的实现说明。
        residual = x + time_bias
        x = residual
        # 步骤 3：应用当前变换。
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        # 步骤 4：按当前协议处理。
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return x + residual


class WTDownBlock(_WTResidualTimeBlock):
    """`WTDownBlock` 组件。"""


class WTUpBlock(_WTResidualTimeBlock):
    """`WTUpBlock` 组件。"""


class WTMiddleBlock(_WTResidualTimeBlock):
    """`WTMiddleBlock` 组件。"""


@dataclass(frozen=True, slots=True)
class WTFlowUNetConfig:
    in_channels: int = 1024
    base_channels: int = 768
    time_emb_dim: int | None = None


class WTFlowMiniUNet(nn.Module):
    """`WTFlowMiniUNet` 组件。"""

    def __init__(self, in_channels: int = 1024, base_channels: int = 768, time_emb_dim: int | None = None) -> None:
        super().__init__()
        if in_channels <= 0 or base_channels <= 0:
            raise ValueError("in_channels and base_channels must be positive")
        if base_channels % 2 != 0:
            raise ValueError("base_channels must be even for the sinusoidal time embedding")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.time_emb_dim = int(time_emb_dim or base_channels)

        # 步骤 1：按当前协议处理。
        self.conv_in = nn.Conv2d(self.in_channels, self.base_channels, kernel_size=1, padding=0)
        # 步骤 2：按当前协议处理。
        self.down1 = nn.ModuleList(
            [
                WTDownBlock(self.base_channels, self.base_channels * 2, self.time_emb_dim),
                WTDownBlock(self.base_channels * 2, self.base_channels * 2, self.time_emb_dim),
            ]
        )
        self.maxpool1 = nn.MaxPool2d(2)
        self.down2 = nn.ModuleList(
            [
                WTDownBlock(self.base_channels * 2, self.base_channels * 4, self.time_emb_dim),
                WTDownBlock(self.base_channels * 4, self.base_channels * 4, self.time_emb_dim),
            ]
        )
        self.maxpool2 = nn.MaxPool2d(2)
        self.middle = WTMiddleBlock(self.base_channels * 4, self.base_channels * 4, self.time_emb_dim)
        # 步骤 3：解码图像。
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up1 = nn.ModuleList(
            [
                WTUpBlock(self.base_channels * 8, self.base_channels * 2, self.time_emb_dim),
                WTUpBlock(self.base_channels * 2, self.base_channels * 2, self.time_emb_dim),
            ]
        )
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.up2 = nn.ModuleList(
            [
                WTUpBlock(self.base_channels * 4, self.base_channels, self.time_emb_dim),
                WTUpBlock(self.base_channels, self.base_channels, self.time_emb_dim),
            ]
        )
        # 步骤 4：按当前协议处理。
        self.conv_out = nn.Conv2d(self.base_channels, self.in_channels, kernel_size=1, padding=0)

    def time_emb(self, time: Tensor, dim: int | None = None) -> Tensor:
        """执行 `time_emb` 所需的处理。"""

        if time.ndim != 1:
            raise ValueError(f"time must have shape [B], got {tuple(time.shape)}")
        dim = int(dim or self.base_channels)
        if dim % 2 != 0:
            raise ValueError("time embedding dimension must be even")
        # 与 WT-Flow 参考实现保持一致。
        time = time * 1000.0
        frequencies = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(time.device)
        return torch.cat((torch.sin(time[:, None] / frequencies), torch.cos(time[:, None] / frequencies)), dim=-1)

    def forward(self, xt: Tensor, time: Tensor) -> Tensor:
        if xt.ndim != 4 or xt.shape[1] != self.in_channels:
            raise ValueError(f"expected xt [B,{self.in_channels},H,W], got {tuple(xt.shape)}")
        if xt.shape[-2] % 4 != 0 or xt.shape[-1] % 4 != 0:
            raise ValueError("WT-Flow Mini-U-Net needs feature H and W divisible by 4")
        if time.ndim != 1 or time.shape[0] != xt.shape[0]:
            raise ValueError(f"time must have shape [B={xt.shape[0]}], got {tuple(time.shape)}")
        # 步骤 5：构建当前对象。
        temb = self.time_emb(time.to(dtype=xt.dtype), self.base_channels)
        x = self.conv_in(xt)
        for layer in self.down1:
            x = layer(x, temb)
        skip1 = x
        x = self.maxpool1(x)
        for layer in self.down2:
            x = layer(x, temb)
        skip2 = x
        x = self.maxpool2(x)
        x = self.middle(x, temb)
        # 步骤 6：按当前协议处理。
        x = torch.cat((self.upsample1(x), skip2), dim=1)
        for layer in self.up1:
            x = layer(x, temb)
        x = torch.cat((self.upsample2(x), skip1), dim=1)
        for layer in self.up2:
            x = layer(x, temb)
        return self.conv_out(x)

    def summary(self) -> dict[str, object]:
        parameters = list(self.parameters())
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "time_emb_dim": self.time_emb_dim,
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "trainable_parameter_count": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        }

    @torch.no_grad()
    def prediction_summary(self, xt: Tensor, time: Tensor) -> dict[str, object]:
        """执行 `prediction_summary` 所需的处理。"""

        was_training = self.training
        self.eval()
        prediction = self(xt, time)
        self.train(was_training)
        return {"prediction": tensor_statistics(prediction)}


# 保持导入顺序兼容。
from .photo_conditioner import ConditionApplication, ConditionMode, PhotoConditioner


@dataclass(slots=True)
class ConditionalUNetOutput:
    """`ConditionalUNetOutput` 组件。"""

    velocity: Tensor
    time_embedding: Tensor
    condition: ConditionApplication
    combined_embedding: Tensor

    def summary(self) -> dict[str, object]:
        """执行 `summary` 所需的处理。"""

        return {
            "velocity": tensor_statistics(self.velocity),
            "time_embedding": tensor_statistics(self.time_embedding),
            "condition": self.condition.summary(),
            "combined_embedding": tensor_statistics(self.combined_embedding),
        }


class EnvFMConditionalMiniUNet(WTFlowMiniUNet):
    """`EnvFMConditionalMiniUNet` 组件。"""

    def __init__(
        self,
        in_channels: int = 1024,
        base_channels: int = 768,
        time_emb_dim: int | None = None,
        *,
        condition_dim: int = 8,
        condition_hidden_dim: int = 128,
    ) -> None:
        if time_emb_dim is not None and time_emb_dim != base_channels:
            raise ValueError("EnvFM requires time_emb_dim == base_channels for additive condition fusion")
        super().__init__(in_channels=in_channels, base_channels=base_channels, time_emb_dim=time_emb_dim)
        # 步骤 1：按当前协议处理。
        self.photo_conditioner = PhotoConditioner(
            condition_dim=condition_dim,
            hidden_dim=condition_hidden_dim,
            output_dim=self.base_channels,
        )

    def load_unconditional_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        strict_base: bool = True,
    ) -> dict[str, list[str]]:
        """执行 `load_unconditional_state_dict` 所需的处理。"""

        # 步骤 2：加载当前输入。
        incompatibility = self.load_state_dict(state_dict, strict=False)
        missing = list(incompatibility.missing_keys)
        unexpected = list(incompatibility.unexpected_keys)
        expected_missing = {f"photo_conditioner.{name}" for name in self.photo_conditioner.state_dict()}
        if strict_base and (set(missing) != expected_missing or unexpected):
            raise RuntimeError(
                "unconditional checkpoint is incompatible with EnvFM base layers: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    @classmethod
    def from_unconditional(
        cls,
        unconditional: WTFlowMiniUNet,
        *,
        condition_dim: int = 8,
        condition_hidden_dim: int = 128,
    ) -> "EnvFMConditionalMiniUNet":
        """执行 `from_unconditional` 所需的处理。"""

        if type(unconditional) is not WTFlowMiniUNet:
            raise TypeError("from_unconditional expects the accepted WTFlowMiniUNet base class")
        reference_parameter = next(unconditional.parameters())
        # 步骤 3：构建当前模块。
        conditional = cls(
            in_channels=unconditional.in_channels,
            base_channels=unconditional.base_channels,
            time_emb_dim=unconditional.time_emb_dim,
            condition_dim=condition_dim,
            condition_hidden_dim=condition_hidden_dim,
        ).to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        conditional.load_unconditional_state_dict(unconditional.state_dict(), strict_base=True)
        conditional.train(unconditional.training)
        return conditional

    def _forward_from_shared_embedding(self, xt: Tensor, shared_embedding: Tensor) -> Tensor:
        """执行 `_forward_from_shared_embedding` 所需的处理。"""

        # 步骤 4：执行当前计算。
        x = self.conv_in(xt)
        for layer in self.down1:
            x = layer(x, shared_embedding)
        skip1 = x
        x = self.maxpool1(x)
        for layer in self.down2:
            x = layer(x, shared_embedding)
        skip2 = x
        x = self.maxpool2(x)
        # 步骤 5：按当前协议处理。
        x = self.middle(x, shared_embedding)
        # 步骤 6：执行当前计算。
        x = torch.cat((self.upsample1(x), skip2), dim=1)
        for layer in self.up1:
            x = layer(x, shared_embedding)
        x = torch.cat((self.upsample2(x), skip1), dim=1)
        for layer in self.up2:
            x = layer(x, shared_embedding)
        return self.conv_out(x)

    def forward_with_details(
        self,
        xt: Tensor,
        time: Tensor,
        photo_condition: Tensor | None,
        *,
        condition_mode: ConditionMode = "correct",
        condition_permutation: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ConditionalUNetOutput:
        """执行 `forward_with_details` 所需的处理。"""

        if xt.ndim != 4 or xt.shape[1] != self.in_channels:
            raise ValueError(f"expected xt [B,{self.in_channels},H,W], got {tuple(xt.shape)}")
        if xt.shape[-2] % 4 != 0 or xt.shape[-1] % 4 != 0:
            raise ValueError("EnvFM Mini-U-Net needs feature H and W divisible by 4")
        if time.ndim != 1 or time.shape[0] != xt.shape[0]:
            raise ValueError(f"time must have shape [B={xt.shape[0]}], got {tuple(time.shape)}")
        if time.device != xt.device:
            raise ValueError(f"time is on {time.device}, but feature tensors are on {xt.device}")

        # 步骤 7：按当前协议处理。
        time_embedding = self.time_emb(time.to(dtype=xt.dtype), self.base_channels)
        # 步骤 8：计算环境条件。
        condition = self.photo_conditioner.forward_with_details(
            photo_condition,
            batch_size=xt.shape[0],
            device=xt.device,
            dtype=time_embedding.dtype,
            mode=condition_mode,
            training=self.training,
            permutation=condition_permutation,
            generator=generator,
        )
        # 步骤 9：按当前协议处理。
        combined_embedding = time_embedding + condition.embedding
        velocity = self._forward_from_shared_embedding(xt, combined_embedding)
        return ConditionalUNetOutput(
            velocity=velocity,
            time_embedding=time_embedding,
            condition=condition,
            combined_embedding=combined_embedding,
        )

    def forward(
        self,
        xt: Tensor,
        time: Tensor,
        photo_condition: Tensor | None,
        *,
        condition_mode: ConditionMode = "correct",
        condition_permutation: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """执行 `forward` 所需的处理。"""

        return self.forward_with_details(
            xt,
            time,
            photo_condition,
            condition_mode=condition_mode,
            condition_permutation=condition_permutation,
            generator=generator,
        ).velocity

    def summary(self) -> dict[str, object]:
        """执行 `summary` 所需的处理。"""

        all_parameters = list(self.parameters())
        condition_parameters = list(self.photo_conditioner.parameters())
        condition_count = sum(parameter.numel() for parameter in condition_parameters)
        total_count = sum(parameter.numel() for parameter in all_parameters)
        return {
            "model": "envfm_conditional_min_unet",
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "time_emb_dim": self.time_emb_dim,
            "unconditional_parameter_count": total_count - condition_count,
            "condition_parameter_count": condition_count,
            "total_parameter_count": total_count,
            "trainable_parameter_count": sum(parameter.numel() for parameter in all_parameters if parameter.requires_grad),
            "fusion": "time_embedding_plus_photo_embedding",
            "supported_condition_modes": ["none", "correct", "zero", "shuffle_train"],
            "photo_conditioner": self.photo_conditioner.summary(),
        }

    @torch.no_grad()
    def conditional_prediction_summary(
        self,
        xt: Tensor,
        time: Tensor,
        photo_condition: Tensor,
        *,
        condition_mode: ConditionMode = "correct",
    ) -> dict[str, object]:
        """执行 `conditional_prediction_summary` 所需的处理。"""

        was_training = self.training
        self.eval()
        output = self.forward_with_details(xt, time, photo_condition, condition_mode=condition_mode)
        self.train(was_training)
        return output.summary()
