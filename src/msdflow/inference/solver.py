"""模式条件正常性流的确定性 Euler 求解器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from msdflow.models import ModeConditionedNormalityFlow


def _mean_l2(value: Tensor) -> float:
    return float(value.detach().float().flatten(1).norm(dim=1).mean().item())


@dataclass(slots=True)
class ModeEulerSolveOutput:
    """正常特征到高斯端点的结果和逐步数值诊断。"""

    final_state: Tensor
    evaluation_times: tuple[float, ...]
    state_mean_l2: tuple[float, ...]
    velocity_mean_l2: tuple[float, ...]
    steps: int

    def summary(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "dt": 1.0 / self.steps,
            "evaluation_times": list(self.evaluation_times),
            "state_mean_l2": list(self.state_mean_l2),
            "velocity_mean_l2": list(self.velocity_mean_l2),
            "final_shape": list(self.final_state.shape),
            "final_finite": bool(torch.isfinite(self.final_state).all().item()),
        }


class ModeNormalityEulerSolver:
    """每一步都传入相同模式条件，从 t=0 积分到 t=1。"""

    def __init__(self, steps: int = 20) -> None:
        if steps <= 0:
            raise ValueError("Euler steps must be positive")
        self.steps = int(steps)

    @torch.no_grad()
    def solve(
        self,
        model: ModeConditionedNormalityFlow,
        feature: Tensor,
        mode_weights: Tensor,
    ) -> ModeEulerSolveOutput:
        if feature.ndim != 4 or mode_weights.ndim != 2 or mode_weights.shape[0] != feature.shape[0]:
            raise ValueError("feature must be [B,C,H,W] and mode_weights [B,K]")
        if feature.device != mode_weights.device:
            raise ValueError("feature and mode weights must share a device")
        model.eval()
        state = feature
        dt = 1.0 / self.steps
        state_norms = [_mean_l2(state)]
        velocity_norms: list[float] = []
        times: list[float] = []
        for step in range(self.steps):
            time_value = step * dt
            time = torch.full((state.shape[0],), time_value, device=state.device, dtype=state.dtype)
            velocity = model.predict_velocity(state, time, mode_weights)
            if velocity.shape != state.shape or not torch.isfinite(velocity).all():
                raise FloatingPointError(f"invalid normality velocity at Euler step {step}")
            state = state + dt * velocity
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"invalid normality state after Euler step {step}")
            times.append(time_value)
            velocity_norms.append(_mean_l2(velocity))
            state_norms.append(_mean_l2(state))
        model.eval()
        return ModeEulerSolveOutput(
            final_state=state,
            evaluation_times=tuple(times),
            state_mean_l2=tuple(state_norms),
            velocity_mean_l2=tuple(velocity_norms),
            steps=self.steps,
        )
