"""公开源码 baseline 的薄协议适配层。

这里不实现第三方算法，只统一三件事：数据目录、正常 reference 划分和结果校准。
"""

from .public_protocol import (
    BaselineProtocolConfig,
    finalize_public_baseline,
    prepare_public_baseline,
)
from .public_registry import PUBLIC_BASELINES, PublicBaselineSpec

__all__ = [
    "BaselineProtocolConfig",
    "PUBLIC_BASELINES",
    "PublicBaselineSpec",
    "finalize_public_baseline",
    "prepare_public_baseline",
]
