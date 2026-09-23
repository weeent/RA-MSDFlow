"""Mode-aware Shift--Defect Flow (MSD-Flow) research implementation.

本包按数据、条件、模型、训练、推理、评估和命令入口拆分。模块之间只传递显式张量和元数据，
便于在不启动完整工业数据实验的情况下逐层测试。
"""

from . import conditions, data, models, training, inference, evaluation, reference_otcfm, score_calibration

__all__ = [
    "conditions",
    "data",
    "evaluation",
    "inference",
    "models",
    "reference_otcfm",
    "score_calibration",
    "training",
]
