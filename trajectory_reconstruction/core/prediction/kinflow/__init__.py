"""
KinFlow: 学习式运动学先验 + 并行非自回归细化

用于不规则采样无人机轨迹预测的紧凑模型（论文 APIN-D-26-14134）。

模块:
  - ContextEncoder:        连续时间上下文编码器（不规则历史 → c）
  - KinematicCorridor:     学习式常数加速度走廊先验（闭式求值）
  - CorridorFlow:          走廊初始化的条件流匹配 + 并行 Euler 采样
  - PhysicsProjector:      非参数可微物理投影器（速度/垂直速率/高度硬约束）
  - KinFlow:               顶层预测模型
"""

from trajectory_reconstruction.core.prediction.kinflow.encoder import (
    ContextEncoder, ContinuousTimeEncoding,
)
from trajectory_reconstruction.core.prediction.kinflow.corridor import KinematicCorridor
from trajectory_reconstruction.core.prediction.kinflow.flow import (
    CorridorFlow, SinusoidalEmbedding, TemporalAttentionMixer,
)
from trajectory_reconstruction.core.prediction.kinflow.projector import (
    KinematicBounds, PhysicsProjector, kinematic_violation_cost,
)
from trajectory_reconstruction.core.prediction.kinflow.model import (
    KinFlow, POSITION_STD_FLOOR,
)

__all__ = [
    "KinFlow",
    "ContextEncoder",
    "ContinuousTimeEncoding",
    "KinematicCorridor",
    "CorridorFlow",
    "SinusoidalEmbedding",
    "TemporalAttentionMixer",
    "PhysicsProjector",
    "KinematicBounds",
    "kinematic_violation_cost",
    "POSITION_STD_FLOOR",
]
