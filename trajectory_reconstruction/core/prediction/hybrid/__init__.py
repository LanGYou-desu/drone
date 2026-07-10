"""
Phy-ODE-Diffusion: 融合扩散模型与 Transformer 的物理约束轨迹预测

Components:
  - TransformerEncoder:    不规则时间序列编码
  - PhysicsODEFunc:        物理结构化 ODE 动力学
  - ODESolver:             定步长 RK4 求解器
  - ODEStateManager:       ODE 演化 + GRU 状态更新
  - GuidedDiffusion:        物理引导扩散生成器
  - NoiseScheduler:         噪声调度器 (Cosine/Linear)
  - PhyODEDiffusion:        顶层预测模型
"""

from trajectory_reconstruction.core.prediction.hybrid.transformer import (
    TransformerEncoder, ContinuousTimeEncoding,
)
from trajectory_reconstruction.core.prediction.hybrid.physics_ode import (
    PhysicsODEFunc, ODESolver,
)
from trajectory_reconstruction.core.prediction.hybrid.diffusion import (
    GuidedDiffusion, NoiseScheduler, NoisePredictionNet,
)
from trajectory_reconstruction.core.prediction.hybrid.ode_manager import (
    ODEStateManager,
)
from trajectory_reconstruction.core.prediction.hybrid.model import (
    PhyODEDiffusion,
)

__all__ = [
    "PhyODEDiffusion",
    "TransformerEncoder",
    "ContinuousTimeEncoding",
    "PhysicsODEFunc",
    "ODESolver",
    "GuidedDiffusion",
    "NoiseScheduler",
    "NoisePredictionNet",
    "ODEStateManager",
]
