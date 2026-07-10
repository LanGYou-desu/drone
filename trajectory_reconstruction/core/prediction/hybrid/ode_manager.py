"""
ODE 状态管理器 — ODEStateManager

管理物理隐状态的初始化、演化和闭环更新。
- evolve():   ODE 演化 → 先验状态
- update():   GRU 融合虚拟观测 → 后验状态
"""

import torch
import torch.nn as nn

from trajectory_reconstruction.core.prediction.hybrid.physics_ode import (
    PhysicsODEFunc, ODESolver,
)


class ObservationEncoder(nn.Module):
    """虚拟观测编码器: (Δt, p, v, a) → u，包含加速度提供更丰富的运动信息"""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden_dim),  # Δt(1) + p(3) + v(3) + a(3)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, dt: torch.Tensor, p_obs: torch.Tensor,
                v_obs: torch.Tensor, a_obs: torch.Tensor = None) -> torch.Tensor:
        if a_obs is None:
            a_obs = torch.zeros_like(v_obs)
        inp = torch.cat([dt.unsqueeze(-1), p_obs, v_obs, a_obs], dim=-1)
        return self.net(inp)


class ODEStateManager(nn.Module):
    """
    ODE 状态管理器。

    职责:
      1. 初始化: h_0 = [p_N, v_N_est, MLP_init(c)]
      2. 演化:   h^- = ODESolve(f_θ, h, t, t+Δt)
      3. 更新:   h_new = GRU(h^-, Encoder(Δt, p_obs, v_obs))
    """

    def __init__(
        self,
        d_context: int = 128,
        d_z: int = 32,
        a_max: float = 30.0,
        hidden_dim: int = 64,
        obs_hidden_dim: int = 32,
    ):
        super().__init__()
        self.d_z = d_z
        self.state_dim = 6 + d_z  # p(3) + v(3) + z(d_z)

        # 初始化映射: [p_N(3), v_N_est(3), context(d_context)] → h_0(state_dim)
        self.init_proj = nn.Sequential(
            nn.Linear(6 + d_context, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.state_dim),
        )

        # ODE 动力学 + 求解器
        self.ode_func = PhysicsODEFunc(d_z=d_z, a_max=a_max, hidden_dim=hidden_dim)
        self.ode_solver = ODESolver(self.ode_func, n_steps=8)

        # 观测编码器
        self.obs_encoder = ObservationEncoder(hidden_dim=obs_hidden_dim)

        # GRU 更新: 输入观测编码(hidden_dim) → 输出状态修正
        self.gru = nn.GRUCell(input_size=obs_hidden_dim, hidden_size=self.state_dim)

    def init_state(
        self,
        p_last: torch.Tensor,      # (B, 3)
        v_est_last: torch.Tensor,  # (B, 3)
        context: torch.Tensor,     # (B, d_context)
    ) -> torch.Tensor:
        """
        初始化 ODE 隐状态。
        h(t_N) = [p_N, v_N_est, MLP_init([p_N, v_N_est, c])]
        """
        inp = torch.cat([p_last, v_est_last, context], dim=-1)
        return self.init_proj(inp)  # (B, state_dim)

    def evolve(
        self,
        h: torch.Tensor,    # (B, state_dim)
        t0: torch.Tensor,   # (B,)
        t1: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        ODE 演化 → 先验状态 h^-
        """
        return self.ode_solver.solve(h, t0, t1)

    def update(
        self,
        h_prior: torch.Tensor,   # (B, state_dim) ODE 先验
        dt: torch.Tensor,        # (B,) 时间步长
        p_obs: torch.Tensor,     # (B, 3) 观测位置
        v_obs: torch.Tensor,     # (B, 3) 观测速度（差分估计）
    ) -> torch.Tensor:
        """
        GRU 融合虚拟观测 → 后验状态 h_new
        """
        u = self.obs_encoder(dt, p_obs, v_obs)    # (B, obs_hidden_dim)
        h_new = self.gru(u, h_prior)              # (B, state_dim)
        return h_new
