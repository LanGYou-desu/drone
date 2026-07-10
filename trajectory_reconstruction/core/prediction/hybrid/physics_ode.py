"""
物理结构化 ODE — Physics-Structured Neural ODE

定义物理隐状态 h = [p, v, z] 的连续时间动力学:
  dp/dt = v
  dv/dt = a_max * tanh(MLP_a(p, v, z))
  dz/dt = MLP_z(p, v, z)

该设计强制位置与速度的运动学耦合，加速度由网络学习并软限幅。
"""

import torch
import torch.nn as nn


class PhysicsODEFunc(nn.Module):
    """
    ODE 右端函数 f_θ: dh/dt = f_θ(h)

    隐状态 h = [p(3), v(3), z(d_z)], 总维度 = 6 + d_z
    """

    def __init__(
        self,
        d_z: int = 32,
        a_max: float = 30.0,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.d_z = d_z
        self.a_max = a_max
        self.state_dim = 6 + d_z

        self.mlp_a = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.mlp_z = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_z),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.mlp_a, self.mlp_z]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) 或标量 — 自治系统不使用
            h: (B, state_dim) 当前状态 [p(3), v(3), z(d_z)]
        Returns:
            dh_dt: (B, state_dim) 状态导数 [v, a, z_dot]
        """
        p = h[..., :3]
        v = h[..., 3:6]

        a_raw = self.mlp_a(h)
        a = self.a_max * torch.tanh(a_raw / self.a_max)
        z_dot = self.mlp_z(h)

        dh_dt = torch.cat([v, a, z_dot], dim=-1)
        return dh_dt


class ODESolver:
    """
    定步长 RK4 ODE 求解器。

    对 PhysicsODEFunc 描述的平滑动力学，RK4 在 4-8 步内即可提供足够精度。
    """

    def __init__(self, ode_func: PhysicsODEFunc, n_steps: int = 8):
        self.ode_func = ode_func
        self.n_steps = n_steps

    def solve(
        self,
        h0: torch.Tensor,       # (B, state_dim) 初始状态
        t0: torch.Tensor,       # (B,) 起始时间
        t1: torch.Tensor,       # (B,) 终止时间
    ) -> torch.Tensor:
        """
        RK4 从 t0 积分到 t1。

        Returns:
            h1: (B, state_dim) t1 时刻的状态
        """
        if isinstance(t0, (int, float)):
            t0 = torch.full((h0.shape[0],), t0, device=h0.device, dtype=h0.dtype)
        if isinstance(t1, (int, float)):
            t1 = torch.full((h0.shape[0],), t1, device=h0.device, dtype=h0.dtype)
        if t0.dim() == 0:
            t0 = t0.unsqueeze(0)
        if t1.dim() == 0:
            t1 = t1.unsqueeze(0)

        dt_total = t1 - t0
        dt_step = dt_total / self.n_steps

        h = h0
        t_current = t0

        for _ in range(self.n_steps):
            k1 = self.ode_func(t_current, h)
            k2 = self.ode_func(t_current + dt_step / 2,
                               h + dt_step.unsqueeze(-1) * k1 / 2)
            k3 = self.ode_func(t_current + dt_step / 2,
                               h + dt_step.unsqueeze(-1) * k2 / 2)
            k4 = self.ode_func(t_current + dt_step,
                               h + dt_step.unsqueeze(-1) * k3)

            h = h + dt_step.unsqueeze(-1) * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t_current = t_current + dt_step

        return h
