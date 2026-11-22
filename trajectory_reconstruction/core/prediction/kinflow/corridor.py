"""
运动学走廊先验 — Kinematic Corridor Prior

KinFlow 的第二个模块（论文 §3.3）。一个小 MLP 把上下文 c 与末位置 p_N
映射为初速度与加速度
    (v_0, a) = MLP_corr(c, p_N)
并在真实未来时间戳上闭式求值常数加速度先验
    X_0[k] = p_N + v_0·Δτ_k + ½·a·Δτ_k²,  Δτ_k = τ_k - t_N

因为求值是闭式的，先验可以在任意（规则或不规则）未来时刻查询，
不需要网格，也不会出现逐点抖动。走廊仅做解析外推，边界满足交给物理投影器。
"""

import torch
import torch.nn as nn


class KinematicCorridor(nn.Module):
    """
    学习式常数加速度先验（运动学走廊）。

    与经典 CA-Kalman 的关键区别: (v_0, a) 由整段历史上下文回归得到，
    而不是由局部有限差分估计。
    """

    def __init__(self, d_context: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.d_context = d_context
        self.mlp = nn.Sequential(
            nn.Linear(d_context + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
        # 末端小初始化: 训练初期先验接近静止点，再逐步学到真实速度/加速度
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        context: torch.Tensor,    # (B, d_context)
        last_pos: torch.Tensor,   # (B, 3) 归一化末位置 p_N
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            v0:  (B, 3) 初始速度（归一化坐标 / 秒）
            acc: (B, 3) 常加速度（归一化坐标 / 秒²）
        """
        out = self.mlp(torch.cat([context, last_pos], dim=-1))
        v0 = out[:, :3]
        acc = out[:, 3:]
        return v0, acc

    @staticmethod
    def extrapolate(
        last_pos: torch.Tensor,   # (B, 3)
        v0: torch.Tensor,         # (B, 3)
        acc: torch.Tensor,        # (B, 3)
        dtau: torch.Tensor,       # (B, K) 未来相对时间（秒）
    ) -> torch.Tensor:
        """
        闭式求值走廊先验。

        Returns:
            X0: (B, K, 3)
        """
        dtau = dtau.unsqueeze(-1)                                  # (B, K, 1)
        return (last_pos.unsqueeze(1)
                + v0.unsqueeze(1) * dtau
                + 0.5 * acc.unsqueeze(1) * dtau * dtau)

    def trajectory(
        self,
        context: torch.Tensor,
        last_pos: torch.Tensor,
        dtau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """一步得到走廊参数与走廊轨迹: (X0, v0, acc)"""
        v0, acc = self.forward(context, last_pos)
        return self.extrapolate(last_pos, v0, acc, dtau), v0, acc


__all__ = ["KinematicCorridor"]
