"""
可微物理投影器 — Differentiable Physics Projector

KinFlow 的第四个模块（论文 §3.5）。非参数、可微、单次前向完成四旋翼硬约束投影:

  1. 水平速度缩放到 v_h,max（水平方向整体缩放，保持朝向不变）
  2. 垂直速度裁剪到 [-v_down,max, v_up,max]（上升/下降能力不对称）
  3. 前向扫描: 逐步限制垂直速度，使到达高度始终落在 [y_min, y_max]，
     避免硬裁剪高度造成的速度尖峰
  4. 以 p_N 为锚点、按 Δτ_k 累加重建位置

默认迭代 R=3 次；由于速度已被约束、位置又由速度重建，第一次之后不再变化，
后续迭代作为分布外输入的保险。

坐标约定: 本项目位置为 [x, y, z]，其中 y 为垂直方向（高度轴），
x/z 构成水平面——与无人机动力学配置 (v_h_max / v_v_up / v_v_down / min_alt / max_alt) 一致。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class KinematicBounds:
    """
    四旋翼运动学边界（物理坐标，单位 m / s / s²）。

    默认值与 config.json → drone_dynamics 对齐（本项目坐标系: X 右 / Y 上 / Z 前，
    y 为高度轴），确保训练数据、运行时轨迹与物理投影使用同一套边界。
    （论文在公开数据集上的取值更严: v_h=10, up=4, down=2, tilt=30, 高度 2~100 m。）
    """

    v_h_max: float = 20.0      # 最大水平速度
    v_up_max: float = 5.0      # 最大垂直上升速度
    v_down_max: float = 3.0    # 最大垂直下降速度
    y_min: float = 1.0         # 最低高度
    y_max: float = 120.0       # 最高高度
    max_tilt: float = 35.0     # 最大倾斜角（度）
    g: float = 9.81            # 重力加速度

    @property
    def a_h_max(self) -> float:
        """水平加速度上限 a_h ≤ g·tan(φ_max)，仅以软损失近似。"""
        import math
        return self.g * math.tan(math.radians(self.max_tilt))


def kinematic_violation_cost(
    traj: torch.Tensor,        # (B, K, 3) 物理坐标轨迹
    anchor: torch.Tensor,      # (B, 3) 物理坐标锚点 p_N
    dtau: torch.Tensor,        # (B, K) 未来相对时间（秒）
    bounds: KinematicBounds,
    anchor_vel: torch.Tensor = None,   # (B, 3) 锚点处速度（物理坐标）
) -> torch.Tensor:
    """
    软物理损失 L_phy: 速度/垂直速度/高度/水平加速度的平方 hinge 违反量。

    速度与高度是单点约束，由投影器硬性保证；加速度耦合相邻步、没有单点闭式投影，
    因此只在这里以软损失近似（论文 §3.9 的设计取舍）。
    """
    dt = dtau.clamp(min=1e-3).unsqueeze(-1)                       # (B, K, 1)

    prev = torch.cat([anchor.unsqueeze(1), traj[:, :-1]], dim=1)  # (B, K, 3)
    vel = (traj - prev) / dt                                      # (B, K, 3)

    v_h = torch.linalg.vector_norm(vel[..., [0, 2]], dim=-1)      # (B, K)
    v_y = vel[..., 1]

    cost = torch.relu(v_h - bounds.v_h_max) ** 2
    cost = cost + torch.relu(v_y - bounds.v_up_max) ** 2
    cost = cost + torch.relu(-v_y - bounds.v_down_max) ** 2

    # 高度约束（y 为垂直轴）
    y = traj[..., 1]
    cost = cost + torch.relu(bounds.y_min - y) ** 2
    cost = cost + torch.relu(y - bounds.y_max) ** 2

    # 水平加速度: 相对锚点速度（历史末速度）起算，其后用相邻步速度差分
    if anchor_vel is None:
        anchor_vh = torch.zeros_like(vel[:, :1, [0, 2]])
    else:
        anchor_vh = anchor_vel[:, [0, 2]].unsqueeze(1)
    prev_vh = torch.cat([anchor_vh, vel[:, :-1, [0, 2]]], dim=1)
    a_h = torch.linalg.vector_norm(vel[..., [0, 2]] - prev_vh, dim=-1) / dtau.clamp(min=1e-3)
    cost = cost + torch.relu(a_h - bounds.a_h_max) ** 2

    return cost.mean()


class PhysicsProjector(nn.Module):
    """
    非参数可微物理投影器（无学习参数，因此不增加参数量）。

    输入为归一化坐标下的采样轨迹，输出物理坐标下满足速度/垂直速率/高度边界的轨迹。
    """

    def __init__(self, bounds: KinematicBounds = None, n_iters: int = 3):
        super().__init__()
        self.bounds = bounds or KinematicBounds()
        self.n_iters = max(int(n_iters), 1)

    # ── 主入口 ────────────────────────────────────────

    def forward(
        self,
        traj: torch.Tensor,        # (B, K, 3) 归一化坐标
        anchor: torch.Tensor,      # (B, 3) 归一化坐标锚点 p_N
        dtau: torch.Tensor,        # (B, K)
        p_mean: torch.Tensor,      # (B, 3) 或 (3,)
        p_std: torch.Tensor,       # (B, 3) 或 (3,)
    ) -> torch.Tensor:
        """
        Returns:
            (B, K, 3) 物理坐标下的投影轨迹
        """
        if p_mean.dim() == 1:
            p_mean = p_mean.unsqueeze(0)
        if p_std.dim() == 1:
            p_std = p_std.unsqueeze(0)

        anchor_phys = anchor * p_std + p_mean                      # (B, 3)
        x = traj * p_std.unsqueeze(1) + p_mean.unsqueeze(1)        # (B, K, 3)

        for _ in range(self.n_iters):
            x = self._project_once(x, anchor_phys, dtau)
        return x

    # ── 单次投影 ──────────────────────────────────────

    def _project_once(
        self,
        x: torch.Tensor,           # (B, K, 3) 物理坐标
        anchor: torch.Tensor,      # (B, 3) 物理坐标
        dtau: torch.Tensor,        # (B, K)
    ) -> torch.Tensor:
        b = self.bounds
        B, K, _ = x.shape
        dt = dtau.clamp(min=1e-3)                                  # (B, K)

        prev = torch.cat([anchor.unsqueeze(1), x[:, :-1]], dim=1)  # (B, K, 3)
        vel = (x - prev) / dt.unsqueeze(-1)                        # (B, K, 3)

        # (1) 水平速度缩放到 v_h_max
        v_h = vel[..., [0, 2]]                                     # (B, K, 2)
        v_h_norm = torch.linalg.vector_norm(v_h, dim=-1, keepdim=True)
        scale = torch.clamp(b.v_h_max / v_h_norm.clamp_min(1e-6), max=1.0)
        v_h = v_h * scale

        # (2) 垂直速度裁剪（上升/下降不对称）
        v_y = vel[..., 1:2].clamp(min=-b.v_down_max, max=b.v_up_max)

        # (3) 前向扫描: 让高度始终位于 [y_min, y_max]
        # 锚点若已在区间之外（例如贴地飞行的输入轨迹），把区间扩展到包含锚点，
        # 只阻止继续向界外运动，避免投影器"凭空"把轨迹抬进或压进包线。
        y = anchor[:, 1:2].unsqueeze(-1)                           # (B, 1, 1)
        y_lo = torch.minimum(anchor[:, 1:2], torch.full_like(anchor[:, 1:2], b.y_min))
        y_hi = torch.maximum(anchor[:, 1:2], torch.full_like(anchor[:, 1:2], b.y_max))
        v_y_bounded = []
        for k in range(K):
            dt_k = dt[:, k:k + 1].unsqueeze(-1)                    # (B, 1, 1)
            lo = (y_lo.unsqueeze(-1) - y) / dt_k                   # vy ≥ lo
            hi = (y_hi.unsqueeze(-1) - y) / dt_k                   # vy ≤ hi
            hi = torch.maximum(hi, lo)                             # 锚点越界时保证区间非空
            v_k = torch.minimum(torch.maximum(v_y[:, k:k + 1], lo), hi)
            y = y + v_k * dt_k
            v_y_bounded.append(v_k)
        v_y = torch.cat(v_y_bounded, dim=1)                        # (B, K, 1)

        # (4) 由受限速度重建位置（锚点 p_N 出发的累加）
        vel_bounded = torch.cat([v_h[..., :1], v_y, v_h[..., 1:]], dim=-1)
        inc = vel_bounded * dt.unsqueeze(-1)
        return anchor.unsqueeze(1) + torch.cumsum(inc, dim=1)

    # ── 诊断 ──────────────────────────────────────────

    @torch.no_grad()
    def violation_rates(
        self,
        traj: torch.Tensor,        # (B, K, 3) 物理坐标
        anchor: torch.Tensor,      # (B, 3) 物理坐标
        dtau: torch.Tensor,        # (B, K)
        anchor_vel: torch.Tensor = None,   # (B, 3) 锚点处速度（物理坐标）
    ) -> dict:
        """返回各约束的违反率（0~1），用于评价与图表。"""
        b = self.bounds
        dt = dtau.clamp(min=1e-3).unsqueeze(-1)
        prev = torch.cat([anchor.unsqueeze(1), traj[:, :-1]], dim=1)
        vel = (traj - prev) / dt

        v_h = torch.linalg.vector_norm(vel[..., [0, 2]], dim=-1)
        v_y = vel[..., 1]
        y = traj[..., 1]
        if anchor_vel is None:
            anchor_vh = torch.zeros_like(vel[:, :1, [0, 2]])
        else:
            anchor_vh = anchor_vel[:, [0, 2]].unsqueeze(1)
        prev_vh = torch.cat([anchor_vh, vel[:, :-1, [0, 2]]], dim=1)
        a_h = torch.linalg.vector_norm(vel[..., [0, 2]] - prev_vh, dim=-1) / dtau.clamp(min=1e-3)

        return {
            "speed": (v_h > b.v_h_max).float().mean().item(),
            "vertical": ((v_y > b.v_up_max) | (v_y < -b.v_down_max)).float().mean().item(),
            "altitude": ((y < b.y_min) | (y > b.y_max)).float().mean().item(),
            "accel": (a_h > b.a_h_max).float().mean().item(),
        }


__all__ = ["KinematicBounds", "PhysicsProjector", "kinematic_violation_cost"]
