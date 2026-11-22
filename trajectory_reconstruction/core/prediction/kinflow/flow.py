"""
走廊初始化的 Rectified Flow — Corridor-Initialized Rectified Flow

KinFlow 的第三个模块（论文 §3.4）。以走廊轨迹 X_0 为源、真实轨迹 X_1 为汇，
定义线性插值路径
    X_t = (1-t)·X_0 + t·X_1,   t ∈ U[0, 1]
并训练速度网络 v_θ 回归路径切向 X_1 - X_0（条件流匹配，CFM）:
    L_flow = E_t[ ‖v_θ(X_t, t, c, τ) - (X_1 - X_0)‖² ]

推理时从走廊出发做少量并行 Euler 积分（默认 4 步），一次前向覆盖全部 K 个未来点，
因此不存在自回归误差累积，也不存在扩散的数百步去噪。

网络结构（论文 §3.4 "Network architecture"）:
  - 全局分支: z_gen = MLP(c, X_0^flat) ∈ R^{d_z}
  - 逐点分支: v_k = MLP_step([x_k; Emb(τ_k); Emb(t); z_gen]) ∈ R³（K 点共享）
  - 时间相干性: K 个 token 上的单头注意力
输出层零初始化，使训练早期的采样结果停在走廊先验上。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """固定正弦嵌入（无学习参数），把任意实数映射到有界的周期特征。"""

    def __init__(self, dim: int, max_period: float = 100.0):
        super().__init__()
        self.dim = dim
        half = max(dim // 2, 1)
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (...,) 标量序列
        Returns:
            (..., dim)
        """
        angles = x.unsqueeze(-1) * self.freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TemporalAttentionMixer(nn.Module):
    """
    K 个时间步 token 上的单头自注意力，为逐点速度提供时间相干性。

    输出投影零初始化，保证模型初始时严格等价于逐点 MLP。
    """

    def __init__(self, hidden_dim: int = 128, out_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, K, H) 逐点隐状态
        Returns:
            delta: (B, K, out_dim) 注意力修正量
        """
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hidden_dim), dim=-1)
        return attn @ v


class CorridorFlow(nn.Module):
    """
    走廊初始化的条件流匹配速度场 v_θ(X_t, t, c, τ)。
    """

    def __init__(
        self,
        d_context: int = 128,
        d_z: int = 128,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        tau_emb_dim: int = 16,
        time_emb_dim: int = 16,
        k_max: int = 10,
    ):
        super().__init__()
        self.d_context = d_context
        self.d_z = d_z
        self.hidden_dim = hidden_dim
        self.k_max = int(k_max)

        # 全局分支: 把走廊整体形状压缩为定长向量
        self.z_net = nn.Sequential(
            nn.Linear(d_context + 3 * self.k_max, d_z),
            nn.SiLU(),
            nn.Linear(d_z, d_z),
        )

        self.tau_embed = SinusoidalEmbedding(tau_emb_dim)
        self.time_embed = SinusoidalEmbedding(time_emb_dim)

        in_dim = 3 + tau_emb_dim + time_emb_dim + d_z
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(n_hidden_layers - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 3))
        self.step_net = nn.Sequential(*layers)
        # 输出层零初始化: 早期采样停留在走廊先验，不破坏先验质量
        nn.init.zeros_(self.step_net[-1].weight)
        nn.init.zeros_(self.step_net[-1].bias)

        self.attention = TemporalAttentionMixer(hidden_dim, out_dim=3)

    # ── 全局条件 ──────────────────────────────────────

    def encode_global(
        self,
        context: torch.Tensor,
        x0: torch.Tensor,
        x0_ref: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        全局分支: z_gen = MLP(c, X_0^flat)。

        走廊长度 K 可变（推理时预测点数可配置）。为了让全局条件与 K 无关，
        训练和推理都使用同一段 k_max 步的参考走廊（x0_ref，缺省为 x0 自身；
        步数非 k_max 时先线性重采样）。否则同一历史在不同预测点数下会得到
        不同的首步预测，界面上的点数滑块会改变预测行为。
        """
        ref = x0 if x0_ref is None else x0_ref
        B, K, _ = ref.shape
        if K != self.k_max:
            ref = F.interpolate(
                ref.transpose(1, 2), size=self.k_max, mode='linear', align_corners=True,
            ).transpose(1, 2)
        return self.z_net(torch.cat([context, ref.reshape(B, -1)], dim=-1))

    # ── 速度场 ────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,           # (B, K, 3) 当前轨迹
        t: torch.Tensor,           # (B,) 或标量 流时间
        z_gen: torch.Tensor,       # (B, d_z) 全局条件
        tau: torch.Tensor,         # (B, K) 未来相对时间（秒）
    ) -> torch.Tensor:
        """
        Returns:
            velocity: (B, K, 3)
        """
        B, K, _ = x.shape
        if not torch.is_tensor(t):
            t = torch.full((B,), float(t), device=x.device)
        elif t.dim() == 0:
            t = t.expand(B)
        t_view = t.reshape(B, 1).expand(B, K)

        h = torch.cat([
            x,
            self.tau_embed(tau),
            self.time_embed(t_view),
            z_gen.unsqueeze(1).expand(B, K, self.d_z),
        ], dim=-1)

        hidden = self.step_net[:-1](h)
        velocity = self.step_net[-1](hidden)
        return velocity + self.attention(hidden)

    # ── 训练: 条件流匹配 ──────────────────────────────

    def flow_matching_loss(
        self,
        x0: torch.Tensor,        # (B, K, 3) 走廊先验
        x1: torch.Tensor,        # (B, K, 3) 真实轨迹（归一化）
        context: torch.Tensor,   # (B, d_context)
        tau: torch.Tensor,       # (B, K) 未来相对时间
        t: torch.Tensor = None,  # (B,) 可选，默认 U[0,1] 采样
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss: 标量 L_flow
            velocity: (B, K, 3) 预测速度（用于物理正则/诊断）
        """
        B = x0.shape[0]
        if t is None:
            t = torch.rand(B, device=x0.device)
        t_view = t.detach().reshape(B, 1, 1)

        x_t = (1.0 - t_view) * x0 + t_view * x1
        target = x1 - x0

        z_gen = self.encode_global(context, x0)
        velocity = self.forward(x_t, t.detach(), z_gen, tau)
        return F.mse_loss(velocity, target), velocity

    # ── 推理: 并行 Euler 采样 ─────────────────────────

    def sample(
        self,
        x0: torch.Tensor,        # (B, K, 3) 走廊先验
        context: torch.Tensor,   # (B, d_context)
        tau: torch.Tensor,       # (B, K)
        n_steps: int = 4,
        sigma: float = 0.0,
        n_samples: int = 1,
        x0_ref: torch.Tensor = None,   # (B, k_max, 3) 全局分支参考走廊
    ) -> torch.Tensor:
        """
        从走廊出发做 n_steps 步并行 Euler 积分（可微，训练时可用于投影器在环）。

        Args:
            n_steps:   Euler 步数（论文默认 4）
            sigma:     起点注入噪声标准差（归一化坐标，0=确定性均值预测）
            n_samples: 采样条数（概率预测，配合 sigma>0）

        Returns:
            (S, B, K, 3) 当 n_samples>1，否则 (B, K, 3)
        """
        B, K, C = x0.shape
        n_samples = max(int(n_samples), 1)

        if n_samples > 1:
            x = x0.unsqueeze(0).expand(n_samples, B, K, C).reshape(-1, K, C)
            ctx = context.unsqueeze(0).expand(n_samples, B, context.shape[-1]).reshape(-1, context.shape[-1])
            tau_r = tau.unsqueeze(0).expand(n_samples, B, K).reshape(-1, K)
        else:
            x, ctx, tau_r = x0, context, tau

        ref = None if x0_ref is None else (
            x0_ref.unsqueeze(0).expand(n_samples, *x0_ref.shape).reshape(-1, x0_ref.shape[1], 3)
            if n_samples > 1 else x0_ref
        )
        z_gen = self.encode_global(ctx, x, ref)
        if sigma > 0:
            x = x + torch.randn_like(x) * sigma

        n_steps = max(int(n_steps), 1)
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i / n_steps, device=x.device)
            x = x + self.forward(x, t, z_gen, tau_r) / n_steps

        x = x.reshape(n_samples, B, K, C)
        return x if n_samples > 1 else x[0]


__all__ = ["CorridorFlow", "SinusoidalEmbedding", "TemporalAttentionMixer"]
