"""
连续时间上下文编码器 — Continuous-Time Context Encoder

KinFlow 的第一个模块（论文 §3.2）。对每个历史采样点构造 7 维特征
    f_i = Linear_{7→d_f}([Δt_i; p_i; v_i^est])
并叠加在采样时刻 t_i 上求值的连续时间正弦编码
    e_i = [sin(ω_1 t_i), cos(ω_1 t_i), ..., sin(ω_{d_f/2} t_i), cos(ω_{d_f/2} t_i)]
再通过带 CLS token 的 4 层 Pre-LN Transformer 把整段历史压缩为上下文向量
    c = MLP_ctx(Transformer(CLS, f_1+e_1, ..., f_N+e_N)_CLS) ∈ R^{d_c}

相对时间说明: 正弦编码在时间平移下仅相差一个相位，编码器本身依赖相对时间；
为保持 float32 在大时间戳下的数值精度，这里传入的是相对窗口起点的时间
(t_i - t_1)，等价于论文的绝对时间编码（同一组频率、同一函数形式）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousTimeEncoding(nn.Module):
    """
    连续时间正弦编码。

    把任意实数时间映射到有界的周期特征，无需分箱、填充或插值，
    因此天然支持不规则采样间隔与任意查询时刻。
    """

    def __init__(self, dim: int = 64, min_freq: float = 0.1, max_freq: float = 10.0):
        super().__init__()
        self.dim = dim
        half = max(dim // 2, 1)
        # 频率在对数尺度上均匀分布，覆盖慢变化到快变化的运动尺度
        freqs = torch.logspace(
            math.log10(min_freq), math.log10(max_freq), half,
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (...,) 时间（秒）
        Returns:
            encoding: (..., dim)
        """
        angles = t.unsqueeze(-1) * self.freqs            # (..., half)
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            enc = F.pad(enc, (0, 1))
        return enc


class ContextEncoder(nn.Module):
    """
    连续时间上下文编码器。

    输入: (t_i, Δt_i, p_i, v_est_i), i = 1..N（长度为 N 的不规则历史）
    输出: c ∈ R^{d_context}
    """

    def __init__(
        self,
        d_feat: int = 64,
        d_context: int = 128,
        n_head: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        min_freq: float = 0.1,
        max_freq: float = 10.0,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.d_context = d_context

        # 特征投影: [Δt(1) + p(3) + v_est(3)] = 7 → d_feat
        self.feature_proj = nn.Sequential(
            nn.Linear(7, d_feat),
            nn.LayerNorm(d_feat),
            nn.SiLU(),
        )
        self.time_encoding = ContinuousTimeEncoding(d_feat, min_freq, max_freq)

        layer = nn.TransformerEncoderLayer(
            d_model=d_feat,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN: 训练更稳定，梯度更平滑
        )
        self.body = nn.TransformerEncoder(
            layer, num_layers=n_layers,
            enable_nested_tensor=False,   # 避免 Pre-LN 下的嵌套张量告警
        )

        # 上下文压缩 MLP: CLS 输出 → c
        self.context_mlp = nn.Sequential(
            nn.Linear(d_feat, d_context),
            nn.LayerNorm(d_context),
            nn.Tanh(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_feat) * 0.02)

    def forward(
        self,
        t: torch.Tensor,          # (B, N) 时间戳（秒）
        dt: torch.Tensor,         # (B, N) 时间间隔（dt_1 = 0）
        pos: torch.Tensor,        # (B, N, 3) 归一化位置
        vel_est: torch.Tensor,    # (B, N, 3) 中心差分估计速度
        mask: torch.Tensor = None,  # (B, N) padding mask, True=pad
    ) -> torch.Tensor:
        """
        Returns:
            context: (B, d_context)
        """
        B, N = t.shape

        feats = torch.cat([dt.unsqueeze(-1), pos, vel_est], dim=-1)  # (B, N, 7)
        tokens = self.feature_proj(feats)                            # (B, N, d_feat)

        # 相对窗口起点的时间（数值稳定，语义等价于绝对时间编码）
        t_rel = t - t[:, :1]
        tokens = tokens + self.time_encoding(t_rel)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)              # (B, 1+N, d_feat)

        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            key_padding_mask = None

        encoded = self.body(tokens, src_key_padding_mask=key_padding_mask)
        cls_out = encoded[:, 0, :]                  # (B, d_feat)
        return self.context_mlp(cls_out)            # (B, d_context)


__all__ = ["ContextEncoder", "ContinuousTimeEncoding"]
