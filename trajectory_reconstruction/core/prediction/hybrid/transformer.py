"""
Transformer 编码器 — 不规则时间序列编码

对历史轨迹点提取特征（时间间隔 + 位置 + 估计速度），
加入连续时间位置编码，通过 Transformer Encoder 压缩为上下文向量。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousTimeEncoding(nn.Module):
    """
    连续时间位置编码。

    对时间戳 t 生成正弦/余弦编码，频率与 t 成比例。
    使 Transformer 能感知绝对时间和时间间隔。
    """

    def __init__(self, d_model: int = 64, max_freq: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
        self.freq_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, N) 时间戳
        Returns:
            encoding: (B, N, d_model)
        """
        B, N = t.shape
        device = t.device

        freqs = torch.logspace(
            0, math.log10(self.max_freq),
            self.d_model // 2, device=device,
        )
        freqs = freqs * self.freq_scale.abs()
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.d_model % 2 == 1:
            encoding = F.pad(encoding, (0, 1))
        return encoding


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器：将不等间隔的历史轨迹编码为固定维度上下文向量。

    输入:  (t_i, Δt_i, p_i, v_est_i) for i=1..N
    输出:  c ∈ R^{d_c}
    """

    def __init__(
        self,
        d_feat: int = 64,
        d_context: int = 128,
        n_head: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_freq: float = 10.0,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.d_context = d_context

        # 输入投影: [dt(1) + p(3) + v_est(3)] = 7 → d_feat
        self.input_proj = nn.Sequential(
            nn.Linear(7, d_feat),
            nn.LayerNorm(d_feat),
            nn.ReLU(),
        )

        self.time_encoding = ContinuousTimeEncoding(d_feat, max_freq)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_feat,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LN: 更稳定的训练，梯度流更好
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,  # Pre-LN 下禁用嵌套张量避免警告
        )

        self.context_proj = nn.Sequential(
            nn.Linear(d_feat, d_context),
            nn.LayerNorm(d_context),
            nn.Tanh(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_feat) * 0.02)

    def forward(
        self,
        t: torch.Tensor,         # (B, N) 时间戳
        dt: torch.Tensor,        # (B, N) 时间间隔（dt_1=0）
        pos: torch.Tensor,       # (B, N, 3) 位置
        vel_est: torch.Tensor,   # (B, N, 3) 估计速度
        mask: torch.Tensor = None,  # (B, N) padding mask (True=pad)
    ) -> torch.Tensor:
        """
        Returns:
            context: (B, d_context) 上下文向量
        """
        B, N = t.shape

        feats = torch.cat([dt.unsqueeze(-1), pos, vel_est], dim=-1)  # (B, N, 7)
        feats = self.input_proj(feats)                                # (B, N, d_feat)

        time_enc = self.time_encoding(t)  # (B, N, d_feat)
        tokens = feats + time_enc

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+N, d_feat)

        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            key_padding_mask = None

        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        cls_out = encoded[:, 0, :]          # (B, d_feat)
        context = self.context_proj(cls_out)  # (B, d_context)
        return context
