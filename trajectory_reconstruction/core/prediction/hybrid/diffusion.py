"""
物理引导扩散生成器 — Physics-Guided Diffusion Predictor

在 R³ 位置空间中执行去噪扩散过程，以 ODE 先验状态为条件生成下一时刻位置。
推理时嵌入基于梯度的物理约束引导，使生成的位置自然满足速度/加速度/高度约束。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 噪声调度器 ──────────────────────────────────────────

class NoiseScheduler:
    """
    扩散噪声调度器 — 支持 Cosine / Linear 调度，DDPM / DDIM 采样。
    """

    def __init__(
        self,
        n_steps: int = 500,
        beta_end: float = 0.02,
        schedule: str = "cosine",
    ):
        self.n_steps = n_steps

        if schedule == "cosine":
            betas = self._cosine_schedule(n_steps, beta_end)
        else:
            betas = torch.linspace(1e-4, beta_end, n_steps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    @staticmethod
    def _cosine_schedule(n_steps: int, max_beta: float = 0.999) -> torch.Tensor:
        steps = torch.arange(n_steps + 1, dtype=torch.float32)
        alpha_bar = torch.cos((steps / n_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(betas, max=max_beta)

    def _gather(self, arr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        gathered = arr[t.clamp(0, len(arr) - 1)]
        return gathered.view(*gathered.shape, *([1] * max(0, 2 - gathered.dim())))

    def to(self, device: torch.device):
        """将所有张量移到指定设备"""
        for key in list(self.__dict__.keys()):
            val = getattr(self, key)
            if isinstance(val, torch.Tensor):
                setattr(self, key, val.to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """前向扩散: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self._gather(self.sqrt_alphas_cumprod, t)
        sqrt_1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return sqrt_a * x0 + sqrt_1ma * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor,
                   noise_pred: torch.Tensor) -> torch.Tensor:
        """x̂_0 = (x_t - √(1-ᾱ_t)·ε) / √(ᾱ_t)"""
        sqrt_a = self._gather(self.sqrt_alphas_cumprod, t)
        sqrt_1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return (x_t - sqrt_1ma * noise_pred) / sqrt_a.clamp(min=1e-8)

    def ddim_step(self, x_t: torch.Tensor, noise_pred: torch.Tensor,
                  t: int, t_prev: int) -> torch.Tensor:
        """DDIM 确定性采样步"""
        device = x_t.device
        alpha_bar_t = self._gather(self.alphas_cumprod,
                                   torch.tensor([t], device=device))
        alpha_bar_prev = (
            self._gather(self.alphas_cumprod, torch.tensor([t_prev], device=device))
            if t_prev >= 0
            else torch.tensor([[1.0]], device=device)
        )
        x0_pred = self.predict_x0(x_t, torch.tensor([t], device=device), noise_pred)
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + \
                 torch.sqrt(1.0 - alpha_bar_prev) * noise_pred
        return x_prev


# ── 嵌入层 ──────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """正弦位置嵌入（用于扩散时间步 τ）"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = max(self.dim // 2, 1)
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32)
                        * -emb_scale)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ── 噪声预测网络 ────────────────────────────────────────

class NoisePredictionNet(nn.Module):
    """
    条件噪声预测网络 ε_φ(x_τ, τ, h_prior, Δt)

    输入含噪位置 x_τ ∈ R³ 及条件信息，预测注入的噪声 ε ∈ R³。
    """

    def __init__(
        self,
        state_dim: int = 38,
        tau_emb_dim: int = 16,
        dt_emb_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.tau_embed = SinusoidalEmbedding(tau_emb_dim)
        self.dt_embed = nn.Sequential(
            nn.Linear(1, dt_emb_dim),
            nn.SiLU(),
            nn.Linear(dt_emb_dim, dt_emb_dim),
        )
        cond_dim = state_dim + tau_emb_dim + dt_emb_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(3 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        # 最后一层趋近零输出
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x_tau: torch.Tensor,        # (B, 3) 含噪位置
        tau: torch.Tensor,          # (B,) 扩散时间步
        h_prior: torch.Tensor,      # (B, state_dim) ODE 先验状态
        dt: torch.Tensor,           # (B,) 物理时间步长
    ) -> torch.Tensor:
        tau_emb = self.tau_embed(tau)
        dt_emb = self.dt_embed(dt.unsqueeze(-1))
        cond = torch.cat([h_prior, tau_emb, dt_emb], dim=-1)
        cond = self.cond_proj(cond)
        inp = torch.cat([x_tau, cond], dim=-1)
        return self.net(inp)


# ── 物理引导扩散生成器 ──────────────────────────────────

class GuidedDiffusion(nn.Module):
    """
    物理引导扩散生成器。

    在自回归预测的每一步，以 ODE 先验状态为条件，
    通过 DDIM 去噪采样生成下一时刻位置，并在采样过程中嵌入物理约束引导。
    """

    def __init__(
        self,
        state_dim: int = 38,
        n_diffusion_steps: int = 500,
        n_inference_steps: int = 50,
        tau_emb_dim: int = 16,
        dt_emb_dim: int = 16,
        hidden_dim: int = 128,
        guidance_eta: float = 0.1,
        v_max: float = 30.0,
        a_max: float = 30.0,
        z_min: float = 0.0,
    ):
        super().__init__()
        self.n_diffusion_steps = n_diffusion_steps
        self.n_inference_steps = n_inference_steps
        self.guidance_eta = guidance_eta
        self.v_max = v_max
        self.a_max = a_max
        self.z_min = z_min
        self.state_dim = state_dim

        self.scheduler = NoiseScheduler(n_steps=n_diffusion_steps, schedule="cosine")
        self.noise_net = NoisePredictionNet(
            state_dim=state_dim,
            tau_emb_dim=tau_emb_dim,
            dt_emb_dim=dt_emb_dim,
            hidden_dim=hidden_dim,
        )

        # 物理约束权重（不可训练，通过配置调整）
        self.register_buffer('lambda_v', torch.tensor(1.0))
        self.register_buffer('lambda_a', torch.tensor(1.0))
        self.register_buffer('lambda_z', torch.tensor(1.0))

    def compute_loss(
        self,
        h_prior: torch.Tensor,
        dt: torch.Tensor,
        prev_p: torch.Tensor,
        x0_gt: torch.Tensor,
        p_mean: torch.Tensor = None,   # (B, 3) 归一化均值，用于反标准化物理约束
        p_std: torch.Tensor = None,    # (B, 3) 归一化标准差
    ) -> dict:
        """
        训练模式：扩散去噪损失 + 可选的物理正则。

        若提供 p_mean/p_std，物理代价在反标准化后的原始空间中计算，
        梯度通过 1/p_std 缩放正确回传。
        """
        B = h_prior.shape[0]
        device = h_prior.device

        noise = torch.randn_like(x0_gt)
        tau = torch.randint(0, self.n_diffusion_steps, (B,), device=device)
        x_tau = self.scheduler.q_sample(x0_gt, tau, noise)
        noise_pred = self.noise_net(x_tau, tau, h_prior, dt)
        diff_loss = F.mse_loss(noise_pred, noise)

        # 物理正则：反标准化后在原始空间计算，梯度正确缩放
        phy_loss = 0.0
        if p_mean is not None and p_std is not None:
            with torch.no_grad():
                x0_pred_norm = self.scheduler.predict_x0(x_tau, tau, noise_pred)
            # 反标准化到物理空间
            x0_pred_phys = x0_pred_norm * p_std + p_mean
            prev_p_phys = prev_p * p_std + p_mean
            dt_phys = dt  # 时间不变
            phy_loss = self._physics_cost(x0_pred_phys, prev_p_phys, dt_phys).mean()

        return {"diff_loss": diff_loss, "physics_loss": phy_loss}

    @torch.no_grad()
    def guided_sampling(
        self,
        h_prior: torch.Tensor,
        dt: torch.Tensor,
        prev_p: torch.Tensor,
        n_steps: int = None,
        p_mean: torch.Tensor = None,   # 反标准化统计量
        p_std: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        物理引导的 DDIM 采样，生成下一时刻位置。

        Args:
            h_prior: (B, state_dim) ODE 先验状态
            dt: (B,) 物理时间步长
            prev_p: (B, 3) 上一时刻位置
            n_steps: DDIM 采样步数

        Returns:
            p_next: (B, 3) 预测位置
        """
        if n_steps is None:
            n_steps = self.n_inference_steps

        B = h_prior.shape[0]
        device = h_prior.device

        x_t = torch.randn(B, 3, device=device)
        times = torch.linspace(self.n_diffusion_steps - 1, 0,
                               n_steps + 1, device=device).long()

        for i in range(n_steps):
            t = times[i].item()
            t_prev = times[i + 1].item()
            tau_t = torch.full((B,), t, device=device, dtype=torch.long)

            noise_pred = self.noise_net(x_t, tau_t, h_prior, dt)

            # 物理引导: 若提供反标准化统计量则在原始空间计算
            guidance_grad = self._compute_guidance(
                x_t, tau_t, h_prior, prev_p, dt, p_mean, p_std,
            )
            guided_noise = noise_pred - guidance_grad

            x_t = self.scheduler.ddim_step(x_t, guided_noise, int(t), int(t_prev))

        return x_t

    def _compute_guidance(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        h_prior: torch.Tensor,
        prev_p: torch.Tensor,
        dt: torch.Tensor,
        p_mean: torch.Tensor = None,
        p_std: torch.Tensor = None,
    ) -> torch.Tensor:
        """物理约束梯度: 若提供 p_mean/p_std 则在原始空间计算"""
        with torch.enable_grad():
            x_tau_grad = x_tau.detach().requires_grad_(True)
            noise_pred_grad = self.noise_net(x_tau_grad, tau, h_prior, dt)

            sqrt_a = self.scheduler._gather(self.scheduler.sqrt_alphas_cumprod, tau)
            sqrt_1ma = self.scheduler._gather(
                self.scheduler.sqrt_one_minus_alphas_cumprod, tau,
            )
            x0_pred_norm = (x_tau_grad - sqrt_1ma * noise_pred_grad) / sqrt_a.clamp(min=1e-8)

            if p_mean is not None and p_std is not None:
                x0_pred = x0_pred_norm * p_std + p_mean
                prev = prev_p * p_std + p_mean
            else:
                x0_pred = x0_pred_norm
                prev = prev_p

            cost = self._physics_cost(x0_pred, prev, dt)
            grad = torch.autograd.grad(cost.sum(), x_tau_grad)[0]
            scale = sqrt_1ma.detach()
            return self.guidance_eta * scale * grad.detach()

    def _physics_cost(
        self,
        p: torch.Tensor,
        prev_p: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        物理约束违反代价 C(p)：
          - 速度 ≤ v_max
          - 加速度 ≤ a_max（基于当前位置速度的变化率）
          - 高度 ≥ z_min
        """
        dt_safe = dt.clamp(min=1e-3).unsqueeze(-1)

        # 速度违反
        v = (p - prev_p) / dt_safe
        v_norm = torch.norm(v, dim=-1)
        cost = self.lambda_v * F.relu(v_norm - self.v_max) ** 2

        # 加速度违反：|v| / dt 作为加速度上界估计
        # 保守估计：从零到当前速度所需的最小加速度
        a_norm = v_norm / dt_safe.squeeze(-1)
        cost = cost + self.lambda_a * F.relu(a_norm - self.a_max) ** 2

        # 高度违反（y < z_min）
        cost = cost + self.lambda_z * F.relu(self.z_min - p[:, 1]) ** 2

        return cost
