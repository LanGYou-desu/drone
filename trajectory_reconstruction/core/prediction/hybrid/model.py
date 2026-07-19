"""
Phy-ODE-Diffusion 顶层模型

完整的自回归轨迹预测流水线:
  1. Transformer 编码历史轨迹 → 上下文向量 c
  2. ODE 状态初始化 h_0 = [p_N, v_N, MLP(c)]
  3. 循环 K 步:
     a. ODE 演化 → h_prior
     b. 物理引导扩散采样 → p_next
     c. 虚拟观测计算 v_obs
     d. GRU 状态更新 → h_new
"""

import torch
import torch.nn as nn
import numpy as np

from trajectory_reconstruction.core.prediction.hybrid.transformer import TransformerEncoder
from trajectory_reconstruction.core.prediction.hybrid.ode_manager import ODEStateManager
from trajectory_reconstruction.core.prediction.hybrid.diffusion import GuidedDiffusion
from trajectory_reconstruction.core.math_utils import estimate_velocity


class PhyODEDiffusion(nn.Module):
    """
    Phy-ODE-Diffusion 混合轨迹预测模型。

    使用示例:
        model = PhyODEDiffusion(...)
        model.load_state_dict(torch.load("models/hybrid_predictor/phy_ode_diffusion.pt"))

        # 预测
        pred_positions, pred_times = model.predict(
            points, timestamps, num_points=20, time_step=0.3,
        )
    """

    def __init__(
        self,
        # Transformer
        d_feat: int = 64,
        d_context: int = 128,
        n_head: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        # ODE
        d_z: int = 32,
        a_max: float = 30.0,
        ode_hidden_dim: int = 64,
        # Diffusion
        n_diffusion_steps: int = 500,
        n_inference_steps: int = 50,
        tau_emb_dim: int = 16,
        dt_emb_dim: int = 16,
        diff_hidden_dim: int = 128,
        guidance_eta: float = 0.1,
        v_max: float = 30.0,
        z_min: float = 0.0,
        z_max: float = 120.0,      # 四旋翼最大高度
        v_v_up: float = 5.0,       # 最大垂直上升速度
        v_v_down: float = 3.0,     # 最大垂直下降速度
        max_tilt: float = 35.0,    # 最大倾斜角
        g: float = 9.81,           # 重力加速度
        # GRU
        obs_hidden_dim: int = 32,
    ):
        super().__init__()
        self.d_context = d_context
        self.d_z = d_z
        self.state_dim = 6 + d_z
        self.v_max = v_max
        self.a_max = a_max
        self.z_min = z_min
        self.z_max = z_max
        self.v_v_up = v_v_up
        self.v_v_down = v_v_down
        self.max_tilt = max_tilt
        self.g = g

        # 子模块
        self.transformer = TransformerEncoder(
            d_feat=d_feat,
            d_context=d_context,
            n_head=n_head,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.state_manager = ODEStateManager(
            d_context=d_context,
            d_z=d_z,
            a_max=a_max,
            hidden_dim=ode_hidden_dim,
            obs_hidden_dim=obs_hidden_dim,
        )

        self.diffusion = GuidedDiffusion(
            state_dim=self.state_dim,
            n_diffusion_steps=n_diffusion_steps,
            n_inference_steps=n_inference_steps,
            tau_emb_dim=tau_emb_dim,
            dt_emb_dim=dt_emb_dim,
            hidden_dim=diff_hidden_dim,
            guidance_eta=guidance_eta,
            v_max=v_max, a_max=a_max,
            z_min=z_min, z_max=z_max,
            v_v_up=v_v_up, v_v_down=v_v_down,
            max_tilt=max_tilt, g=g,
        )

    # ── 训练接口 ──────────────────────────────────────

    def forward(
        self,
        ctx_t: torch.Tensor,
        ctx_dt: torch.Tensor,
        ctx_pos: torch.Tensor,
        ctx_vel: torch.Tensor,
        tgt_t: torch.Tensor,
        tgt_pos: torch.Tensor,
        tgt_vel: torch.Tensor = None,
        mask: torch.Tensor = None,
        p_mean: torch.Tensor = None,   # (B, 3) 归一化统计量
        p_std: torch.Tensor = None,    # (B, 3)
    ) -> dict:
        """训练前向传播（教师强制模式）。

        注意: 此 forward() 方法预留用于标准 PyTorch 训练循环（loss = model(...)），
        当前 train/hybrid_predictor/train.py 分阶段训练脚本直接调用子模块
        （model.transformer / model.state_manager / model.diffusion），
        以便在各阶段使用不同的损失函数和冻结策略。
        """
        # 1. Transformer 编码
        c = self.transformer(ctx_t, ctx_dt, ctx_pos, ctx_vel, mask)

        # 2. 初始化 ODE 状态
        h = self.state_manager.init_state(
            ctx_pos[:, -1, :],    # p_N
            ctx_vel[:, -1, :],    # v_N_est
            c,
        )
        t_now = ctx_t[:, -1]     # (B,)

        total_diff_loss = 0.0
        total_phy_loss = 0.0
        count = 0

        # 3. 逐目标点训练
        for j in range(tgt_t.shape[1]):
            t_next = tgt_t[:, j]          # (B,)
            dt = t_next - t_now           # (B,)

            # ODE 演化
            h_prior = self.state_manager.evolve(h, t_now, t_next)

            # 扩散损失（含物理正则，使用原始空间统计量）
            loss_dict = self.diffusion.compute_loss(
                h_prior, dt,
                ctx_pos[:, -1, :] if j == 0 else tgt_pos[:, j-1, :],
                tgt_pos[:, j, :],
                p_mean=p_mean, p_std=p_std,
            )
            total_diff_loss += loss_dict["diff_loss"]
            total_phy_loss += loss_dict["physics_loss"]

            # 教师强制：用真实观测更新状态
            v_obs = tgt_vel[:, j, :] if tgt_vel is not None else \
                    (tgt_pos[:, j, :] - (ctx_pos[:, -1, :] if j == 0
                                         else tgt_pos[:, j-1, :])) / dt.clamp(min=1e-3).unsqueeze(-1)
            h = self.state_manager.update(h_prior, dt, tgt_pos[:, j, :], v_obs)
            t_now = t_next
            count += 1

        return {
            "diff_loss": total_diff_loss / max(count, 1),
            "physics_loss": total_phy_loss / max(count, 1),
            "total_loss": (total_diff_loss + 0.1 * total_phy_loss) / max(count, 1),
        }

    # ── 推理接口 ──────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        timestamps: list[float],
        positions: list[list[float]],
        num_points: int = 20,
        time_step: float = 0.3,
        device: str = "cpu",
    ) -> tuple[list[list[float]], list[float]]:
        """
        自回归轨迹预测（单条轨迹）。

        Args:
            timestamps: 历史时间戳列表 [t_1, ..., t_N]
            positions:  历史位置列表 [[x,y,z], ...]
            num_points: 预测点数
            time_step:  预测时间步长（秒）
            device:     推理设备

        Returns:
            pred_positions: [[x,y,z], ...]  共 num_points+1 个点（含锚点）
            pred_times:     [t, ...]
        """
        self.eval()
        device_obj = torch.device(device)

        # 预处理
        N = len(positions)
        t_arr = np.array(timestamps, dtype=np.float32)
        p_arr = np.array(positions, dtype=np.float32)

        # 计算 Δt 和估计速度（使用共享工具函数）
        dt_arr = np.zeros(N, dtype=np.float32)
        dt_arr[1:] = t_arr[1:] - t_arr[:-1]
        v_arr = estimate_velocity(p_arr, t_arr)

        # 标准化
        p_mean = p_arr.mean(axis=0)
        p_std = p_arr.std(axis=0).clip(min=1e-3)
        p_norm = (p_arr - p_mean) / p_std
        v_norm = v_arr / p_std

        # 转为 tensor (B=1)
        t_t = torch.from_numpy(t_arr).unsqueeze(0).to(device_obj)
        dt_t = torch.from_numpy(dt_arr).unsqueeze(0).to(device_obj)
        p_t = torch.from_numpy(p_norm).unsqueeze(0).to(device_obj)
        v_t = torch.from_numpy(v_norm).unsqueeze(0).to(device_obj)

        # 编码
        c = self.transformer(t_t, dt_t, p_t, v_t)

        # 初始化 ODE 状态
        h = self.state_manager.init_state(p_t[:, -1, :], v_t[:, -1, :], c)
        t_now = float(t_arr[-1])          # 标量
        dt_step = torch.tensor([time_step], device=device_obj)

        last_pos = p_t[:, -1, :].clone()  # (1, 3)
        preds_norm = [last_pos.squeeze(0).cpu().numpy().tolist()]
        preds_times = [t_now]

        # 自回归预测
        for _ in range(num_points):
            t_next = t_now + time_step

            # ODE 演化
            h_prior = self.state_manager.evolve(h,
                torch.tensor([t_now], device=device_obj),
                torch.tensor([t_next], device=device_obj))

            # 扩散采样（传入归一化统计量用于物理引导反标准化）
            p_mean_t = torch.from_numpy(p_mean).unsqueeze(0).float().to(device_obj)
            p_std_t = torch.from_numpy(p_std).unsqueeze(0).float().to(device_obj)
            p_next_norm = self.diffusion.guided_sampling(
                h_prior, dt_step, last_pos,
                n_steps=self.diffusion.n_inference_steps,
                p_mean=p_mean_t, p_std=p_std_t,
            )

            # 虚拟观测
            v_obs = (p_next_norm - last_pos) / time_step

            # 状态更新
            h = self.state_manager.update(h_prior, dt_step, p_next_norm, v_obs)

            last_pos = p_next_norm
            t_now = t_next

            preds_norm.append(p_next_norm.squeeze(0).cpu().numpy().tolist())
            preds_times.append(preds_times[-1] + time_step)

        # 反标准化
        preds_raw = np.array(preds_norm) * p_std + p_mean
        # NaN 保护：用最后一个有效值填充
        valid_mask = ~np.isnan(preds_raw).any(axis=1)
        if valid_mask.any() and not valid_mask.all():
            last_valid = preds_raw[valid_mask][-1]
            preds_raw[~valid_mask] = last_valid
        elif not valid_mask.any():
            # 全部 NaN，回退到线性外推的结果
            preds_raw = np.tile(p_arr[-1:], (len(preds_norm), 1))
        pred_positions = preds_raw.tolist()
        pred_times_list = [round(t, 6) for t in preds_times]

        return pred_positions, pred_times_list

    def get_model_info(self) -> dict:
        """返回模型元信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "d_context": self.d_context,
            "d_z": self.d_z,
            "state_dim": self.state_dim,
            "v_max": self.v_max,
            "a_max": self.a_max,
            "z_min": self.z_min,
        }
