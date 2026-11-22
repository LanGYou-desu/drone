"""
KinFlow 顶层模型 — 学习式运动学先验 + 并行非自回归细化

完整预测流水线（论文 §3.6，式 (7)）:
    X̂ = Projector ∘ Flow ∘ Corridor ∘ Encoder(history)

  1. ContextEncoder   : 不规则历史 → 上下文向量 c
  2. KinematicCorridor: (c, p_N) → (v_0, a)，在真实未来时刻闭式外推常数加速度先验 X_0
  3. CorridorFlow     : 从 X_0 出发，条件流匹配训练的速度场做少量并行 Euler 步 → X
  4. PhysicsProjector : 物理坐标下硬性投影速度/垂直速率/高度边界 → X̂

整个未来 K 步在一次前向中并行产生：没有自回归、没有逐步迭代，
因此误差不会沿时间累积，延迟与 K 无关。

训练损失（论文 §3.7，式 (8)）:
    L = L_flow + λ_c·MSE(X_0, X_gt) + λ_p·L_phy      (λ_c=1.0, λ_p=0.01)
"""

import numpy as np
import torch
import torch.nn as nn

from trajectory_reconstruction.core.math_utils import estimate_velocity
from trajectory_reconstruction.core.prediction.kinflow.corridor import KinematicCorridor
from trajectory_reconstruction.core.prediction.kinflow.encoder import ContextEncoder
from trajectory_reconstruction.core.prediction.kinflow.flow import CorridorFlow
from trajectory_reconstruction.core.prediction.kinflow.projector import (
    KinematicBounds, PhysicsProjector, kinematic_violation_cost,
)

# 逐轴 Z-score 标准差的物理下限（米）。
# 平飞轨迹的垂直/侧向变化可能只有几厘米，若无下限，归一化坐标会被放大几十倍，
# 使流匹配回归的数值尺度失衡。0.25 m 在保留相对形状的同时避免病态放大。
POSITION_STD_FLOOR = 0.25


class KinFlow(nn.Module):
    """
    KinFlow 无人机轨迹预测模型。

    使用示例::

        model = KinFlow()
        ckpt = torch.load("models/kinflow/kinflow_best_s2.pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        pred_pos, pred_time = model.predict(timestamps, positions, num_points=20, time_step=0.3)
    """

    def __init__(
        self,
        # 编码器
        d_feat: int = 64,
        d_context: int = 128,
        n_head: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        # 运动学走廊
        corridor_hidden: int = 128,
        # 走廊流
        d_z: int = 128,
        flow_hidden: int = 128,
        flow_hidden_layers: int = 3,
        tau_emb_dim: int = 16,
        time_emb_dim: int = 16,
        k_max: int = 10,
        ctx_len: int = 20,
        sample_dt: float = 0.1,
        max_native_steps: int = 48,
        # 采样
        n_euler_steps: int = 4,
        sigma: float = 0.0,
        # 物理边界（默认对齐 config.json → drone_dynamics）
        v_h_max: float = 20.0,
        v_up_max: float = 5.0,
        v_down_max: float = 3.0,
        y_min: float = 1.0,
        y_max: float = 120.0,
        max_tilt: float = 35.0,
        g: float = 9.81,
        n_project_iters: int = 3,
        project: bool = True,
    ):
        super().__init__()

        self.bounds = KinematicBounds(
            v_h_max=v_h_max, v_up_max=v_up_max, v_down_max=v_down_max,
            y_min=y_min, y_max=y_max, max_tilt=max_tilt, g=g,
        )
        self.n_euler_steps = int(n_euler_steps)
        self.sigma = float(sigma)
        self.project = bool(project)
        self.k_max = int(k_max)
        # 训练时的窗口规格: C 点、名义采样间隔 dt。
        # 推理时把任意长度的输入历史重采样到同一时间栅格，保证归一化尺度、
        # 归一化速度与编码器输入分布和训练完全一致。
        self.ctx_len = int(ctx_len)
        self.sample_dt = float(sample_dt)
        # 单次流匹配可并行的最大未来步数（保护长视界推理耗时）
        self.max_native_steps = int(max(max_native_steps, 2))

        self.encoder = ContextEncoder(
            d_feat=d_feat, d_context=d_context, n_head=n_head, n_layers=n_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.corridor = KinematicCorridor(d_context=d_context, hidden_dim=corridor_hidden)
        self.flow = CorridorFlow(
            d_context=d_context, d_z=d_z, hidden_dim=flow_hidden,
            n_hidden_layers=flow_hidden_layers, tau_emb_dim=tau_emb_dim,
            time_emb_dim=time_emb_dim, k_max=self.k_max,
        )
        self.projector = PhysicsProjector(self.bounds, n_iters=n_project_iters)

        # 完整架构配置（推理时据此重建模型）
        self._model_config = {
            "d_feat": d_feat, "d_context": d_context, "n_head": n_head,
            "n_layers": n_layers, "dim_feedforward": dim_feedforward, "dropout": dropout,
            "corridor_hidden": corridor_hidden,
            "d_z": d_z, "flow_hidden": flow_hidden,
            "flow_hidden_layers": flow_hidden_layers,
            "tau_emb_dim": tau_emb_dim, "time_emb_dim": time_emb_dim, "k_max": k_max,
            "ctx_len": ctx_len, "sample_dt": sample_dt,
            "max_native_steps": max_native_steps,
            "n_euler_steps": n_euler_steps, "sigma": sigma,
            "v_h_max": v_h_max, "v_up_max": v_up_max, "v_down_max": v_down_max,
            "y_min": y_min, "y_max": y_max, "max_tilt": max_tilt, "g": g,
            "n_project_iters": n_project_iters, "project": project,
        }

    # ── 前向组件 ──────────────────────────────────────

    def encode(
        self,
        ctx_t: torch.Tensor,
        ctx_dt: torch.Tensor,
        ctx_pos: torch.Tensor,
        ctx_vel: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """历史编码: (t, Δt, p, v_est) → c ∈ R^{d_context}"""
        return self.encoder(ctx_t, ctx_dt, ctx_pos, ctx_vel, mask)

    def corridor_prior(
        self,
        context: torch.Tensor,
        last_pos: torch.Tensor,
        dtau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """运动学走廊: c, p_N, Δτ → (X_0, v_0, a)"""
        return self.corridor.trajectory(context, last_pos, dtau)

    def physics_loss(
        self,
        traj_norm: torch.Tensor,
        anchor_norm: torch.Tensor,
        dtau: torch.Tensor,
        p_mean: torch.Tensor,
        p_std: torch.Tensor,
        anchor_vel_norm: torch.Tensor = None,
    ) -> torch.Tensor:
        """软物理损失（物理坐标下计算，梯度经 1/p_std 回传到归一化空间）。"""
        if p_mean.dim() == 1:
            p_mean = p_mean.unsqueeze(0)
        if p_std.dim() == 1:
            p_std = p_std.unsqueeze(0)
        traj_phys = traj_norm * p_std.unsqueeze(1) + p_mean.unsqueeze(1)
        anchor_phys = anchor_norm * p_std + p_mean
        anchor_vel_phys = None if anchor_vel_norm is None else anchor_vel_norm * p_std
        return kinematic_violation_cost(traj_phys, anchor_phys, dtau, self.bounds,
                                        anchor_vel=anchor_vel_phys)

    # ── 训练前向 ──────────────────────────────────────

    def forward_train(
        self,
        ctx_t: torch.Tensor,
        ctx_dt: torch.Tensor,
        ctx_pos: torch.Tensor,
        ctx_vel: torch.Tensor,
        tgt_t: torch.Tensor,
        tgt_pos: torch.Tensor,
        p_mean: torch.Tensor,
        p_std: torch.Tensor,
        mask: torch.Tensor = None,
        sigma: float = 0.0,
        lambda_corridor: float = 1.0,
        lambda_physics: float = 0.01,
        use_projector: bool = False,
        lambda_projector: float = 0.1,
        n_project_steps: int = 2,
        lambda_sample: float = 0.0,
        n_sample_steps: int = 2,
    ) -> dict:
        """
        单次前向的联合训练损失。

        Args:
            sigma:  走廊先验注入噪声标准差（阶段二用于对齐推理时分布）
            lambda_corridor / lambda_physics: 论文 λ_c / λ_p
            use_projector: 是否把可微投影器放进计算图（论文 §3.7 可选）
            lambda_projector: 投影后监督项权重
        Returns:
            dict 含 flow_loss / corridor_loss / physics_loss / projector_loss / total_loss
        """
        device = ctx_t.device
        B = ctx_t.shape[0]

        # 1. 编码
        context = self.encode(ctx_t, ctx_dt, ctx_pos, ctx_vel, mask)

        # 2. 走廊先验（真实未来时刻上的闭式外推）
        dtau = tgt_t - ctx_t[:, -1:]
        x0, _, _ = self.corridor_prior(context, ctx_pos[:, -1, :], dtau)

        # 阶段二: 向先验注入噪声，使生成器见到与推理时相同的输入分布
        x0_flow = x0
        if sigma > 0:
            x0_flow = x0 + torch.randn_like(x0) * sigma

        # 3. 流匹配损失
        flow_loss, _ = self.flow.flow_matching_loss(x0_flow, tgt_pos, context, dtau)

        # 4. 走廊拟合损失（独立回归目标，避免生成目标主导先验拟合）
        corridor_loss = torch.mean((x0 - tgt_pos) ** 2)

        # 5. 软物理损失: 中点插值 X_0.5 作为可微代理，避免稀疏梯度
        x_mid = 0.5 * (x0_flow + tgt_pos)
        physics_loss = self.physics_loss(
            x_mid, ctx_pos[:, -1, :], dtau, p_mean, p_std,
            anchor_vel_norm=ctx_vel[:, -1, :],
        )

        total = flow_loss + lambda_corridor * corridor_loss + lambda_physics * physics_loss

        # 端到端采样损失: 直接优化"少量 Euler 步后的生成轨迹"，缓解纯速度回归
        # 导致的均值化（预测过于平直）问题，让训练目标与推理过程一致。
        sample_loss = torch.zeros((), device=device)
        if lambda_sample > 0:
            x_hat = self.flow.sample(x0_flow, context, dtau,
                                     n_steps=max(int(n_sample_steps), 1), sigma=0.0)
            sample_loss = torch.mean((x_hat - tgt_pos) ** 2)
            total = total + lambda_sample * sample_loss

        # 6. 可选: 投影器在环（缩小训练/部署差距）
        projector_loss = torch.zeros((), device=device)
        if use_projector:
            x_hat = self.flow.sample(
                x0_flow, context, dtau, n_steps=n_project_steps, sigma=0.0,
            )
            if p_mean.dim() == 1:
                p_mean = p_mean.unsqueeze(0)
            if p_std.dim() == 1:
                p_std = p_std.unsqueeze(0)
            x_proj = self.projector(x_hat, ctx_pos[:, -1, :], dtau, p_mean, p_std)
            x_proj_norm = (x_proj - p_mean.unsqueeze(1)) / p_std.clamp_min(1e-6).unsqueeze(1)
            projector_loss = torch.mean((x_proj_norm - tgt_pos) ** 2)
            total = total + lambda_projector * projector_loss

        return {
            "flow_loss": flow_loss,
            "corridor_loss": corridor_loss,
            "physics_loss": physics_loss,
            "projector_loss": projector_loss,
            "sample_loss": sample_loss,
            "total_loss": total,
            "x0": x0,
            "dtau": dtau,
        }

    # ── 推理 ──────────────────────────────────────────

    def forecast(
        self,
        ctx_t: torch.Tensor,
        ctx_dt: torch.Tensor,
        ctx_pos: torch.Tensor,
        ctx_vel: torch.Tensor,
        dtau: torch.Tensor,
        p_mean: torch.Tensor,
        p_std: torch.Tensor,
        n_steps: int = None,
        sigma: float = None,
        n_samples: int = 1,
        project: bool = None,
    ) -> dict:
        """
        批量预测（已归一化的历史）。返回归一化与物理坐标两套轨迹。

        Returns:
            dict(x_norm, x_phys, corridor_phys)，形状 (S, B, K, 3) 或 (B, K, 3)
        """
        B = ctx_t.shape[0]
        if dtau.dim() == 1:
            dtau = dtau.unsqueeze(0)
        if dtau.shape[0] != B:
            dtau = dtau.expand(B, -1)

        context = self.encode(ctx_t, ctx_dt, ctx_pos, ctx_vel)
        x0, v0, acc = self.corridor_prior(context, ctx_pos[:, -1, :], dtau)

        n_steps = self.n_euler_steps if n_steps is None else int(n_steps)
        sigma = self.sigma if sigma is None else float(sigma)
        project = self.project if project is None else bool(project)

        # 全局条件使用固定 k_max 步的参考走廊，使预测行为与请求的预测点数解耦
        # （推导可见 CorridorFlow.encode_global 的说明）。
        x0_ref = None
        if dtau.shape[1] != self.k_max:
            tau_ref = (torch.arange(1, self.k_max + 1, device=dtau.device,
                                    dtype=dtau.dtype) * self.sample_dt)
            x0_ref = self.corridor.extrapolate(ctx_pos[:, -1, :], v0, acc,
                                               tau_ref.unsqueeze(0).expand(B, -1))

        x_norm = self.flow.sample(x0, context, dtau, n_steps=n_steps,
                                  sigma=sigma, n_samples=n_samples, x0_ref=x0_ref)

        if p_mean.dim() == 1:
            p_mean = p_mean.unsqueeze(0)
        if p_std.dim() == 1:
            p_std = p_std.unsqueeze(0)

        B, K, _ = x0.shape
        if x_norm.dim() == 4:      # (S, B, K, 3)
            S = x_norm.shape[0]
            flat = x_norm.reshape(S * B, K, 3)
            anchor = ctx_pos[:, -1, :].unsqueeze(0).expand(S, B, 3).reshape(S * B, 3)
            mean_flat = p_mean.unsqueeze(0).expand(S, B, 3).reshape(S * B, 3)
            std_flat = p_std.unsqueeze(0).expand(S, B, 3).reshape(S * B, 3)
            if project:
                x_phys = self.projector(flat, anchor, dtau.repeat(S, 1), mean_flat, std_flat)
            else:
                x_phys = flat * std_flat.unsqueeze(1) + mean_flat.unsqueeze(1)
            x_phys = x_phys.reshape(S, B, K, 3)
        else:
            if project:
                x_phys = self.projector(x_norm, ctx_pos[:, -1, :], dtau, p_mean, p_std)
            else:
                x_phys = x_norm * p_std.unsqueeze(1) + p_mean.unsqueeze(1)

        corridor_phys = x0 * p_std.unsqueeze(1) + p_mean.unsqueeze(1)
        return {"x_norm": x_norm, "x_phys": x_phys, "corridor_phys": corridor_phys}

    # ── 长视界运动模型 ──────────────────────────────

    def _estimate_motion_state(self, anchor: np.ndarray, model_pts: np.ndarray,
                               dt: float) -> dict:
        """
        从"锚点 + 模型段"中估计常转率运动状态。

        速度用最后约 1 s 的最小二乘拟合（抑制逐点噪声），转率由航向角斜率估计，
        水平加速度与垂直加速度用短窗线性/二次拟合，均做物理限幅。
        """
        pts = np.vstack([anchor[None, :], model_pts])
        n = len(pts)
        k = int(min(max(round(1.0 / dt), 3), n))
        ts = np.arange(n) * dt

        A = np.vstack([ts[-k:], np.ones(k)]).T
        vel = np.linalg.lstsq(A, pts[-k:], rcond=None)[0][0]

        seg = pts[-k:]
        d = np.diff(seg[:, [0, 2]], axis=0)
        speed = np.linalg.norm(d, axis=1) / dt
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0])) if len(d) else np.zeros(1)
        th = ts[-k + 1:]

        omega = float(np.polyfit(th, heading, 1)[0]) if len(heading) >= 2 else 0.0
        omega = float(np.clip(omega, -1.2, 1.2))
        vh = float(speed[-1]) if len(speed) else float(np.linalg.norm(vel[[0, 2]]))
        vy = float(np.clip(vel[1], -self.bounds.v_down_max, self.bounds.v_up_max))

        # 长视界不叠加水平/垂直加速度：几十秒后加速度会积分到速度上限，
        # 产生"越推越快、一路爬升到高度上限"的非物理轨迹。
        # 保持末速度 + 衰减转率，既延续当前机动，又随时间的推移自然趋于平稳。
        return {
            'pos': pts[-1].astype(np.float64),
            'heading': float(heading[-1]) if len(heading) else 0.0,
            'omega': omega, 'vh': max(vh, 0.2), 'vy': vy,
        }

    def _integrate_motion(self, state: dict, offsets: np.ndarray,
                          sub_dt: float = 0.05, turn_decay_s: float = 6.0) -> np.ndarray:
        """
        常转率 + 加速度积分，转率按指数衰减（长时间后趋于直线但保留初始曲率）。
        """
        pos = state['pos'].copy()
        heading = state['heading']
        t = 0.0
        out = np.empty((len(offsets), 3), dtype=np.float64)
        for i, target in enumerate(offsets):
            span = max(float(target) - t, 1e-6)
            steps = max(int(np.ceil(span / sub_dt)), 1)
            h = span / steps
            for _ in range(steps):
                omega = state['omega'] * np.exp(-t / turn_decay_s)
                heading += omega * h
                vh = float(np.clip(state['vh'], 0.2, self.bounds.v_h_max))
                pos[0] += vh * np.cos(heading) * h
                pos[2] += vh * np.sin(heading) * h
                pos[1] += state['vy'] * h
                t += h
            out[i] = pos
        return out

    @torch.no_grad()
    def predict_with_confidence(
        self,
        timestamps: list[float],
        positions: list[list[float]],
        num_points: int = 20,
        time_step: float = 0.3,
        device: str = "cpu",
        sigma: float = None,
        n_samples: int = 1,
        n_euler_steps: int = None,
    ) -> tuple[list[list[float]], list[float], list[float]]:
        """
        单条轨迹的预测（含长视界分段策略），返回坐标、时间戳与逐点置信度。

        内部流程:
          1. 把输入历史末尾重采样到训练栅格（C 点 / (C-1)·sample_dt）；
          2. 视界 ≤ 模型可靠视界（k_max × sample_dt，默认 2 s）时，直接在请求的
             时间戳上并行生成（流匹配以 τ 为条件，走廊先验闭式求值）；
          3. 视界更长时：模型段预测 2 s，其后用"常转率 + 加速度"运动模型外推
             （转率指数衰减，保留初始曲率而非退化成直线），逐点置信度随时间衰减；
          4. 全段统一经物理投影器约束速度/垂直速率/高度。

        Args:
            timestamps: 历史时间戳 [t_1, ..., t_N]（任意采样间隔）
            positions:  历史位置 [[x, y, z], ...]
            num_points: 请求的预测点数
            time_step:  请求的未来相邻点时间间隔（秒）
            device:     推理设备
            sigma:      起点噪声（0=确定性均值预测；>0 给出单峰概率样本）
            n_samples:  概率样本条数（sigma>0 时有效，返回第一条）

        Returns:
            pred_positions: 共 num_points+1 个点（首个为最后观测点锚点）
            pred_times:     对应时间戳
        """
        self.eval()
        device_obj = torch.device(device)

        t_raw = np.asarray(timestamps, dtype=np.float32)
        p_raw = np.asarray(positions, dtype=np.float32)
        if len(t_raw) != len(p_raw):
            n = min(len(t_raw), len(p_raw))
            t_raw, p_raw = t_raw[:n], p_raw[:n]

        # 统一到训练时的时间栅格: 末尾对齐、跨度 (C-1)·dt、C 点等间隔。
        # 训练窗口是 20 点 @ 0.1 s（1.9 s），而运行时轨迹可能是 0.1/0.2 s 采样、
        # 长度 10~50 点。若不统一，逐轴 Z-score 的标准差会随窗口跨度变化，
        # 归一化后的速度尺度随之漂移（实测可达 5~10 倍），预测会被放大。
        t_arr, p_arr = self._canonical_history(t_raw, p_raw)
        N = len(p_arr)

        # 历史特征: Δt 与中心差分速度
        dt_arr = np.zeros(N, dtype=np.float32)
        if N > 1:
            dt_arr[1:] = t_arr[1:] - t_arr[:-1]
        v_arr = estimate_velocity(p_arr, t_arr)

        # 逐轴 Z-score（仅用历史统计量，与训练一致）
        p_mean = p_arr.mean(axis=0)
        p_std = np.maximum(p_arr.std(axis=0), POSITION_STD_FLOOR)
        p_norm = (p_arr - p_mean) / p_std
        v_norm = v_arr / p_std

        t_t = torch.from_numpy(t_arr).unsqueeze(0).to(device_obj)
        dt_t = torch.from_numpy(dt_arr).unsqueeze(0).to(device_obj)
        p_t = torch.from_numpy(p_norm).unsqueeze(0).to(device_obj)
        v_t = torch.from_numpy(v_norm).unsqueeze(0).to(device_obj)
        mean_t = torch.from_numpy(p_mean).unsqueeze(0).to(device_obj)
        std_t = torch.from_numpy(p_std).unsqueeze(0).to(device_obj)

        req_offset = np.arange(1, num_points + 1, dtype=np.float64) * float(time_step)
        model_horizon = self.k_max * self.sample_dt          # 模型可靠视界（默认 2 s）
        anchor_phys = p_raw[-1].astype(np.float64)

        if req_offset[-1] <= model_horizon + 1e-9:
            # ── 短/中视界：直接在全请求时间戳上并行生成 ──
            internal = req_offset
            if len(internal) > self.max_native_steps:
                internal = np.linspace(internal[0], internal[-1], self.max_native_steps)
            tau_req = torch.from_numpy(internal.astype(np.float32)).unsqueeze(0).to(device_obj)
            out = self.forecast(t_t, dt_t, p_t, v_t, tau_req, mean_t, std_t,
                                n_steps=n_euler_steps, sigma=sigma, n_samples=n_samples,
                                project=False)
            x_norm_req = out["x_norm"]
            if x_norm_req.dim() == 4:
                x_norm_req = x_norm_req[0]
            x_norm_req = x_norm_req[0].cpu().numpy()
            if len(internal) != len(req_offset):
                grid_t = np.concatenate([[0.0], internal])
                traj = np.concatenate([p_norm[-1:], x_norm_req], axis=0)
                x_norm_req = np.stack(
                    [np.interp(req_offset, grid_t, traj[:, axis]) for axis in range(3)], axis=1)
            pos_req = x_norm_req * p_std + p_mean
            conf = np.ones(len(req_offset), dtype=np.float64)

        else:
            # ── 长视界：模型段（≤2 s）+ 运动模型外推段 ──
            # 模型段在原生栅格上生成，保证与训练分布一致；
            # 超出部分用模型段末端的航向/转率/速度/垂直加速度做常转率积分，
            # 既不会退化成直线，也不会像纯回归那样越推越平。
            k_model = self.k_max
            model_offsets = np.arange(1, k_model + 1, dtype=np.float64) * self.sample_dt
            tau_model = torch.from_numpy(model_offsets.astype(np.float32)).unsqueeze(0).to(device_obj)
            out = self.forecast(t_t, dt_t, p_t, v_t, tau_model, mean_t, std_t,
                                n_steps=n_euler_steps, sigma=sigma, n_samples=n_samples,
                                project=False)
            x_norm_model = out["x_norm"]
            if x_norm_model.dim() == 4:
                x_norm_model = x_norm_model[0]
            x_norm_model = x_norm_model[0].cpu().numpy()
            model_phys = x_norm_model * p_std + p_mean

            pos_req = np.empty((len(req_offset), 3), dtype=np.float64)
            in_model = req_offset <= model_horizon + 1e-9
            if in_model.any():
                grid_t = np.concatenate([[0.0], model_offsets])
                traj = np.concatenate([anchor_phys[None, :], model_phys], axis=0)
                for axis in range(3):
                    pos_req[in_model, axis] = np.interp(req_offset[in_model], grid_t, traj[:, axis])

            beyond = ~in_model
            if beyond.any():
                state = self._estimate_motion_state(anchor_phys, model_phys, self.sample_dt)
                ext = self._integrate_motion(state, req_offset[beyond] - model_horizon)
                pos_req[beyond] = ext

            conf = np.where(
                req_offset <= model_horizon + 1e-9, 1.0,
                np.clip(np.exp(-(req_offset - model_horizon) / 8.0), 0.15, 1.0))

        # 统一投影到请求时间栅格，保证速度/垂直速率/高度的硬约束
        pos_norm_req = (pos_req - p_mean) / p_std
        req_norm_t = torch.from_numpy(pos_norm_req.astype(np.float32)).unsqueeze(0).to(device_obj)
        dtau_req = torch.from_numpy(req_offset.astype(np.float32)).unsqueeze(0).to(device_obj)
        if self.project:
            x_phys = self.projector(req_norm_t, p_t[:, -1, :], dtau_req, mean_t, std_t)
        else:
            x_phys = req_norm_t * std_t.unsqueeze(1) + mean_t.unsqueeze(1)
        preds = x_phys[0].cpu().numpy()
        self._last_confidence = conf

        t_last = float(t_arr[-1])
        pred_times = [round(t_last + o, 6) for o in req_offset.tolist()]
        pred_times = [round(t_last, 6)] + pred_times
        pred_positions = [list(map(float, p_raw[-1]))] + preds.tolist()

        # NaN 保护: 用最近一个有效点填充
        pred_positions = np.asarray(pred_positions, dtype=np.float64)
        valid = ~np.isnan(pred_positions).any(axis=1)
        if not valid.all():
            if valid.any():
                last_valid = pred_positions[valid][-1]
                pred_positions[~valid] = last_valid
            else:
                pred_positions = np.tile(p_arr[-1:].astype(np.float64), (num_points + 1, 1))

        conf_list = [1.0] + [float(c) for c in np.asarray(conf, dtype=float).tolist()]
        return [list(map(float, row)) for row in pred_positions], pred_times, conf_list

    @torch.no_grad()
    def predict(
        self,
        timestamps: list[float],
        positions: list[list[float]],
        num_points: int = 20,
        time_step: float = 0.3,
        device: str = "cpu",
        sigma: float = None,
        n_samples: int = 1,
        n_euler_steps: int = None,
    ) -> tuple[list[list[float]], list[float]]:
        """预测入口（保持旧签名）：返回坐标与时间戳。"""
        points, times, _ = self.predict_with_confidence(
            timestamps, positions, num_points=num_points, time_step=time_step,
            device=device, sigma=sigma, n_samples=n_samples, n_euler_steps=n_euler_steps,
        )
        return points, times

    # ── 推理输入规整 ──────────────────────────────────

    def _canonical_history(
        self,
        t_raw: np.ndarray,
        p_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把任意采样历史的末尾片段重采样到训练窗口规格（C 点 / 跨度 (C-1)·dt）。"""
        n = len(t_raw)
        if n < 2:
            return t_raw, p_raw

        order = np.argsort(t_raw, kind='stable')
        t_sorted = t_raw[order].astype(np.float64)
        p_sorted = p_raw[order].astype(np.float64)

        # 去掉重复时间戳（保留最后一个）
        keep = np.concatenate([[True], np.diff(t_sorted) > 0])
        t_sorted, p_sorted = t_sorted[keep], p_sorted[keep]
        if len(t_sorted) < 2:
            return t_raw, p_raw

        c = max(self.ctx_len, 2)
        span = (c - 1) * self.sample_dt
        t_end = float(t_sorted[-1])
        t_start = max(float(t_sorted[0]), t_end - span)
        if t_end - t_start <= 1e-6:
            t_start = float(t_sorted[0])

        grid = np.linspace(t_start, t_end, c)
        pos = np.stack(
            [np.interp(grid, t_sorted, p_sorted[:, axis]) for axis in range(3)], axis=1,
        ).astype(np.float32)
        return grid.astype(np.float32), pos

    @torch.no_grad()
    def count_parameters(self) -> dict:
        """按模块统计参数量（对应论文 Table 1）。"""
        parts = {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "corridor": sum(p.numel() for p in self.corridor.parameters()),
            "flow": sum(p.numel() for p in self.flow.parameters()),
            "projector": sum(p.numel() for p in self.projector.parameters()),
        }
        parts["total"] = sum(p.numel() for p in self.parameters())
        parts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return parts

    def get_model_info(self) -> dict:
        """返回模型元信息（含完整架构参数，用于推理时重建模型）。"""
        info = {"model_type": "KinFlow"}
        info.update(self._model_config)
        info.update(self.count_parameters())
        return info


__all__ = ["KinFlow", "POSITION_STD_FLOOR"]
