"""
Phy-ODE-Diffusion 分阶段训练脚本（含进度条、图表输出和详细日志）

输出目录: train/hybrid_predictor/train_result/（图表/日志/模型均在此目录下）
  - 01_loss_breakdown.png       损失分解曲线
  - 02_convergence_analysis.png 收敛性分析
  - 03_stage_comparison.png     阶段对比（含雷达图）
  - 04_ade_fde.png              ADE/FDE 预测精度指标
  - 05_physics_metrics.png      物理约束违反率
  - 06_dashboard.png            综合仪表盘
  - training_history.json       详细训练日志
  - training_summary.json       训练摘要

阶段一: 训练 Transformer 编码器 + ODE + GRU 更新（无扩散）
阶段二: 固定上述模块，训练扩散模型
阶段三（可选）: 联合微调（计划采样）

用法:
  python train/hybrid_predictor/train.py --stage all --epochs 30 --batch 32 --device cpu
  python train/hybrid_predictor/train.py --stage 1 --epochs 50
"""

import os
import sys
import json
import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 路径（pathlib 绝对路径，无 sys.path 注入）
_MODULE_DIR = Path(__file__).resolve().parent          # .../train/hybrid_predictor/
_PROJECT_ROOT = _MODULE_DIR.parent.parent               # .../drone/
sys.path.insert(0, str(_PROJECT_ROOT))

from train.hybrid_predictor.dataset import (
    load_all_trajectories, TrajectoryDataset, collate_fn,
)
from trajectory_reconstruction.core.prediction.hybrid import PhyODEDiffusion

_RESULTS_DIR = _MODULE_DIR / "train_result"

# 尝试导入 tqdm
try:
    from tqdm import tqdm as _tqdm_import
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 尝试导入 matplotlib
try:
    import matplotlib as _mpl
    _mpl.use('Agg')
    import matplotlib.pyplot as _plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── 配置 ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    "ctx_len": 20,
    "tgt_len": 10,
    "batch_size": 32,
    "num_workers": 4,
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-3,
    "lr_stage3": 1e-4,
    "weight_decay": 1e-4,
    "epochs_stage1": 50,
    "epochs_stage2": 100,
    "epochs_stage3": 20,
    # Warmup 策略（各阶段独立配置，0=跳过 warmup）
    "warmup_epochs_s1": 5,       # 阶段一: Transformer+ODE 需稳定初始化
    "warmup_epochs_s2": 3,       # 阶段二: 扩散模型对初始 LR 敏感
    "warmup_epochs_s3": 2,       # 阶段三: 微调已有基础，短 warmup
    "warmup_start_factor": 0.1,  # warmup 起始 LR = base_lr × factor
    # 标签平滑（回归任务：对目标位置加高斯噪声作为正则化）
    "label_smoothing": 0.005,     # 噪声标准差（相对于数据标准差的比例）
    "checkpoint_interval": 10,    # 每隔 N 个 epoch 在阶段中保存一次检查点（0=仅阶段结束时保存）
    "dataset_dir": "train/hybrid_predictor/dataset",
    "output_dir": "train/hybrid_predictor/train_result/models",
    "device": "cuda:0",
    "val_split": 0.1,
    # 模型架构参数（可通过 config.json → training 覆盖）
    "d_feat": 64,
    "d_context": 128,
    "n_head": 4,
    "n_layers": 6,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "d_z": 64,
    "a_max": 30.0,
    "ode_hidden_dim": 128,
    "n_diffusion_steps": 500,
    "n_inference_steps": 50,
    "val_inference_steps": 10,     # 验证时DDIM步数（配合梯度裁剪，10步可稳定运行）
    "tau_emb_dim": 16,
    "dt_emb_dim": 16,
    "diff_hidden_dim": 128,
    "guidance_eta": 0.1,
    "v_max": 20.0,                  # 最大水平速度 (m/s)，对齐 drone_dynamics
    "a_max": 10.0,                  # 最大加速度 (m/s²)，基于 g·tan(35°)≈6.9 适当放宽
    "z_min": 1.0,
    "z_max": 120.0,
    "v_v_up": 5.0,
    "v_v_down": 3.0,
    "max_tilt": 35.0,
    "g": 9.81,
    "obs_hidden_dim": 32,
    # 物理损失权重（阶段二/三中 physics_loss 的相对权重）
    "physics_weight": 0.01,
}


# ── 工具 ──────────────────────────────────────────────

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


class TrainingLogger:
    """训练日志记录器 — 收集每个 epoch 的指标用于图表生成"""

    def __init__(self):
        self.history: list[dict] = []
        self.stage_labels: dict = {}

    def log_epoch(self, stage: int, epoch: int, train_loss: float,
                  val_loss: float, lr: float, epoch_time: float,
                  extra: dict = None):
        entry = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": lr,
            "epoch_time_s": round(epoch_time, 1),
        }
        if extra:
            entry.update({k: round(v, 6) if isinstance(v, float) else v
                          for k, v in extra.items()})
        self.history.append(entry)
        if stage not in self.stage_labels:
            self.stage_labels[stage] = f"Stage {stage}"

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "training_history.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "history": self.history,
                "stage_labels": self.stage_labels,
            }, f, indent=2, ensure_ascii=False)
        return path

    @property
    def best_val_loss(self) -> float:
        losses = [e["val_loss"] for e in self.history if e["stage"] > 0]
        return min(losses) if losses else float('inf')


# ── 图表生成（单图高分辨率，适合论文发表）────────────────
# 全局设置
_CHART_DPI = 300
_CHART_SIZE = (10, 6)
_CHART_SIZE_WIDE = (12, 6)

# 全局色板
_COLORS = {
    'stage1': '#2196F3', 'stage2': '#FF9800', 'stage3': '#4CAF50',
    'train': '#2196F3', 'val': '#EF5350', 'best': '#4CAF50',
    'speed': '#FF7043', 'accel': '#AB47BC', 'height': '#26C6DA',
}

def _savefig(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=_CHART_DPI, bbox_inches='tight')
    _plt.close(fig)
    print(f"[图表] {name}")
    return path


# ═══════════════════ 01 损失曲线（每阶段一张）═══════════════

def _chart_loss_stage(logger, output_dir, stage, color):
    data = [e for e in logger.history if e["stage"] == stage]
    if not data: return
    epochs = [e["epoch"] for e in data]
    train_vals = [e["train_loss"] for e in data]
    val_vals = [e["val_loss"] for e in data]
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    ax.plot(epochs, train_vals, '-', color=color, lw=2, alpha=0.85, label='Train')
    ax.plot(epochs, val_vals, '--', color=_COLORS['val'], lw=2, alpha=0.85, label='Val')
    # y 轴上限裁剪：取验证loss最大值的3倍，避免训练loss过大压扁验证曲线
    ax.set_ylim(0, 0.8)
    best_i = min(range(len(data)), key=lambda i: data[i]["val_loss"])
    ax.axvline(x=epochs[best_i], color=_COLORS['best'], linestyle=':', alpha=0.5, lw=1)
    ax.annotate(f'Best: {data[best_i]["val_loss"]:.4f}', xy=(epochs[best_i], data[best_i]["val_loss"]),
                xytext=(10, 15), textcoords='offset points', fontsize=11,
                arrowprops=dict(arrowstyle='->', lw=1, color=_COLORS['best']), color=_COLORS['best'], fontweight='bold')
    ax.set_title(f'Stage {stage} — Training Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, f'01{chr(96+stage)}_loss_s{stage}.png')


# ═══════════════════ 02 收敛性分析（单图）═══════════════════

def _chart_global_loss(logger, output_dir, suffix: str = ""):
    all_epochs = list(range(1, len(logger.history) + 1))
    train_losses = [e["train_loss"] for e in logger.history]
    val_losses = [e["val_loss"] for e in logger.history]
    boundaries = [i for i, e in enumerate(logger.history) if i == 0 or e["stage"] != logger.history[i-1]["stage"]]
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    ax.plot(all_epochs, train_losses, '-', color=_COLORS['train'], lw=1.5, alpha=0.8, label='Train')
    ax.plot(all_epochs, val_losses, '-', color=_COLORS['val'], lw=1.5, alpha=0.8, label='Val')
    for b in boundaries[1:]: ax.axvline(x=b, color='gray', linestyle='--', alpha=0.4, lw=0.8)
    # y 轴上限裁剪
    if val_losses:
        ax.set_ylim(0, 0.8)
    ax.set_title('Global Loss Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Global Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, f'02a_global_loss{suffix}.png')


def _chart_overfitting(logger, output_dir, suffix: str = ""):
    all_epochs = list(range(1, len(logger.history) + 1))
    ratios = [e["val_loss"] / max(e["train_loss"], 1e-8) for e in logger.history]
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    ax.plot(all_epochs, ratios, '-', color='#7E57C2', lw=2)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between(all_epochs, 0, ratios, alpha=0.15, color='#7E57C2')
    ax.set_title('Val/Train Ratio (Overfitting Detection)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Global Epoch', fontsize=12); ax.set_ylabel('Ratio', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, f'02b_overfitting{suffix}.png')


def _chart_improvement(logger, output_dir, suffix: str = ""):
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    for stage in sorted(set(e["stage"] for e in logger.history)):
        losses = [e["val_loss"] for e in logger.history if e["stage"] == stage]
        if len(losses) < 2: continue
        imp = [(losses[0] - l) / losses[0] * 100 for l in losses]
        ax.plot(range(1, len(imp)+1), imp, '-o', color=_COLORS.get(f'stage{stage}','#333'), markersize=4, lw=2, label=f'Stage {stage}')
    ax.set_title('Improvement Over Initial Loss (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch (within stage)', fontsize=12); ax.set_ylabel('Improvement %', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, f'02c_improvement{suffix}.png')


def _chart_lr_schedule(logger, output_dir, suffix: str = ""):
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    for stage in sorted(set(e["stage"] for e in logger.history)):
        lrs = [e["lr"] for e in logger.history if e["stage"] == stage]
        ax.semilogy(range(1, len(lrs)+1), lrs, '-o', color=_COLORS.get(f'stage{stage}','#333'), markersize=4, lw=2, label=f'Stage {stage}')
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch (within stage)', fontsize=12); ax.set_ylabel('LR', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, f'02d_lr_schedule{suffix}.png')


def _chart_epoch_time(logger, output_dir, suffix: str = ""):
    all_epochs = list(range(1, len(logger.history) + 1))
    times = [e["epoch_time_s"] for e in logger.history]
    colors_t = [_COLORS.get(f'stage{e["stage"]}','#333') for e in logger.history]
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    ax.bar(all_epochs, times, color=colors_t, alpha=0.7, width=0.8)
    avg_time = sum(times) / len(times) if times else 0
    ax.axhline(y=avg_time, color='red', linestyle='--', alpha=0.5, lw=1, label=f'Avg: {avg_time:.1f}s')
    ax.set_title('Epoch Training Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Global Epoch', fontsize=12); ax.set_ylabel('Time (s)', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout(); _savefig(fig, output_dir, f'02e_epoch_time{suffix}.png')


def _chart_cumulative_time(logger, output_dir, suffix: str = ""):
    all_epochs = list(range(1, len(logger.history) + 1))
    times = [e["epoch_time_s"] for e in logger.history]
    cum_times = [sum(times[:i+1]) / 60 for i in range(len(times))]
    boundaries = [i for i, e in enumerate(logger.history) if i == 0 or e["stage"] != logger.history[i-1]["stage"]]
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    ax.fill_between(all_epochs, 0, cum_times, alpha=0.3, color='#26C6DA')
    ax.plot(all_epochs, cum_times, '-', color='#00838F', lw=2)
    for b in boundaries[1:]: ax.axvline(x=b, color='gray', linestyle='--', alpha=0.4, lw=0.8)
    ax.set_title('Cumulative Training Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Global Epoch', fontsize=12); ax.set_ylabel('Time (min)', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, f'02f_cumulative_time{suffix}.png')


# ═══════════════════ 03 阶段对比（单图）═══════════════════

def _chart_final_vs_best(logger, output_dir):
    stages = sorted(set(e["stage"] for e in logger.history))
    final_losses, best_losses = [], []
    for s in stages:
        s_data = [e for e in logger.history if e["stage"] == s]
        final_losses.append(s_data[-1]["val_loss"] if s_data else 0)
        best_losses.append(min(e["val_loss"] for e in s_data) if s_data else 0)
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    x = range(len(stages)); w = 0.35
    ax.bar([i-w/2 for i in x], final_losses, w, label='Final Loss', color='#FF9800', alpha=0.85)
    bars = ax.bar([i+w/2 for i in x], best_losses, w, label='Best Loss', color='#4CAF50', alpha=0.85)
    for bar in bars: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{bar.get_height():.4f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(list(x)); ax.set_xticklabels([f'S{s}' for s in stages], fontsize=12)
    ax.set_title('Final vs Best Val Loss', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12); ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout(); _savefig(fig, output_dir, '03a_final_vs_best.png')


def _chart_stage_radar(logger, output_dir):
    stages = sorted(set(e["stage"] for e in logger.history))
    if len(stages) < 1: return
    metrics = ['1-Final Loss', '1-Best Loss', 'Speed (1/epoch)', 'Stability']
    all_final = []; all_best = []; all_times = {}
    for s in stages:
        s_data = [e for e in logger.history if e["stage"] == s]
        all_final.append(s_data[-1]["val_loss"]); all_best.append(min(e["val_loss"] for e in s_data))
        all_times[s] = sum(e["epoch_time_s"] for e in s_data) / len(s_data)
    max_loss = max(all_final + all_best) if all_final else 1
    max_time = max(all_times.values()) if all_times else 1
    fig = _plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    angles = [n * 2 * np.pi / len(metrics) for n in range(len(metrics))]; angles += angles[:1]
    for s in stages:
        s_data = [e for e in logger.history if e["stage"] == s]
        final = s_data[-1]["val_loss"]; best = min(e["val_loss"] for e in s_data)
        ratios = [e["val_loss"]/max(e["train_loss"],1e-8) for e in s_data]
        stability = 1.0/(max(ratios)-min(ratios)+1) if len(ratios)>1 else 1
        vals = [1-final/max(max_loss,1e-8), 1-best/max(max_loss,1e-8), 1-all_times[s]/max(max_time,1e-8), stability]
        color = _COLORS.get(f'stage{s}','#333')
        ax.fill(angles, vals+vals[:1], alpha=0.15, color=color)
        ax.plot(angles, vals+vals[:1], 'o-', lw=2, color=color, label=f'Stage {s}')
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title('Stage Comparison Radar', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=10)
    fig.tight_layout(); _savefig(fig, output_dir, '03b_radar.png')


def _chart_stage_boxplot(logger, output_dir):
    stages = sorted(set(e["stage"] for e in logger.history))
    box_data = [[e["val_loss"] for e in logger.history if e["stage"]==s] for s in stages]
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    bp = ax.boxplot(box_data, patch_artist=True, showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    ax.set_xticklabels([f'S{s}' for s in stages], fontsize=12)
    for patch, s in zip(bp['boxes'], stages): patch.set_facecolor(_COLORS.get(f'stage{s}','#333')); patch.set_alpha(0.6)
    ax.set_title('Val Loss Distribution per Stage', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout(); _savefig(fig, output_dir, '03c_boxplot.png')


# ═══════════════════ 04 ADE/FDE（单图）═══════════════════

def _chart_ade(logger, output_dir):
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    for stage in sorted(set(e["stage"] for e in logger.history)):
        s_data = [e for e in logger.history if e["stage"]==stage and "ADE" in e]
        if not s_data: continue
        epochs = [e["epoch"] for e in s_data]; vals = [e["ADE"] for e in s_data]
        ax.plot(epochs, vals, 'o-', color=_COLORS.get(f'stage{stage}','#333'), markersize=4, lw=2, label=f'Stage {stage}')
        best_i = vals.index(min(vals))
        ax.annotate(f'{vals[best_i]:.3f}', xy=(epochs[best_i], vals[best_i]), fontsize=10, color='darkred')
    ax.set_title('ADE (Average Displacement Error)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('ADE', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, '04a_ade.png')


def _chart_fde(logger, output_dir):
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    for stage in sorted(set(e["stage"] for e in logger.history)):
        s_data = [e for e in logger.history if e["stage"]==stage and "FDE" in e]
        if not s_data: continue
        epochs = [e["epoch"] for e in s_data]; vals = [e["FDE"] for e in s_data]
        ax.plot(epochs, vals, 's--', color=_COLORS.get(f'stage{stage}','#333'), markersize=4, lw=2, label=f'Stage {stage}')
        best_i = vals.index(min(vals))
        ax.annotate(f'{vals[best_i]:.3f}', xy=(epochs[best_i], vals[best_i]), fontsize=10, color='darkred')
    ax.set_title('FDE (Final Displacement Error)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('FDE', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, '04b_fde.png')


def _chart_ade_vs_fde(logger, output_dir):
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    for stage in sorted(set(e["stage"] for e in logger.history)):
        s_data = [e for e in logger.history if e["stage"]==stage and "ADE" in e and "FDE" in e]
        if not s_data: continue
        ax.scatter([e["ADE"] for e in s_data], [e["FDE"] for e in s_data], c=[e["epoch"] for e in s_data], cmap='viridis', s=40, alpha=0.7, label=f'S{stage}')
    ax.set_title('ADE vs FDE', fontsize=14, fontweight='bold')
    ax.set_xlabel('ADE', fontsize=12); ax.set_ylabel('FDE', fontsize=12); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, '04c_ade_vs_fde.png')


def _chart_ade_improvement(logger, output_dir):
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    for stage in sorted(set(e["stage"] for e in logger.history)):
        s_data = [e for e in logger.history if e["stage"]==stage and "ADE" in e]
        if len(s_data) < 2: continue
        vals = [e["ADE"] for e in s_data]; imp = [(vals[0]-v)/max(vals[0],1e-8)*100 for v in vals]
        ax.plot([e["epoch"] for e in s_data], imp, '-', color=_COLORS.get(f'stage{stage}','#333'), lw=2, label=f'Stage {stage}')
    ax.set_title('ADE Improvement Over Initial (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Improvement %', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, '04d_ade_improvement.png')


# ═══════════════════ 05 物理指标（单图）═══════════════════

def _chart_physics_single(logger, output_dir, key, title, color, fname, suffix: str = ""):
    if key not in logger.history[0]: return
    epochs = list(range(1, len(logger.history)+1))
    vals = [e.get(key, 0) for e in logger.history]
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    ax.plot(epochs, vals, '-', color=color, lw=2, alpha=0.85)
    ax.fill_between(epochs, 0, vals, alpha=0.1, color=color)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Global Epoch', fontsize=12); ax.set_ylabel('Rate', fontsize=12)
    ax.grid(True, alpha=0.3)
    if vals: best_i = vals.index(min(vals)); ax.annotate(f'{vals[best_i]:.4f}', xy=(best_i+1, vals[best_i]), fontsize=11, color='darkred', fontweight='bold')
    fig.tight_layout(); _savefig(fig, output_dir, fname.replace('.png', '') + suffix + '.png')


def _chart_physics_score(logger, output_dir, suffix: str = ''):
    keys = ["speed_violation", "accel_violation", "height_violation"]
    if not all(k in logger.history[0] for k in keys): return
    epochs = list(range(1, len(logger.history)+1))
    scores = [max(0, 1.0 - sum(e.get(k,0) for k in keys)/3) for e in logger.history]
    fig, ax = _plt.subplots(figsize=_CHART_SIZE)
    ax.plot(epochs, scores, '-', color='#4CAF50', lw=2)
    ax.fill_between(epochs, 0, scores, alpha=0.15, color='#4CAF50')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Composite Physics Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Global Epoch', fontsize=12); ax.set_ylabel('Score', fontsize=12); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); _savefig(fig, output_dir, f'05d_physics_score{suffix}.png')


# ═══════════════════ 06 仪表盘（保留概览）═══════════════════

def _make_training_summary_dashboard(logger: TrainingLogger, output_dir: str,
                                     config: dict = None):
    """综合仪表盘 — 高 DPI 版本"""
    if config is None:
        config = {}
    fig = _plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    fig.suptitle('Phy-ODE-Diffusion Training Dashboard', fontsize=18, fontweight='bold', y=0.98)
    all_epochs = list(range(1, len(logger.history)+1))
    train_losses = [e["train_loss"] for e in logger.history]
    val_losses = [e["val_loss"] for e in logger.history]

    ax = fig.add_subplot(gs[0,:2])
    for stage in sorted(set(e["stage"] for e in logger.history)):
        idxs = [i for i,e in enumerate(logger.history) if e["stage"]==stage]
        ep = [all_epochs[i] for i in idxs]
        color = _COLORS.get(f'stage{stage}','#333')
        ax.plot(ep, [train_losses[i] for i in idxs], '-', color=color, lw=1.2, alpha=0.7)
        ax.plot(ep, [val_losses[i] for i in idxs], '--', color=color, lw=1.5, alpha=0.9, label=f'S{stage} Val')
    ax.set_title('Loss Overview', fontsize=14, fontweight='bold'); ax.set_xlabel('Global Epoch'); ax.set_ylabel('Loss')
    if val_losses:
        ax.set_ylim(0, 0.8)  # 固定y轴上限，确保验证曲线可见
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0,2]); ax.axis('off')
    stats = "Training Statistics\n" + "─"*25 + "\n"
    for s in sorted(set(e["stage"] for e in logger.history)):
        sd = [e for e in logger.history if e["stage"]==s]
        if sd: stats += f"Stage {s}:\n  Epochs: {len(sd)}\n  Best Val: {min(e['val_loss'] for e in sd):.4f}\n  Final: {sd[-1]['val_loss']:.4f}\n"
    stats += f"─"*25 + f"\nTotal: {sum(e['epoch_time_s'] for e in logger.history)/60:.1f} min\nBest Overall: {logger.best_val_loss:.4f}"
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    ax = fig.add_subplot(gs[0,3]); ax.axis('off')
    model_text = ("Model\n" + "─"*25 +
        f"\nTransformer: {config.get('n_layers',6)}L,{config.get('n_head',4)}H"
        f"\nd={config.get('d_feat',64)}, ctx={config.get('d_context',128)}"
        f"\nODE: dz={config.get('d_z',64)}, a={config.get('a_max',30)}"
        f"\nDiffusion: T={config.get('n_diffusion_steps',500)}"
        f"\nDDIM: {config.get('n_inference_steps',50)} steps"
        f"\nGRU: obs={config.get('obs_hidden_dim',32)}"
        f"\nData: ctx={config.get('ctx_len',20)},tgt={config.get('tgt_len',10)}")
    ax.text(0.05,0.95, model_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

    ax = fig.add_subplot(gs[1,:2])
    for stage in sorted(set(e["stage"] for e in logger.history)):
        lrs = [e["lr"] for e in logger.history if e["stage"]==stage]
        ax.semilogy(range(1,len(lrs)+1), lrs, '-o', color=_COLORS.get(f'stage{stage}','#333'), markersize=3, lw=1.2, label=f'S{stage}')
    ax.set_title('LR Schedule', fontsize=13, fontweight='bold'); ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1,2:])
    times_by_stage = {}
    for e in logger.history: times_by_stage.setdefault(e["stage"],[]).append(e["epoch_time_s"])
    totals = [sum(ts)/60 for ts in times_by_stage.values()]
    colors = [_COLORS.get(f'stage{s}','#333') for s in sorted(times_by_stage)]
    ax.pie(totals, labels=[f'S{s}' for s in sorted(times_by_stage)], autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize':10})
    ax.set_title(f'Time ({sum(totals):.1f} min)', fontsize=13, fontweight='bold')

    for idx, s in enumerate(sorted(set(e["stage"] for e in logger.history))):
        ax = fig.add_subplot(gs[2,idx] if idx<3 else gs[2,3])
        sv = [e["val_loss"] for e in logger.history if e["stage"]==s and e["val_loss"] == e["val_loss"]]
        if not sv: continue
        ax.hist(sv, bins=min(15,len(sv)), color=_COLORS.get(f'stage{s}','#333'), alpha=0.6, edgecolor='white')
        ax.axvline(x=min(sv), color='red', linestyle='--', lw=1, label=f'Best:{min(sv):.4f}')
        ax.axvline(x=sv[-1], color='orange', linestyle='--', lw=1, label=f'Final:{sv[-1]:.4f}')
        ax.set_title(f'S{s} Loss Hist', fontsize=12, fontweight='bold'); ax.set_xlabel('Val Loss'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout(); _savefig(fig, output_dir, '06_dashboard.png')


# ═══════════════════ 07 测试评价（单图）═══════════════════

def _make_test_chart(metrics: dict, output_dir: str):
    """测试集评价 — 4 张独立高分辨率图表"""
    import matplotlib.pyplot as plt

    # a) 精度指标
    fig, ax = plt.subplots(figsize=_CHART_SIZE)
    acc = {'Loss': metrics.get('test_loss',0), 'ADE': metrics.get('test_ADE',0), 'FDE': metrics.get('test_FDE',0)}
    bars = ax.barh(list(acc.keys()), list(acc.values()), color=['#2196F3','#FF9800','#EF5350'], alpha=0.85, height=0.5)
    for bar, val in zip(bars, acc.values()): ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=12, fontweight='bold')
    ax.set_title('Test Prediction Accuracy', fontsize=14, fontweight='bold'); ax.set_xlabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x'); ax.set_xlim(0, max(acc.values())*1.2 if max(acc.values())>0 else 1)
    fig.tight_layout(); _savefig(fig, output_dir, '07a_test_accuracy.png')

    # b) 物理违反率
    fig, ax = plt.subplots(figsize=_CHART_SIZE)
    phy = {'Speed': metrics.get('test_speed_violation',0), 'Accel': metrics.get('test_accel_violation',0), 'Height': metrics.get('test_height_violation',0)}
    bars = ax.barh(list(phy.keys()), list(phy.values()), color=['#FF7043','#AB47BC','#26C6DA'], alpha=0.85, height=0.5)
    for bar, val in zip(bars, phy.values()): ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=12, fontweight='bold')
    ax.set_title('Test Physics Violations', fontsize=14, fontweight='bold'); ax.set_xlabel('Rate', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x'); ax.set_xlim(0, max(phy.values())*1.3 if max(phy.values())>0 else 1)
    fig.tight_layout(); _savefig(fig, output_dir, '07b_test_physics.png')

    # c) 文字概要
    fig, ax = plt.subplots(figsize=_CHART_SIZE); ax.axis('off')
    phy_score = 1.0 - sum(phy.values())/3
    summary = "Test Evaluation Summary\n"+"─"*28+"\n"
    for k, v in {**acc, **phy}.items(): summary += f"{k:<18} {v:.4f}\n"
    summary += "─"*28 + f"\nPhysics Score:     {phy_score:.4f}"
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))
    fig.tight_layout(); _savefig(fig, output_dir, '07c_test_summary.png')

    # d) 雷达图
    fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, projection='polar')
    labels = ['1-ADE','1-FDE','1-Speed','1-Accel','1-Height','Physics']
    vals_r = [max(0, 1-acc['ADE']/max(acc['ADE'],0.001)), max(0,1-acc['FDE']/max(acc['FDE'],0.001)),
              max(0,1-phy['Speed']), max(0,1-phy['Accel']), max(0,1-phy['Height']), phy_score]
    angles = [n*2*np.pi/len(labels) for n in range(len(labels))]; angles += angles[:1]; vals_r += vals_r[:1]
    ax.fill(angles, vals_r, alpha=0.2, color='#4CAF50'); ax.plot(angles, vals_r, 'o-', lw=2, color='#4CAF50')
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('Test Radar', fontsize=14, fontweight='bold', pad=20); ax.set_ylim(0, 1.05)
    fig.tight_layout(); _savefig(fig, output_dir, '07d_test_radar.png')


# ═══════════════════ 图表入口 ═══════════════════

def _filtered_logger(logger: TrainingLogger, stage: int) -> TrainingLogger:
    """返回仅包含指定阶段数据的临时日志器（用于分阶段图表）"""
    tmp = TrainingLogger()
    tmp.history = [e for e in logger.history if e["stage"] == stage]
    tmp.stage_labels = {stage: logger.stage_labels.get(str(stage), f"Stage {stage}")}
    return tmp

def _plot_all_charts(logger: TrainingLogger, output_dir: str, config: dict = None,
                     stages_only: bool = False):
    """生成全部训练图表。

    Args:
        stages_only: True=仅生成各阶段独立图表（01/02/04/05 按阶段分开），
                     跳过跨阶段对比图（03/06）。False=生成全部图表。
    """
    if not HAS_MPL: print("[跳过] matplotlib 未安装"); return
    if not logger.history: print("[跳过] 训练历史为空"); return
    if config is None:
        config = {}
    os.makedirs(output_dir, exist_ok=True)

    stages = sorted(set(e["stage"] for e in logger.history))
    # stages_only 模式下跳过跨阶段对比图(03)和仪表盘(06)
    cross_stage = not stages_only

    # 01 损失曲线（每阶段一张）
    for s in stages:
        _chart_loss_stage(logger, output_dir, s, _COLORS.get(f'stage{s}','#333'))

    # 02 收敛性 — 每阶段独立出图
    for s in stages:
        _l = _filtered_logger(logger, s)
        _sfx = f'_s{s}'
        _chart_global_loss(_l, output_dir, _sfx)
        _chart_overfitting(_l, output_dir, _sfx)
        _chart_improvement(_l, output_dir, _sfx)
        _chart_lr_schedule(_l, output_dir, _sfx)
        _chart_epoch_time(_l, output_dir, _sfx)
        _chart_cumulative_time(_l, output_dir, _sfx)

    # 03 阶段对比（跨阶段，仅全局模式）
    if cross_stage and len(stages) >= 2:
        _chart_final_vs_best(logger, output_dir)
        _chart_stage_radar(logger, output_dir)
        _chart_stage_boxplot(logger, output_dir)
    elif not cross_stage:
        print("[图表03] 阶段对比图仅在全局模式下生成，跳过")

    # 04 ADE/FDE（每阶段独立出图）
    if any("ADE" in e for e in logger.history):
        _chart_ade(logger, output_dir)
        _chart_fde(logger, output_dir)
        _chart_ade_vs_fde(logger, output_dir)
        _chart_ade_improvement(logger, output_dir)
    else:
        print("[图表04] 无 ADE/FDE 数据，跳过")

    # 05 物理指标 — 每阶段独立出图
    if logger.history and "speed_violation" in logger.history[0]:
        for s in stages:
            _l = _filtered_logger(logger, s)
            _sfx = f'_s{s}'
            _chart_physics_single(_l, output_dir, "speed_violation",
                                  f"Speed Violation Rate (Stage {s})",
                                  _COLORS['speed'], '05a_speed.png', _sfx)
            _chart_physics_single(_l, output_dir, "accel_violation",
                                  f"Acceleration Violation Rate (Stage {s})",
                                  _COLORS['accel'], '05b_accel.png', _sfx)
            _chart_physics_single(_l, output_dir, "height_violation",
                                  f"Height Violation Rate (Stage {s})",
                                  _COLORS['height'], '05c_height.png', _sfx)
            _chart_physics_score(_l, output_dir, _sfx)
    else:
        print("[图表05] 无物理指标数据，跳过")

    # 06 仪表盘概览（仅全局模式）
    if cross_stage:
        _make_training_summary_dashboard(logger, output_dir, config)
    else:
        print("[图表06] 仪表盘仅在全局模式下生成，跳过")

def _save_training_summary(logger: TrainingLogger, config: dict, output_dir: str):
    """保存训练摘要 JSON"""
    os.makedirs(output_dir, exist_ok=True)

    # 从历史中提取各阶段最佳
    best_per_stage = {}
    for entry in logger.history:
        s = entry["stage"]
        if s not in best_per_stage or entry["val_loss"] < best_per_stage[s]["val_loss"]:
            best_per_stage[s] = {
                "epoch": entry["epoch"],
                "train_loss": entry["train_loss"],
                "val_loss": entry["val_loss"],
            }

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_epochs": len(logger.history),
        "best_val_loss": logger.best_val_loss,
        "best_per_stage": best_per_stage,
        "config": {k: str(v) for k, v in config.items()},
        "total_params": None,  # 由调用方填充
    }

    path = os.path.join(output_dir, "training_summary.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[日志] 训练摘要: {path}")
    return summary


# ── 数据加载 ──────────────────────────────────────────

def _make_loaders(train_trajs, valid_trajs, config):
    train_set = TrajectoryDataset(train_trajs, ctx_len=config["ctx_len"],
                                  tgt_len=config["tgt_len"], augment=True)
    val_set = TrajectoryDataset(
        valid_trajs if valid_trajs else train_trajs[:max(1, int(len(train_trajs) * 0.1))],
        ctx_len=config["ctx_len"], tgt_len=config["tgt_len"], augment=False,
    )
    return (
        DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                   collate_fn=collate_fn, drop_last=True,
                   num_workers=config.get("num_workers", 4),
                   pin_memory=True, prefetch_factor=2),
        DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                   collate_fn=collate_fn, drop_last=False,
                   num_workers=config.get("num_workers", 2),
                   pin_memory=True),
    )


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """构建训练/验证数据加载器。优先级: train/|valid/ > data/"""
    from numpy.random import default_rng

    dataset_root = os.path.join(_PROJECT_ROOT, config["dataset_dir"])

    # 优先 train/ + valid/ 目录
    train_dir = os.path.join(dataset_root, "train")
    valid_dir = os.path.join(dataset_root, "valid")

    train_trajs = load_all_trajectories(synthetic_dir=train_dir) \
        if os.path.isdir(train_dir) else []
    valid_trajs = load_all_trajectories(synthetic_dir=valid_dir) \
        if os.path.isdir(valid_dir) else []

    if len(train_trajs) > 0:
        print(f"[数据] train: {len(train_trajs)} 条轨迹, valid: {len(valid_trajs)} 条轨迹")
        return _make_loaders(train_trajs, valid_trajs, config)

    # 回退 data/ 目录（备份数据）
    data_dir = os.path.join(dataset_root, "data")
    if os.path.isdir(data_dir) and any(
        f.endswith(('.npz', '.dat')) for f in os.listdir(data_dir)
    ):
        all_trajs = load_all_trajectories(synthetic_dir=data_dir)
        if all_trajs:
            rng = default_rng(42)
            indices = list(range(len(all_trajs)))
            rng.shuffle(indices)
            n_val = max(1, int(len(all_trajs) * config.get("val_split", 0.15)))
            train_trajs = [all_trajs[i] for i in indices[n_val:]]
            valid_trajs = [all_trajs[i] for i in indices[:n_val]]
            print(f"[数据] data/ (备份) → train: {len(train_trajs)}, valid: {len(valid_trajs)}")
            return _make_loaders(train_trajs, valid_trajs, config)

    raise RuntimeError(
        f"没有可用的训练数据！请将 .npz 轨迹文件放入以下任一目录:\n"
        f"  • {train_dir}\n"
        f"  • {data_dir}\n"
        f"或使用 generate_synthetic.py 生成合成数据:\n"
        f"  python train/hybrid_predictor/generate_synthetic.py 200"
    )


def build_test_loader(config: dict) -> DataLoader | None:
    """构建测试集数据加载器。若 test/ 目录不存在或为空，返回 None。"""
    dataset_root = os.path.join(_PROJECT_ROOT, config["dataset_dir"])
    test_dir = os.path.join(dataset_root, "test")

    if not os.path.isdir(test_dir):
        print(f"[测试] test/ 目录不存在，跳过测试集评价")
        return None

    test_trajs = load_all_trajectories(synthetic_dir=test_dir)
    if len(test_trajs) == 0:
        print(f"[测试] test/ 为空，跳过测试集评价")
        return None

    test_set = TrajectoryDataset(test_trajs, ctx_len=config["ctx_len"],
                                  tgt_len=config["tgt_len"], augment=False)
    print(f"[测试] {len(test_trajs)} 条轨迹 → {len(test_set)} 个评估窗口")
    return DataLoader(test_set, batch_size=config["batch_size"], shuffle=False,
                      collate_fn=collate_fn, drop_last=False)


def create_model(config: dict, device: torch.device) -> PhyODEDiffusion:
    model = PhyODEDiffusion(
        d_feat=config.get("d_feat", 64),
        d_context=config.get("d_context", 128),
        n_head=config.get("n_head", 4),
        n_layers=config.get("n_layers", 3),
        dim_feedforward=config.get("dim_feedforward", 256),
        dropout=config.get("dropout", 0.1),
        d_z=config.get("d_z", 32),
        a_max=config.get("a_max", 30.0),
        ode_hidden_dim=config.get("ode_hidden_dim", 64),
        n_diffusion_steps=config.get("n_diffusion_steps", 500),
        n_inference_steps=config.get("n_inference_steps", 50),
        tau_emb_dim=config.get("tau_emb_dim", 16),
        dt_emb_dim=config.get("dt_emb_dim", 16),
        diff_hidden_dim=config.get("diff_hidden_dim", 128),
        guidance_eta=config.get("guidance_eta", 0.1),
        v_max=config.get("v_max", 30.0),
        z_min=config.get("z_min", 0.0),
        z_max=config.get("z_max", 120.0),
        v_v_up=config.get("v_v_up", 5.0),
        v_v_down=config.get("v_v_down", 3.0),
        max_tilt=config.get("max_tilt", 35.0),
        g=config.get("g", 9.81),
        obs_hidden_dim=config.get("obs_hidden_dim", 32),
    )
    model.to(device)
    model.diffusion.scheduler.to(device)
    return model


def save_checkpoint(model, out_dir, stage, epoch, val_loss, is_best=False,
                     optimizer=None, scheduler=None, ade=None, fde=None):
    """保存完整检查点：命名含 stage/epoch/loss，最佳模型单独保存"""
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "stage": stage, "epoch": epoch, "val_loss": val_loss,
        "model_info": model.get_model_info(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if ade is not None:
        ckpt["ADE"] = ade
    if fde is not None:
        ckpt["FDE"] = fde

    # 带描述的检查点文件名
    fname = f"phy_ode_diffusion_s{stage}_e{epoch}_v{val_loss:.4f}.pt"
    latest_path = os.path.join(out_dir, fname)
    torch.save(ckpt, latest_path)

    if is_best:
        best_name = f"phy_ode_diffusion_best_s{stage}.pt"
        best_path = os.path.join(out_dir, best_name)
        torch.save(ckpt, best_path)
        print(f"    -> 最佳模型: {best_path}")

        # 同步到统一管理目录 train_result/best/
        _best_central = os.path.join(_RESULTS_DIR, "best")
        os.makedirs(_best_central, exist_ok=True)
        shutil.copy2(best_path, os.path.join(_best_central, best_name))

    return latest_path


def evaluate_test(model, test_loader, device, out_dir: str, val_steps: int = 10):
    """在测试集上评估最终模型，输出详细指标并写入 JSON。"""
    print(f"\n{'='*60}")
    print("  测试集评价")
    print(f"{'='*60}")

    model.eval()
    total_loss, total_ade, total_fde = 0.0, 0.0, 0.0
    total_spd_v, total_acc_v, total_hgt_v = 0.0, 0.0, 0.0
    n_batches, n_viol = 0, 0
    max_a_h = model.g * np.tan(np.radians(model.max_tilt))

    with torch.no_grad():
        for batch in test_loader:
            b = _to_device(batch, device)
            c = model.transformer(b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"])
            h = model.state_manager.init_state(b["ctx_pos"][:, -1, :],
                                                b["ctx_vel"][:, -1, :], c)
            t_now = b["ctx_t"][:, -1]
            diff_loss, count = 0.0, 0
            preds, tgts = [], []
            prev_pos = b["ctx_pos"][:, -1, :]

            for j in range(b["tgt_pos"].shape[1]):
                dt_step = b["tgt_t"][:, j] - t_now
                h_prior = model.state_manager.evolve(h, t_now, b["tgt_t"][:, j])
                # 使用完整扩散采样进行预测（而非仅 ODE 先验）
                prev_p = b["ctx_pos"][:, -1, :] if j == 0 else b["tgt_pos"][:, j-1, :]
                p_mean_t = b.get("p_mean")
                p_std_t = b.get("p_std")
                p_pred = model.diffusion.guided_sampling(
                    h_prior, dt_step, prev_p, n_steps=val_steps,
                    p_mean=p_mean_t, p_std=p_std_t)
                if torch.isnan(p_pred).any():
                    p_pred = h_prior[:, :3]  # NaN 回退到 ODE 先验
                preds.append(p_pred); tgts.append(b["tgt_pos"][:, j, :])
                loss_dict = model.diffusion.compute_loss(
                    h_prior, dt_step, prev_p, b["tgt_pos"][:, j, :],
                    p_mean=p_mean_t, p_std=p_std_t)
                diff_loss += loss_dict["diff_loss"]

                if b.get("p_std") is not None:
                    p_phys = p_pred * b["p_std"] + b.get("p_mean", 0)
                    prev_phys = prev_pos * b["p_std"] + b.get("p_mean", 0)
                    v_phys = (p_phys - prev_phys) / dt_step.clamp(min=1e-3).unsqueeze(-1)
                    # 水平速度（XZ 平面），与 _physics_cost 一致
                    v_h = torch.norm(v_phys[:, [0, 2]], dim=-1)
                    total_spd_v += torch.nn.functional.relu(v_h - model.v_max).mean().item()
                    # 水平加速度：a_h = v_h / dt，与 g·tan(max_tilt) 比较
                    a_h = v_h / dt_step.clamp(min=1e-3)
                    total_acc_v += torch.nn.functional.relu(a_h - max_a_h).mean().item()
                    # 高度违反：同时检查下限和上限
                    total_hgt_v += (torch.nn.functional.relu(model.z_min - p_phys[:, 1]) +
                                    torch.nn.functional.relu(p_phys[:, 1] - model.z_max)).mean().item()
                    n_viol += 1

                h = model.state_manager.update(h_prior, dt_step,
                                               b["tgt_pos"][:, j, :], b["tgt_vel"][:, j, :])
                prev_pos = b["tgt_pos"][:, j, :]
                t_now = b["tgt_t"][:, j]
                count += 1

            total_loss += (diff_loss / max(count, 1)).item()
            if preds:
                errs = torch.norm(torch.stack(preds, 1) - torch.stack(tgts, 1), dim=-1)
                total_ade += errs.mean().item()
                total_fde += errs[:, -1].mean().item()
            n_batches += 1

    n = max(n_batches, 1); nv = max(n_viol, 1)
    metrics = {
        "test_loss": round(total_loss / n, 6),
        "test_ADE": round(total_ade / n, 6),
        "test_FDE": round(total_fde / n, 6),
        "test_speed_violation": round(total_spd_v / nv, 6),
        "test_accel_violation": round(total_acc_v / nv, 6),
        "test_height_violation": round(total_hgt_v / nv, 6),
    }

    print(f"  Loss:             {metrics['test_loss']:.4f}")
    print(f"  ADE:              {metrics['test_ADE']:.4f}")
    print(f"  FDE:              {metrics['test_FDE']:.4f}")
    print(f"  Speed Violation:  {metrics['test_speed_violation']:.4f}")
    print(f"  Accel Violation:  {metrics['test_accel_violation']:.4f}")
    print(f"  Height Violation: {metrics['test_height_violation']:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "test_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  测试指标已保存: {json_path}")

    # 生成测试图表
    if HAS_MPL:
        _make_test_chart(metrics, out_dir)

    model.train()
    return metrics


def _label_smooth(tgt_pos: torch.Tensor, p_std: torch.Tensor,
                  smoothing: float) -> torch.Tensor:
    """
    回归标签平滑：对目标位置添加小幅高斯噪声作为正则化。

    噪声标准差 = smoothing × p_std（适应各样本的数据尺度）。
    smoothing=0 时返回原始数据（无平滑）。
    """
    if smoothing <= 0:
        return tgt_pos
    noise = torch.randn_like(tgt_pos) * smoothing * p_std
    return tgt_pos + noise


def _build_scheduler(optimizer, total_epochs, warmup_epochs, start_factor):
    """
    构建 warmup + cosine 退火复合调度器。

    策略: 前 warmup_epochs 轮 LR 从 start_factor×base_lr 线性增长到 base_lr，
         之后按 CosineAnnealingLR 衰减到 0。
    若 warmup_epochs=0 则仅使用 Cosine 退火。
    """
    if warmup_epochs <= 0 or warmup_epochs >= total_epochs:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def _check_nan(loss, stage, epoch):
    """检测 NaN 损失，返回 True 表示需要跳过"""
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n[警告] Stage {stage} Epoch {epoch}: loss 为 NaN/Inf，跳过此步")
        return True
    return False


# ── 阶段一：编码器 + ODE + GRU ────────────────────────

def train_stage1(model, train_loader, val_loader, config, device, logger: TrainingLogger,
                  resume_ckpt=None):
    print("\n" + "=" * 60)
    print("  Stage 1: 训练 Transformer + ODE + GRU（无扩散）")
    print("=" * 60)

    for param in model.diffusion.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr_stage1"], weight_decay=config["weight_decay"],
    )
    scheduler = _build_scheduler(optimizer, config["epochs_stage1"],
                                  config["warmup_epochs_s1"], config["warmup_start_factor"])

    # 恢复优化器/调度器状态
    start_epoch = 1
    if resume_ckpt is not None and resume_ckpt.get("stage") == 1:
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        print(f"  [RESUME] 从 epoch {start_epoch} 继续训练")

    best_val_loss = float('inf')
    epoch_iter = range(start_epoch, config["epochs_stage1"] + 1)
    if HAS_TQDM:
        epoch_iter = _tqdm_import(epoch_iter, desc="Stage 1", unit="epoch",
                                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        t0 = time.time()

        batch_iter = train_loader
        if HAS_TQDM:
            batch_iter = _tqdm_import(batch_iter, desc=f"Epoch {epoch}", leave=False, unit="batch")

        for batch in batch_iter:
            b = _to_device(batch, device)
            c = model.transformer(b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"])
            h = model.state_manager.init_state(b["ctx_pos"][:, -1, :],
                                                b["ctx_vel"][:, -1, :], c)
            t_now = b["ctx_t"][:, -1]
            mse_loss = 0.0
            phy_loss = 0.0
            count = 0

            for j in range(b["tgt_pos"].shape[1]):
                t_next = b["tgt_t"][:, j]
                dt_step = t_next - t_now
                h_prior = model.state_manager.evolve(h, t_now, t_next)
                p_pred = h_prior[:, :3]
                tgt_s = _label_smooth(b["tgt_pos"][:, j, :], b["p_std"],
                                     config["label_smoothing"])
                mse_loss += torch.nn.functional.mse_loss(p_pred, tgt_s)
                # 物理违反损失：反标准化后在原始空间计算，与验证/推理一致
                prev_p = b["ctx_pos"][:, -1, :] if j == 0 else b["tgt_pos"][:, j-1, :]
                if b.get("p_std") is not None:
                    p_phys = p_pred * b["p_std"] + b.get("p_mean", 0)
                    prev_phys = prev_p * b["p_std"] + b.get("p_mean", 0)
                    v_phys = (p_phys - prev_phys) / dt_step.clamp(min=1e-3).unsqueeze(-1)
                    v_h = torch.norm(v_phys[:, [0, 2]], dim=-1)
                    phy_loss += torch.nn.functional.relu(v_h - model.v_max).mean()
                else:
                    v_pred = (p_pred - prev_p) / dt_step.clamp(min=1e-3).unsqueeze(-1)
                    v_norm = torch.norm(v_pred, dim=-1)
                    phy_loss += torch.nn.functional.relu(v_norm - model.v_max).mean()
                h = model.state_manager.update(h_prior, dt_step,
                                               b["tgt_pos"][:, j, :], b["tgt_vel"][:, j, :])
                t_now = t_next
                count += 1

            loss = mse_loss / max(count, 1) + 0.1 * phy_loss / max(count, 1)
            optimizer.zero_grad()
            loss.backward()
            if not _check_nan(loss, 1, epoch):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            if HAS_TQDM:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train = total_loss / len(train_loader)
        val_loss, ade, fde, spd_v, acc_v, hgt_v = _validate_stage1(model, val_loader, device)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.log_epoch(1, epoch, avg_train, val_loss, lr, epoch_time,
                         extra={"ADE": ade, "FDE": fde,
                                "speed_violation": spd_v, "accel_violation": acc_v,
                                "height_violation": hgt_v})
        is_best = (val_loss == val_loss) and val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # 阶段中定期保存检查点
        save_interval = config.get("checkpoint_interval", 0)
        if save_interval > 0 and (is_best or epoch % save_interval == 0 or epoch == config["epochs_stage1"]):
            _ckpt_out = os.path.join(_PROJECT_ROOT, config["output_dir"])
            save_checkpoint(model, _ckpt_out, 1, epoch, val_loss,
                            is_best=is_best, optimizer=optimizer, scheduler=scheduler,
                            ade=ade, fde=fde)

        status = f"train={avg_train:.4f} val={val_loss:.4f} lr={lr:.2e}"
        if HAS_TQDM:
            epoch_iter.set_postfix_str(status + (" ★" if is_best else ""))
        else:
            print(f"Epoch {epoch:3d}/{config['epochs_stage1']} | {status} | {epoch_time:.1f}s")

    # 阶段一完成，保存完整检查点
    out_dir = os.path.join(_PROJECT_ROOT, config["output_dir"])
    save_checkpoint(model, out_dir, 1, config["epochs_stage1"], best_val_loss,
                    is_best=True, optimizer=optimizer, scheduler=scheduler)

    # 加载阶段一最佳模型（而非最后一个 epoch），作为阶段二的起点
    _best_s1 = os.path.join(out_dir, "phy_ode_diffusion_best_s1.pt")
    if os.path.isfile(_best_s1):
        _ckpt = torch.load(_best_s1, map_location=torch.device('cpu'))
        model.load_state_dict(_ckpt["model_state_dict"])
        print(f"  [LOAD] 已加载阶段一最佳模型 (val_loss={_ckpt.get('val_loss', '?'):.4f})")

    for param in model.diffusion.parameters():
        param.requires_grad = True


def _validate_stage1(model, val_loader, device):
    """返回 (MSE_loss, ADE, FDE, speed_viol, accel_viol, height_viol) 六元组"""
    model.eval()
    total_mse, total_ade, total_fde = 0.0, 0.0, 0.0
    total_spd_v, total_acc_v, total_hgt_v = 0.0, 0.0, 0.0
    n_batches, n_viol = 0, 0
    max_a_h = model.g * np.tan(np.radians(model.max_tilt))
    with torch.no_grad():
        for batch in val_loader:
            b = _to_device(batch, device)
            c = model.transformer(b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"])
            h = model.state_manager.init_state(b["ctx_pos"][:, -1, :],
                                                b["ctx_vel"][:, -1, :], c)
            t_now = b["ctx_t"][:, -1]
            mse, count = 0.0, 0
            preds, tgts = [], []
            prev_pos = b["ctx_pos"][:, -1, :]
            for j in range(b["tgt_pos"].shape[1]):
                t_next = b["tgt_t"][:, j]
                dt_step = t_next - t_now
                h_prior = model.state_manager.evolve(h, t_now, t_next)
                p_pred = h_prior[:, :3]
                mse += torch.nn.functional.mse_loss(p_pred, b["tgt_pos"][:, j, :])
                preds.append(p_pred); tgts.append(b["tgt_pos"][:, j, :])

                # 物理违反率（反标准化到原始空间，与 _physics_cost 一致）
                if b.get("p_std") is not None:
                    p_phys = p_pred * b["p_std"] + b.get("p_mean", 0)
                    prev_phys = prev_pos * b["p_std"] + b.get("p_mean", 0)
                    v_phys = (p_phys - prev_phys) / dt_step.clamp(min=1e-3).unsqueeze(-1)
                    # 水平速度（XZ 平面），与 _physics_cost 一致
                    v_h = torch.norm(v_phys[:, [0, 2]], dim=-1)
                    total_spd_v += torch.nn.functional.relu(v_h - model.v_max).mean().item()
                    # 水平加速度：a_h = v_h / dt，与 g·tan(max_tilt) 比较
                    a_h = v_h / dt_step.clamp(min=1e-3)
                    total_acc_v += torch.nn.functional.relu(a_h - max_a_h).mean().item()
                    # 高度违反：同时检查下限和上限
                    total_hgt_v += (torch.nn.functional.relu(model.z_min - p_phys[:, 1]) +
                                    torch.nn.functional.relu(p_phys[:, 1] - model.z_max)).mean().item()
                    n_viol += 1

                h = model.state_manager.update(h_prior, dt_step,
                                               b["tgt_pos"][:, j, :], b["tgt_vel"][:, j, :])
                prev_pos = b["tgt_pos"][:, j, :]  # 修复：无条件更新，避免速度计算用错前一帧
                t_now = t_next
                count += 1
            total_mse += (mse / max(count, 1)).item()
            if preds:
                errs = torch.norm(torch.stack(preds, 1) - torch.stack(tgts, 1), dim=-1)
                total_ade += errs.mean().item()
                total_fde += errs[:, -1].mean().item()
            n_batches += 1
    model.train()
    n = max(n_batches, 1); nv = max(n_viol, 1)
    return (total_mse / n, total_ade / n, total_fde / n,
            total_spd_v / nv, total_acc_v / nv, total_hgt_v / nv)


# ── 阶段二：扩散模型 ──────────────────────────────────

def train_stage2(model, train_loader, val_loader, config, device, logger: TrainingLogger,
                  resume_ckpt=None):
    print("\n" + "=" * 60)
    print("  Stage 2: 训练扩散模型（固定 Transformer + ODE + GRU）")
    print("=" * 60)

    # 冻结非扩散模块
    for name, param in model.named_parameters():
        if not name.startswith("diffusion"):
            param.requires_grad = False

    optimizer = optim.AdamW(model.diffusion.parameters(),
                            lr=config["lr_stage2"], weight_decay=config["weight_decay"])
    scheduler = _build_scheduler(optimizer, config["epochs_stage2"],
                                  config["warmup_epochs_s2"], config["warmup_start_factor"])

    start_epoch = 1
    if resume_ckpt is not None and resume_ckpt.get("stage") == 2:
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1

    best_val_loss = float('inf')
    epoch_iter = range(start_epoch, config["epochs_stage2"] + 1)
    if HAS_TQDM:
        epoch_iter = _tqdm_import(epoch_iter, desc="Stage 2", unit="epoch",
                                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        t0 = time.time()

        batch_iter = train_loader
        if HAS_TQDM:
            batch_iter = _tqdm_import(batch_iter, desc=f"Epoch {epoch}", leave=False, unit="batch")

        for batch in batch_iter:
            b = _to_device(batch, device)
            with torch.no_grad():
                c = model.transformer(b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"])
                h = model.state_manager.init_state(b["ctx_pos"][:, -1, :],
                                                    b["ctx_vel"][:, -1, :], c)
                t_now = b["ctx_t"][:, -1]

            diff_loss_total = 0.0
            phy_loss_total = 0.0
            count = 0
            for j in range(b["tgt_pos"].shape[1]):
                dt_step = b["tgt_t"][:, j] - t_now
                with torch.no_grad():
                    h_prior = model.state_manager.evolve(h, t_now, b["tgt_t"][:, j])
                prev_p = b["ctx_pos"][:, -1, :] if j == 0 else b["tgt_pos"][:, j-1, :]
                # 扩散模型学习去噪纯净样本，不应用标签平滑
                loss_dict = model.diffusion.compute_loss(
                    h_prior, dt_step, prev_p, b["tgt_pos"][:, j, :],
                    p_mean=b.get("p_mean"), p_std=b.get("p_std"))
                diff_loss_total += loss_dict["diff_loss"]
                phy_loss_total += loss_dict["physics_loss"]
                with torch.no_grad():
                    h = model.state_manager.update(h_prior, dt_step,
                                                   b["tgt_pos"][:, j, :], b["tgt_vel"][:, j, :])
                t_now = b["tgt_t"][:, j]
                count += 1

            loss = (diff_loss_total + config.get("physics_weight", 0.01) * phy_loss_total) / max(count, 1)
            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()
            if not _check_nan(loss, 2, epoch):
                torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        scheduler.step()
        avg_train = total_loss / max(len(train_loader), 1)
        _vsteps = config.get("val_inference_steps", 10)
        val_loss, ade, fde, spd_v, acc_v, hgt_v = _validate_stage2(
            model, val_loader, device, val_steps=_vsteps)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.log_epoch(2, epoch, avg_train, val_loss, lr, epoch_time,
                         extra={"ADE": ade, "FDE": fde,
                                "speed_violation": spd_v, "accel_violation": acc_v,
                                "height_violation": hgt_v})
        is_best = (val_loss == val_loss) and val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # 阶段中定期保存检查点
        save_interval = config.get("checkpoint_interval", 0)
        if save_interval > 0 and (is_best or epoch % save_interval == 0 or epoch == config["epochs_stage2"]):
            _ckpt_out = os.path.join(_PROJECT_ROOT, config["output_dir"])
            save_checkpoint(model, _ckpt_out, 2, epoch, val_loss,
                            is_best=is_best, optimizer=optimizer, scheduler=scheduler,
                            ade=ade, fde=fde)

        status = f"train={avg_train:.4f} val={val_loss:.4f} lr={lr:.2e}"
        if HAS_TQDM:
            epoch_iter.set_postfix_str(status + (" ★" if is_best else ""))
        else:
            print(f"Epoch {epoch:3d}/{config['epochs_stage2']} | {status} | {epoch_time:.1f}s")

    out_dir = os.path.join(_PROJECT_ROOT, config["output_dir"])
    save_checkpoint(model, out_dir, 2, config["epochs_stage2"], best_val_loss,
                    is_best=True, optimizer=optimizer, scheduler=scheduler)

    # 加载阶段二最佳模型，作为阶段三的起点
    _best_s2 = os.path.join(out_dir, "phy_ode_diffusion_best_s2.pt")
    if os.path.isfile(_best_s2):
        _ckpt = torch.load(_best_s2, map_location=torch.device('cpu'))
        model.load_state_dict(_ckpt["model_state_dict"])
        print(f"  [LOAD] 已加载阶段二最佳模型 (val_loss={_ckpt.get('val_loss', '?'):.4f})")

    for param in model.parameters():
        param.requires_grad = True


def _validate_stage2(model, val_loader, device, val_steps: int = 10):
    """返回 (diff_loss, ADE, FDE, speed_viol, accel_viol, height_viol)。
    使用扩散采样进行预测，val_steps 控制验证速度（默认10步）。"""
    model.eval()
    total_loss, total_ade, total_fde = 0.0, 0.0, 0.0
    total_spd_v, total_acc_v, total_hgt_v = 0.0, 0.0, 0.0
    n_batches, n_viol = 0, 0
    max_a_h = model.g * np.tan(np.radians(model.max_tilt))
    with torch.no_grad():
        for batch in val_loader:
            b = _to_device(batch, device)
            c = model.transformer(b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"])
            h = model.state_manager.init_state(b["ctx_pos"][:, -1, :],
                                                b["ctx_vel"][:, -1, :], c)
            t_now = b["ctx_t"][:, -1]
            diff_loss, count = 0.0, 0
            preds, tgts = [], []
            prev_pos = b["ctx_pos"][:, -1, :]
            for j in range(b["tgt_pos"].shape[1]):
                dt_step = b["tgt_t"][:, j] - t_now
                h_prior = model.state_manager.evolve(h, t_now, b["tgt_t"][:, j])
                prev_p = b["ctx_pos"][:, -1, :] if j == 0 else b["tgt_pos"][:, j-1, :]
                # 使用扩散采样进行预测（验证用少步数加速）
                p_mean_t = b.get("p_mean"); p_std_t = b.get("p_std")
                p_pred = model.diffusion.guided_sampling(
                    h_prior, dt_step, prev_p, n_steps=val_steps,
                    p_mean=p_mean_t, p_std=p_std_t)
                # NaN 保护：回退到 ODE 先验位置
                if torch.isnan(p_pred).any():
                    p_pred = h_prior[:, :3]
                preds.append(p_pred); tgts.append(b["tgt_pos"][:, j, :])
                loss_dict = model.diffusion.compute_loss(
                    h_prior, dt_step, prev_p, b["tgt_pos"][:, j, :],
                    p_mean=p_mean_t, p_std=p_std_t)
                diff_loss += loss_dict["diff_loss"]

                # 物理违反（与 _physics_cost 一致）
                if b.get("p_std") is not None:
                    p_phys = p_pred * b["p_std"] + b.get("p_mean", 0)
                    prev_phys = prev_pos * b["p_std"] + b.get("p_mean", 0)
                    v_phys = (p_phys - prev_phys) / dt_step.clamp(min=1e-3).unsqueeze(-1)
                    v_h = torch.norm(v_phys[:, [0, 2]], dim=-1)
                    total_spd_v += torch.nn.functional.relu(v_h - model.v_max).mean().item()
                    a_h = v_h / dt_step.clamp(min=1e-3)
                    total_acc_v += torch.nn.functional.relu(a_h - max_a_h).mean().item()
                    total_hgt_v += (torch.nn.functional.relu(model.z_min - p_phys[:, 1]) +
                                    torch.nn.functional.relu(p_phys[:, 1] - model.z_max)).mean().item()
                    n_viol += 1

                h = model.state_manager.update(h_prior, dt_step,
                                               b["tgt_pos"][:, j, :], b["tgt_vel"][:, j, :])
                prev_pos = b["tgt_pos"][:, j, :]
                t_now = b["tgt_t"][:, j]
                count += 1
            total_loss += (diff_loss / max(count, 1)).item()
            if preds:
                errs = torch.norm(torch.stack(preds, 1) - torch.stack(tgts, 1), dim=-1)
                total_ade += errs.mean().item()
                total_fde += errs[:, -1].mean().item()
            n_batches += 1
    model.train()
    n = max(n_batches, 1); nv = max(n_viol, 1)
    return (total_loss / n, total_ade / n, total_fde / n,
            total_spd_v / nv, total_acc_v / nv, total_hgt_v / nv)


def _validate_stage3(model, val_loader, device, val_steps: int = 10):
    """返回 (diff_loss, ADE, FDE, speed_viol, accel_viol, height_viol)。
    使用扩散采样 + 自回归状态更新，val_steps 控制验证速度。"""
    model.eval()
    total_loss, total_ade, total_fde = 0.0, 0.0, 0.0
    total_spd_v, total_acc_v, total_hgt_v = 0.0, 0.0, 0.0
    n_batches, n_viol = 0, 0
    max_a_h = model.g * np.tan(np.radians(model.max_tilt))
    with torch.no_grad():
        for batch in val_loader:
            b = _to_device(batch, device)
            c = model.transformer(b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"])
            h = model.state_manager.init_state(b["ctx_pos"][:, -1, :],
                                                b["ctx_vel"][:, -1, :], c)
            t_now = b["ctx_t"][:, -1]
            diff_loss, count = 0.0, 0
            preds, tgts = [], []
            prev_pos = b["ctx_pos"][:, -1, :]

            for j in range(b["tgt_pos"].shape[1]):
                dt_step = b["tgt_t"][:, j] - t_now
                h_prior = model.state_manager.evolve(h, t_now, b["tgt_t"][:, j])
                p_mean_t = b.get("p_mean"); p_std_t = b.get("p_std")
                # 使用扩散采样（验证用少步数加速，模拟推理行为）
                p_pred = model.diffusion.guided_sampling(
                    h_prior, dt_step, prev_pos, n_steps=val_steps,
                    p_mean=p_mean_t, p_std=p_std_t)
                if torch.isnan(p_pred).any():
                    p_pred = h_prior[:, :3]  # NaN 回退到 ODE 先验
                preds.append(p_pred); tgts.append(b["tgt_pos"][:, j, :])
                # 自回归：用预测位置（非真实值）估计速度并更新状态
                v_pred = (p_pred - prev_pos) / dt_step.clamp(min=1e-3).unsqueeze(-1)
                h = model.state_manager.update(h_prior, dt_step, p_pred, v_pred)
                prev_pos = p_pred
                t_now = b["tgt_t"][:, j]

                # 扩散损失（仅用于参考，不参与 ADE/FDE 计算）
                prev_p = b["ctx_pos"][:, -1, :] if j == 0 else b["tgt_pos"][:, j-1, :]
                loss_dict = model.diffusion.compute_loss(
                    h_prior, dt_step, prev_p, b["tgt_pos"][:, j, :],
                    p_mean=p_mean_t, p_std=p_std_t)
                diff_loss += loss_dict["diff_loss"]

                # 物理违反
                if b.get("p_std") is not None:
                    p_phys = p_pred * b["p_std"] + b.get("p_mean", 0)
                    prev_phys = prev_pos * b["p_std"] + b.get("p_mean", 0)
                    v_phys = (p_phys - prev_phys) / dt_step.clamp(min=1e-3).unsqueeze(-1)
                    v_h = torch.norm(v_phys[:, [0, 2]], dim=-1)
                    total_spd_v += torch.nn.functional.relu(v_h - model.v_max).mean().item()
                    a_h = v_h / dt_step.clamp(min=1e-3)
                    total_acc_v += torch.nn.functional.relu(a_h - max_a_h).mean().item()
                    total_hgt_v += (torch.nn.functional.relu(model.z_min - p_phys[:, 1]) +
                                    torch.nn.functional.relu(p_phys[:, 1] - model.z_max)).mean().item()
                    n_viol += 1

                count += 1
            total_loss += (diff_loss / max(count, 1)).item()
            if preds:
                errs = torch.norm(torch.stack(preds, 1) - torch.stack(tgts, 1), dim=-1)
                total_ade += errs.mean().item()
                total_fde += errs[:, -1].mean().item()
            n_batches += 1
    model.train()
    n = max(n_batches, 1); nv = max(n_viol, 1)
    return (total_loss / n, total_ade / n, total_fde / n,
            total_spd_v / nv, total_acc_v / nv, total_hgt_v / nv)


# ── 阶段三：联合微调 ──────────────────────────────────

def train_stage3(model, train_loader, val_loader, config, device, logger: TrainingLogger,
                  resume_ckpt=None):
    print("\n" + "=" * 60)
    print("  Stage 3: 联合微调（计划采样）")
    print("=" * 60)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config["lr_stage3"], weight_decay=config["weight_decay"])
    scheduler = _build_scheduler(optimizer, config["epochs_stage3"],
                                  config["warmup_epochs_s3"], config["warmup_start_factor"])

    start_epoch = 1
    if resume_ckpt is not None and resume_ckpt.get("stage") == 3:
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        print(f"  [RESUME] 从 epoch {start_epoch} 继续训练")

    best_val_loss = float('inf')

    epoch_iter = range(start_epoch, config["epochs_stage3"] + 1)
    if HAS_TQDM:
        epoch_iter = _tqdm_import(epoch_iter, desc="Stage 3", unit="epoch",
                                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        t0 = time.time()
        ss_prob = min(0.5, epoch / config["epochs_stage3"] * 0.5)

        batch_iter = train_loader
        if HAS_TQDM:
            batch_iter = _tqdm_import(batch_iter, desc=f"Epoch {epoch} ss={ss_prob:.2f}",
                                      leave=False, unit="batch")

        for batch in batch_iter:
            b = _to_device(batch, device)
            c = model.transformer(b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"])
            h = model.state_manager.init_state(b["ctx_pos"][:, -1, :],
                                                b["ctx_vel"][:, -1, :], c)
            t_now = b["ctx_t"][:, -1]
            diff_loss_total = 0.0
            phy_loss_total = 0.0
            count = 0
            last_pos = b["ctx_pos"][:, -1, :]

            for j in range(b["tgt_pos"].shape[1]):
                dt_step = b["tgt_t"][:, j] - t_now
                h_prior = model.state_manager.evolve(h, t_now, b["tgt_t"][:, j])
                use_gt = torch.rand(1).item() > ss_prob

                if use_gt or j == 0:
                    # 教师强制：使用真实位置（不做平滑，扩散模型需要纯净目标）
                    p_obs = b["tgt_pos"][:, j, :]
                    v_obs = b["tgt_vel"][:, j, :]
                else:
                    # 使用 ODE 先验位置（轻量），而非完整扩散采样（昂贵且不参与梯度）
                    p_obs = h_prior[:, :3].detach()
                    v_obs = (p_obs - last_pos) / dt_step.clamp(min=1e-3).unsqueeze(-1)

                loss_dict = model.diffusion.compute_loss(
                    h_prior, dt_step, last_pos, p_obs,
                    p_mean=b.get("p_mean"), p_std=b.get("p_std"))
                diff_loss_total += loss_dict["diff_loss"]
                phy_loss_total += loss_dict["physics_loss"]
                h = model.state_manager.update(h_prior, dt_step, p_obs, v_obs)
                last_pos = p_obs
                t_now = b["tgt_t"][:, j]
                count += 1

            loss = (diff_loss_total + config.get("physics_weight", 0.01) * phy_loss_total) / max(count, 1)
            optimizer.zero_grad()
            loss.backward()
            if not _check_nan(loss, 3, epoch):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            if HAS_TQDM:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train = total_loss / len(train_loader)

        # 检测到 NaN 则提前终止阶段三
        if avg_train != avg_train:  # NaN check
            print(f"\n[警告] Stage 3 Epoch {epoch}: NaN，提前终止阶段三")
            break

        _vsteps = config.get("val_inference_steps", 10)
        val_loss, ade, fde, spd_v, acc_v, hgt_v = _validate_stage3(
            model, val_loader, device, val_steps=_vsteps)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.log_epoch(3, epoch, avg_train, val_loss, lr, epoch_time,
                         extra={"ADE": ade, "FDE": fde,
                                "speed_violation": spd_v, "accel_violation": acc_v,
                                "height_violation": hgt_v,
                                "scheduled_sampling_prob": round(ss_prob, 3)})
        # NaN 保护：val_loss 为 NaN 时跳过最佳模型判断
        is_best = (val_loss == val_loss) and val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # 阶段中定期保存检查点
        save_interval = config.get("checkpoint_interval", 0)
        if save_interval > 0 and (is_best or epoch % save_interval == 0 or epoch == config["epochs_stage3"]):
            _ckpt_out = os.path.join(_PROJECT_ROOT, config["output_dir"])
            save_checkpoint(model, _ckpt_out, 3, epoch, val_loss,
                            is_best=is_best, optimizer=optimizer, scheduler=scheduler,
                            ade=ade, fde=fde)

        status = f"train={avg_train:.4f} val={val_loss:.4f} ss={ss_prob:.2f}"
        if HAS_TQDM:
            epoch_iter.set_postfix_str(status + (" ★" if is_best else ""))
        else:
            print(f"Epoch {epoch:3d}/{config['epochs_stage3']} | {status} | lr={lr:.2e} | {epoch_time:.1f}s")

    # 阶段三完成（或提前终止），保存完整检查点
    out_dir = os.path.join(_PROJECT_ROOT, config["output_dir"])
    save_checkpoint(model, out_dir, 3, epoch, best_val_loss,
                    is_best=True, optimizer=optimizer, scheduler=scheduler)


# ── 主入口 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phy-ODE-Diffusion 训练")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="1=仅阶段一, 2=阶段一→二, 3=仅阶段三, all=全部")
    parser.add_argument("--epochs", type=int, default=None,
                        help="统一设置三个阶段轮数（快捷方式）")
    parser.add_argument("--epochs-s1", type=int, default=None, help="阶段一轮数")
    parser.add_argument("--epochs-s2", type=int, default=None, help="阶段二轮数")
    parser.add_argument("--epochs-s3", type=int, default=None, help="阶段三轮数")
    parser.add_argument("--batch", type=int, default=32,
                        help="统一设置批次大小（快捷方式）")
    parser.add_argument("--batch-s1", type=int, default=None, help="阶段一批次大小")
    parser.add_argument("--batch-s2", type=int, default=None, help="阶段二批次大小")
    parser.add_argument("--batch-s3", type=int, default=None, help="阶段三批次大小")
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader 多进程数，默认4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ctx-len", type=int, default=20)
    parser.add_argument("--tgt-len", type=int, default=10)
    parser.add_argument("--warmup-s1", type=int, default=None, help="阶段一 warmup 轮数")
    parser.add_argument("--warmup-s2", type=int, default=None, help="阶段二 warmup 轮数")
    parser.add_argument("--warmup-s3", type=int, default=None, help="阶段三 warmup 轮数")
    parser.add_argument("--warmup-factor", type=float, default=None, help="warmup 起始 LR 因子")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help="标签平滑噪声标准差比例")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="阶段中保存检查点的间隔（epoch），0=仅阶段结束时保存")
    parser.add_argument("--no-charts", action="store_true", help="跳过图表生成")
    parser.add_argument("--charts-only", type=str, default=None, metavar="DIR",
                        help="仅从已有 training_history.json 重新生成图表（不训练）。"
                             "传入训练结果目录路径或 JSON 文件路径")
    args = parser.parse_args()

    # ── 仅生成图表模式 ──────────────────────────────────
    if args.charts_only:
        if not HAS_MPL:
            print("[错误] matplotlib 未安装，无法生成图表")
            sys.exit(1)
        _charts_path = args.charts_only
        # 如果是目录，拼接 training_history.json
        if os.path.isdir(_charts_path):
            _json_path = os.path.join(_charts_path, "training_history.json")
        else:
            _json_path = _charts_path
        if not os.path.isfile(_json_path):
            print(f"[错误] 找不到训练历史文件: {_json_path}")
            sys.exit(1)
        with open(_json_path, 'r', encoding='utf-8') as f:
            _hist_data = json.load(f)
        logger = TrainingLogger()
        logger.history = _hist_data.get("history", [])
        logger.stage_labels = _hist_data.get("stage_labels", {})
        # 支持按阶段筛选（--stage 1/2/3）
        _filter_stage = args.stage if args.stage != "all" else None
        if _filter_stage is not None:
            _s = int(_filter_stage)
            logger.history = [e for e in logger.history if e.get("stage") == _s]
            logger.stage_labels = {_s: logger.stage_labels.get(str(_s), f"Stage {_s}")}
            print(f"[图表] 仅生成阶段 {_s} 的图表")
        if not logger.history:
            print("[错误] 训练历史为空")
            sys.exit(1)
        _out_dir = os.path.dirname(_json_path)
        _stage_only = _filter_stage is not None
        print(f"[图表] 从 {_json_path} 重新生成图表 → {_out_dir}")
        _plot_all_charts(logger, _out_dir, stages_only=_stage_only)
        # 测试图表（若有 test_metrics.json）
        _test_json = os.path.join(_out_dir, "test_metrics.json")
        if os.path.isfile(_test_json):
            with open(_test_json, 'r', encoding='utf-8') as f:
                _test_metrics = json.load(f)
            _make_test_chart(_test_metrics, _out_dir)
        print("图表生成完成！")
        return

    config = DEFAULT_CONFIG.copy()

    # 从统一配置读取 training 参数（命令行可覆盖）
    try:
        from trajectory_reconstruction.core.config.config_manager import ensure_config
        _train_cfg = ensure_config().get("training", {})
        for _k in DEFAULT_CONFIG:
            if _k in _train_cfg:
                config[_k] = _train_cfg[_k]
        print("[CONFIG] 已从配置读取训练参数")
    except Exception as e:
        print(f"[WARN] 读取训练配置失败: {e}")

    config["ctx_len"] = args.ctx_len
    config["tgt_len"] = args.tgt_len
    config["batch_size"] = args.batch
    if args.batch_s1 is not None:
        config["batch_size_s1"] = args.batch_s1
    if args.batch_s2 is not None:
        config["batch_size_s2"] = args.batch_s2
    if args.batch_s3 is not None:
        config["batch_size_s3"] = args.batch_s3
    if args.workers is not None:
        config["num_workers"] = args.workers
    config["device"] = args.device

    # 轮次：--epochs 统一设置，--epochs-s1/2/3 可单独覆盖
    if args.epochs:
        config["epochs_stage1"] = args.epochs
        config["epochs_stage2"] = args.epochs
        config["epochs_stage3"] = args.epochs
    if args.epochs_s1 is not None:
        config["epochs_stage1"] = args.epochs_s1
    if args.epochs_s2 is not None:
        config["epochs_stage2"] = args.epochs_s2
    if args.epochs_s3 is not None:
        config["epochs_stage3"] = args.epochs_s3
    if args.lr:
        config["lr_stage1"] = args.lr
        config["lr_stage2"] = args.lr
        config["lr_stage3"] = args.lr * 0.1
    if args.warmup_s1 is not None:
        config["warmup_epochs_s1"] = args.warmup_s1
    if args.warmup_s2 is not None:
        config["warmup_epochs_s2"] = args.warmup_s2
    if args.warmup_s3 is not None:
        config["warmup_epochs_s3"] = args.warmup_s3
    if args.warmup_factor is not None:
        config["warmup_start_factor"] = args.warmup_factor
    if args.label_smoothing is not None:
        config["label_smoothing"] = args.label_smoothing
    if args.checkpoint_interval is not None:
        config["checkpoint_interval"] = args.checkpoint_interval

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[TQDM] {'可用' if HAS_TQDM else '不可用（pip install tqdm）'}")
    print(f"[图表] {'可用' if HAS_MPL else '不可用（pip install matplotlib）'}")

    # 创建本次运行的输出目录（带时间戳，避免覆盖历史结果）
    _run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    _run_dir = os.path.join(_RESULTS_DIR, _run_id)
    out_dir = os.path.join(_run_dir, "models")
    os.makedirs(out_dir, exist_ok=True)
    config["output_dir"] = os.path.relpath(out_dir, _PROJECT_ROOT).replace("\\", "/")
    print(f"[输出] 本次训练结果: {_run_dir}")

    # 数据
    train_loader, val_loader = build_dataloaders(config)

    # 日志器
    logger = TrainingLogger()

    resumed_stage = 0
    resume_ckpt = None
    if args.resume:
        print(f"[RESUME] 从 {args.resume} 恢复完整训练状态")
        resume_ckpt = torch.load(args.resume, map_location=device)
        model = create_model(config, device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        model.diffusion.scheduler.to(device)
        resumed_stage = resume_ckpt.get("stage", 0)
        resume_epoch = resume_ckpt.get("epoch", 0)
        print(f"[RESUME] 已完成: stage={resumed_stage}, epoch={resume_epoch}, "
              f"val_loss={resume_ckpt.get('val_loss', 0.0):.4f}")
        if "ADE" in resume_ckpt and "FDE" in resume_ckpt:
            print(f"[RESUME] 历史指标: ADE={resume_ckpt['ADE']:.4f}, FDE={resume_ckpt['FDE']:.4f}")
    else:
        model = create_model(config, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] 总参数量: {total_params:,}")

    t_total_start = time.time()

    # --stage 语义: 1=阶段一, 2=阶段一二, 3/all=全部
    run_s1 = args.stage in ("1", "2", "all") and resumed_stage < 1
    run_s2 = args.stage in ("2", "all") and resumed_stage < 2
    run_s3 = args.stage in ("3", "all") and resumed_stage < 3

    # 自动加载已完成的前阶段最佳模型（避免重复训练）
    def _load_best_from_central(ckpt_name: str) -> bool:
        _cp = os.path.join(_RESULTS_DIR, "best", ckpt_name)
        if os.path.isfile(_cp):
            _ckpt = torch.load(_cp, map_location=device)
            model.load_state_dict(_ckpt["model_state_dict"])
            model.diffusion.scheduler.to(device)
            print(f"[AUTO-LOAD] {ckpt_name} (val_loss={_ckpt.get('val_loss', '?'):.4f})")
            return True
        return False

    if not args.resume and run_s2 and run_s1 and args.stage != "all":
        # --stage 2: 有历史阶段一最佳模型 → 跳过阶段一
        if _load_best_from_central("phy_ode_diffusion_best_s1.pt"):
            run_s1 = False

    if not args.resume and run_s3 and run_s2 and args.stage != "all":
        # --stage 3: 有历史阶段二最佳模型 → 跳过阶段一二
        if _load_best_from_central("phy_ode_diffusion_best_s2.pt"):
            run_s1 = False
            run_s2 = False

    # 按阶段应用独立的 batch_size（若配置了 batch_sN 则重建 loader）
    def _stage_loader(stage: int, default_loader):
        key = f"batch_size_s{stage}"
        if key in config:
            bs = config[key]
            print(f"[Stage {stage}] 使用专属 batch_size={bs}")
            config["batch_size"] = bs
            return build_dataloaders(config)
        return default_loader

    def _stage_finish(stage_label: str, stage_num: int):
        """阶段结束后保存 JSON + 生成该阶段所有独立图表"""
        os.makedirs(_run_dir, exist_ok=True)
        logger.save(_run_dir)
        if not args.no_charts and HAS_MPL and logger.history:
            print(f"\n[图表] {stage_label}完成，生成阶段图表...")
            _plot_all_charts(logger, _run_dir, config, stages_only=True)

    if run_s1:
        _loader = _stage_loader(1, (train_loader, val_loader))
        train_stage1(model, _loader[0], _loader[1], config, device, logger, resume_ckpt)
        _stage_finish("阶段一", 1)

    if run_s2:
        _loader = _stage_loader(2, (train_loader, val_loader))
        train_stage2(model, _loader[0], _loader[1], config, device, logger, resume_ckpt)
        _stage_finish("阶段二", 2)

    if run_s3:
        _loader = _stage_loader(3, (train_loader, val_loader))
        train_stage3(model, _loader[0], _loader[1], config, device, logger, resume_ckpt)
        _stage_finish("阶段三", 3)

    # 全部阶段完成后生成全局混合图表（跨阶段对比、仪表盘等）
    if not args.no_charts and HAS_MPL and logger.history:
        print(f"\n[图表] 生成全局混合图表...")
        _plot_all_charts(logger, _run_dir)

    total_time = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"  训练完成！总耗时: {total_time/60:.1f} 分钟")
    print(f"{'='*60}")

    # 保存最终权重（各阶段已完成 checkpoint 保存，此处兜底）
    out_dir = os.path.join(_PROJECT_ROOT, config["output_dir"])
    import glob as _pt_glob
    _existing = _pt_glob.glob(os.path.join(out_dir, "*.pt"))
    if not _existing:
        final_loss = logger.history[-1]["val_loss"] if logger.history else 0.0
        save_checkpoint(model, out_dir, 0, 0, final_loss, is_best=True)

    # ── 输出图表和日志到 train/hybrid_predictor/train_result/ ──
    results_dir = _run_dir

    # ── 测试集评价 ──────────────────────────────────────
    test_loader = build_test_loader(config)
    test_metrics = None
    if test_loader is not None:
        # 加载最佳 checkpoint（优先级: s3 > s2 > s1）
        best_ckpt_path = None
        best_ckpt_loss = float('inf')
        for _s in (3, 2, 1):
            _best_p = os.path.join(out_dir, f"phy_ode_diffusion_best_s{_s}.pt")
            if os.path.isfile(_best_p):
                best_ckpt_path = _best_p
                break
        # 若无 best，从普通 checkpoint 中按 loss 最低选取
        if best_ckpt_path is None:
            for _f in _pt_glob.glob(os.path.join(out_dir, "phy_ode_diffusion_s*.pt")):
                try:
                    _loss = float(_f.rsplit('_v', 1)[-1].rstrip('.pt'))
                    if _loss < best_ckpt_loss:
                        best_ckpt_loss = _loss
                        best_ckpt_path = _f
                except (ValueError, IndexError):
                    pass
        _vsteps = config.get("val_inference_steps", 10)
        if best_ckpt_path is not None:
            test_model = create_model(config, device)
            test_ckpt = torch.load(best_ckpt_path, map_location=device)
            test_model.load_state_dict(test_ckpt["model_state_dict"])
            test_model.diffusion.scheduler.to(device)
            test_metrics = evaluate_test(test_model, test_loader, device,
                                          results_dir, val_steps=_vsteps)
        else:
            # 无最佳模型时用当前模型直接评估
            test_metrics = evaluate_test(model, test_loader, device,
                                          results_dir, val_steps=_vsteps)
    os.makedirs(results_dir, exist_ok=True)

    # 保存训练历史 JSON
    history_path = logger.save(results_dir)
    print(f"[日志] 训练历史: {history_path}")

    # 图表已在各阶段完成后生成，此处仅保存摘要
    summary = _save_training_summary(logger, config, results_dir)
    summary["total_params"] = total_params
    summary["total_time_minutes"] = round(total_time / 60, 1)
    if test_metrics is not None:
        summary["test_metrics"] = test_metrics
    with open(os.path.join(results_dir, "training_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 将最佳模型复制到推理目录
    _infer_dir = os.path.join(_PROJECT_ROOT, "models", "hybrid_predictor")
    os.makedirs(_infer_dir, exist_ok=True)
    for _f in os.listdir(out_dir):
        if _f.startswith("phy_ode_diffusion_best") or _f.startswith("phy_ode_diffusion_s"):
            _src = os.path.join(out_dir, _f)
            _dst = os.path.join(_infer_dir, _f)
            shutil.copy2(_src, _dst)
            print(f"[复制] {_f} -> models/hybrid_predictor/")

    print(f"\n所有输出已保存到: {results_dir}")
    print("训练完成！")


if __name__ == "__main__":
    main()
