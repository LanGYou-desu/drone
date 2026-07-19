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
    "dataset_dir": "train/hybrid_predictor/dataset",
    "output_dir": "train/hybrid_predictor/train_result/models",
    "device": "cuda:0",
    "val_split": 0.1,
    # 模型架构参数（可通过 config.json → training 覆盖）
    "d_feat": 64,
    "d_context": 128,
    "n_head": 4,
    "n_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "d_z": 32,
    "a_max": 30.0,
    "ode_hidden_dim": 64,
    "n_diffusion_steps": 500,
    "n_inference_steps": 50,
    "tau_emb_dim": 16,
    "dt_emb_dim": 16,
    "diff_hidden_dim": 128,
    "guidance_eta": 0.1,
    "v_max": 30.0,
    "z_min": 0.0,
    "obs_hidden_dim": 32,
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


# ── 图表生成（多种评价指标）─────────────────────────────

# 全局色板
_COLORS = {
    'stage1': '#2196F3', 'stage2': '#FF9800', 'stage3': '#4CAF50',
    'train': '#2196F3', 'val': '#EF5350', 'best': '#4CAF50',
    'speed': '#FF7043', 'accel': '#AB47BC', 'height': '#26C6DA',
}


def _make_loss_breakdown_chart(logger: TrainingLogger, output_dir: str):
    """图1: 损失分解 — 每阶段 train/val 损失 + 最优标注"""
    stages_data = {}
    for e in logger.history:
        s = e["stage"]
        if s not in stages_data:
            stages_data[s] = {"epochs": [], "train": [], "val": []}
        stages_data[s]["epochs"].append(e["epoch"])
        stages_data[s]["train"].append(e["train_loss"])
        stages_data[s]["val"].append(e["val_loss"])

    n_stages = len(stages_data)
    fig, axes = _plt.subplots(2, n_stages, figsize=(6 * n_stages, 10))
    if n_stages == 1:
        axes = axes.reshape(2, 1)

    for idx, (stage, data) in enumerate(sorted(stages_data.items())):
        color = _COLORS.get(f'stage{stage}', '#333')
        epochs = data["epochs"]

        # 上排: 损失曲线
        ax = axes[0, idx]
        ax.plot(epochs, data["train"], '-', color=color, linewidth=1.8, alpha=0.85, label='Train')
        ax.plot(epochs, data["val"], '--', color=_COLORS['val'], linewidth=1.8, alpha=0.85, label='Val')
        # 最优标注
        best_epoch = epochs[data["val"].index(min(data["val"]))]
        best_val = min(data["val"])
        ax.axvline(x=best_epoch, color=_COLORS['best'], linestyle=':', alpha=0.5, linewidth=1)
        ax.annotate(f'Best: {best_val:.4f}', xy=(best_epoch, best_val),
                    xytext=(10, 15), textcoords='offset points', fontsize=9,
                    arrowprops=dict(arrowstyle='->', lw=1, color=_COLORS['best']),
                    color=_COLORS['best'], fontweight='bold')
        ax.set_title(f'Stage {stage} — Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # 下排: 对数尺度损失（更清晰观察收敛）
        ax = axes[1, idx]
        ax.semilogy(epochs, data["train"], '-', color=color, linewidth=1.5, alpha=0.8, label='Train (log)')
        ax.semilogy(epochs, data["val"], '--', color=_COLORS['val'], linewidth=1.5, alpha=0.8, label='Val (log)')
        ax.set_title(f'Stage {stage} — Loss (log scale)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    _plt.suptitle('Training Loss Breakdown', fontsize=15, fontweight='bold', y=1.01)
    _plt.tight_layout()
    path = os.path.join(output_dir, "01_loss_breakdown.png")
    _plt.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f"[图表1] 损失分解: {path}")


def _make_convergence_chart(logger: TrainingLogger, output_dir: str):
    """图2: 收敛性分析 — val/train 比、损失下降率、学习率"""
    fig, axes = _plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Convergence Analysis', fontsize=14, fontweight='bold')

    # 获取全部 epoch 序列（跨阶段连续）
    all_epochs = list(range(1, len(logger.history) + 1))
    train_losses = [e["train_loss"] for e in logger.history]
    val_losses = [e["val_loss"] for e in logger.history]
    stage_boundaries = []
    current_stage = None
    for i, e in enumerate(logger.history):
        if e["stage"] != current_stage:
            stage_boundaries.append(i)
            current_stage = e["stage"]

    # (0,0) 全局损失曲线
    ax = axes[0, 0]
    ax.plot(all_epochs, train_losses, '-', color=_COLORS['train'], linewidth=1.2, alpha=0.8, label='Train')
    ax.plot(all_epochs, val_losses, '-', color=_COLORS['val'], linewidth=1.2, alpha=0.8, label='Val')
    for b in stage_boundaries[1:]:
        ax.axvline(x=b, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_title('Global Loss Curve')
    ax.set_xlabel('Global Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,1) val/train 比
    ax = axes[0, 1]
    ratios = [v / max(t, 1e-8) for v, t in zip(val_losses, train_losses)]
    ax.plot(all_epochs, ratios, '-', color='#7E57C2', linewidth=1.5)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.fill_between(all_epochs, 0, ratios, alpha=0.15, color='#7E57C2')
    ax.set_title('Val/Train Ratio (Overfitting Detection)')
    ax.set_xlabel('Global Epoch'); ax.set_ylabel('Ratio')
    ax.grid(True, alpha=0.3)

    # (0,2) 损失下降百分比
    ax = axes[0, 2]
    for stage, data in sorted({e["stage"]: [] for e in logger.history}.items()):
        stage_losses = [e["val_loss"] for e in logger.history if e["stage"] == stage]
        if len(stage_losses) < 2:
            continue
        improvements = [(stage_losses[0] - l) / stage_losses[0] * 100 for l in stage_losses]
        color = _COLORS.get(f'stage{stage}', '#333')
        ax.plot(range(1, len(improvements) + 1), improvements, '-o', color=color,
                markersize=3, linewidth=1.5, label=f'Stage {stage}')
    ax.set_title('Improvement Over Initial Loss (%)')
    ax.set_xlabel('Epoch (within stage)'); ax.set_ylabel('Improvement %')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,0) 学习率衰减
    ax = axes[1, 0]
    for stage in sorted(set(e["stage"] for e in logger.history)):
        lrs = [e["lr"] for e in logger.history if e["stage"] == stage]
        epochs_s = list(range(1, len(lrs) + 1))
        color = _COLORS.get(f'stage{stage}', '#333')
        ax.semilogy(epochs_s, lrs, '-o', color=color, markersize=3, linewidth=1.5, label=f'Stage {stage}')
    ax.set_title('Learning Rate Schedule (log)')
    ax.set_xlabel('Epoch (within stage)'); ax.set_ylabel('LR')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,1) 每 epoch 耗时
    ax = axes[1, 1]
    times = [e["epoch_time_s"] for e in logger.history]
    colors_t = [_COLORS.get(f'stage{e["stage"]}', '#333') for e in logger.history]
    ax.bar(all_epochs, times, color=colors_t, alpha=0.7, width=0.8)
    avg_time = sum(times) / len(times) if times else 0
    ax.axhline(y=avg_time, color='red', linestyle='--', alpha=0.5, label=f'Avg: {avg_time:.1f}s')
    ax.set_title('Epoch Training Time')
    ax.set_xlabel('Global Epoch'); ax.set_ylabel('Time (s)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # (1,2) 累积训练时间
    ax = axes[1, 2]
    cum_times = [sum(times[:i+1]) / 60 for i in range(len(times))]
    ax.fill_between(all_epochs, 0, cum_times, alpha=0.3, color='#26C6DA')
    ax.plot(all_epochs, cum_times, '-', color='#00838F', linewidth=2)
    for b in stage_boundaries[1:]:
        ax.axvline(x=b, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_title('Cumulative Training Time')
    ax.set_xlabel('Global Epoch'); ax.set_ylabel('Time (min)')
    ax.grid(True, alpha=0.3)

    _plt.tight_layout()
    path = os.path.join(output_dir, "02_convergence_analysis.png")
    _plt.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f"[图表2] 收敛性分析: {path}")


def _make_stage_comparison_chart(logger: TrainingLogger, output_dir: str):
    """图3: 阶段对比 — 柱状图 + 雷达图 + 箱线图"""
    stages = sorted(set(e["stage"] for e in logger.history))
    if len(stages) < 1:
        return

    fig = _plt.figure(figsize=(18, 6))
    fig.suptitle('Stage Comparison', fontsize=14, fontweight='bold')

    # (左) 柱状图: 各阶段最终/最优损失
    ax = fig.add_subplot(1, 3, 1)
    stage_labels = [f'S{s}' for s in stages]
    final_losses = []
    best_losses = []
    for s in stages:
        s_data = [e for e in logger.history if e["stage"] == s]
        final_losses.append(s_data[-1]["val_loss"] if s_data else 0)
        best_losses.append(min(e["val_loss"] for e in s_data) if s_data else 0)

    x = range(len(stage_labels))
    w = 0.35
    bars1 = ax.bar([i - w/2 for i in x], final_losses, w, label='Final Loss', color='#FF9800', alpha=0.8)
    bars2 = ax.bar([i + w/2 for i in x], best_losses, w, label='Best Loss', color='#4CAF50', alpha=0.8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(list(x)); ax.set_xticklabels(stage_labels)
    ax.set_title('Final vs Best Val Loss')
    ax.set_ylabel('Loss'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    # (中) 雷达图: 各阶段综合评分
    ax = fig.add_subplot(1, 3, 2, projection='polar')
    metrics = ['1-Final\nLoss', '1-Best\nLoss', 'Speed\n(1/epoch)', 'Stability\n(val/train)']
    scores = {}
    for s in stages:
        s_data = [e for e in logger.history if e["stage"] == s]
        if not s_data:
            continue
        final = s_data[-1]["val_loss"]
        best = min(e["val_loss"] for e in s_data)
        avg_time = sum(e["epoch_time_s"] for e in s_data) / len(s_data)
        ratios = [e["val_loss"] / max(e["train_loss"], 1e-8) for e in s_data]
        stability = 1.0 / (max(ratios) - min(ratios) + 1) if len(ratios) > 1 else 1

        # 归一化到 [0,1]
        max_loss = max(final_losses + best_losses) if final_losses else 1
        max_time = max(
            sum(e["epoch_time_s"] for e in logger.history if e["stage"] == ss) / max(len([x for x in logger.history if x["stage"] == ss]), 1)
            for ss in stages
        ) if stages else 1

        scores[s] = [
            1 - final / max(max_loss, 1e-8),
            1 - best / max(max_loss, 1e-8),
            1 - avg_time / max(max_time, 1e-8),
            stability,
        ]

    angles = [n * 2 * np.pi / len(metrics) for n in range(len(metrics))]
    angles += angles[:1]  # 闭合

    for s, values in scores.items():
        values_plot = values + values[:1]
        color = _COLORS.get(f'stage{s}', '#333')
        ax.fill(angles, values_plot, alpha=0.15, color=color)
        ax.plot(angles, values_plot, 'o-', linewidth=2, color=color, label=f'Stage {s}')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_title('Radar: Stage Comparison', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    # (右) 箱线图: 损失分布
    ax = fig.add_subplot(1, 3, 3)
    box_data = [[e["val_loss"] for e in logger.history if e["stage"] == s] for s in stages]
    bp = ax.boxplot(box_data, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    ax.set_xticklabels(stage_labels)
    for i, (patch, s) in enumerate(zip(bp['boxes'], stages)):
        patch.set_facecolor(_COLORS.get(f'stage{s}', '#333'))
        patch.set_alpha(0.6)
    ax.set_title('Val Loss Distribution per Stage')
    ax.set_ylabel('Loss'); ax.grid(True, alpha=0.3, axis='y')

    _plt.tight_layout()
    path = os.path.join(output_dir, "03_stage_comparison.png")
    _plt.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f"[图表3] 阶段对比: {path}")


def _make_ade_fde_chart(logger: TrainingLogger, output_dir: str):
    """图4: ADE/FDE 验证指标曲线"""
    has_ade = any("ADE" in e for e in logger.history)
    if not has_ade:
        print("[图表4] 无 ADE/FDE 数据，跳过")
        return

    fig, axes = _plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Prediction Accuracy Metrics (ADE / FDE)', fontsize=14, fontweight='bold')

    # 按阶段分组
    for stage in sorted(set(e["stage"] for e in logger.history)):
        s_data = [e for e in logger.history if e["stage"] == stage
                  and "ADE" in e and "FDE" in e]
        if not s_data:
            continue
        epochs = [e["epoch"] for e in s_data]
        color = _COLORS.get(f'stage{stage}', '#333')

        # ADE
        ax = axes[0, 0]
        ade_vals = [e["ADE"] for e in s_data]
        ax.plot(epochs, ade_vals, 'o-', color=color, markersize=3, linewidth=1.5,
                label=f'Stage {stage}')
        ax.set_title('ADE (Average Displacement Error)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('ADE')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if ade_vals:
            best_i = ade_vals.index(min(ade_vals))
            ax.annotate(f'{ade_vals[best_i]:.3f}', xy=(epochs[best_i], ade_vals[best_i]),
                        fontsize=8, color='darkred')

        # FDE
        ax = axes[0, 1]
        fde_vals = [e["FDE"] for e in s_data]
        ax.plot(epochs, fde_vals, 's--', color=color, markersize=3, linewidth=1.5,
                label=f'Stage {stage}')
        ax.set_title('FDE (Final Displacement Error)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('FDE')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if fde_vals:
            best_i = fde_vals.index(min(fde_vals))
            ax.annotate(f'{fde_vals[best_i]:.3f}', xy=(epochs[best_i], fde_vals[best_i]),
                        fontsize=8, color='darkred')

        # ADE vs FDE 对比（各阶段数据叠加在同一图上）
        ax = axes[1, 0]
        ax.scatter(ade_vals, fde_vals, c=epochs, cmap='viridis', s=25, alpha=0.7, label=f'S{stage}')

        # ADE 改进率（各阶段数据叠加在同一图上）
        ax = axes[1, 1]
        if len(ade_vals) > 1:
            imp = [(ade_vals[0] - v) / max(ade_vals[0], 1e-8) * 100 for v in ade_vals]
            ax.plot(epochs, imp, '-', color=color, linewidth=1.5, label=f'Stage {stage}')

    # 设置 subplot [1,0] / [1,1] 标题和标签（循环外，避免覆盖）
    axes[1, 0].set_title('ADE vs FDE (color=epoch)')
    axes[1, 0].set_xlabel('ADE'); axes[1, 0].set_ylabel('FDE')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('ADE Improvement Over Initial (%)')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Improvement %')
    axes[1, 1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

    _plt.tight_layout()
    path = os.path.join(output_dir, "04_ade_fde.png")
    _plt.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f"[图表4] ADE/FDE: {path}")


def _make_physics_metrics_chart(logger: TrainingLogger, output_dir: str):
    """图5: 物理指标 — 速度/加速度/高度违反率"""
    has_physics = any("speed_violation" in e for e in logger.history)
    if not has_physics:
        print("[图表5] 无物理指标数据，跳过")
        return

    fig, axes = _plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Physics Constraint Violations', fontsize=14, fontweight='bold')

    all_epochs = list(range(1, len(logger.history) + 1))

    metrics_config = [
        ("speed_violation", "Speed Violation Rate", _COLORS['speed'], axes[0, 0]),
        ("accel_violation", "Acceleration Violation Rate", _COLORS['accel'], axes[0, 1]),
        ("height_violation", "Height Violation Rate", _COLORS['height'], axes[1, 0]),
    ]

    for key, title, color, ax in metrics_config:
        if key in logger.history[0]:
            vals = [e.get(key, 0) for e in logger.history]
            ax.plot(all_epochs, vals, '-', color=color, linewidth=1.8, alpha=0.8)
            ax.fill_between(all_epochs, 0, vals, alpha=0.1, color=color)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Global Epoch'); ax.set_ylabel('Rate')
            ax.grid(True, alpha=0.3)
            if vals:
                best_i = vals.index(min(vals))
                ax.annotate(f'{vals[best_i]:.4f}', xy=(best_i + 1, vals[best_i]),
                            fontsize=9, color='darkred', fontweight='bold')

    ax = axes[1, 1]
    if all(k in logger.history[0] for k in ["speed_violation", "accel_violation", "height_violation"]):
        phys_scores = []
        for e in logger.history:
            score = 1.0 - (e.get("speed_violation", 0) + e.get("accel_violation", 0) + e.get("height_violation", 0)) / 3
            phys_scores.append(max(0, score))
        ax.plot(all_epochs, phys_scores, '-', color='#4CAF50', linewidth=2)
        ax.fill_between(all_epochs, 0, phys_scores, alpha=0.15, color='#4CAF50')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_title('Composite Physics Score (↑ better)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Global Epoch'); ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    _plt.tight_layout()
    path = os.path.join(output_dir, "05_physics_metrics.png")
    _plt.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f"[图表5] 物理指标: {path}")


def _make_training_summary_dashboard(logger: TrainingLogger, output_dir: str):
    """图6: 综合仪表盘 — 一页展示所有关键指标"""
    fig = _plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    fig.suptitle('Phy-ODE-Diffusion Training Dashboard', fontsize=16, fontweight='bold', y=0.98)

    all_epochs = list(range(1, len(logger.history) + 1))
    train_losses = [e["train_loss"] for e in logger.history]
    val_losses = [e["val_loss"] for e in logger.history]

    # Row 0: 损失曲线全景
    ax = fig.add_subplot(gs[0, :2])
    for stage in sorted(set(e["stage"] for e in logger.history)):
        idxs = [i for i, e in enumerate(logger.history) if e["stage"] == stage]
        ep = [all_epochs[i] for i in idxs]
        tr = [train_losses[i] for i in idxs]
        vl = [val_losses[i] for i in idxs]
        color = _COLORS.get(f'stage{stage}', '#333')
        ax.plot(ep, tr, '-', color=color, linewidth=1.2, alpha=0.7)
        ax.plot(ep, vl, '--', color=color, linewidth=1.5, alpha=0.9, label=f'S{stage} Val')
    ax.set_title('Loss Overview (— Train, --- Val)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Global Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 0 right: 训练统计
    ax = fig.add_subplot(gs[0, 2])
    stats_text = "Training Statistics\n" + "─" * 25 + "\n"
    for s in sorted(set(e["stage"] for e in logger.history)):
        s_data = [e for e in logger.history if e["stage"] == s]
        if s_data:
            stats_text += f"Stage {s}:\n"
            stats_text += f"  Epochs: {len(s_data)}\n"
            stats_text += f"  Best Val: {min(e['val_loss'] for e in s_data):.4f}\n"
            stats_text += f"  Final Val: {s_data[-1]['val_loss']:.4f}\n"
            stats_text += f"  Avg Time: {sum(e['epoch_time_s'] for e in s_data)/len(s_data):.1f}s\n"
    total_time = sum(e["epoch_time_s"] for e in logger.history)
    stats_text += f"─" * 25 + f"\nTotal: {total_time/60:.1f} min\n"
    stats_text += f"Best Overall: {logger.best_val_loss:.4f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))
    ax.axis('off')

    # Row 0 right-right: 模型参数
    ax = fig.add_subplot(gs[0, 3])
    ax.axis('off')
    model_text = "Model Architecture\n" + "─" * 25 + "\n"
    model_text += "Transformer: 3 layers, 4 heads\n"
    model_text += "d_feat=64, d_context=128\n"
    model_text += "ODE: d_z=32, a_max=30\n"
    model_text += "Diffusion: T=500, DDIM=50\n"
    model_text += "GRU: obs_dim=32\n"
    model_text += f"─" * 25 + "\n"
    model_text += "Data: ctx=20, tgt=10\n"
    ax.text(0.05, 0.95, model_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

    # Row 1: 学习曲线细节
    ax = fig.add_subplot(gs[1, :2])
    for stage in sorted(set(e["stage"] for e in logger.history)):
        lrs = [e["lr"] for e in logger.history if e["stage"] == stage]
        color = _COLORS.get(f'stage{stage}', '#333')
        ax.semilogy(range(1, len(lrs) + 1), lrs, '-o', color=color, markersize=2,
                     linewidth=1.2, label=f'Stage {stage} LR')
    ax.set_title('Learning Rate Schedule (log scale)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch (within stage)'); ax.set_ylabel('LR')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 1 right: 时间分布
    ax = fig.add_subplot(gs[1, 2:])
    times_by_stage = {}
    for e in logger.history:
        s = e["stage"]
        times_by_stage.setdefault(s, []).append(e["epoch_time_s"])
    stage_labels = [f'Stage {s}' for s in sorted(times_by_stage.keys())]
    totals = [sum(ts) / 60 for ts in times_by_stage.values()]
    colors = [_COLORS.get(f'stage{s}', '#333') for s in sorted(times_by_stage.keys())]
    ax.pie(totals, labels=stage_labels, autopct='%1.1f%%', colors=colors,
           startangle=90, textprops={'fontsize': 10})
    ax.set_title(f'Time Distribution ({sum(totals):.1f} min total)', fontsize=12, fontweight='bold')

    # Row 2: 各阶段损失分布直方图
    for idx, s in enumerate(sorted(set(e["stage"] for e in logger.history))):
        ax = fig.add_subplot(gs[2, idx] if idx < 3 else gs[2, 3])
        s_vals = [e["val_loss"] for e in logger.history if e["stage"] == s]
        color = _COLORS.get(f'stage{s}', '#333')
        ax.hist(s_vals, bins=min(15, len(s_vals)), color=color, alpha=0.6, edgecolor='white')
        ax.axvline(x=min(s_vals), color='red', linestyle='--', linewidth=1, label=f'Best: {min(s_vals):.4f}')
        ax.axvline(x=s_vals[-1], color='orange', linestyle='--', linewidth=1, label=f'Final: {s_vals[-1]:.4f}')
        ax.set_title(f'Stage {s} Loss Histogram', fontsize=11, fontweight='bold')
        ax.set_xlabel('Val Loss'); ax.set_ylabel('Count')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    _plt.tight_layout()
    path = os.path.join(output_dir, "06_dashboard.png")
    _plt.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f"[图表6] 综合仪表盘: {path}")


def _plot_all_charts(logger: TrainingLogger, output_dir: str):
    """生成全部 6 张图表"""
    if not HAS_MPL:
        print("[跳过] matplotlib 未安装")
        return
    if not logger.history:
        print("[跳过] 训练历史为空")
        return

    os.makedirs(output_dir, exist_ok=True)
    _make_loss_breakdown_chart(logger, output_dir)       # 01: 损失分解
    _make_convergence_chart(logger, output_dir)            # 02: 收敛性分析
    _make_stage_comparison_chart(logger, output_dir)       # 03: 阶段对比
    _make_ade_fde_chart(logger, output_dir)                # 04: ADE/FDE
    _make_physics_metrics_chart(logger, output_dir)        # 05: 物理指标
    _make_training_summary_dashboard(logger, output_dir)   # 06: 综合仪表盘


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
                   collate_fn=collate_fn, drop_last=True),
        DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                   collate_fn=collate_fn, drop_last=False),
    )


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """构建训练/验证数据加载器。优先级: data/ > train/|valid/"""
    from numpy.random import default_rng

    dataset_root = os.path.join(_PROJECT_ROOT, config["dataset_dir"])

    # 优先 data/ 目录
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
            print(f"[数据] data/ → train: {len(train_trajs)}, valid: {len(valid_trajs)}")
            return _make_loaders(train_trajs, valid_trajs, config)

    # 回退 train/ | valid/ 目录
    train_dir = os.path.join(dataset_root, "train")
    valid_dir = os.path.join(dataset_root, "valid")

    train_trajs = load_all_trajectories(synthetic_dir=train_dir) \
        if os.path.isdir(train_dir) else []
    valid_trajs = load_all_trajectories(synthetic_dir=valid_dir) \
        if os.path.isdir(valid_dir) else []

    if len(train_trajs) == 0:
        raise RuntimeError(
            f"没有可用的训练数据！请将 .npz 轨迹文件放入以下任一目录:\n"
            f"  • {data_dir}\n"
            f"  • {train_dir}\n"
            f"或使用 generate_synthetic.py 生成合成数据:\n"
            f"  python train/hybrid_predictor/generate_synthetic.py 200"
        )
    print(f"[数据] train: {len(train_trajs)} 条轨迹, valid: {len(valid_trajs)} 条轨迹")
    return _make_loaders(train_trajs, valid_trajs, config)


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
        best_path = os.path.join(out_dir, f"phy_ode_diffusion_best_s{stage}.pt")
        torch.save(ckpt, best_path)
        print(f"    -> 最佳模型: {best_path}")
    return latest_path


def evaluate_test(model, test_loader, device, out_dir: str):
    """在测试集上评估最终模型，输出详细指标并写入 JSON。"""
    print(f"\n{'='*60}")
    print("  测试集评价")
    print(f"{'='*60}")

    model.eval()
    total_loss, total_ade, total_fde = 0.0, 0.0, 0.0
    total_spd_v, total_acc_v, total_hgt_v = 0.0, 0.0, 0.0
    n_batches, n_viol = 0, 0

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
                p_pred = h_prior[:, :3]
                preds.append(p_pred); tgts.append(b["tgt_pos"][:, j, :])
                prev_p = b["ctx_pos"][:, -1, :] if j == 0 else b["tgt_pos"][:, j-1, :]
                loss_dict = model.diffusion.compute_loss(
                    h_prior, dt_step, prev_p, b["tgt_pos"][:, j, :])
                diff_loss += loss_dict["diff_loss"]

                if b.get("p_std") is not None:
                    p_phys = p_pred * b["p_std"] + b.get("p_mean", 0)
                    prev_phys = prev_pos * b["p_std"] + b.get("p_mean", 0)
                    v_norm = torch.norm((p_phys - prev_phys) / dt_step.clamp(min=1e-3).unsqueeze(-1), dim=-1)
                    total_spd_v += torch.nn.functional.relu(v_norm - model.v_max).mean().item()
                    a_h = v_norm / dt_step.clamp(min=1e-3)
                    total_acc_v += torch.nn.functional.relu(a_h - model.a_max).mean().item()
                    total_hgt_v += torch.nn.functional.relu(model.z_min - p_phys[:, 1]).mean().item()
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


def _make_test_chart(metrics: dict, output_dir: str):
    """图7: 测试集评价仪表盘 — 指标柱状图 + 物理违反 + 雷达图"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Test Set Evaluation', fontsize=15, fontweight='bold')

    # (左) 精度指标柱状图
    ax = fig.add_subplot(2, 2, 1)
    acc_metrics = {
        'Loss': metrics.get('test_loss', 0),
        'ADE': metrics.get('test_ADE', 0),
        'FDE': metrics.get('test_FDE', 0),
    }
    colors_acc = ['#2196F3', '#FF9800', '#EF5350']
    bars = ax.barh(list(acc_metrics.keys()), list(acc_metrics.values()),
                   color=colors_acc, alpha=0.85, height=0.5)
    for bar, val in zip(bars, acc_metrics.values()):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Value'); ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(acc_metrics.values()) * 1.2 if max(acc_metrics.values()) > 0 else 1)

    # (右) 物理违反率柱状图
    ax = fig.add_subplot(2, 2, 2)
    phy_metrics = {
        'Speed': metrics.get('test_speed_violation', 0),
        'Accel': metrics.get('test_accel_violation', 0),
        'Height': metrics.get('test_height_violation', 0),
    }
    colors_phy = ['#FF7043', '#AB47BC', '#26C6DA']
    bars = ax.barh(list(phy_metrics.keys()), list(phy_metrics.values()),
                   color=colors_phy, alpha=0.85, height=0.5)
    for bar, val in zip(bars, phy_metrics.values()):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    ax.set_title('Physics Constraint Violations', fontsize=13, fontweight='bold')
    ax.set_xlabel('Rate'); ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(phy_metrics.values()) * 1.3 if max(phy_metrics.values()) > 0 else 1)

    # (左下) 评测概要
    ax = fig.add_subplot(2, 2, 3)
    ax.axis('off')
    summary = "Test Evaluation Summary\n" + "─" * 28 + "\n"
    summary += f"Loss:              {metrics.get('test_loss', 0):.4f}\n"
    summary += f"ADE (avg error):   {metrics.get('test_ADE', 0):.4f}\n"
    summary += f"FDE (final error): {metrics.get('test_FDE', 0):.4f}\n"
    summary += f"Speed  violation:  {metrics.get('test_speed_violation', 0):.4f}\n"
    summary += f"Accel  violation:  {metrics.get('test_accel_violation', 0):.4f}\n"
    summary += f"Height violation:  {metrics.get('test_height_violation', 0):.4f}\n"
    phy_score = 1.0 - sum(phy_metrics.values()) / 3
    summary += f"─" * 28 + f"\nPhysics Score:     {phy_score:.4f}"
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    # (右下) 综合雷达图
    ax = fig.add_subplot(2, 2, 4, projection='polar')
    max_ade = max(metrics.get('test_ADE', 0.001), 0.001)
    max_fde = max(metrics.get('test_FDE', 0.001), 0.001)
    radar_labels = ['1-ADE', '1-FDE', '1-Speed\nViol', '1-Accel\nViol',
                    '1-Height\nViol', 'Physics\nScore']
    radar_values = [
        max(0, 1 - metrics.get('test_ADE', 0) / max_ade),
        max(0, 1 - metrics.get('test_FDE', 0) / max_fde),
        max(0, 1 - metrics.get('test_speed_violation', 0)),
        max(0, 1 - metrics.get('test_accel_violation', 0)),
        max(0, 1 - metrics.get('test_height_violation', 0)),
        phy_score,
    ]
    angles = [n * 2 * np.pi / len(radar_labels) for n in range(len(radar_labels))]
    radar_values += radar_values[:1]
    angles += angles[:1]
    ax.fill(angles, radar_values, alpha=0.2, color='#4CAF50')
    ax.plot(angles, radar_values, 'o-', linewidth=2, color='#4CAF50')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=8)
    ax.set_title('Composite Radar', fontsize=13, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "07_test_evaluation.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表7] 测试评价: {path}")


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
                prev_p = b["ctx_pos"][:, -1, :] if j == 0 else b["tgt_pos"][:, j-1, :]
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
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        status = f"train={avg_train:.4f} val={val_loss:.4f} lr={lr:.2e}"
        if HAS_TQDM:
            epoch_iter.set_postfix_str(status + (" ★" if is_best else ""))
        else:
            print(f"Epoch {epoch:3d}/{config['epochs_stage1']} | {status} | {epoch_time:.1f}s")

    # 阶段一完成，保存完整检查点
    out_dir = os.path.join(_PROJECT_ROOT, config["output_dir"])
    save_checkpoint(model, out_dir, 1, config["epochs_stage1"], best_val_loss,
                    is_best=True, optimizer=optimizer, scheduler=scheduler)

    for param in model.diffusion.parameters():
        param.requires_grad = True


def _validate_stage1(model, val_loader, device):
    """返回 (MSE_loss, ADE, FDE, speed_viol, accel_viol, height_viol) 六元组"""
    model.eval()
    total_mse, total_ade, total_fde = 0.0, 0.0, 0.0
    total_spd_v, total_acc_v, total_hgt_v = 0.0, 0.0, 0.0
    n_batches, n_viol = 0, 0
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

                # 物理违反率（反标准化到原始空间）
                if b.get("p_std") is not None:
                    p_phys = p_pred * b["p_std"] + b.get("p_mean", 0)
                    prev_phys = prev_pos * b["p_std"] + b.get("p_mean", 0)
                    v_phys = (p_phys - prev_phys) / dt_step.clamp(min=1e-3).unsqueeze(-1)
                    v_norm = torch.norm(v_phys, dim=-1)
                    total_spd_v += torch.nn.functional.relu(v_norm - model.v_max).mean().item()
                    total_hgt_v += torch.nn.functional.relu(model.z_min - p_phys[:, 1]).mean().item()
                    # 加速度违反率（修复：此前 total_acc_v 从未被更新，恒为 0）
                    a_h = v_norm / dt_step.clamp(min=1e-3)
                    total_acc_v += torch.nn.functional.relu(a_h - model.a_max).mean().item()
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
            count = 0
            for j in range(b["tgt_pos"].shape[1]):
                dt_step = b["tgt_t"][:, j] - t_now
                with torch.no_grad():
                    h_prior = model.state_manager.evolve(h, t_now, b["tgt_t"][:, j])
                prev_p = b["ctx_pos"][:, -1, :] if j == 0 else b["tgt_pos"][:, j-1, :]
                tgt_smooth = _label_smooth(b["tgt_pos"][:, j, :], b["p_std"],
                                           config["label_smoothing"])
                loss_dict = model.diffusion.compute_loss(
                    h_prior, dt_step, prev_p, tgt_smooth)
                diff_loss_total += loss_dict["diff_loss"]
                with torch.no_grad():
                    h = model.state_manager.update(h_prior, dt_step,
                                                   b["tgt_pos"][:, j, :], b["tgt_vel"][:, j, :])
                t_now = b["tgt_t"][:, j]
                count += 1

            loss = diff_loss_total / max(count, 1)
            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()
            if not _check_nan(loss, 2, epoch):
                torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        scheduler.step()
        avg_train = total_loss / max(len(train_loader), 1)
        val_loss, ade, fde, spd_v, acc_v, hgt_v = _validate_stage2(model, val_loader, device)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.log_epoch(2, epoch, avg_train, val_loss, lr, epoch_time,
                         extra={"ADE": ade, "FDE": fde,
                                "speed_violation": spd_v, "accel_violation": acc_v,
                                "height_violation": hgt_v})
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        status = f"train={avg_train:.4f} val={val_loss:.4f} lr={lr:.2e}"
        if HAS_TQDM:
            epoch_iter.set_postfix_str(status + (" ★" if is_best else ""))
        else:
            print(f"Epoch {epoch:3d}/{config['epochs_stage2']} | {status} | {epoch_time:.1f}s")

    out_dir = os.path.join(_PROJECT_ROOT, config["output_dir"])
    save_checkpoint(model, out_dir, 2, config["epochs_stage2"], best_val_loss,
                    is_best=True, optimizer=optimizer, scheduler=scheduler)

    for param in model.parameters():
        param.requires_grad = True


def _validate_stage2(model, val_loader, device):
    """返回 (diff_loss, ADE, FDE, speed_viol, accel_viol, height_viol)"""
    model.eval()
    total_loss, total_ade, total_fde = 0.0, 0.0, 0.0
    total_spd_v, total_acc_v, total_hgt_v = 0.0, 0.0, 0.0
    n_batches, n_viol = 0, 0
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
                p_pred = h_prior[:, :3]
                preds.append(p_pred); tgts.append(b["tgt_pos"][:, j, :])
                prev_p = b["ctx_pos"][:, -1, :] if j == 0 else b["tgt_pos"][:, j-1, :]
                loss_dict = model.diffusion.compute_loss(
                    h_prior, dt_step, prev_p, b["tgt_pos"][:, j, :])
                diff_loss += loss_dict["diff_loss"]

                # 物理违反
                if b.get("p_std") is not None:
                    p_phys = p_pred * b["p_std"] + b.get("p_mean", 0)
                    prev_phys = prev_pos * b["p_std"] + b.get("p_mean", 0)
                    v_norm = torch.norm((p_phys - prev_phys) / dt_step.clamp(min=1e-3).unsqueeze(-1), dim=-1)
                    total_spd_v += torch.nn.functional.relu(v_norm - model.v_max).mean().item()
                    a_h = v_norm / dt_step.clamp(min=1e-3)
                    total_acc_v += torch.nn.functional.relu(a_h - model.a_max).mean().item()
                    total_hgt_v += torch.nn.functional.relu(model.z_min - p_phys[:, 1]).mean().item()
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
            count = 0
            last_pos = b["ctx_pos"][:, -1, :]

            for j in range(b["tgt_pos"].shape[1]):
                dt_step = b["tgt_t"][:, j] - t_now
                h_prior = model.state_manager.evolve(h, t_now, b["tgt_t"][:, j])
                use_gt = torch.rand(1).item() > ss_prob

                if use_gt or j == 0:
                    p_obs = _label_smooth(b["tgt_pos"][:, j, :], b["p_std"],
                                          config["label_smoothing"])
                    v_obs = b["tgt_vel"][:, j, :]
                else:
                    # 使用 ODE 先验位置（轻量），而非完整扩散采样（昂贵且不参与梯度）
                    p_obs = h_prior[:, :3].detach()
                    v_obs = (p_obs - last_pos) / dt_step.clamp(min=1e-3).unsqueeze(-1)

                loss_dict = model.diffusion.compute_loss(h_prior, dt_step, last_pos, p_obs)
                diff_loss_total += loss_dict["diff_loss"]
                h = model.state_manager.update(h_prior, dt_step, p_obs, v_obs)
                last_pos = p_obs
                t_now = b["tgt_t"][:, j]
                count += 1

            loss = diff_loss_total / max(count, 1)
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

        val_loss, ade, fde, spd_v, acc_v, hgt_v = _validate_stage2(model, val_loader, device)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.log_epoch(3, epoch, avg_train, val_loss, lr, epoch_time,
                         extra={"ADE": ade, "FDE": fde,
                                "speed_violation": spd_v, "accel_violation": acc_v,
                                "height_violation": hgt_v,
                                "scheduled_sampling_prob": round(ss_prob, 3)})
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
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
    parser.add_argument("--no-charts", action="store_true", help="跳过图表生成")
    args = parser.parse_args()

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
    config["device"] = args.device

    if args.epochs:
        config["epochs_stage1"] = args.epochs
        config["epochs_stage2"] = args.epochs
        config["epochs_stage3"] = max(10, args.epochs // 2)
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[TQDM] {'可用' if HAS_TQDM else '不可用（pip install tqdm）'}")
    print(f"[图表] {'可用' if HAS_MPL else '不可用（pip install matplotlib）'}")

    # 数据
    train_loader, val_loader = build_dataloaders(config)

    # 模型
    # 日志器
    logger = TrainingLogger()
    out_dir = os.path.join(_PROJECT_ROOT, config["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

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

    # --stage 语义: 恢复时自动跳过已完成的阶段
    run_s1 = args.stage in ("1", "2", "all") and resumed_stage < 1
    run_s2 = args.stage in ("2", "all") and resumed_stage < 2
    run_s3 = args.stage in ("3", "all") and resumed_stage < 3

    if run_s1:
        train_stage1(model, train_loader, val_loader, config, device, logger, resume_ckpt)

    if run_s2:
        train_stage2(model, train_loader, val_loader, config, device, logger, resume_ckpt)

    if run_s3:
        train_stage3(model, train_loader, val_loader, config, device, logger, resume_ckpt)

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
    results_dir = str(_RESULTS_DIR)

    # ── 测试集评价 ──────────────────────────────────────
    test_loader = build_test_loader(config)
    if test_loader is not None:
        # 加载最佳 checkpoint（优先级: s3 > s2 > s1）
        best_ckpt_path = None
        for _s in (3, 2, 1):
            _p = os.path.join(out_dir, f"phy_ode_diffusion_best_s{_s}.pt")
            if os.path.isfile(_p):
                best_ckpt_path = _p
                break
        if best_ckpt_path is not None:
            test_model = create_model(config, device)
            test_ckpt = torch.load(best_ckpt_path, map_location=device)
            test_model.load_state_dict(test_ckpt["model_state_dict"])
            test_model.diffusion.scheduler.to(device)
            test_metrics = evaluate_test(test_model, test_loader, device, results_dir)
        else:
            # 无最佳模型时用当前模型直接评估
            test_metrics = evaluate_test(model, test_loader, device, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # 保存训练历史 JSON
    history_path = logger.save(results_dir)
    print(f"[日志] 训练历史: {history_path}")

    # 生成多种评价指标图表
    if not args.no_charts and HAS_MPL:
        print(f"\n{'='*50}")
        print("  生成训练图表")
        print(f"{'='*50}")
        _plot_all_charts(logger, results_dir)

    # 保存摘要
    summary = _save_training_summary(logger, config, results_dir)
    summary["total_params"] = total_params
    summary["total_time_minutes"] = round(total_time / 60, 1)
    if test_loader is not None and 'test_metrics' in dir():
        summary["test_metrics"] = test_metrics
    with open(os.path.join(results_dir, "training_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n所有输出已保存到: {results_dir}")
    print("训练完成！")


if __name__ == "__main__":
    main()
