"""
Phy-ODE-Diffusion 分阶段训练脚本（含进度条、图表输出和详细日志）

输出目录: train/hybrid_predictor/results/
  - loss_curves.png         训练/验证损失曲线
  - stage_comparison.png    各阶段对比图
  - training_history.json   详细训练日志
  - training_summary.json   训练摘要

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

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 路径设置
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_DIR = os.path.dirname(_MODULE_DIR)
_PROJECT_ROOT = os.path.dirname(_TRAIN_DIR)
for p in (_PROJECT_ROOT, _TRAIN_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from train.hybrid_predictor.dataset import (
    load_all_trajectories, TrajectoryDataset, collate_fn,
)
from train.hybrid_predictor.generate_synthetic import generate_dataset
from trajectory_reconstruction.core.prediction.hybrid import PhyODEDiffusion

# 输出目录
_RESULTS_DIR = os.path.join(_MODULE_DIR, "train_result")

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
    "dataset_dir": "train/hybrid_predictor/dataset",
    "output_dir": "models/hybrid_predictor",
    "device": "cuda:0",
    "val_split": 0.1,
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

    angles = [n * 2 * _plt.pi / len(metrics) for n in range(len(metrics))]
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
    bp = ax.boxplot(box_data, labels=stage_labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
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


def _make_physics_metrics_chart(logger: TrainingLogger, output_dir: str):
    """图4: 物理指标 — 速度/加速度/高度违反率（如果训练中记录了）"""
    # 检查是否记录了物理指标
    has_physics = any("speed_violation" in e for e in logger.history)
    if not has_physics:
        print("[图表4] 无物理指标数据，跳过")
        return

    fig, axes = _plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Physics Constraint Metrics', fontsize=14, fontweight='bold')

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
            # 标注最小违反
            if vals:
                best_i = vals.index(min(vals))
                ax.annotate(f'{vals[best_i]:.4f}', xy=(best_i + 1, vals[best_i]),
                            fontsize=9, color='darkred', fontweight='bold')

    # 综合物理得分
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
    else:
        ax.text(0.5, 0.5, 'No physics data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Composite Physics Score')

    _plt.tight_layout()
    path = os.path.join(output_dir, "04_physics_metrics.png")
    _plt.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f"[图表4] 物理指标: {path}")


def _make_training_summary_dashboard(logger: TrainingLogger, output_dir: str):
    """图5: 综合仪表盘 — 一页展示所有关键指标"""
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
    path = os.path.join(output_dir, "05_dashboard.png")
    _plt.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f"[图表5] 综合仪表盘: {path}")


def _plot_all_charts(logger: TrainingLogger, output_dir: str):
    """生成全部图表"""
    if not HAS_MPL:
        print("[跳过] matplotlib 未安装")
        return
    if not logger.history:
        print("[跳过] 训练历史为空")
        return

    os.makedirs(output_dir, exist_ok=True)
    _make_loss_breakdown_chart(logger, output_dir)
    _make_convergence_chart(logger, output_dir)
    _make_stage_comparison_chart(logger, output_dir)
    _make_physics_metrics_chart(logger, output_dir)
    _make_training_summary_dashboard(logger, output_dir)


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

def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """构建训练/验证数据加载器 — 仅从 train/hybrid_predictor/dataset/ 加载"""
    dataset_root = os.path.join(_PROJECT_ROOT, config["dataset_dir"])
    train_dir = os.path.join(dataset_root, "train")
    valid_dir = os.path.join(dataset_root, "valid")

    train_exists = os.path.isdir(train_dir) and \
        any(f.endswith(('.npz', '.dat')) for f in os.listdir(train_dir))
    if not train_exists:
        print("[INFO] 训练数据不存在，自动生成 200 条轨迹...")
        generate_dataset(output_dir=dataset_root, n_trajectories=200)

    train_trajs = load_all_trajectories(synthetic_dir=train_dir)
    valid_trajs = load_all_trajectories(synthetic_dir=valid_dir) \
        if os.path.isdir(valid_dir) else []

    if len(train_trajs) == 0:
        raise RuntimeError("没有可用的训练数据！")

    print(f"[数据] train: {len(train_trajs)} 条轨迹, valid: {len(valid_trajs)} 条轨迹")

    train_set = TrajectoryDataset(train_trajs, ctx_len=config["ctx_len"],
                                  tgt_len=config["tgt_len"], augment=True)
    val_set = TrajectoryDataset(
        valid_trajs if valid_trajs else train_trajs[:max(1, int(len(train_trajs) * 0.1))],
        ctx_len=config["ctx_len"], tgt_len=config["tgt_len"], augment=False,
    )

    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn, drop_last=False)
    return train_loader, val_loader


def create_model(config: dict, device: torch.device) -> PhyODEDiffusion:
    model = PhyODEDiffusion(
        d_feat=64, d_context=128, n_head=4, n_layers=3, dim_feedforward=256, dropout=0.1,
        d_z=32, a_max=30.0, ode_hidden_dim=64,
        n_diffusion_steps=500, n_inference_steps=50,
        tau_emb_dim=16, dt_emb_dim=16, diff_hidden_dim=128,
        guidance_eta=0.1, v_max=30.0, z_min=0.0, obs_hidden_dim=32,
    )
    model.to(device)
    model.diffusion.scheduler.to(device)
    return model


def save_final_model(model: PhyODEDiffusion, config: dict, stage: int,
                     epoch: int, loss: float, logger: TrainingLogger):
    """保存最终权重到 models/hybrid_predictor/（只保存一份最佳模型）"""
    out_dir = os.path.join(_PROJECT_ROOT, config["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "stage": stage, "epoch": epoch, "loss": loss,
        "model_info": model.get_model_info(), "config": config,
        "best_val_loss": logger.best_val_loss,
        "total_epochs": len(logger.history),
    }
    fpath = os.path.join(out_dir, "phy_ode_diffusion.pt")
    torch.save(ckpt, fpath)
    print(f"  → 最终模型已保存: {fpath}")
    return fpath


# ── 阶段一：编码器 + ODE + GRU ────────────────────────

def train_stage1(model, train_loader, val_loader, config, device, logger: TrainingLogger):
    print("\n" + "=" * 60)
    print("  Stage 1: 训练 Transformer + ODE + GRU（无扩散）")
    print("=" * 60)

    for param in model.diffusion.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr_stage1"], weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs_stage1"])
    best_val_loss = float('inf')

    epoch_iter = range(1, config["epochs_stage1"] + 1)
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
                mse_loss += torch.nn.functional.mse_loss(p_pred, b["tgt_pos"][:, j, :])
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if HAS_TQDM:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train = total_loss / len(train_loader)
        val_loss = _validate_stage1(model, val_loader, device)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.log_epoch(1, epoch, avg_train, val_loss, lr, epoch_time)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        status = f"train={avg_train:.4f} val={val_loss:.4f} lr={lr:.2e}"
        if HAS_TQDM:
            epoch_iter.set_postfix_str(status + (" ★" if is_best else ""))
        else:
            print(f"Epoch {epoch:3d}/{config['epochs_stage1']} | {status} | {epoch_time:.1f}s")

    for param in model.diffusion.parameters():
        param.requires_grad = True


def _validate_stage1(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            b = _to_device(batch, device)
            c = model.transformer(b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"])
            h = model.state_manager.init_state(b["ctx_pos"][:, -1, :],
                                                b["ctx_vel"][:, -1, :], c)
            t_now = b["ctx_t"][:, -1]
            mse, count = 0.0, 0
            for j in range(b["tgt_pos"].shape[1]):
                t_next = b["tgt_t"][:, j]
                dt_step = t_next - t_now
                h_prior = model.state_manager.evolve(h, t_now, t_next)
                mse += torch.nn.functional.mse_loss(h_prior[:, :3], b["tgt_pos"][:, j, :])
                h = model.state_manager.update(h_prior, dt_step,
                                               b["tgt_pos"][:, j, :], b["tgt_vel"][:, j, :])
                t_now = t_next
                count += 1
            total_loss += (mse / max(count, 1)).item()
    model.train()
    return total_loss / len(val_loader)


# ── 阶段二：扩散模型 ──────────────────────────────────

def train_stage2(model, train_loader, val_loader, config, device, logger: TrainingLogger):
    print("\n" + "=" * 60)
    print("  Stage 2: 训练扩散模型（固定 Transformer + ODE + GRU）")
    print("=" * 60)

    # 冻结非扩散模块
    for name, param in model.named_parameters():
        if not name.startswith("diffusion"):
            param.requires_grad = False

    optimizer = optim.AdamW(model.diffusion.parameters(),
                            lr=config["lr_stage2"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs_stage2"])
    best_val_loss = float('inf')

    epoch_iter = range(1, config["epochs_stage2"] + 1)
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
                loss_dict = model.diffusion.compute_loss(
                    h_prior, dt_step, prev_p, b["tgt_pos"][:, j, :])
                diff_loss_total += loss_dict["diff_loss"]
                with torch.no_grad():
                    h = model.state_manager.update(h_prior, dt_step,
                                                   b["tgt_pos"][:, j, :], b["tgt_vel"][:, j, :])
                t_now = b["tgt_t"][:, j]
                count += 1

            loss = diff_loss_total / max(count, 1)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if HAS_TQDM:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train = total_loss / len(train_loader)
        val_loss = _validate_stage2(model, val_loader, device)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.log_epoch(2, epoch, avg_train, val_loss, lr, epoch_time)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        status = f"train={avg_train:.4f} val={val_loss:.4f} lr={lr:.2e}"
        if HAS_TQDM:
            epoch_iter.set_postfix_str(status + (" ★" if is_best else ""))
        else:
            print(f"Epoch {epoch:3d}/{config['epochs_stage2']} | {status} | {epoch_time:.1f}s")

    for param in model.parameters():
        param.requires_grad = True


def _validate_stage2(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            b = _to_device(batch, device)
            c = model.transformer(b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"])
            h = model.state_manager.init_state(b["ctx_pos"][:, -1, :],
                                                b["ctx_vel"][:, -1, :], c)
            t_now = b["ctx_t"][:, -1]
            diff_loss, count = 0.0, 0
            for j in range(b["tgt_pos"].shape[1]):
                dt_step = b["tgt_t"][:, j] - t_now
                h_prior = model.state_manager.evolve(h, t_now, b["tgt_t"][:, j])
                prev_p = b["ctx_pos"][:, -1, :] if j == 0 else b["tgt_pos"][:, j-1, :]
                loss_dict = model.diffusion.compute_loss(
                    h_prior, dt_step, prev_p, b["tgt_pos"][:, j, :])
                diff_loss += loss_dict["diff_loss"]
                h = model.state_manager.update(h_prior, dt_step,
                                               b["tgt_pos"][:, j, :], b["tgt_vel"][:, j, :])
                t_now = b["tgt_t"][:, j]
                count += 1
            total_loss += (diff_loss / max(count, 1)).item()
    model.train()
    return total_loss / len(val_loader)


# ── 阶段三：联合微调 ──────────────────────────────────

def train_stage3(model, train_loader, val_loader, config, device, logger: TrainingLogger):
    print("\n" + "=" * 60)
    print("  Stage 3: 联合微调（计划采样）")
    print("=" * 60)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config["lr_stage3"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs_stage3"])
    best_val_loss = float('inf')

    epoch_iter = range(1, config["epochs_stage3"] + 1)
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
                    p_obs = b["tgt_pos"][:, j, :]
                    v_obs = b["tgt_vel"][:, j, :]
                else:
                    with torch.set_grad_enabled(True):
                        p_gen = model.diffusion.guided_sampling(
                            h_prior, dt_step, last_pos,
                            n_steps=min(10, model.diffusion.n_inference_steps))
                    p_obs = p_gen.detach()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if HAS_TQDM:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train = total_loss / len(train_loader)
        val_loss = _validate_stage2(model, val_loader, device)
        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        logger.log_epoch(3, epoch, avg_train, val_loss, lr, epoch_time,
                         extra={"scheduled_sampling_prob": round(ss_prob, 3)})
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        status = f"train={avg_train:.4f} val={val_loss:.4f} ss={ss_prob:.2f}"
        if HAS_TQDM:
            epoch_iter.set_postfix_str(status + (" ★" if is_best else ""))
        else:
            print(f"Epoch {epoch:3d}/{config['epochs_stage3']} | {status} | lr={lr:.2e} | {epoch_time:.1f}s")

        pass  # 最终模型统一在训练完成后保存


# ── 主入口 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phy-ODE-Diffusion 训练")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["1", "2", "3", "all"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ctx-len", type=int, default=20)
    parser.add_argument("--tgt-len", type=int, default=10)
    parser.add_argument("--no-charts", action="store_true", help="跳过图表生成")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[TQDM] {'可用' if HAS_TQDM else '不可用（pip install tqdm）'}")
    print(f"[图表] {'可用' if HAS_MPL else '不可用（pip install matplotlib）'}")

    # 数据
    train_loader, val_loader = build_dataloaders(config)

    # 模型
    if args.resume:
        print(f"[RESUME] {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model = create_model(config, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.diffusion.scheduler.to(device)
    else:
        model = create_model(config, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] 总参数量: {total_params:,}")

    # 日志器
    logger = TrainingLogger()
    out_dir = os.path.join(_PROJECT_ROOT, config["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    t_total_start = time.time()

    # ── 阶段训练 ──
    if args.stage in ("1", "all"):
        if not args.resume:
            train_stage1(model, train_loader, val_loader, config, device, logger)

    if args.stage in ("2", "all"):
        if args.stage == "2" and not args.resume:
            s1_path = os.path.join(out_dir, "phy_ode_diffusion.pt")
            if os.path.exists(s1_path):
                print(f"[LOAD] 加载阶段一最佳权重: {s1_path}")
                ckpt = torch.load(s1_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.diffusion.scheduler.to(device)
        train_stage2(model, train_loader, val_loader, config, device, logger)

    if args.stage in ("3", "all"):
        train_stage3(model, train_loader, val_loader, config, device, logger)

    total_time = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"  训练完成！总耗时: {total_time/60:.1f} 分钟")
    print(f"{'='*60}")

    # 只保存最终权重到 models/hybrid_predictor/
    final_loss = logger.history[-1]["val_loss"] if logger.history else 0.0
    save_final_model(model, config, 0, 0, final_loss, logger)

    # ── 输出图表和日志到 train/hybrid_predictor/train_result/ ──
    results_dir = os.path.join(_RESULTS_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
    with open(os.path.join(results_dir, "training_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n所有输出已保存到: {results_dir}")
    print("训练完成！")


if __name__ == "__main__":
    main()
