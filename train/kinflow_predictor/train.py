"""
KinFlow 两阶段训练脚本

输出目录: train/kinflow_predictor/train_result/run_<时间戳>/
  - models/kinflow_s{stage}_e{epoch}_v{loss}.pt   阶段检查点
  - models/kinflow_best_s{stage}.pt               最佳检查点（并同步到 train_result/best/）
  - training_history.json                         逐 epoch 训练历史
  - training_summary.json                         训练摘要
  - test_metrics.json                             测试集指标
  - 0*.png                                        训练曲线与评价图表

阶段划分（论文 §3.7）:
  阶段一 联合训练: L = L_flow + λ_c·MSE(X0, Xgt) + λ_p·L_phy
  阶段二 微调:     向走廊先验注入噪声 σ（对齐推理时的输入分布）、更低学习率，
                   可选把可微投影器放进计算图，缩小训练与部署的差距。

用法:
  python train/kinflow_predictor/train.py --stage all --epochs-s1 60 --epochs-s2 20 --device cuda:0
  python train/kinflow_predictor/train.py --smoke            # 小批量冒烟测试
  python train/kinflow_predictor/train.py --stage 2 --resume train/kinflow_predictor/train_result/best/kinflow_best_s1.pt
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

# 路径（pathlib 绝对路径，无 sys.path 注入风险）
_MODULE_DIR = Path(__file__).resolve().parent          # train/kinflow_predictor/
_PROJECT_ROOT = _MODULE_DIR.parent.parent              # drone/
sys.path.insert(0, str(_PROJECT_ROOT))

from train.kinflow_predictor.dataset import (
    load_trajectory_set, KinFlowWindowDataset, kinflow_collate,
)
from trajectory_reconstruction.core.prediction.kinflow import KinFlow

_RESULTS_DIR = _MODULE_DIR / "train_result"

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import matplotlib as _mpl
    _mpl.use('Agg')
    import matplotlib.pyplot as _plt
    _mpl.rcParams['axes.unicode_minus'] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── 配置 ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    # 数据窗口
    "ctx_len": 40,
    "tgt_len": 20,
    "batch_size": 32,
    "num_workers": 4,
    "dataset_dir": "train/kinflow_predictor/dataset",
    "output_dir": "train/kinflow_predictor/train_result/models",
    "val_split": 0.1,
    # 优化
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-4,
    "weight_decay": 1e-4,
    "epochs_stage1": 60,
    "epochs_stage2": 20,
    "warmup_epochs_s1": 5,
    "warmup_epochs_s2": 0,
    "warmup_start_factor": 0.1,
    "grad_clip": 1.0,
    "checkpoint_interval": 10,
    # 损失权重
    "lambda_corridor": 0.5,
    "lambda_physics": 0.01,
    "lambda_projector": 0.1,
    "lambda_sample": 0.3,
    "projector_in_graph": True,
    "sigma_stage2": 0.1,
    # 编码器
    "d_feat": 64,
    "d_context": 128,
    "n_head": 4,
    "n_layers": 4,
    "dim_feedforward": 256,
    "dropout": 0.1,
    # 走廊
    "corridor_hidden": 128,
    # 走廊流
    "d_z": 128,
    "flow_hidden": 128,
    "flow_hidden_layers": 3,
    "tau_emb_dim": 16,
    "time_emb_dim": 16,
    "k_max": 20,
    # 训练窗口名义采样间隔（推理时按此规格重采样历史；0=自动从数据估计）
    "sample_dt": 0.1,
    # 采样
    "n_euler_steps": 4,
    "sigma": 0.0,
    # 物理边界（与 config.json → drone_dynamics / kinflow_model 对齐）
    "v_h_max": 20.0,
    "v_up_max": 5.0,
    "v_down_max": 3.0,
    "y_min": 1.0,
    "y_max": 120.0,
    "max_tilt": 35.0,
    "g": 9.81,
    "n_project_iters": 3,
    "project": True,
    # 测试期概率采样
    "eval_samples": 8,
    "eval_sigma": 0.1,
    "seed": 42,
    # 不规则采样增强: 历史时间戳抖动幅度（以名义 dt 的比例计，0=关闭）
    "time_jitter": 0.35,
}


# ── 工具 ──────────────────────────────────────────────

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


class TrainingLogger:
    """收集逐 epoch 指标，支撑训练曲线与摘要。"""

    def __init__(self):
        self.history: list[dict] = []
        self.stage_labels: dict = {}

    def log_epoch(self, stage: int, epoch: int, metrics: dict,
                  lr: float, epoch_time: float):
        entry = {"stage": stage, "epoch": epoch, "lr": lr,
                 "epoch_time_s": round(epoch_time, 2)}
        entry.update({k: (round(v, 6) if isinstance(v, float) else v)
                      for k, v in metrics.items()})
        self.history.append(entry)
        self.stage_labels[stage] = f"Stage {stage}"

    def save(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "training_history.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"history": self.history, "stage_labels": self.stage_labels},
                      f, indent=2, ensure_ascii=False)
        return path

    def stage_entries(self, stage: int) -> list[dict]:
        return [e for e in self.history if e.get("stage") == stage]

    @property
    def best_val_loss(self) -> float:
        losses = [e["val_loss"] for e in self.history if "val_loss" in e]
        return min(losses) if losses else float('inf')


# ── 图表 ──────────────────────────────────────────────

_COLORS = {
    "train": "#2196F3", "val": "#EF5350", "best": "#4CAF50",
    "s1": "#2196F3", "s2": "#FF9800", "ade": "#7E57C2", "fde": "#26A69A",
}


def _savefig(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    _plt.close(fig)
    print(f"[图表] {name}")


def _plot_stage_losses(logger: TrainingLogger, out_dir: str):
    stages = sorted(logger.stage_labels)
    if not stages:
        return
    fig, axes = _plt.subplots(1, len(stages), figsize=(6 * len(stages), 5), squeeze=False)
    for ax, stage in zip(axes[0], stages):
        entries = logger.stage_entries(stage)
        epochs = [e["epoch"] for e in entries]
        ax.plot(epochs, [e["train_loss"] for e in entries], color=_COLORS["train"],
                lw=2, marker='o', ms=3, label='Train total')
        ax.plot(epochs, [e["val_loss"] for e in entries], color=_COLORS["val"],
                lw=2, marker='s', ms=3, label='Val total')
        if entries and "flow_train" in entries[0]:
            ax.plot(epochs, [e["flow_train"] for e in entries], '--',
                    color=_COLORS["s1"], lw=1.2, label='Flow (train)')
            ax.plot(epochs, [e["corridor_train"] for e in entries], '--',
                    color=_COLORS["s2"], lw=1.2, label='Corridor (train)')
        best_i = int(np.argmin([e["val_loss"] for e in entries])) if entries else 0
        if entries:
            ax.scatter([epochs[best_i]], [entries[best_i]["val_loss"]], s=90,
                       marker='*', color=_COLORS["best"], zorder=5,
                       label='Best val')
        ax.set_title(f'Stage {stage} — Loss', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_dir, "01_stage_loss.png")


def _plot_global_loss(logger: TrainingLogger, out_dir: str):
    if not logger.history:
        return
    xs = list(range(1, len(logger.history) + 1))
    fig, ax = _plt.subplots(figsize=(10, 5))
    ax.plot(xs, [e["train_loss"] for e in logger.history], color=_COLORS["train"],
            lw=2, label='Train total')
    ax.plot(xs, [e["val_loss"] for e in logger.history], color=_COLORS["val"],
            lw=2, label='Val total')
    for stage in sorted(logger.stage_labels):
        idx = [i for i, e in enumerate(logger.history) if e["stage"] == stage]
        if idx:
            ax.axvspan(min(idx) + 0.5, max(idx) + 1.5, alpha=0.06,
                       color=_COLORS.get(f"s{stage}", "gray"))
    ax.set_title('Global Loss Curve', fontsize=13, fontweight='bold')
    ax.set_xlabel('Global epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_dir, "02_global_loss.png")


def _plot_accuracy(logger: TrainingLogger, out_dir: str):
    entries = [(i + 1, e) for i, e in enumerate(logger.history) if "ade" in e]
    if not entries:
        return
    xs = [x for x, _ in entries]
    entries = [e for _, e in entries]
    fig, ax = _plt.subplots(figsize=(10, 5))
    ax.plot(xs, [e["ade"] for e in entries], color=_COLORS["ade"], lw=2,
            marker='o', ms=3, label='ADE (m)')
    ax.plot(xs, [e["fde"] for e in entries], color=_COLORS["fde"], lw=2,
            marker='s', ms=3, label='FDE (m)')
    best = int(np.argmin([e["ade"] for e in entries]))
    ax.annotate(f'{entries[best]["ade"]:.3f} m',
                xy=(xs[best], entries[best]["ade"]), xytext=(10, 12),
                textcoords='offset points', color='darkred', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='darkred'))
    ax.set_title('Validation ADE / FDE', fontsize=13, fontweight='bold')
    ax.set_xlabel('Global epoch'); ax.set_ylabel('Error (m)')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_dir, "03_accuracy.png")


def _plot_physics(logger: TrainingLogger, out_dir: str):
    keys = ["viol_speed", "viol_vertical", "viol_altitude", "viol_accel"]
    labels = ["Speed", "Vertical rate", "Altitude", "Acceleration (soft)"]
    entries = [(i + 1, e) for i, e in enumerate(logger.history) if keys[0] in e]
    if not entries:
        return
    xs = [x for x, _ in entries]
    entries = [e for _, e in entries]
    fig, ax = _plt.subplots(figsize=(10, 5))
    palette = ['#FF7043', '#AB47BC', '#26C6DA', '#66BB6A']
    for key, label, color in zip(keys, labels, palette):
        ax.plot(xs, [e[key] for e in entries], lw=2, marker='o', ms=2.5,
                label=label, color=color)
    ax.set_title('Physical Constraint Violation Rate (validation)', fontsize=13,
                 fontweight='bold')
    ax.set_xlabel('Global epoch'); ax.set_ylabel('Violation rate')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_dir, "04_physics.png")


def _plot_lr(logger: TrainingLogger, out_dir: str):
    if not logger.history:
        return
    xs = list(range(1, len(logger.history) + 1))
    fig, ax = _plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, [e["lr"] for e in logger.history], color='#5C6BC0', lw=2)
    ax.set_yscale('log')
    ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax.set_xlabel('Global epoch'); ax.set_ylabel('LR (log)')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    _savefig(fig, out_dir, "05_lr_schedule.png")


def _plot_dashboard(logger: TrainingLogger, config: dict, out_dir: str,
                    params: dict = None, total_minutes: float = None):
    if not logger.history:
        return
    fig, axes = _plt.subplots(2, 2, figsize=(13, 9))
    xs = list(range(1, len(logger.history) + 1))
    ax = axes[0][0]
    ax.plot(xs, [e["train_loss"] for e in logger.history], color=_COLORS["train"], lw=2, label='Train')
    ax.plot(xs, [e["val_loss"] for e in logger.history], color=_COLORS["val"], lw=2, label='Val')
    ax.set_title('Loss', fontweight='bold'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[0][1]
    acc = [(i + 1, e) for i, e in enumerate(logger.history) if "ade" in e]
    if acc:
        ax.plot([x for x, _ in acc], [e["ade"] for _, e in acc],
                color=_COLORS["ade"], lw=2, label='ADE')
        ax.plot([x for x, _ in acc], [e["fde"] for _, e in acc],
                color=_COLORS["fde"], lw=2, label='FDE')
        ax.set_xlabel('Global epoch')
        ax.legend(fontsize=9)
    ax.set_title('Accuracy (m)', fontweight='bold'); ax.grid(alpha=0.3)

    ax = axes[1][0]
    phy = [e for e in logger.history if "viol_speed" in e]
    if phy:
        for key, label in [("viol_speed", "Speed"), ("viol_altitude", "Altitude")]:
            ax.plot([e["epoch"] for e in phy], [e[key] for e in phy], lw=2, label=label)
        ax.legend(fontsize=9)
    ax.set_title('Hard-constraint violations', fontweight='bold'); ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.axis('off')
    lines = ["KinFlow training summary", ""]
    if params:
        lines.append(f"total params : {params.get('total', 0):,}")
        for part in ("encoder", "corridor", "flow", "projector"):
            lines.append(f"  {part:<9}: {params.get(part, 0):,}")
        lines.append("")
    lines.append(f"stages        : {sorted(logger.stage_labels)}")
    lines.append(f"best val loss : {logger.best_val_loss:.4f}")
    best_ade = min([e["ade"] for e in logger.history if "ade" in e], default=float('nan'))
    lines.append(f"best val ADE  : {best_ade:.4f} m")
    if total_minutes is not None:
        lines.append(f"total time    : {total_minutes:.1f} min")
    lines.append(f"ctx/tgt len   : {config['ctx_len']}/{config['tgt_len']}")
    lines.append(f"euler steps   : {config['n_euler_steps']}")
    ax.text(0.02, 0.96, "\n".join(lines), va='top', fontsize=10,
            family='monospace', transform=ax.transAxes)
    fig.suptitle('KinFlow Training Dashboard', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, out_dir, "06_dashboard.png")


def _plot_test_metrics(metrics: dict, out_dir: str):
    fig, axes = _plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    names = ["ADE", "FDE", "minADE_8"]
    vals = [metrics.get("ade", 0), metrics.get("fde", 0), metrics.get("min_ade", 0)]
    bars = ax.barh(names, vals, color=['#7E57C2', '#26A69A', '#42A5F5'])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f} m', va='center', fontsize=11, fontweight='bold')
    ax.set_title('Test accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Metres')
    ax.set_xlim(0, max(vals) * 1.18 if max(vals) > 0 else 1)

    ax = axes[1]
    keys = ["viol_speed", "viol_vertical", "viol_altitude", "viol_accel"]
    labels = ["Speed", "Vertical", "Altitude", "Accel (soft)"]
    vals = [metrics.get(k, 0) for k in keys]
    bars = ax.bar(labels, vals, color=['#FF7043', '#AB47BC', '#26C6DA', '#66BB6A'])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('Test violation rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('Rate')
    ax.set_ylim(0, max(1.0, max(vals) * 1.2))
    fig.tight_layout()
    _savefig(fig, out_dir, "07_test_metrics.png")


# ── 数据 ──────────────────────────────────────────────

def _limit(dataset, max_windows: int, seed: int):
    """按固定种子截取子集，用于小批量冒烟测试。"""
    if not max_windows or len(dataset) <= max_windows:
        return dataset
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(dataset), size=max_windows, replace=False)).tolist()
    return Subset(dataset, idx)


def build_dataloaders(config: dict, max_windows: int = None,
                      max_val_windows: int = None, log: bool = True):
    """构建训练/验证 DataLoader。优先 train/|valid/ > data/。"""
    seed = int(config.get("seed", 42))
    dataset_root = os.path.join(_PROJECT_ROOT, config["dataset_dir"])
    train_dir = os.path.join(dataset_root, "train")
    valid_dir = os.path.join(dataset_root, "valid")

    train_trajs = load_trajectory_set(synthetic_dir=train_dir) if os.path.isdir(train_dir) else []
    valid_trajs = load_trajectory_set(synthetic_dir=valid_dir) if os.path.isdir(valid_dir) else []

    if not train_trajs:
        data_dir = os.path.join(dataset_root, "data")
        all_trajs = load_trajectory_set(synthetic_dir=data_dir)             if os.path.isdir(data_dir) else []
        if not all_trajs:
            raise RuntimeError(
                "没有可用的训练数据！请将 .npz 轨迹放入 "
                f"{train_dir}，或先生成合成数据: "
                "python train/kinflow_predictor/generate_synthetic.py 200"
            )
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(all_trajs)).tolist()
        n_val = max(1, int(len(all_trajs) * config.get("val_split", 0.15)))
        valid_trajs = [all_trajs[i] for i in order[:n_val]]
        train_trajs = [all_trajs[i] for i in order[n_val:]]
        if log:
            print(f"[数据] data/ 回退 → train {len(train_trajs)} / valid {len(valid_trajs)} 条轨迹")

    if log:
        print(f"[数据] train {len(train_trajs)} / valid {len(valid_trajs)} 条轨迹")

    # 名义采样间隔: 从数据的中位 Δt 估计，供推理时重采样到训练栅格
    if not config.get("sample_dt"):
        dts = []
        for pos, ts in train_trajs[:50]:
            d = np.diff(ts)
            d = d[d > 0]
            if len(d):
                dts.append(float(np.median(d)))
        config["sample_dt"] = round(float(np.median(dts)), 6) if dts else 0.1
        if log:
            print(f"[数据] 名义采样间隔 sample_dt={config['sample_dt']} s")

    train_set = KinFlowWindowDataset(
        train_trajs, ctx_len=config["ctx_len"], tgt_len=config["tgt_len"],
        augment=True, seed=seed, time_jitter=float(config.get("time_jitter", 0.0)),
    )
    val_set = KinFlowWindowDataset(
        valid_trajs if valid_trajs else train_trajs[:max(1, len(train_trajs) // 10)],
        ctx_len=config["ctx_len"], tgt_len=config["tgt_len"], augment=False, seed=seed,
    )

    train_set = _limit(train_set, max_windows, seed)
    val_set = _limit(val_set, max_val_windows, seed + 1)

    workers = int(config.get("num_workers", 0))
    pin = torch.cuda.is_available() and str(config.get("device", "cpu")).startswith("cuda")

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True,
        collate_fn=kinflow_collate, drop_last=True,
        num_workers=workers, pin_memory=pin,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False,
        collate_fn=kinflow_collate, drop_last=False,
        num_workers=workers, pin_memory=pin,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader


def build_test_loader(config: dict, max_windows: int = None):
    """构建测试集 DataLoader；目录不存在或为空时返回 None。"""
    test_dir = os.path.join(_PROJECT_ROOT, config["dataset_dir"], "test")
    if not os.path.isdir(test_dir):
        print("[测试] test/ 不存在，跳过测试集评价")
        return None
    trajs = load_trajectory_set(synthetic_dir=test_dir)
    if not trajs:
        print("[测试] test/ 为空，跳过测试集评价")
        return None
    test_set = KinFlowWindowDataset(
        trajs, ctx_len=config["ctx_len"], tgt_len=config["tgt_len"], augment=False,
    )
    test_set = _limit(test_set, max_windows, int(config.get("seed", 42)) + 2)
    print(f"[测试] {len(trajs)} 条轨迹 → {len(test_set)} 个评估窗口")
    workers = int(config.get("num_workers", 0))
    return DataLoader(test_set, batch_size=config["batch_size"], shuffle=False,
                      collate_fn=kinflow_collate, drop_last=False, num_workers=workers)


# ── 模型与检查点 ──────────────────────────────────────

def create_model(config: dict, device: torch.device) -> KinFlow:
    """按配置构建 KinFlow 并移到设备。"""
    model = KinFlow(
        d_feat=config["d_feat"], d_context=config["d_context"],
        n_head=config["n_head"], n_layers=config["n_layers"],
        dim_feedforward=config["dim_feedforward"], dropout=config["dropout"],
        corridor_hidden=config["corridor_hidden"],
        d_z=config["d_z"], flow_hidden=config["flow_hidden"],
        flow_hidden_layers=config["flow_hidden_layers"],
        tau_emb_dim=config["tau_emb_dim"], time_emb_dim=config["time_emb_dim"],
        k_max=config["k_max"],
        ctx_len=config["ctx_len"], sample_dt=config.get("sample_dt", 0.1),
        n_euler_steps=config["n_euler_steps"], sigma=config["sigma"],
        v_h_max=config["v_h_max"], v_up_max=config["v_up_max"],
        v_down_max=config["v_down_max"], y_min=config["y_min"], y_max=config["y_max"],
        max_tilt=config["max_tilt"], g=config["g"],
        n_project_iters=config["n_project_iters"], project=config["project"],
    )
    return model.to(device)


def save_checkpoint(model: KinFlow, out_dir: str, stage: int, epoch: int,
                    val_loss: float, is_best: bool = False, metrics: dict = None,
                    optimizer=None, scheduler=None):
    """保存检查点；命名含 stage/epoch/loss，最佳模型单独保存并集中汇总。"""
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_info": model.get_model_info(),
        "stage": stage,
        "epoch": epoch,
        "val_loss": float(val_loss),
        "format": "kinflow-v1",
    }
    if metrics:
        ckpt["metrics"] = metrics
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    path = os.path.join(out_dir, f"kinflow_s{stage}_e{epoch}_v{val_loss:.4f}.pt")
    torch.save(ckpt, path)

    if is_best:
        best_name = f"kinflow_best_s{stage}.pt"
        best_path = os.path.join(out_dir, best_name)
        torch.save(ckpt, best_path)

        # 集中目录 best/ 只接受"更优"的权重，避免冒烟/短训练覆盖已部署的好模型
        central = os.path.join(str(_RESULTS_DIR), "best")
        os.makedirs(central, exist_ok=True)
        central_path = os.path.join(central, best_name)
        replace = True
        if os.path.isfile(central_path):
            try:
                prev = torch.load(central_path, map_location='cpu', weights_only=False)
                prev_loss = float(prev.get('val_loss', float('inf')))
                replace = float(val_loss) < prev_loss
            except Exception:
                replace = True
        if replace:
            shutil.copy2(best_path, central_path)
            print(f"    -> 最佳模型: {best_path} (同步到 {central})")
        else:
            print(f"    -> 最佳模型: {best_path}（未超过集中目录已有权重，保留旧的）")
    return path


def build_scheduler(optimizer, total_epochs: int, warmup_epochs: int,
                    start_factor: float = 0.1):
    """warmup 线性升温 + 余弦退火（论文 §3.7）。"""
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return start_factor + (1.0 - start_factor) * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda)


# ── 训练与评价 ────────────────────────────────────────

def _batch_loss(model: KinFlow, batch: dict, config: dict,
                sigma: float, use_projector: bool) -> dict:
    out = model.forward_train(
        batch["ctx_t"], batch["ctx_dt"], batch["ctx_pos"], batch["ctx_vel"],
        batch["tgt_t"], batch["tgt_pos"], batch["p_mean"], batch["p_std"],
        sigma=sigma,
        lambda_corridor=config["lambda_corridor"],
        lambda_physics=config["lambda_physics"],
        use_projector=use_projector,
        lambda_projector=config["lambda_projector"],
        lambda_sample=config.get("lambda_sample", 0.0),
    )
    return out


def train_epoch(model: KinFlow, loader, optimizer, config: dict,
                device: torch.device, stage: int) -> dict:
    """一个 epoch 的训练；返回按样本数加权的平均损失。"""
    model.train()
    sigma = config["sigma_stage2"] if stage == 2 else 0.0
    use_projector = bool(config["projector_in_graph"]) and stage == 2

    totals = {"loss": 0.0, "flow": 0.0, "corridor": 0.0, "physics": 0.0,
              "projector": 0.0, "sample": 0.0, "n": 0}
    iterator = tqdm(loader, desc=f"Stage {stage}", ncols=110, leave=False) if HAS_TQDM else loader

    for batch in iterator:
        b = _to_device(batch, device)
        out = _batch_loss(model, b, config, sigma, use_projector)
        loss = out["total_loss"]

        if not torch.isfinite(loss):
            print("    [WARN] 非有限损失，跳过该批次")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()

        n = b["ctx_t"].shape[0]
        totals["n"] += n
        totals["loss"] += float(loss.detach()) * n
        totals["flow"] += float(out["flow_loss"].detach()) * n
        totals["corridor"] += float(out["corridor_loss"].detach()) * n
        totals["physics"] += float(out["physics_loss"].detach()) * n
        totals["projector"] += float(out["projector_loss"].detach()) * n
        totals["sample"] += float(out.get("sample_loss", 0.0).detach()) * n

        if HAS_TQDM:
            iterator.set_postfix(loss=f'{totals["loss"] / max(totals["n"], 1):.4f}')

    n = max(totals["n"], 1)
    return {k: totals[k] / n for k in ("loss", "flow", "corridor", "physics", "projector", "sample")}


@torch.no_grad()
def evaluate(model: KinFlow, loader, config: dict, device: torch.device,
             n_samples: int = 1, sigma: float = None) -> dict:
    """
    Val/Test 评价: 损失、ADE/FDE（物理坐标，米）、归一化 MSE、约束违反率。
    开启 n_samples>1 时额外给出 minADE（概率采样的 best-of-K）。
    """
    model.eval()
    sigma = config["sigma"] if sigma is None else sigma
    sums = {"loss": 0.0, "flow": 0.0, "corridor": 0.0, "physics": 0.0,
            "ade": 0.0, "fde": 0.0, "mse": 0.0, "min_ade": 0.0, "n": 0,
            "windows": 0}
    viol = {"speed": 0.0, "vertical": 0.0, "altitude": 0.0, "accel": 0.0}

    for batch in loader:
        b = _to_device(batch, device)
        out = _batch_loss(model, b, config, sigma=0.0, use_projector=False)
        n = b["ctx_t"].shape[0]
        sums["loss"] += float(out["total_loss"]) * n
        sums["flow"] += float(out["flow_loss"]) * n
        sums["corridor"] += float(out["corridor_loss"]) * n
        sums["physics"] += float(out["physics_loss"]) * n
        sums["n"] += n
        sums["mse"] += float(torch.mean((out["x0"] - b["tgt_pos"]) ** 2)) * n

        dtau = b["tgt_t"] - b["ctx_t"][:, -1:]
        fc = model.forecast(
            b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"], dtau,
            b["p_mean"], b["p_std"], n_steps=config["n_euler_steps"],
            sigma=sigma, n_samples=n_samples, project=config["project"],
        )
        x_phys = fc["x_phys"]
        gt = b["tgt_pos"] * b["p_std"].unsqueeze(1) + b["p_mean"].unsqueeze(1)

        if x_phys.dim() == 4:      # (S, B, K, 3)
            err = torch.linalg.vector_norm(x_phys - gt.unsqueeze(0), dim=-1)  # (S,B,K)
            per_sample = err.mean(dim=-1)                                     # (S,B)
            ade_sample = per_sample.mean(dim=-1)                              # (S,)
            sums["ade"] += float(ade_sample.mean()) * n
            sums["fde"] += float(err[..., -1].mean()) * n
            sums["min_ade"] += float(per_sample.min(dim=0).values.mean()) * n
        else:
            err = torch.linalg.vector_norm(x_phys - gt, dim=-1)               # (B,K)
            sums["ade"] += float(err.mean()) * n
            sums["fde"] += float(err[:, -1].mean()) * n

        anchor_phys = b["ctx_pos"][:, -1, :] * b["p_std"] + b["p_mean"]
        anchor_vel_phys = b["ctx_vel"][:, -1, :] * b["p_std"]
        vrates = model.projector.violation_rates(
            x_phys if x_phys.dim() == 3 else x_phys[0],
            anchor_phys, dtau, anchor_vel=anchor_vel_phys,
        )
        for k in viol:
            viol[k] += vrates[k] * n
        sums["windows"] += n

    n = max(sums["n"], 1)
    result = {"val_loss": sums["loss"] / n, "flow_loss": sums["flow"] / n,
              "corridor_loss": sums["corridor"] / n,
              "physics_loss": sums["physics"] / n,
              "ade": sums["ade"] / n, "fde": sums["fde"] / n,
              "mse_norm_corridor": sums["mse"] / n}
    if n_samples > 1:
        result["min_ade"] = sums["min_ade"] / n
    for k in viol:
        result[f"viol_{k}"] = viol[k] / n
    return result


def _looks_better(current: float, best: float) -> bool:
    return bool(np.isfinite(current) and current < best)


def run_stage(model: KinFlow, train_loader, val_loader, config: dict,
              device: torch.device, logger: TrainingLogger, stage: int,
              out_dir: str) -> dict:
    """跑完一个训练阶段，返回该阶段最佳验证指标。"""
    epochs = int(config["epochs_stage1"] if stage == 1 else config["epochs_stage2"])
    lr = float(config["lr_stage1"] if stage == 1 else config["lr_stage2"])
    warmup = int(config["warmup_epochs_s1"] if stage == 1 else config["warmup_epochs_s2"])

    optimizer = optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=config["weight_decay"])
    scheduler = build_scheduler(optimizer, epochs, warmup,
                                config["warmup_start_factor"])

    best_loss = float("inf")
    best_metrics: dict = {}
    interval = int(config.get("checkpoint_interval", 10) or 0)

    print(f"\n{'=' * 62}")
    print(f"  阶段 {stage}: {epochs} epochs, lr={lr}, warmup={warmup}, "
          f"batch={config['batch_size']}, "
          f"sigma={config['sigma_stage2'] if stage == 2 else 0.0}")
    print(f"{'=' * 62}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_epoch(model, train_loader, optimizer, config, device, stage)
        val = evaluate(model, val_loader, config, device)
        epoch_time = time.time() - t0

        metrics = {
            "train_loss": train_stats["loss"],
            "val_loss": val["val_loss"],
            "flow_train": train_stats["flow"],
            "corridor_train": train_stats["corridor"],
            "physics_train": train_stats["physics"],
            "projector_train": train_stats["projector"],
            "ade": val["ade"], "fde": val["fde"],
            "viol_speed": val["viol_speed"], "viol_vertical": val["viol_vertical"],
            "viol_altitude": val["viol_altitude"], "viol_accel": val["viol_accel"],
        }
        logger.log_epoch(stage, epoch, metrics, scheduler.get_last_lr()[0], epoch_time)
        scheduler.step()

        is_best = _looks_better(val["val_loss"], best_loss)
        if is_best:
            best_loss = val["val_loss"]
            best_metrics = dict(val)

        # 保存策略: 最佳必存；其余按里程碑间隔；最后一个 epoch 必存
        keep = (is_best or epoch == epochs
                or (interval > 0 and epoch % interval == 0))
        if keep:
            save_checkpoint(model, out_dir, stage, epoch, val["val_loss"],
                            is_best=is_best, metrics=val, optimizer=optimizer,
                            scheduler=scheduler)

        print(f"[S{stage}][{epoch:3d}/{epochs}] "
              f"loss {train_stats['loss']:.4f} (flow {train_stats['flow']:.4f} "
              f"corr {train_stats['corridor']:.4f} phy {train_stats['physics']:.4f}) | "
              f"val {val['val_loss']:.4f} | ADE {val['ade']:.3f} FDE {val['fde']:.3f} | "
              f"viol s/v/a {val['viol_speed']:.3f}/{val['viol_vertical']:.3f}/"
              f"{val['viol_altitude']:.3f} | {epoch_time:.1f}s"
              + ("  *best*" if is_best else ""))

    print(f"[S{stage}] 完成，最佳验证损失 {best_loss:.4f}")
    return best_metrics


@torch.no_grad()
def evaluate_test(model: KinFlow, loader, config: dict, device: torch.device,
                  out_dir: str) -> dict:
    """测试集评价: 精度 + 物理约束 + 概率采样 minADE + 推理延迟。"""
    print(f"\n{'=' * 62}")
    print("  测试集评价")
    print(f"{'=' * 62}")

    metrics = evaluate(model, loader, config, device)
    print(f"[测试] ADE {metrics['ade']:.4f} m | FDE {metrics['fde']:.4f} m | "
          f"val_loss {metrics['val_loss']:.4f}")

    # 概率采样: sigma>0 的 best-of-K
    n_samples = int(config.get("eval_samples", 8))
    if n_samples > 1:
        prob = evaluate(model, loader, config, device, n_samples=n_samples,
                        sigma=config.get("eval_sigma", 0.1))
        metrics["min_ade"] = prob.get("min_ade", float("nan"))
        metrics["eval_samples"] = n_samples
        metrics["eval_sigma"] = config.get("eval_sigma", 0.1)
        print(f"[测试] minADE_{n_samples} {metrics['min_ade']:.4f} m "
              f"(sigma={metrics['eval_sigma']})")

    # 推理延迟（逐窗口，确定性采样）
    model.eval()
    n_windows, elapsed = 0, 0.0
    for batch in loader:
        b = _to_device(batch, device)
        dtau = b["tgt_t"] - b["ctx_t"][:, -1:]
        t0 = time.perf_counter()
        fc = model.forecast(
            b["ctx_t"], b["ctx_dt"], b["ctx_pos"], b["ctx_vel"], dtau,
            b["p_mean"], b["p_std"], n_steps=config["n_euler_steps"],
            sigma=0.0, project=config["project"],
        )
        _ = fc["x_phys"]
        elapsed += time.perf_counter() - t0
        n_windows += b["ctx_t"].shape[0]
    metrics["latency_ms_per_window"] = 1000.0 * elapsed / max(n_windows, 1)
    metrics["windows"] = n_windows
    print(f"[测试] 推理 {metrics['latency_ms_per_window']:.2f} ms/窗口 "
          f"(euler={config['n_euler_steps']} 步, 共 {n_windows} 窗口)")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if HAS_MPL:
        _plot_test_metrics(metrics, out_dir)
    return metrics


# ── 命令行 ────────────────────────────────────────────

def _apply_smoke(config: dict) -> dict:
    """小批量冒烟配置: 2+1 epoch、512 训练窗口、CPU、单进程。"""
    config.update({
        "epochs_stage1": 2,
        "epochs_stage2": 1,
        "batch_size": 32,
        "num_workers": 0,
        "device": "cpu",
        "checkpoint_interval": 1,
    })
    return config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KinFlow 无人机轨迹预测模型训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", type=str, default="all",
                        choices=["1", "2", "all"],
                        help="1=仅阶段一, 2=阶段一+阶段二, all=同 2")
    parser.add_argument("--epochs", type=int, default=None, help="统一设置两阶段轮数")
    parser.add_argument("--epochs-s1", type=int, default=None, help="阶段一轮数")
    parser.add_argument("--epochs-s2", type=int, default=None, help="阶段二轮数")
    parser.add_argument("--batch", type=int, default=None, help="批次大小")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader 进程数")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda:0")
    parser.add_argument("--lr-s1", type=float, default=None, help="阶段一学习率")
    parser.add_argument("--lr-s2", type=float, default=None, help="阶段二学习率")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--ctx-len", type=int, default=None, help="历史窗口长度 C")
    parser.add_argument("--tgt-len", type=int, default=None, help="预测窗口长度 K")
    parser.add_argument("--euler-steps", type=int, default=None, help="Euler 采样步数")
    parser.add_argument("--sigma", type=float, default=None, help="阶段二走廊噪声 σ")
    parser.add_argument("--max-windows", type=int, default=None, help="训练窗口上限（冒烟用）")
    parser.add_argument("--max-val-windows", type=int, default=None, help="验证窗口上限")
    parser.add_argument("--max-test-windows", type=int, default=None, help="测试窗口上限")
    parser.add_argument("--dataset-dir", type=str, default=None, help="数据集目录（相对项目根）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认带时间戳）")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="里程碑检查点间隔（epoch）")
    parser.add_argument("--no-projector-in-graph", action="store_true",
                        help="阶段二不把投影器放进计算图")
    parser.add_argument("--no-charts", action="store_true", help="跳过图表生成")
    parser.add_argument("--charts-only", type=str, default=None, metavar="DIR",
                        help="仅从已有 training_history.json 重新生成图表（不训练）")
    parser.add_argument("--smoke", action="store_true",
                        help="小批量冒烟测试（CPU、2+1 epoch、512 窗口）")
    return parser


def _regenerate_charts(target: str) -> None:
    """从已有训练历史重新生成全部图表（--charts-only）。"""
    if not HAS_MPL:
        print("[错误] matplotlib 未安装，无法生成图表")
        sys.exit(1)

    json_path = os.path.join(target, "training_history.json") if os.path.isdir(target) else target
    if not os.path.isfile(json_path):
        print(f"[错误] 找不到训练历史文件: {json_path}")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(json_path))
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger = TrainingLogger()
    logger.history = data.get("history", [])
    logger.stage_labels = data.get("stage_labels", {})
    if not logger.history:
        print("[错误] 训练历史为空")
        sys.exit(1)

    print(f"[图表] 从 {json_path} 重新生成图表 → {out_dir}")
    _plot_stage_losses(logger, out_dir)
    _plot_global_loss(logger, out_dir)
    _plot_accuracy(logger, out_dir)
    _plot_physics(logger, out_dir)
    _plot_lr(logger, out_dir)

    params, total_minutes, config = None, None, DEFAULT_CONFIG
    summary_path = os.path.join(out_dir, "training_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        params = summary.get("params")
        total_minutes = summary.get("total_minutes")
        config = summary.get("config", config)
    _plot_dashboard(logger, config, out_dir, params, total_minutes)

    test_path = os.path.join(out_dir, "test_metrics.json")
    if os.path.isfile(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            _plot_test_metrics(json.load(f), out_dir)
    print("图表生成完成！")


def main():
    args = _build_arg_parser().parse_args()

    if args.charts_only:
        _regenerate_charts(args.charts_only)
        return

    # 配置: 默认值 ← config.json training ← CLI
    config = DEFAULT_CONFIG.copy()
    try:
        from trajectory_reconstruction.core.config.config_manager import ensure_config
        user_cfg = ensure_config().get("training", {})
        applied = 0
        for key in DEFAULT_CONFIG:
            if key in user_cfg:
                config[key] = user_cfg[key]
                applied += 1
        if applied:
            print(f"[CONFIG] 已从 config.json → training 读取 {applied} 个参数")
    except Exception as exc:
        print(f"[WARN] 读取 config.json 失败: {exc}")

    if args.smoke:
        _apply_smoke(config)
        config["max_windows"] = 512
        config["max_val_windows"] = 256
        config["max_test_windows"] = 512
        print("[SMOKE] 小批量冒烟模式: 2(阶段一)+1(阶段二) epoch, 512 训练窗口, CPU")

    if args.max_windows is not None:
        config["max_windows"] = args.max_windows
    if args.max_val_windows is not None:
        config["max_val_windows"] = args.max_val_windows
    if args.max_test_windows is not None:
        config["max_test_windows"] = args.max_test_windows
    if args.ctx_len is not None:
        config["ctx_len"] = args.ctx_len
    if args.tgt_len is not None:
        config["tgt_len"] = args.tgt_len
        config["k_max"] = args.tgt_len
    if args.batch is not None:
        config["batch_size"] = args.batch
    if args.workers is not None:
        config["num_workers"] = args.workers
    if args.seed is not None:
        config["seed"] = args.seed
    if args.euler_steps is not None:
        config["n_euler_steps"] = args.euler_steps
    if args.sigma is not None:
        config["sigma_stage2"] = args.sigma
    if args.epochs is not None:
        config["epochs_stage1"] = args.epochs
        config["epochs_stage2"] = args.epochs
    if args.epochs_s1 is not None:
        config["epochs_stage1"] = args.epochs_s1
    if args.epochs_s2 is not None:
        config["epochs_stage2"] = args.epochs_s2
    if args.lr_s1 is not None:
        config["lr_stage1"] = args.lr_s1
    if args.lr_s2 is not None:
        config["lr_stage2"] = args.lr_s2
    if args.checkpoint_interval is not None:
        config["checkpoint_interval"] = args.checkpoint_interval
    if args.dataset_dir is not None:
        config["dataset_dir"] = args.dataset_dir
    if args.no_projector_in_graph:
        config["projector_in_graph"] = False

    # 设备
    if args.device is not None:
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["device"] = str(device)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 输出目录
    if args.output_dir:
        run_dir = args.output_dir if os.path.isabs(args.output_dir)             else os.path.join(str(_PROJECT_ROOT), args.output_dir)
    else:
        run_dir = os.path.join(str(_RESULTS_DIR), datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    out_dir = os.path.join(run_dir, "models")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[输出] {run_dir}")
    print(f"[DEVICE] {device}")

    # —— 数据 ——
    train_loader, val_loader = build_dataloaders(
        config, config.get("max_windows"), config.get("max_val_windows"),
    )
    print(f"[数据] 训练批次 {len(train_loader)} / 验证批次 {len(val_loader)}")

    # —— 模型 ——
    model = create_model(config, device)
    params = model.count_parameters()
    print(f"[MODEL] 参数量 {params['total']:,} "
          f"(encoder {params['encoder']:,} + corridor {params['corridor']:,} "
          f"+ flow {params['flow']:,} + projector {params['projector']:,})")

    logger = TrainingLogger()
    resumed_stage = 0
    if args.resume:
        print(f"[RESUME] {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        resumed_stage = int(ckpt.get("stage", 0))
        print(f"[RESUME] stage={resumed_stage}, epoch={ckpt.get('epoch')}, "
              f"val_loss={ckpt.get('val_loss')}")

    # 自动加载前一阶段最佳模型（--stage 2 时无需重训阶段一）
    if not args.resume and args.stage in ("2", "all"):
        for stage in (1,):
            best_path = os.path.join(str(_RESULTS_DIR), "best", f"kinflow_best_s{stage}.pt")
            if args.stage == "2" and os.path.isfile(best_path):
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"[AUTO-LOAD] {best_path} (val_loss={ckpt.get('val_loss'):.4f})")
                resumed_stage = max(resumed_stage, stage)
                break

    t_start = time.time()
    run_s1 = args.stage in ("1", "2", "all") and resumed_stage < 1
    run_s2 = args.stage in ("2", "all") and resumed_stage < 2

    stage_metrics = {}
    if run_s1:
        stage_metrics[1] = run_stage(model, train_loader, val_loader, config,
                                     device, logger, 1, out_dir)
        logger.save(run_dir)
        if HAS_MPL and not args.no_charts:
            _plot_stage_losses(logger, run_dir)

    if run_s2:
        stage_metrics[2] = run_stage(model, train_loader, val_loader, config,
                                     device, logger, 2, out_dir)
        logger.save(run_dir)

    # —— 汇总图表与摘要 ——
    total_minutes = (time.time() - t_start) / 60.0
    history_path = logger.save(run_dir)
    print(f"[日志] 训练历史: {history_path}")

    if HAS_MPL and not args.no_charts and logger.history:
        _plot_stage_losses(logger, run_dir)
        _plot_global_loss(logger, run_dir)
        _plot_accuracy(logger, run_dir)
        _plot_physics(logger, run_dir)
        _plot_lr(logger, run_dir)
        _plot_dashboard(logger, config, run_dir, params, total_minutes)

    # —— 测试集评价（用最终阶段最佳检查点）——
    test_loader = build_test_loader(config, config.get("max_test_windows"))
    test_metrics = None
    if test_loader is not None:
        best_path = None
        for stage in (2, 1):
            candidate = os.path.join(out_dir, f"kinflow_best_s{stage}.pt")
            if os.path.isfile(candidate):
                best_path = candidate
                break
        if best_path is None:
            candidates = sorted(
                [f for f in os.listdir(out_dir) if f.startswith("kinflow_s") and f.endswith(".pt")],
                key=lambda f: float(f.rsplit("_v", 1)[-1].rstrip(".pt")),
            )
            if candidates:
                best_path = os.path.join(out_dir, candidates[0])

        if best_path:
            print(f"[测试] 载入最佳检查点: {best_path}")
            test_model = create_model(config, device)
            test_model.load_state_dict(
                torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"]
            )
            test_metrics = evaluate_test(test_model, test_loader, config, device, run_dir)
        else:
            print("[测试] 未找到检查点，跳过测试")
    else:
        print("[测试] 无测试集，跳过")

    # —— 训练摘要 ——
    summary = {
        "model": "KinFlow",
        "paper": "APIN-D-26-14134",
        "run_dir": run_dir,
        "device": str(device),
        "config": {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
        "params": params,
        "stages": {str(k): {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                            for kk, vv in v.items()}
                   for k, v in stage_metrics.items()},
        "best_val_loss": logger.best_val_loss,
        "test_metrics": test_metrics,
        "total_minutes": round(total_minutes, 2),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "training_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[日志] 训练摘要: {os.path.join(run_dir, 'training_summary.json')}")

    print(f"\n{'=' * 62}")
    print(f"  训练完成！总耗时 {total_minutes:.1f} 分钟")
    print(f"  产物目录: {run_dir}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
