"""
YOLO 无人机检测模型 — 训练脚本（含图表输出和进度条）

位于 train/yolotrain/ 目录，独立于主项目代码。
训练结果:
  - 模型权重: train/yolotrain/train_result/models/drone_detect_YYYYMMDD_HHMMSS.pt
  - ultralytics 输出: train/yolotrain/train_result/models/drone_detect/
  - 图表日志: train/yolotrain/train_result/

======== 快速开始 ========

  python train/yolotrain/train.py \
      --data train/yolotrain/dataset/data.yaml \
      --model yolov8n.pt \
      --epochs 100 --imgsz 960

      对于 1280×720 图片，推荐 --imgsz 960 或 1280。
      小目标检测建议 960，追求极致精度用 1280（显存消耗更大）。

======== 数据标注格式 (YOLO) ========

每张图片对应一个同名的 .txt 文件，每行一个目标:
  class_id x_center y_center width height
所有值归一化到 [0, 1]（相对于图片宽高）。

推荐标注工具: Roboflow / LabelImg / CVAT

======== 推荐公开数据集 ========

- DUT-Anti-UAV   — 红外无人机
- UAVDT          — 城市无人机
- Drone-vs-Bird  — 无人机 vs 飞鸟
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

# 项目路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_result")


def _plot_training_results(results_dir: str, output_dir: str):
    """
    从 ultralytics 输出的 results.csv 生成多种评价指标图表。
    图表类型:
      1. 损失分解曲线 (box/cls/dfl + val)
      2. 验证指标曲线 (mAP/Precision/Recall/F1)
      3. 收敛性分析 (改进率/过拟合检测)
      4. 综合仪表盘
    """
    import csv

    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.isfile(csv_path):
        print("[跳过] 未找到 results.csv")
        return

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): float(v.strip()) for k, v in row.items()})
    if not rows:
        return

    epochs = list(range(1, len(rows) + 1))
    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 色板
        c_loss = ['#2196F3', '#FF9800', '#9C27B0']
        c_metric = ['#4CAF50', '#FF5722', '#2196F3', '#FF9800']

        # ── 图1: 损失分解 ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('YOLO Training — Loss Decomposition', fontsize=14, fontweight='bold')

        loss_panels = [
            ('train/box_loss', 'Box Loss (Train)', axes[0, 0], c_loss[0]),
            ('train/cls_loss', 'Class Loss (Train)', axes[0, 1], c_loss[1]),
            ('train/dfl_loss', 'DFL Loss (Train)', axes[1, 0], c_loss[2]),
            ('val/box_loss', 'Box Loss (Validation)', axes[1, 1], '#EF5350'),
        ]
        for key, title, ax, color in loss_panels:
            if key in rows[0]:
                vals = [r[key] for r in rows]
                ax.plot(epochs, vals, '-', color=color, linewidth=1.8, alpha=0.85)
                ax.fill_between(epochs, 0, vals, alpha=0.08, color=color)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                # 标注最优
                best_i = vals.index(min(vals))
                ax.annotate(f'{vals[best_i]:.4f}', xy=(best_i+1, vals[best_i]),
                            fontsize=9, color='darkred', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "01_loss_decomposition.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[图表1] 损失分解: {output_dir}/01_loss_decomposition.png")

        # ── 图2: 验证指标 ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('YOLO Validation Metrics', fontsize=14, fontweight='bold')

        metric_panels = [
            ('metrics/precision(B)', 'Precision', axes[0, 0], c_metric[0]),
            ('metrics/recall(B)', 'Recall', axes[0, 1], c_metric[1]),
            ('metrics/mAP50(B)', 'mAP@0.5', axes[1, 0], c_metric[2]),
            ('metrics/mAP50-95(B)', 'mAP@0.5:0.95', axes[1, 1], c_metric[3]),
        ]
        for key, title, ax, color in metric_panels:
            if key in rows[0]:
                vals = [r[key] for r in rows]
                ax.plot(epochs, vals, '-', color=color, linewidth=1.8, alpha=0.85)
                ax.fill_between(epochs, 0, vals, alpha=0.08, color=color)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                best_i = vals.index(max(vals))
                ax.annotate(f'Best: {vals[best_i]:.4f} (e{best_i+1})',
                            xy=(best_i+1, vals[best_i]),
                            xytext=(10, -15), textcoords='offset points',
                            fontsize=8, color='darkred',
                            arrowprops=dict(arrowstyle='->', lw=0.8, color='darkred'))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "02_validation_metrics.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[图表2] 验证指标: {output_dir}/02_validation_metrics.png")

        # ── 图3: 收敛性分析 ──
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('YOLO — Convergence Analysis', fontsize=14, fontweight='bold')

        # 3a. 损失下降百分比
        ax = axes[0, 0]
        for key, label, color in [('train/box_loss', 'Box Loss', c_loss[0]),
                                   ('train/cls_loss', 'Cls Loss', c_loss[1]),
                                   ('train/dfl_loss', 'DFL Loss', c_loss[2])]:
            if key in rows[0]:
                vals = [r[key] for r in rows]
                improvements = [(vals[0]-v)/vals[0]*100 for v in vals]
                ax.plot(epochs, improvements, '-', color=color, linewidth=1.5, label=label)
        ax.set_title('Loss Improvement (%)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Improvement %')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # 3b. 过拟合检测
        ax = axes[0, 1]
        if 'train/box_loss' in rows[0] and 'val/box_loss' in rows[0]:
            train_v = [r['train/box_loss'] for r in rows]
            val_v = [r['val/box_loss'] for r in rows]
            ax.plot(epochs, train_v, '-', color='#2196F3', linewidth=1.2, alpha=0.7, label='Train')
            ax.plot(epochs, val_v, '-', color='#EF5350', linewidth=1.2, alpha=0.7, label='Val')
            ax.fill_between(epochs, train_v, val_v, alpha=0.1, color='gray')
            ax.set_title('Overfitting Detection (Box Loss)')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # 3c. mAP 提升率
        ax = axes[0, 2]
        for key, label, color in [('metrics/mAP50(B)', 'mAP@0.5', c_metric[2]),
                                   ('metrics/mAP50-95(B)', 'mAP@0.5:0.95', c_metric[3])]:
            if key in rows[0]:
                vals = [r[key] for r in rows]
                ax.plot(epochs, vals, '-', color=color, linewidth=1.8, label=label)
        ax.set_title('mAP Progression')
        ax.set_xlabel('Epoch'); ax.set_ylabel('mAP')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # 3d. Precision-Recall 平衡
        ax = axes[1, 0]
        if 'metrics/precision(B)' in rows[0] and 'metrics/recall(B)' in rows[0]:
            p_vals = [r['metrics/precision(B)'] for r in rows]
            r_vals = [r['metrics/recall(B)'] for r in rows]
            ax.plot(epochs, p_vals, '-', color='#4CAF50', linewidth=1.5, label='Precision')
            ax.plot(epochs, r_vals, '-', color='#FF5722', linewidth=1.5, label='Recall')
            # F1 估计
            f1_vals = [2*p*r/(p+r) if (p+r)>0 else 0 for p, r in zip(p_vals, r_vals)]
            ax.plot(epochs, f1_vals, '--', color='#7E57C2', linewidth=1.5, label='F1 (est.)')
            ax.set_title('Precision / Recall / F1 Balance')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # 3e. 指标稳定性（变异系数）
        ax = axes[1, 1]
        for key, label, color in [('metrics/mAP50(B)', 'mAP@0.5', c_metric[2]),
                                   ('metrics/precision(B)', 'Precision', c_metric[0])]:
            if key in rows[0]:
                vals = [r[key] for r in rows]
                window = max(5, len(vals)//5)
                rolling_std = [np.std(vals[max(0,i-window):i+1]) for i in range(len(vals))]
                ax.plot(epochs, rolling_std, '-', color=color, linewidth=1.2, alpha=0.8, label=f'{label} σ')
        ax.set_title('Metric Stability (Rolling Std Dev)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Std Dev')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # 3f. 学习曲线效率
        ax = axes[1, 2]
        if 'metrics/mAP50(B)' in rows[0]:
            mAP_vals = [r['metrics/mAP50(B)'] for r in rows]
            # 每个 epoch 相对最终 mAP 的完成度
            final_map = mAP_vals[-1]
            if final_map > 0:
                completion = [v/final_map*100 for v in mAP_vals]
                ax.fill_between(epochs, 0, completion, alpha=0.3, color='#4CAF50')
                ax.plot(epochs, completion, '-', color='#2E7D32', linewidth=2)
                ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%')
                ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95%')
            ax.set_title('mAP Learning Efficiency (% of final)')
            ax.set_xlabel('Epoch'); ax.set_ylabel('%'); ax.set_ylim(0, 105)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_convergence_analysis.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[图表3] 收敛性分析: {output_dir}/03_convergence_analysis.png")

        # ── 图4: 综合仪表盘 ──
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        fig.suptitle('YOLO Training Dashboard', fontsize=16, fontweight='bold', y=0.98)

        # 全图损失
        ax = fig.add_subplot(gs[0, :2])
        for key, label, color in [('train/box_loss', 'Train Box', '#2196F3'),
                                   ('val/box_loss', 'Val Box', '#EF5350'),
                                   ('train/cls_loss', 'Train Cls', '#FF9800'),
                                   ('train/dfl_loss', 'Train DFL', '#9C27B0')]:
            if key in rows[0]:
                ax.plot(epochs, [r[key] for r in rows], '-', color=color, linewidth=1.2, alpha=0.8, label=label)
        ax.set_title('All Losses Overview', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(fontsize=7, ncol=4); ax.grid(True, alpha=0.3)

        # mAP 综合
        ax = fig.add_subplot(gs[0, 2:])
        for key, label, color in [('metrics/mAP50(B)', 'mAP@0.5', '#4CAF50'),
                                   ('metrics/mAP50-95(B)', 'mAP@0.5:0.95', '#2196F3')]:
            if key in rows[0]:
                ax.plot(epochs, [r[key] for r in rows], '-', color=color, linewidth=2, label=label)
        ax.set_title('Detection Performance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('mAP')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # 训练统计
        ax = fig.add_subplot(gs[1, 0])
        ax.axis('off')
        stats = f"Training Summary\n{'─'*25}\nEpochs: {len(rows)}\n"
        for k, label in [('metrics/mAP50(B)', 'Best mAP@0.5'),
                          ('metrics/mAP50-95(B)', 'Best mAP@0.5:0.95'),
                          ('metrics/precision(B)', 'Best Precision'),
                          ('metrics/recall(B)', 'Best Recall')]:
            if k in rows[0]:
                vals = [r[k] for r in rows]
                best_v, best_e = max(vals), vals.index(max(vals)) + 1
                stats += f"{label}: {best_v:.4f} (e{best_e})\n"
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

        # 各指标最优条形图
        ax = fig.add_subplot(gs[1, 1:3])
        metrics_summary = []
        labels_summary = []
        colors_summary = []
        for k, label, c in [('metrics/mAP50(B)', 'mAP@0.5', '#4CAF50'),
                             ('metrics/mAP50-95(B)', 'mAP@0.5:0.95', '#2196F3'),
                             ('metrics/precision(B)', 'Precision', '#FF9800'),
                             ('metrics/recall(B)', 'Recall', '#FF5722')]:
            if k in rows[0]:
                vals = [r[k] for r in rows]
                metrics_summary.append(max(vals))
                labels_summary.append(label)
                colors_summary.append(c)
        bars = ax.barh(labels_summary, metrics_summary, color=colors_summary, alpha=0.8)
        for bar, v in zip(bars, metrics_summary):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
        ax.set_title('Best Metrics Summary', fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(metrics_summary)*1.15 if metrics_summary else 1)
        ax.grid(True, alpha=0.3, axis='x')

        # Precision-Recall 散点
        ax = fig.add_subplot(gs[1, 3])
        if 'metrics/precision(B)' in rows[0] and 'metrics/recall(B)' in rows[0]:
            p_arr = [r['metrics/precision(B)'] for r in rows]
            r_arr = [r['metrics/recall(B)'] for r in rows]
            scatter = ax.scatter(r_arr, p_arr, c=epochs, cmap='viridis', s=30, alpha=0.7)
            ax.set_title('P-R Space Over Training')
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
            plt.colorbar(scatter, ax=ax, label='Epoch')

        # 损失下降直方图
        ax = fig.add_subplot(gs[2, :2])
        for key, label, color in [('train/box_loss', 'Box', '#2196F3'),
                                   ('train/cls_loss', 'Cls', '#FF9800'),
                                   ('train/dfl_loss', 'DFL', '#9C27B0')]:
            if key in rows[0]:
                vals = [r[key] for r in rows]
                if len(vals) > 1:
                    drops = [vals[i-1] - vals[i] for i in range(1, len(vals))]
                    ax.hist(drops, bins=20, alpha=0.5, color=color, label=f'{label} Δ')
        ax.set_title('Per-Epoch Loss Change Distribution')
        ax.set_xlabel('Loss Change (Δ)'); ax.set_ylabel('Frequency')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

        # mAP 改进积累
        ax = fig.add_subplot(gs[2, 2:])
        if 'metrics/mAP50(B)' in rows[0]:
            mAP_vals = [r['metrics/mAP50(B)'] for r in rows]
            ax.plot(epochs, mAP_vals, '-', color='#4CAF50', linewidth=2)
            # 标注最大提升 epoch
            if len(mAP_vals) > 1:
                diffs = [mAP_vals[i]-mAP_vals[i-1] for i in range(1, len(mAP_vals))]
                top3 = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:3]
                for i in top3:
                    ax.annotate(f'+{diffs[i]:.4f}', xy=(i+2, mAP_vals[i+1]),
                                fontsize=8, color='darkgreen',
                                arrowprops=dict(arrowstyle='->', lw=0.8, color='darkgreen'))
        ax.set_title('mAP@0.5 with Top Gains Annotated')
        ax.set_xlabel('Epoch'); ax.set_ylabel('mAP@0.5')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "04_dashboard.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[图表4] 综合仪表盘: {output_dir}/04_dashboard.png")

    except ImportError:
        print("[跳过] matplotlib 未安装，无法生成图表")
    except Exception as e:
        print(f"[警告] 图表生成异常: {e}")


def _save_training_summary(results_dir: str, output_dir: str, config: dict):
    """保存训练摘要 JSON"""
    import json

    csv_path = os.path.join(results_dir, "results.csv")
    best_metrics = {}
    if os.path.isfile(csv_path):
        import csv
        rows = []
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                rows.append({k.strip(): float(v.strip()) for k, v in row.items()})
        if rows:
            for metric_key in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                               'metrics/precision(B)', 'metrics/recall(B)']:
                if metric_key in rows[0]:
                    values = [r[metric_key] for r in rows]
                    best_idx = np.argmax(values)
                    best_metrics[metric_key] = {
                        "best": round(values[best_idx], 4),
                        "epoch": best_idx + 1,
                        "final": round(values[-1], 4),
                    }

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {k: str(v) for k, v in config.items()},
        "best_metrics": best_metrics,
    }
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[日志] 训练摘要已保存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 无人机检测模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train/yolotrain/train.py --data train/yolotrain/dataset/data.yaml --model yolov8n.pt --epochs 100
  python train/yolotrain/train.py --data ... --model yolov8n.pt --epochs 50 --freeze 10
  python train/yolotrain/train.py --data ... --model runs/detect/train/weights/last.pt --resume
        """,
    )

    parser.add_argument("--data", type=str, default="train/yolotrain/dataset/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960,
                        help="训练输入尺寸。对于 1280×720 图片推荐 960 或 1280，"
                             "可保留更多小目标细节。默认 640 适用于 ≤720p 的图片")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--name", type=str, default="drone_detect")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，确保训练可复现")
    parser.add_argument("--export", type=str, default=None)
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "engine", "tflite", "openvino"])
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--no-charts", action="store_true",
                        help="跳过图表生成")
    args = parser.parse_args()

    # Windows 下多进程 DataLoader 可能不稳定，自动降级
    if sys.platform == "win32" and args.workers > 0:
        print(f"[提示] Windows 检测到 workers={args.workers}，"
              "多进程在 Windows 下可能不稳定，建议 --workers 0")
        # 不强制覆盖，用户可能已经知道并配置了合适值

    # 检查依赖
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: ultralytics 未安装\n运行: pip install ultralytics")
        sys.exit(1)

    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        print("[提示] tqdm 未安装，进度条不可用。pip install tqdm")

    # ── 导出模式 ──
    if args.export:
        print(f"\n{'='*50}")
        print("导出模型")
        print(f"{'='*50}")
        model = YOLO(args.export)
        model.export(format=args.format, imgsz=args.imgsz)
        print(f"模型已导出为 {args.format} 格式")
        return

    # ── 仅验证 ──
    if args.val:
        print(f"\n{'='*50}")
        print("验证模型")
        print(f"{'='*50}")
        model = YOLO(args.model)
        metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        return

    # ── 训练模式 ──
    print(f"\n{'='*60}")
    print(f"  YOLO 无人机检测模型训练")
    print(f"{'='*60}")
    print(f"  数据集:     {args.data}")
    print(f"  预训练:     {args.model}")
    print(f"  训练轮数:   {args.epochs}")
    print(f"  输入尺寸:   {args.imgsz}")
    print(f"  批次大小:   {args.batch}")
    print(f"  设备:       {args.device}")
    if args.freeze:
        print(f"  冻结层数:   {args.freeze}")
    print(f"{'='*60}\n")

    # 检查数据集是否包含实际图片
    _data_yaml_dir = os.path.dirname(os.path.abspath(args.data))
    for _split, _sub in [("训练集", "train/images"), ("验证集", "valid/images")]:
        _img_dir = os.path.join(_data_yaml_dir, _sub)
        if not os.path.isdir(_img_dir):
            print(f"[警告] {_split}目录不存在: {_img_dir}")
            continue
        _imgs = [f for f in os.listdir(_img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not _imgs:
            print(f"[错误] {_split}为空（{_img_dir} 中无图片文件）")
            print("请将标注好的图片放入对应目录后再训练。")
            print("详见: train/yolotrain/dataset/README.md")
            sys.exit(1)
        print(f"[数据] {_split}: {len(_imgs)} 张图片")

    # 模型直接保存到 train/yolotrain/train_result/models/
    yolo_model_dir = os.path.join(_RESULTS_DIR, "models")
    os.makedirs(yolo_model_dir, exist_ok=True)

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr,
        name=args.name,
        project=str(yolo_model_dir),
        resume=args.resume,
        seed=args.seed,
        # 优化策略
        optimizer="AdamW",
        cos_lr=True,
        patience=20,
        # 矩形训练 — 按宽高比分组 batch，减少 padding 浪费
        # 对 1280×720 (16:9) 等非正方形图片效果显著
        rect=True,
        # 数据增强（无人机场景优化）
        # 注: scale 值需配合 imgsz — 高分辨率输入时避免过度缩小小目标
        mosaic=1.0,
        close_mosaic=10,
        degrees=15.0,
        translate=0.1,
        scale=0.3,          # 640→0.5, 960/1280→0.3 防止无人机缩成几个像素
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # 启用详细日志
        plots=True,
        exist_ok=True,
    )

    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    # 开始训练（ultralytics 内部有进度条，这里添加总体时间跟踪）
    t_start = time.time()

    if HAS_TQDM:
        print("训练中... (ultralytics 内置进度条)")
        print("-" * 40)

    results = model.train(**train_kwargs)

    elapsed = time.time() - t_start
    print(f"\n训练完成！耗时: {elapsed/60:.1f} 分钟")

    # 输出结果路径
    save_dir = str(results.save_dir)
    print(f"\n模型保存于: {save_dir}")
    print(f"  最佳模型: {save_dir}/weights/best.pt")
    print(f"  最终模型: {save_dir}/weights/last.pt")

    # ── 生成训练图表 ──
    if not args.no_charts:
        print(f"\n{'='*50}")
        print("生成训练图表和日志")
        print(f"{'='*50}")

        output_dir = _RESULTS_DIR
        _plot_training_results(save_dir, output_dir)
        _save_training_summary(save_dir, output_dir, vars(args))

    # 复制最佳模型到 train_result/models/
    import shutil
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    if os.path.isfile(best_pt):
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(yolo_model_dir, f"drone_detect_{_ts}.pt")
        shutil.copy2(best_pt, dst)
        print(f"\n最佳模型已保存: {dst}")

    # 验证最佳模型
    print(f"\n{'='*50}")
    print("在验证集上评估最佳模型")
    print(f"{'='*50}")
    best_model = YOLO(best_pt)
    metrics = best_model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
