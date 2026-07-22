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
    从 ultralytics 输出的 results.csv 生成高分辨率单图评价指标。
    每张图只包含一个指标，DPI=300，适合论文发表。
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
    _CHART_DPI = 300
    _FS = (10, 6)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # ── 01 损失分解（4 张单图）──────────────────────
        loss_configs = [
            ('train/box_loss', 'Box Loss (Train)', '#2196F3', '01a_box_loss.png'),
            ('train/cls_loss', 'Class Loss (Train)', '#FF9800', '01b_cls_loss.png'),
            ('train/dfl_loss', 'DFL Loss (Train)', '#9C27B0', '01c_dfl_loss.png'),
            ('val/box_loss', 'Box Loss (Validation)', '#EF5350', '01d_val_box_loss.png'),
        ]
        for key, title, color, fname in loss_configs:
            if key in rows[0]:
                fig, ax = plt.subplots(figsize=_FS)
                vals = [r[key] for r in rows]
                ax.plot(epochs, vals, '-', color=color, lw=2, alpha=0.85)
                ax.fill_between(epochs, 0, vals, alpha=0.08, color=color)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
                ax.grid(True, alpha=0.3)
                best_i = vals.index(min(vals))
                ax.annotate(f'{vals[best_i]:.4f}', xy=(best_i+1, vals[best_i]), fontsize=11, color='darkred', fontweight='bold')
                fig.tight_layout(); fig.savefig(os.path.join(output_dir, fname), dpi=_CHART_DPI, bbox_inches='tight'); plt.close(fig)
                print(f"[图表] {fname}")

        # ── 02 验证指标（4 张单图）──────────────────────
        metric_configs = [
            ('metrics/precision(B)', 'Precision', '#4CAF50', '02a_precision.png'),
            ('metrics/recall(B)', 'Recall', '#FF5722', '02b_recall.png'),
            ('metrics/mAP50(B)', 'mAP@0.5', '#2196F3', '02c_mAP50.png'),
            ('metrics/mAP50-95(B)', 'mAP@0.5:0.95', '#FF9800', '02d_mAP50_95.png'),
        ]
        for key, title, color, fname in metric_configs:
            if key in rows[0]:
                fig, ax = plt.subplots(figsize=_FS)
                vals = [r[key] for r in rows]
                ax.plot(epochs, vals, '-', color=color, lw=2, alpha=0.85)
                ax.fill_between(epochs, 0, vals, alpha=0.08, color=color)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Value', fontsize=12)
                ax.grid(True, alpha=0.3)
                best_i = vals.index(max(vals))
                ax.annotate(f'Best: {vals[best_i]:.4f} (e{best_i+1})', xy=(best_i+1, vals[best_i]),
                            xytext=(10, -15), textcoords='offset points', fontsize=10, color='darkred',
                            arrowprops=dict(arrowstyle='->', lw=0.8, color='darkred'))
                fig.tight_layout(); fig.savefig(os.path.join(output_dir, fname), dpi=_CHART_DPI, bbox_inches='tight'); plt.close(fig)
                print(f"[图表] {fname}")

        # ── 03 收敛性分析（6 张单图）───────────────────

        # 03a 损失改进率
        fig, ax = plt.subplots(figsize=_FS)
        for key, label, color in [('train/box_loss','Box', '#2196F3'),('train/cls_loss','Cls', '#FF9800'),('train/dfl_loss','DFL', '#9C27B0')]:
            if key in rows[0]:
                vals = [r[key] for r in rows]
                improvements = [(vals[0]-v)/vals[0]*100 for v in vals]
                ax.plot(epochs, improvements, '-', color=color, lw=2, label=label)
        ax.set_title('Loss Improvement (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Improvement %', fontsize=12)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, '03a_improvement.png'), dpi=_CHART_DPI, bbox_inches='tight'); plt.close(fig)

        # 03b 过拟合检测
        if 'train/box_loss' in rows[0] and 'val/box_loss' in rows[0]:
            fig, ax = plt.subplots(figsize=_FS)
            ax.plot(epochs, [r['train/box_loss'] for r in rows], '-', color='#2196F3', lw=1.5, alpha=0.7, label='Train')
            ax.plot(epochs, [r['val/box_loss'] for r in rows], '-', color='#EF5350', lw=1.5, alpha=0.7, label='Val')
            ax.fill_between(epochs, [r['train/box_loss'] for r in rows], [r['val/box_loss'] for r in rows], alpha=0.1, color='gray')
            ax.set_title('Overfitting Detection (Box Loss)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(output_dir, '03b_overfitting.png'), dpi=_CHART_DPI, bbox_inches='tight'); plt.close(fig)

        # 03c mAP 演进
        fig, ax = plt.subplots(figsize=_FS)
        for key, label, color in [('metrics/mAP50(B)','mAP@0.5','#4CAF50'),('metrics/mAP50-95(B)','mAP@0.5:0.95','#2196F3')]:
            if key in rows[0]:
                ax.plot(epochs, [r[key] for r in rows], '-', color=color, lw=2, label=label)
        ax.set_title('mAP Progression', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('mAP', fontsize=12); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, '03c_map_progression.png'), dpi=_CHART_DPI, bbox_inches='tight'); plt.close(fig)

        # 03d P-R 平衡
        if 'metrics/precision(B)' in rows[0] and 'metrics/recall(B)' in rows[0]:
            fig, ax = plt.subplots(figsize=_FS)
            p_vals = [r['metrics/precision(B)'] for r in rows]; r_vals = [r['metrics/recall(B)'] for r in rows]
            ax.plot(epochs, p_vals, '-', color='#4CAF50', lw=1.5, label='Precision')
            ax.plot(epochs, r_vals, '-', color='#FF5722', lw=1.5, label='Recall')
            f1_vals = [2*p*r/(p+r) if (p+r)>0 else 0 for p, r in zip(p_vals, r_vals)]
            ax.plot(epochs, f1_vals, '--', color='#7E57C2', lw=1.5, label='F1 (est.)')
            ax.set_title('Precision / Recall / F1 Balance', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Value', fontsize=12); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(output_dir, '03d_pr_balance.png'), dpi=_CHART_DPI, bbox_inches='tight'); plt.close(fig)

        # 03e 稳定性
        fig, ax = plt.subplots(figsize=_FS)
        for key, label, color in [('metrics/mAP50(B)','mAP@0.5','#4CAF50'),('metrics/precision(B)','Precision','#FF9800')]:
            if key in rows[0]:
                vals = [r[key] for r in rows]; window = max(5, len(vals)//5)
                rolling_std = [np.std(vals[max(0,i-window):i+1]) for i in range(len(vals))]
                ax.plot(epochs, rolling_std, '-', color=color, lw=1.5, alpha=0.8, label=f'{label}')
        ax.set_title('Metric Stability (Rolling Std Dev)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Std Dev', fontsize=12); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, '03e_stability.png'), dpi=_CHART_DPI, bbox_inches='tight'); plt.close(fig)

        # 03f 学习效率
        if 'metrics/mAP50(B)' in rows[0]:
            fig, ax = plt.subplots(figsize=_FS)
            mAP_vals = [r['metrics/mAP50(B)'] for r in rows]; final_map = mAP_vals[-1]
            if final_map > 0:
                completion = [v/final_map*100 for v in mAP_vals]
                ax.fill_between(epochs, 0, completion, alpha=0.3, color='#4CAF50')
                ax.plot(epochs, completion, '-', color='#2E7D32', lw=2)
                ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%'); ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95%')
            ax.set_title('mAP Learning Efficiency (% of final)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('%', fontsize=12); ax.set_ylim(0, 105); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(os.path.join(output_dir, '03f_efficiency.png'), dpi=_CHART_DPI, bbox_inches='tight'); plt.close(fig)

        # ── 04 仪表盘（保留概览，高 DPI）────────────────
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        fig.suptitle('YOLO Training Dashboard', fontsize=18, fontweight='bold', y=0.98)

        ax = fig.add_subplot(gs[0, :2])
        for key, label, color in [('train/box_loss','Train Box','#2196F3'),('val/box_loss','Val Box','#EF5350'),
                                   ('train/cls_loss','Train Cls','#FF9800'),('train/dfl_loss','Train DFL','#9C27B0')]:
            if key in rows[0]: ax.plot(epochs, [r[key] for r in rows], '-', color=color, lw=1.2, alpha=0.8, label=label)
        ax.set_title('All Losses Overview', fontsize=14, fontweight='bold'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[0, 2:])
        for key, label, color in [('metrics/mAP50(B)','mAP@0.5','#4CAF50'),('metrics/mAP50-95(B)','mAP@0.5:0.95','#2196F3')]:
            if key in rows[0]: ax.plot(epochs, [r[key] for r in rows], '-', color=color, lw=2, label=label)
        ax.set_title('Detection Performance', fontsize=14, fontweight='bold'); ax.set_xlabel('Epoch'); ax.set_ylabel('mAP')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 0]); ax.axis('off')
        stats = f"Training Summary\n{'─'*25}\nEpochs: {len(rows)}\n"
        for k, label in [('metrics/mAP50(B)','Best mAP@0.5'),('metrics/mAP50-95(B)','Best mAP@0.5:0.95'),
                          ('metrics/precision(B)','Best Precision'),('metrics/recall(B)','Best Recall')]:
            if k in rows[0]:
                vals = [r[k] for r in rows]; best_v, best_e = max(vals), vals.index(max(vals))+1
                stats += f"{label}: {best_v:.4f} (e{best_e})\n"
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

        ax = fig.add_subplot(gs[1, 1:3])
        m_sum, l_sum, c_sum = [], [], []
        for k, label, c in [('metrics/mAP50(B)','mAP@0.5','#4CAF50'),('metrics/mAP50-95(B)','mAP@0.5:0.95','#2196F3'),
                             ('metrics/precision(B)','Precision','#FF9800'),('metrics/recall(B)','Recall','#FF5722')]:
            if k in rows[0]: vals = [r[k] for r in rows]; m_sum.append(max(vals)); l_sum.append(label); c_sum.append(c)
        bars = ax.barh(l_sum, m_sum, color=c_sum, alpha=0.8)
        for bar, v in zip(bars, m_sum): ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2, f'{v:.4f}', va='center', fontsize=11, fontweight='bold')
        ax.set_title('Best Metrics Summary', fontsize=14, fontweight='bold'); ax.set_xlim(0, max(m_sum)*1.15 if m_sum else 1); ax.grid(True, alpha=0.3, axis='x')

        ax = fig.add_subplot(gs[1, 3])
        if 'metrics/precision(B)' in rows[0] and 'metrics/recall(B)' in rows[0]:
            sc = ax.scatter([r['metrics/recall(B)'] for r in rows], [r['metrics/precision(B)'] for r in rows], c=epochs, cmap='viridis', s=40, alpha=0.7)
            ax.set_title('P-R Space Over Training'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); plt.colorbar(sc, ax=ax, label='Epoch')

        ax = fig.add_subplot(gs[2, :2])
        for key, label, color in [('train/box_loss','Box','#2196F3'),('train/cls_loss','Cls','#FF9800'),('train/dfl_loss','DFL','#9C27B0')]:
            if key in rows[0]:
                vals = [r[key] for r in rows]
                if len(vals) > 1: drops = [vals[i-1]-vals[i] for i in range(1,len(vals))]; ax.hist(drops, bins=20, alpha=0.5, color=color, label=f'{label}')
        ax.set_title('Per-Epoch Loss Change Distribution'); ax.set_xlabel('Loss Change'); ax.set_ylabel('Frequency'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

        ax = fig.add_subplot(gs[2, 2:])
        if 'metrics/mAP50(B)' in rows[0]:
            mAP_vals = [r['metrics/mAP50(B)'] for r in rows]; ax.plot(epochs, mAP_vals, '-', color='#4CAF50', lw=2)
            if len(mAP_vals) > 1:
                diffs = [mAP_vals[i]-mAP_vals[i-1] for i in range(1,len(mAP_vals))]
                for i in sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:3]:
                    ax.annotate(f'+{diffs[i]:.4f}', xy=(i+2, mAP_vals[i+1]), fontsize=9, color='darkgreen',
                                arrowprops=dict(arrowstyle='->', lw=0.8, color='darkgreen'))
        ax.set_title('mAP@0.5 with Top Gains'); ax.set_xlabel('Epoch'); ax.set_ylabel('mAP@0.5'); ax.grid(True, alpha=0.3)

        fig.tight_layout(); fig.savefig(os.path.join(output_dir, '04_dashboard.png'), dpi=_CHART_DPI, bbox_inches='tight'); plt.close(fig)
        print(f"[图表] 04_dashboard.png")

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


def _make_test_comparison_chart(val_map50, val_map50_95, test_map50, test_map50_95,
                                output_dir):
    """生成验证集 vs 测试集 mAP 对比柱状图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle('Validation vs Test — mAP Comparison', fontsize=13, fontweight='bold')

        labels = ['mAP@0.5', 'mAP@0.5:0.95']
        val_vals = [val_map50, val_map50_95]
        test_vals = [test_map50, test_map50_95]

        x = range(len(labels))
        w = 0.3
        bars1 = ax.bar([i - w/2 for i in x], val_vals, w, label='Validation',
                       color='#2196F3', alpha=0.85)
        bars2 = ax.bar([i + w/2 for i in x], test_vals, w, label='Test',
                       color='#4CAF50', alpha=0.85)

        for bar, val in zip(bars1, val_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
        for bar, val in zip(bars2, test_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('mAP'); ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ymax = max(val_vals + test_vals) * 1.15
        ax.set_ylim(0, ymax if ymax > 0 else 1)

        plt.tight_layout()
        path = os.path.join(output_dir, "05_test_comparison.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[图表5] 测试对比: {path}")
    except Exception as e:
        print(f"[警告] 测试对比图生成失败: {e}")


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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--name", type=str, default="drone_detect")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，确保训练可复现")
    parser.add_argument("--test-only", action="store_true",
                        help="仅用测试集评估已有模型，不训练")
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

    # ── 仅测试模式 ──
    if args.test_only:
        os.makedirs(_RESULTS_DIR, exist_ok=True)

        # data.yaml 转为绝对路径（ultralytics 需要）
        args.data = os.path.abspath(args.data)

        # 找到模型文件
        model_path = args.model
        if not os.path.isfile(model_path):
            # 尝试 train_result 中的 best.pt
            alt = os.path.join(_RESULTS_DIR, "models", "weights", "best.pt")
            if os.path.isfile(alt):
                model_path = alt
                print(f"使用模型: {model_path}")
            else:
                print(f"错误: 模型文件不存在: {model_path}")
                sys.exit(1)

        model = YOLO(model_path)

        # 检查测试集
        from pathlib import Path
        _data_dir = Path(args.data).parent
        _test_dir = _data_dir / "test" / "images"
        if not _test_dir.is_dir() or not any(
            f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
            for f in _test_dir.iterdir()
        ):
            print("错误: test/images/ 中没有图片，无法评估")
            sys.exit(1)

        print(f"\n{'='*50}")
        print("在测试集上评估模型")
        print(f"{'='*50}")

        # 切到数据集目录（ultralytics 要求 path 相对 yaml 所在目录）
        _prev_cwd = os.getcwd()
        os.chdir(_data_dir)

        try:
            # 验证集评估
            print("\n[1/2] 验证集评估...")
            val_metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
            val_map50 = val_metrics.box.map50
            val_map50_95 = val_metrics.box.map
            print(f"  mAP@0.5:      {val_map50:.4f}")
            print(f"  mAP@0.5:0.95: {val_map50_95:.4f}")

            # 测试集评估
            print("\n[2/2] 测试集评估...")
            import yaml as _yaml_lib
            _orig_cfg = {}
            try:
                with open(args.data, 'r', encoding='utf-8') as _f:
                    _orig_cfg = _yaml_lib.safe_load(_f) or {}
            except Exception:
                pass
            _ds_cfg = {
                'path': _orig_cfg.get('path', '.'),
                'train': 'test/images',
                'val': 'test/images',
            }
            if 'names' in _orig_cfg:
                _ds_cfg['names'] = _orig_cfg['names']
            elif 'nc' in _orig_cfg:
                _ds_cfg['nc'] = _orig_cfg['nc']
            _test_yaml = str(_data_dir / "_test_eval.yaml")
            with open(_test_yaml, 'w') as _f:
                _yaml_lib.dump(_ds_cfg, _f)

            try:
                test_metrics = model.val(data=_test_yaml, imgsz=args.imgsz, device=args.device)
                test_map50 = test_metrics.box.map50
                test_map50_95 = test_metrics.box.map
                print(f"  mAP@0.5:      {test_map50:.4f}")
                print(f"  mAP@0.5:0.95: {test_map50_95:.4f}")

                # 生成对比图
                _make_test_comparison_chart(
                    val_map50, val_map50_95,
                    test_map50, test_map50_95,
                    _RESULTS_DIR,
                )
                print(f"\n图表已保存到 {_RESULTS_DIR}/05_test_comparison.png")
            finally:
                if os.path.isfile(_test_yaml):
                    os.remove(_test_yaml)
        finally:
            os.chdir(_prev_cwd)

        sys.exit(0)

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

    training_failed = False
    try:
        results = model.train(**train_kwargs)
    except Exception as e:
        print(f"\n[错误] 训练过程中出现异常: {e}")
        print("尝试从已保存的中间结果生成图表...")
        training_failed = True
        import glob as _glob
        # ultralytics 自动保存到 project/name/ 目录
        _run_dirs = sorted(
            _glob.glob(os.path.join(yolo_model_dir, args.name + "*")),
            key=os.path.getmtime, reverse=True,
        )
        if _run_dirs:
            # 构造一个伪 results 对象用于后续步骤
            class _FakeResults:
                save_dir = _run_dirs[0]
            results = _FakeResults()
        else:
            print("未找到 ultralytics 输出目录，跳过后续步骤")
            sys.exit(1)

    elapsed = time.time() - t_start
    if training_failed:
        print(f"\n训练异常终止！已耗时: {elapsed/60:.1f} 分钟")
    else:
        print(f"\n训练完成！耗时: {elapsed/60:.1f} 分钟")

    # 输出结果路径
    save_dir = str(results.save_dir)
    print(f"\n模型保存于: {save_dir}")
    print(f"  最佳模型: {save_dir}/weights/best.pt")
    print(f"  最终模型: {save_dir}/weights/last.pt")

    # ── 生成训练图表 ──（训练失败时也尝试从已有数据生成）
    if not args.no_charts:
        print(f"\n{'='*50}")
        print("生成训练图表和日志")
        print(f"{'='*50}")

        output_dir = _RESULTS_DIR
        _plot_training_results(save_dir, output_dir)
        _save_training_summary(save_dir, output_dir, vars(args))

    if training_failed:
        print("\n训练未完成，跳过模型复制和验证步骤")
    else:
        # 复制最佳模型到 train_result/models/
        import shutil
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        if os.path.isfile(best_pt):
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = os.path.join(yolo_model_dir, f"drone_detect_{_ts}.pt")
            shutil.copy2(best_pt, dst)
            print(f"\n最佳模型已保存: {dst}")

            # 验证最佳模型（验证集）
            print(f"\n{'='*50}")
            print("在验证集上评估最佳模型")
            print(f"{'='*50}")
            best_model = YOLO(best_pt)
            metrics = best_model.val(data=args.data, imgsz=args.imgsz, device=args.device)
            val_map50 = metrics.box.map50
            val_map50_95 = metrics.box.map
            print(f"  mAP@0.5:      {val_map50:.4f}")
            print(f"  mAP@0.5:0.95: {val_map50_95:.4f}")

            # 测试集评价
            _test_img_dir = os.path.join(_data_yaml_dir, "test", "images")
            if os.path.isdir(_test_img_dir) and any(
                f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                for f in os.listdir(_test_img_dir)
            ):
                print(f"\n{'='*50}")
                print("在测试集上评估最佳模型")
                print(f"{'='*50}")
                # 构建测试集 data yaml（复用验证集路径，替换为测试集）
                import yaml as _yaml_lib
                try:
                    with open(args.data, 'r', encoding='utf-8') as _f:
                        _ds_cfg = _yaml_lib.safe_load(_f)
                except Exception:
                    _ds_cfg = {}
                _ds_cfg['train'] = 'test/images'
                _ds_cfg['val'] = 'test/images'
                _test_yaml = os.path.join(_data_yaml_dir, "_test_eval.yaml")
                with open(_test_yaml, 'w') as _f:
                    _yaml_lib.dump(_ds_cfg, _f)
                try:
                    test_metrics = best_model.val(
                        data=_test_yaml, imgsz=args.imgsz, device=args.device,
                    )
                    test_map50 = test_metrics.box.map50
                    test_map50_95 = test_metrics.box.map
                    print(f"  mAP@0.5:      {test_map50:.4f}")
                    print(f"  mAP@0.5:0.95: {test_map50_95:.4f}")

                    # 生成验证 vs 测试对比图
                    _make_test_comparison_chart(
                        val_map50, val_map50_95,
                        test_map50, test_map50_95,
                        _RESULTS_DIR,
                    )
                finally:
                    if os.path.isfile(_test_yaml):
                        os.remove(_test_yaml)
            else:
                print("\n[提示] 测试集为空，跳过测试集评价")
        else:
            print("\n[警告] best.pt 不存在，跳过模型验证")


if __name__ == "__main__":
    main()
