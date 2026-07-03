"""
YOLO 无人机检测模型 — 训练脚本

======== 快速开始 ========

1. 准备数据集（推荐 Roboflow 或自定义标注）:

   dataset/
   ├── data.yaml          # 数据集描述
   ├── train/
   │   ├── images/        # 训练图片
   │   └── labels/        # YOLO 格式标注（每图一个 .txt）
   └── valid/
       ├── images/        # 验证图片
       └── labels/        # 验证标注

2. 编写 data.yaml:

   path: ./dataset
   train: train/images
   val: valid/images
   names:
     0: drone             # 至少包含 drone 类
     # 1: bird            # 可选：区分飞鸟
     # 2: airplane        # 可选：区分飞机

3. 开始训练:

   python -m trajectory_recognition.train \
       --data dataset/data.yaml \
       --model yolov8n.pt \
       --epochs 100 \
       --imgsz 640

======== 数据标注格式 (YOLO) ========

每张图片对应一个同名的 .txt 文件，每行一个目标:
  class_id x_center y_center width height

所有值归一化到 [0, 1]（相对于图片宽高）。
示例:
  0 0.523 0.418 0.082 0.065   # drone 在图中偏右上位置

推荐标注工具:
  - Roboflow (https://roboflow.com)        — 在线，支持导出
  - LabelImg (https://github.com/HumanSignal/labelImg)  — 本地
  - CVAT (https://www.cvat.ai)             — 在线/自部署

======== 推荐公开数据集 ========

- DUT-Anti-UAV (https://github.com/wangdongdut/DUT-Anti-UAV)   — 红外无人机
- UAVDT (https://sites.google.com/view/grli-uavdt)              — 城市无人机
- Drone-vs-Bird (https://github.com/DroneDetection/Drone-vs-Bird) — 无人机 vs 飞鸟

======== 数据增强建议 ========

无人机检测场景特殊，建议开启以下增强:
  - 小目标增强: mosaic=1.0, scale=0.5  (无人机在画面中通常很小)
  - 运动模糊:    blur 增强模拟高速运动
  - 光照变化:    HSV 增强应对不同天气
  - 背景融合:    copy_paste 提高复杂背景泛化

======== 训练策略 ========

推荐流程:
  1. 用预训练 yolov8n.pt 作为起点
  2. 冻结 backbone，仅训练 head 10 epochs (freeze=10)
  3. 解冻全网络，训练 50-100 epochs
  4. 根据 mAP@0.5 选择最佳 checkpoint

======== 导出加速推理 ========

训练完成后导出为 ONNX / TensorRT:
  yolo export model=runs/detect/train/weights/best.pt format=onnx
  yolo export model=runs/detect/train/weights/best.pt format=engine  # TensorRT

ONNX 推理速度可提升 2-3x，TensorRT 可提升 3-5x。
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 无人机检测模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从头训练 drone 检测器
  python -m trajectory_recognition.train --data dataset/data.yaml --model yolov8n.pt --epochs 100

  # 冻结 backbone 快速微调
  python -m trajectory_recognition.train --data dataset/data.yaml --model yolov8n.pt --epochs 50 --freeze 10

  # 继续上次训练
  python -m trajectory_recognition.train --data dataset/data.yaml --model runs/detect/train/weights/last.pt --resume

  # 导出 ONNX 加速
  python -m trajectory_recognition.train --export runs/detect/train/weights/best.pt --format onnx
        """,
    )

    parser.add_argument(
        "--data", type=str, default="dataset/data.yaml",
        help="数据集描述文件路径 (data.yaml)",
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="预训练模型 (yolov8n.pt / yolov8s.pt / ...) 或 checkpoint 路径",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="训练轮数（推荐 50-200）",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="输入尺寸（推荐 640，小目标可增至 1280）",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="批次大小（根据 GPU 显存调整，16GB → 32, 8GB → 16, 4GB → 8）",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="训练设备 (0 / 1 / cpu)",
    )
    parser.add_argument(
        "--freeze", type=int, default=None,
        help="冻结前 N 层（微调推荐 10）",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="初始学习率",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从 --model 指定的 checkpoint 继续训练",
    )
    parser.add_argument(
        "--name", type=str, default="drone_detect",
        help="实验名称（输出到 runs/detect/<name>）",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="数据加载线程数",
    )

    # 导出
    parser.add_argument(
        "--export", type=str, default=None,
        help="导出模式：指定 .pt 文件路径并配合 --format",
    )
    parser.add_argument(
        "--format", type=str, default="onnx",
        choices=["onnx", "engine", "tflite", "openvino"],
        help="导出格式 (onnx/tensorrt/tflite/openvino)",
    )

    # 验证
    parser.add_argument(
        "--val", action="store_true",
        help="仅验证模型（不训练），需指定 --model 和 --data",
    )

    args = parser.parse_args()

    # 检查依赖
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: ultralytics 未安装")
        print("运行: pip install ultralytics")
        sys.exit(1)

    # ── 导出模式 ──
    if args.export:
        model = YOLO(args.export)
        model.export(format=args.format, imgsz=args.imgsz)
        print(f"模型已导出为 {args.format} 格式")
        return

    # ── 仅验证 ──
    if args.val:
        model = YOLO(args.model)
        metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        return

    # ── 训练模式 ──
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
        resume=args.resume,
        # 数据增强（无人机场景优化）
        mosaic=1.0,            # 马赛克增强（小目标友好）
        scale=0.5,             # 缩放增强
        flipud=0.0,            # 不翻转上下（天空在上）
        hsv_h=0.015,           # HSV 色调扰动
        hsv_s=0.7,             # HSV 饱和度扰动
        hsv_v=0.4,             # HSV 亮度扰动
    )

    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    print(f"\n{'='*50}")
    print(f"开始训练 YOLO 无人机检测模型")
    print(f"  数据集: {args.data}")
    print(f"  预训练: {args.model}")
    print(f"  轮数:   {args.epochs}")
    print(f"  尺寸:   {args.imgsz}")
    print(f"  批次:   {args.batch}")
    print(f"  设备:   {args.device}")
    if args.freeze:
        print(f"  冻结:   前 {args.freeze} 层")
    print(f"{'='*50}\n")

    results = model.train(**train_kwargs)

    # 输出结果路径
    save_dir = results.save_dir
    print(f"\n训练完成！模型保存于: {save_dir}")
    print(f"最佳模型: {save_dir}/weights/best.pt")
    print(f"最终模型: {save_dir}/weights/last.pt")
    print(f"\n复制到项目 models/ 目录:")
    print(f"  cp {save_dir}/weights/best.pt models/drone_detect.pt")

    # 验证最佳模型
    print("\n在验证集上评估最佳模型...")
    best_model = YOLO(f"{save_dir}/weights/best.pt")
    metrics = best_model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.p:.4f}" if hasattr(metrics.box, 'p') else "")
    print(f"  Recall:       {metrics.box.r:.4f}" if hasattr(metrics.box, 'r') else "")


if __name__ == "__main__":
    main()
