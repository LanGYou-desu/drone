# YOLO 无人机检测模型 — 训练模块

独立于主项目的 YOLO 模型训练目录。包含训练脚本和数据集。

## 目录结构

```
train/yolotrain/
├── README.md               # 本文件
├── train.py                # 训练脚本（独立运行）
└── dataset/                # 训练数据集
    ├── data.yaml           # 数据集配置
    ├── train/
    │   ├── images/         # 训练图片
    │   └── labels/         # YOLO 格式标注
    └── valid/
        ├── images/         # 验证图片
        └── labels/         # 验证标注
```

## 快速开始

```bash
# 在项目根目录下运行
python train/yolotrain/train.py \
    --data train/yolotrain/dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 640
```

## 训练完成后

将最佳模型复制到项目 `models/yolo/` 目录供推理使用：

```bash
cp runs/detect/drone_detect/weights/best.pt models/yolo/drone_detect.pt
```

然后在 `config.json` 中配置模型路径：

```json
{
    "detection": {
        "model": "models/yolo/drone_detect.pt"
    }
}
```
