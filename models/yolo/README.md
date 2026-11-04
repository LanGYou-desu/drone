# 模型文件目录

存放 YOLO 模型权重文件 (.pt)，位于 models/yolo/ 目录。

## 自动下载预训练模型

首次运行时 ultralytics 会自动下载预训练模型到当前目录：

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # 自动下载 yolov8n.pt
```

## 放置自定义训练的模型

训练完成后，将模型从训练输出目录复制到此：

```bash
cp train/yolotrain/train_result/models/drone_detect_YYYYMMDD_HHMMSS.pt models/yolo/drone_detect.pt
```

然后在 `config.json` 中配置：

```json
{
    "detection": {
        "model": "models/yolo/drone_detect.pt"
    }
}
```

## 本目录现有模型（2026-11-04）

| 文件 | 大小 | 类别 | 说明 |
|------|------|------|------|
| detect.pt | 6.0 MB | drone | 项目自训练（YOLOv8n，单类）。验证集 mAP@0.5 0.916，但训练/验证同源、目标极小且居中，换真实场景易漏检、且会误报 |
| drone_yolo11n.pt | 5.2 MB | drone | 开源权重（Hugging Face: greatkoko/drone-detection-yolo11n，YOLO11n）。真实航拍机/四旋翼可整机框出，置信度 0.67–0.72，明显优于 detect.pt；鸟/客机仍会误报 |

两个模型都只输出 1 类 `drone`（class_id = 0），在"检测设置"页的下拉框中可直接切换，互不影响。

实测对比见 `docs/模型测试/README.md` 第 6 节。

> 注意：`drone_yolo11n.pt` 来自公开模型，商用前请确认其许可证。

## 可用预训练模型

| 模型 | 大小 | mAP | 速度 (CPU) | 速度 (GPU) |
|------|------|-----|-----------|-----------|
| yolov8n.pt | 6 MB | 37.3 | 80 ms | 3 ms |
| yolov8s.pt | 22 MB | 44.9 | 140 ms | 5 ms |
| yolov8m.pt | 52 MB | 50.2 | 230 ms | 8 ms |
| yolov8l.pt | 88 MB | 52.9 | 340 ms | 12 ms |
| yolov11n.pt | 5 MB | 39.5 | 70 ms | 2.5 ms |

> 速度基于 640×640 输入，CPU 为 i7-13700，GPU 为 RTX 4060。
> 无人机检测场景建议使用 n/s 型号，平衡精度与实时性。
