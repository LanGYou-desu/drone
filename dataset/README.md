# 无人机检测数据集

## 目录结构

```
dataset/
├── data.yaml              # 数据集配置
├── train/
│   ├── images/            # 训练图片 (.jpg, .png)
│   └── labels/            # YOLO 标注 (.txt，与图片同名)
└── valid/
    ├── images/            # 验证图片
    └── labels/            # 验证标注
```

## 标注格式

每张图片对应一个同名 `.txt` 文件，每行一个目标：

```
class_id x_center y_center width height
```

所有值归一化到 [0, 1]（相对于图片宽高）。

示例 `drone_001.txt`：
```
0 0.523 0.418 0.082 0.065
```

## 快速开始

1. 将图片放入 `train/images/` 和 `valid/images/`
2. 用标注工具标注目标，导出 YOLO 格式到 `train/labels/` 和 `valid/labels/`
3. 执行训练：

```bash
python -m trajectory_recognition.train --data dataset/data.yaml --model models/yolov8n.pt --epochs 100
```

## 标注工具推荐

| 工具 | 地址 |
|------|------|
| Roboflow | https://roboflow.com |
| LabelImg | https://github.com/HumanSignal/labelImg |
| CVAT | https://www.cvat.ai |
