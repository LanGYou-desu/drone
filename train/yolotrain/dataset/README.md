# 无人机检测数据集

位于 `train/yolotrain/dataset/` — YOLO 训练专用数据集目录。

## 目录结构

```
train/yolotrain/dataset/
├── data.yaml              # 数据集配置（训练入口）
├── data/
│   └── data.yaml          # 备用配置（类别一致）
├── train/
│   ├── images/            # 训练图片 (.jpg, .png)
│   └── labels/            # YOLO 标注 (.txt，与图片同名)
└── valid/
    ├── images/            # 验证图片
    └── labels/            # 验证标注
```

## 图片分辨率要求

**推荐 1280×720 (HD)**，也支持其他分辨率。

训练脚本已针对 1280×720 优化（`rect=True` 矩形训练 + 调整后的数据增强参数），
更大分辨率（如 4K）建议先缩放到 1280×720 再训练。

## 标注格式

每张图片对应一个同名 `.txt` 文件，每行一个目标：

```
class_id x_center y_center width height
```

所有值**归一化到 [0, 1]**（相对于图片宽高）。

示例 `drone_001.txt`：
```
0 0.523 0.418 0.082 0.065
```

> **提示**：归一化坐标与图片分辨率无关，因此同一份标注可以在不同 `--imgsz` 下训练，
> 无需重新标注。

## 快速开始

### 1. 准备数据

将图片放入对应目录：
- 训练图片 → `train/images/`
- 验证图片 → `valid/images/`

用标注工具标注目标，导出 YOLO 格式（归一化坐标）到：
- 训练标注 → `train/labels/`
- 验证标注 → `valid/labels/`

### 2. 执行训练

```bash
# 默认 960（推荐 1280×720 图片，平衡精度与显存）
python train/yolotrain/train.py \
    --data train/yolotrain/dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 100

# 极致精度（小目标无人机检测，显存需求更大）
python train/yolotrain/train.py \
    --data train/yolotrain/dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 100 --imgsz 1280 --batch 8

# 低显存回退
python train/yolotrain/train.py \
    --data train/yolotrain/dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 100 --imgsz 640 --batch 32
```

### 3. 训练参数选择指南

| 场景 | `--imgsz` | `--batch` | 说明 |
|------|-----------|-----------|------|
| 1280×720 推荐 | **960** | 16 | 默认值，平衡精度与速度 |
| 小目标优先 | **1280** | 8 | 原始分辨率，无人机细节最完整 |
| 快速实验 | **640** | 32 | 训练最快，但小目标细节丢失较多 |

## 标注工具推荐

| 工具 | 地址 | 特点 |
|------|------|------|
| Roboflow | https://roboflow.com | 在线标注 + 数据集管理 + 自动导出 YOLO 格式 |
| LabelImg | https://github.com/HumanSignal/labelImg | 离线桌面工具，轻量 |
| CVAT | https://www.cvat.ai | 在线/自部署，支持团队协作 |
| Label Studio | https://labelstud.io | 开源，支持多种标注类型 |

## 注意事项

1. **图片文件名**：建议使用英文/数字命名，避免中文路径问题
2. **标注完整性**：每张图片必须有同名的 `.txt` 标注文件（无目标则为空文件）
3. **类别 ID**：`data.yaml` 中类别顺序必须与标注文件中的 `class_id` 一致
4. **训练/验证比例**：建议 8:2 或 9:1
5. **默认类别**：当前仅定义 `0: drone`，如需添加飞鸟等负样本，取消 `data.yaml` 中 `#1: bird` 的注释并同步更新标注
