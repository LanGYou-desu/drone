# 鹰眼长空

基于多平台协同的低空无人机智能监测系统。融合可见光、红外、雷达三种传感平台的轨迹数据，提供双目立体视觉（YOLO + 三角测量）无人机检测、3D 可视化、加权合成、轨迹预测、捕捉时机分析和 AI 策略生成。

**技术栈：** Python Flask · pywebview · Three.js · ECharts · YOLO (ultralytics) · OpenCV · 双目立体视觉 · OpenAI 兼容 AI 接口

## 快速开始

```bash
pip install -r requirements.txt

python main.py               # 默认同时启动两个模块（桌面窗口）
python main.py --headless    # 纯 HTTP 模式
python main.py recon         # 仅重建分析 → :5000
python main.py recog         # 仅轨迹识别 → :5001
```

首次启动识别模块时会自动下载 YOLO 模型。编辑 `config.json` 可切换模型、配置双目参数等。

## 项目结构

```
drone/
├── main.py                              # 统一入口
├── config.json                          # 运行时配置
├── requirements.txt
├── models/                              # YOLO 模型权重
│
├── trajectory_reconstruction/           # 轨迹重建与分析 (:5000)
│   ├── core/                            #   领域逻辑
│   │   ├── config/                      #   配置管理
│   │   ├── io/                          #   数据 I/O
│   │   ├── prediction/                  #   预测算法
│   │   ├── ai/                          #   AI 策略
│   │   ├── state.py                     #   全局状态
│   │   └── math_utils.py                #   数学工具
│   ├── services/                        #   业务编排
│   │   ├── data_service.py              #   数据加载/合成
│   │   ├── predict_service.py           #   预测编排
│   │   └── backup_service.py            #   备份管理
│   ├── views/                           #   HTTP 接口
│   └── frontend/                        #   页面 + JS
│
├── trajectory_recognition/              # 无人机检测 (:5001)
│   ├── detection/                       #   双目立体视觉检测引擎
│   │   ├── engine.py                    #   YOLO 模型加载与推理
│   │   ├── stereo.py                    #   双目三角测量 + 跨目匹配
│   │   ├── tracker.py                   #   多目标跟踪 (ByteTrack/IOU)
│   │   └── preprocess.py                #   视频抽帧预处理
│   ├── services/                        #   业务编排
│   │   ├── detection_service.py         #   检测会话管理 + 世界坐标变换
│   │   └── data_bridge.py               #   轨迹数据桥接 → data/fact/
│   ├── views/                           #   HTTP 接口 + 页面路由
│   ├── train.py                         #   YOLO 训练脚本
│   └── frontend/                        #   页面模板 + JS
│
├── templates/                           # 共享前端资源
│   ├── frontend/shared/                 #   base.html + icons.html
│   ├── frontend/static/                 #   CSS + JS + 第三方库
│   └── config_template.json
│
├── data/                                # 运行时数据
│   ├── fact/                            #   轨迹数据 (.dat)
│   ├── predict/                         #   预测结果
│   ├── backup/                          #   快照备份
│   └── uploads/                         #   临时视频上传
│
├── reports/                             # AI 策略报告
└── docs/                                # 文档
```

## 功能模块

### 轨迹重建与分析 (:5000)
- 3D 轨迹可视化（Three.js）
- 多平台加权合成
- 轨迹预测（线性外推）
- 运动学分析图表（ECharts）
- 最佳捕捉时机评分
- AI 反制策略生成
- 数据备份/恢复

### 无人机检测 (:5001)
- 双目视频 YOLO 实时检测
- 多目标跟踪（ByteTrack/IOU）
- 立体视觉 3D 定位
- 检测结果自动保存为轨迹数据
- 自定义模型训练

## 模型训练

```bash
# 准备数据集（YOLO 格式标注）
# 详见 DEVELOPER.md 第 16 节

# 训练无人机检测模型
python -m trajectory_recognition.train \
    --data dataset/data.yaml \
    --model models/yolov8n.pt \
    --epochs 100 --imgsz 640
```

## 文档

| 文档 | 说明 |
|------|------|
| [使用手册](docs/manual.md) | 用户操作指南 |
| [开发文档](docs/DEVELOPER.md) | 架构、API、开发指南 |
