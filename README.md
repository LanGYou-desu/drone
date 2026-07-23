# 鹰眼长空

基于多平台协同的低空无人机智能监测系统。融合可见光、红外、雷达三种传感平台的轨迹数据，提供双目立体视觉（YOLO + 三角测量）无人机检测、3D 可视化、加权合成、轨迹预测、捕捉时机分析和 AI 策略生成。

**技术栈：** Python Flask · pywebview · Three.js · ECharts · YOLO (ultralytics) · OpenCV · PyTorch · 双目立体视觉 · OpenAI 兼容 AI 接口

## 快速开始

```bash
# CPU 版 PyTorch
pip install -r requirements.txt

# GPU 版 PyTorch（CUDA 12.4）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 启动
python main.py               # 默认同时启动两个模块（桌面窗口）
python main.py --headless    # 纯 HTTP 模式
python main.py recon         # 仅重建分析 → :5000
python main.py recog         # 仅轨迹识别 → :5001
```

首次启动识别模块时会自动下载 YOLO 模型。编辑 `config.json` 可切换模型、配置双目参数、AI 接口和预测模型参数。

## 项目结构

```
drone/
├── main.py                              # 统一入口
├── config.json                          # 运行时配置
├── requirements.txt
├── models/                              # 模型权重文件
│   ├── yolo/                            # YOLO 模型权重 (.pt)
│   └── hybrid_predictor/                # 轨迹预测模型权重 (.pt)
│
├── trajectory_reconstruction/           # 轨迹重建与分析 (:5000)
│   ├── core/                            #   领域逻辑
│   │   ├── config/                      #   配置管理
│   │   ├── io/                          #   数据 I/O
│   │   ├── prediction/                  #   预测算法
│   │   │   ├── prediction.py            #     预测入口（线性外推 + 混合模型）
│   │   │   └── hybrid/                  #     Phy-ODE-Diffusion 模型定义
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
│   └── frontend/                        #   页面模板 + JS
│
├── train/                               # 模型训练
│   ├── yolotrain/                       # YOLO 无人机检测训练
│   │   ├── dataset/                     #   训练/验证数据
│   │   ├── train.py                     #   训练脚本（含图表输出）
│   │   └── train_result/                #   训练结果图表和日志
│   └── hybrid_predictor/                # 轨迹预测模型训练
│       ├── dataset/                     #   数据集目录
│       │   ├── train/                   #   训练轨迹 (.npz / .dat)
│       │   ├── valid/                   #   验证轨迹
│       │   ├── test/                    #   测试轨迹（可选）
│       │   └── data/                    #   后备数据目录
│       ├── dataset.py                   #   数据加载 + 滑动窗口采样
│       ├── generate_synthetic.py        #   四旋翼动力学轨迹生成器
│       ├── train.py                     #   分阶段训练脚本（含进度条+6张图表）
│       └── train_result/                #   训练结果图表和日志
│
├── templates/                           # 共享前端资源
│   ├── frontend/shared/                 #   base.html + icons.html
│   ├── frontend/static/                 #   CSS + JS + 第三方库
│   └── config_template.json
│
├── data/                                # 运行时数据
│   ├── fact/                            #   轨迹数据 (.npz 默认, .dat 兼容)
│   ├── predict/                         #   预测结果 (.npz)
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
- **轨迹预测 — 双模式**:
  - **Phy-ODE-Diffusion 混合模型**: Transformer + 物理 ODE + 扩散生成，物理约束自回归预测
  - **线性外推**: 快速兜底方案，模型不可用时自动回退
- 运动学分析图表（ECharts）— 高度/速度/加速度/曲率
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

### YOLO 无人机检测模型

```bash
# 准备数据集（YOLO 格式标注）
# 详见 DEVELOPER.md

# 训练无人机检测模型
python train/yolotrain/train.py \
    --data train/yolotrain/dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 100 --imgsz 640
```

训练结果图表自动保存到 `train/yolotrain/train_result/`，包含：
- 损失分解曲线、验证指标曲线、收敛性分析、综合仪表盘

### 轨迹预测模型 (Phy-ODE-Diffusion)

> 模型架构：6 层 Transformer 编码器 + 物理结构 ODE (d_z=64) + 物理引导扩散，~482K 参数

```bash
# 1. 生成合成训练数据（或放入 .dat 格式真实数据到 dataset/train/）
python train/hybrid_predictor/generate_synthetic.py 200

# 2. 阶段一+二训练（推荐 --stage 2）
python train/hybrid_predictor/train.py --stage 2 \
    --epochs-s1 100 --epochs-s2 100 \
    --batch-s1 128 --batch-s2 128 \
    --device cuda:0

# 3. 阶段三联合微调（可选）
python train/hybrid_predictor/train.py --stage 3 \
    --epochs-s3 20 --batch-s3 64 --device cuda:0

# 4. 全部三阶段一键训练（各阶段独立参数）
python train/hybrid_predictor/train.py --stage all \
    --epochs-s1 100 --epochs-s2 100 --epochs-s3 20 \
    --batch-s1 128 --batch-s2 128 --batch-s3 64 \
    --device cuda:0

# 5. 恢复训练（自动跳过已完成阶段，恢复优化器/调度器状态）
python train/hybrid_predictor/train.py --stage 2 --resume train/hybrid_predictor/train_result/best/phy_ode_diffusion_best_s2.pt
```

**训练特性：**
- 每阶段结束自动保存完整检查点（含优化器/调度器状态）
- 最佳模型单独保存为 `phy_ode_diffusion_best_s{stage}.pt` 不被覆盖
- 每阶段结束后立即生成训练图表（01~06），及时掌握训练进展
- `--epochs-sN` / `--batch-sN` 独立控制各阶段轮次和批次大小
- NaN 异常自动跳过梯度更新，阶段三检测到 NaN 自动提前终止
- 物理约束损失在阶段二/三中参与梯度回传（`physics_weight=0.01`）

**评估指标：** MSE + ADE + FDE + Speed/Accel/Height 物理违反率（含上下限）。

预测服务自动查找最新/最佳权重文件（优先级：`best_s*` > `s*_e*`）。
`config.json` → `hybrid_model` 可在设置页面直接调整推理物理约束参数。

> 详细数学原理和训练方法见 [`docs/hybrid_prediction_model.md`](docs/hybrid_prediction_model.md)

## 文档

| 文档 | 说明 |
|------|------|
| [使用手册](docs/manual.md) | 用户操作指南 |
| [配置说明](docs/CONFIG.md) | config.json 完整参数说明 |
| [混合预测模型](docs/hybrid_prediction_model.md) | Phy-ODE-Diffusion 数学模型、架构、训练方法 |
| [开发文档](docs/DEVELOPER.md) | 架构、API、开发指南 |
