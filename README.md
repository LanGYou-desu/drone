# 鹰眼长空

基于多平台协同的低空无人机智能监测系统。融合可见光、红外、雷达三种传感平台的轨迹数据，提供双目立体视觉（YOLO + 三角测量）无人机检测、时间同步/关联融合/三维重建、KinFlow 轨迹预测（短视界模型 + 长视界运动外推）、六因子风险评估、捕捉时机分析和 AI 策略生成。

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

```text
drone/
├── main.py                              # 统一入口
├── config.json                          # 运行时配置
├── requirements.txt
├── models/                              # 模型权重文件
│   ├── yolo/                            # YOLO 模型权重 (.pt)
│   └── kinflow/                         # KinFlow 轨迹预测权重 (.pt)
├── tools/                               # 数据/演示工具
│   └── make_runtime_demo.py             # 生成同目标三路异构观测演示数据
│
├── trajectory_reconstruction/           # 轨迹重建与分析 (:5000)
│   ├── core/                            #   领域逻辑
│   │   ├── config/                      #   配置管理
│   │   ├── io/                          #   数据 I/O
│   │   ├── prediction/                  #   预测算法
│   │   │   ├── prediction.py            #     预测入口（线性外推 + KinFlow）
│   │   │   └── kinflow/                 #     KinFlow 模型定义
│   │   │       ├── encoder.py           #       连续时间上下文编码器
│   │   │       ├── corridor.py          #       运动学走廊先验
│   │   │       ├── flow.py              #       走廊初始化流匹配
│   │   │       ├── projector.py         #       可微物理投影器
│   │   │       └── model.py             #       KinFlow 顶层模型
│   │   ├── timesync.py                  #   时间步同步（时钟偏置/主时间栅格）
│   │   ├── fusion.py                    #   多传感器关联融合 + 健康度
│   │   ├── reconstruction.py            #   三维轨迹重建（剔除/补洞/平滑/置信度）
│   │   ├── risk.py                      #   智能风险评估（六因子评分）
│   │   ├── sensor_sim.py                #   同目标多传感器协同仿真
│   │   ├── ai/                          #   AI 策略
│   │   ├── state.py                     #   全局状态
│   │   └── math_utils.py                #   数学工具
│   ├── services/                        #   业务编排
│   │   ├── data_service.py              #   数据加载/合成
│   │   ├── predict_service.py           #   预测编排
│   │   ├── sensor_service.py            #   协同/重建/评估编排
│   │   └── backup_service.py            #   备份管理
│   ├── views/                           #   HTTP 接口（含 /api/sensors/*）
│   └── frontend/                        #   页面 + JS（总览/预测/分析评估/AI 策略/数据/帮助/设置）
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
│   └── kinflow_predictor/               # KinFlow 轨迹预测模型训练
│       ├── dataset/                     #   数据集目录
│       │   ├── train/                   #   训练轨迹 (.npz / .dat)
│       │   ├── valid/                   #   验证轨迹
│       │   ├── test/                    #   测试轨迹
│       │   └── data/                    #   后备数据目录
│       ├── dataset.py                   #   数据加载 + 滑动窗口采样 + 增强
│       ├── generate_synthetic.py        #   四旋翼动力学轨迹生成器
│       ├── train.py                     #   两阶段训练脚本（含进度条+图表）
│       └── train_result/                #   训练结果图表、日志与最佳权重
│
├── backup/                              # 历史版本归档（当前代码不引用）
│   └── legacy_phy_ode/                  #   旧 Phy-ODE-Diffusion 模型代码/权重/产出
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
- 3D 轨迹可视化（Three.js），含圆柱形防护区
- **综合重建（总览内）**: 时间步同步 → 一致子集关联 → 逆方差融合 → 三维重建，直接生成"综合"轨迹（青色管状体）；左侧面板显示质量指标并可直接调整同步频率/关联门限/平滑窗口/最大补洞后重建，3D 场景显示防护区
- **轨迹预测（与重建联动）**: 默认预测综合轨迹；视界 ≤2 s 用 KinFlow 模型，>2 s 由运动模型外推并给出末端置信度；支持 3~60 点、0.05~5 s 步长
- **分析评估（合并页）**: 运动学图表 + 最佳捕捉时机 + 六因子可解释风险评估、风险时间线、分级处置建议
- **协同仿真自检**: 同一目标的多传感器观测仿真（异构采样率/噪声/时钟偏置/遮挡/丢帧），定量验证协同增益
- **线性外推**: 模型不可用时的兜底方案
- 运动学分析图表（ECharts）— 高度/速度/加速度/曲率
- 最佳捕捉时机评分（预测生成后即可用）
- AI 反制策略生成
- 数据备份/恢复；重建/评估参数持久化（刷新、重启后保持）

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

### 轨迹预测模型 (KinFlow)

> 模型架构：4 层 Pre-LN 连续时间 Transformer 编码器（40 点 / 3.9 s 历史）+ 运动学走廊先验
> + 走廊初始化流匹配（2 s 目标视界）+ 非参数可微物理投影器，共 ~372K 参数。
> 测试集（2 s 视界）：ADE 0.495 m / FDE 1.324 m；短视界优于线性外推约 4 倍，长视界由运动模型外推。
> 参考 *KinFlow: A Learned Kinematic Prior with Parallel Non-Autoregressive Refinement for
> Irregularly-Sampled UAV Trajectory Prediction*（APIN-D-26-14134）。

```bash
# 1. 生成合成训练数据（或放入 .npz/.dat 真实数据到 dataset/train/）
python train/kinflow_predictor/generate_synthetic.py 200

# 2. 冒烟测试：小批量跑通两阶段（CPU，约 1~3 分钟）
python train/kinflow_predictor/train.py --smoke

# 3. 正式训练（阶段一联合训练 + 阶段二走廊噪声微调）
python train/kinflow_predictor/train.py --stage all \
    --epochs-s1 60 --epochs-s2 20 \
    --ctx-len 40 --tgt-len 20 --batch 64 --device cuda:0

# 4. 仅阶段二微调（自动载入 train_result/best/kinflow_best_s1.pt）
python train/kinflow_predictor/train.py --stage 2 --epochs-s2 20 --device cuda:0

# 5. 恢复训练（恢复优化器/调度器/阶段进度）
python train/kinflow_predictor/train.py --stage 2 \
    --resume train/kinflow_predictor/train_result/best/kinflow_best_s1.pt
```

**训练特性：**
- 两阶段训练：阶段一联合优化 `L = L_flow + λ_c·MSE(X0, Xgt) + λ_p·L_phy`（λ_c=0.5），
  阶段二注入走廊噪声 σ、降低学习率、可选投影器在环；另含端到端采样损失 `λ_sample=0.3`
- 每 epoch 保存检查点 `kinflow_s{stage}_e{epoch}_v{loss}.pt`；最佳模型 `kinflow_best_s{stage}.pt` 不被覆盖并同步到 `train_result/best/`
- 自动生成训练曲线（阶段损失、全局损失、ADE/FDE、物理违反率、学习率、仪表盘）与测试集评价图表
- NaN/Inf 损失自动跳过该批次；梯度裁剪 max-norm 1.0；AdamW + warmup 余弦退火
- `--max-windows` / `--max-val-windows` 可限制窗口数做快速实验

**评估指标：** 归一化 MSE + 物理坐标 ADE/FDE + minADE_8（σ=0.1 概率采样）+ 速度/垂直速率/高度/加速度违反率 + 单窗口推理延迟。

预测服务自动查找可用权重（优先级：`train_result/best/kinflow_best_s2|s1` > `models/kinflow/kinflow_best_s*` > `models/kinflow/kinflow_s*_v*`）。
`config.json` → `kinflow_model` 可在设置页面直接调整推理物理边界、Euler 步数、起点噪声与设备。

> 详细数学原理和训练方法见 [`docs/kinflow_prediction_model.md`](docs/kinflow_prediction_model.md)

## 文档

| 文档 | 说明 |
|------|------|
| [使用手册](docs/manual.md) | 用户操作指南 |
| [配置说明](docs/CONFIG.md) | config.json 完整参数说明 |
| [KinFlow 预测模型](docs/kinflow_prediction_model.md) | KinFlow 数学模型、架构、训练与评估方法 |
| [多传感协同与智能评估](docs/collaboration.md) | 时间步同步、关联融合、三维重建、预测联动、风险评估的算法/接口/验收 |
| [变更记录](docs/CHANGELOG.md) | 版本迭代与文档更新说明 |
| [开发文档](docs/DEVELOPER.md) | 架构、API、开发指南 |
| [程序设计说明书](docs/程序设计说明书/main.pdf) | 系统总体设计、模块详细设计、流程图、数据结构与接口、运行截图（LaTeX 源码同目录） |
