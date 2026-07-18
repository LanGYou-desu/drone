# 鹰眼长空 — 使用手册

基于多平台协同的低空无人机智能监测系统，集成双目 YOLO 无人机检测与轨迹分析。

## 快速开始

```bash
pip install -r requirements.txt
python main.py               # 桌面窗口（同时启动两个模块）
python main.py --headless    # 纯 HTTP 模式
python main.py recon         # 仅重建分析 → :5000
python main.py recog         # 仅轨迹识别 → :5001
```

编辑 `config.json` 填入 API Key 后可使用 AI 策略。

## 界面总览

系统包含两个模块，各 5 个功能页面，顶部导航栏切换。支持深色/浅色双主题（设置页面切换）。

### 轨迹重建与分析 (:5000)

**总览** — 全屏 3D 轨迹视图。左侧面板切换平台显隐，底部播放控制条。

**操作：** 鼠标左键旋转 · 滚轮缩放 · 右键平移 · WASD 飞行 · QE 升降 · Space 播放 · 悬停轨迹点查看坐标和时间

**预测** — 选择平台，调整点数/步长，点击「开始预测」。系统自动选择最优预测引擎：
- **Phy-ODE-Diffusion 混合模型**：基于深度学习的物理约束预测（需训练权重）
- **线性外推**：兜底方案（始终可用）

所有平台预测时间自动对齐，综合轨迹由各平台预测加权合成。

**分析** — 高度/速度/加速度/曲率四张图表，原始+预测拼接显示。底部卡片给出最佳捕捉时机（需先预测）。

**AI 策略** — 勾选平台，生成捕捉策略，可保存为报告。

**数据** — 上传 `.dat` 文件创建自选平台；备份管理支持完整/部分恢复；更新全部数据重置出厂轨迹。

### 无人机检测 (:5001)

**检测** — 双目视频上传（拖拽或浏览），选择目标平台，YOLO 实时检测无人机，双目三角测量生成 3D 轨迹。检测完成后可保存到 `data/fact/` 供重建模块使用。

**检测历史** — 查看已保存的检测轨迹文件和备份列表。

**设置** — 配置 YOLO 模型、检测参数、双目标定参数（基线距离、视场角等）。

## 配置要点

`config.json` 关键字段：

| 字段 | 说明 |
|------|------|
| `ai.api_key` | AI 接口密钥 |
| `detection_methods.<id>.weight` | 综合轨迹合成权重（默认 1.0） |
| `hybrid_model.enabled` | 是否启用混合预测模型 |
| `hybrid_model.v_max` / `a_max` / `z_min` | 物理约束参数（速度/加速度/高度） |
| `hybrid_model.guidance_eta` | 物理引导强度（0.01-0.50） |
| `hybrid_model.inference_steps` | DDIM 推理步数（10-200） |
| `hybrid_model.device` | 推理设备（cpu / cuda:0） |
| `capture_weights` | 捕捉时机评分权重 |
| `camera_speed` | WASD 移动速度（默认 0.12） |
| `theme` | 默认主题（dark / light） |
| `detection.model` | YOLO 模型路径 |
| `detection.confidence_threshold` | 检测置信度阈值 |
| `stereo.baseline` | 双目基线距离（米） |

完整配置说明见 [`docs/CONFIG.md`](CONFIG.md)

## 数据格式

`.dat` 文件，每行空格分隔 `x y z t`（坐标 + 秒）。

示例：
```
12.3456 -3.2100 150.0000 0.0000
12.5120 -3.1850 150.1230 0.3000
12.6890 -3.1420 150.2450 0.6000
```

此格式同时用于：
- 运行时轨迹数据（`data/fact/*.dat`）
- 训练数据导入（放入 `train/hybrid_predictor/dataset/train/` 或 `valid/`）

## 模型训练

### YOLO 无人机检测模型

```bash
# 训练
python train/yolotrain/train.py \
    --data train/yolotrain/dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 100

# 查看训练结果
# 图表保存在 train/yolotrain/train_result/ 中
```

详见开发文档。

### 轨迹预测模型 (Phy-ODE-Diffusion)

```bash
# 1. 生成合成训练数据
python train/hybrid_predictor/generate_synthetic.py 200

# 2. 阶段一+二训练（推荐）
python train/hybrid_predictor/train.py --stage 2 --epochs 40 --batch 64 --device cuda:0

# 3. 阶段三联合微调（可选）
python train/hybrid_predictor/train.py --stage 3 --epochs 20 --batch 32 --device cuda:0

# 4. 恢复训练
python train/hybrid_predictor/train.py --stage 2 --resume train/hybrid_predictor/train_result/models/phy_ode_diffusion_best_s2.pt

# 训练完成后权重自动保存，重启服务即生效
```

训练结果图表保存在 `train/hybrid_predictor/train_result/`，包含：
- 损失分解曲线（train/val，对数尺度）
- 收敛性分析（val/train 比、损失下降率、学习率衰减、耗时分布）
- 阶段对比（柱状图 + 雷达图 + 箱线图）
- 物理指标（速度/加速度/高度违反率）
- 综合仪表盘（一页概览）

详细原理见 [`docs/hybrid_prediction_model.md`](hybrid_prediction_model.md)

## 常见问题

**预测不准？** 
- 检查是否启用了混合模型（`hybrid_model.enabled`），模型权重是否存在
- 无权重时自动回退到线性外推，精度有限
- 使用更多训练数据（真实 `.dat` 轨迹放入 `train/hybrid_predictor/dataset/train/`）可提升精度

**检测不到无人机？** COCO 预训练模型无 drone 类，需训练自定义模型或使用 airplane/bird 代理。配置中 `target_classes` 设为 `[4, 14]` 可检测飞行器和飞鸟。

**预测后分析页没数据？** 需先执行预测。切换页面或重启后预测文件会清空，需重新生成。

**备份恢复没反应？** 恢复前确认选中了备份项。完整覆盖会清空全部数据后替换。

**自选平台不显示？** 上传 .dat 后自动出现，重启清空。

**切换主题？** 设置页面点击切换，自动保存到 localStorage。

**检测历史显示"空"？** 旧格式备份已自动迁移，新检测备份正常显示平台名。

**混合模型未加载？**
- 检查 `models/hybrid_predictor/phy_ode_diffusion.pt` 是否存在
- 查看控制台 `[预测]` 日志了解回退原因
- 历史点不足 10 个时自动回退到线性外推
- 设置页面可禁用混合模型或调整物理约束参数

**训练时 loss 显示 NaN？**
- 阶段三 scheduled sampling 可能导致梯度爆炸，阶段一+二训练即可获得可用模型
- 训练脚本自动跳过 NaN 梯度更新，阶段三检测到 NaN 会提前终止并保存检查点
- 减少批次大小或降低学习率可提升稳定性

**如何恢复中断的训练？**
- `--resume models/.../phy_ode_diffusion_best_s2.pt` 可恢复优化器/调度器/epoch 状态
- 自动跳过已完成阶段，从断点继续训练

**训练结果如何评估？**
- 每 epoch 输出 MSE + ADE (平均位移误差) + FDE (终点位移误差)
- 训练完成后运行图表生成脚本可视化所有指标
