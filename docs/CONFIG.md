# config.json 配置说明

## 一、ai — AI 大模型配置

| 字段 | 类型 | 说明 |
|------|------|------|
| `provider` | string | API 提供商名称 |
| `api_key` | string | API 密钥，从控制台获取 |
| `url` | string | API 端点地址 |
| `model` | string | 模型名称，格式 `provider/model`（如 `deepseek-ai/DeepSeek-V3`） |

---

## 二、detection_methods — 多源检测手段配置

每种检测手段（可见光/红外/雷达/自选/综合）的元信息：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 显示名称 |
| `color` | string | UI 中的显示颜色（hex） |
| `visible` | bool | 是否在 UI 中可见 |
| `enabled` | bool | 是否启用该检测手段 |
| `weight` | float | 综合研判时的权重系数（synthetic 无此字段） |

---

## 三、prediction_settings — 轨迹预测参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_points` | int | 10 | 预测所需最少轨迹点数 |
| `max_points` | int | 50 | 预测使用最大轨迹点数 |
| `default_points` | int | 20 | 默认预测点数 |
| `time_step` | float | 0.3 | 预测时间步长（秒） |

---

## 四、hybrid_model — 混合预测模型配置

Phy-ODE-Diffusion 混合轨迹预测模型的运行时参数。

> 这些参数可在 **设置页面** 直接修改，无需手动编辑配置文件。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用混合模型（权重不存在时自动回退线性外推） |
| `v_max` | float | 30 | 最大速度约束 (m/s)，范围 0-100 |
| `a_max` | float | 30 | 最大加速度约束 (m/s²)，范围 0-100 |
| `z_min` | float | 0 | 最小高度约束 (m)，范围 0-100 |
| `guidance_eta` | float | 0.1 | 物理引导强度（0.01-0.50，越大约束越硬） |
| `inference_steps` | int | 50 | DDIM 推理采样步数（10-200，越多越精细越慢） |
| `device` | string | `"cpu"` | 推理设备 (`"cpu"` / `"cuda:0"`) |

### 预测模式说明

系统提供两种预测方式，自动选择：

| 模式 | 条件 | 特点 |
|------|------|------|
| **Phy-ODE-Diffusion** | `models/hybrid_predictor/phy_ode_diffusion.pt` 存在且 `enabled=true` | 高精度，物理约束，适应不规则采样 |
| **线性外推** | 模型不可用时的兜底方案 | 简单快速，仅用最后两点 |

### 权重文件

训练每阶段结束保存带描述的检查点：`phy_ode_diffusion_s{stage}_e{epoch}_v{loss}.pt`。
最佳模型单独保存：`phy_ode_diffusion_best_s{stage}.pt`，不被覆盖。
推理时自动查找优先级：`best_s*` > `s*_e*` > 旧命名 `phy_ode_diffusion.pt`。
训练方法详见 `docs/hybrid_prediction_model.md`。

---

## 五、drone_dynamics — 四旋翼动力学参数

合成轨迹生成器和物理约束使用的无人机动力学参数。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `g` | float | 9.81 | 重力加速度 (m/s²) |
| `v_h_max` | float | 20.0 | 最大水平速度 (m/s) |
| `v_v_up` | float | 5.0 | 最大垂直上升速度 (m/s) |
| `v_v_down` | float | 3.0 | 最大垂直下降速度 (m/s) |
| `max_tilt` | float | 35.0 | 最大倾斜角 (deg)，限制水平加速度 a_h ≤ g·tan(max_tilt) |
| `max_alt` | float | 120.0 | 最大飞行高度 (m)，法规限制 |
| `min_alt` | float | 1.0 | 最低飞行高度 (m)，地面约束 |
| `thrust_max` | float | 25.0 | 最大推力加速度 (m/s²)，约 2.5g |
| `thrust_hover` | float | 9.81 | 悬停推力加速度 = g |

---

## 六、training — 训练超参配置

Phy-ODE-Diffusion 模型训练的完整参数。训练脚本启动时优先从 `config.json` 读取，命令行 `--*` 参数可覆盖。

### 数据与批次

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ctx_len` | int | 20 | Transformer 上下文长度（历史轨迹点数） |
| `tgt_len` | int | 10 | 目标预测长度（每窗口预测步数） |
| `batch_size` | int | 32 | 批次大小（GPU 内存不足时减小） |

### 学习率与优化器

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lr_stage1` | float | 0.001 | 阶段一学习率（Transformer + ODE + GRU） |
| `lr_stage2` | float | 0.001 | 阶段二学习率（扩散模型） |
| `lr_stage3` | float | 0.0001 | 阶段三学习率（联合微调，较小） |
| `weight_decay` | float | 0.0001 | AdamW 权重衰减系数 |

### 训练轮数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epochs_stage1` | int | 50 | 阶段一训练轮数 |
| `epochs_stage2` | int | 100 | 阶段二训练轮数 |
| `epochs_stage3` | int | 20 | 阶段三训练轮数 |

### Warmup 策略

每阶段前 N 轮学习率从 `factor × base_lr` 线性增长到 `base_lr`，之后按 Cosine 退火衰减。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `warmup_epochs_s1` | int | 5 | 阶段一 warmup 轮数（Transformer+ODE 需稳定初始化） |
| `warmup_epochs_s2` | int | 3 | 阶段二 warmup 轮数（扩散模型对初始 LR 敏感） |
| `warmup_epochs_s3` | int | 2 | 阶段三 warmup 轮数（微调已有基础） |
| `warmup_start_factor` | float | 0.1 | warmup 起始 LR 因子（base_lr × factor） |

设置 `warmup_epochs_* = 0` 可跳过对应阶段的 warmup。

### 正则化

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `label_smoothing` | float | 0.005 | 标签平滑噪声系数（0=关闭）。对目标位置添加 `N(0, σ²)` 噪声，其中 `σ = smoothing × p_std` |

### 训练命令示例

```bash
# 使用 config.json 中所有默认值
python train/hybrid_predictor/train.py --stage 2 --device cuda:0

# 命令行覆盖特定参数
python train/hybrid_predictor/train.py --stage 2 --epochs 30 --batch 16 --device cuda:0

# 关闭 label_smoothing 和 warmup
python train/hybrid_predictor/train.py --stage 2 --warmup-s1 0 --warmup-s2 0 --label-smoothing 0
```

---

## 六、检测 UI 与采集

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `theme` | string | `"dark"` | 主题模式：`dark` / `light` |
| `camera_speed` | float | 0.1 | 3D 场景相机旋转速度 |

### capture_weights — 轨迹采集特征权重

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `height` | float | 1.0 | 高度变化权重 |
| `speed` | float | 1.0 | 速度变化权重 |
| `acceleration` | float | 1.0 | 加速度变化权重 |
| `curvature` | float | 1.0 | 曲率变化权重 |

---

## 六、detection — YOLO 目标检测参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | `models/yolo/yolov8n.pt` | YOLO 模型权重文件路径 |
| `confidence_threshold` | float | 0.5 | 检测置信度阈值，低于此值的结果被过滤 |
| `nms_threshold` | float | 0.45 | NMS 非极大抑制 IoU 阈值 |
| `frame_interval` | int | 3 | 抽帧间隔，每隔 N 帧检测一次 |
| `tracker` | string | `bytetrack` | 多目标跟踪算法：`bytetrack` / `botsort` |
| `input_width` | int | 640 | 模型输入宽度（像素） |
| `input_height` | int | 640 | 模型输入高度（像素） |
| `device` | string | `cpu` | 推理设备：`cpu` / `cuda:0` / `auto` |
| `auto_save` | bool | false | 检测完成后是否自动保存结果 |
| `target_classes` | int[] | `[0]` | 关注的类别 ID 列表（0=drone, 4=airplane, 14=bird） |
| `target_class_id` | int | 0 | 双目匹配时用于类别约束的目标类 ID |

---

## 七、platforms — 各平台双目相机参数

每个平台（visible/infrared/radar/self）独立配置，字段如下：

### 相机内参

| 字段 | 类型 | 说明 |
|------|------|------|
| `focal_length_px` | float | 焦距（像素），设为 0 则根据 FOV 自动计算 |
| `fov_horizontal` | float | 水平视场角（度） |
| `fov_vertical` | float | 垂直视场角（度） |
| `resolution_width` | int | 图像宽度（像素） |
| `resolution_height` | int | 图像高度（像素） |

### 双目外参

| 字段 | 类型 | 说明 |
|------|------|------|
| `baseline` | float | 双目基线距离（米） |
| `convergence_angle` | float | 双目会聚角（度），0 表示平行光轴 |
| `tilt_angle` | float | 相机俯角（度），正=向下倾斜 |

### 平台位姿（世界坐标系）

| 字段 | 类型 | 说明 |
|------|------|------|
| `pos_x` | float | X 坐标（米），右为正 |
| `pos_y` | float | Y 坐标（米），上为正 |
| `pos_z` | float | Z 坐标（米），前为正 |
| `pitch` | float | 俯仰角（度），正=抬头 |
| `yaw` | float | 偏航角（度），正=右转 |
| `roll` | float | 翻滚角（度），正=右滚 |

### 各平台默认值一览

| 参数 | visible | infrared | radar | self |
|------|---------|----------|-------|------|
| 基线 (m) | 1.0 | 1.2 | 2.0 | 1.0 |
| 水平 FOV | 90° | 60° | 120° | 90° |
| 垂直 FOV | 60° | 45° | 90° | 60° |
| 分辨率 | 1920×1080 | 640×480 | 1024×768 | 1920×1080 |
| 位置 (x,y,z) | (0,0,0) | (2,1.5,0) | (-2,1.5,0) | (0,1.5,0) |
