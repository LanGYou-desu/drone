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
| `min_points` | int | 3 | 预测最少点数 |
| `max_points` | int | 60 | 预测最多点数 |
| `default_points` | int | 10 | 默认预测点数 |
| `time_step` | float | 0.2 | 默认预测时间步长（秒，0.05~5） |

> 视界的组合由"点数 × 步长"决定。≤2 s（`k_max × sample_dt`）为 KinFlow 模型预测；
> 更长时前 2 s 仍由模型生成，其后由运动模型外推，逐点置信度随时间衰减。

---

## 四、kinflow_model — KinFlow 预测模型配置

KinFlow（学习式运动学先验 + 并行非自回归细化）轨迹预测模型的运行时参数。

> 这些参数可在 **设置页面 → KinFlow 预测模型** 直接修改，无需手动编辑配置文件。
> 保存后进程内模型缓存会被清空，下一次预测按新参数重新加载权重。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用 KinFlow 模型 |
| `v_h_max` | float | 20.0 | 最大水平速度 (m/s)，物理投影器硬约束 |
| `v_up_max` | float | 5.0 | 最大垂直上升速度 (m/s) |
| `v_down_max` | float | 3.0 | 最大垂直下降速度 (m/s) |
| `max_tilt` | float | 35.0 | 最大倾斜角 (deg)，决定水平加速度上限 a_h ≤ g·tan(max_tilt)，仅软约束 |
| `y_min` | float | 1.0 | 最低高度 (m)，y 为垂直轴 |
| `y_max` | float | 120.0 | 最高高度 (m) |

> 默认值与 `drone_dynamics`（`v_h_max` / `v_v_up` / `v_v_down` / `max_tilt` / `min_alt` / `max_alt`）
> 对齐，保证训练数据、运行时轨迹与投影器使用同一套物理包线。论文在公开数据集上使用更严的
> 10 / 4 / 2 m/s、30°、2~100 m。
| `n_euler_steps` | int | 4 | 并行 Euler 采样步数（1-16），越大越精细、越慢 |
| `sigma` | float | 0.0 | 起点噪声标准差（0=确定性均值预测，0.1=单峰概率采样） |
| `project` | bool | `true` | 是否启用可微物理投影器（关闭后不保证硬约束） |
| `device` | string | `"cpu"` | 推理设备 (`cpu` / `cuda:0`) |

### 预测模式说明

系统提供两种预测方式，自动选择：

| 模式 | 条件 | 特点 |
|------|------|------|
| **KinFlow** | 权重文件存在且 `enabled=true`，历史点 ≥ 5 | 并行生成全部未来点、物理硬约束、适应不规则采样 |
| **线性外推** | 模型缺失/禁用/推理异常/历史点不足 | 简单快速，仅用最后两点 |

### 权重文件

训练每个 epoch 保存检查点：`kinflow_s{stage}_e{epoch}_v{loss}.pt`。
最佳模型单独保存：`kinflow_best_s{stage}.pt`，不被覆盖，并同步到
`train/kinflow_predictor/train_result/best/`。

推理时自动查找优先级：

1. `train/kinflow_predictor/train_result/best/kinflow_best_s2.pt` → `..._s1.pt`
2. `models/kinflow/kinflow_best_s2.pt` → `..._s1.pt`
3. `models/kinflow/kinflow_s*_v{loss}.pt`（按 loss 升序取最优）
4. `models/kinflow/kinflow.pt`

训练方法详见 `docs/kinflow_prediction_model.md`。

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

KinFlow 模型训练的完整参数。训练脚本启动时优先从 `config.json` 读取，命令行 `--*` 参数可覆盖。

### 数据与批次

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ctx_len` | int | 40 | 历史窗口长度 C（轨迹点数，3.9 s @ 0.1 s） |
| `tgt_len` | int | 20 | 目标预测长度 K（每窗口预测步数，2 s @ 0.1 s） |
| `batch_size` | int | 32 | 批次大小（GPU 内存不足时减小） |
| `num_workers` | int | 4 | DataLoader 进程数（Windows 冒烟测试建议 0） |
| `time_jitter` | float | 0.35 | 训练期历史时间戳抖动幅度（按名义 dt 比例，0=关闭），用于不规则采样鲁棒性 |

### 两阶段学习率与优化器

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lr_stage1` | float | 0.001 | 阶段一联合训练学习率 |
| `lr_stage2` | float | 0.0001 | 阶段二微调学习率（论文为阶段一的 1/10） |
| `weight_decay` | float | 0.0001 | AdamW 权重衰减系数 |
| `grad_clip` | float | 1.0 | 梯度最大范数裁剪 |

### 训练轮数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epochs_stage1` | int | 60 | 阶段一联合训练轮数 |
| `epochs_stage2` | int | 20 | 阶段二走廊噪声微调轮数 |
| `checkpoint_interval` | int | 10 | 里程碑检查点间隔（epoch） |

### Warmup 策略

每阶段前 N 轮学习率从 `factor × base_lr` 线性增长到 `base_lr`，之后按 Cosine 退火衰减。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `warmup_epochs_s1` | int | 5 | 阶段一 warmup 轮数 |
| `warmup_epochs_s2` | int | 0 | 阶段二 warmup 轮数（微调无需预热） |
| `warmup_start_factor` | float | 0.1 | warmup 起始 LR 因子（base_lr × factor） |

设置 `warmup_epochs_* = 0` 可跳过对应阶段的 warmup。

### 损失权重

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lambda_corridor` | float | 0.5 | 走廊先验拟合损失 λ_c（MSE(X0, Xgt)） |
| `lambda_physics` | float | 0.01 | 软物理损失 λ_p（只做柔性引导） |
| `lambda_projector` | float | 0.1 | 阶段二投影器在环的监督项权重 |
| `lambda_sample` | float | 0.3 | 端到端采样损失权重（少量 Euler 步后的轨迹 MSE） |
| `projector_in_graph` | bool | `true` | 阶段二是否把可微投影器放进计算图 |
| `sigma_stage2` | float | 0.1 | 阶段二向走廊注入的噪声标准差 σ |

### 模型架构（可在 training 中覆盖）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_feat` | int | 64 | 编码器 token 维度 d_f |
| `d_context` | int | 128 | 上下文向量维度 d_c |
| `n_head` | int | 4 | 注意力头数 |
| `n_layers` | int | 4 | Transformer 编码器层数 |
| `dim_feedforward` | int | 256 | 前馈网络维度 |
| `dropout` | float | 0.1 | Dropout |
| `corridor_hidden` | int | 128 | 走廊 MLP 隐藏层维度 |
| `d_z` | int | 128 | 流全局条件 z_gen 维度 |
| `flow_hidden` | int | 128 | 流逐点 MLP 隐藏层维度 |
| `flow_hidden_layers` | int | 3 | 流逐点 MLP 隐藏层数 |
| `tau_emb_dim` / `time_emb_dim` | int | 16 | 未来时刻 / 流时间正弦嵌入维度 |
| `k_max` | int | 20 | 全局分支展平前的参考步数 = 模型可靠视界（20 × 0.1 s = 2 s） |

### 采样与物理边界（训练侧）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_euler_steps` | int | 4 | 训练/验证采样 Euler 步数 |
| `sigma` | float | 0.0 | 验证时起点噪声（0 = 确定性） |
| `v_h_max` | float | 20.0 | 最大水平速度 (m/s) |
| `v_up_max` | float | 5.0 | 最大垂直上升速度 (m/s) |
| `v_down_max` | float | 3.0 | 最大垂直下降速度 (m/s) |
| `y_min` / `y_max` | float | 1.0 / 120.0 | 高度范围 (m) |

> 演示数据的默认防护区已改为 `center=[8,12,4], radius=6, y_min=5, y_max=25`，与
> `tools/make_runtime_demo.py` 生成的低空演示轨迹匹配；实际部署按要地重新配置。
| `max_tilt` | float | 35.0 | 最大倾斜角 (deg) |
| `g` | float | 9.81 | 重力加速度 (m/s²) |
| `n_project_iters` | int | 3 | 投影器迭代次数 |
| `project` | bool | `true` | 是否启用投影器 |
| `eval_samples` | int | 8 | 测试集概率采样条数（minADE_K） |
| `eval_sigma` | float | 0.1 | 测试集概率采样噪声 |
| `seed` | int | 42 | 随机种子 |

### 训练命令示例

```bash
# 使用 config.json 中所有默认值
python train/kinflow_predictor/train.py --stage all --device cuda:0

# 小批量冒烟（CPU、2+1 epoch、512 训练窗口）
python train/kinflow_predictor/train.py --smoke

# 两阶段分别控制轮数与批次
python train/kinflow_predictor/train.py --stage all \
    --epochs-s1 60 --epochs-s2 20 --batch 128 --device cuda:0

# 命令行覆盖特定参数
python train/kinflow_predictor/train.py --stage 2 --epochs-s2 10 --batch 16 \
    --euler-steps 8 --device cpu

# 仅从已有训练历史重新生成图表
python train/kinflow_predictor/train.py --charts-only train/kinflow_predictor/train_result/run_xxx
```

---

## 七、多传感器协同与智能评估

以下四节由**总览页（综合重建）**、**分析评估页**与 `/api/sensors/*` 使用，
也可在 `core/config/config_manager.py` 的默认值中调整。

> **持久化**：总览页重建时会把 `synchronization.hz`、`sensor_fusion.gate_m`、
> `reconstruction.smooth_window/max_gap_s` 写回本文件；分析评估页会写回
> `assessment.zone` 与 `assessment.prediction_horizon_s`。因此刷新页面或重启程序后，
> 重建/评估面板保持上次设置，启动时的自动重建也使用这些参数。

### 7.1 synchronization — 时间步同步

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hz` | float | 10.0 | 主时间栅格频率（Hz） |
| `max_offset_s` | float | 0.6 | 时钟偏置搜索范围（秒） |
| `max_sample_gap_s` | float | 0.5 | 超过该采样间隔视为丢帧（秒） |

### 7.2 sensor_fusion — 多传感器融合

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gate_m` | float | 3.0 | 数据关联门限（米） |
| `default_sigma` | float | 1.5 | 未列出传感器的默认测量噪声（米） |
| `sigmas` | object | `{visible:1.0, infrared:1.5, radar:0.6, self:2.0}` | 各传感器测量噪声（米），用于逆方差加权 |

### 7.3 reconstruction — 三维重建

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_gap_s` | float | 0.8 | 允许补全的最长空洞（秒） |
| `smooth_window` | int | 5 | 加权平滑窗口（奇数，1=不平滑） |
| `outlier_sigma` | float | 3.5 | 异常点剔除的鲁棒阈值倍数 |

### 7.4 assessment — 智能评估

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `zone.center` | float[3] | `[8.0, 12.0, 4.0]` | 防护区中心（x, y, z），演示数据默认值 |
| `zone.radius` | float | 6.0 | 防护区水平半径（米） |
| `zone.y_min` / `y_max` | float | 5.0 / 25.0 | 防护区高度带（米） |
| `thresholds` | float[3] | `[75, 50, 25]` | 分级阈值（紧急/高/关注） |
| `weights` | object | 见下 | 六因子权重 |
| `prediction_horizon_s` | float | 2.0 | 评估时使用的预测视界（秒） |
| `prediction_points` | int | 10 | 预测点数 |

默认权重：`proximity 0.30 / closing 0.25 / altitude 0.10 / maneuver 0.10 / prediction 0.20 / dwell 0.05`。

---

## 八、检测 UI 与采集

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

## 九、detection — YOLO 目标检测参数

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

## 十、platforms — 各平台双目相机参数

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
