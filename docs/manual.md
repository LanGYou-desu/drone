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

系统包含两个模块：轨迹重建与分析模块 5 个功能页面（总览 / 预测 / 分析评估 / AI 策略 / 数据），无人机检测模块 3 个页面，顶部导航栏切换。支持深色/浅色双主题（设置页面切换）。

### 轨迹重建与分析 (:5000)

**总览** — 全屏 3D 轨迹视图。左侧面板切换平台显隐，并显示「综合重建」质量（帧数、参与传感器、融合不确定度、重建置信度、补洞/剔除）；可直接调整同步频率/关联门限/平滑窗口/最大补洞后一键重建（参数会保存，刷新/重启后保持本次设置）；综合轨迹以青色管状体渲染，场景中同时显示红色圆柱形防护区。底部播放控制条（隐藏的平台在播放时不会出现目标点）。

**操作：** 鼠标左键旋转 · 滚轮缩放 · 右键平移 · WASD 飞行 · QE 升降 · Space 播放 · 悬停轨迹点查看坐标和时间

**预测** — 默认预测重建后的综合轨迹（页面会显示"综合轨迹来自 N 路传感器重建"），也可选择其它平台。调整点数（3~60）与步长（0.05~5 s）后点击「开始预测」：
- 视界 ≤ 2 s：KinFlow 学习式模型并行生成，物理约束硬投影；
- 视界 > 2 s：前 2 s 为模型预测，其后由运动模型外推，面板会标注末端置信度；
- 模型不可用时回退线性外推。

所有平台预测时间自动对齐。

**分析评估** — 分析与评估合并：高度/速度/加速度/曲率四张图表、最佳捕捉时机，以及智能风险评估。设置防护区中心/半径/高度上限与预测视界后点击「重新评估」，输出风险总分（0~100）与等级（低/关注/高/紧急）、六因子拆解（空间接近/接近态势/高度威胁/机动强度/预测入侵/滞留时长）、风险时间线、关键事件与分级处置建议；同时显示证据置信度、传感器状态与 KinFlow 预测来源。

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
| `kinflow_model.enabled` | 是否启用 KinFlow 预测模型 |
| `kinflow_model.v_h_max` / `v_up_max` / `v_down_max` | 速度物理边界（水平/上升/下降） |
| `kinflow_model.y_min` / `y_max` | 高度边界（y 为垂直轴） |
| `kinflow_model.n_euler_steps` | 并行 Euler 采样步数（1-16，默认 4） |
| `kinflow_model.sigma` | 起点噪声（0=确定性预测，0.1=概率采样） |
| `kinflow_model.device` | 推理设备（cpu / cuda:0） |
| `synchronization.hz` | 主时间栅格频率（默认 10 Hz） |
| `sensor_fusion.gate_m` | 多传感器关联门限（默认 3 m） |
| `sensor_fusion.sigmas` | 各传感器测量噪声，用于逆方差融合 |
| `reconstruction.max_gap_s` / `smooth_window` | 重建补洞上限与平滑窗口 |
| `assessment.zone` | 防护区中心/半径/高度带 |
| `assessment.weights` / `thresholds` | 六因子权重与风险分级阈值 |
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
- 训练数据导入（放入 `train/kinflow_predictor/dataset/train/` 或 `valid/`）

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

### 轨迹预测模型 (KinFlow)

```bash
# 1. 生成合成训练数据
python train/kinflow_predictor/generate_synthetic.py 200

# 2. 冒烟测试（CPU，约 1~3 分钟，验证流水线可跑通）
python train/kinflow_predictor/train.py --smoke

# 3. 正式训练（阶段一联合训练 + 阶段二走廊噪声微调）
python train/kinflow_predictor/train.py --stage all \
    --epochs-s1 40 --epochs-s2 20 --batch 64 --device cuda:0

# 4. 仅微调（自动载入 train_result/best/kinflow_best_s1.pt）
python train/kinflow_predictor/train.py --stage 2 --epochs-s2 20 --device cuda:0

# 5. 恢复训练
python train/kinflow_predictor/train.py --stage 2 \
    --resume train/kinflow_predictor/train_result/best/kinflow_best_s1.pt

# 训练完成后权重自动保存到 train_result/best/，重启服务即生效
```

训练结果保存在 `train/kinflow_predictor/train_result/run_<时间戳>/`，包含：
- 阶段损失曲线（train/val/flow/corridor）
- 全局损失曲线、学习率调度曲线
- 验证集 ADE / FDE 曲线
- 物理约束违反率曲线（速度/垂直速率/高度/加速度）
- 综合仪表盘（一页概览）
- 测试集指标图表与 `test_metrics.json`

详细原理见 [`docs/kinflow_prediction_model.md`](kinflow_prediction_model.md)

## 常见问题

**预测不准？** 
- 检查是否启用了 KinFlow（`kinflow_model.enabled`），模型权重是否存在
- 无权重时自动回退到线性外推，精度有限
- 使用更多训练数据（真实 `.dat` 轨迹放入 `train/kinflow_predictor/dataset/train/`）可提升精度

**总览里的轨迹看起来是几条噪声带？**
- 细线是各传感器的原始观测（含测量噪声），青色管是重建后的综合轨迹，两者叠加属正常现象；
- 可在左侧调整同步频率/平滑窗口后点"重新重建综合轨迹"观察效果；参数会保存，刷新后仍保持。

**"参与传感器"少于总数？**
- 说明某传感器与其它传感器的中位距离长期超过关联门限（默认 3 m），被判定为不同目标而排除，
  这是"一致子集选择"的预期行为，可避免融合轨迹在多个目标之间来回跳；
- 演示数据由 `tools/make_runtime_demo.py` 生成，三路观测同一目标，正常情况下应为 3/3。

**评估分数怎么理解？**
- 0~100 是六因子加权和，不是概率；等级阈值为 75/50/25。分数高不代表误报风险高，
  证据置信度（传感器数量、融合不确定度）单独显示，置信度低时应人工复核；
- 修改防护区或预测视界后点「重新评估」即可刷新。

**检测不到无人机？** COCO 预训练模型无 drone 类，需训练自定义模型或使用 airplane/bird 代理。配置中 `target_classes` 设为 `[4, 14]` 可检测飞行器和飞鸟。

**预测后分析页没数据？** 需先在预测页执行一次预测（结果会落盘，分析/捕捉时机读取同一份文件）。
重启程序时 `initialize_data()` 会清空 `data/predict/`，重启后需重新预测；切换页面不影响。

**备份恢复没反应？** 恢复前确认选中了备份项。完整覆盖会清空全部数据后替换。

**自选平台不显示？** 上传 .dat 后自动出现，重启清空。

**切换主题？** 设置页面点击切换，自动保存到 localStorage。

**检测历史显示"空"？** 旧格式备份已自动迁移，新检测备份正常显示平台名。

**KinFlow 未加载？**
- 检查 `train/kinflow_predictor/train_result/best/kinflow_best_s*.pt` 或
  `models/kinflow/kinflow_best_s*.pt` 是否存在
- 查看控制台 `[预测]` 日志了解回退原因
- 历史点不足 5 个时自动回退到线性外推
- 设置页面可禁用 KinFlow 或调整物理边界 / Euler 步数 / 起点噪声

**训练时 loss 显示 NaN？**
- 训练脚本会自动跳过非有限损失的批次，不会中断训练
- 减少批次大小或降低学习率可提升稳定性
- 若频繁出现，检查输入数据是否含 NaN（`numpy.load` 后 `np.isfinite` 校验）

**如何恢复中断的训练？**
- `--resume train/kinflow_predictor/train_result/best/kinflow_best_s1.pt` 可恢复模型与阶段进度
- `--stage 2` 会自动载入 `train_result/best/kinflow_best_s1.pt`，跳过阶段一

**训练结果如何评估？**
- 每 epoch 输出总损失 + 流/走廊/物理分解 + 物理坐标 ADE/FDE + 约束违反率
- 训练结束后自动在测试集上评估，并额外给出 minADE_8 与单窗口推理延迟
- `--charts-only <结果目录>` 可仅重建图表而不重新训练
