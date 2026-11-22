# KinFlow 轨迹预测模型训练

位于 `train/kinflow_predictor/` — **KinFlow**（学习式运动学先验 + 并行非自回归细化）
无人机轨迹预测模型的训练模块。实现依据 Applied Intelligence 稿件 APIN-D-26-14134。

## 架构概览

```text
历史轨迹 (t_i, p_i, Δt_i, v_i^est)   —— 允许不规则采样
  │
  ├─ 1. ContextEncoder   → 上下文 c ∈ R^128
  │     7 维特征 + 连续时间正弦编码 + 4 层 Pre-LN Transformer + CLS
  │
  ├─ 2. KinematicCorridor → 走廊先验 X0
  │     MLP(c, p_N) → (v0, a)，X0[k] = p_N + v0·Δτ_k + ½a·Δτ_k²（闭式，任意 τ）
  │
  ├─ 3. CorridorFlow     → 细化轨迹 X
  │     条件流匹配速度场，从 X0 出发并行 Euler 4 步（一次覆盖全部 K 步）
  │
  └─ 4. PhysicsProjector → 物理可行轨迹 X̂
        速度/垂直速率/高度硬投影，可微、非参数、3 次迭代
```

## 目录结构

```text
train/kinflow_predictor/
├── README.md                   # 本文件
├── __init__.py
├── dataset.py                  # 轨迹加载 + 滑动窗口采样 + 数据增强
├── generate_synthetic.py       # 四旋翼动力学合成轨迹生成器
├── train.py                    # 两阶段训练脚本（主入口）
├── dataset/
│   ├── train/                  # 训练轨迹 (.npz / .dat)
│   ├── valid/                  # 验证轨迹
│   ├── test/                   # 测试轨迹（训练后自动评估）
│   └── data/                   # 回退目录（无 train/ 时按 85/15 切分）
└── train_result/
    ├── best/                   # 各阶段最佳权重（推理服务优先读取）
    │   ├── kinflow_best_s1.pt
    │   └── kinflow_best_s2.pt
    └── run_<时间戳>/
        ├── models/             # 本次运行的检查点
        ├── 01_stage_loss.png
        ├── 02_global_loss.png
        ├── 03_accuracy.png
        ├── 04_physics.png
        ├── 05_lr_schedule.png
        ├── 06_dashboard.png
        ├── 07_test_metrics.png
        ├── training_history.json
        ├── training_summary.json
        └── test_metrics.json
```

## 数据格式

每条轨迹一个 `.npz` 文件：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `positions` | (N, 3) | float32 | 3D 位置 [x, y, z]，**y 为垂直轴** |
| `timestamps` | (N,) | float32 | 时间戳（秒，支持非等距） |

也兼容 `.dat`（每行 `x y z t`）。数据按**轨迹级**切分为 train/valid/test，
避免同一轨迹的窗口跨集合泄漏。

## 训练流程

### 阶段一 — 联合训练

```text
L = L_flow + λ_c·MSE(X0, Xgt) + λ_p·L_phy + λ_sample·MSE(X̂_sample, Xgt)
                                          λ_c=0.5, λ_p=0.01, λ_sample=0.3
```

- `L_flow`：条件流匹配，回归插值路径 `X_t = (1-t)X0 + tX1` 的切向 `X1 - X0`
- 走廊项：独立回归先验，避免生成目标主导先验拟合
- `L_phy`：在中点插值 `X_0.5` 的物理坐标上计算速度/垂直速率/高度/加速度的平方 hinge 违反量

### 阶段二 — 微调

- 学习率降至 `1e-4`
- 向走廊先验注入 `σ = 0.1` 噪声，使生成器见到推理时的输入分布
- 可选把可微投影器放进计算图（`projector_in_graph`），2 步采样 + 投影后与真值比较
- `λ_sample`：对 2 步 Euler 采样结果做 MSE 监督，缩小"训练用真值走廊、推理用预测走廊"的差距

### 优化设置

AdamW（weight decay `1e-4`）、梯度裁剪 max-norm `1.0`、warmup + 余弦退火、
默认批次 64、loss 非有限时跳过该批次。数据集规模建议至少数千个窗口；
默认窗口为 `ctx_len=40`（3.9 s）→ `tgt_len=20`（2 s），`k_max=20`。

## 常用命令

```bash
# 生成合成数据（200 条，自动切分 train/valid/test）
python train/kinflow_predictor/generate_synthetic.py 200

# 冒烟测试：CPU、2+1 epoch、512 训练窗口（约 1~3 分钟）
python train/kinflow_predictor/train.py --smoke

# 正式训练
python train/kinflow_predictor/train.py --stage all \
    --epochs-s1 60 --epochs-s2 20 --batch 128 --device cuda:0

# 仅阶段二微调（自动载入 train_result/best/kinflow_best_s1.pt）
python train/kinflow_predictor/train.py --stage 2 --epochs-s2 20 --device cuda:0

# 恢复训练
python train/kinflow_predictor/train.py --stage 2 \
    --resume train/kinflow_predictor/train_result/best/kinflow_best_s1.pt

# 小规模快速实验
python train/kinflow_predictor/train.py --stage 1 --epochs-s1 2 \
    --max-windows 512 --batch 32 --workers 0 --device cpu --no-charts

# 仅重建图表
python train/kinflow_predictor/train.py --charts-only train/kinflow_predictor/train_result/run_xxx
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--stage {1,2,all}` | 1=仅阶段一；2=阶段一+阶段二；all=同 2 |
| `--epochs-s1` / `--epochs-s2` / `--epochs` | 阶段轮数（`--epochs` 同时设置两阶段） |
| `--batch` / `--workers` / `--device` | 批次、DataLoader 进程数、设备 |
| `--lr-s1` / `--lr-s2` | 两阶段学习率 |
| `--ctx-len` / `--tgt-len` | 历史窗口 C / 预测窗口 K（同时更新 `k_max`） |
| `--euler-steps` / `--sigma` | 采样步数 / 阶段二走廊噪声 |
| `--max-windows` / `--max-val-windows` / `--max-test-windows` | 窗口数上限（冒烟、快速实验） |
| `--dataset-dir` / `--output-dir` | 数据集目录 / 输出目录 |
| `--resume` | 从检查点恢复 |
| `--no-projector-in-graph` | 阶段二不启用投影器在环 |
| `--no-charts` / `--charts-only` | 跳过图表 / 仅重建图表 |
| `--smoke` | 冒烟模式（CPU、2+1 epoch、512 窗口） |

## 训练输出

- 检查点：`kinflow_s{stage}_e{epoch}_v{loss}.pt`，最佳模型 `kinflow_best_s{stage}.pt`
  （同时复制到 `train_result/best/`，推理服务优先读取该目录）
- 测试集指标：ADE / FDE / minADE_8 / 违反率 / 单窗口推理延迟 → `test_metrics.json`
- 训练历史：`training_history.json`；摘要（含参数量、配置、测试指标）：`training_summary.json`

## 推理对齐（训练窗口规格）

训练会把 `ctx_len`（默认 40）、`sample_dt`（默认 0.1 s）与 `k_max`（默认 20）写入
检查点的 `model_info`。推理时模型据此把任意输入历史重采样到同一规格（末尾 3.9 s、
40 点等间隔），再在原生栅格（`k_max` 步 × `sample_dt` = 2 s）上并行生成。
**加载器必须把这些字段传给模型**，否则会用构造函数默认值（20 点）推理，
实测 ADE 会从 0.2 m 恶化到 1 m 以上。

- 输入历史可以是 10~50 点、0.1~0.2 s 不等间隔，预测尺度不会被归一化窗口拉偏；
- 视界 ≤ 2 s：直接在界面请求的时间戳上生成，支持 3~60 点、0.05~5 s 步长；
- 视界 > 2 s：前 2 s 为模型预测，其后由**常转率 + 恒定速度**的运动模型外推
  （转率按 6 s 指数衰减，保留初始曲率），逐点置信度 `exp(-(t-2)/8)` 衰减；
- 全段统一经物理投影器约束速度/垂直速率/高度。

## 当前权重与指标（2 s 视界）

| 指标 | 数值 |
|------|------|
| 测试集 ADE | 0.4946 m |
| 测试集 FDE | 1.3235 m |
| minADE_8（σ=0.1） | 0.4550 m |
| 最佳验证损失 | 0.2407（阶段二） |
| 参数量 | 371,916 |

权重位于 `train_result/best/kinflow_best_s{1,2}.pt`，并已复制到 `models/kinflow/`。
训练脚本只会用更好的验证损失覆盖 `train_result/best/`，冒烟/短训练不会覆盖已部署的权重。


## 与推理服务的关系

推理侧（`trajectory_reconstruction/core/prediction/prediction.py`）按以下顺序查找权重：

```text
train/kinflow_predictor/train_result/best/kinflow_best_s2.pt → s1.pt
models/kinflow/kinflow_best_s2.pt → s1.pt
models/kinflow/kinflow_s*_v{loss}.pt（按 loss 升序）
models/kinflow/kinflow.pt
```

权重缺失、模型禁用或推理异常时自动回退线性外推。

> 数学模型与设计取舍详见 [`docs/kinflow_prediction_model.md`](../../docs/kinflow_prediction_model.md)
