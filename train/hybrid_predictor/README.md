# Phy-ODE-Diffusion 轨迹预测模型训练

位于 `train/hybrid_predictor/` — 融合物理 ODE 与扩散模型的无人机轨迹预测训练模块。

## 架构概览

```
输入: 历史轨迹 (位置+时间戳)
  │
  ├─ 1. Transformer 编码器 ──→ 上下文向量 c
  │     (不规则时间序列编码)
  │
  ├─ 2. ODE 状态管理器 ──→ 先验状态 h_prior
  │     (Neural ODE 演化 + GRU 更新)
  │
  └─ 3. 物理引导扩散模型 ──→ 下一时刻位置 p_next
        (DDIM 采样 + 速度/加速度/高度约束引导)
```

## 目录结构

```
train/hybrid_predictor/
├── README.md                   # 本文件
├── __init__.py                 # 模块入口
├── train.py                    # 分阶段训练脚本（主入口）
├── dataset.py                  # 轨迹数据集 + 滑动窗口采样
├── generate_synthetic.py       # 四旋翼动力学生成合成数据
├── dataset/
│   ├── data/                   # 训练数据 (.npz 格式，优先加载)
│   ├── train/                  # 回退：训练集目录
│   └── valid/                  # 回退：验证集目录
└── train_result/               # 训练输出
    ├── models/                 # 模型权重（检查点）
    │   ├── phy_ode_diffusion_best_s1.pt
    │   └── phy_ode_diffusion_best_s2.pt
    ├── 01_loss_breakdown.png
    ├── 02_convergence_analysis.png
    ├── 03_stage_comparison.png
    ├── 04_ade_fde.png
    ├── 05_physics_metrics.png
    ├── 06_dashboard.png
    ├── training_history.json
    └── training_summary.json
```

## 数据格式

每条轨迹保存为一个 `.npz` 文件，包含两个数组：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `positions` | (N, 3) | float32 | 3D 位置序列 [x, y, z] |
| `timestamps` | (N,) | float32 | 时间戳序列（秒，支持非等距） |

也兼容 `.dat` 格式（每行 `x y z t`，空格分隔），优先使用 `.npz`。

> **提示**：数据归一化在数据集加载时自动完成（按整条轨迹统计量），训练与推理一致。

## 物理约束

训练和推理过程中施加以下四旋翼飞行约束：

| 约束 | 阈值 | 施加方式 |
|------|------|----------|
| 水平速度 ≤ 30 m/s | v_max | Stage1 损失惩罚 + 扩散物理正则 |
| 垂直上升 ≤ 5 m/s | v_v_up | 扩散物理正则 + 推理引导 |
| 垂直下降 ≤ 3 m/s | v_v_down | 扩散物理正则 + 推理引导 |
| 水平加速度 ≤ g·tan(35°) ≈ 6.9 m/s² | max_tilt | 扩散物理正则 + 推理引导 |
| 高度 ∈ [0, 120] m | z_min / z_max | 扩散物理正则 + 推理引导 |
| 合成数据速度 ≤ 20 m/s | v_h_max | 数据生成阶段内置 |

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy tqdm matplotlib
```

### 2. 生成合成训练数据

如无真实数据，使用内置生成器创建合成轨迹：

```bash
python train/hybrid_predictor/generate_synthetic.py 200
```

这会在 `dataset/train/` 和 `dataset/valid/` 下生成 200 条基于四旋翼动力学模型的轨迹。

### 3. 执行训练

`--stage` 语义：

| 参数 | 实际执行 | 说明 |
|------|---------|------|
| `--stage 1` | 阶段一 | 仅训练 Transformer + ODE + GRU |
| `--stage 2` | 阶段一 → 阶段二 | 先跑阶段一，再跑阶段二（扩散模型） |
| `--stage 3` | 阶段三 | 仅联合微调（需预训练权重） |
| `--stage all` | 阶段一 → 阶段二 → 阶段三 | 完整三阶段流水线 |

#### 完整训练

```bash
# CPU — 快速验证（每阶段只跑少量轮数）
python train/hybrid_predictor/train.py --stage all --epochs 5 --batch 16 --device cpu

# CPU — 完整训练（默认轮数: S1=50, S2=100, S3=20）
python train/hybrid_predictor/train.py --stage all --batch 32 --device cpu

# GPU — 完整训练
python train/hybrid_predictor/train.py --stage all --batch 64 --device cuda:0

# GPU — 自定义轮数（覆盖各阶段轮数: S1=80, S2=80, S3=40）
python train/hybrid_predictor/train.py --stage all --epochs 80 --batch 64 --device cuda:0
```

#### 单独执行

```bash
# ===== 只跑阶段一: Transformer + ODE + GRU =====
# CPU
python train/hybrid_predictor/train.py --stage 1 --epochs 50 --batch 32 --device cpu
# GPU
python train/hybrid_predictor/train.py --stage 1 --epochs 50 --batch 64 --device cuda:0

# ===== 跑阶段一+二: 编码器 + 扩散模型 =====
# CPU
python train/hybrid_predictor/train.py --stage 2 --epochs 100 --batch 32 --device cpu
# GPU
python train/hybrid_predictor/train.py --stage 2 --epochs 100 --batch 64 --device cuda:0

# ===== 只跑阶段三: 联合微调（需要已有预训练权重）=====
# CPU
python train/hybrid_predictor/train.py --stage 3 --epochs 20 --batch 16 --device cpu
# GPU
python train/hybrid_predictor/train.py --stage 3 --epochs 20 --batch 32 --device cuda:0
```

#### 分步执行（检查中间结果后继续）

```bash
# Step 1: 只跑阶段一，检查 loss 是否收敛
python train/hybrid_predictor/train.py --stage 1 --epochs 50 --batch 32 --device cpu

# Step 2: 在阶段一基础上继续跑阶段二（自动从已保存权重开始）
python train/hybrid_predictor/train.py --stage 2 --epochs 100 --batch 32 --device cpu

# Step 3: 联合微调
python train/hybrid_predictor/train.py --stage 3 --epochs 20 --batch 16 --device cpu
```

#### 从检查点恢复训练

```bash
# 恢复阶段一（保留 optimizer/scheduler 状态）
python train/hybrid_predictor/train.py --stage 1 \
    --resume train/hybrid_predictor/train_result/models/phy_ode_diffusion_best_s1.pt

# 恢复阶段一+二
python train/hybrid_predictor/train.py --stage 2 \
    --resume train/hybrid_predictor/train_result/models/phy_ode_diffusion_best_s1.pt

# 完整训练（自动跳过已完成阶段）
python train/hybrid_predictor/train.py --stage all \
    --resume train/hybrid_predictor/train_result/models/phy_ode_diffusion_best_s2.pt
```

#### 常用组合速查

| 场景 | 命令 |
|------|------|
| CPU 快速试跑 | `--stage all --epochs 5 --batch 16 --device cpu` |
| CPU 完整训练 | `--stage all --batch 32 --device cpu` |
| GPU 完整训练 | `--stage all --batch 64 --device cuda:0` |
| 只训编码器+ODE | `--stage 1 --epochs 50 --device cpu` |
| 训编码器+扩散 | `--stage 2 --epochs 100 --device cpu` |
| 只做微调 | `--stage 3 --epochs 20 --batch 16 --device cpu` |
| 断点续训 | `--stage all --resume <checkpoint.pt>` |
| 跳过图表 | 加 `--no-charts` |

## 三阶段训练说明

| 阶段 | 训练目标 | 冻结模块 | 默认轮数 | LR |
|------|---------|---------|---------|-----|
| **Stage 1** | Transformer + ODE + GRU (教师强制) | Diffusion | 50 | 1e-3 |
| **Stage 2** | 扩散模型（噪声预测） | Transformer + ODE + GRU | 100 | 1e-3 |
| **Stage 3** | 联合微调（计划采样） | 全部解冻 | 20 | 1e-4 |

- **Stage 1**: 直接 MSE 回归，学习基本的动力学演化
- **Stage 2**: 固定编码器/ODE，训练扩散模型去噪预测位置
- **Stage 3**: 端到端微调，逐步提高计划采样概率（0 → 0.5），使模型适应自回归推理

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage` | `all` | 训练阶段: `1`, `2`, `3`, `all` |
| `--epochs` | 按阶段 | 统一覆盖各阶段轮数 |
| `--batch` | 32 | 批次大小 |
| `--device` | `cpu` | 训练设备 (`cpu`, `cuda:0`) |
| `--lr` | 按阶段 | 统一覆盖各阶段学习率 |
| `--ctx-len` | 20 | 历史窗口长度 |
| `--tgt-len` | 10 | 预测目标长度 |
| `--resume` | — | 检查点路径（恢复完整训练状态） |
| `--warmup-s1` | 5 | 阶段一 warmup 轮数 |
| `--warmup-s2` | 3 | 阶段二 warmup 轮数 |
| `--warmup-s3` | 2 | 阶段三 warmup 轮数 |
| `--label-smoothing` | 0.005 | 标签平滑噪声强度 |
| `--no-charts` | — | 跳过图表生成 |

## 训练输出

训练完成后生成 **6 张图表** + **2 个 JSON 日志**：

| 文件 | 内容 |
|------|------|
| `01_loss_breakdown.png` | 各阶段 train/val 损失曲线 + 对数尺度 |
| `02_convergence_analysis.png` | 全局损失、过拟合检测、LR 调度、耗时分布 |
| `03_stage_comparison.png` | 阶段间柱状图 + 雷达图 + 箱线图对比 |
| `04_ade_fde.png` | ADE/FDE 预测精度指标曲线 |
| `05_physics_metrics.png` | 速度/加速度/高度违反率变化 |
| `06_dashboard.png` | 综合仪表盘（一页概览） |
| `training_history.json` | 每 epoch 详细指标 |
| `training_summary.json` | 训练摘要（最优指标 + 配置） |

训练完成后所有产物均在 `train/hybrid_predictor/train_result/` 下：
- `phy_ode_diffusion_best_s1.pt` — 阶段一最佳
- `phy_ode_diffusion_best_s2.pt` — 阶段二最佳
- 带 epoch/loss 的检查点文件用于断点续训

## 自定义数据

使用你自己的轨迹数据（替换 `dataset/data/` 中的文件）：

```python
import numpy as np

# 每条轨迹: (N, 3) positions + (N,) timestamps
positions = np.array([[x1,y1,z1], ...], dtype=np.float32)
timestamps = np.array([t1, ...], dtype=np.float32)
np.savez("train/hybrid_predictor/dataset/data/traj_custom_001.npz",
         positions=positions, timestamps=timestamps)
```

数据加载优先级：`data/` > `train/` + `valid/`。三个目录均无数据时报错退出。

## 注意事项

1. **数据质量**：轨迹应包含真实的飞行动态（加速/减速/转弯/爬升），纯匀速直线运动的训练效果差
2. **数据量**：建议至少 50-100 条轨迹，合成数据默认 200 条（train 170 + valid 30）
3. **序列长度**：每条轨迹 ≥ `ctx_len + tgt_len` 个点（默认 30 个点）
4. **非等距采样**：合成数据默认使用非等距时间戳，增强模型对不规则采样的鲁棒性
5. **显存需求**：约 255K 参数，batch_size=32 时约需 ~2GB 显存
6. **Windows 用户**：建议用 `--device cpu` 或确认 CUDA 可用后再用 GPU
