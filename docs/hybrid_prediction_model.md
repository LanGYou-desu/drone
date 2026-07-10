# Phy-ODE-Diffusion 混合轨迹预测模型

> 融合扩散模型与 Transformer 的物理约束无人机轨迹预测方案

---

## 目录

1. [问题定义](#1-问题定义)
2. [数学模型](#2-数学模型)
   - [2.1 历史序列编码](#21-历史序列编码)
   - [2.2 物理结构化隐状态与 ODE 动力学](#22-物理结构化隐状态与-ode-动力学)
   - [2.3 ODE 状态初始化与闭环更新](#23-ode-状态初始化与闭环更新)
   - [2.4 物理引导扩散生成器](#24-物理引导扩散生成器)
   - [2.5 自回归预测闭环](#25-自回归预测闭环)
3. [模型架构](#3-模型架构)
4. [训练方法](#4-训练方法)
5. [代码结构](#5-代码结构)
6. [配置参数](#6-配置参数)
7. [API 使用方法](#7-api-使用方法)
8. [训练命令](#8-训练命令)
9. [输出文件](#9-输出文件)
10. [创新点](#10-创新点)

---

## 1. 问题定义

### 输入

- **历史轨迹**: 不等间隔的无人机位置序列 $\{(t_i, \mathbf{p}_i)\}_{i=1}^{N}$，其中 $\mathbf{p}_i = (x_i, y_i, z_i) \in \mathbb{R}^3$
- **预测参数**: 未来时间步长 $\Delta t$，预测步数 $K$

### 输出

自回归生成未来位置序列 $\mathbf{p}_{N+1}, \mathbf{p}_{N+2}, \dots, \mathbf{p}_{N+K}$

### 核心挑战

| 挑战 | 说明 |
|------|------|
| **不规则采样** | 各平台（可见光/红外/雷达）采样频率不同，时间间隔不等 |
| **长时自回归** | 预测 $K$ 步时误差随步数累积，可能发散 |
| **物理可行性** | 预测位置必须满足速度 $v \leq v_{\max}$、加速度 $a \leq a_{\max}$、高度 $z \geq z_{\min}$ 等硬约束 |

### 与线性外推的对比

| 维度 | 线性外推（旧方案） | Phy-ODE-Diffusion（新方案） |
|------|---------------------|------------------------------|
| 原理 | 基于最后两点的恒定速度外推 | Transformer 捕捉长程模式 + ODE 物理演化 + 扩散生成 |
| 历史利用 | 仅用最后 2 个点 | 利用全部历史序列（注意力机制） |
| 物理约束 | 无 | 速度/加速度/高度硬约束嵌入生成过程 |
| 不确定性 | 确定性（无变化） | 概率性（扩散天然支持多模态） |
| 自回归稳定性 | 差，易发散 | ODE+GRU 闭环校正抑制发散 |
| 不规则采样 | 不支持 | 连续时间编码天然支持 |

---

## 2. 数学模型

### 2.1 历史序列编码

#### 输入特征构造

对历史点 $i$（$i = 1, \dots, N$），构造特征向量：

$$\mathbf{f}_i = \text{Linear}\big( \Delta t_i,\ \mathbf{p}_i,\ \mathbf{v}_i^{\text{est}} \big) \in \mathbb{R}^{d_f}$$

其中：
- $\Delta t_i = t_i - t_{i-1}$（令 $\Delta t_1 = 0$），捕捉时间间隔信息
- $\mathbf{v}_i^{\text{est}}$ 由中心差分近似：$\mathbf{v}_i^{\text{est}} \approx \frac{\mathbf{p}_{i+1} - \mathbf{p}_{i-1}}{t_{i+1} - t_{i-1}}$

输入维度为 $1 + 3 + 3 = 7$，通过线性层投影到 $d_f = 64$ 维。

#### 连续时间位置编码

由于轨迹采样是不等间隔的，标准的位置编码（假设等间距）不适用。我们采用连续时间正弦编码：

$$\mathbf{e}_i = \text{TimeEmbedding}(t_i) = \big[\sin(\omega_1 t_i), \cos(\omega_1 t_i), \dots, \sin(\omega_{d_f/2} t_i), \cos(\omega_{d_f/2} t_i)\big]$$

其中频率 $\omega_k$ 在对数尺度上均匀分布：$\omega_k \in [1, f_{\max}]$，$f_{\max} = 10$ Hz。这种编码使 Transformer 能感知绝对时间值和相对时间差。

#### Transformer Encoder

输入 token 为 $\mathbf{f}_i + \mathbf{e}_i$，通过多层 Transformer Encoder（$L=3$ 层，$H=4$ 头），取前置 CLS token 的输出经投影得到上下文向量：

$$\mathbf{c} = \text{MLP}_{\text{ctx}}\big(\text{Transformer}(\text{CLS}, \mathbf{f}_1+\mathbf{e}_1, \dots, \mathbf{f}_N+\mathbf{e}_N)_{\text{CLS}}\big) \in \mathbb{R}^{d_c}$$

其中 $d_c = 128$。上下文向量 $\mathbf{c}$ 是整个历史轨迹的固定维度摘要。

---

### 2.2 物理结构化隐状态与 ODE 动力学

#### 状态定义

物理隐状态显式分解为位置、速度和潜在特征：

$$\mathbf{h}(t) = [\mathbf{p}(t), \mathbf{v}(t), \mathbf{z}(t)]^\top \in \mathbb{R}^{6 + d_z}$$

其中 $\mathbf{p}, \mathbf{v} \in \mathbb{R}^3$，$\mathbf{z} \in \mathbb{R}^{d_z}$ 为 $d_z = 32$ 维潜在特征。

#### 连续时间动力学（神经常微分方程）

状态演化由以下 ODE 规定：

$$\frac{d}{dt} \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \\ \mathbf{z} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ a_{\max} \cdot \tanh\!\big(\text{MLP}_a(\mathbf{p},\mathbf{v},\mathbf{z}) / a_{\max}\big) \\ \text{MLP}_z(\mathbf{p},\mathbf{v},\mathbf{z}) \end{bmatrix} = f_\theta(\mathbf{h})$$

**关键设计**:
- **运动学一致性**: $\frac{d\mathbf{p}}{dt} = \mathbf{v}$，强制位置与速度的正确物理关系
- **加速度软限幅**: $\tanh$ 函数将加速度限制在 $[-a_{\max}, a_{\max}]$（$a_{\max} = 30\ \text{m/s}^2$）
- **潜在动力学**: $\mathbf{z}$ 的演化由网络自由学习，捕获非运动学的高阶模式（如转弯意图、风扰等）

#### ODE 数值求解

使用 **4 阶 Runge-Kutta (RK4)** 方法从 $t_0$ 积分到 $t_1$（$n_{\text{steps}} = 8$ 个子步）：

$$\mathbf{h}(t_1) = \text{RK4}(f_\theta, \mathbf{h}(t_0), t_0, t_1)$$

RK4 的局部截断误差为 $\mathcal{O}(\Delta t^5)$，在 8 个子步内可提供足够的精度。

---

### 2.3 ODE 状态初始化与闭环更新

#### 初始化

利用上下文向量 $\mathbf{c}$ 和最后时刻的运动状态初始化 ODE：

$$\mathbf{h}(t_N) = \text{MLP}_{\text{init}}\big([\mathbf{p}_N, \mathbf{v}_N^{\text{est}}, \mathbf{c}]\big) \in \mathbb{R}^{6+d_z}$$

#### 演化（预测步）

在当前时刻 $t$，将状态演化为先验预测 $\mathbf{h}^-$（尚未融合新观测）：

$$\mathbf{h}^- = \text{ODESolve}(f_\theta, \mathbf{h}, t, t+\Delta t)$$

#### 更新（校正步）

获得新位置观测 $\mathbf{p}_{\text{obs}}$ 后（由扩散生成），计算虚拟观测速度 $\mathbf{v}_{\text{obs}} = (\mathbf{p}_{\text{obs}} - \mathbf{p}_t) / \Delta t$，然后通过 GRU 融合：

$$\mathbf{u} = \text{Encoder}(\Delta t, \mathbf{p}_{\text{obs}}, \mathbf{v}_{\text{obs}})$$
$$\mathbf{h}_{\text{new}} = \text{GRU}(\mathbf{h}^-, \mathbf{u})$$

这种 **先演化-后更新** 的设计类似于 Kalman 滤波的预测-校正框架，使模型既能遵循物理定律外推，又能用新数据修正不确定性。

---

### 2.4 物理引导扩散生成器

#### 为什么用扩散模型

在自回归预测中，每步需要根据 ODE 先验状态 $\mathbf{h}^-$ 生成下一时刻位置 $\mathbf{p}_{t+\Delta t}$。扩散模型具有以下优势：

- **生成质量高**: 去噪扩散过程自然逼近复杂条件分布
- **天然概率性**: 可以表达多模态不确定性（如分叉路段）
- **可引导**: 在去噪过程中可嵌入基于梯度的约束，强制物理可行性

#### 前向扩散过程

在 $\mathbb{R}^3$ 位置空间中进行扩散。给定清洁位置 $\mathbf{p}^{(0)} = \mathbf{p}_{t+\Delta t}$，前向过程为：

$$\mathbf{x}_\tau = \sqrt{\bar{\alpha}_\tau} \cdot \mathbf{p}^{(0)} + \sqrt{1-\bar{\alpha}_\tau} \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

其中 $\tau \in [0, T-1]$ 为扩散时间步（$T = 500$），$\bar{\alpha}_\tau$ 由余弦噪声调度定义。

#### 反向去噪（训练）

训练一个条件噪声预测网络 $\boldsymbol{\epsilon}_\phi(\mathbf{x}_\tau, \tau, \mathbf{h}^-, \Delta t)$：

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{\tau, \boldsymbol{\epsilon}}\left[ \|\boldsymbol{\epsilon}_\phi(\mathbf{x}_\tau, \tau, \mathbf{h}^-, \Delta t) - \boldsymbol{\epsilon}\|^2 \right]$$

噪声预测网络结构：
- 输入: $\mathbf{x}_\tau(3) + \tau\text{-emb}(16) + \mathbf{h}^-(38) + \Delta t\text{-emb}(16) = 73$
- 架构: 条件投影 → 4层 MLP ($128 \to 128 \to 128 \to 3$)

#### 推理时的物理引导

标准 DDPM/DDIM 采样仅保证数据分布拟合，不保证物理约束。引入 **基于梯度的分类器引导**，在去噪步修改预测噪声：

$$\tilde{\boldsymbol{\epsilon}}_\tau = \boldsymbol{\epsilon}_\phi(\mathbf{x}_\tau, \tau, \mathbf{h}^-, \Delta t) - \sqrt{1-\bar{\alpha}_\tau} \cdot \eta \cdot \nabla_{\mathbf{x}_\tau} \mathcal{C}(\hat{\mathbf{p}}^{(0)})$$

其中 $\hat{\mathbf{p}}^{(0)}(\mathbf{x}_\tau)$ 是从当前噪声样本估计的清洁位置：

$$\hat{\mathbf{p}}^{(0)}(\mathbf{x}_\tau) = \frac{1}{\sqrt{\bar{\alpha}_\tau}}\left(\mathbf{x}_\tau - \sqrt{1-\bar{\alpha}_\tau} \cdot \boldsymbol{\epsilon}_\phi(\mathbf{x}_\tau, \tau, \dots)\right)$$

#### 物理约束函数

$$\mathcal{C}(\mathbf{p}) = \lambda_v \max(0, \|\mathbf{v}\|-v_{\max})^2 + \lambda_a \max(0, \|\mathbf{a}\|-a_{\max})^2 + \lambda_z \max(0, z_{\min}-z)^2$$

其中:
- $v_{\max} = 30$ m/s（最大速度）
- $a_{\max} = 30$ m/s²（最大加速度）
- $z_{\min} = 0$ m（最小高度，即地面）
- 速度 $\mathbf{v} \approx (\mathbf{p} - \mathbf{p}_t) / \Delta t$（从上一时刻位置差分）
- 引导强度 $\eta = 0.1$

#### DDIM 加速采样

推理时使用确定性 DDIM 采样，$50$ 步即可生成高质量位置（训练时 $T=500$ 步）。

---

### 2.5 自回归预测闭环

第 $k$ 步预测（$k = 1, \dots, K$）:

| 步骤 | 操作 | 数学表达 |
|------|------|----------|
| 1 | 当前时刻和状态 | $t = t_N + (k-1)\Delta t$，ODE 状态 $\mathbf{h}$ |
| 2 | ODE 演化 | $\mathbf{h}^- = \text{ODESolve}(f_\theta, \mathbf{h}, t, t+\Delta t)$ |
| 3 | 物理引导扩散采样 | $\mathbf{p}_{t+\Delta t} = \text{DDIM-Guided}(\mathbf{h}^-, \Delta t, \mathbf{p}_t)$ |
| 4 | 计算虚拟观测速度 | $\mathbf{v}_{\text{obs}} = (\mathbf{p}_{t+\Delta t} - \mathbf{p}_t) / \Delta t$ |
| 5 | GRU 状态更新 | $\mathbf{h} \leftarrow \text{GRU}(\mathbf{h}^-, \text{Encoder}(\Delta t, \mathbf{p}_{t+\Delta t}, \mathbf{v}_{\text{obs}}))$ |
| 6 | 推进时间 | $\mathbf{p}_t = \mathbf{p}_{t+\Delta t}$，$t = t+\Delta t$ |

整个过程仅依赖历史和已预测点，满足严格的自回归要求。

---

## 3. 模型架构

```
                          ┌────────────────────────────────────┐
历史轨迹                    │  Transformer Encoder               │
{(t_i, p_i)} ──────────►  │  ┌─────────────────────────────┐   │
                          │  │ ContinuousTimeEncoding(t_i)  │   │
                          │  │ InputProj(dt_i, p_i, v_i)   │   │
                          │  │ CLS Token → 3×Self-Attn     │   │
                          │  └─────────────────────────────┘   │
                          │             ↓ 上下文向量 c          │
                          └────────────────┬───────────────────┘
                                           │
                          ┌────────────────▼───────────────────┐
                          │  ODE State Manager                  │
                          │  ┌────────────────────────────────┐ │
                          │  │ Init: h_0 = [p_N, v_N, MLP(c)] │ │
                          │  │        ┌───────────────────┐   │ │
                          │  │        │ Physics ODE Func   │   │ │
                          │  │        │ dp/dt = v          │   │ │
                          │  │        │ dv/dt = a_max·tanh │   │ │
                          │  │        │ dz/dt = MLP_z      │   │ │
                          │  │        └───────┬───────────┘   │ │
                          │  │           RK4 Solver (8 steps)  │ │
                          │  │                ↓  h_prior       │ │
                          │  │        ┌───────────────────┐   │ │
                          │  │        │GRU Update          │   │ │
                          │  │        │Obs_enc(dt,p_obs,v) │   │ │
                          │  │        │ → GRU(h_prior, u)  │   │ │
                          │  │        └───────────────────┘   │ │
                          │  └────────────────────────────────┘ │
                          └────────────────┬───────────────────┘
                                           │
                          ┌────────────────▼───────────────────┐
                          │  Guided Diffusion Predictor         │
                          │  ┌────────────────────────────────┐ │
                          │  │ NoiseScheduler (Cosine, T=500) │ │
                          │  │ NoisePredictionNet:            │ │
                          │  │   MLP(x_tau, τ, h_prior, Δt)  │ │
                          │  │        → ε_pred ∈ R³           │ │
                          │  │                                │ │
                          │  │ DDIM Sampling (50 steps)       │ │
                          │  │   + Physics Guidance Gradient  │ │
                          │  │   ∇_x C(p_pred) → corrected ε  │ │
                          │  └────────────────┬───────────────┘ │
                          │                  ↓ p_{t+Δt}         │
                          └────────────────────────────────────┘
                                           │
                          ┌────────────────▼───────────────────┐
                          │  自回归循环 (K 次)                   │
                          │  虚拟观测 → GRU → 更新 h → 下一轮   │
                          └────────────────────────────────────┘
```

### 模型参数统计

| 模块 | 参数量 (约) |
|------|-------------|
| Transformer Encoder | ~60,000 |
| Physics ODE | ~12,000 |
| ODE State Manager (GRU) | ~8,000 |
| Noise Prediction Net | ~40,000 |
| 其他 (嵌入/投影) | ~15,000 |
| **总计** | **~135,000** |

---

## 4. 训练方法

### 4.1 训练数据

#### 数据来源

| 来源 | 格式 | 说明 |
|------|------|------|
| 合成轨迹 | `.npz` | `generate_synthetic.py` 生成，含 4 种机动模式 |
| 用户导入 | `.dat` | 放入 `dataset/train/` 或 `dataset/valid/` |

#### 合成轨迹的机动模式

| 模式 | 占比 | 描述 |
|------|------|------|
| 巡航 (cruise) | ~35% | 匀速直线，缓慢随机转向 |
| 转弯 (turn) | ~25% | 绕 Y 轴水平转弯 |
| 爬升/下降 (climb) | ~20% | 高度变化 + 水平移动 |
| 加速/减速 (accel) | ~20% | 匀加速直线运动 |

#### 滑动窗口采样

从每条轨迹中滑动提取 (上下文, 目标) 对：

```
轨迹: [p₁, p₂, ..., p_N]
      ├── ctx_len ──┤├─ tgt_len ─┤
窗口: [p₁, ..., p_C] [p_{C+1}, ..., p_{C+M}]
步长: stride = min(ctx_len//2, tgt_len//2)
```

默认：`ctx_len=20, tgt_len=10`

#### 数据增强

- 随机旋转（绕 Y 轴）
- 随机缩放（±10%）
- 小幅高斯噪声（σ=0.01）

#### 标准化

每个窗口基于上下文统计独立标准化（零均值、单位方差），时间戳不变。

### 4.2 分阶段训练策略

由于系统包含多个异质模块，采用三阶段训练以稳定收敛。

#### 阶段一：Transformer + ODE + GRU（无扩散）

| 参数 | 值 |
|------|------|
| 训练模块 | Transformer Encoder、Physics ODE、GRU |
| 冻结模块 | Diffusion（全部参数） |
| 损失函数 | $\mathcal{L} = \text{MSE}(\mathbf{p}_{\text{pred}}, \mathbf{p}_{\text{gt}}) + 0.1 \cdot \mathcal{L}_{\text{phy}}$ |
| 预测方式 | 直接从 ODE 先验状态取位置分量 $\mathbf{h}^-[:3]$ |
| 学习率 | $1 \times 10^{-3}$，Cosine 退火 |
| 优化器 | AdamW，weight_decay=$1 \times 10^{-4}$ |
| 梯度裁剪 | max_norm=1.0 |

#### 阶段二：训练扩散模型

| 参数 | 值 |
|------|------|
| 训练模块 | Diffusion（NoisePredictionNet） |
| 冻结模块 | Transformer、Physics ODE、GRU（全部参数） |
| 损失函数 | $\mathcal{L} = \text{MSE}(\boldsymbol{\epsilon}_{\text{pred}}, \boldsymbol{\epsilon}_{\text{true}}) + 0.1 \cdot \mathcal{L}_{\text{phy}}$ |
| 教师强制 | 使用真实目标位置进行 ODE 状态更新 |
| 学习率 | $1 \times 10^{-3}$，Cosine 退火 |

#### 阶段三（可选）：联合微调

| 参数 | 值 |
|------|------|
| 训练模块 | 全部参数 |
| 计划采样 | 概率 $p_{\text{ss}}$ 从 0 线性增长到 0.5 |
| 学习率 | $1 \times 10^{-4}$，Cosine 退火 |

计划采样（Scheduled Sampling）：以概率 $p_{\text{ss}}$ 使用模型生成的（而非真实的）位置进行状态更新，使模型逐步适应自回归推理时的误差分布。

### 4.3 验证指标

| 指标 | 公式 | 说明 |
|------|------|------|
| ADE | $\frac{1}{K}\sum_{k=1}^{K}\|\hat{\mathbf{p}}_{N+k} - \mathbf{p}_{N+k}\|$ | 平均位移误差 |
| FDE | $\|\hat{\mathbf{p}}_{N+K} - \mathbf{p}_{N+K}\|$ | 最终位移误差 |
| Speed Violation | $\max(0, \|\mathbf{v}\| - v_{\max})$ | 速度违反 |
| Accel Violation | $\max(0, \|\mathbf{a}\| - a_{\max})$ | 加速度违反 |
| Height Violation | $\max(0, z_{\min} - z)$ | 高度违反 |

---

## 5. 代码结构

```
trajectory_reconstruction/core/prediction/
├── prediction.py                        # 预测入口（自动选择模型/线性外推）
└── hybrid/
    ├── __init__.py                      # 模块导出
    ├── model.py                         # PhyODEDiffusion 顶层模型
    │   ├── forward()                    # 训练前向（教师强制）
    │   ├── predict()                    # 推理（自回归预测）
    │   └── get_model_info()             # 模型元信息
    ├── transformer.py                   # TransformerEncoder + ContinuousTimeEncoding
    ├── physics_ode.py                   # PhysicsODEFunc + ODESolver(RK4)
    ├── diffusion.py                     # GuidedDiffusion + NoiseScheduler + NoisePredictionNet
    └── ode_manager.py                   # ODEStateManager (init/evolve/update)

train/hybrid_predictor/
├── dataset/
│   ├── train/                           # 训练数据（.npz 或用户导入 .dat）
│   └── valid/                           # 验证数据
├── dataset.py                           # TrajectoryDataset + 数据加载 + 速度估计
├── generate_synthetic.py                # 合成训练数据生成器
├── train.py                             # 分阶段训练脚本（含进度条和图表输出）
└── train_result/                        # 训练输出（图表 + 日志）

models/hybrid_predictor/
└── phy_ode_diffusion.pt                 # 训练后的模型权重
```

---

## 6. 配置参数

### 预测配置（`config.json` → `hybrid_model`）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用混合模型 |
| `v_max` | float | 30 | 最大速度约束 (m/s) |
| `a_max` | float | 30 | 最大加速度约束 (m/s²) |
| `z_min` | float | 0 | 最小高度约束 (m) |
| `guidance_eta` | float | 0.1 | 物理引导强度 |
| `inference_steps` | int | 50 | DDIM 推理步数 |
| `device` | str | `"cpu"` | 推理设备 (`cpu` / `cuda:0`) |

### 训练配置（`train.py` → `DEFAULT_CONFIG`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ctx_len` | 20 | Transformer 上下文长度 |
| `tgt_len` | 10 | 目标预测长度 |
| `batch_size` | 32 | 批次大小 |
| `lr_stage1` | 1e-3 | 阶段一学习率 |
| `lr_stage2` | 1e-3 | 阶段二学习率 |
| `lr_stage3` | 1e-4 | 阶段三学习率 |
| `epochs_stage1` | 50 | 阶段一轮数 |
| `epochs_stage2` | 100 | 阶段二轮数 |
| `epochs_stage3` | 20 | 阶段三轮数 |

### 模型超参（`PhyODEDiffusion.__init__`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_feat` | 64 | Transformer 特征维度 |
| `d_context` | 128 | 上下文向量维度 |
| `n_head` | 4 | 注意力头数 |
| `n_layers` | 3 | Transformer 层数 |
| `d_z` | 32 | ODE 潜在特征维度 |
| `n_diffusion_steps` | 500 | 训练时扩散步数 |
| `n_inference_steps` | 50 | 推理时 DDIM 步数 |

---

## 7. API 使用方法

### 自动使用（推荐）

启动服务后，预测 API 自动检测混合模型权重文件：

```
models/hybrid_predictor/phy_ode_diffusion.pt 存在 → 使用混合模型
                                             不存在 → 自动回退线性外推
```

无需修改任何前端或 API 代码。

### 手动调用

```python
from trajectory_reconstruction.core.prediction.prediction import (
    generate_prediction,           # 线性外推（始终可用）
    generate_prediction_hybrid,    # 混合模型（自动回退）
    is_hybrid_model_available,      # 检查权重文件
    load_hybrid_model,              # 手动加载模型
)

# 自动模式（推荐）
pred_pts, pred_times = generate_prediction_hybrid(
    points=history_points,      # [[x,y,z], ...]
    timestamps=history_times,   # [t, ...]
    num_points=20,              # 预测步数
    time_step=0.3,              # 时间步长
)

# 检查模型可用性
if is_hybrid_model_available():
    print("混合模型可用")
```

### 预测服务集成

`predict_service.py` 已自动集成，优先使用混合模型：

```python
def predict_single(method_id, num_points, time_step):
    if method_id != 'synthetic' and is_hybrid_model_available():
        pred_points, pred_times = generate_prediction_hybrid(...)
    else:
        pred_points, pred_times = generate_prediction(...)
```

> **注意**: 综合轨迹（`synthetic`）不做预测，仅对各平台分别预测后加权合成。

---

## 8. 训练命令

```bash
# ========== 1. 生成合成训练数据 ==========
python train/hybrid_predictor/generate_synthetic.py 200
# 输出: dataset/train/ (170) + dataset/valid/ (30)

# ========== 2. 分阶段训练 ==========

# GPU 环境（推荐使用 conda drone 环境）
conda activate drone

# 阶段一: Transformer + ODE + GRU
python train/hybrid_predictor/train.py --stage 1 --epochs 50 --batch 64 --device cuda:0

# 阶段二: 扩散模型（推荐到此为止，稳定收敛）
python train/hybrid_predictor/train.py --stage 2 --epochs 40 --batch 64 --device cuda:0

# 阶段三: 联合微调（可选，scheduled sampling 可能导致 NaN）
python train/hybrid_predictor/train.py --stage 3 --epochs 20 --batch 32 --device cuda:0

# ========== 3. CPU 训练（较慢）==========
python train/hybrid_predictor/train.py --stage 2 --epochs 30 --batch 32 --device cpu

# ========== 4. 恢复训练 ==========
python train/hybrid_predictor/train.py --stage 2 --resume models/hybrid_predictor/phy_ode_diffusion.pt

# ========== 注意事项 ==========
# - 每阶段结束自动保存权重到 models/hybrid_predictor/
# - NaN 异常自动跳过，阶段三检测到 NaN 会提前终止并保存
# - 推荐使用 --stage 2 跳过阶段三获得稳定模型
```

---

## 9. 输出文件

### 训练输出目录 `train/hybrid_predictor/train_result/`

```
train_result/
└── run_20260710_143052/
    ├── loss_curves.png          # 分阶段损失曲线
    ├── dashboard.png            # 训练仪表盘（4面板）
    ├── training_history.json    # 每个 epoch 的详细指标
    └── training_summary.json    # 训练摘要（最优指标、配置）
```

### 模型权重目录 `models/hybrid_predictor/`

```
models/hybrid_predictor/
├── phy_ode_diffusion.pt        # 最佳模型（推理时自动加载）
├── phy_ode_diffusion_s1_e10.pt # 阶段一 epoch 10 检查点
└── phy_ode_diffusion_s2_e50.pt # 阶段二 epoch 50 检查点
```

---

## 10. 创新点

| # | 创新点 | 说明 |
|---|--------|------|
| 1 | **Transformer 连续时间编码器** | 完全适应不规则采样，无需插值，注意力机制捕捉长程飞行模式 |
| 2 | **物理结构化 ODE 状态空间** | 显式分解 $\mathbf{p}, \mathbf{v}$，为扩散生成器提供运动学一致的强先验 |
| 3 | **物理引导扩散预测** | 在去噪过程中引入可微物理约束的梯度，将硬约束强制嵌入生成过程，避免事后修补 |
| 4 | **闭环 ODE 校正更新** | 将扩散生成的结果视为虚拟观测，通过 GRU 滤波机制修正隐状态，显著抑制自回归发散 |
| 5 | **分阶段训练策略** | 保证多模块复杂系统的稳定收敛，各阶段目标清晰，易于调试 |
| 6 | **轻量级 R³ 扩散** | 仅在位置空间中扩散（非图像），噪声预测网络为 MLP 而非 UNet，推理速度极快 |
