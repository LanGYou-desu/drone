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
   - [4.1 训练数据](#41-训练数据)
   - [4.2 分阶段训练策略](#42-分阶段训练策略)
   - [4.3 验证指标](#43-验证指标)
   - [4.4 恢复训练](#44-恢复训练)
5. [代码结构](#5-代码结构)
6. [配置参数](#6-配置参数)
7. [API 使用方法](#7-api-使用方法)
8. [训练命令](#8-训练命令)
9. [输出文件](#9-输出文件)
10. [关键设计决策与修复记录](#10-关键设计决策与修复记录)
11. [创新点](#11-创新点)

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
| 物理约束 | 无 | 速度/加速度/高度硬约束嵌入生成过程（反标准化到物理空间） |
| 不确定性 | 确定性（无变化） | 概率性（扩散天然支持多模态） |
| 自回归稳定性 | 差，易发散 | ODE+GRU 闭环校正抑制发散 |
| 不规则采样 | 不支持 | 连续时间编码天然支持 |
| 回退机制 | — | 权重缺失/禁用时自动回退线性外推 |

---

## 2. 数学模型

### 2.1 历史序列编码

#### 输入特征构造

对历史点 $i$（$i = 1, \dots, N$），构造特征向量：

$$\mathbf{f}_i = \text{Linear}\big( \Delta t_i,\ \mathbf{p}_i,\ \mathbf{v}_i^{\text{est}} \big) \in \mathbb{R}^{d_f}$$

其中：
- $\Delta t_i = t_i - t_{i-1}$（令 $\Delta t_1 = 0$），捕捉时间间隔信息
- $\mathbf{v}_i^{\text{est}}$ 由中心差分近似（内部点），端点用前向/后向差分避免边界失真
- 输入维度为 $1 + 3 + 3 = 7$，通过线性层投影到 $d_f = 64$ 维

#### 连续时间位置编码

由于轨迹采样是不等间隔的，标准的位置编码（假设等间距）不适用。采用连续时间正弦编码：

$$\mathbf{e}_i = \text{TimeEmbedding}(t_i) = \big[\sin(\omega_1 t_i), \cos(\omega_1 t_i), \dots, \sin(\omega_{d_f/2} t_i), \cos(\omega_{d_f/2} t_i)\big]$$

其中频率 $\omega_k$ 在对数尺度上均匀分布：$\omega_k \in [1, f_{\max}]$，$f_{\max} = 10$ Hz，乘以可学习的缩放因子。

#### Transformer Encoder

采用 **Pre-LN (Pre-Layer Normalization)** 架构：输入 token 为 $\mathbf{f}_i + \mathbf{e}_i$，前置可学习的 CLS token，通过 6 层 Transformer Encoder（4 头注意力，GELU 激活，Pre-LN 结构，batch_first）。Pre-LN 比 Post-LN 训练更稳定，梯度从顶层直通底层。取 CLS token 输出经 Tanh 投影：

$$\mathbf{c} = \text{MLP}_{\text{ctx}}\big(\text{Transformer}(\text{CLS}, \mathbf{f}_1+\mathbf{e}_1, \dots, \mathbf{f}_N+\mathbf{e}_N)_{\text{CLS}}\big) \in \mathbb{R}^{d_c}$$

其中 $d_c = 128$。上下文向量 $\mathbf{c}$ 是整个历史轨迹的固定维度摘要。

---

### 2.2 物理结构化隐状态与 ODE 动力学

#### 状态定义

物理隐状态显式分解为位置、速度和潜在特征：

$$\mathbf{h}(t) = [\mathbf{p}(t), \mathbf{v}(t), \mathbf{z}(t)]^\top \in \mathbb{R}^{6 + d_z}$$

其中 $\mathbf{p}, \mathbf{v} \in \mathbb{R}^3$，$\mathbf{z} \in \mathbb{R}^{d_z}$ 为 $d_z = 64$ 维潜在特征。

#### 连续时间动力学（神经常微分方程）

$$\frac{d}{dt} \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \\ \mathbf{z} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ a_{\max} \cdot \tanh\!\big(\text{MLP}_a(\mathbf{p},\mathbf{v},\mathbf{z}) / a_{\max}\big) \\ \text{MLP}_z(\mathbf{p},\mathbf{v},\mathbf{z}) \end{bmatrix} = f_\theta(\mathbf{h})$$

**关键设计**:
- **运动学一致性**: $\frac{d\mathbf{p}}{dt} = \mathbf{v}$，强制位置与速度的正确物理关系
- **加速度软限幅**: $\tanh$ 将加速度限制在 $[-a_{\max}, a_{\max}]$（默认 $a_{\max} = 30\ \text{m/s}^2$）
- **潜在动力学**: $\mathbf{z}$ 的演化由 MLP 自由学习，捕获高阶模式（转弯意图、风扰等）

#### ODE 数值求解

使用 **4 阶 Runge-Kutta (RK4)** 方法，$n_{\text{steps}} = 8$ 个子步：
$$\mathbf{h}(t_1) = \text{RK4}(f_\theta, \mathbf{h}(t_0), t_0, t_1)$$

RK4 局部截断误差 $\mathcal{O}(\Delta t^5)$，无需额外依赖（未使用 torchdiffeq）。

---

### 2.3 ODE 状态初始化与闭环更新

#### 初始化

$$\mathbf{h}(t_N) = \text{MLP}_{\text{init}}\big([\mathbf{p}_N, \mathbf{v}_N^{\text{est}}, \mathbf{c}]\big) \in \mathbb{R}^{6+d_z}$$

#### 演化（预测步）

$$\mathbf{h}^- = \text{ODESolve}(f_\theta, \mathbf{h}, t, t+\Delta t)$$

#### 更新（校正步）—— GRU 融合虚拟观测

观测编码器接受 10 维输入（$\Delta t$ + 位置 $\mathbf{p}_{\text{obs}}$ + 速度 $\mathbf{v}_{\text{obs}}$ + 加速度 $\mathbf{a}_{\text{obs}}$），生成观测特征 $\mathbf{u} \in \mathbb{R}^{32}$：

$$\mathbf{u} = \text{Encoder}(\Delta t, \mathbf{p}_{\text{obs}}, \mathbf{v}_{\text{obs}}, \mathbf{a}_{\text{obs}})$$
$$\mathbf{h}_{\text{new}} = \text{GRU}(\mathbf{h}^-, \mathbf{u})$$

其中 $\mathbf{a}_{\text{obs}} = (\mathbf{v}_{\text{obs}} - \mathbf{v}_{\text{prior}}) / \Delta t$，由观测速度与 ODE 先验速度差分得到。这种 **先演化-后更新** 的设计类似 Kalman 滤波的预测-校正框架。

---

### 2.4 物理引导扩散生成器

#### 前向扩散过程（R³ 空间）

给定清洁位置 $\mathbf{p}^{(0)} = \mathbf{p}_{t+\Delta t}$：

$$\mathbf{x}_\tau = \sqrt{\bar{\alpha}_\tau} \cdot \mathbf{p}^{(0)} + \sqrt{1-\bar{\alpha}_\tau} \cdot \mathbf{\epsilon}, \quad \mathbf{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

$\tau \in [0, T-1]$，$T = 500$，$\bar{\alpha}_\tau$ 由余弦调度定义。

#### 反向去噪（训练）

条件噪声预测网络 $\mathbf{\epsilon}_\phi(\mathbf{x}_\tau, \tau, \mathbf{h}^-, \Delta t)$：

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{\tau, \mathbf{\epsilon}}\left[ \|\mathbf{\epsilon}_\phi(\mathbf{x}_\tau, \tau, \mathbf{h}^-, \Delta t) - \mathbf{\epsilon}\|^2 \right]$$

网络结构：输入 $\mathbf{x}_\tau(3) + \tau\text{-emb}(16) + \mathbf{h}^-(70) + \Delta t\text{-emb}(16)$ → 条件投影 → 4 层 MLP($128 \to 128 \to 128 \to 3$)，最后一层初始化为零。

**物理正则**：训练时若提供归一化统计量 $(\mathbf{p}_{\text{mean}}, \mathbf{p}_{\text{std}})$，则反标准化后在原始物理空间计算约束代价，梯度通过 $1/\mathbf{p}_{\text{std}}$ 正确缩放回传。

#### 推理时的物理引导（反标准化）

推理时，`guided_sampling` 接收 $(\mathbf{p}_{\text{mean}}, \mathbf{p}_{\text{std}})$，在去噪过程中先将估计的清洁位置反标准化到物理空间，再计算物理约束梯度：

$$\tilde{\mathbf{\epsilon}}_\tau = \mathbf{\epsilon}_\phi(\mathbf{x}_\tau, \tau, \mathbf{h}^-, \Delta t) - \sqrt{1-\bar{\alpha}_\tau} \cdot \eta \cdot \nabla_{\mathbf{x}_\tau} \mathcal{C}(\hat{\mathbf{p}}^{(0)}_{\text{phys}})$$

其中 $\hat{\mathbf{p}}^{(0)}_{\text{phys}} = \hat{\mathbf{p}}^{(0)}_{\text{norm}} \cdot \mathbf{p}_{\text{std}} + \mathbf{p}_{\text{mean}}$

#### 四旋翼动力学约束

物理约束基于四旋翼飞行器的简化刚体动力学模型。四旋翼通过倾斜机体产生水平加速度：

$$\begin{aligned}
\dot{\mathbf{p}} &= \mathbf{v} \\
\dot{\mathbf{v}} &= \frac{T}{m} \cdot R(\phi,\theta) \cdot \mathbf{e}_{\text{up}} - g \cdot \mathbf{e}_{\text{up}}
\end{aligned}$$

其中 $T$ 为推力，$m$ 为质量，$R$ 为旋转矩阵（roll $\phi$, pitch $\theta$），$g$ 为重力加速度。

**水平运动**：倾斜机体使推力产生水平分量，最大水平加速度受倾斜角限制：
$$a_{h,\max} = g \cdot \tan(\phi_{\max}) \approx g \cdot \tan(35°) \approx 6.9\ \text{m/s}^2$$

**垂直运动**：推力与重力之差驱动升降，上下行速度不对称（受空气动力学和法规限制）：
$$v_{y,\max}^{\text{up}} = 5\ \text{m/s},\quad v_{y,\max}^{\text{down}} = 3\ \text{m/s}$$

**高度限制**：法规上限 120m，地面约束 1m。

#### 物理约束函数（原始物理空间，四旋翼扩展版）

$$\mathcal{C}(\mathbf{p}) = \mathcal{C}_v + \mathcal{C}_a + \mathcal{C}_z$$

**速度约束** $\mathcal{C}_v$（水平 + 上升 + 下降分离）：

$$\mathcal{C}_v = \lambda_v \max(0, \|\mathbf{v}_{xz}\|-v_{h,\max})^2 + \lambda_v \max(0, v_y - v_{v,\text{up}})^2 + \lambda_v \max(0, -v_y - v_{v,\text{down}})^2$$

**加速度约束** $\mathcal{C}_a$（水平加速度受倾斜角限制）：

$$\mathcal{C}_a = \lambda_a \max(0, \|\mathbf{a}_{xz}\| - g \cdot \tan(\phi_{\max}))^2$$

**高度约束** $\mathcal{C}_z$（上下界）：

$$\mathcal{C}_z = \lambda_z \max(0, z_{\min} - y)^2 + \lambda_{z,\max} \max(0, y - z_{\max})^2$$

- 参数从 `config.json` → `drone_dynamics` 和 `hybrid_model` 读取
- 速度 $\mathbf{v} \approx (\mathbf{p} - \mathbf{p}_t) / \Delta t$
- 推理时反标准化到物理空间后计算，引导强度 $\eta$ 默认 0.1

#### DDIM 加速采样

推理时使用确定性 DDIM，默认 $50$ 步（训练时 $T=500$ 步）。

---

### 2.5 自回归预测闭环

第 $k$ 步预测（$k = 1, \dots, K$）：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | $t = t_N + (k-1)\Delta t$ | 当前时间，ODE 状态 $\mathbf{h}$ |
| 2 | $\mathbf{h}^- = \text{ODESolve}(f_\theta, \mathbf{h}, t, t+\Delta t)$ | ODE 演化到下一时刻 |
| 3 | $\mathbf{p}_{t+\Delta t} = \text{DDIM-Guided}(\mathbf{h}^-, \Delta t, \mathbf{p}_t, \mathbf{p}_{\text{mean}}, \mathbf{p}_{\text{std}})$ | 物理引导扩散采样（反标准化空间） |
| 4 | $\mathbf{v}_{\text{obs}} = (\mathbf{p}_{t+\Delta t} - \mathbf{p}_t) / \Delta t$ | 虚拟观测速度 |
| 5 | $\mathbf{h} \leftarrow \text{GRU}(\mathbf{h}^-, \text{Encoder}(\Delta t, \mathbf{p}_{t+\Delta t}, \mathbf{v}_{\text{obs}}))$ | 闭环状态更新 |
| 6 | $\mathbf{p}_t = \mathbf{p}_{t+\Delta t}$，$t = t+\Delta t$ | 推进 |

全过程仅依赖历史+已预测点，严格自回归。预测输出含 NaN 保护：用最后有效值填充或回退到历史最后点。

---

## 3. 模型架构

```
                          ┌────────────────────────────────────┐
历史轨迹                    │  Transformer Encoder               │
{(t_i, p_i)} ──────────►  │  ┌─────────────────────────────┐   │
                          │  │ ContinuousTimeEncoding(t_i)  │   │
                          │  │ InputProj(dt_i, p_i, v_i)   │   │
                          │  │ CLS → 6×Self-Attn (4 heads) │   │
                          │  └─────────────────────────────┘   │
                          │             ↓ c ∈ R¹²⁸             │
                          └────────────────┬───────────────────┘
                                           │
                          ┌────────────────▼───────────────────┐
                          │  ODE State Manager                  │
                          │  ┌────────────────────────────────┐ │
                          │  │ Init: MLP([p_N, v_N, c]) → h₀       │ │
                          │  │        ┌───────────────────┐   │ │
                          │  │        │ Physics ODE Func   │   │ │
                          │  │        │ dp/dt = v          │   │ │
                          │  │        │ dv/dt = a_max·tanh │   │ │
                          │  │        │ dz/dt = MLP_z      │   │ │
                          │  │        └───────┬───────────┘   │ │
                          │  │         RK4 (8 steps)           │ │
                          │  │              ↓ h_prior          │ │
                          │  │        ┌───────────────────┐   │ │
                          │  │        │GRU Update          │   │ │
                          │  │        │ObsEnc(dt,p,v,a)    │   │ │
                          │  │        │→ GRU(h_prior, u)   │   │ │
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
                          │  │   + Physics Guidance (p_mean,  │ │
                          │  │     p_std → 反标准化梯度)       │ │
                          │  │   ∇_x C(p_phys) → corrected ε  │ │
                          │  └────────────────┬───────────────┘ │
                          │                  ↓ p_{t+Δt}         │
                          └────────────────────────────────────┘
                                           │
                          ┌────────────────▼───────────────────┐
                          │  自回归循环 (K 次)                   │
                          │  虚拟观测→GRU→更新h→下一轮          │
                          │  NaN保护：最后有效值填充            │
                          └────────────────────────────────────┘
```

### 模型参数统计

| 模块 | 参数量 |
|------|--------|
| Transformer Encoder (6 layers, 4 heads, d_feat=64) | ~309,000 |
| Physics ODE (MLP_a + MLP_z, d_z=64, hidden=128) | ~60,000 |
| ODE State Manager (GRU + ObsEncoder + Init Proj) | ~50,000 |
| Noise Prediction Net (4-layer MLP, hidden=128) | ~64,000 |
| **总计** | **~482,000** |

---

## 4. 训练方法

### 4.1 训练数据

#### 数据来源

| 来源 | 格式 | 说明 |
|------|------|------|
| 合成轨迹 | `.npz` | `generate_synthetic.py` 生成 |
| 用户导入 | `.dat` | 放入 `dataset/train/` 或 `dataset/valid/`（x y z t 格式） |

#### 合成轨迹的机动模式

| 模式 | 占比 | 描述 |
|------|------|------|
| 巡航 (cruise) | ~35% | 匀速直线，缓慢随机转向 |
| 转弯 (turn) | ~25% | 绕 Y 轴水平转弯 |
| 爬升/下降 (climb) | ~20% | 高度变化 + 水平移动 |
| 加速/减速 (accel) | ~20% | 匀加速直线运动 |

**平滑过渡**：模式切换后 0.5 秒内线性插值混合新旧 (speed, direction)，轨迹更加自然。

#### 滑动窗口采样

```
轨迹: [p₁, p₂, ..., p_N]
      ├── ctx_len ──┤├─ tgt_len ─┤
窗口: [p₁, ..., p_C] [p_{C+1}, ..., p_{C+M}]
步长: stride = min(ctx_len//2, tgt_len//2)
```

默认 `ctx_len=20, tgt_len=10`。每个窗口保存整条轨迹的归一化统计量 `(traj_mean, traj_std)`，确保训练与推理时归一化一致。

#### 数据增强

| 增强方式 | 说明 |
|----------|------|
| 绕 Y 轴旋转 | 仅在爬升角 < 30° 时应用，避免垂直轨迹失真 |
| 随机缩放 | ±10%，对位置和速度同步缩放 |
| 小幅加噪 | 幅度自适应数据标准差（0.5%），避免过度扰动 |

#### 标准化

使用**整条轨迹**的均值/标准差归一化（而非每窗口独立归一化），与推理时 `model.predict()` 的归一化方式一致，消除 train/inference 分布不匹配。

#### 速度与加速度估计

- 内部点：中心差分 `(p_{i+1} - p_{i-1}) / (t_{i+1} - t_{i-1})`（O(Δt²) 精度）
- 端点：前向/后向差分（O(Δt) 精度），而非简单复制相邻值
- 加速度：速度的时间差分

### 4.2 分阶段训练策略

#### 阶段一：Transformer + ODE + GRU（无扩散）

| 参数 | 值 |
|------|------|
| 训练模块 | Transformer Encoder、Physics ODE、GRU |
| 冻结模块 | Diffusion（全部参数） |
| 损失函数 | $\mathcal{L} = \text{MSE}(\mathbf{h}^-[:3], \mathbf{p}_{\text{gt}}) + 0.1 \cdot \mathcal{L}_{\text{phy}}$ |
| 标签平滑 | 对目标位置添加自适应高斯噪声（σ = `label_smoothing` × p_std），正则化防过拟合 |
| 学习率 | Warmup(5 epochs, 起始 0.1×) + Cosine 退火 |
| 优化器 | AdamW，weight_decay=$1 \times 10^{-4}$ |
| 梯度裁剪 | max_norm=1.0 |
| 验证指标 | MSE + ADE + FDE + Speed/Accel/Height 违反率 |

阶段一完成后自动保存完整检查点（含优化器/调度器状态）。

> **验证方式**（`_validate_stage1`）：使用 ODE 先验位置 `h_prior[:, :3]` 作为预测值，GRU 用真实轨迹更新（教师强制）。物理违反率反标准化到原始空间（m/s）后计算，速度/加速度取水平分量，高度同时检查上下限。

#### 阶段二：训练扩散模型

| 参数 | 值 |
|------|------|
| 训练模块 | Diffusion（NoisePredictionNet） |
| 冻结模块 | Transformer、Physics ODE、GRU（全部参数） |
| 损失函数 | $\mathcal{L}_{\text{diff}} = \text{MSE}(\mathbf{\epsilon}_{\text{pred}}, \mathbf{\epsilon}_{\text{true}}) + \lambda_{\text{phy}} \cdot \mathcal{C}(\mathbf{p}_{\text{phys}})$ |
| 物理正则 | 传入 `p_mean/p_std` 在原始物理空间计算，梯度通过 1/p_std 正确缩放回传 |
| 物理权重 | `physics_weight=0.01`，可在 config.json → training 中调整 |
| 验证指标 | Diff Loss + ADE + FDE + Speed/Accel/Height 违反率 |
| 学习率 | Warmup(3 epochs, 起始 0.1×) + Cosine 退火 |

阶段二完成后自动保存完整检查点。

> **验证方式**（`_validate_stage2`）：使用完整扩散采样 `guided_sampling()` 作为预测值（DDIM 50 步 + 物理引导），GRU 用真实轨迹更新（教师强制）。因此验证较慢，每个预测点需跑完整去噪过程。

#### 阶段三（可选）：联合微调

| 参数 | 值 |
|------|------|
| 训练模块 | 全部参数 |
| 计划采样 | $p_{\text{ss}}$ 从 0 线性增长到 0.5 |
| 采样方式 | **ODE 先验位置**（轻量，与阶段一一致），非完整扩散采样（避免昂贵且不参与梯度的计算） |
| 学习率 | $1 \times 10^{-4}$，Cosine 退火 |
| 验证指标 | 位置 MSE + ADE + FDE（与联合训练目标一致） |
| NaN 保护 | 跳过 NaN 梯度更新；检测到全 epoch NaN 时提前终止并保存 |

> **注意**: 阶段三使用 ODE 先验位置（`h_prior[:, :3]`）而非扩散采样作为计划采样的生成样本，因为扩散采样在训练中不参与梯度回传却消耗大量计算。

> **验证方式**（`_validate_stage3`）：使用完整扩散采样 + **自回归** GRU 更新（用预测位置估计速度，不回退真实值）。与 `_validate_stage2` 的最大区别是模拟真实推理的闭环行为，因此验证 loss 会高于阶段二，但更能反映实际部署精度。

### 4.3 Warmup 学习率策略

三个阶段采用独立的 warmup + cosine 退火策略，避免训练初期梯度震荡：

| 阶段 | warmup 轮数 | 起始 LR 因子 | 说明 |
|------|-------------|-------------|------|
| 阶段一 | 5 | 0.1 | Transformer + ODE 需稳定初始化，LR 从 $0.1\times 10^{-3}$ 线性增长到 $10^{-3}$ |
| 阶段二 | 3 | 0.1 | 扩散模型对初始 LR 敏感，短 warmup 后进入 Cosine 退火 |
| 阶段三 | 2 | 0.1 | 全参数微调已有良好基础，短 warmup 即可 |

**调度曲线**：

```
LR
│  warmup (线性增长)     cosine 退火
│  ╱                     ╲
│ ╱                       ╲
│╱                         ╲___
├────────────────────────────── epochs
│← warmup →←─── T_max ──────→
```

CLI 参数：`--warmup-s1 5 --warmup-s2 3 --warmup-s3 2 --warmup-factor 0.1`

### 4.4 验证指标

训练时每 epoch 输出：

| 指标 | 公式 | 说明 |
|------|------|------|
| **MSE** | $\frac{1}{K}\sum\|\hat{\mathbf{p}}_k - \mathbf{p}_k\|^2$ | 均方位置误差（阶段一/三） |
| **ADE** | $\frac{1}{K}\sum_{k=1}^{K}\|\hat{\mathbf{p}}_{N+k} - \mathbf{p}_{N+k}\|$ | 平均位移误差 |
| **FDE** | $\|\hat{\mathbf{p}}_{N+K} - \mathbf{p}_{N+K}\|$ | 最终位移误差（终点精度） |
| **Speed Violation** | $\max(0, \|\mathbf{v}_{xz}\| - v_{\max})$ | 水平速度约束违反（XZ 平面） |
| **Accel Violation** | $\max(0, \|\mathbf{a}\| - a_{\max})$ | 加速度约束违反 |
| **Height Violation** | $\max(0, z_{\min} - z) + \max(0, z - z_{\max})$ | 高度约束违反（下限+上限） |

### 4.5 恢复训练

`--resume` 参数支持完整恢复：

1. 加载 `model_state_dict`
2. 恢复 `optimizer_state_dict`（学习率/动量不丢失）
3. 恢复 `scheduler_state_dict`（Cosine 退火延续）
4. 从 `checkpoint["epoch"] + 1` 继续训练
5. 自动跳过已完成的阶段（`resumed_stage` 检测）

```
# 从阶段二最佳模型恢复
python train/hybrid_predictor/train.py --stage 2 --resume train/hybrid_predictor/train_result/models/phy_ode_diffusion_best_s2.pt
```

---

## 5. 代码结构

```
trajectory_reconstruction/core/prediction/
├── prediction.py                           # 预测入口
│   ├── generate_prediction()               # 线性外推（兜底，始终可用）
│   ├── generate_prediction_hybrid()        # 混合模型（自动回退）
│   ├── load_hybrid_model()                 # 模型加载（含缓存 + config读取）
│   ├── _get_model_path()                   # 智能查找最新/最佳权重
│   └── is_hybrid_model_available()         # 权重文件检查
│
└── hybrid/
    ├── __init__.py
    ├── model.py                            # PhyODEDiffusion 顶层
    │   ├── __init__()                      # 子模块组装
    │   ├── forward()                       # 训练前向（教师强制，接收 p_mean/p_std）
    │   ├── predict()                       # 推理（自回归 + 反标准化 + NaN保护）
    │   └── get_model_info()                # 完整模型元信息（含架构参数）
    │
    ├── transformer.py                      # 不规则时间序列编码
    │   ├── ContinuousTimeEncoding          # 连续时间正弦位置编码
    │   └── TransformerEncoder              # 6层 Transformer + CLS token
    │
    ├── physics_ode.py                      # 物理 ODE 动力学
    │   ├── PhysicsODEFunc                  # f_θ: dp/dt=v, dv/dt=a_max·tanh, dz/dt=MLP
    │   └── ODESolver                       # RK4 定步长求解器（8子步）
    │
    ├── diffusion.py                        # 物理引导扩散
    │   ├── NoiseScheduler                  # Cosine 调度 + DDIM 采样
    │   ├── SinusoidalEmbedding             # 扩散时间步 τ 编码
    │   ├── NoisePredictionNet              # 条件 MLP（x_tau, τ, h_prior, Δt → ε ∈ R³）
    │   ├── GuidedDiffusion                 # 扩散生成器
    │   │   ├── compute_loss()              # 训练损失（含反标准化物理正则）
    │   │   ├── guided_sampling()           # DDIM 推理（含反标准化物理引导）
    │   │   ├── _compute_guidance()         # 物理约束梯度计算
    │   │   └── _physics_cost()             # C(p): 速度/加速度/高度违反代价
    │
    └── ode_manager.py                      # ODE 状态管理
        ├── ObservationEncoder              # (Δt, p, v, a) → u (10维输入)
        ├── ODEStateManager.init_state()    # h₀ = [p_N, v_N, MLP(c)]
        ├── ODEStateManager.evolve()        # RK4 演化 → h_prior
        └── ODEStateManager.update()        # GRU 融合虚拟观测 → h_new

train/hybrid_predictor/
├── dataset/
│   ├── train/                              # 训练数据 (.npz / .dat)
│   └── valid/                              # 验证数据
├── dataset.py
│   ├── estimate_velocity()                 # 速度估计（中心差分 + 端点前向/后向）
│   ├── load_trajectory_from_dat()          # .dat 文件加载
│   ├── load_all_trajectories()             # 批量加载（.npz + .dat）
│   ├── sliding_window_sample()             # 窗口采样（保存 traj_mean/std）
│   ├── TrajectoryDataset                   # PyTorch Dataset（含增强 + 归一化）
│   └── collate_fn()                        # 批次整理
├── generate_synthetic.py                   # 合成轨迹生成（4种机动 + 平滑过渡）
├── train.py                                # 训练脚本
│   ├── TrainingLogger                      # 训练日志（支持 ADE/FDE）
│   ├── save_checkpoint()                   # 完整检查点（含 optimizer/scheduler）
│   ├── _check_nan()                        # NaN 梯度保护
│   ├── _to_device()                        # 批次设备转移
│   ├── build_dataloaders()                 # 数据加载器构建
│   ├── train_stage1/2/3()                  # 三阶段训练（支持 resume_ckpt）
│   ├── _validate_stage1/2/3()              # 验证函数（返回 loss+ADE+FDE+物理违反率）
│   └── _plot_all_charts()                  # 5张训练图表生成
└── train_result/                           # 训练输出（图表+日志）

train/hybrid_predictor/train_result/models/
├── phy_ode_diffusion_s1_e50_v0.1234.pt     # 阶段一检查点（含描述）
├── phy_ode_diffusion_s2_e100_v0.0456.pt    # 阶段二检查点
├── phy_ode_diffusion_best_s1.pt             # 阶段一最佳
├── phy_ode_diffusion_best_s2.pt             # 阶段二最佳
└── phy_ode_diffusion.pt                     # 兼容
```

---

## 6. 配置参数

### 运行时配置（`config.json` → `hybrid_model`，设置页面可调）

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `enabled` | bool | `true` | — | 是否启用混合模型（权重不存在时自动回退） |
| `v_max` | float | 30 | 5–40 | 最大水平速度 (m/s) |
| `v_v_up` | float | 5.0 | 1–10 | 最大垂直上升速度 (m/s) |
| `v_v_down` | float | 3.0 | 1–10 | 最大垂直下降速度 (m/s) |
| `a_max` | float | 30 | 5–50 | 最大加速度 (m/s²) |
| `max_tilt` | float | 35 | 10–60 | 最大倾斜角 (deg) |
| `z_min` | float | 0 | 0–30 | 最低高度 (m) |
| `z_max` | float | 120 | 30–250 | 最高高度 (m) |
| `guidance_eta` | float | 0.1 | 0.01–0.50 | 物理引导强度 |
| `inference_steps` | int | 50 | 10–200 | DDIM 推理步数 |
| `device` | str | `"cpu"` | cpu/cuda:0 | 推理设备 |

### 四旋翼动力学配置（`config.json` → `drone_dynamics`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `g` | 9.81 | 重力加速度 (m/s²) |
| `v_h_max` | 20.0 | 最大水平速度 (m/s) |
| `v_v_up` | 5.0 | 最大垂直上升速度 (m/s) |
| `v_v_down` | 3.0 | 最大垂直下降速度 (m/s) |
| `max_tilt` | 35.0 | 最大倾斜角 (deg)，限制 a_h ≤ g·tan(max_tilt) |
| `max_alt` | 120.0 | 最大飞行高度 (m) |
| `min_alt` | 1.0 | 最低飞行高度 (m) |
| `thrust_max` | 25.0 | 最大推力加速度 (m/s²) |
| `thrust_hover` | 9.81 | 悬停推力加速度 = g |

### 训练配置（`train.py` → `DEFAULT_CONFIG` / `config.json` → `training`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ctx_len` | 20 | Transformer 上下文长度 |
| `tgt_len` | 10 | 目标预测长度 |
| `batch_size` | 32 | 批次大小 |
| `lr_stage1` | 1e-3 | 阶段一学习率（warmup 峰值） |
| `lr_stage2` | 1e-3 | 阶段二学习率（warmup 峰值） |
| `lr_stage3` | 1e-4 | 阶段三学习率（warmup 峰值） |
| `weight_decay` | 1e-4 | AdamW 权重衰减 |
| `epochs_stage1` | 50 | 阶段一轮数 |
| `epochs_stage2` | 100 | 阶段二轮数 |
| `epochs_stage3` | 20 | 阶段三轮数 |
| `warmup_epochs_s1` | 5 | 阶段一 warmup 轮数 |
| `warmup_epochs_s2` | 3 | 阶段二 warmup 轮数 |
| `warmup_epochs_s3` | 2 | 阶段三 warmup 轮数 |
| `warmup_start_factor` | 0.1 | warmup 起始 LR = base_lr × factor |
| `label_smoothing` | 0.005 | 标签平滑噪声系数（0=关闭） |

### 模型超参（训练默认值，可由 config.json → training 覆盖）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_feat` | 64 | Transformer 特征维度 |
| `d_context` | 128 | 上下文向量维度 |
| `n_head` | 4 | 注意力头数 |
| `n_layers` | 6 | Transformer 层数 |
| `dim_feedforward` | 256 | FFN 维度 |
| `d_z` | 64 | ODE 潜在特征维度 |
| `ode_hidden_dim` | 128 | ODE MLP 隐藏层维度 |
| `n_diffusion_steps` | 500 | 训练时扩散步数 |
| `n_inference_steps` | 50 | 推理时 DDIM 步数 |
| `a_max` | 30.0 | 加速度软限幅 (m/s²) |
| `v_max` | 30.0 | 速度约束 (m/s) |
| `z_min` | 0.0 | 高度下限 (m) |
| `z_max` | 120.0 | 高度上限 (m) |
| `physics_weight` | 0.01 | 物理损失权重（阶段二/三） |

---

## 7. API 使用方法

### 自动模式（推荐）

启动服务后，预测 API 自动检测模型权重文件：

```python
# 优先级: best_s* > s*_e* > 旧命名
_get_model_path()  →  自动查找 models/hybrid_predictor/

权重存在 + enabled=true  →  使用混合模型
权重不存在或 disabled    →  自动回退线性外推
CUDA 不可用              →  自动回退 CPU
```

### 手动调用

```python
from trajectory_reconstruction.core.prediction.prediction import (
    generate_prediction,            # 线性外推（始终可用）
    generate_prediction_hybrid,     # 混合模型（自动回退）
    is_hybrid_model_available,      # 检查权重文件
    load_hybrid_model,              # 手动加载（可指定 device）
)

# 自动模式（推荐）
pred_pts, pred_times = generate_prediction_hybrid(
    points=history_points,       # [[x,y,z], ...]
    timestamps=history_times,    # [t, ...]
    num_points=20,               # 预测步数（用户输入）
    time_step=0.3,               # 时间步长
)
# 自动选择：权重可用 + 历史≥10点 → 混合模型，否则 → 线性外推
```

### 预测服务集成

`predict_service.py` 已自动集成：

```python
def predict_single(method_id, num_points, time_step):
    # 综合轨迹(synthetic)不参与预测
    if method_id != 'synthetic' and is_hybrid_model_available():
        cfg = ensure_config()
        device = cfg.get('hybrid_model', {}).get('device', 'cpu')
        return generate_prediction_hybrid(points, timestamps, num_points, time_step, device=device)
    else:
        return generate_prediction(points, timestamps, num_points, time_step)
```

---

## 8. 训练命令

```bash
# ========== 1. 生成合成训练数据 ==========
python train/hybrid_predictor/generate_synthetic.py 200
# 输出: dataset/train/ (~170) + dataset/valid/ (~30)

# ========== 2. GPU 训练（推荐 conda drone 环境）==========
conda activate drone

# 阶段一+二（推荐，稳定收敛）
python train/hybrid_predictor/train.py --stage 2 --epochs 40 --batch 64 --device cuda:0

# 阶段三联合微调（可选）
python train/hybrid_predictor/train.py --stage 3 --epochs 20 --batch 32 --device cuda:0

# ========== 3. 恢复训练 ==========
python train/hybrid_predictor/train.py --stage 2 --resume train/hybrid_predictor/train_result/models/phy_ode_diffusion_best_s2.pt

# ========== 4. CPU 训练 ==========
python train/hybrid_predictor/train.py --stage 2 --epochs 30 --batch 32 --device cpu

# ========== 5. 快速测试（少数据、少轮次）==========
python train/hybrid_predictor/generate_synthetic.py 30
python train/hybrid_predictor/train.py --stage 2 --epochs 5 --batch 32 --device cuda:0 --ctx-len 10 --tgt-len 5

# --stage 语义:
#   1    = 仅阶段一
#   2    = 阶段一 → 阶段二（推荐）
#   3    = 仅阶段三
#   all  = 全部三个阶段
```

---

## 9. 输出文件

### 训练输出目录 `train/hybrid_predictor/train_result/`

| 文件 | 内容 | 对应指标 |
|------|------|----------|
| `01_loss_breakdown.png` | 每阶段 train/val 损失曲线 + 对数尺度 | MSE |
| `02_convergence_analysis.png` | 全局损失、val/train 比、损失下降率、LR 衰减、累积耗时 | 过拟合检测 |
| `03_stage_comparison.png` | 柱状图对比 + 雷达图 + 箱线图 | 阶段综合 |
| `04_ade_fde.png` | ADE/FDE 分阶段曲线、ADE vs FDE 散点图、ADE 改进率 | ADE、FDE |
| `05_physics_metrics.png` | 速度/加速度/高度违反率趋势 + 综合物理得分 | Speed/Accel/Height Violation |
| `06_dashboard.png` | 全景仪表盘：损失总览、LR 调度、时间饼图、损失直方图 | 综合概览 |
| `training_history.json` | 每 epoch 的 train_loss/val_loss/ADE/FDE/lr/time + 物理违反率 | 全部指标 |
| `training_summary.json` | 各阶段最优指标、总参数、总耗时 | 摘要 |

### 模型权重目录 `train/hybrid_predictor/train_result/models/`

| 文件 | 说明 |
|------|------|
| `phy_ode_diffusion_s1_e50_v0.1234.pt` | 阶段一 epoch 50 检查点（含优化器/调度器状态） |
| `phy_ode_diffusion_s2_e100_v0.0456.pt` | 阶段二 epoch 100 检查点 |
| `phy_ode_diffusion_best_s1.pt` | 阶段一最佳模型 |
| `phy_ode_diffusion_best_s2.pt` | 阶段二最佳模型 |
| `phy_ode_diffusion.pt` | 兼容 |

训练完成后如需部署，将权重复制到 `models/hybrid_predictor/`。推理时自动查找优先级：`best_s*` > `s*_e*` > 旧命名。

---

## 10. 关键设计决策与修复记录

### 归一化一致性

**问题**: 训练时每窗口独立归一化，推理时整条轨迹归一化，分布不匹配。

**方案**: 训练窗口保存整条轨迹的 `(traj_mean, traj_std)` 进行归一化，与推理完全一致。

### 物理约束反标准化

**问题**: 物理约束在标准化空间计算（无量纲），`v_max=30 m/s` 与实际值不匹配。

**方案**: 训练 `compute_loss` 和推理 `guided_sampling` 均接收 `(p_mean, p_std)`，反标准化后在原始物理空间计算。

### 速度估计端点

**问题**: 端点速度直接复制相邻值 `vel[0]=vel[1]`, `vel[-1]=vel[-2]`，不准确。

**方案**: 改用前向差分 `(p[1]-p[0])/dt` 和后向差分 `(p[-1]-p[-2])/dt`（O(Δt) 精度）。

### 数据增强旋转

**问题**: 无条件绕 Y 轴旋转，爬升轨迹被错误投影到水平面。

**方案**: 用爬升角 `arctan2(v_y, v_xz) < 30°` 判断，仅水平运动为主时旋转。

### 检查点管理

**问题**: 仅保存单一文件，最佳模型被覆盖，不支持恢复优化器状态。

**方案**:
- 命名含描述: `phy_ode_diffusion_s{stage}_e{epoch}_v{loss}.pt`
- 最佳模型单独保存不被覆盖: `phy_ode_diffusion_best_s{stage}.pt`
- 含 `optimizer_state_dict` + `scheduler_state_dict`
- `--resume` 恢复完整训练状态

### 阶段三优化

**问题**: 训练循环内调用完整扩散采样（50步DDIM+物理引导），但不参与梯度回传，严重拖慢训练。

**方案**: 计划采样时使用 ODE 先验位置 `h_prior[:, :3]` 替代扩散采样，与阶段一一致。扩散采样仅用于最终推理。

### 观测编码器

**问题**: 仅接受 `(Δt, p, v)` 7维输入，信息有限。

**方案**: 扩展到 `(Δt, p, v, a)` 10维输入，加速度由速度差分得到，为 GRU 提供更丰富的运动信息。

### 合成数据平滑过渡

**问题**: 机动模式切换时速度和方向突变，轨迹不够真实。

**方案**: 切换后 0.5 秒内线性插值混合新旧 `(speed, direction)`。

### Pre-LN Transformer

**问题**: Post-LN（`norm_first=False`）残差路径上有两个 LayerNorm，深层梯度衰减明显。

**方案**: 采用 Pre-LN（`norm_first=True`），残差路径上无 LayerNorm，梯度从顶层直通底层，训练更稳定收敛更快。

### 标签平滑（回归正则化）

**问题**: 回归任务没有分类任务中的标签平滑机制，模型可能过拟合精确坐标。

**方案**: 对目标位置添加自适应高斯噪声（σ = `label_smoothing` × p_std），噪声幅度与数据尺度匹配。仅在阶段一确定性回归中使用，阶段二/三扩散目标不做平滑。

### Warmup 学习率调度

**问题**: 训练初期 LR 直接从峰值开始，梯度震荡大，不利于复杂多模块系统收敛。

**方案**: 三阶段独立 warmup：阶段一 5 epoch（Transformer+ODE 需稳定初始化），阶段二 3 epoch（扩散模型对 LR 敏感），阶段三 2 epoch（微调已有基础）。warmup 后接 Cosine 退火。

### 验证指标全量图表化

**问题**: ADE/FDE 和物理违反率仅在日志中记录，未可视化。

**方案**: 验证时反标准化后在原始空间计算物理违反率，全部 6 项指标（MSE/ADE/FDE/Speed/Accel/Height）通过 6 张图表可视化。

### 扩散评估使用完整管道（2026-07）

**问题**: 阶段二/三验证和测试评估使用 ODE 先验 `h_prior[:, :3]` 而非完整扩散采样计算 ADE/FDE，导致基于错误指标选择"最佳"模型。

**方案**: `_validate_stage2/stage3` 和 `evaluate_test` 改用 `guided_sampling()` 进行位置预测。新增 `_validate_stage3` 使用自回归状态更新模拟真实推理。

### 物理损失梯度修复

**问题**: `compute_loss()` 中 `torch.no_grad()` 阻止物理约束代价梯度回传，且阶段二/三调用方忽略 `physics_loss`——扩散模型训练中物理正则完全无效。

**方案**: 移除 `no_grad()` 使梯度正确回传，阶段二/三损失中加入 `physics_weight * phy_loss`。权重默认 0.01，可在配置中调整。

### 高度约束完整检查

**问题**: 三个评估函数仅检查 `z_min`（下限），不检查 `z_max`（上限），而 `_physics_cost()` 正确检查两者。

**方案**: 所有评估函数增加 `relu(p[:, 1] - z_max)` 上限检查。

### 加速度违反公式对齐

**问题**: 评估函数使用全速度大小 `v_norm/dt` 计算加速度，与 `_physics_cost()` 的水平分量 `v_h/dt` 不一致。

**方案**: 统一使用水平速度分量 `v_h = norm(v[:, [0,2]])` 计算加速度，与 `g·tan(max_tilt)` 比较。

### 扩散目标标签平滑移除

**问题**: 阶段二对扩散目标应用标签平滑（高斯噪声），对去噪扩散模型概念上不正确。

**方案**: 阶段二/三的扩散目标移除标签平滑，仅阶段一确定性回归保留。

### 检查点保存完整架构参数

**问题**: `load_hybrid_model()` 仅传物理参数，架构参数依赖默认值，非默认架构训练后加载失败。

**方案**: `PhyODEDiffusion` 保存 `_model_config`，`get_model_info()` 扩展为完整参数，推理时从检查点读取。

### 模型容量提升

**问题**: 原 3 层 Transformer + d_z=32 + ode_hidden=64（~255K 参数）容量有限。

**方案**: 增至 6 层 Transformer + d_z=64 + ode_hidden=128（~482K 参数），显著增强时序建模和动力学学习能力。

---

## 11. 创新点

| # | 创新点 | 说明 |
|---|--------|------|
| 1 | **Transformer 连续时间编码器** | 正弦编码 + 可学习频率缩放，完全适应不规则采样，无需插值 |
| 2 | **物理结构化 ODE 状态空间** | 显式分解 p/v/z，RK4 求解保证运动学一致性，加速度 tanh 软限幅 |
| 3 | **反标准化物理引导扩散** | 推理时传入归一化统计量，在原始 m/s 空间计算约束梯度，量纲正确 |
| 4 | **闭环 ODE 校正更新** | GRU 融合虚拟观测（含加速度），类似 Kalman 滤波的预测-校正框架 |
| 5 | **分阶段+可恢复训练** | 三阶段解耦训练 + 完整检查点（优化器/调度器）+ 自动跳过已完成阶段 |
| 6 | **轻量级 R³ 扩散** | 仅在 3D 位置空间扩散，MLP 噪声预测网络而非 UNet，推理 50 步 |
| 7 | **训练/推理归一化一致** | 窗口使用轨迹级统计量归一化，消除分布不匹配 |
| 8 | **自适应数据增强** | 爬升角判断 + 自适应噪声幅度 + 模式切换平滑过渡 |
