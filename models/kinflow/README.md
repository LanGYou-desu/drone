# models/kinflow — KinFlow 推理权重

放置训练完成、用于推理的 KinFlow 权重：

```text
models/kinflow/
├── kinflow_best_s2.pt      # 阶段二最佳（推荐，优先加载）
├── kinflow_best_s1.pt      # 阶段一最佳（备选）
└── kinflow_s{stage}_e{epoch}_v{loss}.pt   # 阶段检查点（可选）
```

## 当前权重

| 文件 | 阶段 | 验证损失 | 参数量 | 说明 |
|------|------|----------|--------|------|
| `kinflow_best_s2.pt` | 2 | 0.2407 | 371,916 | 推荐；2 s 视界，测试 ADE 0.495 m / FDE 1.324 m |
| `kinflow_best_s1.pt` | 1 | 0.2640 左右 | 371,916 | 阶段一最佳，可作对照 |

检查点内的 `model_info` 记录 `ctx_len=40`、`sample_dt=0.1`、`k_max=20`、`v_h_max` 等，
推理时据此重建模型；**不要手动用默认参数构造模型加载**，否则窗口规格不一致会显著掉精度。

## 加载顺序

训练脚本会把最佳权重自动复制到 `train/kinflow_predictor/train_result/best/`，
推理服务优先读取那里；本目录用于手动部署或跨机器分发。

1. `train/kinflow_predictor/train_result/best/kinflow_best_s2.pt` → `..._s1.pt`
2. `models/kinflow/kinflow_best_s2.pt` → `..._s1.pt`
3. `models/kinflow/kinflow_s*_v{loss}.pt`（按 loss 升序）
4. `models/kinflow/kinflow.pt`

模型加载失败或权重缺失时会自动回退到线性外推，不会中断预测。
短视界（≤2 s）为模型预测，更长视界由运动模型外推并在界面显示逐点置信度。

> 旧 Phy-ODE-Diffusion 权重与 KinFlow 的 `state_dict` 不兼容，已归档在
> `backup/legacy_phy_ode/models/hybrid_predictor/`。
