"""
轨迹重建与分析模块 — Trajectory Reconstruction & Analysis

负责无人机轨迹数据的加载、重建、预测与运动学分析。

分层架构:
  core/      — 核心领域逻辑（config / data / predict / ai / state）
  services/  — 业务逻辑层（data_service / backup_service / predict_service）
  routes/    — Flask 路由/蓝图（HTTP 接口层）
  web/       — 前端资源（HTML / CSS / JS）
"""
