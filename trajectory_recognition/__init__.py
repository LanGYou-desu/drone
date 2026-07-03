"""
轨迹识别模块 — Trajectory Recognition

基于 YOLO 模型的无人机视频检测与轨迹模式识别系统。

核心流程:
  视频输入 → 抽帧预处理 → YOLO 检测 → 多目标跟踪 → data/ 落盘 → 轨迹分类

子模块:
  detection/   — YOLO 检测引擎（模型推理 / 多目标跟踪 / 视频预处理）
  features/    — 轨迹特征提取（运动学 / 几何 / 频域）
  models/      — 识别模型定义（规则 / 机器学习 / 深度学习）
  classifier/  — 轨迹分类器（推理流水线 / 标签定义）
  services/    — 业务编排（检测会话管理 / 数据桥接）
  views/       — HTTP 视图（页面路由 / 检测 API）
"""
