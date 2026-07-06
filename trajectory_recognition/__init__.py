"""
轨迹识别模块 — Trajectory Recognition

基于 YOLO 模型的无人机视频检测与双目立体视觉系统。

核心流程:
  双目视频输入 → 抽帧预处理 → YOLO 检测(左右目独立) → 双目匹配
  → 三角测量(3D) → 多目标跟踪 → 世界坐标变换 → data/fact/ 落盘

子模块:
  detection/   — 检测引擎（YOLO推理 / 双目立体视觉 / 多目标跟踪 / 视频预处理）
  services/    — 业务编排（检测会话管理 / 数据桥接）
  views/       — HTTP 视图（页面路由 / 检测 API）
"""
