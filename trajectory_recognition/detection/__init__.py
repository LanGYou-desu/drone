"""
检测引擎模块 — Detection Engine

基于 YOLO 模型的无人机视频检测，包含模型推理、多目标跟踪、视频预处理。

子模块:
  engine/       — YOLO 模型加载与推理
  tracker/      — 多目标跟踪（ByteTrack/DeepSORT）
  preprocess/   — 视频抽帧、缩放、归一化
"""
