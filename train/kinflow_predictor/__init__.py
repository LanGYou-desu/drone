"""
KinFlow 训练模块

- dataset.py:            轨迹数据集 + 滑动窗口采样 + 增强
- generate_synthetic.py: 四旋翼动力学合成数据生成
- train.py:              两阶段训练脚本（联合训练 + 微调）
"""

__all__ = ["dataset", "generate_synthetic", "train"]
