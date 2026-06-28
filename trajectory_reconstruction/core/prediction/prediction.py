"""
轨迹预测算法 — 基于最近两点速度向量的线性外推

数学原理:
  速度向量 v = (P_last - P_prev) / Δt
  预测点 P_i = P_last + v × i × time_step   (i = 1..num_points)

局限性:
  - 仅反映瞬时速度，不体现加速度变化
  - 转弯后预测精度下降
  - 预测步数越多误差越大
"""
from typing import Optional


def generate_prediction(
    points: list[list[float]],
    timestamps: list[float],
    num_points: int = 5,
    time_step: float = 0.5,
) -> tuple[list[list[float]], list[float]]:
    """
    基于最后两点线性外推，生成未来轨迹预测点

    Args:
        points:     历史轨迹点 [[x, y, z], ...]，至少 2 个点
        timestamps: 各点时间戳 [t, ...]，长度需与 points 一致
        num_points: 预测点数
        time_step:  预测点之间的时间间隔（秒）

    Returns:
        pred_points: 预测坐标 [[x, y, z], ...]
        pred_times:  对应时间戳 [t, ...]

        若 points < 2，返回两个空列表
    """
    if len(points) < 2:
        return [], []

    last = points[-1]
    second_last = points[-2]
    last_t = timestamps[-1] if timestamps else 0.0
    second_last_t = timestamps[-2] if len(timestamps) >= 2 else last_t - 1.0

    dt = last_t - second_last_t
    if dt <= 0:
        dt = 1.0  # 防止除零和时间倒流

    # 计算速度分量
    vx = (last[0] - second_last[0]) / dt
    vy = (last[1] - second_last[1]) / dt
    vz = (last[2] - second_last[2]) / dt

    # 沿速度方向生成预测点
    pred_points: list[list[float]] = []
    pred_times: list[float] = []

    for i in range(1, num_points + 1):
        t = last_t + i * time_step
        pred_points.append([
            last[0] + vx * (t - last_t),
            last[1] + vy * (t - last_t),
            last[2] + vz * (t - last_t),
        ])
        pred_times.append(t)

    return pred_points, pred_times
