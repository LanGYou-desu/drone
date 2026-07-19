"""
纯数学工具 — 三维轨迹的插值、平均、平滑、速度估计

这些是无副作用的纯函数，不依赖全局状态或 Flask。
所有函数对输入数据做防御性拷贝，不会修改原始数据。
"""
import bisect
import numpy as np


def _lerp_3d_impl(pts: list, ts_arr: list, t: float):
    """内部：范围内线性插值（要求 t 在 [ts[0], ts[-1]] 内）"""
    i = bisect.bisect_left(ts_arr, t)
    if i <= 0 or i >= len(ts_arr):
        return list(pts[i] if i < len(pts) else pts[-1])
    t0, t1 = ts_arr[i - 1], ts_arr[i]
    if t1 == t0:
        return list(pts[i])
    r = (t - t0) / (t1 - t0)
    p0, p1 = pts[i - 1], pts[i]
    return [p0[j] + (p1[j] - p0[j]) * r for j in range(3)]


def lerp_3d(pts: list, ts_arr: list, t: float):
    """
    在时间点 t 处对三维点序列进行线性插值（超出范围夹紧到端点）

    Args:
        pts:    轨迹点列表 [[x, y, z], ...]，至少 2 个点
        ts_arr: 并行时间戳列表（已排序）
        t:      目标时间

    Returns:
        [x, y, z] 插值结果，数据不足返回 None
    """
    if not pts or not ts_arr or len(pts) < 2:
        return None
    if t <= ts_arr[0]:
        return list(pts[0])
    if t >= ts_arr[-1]:
        return list(pts[-1])
    return _lerp_3d_impl(pts, ts_arr, t)


def lerp_3d_extrapolate(pts: list, ts_arr: list, t: float):
    """
    在时间点 t 处对三维点序列进行插值或速度外推

    与 lerp_3d 不同：t 超出数据范围时不会夹紧到端点，而是用最近两点的
    速度向量外推，保证轨迹运动趋势的连续性。

    Args:
        pts:    轨迹点列表 [[x, y, z], ...]，至少 2 个点
        ts_arr: 并行时间戳列表（已排序）
        t:      目标时间

    Returns:
        [x, y, z] 插值/外推结果，数据不足返回 None
    """
    if not pts or not ts_arr or len(pts) < 2:
        return None

    # 范围外 — 速度外推
    if t < ts_arr[0]:
        dt = ts_arr[1] - ts_arr[0]
        if dt > 0:
            v = [(pts[1][j] - pts[0][j]) / dt for j in range(3)]
            return [pts[0][j] + v[j] * (t - ts_arr[0]) for j in range(3)]
        return list(pts[0])

    if t > ts_arr[-1]:
        dt = ts_arr[-1] - ts_arr[-2]
        if dt > 0:
            v = [(pts[-1][j] - pts[-2][j]) / dt for j in range(3)]
            return [pts[-1][j] + v[j] * (t - ts_arr[-1]) for j in range(3)]
        return list(pts[-1])

    # 范围内 — 线性插值
    return _lerp_3d_impl(pts, ts_arr, t)


def nearest_avg_3d(pts: list, ts_arr: list, t: float):
    """
    t 时刻前后最近两点的坐标平均值（lerp_3d 的兜底方案）

    Args:
        pts:    轨迹点列表 [[x, y, z], ...]
        ts_arr: 并行时间戳列表（已排序）
        t:      目标时间

    Returns:
        [x, y, z] 平均坐标，或最近单点，或 None
    """
    if not pts or not ts_arr or len(pts) < 2:
        return None

    prev_pt, next_pt = None, None
    for i, ti in enumerate(ts_arr):
        if ti <= t:
            prev_pt = pts[i]
        if ti >= t and next_pt is None:
            next_pt = pts[i]

    if prev_pt and next_pt:
        return [(prev_pt[j] + next_pt[j]) / 2 for j in range(3)]
    target = prev_pt or next_pt
    return list(target) if target else None


def estimate_velocity(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    用中心差分 + 端点前向/后向差分估计各点的速度向量。

    内部点用中心差分（O(Δt²) 精度），端点用单侧差分（O(Δt) 精度），
    避免简单复制相邻值导致的边界速度失真。

    Args:
        positions: (N, 3) 位置序列
        timestamps: (N,) 时间戳

    Returns:
        velocities: (N, 3) 速度估计
    """
    N = len(positions)
    vel = np.zeros((N, 3), dtype=np.float32)
    if N < 2:
        return vel

    if N >= 3:
        # 内部点: 中心差分
        for i in range(1, N - 1):
            dt_span = timestamps[i + 1] - timestamps[i - 1]
            if dt_span > 0:
                vel[i] = (positions[i + 1] - positions[i - 1]) / dt_span
        # 起点: 前向差分
        dt_fwd = timestamps[1] - timestamps[0]
        if dt_fwd > 0:
            vel[0] = (positions[1] - positions[0]) / dt_fwd
        # 终点: 后向差分
        dt_bwd = timestamps[-1] - timestamps[-2]
        if dt_bwd > 0:
            vel[-1] = (positions[-1] - positions[-2]) / dt_bwd
    else:  # N == 2
        dt_span = timestamps[1] - timestamps[0]
        if dt_span > 0:
            v = (positions[1] - positions[0]) / dt_span
            vel[0] = vel[1] = v
    return vel


def smooth_points_3d(points: list, passes: int = 2) -> list:
    """
    对三维轨迹点序列做 N 次三点移动平均平滑

    每遍平滑保持首尾点不变，中间点取前后三点的算术平均。

    Args:
        points: 轨迹点列表 [[x, y, z], ...]
        passes: 平滑遍数（默认 2）

    Returns:
        新列表，长度与输入相同
    """
    smoothed = [p[:] for p in points]
    for _ in range(passes):
        result = []
        for i in range(len(smoothed)):
            if i == 0 or i == len(smoothed) - 1:
                result.append(smoothed[i][:])
            else:
                result.append([
                    (smoothed[i - 1][j] + smoothed[i][j] + smoothed[i + 1][j]) / 3
                    for j in range(3)
                ])
        smoothed = result
    return smoothed
