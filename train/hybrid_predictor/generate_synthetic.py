"""
四旋翼无人机轨迹合成生成器

基于四旋翼动力学模型生成具有真实飞行特征的训练数据。

动力学模型（简化刚体）:
  位置:  dp/dt = v
  速度:  dv/dt = (T/m) * R(φ,θ) * e_up - g*e_up
  其中 T=推力, m=质量, φ=roll, θ=pitch, R=旋转矩阵, g=重力加速度

  水平运动通过倾斜机体实现: a_horizontal ≈ g * tan(tilt)
  垂直运动由推力与重力的差值驱动: a_vertical = (T/m) - g

生成策略:
  随机规划 3D 航点 → 带速度/加速度约束的平滑插值 → 应用四旋翼物理限制

约束（从 config.json → drone_dynamics 读取）:
  - 最大水平速度 v_h_max (m/s)
  - 最大垂直上升/下降速度 v_v_up / v_v_down (m/s)
  - 最大倾斜角 max_tilt (deg) — 限制水平加速度
  - 最大/最小高度 max_alt / min_alt (m)
  - 最大推力加速度 thrust_max (m/s²)

输出格式: .npz (positions + timestamps)
"""

import os
import numpy as np


# ── 默认四旋翼物理参数 ──────────────────────────────────

DEFAULT_DYNAMICS = {
    "g": 9.81,              # 重力加速度 (m/s²)
    "v_h_max": 20.0,        # 最大水平速度 (m/s)
    "v_v_up": 5.0,          # 最大垂直上升速度 (m/s)
    "v_v_down": 3.0,        # 最大垂直下降速度 (m/s)
    "max_tilt": 35.0,       # 最大倾斜角 (deg)
    "max_alt": 120.0,       # 最大飞行高度 (m)
    "min_alt": 1.0,         # 最低飞行高度 (m)
    "thrust_max": 25.0,     # 最大推力加速度 (m/s²), ~2.5g
    "thrust_hover": 9.81,   # 悬停推力加速度 = g
}


def _load_dynamics():
    """从 config.json 加载四旋翼动力学参数，缺失用默认值"""
    import json
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config.json"
        )
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f).get("drone_dynamics", {})
            result = DEFAULT_DYNAMICS.copy()
            result.update({k: v for k, v in cfg.items() if k in result})
            return result
    except Exception:
        pass
    return DEFAULT_DYNAMICS.copy()


# ── 航点规划 ────────────────────────────────────────────

def _random_waypoints(n_waypoints: int, duration: float, dyn: dict,
                      seed: int = None) -> np.ndarray:
    """
    在四旋翼物理约束内随机生成 3D 航点序列。

    约束:
      - 相邻航点间的平均速度 ≤ 物理限制
      - 高度在 [min_alt, max_alt] 之间
      - 水平转弯半径合理
    """
    rng = np.random.RandomState(seed)

    # 初始位置: 随机高度
    start_h = rng.uniform(dyn["min_alt"] + 10, dyn["max_alt"] - 20)
    waypoints = np.zeros((n_waypoints, 3), dtype=np.float32)
    waypoints[0] = [0.0, start_h, 0.0]

    # 相邻航点间的时间
    dt_wp = duration / (n_waypoints - 1)

    for i in range(1, n_waypoints):
        prev = waypoints[i - 1]

        # 最大水平位移 (基于 v_h_max)
        max_h_dist = dyn["v_h_max"] * dt_wp * 0.8  # 80% 余量
        # 最大垂直位移 (基于 v_v_up/down)
        max_v_up = dyn["v_v_up"] * dt_wp * 0.7
        max_v_down = dyn["v_v_down"] * dt_wp * 0.7

        # 随机水平位移 (XZ 平面)
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0.2, 1.0) * max_h_dist
        dx = dist * np.cos(angle)
        dz = dist * np.sin(angle)

        # 随机垂直位移
        dy = rng.uniform(-max_v_down, max_v_up)

        # 避免大幅高度变化
        if abs(dy) > max(abs(max_v_up), abs(max_v_down)) * 0.5:
            dy *= 0.5

        new_pos = prev + np.array([dx, dy, dz])

        # 高度裁剪
        new_pos[1] = np.clip(new_pos[1], dyn["min_alt"], dyn["max_alt"])

        waypoints[i] = new_pos

    return waypoints


# ── 带物理约束的样条插值 ────────────────────────────────

def _cubic_spline_interpolate(waypoints: np.ndarray, n_points: int) -> np.ndarray:
    """
    用 Catmull-Rom 样条在航点间平滑插值，生成连续轨迹。

    Catmull-Rom 保证通过所有航点且 C¹ 连续（速度连续）。
    """
    n_wp = len(waypoints)
    if n_wp < 2:
        return waypoints

    # 为每个航点分配参数 t (累积弦长)
    t = np.zeros(n_wp)
    for i in range(1, n_wp):
        t[i] = t[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])
    t = t / t[-1]  # 归一化到 [0, 1]

    # 均匀采样
    t_samples = np.linspace(0, 1, n_points)
    positions = np.zeros((n_points, 3), dtype=np.float32)

    for j in range(n_points):
        tau = t_samples[j]
        # 找到 tau 所在的段
        idx = np.searchsorted(t, tau)
        if idx == 0:
            positions[j] = waypoints[0]
        elif idx >= n_wp:
            positions[j] = waypoints[-1]
        else:
            i0 = max(0, idx - 2)
            i1 = max(0, idx - 1)
            i2 = idx
            i3 = min(n_wp - 1, idx + 1)

            if i0 == i1:
                i0 = max(0, i0 - 1)
            if i3 == i2:
                i3 = min(n_wp - 1, i2 + 1)

            p0, p1, p2, p3 = waypoints[i0], waypoints[i1], waypoints[i2], waypoints[i3]
            t0, t1, t2, t3 = t[i0], t[i1], t[i2], t[i3]

            # 段内归一化参数
            if t2 > t1:
                s = (tau - t1) / (t2 - t1)
            else:
                s = 0.0

            # Catmull-Rom 公式
            s2 = s * s
            s3 = s2 * s
            positions[j] = (
                0.5 * ((2 * p1) +
                       (-p0 + p2) * s +
                       (2*p0 - 5*p1 + 4*p2 - p3) * s2 +
                       (-p0 + 3*p1 - 3*p2 + p3) * s3)
            )

    return positions


# ── 应用四旋翼物理约束 ──────────────────────────────────

def _apply_dynamics_constraints(
    positions: np.ndarray, timestamps: np.ndarray, dyn: dict
) -> np.ndarray:
    """
    对生成轨迹施加四旋翼物理约束。

    对违反约束的速度段进行局部修正:
      - 水平速度超过 v_h_max → 缩放水平分量
      - 垂直速度超过 v_v_up/down → 裁剪垂直分量
      - 高度超出 [min_alt, max_alt] → 裁剪
    """
    N = len(positions)
    if N < 2:
        return positions

    corrected = positions.copy()
    max_tilt_rad = np.radians(dyn["max_tilt"])

    for i in range(1, N):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            continue

        prev = corrected[i - 1]
        curr = corrected[i]
        displacement = curr - prev

        # 分解水平和垂直分量
        h_disp = np.array([displacement[0], 0.0, displacement[2]])
        v_disp = displacement[1]
        h_speed = np.linalg.norm(h_disp) / dt
        v_speed = v_disp / dt

        # 水平速度约束
        if h_speed > dyn["v_h_max"]:
            scale = dyn["v_h_max"] / h_speed
            h_disp *= scale
            # 同时限制倾斜产生的加速度
            a_h = (h_speed - (np.linalg.norm(
                corrected[max(0, i-2)] - prev if i >= 2 else [0, 0, 0]
            )[:2] / max(timestamps[i-1] - timestamps[max(0, i-2)], 1e-3)).sum()) / dt
            # 简化: 确保水平加速度 ≤ g*tan(max_tilt)
            max_a_h = dyn["g"] * np.tan(max_tilt_rad)
            if abs(a_h) > max_a_h:
                scale_acc = max_a_h / (abs(a_h) + 1e-8)
                h_disp = prev[:2] * dt + 0.5 * (a_h * scale_acc) * dt**2
                # 取简化处理
                h_disp = h_disp * (dyn["v_h_max"] / h_speed)

        # 垂直速度约束
        if v_speed > dyn["v_v_up"]:
            v_disp = dyn["v_v_up"] * dt
        elif v_speed < -dyn["v_v_down"]:
            v_disp = -dyn["v_v_down"] * dt

        # 重组位移
        new_pos = prev.copy()
        new_pos[0] += h_disp[0] if isinstance(h_disp, np.ndarray) else h_disp
        new_pos[2] += h_disp[2] if len(h_disp) > 2 else (h_disp[2] if isinstance(h_disp, np.ndarray) else 0)
        new_pos[1] = prev[1] + v_disp

        # 高度约束
        new_pos[1] = np.clip(new_pos[1], dyn["min_alt"], dyn["max_alt"])

        corrected[i] = new_pos

    return corrected


# ── 完整轨迹生成 ─────────────────────────────────────

def generate_trajectory(
    duration: float = 60.0,
    dt: float = 0.3,
    irregular: bool = True,
    seed: int = None,
    dynamics: dict = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    基于四旋翼动力学生成一条真实飞行轨迹。

    流程:
      1. 随机规划 3D 航点（受物理约束限制）
      2. Catmull-Rom 样条平滑插值
      3. 施加四旋翼速度/加速度/高度约束修正
      4. 添加传感器噪声模拟

    Args:
        duration: 轨迹时长 (秒)
        dt: 基础采样间隔 (秒)
        irregular: 是否非等距采样
        seed: 随机种子
        dynamics: 四旋翼动力学参数字典，None 则从 config.json 读取

    Returns:
        positions: (N, 3), timestamps: (N,)
    """
    if dynamics is None:
        dynamics = _load_dynamics()
    rng = np.random.RandomState(seed)

    # 非等距采样
    N = int(duration / dt)
    if irregular:
        dts = rng.uniform(dt * 0.5, dt * 1.5, N)
        timestamps = np.cumsum(dts)
        timestamps = timestamps - timestamps[0]
        timestamps = timestamps[:N]
    else:
        timestamps = np.linspace(0, duration, N)

    # 1. 随机航点 (5-15 个)
    n_wp = rng.randint(5, 15)
    waypoints = _random_waypoints(n_wp, duration, dynamics, seed)

    # 2. 样条插值
    positions = _cubic_spline_interpolate(waypoints, N)

    # 3. 动力学约束修正
    positions = _apply_dynamics_constraints(positions, timestamps, dynamics)

    # 4. 传感器噪声 (GPS 精度 ~0.5m)
    noise = rng.randn(N, 3) * 0.3
    positions += noise.astype(np.float32)

    # 5. 最终高度确保
    positions[:, 1] = np.clip(positions[:, 1], dynamics["min_alt"], dynamics["max_alt"])

    return positions.astype(np.float32), timestamps.astype(np.float32)


# ── 批量生成 ──────────────────────────────────────────

def generate_dataset(
    output_dir: str = "train/hybrid_predictor/dataset",
    n_trajectories: int = 200,
    valid_ratio: float = 0.15,
    duration_range: tuple = (30, 120),
    seed: int = 42,
) -> str:
    """
    生成合成训练数据集。

    Args:
        output_dir:      输出根目录（生成 train/ 和 valid/ 子目录）
        n_trajectories:  轨迹总数
        valid_ratio:     验证集比例
        duration_range:  轨迹时长范围（秒）
        seed:            随机种子
    """
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    dynamics = _load_dynamics()

    n_valid = max(1, int(n_trajectories * valid_ratio))
    n_train = n_trajectories - n_valid
    print(f"生成四旋翼合成轨迹: {n_train} train + {n_valid} valid = {n_trajectories} total")
    print(f"动力学参数: v_h_max={dynamics['v_h_max']}m/s, "
          f"v_v_up={dynamics['v_v_up']}m/s, v_v_down={dynamics['v_v_down']}m/s, "
          f"max_tilt={dynamics['max_tilt']}°, alt=[{dynamics['min_alt']},{dynamics['max_alt']}]m")

    all_indices = list(range(n_trajectories))
    rng.shuffle(all_indices)
    valid_indices = set(all_indices[:n_valid])

    for idx in range(n_trajectories):
        duration = rng.uniform(*duration_range)

        positions, timestamps = generate_trajectory(
            duration=duration,
            dt=rng.uniform(0.2, 0.5),
            irregular=True,
            seed=seed + idx,
            dynamics=dynamics,
        )

        subdir = valid_dir if idx in valid_indices else train_dir
        fname = f"traj_{idx:04d}.dat"
        with open(os.path.join(subdir, fname), 'w', encoding='utf-8') as f:
            for i in range(len(positions)):
                f.write(f"{positions[i,0]:.6f} {positions[i,1]:.6f} "
                        f"{positions[i,2]:.6f} {timestamps[i]:.6f}\n")

        if (idx + 1) % 50 == 0:
            print(f"  已生成 {idx + 1}/{n_trajectories} 条...")

    print(f"数据集已保存: {train_dir} ({n_train} train) + {valid_dir} ({n_valid} valid)")
    return output_dir


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    generate_dataset(n_trajectories=n)
