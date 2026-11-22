"""
四旋翼无人机轨迹合成生成器（KinFlow 训练数据）

基于简化的四旋翼刚体动力学生成具有真实飞行特征的轨迹:

    位置:  dp/dt = v
    速度:  dv/dt = (T/m)·R(φ,θ)·e_up - g·e_up

  水平运动通过倾斜机体实现: a_h ≈ g·tan(φ_max)
  垂直运动由推力与重力的差值驱动: a_v = T/m - g

生成流程: 随机规划 3D 航点 → Catmull-Rom 样条平滑 → 施加四旋翼速度/高度约束 → 传感器噪声

约束（默认读取 config.json → drone_dynamics）:
  - 水平速度 v_h_max，垂直上升/下降 v_v_up / v_v_down
  - 最大倾斜角 max_tilt（限制水平加速度）
  - 高度范围 min_alt / max_alt

输出格式: .npz（positions (N,3) + timestamps (N,)），支持非等距采样。
"""

import json
import os
from pathlib import Path

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent      # train/kinflow_predictor/
_PROJECT_ROOT = _MODULE_DIR.parent.parent          # drone/


# ── 默认四旋翼物理参数 ──────────────────────────────────

DEFAULT_DYNAMICS = {
    "g": 9.81,              # 重力加速度 (m/s²)
    "v_h_max": 20.0,        # 最大水平速度 (m/s)
    "v_v_up": 5.0,          # 最大垂直上升速度 (m/s)
    "v_v_down": 3.0,        # 最大垂直下降速度 (m/s)
    "max_tilt": 35.0,       # 最大倾斜角 (deg)
    "max_alt": 120.0,       # 最高飞行高度 (m)
    "min_alt": 1.0,         # 最低飞行高度 (m)
    "thrust_max": 25.0,     # 最大推力加速度 (m/s²)
    "thrust_hover": 9.81,   # 悬停推力加速度 = g
}


def load_drone_dynamics() -> dict:
    """从 config.json → drone_dynamics 读取参数，缺失字段用默认值。"""
    config_path = _PROJECT_ROOT / "config.json"
    if config_path.is_file():
        try:
            with open(str(config_path), 'r', encoding='utf-8') as f:
                cfg = json.load(f).get("drone_dynamics", {})
            result = DEFAULT_DYNAMICS.copy()
            result.update({k: v for k, v in cfg.items() if k in result})
            return result
        except Exception:
            pass
    return DEFAULT_DYNAMICS.copy()


# ── 航点规划 ────────────────────────────────────────────

def _plan_waypoints(n_waypoints: int, duration: float, dyn: dict,
                    seed: int = None) -> np.ndarray:
    """
    在四旋翼物理约束内随机生成 3D 航点序列。

    约束: 相邻航点平均速度不超限、高度位于 [min_alt, max_alt]。
    """
    rng = np.random.default_rng(seed)

    start_h = rng.uniform(dyn["min_alt"] + 10, max(dyn["max_alt"] - 20, dyn["min_alt"] + 11))
    waypoints = np.zeros((n_waypoints, 3), dtype=np.float32)
    waypoints[0] = [0.0, start_h, 0.0]

    dt_wp = duration / max(n_waypoints - 1, 1)

    for i in range(1, n_waypoints):
        prev = waypoints[i - 1]

        max_h_dist = dyn["v_h_max"] * dt_wp * 0.8       # 80% 余量
        max_v_up = dyn["v_v_up"] * dt_wp * 0.7
        max_v_down = dyn["v_v_down"] * dt_wp * 0.7

        # 水平位移（XZ 平面）
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0.2, 1.0) * max_h_dist
        dx, dz = dist * np.cos(angle), dist * np.sin(angle)

        # 垂直位移，避免大幅爬升/俯冲
        dy = rng.uniform(-max_v_down, max_v_up)
        if abs(dy) > max(abs(max_v_up), abs(max_v_down)) * 0.5:
            dy *= 0.5

        new_pos = prev + np.array([dx, dy, dz], dtype=np.float32)
        new_pos[1] = np.clip(new_pos[1], dyn["min_alt"], dyn["max_alt"])
        waypoints[i] = new_pos

    return waypoints


# ── 样条插值 ────────────────────────────────────────────

def _spline_interpolate(waypoints: np.ndarray, n_points: int) -> np.ndarray:
    """
    用 Catmull-Rom 样条在航点间平滑插值（通过全部航点、C¹ 连续）。
    """
    n_wp = len(waypoints)
    if n_wp < 2:
        return waypoints

    t = np.zeros(n_wp)
    for i in range(1, n_wp):
        t[i] = t[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])
    if t[-1] <= 0:
        return np.repeat(waypoints[:1], n_points, axis=0)
    t = t / t[-1]

    positions = np.zeros((n_points, 3), dtype=np.float32)
    for j, tau in enumerate(np.linspace(0, 1, n_points)):
        idx = int(np.searchsorted(t, tau))
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
            t1, t2 = t[i1], t[i2]
            s = (tau - t1) / (t2 - t1) if t2 > t1 else 0.0
            s2, s3 = s * s, s * s * s
            positions[j] = 0.5 * ((2 * p1)
                                  + (-p0 + p2) * s
                                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2
                                  + (-p0 + 3 * p1 - 3 * p2 + p3) * s3)

    return positions


# ── 物理约束 ────────────────────────────────────────────

def _enforce_dynamics(positions: np.ndarray, timestamps: np.ndarray, dyn: dict) -> np.ndarray:
    """
    逐点修正轨迹使其满足四旋翼速度/高度约束。

    水平速度超限 → 等比缩放水平位移；垂直速度超限 → 裁剪；
    高度越界 → 夹紧到 [min_alt, max_alt]。
    """
    N = len(positions)
    if N < 2:
        return positions

    corrected = positions.copy()
    for i in range(1, N):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            continue

        prev, curr = corrected[i - 1], corrected[i]
        displacement = curr - prev

        h_disp = np.array([displacement[0], 0.0, displacement[2]], dtype=np.float32)
        v_disp = displacement[1]

        h_speed = np.linalg.norm(h_disp) / dt
        v_speed = v_disp / dt

        if h_speed > dyn["v_h_max"]:
            h_disp = h_disp * (dyn["v_h_max"] / h_speed)
        if v_speed > dyn["v_v_up"]:
            v_disp = dyn["v_v_up"] * dt
        elif v_speed < -dyn["v_v_down"]:
            v_disp = -dyn["v_v_down"] * dt

        new_pos = prev.copy()
        new_pos[0] += h_disp[0]
        new_pos[2] += h_disp[2]
        new_pos[1] = np.clip(prev[1] + v_disp, dyn["min_alt"], dyn["max_alt"])
        corrected[i] = new_pos

    return corrected


# ── 单条轨迹 ────────────────────────────────────────────

def simulate_trajectory(
    duration: float = 60.0,
    dt: float = 0.3,
    irregular: bool = True,
    seed: int = None,
    dynamics: dict = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成一条受物理约束的四旋翼飞行轨迹。

    Args:
        duration:  轨迹时长（秒）
        dt:        基础采样间隔（秒）
        irregular: 是否非等距采样（KinFlow 的训练目标就是不规则采样）
        seed:      随机种子
        dynamics:  四旋翼动力学参数，None 则读取 config.json

    Returns:
        positions: (N, 3), timestamps: (N,)
    """
    if dynamics is None:
        dynamics = load_drone_dynamics()
    rng = np.random.default_rng(seed)

    N = max(int(duration / dt), 8)
    if irregular:
        dts = rng.uniform(dt * 0.5, dt * 1.5, N)
        timestamps = np.cumsum(dts)
        timestamps = (timestamps - timestamps[0])[:N]
    else:
        timestamps = np.linspace(0, duration, N)
    timestamps = timestamps.astype(np.float32)

    n_wp = int(rng.integers(5, 15))
    waypoints = _plan_waypoints(n_wp, duration, dynamics, seed)
    positions = _spline_interpolate(waypoints, N)
    positions = _enforce_dynamics(positions, timestamps, dynamics)

    # 传感器噪声（GPS 量级 ~0.3 m）
    positions = positions + rng.standard_normal((N, 3)).astype(np.float32) * 0.3
    positions[:, 1] = np.clip(positions[:, 1], dynamics["min_alt"], dynamics["max_alt"])

    return positions.astype(np.float32), timestamps


# ── 数据集批量生成 ──────────────────────────────────────

def generate_synthetic_dataset(
    output_dir: str = None,
    n_trajectories: int = 200,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.10,
    duration_range: tuple = (30.0, 120.0),
    seed: int = 42,
) -> str:
    """
    生成合成训练数据集，按轨迹级切分为 train / valid / test。

    轨迹级切分保证同一轨迹的窗口不会跨集合泄漏（论文 §4.1）。
    """
    if output_dir is None:
        output_dir = str(_MODULE_DIR / "dataset")
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")
    for d in (train_dir, valid_dir, test_dir):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(seed)
    dynamics = load_drone_dynamics()

    n_valid = max(1, int(n_trajectories * valid_ratio))
    n_test = max(1, int(n_trajectories * test_ratio))
    n_train = max(1, n_trajectories - n_valid - n_test)

    print(f"生成四旋翼合成轨迹: {n_train} train + {n_valid} valid + {n_test} test "
          f"= {n_train + n_valid + n_test} total")
    print(f"动力学参数: v_h_max={dynamics['v_h_max']} m/s, "
          f"v_v_up={dynamics['v_v_up']} m/s, v_v_down={dynamics['v_v_down']} m/s, "
          f"max_tilt={dynamics['max_tilt']}°, "
          f"alt=[{dynamics['min_alt']}, {dynamics['max_alt']}] m")

    order = list(range(n_trajectories))
    rng.shuffle(order)
    split_of = {}
    for rank, idx in enumerate(order):
        if rank < n_test:
            split_of[idx] = test_dir
        elif rank < n_test + n_valid:
            split_of[idx] = valid_dir
        else:
            split_of[idx] = train_dir

    for idx in range(n_trajectories):
        duration = float(rng.uniform(*duration_range))
        positions, timestamps = simulate_trajectory(
            duration=duration,
            dt=float(rng.uniform(0.2, 0.5)),
            irregular=True,
            seed=seed + idx,
            dynamics=dynamics,
        )
        subdir = split_of[idx]
        np.savez(os.path.join(subdir, f"traj_{idx:05d}.npz"),
                 positions=positions, timestamps=timestamps)
        if (idx + 1) % 50 == 0:
            print(f"  已生成 {idx + 1}/{n_trajectories} 条...")

    print(f"数据集已保存: {output_dir}")
    return output_dir


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="生成四旋翼合成轨迹数据集")
    parser.add_argument("n", nargs="?", type=int, default=200, help="轨迹条数")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    args = parser.parse_args()
    generate_synthetic_dataset(
        output_dir=args.output_dir, n_trajectories=args.n,
        valid_ratio=args.valid_ratio, test_ratio=args.test_ratio, seed=args.seed,
    )


if __name__ == "__main__":
    _main()
