"""
合成轨迹数据生成器

生成具有真实飞行特征的无人机轨迹数据用于训练。
特征包括：巡航、转弯、上升/下降、悬停、加速等机动模式。

输出格式：每个轨迹保存为 .npz 文件 (positions + timestamps)
"""

import os
import numpy as np


# ── 机动基元 ──────────────────────────────────────────

def _cruise(t: np.ndarray, speed: float, direction: np.ndarray,
            start_pos: np.ndarray) -> np.ndarray:
    """匀速直线巡航"""
    return start_pos + np.outer(t, direction * speed)


def _turn(t: np.ndarray, radius: float, angular_vel: float,
          turn_axis: int = 0) -> np.ndarray:
    """转弯轨迹（在 XZ 平面）"""
    theta = angular_vel * t
    if turn_axis == 0:
        return np.column_stack([radius * np.sin(theta),
                                 np.zeros_like(t),
                                 radius * np.cos(theta)])
    else:
        return np.column_stack([radius * np.cos(theta),
                                 np.zeros_like(t),
                                 radius * np.sin(theta)])


def _climb(t: np.ndarray, climb_rate: float, start_pos: np.ndarray,
           direction: np.ndarray, speed: float) -> np.ndarray:
    """爬升/下降"""
    horizontal = start_pos + np.outer(t, direction * speed)
    horizontal[:, 1] += climb_rate * t
    return horizontal


def _accelerate(t: np.ndarray, v0: float, a: float,
                direction: np.ndarray, start_pos: np.ndarray) -> np.ndarray:
    """匀加速直线运动"""
    s = v0 * t + 0.5 * a * t ** 2
    return start_pos + np.outer(s, direction)


# ── 完整轨迹生成 ─────────────────────────────────────

def generate_trajectory(
    duration: float = 60.0,
    dt: float = 0.3,
    irregular: bool = True,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成一条具有多种机动模式的无人机轨迹。

    机动序列（随机拼接）:
      - 巡航 (30-40%)
      - 转弯 (20-30%)
      - 爬升/下降 (15-20%)
      - 加速/减速 (10-20%)

    Returns:
        positions: (N, 3), timestamps: (N,)
    """
    if seed is not None:
        np.random.seed(seed)

    N = int(duration / dt)

    # 如果非等距采样，则生成不等间隔时间戳
    if irregular:
        dts = np.random.uniform(dt * 0.5, dt * 1.5, N)
        timestamps = np.cumsum(dts)
        timestamps = timestamps - timestamps[0]
        timestamps = timestamps[:N]
    else:
        timestamps = np.linspace(0, duration, N)

    positions = np.zeros((N, 3), dtype=np.float32)

    # 初始状态
    pos = np.array([0.0, np.random.uniform(50, 200), 0.0])  # 起始位置
    direction = np.array([np.random.uniform(-1, 1), 0.0, 1.0])
    direction = direction / np.linalg.norm(direction)
    speed = np.random.uniform(5, 20)  # m/s

    positions[0] = pos
    i = 1

    mode_duration = 0
    current_mode = None

    # 平滑过渡状态
    transition_time = 0.0
    prev_speed = speed
    prev_direction = direction.copy()

    while i < N:
        # 选择机动模式
        if mode_duration <= 0:
            r = np.random.random()
            if r < 0.35:
                current_mode = "cruise"
                mode_duration = np.random.uniform(5, 15)
            elif r < 0.60:
                current_mode = "turn"
                mode_duration = np.random.uniform(3, 10)
            elif r < 0.80:
                current_mode = "climb"
                mode_duration = np.random.uniform(3, 8)
            else:
                current_mode = "accel"
                mode_duration = np.random.uniform(2, 5)

            # 记录切换前状态用于平滑过渡
            prev_speed = speed
            prev_direction = direction.copy()
            transition_time = min(0.5, mode_duration * 0.1)  # 0.5秒过渡

            # 模式参数
            if current_mode == "turn":
                turn_radius = np.random.uniform(50, 500)
                turn_omega = np.random.uniform(0.1, 0.5) * np.random.choice([-1, 1])
            elif current_mode == "climb":
                climb_rate = np.random.uniform(-10, 10)
                h_speed = np.random.uniform(3, 15)
            elif current_mode == "accel":
                a_val = np.random.uniform(-5, 5)
                v0 = speed

        # 执行一个时间步
        t_now = timestamps[i] - timestamps[i - 1]

        if current_mode == "cruise":
            pos = pos + direction * speed * t_now
            # 缓慢随机转向
            angle_drift = np.random.uniform(-0.02, 0.02)
            cos_a, sin_a = np.cos(angle_drift), np.sin(angle_drift)
            direction = np.array([
                cos_a * direction[0] + sin_a * direction[2],
                direction[1],
                -sin_a * direction[0] + cos_a * direction[2],
            ])
            speed += np.random.uniform(-0.3, 0.3)
            speed = np.clip(speed, 3, 30)

        elif current_mode == "turn":
            # 随机转弯方向
            turn_dir = 1 if np.random.random() > 0.5 else -1
            angle = np.random.uniform(0.02, 0.08) * turn_dir
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            direction = np.array([
                cos_a * direction[0] + sin_a * direction[2],
                direction[1],
                -sin_a * direction[0] + cos_a * direction[2],
            ])
            pos = pos + direction * speed * t_now

        elif current_mode == "climb":
            pos[1] += climb_rate * t_now
            pos = pos + direction * speed * t_now

        elif current_mode == "accel":
            speed += a_val * t_now
            speed = np.clip(speed, 3, 30)
            pos = pos + direction * speed * t_now

        # 平滑过渡：在模式切换后的短暂窗口内混合新旧状态
        if transition_time > 0:
            alpha = min(1.0, t_now / max(transition_time, 1e-6))
            speed = prev_speed * (1 - alpha) + speed * alpha
            direction = prev_direction * (1 - alpha) + direction * alpha
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            transition_time -= t_now

        # 添加小幅随机扰动
        pos += np.random.randn(3) * 0.1

        # 高度限制
        pos[1] = max(pos[1], 10.0)

        positions[i] = pos
        i += 1
        mode_duration -= t_now

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
    生成合成训练数据集，仿照 YOLO 结构分 train/valid 两个子目录。

    Args:
        output_dir:   输出根目录（生成 train/ 和 valid/ 子目录）
        n_trajectories: 轨迹总数
        valid_ratio:   验证集比例
        duration_range: 轨迹时长范围（秒）
        seed:          随机种子

    Returns:
        output_dir 路径
    """
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    np.random.seed(seed)

    n_valid = max(1, int(n_trajectories * valid_ratio))
    n_train = n_trajectories - n_valid
    print(f"生成合成轨迹: {n_train} train + {n_valid} valid = {n_trajectories} total")

    all_indices = list(range(n_trajectories))
    np.random.shuffle(all_indices)
    valid_indices = set(all_indices[:n_valid])

    for idx in range(n_trajectories):
        duration = np.random.uniform(*duration_range)
        irregular = np.random.random() > 0.3

        positions, timestamps = generate_trajectory(
            duration=duration,
            dt=np.random.uniform(0.2, 0.5),
            irregular=irregular,
            seed=seed + idx,
        )

        subdir = valid_dir if idx in valid_indices else train_dir
        fname = f"traj_{idx:04d}.npz"
        np.savez(
            os.path.join(subdir, fname),
            positions=positions,
            timestamps=timestamps,
        )

        if (idx + 1) % 50 == 0:
            print(f"  已生成 {idx + 1}/{n_trajectories} 条...")

    print(f"数据集已保存: {train_dir} ({n_train} train) + {valid_dir} ({n_valid} valid)")
    return output_dir


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    generate_dataset(n_trajectories=n)
