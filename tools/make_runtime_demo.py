# -*- coding: utf-8 -*-
"""生成运行时演示数据：一条真值轨迹 + 三路异构传感器观测（同一目标）

用法: python tools/make_runtime_demo.py [--seed 20261122]
产物:
  data/fact/visible.dat / infrared.dat / radar.dat   —— 三路传感器观测（x y z t）
  data/test.dat                                      —— 第四路观测（用于"自选"上传演示）
  docs/acceptance/demo_truth.npz                     —— 真值（仅用于验收对比）
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.kinflow_predictor.generate_synthetic import simulate_trajectory, DEFAULT_DYNAMICS
from trajectory_reconstruction.core import sensor_sim

# 低空演示场景：高度 3~25 m、水平速度 ≤ 12 m/s，与防护区默认参数匹配
# 速度/机动水平与训练集同量级（2~4 m/s），演示时预测精度更具代表性
DEMO_DYNAMICS = dict(
    DEFAULT_DYNAMICS,
    v_h_max=6.0, v_v_up=3.0, v_v_down=2.0,
    max_tilt=30.0, min_alt=3.0, max_alt=25.0,
)

SENSOR_SPECS = [
    {'id': 'visible', 'name': '可见光', 'color': '#ff6b6b', 'rate': 5.0, 'sigma': 0.35,
     'offset': 0.10, 'occlusion': [[4.0, 5.2]], 'dropout': 0.03, 'outlier': 0.0,
     'bias': [0.05, 0.0, 0.0]},
    {'id': 'infrared', 'name': '红外', 'color': '#4ecdc4', 'rate': 4.0, 'sigma': 0.60,
     'offset': 0.20, 'occlusion': [[8.5, 10.0]], 'dropout': 0.05, 'outlier': 0.0,
     'bias': [0.12, -0.08, 0.0]},
    {'id': 'radar', 'name': '雷达', 'color': '#ffe66d', 'rate': 10.0, 'sigma': 0.25,
     'offset': 0.0, 'occlusion': [], 'dropout': 0.02, 'outlier': 0.0,
     'bias': [-0.08, 0.05, 0.06]},
]

SELF_SPEC = {'id': 'self', 'name': '自选', 'color': '#ff9500', 'rate': 8.0, 'sigma': 0.45,
             'offset': 0.05, 'occlusion': [[12.0, 13.0]], 'dropout': 0.02, 'outlier': 0.0,
             'bias': [0.06, 0.03, -0.05]}


def smooth(points: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return points.copy()
    half = window // 2
    out = points.copy()
    for i in range(len(points)):
        lo, hi = max(0, i - half), min(len(points), i + half + 1)
        out[i] = points[lo:hi].mean(axis=0)
    return out


def write_dat(path: Path, times: np.ndarray, points: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([points, times]), fmt='%.4f %.4f %.4f %.3f')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20261122)
    parser.add_argument('--duration', type=float, default=24.0)
    args = parser.parse_args()

    # simulate_trajectory 返回 (positions, timestamps)
    truth_p, truth_t = simulate_trajectory(
        duration=args.duration, dt=0.1, irregular=False,
        seed=args.seed, dynamics=DEMO_DYNAMICS,
    )
    truth_p = smooth(np.asarray(truth_p, dtype=float), 5)
    truth_t = np.asarray(truth_t, dtype=float)

    # 平移到演示可视区域：水平中心 (8, ・, 4)，最低高度 7 m
    truth_p = truth_p - truth_p.mean(axis=0) + np.array([8.0, 12.0, 4.0])
    truth_p[:, 1] += 7.0 - truth_p[:, 1].min()

    sim = sensor_sim.simulate_multi_sensor(truth_t, truth_p, specs=SENSOR_SPECS, seed=args.seed % 1000)
    for sensor in sim.sensors:
        write_dat(ROOT / 'data' / 'fact' / f'{sensor.id}.dat', sensor.times, sensor.positions)
        print(f'  {sensor.id:9s} {len(sensor.times):3d} 点  {sensor.times[0]:.2f}~{sensor.times[-1]:.2f}s')

    self_sim = sensor_sim.simulate_multi_sensor(truth_t, truth_p, specs=[SELF_SPEC], seed=7)
    write_dat(ROOT / 'data' / 'test.dat', self_sim.sensors[0].times, self_sim.sensors[0].positions)
    print(f'  self      {len(self_sim.sensors[0].times):3d} 点（data/test.dat）')

    np.savez(ROOT / 'docs' / 'acceptance' / 'demo_truth.npz',
             positions=truth_p.astype(np.float32), timestamps=truth_t.astype(np.float32))
    print(f'  真值 {len(truth_t)} 点，范围 x[{truth_p[:,0].min():.1f},{truth_p[:,0].max():.1f}] '
          f'y[{truth_p[:,1].min():.1f},{truth_p[:,1].max():.1f}] '
          f'z[{truth_p[:,2].min():.1f},{truth_p[:,2].max():.1f}]')
    print('中段坐标:', np.round(truth_p[len(truth_p)//2], 2).tolist())


if __name__ == '__main__':
    main()
