"""
数据桥接 — Data Bridge

将跟踪轨迹写入 data/ 目录，供 trajectory_reconstruction 模块加载。

平台 → 文件映射:
  visible  → visible.dat
  infrared → infrared.dat
  radar    → radar.dat
  self     → self.dat

.dat 格式: 每行 "x y z t"（空格分隔浮点数）
"""

import json
import os
import shutil
from datetime import datetime
from typing import Optional

PLATFORM_FACT_MAP = {
    "visible":  "visible.dat",
    "infrared": "infrared.dat",
    "radar":    "radar.dat",
    "self":     "self.dat",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def backup_existing_fact(
    source_dir: str = "data/fact/",
    backup_dir: str = "data/backup/",
    label: str = "auto",
    filenames: Optional[list] = None,
    platform_id: str = "",
) -> Optional[str]:
    """
    备份 data/fact/ 中的指定 .dat 文件到 data/backup/。

    未指定 filenames 则备份全部 .dat 文件。
    格式: data/backup/{YYYYmmdd_HHMMSS}_{label}/
    """
    # label 用中文平台名，如 "可见光"
    PLATFORM_NAMES = {"visible": "可见光", "infrared": "红外", "radar": "雷达", "self": "自选"}
    if platform_id and label == "auto":
        label = PLATFORM_NAMES.get(platform_id, platform_id)
    src = os.path.join(PROJECT_ROOT, source_dir)
    if not os.path.isdir(src):
        return None

    if filenames:
        dat_files = [f for f in filenames if os.path.isfile(os.path.join(src, f))]
    else:
        dat_files = [f for f in os.listdir(src) if f.endswith('.dat')]
    if not dat_files:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(PROJECT_ROOT, backup_dir, f"{timestamp}_{label}")
    fact_dst = os.path.join(backup_path, "fact")
    os.makedirs(fact_dst, exist_ok=True)

    # 复制 .dat 文件
    for f in dat_files:
        shutil.copy2(os.path.join(src, f), os.path.join(fact_dst, f))

    # 写 manifest（与 trajectory_reconstruction 统一格式）
    # 从文件名推断平台名 + 统计点数
    PLATFORM_NAMES = {"visible":"可见光","infrared":"红外","radar":"雷达","self":"自选"}
    methods = {}
    for f in dat_files:
        for pid, fname in PLATFORM_FACT_MAP.items():
            if f == fname:
                cnt = 0
                fpath = os.path.join(src, f)
                try:
                    with open(fpath, 'r') as fh:
                        cnt = sum(1 for _ in fh)
                except Exception:
                    pass
                methods[pid] = {"name": PLATFORM_NAMES.get(pid, pid), "point_count": cnt}

    manifest = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "methods": methods,
        "files": {
            "fact": dat_files,
            "predict": [],
            "memory": [os.path.basename(f) for f in dat_files],
        },
    }
    with open(os.path.join(backup_path, "manifest.json"), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"[data_bridge] 已备份到 {backup_path}")
    return backup_path


def tracks_to_dat(
    tracks: list,
    platform_id: str = "visible",
    output_dir: str = "data/fact/",
    auto_backup: bool = True,
) -> list[str]:
    """
    将双目融合后的 3D 轨迹合并写入一个 .dat 文件。

    所有 track 的 3D 点按时序合并，输出格式:
      x y z t

    平台映射:
      visible → visible.dat    infrared → infrared.dat
      radar   → radar.dat      self     → self.dat
    """
    out = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(out, exist_ok=True)

    filename = PLATFORM_FACT_MAP.get(platform_id, f"{platform_id}.dat")

    # 自动备份（只备份即将覆盖的文件，用平台名标记）
    if auto_backup:
        backup_existing_fact(source_dir=output_dir, filenames=[filename], platform_id=platform_id)
    fpath = os.path.join(out, filename)

    # 收集所有 track 的 (x, y, z, t) 点，按时序合并
    all_points = []  # [(t, x, y, z), ...]
    for track in tracks:
        if not track.positions:
            continue
        for i, pos in enumerate(track.positions):
            ts = track.timestamps[i] if i < len(track.timestamps) else 0.0
            if len(pos) >= 3:
                all_points.append((ts, pos[0], pos[1], pos[2]))

    if not all_points:
        # 双目未定位到 3D 点，降级写 2D
        for track in tracks:
            for i, xy in enumerate(track.positions_2d):
                ts = track.timestamps[i] if i < len(track.timestamps) else 0.0
                all_points.append((ts, xy[0], xy[1], 0.0))

    # 按时序排序
    all_points.sort(key=lambda p: p[0])

    with open(fpath, 'w', encoding='utf-8') as f:
        for ts, x, y, z in all_points:
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {ts:.4f}\n")

    print(f"[data_bridge] 合并写入 {fpath} ({len(all_points)} 个 3D 点)")
    return [fpath]


def list_detect_files(data_dir: str = "data/fact/") -> list[dict]:
    """列出 data/fact/ 中的所有轨迹文件"""
    d = os.path.join(PROJECT_ROOT, data_dir)
    if not os.path.isdir(d):
        return []

    files = []
    for fname in sorted(os.listdir(d)):
        if not fname.endswith('.dat'):
            continue
        fpath = os.path.join(d, fname)
        try:
            point_count = sum(1 for _ in open(fpath, 'r'))
        except Exception:
            point_count = 0

        files.append({
            "name": fname,
            "path": f"data/fact/{fname}",
            "point_count": point_count,
            "size_kb": round(os.path.getsize(fpath) / 1024, 1),
            "modified": datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat(),
        })

    return files


def notify_reconstruction():
    """通知重建模块刷新数据"""
    try:
        import requests
        requests.post('http://127.0.0.1:5000/api/refresh_data', timeout=2)
    except Exception:
        pass  # 重建模块未运行，静默忽略
