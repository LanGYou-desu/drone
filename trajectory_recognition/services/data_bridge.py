"""
数据桥接 — Data Bridge

将跟踪轨迹写入 data/ 目录，供 trajectory_reconstruction 模块加载。

平台 → 文件映射:
  visible  → fact1.dat
  infrared → fact2.dat
  radar    → fact3.dat
  self     → self.dat

.dat 格式: 每行 "x y z t"（空格分隔浮点数）
"""

import json
import os
import shutil
from datetime import datetime
from typing import Optional

PLATFORM_FACT_MAP = {
    "visible":  "fact1.dat",
    "infrared": "fact2.dat",
    "radar":    "fact3.dat",
    "self":     "self.dat",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def backup_existing_fact(
    source_dir: str = "data/fact/",
    backup_dir: str = "data/backup/",
    label: str = "auto",
) -> Optional[str]:
    """
    备份 data/fact/ 中的现有 .dat 文件到 data/backup/。

    格式: data/backup/{YYYYmmdd_HHMMSS}_{label}/
    """
    src = os.path.join(PROJECT_ROOT, source_dir)
    if not os.path.isdir(src):
        return None

    dat_files = [f for f in os.listdir(src) if f.endswith('.dat')]
    if not dat_files:
        return None  # 空目录不备份

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(PROJECT_ROOT, backup_dir, f"{timestamp}_{label}")
    fact_dst = os.path.join(backup_path, "fact")
    os.makedirs(fact_dst, exist_ok=True)

    # 复制 .dat 文件
    for f in dat_files:
        shutil.copy2(os.path.join(src, f), os.path.join(fact_dst, f))

    # 写 manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "label": label,
        "files": dat_files,
        "point_counts": {},
    }
    for f in dat_files:
        fpath = os.path.join(src, f)
        try:
            with open(fpath, 'r') as fh:
                manifest["point_counts"][f] = sum(1 for _ in fh)
        except Exception:
            manifest["point_counts"][f] = 0

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
    将跟踪目标的 3D 轨迹点写入平台的 .dat 文件。

    每行格式: x y z t
    仅写入有 3D positions 的 track，跳过仅有 2D 的目标。
    """
    out = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(out, exist_ok=True)

    # 自动备份
    if auto_backup:
        backup_existing_fact(source_dir=output_dir)

    base_name = PLATFORM_FACT_MAP.get(platform_id, f"{platform_id}.dat")
    written = []

    tracks_with_3d = [t for t in tracks if t.positions and len(t.positions) > 0]

    if not tracks_with_3d:
        # 没有 3D 数据，写空文件标记
        fpath = os.path.join(out, base_name)
        with open(fpath, 'w') as f:
            pass
        return [fpath]

    for i, track in enumerate(tracks_with_3d):
        filename = base_name if i == 0 else base_name.replace('.dat', f'_track{track.track_id}.dat')
        fpath = os.path.join(out, filename)

        with open(fpath, 'w', encoding='utf-8') as f:
            for j, (pos, ts) in enumerate(zip(track.positions, track.timestamps)):
                if len(pos) >= 3:
                    f.write(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} {ts:.4f}\n")
                elif j < len(track.positions_2d) and len(track.positions_2d[j]) >= 2:
                    # 降级：仅有 2D 坐标时 z=0
                    xy = track.positions_2d[j]
                    f.write(f"{xy[0]:.4f} {xy[1]:.4f} 0.0000 {ts:.4f}\n")

        written.append(fpath)
        print(f"[data_bridge] 写入 {fpath} ({len(track.positions)} 个 3D 点)")

    return written


def tracks_to_memory(tracks: list, target_methods: dict) -> dict:
    """将跟踪轨迹注入 detection_methods 内存结构"""
    for track in tracks:
        if not track.positions:
            continue
        mid = f"detect_{track.track_id}"
        target_methods[mid] = {
            "name": f"检测目标 {track.track_id}",
            "color": "#58a6ff",
            "visible": True,
            "weight": 1.0,
            "points": [list(p) for p in track.positions],
            "timestamps": list(track.timestamps),
        }
    return target_methods


def merge_tracks(tracks: list) -> list[list[float]]:
    """多目标轨迹合并为综合轨迹（按时序质心）"""
    all_ts = set()
    for t in tracks:
        for ts in t.timestamps:
            all_ts.add(ts)
    if not all_ts:
        return []

    sorted_ts = sorted(all_ts)
    merged = []
    for ts in sorted_ts:
        points_at_t = []
        for t in tracks:
            if ts in t.timestamps:
                idx = t.timestamps.index(ts)
                if idx < len(t.positions):
                    points_at_t.append(t.positions[idx])
        if points_at_t:
            avg = [sum(c) / len(points_at_t) for c in zip(*points_at_t)]
            merged.append(avg)

    return merged


def save_detection_metadata(
    tracks: list,
    session_info: dict,
    output_dir: str = "data/fact/",
) -> str:
    """保存检测元信息 JSON"""
    out = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(out, exist_ok=True)
    fpath = os.path.join(out, "detect_manifest.json")

    meta = {
        "saved_at": datetime.now().isoformat(),
        "session": session_info,
        "tracks": [t.to_summary() if hasattr(t, 'to_summary') else {} for t in tracks],
    }
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return fpath


def load_detection_tracks(data_dir: str = "data/fact/") -> list[dict]:
    """加载之前保存的检测轨迹"""
    d = os.path.join(PROJECT_ROOT, data_dir)
    if not os.path.isdir(d):
        return []

    tracks = []
    for fname in sorted(os.listdir(d)):
        if not fname.startswith("detect_") or not fname.endswith(".dat"):
            # 也检查平台文件
            if fname not in PLATFORM_FACT_MAP.values():
                continue
            if fname == "self.dat":
                continue  # 跳过自选

        fpath = os.path.join(d, fname)
        points = []
        try:
            with open(fpath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        points.append([float(p) for p in parts[:4]])
        except Exception:
            continue

        if points:
            tracks.append({
                "name": fname,
                "path": fpath,
                "point_count": len(points),
                "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                "modified": datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat(),
            })

    return tracks


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
