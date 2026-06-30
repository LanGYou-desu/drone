// 共享工具函数

/** 在时序数据中按时间 t 线性插值坐标 */
export function lerp(pts, times, t) {
    if (!pts?.length || !times?.length) return null;
    if (t <= times[0]) return [...pts[0]];
    if (t >= times[times.length - 1]) return [...pts[times.length - 1]];
    let i = 0;
    while (i < times.length - 1 && times[i + 1] < t) i++;
    const r = (t - times[i]) / (times[i + 1] - times[i]);
    return [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * r,
            pts[i][1] + (pts[i + 1][1] - pts[i][1]) * r,
            pts[i][2] + (pts[i + 1][2] - pts[i][2]) * r];
}
