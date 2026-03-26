import os

def load_dat_file(file_path):
    """加载 .dat 文件，每行格式：x y z t，返回点列表和时间戳列表"""
    points = []
    timestamps = []
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z, t = map(float, parts[:4])
                    points.append([x, y, z])
                    timestamps.append(t)
        print(f"✅ 成功加载 {file_path}: {len(points)} 个点")
    except Exception as e:
        print(f"❌ 加载文件失败 {file_path}: {e}")
    return points, timestamps

def save_predict_data(method_id, points, timestamps=None):
    mapping = {'visible': '1', 'infrared': '2', 'radar': '3'}
    num = mapping.get(method_id, 'self')
    filename = f"pre{num}.dat"
    file_path = os.path.join('data', 'predict', filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, p in enumerate(points):
            t = timestamps[i] if timestamps and i < len(timestamps) else i * 0.5
            f.write(f"{p[0]} {p[1]} {p[2]} {t}\n")
    print(f"📁 预测数据已保存至 {file_path}")

def load_default_data():
    method_ids = ['visible', 'infrared', 'radar']
    file_names = ['fact1.dat', 'fact2.dat', 'fact3.dat']
    methods = {}
    for mid, fname in zip(method_ids, file_names):
        file_path = os.path.join('data', 'fact', fname)
        points, timestamps = load_dat_file(file_path)
        methods[mid] = {'points': points, 'timestamps': timestamps}
    return methods