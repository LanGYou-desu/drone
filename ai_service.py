import requests

def get_ai_suggestion(methods_data, api_key, url, model):
    """调用硅基流动 API，返回捕捉建议"""
    # 构造提示词
    prompt = "你是一个无人机捕捉策略专家。以下是多种检测手段捕获的无人机轨迹数据（三维坐标 x, y, z）：\n\n"
    for method_id, info in methods_data.items():
        prompt += f"检测手段: {info['name']}\n"
        points = info['points']
        prompt += "轨迹点序列 (x, y, z):\n"
        for i, p in enumerate(points):
            prompt += f"  {i+1}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})\n"
        prompt += "\n"
    prompt += "请根据这些轨迹数据，分析无人机的运动模式，并给出捕捉建议（例如：推荐使用的捕捉设备、最佳拦截点、时间窗口等）。建议要具体、可操作。"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个专业的无人机反制专家，擅长根据多传感器轨迹数据给出捕捉策略。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']