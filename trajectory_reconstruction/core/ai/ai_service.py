"""
AI 建议服务 — 调用硅基流动 API 生成无人机捕捉策略
"""
import json

import requests
from requests.exceptions import RequestException, Timeout


# 系统提示词
SYSTEM_PROMPT = (
    '你是一个专业的无人机反制专家，擅长根据多传感器（可见光、红外、雷达）'
    '轨迹数据给出具体的捕捉策略。请提供可操作的建议，包括推荐捕捉设备、'
    '最佳拦截位置、时间窗口等。'
)


def _build_prompt(methods_data: dict) -> str:
    """根据多平台轨迹数据构建分析提示词"""
    prompt = (
        '你是一个无人机捕捉策略专家。以下是多种检测手段捕获的无人机轨迹数据'
        '（三维坐标 x, y, z，时间 t）：\n\n'
    )
    for method_id, info in methods_data.items():
        prompt += f'检测手段: {info["name"]}\n'
        points = info.get('points', [])
        timestamps = info.get('timestamps', [])
        prompt += '轨迹点序列 (x, y, z, t):\n'
        for i, p in enumerate(points):
            t = timestamps[i] if i < len(timestamps) else 0.0
            prompt += f'  {i + 1}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}, {t:.2f})\n'
        prompt += '\n'

    prompt += (
        '请根据这些轨迹数据，分析无人机的运动模式，并给出捕捉建议'
        '（例如：推荐使用的捕捉设备、最佳拦截点、时间窗口等）。'
        '建议要具体、可操作。'
    )
    return prompt


def get_ai_suggestion(
    methods_data: dict,
    api_key: str,
    url: str,
    model: str,
    timeout: int = 30,
) -> str:
    """
    调用大模型 API 获取无人机捕捉策略建议

    Args:
        methods_data: { methodId: { name, points, timestamps }, ... }
        api_key:      硅基流动 API Key
        url:          API 端点
        model:        模型名称
        timeout:      请求超时秒数

    Returns:
        AI 生成的建议文本

    Raises:
        RequestException: 网络异常
        ValueError:       API 响应格式异常
    """
    prompt = _build_prompt(methods_data)

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': 0.7,
        'max_tokens': 800,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    except Timeout:
        raise RequestException('AI 服务响应超时，请稍后重试')
    except RequestException as e:
        raise RequestException(f'AI 服务请求失败: {e}')

    try:
        result = response.json()
    except json.JSONDecodeError:
        raise ValueError('AI 返回数据格式异常（非 JSON 响应）')

    # 校验响应结构
    if 'choices' not in result or not result['choices']:
        raise ValueError('AI 返回数据格式异常')

    content = result['choices'][0].get('message', {}).get('content', '')
    if not content:
        raise ValueError('AI 未返回有效建议')

    return content
