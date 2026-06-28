"""
报告保存 API — 将 AI 策略保存为报告文件
"""
import os
import time
from flask import Blueprint, jsonify, request

report_bp = Blueprint('report', __name__)
REPORTS_DIR = os.path.join('reports')


@report_bp.route('/api/save_report', methods=['POST'])
def save_report():
    """保存捕捉策略报告到 reports/ 目录，文件名含时间戳"""
    data = request.get_json(silent=True) or {}
    content = data.get('content', '').strip()
    platforms = data.get('platforms', '')

    if not content:
        return jsonify({'success': False, 'error': '报告内容为空'}), 400

    os.makedirs(REPORTS_DIR, exist_ok=True)
    filename = time.strftime('capture_report_%Y%m%d_%H%M%S.txt')
    filepath = os.path.join(REPORTS_DIR, filename)

    header = f"鹰眼长空 — 无人机捕捉策略报告\n"
    header += f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    if platforms:
        header += f"分析平台: {platforms}\n"
    header += f"{'='*50}\n\n"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + content)

    return jsonify({'success': True, 'filepath': filepath, 'filename': filename})
