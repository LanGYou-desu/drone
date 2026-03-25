# -*- coding: utf-8 -*-
from flask import Flask, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Web UI 框架 - 带按钮</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background: #f0f2f5;
        }
        .container {
            max-width: 800px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1e3c72;
            border-left: 4px solid #007bff;
            padding-left: 15px;
        }
        .button-group {
            margin: 30px 0;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover {
            background-color: #0056b3;
        }
        button.secondary {
            background-color: #6c757d;
        }
        button.secondary:hover {
            background-color: #5a6268;
        }
        .info {
            margin-top: 30px;
            padding: 15px;
            background: #e9ecef;
            border-radius: 8px;
            color: #495057;
        }
        .info p {
            margin: 0;
        }
        .output {
            margin-top: 20px;
            padding: 10px;
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Web UI 框架</h1>
        <p>这是一个带按钮的界面框架，功能暂未实现（仅前端占位）。</p>

        <div class="button-group">
            <button id="btn1">功能 1</button>
            <button id="btn2">功能 2</button>
            <button id="btn3" class="secondary">功能 3</button>
        </div>

        <div class="info">
            <p>📌 点击按钮会显示提示，后续可将这些按钮绑定到实际后端功能。</p>
        </div>

        <div class="output" id="output">
            等待操作...
        </div>
    </div>

    <script>
        // 获取元素
        const btn1 = document.getElementById('btn1');
        const btn2 = document.getElementById('btn2');
        const btn3 = document.getElementById('btn3');
        const outputDiv = document.getElementById('output');

        // 按钮点击事件
        btn1.addEventListener('click', () => {
            outputDiv.innerHTML = '✅ 你点击了「功能 1」，功能开发中...';
        });
        btn2.addEventListener('click', () => {
            outputDiv.innerHTML = '✅ 你点击了「功能 2」，功能开发中...';
        });
        btn3.addEventListener('click', () => {
            outputDiv.innerHTML = '✅ 你点击了「功能 3」，功能开发中...';
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    # Development server; use a production WSGI server (e.g. gunicorn/waitress) in production.
    app.run(host='0.0.0.0', port=5000, debug=True)