# 鹰眼长空 — 开发文档

> 面向新开发者的上手指南

---

## 1. 项目概述

双模块无人机轨迹监测系统：轨迹重建分析 + 轨迹识别。

**技术栈：**

| 层 | 技术 |
|----|------|
| 后端 | Python Flask |
| 桌面 | pywebview |
| 3D | Three.js（本地） |
| 图表 | ECharts 5（本地） |
| AI | OpenAI 兼容接口 |
| 样式 | Apple HIG 深色 CSS |

---

## 2. 环境搭建

```bash
pip install flask pywebview requests
```

### 配置

```json
{
    "ai": {
        "provider": "siliconflow",
        "api_key": "你的密钥",
        "url": "https://api.siliconflow.cn/v1/chat/completions",
        "model": "Qwen/Qwen2.5-7B-Instruct"
    }
}
```

`ai` 字段支持任意 OpenAI 兼容接口（DeepSeek、OpenRouter、Ollama 等）。

### 启动

```bash
python main.py               # 默认同时启动两个模块
python main.py --headless    # 纯 HTTP
python main.py recon         # 仅重建分析 → :5000
python main.py recog         # 仅轨迹识别 → :5001
python -m trajectory_reconstruction --headless
python -m trajectory_recognition --headless
```

---

## 3. 架构

```
trajectory_reconstruction/          trajectory_recognition/
├── core/          (领域逻辑)        ├── features/    (待实现)
├── services/      (业务编排)        ├── models/      (待实现)
├── views/         (HTTP 接口)       ├── classifier/  (待实现)
└── frontend/      (HTML/CSS/JS)     └── frontend/    (页面)
         ↓                                    ↓
       :5000           data/ reports/         :5001
```

分层：`views → services → core`（单向依赖）

---

## 4. API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/refresh_data` | 重载轨迹 |
| POST | `/api/load_data` | 上传 .dat |
| POST | `/api/clear_all_data` | 清理并备份 |
| POST | `/api/predict` | 单平台预测 |
| POST | `/api/predict_all` | 全平台预测 |
| POST | `/api/ai_suggestion` | AI 策略 |
| POST | `/api/save_report` | 保存报告 |
| GET | `/api/list_backups` | 列出备份 |
| POST | `/api/restore_backup` | 恢复备份 |
| POST | `/api/backup/create` | 手动备份 |
| POST | `/api/backup/delete` | 删除备份 |
| GET | `/analysis/data` | 分析数据 |

---

## 5. 前端页面

| 页面 | 路由 | 引擎 |
|------|------|------|
| 总览 | `/` | Three.js |
| 预测 | `/predict` | Three.js |
| 分析 | `/analysis` | ECharts |
| AI | `/ai` | DOM |
| 数据 | `/data` | DOM |
| 识别 | `:5001/` | DOM |

Three.js 和 ECharts 已本地化到 `static/js/lib/`，离线可用。

---

## 6. 添加功能

**新 API：** 在 `views/api.py` 加 `@api_bp.route(...)`  
**新页面：** `frontend/pages/new.html` + `views/main.py` 加路由  
**新算法：** `core/` 下新建模块，保持纯函数  
**切换 AI：** 改 `config.json` 的 `ai.url` 和 `ai.model`

## 7. 调试

```bash
# 查看路由
python -c "from trajectory_reconstruction.app import create_app; [print(r) for r in create_app().url_map.iter_rules()]"

# 备份内容
ls -R data/backup/ && cat data/backup/*/manifest.json

# 恢复数据
git restore data/fact/
```
