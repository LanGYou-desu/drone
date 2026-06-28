# 鹰眼长空 — 基于多平台协同的低空无人机智能监测系统

## 项目结构

```
drone/
├── main.py                              # 主入口 (Flask + pywebview)
├── config.json                          # 运行时配置（不纳入版本控制）
├── requirements.txt                     # Python 依赖
│
├── trajectory_reconstruction/           # 轨迹重建与分析模块
│   ├── app.py                           #   Flask 应用工厂
│   ├── __main__.py                      #   独立启动入口
│   ├── core/                            #   核心领域逻辑
│   │   ├── config/                      #     配置管理
│   │   ├── io/                          #     轨迹文件读写
│   │   ├── prediction/                  #     轨迹预测算法（线性外推）
│   │   ├── ai/                          #     AI 建议服务（硅基流动 API）
│   │   └── state.py                     #     共享运行时状态
│   ├── services/                        #   业务逻辑层
│   │   ├── data_service.py
│   │   ├── predict_service.py
│   │   └── backup_service.py
│   ├── views/                           #   HTTP 视图层（页面 + API）
│   │   ├── main.py                      #     页面视图
│   │   ├── api.py                       #     数据 & 备份 API
│   │   ├── api_predict.py               #     预测 & AI API
│   │   ├── api_report.py                #     报告保存 API
│   │   └── analysis.py                  #     运动学分析 API
│   └── frontend/                        #   前端资源
│       ├── pages/                       #     HTML 模板
│       └── static/                      #     CSS / JS
│
├── trajectory_recognition/              # 轨迹识别模块（框架）
│   ├── app.py                           #   Flask 应用工厂
│   ├── __main__.py                      #   独立启动入口
│   ├── features/                        #   特征提取
│   ├── models/                          #   识别模型定义
│   └── classifier/                      #   分类器
│
├── data/                                # 共享运行时数据（两模块交互）
│   ├── fact/                            #   实际轨迹数据 (.dat)
│   ├── predict/                         #   预测结果
│   └── backup/                          #   备份
│
├── reports/                             # 捕捉策略报告
├── templates/                           # 配置模板
└── docs/                                # 文档
```

## 架构说明

两个模块通过根目录 `data/` 共享轨迹数据：

- **trajectory_reconstruction** — 负责轨迹加载、重建、预测与运动学分析，将结果写入 `data/predict/`
- **trajectory_recognition** — 读取 `data/` 下的轨迹数据进行模式识别与分类

## 快速开始

```bash
pip install -r requirements.txt
```

### 统一入口

```bash
python main.py               # 轨迹重建与分析（桌面窗口）
python main.py --headless    # 轨迹重建与分析（纯 HTTP）
python main.py recon         # 轨迹重建与分析
python main.py recog         # 轨迹识别
python main.py all           # 同时启动两个模块
```

### 独立启动各模块

```bash
# 轨迹重建与分析 → http://127.0.0.1:5000
python -m trajectory_reconstruction --headless

# 轨迹识别 → http://127.0.0.1:5001
python -m trajectory_recognition
```

## API 端点

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 总览仪表板（3D 轨迹视图） |
| `/predict` | GET | 轨迹预测页面 |
| `/ai` | GET | AI 策略建议页面 |
| `/data` | GET | 数据管理页面 |
| `/analysis/` | GET | 运动学分析页面 |
| `/analysis/data` | GET | 分析数据 JSON 接口 |
| `/api/refresh_data` | POST | 重新加载默认轨迹 |
| `/api/load_data` | POST | 上传自选数据 |
| `/api/clear_all_data` | POST | 清理并备份全部数据 |
| `/api/predict` | POST | 单平台轨迹预测 |
| `/api/predict_all` | POST | 全平台轨迹预测 |
| `/api/ai_suggestion` | POST | 获取 AI 捕捉策略 |
| `/api/save_report` | POST | 保存捕捉报告 |
| `/api/list_backups` | GET | 列出备份 |
| `/api/restore_backup` | POST | 恢复指定备份 |
| `/api/restore_all_backups` | POST | 一键恢复最新备份 |
