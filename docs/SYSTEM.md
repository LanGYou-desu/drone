# 鹰眼长空 — 系统架构

> 基于多平台协同的低空无人机智能监测系统 · 高内聚低耦合

---

## 快速启动

```bash
# 桌面窗口模式
python main.py

# 纯浏览器模式（开发调试）
python main.py --no-window
# → http://127.0.0.1:5000
```

## 完整目录

```
drone/
├── main.py                       # 入口：Flask + pywebview 桌面窗口
├── config.json                   # 运行配置（API Key / 平台 / 预测参数）
├── .gitignore
│
├── modules/                      # 后端（三层架构）
│   ├── state.py                  # 共享运行时状态（跨切面）
│   ├── config/config_manager.py  # 配置读写 + 校验 + 默认值
│   ├── data/data_loader.py       # .dat 文件 I/O
│   ├── predict/prediction.py     # 线性外推预测算法
│   ├── ai/ai_service.py          # 硅基流动 AI 调用
│   ├── services/                 # 业务逻辑层
│   │   ├── data_service.py       #   数据刷新/上传/清理
│   │   ├── predict_service.py    #   预测编排 + 参数校验
│   │   └── backup_service.py     #   备份列表/恢复
│   └── routes/                   # HTTP 路由层（薄层）
│       ├── __init__.py           #   蓝图注册 + 错误处理
│       ├── main.py               #   5 个页面渲染
│       ├── api.py                #   数据 + 备份 API（6个端点）
│       ├── api_predict.py        #   预测 + AI API（3个端点）
│       └── analysis.py           #   分析数据 API
│
├── web/                           # Web 资源
│   ├── pages/                     #   多页面 Jinja2 模板
│   │   ├── base.html              #     基础布局（导航栏 + toast）
│   │   ├── index.html             #     总览（全屏 3D + 覆盖层）
│   │   ├── predict.html           #     预测（参数 + 3D 预览）
│   │   ├── analysis.html          #     分析（ECharts 四图表）
│   │   ├── ai.html                #     AI 策略
│   │   └── data.html              #     数据管理
│   ├── static/                    #   前端 CSS + JS
│   │   ├── css/main.css           #     深色 Apple 全局样式
│   │   └── js/
│   │       ├── common/toast.js    #     共享通知组件
│   │       └── pages/
│   │           ├── dashboard.js   #     总览 3D 引擎
│   │           ├── predict.js     #     预测页
│   │           ├── ai.js          #     AI 页
│   │           └── data.js        #     数据页
│
├── templates/                    # 配置模板文件
│   └── config_template.json      #   config.json 模板
│
├── data/
│   ├── fact/                     # 默认轨迹（fact1/2/3.dat）
│   ├── predict/                  # 预测结果
│   └── backup/                   # 备份
│
└── docs/
    ├── API.md                    # API 接口文档
    └── SYSTEM.md                 # 系统架构（本文档）
```

---

## 页面功能

| 页面 | 路由 | 说明 |
|------|------|------|
| 总览 | `/` | 全屏 3D 视图，发光轨迹 + 带刻度坐标轴 + 拖尾粒子 + 时间轴动画 |
| 预测 | `/predict` | 参数配置 + 单/全平台线性外推预测 + 3D 预览 |
| 分析 | `/analysis` | ECharts 四图表：高度 / 速度 / 加速度 / 曲率 |
| AI | `/ai` | 多平台选择 + 硅基流动大模型捕捉策略生成 |
| 数据 | `/data` | .dat 上传 + 备份恢复 + 当前数据表格 |

## API 端点（10个）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/refresh_data` | 刷新默认轨迹 |
| POST | `/api/load_data` | 上传 .dat 文件 |
| POST | `/api/clear_all_data` | 清理并备份 |
| POST | `/api/predict` | 单平台预测 |
| POST | `/api/predict_all` | 全平台预测 |
| POST | `/api/ai_suggestion` | AI 建议 |
| GET | `/api/list_backups` | 列出备份 |
| POST | `/api/restore_backup` | 恢复备份 |
| POST | `/api/restore_all_backups` | 一键恢复 |
| GET | `/analysis/data` | 运动学分析数据 |

## 3D 渲染特性

- 深色渐变背景 + 线性雾（远距裁剪）
- ACES 色调映射 + PCF 软阴影
- 自定义坐标轴：带刻度标记、数字标签、方向箭头、轴字母
- CatmullRom 发光轨迹 + GPU 拖尾粒子
- OrbitControls（手动旋转/缩放，无自动旋转）

## 数据格式

`.dat` 文件每行 4 个字段：
```
x y z t
```

- **x, z**: 水平位置（m）
- **y**: 高度（m）
- **t**: 时间戳（s）

测试数据（`data/fact/`）包含 3 组各 60 点的虚构轨迹，可替换为真实数据。

## 设计原则

- **高内聚**：模块/页面功能内聚，边界清晰
- **低耦合**：页面间无直接依赖；Routes → Services → Modules 单向依赖
- **薄路由**：路由层不写业务逻辑
- **纯函数**：算法/I/O 模块无副作用
