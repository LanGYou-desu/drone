# 鹰眼长空 — 系统架构

> 基于多平台协同的低空无人机智能监测系统 · 高内聚低耦合

---

## 完整目录

```
drone/
├── main.py                          # 入口：Flask + pywebview
├── .gitignore
│
├── modules/                         # 后端（三层架构）
│   ├── config/config_manager.py     # 配置读写+校验
│   ├── data/data_loader.py          # .dat 文件 I/O
│   ├── predict/prediction.py        # 线性外推预测算法
│   ├── ai/ai_service.py             # 硅基流动 AI 调用
│   ├── services/                    # 业务逻辑层
│   │   ├── state.py                 #   共享运行时状态
│   │   ├── data_service.py          #   数据刷新/上传/清理
│   │   ├── predict_service.py       #   预测编排+参数校验
│   │   └── backup_service.py        #   备份列表/恢复
│   └── routes/                      # HTTP 路由层（薄层）
│       ├── __init__.py              #   蓝图注册+错误处理
│       ├── main.py                  #   5 个页面渲染入口
│       ├── api_data.py              #   数据管理 API
│       ├── api_predict.py           #   预测+AI API
│       ├── api_backup.py            #   备份管理 API
│       └── analysis.py              #   分析数据 API
│
├── templates/                       # 多页面 Jinja2 模板
│   ├── base.html                    #   基础布局（导航+toast）
│   ├── index.html                   #   总览（全屏 3D）
│   ├── predict.html                 #   预测（参数+3D 预览）
│   ├── analysis.html                #   分析（四图表）
│   ├── ai.html                      #   AI 策略
│   └── data.html                    #   数据管理
│
├── static/                          # 前端资源
│   ├── css/main.css                 #   深色 Apple 全局样式
│   └── js/
│       ├── common/toast.js          #   共享通知组件
│       └── pages/
│           ├── dashboard.js         #   总览 3D 引擎
│           ├── predict.js           #   预测页逻辑
│           ├── ai.js                #   AI 页逻辑
│           └── data.js              #   数据页逻辑
│
├── data/fact/                       # 默认轨迹数据
└── docs/
    ├── API.md                       # API 接口文档
    └── SYSTEM.md                    # 系统架构（本文档）
```

---

## 架构设计

### 后端三层

```
Routes (HTTP 薄层)    →  只做解析+响应，不写业务逻辑
    ↓
Services (核心)       →  业务编排，不依赖 Flask
    ↓
Modules (纯函数)      →  算法/IO/API，无状态可测试
```

### 前端模块隔离

每个页面是自包含模块，页面间无运行时耦合：

| 页面 | 路由 | JS 模块 | 说明 |
|------|------|---------|------|
| 总览 | `/` | dashboard.js | 全屏 3D：发光轨迹+阴影+粒子+动画 |
| 预测 | `/predict` | predict.js | 参数配置+单/全预测+3D 预览 |
| 分析 | `/analysis` | 内联脚本 | ECharts 速度/加速度/曲率/高度 |
| AI | `/ai` | ai.js | 平台选择+大模型策略生成 |
| 数据 | `/data` | data.js | 上传+备份恢复+数据表格 |

### 3D 投影特性

- 渐变背景 + 指数雾（深度感）
- ACES 色调映射（电影级色彩）
- PCF 软阴影（方向光）
- CatmullRom 平滑曲线 + 发光光晕（AdditiveBlending）
- GPU 拖尾粒子系统（2000+/平台）
- OrbitControls 自动旋转 + 阻尼惯性
- CSS2D 标签（起点/终点/平台名）

### 设计原则

- 高内聚：每个模块/页面功能内聚，边界清晰
- 低耦合：页面间无直接依赖，通过 HTTP API 通信
- 薄路由：路由层不写业务逻辑
- 纯函数模块：算法模块无副作用
