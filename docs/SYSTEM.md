# 鹰眼长空 — 系统架构

> 基于多平台协同的低空无人机智能监测系统 · 双模块架构

---

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 桌面窗口模式
python main.py

# 纯 HTTP 服务模式
python main.py --headless
# → http://127.0.0.1:5000

# 模块单独启动
python main.py recon       # 轨迹重建与分析
python main.py recog       # 轨迹识别
python main.py all         # 同时启动两个

# 独立模块入口
python -m trajectory_reconstruction --headless   # → :5000
python -m trajectory_recognition                  # → :5001
```

## 完整目录

```
drone/
├── main.py                              # 入口：Flask + pywebview 桌面窗口
├── config.json                          # 运行配置（API Key / 平台 / 预测参数）
├── requirements.txt
├── .gitignore
│
├── trajectory_reconstruction/           # 轨迹重建与分析模块
│   ├── __init__.py                      #   模块入口
│   ├── core/                            #   核心领域逻辑
│   │   ├── state.py                     #     共享运行时状态
│   │   ├── config/                      #     配置管理
│   │   │   └── config_manager.py        #       配置读写 + 校验 + 默认值
│   │   ├── io/                          #     数据文件读写
│   │   │   └── data_loader.py           #       .dat 文件 I/O
│   │   ├── prediction/                  #     轨迹预测算法
│   │   │   └── prediction.py            #       线性外推预测
│   │   └── ai/                          #     AI 建议服务
│   │       └── ai_service.py            #       硅基流动 API 调用
│   ├── services/                        #   业务逻辑层
│   │   ├── data_service.py              #     数据刷新/上传/清理
│   │   ├── predict_service.py           #     预测编排 + 参数校验
│   │   └── backup_service.py            #     备份列表/恢复
│   ├── views/                           #   HTTP 视图层
│   │   ├── __init__.py                  #     蓝图注册 + 错误处理
│   │   ├── main.py                      #     5 个页面渲染
│   │   ├── api.py                       #     数据 + 备份 API
│   │   ├── api_predict.py               #     预测 + AI API
│   │   ├── api_report.py                #     报告保存 API
│   │   └── analysis.py                  #     运动学分析数据 API
│   └── frontend/                        #   前端资源
│       ├── pages/                       #     多页面 Jinja2 模板
│       │   ├── base.html                #       基础布局
│       │   ├── index.html               #       总览（全屏 3D）
│       │   ├── predict.html             #       预测
│       │   ├── analysis.html            #       分析
│       │   ├── ai.html                  #       AI 策略
│       │   └── data.html                #       数据管理
│       └── static/                      #     CSS + JS
│           ├── css/main.css
│           └── js/
│               ├── common/toast.js
│               └── pages/
│                   ├── dashboard.js
│                   ├── predict.js
│                   ├── ai.js
│                   └── data.js
│
├── trajectory_recognition/              # 轨迹识别模块（框架）
│   ├── __init__.py                      #   模块入口
│   ├── features/                        #   特征提取（TODO）
│   ├── models/                          #   识别模型定义（TODO）
│   └── classifier/                      #   分类器（TODO）
│
├── data/                                # 共享运行时数据（两模块交互）
│   ├── fact/                            #   实际轨迹数据 (.dat)
│   ├── predict/                         #   预测结果
│   └── backup/                          #   备份
│
├── reports/                             # 捕捉策略报告
├── templates/                           # 配置模板
│   └── config_template.json             #   config.json 模板
│
└── docs/
    ├── API.md                           # API 接口文档
    └── SYSTEM.md                        # 系统架构（本文档）
```

---

## 模块职责

### trajectory_reconstruction（轨迹重建与分析）

负责轨迹数据的全生命周期管理：

| 子模块 | 职责 |
|--------|------|
| `core/config/` | 配置文件读写、校验、默认值补全 |
| `core/io/` | .dat 格式轨迹文件 I/O |
| `core/prediction/` | 基于最近两点速度向量的线性外推预测 |
| `core/ai/` | 调用硅基流动大模型生成捕捉策略 |
| `core/state.py` | 全局内存缓存（detection_methods + API 配置） |
| `services/` | 业务逻辑编排（数据/预测/备份） |
| `views/` | Flask 蓝图，页面渲染 + REST API |
| `frontend/` | 前端 UI（3D 可视化 + 数据管理） |

### trajectory_recognition（轨迹识别）

读取 `data/` 下的轨迹数据，进行模式识别与分类（框架预留，待实现）。

---

## 数据流

```
data/fact/*.dat          ← 原始轨迹数据
    ↓ load_default_data()
detection_methods (内存)  ← 全局状态
    ↓ generate_prediction()
data/predict/*.dat       ← 预测结果
    ↓
trajectory_recognition   ← 未来：读取 data/ 进行识别
```

## 页面功能

| 页面 | 路由 | 说明 |
|------|------|------|
| 总览 | `/` | 全屏 3D 视图，发光轨迹 + 带刻度坐标轴 + 拖尾粒子 + 时间轴动画 |
| 预测 | `/predict` | 参数配置 + 单/全平台线性外推预测 + 3D 预览 |
| 分析 | `/analysis` | ECharts 四图表：高度 / 速度 / 加速度 / 曲率 |
| AI | `/ai` | 多平台选择 + 硅基流动大模型捕捉策略生成 |
| 数据 | `/data` | .dat 上传 + 备份恢复 + 当前数据表格 |

## API 端点（11个）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/refresh_data` | 刷新默认轨迹 |
| POST | `/api/load_data` | 上传 .dat 文件 |
| POST | `/api/clear_all_data` | 清理并备份 |
| POST | `/api/predict` | 单平台预测 |
| POST | `/api/predict_all` | 全平台预测 |
| POST | `/api/ai_suggestion` | AI 建议 |
| POST | `/api/save_report` | 保存捕捉报告 |
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

- **双模块架构**：重建分析 + 识别分离，通过 `data/` 和 `reports/` 共享数据
- **分层设计**：core（领域逻辑）→ services（业务编排）→ views（HTTP 接口），单向依赖
- **高内聚**：模块/页面功能内聚，边界清晰
- **低耦合**：页面间无直接依赖；Views → Services → Core 单向依赖
- **薄路由**：路由层不写业务逻辑
- **纯函数**：算法/I/O 模块无副作用
