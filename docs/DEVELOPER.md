# 鹰眼长空 — 开发文档

> 面向新开发者的完整上手指南，涵盖架构、模块详解、实现方式与未开发部分的规划。

---

## 目录

1. [项目概述](#1-项目概述)
2. [环境搭建](#2-环境搭建)
3. [启动流程](#3-启动流程)
4. [目录结构](#4-目录结构)
5. [分层架构](#5-分层架构)
6. [配置系统](#6-配置系统)
7. [数据系统](#7-数据系统)
8. [轨迹合成引擎](#8-轨迹合成引擎)
9. [预测引擎](#9-预测引擎)
10. [AI 策略引擎](#10-ai-策略引擎)
11. [备份系统](#11-备份系统)
12. [前端架构](#12-前端架构)
13. [3D 渲染系统](#13-3d-渲染系统)
14. [分析系统](#14-分析系统)
15. [API 完整参考](#15-api-完整参考)
16. [识别模块（未开发）](#16-识别模块未开发)
17. [开发指南](#17-开发指南)
18. [调试与测试](#18-调试与测试)
19. [代码规范](#19-代码规范)

---

## 1. 项目概述

双模块无人机轨迹监测系统：

| 模块 | 端口 | 职责 | 状态 |
|------|------|------|------|
| `trajectory_reconstruction` | 5000 | 轨迹加载、3D 可视化、预测、AI 策略、分析 | **已完成** |
| `trajectory_recognition` | 5001 | 轨迹模式识别与分类 | **框架（待开发）** |

**技术栈：** Python Flask · pywebview · Three.js (本地) · ECharts (本地) · OpenAI 兼容 AI 接口

---

## 2. 环境搭建

### 依赖

```bash
pip install flask pywebview requests
```

### 配置 `config.json`

```json
{
    "ai": { "api_key": "sk-xxx", "url": "https://api.siliconflow.cn/v1/chat/completions", "model": "Qwen/Qwen2.5-7B-Instruct" },
    "detection_methods": {
        "visible": { "name": "可见光", "color": "#ff6b6b", "visible": true, "weight": 1.0 },
        "infrared": { "name": "红外", "color": "#4ecdc4", "visible": true, "weight": 1.0 },
        "radar": { "name": "雷达", "color": "#ffe66d", "visible": true, "weight": 1.0 },
        "self": { "name": "自选", "color": "#FF9500", "visible": true, "weight": 1.0 }
    },
    "prediction_settings": { "min_points": 10, "max_points": 50, "default_points": 20, "time_step": 0.3 },
    "capture_weights": { "height": 0.3, "speed": 0.3, "acceleration": 0.2, "curvature": 0.2 },
    "camera_speed": 0.12,
    "theme": "dark"
}
```

兼容任意 OpenAI Chat Completions 接口，切换模型只需改 `ai.url` 和 `ai.model`。

### 启动

```bash
python main.py               # 桌面窗口
python main.py --headless    # 纯 HTTP
python main.py recon         # 仅重建分析 → :5000
```

---

## 3. 启动流程

```
main.py
  ├─ threading.Thread(target=start_recog)    # 识别模块 :5001
  └─ start_recon(headless)
       └─ app.create_app()
            ├─ os.chdir(PROJECT_ROOT)          # 统一工作目录
            ├─ ensure_config()                 # 校验/创建 config.json
            ├─ migrate_legacy()                # 清理旧版备份文件
            ├─ init_from_config()              # 初始化 detection_methods 元信息
            ├─ initialize_data()               # 加载 fact/*.dat → 自动合成综合轨迹
            ├─ register_blueprints()           # 注册 5 个蓝图
            └─ register_error_handlers()       # 400/404/500
```

**关键点：** `initialize_data()` 在启动时清除 `data/predict/` 和 `data/fact/self.dat`，确保每次启动都是干净状态。

---

## 4. 目录结构

```
drone/
├── main.py                              # 统一入口
├── config.json                          # 运行时配置
├── requirements.txt
│
├── trajectory_reconstruction/           # ===== 重建分析模块 =====
│   ├── app.py                           # Flask 应用工厂 (create_app)
│   ├── __main__.py                      # python -m 独立入口
│   │
│   ├── core/                            # ---- 领域逻辑层 ----
│   │   ├── state.py                     # 全局状态 detection_methods + AI 配置
│   │   ├── config/
│   │   │   └── config_manager.py        # 配置读写、校验、默认值补全
│   │   ├── io/
│   │   │   └── data_loader.py           # .dat 文件 I/O（读写、默认加载）
│   │   ├── prediction/
│   │   │   └── prediction.py            # 线性外推预测算法
│   │   └── ai/
│   │       └── ai_service.py            # 大模型 API 调用
│   │
│   ├── services/                        # ---- 业务编排层 ----
│   │   ├── data_service.py              # 数据加载/刷新/合成/清理
│   │   ├── predict_service.py           # 预测编排 + 综合预测合成
│   │   └── backup_service.py            # 快照备份/列表/恢复/删除
│   │
│   ├── views/                           # ---- HTTP 视图层 ----
│   │   ├── __init__.py                  # 蓝图注册 + 错误处理
│   │   ├── main.py                      # 页面路由 (/, /predict, /ai, /data, /docs, /settings)
│   │   ├── api.py                       # 数据 & 备份 API + 平台切换 + 合成
│   │   ├── api_predict.py               # 预测 & AI API
│   │   ├── api_report.py                # 报告保存 API
│   │   └── analysis.py                  # 分析数据 + 捕捉时机 API
│   │
│   └── frontend/                        # ---- 前端资源（模块专属）----
│       └── pages/                       # 7 个页面模板
│           ├── index.html               # 总览（全屏 3D）
│           ├── predict.html             # 预测
│           ├── analysis.html            # 分析（ECharts + 捕捉时机卡片）
│           ├── ai.html                  # AI 策略
│           ├── data.html                # 数据管理
│           ├── docs.html                # 帮助文档
│           └── settings.html            # 设置（主题切换）
│       └── static/js/pages/             # 页面 JS 模块
│           ├── dashboard.js             # 3D 场景/轨迹/动画
│           ├── predict.js               # 预测 + 预览/动画
│           ├── ai.js                    # AI 策略交互
│           └── data.js                  # 备份管理
│
├── templates/                           # ---- 共享前端 + 配置 ----
│   ├── frontend/
│   │   ├── shared/
│   │   │   ├── base.html                # 基础布局（导航栏 + toast + 主题）
│   │   │   └── icons.html               # 25 个 SVG 图标宏
│   │   └── static/
│   │       ├── css/main.css             # GitHub 双主题样式
│   │       └── js/
│   │           ├── lib/                 # Three.js + ECharts（离线）
│   │           └── common/
│   │               ├── three-utils.js   # 坐标轴/光照/星空/主题感知
│   │               ├── three-input.js   # WASD/悬停/resize
│   │               ├── toast.js         # Toast 通知
│   │               └── utils.js         # lerp 工具
│
├── trajectory_recognition/              # ===== 识别模块（框架）=====
│   ├── app.py                           # Flask 应用工厂 (:5001)
│   ├── __main__.py                      # 独立入口
│   ├── frontend/
│   │   ├── pages/index.html             # 主页面
│   │   └── static/js/pages/recognition.js
│   ├── features/__init__.py             # 特征提取（待实现）
│   ├── models/__init__.py               # 模型定义（待实现）
│   └── classifier/__init__.py           # 分类器（待实现）
│
├── data/                                # 共享运行时数据
│   ├── fact/                            # 实际轨迹 (.dat)
│   │   ├── fact1.dat                    # 可见光（60 点，Z 字起伏）
│   │   ├── fact2.dat                    # 红外（60 点，盘旋上升）
│   │   └── fact3.dat                    # 雷达（60 点，S 形加速）
│   ├── predict/                         # 预测结果（启动时清除）
│   └── backup/                          # 快照备份
│       └── <YYYYmmdd_HHMMSS>_<label>/
│           ├── manifest.json            # 元信息
│           ├── fact/                    # 复制自 data/fact/
│           ├── predict/                 # 复制自 data/predict/
│           └── memory/                  # detection_methods dump
│
├── reports/                             # AI 策略报告输出
├── templates/                           # config_template.json
└── docs/                                # 文档
```

---

## 5. 分层架构

```
frontend/     (HTML/CSS/JS)      ← 浏览器渲染
views/        (Flask 蓝图)       ← 仅 HTTP 参数解析和 JSON 序列化
services/     (业务编排)          ← 参数校验、流程控制、组合调用
core/         (领域逻辑)          ← 纯函数，不依赖 Flask、不依赖 services
```

**依赖方向：** views → services → core（单向，不可逆）

**原则：**
- `core/` 不导入 Flask、services、views
- `services/` 不导入 Flask、views
- `views/` 只做参数提取和 JSON 序列化，业务逻辑委托给 services

---

## 6. 配置系统

### 6.1 文件：`core/config/config_manager.py`

**职责：** 配置文件的全生命周期管理。

**核心函数：**

| 函数 | 输入 | 输出 | 实现 |
|------|------|------|------|
| `ensure_config()` | 无 | `dict` | 若 config.json 不存在，从模板复制或使用 DEFAULT_CONFIG 创建；读取后调用 `_validate_config()` 补全缺失字段 |
| `save_config(cfg)` | `dict` | 无 | `json.dump` 写入 config.json，indent=4 |
| `_validate_config(cfg)` | `dict` | `dict` | 遍历 DEFAULT_CONFIG 所有键，递归补全缺失的顶层/子字段（ai, detection_methods, prediction_settings, theme, camera_speed, capture_weights, synthesis_weights） |

**实现细节：**
- DEFAULT_CONFIG 包含所有字段的硬编码默认值，作为最后兜底
- `config_template.json` 优先于 DEFAULT_CONFIG（模板存在时从模板复制）
- 校验采用"存在即跳过，缺失即补全"策略，不覆盖已有值
- 配置文件编码为 UTF-8

### 6.2 配置字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ai.api_key` | str | `""` | API 密钥 |
| `ai.url` | str | siliconflow URL | OpenAI 兼容端点 |
| `ai.model` | str | `Qwen/Qwen2.5-7B-Instruct` | 模型名称 |
| `detection_methods.<id>.weight` | float | 1.0 | 综合轨迹合成权重 |
| `prediction_settings.time_step` | float | 0.3 | 预测时间步长（秒） |
| `capture_weights.height` | float | 0.3 | 捕捉评分高度权重 |
| `capture_weights.speed` | float | 0.3 | 捕捉评分速度权重 |
| `capture_weights.acceleration` | float | 0.2 | 捕捉评分加速度权重 |
| `capture_weights.curvature` | float | 0.2 | 捕捉评分曲率权重 |
| `camera_speed` | float | 0.12 | WASD 移动速度 |
| `theme` | str | `"dark"` | 默认主题 |

---

## 7. 数据系统

### 7.1 数据文件 I/O：`core/io/data_loader.py`

| 函数 | 实现 |
|------|------|
| `load_dat_file(path)` | 逐行读取 `x y z t`，空格分隔，异常返回空列表 |
| `save_predict_data(mid, pts, ts)` | 按 method_id 映射文件名（visible→pre1.dat 等），写入 data/predict/ |
| `load_default_data()` | 加载 fact1/2/3.dat 到 {visible, infrared, radar} 字典 |

**文件映射：**
```
visible  → fact1.dat / pre1.dat
infrared → fact2.dat / pre2.dat
radar    → fact3.dat / pre3.dat
self     → self.dat / preself.dat
synthetic → 不落盘 / presyn.dat
```

### 7.2 数据服务：`services/data_service.py`

**核心函数：**

| 函数 | 实现 |
|------|------|
| `initialize_data()` | 清除 predict/ 和 self.dat → 加载 fact/*.dat → 清空 self 内存 → 自动合成综合轨迹 → save_metadata() |
| `refresh_fact_data()` | 重载 fact 文件 → 清空 self 和预测 → 重新合成综合轨迹 |
| `load_self_data(pts, ts)` | 创建/更新 self 平台 → 保存 self.dat → 自动合成 → save_metadata() |
| `clear_all_data()` | create_backup("auto") → 删除 fact/ 和 predict/ 文件 → 清空内存 |
| `save_metadata()` | 将 detection_methods 元信息（name/color/visible/weight）写回 config.json，不包含轨迹点 |
| `_auto_synthesize()` | 静默调用 synthesize_trajectory()，异常时 pass |

### 7.3 全局状态：`core/state.py`

```python
detection_methods = {
    "visible": {"name": "可见光", "color": "#ff6b6b", "weight": 1.0,
                "visible": True, "points": [[x,y,z],...], "timestamps": [t,...]},
    "infrared": {...}, "radar": {...}, "self": {...}, "synthetic": {...}
}
AI_API_KEY, AI_URL, AI_MODEL  # 从 config.json ai 字段读取
```

`init_from_config()` 按 `_METHOD_ORDER` 固定顺序初始化，确保各页面图例顺序一致。

---

## 8. 轨迹合成引擎

### 8.1 文件：`services/data_service.py` → `synthesize_trajectory()`

**算法流程：**

```
1. 收集活跃平台及其权重
   for mid in ['visible', 'infrared', 'radar', 'self']:
       if 有数据且 weight > 0: active[mid] = weight

2. 构建统一时间轴
   收集所有活跃平台的时间戳 → set 去重 → sorted 排序

3. 逐时间点加权平均
   for t in sorted_ts:
       for mid in active:
           p = _interpolate(mid, t) or _nearest_avg(mid, t)
           weighted_sum += p * weight
       if weight_sum > 0: syn_points.append(weighted_sum / weight_sum)

4. 平滑处理（两次三点移动平均）
   for _ in range(2):
       smoothed[i] = (points[i-1] + points[i] + points[i+1]) / 3

5. 更新 detection_methods['synthetic']
```

**插值函数 `_interpolate(mid, t)`：**
- 二分查找时间戳区间
- 线性插值：`ratio = (t - t0) / (t1 - t0)`，`result = p0 + (p1-p0) * ratio`
- 边界：t 在范围外返回最近端点

**兜底函数 `_nearest_avg(mid, t)`：**
- 当 `_interpolate` 返回 None 时触发
- 找到 t 前后的最近数据点，取坐标平均值

**触发时机：**
- `initialize_data()` — 启动时自动
- `refresh_fact_data()` — 更新数据时
- `load_self_data()` — 上传自选后
- `restore_backup()` — 恢复备份后

---

## 9. 预测引擎

### 9.1 核心算法：`core/prediction/prediction.py`

```python
def generate_prediction(points, timestamps, num_points=5, time_step=0.5):
    last, prev = points[-1], points[-2]
    dt = timestamps[-1] - timestamps[-2]
    v = (last - prev) / dt          # 速度向量
    pred[0] = last                   # 首点 = 历史末点（保证连接）
    for i in 1..num_points:
        pred[i] = last + v * i * time_step
    return pred_points, pred_times  # 共 num_points + 1 个点
```

**数学本质：** 常数速度线性外推，假设最近的运动趋势不变。

### 9.2 编排服务：`services/predict_service.py`

| 函数 | 实现 |
|------|------|
| `predict_single(mid, n, ts)` | 从 detection_methods 取历史点 → generate_prediction → save_predict_data |
| `predict_all(n, ts)` | 遍历所有可见平台调用 predict_single → 合成综合预测 → save_predict_data('synthetic') |
| `clamp_params(n, ts)` | 将 num_points 限制在 [min, max]，time_step 填充默认值 |

**综合预测合成 `_synthesize_predictions()`：**
- 与 `synthesize_trajectory()` 算法完全一致
- 区别：输入为各平台的预测结果（而非历史数据）
- 包含相同的插值 + 兜底 + 平滑流程

---

## 10. AI 策略引擎

### 10.1 文件：`core/ai/ai_service.py`

```python
def get_ai_suggestion(methods_data, api_key, url, model) -> str:
    prompt = 构建包含轨迹统计的 system prompt
    response = requests.post(url, headers, json={
        "model": model,
        "messages": [
            {"role": "system", "content": "无人机反制专家"},
            {"role": "user", "content": prompt}
        ]
    }, timeout=30)
    return response["choices"][0]["message"]["content"]
```

**Prompt 构建：** 将各平台的名称、点数、高度范围、速度范围、坐标范围注入 prompt，要求输出运动模式识别、推荐捕捉设备、最佳拦截点、时间窗口建议。

---

## 11. 备份系统

### 11.1 文件：`services/backup_service.py`

**快照结构：**
```
data/backup/20260630_143052_manual/
├── manifest.json        # {created, label, methods: {id: {name, color, point_count}}}
├── fact/                # shutil.copy2 复制 data/fact/
├── predict/             # 复制 data/predict/（备份时保存，恢复时跳过）
└── memory/              # detection_methods dump (<id>.dat 格式)
```

**核心函数：**

| 函数 | 实现 |
|------|------|
| `create_backup(label)` | 创建时间戳目录 → 复制 fact/predict → dump memory → 写 manifest |
| `list_backups()` | 遍历 data/backup/ → 读 manifest → 按时间倒序返回 |
| `restore_backup(name)` | `_clear_all()` → 复制 fact 文件 → 加载 memory dump → `synthesize_trajectory()` |
| `restore_all_latest()` | 取 `list_backups()[0]` 调用 `restore_backup` |
| `delete_backup(name)` | `shutil.rmtree` |
| `migrate_legacy()` | 启动时清理旧扁平 .dat 文件和旧 fact/predict 子目录 |
| `_clear_all()` | 清空 detection_methods 内存点 → 删除 self/synthetic 键 → 删除 fact/predict 磁盘文件 |

**注意：** 恢复时不还原 predict 文件——预测数据应在需要时重新生成。

---

## 12. 前端架构

### 12.1 模板系统

模板分为两层：
- **共享层** `templates/frontend/shared/` — base.html + icons.html，两模块共用
- **模块层** `trajectory_reconstruction/frontend/pages/` — 7 个页面模板

通过 `app.jinja_loader.searchpath` 将共享目录加入搜索路径，各页面 `{% extends "base.html" %}` 和 `{% from "icons.html" import icon_xxx %}` 无需修改路径。

**icons.html：** 25 个纯几何 SVG 图标宏，`currentColor` 继承文本色。

### 12.2 CSS 架构

**文件：** `templates/frontend/static/css/main.css`（GitHub 风格双主题）

- `:root` 定义深色主题变量（--bg-root: #0d1117, --blue: #58a6ff, --green: #3fb950 等）
- `[data-theme="light"]` 覆盖浅色主题变量
- 所有颜色/圆角/阴影/过渡通过 CSS 变量定义
- 组件：导航栏（毛玻璃）、卡片、按钮（primary/danger/secondary/ghost）、表单、表格、模态框、toast

### 12.3 JS 架构

**模块化原则：**
- 每页独立 ES6 模块（`type="module"`）
- 共享工具提取到 `common/`
- Three.js 通过 `importmap` 加载

**共享模块：**

| 文件 | 导出函数 | 功能 |
|------|---------|------|
| `three-utils.js` | `buildAxes`, `buildLights`, `buildStarfield`, `getSceneBackground`, `getFogColor`, `getGridColor` | 3D 场景构建 + 双主题感知 |
| `three-input.js` | `bindWASD`, `bindCoordTooltip`, `bindResize` | 键盘控制 + 鼠标悬停 + 窗口适配 |
| `toast.js` | `toast.show/error/success/warning` | 轻量通知 |
| `utils.js` | `lerp` | 线性插值 |

**页面模块：**

| 文件 | 行数 | 功能 |
|------|------|------|
| `dashboard.js` | ~500 | 3D 场景初始化、轨迹渲染、时间轴动画、平台切换、鼠标拾取 |
| `predict.js` | ~500 | 独立 3D 场景、预测渲染、动画播放 |
| `ai.js` | ~100 | AI 请求 + sessionStorage 持久化 + 报告保存 |
| `data.js` | ~200 | 上传/备份列表/恢复/删除/更新 |
| `analysis.html` (内嵌 JS) | ~150 | ECharts 图表渲染 + 捕捉时机卡片 |

---

## 13. 3D 渲染系统

### 13.1 场景初始化（dashboard.js / predict.js）

```
1. PerspectiveCamera(50, W/H, 0.1, 500)
2. WebGLRenderer { antialias: true, shadowMap: PCFSoft, toneMapping: ACESFilmic }
3. CSS2DRenderer（用于文字标签）
4. OrbitControls { enableDamping, minDistance: 1, maxDistance: 200 }
5. 场景背景 + 雾（getSceneBackground / getFogColor 主题感知）
```

### 13.2 光照（three-utils.js → buildLights）

- 深色模式：冷蓝调环境光 + 半球光 + 暖色方向光（带阴影）+ 蓝色补光
- 浅色模式：亮度提升、色温偏暖

### 13.3 轨迹渲染（dashboard.js）

- `buildTrailMesh(points, color)` — CatmullRom 曲线 + 外发光层（AdditiveBlending）
- `buildSpheres(points, color, size, opacity, timestamps)` — 球体标记，存 userData {pts, ts, idx}
- 预测线：`LineDashedMaterial` + 半透明球体
- 拖尾粒子：2000 粒子/平台，AdditiveBlending，life 衰减

### 13.4 坐标轴（three-utils.js → buildAxes）

- 自定义三轴（X 红 Y 绿 Z 蓝），深/浅模式颜色自适应
- 正半轴实线 + 负半轴虚线 + 刻度标记 + 数字标签 + 锥形箭头 + 轴字母
- 使用 CSS2DRenderer 渲染文字标签

### 13.5 动画系统（dashboard.js / predict.js）

```
startAnim(fromStart)
  animActive = true
  requestAnimationFrame(step)
    animElapsed = (now - startWall) / 1000 * speed
    ts = timeRange.start + animElapsed
    animStep(ts)
      → lerp 各平台历史/预测坐标
      → 更新球体位置 + 拖尾粒子
      → 更新状态面板
```

- 支持暂停/恢复、调速（0.5×/1×/2×/4×）、进度条拖拽

### 13.6 WASD 自由视角（three-input.js → bindWASD）

```
每帧读取按键状态
W/S → camera.position + controls.target 沿视线方向前后移动
A/D → 沿视线右向量左右移动
Q/E → 垂直升降
速度从 window._PAGE_DATA_.cameraSpeed 读取（config 可配）
```

### 13.7 坐标拾取（three-input.js → bindCoordTooltip）

```
mousemove → raycaster.setFromCamera
  → 收集 lines + predLines 中所有球体（排除非标准 geometry）
  → intersectObjects → 命中则放大 2.5× + 增强自发光
  → 读取 userData.ts[userData.idx] 获取时间戳
  → 渲染 HTML tooltip: X(红) Y(绿) Z(蓝) T(橙)
```

---

## 14. 分析系统

### 14.1 运动学分析 API：`GET /analysis/data`

**实现（analysis.py）：**
1. 遍历 `_METHOD_ORDER` → 过滤 visible + points>=2 + self检查
2. 拼接原始点 + 预测点（跳过重复首点，用时间戳判断）
3. 调用 `_compute_metrics(points, timestamps)` 统一计算

**_compute_metrics 算法：**
- **速度：** `sqrt(dx² + dz²) / dt`（仅水平速度）
- **加速度：** `speed[i] - speed[i-1]`（相邻速度差）
- **曲率：** 三点法 `|d1 × d2| / |d1|³`
- **高度：** 直接提取 `points[i][1]`

### 14.2 最佳捕捉时机 API：`GET /analysis/capture`

**实现（analysis.py）：**
1. 仅分析综合轨迹（synthetic），无预测数据时返回空数组
2. 读取 `capture_weights` 配置
3. 对预测部分的每个点计算评分：

```
h_norm = 1 - height / max_height     # 越低越好
s_norm = 1 - speed / max_speed       # 越低越好
a_norm = 1 - |accel| / max_accel     # 越平稳越好
c_norm = 1 - curvature / max_curv    # 越直越好
score = w_h*h_norm + w_s*s_norm + w_a*a_norm + w_c*c_norm
```

4. 排序取前三，返回 `[{rank, time, position, score, height, speed}]`

### 14.3 前端图表（analysis.html）

- ECharts 深色主题，共享时间轴（去重排序）
- 每平台一条连续曲线（原始+预测拼接）
- 图例按 `LEGEND_ORDER` 固定排序
- 捕捉时机卡片：预测后显示前三名（#1/#2/#3，金银铜色）

---

## 15. API 完整参考

18 个端点，统一格式：`{"success": true, ...}` 或 `{"success": false, "error": "..."}`

### 数据管理（api.py）

| 方法 | 端点 | 请求体 | 返回 |
|------|------|--------|------|
| POST | `/api/refresh_data` | - | `{success}` |
| POST | `/api/load_data` | multipart: file + method_id=self | `{success, name, method_id}` |
| POST | `/api/clear_all_data` | - | `{success, backup_name}` |
| POST | `/api/toggle_method` | `{method_id}` | `{success, visible}` |
| POST | `/api/synthesize` | - | `{success, point_count, platforms, weights}` |

### 预测 & AI（api_predict.py）

| 方法 | 端点 | 请求体 | 返回 |
|------|------|--------|------|
| POST | `/api/predict` | `{method_id, num_points?, time_step?}` | `{success, prediction, pred_times}` |
| POST | `/api/predict_all` | `{num_points?, time_step?}` | `{success, results: {id: {prediction, pred_times}}}` |
| POST | `/api/ai_suggestion` | `{methods_data}` | `{success, suggestion}` |
| POST | `/api/save_report` | `{content, platforms?}` | `{success, filepath}` |

### 备份（api.py）

| 方法 | 端点 | 请求体 | 返回 |
|------|------|--------|------|
| GET | `/api/list_backups` | - | `{success, backups: [{filename, timestamp, label, method, point_count}]}` |
| POST | `/api/restore_backup` | `{backup_file}` | `{success, message}` |
| POST | `/api/restore_all_backups` | - | `{success, message, restored}` |
| POST | `/api/backup/create` | - | `{success, message, name}` |
| POST | `/api/backup/delete` | `{backup_name}` | `{success}` |

### 分析（analysis.py）

| 方法 | 端点 | 返回 |
|------|------|------|
| GET | `/analysis/data` | `{id: {name, color, speeds, accelerations, curvatures, heights, time_steps}}` |
| GET | `/analysis/capture` | `[{rank, time, position, score, height, speed}]` 或 `[]` |

### 其他（main.py）

| 方法 | 端点 | 说明 |
|------|------|------|
| GET/POST | `/api/theme` | 读写主题设置 |

---

## 16. 识别模块（未开发）

### 16.1 现状

`trajectory_recognition/` 当前为**纯框架**：
- `app.py` — Flask 应用工厂，端口 5001，提供 `/` 页面和 `/health` `/api/status` 端点
- `frontend/pages/index.html` — 展示"待实现"状态卡片
- `recognition.js` — 尝试跨域请求 `:5000/analysis/data`（已失效）
- `features/` `models/` `classifier/` — `__init__.py` 仅有 TODO 注释

### 16.2 规划架构

```
trajectory_recognition/
├── app.py
├── features/                    # 特征提取
│   ├── __init__.py
│   ├── kinematics.py            # 运动学特征：速度/加速度/曲率统计量
│   ├── geometry.py              # 几何特征：轨迹形状描述符
│   └── frequency.py             # 频域特征：FFT 频谱分析
├── models/                      # 识别模型
│   ├── __init__.py
│   ├── rule_based.py            # 规则分类器（基线）
│   ├── ml_classifier.py         # 机器学习（SVM/随机森林）
│   └── dl_classifier.py         # 深度学习（LSTM/Transformer）
└── classifier/                  # 分类器
    ├── __init__.py
    ├── pipeline.py              # 推理流水线
    └── labels.py                # 标签定义（悬停/巡航/俯冲/盘旋等）
```

### 16.3 开发步骤

**Phase 1 — 特征提取（2-3 天）**

1. 实现 `kinematics.py`：
   - 从已有分析 API (`/analysis/data`) 读取数据
   - 提取统计特征：速度均值/方差/最大值、加速度均值/方差、曲率均值
   - 提取时序特征：速度单调性、高度变化趋势

2. 实现 `geometry.py`：
   - 轨迹总长度、直线距离/轨迹长度比（效率比）
   - 包围盒体积、轨迹平面拟合残差

**Phase 2 — 规则分类器（1-2 天）**

3. 实现 `rule_based.py`：
   - 定义阈值规则：悬停（速度<0.5m/s）、巡航（匀速直线）、俯冲（高度单调递减+速度>阈值）
   - 输出标签 + 置信度

**Phase 3 — 机器学习（3-5 天）**

4. 实现 `ml_classifier.py`：
   - 训练数据：从 fact/*.dat 提取特征 + 人工标注
   - 模型选择：随机森林（特征可解释）+ SVM（小样本）
   - 保存模型到 `trajectory_recognition/models/saved/`

5. 实现推理流水线 `pipeline.py`：
   - 加载模型 → 提取特征 → 推理 → 返回标签+概率

**Phase 4 — 前端集成（1-2 天）**

6. 更新 `recognition.js`：
   - 通过 `/api/classify` 获取分类结果
   - 在页面上展示：轨迹类型、飞行模式、异常检测
   - 使用 ECharts 渲染分类置信度雷达图

7. 注册新的 API 端点：
   - `POST /api/extract_features` — 特征提取
   - `POST /api/classify` — 轨迹分类

### 16.4 数据共享

识别模块通过共享 `data/` 目录读取重建模块的数据：
- `data/fact/*.dat` — 原始轨迹
- `data/predict/*.dat` — 预测轨迹
- 也可通过 HTTP 调用 `:5000/analysis/data` 获取分析数据

### 16.5 改进方案

当前 `recognition.js` 硬编码 `http://127.0.0.1:5000` CORS 请求。建议改为：
1. 服务端渲染时注入 `window.ANALYSIS_URL`
2. 或识别模块直接读取共享 `data/` 文件

---

## 17. 开发指南

### 17.1 添加新 API 端点

```python
# 在 views/api.py 或新建蓝图文件
@api_bp.route('/api/my_feature', methods=['POST'])
def my_feature():
    data = request.get_json(silent=True) or {}
    if not data.get('required_param'):
        return jsonify({'success': False, 'error': '缺少参数'}), 400
    result = some_service.do_work(data)
    return jsonify({'success': True, 'result': result})

# 在 views/__init__.py 注册蓝图（如果是新文件）
app.register_blueprint(my_bp)
```

### 17.2 添加新页面

1. `frontend/pages/new.html` → `{% extends "base.html" %}` + `{% block content %}` + `{% block scripts %}`
2. `frontend/static/js/pages/new.js` → ES6 模块
3. `views/main.py` → `@main_bp.route('/new')` → `render_template('new.html', **ctx)`
4. `base.html` 导航栏添加链接 + 图标

### 17.3 添加新检测平台

1. `config.json` 的 `detection_methods` 添加条目（含 name/color/weight）
2. `data/fact/` 放入对应 .dat 文件
3. `core/io/data_loader.py` 的 `load_default_data()` 添加文件映射
4. `views/main.py` 和 `views/analysis.py` 的 `_METHOD_ORDER` 添加新 key

### 17.4 添加新图标

在 `frontend/pages/icons.html` 添加 Jinja2 宏：
```jinja2
{% macro icon_new(size='1em') -%}
<svg width="{{ size }}" height="{{ size }}" viewBox="0 0 16 16" fill="none"
     stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
  <!-- 几何图形 -->
</svg>
{%- endmacro %}
```
在模板中：`{% from "icons.html" import icon_new %}` → `{{ icon_new() }}`

---

## 18. 调试与测试

```bash
# 查看所有路由
python -c "
from trajectory_reconstruction.app import create_app
for r in sorted(create_app().url_map.iter_rules(), key=lambda x: x.rule):
    print(f'{r.rule:40s} {sorted(r.methods)}')"

# 查看全局状态
python -c "
from trajectory_reconstruction.core.state import init_from_config, detection_methods
from trajectory_reconstruction.services.data_service import initialize_data
init_from_config(); initialize_data()
for mid, d in detection_methods.items():
    print(f'{mid}: {len(d[\"points\"])} pts, visible={d[\"visible\"]}, weight={d.get(\"weight\",1)}')"

# 查看备份内容
ls -R data/backup/ && cat data/backup/*/manifest.json

# 重置环境
rm -rf data/backup/* data/predict/* reports/* && git restore data/fact/

# JS 语法检查
node --check trajectory_reconstruction/frontend/static/js/pages/dashboard.js
```

---

## 19. 代码规范

- **Python：** 类型注解，导入顺序：标准库 → 第三方 → 项目模块；`core/` 不导入 Flask
- **JavaScript：** ES6 模块，async/await，共享逻辑提取到 `common/`；无全局变量污染
- **CSS：** 所有值使用 `var(--xxx)`，支持 `[data-theme="light"]`；不硬编码颜色
- **分层：** core → services → views 单向依赖
- **提交：** 中文描述，简洁准确
- **图标：** 全部使用 SVG，禁止 emoji
