---
title: "鹰眼长空 — 使用手册"
subtitle: "基于多平台协同的低空无人机智能监测系统"
author: "LanGYou"
date: "2026-06-30"
lang: zh-CN
toc: true
toc-depth: 3
mainfont: Microsoft YaHei
monofont: Cascadia Code
CJKmainfont: Microsoft YaHei
fontsize: 11pt
geometry: margin=2.5cm
---

# 系统概述

**鹰眼长空**是一个桌面端无人机轨迹监测与预测系统，融合可见光、红外、雷达三种传感平台的轨迹数据，提供 3D 可视化、加权合成、轨迹预测、捕捉时机分析和 AI 策略生成功能。

**技术栈：** Python Flask + pywebview + Three.js + ECharts + 硅基流动 AI

---

# 快速启动

## 环境要求

- Python 3.10+
- 依赖包：`flask`, `pywebview`, `requests`

## 安装与运行

```bash
pip install -r requirements.txt
python main.py               # 桌面窗口模式
python main.py --headless    # 纯 HTTP → http://127.0.0.1:5000
python main.py recon         # 仅重建分析 → :5000
```

## 配置文件

编辑 `config.json` 填入 API Key 后可使用 AI 策略功能。主要字段：

```json
{
  "ai": { "api_key": "你的密钥", "url": "...", "model": "..." },
  "detection_methods": {
    "visible": { "color": "#ff6b6b", "weight": 1.0 },
    "infrared": { "color": "#4ecdc4", "weight": 1.0 },
    "radar": { "color": "#ffe66d", "weight": 1.0 },
    "self": { "color": "#FF9500", "weight": 1.0 }
  },
  "capture_weights": { "height": 0.3, "speed": 0.3, "acceleration": 0.2, "curvature": 0.2 },
  "camera_speed": 0.12,
  "theme": "dark"
}
```

- `detection_methods[].weight` — 综合轨迹合成权重，越大越重要
- `capture_weights` — 最佳捕捉时机评分权重
- `camera_speed` — WASD 移动速度
- `theme` — 默认主题（dark / light）

---

# 界面总览

系统采用 GitHub 风格双主题 UI，顶部导航栏切换 5 个功能页面。

## 总览

路由：`/`

全屏 3D 轨迹视图，综合展示所有检测平台。

**3D 操作：**

| 操作 | 说明 |
|------|------|
| 鼠标左键拖拽 | 旋转视角 |
| 鼠标滚轮 | 缩放 |
| 鼠标右键拖拽 | 平移视角 |
| W/A/S/D | 摄像头前后左右平移 |
| Q/E | 摄像头升降 |
| 鼠标悬停轨迹点 | 球体放大，显示坐标 (X/Y/Z) 和时间 (T) |
| Space | 播放 / 暂停轨迹动画 |

**覆盖层：**

- **左上角 — 检测平台**：点击切换平台显隐，影响后续所有页面
- **右上角 — 平台状态**：实时速度/高度/航向角，点击标题折叠
- **底部 — 播放控制条**：播放/暂停、倍速、进度条、预测入口

**3D 渲染：** 自定义坐标轴 + CatmullRom 发光轨迹线 + 拖尾粒子系统 + ACES 色调映射

## 预测

路由：`/predict`

基于最后两点速度向量线性外推。各平台独立预测，综合轨迹由其他平台预测加权合成。

1. 选择目标平台（单个或所有）
2. 调整预测点数（10-50）和时间步长（0.1-5s）
3. 点击"开始预测"
4. 预测结果以虚线和半透明球体显示
5. 播放动画查看历史+预测全程

## 分析

路由：`/analysis`

多维运动学分析：

| 图表 | 计算方式 |
|------|---------|
| 高度变化 | 原始 Y 坐标 |
| 速度变化 | 水平位移 / 时间差 |
| 加速度变化 | 相邻速度差 |
| 曲率变化 | 三点法平面曲率 |

原始数据与预测数据拼接为连续曲线。图例固定排序：可见光 → 红外 → 雷达 → 自选 → 综合。

**最佳捕捉时机：** 需先执行预测，基于高度/速度/加速度/曲率加权评分，给出前三名捕捉时间点与坐标。

## AI 策略

路由：`/ai`

接入大模型生成无人机反制捕捉建议。勾选平台 → 生成策略 → 保存报告至 `reports/`。

## 数据管理

路由：`/data`

| 区域 | 功能 |
|------|------|
| 上传 | `.dat` 文件创建自选平台 |
| 数据更新 | 重置出厂轨迹，清空自选和预测 |
| 备份管理 | 创建快照 / 一键恢复 / 查看列表 |
| 数据概览 | 各平台点数/时间范围/起止坐标 |

---

# 综合轨迹

系统启动或数据更新后，自动将可见光、红外、雷达（及自选）加权合成为白色综合轨迹。权重在各平台 `weight` 字段配置。合成使用时间对齐插值 + 两次平滑处理。

---

# API 接口

统一响应格式：`{"success": true, ...}` 或 `{"success": false, "error": "..."}`

## 数据管理

| 端点 | 说明 |
|------|------|
| `POST /api/refresh_data` | 重载出厂轨迹，清空自选和预测 |
| `POST /api/load_data` | 上传自选数据（multipart: file + method_id=self） |
| `POST /api/clear_all_data` | 备份后清空全部数据 |

## 预测

| 端点 | 说明 |
|------|------|
| `POST /api/predict` | 单平台预测 |
| `POST /api/predict_all` | 全平台预测（综合由加权合成） |

## AI

| 端点 | 说明 |
|------|------|
| `POST /api/ai_suggestion` | 生成捕捉策略 |
| `POST /api/save_report` | 保存报告至 reports/ |

## 备份

| 端点 | 说明 |
|------|------|
| `GET /api/list_backups` | 列出备份列表 |
| `POST /api/restore_backup` | 恢复指定备份 |
| `POST /api/restore_all_backups` | 一键恢复最新 |
| `POST /api/backup/create` | 手动创建备份 |
| `POST /api/backup/delete` | 删除指定备份 |

## 分析

| 端点 | 说明 |
|------|------|
| `GET /analysis/data` | 运动学数据（速度/加速度/曲率/高度） |
| `GET /analysis/capture` | 最佳捕捉时机（预测后可用，前三名） |

## 其他

| 端点 | 说明 |
|------|------|
| `GET/POST /api/theme` | 读写主题设置 |
| `POST /api/toggle_method` | 切换平台可见性 |
| `POST /api/synthesize` | 手动触发综合轨迹合成 |

---

# 数据格式

## .dat 轨迹文件

每行 4 个空格分隔数值：

```
x y z t
```

| 字段 | 含义 | 单位 |
|------|------|------|
| x | 东西方向 | 米 |
| y | 高度 | 米 |
| z | 南北方向 | 米 |
| t | 时间戳 | 秒 |

## 测试数据

`data/fact/` 下有 3 组测试轨迹（各 60 点）：fact1（可见光，Z 字起伏）、fact2（红外，盘旋上升）、fact3（雷达，S 形加速）。

---

# 常见问题

**Q: 预测后分析页综合轨迹没数据？**
A: 综合轨迹的预测由各平台预测加权合成，确保至少有两个平台有预测数据。

**Q: 如何切换明暗主题？**
A: 设置页面点击主题切换，自动保存到 config.json 和浏览器。

**Q: 自选平台不显示？**
A: 上传 .dat 数据后自动创建，启动时临时数据清除。

**Q: 预测动画球体消失了？**
A: 暂停后球体保持在当前位置。如需清除，刷新页面。
