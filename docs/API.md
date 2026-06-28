# 鹰眼长空 API 接口文档

> 低空无人机智能监测系统 · 后端 API 参考

---

## 目录

- [概述](#概述)
- [基础信息](#基础信息)
- [接口列表](#接口列表)
  - [1. 刷新轨迹数据](#1-刷新轨迹数据)
  - [2. 加载外部数据](#2-加载外部数据)
  - [3. 清理所有数据并备份](#3-清理所有数据并备份)
  - [4. 预测（单平台）](#4-预测单平台)
  - [5. 预测（所有平台）](#5-预测所有平台)
  - [6. 获取 AI 建议](#6-获取-ai-建议)
  - [7. 保存报告](#7-保存报告)
  - [8. 列出备份](#8-列出备份)
  - [9. 恢复指定备份](#9-恢复指定备份)
  - [10. 一键恢复最新](#10-一键恢复最新)
  - [11. 创建备份](#11-创建备份)
  - [12. 删除备份](#12-删除备份)
  - [13. 获取分析数据](#13-获取分析数据)
- [数据格式说明](#数据格式说明)
- [错误处理](#错误处理)

---

## 概述

系统提供 RESTful API，用于无人机轨迹数据的管理、预测和分析。所有接口基于 Flask 蓝图注册，返回 JSON 格式数据。

**技术栈:** Python Flask · 蓝图路由 · JSON API

---

## 基础信息

| 项目 | 说明 |
|------|------|
| 协议 | HTTP/1.1 |
| 数据格式 | JSON |
| 编码 | UTF-8 |
| 基础路径 | `http://127.0.0.1:5000` |

### 统一响应格式

```json
{
    "success": true,       // 布尔值，请求是否成功
    "message": "...",      // 字符串，描述信息（部分接口）
    "error": "..."         // 字符串，错误描述（仅失败时）
}
```

---

## 接口列表

### 1. 刷新轨迹数据

重新加载 `data/fact/` 目录下的默认轨迹文件。

```
POST /api/refresh_data
```

**请求体:** 无

**响应示例:**
```json
{
    "success": true
}
```

**前端调用:**
```javascript
const response = await fetch('/api/refresh_data', { method: 'POST' });
const result = await response.json();
// result.success === true
```

---

### 2. 加载外部数据

上传用户自选的 `.dat` 轨迹文件，创建"自选"检测手段。

```
POST /api/load_data
Content-Type: multipart/form-data
```

**表单参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | ✅ | `.dat` 格式文件（每行: `x y z t`） |
| `method_id` | String | ✅ | 固定值 `"self"`（表示自选平台） |

**响应示例:**
```json
{
    "success": true,
    "method_id": "self",
    "name": "自选"
}
```

**前端调用:**
```javascript
const formData = new FormData();
formData.append('file', fileObject);
formData.append('method_id', 'self');

const response = await fetch('/api/load_data', {
    method: 'POST',
    body: formData
});
```

**`.dat` 文件格式说明:**
```
x1 y1 z1 t1
x2 y2 z2 t2
...
```
每行 4 个浮点数：三维坐标 (x, y, z) 和时间戳 (t)，用空格分隔。

---

### 3. 清理所有数据并备份

自动创建快照备份后清空全部数据。

```
POST /api/clear_all_data
```

**请求体:** 无

**响应示例:**
```json
{
    "success": true,
    "message": "数据已清理，备份: 20260629_143052_auto",
    "backup_name": "20260629_143052_auto"
}
```

---

### 4. 预测（单平台）

对指定检测平台的轨迹进行线性外推预测。

```
POST /api/predict
Content-Type: application/json
```

**请求体:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `method_id` | String | ✅ | - | 平台 ID（`visible`/`infrared`/`radar`/`self`） |
| `points` | Array | ✅ | - | 历史轨迹点 `[[x,y,z], ...]` |
| `timestamps` | Array | ❌ | `[]` | 各点对应时间戳 |
| `num_points` | Integer | ❌ | `6` | 预测点数（受 min/max 约束） |
| `time_step` | Float | ❌ | `0.5` | 预测点时间间隔（秒） |

**响应示例:**
```json
{
    "success": true,
    "prediction": [
        [7.5, 12.3, 9.1],
        [8.2, 13.1, 10.3],
        [8.9, 13.9, 11.5],
        [9.6, 14.7, 12.7],
        [10.3, 15.5, 13.9],
        [11.0, 16.3, 15.1]
    ],
    "pred_times": [5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
}
```

**前端调用:**
```javascript
const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        method_id: 'visible',
        points: [[1,2,3], [2,3,4], [3,4,5]],
        timestamps: [0.0, 0.5, 1.0],
        num_points: 6,
        time_step: 0.5
    })
});
```

---

### 5. 预测（所有平台）

对所有可见检测手段同时进行预测。

```
POST /api/predict_all
Content-Type: application/json
```

**请求体:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `num_points` | Integer | ❌ | `6` | 每个平台的预测点数 |
| `time_step` | Float | ❌ | `0.5` | 预测点时间间隔（秒） |

**响应示例:**
```json
{
    "success": true,
    "results": {
        "visible": {
            "prediction": [[7.5, 12.3, 9.1], ...],
            "pred_times": [5.5, 6.0, ...]
        },
        "infrared": {
            "prediction": [[6.8, 11.5, 8.2], ...],
            "pred_times": [5.5, 6.0, ...]
        }
    }
}
```

**说明:** 仅对 `visible: true` 且有 ≥2 个历史点的平台进行预测。

---

### 6. 获取 AI 建议

调用硅基流动大模型 API，根据多平台轨迹数据生成无人机捕捉策略建议。

```
POST /api/ai_suggestion
Content-Type: application/json
```

**请求体:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `methods_data` | Object | ✅ | 所有检测手段数据 |

**`methods_data` 结构:**
```json
{
    "visible": {
        "name": "可见光",
        "points": [[x,y,z], ...],
        "timestamps": [t, ...]
    },
    "infrared": { ... },
    "radar": { ... }
}
```

**响应示例:**
```json
{
    "success": true,
    "suggestion": "根据多平台轨迹分析，无人机呈东南方向匀速飞行...\n建议:\n1. 使用网捕设备在坐标(8.5, 13.0, 10.0)附近拦截\n2. 最佳拦截时间窗口: t=5.0-7.0s\n3. 可见光+红外双确认后发起捕捉..."
}
```

**前端调用:**
```javascript
const response = await fetch('/api/ai_suggestion', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ methods_data: detectionMethods })
});
```

---

### 7. 保存报告

将 AI 生成的捕捉策略保存为报告文件（`reports/` 目录）。

```
POST /api/save_report
Content-Type: application/json
```

**请求体:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `content` | String | ✅ | 报告文本内容 |
| `platforms` | String | ❌ | 分析平台名称 |

**响应示例:**
```json
{
    "success": true,
    "filepath": "reports/capture_report_20260629_143052.txt",
    "filename": "capture_report_20260629_143052.txt"
}
```

---

### 8. 列出备份

列出 `data/backup/` 下所有快照备份，按时间倒序。

```
GET /api/list_backups
```

**响应示例:**
```json
{
    "success": true,
    "backups": [
        {
            "filename": "20260629_143052_auto",
            "timestamp": "2026-06-29 14:30:52",
            "label": "auto",
            "method": "可见光 + 红外 + 雷达",
            "point_count": 180
        }
    ]
}
```

---

### 9. 恢复指定备份

先清空当前数据，再从快照完整恢复。自选平台也会正确恢复。

```
POST /api/restore_backup
Content-Type: application/json
```

**请求体:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `backup_file` | String | ✅ | 快照目录名（如 `20260629_143052_auto`） |

**响应示例:**
```json
{
    "success": true,
    "message": "已从 20260629_143052_auto 恢复 6 个平台"
}
```

---

### 10. 一键恢复最新

恢复最新的快照备份。

```
POST /api/restore_all_backups
```

**响应示例:**
```json
{
    "success": true,
    "message": "已从 20260629_143052_auto 恢复 6 个平台",
    "restored": ["20260629_143052_auto"]
}
```

---

### 11. 创建备份

手动创建当前数据的快照备份。

```
POST /api/backup/create
```

**响应示例:**
```json
{
    "success": true,
    "message": "备份已创建: 20260629_150000_manual",
    "name": "20260629_150000_manual"
}
```

---

### 12. 删除备份

删除指定快照备份。

```
POST /api/backup/delete
Content-Type: application/json
```

**请求体:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `backup_name` | String | ✅ | 快照目录名 |

**响应示例:**
```json
{
    "success": true,
    "message": "已删除备份 20260629_150000_manual"
}
```

---

### 13. 获取分析数据

获取各检测手段的详细分析数据（速度、加速度、曲率、高度）。

```
GET /analysis/data
```

**响应示例:**
```json
{
    "visible": {
        "name": "可见光",
        "color": "#ff6b6b",
        "speeds": [2.5, 3.1, 2.8, ...],
        "accelerations": [0.6, -0.3, 0.2, ...],
        "curvatures": [0.05, 0.08, 0.03, ...],
        "heights": [10.0, 10.5, 11.2, ...],
        "time_steps": [0.0, 0.5, 1.0, ...]
    },
    "infrared": { ... },
    "radar": { ... }
}
```

**字段说明:**

| 字段 | 说明 | 计算方式 |
|------|------|---------|
| `speeds` | 速度序列 (m/s) | 水平位移 ÷ 时间差 |
| `accelerations` | 加速度序列 (m/s²) | 相邻速度差 |
| `curvatures` | 曲率序列 (1/m) | 三点法平面曲率 |
| `heights` | 高度序列 (m) | Y 坐标直接提取 |
| `time_steps` | 时间戳序列 (s) | 原始时间数据 |

---

## 数据格式说明

### 三维坐标点

```json
[x, y, z]   // 例: [3.5, 12.0, 7.2]
```

| 轴 | 含义 |
|----|------|
| X | 水平位置 (东西方向) |
| Y | 高度 (垂直方向) |
| Z | 水平位置 (南北方向) |

### 时间戳

浮点数，单位为**秒**，从轨迹起始时刻开始递增。

---

## 错误处理

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

### 错误响应格式

所有接口在出错时返回:
```json
{
    "success": false,
    "error": "详细错误描述"
}
```

### 前端错误处理建议

```javascript
async function apiCall() {
    try {
        const response = await fetch('/api/predict', { ... });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || '未知错误');
        }
        // 处理成功结果
        return result;
    } catch (error) {
        console.error('API 调用失败:', error.message);
        // 显示用户友好提示
    }
}
```

---

## 预测算法说明

预测使用**线性外推法**：取轨迹最后两点计算速度向量，沿该方向以固定时间步长生成预测点。

```
v = (P_last - P_prev) / Δt
P_pred[i] = P_last + v × (i × time_step)
```

**局限性:**
- 仅根据瞬时速度线性外推，不反映加速度变化
- 不适合大幅转弯后的轨迹预测
- 预测精度随时间步数增加而下降

**建议:** 预测点数（`num_points`）不宜过多，默认 6 个点在大多数场景下较为合适。
