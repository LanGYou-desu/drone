# 鹰眼长空

基于多平台协同的低空无人机智能监测系统。融合可见光、红外、雷达传感数据，提供 3D 轨迹可视化、加权合成、轨迹预测、捕捉时机分析和 AI 策略生成。

**技术栈：** Python Flask · pywebview · Three.js · ECharts · 硅基流动 AI

## 快速开始

```bash
pip install -r requirements.txt

python main.py               # 桌面窗口模式（同时启动两个模块）
python main.py --headless    # 纯 HTTP，重建模块 :5000，识别模块 :5001
python main.py recon         # 仅轨迹重建与分析 → :5000
```

编辑 `config.json` 填入 API Key 后即可使用 AI 策略功能。

## 功能

| 页面 | 功能 |
|------|------|
| **总览** | 3D 轨迹可视化，WASD 自由视角，鼠标悬停坐标+时间，平台显隐切换，动画播放 |
| **预测** | 线性外推轨迹预测，综合轨迹由各平台加权合成，历史+预测全程动画 |
| **分析** | 高度/速度/加速度/曲率多维曲线，原始+预测拼接连续分析，最佳捕捉时机评分 |
| **AI 策略** | 大模型分析多平台数据，自动生成无人机反制建议，报告保存 |
| **数据** | 上传自选数据，备份/恢复/更新，数据概览表格 |
| **设置** | 深色/浅色主题切换，摄像头移动速度配置 |

## 快捷键

| 按键 | 功能 |
|------|------|
| W/A/S/D | 摄像头前后左右平移 |
| Q/E | 摄像头升降 |
| Space | 播放/暂停轨迹动画 |
| 鼠标左键 | 旋转视角 |
| 鼠标右键 | 平移视角 |
| 滚轮 | 缩放 |

## 项目结构

```
drone/
├── main.py                              # 统一入口
├── config.json                          # 运行时配置（API Key、权重、主题等）
│
├── trajectory_reconstruction/           # 轨迹重建与分析
│   ├── core/                            #   领域逻辑（config/io/prediction/ai）
│   ├── services/                        #   业务编排（数据/预测/备份）
│   ├── views/                           #   Flask 路由与 API
│   └── frontend/                        #   前端（HTML/CSS/JS）
│
├── trajectory_recognition/              # 轨迹识别（框架，待实现）
│
├── data/                                # 运行时数据（fact/predict/backup）
├── reports/                             # AI 策略报告输出
└── templates/                           # 配置模板
```

## 配置

`config.json` 主要字段：

```json
{
  "ai": { "api_key": "...", "url": "...", "model": "..." },
  "detection_methods": {
    "visible": { "color": "#ff6b6b", "weight": 1.0 },
    "infrared": { "color": "#4ecdc4", "weight": 1.0 },
    "radar": { "color": "#ffe66d", "weight": 1.0 }
  },
  "capture_weights": { "height": 0.3, "speed": 0.3, "acceleration": 0.2, "curvature": 0.2 },
  "camera_speed": 0.12
}
```

- `detection_methods[].weight` — 综合轨迹合成权重
- `capture_weights` — 最佳捕捉时机评分权重
- `camera_speed` — WASD 移动速度
