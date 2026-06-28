# 鹰眼长空

基于多平台协同的低空无人机智能监测系统。融合可见光、红外、雷达三种传感平台的轨迹数据，提供 3D 可视化、轨迹预测、AI 策略生成和多维运动学分析。

**技术栈：** Python Flask · pywebview · Three.js · ECharts · 硅基流动 AI

## 快速开始

```bash
pip install -r requirements.txt

python main.py               # 桌面窗口
python main.py recon --headless   # HTTP 服务 → :5000
python main.py recog         # 轨迹识别 → :5001
```

编辑 `config.json` 填入硅基流动 API Key 后即可使用 AI 策略功能。

## 项目结构

```
drone/
├── main.py                              # 统一入口
├── config.json                          # 运行时配置
│
├── trajectory_reconstruction/           # 轨迹重建与分析
│   ├── core/                            #   领域逻辑（config/io/prediction/ai）
│   ├── services/                        #   业务编排
│   ├── views/                           #   HTTP 接口
│   └── frontend/                        #   前端资源
│
├── trajectory_recognition/              # 轨迹识别（框架）
│
├── data/                                # 共享运行时数据
├── reports/                             # AI 策略报告
├── templates/                           # 配置模板
└── docs/                                # 文档
```

## 文档

| 文档 | 说明 |
|------|------|
| [使用手册](docs/鹰眼长空使用手册.pdf) | 用户操作指南 |
| [开发文档](docs/DEVELOPER.md) | 架构、API、开发指南（新人必读） |
