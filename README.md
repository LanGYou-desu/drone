# 鹰眼长空

基于多平台协同的低空无人机智能监测系统。融合可见光、红外、雷达三种传感平台的轨迹数据，提供 3D 可视化、加权合成、轨迹预测、捕捉时机分析和 AI 策略生成。

**技术栈：** Python Flask · pywebview · Three.js · ECharts · 硅基流动 AI

## 快速开始

```bash
pip install -r requirements.txt

python main.py               # 默认同时启动两个模块（桌面窗口）
python main.py --headless    # 纯 HTTP 模式
python main.py recon         # 仅重建分析 → :5000
python main.py recog         # 仅轨迹识别 → :5001
```

编辑 `config.json` 填入硅基流动 API Key 后即可使用 AI 策略功能。

## 项目结构

```
drone/
├── main.py                              # 统一入口
├── config.json                          # 运行时配置（API Key、权重、主题等）
│
├── trajectory_reconstruction/           # 轨迹重建与分析
│   ├── core/                            #   领域逻辑（config/io/prediction/ai）
│   ├── services/                        #   业务编排（数据/预测/备份）
│   ├── views/                           #   HTTP 接口
│   └── frontend/                        #   前端资源（HTML/CSS/JS）
│
├── trajectory_recognition/              # 轨迹识别（框架）
│
├── data/                                # 共享运行时数据（fact/predict/backup）
├── reports/                             # AI 策略报告
├── templates/                           # 配置模板
└── docs/                                # 文档
```

## 文档

| 文档 | 说明 |
|------|------|
| [使用手册](docs/manual.md) | 用户操作指南 |
| [开发文档](docs/DEVELOPER.md) | 架构、API、开发指南（新人必读） |
