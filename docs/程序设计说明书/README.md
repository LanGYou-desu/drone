# 程序设计说明书（LaTeX 源码）

本目录为“鹰眼长空 — 基于多平台协同的低空无人机智能监测系统”V1.0 的《程序设计说明书》LaTeX 工程。

## 目录结构

```
程序设计说明书/
├── main.tex                    # 主文件（封面、文档信息、目录、章节装配）
├── preamble.tex                # 导言区：字体、版式、图表、TikZ 样式、代码清单样式
├── chapters/                   # 各章节源文件（第 1–12 章与附录）
├── figures/ui/                 # 软件运行截图（Playwright 自动采集，深色主题）
├── figures/yolo/               # YOLO 训练与验证结果图
├── tools/                      # 截图、渲染、日志检查脚本
├── build.ps1                   # 一键编译
└── main.pdf                    # 编译产物
```

## 编译

环境要求：TeX Live 2025（含 ctex、tikz 等宏包），XeLaTeX 引擎。

```powershell
# 方式一：一键编译（推荐）
.\build.ps1

# 方式二：手动编译（需 3 遍以生成目录与交叉引用）
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

输出文件为 main.pdf。

## 排版约定

- 正文白底黑字；插图（TikZ）使用低饱和度配色区分层次。
- 图表编号采用“章-序”形式（如图 3-1、表 5-2），公式同。
- 代码清单使用 listings，路径统一用 \file{}（可断行），标识符用 \code{}。
- 章节文件均以 \section 起头，由 main.tex 通过 \input 装配。

## 截图与数据来源

- 界面截图：启动 python main.py --headless（Conda 环境 drone），用 tools/shots.py 通过 Playwright（Edge 内核）采集 10 个页面。
- YOLO 训练结果：来自 D:\作业&资料\大创\result\yolo，已复制到 figures/yolo/。

## 修改提示

- 新增图：使用 figure 环境 + \caption + \label，TikZ 统一使用 preamble.tex 中定义的 proc/deci/term/layer/arr 等样式。
- 新增长表：使用 longtable + p{} 列宽（本版 tabularx 与 xltabular 不兼容）。
- 若出现 Missing $ inserted，通常是 \code{} / \file{} 之外出现了未转义的下划线。
