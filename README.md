> 修改记录：2026-11-06 13:38 CST - 补充 GitHub 快照包含范围与本地实验资产说明。

> 修改记录：2026-11-05 19:40 CST - 按计算机视觉工程分层重构，合并当前指南并建立历史来源映射。

> 修改记录：2026-11-05 19:21 CST - 项目更名 PortalLayout，整理代码导航与统一命令入口；原首页及修改记录完整保存在历史说明。

# PortalLayout

**开口引导的跨房间布局重建 · Opening-Guided Cross-Room Layout Reconstruction**

从整层全景 RGB 预测单图布局与开口，学习跨图对应，估计相对位姿并组装楼层布局。基于原始 Bi-Layout 前端扩展；代码实现、开发实验和正式效果结论分开记录。

## 快速开始

```bash
conda activate bi_layout
python -m pip install --no-deps --no-build-isolation -e .
python -m portallayout --help
python -m portallayout floor run infer --help
```

在仓库根目录运行。本机解释器仍是 `/home/feixia/anaconda3/envs/bi_layout/bin/python`。可编辑安装注册 `portallayout` 命令；配置、权重与数据依赖源码工作区，尚未提供包含这些资源的独立部署包。

## 项目结构

```text
PortalLayout/
├── portallayout/         # 核心 Python 包
│   ├── models/          # Bi-Layout、开口表征、匹配器与辅助网络
│   ├── datasets/        # Dataset、构建器、标签与数据契约
│   ├── losses/          # 训练损失
│   ├── engine/          # 训练、标定、推理流程
│   ├── geometry/        # 位姿、图优化与布局组装
│   ├── evaluation/      # 匹配、位姿与楼层指标
│   ├── preprocessing/  # 前处理
│   ├── postprocessing/ # 后处理
│   ├── visualization/  # 绘图与查看器
│   ├── config/         # 配置加载逻辑
│   └── utils/          # 缓存、checkpoint、日志等通用工具
├── configs/             # 实验 JSON；layout/ 为单图 YAML
├── tools/               # CLI、诊断及实验入口
├── tests/               # 回归与兼容性测试
├── data/                # 实际数据，默认不提交
├── checkpoints/         # 模型权重，默认不提交
├── outputs/             # 运行输出，按现有规则保留报告
├── experiments/         # 冻结实验与证据
├── docs/                # 当前指南、六份历史合订本和审计清单
├── deliverables/        # PPT、图像及演示交付物
├── pyproject.toml       # 包元数据、可编辑安装与测试入口
└── requirements*.txt    # 运行与开发依赖
```

根目录原 `models/`、`dataset/`、`utils/` 等目录仅保留导入兼容桥；`src/config`、`src/dataset`、`src/output` 是兼容链接。`main.py`、`inference.py`、`convert_ckpt.py` 是原单图入口；新代码统一导入 `portallayout.*`。完整兼容说明见 [开发指南](docs/development.md)。

## 文档

从 [文档总入口](docs/README.md) 阅读当前架构、数据、训练、推理与评估。

GitHub 保存代码、配置、测试、交付材料和精选实验报告；完整数据集、权重、逐样本预测及缓存保留在本机。实验记录中的本机路径不能在其他机器直接使用，复现实验前请按 [快照说明](docs/project/github_snapshot.md) 准备对应资产。

本次重构的 [文件清理建议](docs/cleanup.md) 与 [全量 Markdown 审计](docs/project/audit/markdown_inventory.csv) 列出已合并资料、应保留证据以及可清理候选。

## 研究边界与来源

新版 matcher 使用 log-Sinkhorn 与 dustbin，legacy matcher 保留 softmax。strict 与 exploratory 分别记录，不以探索性拼接冒充严格校准结果。候选可解性、正确匹配、位姿准确率与楼层指标分别统计，保留 No Match、失败和断连分量。

原始 Bi-Layout 作者、论文引用、致谢与安装历史完整保存在 [上游说明](docs/archive/upstream.md)，许可见 [LICENSE](LICENSE)。本次结构重构没有重训或更新模型成绩。
