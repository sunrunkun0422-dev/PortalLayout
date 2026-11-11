> 修改记录：2026-11-05 19:40 CST - 按计算机视觉工程分层重构，合并当前指南并建立历史来源映射。

# PortalLayout 文档入口

当前维护文档只在下表更新。历史阶段记录保留原日期，不再作为“当前实现”的重复入口。

| 文档 | 解决的问题 |
| --- | --- |
| [架构](architecture.md) | 模块、数据流与已实现边界 |
| [数据](datasets.md) | FloorSet / BiPair / ObservablePairs、GT 隔离与构建 |
| [训练](training.md) | 前端缓存、监督、训练与标定 |
| [推理](inference.md) | RGB / 缓存入口、输出和 strict/exploratory |
| [评估与实验](evaluation.md) | 指标口径与已有证据入口 |
| [开发与目录迁移](development.md) | 安装、测试、兼容与源码映射 |
| [清理清单](cleanup.md) | 可删除、条件清理与必须保留 |

## 历史资料已合并

33 份散落的 Markdown 合并为 6 份来源合订本；历史正文、修改记录与日期边界保留，相对链接已调整。当前指南提炼主线，合订本承担原文追溯。

- [上游说明](archive/upstream.md)
- [早期学习与项目手册](archive/legacy_handbook.md)
- [数据设计与构建](archive/datasets.md)
- [研究设计与协议草案](archive/research_design.md)
- [阶段实现与验收](archive/implementation.md)
- [项目整理与早期结果](archive/project_history.md)

逐文件来源、原 SHA-256 和归档锚点见 [合并映射](project/document_migration.json)。重构前全部 391 份 Markdown（包括被 Git 忽略的数据/权重说明）均进入 [审计 CSV](project/audit/markdown_inventory.csv)。每个文件都有去向与处置分类。

## 不合并的材料

`experiments/`、`outputs/` 中的逐楼层报告依附不同运行，必须与 JSON/CSV、参数和失败记录一起保留。`data/` 与 `checkpoints/` 中的说明属于对应数据/权重；`deliverables/` 中的讲稿与交付说明属于演示包。即使正文相同，也不能仅按文本相等删除其中一个上下文。

证据包和开题材料保留在 [方法证据](method_advantage_evidence_20261028/README.md)、[方法比较](proposal_method_comparison/README.md)。它们的历史结论不会因为本次目录整理自动升级为正式实验结果。
