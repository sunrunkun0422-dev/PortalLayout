> 修改记录：2026-11-05 20:07 CST - 记录 CRC 校验后删除冗余 MP3D 压缩包和批量训练可视化，以及缓存清理。

> 修改记录：2026-11-05 19:44 CST - 根据磁盘、导入引用、Markdown 哈希和归档映射列出文件处置建议；未删除缓存、数据、权重或实验。

# 文件清理与保留清单

2026-11-05 已按用户授权执行一轮清理，详见下方记录；其余建议保留审查状态。表格中的目录体积是清理前的盘点值。**文件大、名字旧或没有被 import，都不能单独证明没用。**

## 确定不参与研究结果，可清理

| 文件/目录 | 依据 | 处理建议 |
| --- | --- | --- |
| `portallayout/datasets/.DS_Store` | macOS Finder 元数据，6,148 字节 | 可删除 |
| `portallayout/models/.DS_Store` | macOS Finder 元数据，6,148 字节 | 可删除 |
| `src/.DS_Store` | macOS Finder 元数据，6,148 字节 | 可删除 |
| 各目录 `__pycache__/`、`*.pyc` | Python 可重新生成的字节码 | 停止相关运行后可清理；不会释放大量空间 |
| `.pytest_cache/` | 测试运行缓存 | 可删除，下次测试重建 |

完整缓存文件路径与体积见 [cleanup_candidates.json](project/audit/cleanup_candidates.json)。以上三处 Finder 元数据与当时存在的缓存已清理；Python 和 pytest 后续运行可能重新生成缓存。

## 有条件可删或归档，优先审查大文件

| 路径 | 当前证据 | 删除前条件 |
| --- | --- | --- |
| `mp3d.zip`，约 1.25 GiB | 压缩包 6,890 个文件均在 `data/mp3d/` 有同名同大小文件 | 再核对内容 CRC/哈希，并确认不需保留分发/离线恢复包；本次仅核对名称与大小 |
| `checkpoints/Bi_Layout_Net/mp3d/`，约 133G | 多批次模型及训练产物，是最大空间来源 | 确认每轮最终 best、需要的 last/恢复点和已发表结果依赖后，再按清单删中间 epoch；不能直接删整个目录 |
| 其余 `checkpoints/`，整体约 166G | 包含原前端、matcher、辅助模型与缓存 | 被当前配置、bundle、实验报告引用的权重必须保留；仅凭文件名不足以判断 |
| `experiments/`，约 15G；`outputs/`，约 5.4G | 存在缓存、逐层输出、证据与失败反例 | 成套归档并保留来源和必要权重后才可考虑删冗余可再生部分 |

`mp3d.zip` 已在全部文件的大小与 CRC32 校验通过后删除。MP3D 训练批量可视化已每种布局保留 10 张，其余 PNG 已删除；没有删除模型权重。下表保留原审查记录，便于追溯。

## 没有静态直接引用，建议复核的源码

| 文件 | 本次检查 | 建议 |
| --- | --- | --- |
| `portallayout/evaluation/subset_selection.py` | 硬编码 `/media/Pluto/frank/...`，导入即读写旧机器文件；当前工程无直接引用 | 优先从活跃评估包中退役/归档；若仍需功能，先改成显式参数 CLI |
| `portallayout/models/modules/conv_transformer.py` | `ConvTransformer` 未由模块注册入口导出，也未发现当前配置/源码调用 | 旧实验候选；确认没有外部 checkpoint 类依赖后可删除或归档 |
| `portallayout/models/other/scheduler.py` | `WarmupScheduler` 无当前直接引用 | 复核旧训练恢复需求后可归档；不是当前训练必须代码的已验证依据 |
| `portallayout/datasets/zind_dataset_dop.py` | 当前数据构建入口没有直接使用该加载器 | 旧数据实现候选；确认不再复现对应版本后归档 |
| `portallayout/evaluation/analyse_layout_type.py` | 可独立运行的原始分析脚本 | 不属于主线，但不等于无用；保留为历史分析工具 |
| `portallayout/evaluation/eval_visible_iou.py` | 可独立运行的原始评价脚本 | 保留独立评估用途，不能因没有 import 删除 |
| `portallayout/visualization/visualizer/visualizer.py` | 组件 README 明确说明的 GUI 入口 | 有用途，保留；静态无引用是正常现象 |

上述 7 项来自 AST 导入、相对导入及字符串引用检查。动态注册、外部 notebook、用户手工运行与 pickle 依赖未穷尽。没有把任何算法源码自动删除。

## Markdown 重复与合并

33 份旧独立指南已合并为 6 份历史合订本，旧独立文件已移除；内容与修改记录保留在合订本。当前维护以 [文档入口](README.md) 的主题指南为准，逐文件对应见 [合并映射](project/document_migration.json)。这是本次已实施的去重维护，不是仅列计划。

另外发现 3 组逐字相同的 Markdown：

- `experiments/zind_observable_v1/results/训练结果摘要.md` 与 `outputs/zind_observable_training_20260910/训练结果摘要.md`：内容重复，分别属于实验与导出展示包。若停止维护展示包，可在归档整个包后删除其中的重复副本。
- `outputs/zind_observable_pairs_v1/review_pilot/README.md`、`review_pilot_v2/README.md`、`review_pilot_final/README.md`：同样说明文字对应不同审核目录，不能据此判断审核数据重复。
- `data/ZInD-FloorSet-v1/preflight_train25_20261021/README.md` 与 `smoke_train_0665_20261021/README.md`：说明模板相同，数据集身份不同，不单独删除。

全量 [CSV 清单](project/audit/markdown_inventory.csv) 可按 `category`、`integrity` 过滤；当前去向含归档锚点。没有把每个 `floor_report.md` 合并成一个报告，否则会丢失实验/样本归属。

## 必须保留

- `LICENSE`、上游作者与论文引用。
- `data/`、已使用权重、冻结实验的 JSON/CSV、失败记录、人工“不确定”与待审核标签。
- 根目录兼容包与 `src/` 符号链接：它们仍承担旧导入、pickle 或路径兼容职责，不是迁移遗留垃圾。
- `.git/`：约 8.4G 是版本库对象，不直接手动清空。
- `deliverables/`、`atlanta_subset_report/`、`bilayout_test_382_report/`：独立可读报告/交付物，没有源码引用不代表无用。

重构前源码与文档备份位于 `/tmp/portallayout_before_refactor_20261105_193126.tar.gz`，仅为本机短期恢复点，不是永久归档。数据与权重没有复制到该包，也没有因重构被修改。

## 2026-11-05 已执行清理

- `mp3d.zip`：6,890 个解压文件逐个通过文件大小与 CRC32 核验后删除；`data/mp3d/` 完整保留。
- `checkpoints/Bi_Layout_Net/mp3d/results/train_best/`：删除 187,492 张批量可视化 PNG，enclosed 和 extended 各保留 10 张样例。保留所有模型权重、数据和其他实验产物。
- 删除三个 Finder 元数据文件及当时存在的 Python/pytest 缓存。
- 逐类数量、字节数、样例路径和校验结果见 [机器可读清理记录](project/validation/cleanup_20261105.json)。
