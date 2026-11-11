> 修改记录：2026-11-05 19:40 CST - 按计算机视觉工程分层重构，合并当前指南并建立历史来源映射。

# 数据组织与输入契约

| 数据集 | 用途 | 当前加载代码 |
| --- | --- | --- |
| ZInD-FloorSet-v1 | 整层全部观测、独立 GT、lazy 全无序对 | `portallayout/datasets/zind_floor_dataset.py` |
| ZInD-BiPair-v1 | 同一 complete room 内不同 partial-room 的早期 pair 任务 | `portallayout/datasets/zind_bipair_dataset.py` |
| ZInD-ObservablePairs-v1 | 可观察候选与历史端到端训练 | `portallayout/datasets/zind_observable_pairs.py` |
| MP3D / ZInD 原单图数据 | Bi-Layout 单图训练与测试 | `portallayout/datasets/` 中对应 Dataset |

`data/` 存数据，`portallayout/datasets/` 存代码。`src/dataset` 保留为兼容链接，不是第二份数据。

## FloorSet 主契约

```text
inputs/<sample_id>/manifest.json                 # 公开 RGB 输入
labels/<floor_uid>/floor_gt.json                 # 楼层 GT
labels/<floor_uid>/observations_gt.jsonl         # 观测 GT
labels/<floor_uid>/portal_links.jsonl            # 几何关系
labels/<floor_uid>/pairs_policy.json             # lazy/materialized 策略
samples/*.jsonl                                 # 采样视图
splits/*.json                                   # 房屋划分
schemas/、audit/、dataset_meta.json              # 契约与来源
```

公开输入不包含真实房间分组或 GT 位姿。RGB 有效但 GT 缺失的观测仍保留；尺度、位姿、几何有效性独立记录。unknown 不变成负例。默认 `lazy_all_unordered` 从单图标签和稀疏关系恢复全部 pair；空 `pairs_gt.jsonl` 不代表数据损坏。

`all_panoramas` 是主任务视图；`one_per_space` 使用 GT 分组辅助采样，只能作为简化基线。原 RGB 按相对路径引用，`raw-root` 指向原始 ZInD 目录。

```python
from portallayout.datasets.zind_floor_dataset import ZInDFloorDataset

dataset = ZInDFloorDataset(
    "data/ZInD-FloorSet-v1/train_20261021",
    training_subset="dev", load_images=False,
)
inputs = dataset[0]
# 只有训练/评价调用方才显式读取独立 supervision。
```

## 构建与验证

先查参数，输出使用新目录：

```bash
python -m portallayout data build --help
python -m portallayout data validate --help
```

```text
python -m portallayout data build --root <ZInD原图目录> --partition <zind_partition.json> --splits train --output <新数据目录>
python -m portallayout data validate <数据目录> --raw-root <ZInD原图目录> --check-images --all-pairs --report <新报告.json>
```

构建恢复会检查代码、配置、来源与已写文件。目录重构改变源码哈希，因此不能无条件续写重构前未完成的构建；使用原版本恢复，或在新目录重新构建。完成的数据集本次仅移动目录，没有改写索引和标签。

各数据集完整字段、历史规模与构建审计合并在 [数据档案](archive/datasets.md)，不将旧规模直接视为当前数据盘点结果。
