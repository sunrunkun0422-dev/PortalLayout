# ZInD v2 训练前预检报告

- 兼容性关卡：**PASS**
- G1 候选关卡：**未达到目标**（按预定规则选出 G1，但联合召回仅提升 1.2042 个百分点，未达到 +5 个百分点挑战目标）
- 小样本可学习性关卡：**PASS**

本轮只运行了真实 validation 诊断和固定 train 子集的小样本 overfit；未读取或评价 test split，未运行完整 epoch 或正式训练，也未提交、推送或改动数据划分。

## 1. 基线、checkpoint 和数据契约

- 分支：`agent/zind-bipair-opening-matcher`
- 基准/当前 HEAD：`74509267971a4cc7097923774d3d26b652f5684d`
- 历史 checkpoint：`checkpoints/End_To_End/zind_matching_v1/best.pt`
- SHA256：`9406f52e209ffb7645b1187dcf21097e8ea0d924685c1521d74657bd6c7de2f8`
- 选择证据：完整端到端任务的 best epoch 为 zero-based epoch 8；`last.pt` 已运行到 epoch 14，但连续 6 个 stale epoch，内嵌 best 仍是 epoch 8，因此没有按文件时间选 `last.pt`。
- validation manifest：`src/dataset/ZInD_matching/val.json`，3630 pairs、157 houses。
- validation feature cache：`checkpoints/Cross_Scene_Matcher/zind_matching_feature_cache/val`
- G0：threshold `0.47`、最小组件宽度 `2` tokens、`max_candidates=12`、candidate target IoU `0.30`。当前生成器没有缺口连接参数。
- 候选调用链：`OpeningSignalHead.opening_probability` → `predicted_candidate_masks` → `opening_probabilities_to_intervals` → `candidate_targets_from_affinity`（真实 IoU 判定 `>=0.30`）→ matcher。
- 历史训练配置：epoch 0 使用 GT guidance + GT candidates；epoch 1 使用 mixed guidance + predicted candidates；后续使用 predicted + predicted。因此本轮 predicted + predicted T0 是“新一轮公平基线”，不是把整个历史 schedule 宣称为完全相同。
- 历史正式配置：batch 4，AdamW，最大 24 epochs，Bi-Layout/Opening/Matcher LR 分别为 `2e-6/1e-5/3e-5`，seed 123，无 AMP，无 activation checkpointing。

完整解析结果见 `resolved_checkpoint_and_config.json`。

## 2. 关卡 A：真实数据兼容性

在同一历史 checkpoint、同一 batch 顺序、seed 123、`model.eval()`、predicted guidance、predicted candidates 下，以旧内联公式独立重建 reference。固定真实样本为：

1. `0340_floor_01_pano_25_pano_51_8`
2. `0340_floor_01_pano_33_pano_25_7`
3. `0340_floor_01_pano_37_pano_25_3`
4. `0340_floor_01_pano_37_pano_41_4`

容差为 `atol=1e-6, rtol=1e-5`。以下所有项的最大绝对误差均为 **0.0**：candidate assignment logits、AB/BA assignment、mutual affinity、opening confidence、candidate pair score、best candidate pair、Top-1 排序，以及 candidate/其他原有/matching/full total loss。

真实 batch 的 mask 行为也通过：

| 情形 | pair ID | legacy assignment | masked assignment | dustbin grad | 结论 |
|---|---|---:|---:|---:|---|
| 正样本、无 target | `0340_floor_01_pano_25_pano_51_8` | 0.099222 | 0.000000 | 0 | assignment 精确屏蔽；其他 loss 最大误差 0，三个模块梯度均非零 |
| 正样本、有 target | `0340_floor_01_pano_33_pano_25_7` | 2.582340 | 2.582340 | 0.516408 | 与旧损失完全一致，未匹配行仍学习 dustbin |
| 真负样本、无 target | `0340_floor_01_pano_26_pano_25_negative` | 0.099795 | 0.099795 | -0.090885 | 负样本 dustbin loss 和梯度保留 |

全屏蔽 graph-connected zero 的 backward 也由单测覆盖且通过。CUDA bilinear backward 在 PyTorch 1.12 只提供 deterministic warn-only 提示，但本次需要核验的前向张量和 loss reference 都达到零误差。

## 3. 关卡 B：G0、有限扫描与 G1

### G0 完整 validation

| 指标 | G0 |
|---|---:|
| Opening token Precision / Recall / F1 | 85.3283% / 92.3280% / 88.6903% |
| A / B / 联合候选召回 | 67.4330% / 89.8194% / 58.8944% |
| positive missing target | 41.1056% |
| 每视图候选数 mean / P50 / P90 / max | 2.4534 / 2 / 5 / 12 |
| 有 target 条件下纯排序 Top-1 / Top-3 / MRR | 67.3792% / 90.1487% / 0.7947 |
| 有 target 条件下“排序第一且被接受” | 43.1227% |
| Recall@3 | 90.1487% |
| 端到端 Top-1 | 25.3968% |
| 正样本接受 / 负样本拒绝 | 34.3733% / 93.7327% |
| Harmonic / Balanced flow | 0.399651 / 0.595647 |

历史 best 中的离散 validation 指标得到精确复现；总损失误差约 `1.0e-6`，Opening token 指标误差约 `2.2e-6`。此前约 41.81% missing / 58.19% joint recall 不能作为本 validation 的复现值；本次从 1827 个真实 validation positives 独立算得 41.1056% / 58.8944%，未读取 test split。

### G0 → G1

扫描只覆盖既有参数：threshold `{0.30,0.35,0.40,0.47,0.55}` × min width `{1,2}`；checkpoint、IoU `0.30`、`max_candidates=12`、样本顺序和接受阈值固定，Opening 概率缓存后复用。

| 指标 | G0: 0.47 / width 2 | G1: 0.55 / width 2 | 变化 |
|---|---:|---:|---:|
| A 候选召回 | 67.4330% | 68.4729% | +1.0399 pp |
| B 候选召回 | 89.8194% | 89.9288% | +0.1095 pp |
| 联合候选召回 | 58.8944% | 60.0985% | **+1.2042 pp** |
| missing target | 41.1056% | 39.9015% | -1.2042 pp |
| 候选 mean / P90 / max | 2.4534 / 5 / 12 | 2.3956 / 5 / 12 | -0.0579 / 0 / 0 |
| 纯排序 Top-1 / Top-3 / MRR | 67.3792% / 90.1487% / 0.7947 | 66.5756% / 90.2550% / 0.7916 | -0.8036 pp / +0.1063 pp / -0.0031 |
| 接受后条件 Top-1 | 43.1227% | 42.1676% | -0.9551 pp |
| 端到端 Top-1 | 25.3968% | 25.3421% | -0.0547 pp |
| 正样本接受 / 负样本拒绝 | 34.3733% / 93.7327% | 35.0848% / 93.5663% | +0.7115 pp / -0.1664 pp |
| Harmonic / Balanced flow | 0.399651 / 0.595647 | 0.398822 / 0.594542 | -0.000829 / -0.001106 |

G1 满足平均候选数不超过 G0 2 倍、负样本拒绝率不低于 90%。threshold 0.55 / width 1 的联合召回为 60.3175%，只比 width 2 高 0.2189 个百分点，落在 0.5 pp tie window 内；width 2 的条件 Top-1 更高（42.1676% vs 41.9238%）且候选更少（2.3956 vs 2.5291），所以按预定规则选择 width 2。由于联合召回只增加 1.2042 pp，G1 的 +5 pp 挑战目标未达到；同时 flow 和端到端 Top-1 没有提升，不能把 G1 描述为 validation 改善。

### 失败分类迁移

分母均为 1827 个 positives，分类互斥：

| 类别 | G0 count / ratio | G1 count / ratio | 迁移 | 代表 ID |
|---|---:|---:|---:|---|
| 两侧均缺候选 | 30 / 1.6420% | 31 / 1.6968% | +1 | `0209_floor_01_pano_4_pano_9_0` |
| A 侧缺候选 | 565 / 30.9250% | 545 / 29.8303% | -20 | `0613_floor_01_pano_21_pano_31_5` |
| B 侧缺候选 | 156 / 8.5386% | 153 / 8.3744% | -3 | `0340_floor_01_pano_25_pano_51_8` |
| 有 target、rank 1 但被 dustbin 拒绝 | 261 / 14.2857% | 268 / 14.6689% | +7 | `0340_floor_01_pano_37_pano_25_3` |
| 有 target、正确候选 rank > 1 | 351 / 19.2118% | 367 / 20.0876% | +16 | `0340_floor_01_pano_33_pano_25_7` |
| 成功 | 464 / 25.3968% | 463 / 25.3421% | -1 | `0340_floor_01_pano_37_pano_41_4` |
| 被 `max_candidates` 截断 | 0 / 0% | 0 / 0% | 0 | 无 |
| 有近似候选但 IoU < 0.30 | 0 / 0% | 0 / 0% | 0 | 无 |
| 其他可核实失败 | 0 / 0% | 0 / 0% | 0 | 无 |

G1 净减少 22 个 missing-target positives，但新增 target 主要迁移到了 rank>1 或 rank1-rejected，最终成功数反而少 1；这解释了召回上升但端到端指标未升。

## 4. 关卡 C：四种 source 的误差隔离

以下均使用同一 checkpoint、完整 validation 和 G1。条件 Top-1 同时列出“接受后 / 纯排序”；只有最后一行是真实端到端。

| Guidance / Candidates | 候选召回 | 条件 Top-1（接受/纯排序） | Top-3 | MRR | 正接受 | 负拒绝 | Harmonic / Balanced | 定位 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GT / GT | 100.0000% | 26.9841% / 46.3054% | 69.9507% | 0.6133 | 37.9858% | 94.6201% | 0.419926 / 0.608021 | Matcher 理想输入诊断 |
| Predicted / GT | 100.0000% | 26.8747% / 45.2107% | 68.9655% | 0.6026 | 38.6426% | 94.2873% | 0.418273 / 0.605810 | 隔离 guidance 误差 |
| GT / Predicted | 60.0985% | 42.7140% / 69.4900% | 93.7158% | 0.8188 | 35.7417% | 94.6755% | 0.403897 / 0.601730 | 隔离候选生成误差 |
| Predicted / Predicted | 60.0985% | 42.1676% / 66.5756% | 90.2550% | 0.7916 | 35.0848% | 93.5663% | 0.398822 / 0.594542 | **真实端到端** |

切换 guidance 时 candidate tensor 来源保持不变；切换 candidate source 时 guidance tensor 来源保持不变，并有单测验证 predicted/predicted 不读取 manifest candidate 或 GT guidance。主要上游瓶颈是 predicted candidates 将召回从 100% 降到 60.10%；predicted guidance 在同为 predicted candidates 时又使纯排序 Top-1 下降约 2.91 pp。

## 5. 关卡 D：固定 train 子集 T0–T3

子集为固定 24 pairs：8 个 positive-with-target、8 个 positive-without-target、8 个 negatives，并覆盖候选总数 2 到 18。全部样本 ID、索引、类别和固定顺序保存在 `overfit_subset.json`。四组均从同一历史 checkpoint 独立重载，使用同一 G1、seed、batch 顺序、optimizer、LR 和可训练模块；未使用 validation/test 训练。

初始十步 smoke 在 T2 第 1 步发现明确阻断错误：padding candidate 上的零乘积直接进入 `sqrt`，listwise ranking 反向时形成 `0 × inf`，导致 Bi-Layout 上游梯度非有限。实验当即停止，没有提高 loss 权重或改模型结构。最小补丁只在 `pair_valid & product>0` 的元素上求平方根，并让 listwise loss 只对有效 pair 求 log；补充真实形态 padding 的回归测试后，38 个相关测试全部通过，兼容性再次得到零误差 PASS。

正式结果为每组 100 optimizer steps；表中是第 1–10 步均值 → 第 91–100 步均值：

| Exp | policy / rank weight | total loss | assignment | ranking | score margin | rank / Top-1 / MRR | 最后窗口梯度范数（backbone/opening/matcher） | 结果 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| T0 | legacy / 0.0 | 8.4345 → 7.5942 | 0.07463 → 0.01986 | 0 → 0 | 0.7686 → 0.9256 | 1 / 1 / 1 → 1 / 1 / 1 | 5.796 / 0.588 / 0.827 | finite |
| T1 | mask / 0.0 | 8.4393 → 7.5970 | 0.07856 → 0.02192 | 0 → 0 | 0.7751 → 0.9290 | 1 / 1 / 1 → 1 / 1 / 1 | 6.320 / 0.591 / 0.826 | finite，masked contribution=0 |
| T2 | legacy / 0.1 | 8.4431 → 7.5931 | 0.07422 → 0.01754 | **0.08993 → 0.01390** | **0.7696 → 0.9349** | 1 / 1 / 1 → 1 / 1 / 1 | 5.953 / 0.590 / 0.828 | PASS |
| T3 | mask / 0.1 | 8.4477 → 7.5987 | 0.07818 → 0.02125 | **0.08790 → 0.01912** | **0.7759 → 0.9306** | 1 / 1 / 1 → 1 / 1 / 1 | 6.397 / 0.593 / 0.829 | PASS |

- 四组 loss、梯度、backward 和 optimizer step 均 finite；峰值显存约 11.80 GiB，平均单步约 1.01 秒。
- T1/T3 的 masked-positive assignment 最大绝对贡献始终为 `0.0`。
- T0/T2 的真负样本 dustbin 梯度在 100/100 steps 均非零。
- T2 ranking loss 下降 84.54%，T3 下降 78.25%。
- 子集初始 rank/Top-1/MRR 已饱和为 1/1/1，因此没有离散提升空间；连续 score margin 分别提升 0.1653 和 0.1548，满足预定“至少一个排序方向改善”的条件。
- 所有实验中 Bi-Layout 解冻 backbone、Opening Head、Matcher 均观察到有限非零梯度。

该结果只证明实现具备小样本可学习性，不能证明 T3 优于 T0，也不是 validation 提升。

## 6. 是否可以启动正式 T0–T3

**实现层面具备启动条件，但本轮按范围限制没有启动。** Gate A 和 Gate D 均 PASS，数值阻断错误已有最小修复和回归测试，当前没有训练实现 blocker。Gate B 的 +5 pp 是挑战目标而非硬性可行条件；它没有达到，并且 G1 的 flow/端到端 Top-1 略低于 G0。因此可以把正式 T0–T3 作为受控消融来检验 loss 方案，但在完整 validation 结果出现前不能宣称 G1 或 T3 更优。若项目把“G1 必须 +5 pp”提升为硬门槛，则准确阻塞点是现有有限候选参数只能带来 +1.2042 pp，且按本轮限制不能引入更复杂候选算法。

## 7. 产物与复现命令

目录：`experiments/zind_e2e_v2/preflight/`

- `environment.txt`
- `resolved_checkpoint_and_config.json`
- `compatibility_comparison.json`
- `g0_validation_metrics.json`
- `candidate_threshold_scan.csv`
- `candidate_failures_g0_g1.csv`
- `candidate_failure_summary_g0_g1.csv`
- `teacher_forcing_2x2.csv`
- `overfit_subset.json`
- `overfit_t0_t3.csv`
- `overfit_summary.json`
- `overfit_gate.json`
- `analysis_summary.json`
- `validation_opening_probabilities.pt`
- `commands.txt`
- `ZInD_v2_preflight.md`

主要实际命令（完整历史也保存在 `commands.txt`）：

```bash
git status --short
git diff --stat
git diff --check
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

env MPLCONFIGDIR=/tmp/bi_layout_mpl XDG_CACHE_HOME=/tmp/bi_layout_cache \
  /home/feixia/anaconda3/envs/bi_layout/bin/python -m unittest \
  tests.test_zind_end_to_end_candidate_fixes tests.test_cross_scene_matcher \
  tests.test_zind_matcher_training tests.test_zind_matching_feature_dataset -v

env MPLCONFIGDIR=/tmp/bi_layout_mpl XDG_CACHE_HOME=/tmp/bi_layout_cache CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/feixia/anaconda3/envs/bi_layout/bin/python tools/analyze_zind_candidate_recall.py \
  --output_dir experiments/zind_e2e_v2/preflight --device cuda:0 --batch_size 4 --workers 4 \
  --thresholds 0.30,0.35,0.40,0.47,0.55 --min_widths 1,2

env MPLCONFIGDIR=/tmp/bi_layout_mpl XDG_CACHE_HOME=/tmp/bi_layout_cache CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/feixia/anaconda3/envs/bi_layout/bin/python tools/run_zind_small_overfit.py \
  --output_dir experiments/zind_e2e_v2/preflight --device cuda:0 --batch_size 4 \
  --subset_size 24 --steps 100 --seed 123 --workers 0 --selection_scan_limit 256
```

## 8. 工作区变更与校验

本轮结束时：

```text
branch: agent/zind-bipair-opening-matcher
HEAD:   74509267971a4cc7097923774d3d26b652f5684d
git diff --check: clean
38 focused tests: PASS
```

工作区中的相关文件：

- `models/cross_scene_matcher.py`：保留已有 candidate 修复；另对本轮发现的 padding `sqrt` 非有限梯度做最小补丁。
- `tools/train_zind_end_to_end.py`：保留进入本轮前已有的 source、mask 和 ranking 接线修改。
- `tests/test_zind_end_to_end_candidate_fixes.py`：candidate mask/rank/source 测试，并新增 padding ranking backward finite 回归。
- `tools/analyze_zind_candidate_recall.py`：新增真实兼容性、validation 扫描、失败分类和 2×2 诊断。
- `tools/run_zind_small_overfit.py`：新增固定 train 子集 T0–T3 工具。
- `experiments/zind_e2e_v2/preflight/`：本报告列出的全部实验产物。

`git diff --stat`（不包含 untracked 文件）为：

```text
 models/cross_scene_matcher.py  | 161 +++++++++++++++++++++++--
 tools/train_zind_end_to_end.py | 265 ++++++++++++++++++++++++++++++++++++++---
 2 files changed, 399 insertions(+), 27 deletions(-)
```

现有 `src/output/...` untracked 目录与本轮无关，均未改动。未执行 commit 或 push。
