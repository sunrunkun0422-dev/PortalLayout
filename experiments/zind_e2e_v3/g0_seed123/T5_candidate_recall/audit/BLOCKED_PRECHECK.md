# T5 candidate failure audit — artifact precheck

结论：**本轮按协议停在 GPU 前向授权门前。** 现有冻结产物足以在 CPU 上重放
完整 val 的 legacy / Recall-TopK proposal，并可重建 proposal 的 trace-only 中间阶段；
但它们不足以完成“全部 train positive”漏斗以及 O1–O4 / Recall-TopK Matcher 重打分。
没有启动模型前向、训练、optimizer step 或后台任务。

## 状态（2026-08-19，Asia/Shanghai）

- branch: `agent/zind-bipair-opening-matcher`
- HEAD: `50a6ae5ca403928627fb4de18ae918d59c03d479`
- worktree: dirty；全部既有修改原样保留，没有 reset / checkout / clean。
- GPU 0: NVIDIA GeForce RTX 4060 Ti，749 / 16380 MiB，utilization 38%；
  `--query-compute-apps` 没有返回计算进程。
- disk: 382 GiB available（77% used）。
- active splits inspected: `train`, `val`
- `test_data_accessed=false`

## Checkpoint 不变证据

| Phase | SHA-256 | Size | mtime (+08:00) |
|---|---|---:|---|
| T0 | `2504cc25391e356cbc2fc7566fd0efea1040c916e2c85be6fbbc488c84ee5120` | 418836277 | 2026-08-11 11:23:11.863426196 |
| T1 | `171052202a4a40b4d2dafc685aa38256b16f904a79d4f4dac993169f098c51bf` | 418836277 | 2026-08-12 15:54:30.460200002 |
| T2 | `966af20b2b83b3d170b014e122fac206518de161d88668222b72dff646d1dd51` | 418836277 | 2026-08-13 22:22:57.604574390 |
| T3 | `ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d` | 418836277 | 2026-08-15 02:54:10.604347717 |

T3 path:
`checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best.pt`.

## 可直接复用

- `src/output/zind_geometry_val_full/t3_frozen_m2/frozen_predictions/`：3630/3630
  val pair，均保存 256-token `opening_probability_A/B`、legacy final interval、候选数、
  legacy score matrix、best pair、pair match/no-match probability、layout/M2 所需几何状态。
- `src/output/zind_geometry_val_full/t3_frozen_m2/per_pair.jsonl`：包含正式 GT target
  mapping 结果与 legacy Matcher/geometry 评价。
- `src/dataset/ZInD_matching/train.json` / `val.json`：GT opening interval、候选类型、
  target candidate 与 token affinity 可重建；window 不在正 target 中使用。
- full-val proposal parity 已有 CPU 证据：legacy 1080/1827；Recall-TopK
  1107/1827。正式审计仍需在 trace tool 中重新做逐元素硬门禁。
- raw peak、threshold、expansion、wrap、dedup、cap 虽未序列化，但可仅从冻结概率
  通过 trace-only 接口确定性重建，无需 GPU，也不需要改变正式 generator 返回值。
- medium cache 只覆盖 train 512 positive、val 256 positive，不能代表“全部 positive”。

Artifact SHA-256:

- full-val `summary.json`: `b342f316e064fc9d3577cd03150bc9bc9ae17137f36488ddbef68e3bc12c5caa`
- full-val `per_pair.jsonl`: `58a06662c8985d5de6975a7fdcbb39f809b62a448dc9bbe2711011b8c80ae3da`
- T4 recall `audit.json`: `cc10df6101ad661c3d1d11c6a70e4f403d0ce0c08c452102ea361b1040dca016`
- medium cache `schema.json`: `90edf541c263ad1a5b458871e28468711a19095e411779d70c91203f1b8d82a2`

## 缺失字段和受影响范围

### Full train

train 有 14741 个 positive pair（29482 sides；20603 个不同的
house/floor/room/pano）。除 medium cache 中 512 个 positive 外，没有同一 T3
checkpoint 的完整逐 pair：

- `opening_probability_A/B`
- T3 live `layout_feature_A/B`
- `enclosed_depth_A/B`, `extended_depth_A/B`
- legacy / Recall-TopK candidate masks 与各阶段 trace
- legacy Matcher score/dustbin decision 与 Top-1

因此缺 14229 个 positive pair，无法生成完整 `stage_funnel_train.json`、
`reason_counts_train.json`，也无法做 train/val 的同源比较。

### Full val Oracle / Recall-TopK Matcher

full-val compact artifact 没有序列化：

- T3 live `layout_feature_A/B`
- `enclosed_depth_A/B`, `extended_depth_A/B`
- candidate assignment logits、row dustbin、column dustbin logits
- GT-injected / raw-proposal / Recall-TopK candidate 集合的 Matcher 输出

所以 O0 legacy 可直接复用，O5 可离线计算；但 O1、O2、O3、O4 不能由现有
legacy score matrix外推。新增 27 个 Recall-TopK positive 的“Matcher 是否接受/选对、
为何未形成 S3”和“legacy 正确变错误”也必须重新用 Recall-TopK candidate masks
调用冻结 T3 Matcher。medium val cache 只覆盖 256/1827 positive，仍缺 1571。

## 预计需要的前向量（等待授权，不在本轮执行）

最小补洞方案，复用 medium cache：

- upstream T3 pair samples：train 14229 + val 1571 = **15800**；若 batch size=4
  且 split 分开，约 **3951 batched steps**。
- 为保证单一、同质 cache 而完整重建：train 14741 + val 1827 = **16568**
  positive pair samples；batch size=4 时约 **4143 batched steps**。
- val live feature 冻结后，O1–O4 四个 Matcher-only 视图需 4×1827 = **7308**
  positive pair evaluations；Recall-TopK 对照另需 **1827**，共 **9135** 个
  Matcher-only pair evaluations。这部分可在冻结 feature 后优先 CPU 执行。
- 若按 medium cache 的约 1.34 MiB/pair schema原样保存，完整 positive cache约
  22 GiB；正式实现应只保存审计必要张量以缩小体积。

另一种实现是先按 `(split, house, floor, complete_room, pano)` 去重上游：train
20603、val 2546，共 23149 个 single-view item，再展开 pair 并仅运行 Matcher。
采用哪种缓存布局需要用户确认；它不改变 checkpoint、候选配置或 GT 语义。

## 为什么现在停止

请求明确规定：现有 artifact 无法完成审计且必须重新运行 T3 GPU 前向时，要先列出
缺失字段和预计调用次数，并在本轮停止等待确认。该条件已经满足。现在继续生成要求的
`REPORT.md`、stage funnel、O0–O5 或可视化，会把不完整的 medium subset 伪装成
full train/val 结论，因此未创建这些正式产物。

下一轮若获授权，只做冻结 `eval()` / `torch.no_grad()` inference cache；不训练、
不创建 optimizer、不访问 test、不修改正式默认值、不继续 T4。
