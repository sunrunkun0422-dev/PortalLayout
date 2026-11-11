# T4 recall-only 预检报告

## 1. 一句话结论

T4 的 proposal recall、CPU 可学习性和 2-epoch GPU 信号均通过，但固定 train 子集 overfit 的总 loss 未稳定下降，因此 `do_not_start_full_training=true`，未启动完整训练。

## 2. branch / HEAD / worktree

- branch: `agent/zind-bipair-opening-matcher`
- HEAD: `50a6ae5ca403928627fb4de18ae918d59c03d479`
- 开始时工作区已 dirty；所有既有修改均保留。
- 正式 T4 checkpoint 目录 `checkpoints/End_To_End/zind_e2e_v3/g0_seed123/T4_recall/` 未创建。
- 剩余磁盘约 385 GiB。

## 3. full-val recall decomposition

| 项目 | 数量 / 比例 |
|---|---:|
| positive | 1827 |
| candidate recalled | 1080 / 1827 = 59.11% |
| candidate not recalled | 747 |
| recalled 后 matcher no-match | 437 / 1080 = 40.46% |
| recalled 后选错 | 142 / 1080 = 13.15% |
| selected correct / S3 | 501 / 1827 = 27.42% |
| recalled 条件下 correct | 501 / 1080 = 46.39% |

P0 重放逐元素兼容：3630/3630 pair 的 interval 完全一致，1827/1827 positive 的 recall boolean 完全一致。

## 4. candidate proposal audit

预先固定了 9 个 P0–P3 配置，没有无界搜索。主要结果：

| 配置 | recalled | recall | K mean / p95 | `K_A*K_B` mean / p95 | negative product mean / p95 |
|---|---:|---:|---:|---:|---:|
| P0 t=.47 legacy max12 | 1080 | 59.11% | 2.43 / 6 | 5.72 / 16 | 3.57 / 11.8 |
| P1 t=.42 legacy max12 | 1058 | 57.91% | 2.45 / 6 | 5.84 / 18 | 3.66 / 12 |
| P2 t=.47 Top-2 max12 | 1104 | 60.43% | 2.96 / 8 | 8.47 / 28 | 5.14 / 16 |
| **P2 t=.47 Top-4 max12** | **1107** | **60.59%** | **3.08 / 8** | **9.23 / 30** | **5.54 / 18** |
| P3 t=.42 Top-4 max12 | 1080 | 59.11% | 3.05 / 8 | 9.07 / 30 | 5.50 / 18 |

选择 P2 Top-4：比 P0 多 27 个 positive，候选数有界且 `K_A*K_B` p95=30，低于预设的 2x P0 门限 32。wrap recall 从 408/616 提升到 422/616。

接口分层（样本极不平衡）：door-door 1103/1818，door-opening 1/1，opening-door 2/4，opening-opening 1/4。稀有类型不能据此作稳定结论。

## 5. dustbin score / margin audit

- T3 learned dustbin pre-softmax scalar：`1.3945990800857544`。
- 冻结 artifacts 有 true candidate pair score、pair match/no-match probability；没有逐行/逐列 assignment logits。因此没有伪造 row/column true-to-dustbin margin。
- recalled positive 的 true score：median 0.3862；accepted median 0.7356；rejected median 0.0618。
- rejected positive pair margin：median -0.7524，p95 -0.3029；negative pair margin：median -0.9804，p95 0.0150。
- 纯事后 pair-margin=-0.5 sweep：positive recall 53.20%，negative rejection 90.29%，precision 84.74%，S3 proxy 29.83%。相对原 S3 27.42% 有约 44-pair 的潜在增量，但这是 pair-level 近似校准，不等价于正式 bidirectional row/column 决策，因此没有直接改正式 threshold。

## 6. wrong-candidate audit

- 142 个 wrong selection 的 selected-minus-best-true gap：mean 0.4243，median 0.4305，p95 0.7986。
- wrong pair 的候选乘积 mean 9.20 / p95 20；correct pair 为 mean 4.94 / p95 14，错误明显集中在候选较多的 pair。
- 推断的 selected interface：140 door-door、2 opening-door。
- T3 ranking loss 只比较 real-real `candidate_pair_score`，不包含 dustbin。

## 7. 最终 T4 设计

只包含两类改动：

1. `candidate_proposal_mode=recall_topk`：threshold=.47、Top-K=4、peak min=.20、expansion boundary=.12、dedup IoU=.50、max candidates=12；train 和 trainer val 使用相同 predicted proposal。train 可做 target-only injection，val 强制 predicted-only。
2. positive-to-bidirectional-dustbin margin：margin=.20、weight=.20，只对 `is_match && has_valid_target` 生效；negative 仍由原 assignment loss 监督，missing positive 继续由 `mask_positive` 屏蔽，原 ranking loss 保留。

训练 proposal contract hash：`4e607c1a22a428621cd334f38536a6cd5d4b0b7189c66b01c357435f7e1a8730`。审计行（含 audit id/target IoU）的 hash：`ec3e678eaf0aa05160149314f8eaabbfd5862cd244099cc1a6fbf61424c3cbb5`。

## 8. 修改文件

- `models/cross_scene_matcher.py`：circular IoU、recall Top-K proposal、positive dustbin margin loss。
- `tools/train_zind_cross_scene_matcher.py`：`predicted_candidate_masks` 增加显式 proposal mode，默认 legacy。
- `tools/train_zind_end_to_end.py`：T4 参数、train-only injection、margin 接线、candidate count/GT-use 记录、finite preflight gate、config hash。
- `tools/audit_zind_t4_recall.py`：val-only frozen recall / score / wrong-candidate audit。
- `tools/preflight_zind_t4_cpu.py`：CPU learnability fixture。
- `tools/prepare_zind_t4_preflight_subsets.py`：只引用原 train/val 的确定性小子集。
- `tools/summarize_zind_t4_preflight.py`：14 项 launch gate 汇总。
- `tests/test_zind_t4_recall.py`：T4 proposal、injection、margin、padding、复现性测试。

## 9. CPU 测试

- 完整 CPU 回归：187/187 PASS。
- 新增定向测试：7/7 PASS。
- 覆盖 legacy 逐元素兼容、deterministic Top-K、wrap、dedup、cap、padding、train/val GT 隔离、margin mask、negative assignment、ranking 不变、finite gradient。
- 单元测试中的 test split 仅为 `/tmp` 合成 fixture；未读取项目 ZInD test manifest/cache。

## 10. learnability / overfit

CPU learnability PASS：

- `L_db`: 0.20 → 0.00；
- total: 1.1814 → 0.0383；
- true-to-dustbin margin: 0.00 → 3.5981；
- positive acceptance: 0 → 1；
- negative rejection: 1 → 1；
- 所有 loss / gradient finite。

固定 train subset：8 recalled positive、8 missing positive、8 high-burden negative；16 个 positive target 跨 seam。fixed predicted-only、无 rotation 的 4-epoch overfit：

- total loss：8.0993 → 8.1177（**未下降，FAIL**）；
- `L_db`：2.6603 → 2.5665（下降）；
- positive acceptance：0.25 → 0.3125（上升）；
- negative rejection：0.875 → 1.0（未崩）；
- candidate max=12，无 OOM/NaN。

## 11. GPU preflight

单 batch forward/backward：PASS；loss=1.7104；峰值 10.95 GiB；Bi-Layout feature/transformer/depth、Opening、Matcher 梯度全部非零且有限。

固定 24-train / 32-val 的 2-epoch 结果与冻结 T3 同 pair baseline：

| 指标 | T3 frozen | T4 epoch 1 | T4 epoch 2 |
|---|---:|---:|---:|
| candidate recall | 0.5000 | 0.5000 | 0.5625 |
| positive acceptance | 0.4375 | 0.3750 | 0.3750 |
| negative rejection | 0.7500 | 0.8750 | 0.8125 |
| conditional Top-1 | 0.3750 | 0.3750 | 0.3333 |
| end-to-end Top-1 | 0.1875 | 0.1875 | 0.1875 |
| Harmonic | 0.3000 | 0.3088 | 0.3047 |

有 recall 改善信号，negative 未崩，但没有 end-to-end Top-1 提升。

## 12. 完整训练启动门槛

14 项中 13 项通过。唯一失败：`7_small_overfit`，具体为 fixed-input total loss 未下降。完整机器可读证据见 `launch_gate_report.json`。

## 13. 是否启动

```text
full_training_started = false
do_not_start_full_training = true
```

未创建正式 checkpoint 目录，未产生正式 PID。

## 14. 后台训练状态与命令

没有后台 T4 full-training 进程；没有 `.running/.completed/.failed`，因为硬门禁前已停止。GPU 预检进程均已退出，最终 `nvidia-smi` 无 compute process。

若后续修复 gate 7，完整命令必须复用 T3 的 seed/batch/optimizer/scheduler/24 epoch/Harmonic selection 和三个独立初始 checkpoint，只增加第 7 节两类 T4 参数，并写入全新正式目录；本轮没有执行该命令。

## 15. 未访问 test 的证据

- 所有真实数据命令都显式 `--skip_test`。
- GPU report active splits 只有 `train,val`。
- audit/subset/CPU/GPU artifact 的 `test_data_accessed=false`。
- 没有解析项目 test manifest 或 test cache。

## 16. T0–T3 / M2 未被破坏的证据

- T3 best SHA-256 仍为 `ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d`，mtime 仍为 2026-08-15 02:54:10 +08:00。
- 初始 checkpoint SHA-256：Bi-Layout `377b84d1...`, Opening `93a86b10...`, Matcher `5da09959...`。
- 187 项回归包含 M2、joint-layout、三房间 group、full-val evaluator 回归并全部通过。
- 本轮未修改 M2 参数、scale、joint BEV、layout 或 T0–T3 checkpoint/log/output。

最小修复建议：先用相同 fixed subset 做一次只训练 matcher/dustbin（冻结 Bi-Layout/Opening）的诊断，确认总 matching loss 能否稳定下降；若能，再判断联合训练中是多任务 loss 方差还是 margin weight/学习率比例导致 gate 7 失败。该诊断属于下一轮，不应在本轮继续调参或占用 GPU。
