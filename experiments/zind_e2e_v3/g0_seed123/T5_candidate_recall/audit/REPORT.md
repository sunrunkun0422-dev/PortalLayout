# ZInD candidate failure audit

## 1. 一句话结论

主要决策为 **inconclusive**：do not change model yet; request full-train expansion。train-4096 与 full-val 的稳定性门禁为
`False`；是否建议补跑 full train：`True`。

## 2. branch、HEAD、worktree

- branch: `agent/zind-bipair-opening-matcher`
- HEAD: `50a6ae5ca403928627fb4de18ae918d59c03d479`
- worktree 保持 dirty，未 reset/checkout/clean。
- 审计开始时已有修改及最终 `git status --short` 原样保留：

```text
 M .gitignore
 M models/cross_scene_matcher.py
 M postprocessing/post_process.py
 M tests/test_zind_end_to_end_candidate_fixes.py
 M tools/debug_cross_scene_flow.py
 M tools/train_zind_cross_scene_matcher.py
 M tools/train_zind_end_to_end.py
 M utils/joint_layout.py
?? experiments/zind_e2e_v2/full_ablation_g0_seed123/
?? experiments/zind_e2e_v3/
?? src/dataset/
?? src/output/cross_scene_flow_zind_bipair_v1/
?? src/output/cross_scene_format_smoke/
?? src/output/progress_report_20260716/cross_scene_pair/
?? src/output/progress_report_20260716/zind_adjacent_pairs_eval/
?? src/output/zind_bipair_v1_smoke/
?? src/output/zind_cross_scene_eval_zind_bipair_v1/
?? src/output/zind_geometry_val_full/
?? src/output/zind_geometry_val_smoke/
?? src/output/zind_matching_dataset_smoke/
?? src/output/zind_opening_recall/
?? src/output/zind_room_group_debug/
?? tests/test_audit_zind_candidate_failures.py
?? tests/test_audit_zind_joint_geometry.py
?? tests/test_audit_zind_portal_geometry.py
?? tests/test_candidate_local_portal_geometry.py
?? tests/test_evaluate_zind_portal_geometry_val.py
?? tests/test_evaluate_zind_portal_geometry_val_full.py
?? tests/test_evaluate_zind_room_group.py
?? tests/test_frame_preserving_manhattan.py
?? tests/test_zind_t4_matcher_only.py
?? tests/test_zind_t4_medium.py
?? tests/test_zind_t4_recall.py
?? tools/audit_zind_candidate_failures.py
?? tools/audit_zind_joint_geometry.py
?? tools/audit_zind_portal_geometry.py
?? tools/audit_zind_t4_recall.py
?? tools/diagnose_zind_t4_medium.py
?? tools/evaluate_zind_portal_geometry_val.py
?? tools/evaluate_zind_portal_geometry_val_full.py
?? tools/evaluate_zind_room_group.py
?? tools/preflight_zind_t4_cpu.py
?? tools/preflight_zind_t4_matcher_only.py
?? tools/prepare_zind_t4_preflight_subsets.py
?? tools/summarize_zind_t4_preflight.py
?? utils/candidate_failure_trace.py
?? utils/candidate_local_portal_geometry.py
```

## 3. 授权范围内实际执行内容

只执行 T3 `eval()` + `torch.inference_mode()` 的 unique-room 前向、冻结 Matcher
重评分、CPU proposal/GT 后验审计和可视化。未创建 optimizer、未 backward、未训练。

## 4. room-level 去重前后工作量

- train-4096: 8192 sides → 6939 rooms，节省 15.30%。
- full-val positive: 3654 sides → 2546 rooms，节省 30.32%。
- 总 unique room: 9485。

## 5. Phase 0 时间、显存、磁盘预测

- PASS=`True`；64-room batch 平均 0.2151s。
- peak allocated=0.62 GiB。
- 预计缓存=4.72 GiB，约占当时可用空间 1.24%。
- opening/feature/depth、legacy candidate 与 pair regroup parity 均通过 2e-5 门禁。

## 6. GPU 实际运行时间和缓存完成率

Phase 1 完成 9485/9485 rooms，新增运行
14.19 分钟，写入 4.69 GiB。

## 7. train-4096 house 覆盖

覆盖 1260 个 train house；抽样顺序 hash=`9f9468d8e62d225c9ed206ddb6264e6866c7db01256df539654ecde642d17233`。

## 8. legacy / Recall-TopK parity

- legacy: 1080/1827（exact=True）
- Recall-TopK: 1107/1827（exact=True）

## 9. train-4096 与 full-val stage funnel

完整逐阶段、interface、wrap、house 与 unique-side 统计见
`stage_funnel_train.json` 和 `stage_funnel_val.json`。

## 10. 747 个 missing candidate 原因分解

full-val legacy missing=747；互斥主因：
`{"below_threshold_A": 14, "below_threshold_B": 5, "candidate_cap_removed": 1, "dedup_removed": 4, "interval_expansion_failure": 1, "interval_merge": 413, "interval_split": 16, "no_opening_signal_A": 223, "no_opening_signal_B": 63, "no_opening_signal_both": 7}`。

## 11. O0–O5 结果

| Oracle | candidate recall | predicted match | conditional Top-1 | S3 |
|---|---:|---:|---:|---:|
| O0 | 1080/1827 | 806/1827 | 0.4639 | 501/1827 |
| O1 | 1643/1827 | 691/1827 | 0.3670 | 603/1827 |
| O2 | 1235/1827 | 788/1827 | 0.4405 | 544/1827 |
| O3 | 1827/1827 | 652/1827 | 0.3569 | 652/1827 |
| O4 | 904/1827 | 681/1827 | 0.5033 | 455/1827 |
| O5 | 1080/1827 | 1080/1827 | 1.0000 | 1080/1827 |

所有 O1–O5 artifact 均标记 `oracle_only=true`, `valid_for_inference=false`。

## 12. train/val house-bootstrap 及一致性

- 2,000 次 house-cluster bootstrap（全体 A/B/C/D failure，success 不进分母）；train/val top reason 相同=`False`。
- top reason 最大 95% CI 宽度=0.0666；情形排序一致=`False`。
- train decision=`B_proposal_logic`；val decision=`D_matcher`。

## 13. 推荐情形 A/B/C/D

推荐 **inconclusive**。分类计数：`{"train": {"A_opening_head_signal": 620, "B_proposal_logic": 1490, "C_target_mapping": 1, "D_matcher": 320, "success": 1665}, "val": {"A_opening_head_signal": 293, "B_proposal_logic": 454, "D_matcher": 579, "success": 501}}`。

## 14. 是否需要补跑完整 train

`True`。本轮没有也不会自动扩展至 14,741 train positives。

## 15. 下一阶段最小方案

do not change model yet; request full-train expansion。本报告只提出方案，不执行修复或训练。

## 16. test 隔离、无训练及 checkpoint 保护证据

- active splits=`train,val`；`test_data_accessed=false`。
- training/optimizer/backward/parameter update 均为 false。
- T3 SHA=`ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d`，size/mtime 未变。
- M2、scale、layout、joint BEV、正式 proposal defaults 和 T4 输出均未修改。

## 17. PID、状态文件和 GPU 最终状态

PID、完整/恢复命令、`.started/.completed/.failed`、room/pair progress 位于本目录。
本轮采用前台分阶段执行；最终 report 的隔离命名空间 PID=`2`，所有阶段均已退出。
完整分阶段命令与恢复命令见 `executed_commands.json`；`.completed` 存在，`.failed` 不存在。
2026-08-19T19:50:39+08:00 的只读 `nvidia-smi`：RTX 4060 Ti，741/16380 MiB，
compute process 数为 0。结构化记录见 `gpu_final_status.json`。
