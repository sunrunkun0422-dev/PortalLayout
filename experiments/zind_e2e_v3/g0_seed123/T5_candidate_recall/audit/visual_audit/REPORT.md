# Full-val Opening Proposal 与 Matcher 失败样本视觉审计

## 结论摘要

本报告只总结确定性抽取并逐图查看的 52 个案例。候选缺失样本为 30/747；Matcher 失败样本为 16/579；另有 6 个正确对照。抽样比例**不得直接外推**为全部 747 或 579 个样本的真实比例。

- 30 个候选缺失自动归因的视觉核对：confirmed=10，partially_confirmed=14，contradicted=0，indeterminate=6。
- interval_merge（8例）主要视觉机制：`broad response bridge`。该自动标签不是显式 merge 操作，只表示一个阈值连通区相对 GT 过宽。
- no_opening_signal（8例）：GT 开口 clear=1；partial/not_visible/uncertain 合计=7。
- dustbin rejection（8例）的视觉/分数判断见逐例记录；主要机制计数：`{"annotation ambiguity": 3, "candidate score calibration": 2, "repeated door appearance": 3}`。raw-logit dustbin margin 与 candidate_pair_score 严格分开。
- wrong selection（8例）的主要机制计数：`{"annotation ambiguity": 3, "repeated door appearance": 4, "wrap/seam handling": 1}`；重复门外观只能在逐图可见时报告。
- Dustbin 结论：8例中 6 例真实 pair 的 real-real rank≤2，但不能据此称为‘高绝对分数被保守拒绝’。只有 0780 的 raw true-pair logit 1.3846 与共享 dustbin 1.3946 几乎贴边；0417、0654、0780 三例的视觉对应清楚而拒绝不具说服力，0139、0286、1084 三例视觉上可辩护，0590、0888 两例介于两者之间。
- Wrong-selection 结论：重复门外观是 4/8 的首要视觉机制，是最大单类但不是绝对多数；标注/投影歧义为 3/8，seam 为 1/8。除 1095 的 selected-vs-true gap 仅 0.0055 外，其余 7 例 gap 为 0.1634–0.9233。

## 可复现性与安全边界

- branch: `agent/zind-bipair-opening-matcher`
- HEAD: `50a6ae5ca403928627fb4de18ae918d59c03d479`
- active split: `val`; `test_data_accessed=false`。
- GPU/model forward/training/optimizer/backward: `false`；未修改 Opening threshold、Proposal、Matcher、M2、T0–T4 或正式默认值。
- T3 before/after SHA: `ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d` / `ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d`；size=418836277；mtime_ns=1786733650604347717；identity unchanged=`True`。
- pair-list SHA: `a75af0cf7e3c2e9e3ece66e81ed25508ec92501e8bac9966d7594a2134a89997`；selection SHA: `100d56316b575306b8b9dd04c17c70921444b0a6fb4ab971f9973743665f7fec`。
- observation inputs: `experiments/zind_e2e_v3/g0_seed123/T5_candidate_recall/audit/visual_audit/manual_visual_observations.jsonl`。

## 冻结统计门禁

`1827 / 1080 / 747 / 501 / 437 / 142` 与七类 `413 / 293 / 19 / 16 / 4 / 1 / 1` 均由 per-pair artifact 重算并与冻结汇总零差异。

## Artifact 限制与直接证据边界

- 图像直接支持：RGB 可见性、视角、遮挡、照明、邻近/重复开口外观，以及曲线/区间/score 面板中实际显示的数值关系。
- 推断：把可见困难归为 Opening Head 原因、把相似门归为 Matcher 错选原因、或把高分拒绝归为校准问题；这些都在 JSONL 的 `inferences` 单列。
- 证据不足：未保存 predicted candidate 语义类别、未扩张 raw interval、显式 merge lineage、candidate-local M2 endpoint；不得从图中伪造。
- O0 dustbin compact serialization 有 batch-padding 定位缺陷。可信直接值全量唯一为共享 logit `1.394599080`；离线恢复后 pair no-match 最大误差 `1.21e-07`，best pair 复现 `1827/1827`。每例保留 direct/recovered provenance，未使用 -FLT_MAX sentinel。

## 责任域解释

- `missing visual response` 属于 Opening Head 可见响应现象；是否应训练仍需独立实验。
- `broad response bridge`、`adjacent openings merged`、`split`、`below threshold`、dedup/cap 属于 Proposal 规则的事后现象；本轮未调阈值。
- `candidate score calibration` 与错误候选排序属于 Matcher；视觉相似性只是解释线索，不是因果证明。
- `annotation ambiguity` 只在 RGB 与投影 GT 明显不一致时作为可能性，不等同于已证实标注错误。

## 重点问题的逐类回答

- **Opening Head**：no_opening_signal 的 8 例中只有 1 例 GT 清晰可见；5 例主现象是 GT 内缺失/错位响应，2 例其实有亚阈值响应，1 例以 edge-on 视角为主。
- **Proposal**：interval_merge 的 7/8 是单个过宽响应桥接，只有 1/8 能较明确看到相邻开口共同落入宽区间；split 4/4 的 GT 内可见分离片段，below-threshold 4/4 可见冻结阈值以下响应。dedup/cap/expansion 的小样本多伴随 GT 语义歧义，不能仅由 RGB 证明规则内部因果。
- **Matcher**：dustbin 样本常在 real-real 排序中靠前（6/8 rank≤2），但绝对 raw-logit margin 并非普遍接近 dustbin；wrong-selection 以重复门和含混 GT 为主，候选上下文不足/分数排序是推断而非图像直接因果。
- **标注歧义**：墙面、柜体、窗/凹位或超宽跨 seam GT 在多个样本中直接可见；这里只标记 `possible/strong`，没有宣称标注错误已经证实。

## 代表性图片

- **interval_merge**: [`0059_floor_01_pano_17_pano_14_6`](figures/interval_merge/0059_floor_01_pano_17_pano_14_6.png), [`0527_floor_01_pano_15_pano_36_0`](figures/interval_merge/0527_floor_01_pano_15_pano_36_0.png)
- **no_opening_signal**: [`1125_floor_00_pano_34_pano_43_9`](figures/no_opening_signal/1125_floor_00_pano_34_pano_43_9.png), [`0184_floor_01_pano_34_pano_50_10`](figures/no_opening_signal/0184_floor_01_pano_34_pano_50_10.png)
- **interval_split**: [`1359_floor_01_pano_6_pano_15_11`](figures/interval_split/1359_floor_01_pano_6_pano_15_11.png), [`0547_floor_01_pano_16_pano_17_9`](figures/interval_split/0547_floor_01_pano_16_pano_17_9.png)
- **below_threshold**: [`0761_floor_01_pano_8_pano_5_2`](figures/below_threshold/0761_floor_01_pano_8_pano_5_2.png), [`1327_floor_01_pano_4_pano_25_0`](figures/below_threshold/1327_floor_01_pano_4_pano_25_0.png)
- **dedup_removed**: [`0161_floor_01_pano_25_pano_32_8`](figures/dedup_removed/0161_floor_01_pano_25_pano_32_8.png), [`0664_floor_02_pano_29_pano_31_5`](figures/dedup_removed/0664_floor_02_pano_29_pano_31_5.png)
- **candidate_cap_removed**: [`1037_floor_01_pano_12_pano_20_1`](figures/candidate_cap_removed/1037_floor_01_pano_12_pano_20_1.png), [category montage](montages/candidate_cap_removed.png)
- **interval_expansion_failure**: [`0492_floor_02_pano_46_pano_90_11`](figures/interval_expansion_failure/0492_floor_02_pano_46_pano_90_11.png), [category montage](montages/interval_expansion_failure.png)
- **target_present_dustbin_reject**: [`0286_floor_02_pano_33_pano_34_8`](figures/target_present_dustbin_reject/0286_floor_02_pano_33_pano_34_8.png), [`0417_floor_01_pano_45_pano_43_14`](figures/target_present_dustbin_reject/0417_floor_01_pano_45_pano_43_14.png)
- **target_present_wrong_selection**: [`0062_floor_01_pano_21_pano_7_4`](figures/target_present_wrong_selection/0062_floor_01_pano_21_pano_7_4.png), [`0220_floor_01_pano_40_pano_37_13`](figures/target_present_wrong_selection/0220_floor_01_pano_40_pano_37_13.png)
- **target_present_correct**: [`0461_floor_02_pano_76_pano_83_0`](figures/target_present_correct/0461_floor_02_pano_76_pano_83_0.png), [`0712_floor_01_pano_34_pano_36_2`](figures/target_present_correct/0712_floor_01_pano_34_pano_36_2.png)

## 逐例观察

完整字段见 `visual_observations.csv` 与 `visual_observations.jsonl`；`automated_vs_visual_confusion.csv` 为自动原因与人工主机制交叉表。所有 `visual_evidence` 仅描述图中或冻结分数面板中可见内容。

## 实际执行命令

```bash
env CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg MPLCONFIGDIR=/tmp/bi_layout_visual_audit_mpl /home/feixia/anaconda3/envs/bi_layout/bin/python tools/visual_audit_zind_candidate_failures.py render
env CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg MPLCONFIGDIR=/tmp/bi_layout_visual_audit_mpl /home/feixia/anaconda3/envs/bi_layout/bin/python -m unittest tests.test_visual_audit_zind_candidate_failures -v
env CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg MPLCONFIGDIR=/tmp/bi_layout_visual_audit_mpl /home/feixia/anaconda3/envs/bi_layout/bin/python tools/visual_audit_zind_candidate_failures.py finalize --observations experiments/zind_e2e_v3/g0_seed123/T5_candidate_recall/audit/visual_audit/manual_visual_observations.jsonl
git diff --check
```

本轮新增/修改范围：`tools/visual_audit_zind_candidate_failures.py`、`tests/test_visual_audit_zind_candidate_failures.py` 与本目录审计产物。其余 dirty worktree 为既有内容并原样保留，未 reset/checkout/clean。最终 `git status --short` 与额外验证命令由交付摘要报告。
