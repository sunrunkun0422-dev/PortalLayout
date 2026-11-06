# Matcher 汇报定向修改记录

日期：2026-09-10。主汇报 10 页，另保留 6 页 Backup / Appendix。

基于 `../matcher_progress_ppt_20260909/build_ppt.py` 对应的《Matcher阶段汇报_五类案例与修改建议》可编辑源文件修改，同时核对用户提供的 `../matcher_progress_ppt_20260909/Matcher阶段汇报.pptx`。沿用 16:9、白底、暗红 `#853E45`、Arial / Noto Sans CJK SC 以及原页脚结构，日期更新为 2026.09.10。原目录及其所有内容文件均未覆盖或删除。

输入 PPTX 与源文件存在差异：输入 PPTX 第 8、9 页重复了 Negative False Accept，第 8 页全量分析和原第 12 页阶段总结未保留在其对应位置。本次以已有生成源文件和 `slide_specs.json` 恢复相应内容：T3 失败分解进入 P8，原 E1 对照进入 A6，阶段总结进入 A5；重复案例不再占据主汇报页数。原开口类型页删去“教师建议”前缀的措辞保留。

## P1 · Matcher 任务定义

- 修改：保留“双全景 → Bi-Layout / Opening prediction → Matcher → Accept / No Match → Relative Pose”，将核心信息压缩为 Candidate Recall、Correspondence Ranking、Pair Decision 三个研究问题。底部改为指定的冻结 T3 评价范围。
- 原因：让第一页解释研究问题及 Matcher 的职责，减少系统细节。
- 来源：原可编辑源文件 P1；现有 `models/cross_scene_matcher.py` 的排序和双向 dustbin 决策逻辑。
- 实验结论：未改变；仅收紧任务表述。

## P2 · T3 全量结果

- 修改：保留 27.42%（501/1827）、46.39%（501/1080）、92.79%（1673/1803）及 T0–T3 完整表格；分别明确最终 opening 选择、GT 已召回条件和负样本 No Match 拒绝的分母。底部改为“T3 在当前四个配置中取得最高 Harmonic，但正样本选择能力仍明显受限”。不再突出 +2.19 pp。
- 原因：避免将 S3 误读为 pair classification accuracy，避免过度解释单 seed 增益。
- 来源：`checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T0–T3/best_metrics.json` 中的 `validation.predicted_openings`；`src/output/zind_geometry_val_full/t3_frozen_m2/summary.json` 与 `per_pair.jsonl`。
- 实验结论：原始数值不变；降低措辞强度。Conditional success 为 recalled positive 中最终选对的比例，并非不含拒绝的 Raw Top-1。

## P3 · Correct Match

- 修改：保留原案例；中心顺序统一为矩阵、GT Pair / score / rank、Raw Rank-1、Final Decision。补充指定 Observation 与 Implication。
- 原因：用同一模板说明正确 correspondence 的具体行为。
- 来源：`src/output/zind_matcher_visual_report/case_selection.json`；案例 `0461_floor_02_pano_76_pano_83_0` 的原始图片、冻结预测和 score artifact。GT / Raw / Final 均为 A1–B0，real-real score 显示为 0.603。
- 实验结论：未改变；保留 `GT is used for evaluation only and is not model input.`

## P4 · Recalled → Dustbin Reject

- 修改：保留原案例；突出 GT A0–B0、GT Rank 1、real-real score 0.057、Final REJECT / No Match、no-match / dustbin score 0.925。加入“Rank-1 ≠ 高置信度”。Observation 与 Implication 明确区分候选排序和最终接受 / 拒绝。
- 原因：清楚呈现“候选已召回、排序第一、最终仍拒绝”的失败边界。
- 来源：案例 `0417_floor_01_pano_45_pano_43_14` 的原图、冻结预测、score artifact 和已有视觉审计。
- 实验结论：数值和类别不变；“错误发生在最终决策”是行为分解，不声称只调阈值就能解决全部同类错误。

## P5 · Recalled → Wrong Selection

- 修改：移出有 `possible` 标注歧义记录的旧案例 `0220_floor_01_pano_40_pano_37_13`，使用已有冻结 val 案例 `0769_floor_02_pano_46_pano_71_6`。新例 GT A1–B1，GT score 0.080152（显示 0.080），GT Rank 3；Raw Rank-1 / Final A0–B1，score 0.553529（显示 0.554），最终 ACCEPT。图像、候选索引与预测未修改。
- 原因：提供 GT 清楚、两侧已召回、最终接受但明确错选、且不存在已知物理兼容替代歧义的主汇报案例。
- 来源：`T5_candidate_recall/audit/visual_audit/visual_observations.jsonl` 中 `reviewed=true`、`gt_opening_visibility=clear`、`annotation_ambiguity=none`；`audit/per_pair_val.jsonl` 中两侧 `present`、`multiple_shared_openings=false`，仅一组物理共享 opening；`T5_bev_verifier/preflight/shortlist_pairs_val.jsonl` 中严格映射唯一且 selected physical correctness=false；`phase_B_pair_meta_val.jsonl` 中物理目标位置与严格 GT 目标位置一致，baseline physical correctness=false。展示图片复用 `candidate_review/case_03_option_1.png`。
- 实验结论：更换定性例子，不改变 full-val 错选数 142 或任何实验统计。“无已知歧义”限于已有标注与物理兼容审计协议。原有歧义案例文件仍在原目录保留。
- 补充：严格 GT 索引与预测候选索引是不同空间；没有将标注 `[0,0]` 直接当作预测 A0–B0。使用现有明确映射后的 GT A1–B1。

## P6 · Negative Correct Reject

- 修改：保留原案例；删除负样本的绿色 GT shared opening 图例，采用蓝色 Predicted Candidate、紫色 Raw Rank-1、橙色 Final Selection。明确 `GT Label: Non-adjacent`、`Shared Opening: None`、Raw A1–B0、Final REJECT / No Match。补充指定 Observation 与 Implication。
- 原因：负样本没有 GT shared opening；同时说明 No Match 是必要输出。
- 来源：案例 `0003_floor_01_pano_95_pano_34_negative` 的原图、冻结 val label 与 score / decision artifact。
- 实验结论：未改变。

## P7 · Negative False Accept

- 修改：保留原案例；同 P6 删除绿色 GT 图例，明确 Non-adjacent、Shared Opening None、Raw A1–B0、Final ACCEPT A1–B0，增加指定观察与 false positive 约束。
- 原因：提高正样本接受率时必须同时追踪错误连接。
- 来源：案例 `0293_floor_01_pano_36_pano_30_negative` 的原图与冻结 val artifact；real-real score 显示为 0.686。
- 实验结论：标签、分数与决定不变。局部外观相似为定性观察，不作为单例因果证明。

## P8 · 性能损失分解

- 修改：以 T3 自身结构为主：Candidate Missing 747 / 40.89%；Recalled → Reject 437 / 23.92%；Recalled → Wrong 142 / 7.77%；Correct 501 / 27.42%。右侧单列 1080 recalled positive = 501 correct + 437 rejected + 142 wrong selection。强调 747 个 candidate missing 不能归因于 Matcher，以及不能仅用 27.42% 评价 Matcher 本身。E1 原对照移至 A6。
- 原因：分开上游召回损失和 Matcher 排序 / 最终决策损失，使其成为主汇报最重要的 error analysis。
- 来源：`src/output/zind_geometry_val_full/t3_frozen_m2/summary.json` 的 `failure_decomposition`；CPU 重新汇总 `per_pair.jsonl` 的 3630 条冻结记录，所有计数完全一致。
- 实验结论：没有新增实验或改变统计；仅改变论述重点。

## P9 · Oracle Pose

- 修改：新增两例并列的 current M2 预测流程与 Oracle 对照。Yaw / Translation 作为主指标，log-scale 次级显示。
- 原因：验证正确 opening 与 correspondence 对后续现有几何求解有实际价值，同时保留证据边界。
- 来源：`src/output/zind_pose_oracle_gt_opening_0665_floor03/summary.json`、`per_pair.json` 与既有 `REPORT.md`。Baseline 只取 `baseline_frozen_metrics`；Oracle 取 `metrics`。没有使用 `historical_baseline.json` 或旧 polygon baseline。
- 实验结论：将已有真实 Oracle 结果加入汇报，没有重新运行实验。两例属于 **train**，不是冻结 val 中的两个示例。两例同时替换 GT opening geometry 与 GT correspondence，仍保留原 current M2 冻结预测轮廓用于端点消歧；不能单独归因于 Matcher，也不能据此判断 full-val 全局主要瓶颈。页面同时显示中英文限定语。

| Pair | Setting | Yaw error | Translation error | Log-scale error |
| --- | --- | ---: | ---: | ---: |
| 04–01 | Pred opening + T3 + current M2 | 4.3563° | 0.60094 m | 0.247696 |
| 04–01 | GT opening + GT correspondence | ≈ 0° | 0.08743 m | 0.023759 |
| 01–05 | Pred opening + T3 + current M2 | 9.3342° | 0.42652 m | 0.119119 |
| 01–05 | GT opening + GT correspondence | ≈ 0° | 0.05266 m | 0.032773 |

第一例 Oracle yaw 原始值为 `2.5444437451708134e-14°`，第二例为 `0°`。Translation 使用 `translation_error_meters`，未将原坐标单位误当米。

## P10 · 下一步

- 修改：仅保留 Step 1 改善 opening candidate recall；Step 2 改善 recalled candidates 上的 final decision，并区分 142 个 Ranking Error 与 437 个 Dustbin Reject Error。统一列出 S3、Conditional Top-1、Top-k ranking、Negative rejection、False Positive、Original 501 success retention。
- 原因：围绕已有、最明确的失败边界决定下一步，避免在主汇报同时展开多个未验证方向。
- 来源：冻结 T3 失败分解；原下一步页中的评价约束。Conditional Top-1 在本汇报定义为 recalled positive 中最终选对率；Top-k 单列原始排序，避免混用。
- 实验结论：下一步范围收紧；没有执行新实验或声称改善已实现。

## A1 · Physics / BEV Verifier

- 修改：原 P9 原生可编辑内容移至附录，增加 `Backup / Appendix` 与 `Proposed / Not yet validated` 标识。
- 原因：保留原有方案和既有消融证据，但不占用当前主线。
- 来源：原 `build_ppt.py` P9、原备注和引用；原 Phase A 的 score-only 731 / full 729 对照原样保留。
- 实验结论：未改变；未将物理 / BEV 增益写成已验证。既有文献链接保留于备注，本轮未新增文献主张。

## A2 · Opening Type

- 修改：原 P10 移至附录，增加 `Proposed / Not yet validated`；保留软类型、unknown、标签边界和原消融建议。
- 原因：完整保存尚未实现的 proposal，同时避免与主实验结果混淆。
- 来源：原可编辑 P10 和对应备注；用户输入 PPTX 中“不同开口采用不同特征”的措辞。
- 实验结论：未改变；当前仍非已实现 / 已验证结果。

## A3 · Long Corridor Idea

- 修改：原 P13 的走廊原生示意图、文字和备注完整移至附录，增加 `Proposed / Not yet validated`。
- 原因：保留设想，避免主汇报扩展新研究方向。
- 来源：原可编辑 P13。
- 实验结论：未改变；仍为概念示意，不作为实验数据。

## A4 · 详细实验计划

- 修改：原 P11 的 A/B/C 详细计划、评价约束和 T4 反例移至附录，标注 `Proposed / Not yet validated`。备注明确其为原备份内容，不是本次主汇报承诺。
- 原因：保留旧计划文件内容，主线最后一页只讲当前两个优先事项。
- 来源：原可编辑 P11 和备注。
- 实验结论：旧证据不变；实际下一阶段优先级以新 P10 为准。

## A5 · 原阶段总结

- 修改：从可编辑源文件恢复原 P12，作为“历史阶段记录 / 含未验证方案”保存在附录。
- 原因：用户要求保留原阶段总结；输入 PPTX 中该页已不在原位置。
- 来源：原 `build_ppt.py` P12 和 `slide_specs.json`。
- 实验结论：未改变，T3 已完成、E1 独立预检、物理 / 语义未完成等状态原样保留。

## A6 · E1 预检结果

- 修改：原 P8 的 E1 表格、收益代价分解与备注移至附录，标为“已有独立预检 / 非 T3 主结果”。
- 原因：P8 优先讲清 T3 失败结构，同时保留原 E1 数据。
- 来源：原可编辑 P8；`T5_bev_verifier/preflight/phase_B_summary.json`，冻结 T3 summary。
- 实验结论：未改变：E1 正确匹配 +25、负样本 FP +20；恢复拒绝 31 + 修复错选 8 − 丢失成功 14 = 25。未将其解释为物理特征的独立因果增益。

## 全局 score 表达核对

`models/cross_scene_matcher.py` 的既有逻辑：`candidate_pair_score = mutual_affinity × two-sided opening confidence`；展示的 `pair_no_match_probability` 是两侧平均 dustbin 值的几何均值。它们不构成一对互补的邻接概率，且最终决策不是简单比较这两个汇总量。

Final 在通过双向 dustbin 条件的候选中取最高 real-real score；若没有通过的候选则 No Match。本次仅阅读代码并核对冻结 artifact，没有导入或运行 Matcher。P3–P7 均使用 Real-real score、No-match / dustbin score、Raw Rank-1、Final Decision / Final 等术语，不使用 match probability 或 no-match probability 作为页面解释。

## 输出与复现

- `Matcher阶段汇报_收紧主线与OraclePose.pptx`：10 页主汇报 + 6 页附录，可编辑。
- `Matcher阶段汇报_收紧主线与OraclePose.pdf`：对应完整 PDF。
- `Matcher阶段汇报_收紧主线与OraclePose_主汇报10页.pptx/.pdf`：同内容的独立 10 页主汇报。
- `previews/`：完整 16 页的 PNG，主汇报 P01–P10、附录 A01–A06。
- `整套预览.png`：所有页面的缩略总览。
- `build_ppt.py`、`layout_helpers.py`、`appendix_original.py`：本次可编辑生成源。
- `case_selection_revised.json`、`slide_specs.json`：案例选择与逐页内容 / 备注。
- `data_verification.json`、`verification.json`、`source_manifest.json`：数据核对、导出检查和来源哈希。

本轮未训练、未运行任何模型前向、未加载或修改 checkpoint，未修改 Matcher、预测、threshold，未访问 test。仅进行冻结结果读取、CPU 汇总与文件排版导出。所有已读取原始来源文件在生成结束时重新核对 SHA-256 未变。
