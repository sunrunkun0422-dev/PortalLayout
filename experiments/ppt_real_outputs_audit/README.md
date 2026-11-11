# PPT 真实实验输出图审计摘录

审计日期：2026-08-21

本目录只收录仓库中已经存在的图，文件均为原样复制，不修改像素、矢量内容或标注。当前没有一张图同时满足“全景输入 + predicted layout/openings + Rank-1/拒绝分数 + 三房间拓扑/联合 BEV + GT 对照”的完整 PPT 版式；因此本目录中的图均属于“真实结果，但放入正式汇报前需要排版或补充文字标注”。

## 统一模型与数据口径

- 指定三房间样本：ZInD `train`，house `0665`，`floor_03`，rooms `04-01-05`，panoramas `37-34-30`。
- 三房间图使用的端到端 checkpoint：`/home/feixia/pythonProject/AAAsrk/Bi_Layout/checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best.pt`，completed epoch 8。
- T3 SHA-256：`ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d`（由完整 validation 运行元数据记录）。
- Bi-Layout：`checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl`；配置：`src/config/zind_all.yaml`。
- Opening Head：`checkpoints/Opening_Head/zind_matching_v1/best.pt`；Matcher：`checkpoints/Cross_Scene_Matcher/zind_matching_v1/best.pt`。
- 三房间推理契约：predicted candidate + predicted opening guidance + learned bidirectional dustbin；GT 不参与模型判定或 candidate-local M2 几何，只在推理结束后用于评估/对照。
- 该指定样本是 train-only qualitative/debug case；原始 summary 明确写有 `debug/qualitative only, not a scientific evaluation`。PPT 中不能把它表述为 test 泛化结果或总体定量结论。

## 收录图片

### `positive_match_01_from_04.png`

- 原图：`/home/feixia/pythonProject/AAAsrk/Bi_Layout/src/output/zind_room_group_debug/t3_best_0665_floor03_candidate_local_m2/pairs/0665_floor_03_pano_34_pano_37_3_pre_manhattan_joint.png`
- 样本：`complete_room_01 <- complete_room_04`，`pano_34 <- pano_37`。
- 真实模型证据：predicted candidates `(1,0)`，Rank-1，pair match probability `0.8816207647`；relative pose 由模型选中的 predicted opening candidates 经 candidate-local M2 几何求得，GT 未参与。
- 图中内容：预测的 pre-Manhattan 双房间联合 BEV 和 shared opening。
- 可用表述：`T3 在 predicted openings 上选中 Rank-1 对应，并由开口局部几何生成相对对齐。`
- 必须补充：panorama、候选编号、`p=0.8816`、yaw/translation/scale，以及“opening-geometry pose”标记。当前图中文字重叠，不宜原样直接上最终 PPT。

### `positive_match_01_from_04_frame_preserving.png`

- 原图：同一 pair 的 `..._frame_preserving_joint.png`。
- 数据、checkpoint、candidate 与 GT 使用方式同上。
- 图中内容：同一推理的 Manhattan/frame-preserving 表示；它不是另一套 checkpoint，也不是 GT 对齐。
- 建议：只用于解释 pre-Manhattan 与 frame-preserving 表示差异；当前轮廓重叠和文字遮挡明显。

### `positive_match_01_from_05.png`

- 原图：`/home/feixia/pythonProject/AAAsrk/Bi_Layout/src/output/zind_room_group_debug/t3_best_0665_floor03_candidate_local_m2/pairs/0665_floor_03_pano_34_pano_30_4_pre_manhattan_joint.png`
- 样本：`complete_room_01 <- complete_room_05`，`pano_34 <- pano_30`。
- 真实模型证据：predicted candidates `(0,2)`，Rank-1，pair match probability `0.9060317874`；candidate-local M2 使用 predicted geometry，GT 未参与。
- 图中内容：预测的 pre-Manhattan 双房间联合 BEV 和 shared opening。
- 可用表述：`T3 对第二条相邻边同样给出 Rank-1 正匹配，并生成开口几何对齐。`
- 必须补充：panorama、候选编号、`p=0.9060`、yaw/translation/scale；当前 Room 05 标签被裁切。

### `positive_match_01_from_05_frame_preserving.png`

- 原图：同一 pair 的 `..._frame_preserving_joint.png`。
- 数据、checkpoint、candidate 与 GT 使用方式同上。
- 仅作为 frame-preserving 表示的真实输出；当前文字遮挡，不能无处理直接上最终 PPT。

### `before_alignment_01_04.svg` / `before_alignment_01_05.svg`

- 原图：T3 candidate-local M2 目录中两个 positive pair 的 `..._independent.svg`。
- 图中内容：两个房间各自的 predicted polygon，尚未进行跨房间相对对齐。
- 不含 GT、不含 opening candidate、不含匹配分数。
- 建议：与相应 `positive_match_*.png` 组成“匹配前 / 匹配后”两栏；必须在 PPT 中明确它们只是预测房间轮廓的独立局部坐标展示。

### `negative_pair_04_05_independent_only.svg`

- 原图：`/home/feixia/pythonProject/AAAsrk/Bi_Layout/src/output/zind_room_group_debug/t3_best_0665_floor03_candidate_local_m2/pairs/0665_floor_03_pano_37_pano_30_negative_independent.svg`
- 样本：`complete_room_04 <- complete_room_05`，`pano_37 <- pano_30`。
- 配套真实 JSON 给出：best pair `(-1,-1)`，match probability `0.0061624986`，no-match probability `0.9845531583`，未调用融合，未生成 joint BEV。
- 重要限制：SVG 本身只画两个 independent predicted polygons，图内没有 `no-match` 或 `0.9846`；它不能单独作为负样本拒绝证据。需要新增渲染/标注后才能用于最终 PPT。

### `single_room_pred_vs_gt_01.svg` / `single_room_pred_vs_gt_04.svg` / `single_room_pred_vs_gt_05.svg`

- 原图：T3 candidate-local M2 目录 `rooms/complete_room_{01,04,05}.svg`。
- 图中内容：单房间 predicted polygon（蓝）与 GT complete polygon（绿）的后验对照。
- GT 只用于推理后可视化；不是 GT candidate 注入，也不是模型输入。
- 重要限制：没有 panorama 和 predicted opening candidates；predicted 与 GT 使用各自局部坐标直接绘制，视觉差异很大，必须注明坐标/尺度口径，不能声称是已配准的 IoU 图。

### `relative_pose_pred_vs_gt_val_example.svg`

- 原图：`/home/feixia/pythonProject/AAAsrk/Bi_Layout/src/output/zind_geometry_val_full/t3_frozen_m2/visualizations/02_yaw_improved_0989_floor_01_pano_3_pano_5_1.svg`
- 样本：ZInD `val`，pair `0989_floor_01_pano_3_pano_5_1`，不是 house 0665。
- checkpoint：同一 T3 `best.pt`；predicted candidates，GT 不参与 prediction/geometry，只作推理后虚线对照。
- 图中内容：GT A/B、frame-polygon B 与 candidate-local M2 B 的对照；该样本是 post-hoc `yaw_improved` 选择。
- 可用表述：`完整 validation 冻结推理中的一个 M2 yaw-improved 诊断样本。`
- 不能表述为随机样本、总体平均效果或 house 0665 的输出；正式 PPT 应补 pair ID、选择规则与数值。

### `candidate_recall_predicted_vs_recall_topk_val_example.png`

- 原图：`/home/feixia/pythonProject/AAAsrk/Bi_Layout/experiments/zind_e2e_v3/g0_seed123/T5_candidate_recall/audit/visualizations/target_present_correct/0003_floor_01_pano_28_pano_31_9.png`
- 样本：ZInD `val`，pair `0003_floor_01_pano_28_pano_31_9`。
- 曲线是真实 T3 frozen opening probability；legacy candidate 和 Recall-TopK 是从预测概率生成的候选。绿色 GT interval 是推理后的诊断覆盖，不是模型输入。
- 图中没有 aggregate `58.89% -> 60.10%`。图顶部标题/legend 有遮挡，需要重新排版。
- 不能把 Recall-TopK 诊断称为已经部署的 T3 production output；相应审计把 oracle/proposal 对比标记为 post-hoc/oracle-only。

## 未收录，因为当前没有合格现成图片

- `negative_no_match_04_05.png`：只有 JSON 数值和 independent SVG，没有带拒绝标签/分数的真实结果图。
- `three_room_topology_0665_floor03.png`：只有 `adjacency_pred.json`/`group_summary.json`，没有 topology renderer 输出。
- `three_room_joint_bev_pred_vs_gt.png`：只有两张 pairwise joint BEV，没有把 04-01-05 放入同一全局坐标的联合 BEV，也没有三房间 pred-vs-GT overlay。
- `t0_t3_metrics.png`：T0-T3 有 `metrics.jsonl` 与 `best_metrics.json`，没有 PNG/SVG/PDF 或对应 TensorBoard event。
- `candidate_recall_comparison.png`：`58.8944% -> 60.0985%` 只有 JSON/CSV/Markdown 表。该变化来自历史兼容 checkpoint 的 G0/G1 threshold scan，不是 T3 best 曲线；当前 T3 best validation candidate recall 为 `59.1133%`。
- Panorama A/B + predicted layout + boundary opening candidates 的组合图：现有 room renderer 只画 predicted-vs-GT polygons；candidate audit renderer 只画 token probability/interval，不画 panorama/layout/BEV。

## 严禁误用

- 文件名包含 `gt_target`、目录包含 `geometry_oracle` 的图不能称为模型预测。
- `single_room_pred_vs_gt_*` 和 `relative_pose_pred_vs_gt_val_example.svg` 含 GT 后验叠加，必须显式标注 GT。
- 0665 三房间样本是 train debug case，不能写成 val/test 定量例证。
- candidate-local M2 的 yaw/translation/scale 是在模型选中 opening pair 后由预测开口几何计算，不是独立 learned pose head 的直接输出。
