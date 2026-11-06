# Pose Oracle Bottleneck Experiment — 0665 / floor_03

日期：2026-09-08。仅 train 的两个固定正边，非总体泛化结论。

## 实验与输入隔离

调用现有 `utils.joint_layout.build_joint_layout_from_endpoints`，未修改 solver。
只替换 GT opening 和 GT correspondence；保留当前 M2 冻结预测的 pre-Manhattan
房间轮廓，用于原有质心异侧端点消歧。没有使用 GT 房间轮廓。
这测试的是“GT opening/correspondence + 原预测轮廓消歧”的 solver，不是输入完全无误的房间重建 oracle。

GT 来自 `0665/zind_data.json#/merger/floor_03` 的三个指定 panorama。
复用 `dataset.zind_pair_mining._interfaces`：优先 `layout_complete`，否则
`layout_raw`。本案例二者的 opening 数组并不相同，不能混用。
doors/openings 按三条记录一组读取；本实验只使用每组前两条二维端点，
不将第三条附加记录当作第三个平面点。
GT correspondence 取 train manifest 的 `supervision.target_candidate_a/b`，
并核对原始标注端点、manifest local/global endpoints 和既有 0.06 global-coordinate
physical compatibility 协议。GT 索引不是 Matcher predicted candidate index。

Solver 的独立函数只接收四项：两组 local endpoints、两组 frozen predicted boundaries。
不接受 GT transform、global endpoints、尺度换算因子、Matcher score 或预测开口索引。
GT 相对变换在求解完成后才构造，使用现有 `relative_zind_transform`；
metric factor 只用于评估和输出米制量。

## 当前 M2 对照

| Pair | Setting | Yaw error (deg) | Translation error (m) | Absolute log-scale error |
| --- | --- | ---: | ---: | ---: |
| 04-01 | Pred opening + T3 + current M2 | 4.356263 | 0.600936 | 0.247696 |
| 04-01 | GT opening + GT correspondence | 0.000000 | 0.087429 | 0.023759 |
| 01-05 | Pred opening + T3 + current M2 | 9.334189 | 0.426519 | 0.119119 |
| 01-05 | GT opening + GT correspondence | 0.000000 | 0.052658 | 0.032773 |

Oracle scale 分别为 1.024043498、1.033316168；GT scale 均约为 1。
相对 scale error 为 2.40435%、3.33162%。第一条 yaw error 实际为
2.54e-14 deg，第二条为 0（浮点精度），不能据此声称任意房间精确恢复。

相较当前 M2：translation error 降低 85.4513%、87.6540%；log-scale error
降低 90.4080%、72.4870%。绝对 translation 改善 0.513508m、0.373861m。

## 历史对照识别

用户列出的 101.178516deg / 3.420183m / 0.599943 和
78.364440deg / 4.574525m / 1.221219 来自旧目录
`src/output/zind_room_group_debug/t3_best_0665_floor03/pairs/`。
当前主对照来自 `t3_best_0665_floor03_candidate_local_m2/pairs/`。
前者不得标作 current M2；真实历史 JSON 的评价摘录另存 `historical_baseline.json`。
两条当前冻结 baseline 的三项指标均经现有 evaluator 重算，与原 JSON 在 1e-10 容差内一致。

## 十项坐标/求解诊断

| 检查 | 证据与结论 |
| --- | --- |
| 1. endpoint order / 180° ambiguity | 保留标注原始顺序；solver 自行比较 same/reversed，并按预测质心异侧规则选解。分别反转 A、B、两侧后的 transform 最大差为 0 与 2.78e-17。没有 GT pose 选解。 |
| 2. A→B / B→A | 两条均以 room01 为 A，B 分别是 04、05；solver 明确输出 B_to_A，GT 用 inv(T_A_global) @ T_B_global。保持原冻结方向，不将无向边标签当变换方向。 |
| 3. yaw sign / wrap | 二维矩阵采用标准 [cos,-sin;sin,cos]；复用 atan2(R10,R00) 和既有 wrapped evaluator。359° 与 1° 的合成测试得到 2°。 |
| 4. pano / room frame | GT localEndpoints 来自 pano34/37/30 对应的标注节点；预测边界使用 panorama-local pre-Manhattan，调用既有 Bi→ZInD x 取反适配，不使用 Manhattan canonical frame。 |
| 5. normalized units / meters | 求解在 ZInD panorama-A local xy 中完成；米制转换只乘一次 A.floor_plan_transformation.scale × floor03.scale_meters_per_coordinate。原始 local translation 也保留。 |
| 6. repeated scale | 相对 scale 仅由既有 solver 的 width_A/width_B 计算。GT room scale 不进入求解器，仅用于评价米制单位和构造 GT transform。 |
| 7. tangent / normal | tangent=endpoint1-endpoint0；没有额外 normal 输入或法线翻转操作。记录 left normal 只为诊断，不参与选解。 |
| 8. GT/pred same frame | 两者均为 B→A、A=room01/pano34；annotation-derived GT transform 与 manifest transform 在 1e-12 容差内一致。 |
| 9. translation after yaw | 复用 _fit_similarity 的 t=mid_A-sR·mid_B，translation 已在 A frame，不能再次旋转。没有重新实现该公式。 |
| 10. multiple physical correspondence | 复用既有 0.06 global-coordinate 匹配协议，两条各只有一组：[2,0] 和 [0,2]。没有按 GT pose error 选择 opening。 |

两组 GT opening 的局部宽度分别为 0.456640/0.445919 和
0.495124/0.479160，解释了 width-ratio solver 的非单位 scale。
GT floor-global endpoint discrepancy 分别为 0.026543、0.022829 个楼层坐标单位，
即约 0.0960m、0.0826m。因此 GT 配对满足已有物理容差，但不是两个完全相等的线段。
残留厘米级 translation / 非零 scale error 与此不一致性相容；不能通过改成 unit scale
或 GT pose 选 endpoint mapping 来美化结果。本轮不修改几何算法。

## 研究结论

Q1：固定两边的 yaw 误差接近零，translation error 为 8.74cm、5.27cm，
log-scale error 为 0.023759、0.032773。

Q2：相较当前 M2，translation error 分别改善 0.5135m、0.3739m；详见上表及 JSON。
旧 polygon 结果另列，不把已完成的 M2 改善混入 Oracle 收益。

Q3：对“整个系统主要瓶颈”的结论选择 D：两例不足以全局判断。
本实验同时替换 opening geometry 和 correspondence，因此单凭这次干预无法严格区分 A/B。
它说明这两个案例的现有 solver 在正确开口条件下可达到较好精度，不支持 C 是这两例的主要瓶颈。
后续如需区分 A/B，应另行授权固定 GT correspondence、仅改变几何输入的受控对照；
本轮未扩展到完整 val。

## 执行与验证

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/bi_layout_oracle_mpl /home/feixia/anaconda3/envs/bi_layout/bin/python -B tools/eval_zind_gt_opening_pose_oracle.py
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/bi_layout_oracle_test_mpl /home/feixia/anaconda3/envs/bi_layout/bin/python -B -m unittest tests.test_zind_gt_opening_pose_oracle tests.test_candidate_local_portal_geometry tests.test_frame_preserving_manhattan -v
```

入口拒绝覆盖已有输出目录；重现时用 `--output_dir` 指定新的空目录。
上述相关测试 19/19 通过（其中新增 3 项）；git diff --check 通过，正式已跟踪文件无修改。
初次执行在写文件前因 source hash 路径未 resolve 失败；修正后成功，未改变实验算法或协议。
输出包括 per_pair.json、summary.json，另附本人工审计报告及只读提取的 historical_baseline.json。
summary 保存输入 SHA-256、主脚本/solver/evaluator SHA-256，输入在运行末尾重新核对未变。
仅测试两条正边；未运行 04-05、完整 val、test、训练或 GPU/model forward。
未加载或修改任何 checkpoint，未修改 threshold、Matcher、dustbin、verifier 或正式几何模块。
