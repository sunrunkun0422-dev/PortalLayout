# 正确匹配全景图对：0665 / floor_03 / pano 34 与 37

- A：房间01，全景34；B：房间04，全景37。
- 历史T3 best.pt，completed epoch 8；官方train上的定性案例，并非最新ObservablePairs模型或独立测试结果。
- 预测候选A1 ↔ B0，接受匹配；对应分数0.881621（非校准概率）。已有评估：正确候选已召回，GT候选排名1，正负分类正确；所选候选也是实际分数第一。
- GT标注候选索引是A2 ↔ B0，和预测候选A1 ↔ B0属于不同候选集合，索引不能直接比较。正确性按原评估的区间对应映射判定。

## 文件

直接打开index.html可查看全部图片。

- A_pano_34.jpg、B_pano_37.jpg：原始全景，不裁剪、不拉伸。
- A/B_estimated_layout.svg：从已有JSON重新绘制单图预测；蓝色闭合布局、橙色外延布局、红色选中开口。使用pre-Manhattan预测坐标、等比例绘制；不含GT。不将两张独立图的显示缩放解释为相对尺度。
- A/B_prediction.json：完整单图估计及开口概率、候选信息，不含GT。
- joint_layout_clean.svg：已有联合布局SVG的相同多边形、接口与相机标记，仅移除重叠英文标签并重排说明。
- joint_layout_original.png/svg：未改动的原始联合布局；joint_layout_manhattan_original.png为另一种后处理表示。
- pair_prediction_only.json：匹配、B→A变换、预测联合布局等全部模型输出。
- 带original_with_GT字样的JSON含离线GT对照，仅用于核验。

## 对齐结果与边界

B→A旋转约−173.178°，平移[−1.042097, −0.673623]（A局部坐标，不能直接写成米），相对尺度0.780597。位姿由预测开口的candidate-local M2几何得到，非GT对齐。

已有后验评价：旋转误差4.356°；平移误差0.600936米（仅作为该历史记录已有米制评价）；绝对log尺度误差0.247696。匹配正确并不表示拼接无误差。单图与联合图均为估计结果，没有进行新训练、模型推理或用GT替换预测。
