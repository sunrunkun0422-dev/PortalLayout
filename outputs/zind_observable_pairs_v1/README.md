# ZInD-ObservablePairs-v1 交付入口

正式数据快照：`src/dataset/ZInD-ObservablePairs-v1/builds/train_auto_v2`。

1. [构建报告与复现命令](../../../ZInD-ObservablePairs-v1构建报告.md)
2. [第一阶段盲审 ZIP](review_pilot_final/stage1_package.zip) / [直接打开页面](review_pilot_final/stage1_blind/index.html)
3. [第二阶段 GT ZIP](review_pilot_final/stage2_package.zip)：第一阶段完整提交归档后再发放。
4. [GT 核验图册](figures/index.html)：10 张真实案例图和统计图，不提前提供给盲审者。
5. [索引/图像全量审计](audit/audit.json) / [坐标对照](audit/coordinate_contract.json) / [真人审核状态](audit/human_review_status.json)

251,650 对 = 41,169 几何 positive + 98,299 negative + 112,182 unknown。35,659 对通过双侧几何观测代理。没有已完成真人审核，`core_verified=0`，val/test 等待训练校准冻结。没有运行 Matcher、模型前向或训练。

`review_pilot`、`review_pilot_v2` 为本次开发中间材料，正式使用 `review_pilot_final`。`provenance` 保存修复重复裸 pano 编号前的源代码，仅用于追溯，不作为可运行入口。原 BiPair 数据及旧实验未更改。
