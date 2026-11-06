> 修改记录：2026-11-06 00:59 CST — 修正位姿标签灰区统计并记录固定分母。

# R02：GT 开口位姿上限（灰区修正版）

在 R01 固定队列的 40 个 fit 和 10 个 selection 房屋上，使用标注开口和真实相对位姿做离线候选上限检查；不读取 calibration、audit 或官方测试集。这不是模型预测或人工确认。

50 层共 4,757 条非歧义标注开口关联。selection 的 265 条跨房关联中，**213 条有合格候选、10 条为确定失败、42 条处于标签灰区**，覆盖 8 个有跨房关联的房屋。unknown 保留在 265 的固定分母中，不能当作负样本。所有链接均为几何代理，人工确认数为 0。

合格条件沿用 `pose_target`：正反向平移误差均不超过 0.25 个相机高度单位、旋转误差不超过 10°、相对尺度误差不超过 10%；不能解释为 50 cm 米制验收。4,251 个同源标注投影检查的最大数值误差为 0，但它不是独立传感器标定。

[逐层汇总和来源哈希](report.json) 与 `links.jsonl` 保留全部结果。预测开口和位姿消歧的分解尚未完成；[v2](../R02_gt_oracle_v2/README.md) 原样保留并标记修正。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/feixia/anaconda3/envs/bi_layout/bin/python tools/research10_gt_pose_oracle.py \
  --cohort experiments/portallayout_design10_20261105/R01/cohort.json \
  --output experiments/portallayout_design10_20261105/R02_gt_oracle_new
```
