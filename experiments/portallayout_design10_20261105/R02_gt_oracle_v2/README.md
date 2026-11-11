> 修改记录：2026-11-06 00:59 CST — 标记灰区统计修正，保留旧版结果与记录。

> 修改记录：2026-11-06 00:48 CST — 记录新开发队列的 GT 开口位姿上限及评估限制。

# R02：GT 开口位姿上限（离线诊断）

**本版的失败／未知分类已被 [v3 修正](../R02_gt_oracle_v3/README.md)：42 个灰区关联不应视为确定失败；213/265 的合格候选数和固定分母不变。**

针对 R01 固定队列的 40 个 fit 和 10 个 selection 房屋，共 50 层，使用标注开口及真实相对位姿进行离线上限分析。仅用于检查几何条件下的候选可解性，不是模型预测、人工确认或正式测试结果；未读取 calibration、audit 或官方测试集。

- 50 层共 4,757 条非歧义标注开口关联；逐条结果、失败原因及数据哈希见 [report.json](report.json) 和 `links.jsonl`。单位是开口关联，不是去重的全景图对。
- selection 中跨房间共有 265 条关联，213 条含合格位姿候选、52 条不含，覆盖 8 个有跨房关联的房屋；其余 2 个 selection 房屋没有计入这一条件分母。关联标签均为几何代理，人工确认数为 0。
- 合格定义沿用现有 `pose_target`：正反向平移误差均不超过 0.25 个相机高度单位、旋转误差不超过 10°、相对尺度误差不超过 10%；它不是 50 cm 米制验收。
- 4,251 个标注投影检查通过，最大数值误差为 0。被核对的投影使用相关联的同源标注，不能作为独立传感器标定证明。
- 对历史固定的 711 个检索对，原几何、仅单位尺度先验、完整方案的可解数分别为 30、80、81；新旧队列和不同分母不能混为一谈。历史结果复算见 [R02/R03 诊断](../R02_R03_historical/diagnostic.json)。

本轮尚未对新队列完成预测开口、匹配与位姿消歧分解，R02 和 R03 整体仍为进行中。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/feixia/anaconda3/envs/bi_layout/bin/python tools/research10_gt_pose_oracle.py \
  --cohort experiments/portallayout_design10_20261105/R01/cohort.json \
  --output experiments/portallayout_design10_20261105/R02_gt_oracle_new
```
