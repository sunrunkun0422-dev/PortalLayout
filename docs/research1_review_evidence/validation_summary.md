# 本轮实际验证摘要

日期：2026-10-21；工作区 HEAD：`8484b620d1850a27776445178c7f0f2e753213ac`。本轮仅新增审查、设计与验证产物，未修改模型和训练算法，未启动新的真实数据全量训练。

## 1. 回归与环境

- `/home/feixia/anaconda3/envs/bi_layout/bin/python -m unittest discover -s tests`：**260 项，23.441 秒，OK**。
- 使用 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`。
- 完整记录：[unittest.log](unittest.log)。测试内的小型临时训练不是真实全量训练。
- 当前 CUDA 可用，GPU 为 RTX 4060 Ti 16GB。本次真实预测用 CPU；没有测新架构的 GPU 峰值显存。
- 已验证旧训练和双房间几何入口的 `--help`，见 [train_help.txt](train_help.txt)、[pair_help.txt](pair_help.txt)。

## 2. 当前权重的真实三房间样例

输入：现有 `debug_train_0665_floor_03_rooms_04_01_05`，三个 complete-room 代表 panorama，train-only。使用原始 RGB 实时前向，非 teacher-cache 预测。先 dry-run，再真实预测。

权重：`checkpoints/End_To_End/zind_observable_v1/best1_balanced_seed123_20260910/best.pt`，最后加载覆盖模块化初始化，completedEpoch=14。

参数：`layout_frame_mode=panorama_preserving`、`portal_geometry_mode=candidate_local_m2`、`device=cpu`、opening threshold=0.47，来源为端到端 checkpoint。此处 M2 是现有路径核验，不代表其优于 raw endpoints。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/feixia/anaconda3/envs/bi_layout/bin/python tools/evaluate_zind_room_group.py \
  --device cpu \
  --end_to_end_checkpoint checkpoints/End_To_End/zind_observable_v1/best1_balanced_seed123_20260910/best.pt \
  --layout_frame_mode panorama_preserving \
  --portal_geometry_mode candidate_local_m2 \
  --output_dir docs/research1_review_evidence/three_room_prediction
```

实际输出：

| Pair | GT 直接连接 | 预测 | 位姿 |
| --- | --- | --- | --- |
| room01/pano34 – room04/pano37 | 是 | 拒绝；正确候选存在，GT rank=2 | null，不做融合 |
| room01/pano34 – room05/pano30 | 是 | 接受 A0–B2；GT candidate rank=1 | 有输出；事后 yaw 34.5619°、translation 1.20127m |
| room04/pano37 – room05/pano30 | 否 | 拒绝 | null，不做融合 |

结果：TP=1、TN=1、FP=0、FN=1，生成一个双房间 joint BEV。每个房间单图前向次数均为 1，共评估 3 对。脚本记录运行耗时 **16.285 秒**，为该脚本计时范围的 CPU 时间，不代表新架构性能。

`pose_valid=True` 在现有脚本里说明算法产生有效变换，**不等于达到精度阈值**；该接受边的真实误差说明位姿质量仍是关键瓶颈。对于被拒绝的正边，没有用 GT 强制补边。

证据：

- [group_summary.json](three_room_prediction/group_summary.json)：次数、邻接、分类、配置、权重加载、计时。
- [pairs.json](three_room_prediction/pairs.json)：逐 pair 预测和事后 GT 评价。
- [adjacency_pred.json](three_room_prediction/adjacency_pred.json)：预测直接图，room_order 明确。
- [overview.html](three_room_prediction/overview.html)：已有三房间调试可视化；包含 GT 对照，不是无 GT 楼层产品界面。
- [真实预测日志](three_room_prediction.log)、[dry-run 日志](three_room_dry_run.log)。

只执行了固定 train 样例，没有访问正式 test。GT 通过现有评估脚本在预测后用于计分和可视化；这不等于脚本的 manifest 已经支持无标注、任意 N 推理。

## 3. 没有验证的项目

- 新二维候选表征、候选对检索、Sinkhorn 及其学习效果。
- 任意 N 的统一楼层坐标、分量、鲁棒 PGO 和七指标实现。
- 正式 ObservablePairs val/test、外部方法、完整楼层泛化。
- 研究点二的受约束局部修正。

因此不能将此样例的一个 joint BEV、260 项回归或已有 60.37% 内部匹配成绩，写成“多房间系统完成”或最终七指标成绩。
