# Frozen T3/M2 full ZInD val diagnostic

## 1. 一句话结论

M2 在正确候选的 S3 子集上显著改善 yaw、translation、scale 与 BEV，但 72.58% 的 GT positive 在进入正确候选几何前已因 candidate recall、dustbin 或选错候选丢失，因此端到端第一瓶颈仍是 matcher/candidate recall。

## 2. 数据规模和 house 分布

- S0: 3630 pairs；S1: 1827 positives；negative: 1803。
- val houses: 157；train houses: 1260；交叉: 0。
- S3: 501 pairs / 139 houses。

## 3. 代码、配置、checkpoint 和 manifest hashes

- branch/HEAD: `agent/zind-bipair-opening-matcher` / `50a6ae5ca403928627fb4de18ae918d59c03d479`；运行时 worktree 为 dirty，已有修改已保留。
- inference evaluator snapshot: `c1da154d817b07870587431a378cd299f7e313da1ca2f29bb54cd9672a6754c0`；current source after report-only formatting: `42ee45cbd1ab7fd3a5c48863c87c019020c282c24f680444401ce0c927736ae6`。
- T3: `ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d` (`/home/feixia/pythonProject/AAAsrk/Bi_Layout/checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best.pt`)。
- M2 config: `cdce025fd5ba709a3c5afc827137860d1082c7ab44e27a5467ea6686a7a02a4c`。
- Bi-Layout config: `/home/feixia/pythonProject/AAAsrk/Bi_Layout/src/config/zind_all.yaml` / `867506b93ae973b3517d20a9b2e9899964684366b3644d1facc22110e1382cbd`。
- full manifest: `7d2d9412f2fda7c8238d8241f72e9f2162521917e85b4bed6d25ff28899087a9`。
- val manifest: `acae7ecbd55d4f8b8aea7c7a8db4b8cbb434fe92992858ac0a4d4a26da7da1ec`。
- original metrics: `75e410007556eb61fdf1ba2e8121b6c4891a6505ba65f22e75622ad86b3520d3`。
- opening threshold: 0.47，来源为 T3 checkpoint。

## 4. test 未访问证据

- split 固定为 `val`，CLI 不提供 test split；输出声明 `test_data_accessed=false`。
- 只打开 train manifest 做 house-disjoint 检查；未打开 test manifest、图像或 cache。

## 5. 运行命令、时间和恢复次数

- `python tools/evaluate_zind_portal_geometry_val_full.py --device cuda:0 --progress_every 25`
- 记录运行时间: 70.54 min；GPU process invocations: 1；恢复次数: 0。

## 6. 完成率和 forward 唯一性

- completed: 3630/3630。
- frozen forward count: 3630；每 pair 均为 1。
- candidate score/probability branch max difference 均为 0；best candidate 一致率 100%。

## 7. T3 matcher evaluator parity

- parity: PASS；容差 1e-09。
- candidate recall、positive acceptance、conditional/end-to-end Top-1、negative rejection、Balanced、Harmonic 的绝对差全部为 0。
- Harmonic: 0.423333166661。

## 8. S0–S5 样本量

- S0=3630，S1=1827，S2=806，S3=501。
- S4 pose-valid: G0=501，G1=501，G2=497；S5=130。

## 9. Matcher/candidate 指标

- TP/TN/FP/FN: 806/1673/130/1021。
- precision=86.11%，recall=44.12%，F1=58.34%，negative rejection=92.79%。
- candidate recall=59.11%，conditional Top-1=46.39%，end-to-end Top-1=27.42%。
- Balanced=0.601059，Harmonic=0.423333。

## 10. G0/G1/G2 geometry（共同 S3）

| mode | valid | yaw median/mean | translation median/mean (m) | |log-scale| median/mean |
| --- | ---: | ---: | ---: | ---: |
| legacy_polygon | 501/501 | 26.29/44.59 | 1.696/2.772 | 0.475/0.638 |
| frame_polygon | 501/501 | 33.24/48.59 | 1.760/2.822 | 0.542/0.692 |
| candidate_local_m2 | 497/501 | 9.34/28.29 | 0.719/1.467 | 0.184/0.406 |

M2−G1 paired W/T/L: yaw 350/0/147；translation(m) 354/0/82；log-scale 342/0/155。

## 11. End-to-end 与 conditional pose

- legacy_polygon: yaw<15° & translation<1m，S1=4.11%，S3=14.37%。
- frame_polygon: yaw<15° & translation<1m，S1=2.96%，S3=10.38%。
- candidate_local_m2: yaw<15° & translation<1m，S1=12.32%，S3=44.11%。

## 12. D0/D1/D2 scale 诊断

- 子集: S3 correct-candidate pairs with valid formal M2 pose，n=497；正式 G2 未改变。
- D0 translation mean/median=1.467/0.719m；frame/raw BEV mean=0.681/0.772。
- D1 unit translation mean/median=1.109/0.707m；frame/raw BEV mean=0.688/0.776。
- D2 oracle translation mean/median=1.109/0.707m；frame/raw BEV mean=0.688/0.776。
- D1 明显优于 D0，而 D2 与 D1 几乎相同；但 S3 中 500/501 为近 unit-scale，不能据此把 unit scale 覆盖到一般非单位样本。

## 13. 双 polygon BEV

- frame_polygon: frame mean/median=0.624/0.642，raw mean/median=0.644/0.603；invalid=0/0。
- candidate_local_m2: frame mean/median=0.681/0.691，raw mean/median=0.772/0.849；invalid=0/1。

## 14. M2 hard/soft quality

- all predicted-match hard failures: 9/936 (0.96%)。
- soft-fail: 685/936 (73.18%)；RANSAC/TLS sides=614/998。
- soft-fail 最主要原因是 angular dispersion；quality/error correlation 仅作事后描述，没有用于校准 gate。

## 15. House-clustered bootstrap

- S3 houses=139，seed=123，resamples=10000。
- yaw M2−G1 median delta=-11.08°，95% CI [-15.92, -6.21]。
- translation median delta=-0.839m，95% CI [-0.980, -0.738]。
- log-scale median delta=-0.249，95% CI [-0.345, -0.170]。
- frame BEV median delta=0.050，95% CI [0.035, 0.066]。

## 16. 成功和失败类型

- candidate 未召回: 747/1827；召回后 no-match: 437/1080；召回后选错: 142/1080。
- M2 S3 hard failure: 4/501；valid 但 medium-pose 失败: 276。

## 17. M2 得到支持的部分

- yaw 泛化成立：501 个 S3 pair、139 houses，paired median 和 mean 均改善，house bootstrap 区间不跨 0。
- conditional overall pose 也成立：translation、log-scale、BEV 与三档 joint success 均改善；但这不等于端到端已解决。

## 18. 当前系统第一瓶颈

matcher/candidate recall。只有 501/1827（27.42%）GT positive 到达正确候选 S3；747 未召回、437 召回后被 dustbin/no-match 拒绝、142 召回后选错。

## 19. 下一阶段唯一优先方向

优先改 matcher recall（含 predicted candidate recall 与 dustbin positive acceptance），而不是先改 scale。scale/unit-prior 是明确的第二阶段诊断方向，但当前影响的样本远少于进入正确候选前的损失。

## 20. 尚未执行的内容

- 未访问 test，未训练，未调 matcher threshold，未校准 M2 gate。
- 未把 D1/D2 写回正式 G2，未实现 clamp/shrinkage/learned scale、M4、SALVe 或 BEV verifier。
- 本结果为 val-only 系统诊断，不作为论文最终 test 统计。
