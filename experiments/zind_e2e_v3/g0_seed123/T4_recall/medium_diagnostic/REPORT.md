# T4 matcher-only 中等规模泛化诊断

结论：中等门禁失败，未启动正式 `T4_recall_matcher_ft`。正式配置对应的
`I1_M1` 虽提高 positive acceptance，但 S3 与相同 Recall-TopK proposal 下的
T3 baseline 同时下降，Harmonic 从 `0.419463` 降至 `0.400295`。

## 状态与隔离

- branch：`agent/zind-bipair-opening-matcher`
- HEAD：`50a6ae5ca403928627fb4de18ae918d59c03d479`
- T3：`ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d`
  / 418836277 bytes / `2026-08-15 02:54:10.604347717 +0800`
- dirty worktree 保留；没有 reset、checkout、clean。
- active splits 始终为 `train,val`；所有新 artifact 都记录
  `test_data_accessed=false`。
- 正式目录 `checkpoints/End_To_End/zind_e2e_v3/g0_seed123/T4_recall_matcher_ft/`
  不存在。

## 32-pair epoch-0 parity

- 旧 `T3 subset baseline` 与 P0/legacy 完全一致，不是 Recall-TopK baseline。
- fresh T3 的 Recall-TopK 前向重复运行，在 opening probability、interval、candidate
  mask/order、candidate feature、score matrix、含双向 dustbin 的 assignment、best
  candidate 与 match/no-match 上为 32/32 逐元素一致，0 mismatch。
- T4 artifact 记录的初始化 checkpoint SHA 与 T3 完全一致；T3/T4 的 Bi-Layout、
  Opening state 完全一致。
- P0 legacy：acceptance `7/16`，S3 `3/16`，negative rejection `12/16`，
  Harmonic `0.300000`。
- P1 Recall-TopK：acceptance `7/16`，S3 `3/16`，negative rejection `11/16`，
  Harmonic `0.294643`。
- Recall-TopK 在小集没有新增 candidate recall，却令负样本
  `1147_floor_01_pano_8_pano_39_negative` 从 no-match 变为 false match。
- T3 P1 到 T4 best：2 个 match→no-match，其中一个是正样本真实退化、一个是负样本
  修复；另有 1 个 S3→non-S3。正样本
  `0003_floor_01_pano_28_pano_31_9` 的 row/column true-to-dustbin margin 从
  `+0.082934` 降到 `-0.111430`，失败类型从 success 变成 dustbin reject。
- 旧 run 只保留 epoch 1 best 与 epoch 5 last，没有逐 epoch matcher state；报告中的
  “首次出现”是 first observable checkpoint epoch=1，不伪造不可恢复的中间轨迹。

## Medium sample 与 cache

- train：1024 pairs = 512 positive + 512 negative，覆盖 1024 个 house。
- val：512 pairs = 256 positive + 256 negative，覆盖全部 157 个 val house。
- train/val house overlap：0。
- 选择规则：SHA256(seed=123, split, ID) 的 house-first round-robin；不使用分数、
  recall、margin 或错误类型。
- pair order SHA：train
  `7e4899d0ca973a21eedec28a7276e25c394193832b527d78304e0ee4224bf6c8`；val
  `4abd87cee4547537e801bbf9d6f2d01ad30b0a488a81a25b17e48d1758766f2c`。
- T3 upstream 只计算一次；cache 约 2.0 GiB、train 256 shards、val 128 shards。
- cache schema content SHA：
  `2be48272c7dcc62421be77b5bee717de81fdab47f334c9d80b8deb2191eb4fab`。
- val cache 只保留 `legacy_predicted` 与 `recall_predicted`。一次 schema 收尾审计删除了
  128 个 val shard 中从未消费的 injected diagnostic view；所有 predicted-view tensor
  signature 不变，train shard hash 不变，四格结果不变。

## Recall-TopK 独立影响（medium val）

| 指标 | P0 legacy | P1 Recall-TopK |
|---|---:|---:|
| candidate recall ceiling | 151/256 = 0.589844 | 155/256 = 0.605469 |
| candidate count mean / median / p95 | 2.364 / 2 / 6 | 3.038 / 2 / 8 |
| K_A×K_B mean / median / p95 | 5.504 / 4 / 15 | 9.092 / 5 / 32.45 |
| positive acceptance | 107/256 = 0.417969 | 110/256 = 0.429688 |
| conditional Top-1 | 0.450331 | 0.445161 |
| end-to-end S3 | 68/256 = 0.265625 | 69/256 = 0.269531 |
| negative rejection | 243/256 = 0.949219 | 242/256 = 0.945313 |
| precision / recall / F1 | 0.8917 / 0.4180 / 0.5691 | 0.8871 / 0.4297 / 0.5789 |
| Harmonic | 0.415092 | 0.419463 |

pair-level 为 1 win / 510 tie / 1 loss。新增的 4 个 recalled positive 全部没有转化为
S3；没有 legacy S3 因新增候选而直接丢失。Recall-TopK 的总体 S3 +1 来自既有 recalled
样本的选择变化，同时付出一个 negative rejection。

## 2×2 因子实验

所有格从同一 T3 初始化，使用完全相同的 6 个 epoch order hash、AdamW、LR `3e-5`、
weight decay `1e-4`、cosine scheduler、batch 4、seed 123；Matcher 为 train state，
Bi-Layout/Opening frozen+eval，val 只使用 `recall_predicted`，val injection count=0。

| 因子 | Injection | Margin | Best epoch | Train S3@best | Val acceptance | Val S3 | Neg reject | Harmonic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I0_M0 | off | off | 1 | 0.535156 | 0.339844 | 0.230469 | 0.957031 | 0.371479 |
| I0_M1 | off | on | 1 | 0.539063 | 0.355469 | 0.230469 | 0.957031 | 0.371479 |
| I1_M0 | on | off | 4 | 0.730469 | 0.402344 | 0.253906 | 0.945313 | 0.400295 |
| I1_M1 | on | on | 4 | 0.771484 | 0.472656 | 0.253906 | 0.945313 | 0.400295 |

所有 epoch/loss/gradient finite；Bi-Layout 与 Opening gradient tensor 均为 0；Matcher
14/14 gradient tensor finite 且非零；四格 upstream hash 均不变。

## 原因判定

1. **Recall-TopK 增加干扰候选：成立，但不是主因。** medium 中 ceiling 只增加
   4/256，新增 recall 的 S3 转化为 0；候选乘积均值增加 65%。小集还直接造成一个
   negative false match。
2. **GT injection 造成分布差异：明确成立。** train predicted-only 仅有
   310/512 target-present；injection 将其变为 512/512，救回 202 个 positive，并注入
   209 个 candidate。val 保持 predicted-only 155/256。Injection 相对 no-injection
   确实改善 val S3 6/256 和 Harmonic 0.0288，所以它不是单独的负效应；但它把 train
   S3/val S3 gap 从约 0.31 扩大到约 0.52，是最明显的训练/验证分布错位。
3. **dustbin margin 引起校准偏移：成立。** 无 injection 时 margin 令 acceptance +4，
   有 injection 时 +18；两种情况下 S3、negative rejection 和 Harmonic 均完全不变。
   它主要把 positive 从 dustbin 推向 match，而没有改善 candidate selection。
4. **24-pair 太小：成立但不充分。** 扩到 1024 train 后四格仍表现出明显过拟合；即使
   I0_M0，train S3 0.535 对 val 0.230。问题不是只由 24-pair 的随机波动造成。
5. **epoch-0 baseline proposal 不一致：成立。** 旧 baseline 用 legacy。改为相同
   Recall-TopK 后，小集 baseline Harmonic 只从 0.3000 改为 0.2946，不能解释 T4
   0.2143 的大部分下降；medium 比较已完全同 proposal。

综合判断：主要失败边界是 **Matcher 在 predicted candidates 上的跨 house candidate
selection 泛化**。Train-only injection 缓解缺 target 的训练困难，但制造了 100% vs
60.5% target-present 的条件差异；margin 又提升了 acceptance 而没有提升 S3。因此当前
配置更会“敢于匹配”，却没有更会“选对候选”。

## 门禁与启动决定

正式配置 I1_M1 相对同-proposal T3：

- acceptance：110/256 → 121/256（+11）；
- S3：69/256 → 65/256（-4）；
- conditional Top-1：0.445161 → 0.419355；
- negative rejection：242/256 → 242/256；
- Harmonic：0.419463 → 0.400295。

沿用上一轮 GPU small gate 的最低 non-regression 条件，
`harmonic_not_below_same_proposal_t3=false`。决定为 `STOP_MEDIUM_GATE`，正式训练未启动，
无 PID、无 `.running/.completed/.failed`，也没有正式日志或 checkpoint。

## 验证

- `python -m unittest discover -s tests -p 'test_*.py'`：194/194 PASS。
- `git diff --check`：PASS。
- 全部 medium JSON 可被 `jq` 解析，未发现 NaN/Infinity。
- T0/T1/T2/T3 SHA、size、mtime 与审计前一致。
- GPU 结束状态：无 compute process，显存回落到约 712 MiB。
- 磁盘结束状态：382 GiB available。
