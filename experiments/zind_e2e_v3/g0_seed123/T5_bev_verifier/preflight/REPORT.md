# SALVe-style candidate-conditioned verifier preflight

## 1. 一句话结论

Phase A和Phase B均通过硬门禁；E1将完整val S3从501提高到526、Harmonic从0.423333提高到0.438200，同时negative rejection仅下降1.11个百分点。已实现predicted-only raster原型，但未训练CNN；Phase A消融仍表明独立几何排序贡献弱，不能把全部收益归因于BEV geometry。

## 2. branch、HEAD和worktree

- branch `agent/zind-bipair-opening-matcher`；HEAD `50a6ae5ca403928627fb4de18ae918d59c03d479`；dirty worktree原样保留。
- GPU：cuda:0冻结推理smoke `True`，peak allocated `636.5 MiB`；开始前磁盘剩余 `372.4 GiB`。

## 3. Top-N覆盖率及固定N

- train Top-1/3/5/8：1848/1985、1941/1985、1968/1985、1975/1985。
- val Top-1/3/5/8：729/1080、964/1080、1034/1080、1059/1080。
- 仅依据train固定N=`3`。

## 4. predicted-only feature schema

- base=91，missing masks=91，输入维度=182；GT feature=0；坐标契约为predicted B→A / ZInD pano-A local xy。

## 5. M2 geometry valid/hard/soft failure统计

- train: valid=5607 hard=50 soft=3810。
- val: valid=2990 hard=26 soft=2003。

## 6. Phase A V0/V1/V2结果

| 模型 | strict Top-1 | physical-compatible Top-1 | MRR | mean rank | 原501保留率 | Gate |
|---|---:|---:|---:|---:|---:|---:|
| V0_T3_final | 500/1080 | 501/1080 | 0.7748 | 1.5935 | 1.0000 | baseline |
| V1_logistic | 726/1080 | 728/1080 | 0.7745 | 1.5898 | 1.0000 | True |
| V2_mlp | 723/1080 | 725/1080 | 0.7702 | 1.6102 | 0.9960 | True |

## 7. wrong-selection恢复情况

- V1=2/142；V2=3/142。主要提升来自移除positive dustbin reject，而非候选内几何重排。

## 8. Phase A门禁

- passed=`True`；V1/V2逐项门禁全通过。该门禁相对的是dustbin-aware final T3=501，不等价于击败raw real-real score。

## 9. Phase B端到端结果

| 模型 | S3 | Harmonic | Negative rejection | Positive recall | 原501保留 | Gate |
|---|---:|---:|---:|---:|---:|---:|
| E0 T3 | 501/1827 | 0.423333 | 92.7898% | 44.1160% | 100.00% | baseline |
| E1 Logistic | 526/1827 | 0.438200 | 91.6805% | 47.0170% | 97.2056% | True |
| E2 MLP | 487/1827 | 0.414693 | 93.3444% | 40.0657% | 95.2096% | False |

- E1通过全部硬门禁：S3 +25，Harmonic +0.014867，negative rejection下降1.1093%；candidate ceiling=1080/1827，precision/recall/F1=85.1338%/47.0170%/60.5783%，Balanced Accuracy=0.693488；E2失败。

## 10. dustbin恢复和negative false positive

- Phase A V1/V2 dustbin恢复=225/223。
- Phase B E1 dustbin恢复=31，wrong-selection恢复=8，negative FP=150。

## 11. A0-A4消融

- physical-compatible Top-1：{'A0_T3_score': 731, 'A1_M2': 419, 'A2_BEV': 455, 'A3_T3_M2': 727, 'A4_full': 729}。A4比A0少2例，当前scalar geometry没有提供独立排序增益。

## 12. house-bootstrap

- V1={'ci95_high': 0.2363125491223474, 'ci95_low': 0.1832834959605498, 'house_count': 155, 'observed': 0.2101851851851852, 'replicates': 2000}；house win/tie/loss={'loss': 0, 'tie': 36, 'win': 119}。
- V2={'ci95_high': 0.23398576466046966, 'ci95_low': 0.17887721922984035, 'house_count': 155, 'observed': 0.2074074074074074, 'replicates': 2000}；house win/tie/loss={'loss': 1, 'tie': 37, 'win': 117}。
- Phase B E1={'ci95_high': 0.022513139319918003, 'ci95_low': 0.004944851500666092, 'house_count': 157, 'observed': 0.013683634373289545, 'replicates': 2000}；E2={'ci95_high': -0.0016101552610504025, 'ci95_low': -0.01376008292745337, 'house_count': 157, 'observed': -0.007662835249042145, 'replicates': 2000}。

## 13. 是否支持SALVe风格verifier

- 系统级轻量verifier假设获得支持：E1完整val同时提高S3/Harmonic并满足negative/retention/bootstrap门禁。
- 几何独立归因仍不充分：Phase A A0=731、A3=727、A4=729；E1包含T3+M2+relative+BEV，尚未证明Phase B增益单独来自BEV。

## 14. 是否允许进入栅格BEV CNN

- Phase A+B已通过，因此按授权实现了7-channel raster原型：grid=128，范围=±6.0个predicted portal width，examples=8，mean/max clipping=0.07183383929808967/0.2486762285560352。
- CNN训练仍未授权且未启动。

## 15. 测试、安全和GT隔离

- active splits=train,val；test_data_accessed=false；T0–T3 checkpoint、M2、正式proposal与T4均未修改；optimizer只属于CPU轻量V1/V2/A1–A4/E1/E2。
- val未用于标准化、early stopping、阈值或模型选择；所有JSON使用strict finite序列化。
- 新verifier/raster测试11/11、相关旧回归45/45通过；`git diff --check`通过。

## 16. 下一阶段最小建议

- 保持E1与raster配置冻结；等待用户确认是否设计小规模CNN训练，不自动训练。
