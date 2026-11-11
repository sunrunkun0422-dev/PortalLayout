# 研究点一第二轮实验结果

完成时间：2026-10-25T01:23:47.999400+08:00

| 组别 | 目标 | 步数 | 所选步数 | 跨房间房屋宏平均 AP | 整体 AP |
| --- | --- | ---: | ---: | ---: | ---: |
| R1 | assignment | 4000 | 1500 | 0.1397 | 0.1100 |
| R2 | positive_margin | 6000 | 4000 | 0.2014 | 0.1317 |
| R3 | known_cells | 6000 | 5000 | 0.2251 | 0.2309 |
| R4 | known_cells | 6000 | 3500 | 0.1408 | 0.0717 |

selection 选出的 new 组：R3。

- R3：stopped_opening_quality_gate
- R4：stopped_opening_quality_gate

原始数据和缓存保持冻结；unknown 未转为负例。
人工结论须由用户填写 review/review_template.json，不能将自动审计计为人工通过。
当前仍是内部开发集实验，不是独立留出结果。room、多 seed 与表征消融未启动。
