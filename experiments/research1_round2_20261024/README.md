> 修改记录：2026-11-05 19:42 CST - 仅将说明链接指向合并档案，保留原实验正文与结果。

# 研究点一第二轮

实现与操作说明：[第二轮实施文档](../../docs/archive/implementation.md#source-0c8ca3bd96)。

本目录保留独立实验、权重和服务标识，不覆盖上一轮。R1–R4 顺序由 `control.py` 托管，主服务 `research1-round2-20261024.service`，每五分钟状态写入 `monitor_status.json`。实时阶段以 `controller_state.json` 为准；四组结果尚未完成时不能把 smoke 视为效果验证。

- `run_contract.json`：冻结代码、缓存、弱监督和 house 协议。
- `pretraining_acceptance.json`：自动审计、回归与两种 200 步 smoke 的启动依据。
- `baseline_matching.json`：上一轮 new 权重在同口径下的 fit/selection 诊断。
- [人工核查包](review/index.html)：96 例，结论待人工填写 [独立模板](review/review_template.json)。
- `R1`–`R4`：正式组的质量最优、assignment 最优、恢复权重和历史。
- `comparison.json`、`final_report.md`：完成后生成；只有通过严格开口门槛才追加 pose、森林及 PGO。

smoke 均衡损失下降、梯度及参数更新正常；原 assignment probe 未下降的诊断原样保留。unknown 未转为负例，人工核查未被自动标为通过。
