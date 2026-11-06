# 研究点一扩大数据实验（2026-10-24）

按 docs/research1_execution_plan_20261024.md 执行。全部产物使用新目录，冻结前端与原数据集不覆盖。

- E0：baseline.json 保存启动时源码/配置指纹；cohort_verified.json 确认与计划清单一致。
- E1：60 层、1,551 图、23,942 对；9,572 对有分配监督。preflight.json 为 ready_for_training；sampler_support.json 确认纯无匹配池。缓存无输入错误，3,163 个源文件验证通过。
- E2：200 步；原 fit probe loss 0.3824→0.4635，状态 diagnostic_loss_not_decreased。原探针为 24 正例/232 无匹配。独立固定分层诊断的正例损失 1.4349→1.1399，无匹配 0.3059→0.4172，均衡损失 0.8704→0.7786。有限梯度与权重更新通过；保留原诊断状态。smoke 权重不用于正式训练初始化或校准。
- E3：new 从 seed123 初始化开始正式训练，最多20,000步，500步验证，5次不改善早停。接着运行相同预算的 legacy。
- 当前完整回归为391项通过（regression.log）。新增离线匹配报告支持全弱正对应召回、unknown、同房/跨房分层与 selection PR，不改变训练和推理。

训练及后续校准状态查看 progress.json；最终完成范围以 final_report.md 为准。未校准模型不得视作已验证多房间重建。

## 机器托管

训练由用户级 systemd 服务 `research1-main-20261024.service` 执行，控制脚本为 `control.py`。已启用用户 linger，进程不依赖本次对话；训练有 last.pt 时可按原契约恢复，非训练阶段遇到不完整产物会明确失败并保留文件。

状态以 `controller_state.json` 为准；`research1-main-20261024-monitor.timer` 每5分钟写入 `monitor_status.json`。这属于机器上的状态检查，不代表自动发送聊天通知。日志为各阶段同名 `.log`，控制器异常写入 `controller.log` 和状态文件。

本次托管覆盖 B0/B1、各自全对评价与开口工作点，通过后自动训练 pose、校准并比较森林/PGO；不自动放宽阈值。研究点一结果和断连情况需先汇总，房间应用及消融仍为后续阶段。

检查命令：`systemctl --user status research1-main-20261024.service`。每5分钟检查任务为 `research1-main-20261024-monitor.timer`。全部阶段完成后自动写出 `controller_report.md`。
