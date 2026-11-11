> 修改记录：2026-11-05 19:42 CST - 仅将说明链接指向合并档案，保留原实验正文与结果。

# 冻结 R3：位姿验证器实验

**本轮已完成，2026-10-26补齐报告并核验全部产物。** [最终结论](final_report.md) · [素材浏览](presentation/index.html) · [PPT素材包](presentation_bundle.zip)。工作点未通过；仅排序开发验证IoU为29.90%，正确定位覆盖率5.58%。

本轮先核查候选，再训练验证器过滤错拼。仅训练位姿验证器；不训练开口匹配器与room。所有实验边未通过严格校准，结果允许断连。

- [实现说明](../../docs/archive/implementation.md#source-96a8707835)
- [控制器实时状态](controller_state.json) · [五分钟快照](monitor_status.json) · [日志](controller.log)
- [坐标预检](coordinate_preflight.json)：24个合成变换，4497个真实标注投影。
- [全量回归日志](regression.log)：418项通过。
- [最终针对性复测](final_targeted.log)：20项通过。
- [0430导出试运行](integration_probe.log)：126个候选位姿样本，78对。

阶段产物依次生成：

1. `pose_supervision/candidate_audit.json`：fit/selection四级候选上限；GT仅离线诊断。
2. `pose_supervision/supervision.json`、`training_support.json`：独立目标及类别覆盖。
3. `smoke/`、`train/`：200步smoke、最多5000步正式训练，检查点和曲线。
4. `selection_workpoint.json`、`experimental_policy.json`、`development_check.json`：工作点及冻结检查，无可行点不放宽门槛。
5. `comparison/`、`backend_diagnostics.json`、`presentation/index.html`、`final_report.md`：20层四组对照及PPT素材。

后台服务：`research1-pose-verifier-20261025.service`。每五分钟状态检查：`research1-pose-verifier-20261025-monitor.timer`。运行失败时控制器记录原因并停止，不自动进入下一训练任务。旧缓存和R3权重保持只读。

修复后全量419项回归测试通过，见 [回归日志](recovery_regression.log)。

本次运行已结束，不会自动继续训练。原控制器因末尾报告变量重名失败，恢复过程保留在 [修复记录](finalization_recovery.json)；原运行契约不覆盖。
