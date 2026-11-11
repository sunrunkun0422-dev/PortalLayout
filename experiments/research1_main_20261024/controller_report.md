# 本机自动执行结果

完成时间：2026-11-05T15:04:02.898060+08:00

执行范围：new/legacy 正式训练、完整全对匹配评价、开口校准；通过门槛后执行 pose 训练和森林/PGO 定位评价。
房间模块及消融不自动开始，先检查研究点一的全正例召回、跨房间质量与定位结果。

- new：stopped_opening_calibration_infeasible
- legacy：stopped_pose_calibration_infeasible

机器状态：controller_state.json；每5分钟检查：monitor_status.json。
匹配报告分别保存于 new_matching_report.json / legacy_matching_report.json。
这些房屋属于内部 fit/selection/calibration；尚非独立留出评测。
