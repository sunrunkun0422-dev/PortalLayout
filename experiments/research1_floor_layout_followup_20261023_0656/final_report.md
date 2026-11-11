# 整层布局工程试点续作报告

本次完成中断训练恢复、辅助训练、几何校准、12 层推理和独立七指标评价。这不是完整数据集的正式基准实验。

校准结果：`calibration_infeasible`。不得通过降低既定门槛、补边或 GT 分组制造已验证的多房间结果。

| 模块 | 结果 | 原因 |
| --- | --- | --- |
| opening | calibration_infeasible | no_threshold_meets_operating_point |
| pose | calibration_infeasible | opening_gate_infeasible |
| room | calibration_infeasible | insufficient_class_or_house_support |

匹配器训练 3000 步，fit probe loss 0.479285 → 0.220171，选择集 loss 0.714426。

- pose：3000 步，fit loss 0.758150 → 0.180744，有效训练 374 / 选择 215。
- room：5000 步，fit loss 0.697617 → 0.000000，有效训练 17 / 选择 4。

| 数据角色 | 层数 | 整体 IoU ↑ | 定位覆盖率 ↑ | 平移误差 cm ↓ | 旋转误差 ° ↓ | Wall F1 ↑ | Junction F1 ↑ | 墙体距离 cm ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fit | 4 | 0.1368 | 0.0000 | N/A | N/A | 0.1139 | 0.1242 | 82.8113 |
| selection | 3 | 0.1508 | 0.0000 | N/A | N/A | 0.0933 | 0.1142 | 74.3147 |
| calibration | 5 | 0.0887 | 0.0000 | N/A | N/A | 0.1320 | 0.1533 | 83.5743 |

上表按角色分别汇总；训练、模型选择、校准集不能充当独立留出测试集。若没有定位的非锚点，位姿误差为 N/A，而不是零误差；孤立主分量的轮廓指标也不代表整层重建成功。

## 完整性和隔离

- 12 层、222 张观测、2033 个无序图像对；每层结果见 `pilot_inference/`。
- 冻结源缓存校验文件数：457；推理不读 GT。
- 真实 20 图隔离检查见 `real20_isolation.json`；恢复一致性和回归日志见 `resume_exact_tests.log`、`resume_regression.log`。

## 尚未完成的正式实验

需要扩大 fit/selection/calibration 的有效监督覆盖，尤其是同房间正负例；保持 house 隔离，重新训练和校准。随后才能完成冻结后的留出集评测、legacy/new matcher 同协议对照及各消融结果。当前试点不能证明达到七项指标的性能目标。

清理建议见 `docs/research1_code_cleanup_audit.md`，本次未删除或合并历史对照代码、数据和权重。

最新完整回归：380 项通过（`resume_regression.log`）。另已逐层核对全部 2,033 个图像对数量与唯一 ID。诊断图见 [校准分数分布](opening_score_diagnostic.png)。
