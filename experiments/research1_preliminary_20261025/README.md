> 修改记录：2026-11-05 19:42 CST - 仅将说明链接指向合并档案，保留原实验正文与结果。

# 整层布局初步实验（2026-10-25）

**未经严格校准，可能存在明显错拼；完整推理不代表已完成联合端到端训练。**

- [实现说明](../../docs/archive/implementation.md#source-c292b6f473)
- [第一批三层指标](preview_metrics/metrics.csv)
- [三层完成记录](preview_ready.json)
- [实时状态](monitor_status.json)
- [最终报告与七指标](final_report.md)
- [素材浏览页](presentation/index.html) · [PPT 素材包](presentation_bundle.zip)
- [完成验收](acceptance.json) · [优化回退诊断](backend_diagnostics.json)

首批结果目录：

| 房屋 | 观测数 | 布局图 | 独立 GT 对照 |
| --- | ---: | --- | --- |
| 0430 | 13 | [PNG](predictions/exploratory_t50_robust_rooms/floor_a71328a43f233a1478b275ac/floor_bev.png) | [PNG](predictions/exploratory_t50_robust_rooms/floor_a71328a43f233a1478b275ac/gt_comparison.png) |
| 1236 | 14 | [PNG](predictions/exploratory_t50_robust_rooms/floor_613a83e984e35bcd8b75a433/floor_bev.png) | [PNG](predictions/exploratory_t50_robust_rooms/floor_613a83e984e35bcd8b75a433/gt_comparison.png) |
| 0641 | 18 | [PNG](predictions/exploratory_t50_robust_rooms/floor_70fdcbd54a8bf40ef0801e8b/floor_bev.png) | [PNG](predictions/exploratory_t50_robust_rooms/floor_70fdcbd54a8bf40ef0801e8b/gt_comparison.png) |

三层平均 IoU 为 15.36%、定位覆盖率为 1.96%；平移 pooled median 731.78 cm。所有观测连成一分量仍伴随较大位姿误差，不能将连通率当作定位覆盖率。

全部实验已完成，selection 选定阈值 0.7，完成 20 层的严格参考、森林、PGO、PGO＋几何融合对照。每组保留所有 585 张全景和 9,400 对。开发验证集完整系统 IoU 19.30%、正确定位覆盖率 1.75%，效果仍较差；PGO 有回退且未带来稳定提升。`presentation/` 提供流程图、指标表及按 selection 预定规则选取的较好/中位/失败案例，均为 SVG/PNG。上方首批三层是阈值 0.5 的历史预览，最终比较请看最终报告。
