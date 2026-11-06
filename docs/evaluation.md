> 修改记录：2026-11-05 19:40 CST - 按计算机视觉工程分层重构，合并当前指南并建立历史来源映射。

# 评估口径与实验入口

模型推理和 GT 评估分开进行：

```text
python -m portallayout floor run evaluate --dataset <楼层数据目录> --predictions <推理目录> --output <新评估目录>
```

指标实现为 `portallayout/evaluation/floor_metrics.py`，协议对象为 `FloorMetricProtocol`。常用七指标为整体 IoU、正确定位覆盖率、平移误差、旋转误差、Wall F1、Junction F1、墙距。尺度轨道通过 `--scale-track S/M` 选择；缺失米制尺度时不能编造厘米指标。

只按协议进行锚点规范，不逐房/逐分量对齐 GT 消除真实误差。保留失败、缺失、断连样本，输出总分母及各指标有效样本数。无有效误差样本时使用 N/A，而不是零误差。

## 不同层次分别报告

| 层次 | 解释与限制 |
| --- | --- |
| 候选可解性 | 给定候选能否包含可行解，不能代表选择器选对 |
| 开口对应 | 同候选上的已知标签、unknown、召回与误接受 |
| 相对位姿 | 匹配是否正确与位姿是否准确分别计数 |
| 楼层重建 | 定位覆盖、布局/墙体质量及失败分母 |
| 人工审核 | 正确、不确定、待审核分别保留 |

独立诊断用 `openings report`；`evidence build` 复算固定历史产物，不执行新训练，也不是正式 test 评估入口。

## 已有证据位置

- [ObservablePairs 训练记录](../experiments/zind_observable_v1/README.md)
- [第二轮开口训练](../experiments/research1_round2_20261024/README.md)
- [初步楼层实验](../experiments/research1_preliminary_20261025/final_report.md)
- [位姿验证器对照](../experiments/research1_pose_verifier_20261025/final_report.md)
- [冻结几何实验](../experiments/research1_pose_geometry_20261026_0245/final_report.md)
- [方法优势证据包](method_advantage_evidence_20261028/README.md)
- [SALVe 原复现记录](../experiments/salve_reproduction_20261028/README.md) 与 [DAV2 适配记录](../experiments/salve_dav2_20261029/README.md)

这些入口保留历史实验身份。原 SALVe 与 DAV2 适配不能视为相同方法；优势判断需要同房间、同候选与可比协议。当前文档整理不新增性能结论。
