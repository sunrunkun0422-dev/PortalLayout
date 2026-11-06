修改记录：2026-10-28 23:49:56（Asia/Shanghai）从冻结原始数据复算证据及执行条件。

# 方法优势证据包

入口：[可视化报告](index.html)。本目录是首轮现成数据取证交付，未运行新训练/推理，也没有修改模型或历史实验。

复算命令（在仓库根目录）：

```bash
/home/feixia/anaconda3/envs/bi_layout/bin/python tools/build_method_advantage_evidence.py
python -m pytest -q tests/test_method_advantage_evidence.py
```

脚本只写入 --output 指定目录。统计文件确定性生成，Markdown 追加修改记录；生成前标记检查状态为 running，失败不能沿用旧 passed。图表为本地 matplotlib 生成的 PNG/SVG。页面引用仓库中的原始案例，迁移目录时应保留相对路径。

- evidence_claims.json：主张、值、分母及适用边界；source_manifest.json：实际读取来源与 SHA256。
- geometry_paired_selection.csv：711 个 R3 对和 203 个弱对应对；geometry_recomputed.csv：九组完整分层结果。
- geometry_contrasts.json / floor_bootstrap.json：2000 次房屋配对区间，开发数据上的探索性统计。
- all_floor_metrics.csv：四组、两种划分、全部 80 行；paired_floor_metrics.csv：20 层排序对照。
- model_provenance.json：冻结匹配器、V0/V1 权重、几何配置、所选步数、工作点状态。
- review_status.json / data_audit.json：人工审核进度、划分、数据身份与未核实事项。
- ablation_configs/：四组配置规范，不是现有训练 CLI 可直接执行的参数；实现缺口及验收条件见核心消融执行表。

下一步需要真实审核者完成原 96 例审核包。计划约定人工审核先于核心实验；不会自动把待审核改成通过。四组独立开关及跨组共享验证器仍需后续实现，未将规范落盘等同于实验完成。

