# Matcher 阶段汇报 · 收紧主线与 Oracle Pose

保留原白底、暗红、字体和页脚体系。主汇报 10 页，附录 6 页；文字、分数矩阵、表格、流程和图表均为可编辑 PowerPoint 对象，案例 RGB 面板复用已有冻结图片。

- [完整可编辑 PPTX（10 + 6 页）](Matcher阶段汇报_收紧主线与OraclePose.pptx)
- [完整 PDF（10 + 6 页）](Matcher阶段汇报_收紧主线与OraclePose.pdf)
- [仅主汇报 PPTX（10 页）](Matcher阶段汇报_收紧主线与OraclePose_主汇报10页.pptx)
- [仅主汇报 PDF（10 页）](Matcher阶段汇报_收紧主线与OraclePose_主汇报10页.pdf)
- [每页 PNG 预览 ZIP](每页PNG预览.zip) · [浏览每页](previews/index.html) · [整套预览](整套预览.png)
- [逐页 CHANGELOG](CHANGELOG.md)

P5 已改用 0769 的无已知标注 / 物理兼容替代歧义案例：GT A1–B1，rank 3，最终接受错误 A0–B1。P9 只引用已有 Oracle JSON 中两个 train 正边的真实 current M2 对照，并保留不能外推 full-val 的中英文限定。原物理 / BEV、opening type、长走廊、详细计划、阶段总结及 E1 对照保存在附录。

来源、数值与边界见 `source_manifest.json`、`data_verification.json`、`case_selection_revised.json` 和 `slide_specs.json`。`verification.json` 保存最终导出的逐页检查及文件哈希。所有原始来源在结束时核对 SHA-256 未变。

## 复现

在仓库根目录执行：

```bash
/home/feixia/anaconda3/envs/bi_layout/bin/python -B src/output/matcher_progress_ppt_20260910/build_ppt.py
```

生成源沿用原版的原生布局函数（`layout_helpers.py`）；`appendix_original.py` 保留旧页内容。依赖本次环境已有的 `/tmp/bi_layout_ppt_deps` 中 `python-pptx`，未改变项目训练依赖。用本机 LibreOffice 将两份 PPTX 转成同目录 PDF 后执行：

```bash
/home/feixia/anaconda3/envs/bi_layout/bin/python -B src/output/matcher_progress_ppt_20260910/render_and_verify.py
```

该脚本只进行 PDF 文本核对、PNG 导出、预览打包及哈希核对。主汇报独立版与完整版前 10 页内容一致。原汇报文件未覆盖；未训练、未运行模型或 GPU inference、未加载或修改 checkpoint、未修改预测 / Matcher / threshold、未访问 test。
