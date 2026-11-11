> 修改记录：2026-11-05 19:40 CST - 按计算机视觉工程分层重构，合并当前指南并建立历史来源映射。

# 推理与输出

楼层推理入口是 `python -m portallayout floor run infer`，核心实现为 `portallayout.engine.floor_layout_pipeline.estimate_floor`。

两种输入互斥：已有 `--prediction-cache`；或 `--manifest` 加 `--raw-root` 和新 `--frontend-cache-output`，现场运行 RGB 前端。必须提供 `--bundle` 或 `--pair-checkpoint`，详细参数用 `--help` 查看。

```text
python -m portallayout floor run infer --prediction-cache <预测缓存> --bundle <bundle.json> --output <新推理目录>
python -m portallayout floor run infer --manifest <公开manifest.json> --raw-root <原图目录> --frontend-cache-output <新缓存目录> --bundle <bundle.json> --output <新推理目录>
```

尖括号是需要替换的路径，不是可直接运行的示例数据。

## strict 与 exploratory

`strict` 是默认模式，沿用原标定和接受条件。显式 `--inference-mode exploratory` 可运行未经严格标定的探索性候选/几何或 verifier 策略；输出保留 tentative、校准状态和 provenance。存在结果文件、图连通或运行 complete，均不等同于重建通过验收。

No Match、孤立观测和多个连通分量是合法结果。预测失败会记录为失败楼层，评价保留相应分母。布局展示时移动分量的排版位置不用于计算指标。

## 输出

每层包含 `floor_result.json`、pair 结果、预测邻接、连通分量、BEV 和报告；批量入口另写 `run_report.json`。结果记录权重、缓存、训练状态、推理模式与策略。实际文件集合以导出器和运行报告为准。

默认输出目录集中在 `outputs/`，但脚本允许显式指定新目录。旧 `src/output` 指向同一数据，不是拷贝。实验比较继续在命名独立的 `experiments/` 目录冻结输出。

早期双房间 JSON 对齐可用 `python -m portallayout legacy cross-scene --help`。单图原推理入口 `inference.py` 保留兼容，不与整层 RGB 推理混称一个任务。
