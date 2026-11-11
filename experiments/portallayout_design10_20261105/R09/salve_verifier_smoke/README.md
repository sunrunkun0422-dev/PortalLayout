> 修改记录：2026-11-06 00:51 CST - 固化无标签 manifest 的 SALVe 官方验证器 fixture 烟雾检查。

# SALVe 验证器样例：无标签输入

结果为 **passed_fixture_smoke**：官方四图 fixture 使用中性输入文件名，模型 class-1 概率 `0.9624085426`；历史结果 `0.9624085426`，绝对差 `0`。结果仅验证这一个固定图像四元组的验证器加载和评分，**不是端到端 SALVe 复现、正式基准或同候选公平对比**。

- `manifest.json`：显式无标签四图 manifest；`inputs/`：与官方 fixture 字节相同的图片。
- `scores.json`：评分结果、官方权重 MD5 和 SHA256；`validation.json`：与历史结果对照。
- `input_sources_offline.json`：官方 fixture 来源及原图与副本 SHA256，仅用于离线溯源，未输入验证器；`artifact_sha256.json`：本目录产物哈希，不包含自身与本 README。
- `environment.json`：SALVe 源码版本、工作区修改状态、环境与关键源码哈希；`run_command.json` 和日志记录实际运行。当前 SALVe 工作区存在本地修改。

在仓库根目录复跑（选择不存在的新输出文件）：

```bash
../salve/.venv/bin/python tools/research10_salve_scoring.py \
  --salve-root ../salve \
  --manifest experiments/portallayout_design10_20261105/R09/salve_verifier_smoke/manifest.json \
  --output /tmp/salve_verifier_smoke_new.json --device cpu --threads 2
```

原始 SALVe HoHoNet 深度权重仍未取得；候选生成、BEV 渲染和后端尚未完成无 GT 依赖审计。本结果不改变这些限制。
