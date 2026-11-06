修改记录：2026-10-28（Asia/Shanghai）—记录 SALVe 原版模块复现、消失角读取修复、真实运行证据和端到端阻塞项。
修改记录：2026-10-29（Asia/Shanghai）—完整 MHNet 归档下载、gzip 校验、解压和候选重跑完成；记录权重镜像检索结果。

修改记录：2026-10-28（Asia/Shanghai）—补充 6 项预测读取测试、候选重跑一致性和下载最终状态。

# SALVe 复现记录

**状态：原版模块验证已完成；从原始全景到最终楼层的端到端复现尚未完成，也未复现论文全测试集指标。**

## 2026-10-29 续跑结果（当前状态）

- MHNet 归档已完整下载：732377373 字节，`gzip -t` 成功；SHA256 见 `archive_verification.json`（本地审计哈希，并非与官方发布哈希比对）。
- 已解压至 `/home/feixia/pythonProject/AAAsrk/salve/data/ZInD_HorizonNet_predictions`，1575 个房屋目录、67449 个预测 JSON；消失角已按官方 CSV 生成。
- 使用完整归档重跑 building 0000：2773 个 JSON，与之前 fixture 运行内容逐文件一致。日志 `full_archive_hypotheses.log`，结果 `full_archive_hypotheses/`。
- 下载已结束，无下载进程。下文关于下载中断和部分文件的描述是 10 月 28 日历史记录，以本节和 `status.json` 为准。
- HoHoNet 权重仍缺失。GitHub Releases 和 Hugging Face 按 HoHoNet 名称查询均为空；检查 30 个分支中的 29 个可访问文件树和 README，未发现目标权重或新的下载地址，1 个分支返回 404。部分分支有 `label13_weight.pth`，这是语义类别权重，不能替代深度网络参数。
- 检索原始结果：`asset_search_20261029.json`、`fork_search_20261029.json`、`fork_readme_links_20261029.json`。随后扩展 issues API 查询遭遇速率限制，见 `upstream_asset_issues.json`；不能声称所有网络渠道已穷尽。
- 上游 [HoHoNet issue 18](https://github.com/sunset1995/HoHoNet/issues/18) 同样报告权重访问受限，本次未查到维护者提供替代链接。
- 本次没有替换深度模型、没有对外发消息或创建 issue；端到端状态继续为未完成。

完整预测文件的单房屋候选复跑命令（输出选择新目录）：

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=4 ../salve/.venv/bin/python tools/reproduce_salve_hypotheses.py \
  --raw-root ../zind/data \
  --predictions-root ../salve/data/ZInD_HorizonNet_predictions \
  --building-id 0000 \
  --output experiments/salve_reproduction_20261028/full_archive_hypotheses_new
```

## 实际环境和代码

- 官方仓库：https://github.com/zillow/salve
- 本地：`/home/feixia/pythonProject/AAAsrk/salve`
- commit：`7bb3e78d79892f805d3662f1f86f080da47918a1`
- Python：`../salve/.venv/bin/python`；PyTorch `1.12.1+cpu`。本次执行使用 CPU。
- 仓库原来已有本地修改，包括兼容修复和测试修改；执行前快照见 `salve_existing_changes.patch`，不能称为完全未经修改的官方仓库。
- 本次新增修复见 `vanishing_angle_fix.patch`：`hnet_prediction_loader.py` 按图片文件名 stem 读取消失角，与官方 CSV 拆分脚本一致。原来的整数键导致 `KeyError: 15`。未用默认角度掩盖缺失数据。
- 完整依赖见 `environment.txt`。没有修改 Bi_Layout 模型、训练入口或既有研究实验。

## 已验证结果

| 环节 | 输入 | 实际结果 | 证据 |
|---|---|---|---|
| 几何/位姿模块 | 当前 SALVe checkout 的 algorithms/common 测试 | 54 passed | `geometry_tests.log` |
| MHNet 预测读取 | 当前 checkout 的 prediction/loader 测试 | 6 passed | `prediction_tests.log` |
| W/D/O 候选生成 | 官方 MHNet 测试样例预测 + 本地 ZInD building 0000 + 官方消失角 | floor_01 共 2773 个 JSON | `hypotheses_summary.json`, `hypotheses.log` |
| 原版 ResNet-152 验证器 | 官方四张 ceiling/floor RGB BEV 样例 | 正确对齐类概率 0.96240854，预测类 1 | `verifier_smoke.json` |
| 布局拼接 | 官方自带预测位姿和布局样例 | 样例 IoU 0.81194802；已导出图片 | `stitching/score.json`, `stitching/cluster_0.png` |

候选文件分布：`gt_alignment_approx=117`、`incorrect_alignment=2555`、`gt_alignment_exact=101`。最后一类是官方生成器额外保存的精确 GT 对齐，不能计为模型预测。GT 用于官方候选标注及评估；候选的布局/WDO 来源是 MHNet 预测。

验证器权重为 `1200ffbe47d836557d88fef052952337.pth`，流式 MD5 与官方文件名一致，state_dict 严格加载成功。使用官方确定性 resize、center crop、ImageNet normalization 和 ceiling1/ceiling2/floor1/floor2 输入顺序。

这些模块使用不同的官方样例，不是同一楼层贯通的流水线。0.9624 是一个样例的概率，不是准确率；0.8119 是自带位姿的拼接样例 IoU，不是原始全景端到端效果。拼接脚本含 GT 对齐步骤。

## 可重复执行

以下命令在 Bi_Layout 根目录执行：

```bash
# 官方验证器权重 + 四张官方 RGB BEV 样例
../salve/.venv/bin/python tools/reproduce_salve_smoke.py \
  --output experiments/salve_reproduction_20261028/verifier_smoke.json

# 单房屋候选；输出必须是一个新目录。fixture_predictions 已准备好官方消失角。
MPLBACKEND=Agg ../salve/.venv/bin/python tools/reproduce_salve_hypotheses.py \
  --raw-root ../zind/data \
  --predictions-root experiments/salve_reproduction_20261028/fixture_predictions \
  --building-id 0000 \
  --output experiments/salve_reproduction_20261028/hypotheses_new

# 官方拼接样例
MPLBACKEND=Agg ../salve/.venv/bin/python ../salve/scripts/stitch_floor_plan.py \
  --output-dir experiments/salve_reproduction_20261028/stitching \
  --path-clusters ../salve/tests/test_data/example_input_stiching/cluster_pred.json \
  --pred-dir ../salve/tests/test_data/example_input_stiching/pano \
  --path-gt ../salve/tests/test_data/example_input_stiching/floor_map_gt.json

# 官方预测归档断点续传（如代理不稳定可使用 --direct）
python3 tools/download_salve_predictions.py --direct \
  ../salve/data/ZInD_HorizonNet_predictions.tar.gz
```

候选生成封装已重跑验证，全部 2773 个 JSON 内容与首次成功运行一致。归档续传因长期无进展已停止，没有遗留下载进程；保留 275902032 字节部分文件，下载器现有单请求 60 秒硬超时和连续失败重试上限。最终状态见 `status.json`。

## 端到端阻塞项

1. 缺少官方 HoHoNet 深度权重：`mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth`。
   - SALVe 下载脚本的 Google Drive 文件 ID `1kZFPwdo36Uk7qP96yYUyQebZtGjsEabL` 实测 HTTP 404。
   - HoHoNet README 的 Google Drive 文件夹实测 HTTP 404。
   - README 的 Dropbox 链接 HTTP 200，但网页标题是 `Dropbox - File Deleted - Simplify your life`；并非权重下载成功。
   - 上游已有相关访问问题：https://github.com/sunset1995/HoHoNet/issues/18 。本次检查未发现评论提供替代下载。
2. MHNet 完整归档：官方 URL 可访问，预期大小 732377373 字节；首次下载在 275902032 字节中断。保留 `.part` 并提供校验 HTTP Range 的续传脚本。以 `download.log` 和 `status.json` 中最终记录为准，不能把部分文件当作完整归档。
3. 获得深度权重后，需要设置 HoHoNet 的 Python 导入路径，执行深度预测 → RGB BEV 渲染 → 同一楼层验证器推理 → PGO → 拼接。当前 54 项测试仅验证几何/位姿模块，不能替代该流水线。
4. 当前 SALVe 环境为 CPU 版 PyTorch。单元测试及样例推理可运行；全数据复现应另配 GPU 环境，并验证 HoHoNet 依赖兼容性。

不以 Bi_Layout 深度、GT 深度、随机权重或几何栅格替代 HoHoNet RGB BEV 后声称完成原版复现。官方 MHNet 权重未公开，原生 ZInD 复现应使用公开预测文件；任意自定义全景上的前端训练属于另一项工作。

## 后续验收

先用一个有齐全预测和 RGB 的楼层完成所有阶段，记录输入全景、候选数、接受边数、定位节点数、分量和输出平面图；再按官方划分扩大评估。当前没有可与 Bi_Layout 做性能比较的 SALVe 全系统分数。
