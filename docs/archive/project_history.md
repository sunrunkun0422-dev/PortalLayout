> 修改记录：2026-11-05 19:36 CST - 合并历史资料，保留来源、原修改记录和日期边界，调整迁移后的文档链接。

# 项目更名、整理与早期结果记录

本文为历史来源合订本。原文中的“当前”“待实现”“正在运行”均以原日期为准；当前维护文档从 [文档入口](../README.md) 阅读。代码与数据路径可能保留历史写法，迁移对应表见 [项目结构](../development.md)。

- [docs/research1_code_cleanup_audit.md](#source-1811e05f4f)
- [docs/project/code_map.md](#source-9861032b1d)
- [docs/project/workflows.md](#source-28e482e7ae)
- [docs/project/renaming.md](#source-99e1e5b372)
- [ZInD端到端训练结果.md](#source-4242817bc1)

---

<a id="source-1811e05f4f"></a>

## 来源：docs/research1_code_cleanup_audit.md

合并前 SHA-256：`3938dbe7b15d540eea0d1ddfe8e5fdc2bfce0c4108ea92df2613766dcb2f69fa`。原文与修改记录如下（仅相对链接重定位）：

修改记录：2026-10-23 07:00（Asia/Shanghai）新增：整层续作期间的保守清理候选审计；仅清单，未删除或合并。

# 代码清理候选审计

当前仍处于实现和验收分支，尚不满足“全部完成后清理”的前提。以下候选不构成删除授权，不删除实验结果、用户文件、缓存、数据、权重或回归对照。引用证据保存于 `experiments/research1_floor_layout_followup_20261023_0656/cleanup_references.txt`；静态搜索未发现调用不保证外部脚本没有调用。

| 推荐顺序 | 路径/符号 | 调用与引用依据 | 替代实现 | legacy 对照 | 配置与权重兼容性 | 风险及建议 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | utils/floor_assembly.py:FloorAssembler | 仓库 Python 搜索只有类定义；主入口直接调用 assemble_floor | assemble_floor | 否，新接口包装 | 不含参数或 state_dict；保留外部类导入别名/过渡包装 | 低至中；先查外部 API 使用再去除，不直接删整个模块 |
| 1 | utils/floor_pose_graph.py:FloorPoseGraph | 搜索只有类定义；流程调用 optimize_floor_graph | optimize_floor_graph + PoseGraphConfig | 否；用户设计采用此名称 | 保留类名兼容包装和 none/robust 配置 | 中；设计文档可能依赖类接口，优先保留而非为少量代码删 API |
| 2 | utils/floor_training.py:_state_hash 与 utils/opening_pair_checkpoint.py:weights_hash | 前者用于辅助参数更新检查，后者用于 checkpoint 校验；tools/preflight_zind_t4_matcher_only.py:_state_hash 亦被多脚本引用 | 可提取统一 tensor 字节迭代辅助函数，保留每个原摘要格式 | 旧工具属于历史验证链 | 必须逐字节保持名称前缀、排序和摘要算法；不可直接换函数导致旧权重校验失败 | 中；先固定旧 digest 回归；只共用底层，不合并不等价 hash 语义 |
| 3 | utils/floor_auxiliary_features.py:_directed 中 boundary_mean/p90 与 nonseed_boundary_mean/p90 | 两组由同一 mean/p90 写入；models/floor_auxiliary.py:GEOMETRY_NAMES 将它们作为不同输入列 | 新 feature_version 可只保留 nonseed 两列 | 否，新辅助特征冗余 | 当前视觉融合输入维度和 mask 依赖全部列；必须保留 v1 loader，版本升级并重训后才能移除 | 高；不能原地删列或更改权重输入维度 |
| 4 | utils/opening_pair_training.py:train_pairs 与 utils/floor_training.py:train_floor_pairs | run_opening_pairs 与 run_floor_layout 各自调用；test_opening_pair_pipeline 覆盖旧恢复流程 | 提取优化器/随机状态/批次编码等公共小函数 | 是，旧双图 smoke 回归 | 保留原 CLI、监督格式、checkpoint metadata、human-reviewed 校准来源 | 高；两个训练协议不同，禁止直接以新函数替换旧函数 |
| 4 | utils/opening_pair_training.py:calibrate_pairs 与 utils/floor_calibration.py:calibrate_floor_models | 两个入口仍有调用；前者 human_reviewed，后者独立 geometry_dev | 仅可共享纯统计函数，工作点和标签掩码继续分开 | 是，历史人工审核校准对照 | 不放宽旧权限语义，不将 geometry_dev 改名为 human_reviewed；兼容旧 checkpoint | 高；不是可删除的重复功能 |

当前没有充分证据支持删除整个代码文件。训练代码、主匹配器、旧双图入口与实验工具仍承担回归/对照职责。新增整层接口稳定并完成正式验收前，建议只保留此清单。

修改记录：2026-10-23 07:11（Asia/Shanghai）补充本轮版本变化。

本轮为修复校准/推理特征一致性，辅助 feature_version 已由 v1 升至 v2；表内特征去重建议仍然仅为未来候选，若实施还需再次升级版本并显式保留旧版本策略。当前没有删除列、合并训练器或删除包装类。新增 PGO、目录初始化和冲突关系修复属于功能修复，不是清理动作。


---

<a id="source-9861032b1d"></a>

## 来源：docs/project/code_map.md

合并前 SHA-256：`7e82d14742108ea051866f42fdd9ae77b0165211c8893548f4b12755197db9b2`。原文与修改记录如下（仅相对链接重定位）：

> 修改记录：2026-11-05 19:22 CST - 根据当前工作区源码建立 PortalLayout 代码分层与实现状态索引。

# PortalLayout 代码地图

本次依据当前工作区（含未提交文件）核查，不能仅用 Git HEAD 代表该版本。按职责组织导航，保留已有 Python 导入路径。

## 主线模块

| 层次 | 主要源码 | 当前实现 |
| --- | --- | --- |
| 数据 | `dataset/zind_floor_builder.py`、`zind_floor_dataset.py`、`zind_floor_schemas.py` | 楼层构建、公开输入 manifest、GT 与数据校验 |
| 单图前端 | `models/bi_layout.py`、`utils/floor_observation_pipeline.py` | 双布局预测、Opening Head 与观测缓存 |
| 开口表征 | `models/opening_representation.py` | 环形采样、结构与上下文特征、注意力与融合 |
| 新版匹配 | `models/opening_pair_matcher.py` | log-Sinkhorn、dustbin、收敛检查与 mutual matching |
| 匹配管线 | `utils/opening_pair_pipeline.py`、`opening_pair_training.py` | 匹配、训练与阈值标定 |
| 位姿与图 | `utils/floor_geometry.py`、`floor_pose_graph.py` | 几何假设、连通分量与图优化 |
| 楼层管线 | `utils/floor_layout_pipeline.py`、`floor_assembly.py` | strict/exploratory 推理、组装与输出 |
| 辅助训练 | `models/floor_auxiliary.py`、`utils/floor_training.py`、`floor_calibration.py` | pose/room 辅助任务、训练与标定 |
| 评估 | `evaluation/opening_matching.py`、`pose_geometry.py`、`floor_metrics.py` | 匹配、位姿和楼层指标 |

以上表示代码已经实现；本次未重跑训练与数据集评测，不表示这些模块已经达到论文目标。

## 两代匹配器与历史入口

`models/cross_scene_matcher.py` 的 `OpeningGuidedCrossAttentionMatcher` 是早期 token 匹配器，使用 softmax 与 dustbin。`models/legacy_floor_matcher.py` 将该方案适配到相同冻结候选，概率需要独立标定。

`models/opening_pair_matcher.py` 的 `OpeningPairMatcher` 使用带未匹配槽的 log-Sinkhorn；`utils/floor_layout_pipeline.py` 会记录 matcher 类型和 assignment 方法。不能把两代模型写成同一个实现，也不能直接共用工作阈值。

`main.py` / `inference.py` 保留原单图入口；`tools/train_zind_end_to_end.py` 保留早期 pair 联合训练；`tools/estimate_cross_scene_layout.py` 与 `utils/cross_scene_pipeline.py` 保留双房间工程管线。

## 配置与产物

| 位置 | 用途 |
| --- | --- |
| `config/` | Python 默认配置与配置合并逻辑 |
| `src/config/` | 原单图模型 YAML |
| `configs/` | 楼层数据、前端、表征与 ObservablePairs JSON |
| `src/dataset/` | 实际数据；与 `dataset/` 源码区分 |
| `checkpoints/` | 模型权重，保留原名称与路径 |
| `experiments/` | 分日期实验及证据，沿用原实验身份 |
| `src/output/` | 历史运行输出 |
| `docs/` | 方法说明、证据包与当前项目导航 |
| `deliverables/` | PPT、导出图及可交付材料 |

`configs/research1_floor_frontend.json` 仍引用 Bi-Layout 与 Opening Head 权重；改名不修改这些引用。不要把 `config/`、`configs/`、`src/config/` 直接合并，它们分别承担不同职责。

## 修改代码时的落点

- 网络改动放 `models/`；数据与标签放 `dataset/`。
- 推理、缓存与几何逻辑放 `utils/`；指标放 `evaluation/`。
- CLI 参数及运行编排放 `tools/`，再按需注册到 `portallayout/__main__.py`。
- 对应测试放 `tests/`；实验结果进入独立输出目录，并记录配置、权重与来源。

评估继续保留 No Match、失败楼层、断连分量与不确定标签。严格推理与探索性结果分开报告，优势比较需要相同房间与候选条件。


---

<a id="source-28e482e7ae"></a>

## 来源：docs/project/workflows.md

合并前 SHA-256：`441a14d1ec73be1515caaed5af29fb7babaf8e1c752b791c3f0699de8f16d6bd`。原文与修改记录如下（仅相对链接重定位）：

> 修改记录：2026-11-05 19:22 CST - 增加 PortalLayout 分组命令及兼容运行说明。

# PortalLayout 运行入口

先在仓库根目录激活 `bi_layout` 环境。命令分发使用同一 Python 解释器，参数原样传给旧脚本，退出码向上传递。未打包发布，当前使用方式是仓库内 `python -m portallayout`。

| 新入口（后接原参数） | 原脚本 |
| --- | --- |
| `data build` | `tools/build_zind_floor_dataset.py` |
| `data validate` | `tools/validate_zind_floor_dataset.py` |
| `openings predict` | `tools/predict_floor_observations.py` |
| `openings encode` | `tools/encode_floor_openings.py` |
| `openings pairs` | `tools/run_opening_pairs.py` |
| `openings report` | `tools/report_opening_matches.py` |
| `floor run` | `tools/run_floor_layout.py` |
| `floor preflight` | `tools/preflight_floor_training.py` |
| `evidence build` | `tools/build_method_advantage_evidence.py` |
| `legacy layout` | `main.py` |
| `legacy cross-scene` | `tools/estimate_cross_scene_layout.py` |
| `legacy train-pairs` | `tools/train_zind_end_to_end.py` |

表中所有新入口均加前缀 `python -m portallayout`。其他研究脚本仍可直接通过 `python tools/脚本名.py` 使用。

## 查阅参数

```bash
python -m portallayout --help
python -m portallayout data build --help
python -m portallayout openings predict --help
python -m portallayout openings pairs train --help
python -m portallayout floor run infer --help
python -m portallayout floor run evaluate --help
```

## 楼层研究流程

1. `data build` 构建楼层数据，`data validate` 检查数据契约。
2. `openings predict` 生成预测观测缓存。该入口当前选择 official-train 样本；不要把它当作完整 test 推理入口。
3. `openings encode` 可单独导出表征用于分析；楼层训练/推理也有内部编码路径，不能仅因文件名相近就互换缓存。
4. `openings pairs` 提供独立开口任务；`floor run prepare/train` 提供楼层训练工作流。运行前检查对应 supervision 与缓存契约。
5. `floor run calibrate` 使用指定权重和监督标定模型，输出 bundle。
6. `floor run infer` 使用预测缓存，或通过 manifest、raw-root、frontend-cache-output 运行 RGB 前端。必须提供 bundle 或 pair checkpoint。
7. `floor run evaluate` 单独读取 GT 计算指标。

这些是独立操作的顺序说明，不是无需数据和 checkpoint 的一键演示。实际路径由数据清单、缓存指纹与训练产物决定。

以下为使用已有产物的命令模板，尖括号占位符需要替换；输出目录应为新目录：

```text
python -m portallayout floor run infer --prediction-cache <预测缓存> --bundle <bundle.json> --output <新推理目录>
python -m portallayout floor run evaluate --dataset <楼层数据目录> --predictions <推理目录> --output <新评估目录>
```

`strict` 是推理默认值。探索性几何或 verifier 策略需要原脚本相应参数，并在结果中保留 `exploratory` 标记。`evidence build` 复算固定历史产物，不用于证明新训练模型的结果。


---

<a id="source-99e1e5b372"></a>

## 来源：docs/project/renaming.md

合并前 SHA-256：`cddedd9baf177b27683be0f619c4279c9cf45f72ef38d65fc50b09be67f45430`。原文与修改记录如下（仅相对链接重定位）：

> 修改记录：2026-11-05 19:23 CST - 本地目录改为 PortalLayout，保留旧路径兼容链接。

> 修改记录：2026-11-05 19:22 CST - 确认 PortalLayout 项目名，记录本次代码整理与兼容边界。

# 项目更名与整理记录

用户确认的新名称为 **PortalLayout**，中文为 **开口引导的跨房间布局重建**，英文描述为 **Opening-Guided Cross-Room Layout Reconstruction**。Portal 指可连接房间的开口，Layout 指房间及楼层布局。

本次落地：重写项目首页；增加当前代码地图、运行指南与统一命令包 `portallayout/`；在旧索引和手册前增加当前入口；将改名前首页完整保存在根目录 `README_BiLayout_历史说明.md`，原相对链接仍有效。

兼容接口继续保留：

- 本地目录已改为 `/home/feixia/pythonProject/AAAsrk/PortalLayout`，环境仍为 `bi_layout`。旧目录 `Bi_Layout` 保留兼容符号链接，使历史实验中的绝对路径继续可用；新工作区使用新目录。
- 原模型类、导入路径、checkpoint key 和配置文件名不变；Bi-Layout 仍是上游模型名称。
- 原 `tools/` 脚本可独立运行，新入口只做分组与参数转发。
- 数据集名称、历史实验目录和既有指标不做品牌替换，保留复现身份。
- 上游许可、论文引用和致谢保留；新项目名不改变原作者归属。

当前工作区有大量既有修改与未跟踪研究文件。本次没有删除、归档搬移或提交这些文件，也没有修改训练算法。物理目录重构需要逐项处理导入、配置与实验路径，不能用全局字符串替换完成。


---

<a id="source-4242817bc1"></a>

## 来源：ZInD端到端训练结果.md

合并前 SHA-256：`87acf6b7cbff563f116a0f475377aaaf82025085c62286d8e535ae0507cb0be6`。原文与修改记录如下（仅相对链接重定位）：

> 统一文档已合并至 [项目总览与学习手册](legacy_handbook.md#source-5de1910be5)（2026-09-07）。本文件保留为历史资料；其中“当前状态”和“下一步”以原记录日期为准，后续阅读与维护以统一手册为入口。

# ZInD 端到端联合训练结果

> 记录日期：2026-08-04
>
> 对应代码提交：`936d0a73f52008c7909f5946e423306b17fcce7e`
>
> 训练入口：[`tools/train_zind_end_to_end.py`](../../tools/train_zind_end_to_end.py)

## 1. 结论

`Bi-Layout → OpeningSignalHead → Cross-Scene Matcher` 已完成一次真正的端到端联合微调。每个训练 batch 都从原始双全景图开始运行 live Bi-Layout，Opening 与匹配损失能够反向传播到 Bi-Layout 的 CNN、Transformer 和深度输出头，不使用冻结特征替代 live forward。

训练最多计划 24 epochs，实际完成 15 epochs；验证集连续 6 轮没有刷新最佳综合指标后正常早停。最佳模型出现在显示的第 9 轮（checkpoint 内部使用零基 `epoch=8`）。总 GPU 运行时间为 28.35 小时。

端到端训练相对模块化 Matcher 基线有明确提升，但测试集端到端 Top-1 仍只有 **25.65%**，尚不足以视为可靠的自动匹配系统。

## 2. 数据与训练配置

| 项目 | 配置 |
| --- | --- |
| 数据集 | `ZInD_matching` complete-room pairs |
| Train / Val / Test | 29,325 / 3,630 / 3,689 pairs |
| House split | 1,260 / 157 / 158 houses，互不重叠 |
| 输入 | 两张 `3 × 512 × 1024` 原始全景图 |
| Token / feature | 256 tokens / 512 channels |
| GPU | NVIDIA RTX 4060 Ti 16GB |
| 数值精度 | GPU FP32 |
| Batch size | 4 |
| 最大 epoch / 早停 | 24 / patience 6 |
| Bi-Layout LR | `2e-6` |
| Opening Head LR | `1e-5` |
| Matcher LR | `3e-5` |
| LR schedule | cosine，最低为初始 LR 的 10% |
| Opening threshold | 0.47，来自 Opening checkpoint |
| 数据增强 | A/B 独立环形旋转，图像与所有 token/affinity 标签同步 |
| 稳定约束 | 冻结 cache 只作为低权重 feature/depth teacher regularizer |
| 峰值显存 | 10.95 GiB（batch 4 预检） |

FP16 预检会令 Bi-Layout 上游梯度变成非有限值；当前 PyTorch 1.12 的骨干双线性上采样又不支持 CUDA BF16，因此正式训练使用 GPU FP32。FP32 预检确认梯度能够到达：

- Bi-Layout feature extractor
- Bi-Layout Transformer
- Bi-Layout depth heads
- Opening Head
- Cross-Scene Matcher

训练调度为：第 1 轮使用 GT opening guidance/candidates 热身，第 2 轮使用混合 guidance 与预测 candidates，从第 3 轮开始完全使用预测 guidance/candidates。

## 3. 最佳验证结果

最佳 checkpoint 位于显示的第 9 轮：

| 验证指标（predicted openings） | 结果 |
| --- | ---: |
| Opening token F1 | 0.8869 |
| Positive candidate recall | 0.5889 |
| End-to-end candidate Top-1 | 0.2540 |
| Negative rejection rate | 0.9373 |
| Harmonic flow score | **0.3997** |
| Balanced flow score | **0.5956** |

checkpoint 选择首先比较 `harmonic_flow_score`，再比较 balanced score、端到端 Top-1 和 Opening F1。

## 4. 最佳模型测试结果

### 4.1 完全预测门洞候选

| 测试指标 | 结果 |
| --- | ---: |
| Opening token precision | 0.8682 |
| Opening token recall | 0.9226 |
| Opening token F1 | **0.8946** |
| Positive candidate recall | **0.5819** |
| Conditional candidate Top-1 | **0.4407** |
| Conditional candidate Top-3 | **0.9102** |
| End-to-end candidate Top-1 | **0.2565** |
| Negative rejection rate | **0.9296** |
| Harmonic flow score | **0.4020** |
| Balanced flow score | **0.5930** |
| Mean portal-shift error | 65.11° |

### 4.2 Teacher-forced manifest 候选

| 测试指标 | 结果 |
| --- | ---: |
| End-to-end candidate Top-1 | 0.2796 |
| Negative rejection rate | 0.9329 |
| Harmonic flow score | 0.4303 |
| Balanced flow score | 0.6063 |

## 5. 与模块化 Matcher 基线对比

| 测试指标 | 模块化 Matcher | 端到端最佳模型 | 变化 |
| --- | ---: | ---: | ---: |
| Positive candidate recall | 0.5760 | 0.5819 | +0.0059 |
| End-to-end Top-1 | 0.2338 | 0.2565 | +0.0226 |
| Negative rejection rate | 0.9291 | 0.9296 | +0.0005 |
| Harmonic flow score | 0.3736 | 0.4020 | +0.0284 |
| Balanced flow score | 0.5815 | 0.5930 | +0.0116 |

端到端微调有效，但改进幅度不足以消除主要瓶颈。

## 6. 为什么 Top-1 仍然低

测试集端到端 Top-1 可以精确分解为：

```text
0.5819 positive candidate recall
× 0.4407 conditional Top-1
= 0.2565 end-to-end Top-1
```

这说明有两个连续瓶颈：

1. **门洞组件召回瓶颈。** Token F1 达到 0.8946，但完整候选组件只覆盖 58.19% 的正样本。Token 指标高并不等价于候选区间召回高。
2. **第一名排序与拒绝校准瓶颈。** 目标候选存在时 Top-1 为 44.07%，而 Top-3 达到 91.02%。正确答案通常已进入前三，但 Matcher/dustbin 没有稳定地把它排到第一。

负样本拒绝率达到 92.96%，表明模型整体偏保守。继续机械增加 epoch 不太可能解决问题：训练已经由验证集早停，下一步应修改候选生成、排序目标和几何重排。

## 7. 下一步优先级

1. 在验证集上按端到端 Top-1/Harmonic flow 扫描 Opening threshold，不再只依赖 token F1 选择阈值。
2. 提高 candidate ranking loss 权重，降低 token-affinity 与 portal-shift 项对总损失的相对支配。
3. 加入多门洞 hard-negative ranking，并重新校准正样本接受与 dustbin 拒绝。
4. 利用 M6–M9 的几何一致性从 Top-3 候选中重排；条件 Top-3 为 91.02%，这是当前最有潜力的提升路径。
5. 单独检查 65.11° 的 portal-shift 误差，避免把 token shift 与相机相对 yaw 混淆。

## 8. 模型产物

权重按 `.gitignore` 保留在本地，不提交到普通 Git 历史：

| 文件 | 大小 | SHA-256 | 用途 |
| --- | ---: | --- | --- |
| `checkpoints/End_To_End/zind_matching_v1/best.pt` | 418,835,381 bytes | `9406f52e209ffb7645b1187dcf21097e8ea0d924685c1521d74657bd6c7de2f8` | 最佳 Bi-Layout + Matcher 统一权重 |
| `checkpoints/End_To_End/zind_matching_v1/last.pt` | 1,255,777,484 bytes | `49fc6133b09f11593d6763a901d3a7f8f87a75c18dc39f6c087a687596925339` | 含 optimizer/scheduler/scaler 的恢复 checkpoint |

本地完整指标文件：

- `checkpoints/End_To_End/zind_matching_v1/test_metrics.json`
- `checkpoints/End_To_End/zind_matching_v1/best_metrics.json`
- `checkpoints/End_To_End/zind_matching_v1/metrics.jsonl`

## 9. 复现实验

在 CUDA 驱动环境正常且已准备 manifest、feature cache 和三个初始化 checkpoint 的前提下：

```bash
conda run -n bi_layout python tools/train_zind_end_to_end.py \
  --epochs 24 \
  --early_stop_patience 6 \
  --batch_size 4 \
  --accumulation_steps 1 \
  --workers 4 \
  --no_activation_checkpoint \
  --progress_every 250
```

该命令会先执行一次真实端到端梯度预检，再开始完整训练。默认输出目录为：

```text
checkpoints/End_To_End/zind_matching_v1/
```

## 10. 范围说明

本报告完成的是 M1/M2 Bi-Layout、M4 Opening Head、M5 Matcher 的联合梯度训练与测试。M6 pair geometry refiner、M7 relative pose、M8 geometry selector 和 M9 joint fusion 属于分析式/离散几何链路，不参与反向传播；它们仍需使用最佳 checkpoint 单独执行完整整链评估。

