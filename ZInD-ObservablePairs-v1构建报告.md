# ZInD-ObservablePairs-v1：构建报告与数据契约

> GitHub 发布入口：[完整压缩索引与最终训练结果](experiments/zind_observable_v1/README.md)。审核 RGB 包由本地工具生成，不随仓库发布。

更新：2026-09-10。交付快照：`train_auto_v2`；数据集版本仍为 `ZInD-ObservablePairs-v1`。

**结论：已经完成训练房屋的自动构建、可复现索引和双阶段待审核材料。尚未完成真人可辨识性核验和阈值校准，不能称为“已验证可观测数据集”，不能据此宣称 Matcher 的视觉成功率上限。**

## 1. 交付入口

- [正式自动构建统计](experiments/zind_observable_v1/dataset/summary.json)
- [主索引 JSONL](experiments/zind_observable_v1/release_manifest.json)：唯一标签来源；约 1.1 GB。
- RGB 输入白名单索引（本地文件：`experiments/zind_observable_v1/dataset/rgb_input_index.jsonl`）
- [连接索引](experiments/zind_observable_v1/dataset/connections.jsonl.gz) / [全部视点组合](experiments/zind_observable_v1/dataset/all_combinations.jsonl.gz)
- 第一阶段盲审包（本地生成：`src/output/zind_observable_pairs_v1/review_pilot_final/stage1_package.zip`） / 盲审页面（本地生成：`src/output/zind_observable_pairs_v1/review_pilot_final/stage1_blind/index.html`）
- 第二阶段 GT 包（本地生成：`src/output/zind_observable_pairs_v1/review_pilot_final/stage2_package.zip`）：只在第一阶段提交归档后发放。
- 实际 RGB / GT 核验图册（本地生成：`src/output/zind_observable_pairs_v1/figures/index.html`）：含 GT，不提前发给盲审者。
- [全量索引审计](experiments/zind_observable_v1/audit/audit.json) / [官方坐标数值对照](experiments/zind_observable_v1/audit/coordinate_contract.json)
- [阈值配置](configs/zind_observable_pairs_v1.json) / [构建代码](tools/build_zind_observable_pairs.py) / [核心实现](dataset/zind_observable_pairs.py)

旧 BiPair 数据、Matcher 实验和可视化保留，未训练或修改 Matcher。`train_auto_v1` 是审计发现重复视点编号前的中间产物；请使用 `train_auto_v2` 和 `review_pilot_final`，不要混用旧审核包。

## 2. 实际环境与契约核对

仓库：`/home/feixia/pythonProject/AAAsrk/Bi_Layout`。原始数据：相邻仓库 `../zind/data`；划分文件：`../zind/zind_partition.json`。官方划分为 train / val / test = **1260 / 157 / 158 个房屋**，房屋集合互斥。这次构建只读取 train 的注释和图像；读取完整划分名单用于泄漏检查，不等于读取测试样本。

构建前检查了 Git 状态和项目约定，重新阅读了 `dataset/zind_pair_mining.py`、已有 ZInD 数据适配/构建代码，以及官方 `code/transformations.py`、`code/render.py` 和本地真实 JSON。旧 miner 的 `_primary_panoramas` 按 partial room 选择 primary，并使用 complete/raw 开口构造几何对；这些选择不构成双 RGB 可见或可辨识的证据。新实现只复用已经数值核对的坐标函数，数据枚举、标签和审核协议独立。

官方依据：[ZInD README](https://github.com/zillow/zind)、[数据组织说明](https://github.com/zillow/zind/blob/main/data_organization.md)、[坐标实现](https://github.com/zillow/zind/blob/main/code/transformations.py)、[渲染实现](https://github.com/zillow/zind/blob/main/code/render.py)。本地对应文件的 SHA-256 已保存，结论不依赖旧报告行号。

| 字段 / 概念 | 核对后的含义 | 本版本使用方式 |
|---|---|---|
| `complete_room` | 合并 partial 几何后的完整房间单元 | 不同 complete room 的连接标记 `cross_complete_room` |
| `partial_room` | 完整房间内的标注子空间，不能直接当成独立完整房间 | 同一 complete 内不同 partial 的连接标记 `within_complete_partial` |
| `layout_raw` | 原始人工标注几何 | 保留子空间边界和人工 opening |
| `layout_complete` | 合并 partial 后的派生完整布局 | 跨完整房间几何、墙体射线代理 |
| `layout_visible` | 当前视点的派生可见布局 | 支持度代理；不等价于 RGB 无遮挡标签 |
| primary / secondary | primary 用于人工布局标注；secondary 为额外定位视点 | 两类都参与组合枚举，不假定 primary 对目标共享开口最优 |
| door | 门位置标注 | 保留 `door` 来源；没有可靠门开闭状态 |
| opening | 官方定义的人工子空间分界，可在合并处理中移除 | 保留 `opening` 语义，特别用于完整房间内部子空间连接 |
| window | 窗位置标注 | 解析验证，但不作为可通行共享连接 GT |
| `internal` | 不连接外墙的内部多边形，如岛台 | 仅用于已标注内部结构的 2D 遮挡代理 |
| `redraw` | 面向最终平面图的清理结果 | 不替代 merger 几何，也不作为模型输入 |

**“opening”不应直接解释为物理门洞类型的全部集合。** 本项目的共享开口任务同时保留门连接与人工子空间连接，但分开统计；后者即便几何清楚，也可能缺乏独立可见边框或唯一视觉语义。长走廊的额外自主切分尚未在本版本引入。

### 坐标、尺度和投影

设局部全景平面坐标为 `p`、相似变换尺度为 `s`，楼层米制比例为 `m`：

```text
p_floor = s * R(theta) * p_local + translation
p_meters = m * p_floor
local_length_m = local_length * s * m
T_B_to_A = inverse(T_A_to_floor) @ T_B_to_floor
```

官方使用行向量右乘旋转，新代码使用等价列向量矩阵。相机高度通常归一化为 1，而不是默认为真实 1 米。W/D/O 每条标注占三个二元组：`[左端点 xy, 右端点 xy, [底部 z, 顶部 z]]`；第三项是相对相机的高度，不是第三个平面点。换到楼层高度时用 `(z + camera_height) * s`。

`T_B_to_A` 是归一化局部坐标之间的相似变换，可能带 `s_B/s_A`，不是直接的米制 SE(2)。后续米制位姿评价应使用相对旋转 `R_A.T @ R_B`，并将其平移乘以 `a_local_to_meters = m*s_A`；缺 `m` 时不能报告米制平移误差。

全景经度 `atan2(-x,y)`，归一化横坐标 `u = theta/(2*pi)+0.5`，纵坐标 `v=0.5-atan2(z,r_xy)/pi`；像素使用 `(W-1,H-1)`。区间采用圆周 `start_u + length_u`，支持左右接缝。与官方代码比较 388 个训练全景：楼层坐标最大绝对误差 `8.88e-16`，投影最大误差 `1.42e-13` 像素。仅在独立测试进程给官方旧式 `np.float` 类型注解提供兼容别名，未改官方文件。

真实训练数据共有 **211 个楼层缺少有效米制比例**。它们不会设为 1；关系保留 unknown，缺米制的位姿也不进入米制评价。

### 审计发现的真实编号冲突

`0363 / floor_02 / complete_room_05` 的 `partial_room_05/pano_34` 和 `partial_room_08/pano_34` 具有不同图像、布局和变换，裸 `pano_id` 不能作为楼层唯一键。初次构建有 8 条重复 ID。修订版在碰撞时使用 complete/partial/pano 完整路径，保留两个视点。

修订采用受约束的增量重建：固定修复前后代码哈希，逐个检查全部训练房屋的原始编号；仅重建有碰撞的 `0363`，复用其余 1259 个未进入修复分支的房屋分片。证明条件、分片哈希和父快照见 [增量来源记录](experiments/zind_observable_v1/dataset/incremental_provenance.json)。全新完整构建不依赖父快照，命令见下文。

## 3. 三个标签维度与推理输入隔离

| 维度 | 标签 | 当前自动快照 |
|---|---|---|
| adjacency | positive / negative / unknown | 根据标注几何提出，仍为 auto_proposed |
| visibility | clear / partial / not_visible / uncertain；逐 GT、逐侧 | 真实像素可见性保留 uncertain；负样本无目标时不适用 |
| identifiability | clear_evidence / multiple_plausible / insufficient_evidence / unreviewed | 全部 unreviewed |

几何代理单独保存在 `auto_proposal`、`gt.correspondences[*].A/B`，不覆盖人审标签。几何 positive 但看不清仍为 positive；歧义、候选漏检、模型 abstain 均不能改成真实 no_match。`no_match` 是盲审者/模型的回答；只有标签可靠的 negative 才是非相邻监督。

普通配置为 `rgb_pair`，固定 **1024×512 / 张**。`public_input` 和 `ObservablePairsDataset.__getitem__` 只返回 RGB、路径、分辨率和不编码类别的 ID。路径只负责定位文件，不应编码成模型特征。GT 位姿、布局、对应、房间归属在离线 `supervision()` 中。

`gt_candidate_diagnostic()` 明示 `gt_assisted=true`，用于 GT 候选诊断，不能当普通 RGB 实验。当前没有接入预测 layout / 开口 / 类型；以后如接入应另命名配置，并使用训练分割外生成预测，报告候选召回与对应效果。

## 4. 自动构建流程与边界

1. **同房屋、同楼层、按两类房间单元枚举。** 不同 complete 之间比较完整布局；同 complete 内不同 partial 比较 raw 布局。不产生跨房屋简单负样本。
2. **几何确认使用条件交集。** 同类 door/opening，端点反转最小误差、宽度比、方向、垂直高度重叠、双方边界支持、局部法向两侧内部关系和多边形重叠共同约束。多视点重复标注合并并保存来源。多条真实对应保留集合。
3. **冲突进入 unknown。** 近阈值不一致、一对多几何对应、缺尺度、布局或开口清单不一致等不能硬判负。未找到匹配本身不是负证据。负样本要求标注清单一致且两空间多边形有足够间隙，使共享边界在这份标注下不成立；证书明确不是人审事实。
4. **全部视点组合记录。** 主体宽度、角宽、距离、入射角、两端点射线、图像解码/长宽比/灰度分布、visible 支持共同形成代理。沿开口采样射线区分目标接触与前方墙/内部结构，允许数值容差并处理共线遮挡。
5. **视点重选。** 优先通过代理、提高较差一侧得分；后续组合增加低权重的视点互补项，每个房间单元对最多 3 组。保留未入选组合、得分与原因。多 GT 时自动通过只要求至少一条对应双侧通过；严格 core 则要求完整 GT 集合得到核验，二者不要混淆。
6. **核验与分层。** 300 对训练试点；人工结果经独立导入形成新的 reviewed 主索引与 ID 视图，不回写原自动快照。

图像质量目前为粗代理，不是完善的模糊/异常检测。2D 墙体射线没有家具和门板三维模型；其“挡住”也不是像素层面的最终真值。人工 opening 被合并移除时，使用 `layout_visible` 派生 footprint 的支持并单独标来源，证据弱于显式开口线支持。CoVis 未用于选择或裁决；仅核对官方存在评分表，不把全局重叠替代共享开口可见性，也不宣称已经复现其定义。

所有业务阈值均在配置中，当前为 **draft_train_only / 待校准**：端点 0.20 m、疑似冲突 0.45 m、宽度相对差 0.25、方向 15°、负样本空间间隙 0.75 m；最小宽 16 px、角宽 5.625°、距离上限 10 m、入射余弦 0.25、射线通过比例 0.85、visible 支持 0.80。数值不是已经验证的最优值。

## 5. 实际训练数据统计

读取 54,034 个全景标注条目：primary 26,606，secondary 27,428；raw/complete 各 54,034 条，visible 51,602 条。编号碰撞涉及不同图像，条目数不因此减少。

| 关系范围 | 几何 positive | 几何 negative | unknown | 总计 |
|---|---:|---:|---:|---:|
| 不同完整房间 | 25,789 | 76,625 | 90,157 | 192,571 |
| 同完整房间内子空间 | 15,380 | 21,674 | 22,025 | 59,079 |
| 合计 | **41,169** | **98,299** | **112,182** | **251,650** |

这些是选中的 **panorama pair 数量**，不是独立房间连接数，更不是人工已确认样本数。共枚举 106,404 个房间单元对、850,243 个 panorama 组合。关系枚举包含非连接和 unknown，不能把 106,404 都称作真实连接。

单元对层面为 positive 15,591、negative 43,180、unknown 47,633。选中正样本中 517 对包含多个 GT；共有 25,946 个 GT 侧投影跨全景接缝，未因跨界删除。28,273 对缺少楼层米制尺度并保留 unknown。

全量验收通过：251,650 个唯一 ID，1,260 份源注释及 53,581 张入选图像的 SHA-256 均核对；850,243 个组合与最多 3 组的选择结果一致，全部 ID 视图、输入白名单和试点概率抽样可复现。主索引 SHA-256：`0eba445d9c0d3db169a5b12deef1a0028501cd0700730c821e13ccef686f6854`。几何/原 miner 回归与人审流程共 27 项不同测试通过；浏览器测试使用显式 synthetic 临时数据，不计入真人审核。

| ID 视图 | 数量 | 用途与限制 |
|---|---:|---|
| `auto_observable` | **35,659** | 几何 positive 且至少一条 GT 双侧代理通过；待审核 |
| `core_verified` | **0** | 没有人审，不能自动填充 |
| `pending_review` | 251,650 | 当前全部样本 |
| `geometry_positive / negative / unknown` | 41,169 / 98,299 / 112,182 | 保留关系三态 |
| `gt_candidate_diagnostic` | 41,169 | 仅 GT 候选辅助协议 |
| `hard_or_ambiguous` | 215,991 | 当前是 auto_observable 的补集，含普通负样本；不是纯困难集 |

负样本中 84,749 对带至少一个困难代理，13,550 对为普通负样本。困难标记包括空间较近、door 类型/宽度相似、双方门清单较多。**后两项不是 RGB 外观相似和重复纹理的人工确认**，必须在第二阶段核验；标记可重叠，不能直接相加。

同一批 15,591 个几何 positive 单元对中，primary 优先、最近相机的参照方案有 13,172 个代理通过；重选首对有 14,853 个通过，增加 1,681 个，约 **+10.78 pp**。9,273 个更换了视点组合。该对照固定的是新几何标签，不是旧数据集构建的精确复放；效果也是几何代理改善，不是 Matcher accuracy。

## 6. 300 对双阶段审核与加权分析

抽样覆盖 261 个训练房屋，10 个分层各 30 对：两种关系范围 × 正样本通过/未通过、负样本困难代理/普通，以及 unknown。正样本 120 对、负样本 120 对、unknown 60 对。分层内先均匀抽房屋，再在房屋内抽一对，完全不使用 Matcher 结果。

若某层有 H 个房屋，抽 q 个，房屋 h 有 N_h 对，则逐样本包含概率 `pi=(q/H)/N_h`，权重 `1/pi`。抽样文件保存原始数量、概率、权重与确定性随机种子。目标总体是**已选择的主索引**，不是全部 850,243 个组合。不同层/组合可能共享房屋，后续置信区间必须考虑房屋聚类。

协调者流程：

1. 分别发第一阶段 ZIP 给两名独立审核者。完成整批第一阶段；无把握选 uncertain，不猜“非相邻”。允许在完整 RGB 任意位置自由框选多组对应，跨接缝可选外侧区间。
2. 回收第一阶段 JSON 并归档；之后再发第二阶段 ZIP。若第一阶段只完成部分，应标缺失，看到 GT 后不能再把其余样本补成独立盲审。
3. 第二阶段导入本人第一阶段文件并锁定 SHA-256，检查完整 GT 对应集合、两侧可见性、归属/投影错误和歧义，保留第一次选择不变。审核者必须明确确认独立完成。
4. 导入工具校验版本、审核者、阶段顺序、原文件哈希、坐标范围和证据文本。一对多 GT 采用集合的一对一匹配；选中另一条正确 GT 不算错，但 strict core 要求完整集合。
5. 分歧保留两份原始答案，第三人提交显式 `resolution` 和依据。裁决可以更新诊断标签，但不替代缺少的独立盲审证据；这些记录留在 `adjudicated_pending_blind_confirmation`。GT 标注需要修改时另建版本，不能在审核文件中偷偷换 GT。

core 的正样本要求：原几何 positive、两人独立盲审完整集合正确、GT 阶段一致、所有真实对应双侧清楚、明确可辨识且标注检查通过。负样本要求原有非相邻证书、两人独立 no_match 及第二阶段一致确认。自动流程没有真人身份认证能力；独立性由审核组织和本人声明保证。

[加权汇总工具](tools/summarize_zind_observable_review.py) 只在试点审核完整解决后输出 HT 加权总量 / Hájek 比例；缺审核时仅给响应情况和受审者诊断，不假定缺失随机。当前 **0/300 已审核，总体可辨识率不可估计**。人类识别证据也不是严格信息论上限。

## 7. 复现、加载与后续操作

在仓库根目录使用已有环境；以下命令均选新输出目录，不覆盖旧数据：

```bash
conda activate bi_layout
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/observable_mpl

# 完整从原始训练房屋复现，不需要任何旧分片
python -B tools/build_zind_observable_pairs.py \
  --root ../zind/data --partition ../zind/zind_partition.json \
  --config configs/zind_observable_pairs_v1.json --split train \
  --output src/dataset/ZInD-ObservablePairs-v1/builds/train_reproduction

# 中断后用相同参数增加 --resume；代码、配置和注释哈希不一致时拒绝续建
python -B tools/audit_zind_observable_build.py \
  --build src/dataset/ZInD-ObservablePairs-v1/builds/train_reproduction \
  --output src/output/zind_observable_reproduction/audit
python -B tools/prepare_zind_observable_review.py \
  --build src/dataset/ZInD-ObservablePairs-v1/builds/train_reproduction \
  --output src/output/zind_observable_reproduction/review
python -B tools/visualize_zind_observable_build.py \
  --build src/dataset/ZInD-ObservablePairs-v1/builds/train_reproduction \
  --audit src/output/zind_observable_reproduction/audit \
  --output src/output/zind_observable_reproduction/figures

python -B -m unittest tests.test_zind_observable_pairs \
  tests.test_zind_observable_reviews tests.test_zind_pair_mining -v
```

模型加载示意：

```python
from dataset.zind_observable_pairs import ObservablePairsDataset
base = 'src/dataset/ZInD-ObservablePairs-v1/builds/train_auto_v2'
data = ObservablePairsDataset(base+'/main_index.jsonl', '../zind/data',
                              view=base+'/views/auto_observable.json')
rgb_only = data[0]                 # 不含 GT 监督
offline = data.supervision(0)      # 只用于离线监督/评估
```

当前读取器会将主索引读入内存，完整索引需要数 GB RAM；批量构建与审计按 JSONL/房屋分片工作。大索引和原始图片不应作为代码 PR 附件；保留本地版本化目录与哈希即可。

实际审核文件回收后：

```bash
python -B tools/import_zind_observable_reviews.py \
  --build src/dataset/ZInD-ObservablePairs-v1/builds/train_auto_v2 \
  --mapping src/output/zind_observable_pairs_v1/review_pilot_final/coordinator_mapping.json \
  --stage1 /path/to/reviewer_A_stage1.json --stage1 /path/to/reviewer_B_stage1.json \
  --stage2 /path/to/reviewer_A_stage2.json --stage2 /path/to/reviewer_B_stage2.json \
  --output src/dataset/ZInD-ObservablePairs-v1/builds/train_reviewed_v1
python -B tools/summarize_zind_observable_review.py \
  --build src/dataset/ZInD-ObservablePairs-v1/builds/train_auto_v2 \
  --reviewed src/dataset/ZInD-ObservablePairs-v1/builds/train_reviewed_v1 \
  --output src/output/zind_observable_pairs_v1/review_weighted_summary.json
```

阈值校准应使用训练试点中真实核验的通过/未通过证据，比较误筛与漏筛，记录变更原因、房屋名单和配置哈希。完成后人工写入真实冻结记录：`status=frozen`、`calibration_split=train`、`calibration_houses`、`config_sha256`、`evidence`。`--split val/test` 必须传匹配的 `--freeze-record`，否则在读取样本和创建输出前报错。当前没有伪造冻结记录，因此 **val/test 未构建**。

## 8. 当前研究判断与未完成事项

现在可以量化“几何对应 → 几何观测代理”的损失，并审核视点重选是否有用；还不能量化“真实可见 → 唯一可辨识”的损失。大量 unknown 被显式保留是证据质量问题的暴露，不宜为了提高正负比例而删除或改负。

下一步优先级：两名审核者完成试点 → 按抽样权重检查真实可见性/可辨识性及错误来源 → 只在训练房屋校准阈值并冻结 → 构建 val/test → 按跨完整房间和子空间分别评价 RGB Matcher。报告指标应包括 adjacency、集合对应 precision/recall、abstention/coverage、GT 候选与预测候选召回差异、可用端点的 pose 误差，并以房屋/连接聚合，避免把最多三个相关视点对当独立证据。

本轮完成自动构建和审核工具；真人审核、困难外观的确认、校准冻结、验证/测试最终索引及任何新训练结果均未完成。ZInD 的住宅分布和人工子空间标注也不能单独证明对宿舍长走廊等场景覆盖充分。

需求文本在第 6 节 `core_verified` 的“负样本难度”处截断，因此其他 ID 视图名称与规则是本次明确列出的可调整初稿，没有假装已实现未收到的后续条款。
