这是用户明确授权的五小时后一次性项目续做任务。用户原话：“若这个实验没做完，五个小时之后继续做，若这个实验做完了，五个小时之后，对整体项目进行代码编写”。请实际检查代码和结果并执行工作，不停留在计划或请求再次确认。

工作目录：/home/feixia/pythonProject/AAAsrk/Bi_Layout。
先读 AGENTS.md（如有）、docs/research1_opening_pair_implementation.md、experiments/research1_opening_pairs_20261023/validation.json、docs/research1_full_experiment_plan.md、docs/research1_metrics_protocol.md，并检查 git status。仓库有许多用户和之前阶段的未提交改动，必须保留；不要 reset/clean、覆盖数据/旧权重、提交或推送代码。

条件分支：
1. 如果双图开口匹配修复实现或其短实验仍有遗留错误，优先完成修复、针对性测试、真实20图190对验证、证据记录。已有结果若有效则复用，不重复无必要训练。
2. 如果本轮交付已完成，则继续整体项目代码：从当前全景观测图和多位姿假设接口出发，优先实现每连通分量的鲁棒 Sim(2) 位姿优化及楼层级入口，再按已有方案补齐同空间观测关联/多视点布局融合和统一七指标评估。先审查已有相关模块并复用，不重复创建冲突实现。逐步实现、运行合理测试、更新文档和交付记录。遇到数据或人工标签不足时，继续完成不依赖该条件的代码和合成验证，并清楚标记真实效果未验证。

任务口径：输入整层全部 RGB 全景，前端预测每图布局和开口，再进行开口匹配、相对位姿和全局布局。节点是 panorama，pano不等于房间；未知GT位姿、真实房间分组和米制尺度都不能进入推理。固定现有前端与开口候选作为研究对照。不得以随机或smoke训练权重输出冒充校准对应，不得为了连通强行补边；保留孤立节点、多个分量及未解决假设。GT只在独立离线监督构建或评价路径使用。

当前已知交付：
- 单图增强 OpeningRepresentationEncoder、双图 OpeningPairMatcher、自适应 log-Sinkhorn、几何弱监督构造、训练/恢复/校准入口、位姿多假设和190对调度已实现。
- 真实楼层 house0665 floor03：20图51候选，40个唯一关联、11个歧义；190对中138对可训练，75个弱正cell、633个弱负cell、513个unknown。
- 200步短训练：同批fit平均损失0.62461降至0.32879；checkpoint为smoke_trained，尚无人工审核dev标签，正式校准未执行。
- 推理190对完成，0个数值不收敛；12组诊断图；42个源缓存文件哈希不变；GT I/O拦截通过。由于未校准，verified边为0，保留20个孤立观测。
- 最后回归344项通过，35项为新增测试。先确认这些证据未被后续修改失效。

环境与路径：
- Python /home/feixia/anaconda3/envs/bi_layout/bin/python，torch1.12；无pytest，使用unittest。设置OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1；torch线程2。
- GPU RTX4060Ti16GB；不要下载或启动未经要求的大型训练。
- 数据集 src/dataset/ZInD-FloorSet-v1/train_20261021；原始RGB /home/feixia/pythonProject/AAAsrk/zind/data。
- 冻结前端缓存 src/output/research1_floor_frontend/smoke_train_0665_floor03_20261022。
- 本轮输出 src/output/research1_opening_pairs/smoke_train_0665_floor03_20261023，含weak_supervision.json、training/checkpoint.pt、inference/run_report.json。
- 单图原始诊断缓存 src/output/research1_opening_representation/smoke_train_0665_floor03_20261022，保持未修改。

当前是实际实施任务。遵循可用权限；若工具需要不可取得的审批，不要绕过权限，记录阻塞并完成其他已授权部分。不要调用邮件/消息工具，不需要其他代理。最终用中文报告修改内容、验证结果、产物路径和真实尚未完成的工作。把详细进度写入docs或本次新的experiments目录。不要再建立重复定时任务；本次只运行一次。
