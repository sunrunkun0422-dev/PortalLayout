这是用户明确要求的五小时后一次性任务，请实际检查并执行。用户原话：“设置一个五个小时以后的定时任务，若当前任务未完成，则继续完成，若当前任务已完成，则对已有代码进行测验，列出可以删除合并的代码内容。”

当前任务是实施“整层布局估计：整体代码设计与实施方案”，不是此前已经完成的双图冒烟实验。先读 AGENTS.md（如有）、docs/research1_floor_layout_implementation.md、docs/research1_metrics_protocol.md、docs/research1_full_experiment_plan.md，检查 experiments/research1_floor_layout_20261023 中的进度、验证记录、git status 和真实代码。不能只因有旧的344项测试或smoke模型就认为新任务完成。

如果实施仍未完成或验收存在问题，继续实现、修复、测试和运行已授权实验，直到可交付；如果完整实现及验收已完成，则运行与整层数据流相关的针对性测试与必要回归，并审计已有代码中可删除、可合并的内容。每个候选列出路径/符号、调用与引用依据、替代实现、是否为legacy对照、保留的配置与权重兼容性、风险和推荐顺序。仅列清单，不自动删除或合并，不删除实验结果、用户文件、缓存、数据、权重和回归对照。存在运行中的前台实现时，不同时写相同文件，先检查状态并等待其退出或明确交接。

最新已确认设计：
- 输入整层全部RGB，pano是定位节点，未知真实房间分组；复用冻结的Bi-Layout、Opening Head、候选及单图增强/双图Sinkhorn。每图编码一次，全对枚举，按楼层流式处理。
- 新入口 tools/run_floor_layout.py，后端 utils/floor_layout_pipeline.py，独立评估 evaluation/floor_metrics.py。公共接口estimate_floor、assemble_floor、evaluate_floor。
- 新PoseHypothesisVerifier：两种端点Sim(2)假设，7通道几何BEV CNN 7→32→64→128，与视觉/几何及有效mask融合MLP 256→128→1，双向共享并对称评分。拟合端点及同墙方向不是独立证据；保留歧义，不能用质心异侧强选。已连接节点可用排除本pair的独立路径消歧，不能为了连通补边。
- FloorPoseGraph：分量内最大置信森林初始化，固定稳定ID锚点，所有可靠非树边参与SciPy Sim(2)鲁棒PGO，Huber及归一化残差，最多三轮；冲突先降权，有独立路径证据才剔除并重算分量；优化失败回退可靠森林。p_A=T_B_to_A*p_B，T_B=T_A*T_B_to_A。保留pose_opt=none/robust。
- SameSpaceAssociator独立CNN+MLP，在已定位分量内预测同complete-room；完整链接合并，未知/冲突阻止合并。每房间对成员预测区域求并集，允许MultiPolygon，不填未观测洞。保留原局部几何及来源。
- 结构输出用房间分隔边界图，含内部墙与门洞处虚拟分隔；开口单独挂接，不用union外轮廓代替内部结构。证据图、位姿图、可达关系分开，孤立点和多分量都输出。
- 用户已明确允许几何开发集校准，不再必须等人工审核；也允许使用经审计GT位姿及confirmed房间归属离线训练独立辅助模块。主匹配器仍几何弱监督，原视觉unknown及原pose_training掩码保持原样，新监督独立记录来源、源缓存指纹、有效mask。GT不得进入推理、选边、聚类、锚点或主分量选择。
- 位姿辅助正例：双向平移<=0.25相机高度单位、yaw<=10度、scale<=10%；负例：任一平移>=0.5/yaw>=20度/scale>=20%；中间忽略。房间监督只用confirmed且预测位姿合格样本。
- 原house级fit/dev不变，dev seed123稳定排序分成模型选择/校准各半。默认AdamW lr=wd=1e-4，clip1，matcher batch2最多20000步，辅助batch32各最多5000步，500步验证、5次不改善停；200步仍为smoke不能冒充正式权重。
- 校准几何开口/位姿precision>=95%、明确负例FPR<=1%；房间合并precision>=98%；不可行就报告calibration_infeasible，不放宽或伪造。所有校准绑定权重、划分、配置并报告数量与置信区间。
- 输出floor_result.json、pair_results.json、adjacency_pred.json、components.json、floor_bev.svg/png、floor_report.md；评估独立输出metrics.json/csv。
- 主分量按节点数/边质量和/稳定ID确定，锚点最小ID。S尺度轨道仅evaluator通过锚点一次尺度/旋转/平移规范；不能逐房或逐分量GT对齐。缺米制尺度，依赖cm和cm阈值的指标N/A。
- 七项：全目标区域IoU；非锚观测50cm/10度正确定位覆盖；所有有效非锚观测平移/圆周旋转中位误差；Wall最大基数一对一(端点20cm/方向5度/重叠>=.5)；Junction一对一10cm；墙距双向长度加权点到线段、2cm采样、200cm截断。预测GT分别graphify(5cm吸附/交点切分/去重/3度度二共线简化)。缺失目标留分母，失败不能删样本。另报房间IoU>=.5一对一恢复、重复房间、纯组装覆盖、mean/P90和计数。
- 验收：合成完美/1m偏移/删房/单位/镜像/端点翻转/不同尺度/坏闭环/桥边/多分量/打乱顺序；训练梯度更新恢复；真实20图190对；GT文件访问拦截与旧缓存哈希不变；跨房屋验证。未校准或全孤立不称已验证多房间重建。布局几何修正属于研究点二，不混入默认实验。

环境：工作目录 /home/feixia/pythonProject/AAAsrk/Bi_Layout。Python /home/feixia/anaconda3/envs/bi_layout/bin/python，torch1.12，无pytest，使用unittest；线程2，OPENBLAS/OMP/MKL线程1；RTX4060Ti16GB。数据src/dataset/ZInD-FloorSet-v1/train_20261021；原图/home/feixia/pythonProject/AAAsrk/zind/data；20图冻结缓存src/output/research1_floor_frontend/smoke_train_0665_floor03_20261022；旧双图结果src/output/research1_opening_pairs/smoke_train_0665_floor03_20261023。

保留所有既有未提交改动，不reset/clean、不覆盖旧权重和数据、不提交推送。复用已有结果；记录真实未完成工作，不编造效果或满足指标的数值。不要另建定时任务，不发送外部消息，不调用其他代理。详细结果写入新的实验目录和docs/research1_code_cleanup_audit.md（测试与清理审计分支），最后用中文报告代码、测试、产物和剩余问题。
