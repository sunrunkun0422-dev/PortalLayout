> 修改记录：2026-11-06 09:57 CST — 记录新开口头隔离清单及 MP3D 候选骨干的装载检查。

> 修改记录：2026-11-06 09:34 CST — 补充 Opening Head 训练缓存的确定房屋和图像重叠。

> 修改记录：2026-11-05 21:22 CST — 补充可复用的四角色房屋协议及其来源。

> 修改记录：2026-11-05 21:21 CST — 新增 R01 数据与历史结果审计，记录权重暴露限制和缓存校验。

# R01：基线、数据与权重审计

状态：**diagnostic_only_frontend_exposed**。当前队列按房屋互斥，但 Opening Head 训练缓存已确认覆盖复核房屋和部分复核图像；Bi-Layout 单图权重的房屋暴露仍未知。因此不能认定端到端未见房屋验证通过，也不能据此宣布正式训练就绪。

## 新队列

[固定成员](cohort.json)及[可复用协议](house_protocol.json)：官方训练划分内 70 个不同房屋，每房随机确定一层；训练、选型、校准和复核分别为 40/10/10/10 层，对应 947/278/299/318 张全景。排除已知历史 60 房屋；固定顺序由 SHA256 决定，仅使用房屋、楼层 ID 与观测数量，未按标注或效果挑选。检查的 1,842 张 RGB 均存在。70 份输入清单已按严格 RGB-only schema 验证并复制到 `manifests/`；未将监督写入输入。

警告：旧版 `house_protocol` 没有 audit 角色。复核房屋在旧协议中仍归属 selection 或 calibration，后续训练、监督导出和标定必须显式排除 `cohort.json` 中 audit 房屋，否则会泄漏。为证明端到端未见房屋，还须明确前端权重训练来源，或者使用能证明来源的初始化并按新划分重新训练前端。

## 权重与缓存

[audit.json](audit.json) 给出前端、历史 R3、SALVe 验证器的实际路径和 SHA256。前端和开口权重哈希与历史缓存 contract 一致（[对照](cache_weight_check.json)）；这只能证明文件身份，不能证明训练数据独立。历史缓存 1,551 条元信息引用的 1,551 个张量逐条读取并核验哈希，错误数 0；缓存记录仍绑定重构前源码 contract，不能绕过身份检查强行续跑。

[ObservablePairs 暴露记录](observable_exposure_check.json)：历史联合训练权重对新选型 10 房屋中的 7 个训练、3 个监控；新校准为 9/1，新复核为 7/3。该联合权重不能用于声称这些房屋未见；该结论不等于已确定模块化前端权重的暴露范围。

[Opening Head 暴露审计](opening_exposure_audit.json)：其 checkpoint 中训练缓存合同和匹配训练 manifest 的哈希与当前文件一致，训练选择未限制视图；记录的训练步数与 21 轮遍历 21,840 个缓存视图一致。该缓存覆盖复核集 10/10 个房屋，其中复核楼层 318 张图中的 126 张也在训练缓存；选型 10/10 房屋、278 张图中 111 张在训练缓存。该证据针对 Opening Head，不能用于推断布局模型权重的训练名单。复算使用 `tools/research10_opening_exposure.py --output <新文件>`；审计未读取官方测试集或 RGB 内容。

已构建[隔离开口头清单](isolated_opening_manifests/provenance.json)：仅含 fit 40 房屋的 685 图和 selection 10 房屋的 196 图，校准及复核房屋未进入；两份清单通过现有数据读取器的路径检查。候选 MP3D 布局权重经[架构装载检查](mp3d_candidate_smoke.json)无缺失或多余参数；本地日志指向 Matterport 训练，但完整初始化链尚未通过哈希验证，不能据此宣称骨干未见 ZInD 房屋。新 Opening Head 尚未训练；后续必须禁用被污染的 warm start。

## 历史指标复核

根据固定逐图像对 CSV 重新累计，历史 selection R3 跨房候选中，原几何 30/711、新几何 81/711；已知弱对应分别 47/203 和 182/203。它们是候选可解上限，不是正确接边率或楼层重建率。

历史 ObservablePairs 预测开口的内部监控集：正负各 1,176 对，候选召回 80.36%、条件 Top-1 75.13%、总体 Top-1 60.37%、负样本拒绝 93.03%。数值读取原始训练指标，范围为内部监控，并非官方测试。历史 R3 校准状态仍为不可行，人工审查为 7 正确、3 不确定、86 待审。

## 重现与测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/feixia/anaconda3/envs/bi_layout/bin/python tools/research10_r01.py \
  --output experiments/portallayout_design10_20261105/R01_reproduction
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/feixia/anaconda3/envs/bi_layout/bin/python -m pytest -q tests/test_research10_r01.py
```

脚本拒绝覆盖非空目录；测试覆盖队列互斥和确定性、样本不足、缓存损坏和候选分母。新队列尚未运行前端预测或训练。本报告的验证不等同于 R02–R10 的完成。
