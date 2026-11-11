> 修改记录：2026-11-06 10:05 CST — 记录当前 SALVe 候选上游的 GT 依赖和隔离边界。

# SALVe 候选上游审计

以下检查针对本机 `../salve` 工作区，包含本地修改；源码哈希见文末。该审计没有运行候选生成或执行系统对照。

1. `scripts/export_alignment_hypotheses.py` 在预测开口模式下仍读取房屋 `zind_data.json`，通过 GT 房间对象确定全景成员、可见邻接和相对位姿，再把预测候选按 GT 写入 `gt_alignment_approx`／`incorrect_alignment`。因此旧目录路径本身泄露候选标签，不能直接作为推理输入。
2. `salve/dataset/hnet_prediction_loader.py` 的 `load_hnet_predictions` 使用 GT 位姿图枚举全景；`load_inferred_floor_pose_graphs` 再读取 GT 图及其比例，并传入 `convert_to_pano_data`。改用预测 W/D/O 不足以消除 GT 依赖。
3. `salve/dataset/mhnet_prediction.py` 的 `convert_to_pano_data` 先读取 GT 相机高度但随后硬编码为 1.0，同时将 GT 全局变换写进预测 W/D/O 和 pano 对象。不能将这里的 1.0 解释为已知米制高度，也不能假定 GT 变换未被后续步骤使用。
4. `salve/utils/wdo_alignment.py` 的 `align_rooms_by_wd` 可以基于传入的预测对象提出几何候选，但仍需核实其调用路径及对象的每个属性是否完全脱离 GT；单独存在该函数并不证明完整候选管线可用。

后续需要用冻结的无标签图像／预测清单枚举楼层与视图，构造不携带 GT 位姿的预测几何对象，生成中性候选，逐候选记录不支持和失败，再以合成与真实数据分别核实 PortalLayout 和 SALVe 坐标、尺度及相对变换方向。候选渲染和后端需要独立完成同样的输入审计。历史已按标签分类的候选不用于推理，也不通过重命名假装来源无标签。

| 本机源码 | SHA-256 |
|---|---|
| `scripts/export_alignment_hypotheses.py` | `5dc0d77be03787502ed29930a8ad2fe77195530daf3a09964a83ae33573f4c8f` |
| `salve/dataset/hnet_prediction_loader.py` | `bda04aceec799772988dcbc7882e2a65b10e2d746f938a012e09350bce6b2ad9` |
| `salve/dataset/mhnet_prediction.py` | `60e5e060c515896afcc8fa7ee69c7282b2e247ea6e529bc94e2904b54e609b45` |
| `salve/utils/wdo_alignment.py` | `fb39ef174d2350bb58715d57f90fbe9ba2ce5af07d1aafe7f40227b9ecbfea87` |
