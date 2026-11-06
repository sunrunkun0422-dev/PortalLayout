修改记录：2026-10-29（Asia/Shanghai）—新增 DAV2 米制深度适配、单楼层贯通验证、真实失败结果及复跑命令。

# SALVe + Depth Anything V2 适配实验

**替换和单楼层执行链已完成；重建质量失败。不是原版 SALVe 论文复现结果。**

入口：[分量平面图](reconstruction/floorplan.png) / [SVG](reconstruction/floorplan.svg) / [详细结果](reconstruction/summary.json)。每个分量独立坐标系，图中并排位置不代表真实空间关系。

## 实际运行结果

固定 building `0007`、`floor_02`，8 张全景；在评分前选定，用于规模可控的冒烟实验，没有依据效果选楼层。候选生成同时产生了本建筑另一楼层的数据，但只有 floor_02 进入深度、渲染、验证和重建。

| 阶段 | 结果 | 证据 |
|---|---|---|
| DAV2 室内米制深度 | 8 张 512×1024 uint16 毫米径向深度 | `depth/0007/manifest.json` |
| W/D/O 几何候选 | 63 个预测对齐；另有 6 个精确 GT 项未送入验证器 | `hypotheses/0007/floor_02/` |
| 官方 RGB BEV 渲染 | 252 张图，63 个完整 ceiling1/ceiling2/floor1/floor2 四元组 | `bev/`、`render.log` |
| 官方 ResNet-152 预训练验证器 | 63 个全部评分，无不完整四元组 | `verifier/audit.json` |
| 固定阈值 0.93 | 接受 4 个假设，对应 2 条不同全景对边 | `verifier/batch_0.json` |
| 原版 PGO 算法 + 原版置信度形状融合 | 两个 2 节点分量；4 个孤立节点保留 | `reconstruction/summary.json` |
| 几何适配测试 | 2 passed：已知球面距离、透视采样方向/全景接缝与极点 | `geometry_tests.log` |

**接受的 4 个假设均被官方候选标签标为错误。** 原始相对旋转误差约 89.8°–179.7°，不是单纯可视化偏移。`connected_panos=4` 只说明图中连边，不能解释为正确定位了 4 张全景。本次没有报告整层 IoU 或宣称优于任何基线。

该结果说明直接替换深度后，冻结验证器在这个样例上不能可靠拒绝错误对齐；不能单凭一个楼层断言是 DAV2 模型本身造成，需要分开核查深度尺度、透视域差异、BEV 纹理分布和几何候选质量。

## 替换细节与可比性

- 原 HoHoNet 配置和已有复现目录保留。新深度全部写在此目录，不混入原版缓存。
- 模型：[Depth-Anything-V2-Metric-Hypersim-Small 官方权重](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small)。使用米制室内模型，不能把本机已有的相对深度权重直接当米制深度。
- DAV2 本地源码 commit `e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1`；权重 SHA256 在 depth manifest 中。使用 `dav2` 环境 GPU 推理；SALVe 使用原独立环境 CPU 验证器及 GTSAM 后端。
- 全景转成六个 90° FOV 透视面，每面 512×512。网络预测光轴 z 深度，再除以视线与该面光轴夹角的余弦，转换为球面径向距离。每个 ERP 像素选择最接近光轴的面，不进行学习式融合，面接缝是已知限制。
- ERP 水平周期采样，垂直极点 clamp，避免天顶与地底互相环绕。测试发现并修复了此极点问题，全部深度已重算。
- 深度不使用 GT 尺度或 GT 深度；沿用 SALVe renderer 的固定 1.5 m 相机高度/平移尺度假设。米制模型的尺度偏差与实际拍摄高度差异仍可能影响外观对齐。
- 验证器权重保持 `1200ffbe47d836557d88fef052952337.pth`，不重新训练，不调低 0.93 阈值。
- 后端调用官方 `greedily_construct_st_Sim2`、消失角校正、`planar_slam`、`group_panos_by_room`、`refine_predicted_shape`。适配封装逐个处理所有连通分量，而非只保留最大分量；孤立节点保持独立。
- 拼接用 MHNet 预测 dense boundary 和 uncertainty。PGO 单位按相机高度归一化；转换到官方 stitching 内部尺度 `DEFAULT_CAMERA_HEIGHT=0.4042260417272217`，顺时针角度取 `-theta`。输出 GeoJSON 是局部内部坐标，不是地理坐标或真实米制定位。
- 官方 loader 读取 ZInD 注释获取节点/元数据，候选按官方流程用 GT 分目录标注；后端读取标签只定位候选文件。GT 位姿和候选标签不参与接边选择、PGO 或形状融合；最后独立读取 GT 做错误诊断。
- 修复官方 BEV 代码两处未定义函数引用，补丁见 `render_fixes.patch`。这属于运行修复，不改变几何公式。

## 复跑命令

在 Bi_Layout 根目录执行。若修改模型、投影或尺度，必须使用新实验目录并重建 BEV，不能复用旧缓存。以下命令对应当前固定配置。

```bash
# 1. 替代深度
/home/feixia/anaconda3/envs/dav2/bin/python tools/salve_dav2_depth.py \
  --checkpoint ../salve/checkpoints/depth_anything_v2/metric_hypersim_vits.pth \
  --images ../zind/data/0007/panos --floor floor_02 \
  --output experiments/salve_dav2_20261029/depth/0007

# 2. 官方候选生成；此目录已存在时脚本会拒绝覆盖，复跑需选新目录
MPLBACKEND=Agg ../salve/.venv/bin/python tools/reproduce_salve_hypotheses.py \
  --raw-root ../zind/data --predictions-root ../salve/data/ZInD_HorizonNet_predictions \
  --building-id 0007 --output experiments/salve_dav2_20261029/hypotheses

# 3. 预计算深度 -> 官方 RGB BEV；缺深度会明确报错，不回退到 HoHoNet
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
../salve/.venv/bin/python tools/salve_adapted_render.py \
  --salve-root ../salve --raw-root ../zind/data --run experiments/salve_dav2_20261029

# 4. 冻结官方验证器
../salve/.venv/bin/python tools/salve_adapted_verifier.py \
  --salve-root ../salve --bev-root experiments/salve_dav2_20261029/bev \
  --output experiments/salve_dav2_20261029/verifier

# 5. PGO、预测布局融合、分量可视化、事后错误诊断
MPLBACKEND=Agg ../salve/.venv/bin/python tools/salve_adapted_backend.py \
  --salve-root ../salve --raw-root ../zind/data \
  --predictions-root ../salve/data/ZInD_HorizonNet_predictions \
  --run experiments/salve_dav2_20261029

# 几何单测
../salve/.venv/bin/python -m pytest tests/test_salve_dav2_geometry.py -q
```

下一步应先审计已接受错误候选的尺度和纹理，再在独立训练/开发数据上研究验证器适配；本次未进行阈值调参、再训练或正式测试集性能比较。
