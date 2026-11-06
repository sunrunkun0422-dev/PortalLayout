> 修改记录：2026-11-05 19:40 CST - 按计算机视觉工程分层重构，合并当前指南并建立历史来源映射。

# 训练与标定

使用 `bi_layout` 环境，在仓库根目录运行。核心流程位于 `portallayout/engine/`；CLI 位于 `tools/`。既有训练算法和 checkpoint 字段保持原语义。

## 主线顺序

1. `openings predict`：根据单图 YAML、Bi-Layout 和 Opening Head 权重生成预测观测缓存。该独立脚本当前选择 official-train 样本。
2. `floor run prepare`：独立生成开口弱监督，或 pose/room 辅助监督。GT 只存在于监督构造，不写入预测输入。
3. `floor run train --stage pairs`：选择 new/legacy、损失目标与协议 seed；编码器与 matcher 的训练状态写入 checkpoint。
4. `floor run train --stage auxiliary`：使用冻结 pair 模型产生的监督训练辅助任务；pose 与 room 分开记录。
5. `floor run calibrate`：使用独立校准房屋和绑定的开口报告，输出校准结果及 bundle；无可行工作点时保留失败状态。

具体必选参数从原 parser 获取：

```bash
python -m portallayout openings predict --help
python -m portallayout floor preflight --help
python -m portallayout floor run prepare --help
python -m portallayout floor run train --help
python -m portallayout floor run calibrate --help
```

`openings pairs` 是独立 pair 构建/训练/匹配/标定入口。其监督和校准协议与整层训练并不相同，不能因为功能名字相近就删除或互换。`openings encode` 可用于单独导出表征；整层训练也有现场编码路径，独立表征导出不是所有流程的强制步骤。

## 配置与可复现性

- `configs/layout/*.yaml`：单图模型配置，文件内容保留以兼容历史身份。
- `configs/research1_floor_frontend.json`：前端权重与观测设置。
- `configs/research1_opening_representation.json`：表征维度、注意力等。
- `configs/zind_floorset_v1.json`：数据构建协议。

配置中的历史 `src/...` 路径仍通过符号链接解析，避免仅为路径美化改动配置指纹。新运行默认使用规范目录，权重文件仍在 `checkpoints/`。

每次训练必须记录权重、数据、候选、房屋划分和源码身份。源码重构后新缓存/训练 contract 会变化；旧冻结缓存可以按原 contract 读取，但需要完全一致源码身份的 `--resume` 不能强行绕过。不同 contract 应使用新输出目录，不能修改旧 JSON 中的指纹来继续。

## 单图与历史 pair 训练

原单图入口仍为 `main.py`，也可使用：

```bash
python -m portallayout legacy layout --cfg configs/layout/mp3d.yaml --mode train
python -m portallayout legacy train-pairs --help
```

旧模型结构开关必须与训练时一致，改项目名称不改变权重结构。历史超参数、阶段损失和正负结果在 [实施记录](archive/implementation.md) 与 [实验资料](evaluation.md)，本次没有重训。
