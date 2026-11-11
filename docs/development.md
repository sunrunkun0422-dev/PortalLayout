> 修改记录：2026-11-05 19:49 CST - 补充完整回归与真实旧权重单层兼容烟雾检查。

> 修改记录：2026-11-05 19:40 CST - 按计算机视觉工程分层重构，合并当前指南并建立历史来源映射。

# 开发、安装与迁移

## 环境

解释器仍为 `/home/feixia/anaconda3/envs/bi_layout/bin/python`。包名为 `portallayout`，项目目录为 `PortalLayout`，Conda 环境名仍为 `bi_layout`。

```bash
conda activate bi_layout
python -m pip install --no-deps --no-build-isolation -e .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

本轮补装 `pytest==8.3.5` 及其缺失依赖，并写入开发依赖；未升级模型运行依赖。已有 `requirements.txt` 是历史环境约束，PyTorch/CUDA 的安装仍以实际运行环境及上游说明为准。本次验证可编辑安装，没有宣称验证全新机器安装或独立 wheel 部署。

## 代码落点

| 旧位置 | 新位置 |
| --- | --- |
| `models/` | `portallayout/models/` |
| `dataset/` | `portallayout/datasets/` |
| `loss/` | `portallayout/losses/` |
| `evaluation/` | `portallayout/evaluation/` |
| `config/` | `portallayout/config/` |
| `utils/` 的训练/推理 | `portallayout/engine/` |
| `utils/` 的位姿/组装/坐标 | `portallayout/geometry/` |
| `utils/` 的缓存/checkpoint/日志 | `portallayout/utils/` |
| 前处理、后处理、可视化 | `portallayout/` 下同名子目录 |
| `src/config/` | `configs/layout/` |
| `src/dataset/` | `data/` |
| `src/output/` | `outputs/` |

完整逐模块映射见 [module_migration.json](project/module_migration.json)。业务逻辑进入核心包，参数解析与运行编排进入 `tools/`，实验产物留在实验目录。`portallayout/__main__.py` 注册分组 CLI。

## 兼容机制

原根目录包只含导入桥。`portallayout/_compat.py` 根据模块映射把旧路径解析到同一个新模块对象，避免重复类定义、mock 不生效或旧 pickle 无法解析。旧路径只用于兼容；新代码不得继续添加旧根包导入。

`src/` 的三个符号链接和仓库父目录的 `Bi_Layout → PortalLayout` 保留历史绝对/相对路径。历史 shell 脚本与冻结产物中的路径不做批量替换。旧模块的源文件路径已变化；通过旧文件路径直接打开 `.py` 的外部脚本需按迁移表更新。

源码哈希引用已经调整到真实实现文件，不对兼容桥计算算法身份。源码内容和位置变化会生成新的 contract；不伪造历史哈希，不放宽缓存、训练或构建恢复的来源校验。

## 测试与范围

迁移前基线为 434 项测试通过，迁移后 438 项通过；最终验证与命令日志存于 `docs/project/validation/`。新增兼容测试覆盖模块对象一致、旧路径 patch、旧 pickle 类引用、全新进程先导入旧包及资源定位。

真实兼容烟雾检查加载历史 legacy bundle 和 60 层缓存，对其中一层执行 strict 推理：13 张观测、78 对，结果保留 13 个孤立分量，3,163 个源文件验证通过，未读取 GT。该结果仅验证管线能运行，不代表有效连接或精度提升；详见 [烟雾检查](project/validation/real_inference_smoke.json)。

已有未提交改动先随原文件迁移，未 reset 或覆盖为 Git HEAD。结构变更以当前工作区为起点。本次不重训，不替换模型权重，不删除数据或实验。

## Markdown 维护

当前指南在 `docs/`；历史来源在 `docs/archive/`；实验报告保持就地。每次编辑 Markdown 在第一行新增带时间与摘要的修改记录，保留已有记录。

运行 `python tools/audit_project.py --output docs/project/audit` 可重新生成全量 Markdown 处置索引、重复文本分组与静态代码引用候选。静态无引用不等于无用；CLI、动态模型注册和外部使用均需另查。
