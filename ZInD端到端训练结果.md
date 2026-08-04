# ZInD 端到端联合训练结果

> 记录日期：2026-08-04
>
> 对应代码提交：`936d0a73f52008c7909f5946e423306b17fcce7e`
>
> 训练入口：[`tools/train_zind_end_to_end.py`](tools/train_zind_end_to_end.py)

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
