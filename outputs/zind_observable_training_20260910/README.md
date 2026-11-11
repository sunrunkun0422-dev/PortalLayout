# ObservablePairs 端到端联合训练

> 训练已于 2026-09-12 完成：共 21 轮，最佳第 15 轮，Top-1 60.37%。下面运行中快照保留用于追溯；请以 [最终结果摘要](训练结果摘要.md) 为准。

状态快照：2026-09-11T03:32:49.660270+08:00。已完成 6 轮；后台继续训练。实时状态以 [job_status.json](job_status.json) 和 [train.log](train.log) 为准。

## 实验设置

- 数据：ZInD-ObservablePairs-v1 / train_auto_v2，每个连接最佳代理通过视点，按关系类型选等量可靠负样本；unknown 不用于二分类监督。
- 训练 27,354 对（13,677 正 + 13,677 负）；内部监控 2,352 对（1,176 正 + 1,176 负）。
- 内部监控来自 100 个官方训练房屋，与本轮训练房屋互斥，并排除人工校准试点房屋。不是正式 val；预训练权重可能见过这些房屋，不能据此报告未见房屋泛化。
- 完整保留多个 GT 对应；单图开口监督使用 complete/raw 的一致并集。开口候选漏掉真实目标时屏蔽该正样本的候选分类损失，不改成负样本。
- 沿用 T3 的联合优化配置：FP32，batch 4，24 轮上限，patience 6，候选排序权重 0.1。初始化使用原模块化三个 checkpoint，不接着旧端到端优化器训练。
- Bi-Layout、Opening Head、Matcher 的梯度预检均通过，峰值显存 10.95 GiB。M6–M9 分析式几何不参与反向传播。
- 正式验证集和测试集未使用；所有新标签仍为 auto_proposed / pending_review。

## 当前最佳内部监控结果（第 5 轮）

| 指标 | 数值 |
|---|---:|
| 预测候选端到端 Top-1 | 57.40% |
| 正样本候选召回率 | 81.12% |
| 条件 Top-1 | 70.75% |
| 负样本拒绝率 | 93.45% |
| Harmonic | 71.12% |
| Opening token F1 | 90.81% |

这些是运行中的内部监控指标，不是最终测试结果，也不能与旧数据集分数直接作提升比较。

## 持续运行与产物

进程使用独立会话和本地日志，忽略 SIGHUP；不依赖网络连接。网络或当前对话断开不会终止后台任务。进程异常时最多自动尝试 3 次：缓存从完成行继续，训练从 last.pt 的完整 epoch 继续。主机掉电或重启不会自动重启进程；可用下述命令恢复，最近未保存的 epoch 会重跑。

```bash
cd /home/feixia/pythonProject/AAAsrk/Bi_Layout
nohup /home/feixia/anaconda3/envs/bi_layout/bin/python -u -B tools/run_zind_observable_training_job.py \
  --config src/output/zind_observable_training_20260910/job_config.json \
  >> src/output/zind_observable_training_20260910/supervisor.log 2>&1 < /dev/null &
```

运行锁防止重复训练。当前任务仍在运行时不要重复启动。

- 权重目录：`checkpoints/End_To_End/zind_observable_v1/best1_balanced_seed123_20260910`，best.pt 为最佳模型，last.pt 为优化器/调度器等恢复状态。
- [精确命令](job_config.json)、[代码哈希](code_hashes.json)、[梯度预检](gradient_preflight.json)、[25 项接入与回归测试](adapter_tests.log)。
- 正常完成后自动生成 COMPLETED.json 和训练完成报告.md。未出现这些文件不代表训练已经结束。
