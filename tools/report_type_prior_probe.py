"""Generate the Chinese report from measured probe artifacts."""
from tools.validate_type_conditioned_prior import *
from tools.type_prior_probe_evaluation import instances,summarize_pairs

def main():
    out=ROOT/'experiments/type_prior_probe_20260923'
    protocol=json.loads((out/'protocol.json').read_text());sel=json.loads((out/'selection.json').read_text())
    res=json.loads((out/'results.json').read_text());rows=json.loads((out/'candidate_predictions.json').read_text())
    pairs=json.loads((out/'pair_predictions.json').read_text());boot=json.loads((out/'house_bootstrap.json').read_text())
    full=json.loads((out/'full_annotation_counts.json').read_text());views=sel['views']
    diag=list(csv.DictReader((out/'gt_instance_diagnostics.csv').open()))
    pred=torch.load(out/'adapter_predictions.pt',map_location='cpu')
    mon={k:v for k,v in views.items() if v['group']=='monitor'}
    # One common semantic readout permits localization comparisons for untyped A0/A1.
    # This is a separate offline diagnostic, not native semantic output of A0/A1.
    typed={}
    for name,pr in pred.items():
        seed=name.split('_')[1];reference=pred['A3_'+seed]
        shared={k:dict(v,q=reference[k]['q']) for k,v in pr.items()}
        rr,ss=instances(shared,mon,.47)
        typed[name]={'class_agnostic':ss}
        for typ in ['door','opening']:
            subset=sorted([r for r in rr if r['pred_type']==typ],key=lambda r:-r['score'])
            good=np.array([r['tp'] and r['gt_type']==typ for r in subset],int)
            precision=np.cumsum(good)/np.arange(1,len(good)+1) if len(good) else np.array([])
            envelope=np.maximum.accumulate(precision[::-1])[::-1] if len(good) else precision
            denominator=sum(c['type']==typ for v in mon.values() for c in v['candidates'])
            typed[name][typ]=dict(precision=float(good.sum()/max(1,len(good))),recall=float(good.sum()/max(1,denominator)),
                f1=float(2*good.sum()/max(1,len(good)+denominator)),proposal_AP=float((envelope*good).sum()/max(1,denominator)),
                readout='A3 same-seed frozen type readout applied equally; not native A0/A1 semantics')
    write(out/'shared_type_readout.json',typed)
    # House-paired bootstrap for instance localization, including negative proposals.
    rng=np.random.RandomState(123);iboot={};houses=sorted({v['house'] for v in mon.values()})
    denominators={h:sum(len(v['candidates']) for v in mon.values() if v['house']==h) for h in houses}
    for seed in [123,456,789]:
        for base in ['A0_123','A1_'+str(seed),'A2_'+str(seed)]:
            for setting in ['fixed','calibrated']:
                an='A3_'+str(seed)+'_'+setting;bn=base+'_'+setting
                def stat(rs,h):
                    subset=[r for r in rs if r['house']==h];return [sum(r['tp'] for r in subset),len(subset),denominators[h]]
                aa=np.array([stat(rows[an],h) for h in houses]);bb=np.array([stat(rows[bn],h) for h in houses])
                dd=[]
                for _ in range(2000):
                    ix=rng.randint(len(houses),size=len(houses));x=aa[ix].sum(0);y=bb[ix].sum(0)
                    measure=lambda z:np.array([z[0]/max(1,z[1]),z[0]/max(1,z[2]),2*z[0]/max(1,z[1]+z[2])])
                    dd.append(measure(x)-measure(y))
                iboot[an+'_vs_'+base]=dict(metrics=['precision','recall','f1'],ci95=np.quantile(dd,[.025,.975],axis=0).tolist(),houses=len(houses))
    write(out/'instance_house_bootstrap.json',iboot)
    # GT-only union diagnostic exposes loss of instance identity before any model error.
    union={};overlap={}
    for g in ['fit','calibration','monitor']:
        vv={k:v for k,v in views.items() if v['group']==g};pp={k:dict(p=labels(v['candidates'])[0],q=None) for k,v in vv.items()}
        rr,ss=instances(pp,vv,.5)
        union[g]=dict(instances=ss['tp']+ss['fn'],connected_components=ss['tp']+ss['fp'],iou03_recall=ss['recall'],
            warning='GT union offline diagnostic, not deployable oracle or an optimized upper bound')
        overlap[g]=dict(conflict_tokens=sum(int((labels(v['candidates'])[1]==-100).sum()) for v in vv.values()),tokens=len(vv)*256)
    write(out/'gt_union_diagnostic.json',union);write(out/'sample_conflicts.json',overlap)
    # Predetermined width bins; distance cutoffs from fit only, GT local units (not meters).
    distance_edges=np.quantile([float(r['gt_local_distance']) for r in diag if r['group']=='fit'],[1/3,2/3]).tolist()
    strata=[]
    for group in ['fit','calibration','monitor']:
        for dimension in ['width','distance']:
            edges=[8,16,32] if dimension=='width' else distance_edges
            field='gt_width_tokens' if dimension=='width' else 'gt_local_distance'
            for binid in range(len(edges)+1):
                for typ in ['door','opening']:
                    rs=[r for r in diag if r['group']==group and r['type']==typ and int(np.searchsorted(edges,float(r[field]),side='right'))==binid]
                    strata.append(dict(group=group,dimension=dimension,bin=binid,type=typ,n=len(rs),
                        mean_h=float(np.mean([float(r['mean_h']) for r in rs])) if rs else None,
                        low_relative_delta_fraction=float(np.mean([float(r['mean_relative_delta'])<.1 for r in rs])) if rs else None))
    write(out/'stratified_diagnostics.json',dict(width_edges_tokens=[8,16,32],distance_edges_GT_local_units=distance_edges,rows=strata))
    # Relation scope diagnosed on distinct panorama/scope tuples, not counted as new samples.
    scopes=defaultdict(set)
    for group,rs in sel['pairs'].items():
        for r in rs:
            for side in ['a','b']:scopes[r['image_'+side]].add(r['relation_scope'])
    scope_diag=[]
    for scope in ['cross_complete_room','within_complete_partial']:
        for typ in ['door','opening']:
            rs=[r for r in diag if r['group']=='monitor' and r['type']==typ and scope in scopes[r['image']]]
            scope_diag.append(dict(scope=scope,type=typ,n=len(rs),mean_h=float(np.mean([float(r['mean_h']) for r in rs])) if rs else None))
    write(out/'relation_scope_diagnostics.json',scope_diag)
    with (out/'candidate_predictions.csv').open('w') as f:
        fields=['run']+list(next(iter(rows.values()))[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name,rs in rows.items():w.writerows(dict(run=name,**r) for r in rs)
    percent=lambda x:'{:.2f}%'.format(100*x)
    lines=['# 开口类型条件几何先验：小规模验证报告','',
        '实验发起日期：2026-09-23；报告生成日期：2026-09-24。所有表格来自本地实际运行，不混入历史实验分数。','',
        '**判断：证据不足，当前不支持将“door/opening 两个先验标量”作为已验证创新。** A3 在固定阈值下最多多命中 1 个单图实例，未带来任何正样本匹配恢复；重新校准 A0 即可达到与 A3 相同的匹配结果。语义类型可以预测，但类型可预测不等于类型条件先验有增益。','',
        '## 1. 创新点应该怎样定位','',
        '可以保留“预测类型条件下的几何证据可靠性”这个研究问题。模型应从 RGB 预测特征与预测深度判断类型及可信度，软组合先验；不能读取 GT 类型、GT 宽度或 GT 位置完成推理。首轮只改变先验强度，其他模块冻结，是合理的归因起点。','',
        'SALVe 已经利用 windows/doors/openings 提出对齐假设，因此仅添加类别分支并设置不同阈值不构成充分差异。本轮没有训练 SALVe，也没有运行 RGB BEV verifier；当前项目的七通道几何栅格不能当作 SALVe 纹理输入。参考：[SALVe 官方仓库](https://github.com/zillow/salve)、[ECCV 2022 原论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910632.pdf)。','',
        '若后续继续：先解决单图实例候选表示和阈值校准，再把“候选生成”和“SALVe 式候选验证”作为两个独立实验因素。冻结同一候选集合比较 verifier，再冻结 verifier 比较类型先验；不能一次改变二者后归因于类型。','',
        '## 2. 可复现身份与实际资源','',
        '- 分支：`'+protocol['git_branch']+'`；HEAD：`'+protocol['git_head']+'`。未发现适用的 AGENTS.md。初始未提交文件清单保存在 protocol.json，原文件和权重保留。',
        '- checkpoint：`'+protocol['checkpoint']+'`。',
        '- SHA-256：`'+protocol['checkpoint_sha256']+'`。',
        '- 当前 checkpoint 的初始化合同确认 λ0=4、prior_relative_scale=0.1、原阈值 0.47；不是历史 T3 权重。',
        '- train manifest SHA-256：`'+protocol['manifest_sha256']['train']+'`。',
        '- 内部监控 manifest SHA-256：`'+protocol['manifest_sha256']['val']+'`。',
        '- GPU 不可用：NVML driver/library mismatch，PyTorch CUDA error 804。本轮 CPU FP32、4 线程；峰值显存为不适用，不是缺失测量伪填 0。',
        '- 真实 RGB 冻结缓存耗时 {:.2f} 秒；小模块训练及评估主循环 {:.2f} 秒（不含之后 bootstrap、绘图、标注解析和工具等待）。'.format(json.loads((out/'cache_run.json').read_text())['seconds'],json.loads((out/'experiment_runtime.json').read_text())['seconds']),
        '- 冻结主干、Opening Head、Matcher；eval/no_grad，缓存前后所有主干 buffer 逐项一致。未读取旧 teacher_cache 作为当前预测。','',
        '## 3. 标注覆盖、拆分及口径','',
        '|完整 manifest|实际房屋|去重全景|图像对|door 实例|opening 实例|类型冲突 token 比例|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split in ['train','val']:
        s=full[split];lines.append('|{}|{}|{}|{}|{}|{}|{}|'.format(split,s['houses'],s['panoramas'],s['pairs'],s['door']['view_instances'],s['opening']['view_instances'],percent(s['conflict_tokens']/s['total_tokens'])))
    lines+=['','实例计数是“去重全景内的独立候选标注”，同一物理门在不同全景里的观测仍是不同实例；不宣称是跨视点独立物理连接数。完整房屋/全景/图像对的分类覆盖见 full_annotation_counts.json，类型缺失为 0。door/opening 可同时出现在同一图像对，比例不可相加。监控预选名单有 100 屋，最终 manifest 实际只有 92 屋。','',
        '|集合|房屋|图像对|去重全景|door 实例|opening 实例|','|---|---:|---:|---:|---:|---:|']
    for g,s in protocol['sample_counts'].items():
        count=Counter(r['type'] for r in diag if r['group']==g)
        lines.append('|{}|{}|{}|{}|{}|{}|'.format(g,s['houses'],s['pairs'],s['panoramas'],count['door'],count['opening']))
    lines+=['','按 SHA-256(seed=123, house) 固定抽屋，每屋按正/负和 relation_scope 选取样本；训练与校准房屋互斥，独立全景只缓存和训练一次。所有完整单图候选参加监督，不只使用 pair 共享开口。类型重叠冲突 token 使用 ignore；background 只指完整 complete/raw 标注并集外的标注背景，未进行 RGB 可见性人工确认。window 不作为可通行正类。','',
        '主干此前训练见过 fit/calibration 房屋；内部监控此前用于选择 checkpoint，预训练也可能见过这些房屋。本报告不是未见房屋泛化、正式 val/test 或完整拼接结论。','',
        '## 4. 实验配置与对照','',
        'A0 保留原先验；A1 学一个共享 λ；A2/A3 使用相同 515→16 的小特征分支和相同类型辅助交叉熵。A2 通过独立连续函数产生 λ；A3 通过 q_door/q_opening/q_neutral 软组合两类标量和 λ0。A2 类型输出不进入组合。全部 λ 用 8×sigmoid 限制到 (0,8)，初始化为 4。','',
        '每组每种子最多 5 epoch，AdamW lr=0.01、weight_decay=1e-4、batch=8、梯度裁剪 1；原 opening BCE+Tversky 损失与 0.2×类型损失。校准集二值 opening loss 选择 epoch。A1 有 1 个参数，A2 有 {}，A3 有 {}。输入相同，A2/A3 仅差 15 个参数。没有根据监控结果再调参。'.format(res['A2_123']['training']['parameters'],res['A3_123']['training']['parameters']),
        '', 'GT 仅在监督/评价区；推理接口只有预测特征 x、预测 h 与原 logit。新增模块未接入生产代码，可通过 A0 完全关闭。原 expansion_depth 按新概率比例重算并保存，delta=0 时仍为 0。未运行下游几何求解器，因此不声称补回了缺失几何。','',
        '候选提取固定为项目 legacy 圆周连通分量、最小宽度 2 token、最多 12 个；置信度降序、逐 GT 一对一贪心匹配。实例 IoU 使用 circular 1D，主表 0.3，补充 0.5/0.7。Pair R/C/E 复用原 candidate_targets_from_affinity 的 0.3 IoU gating 与全部 GT 对应集合。','',
        'candidate_only 保持原概率 guidance，仅换候选；normal_guidance 把新预测概率同时用于 attention 和候选评分。现有 API 外部 guidance 参数名为 gt，但传入的是预测数组，专项测试已验证。normal_guidance 只指匹配链，未包括位姿/拼接。','',
        '## 5. 固定阈值 0.47：实例与匹配结果','',
        '|组/种子|P|R|F1|FP|door 定位 R|opening 定位 R|pair R|pair C|pair E|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,v in res.items():
        s=v['fixed']['0.3'];p=v['fixed']['candidate_only']
        lines.append('|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|'.format(k,percent(s['precision']),percent(s['recall']),percent(s['f1']),s['fp'],percent(s['door']['localization_recall']),percent(s['opening']['localization_recall']),percent(p['R']),percent(p['C']),percent(p['E'])))
    lines+=['','监控有 364 个单图标注实例、12 个正 pair 和 12 个负 pair。固定阈值下所有组：R=9/12，C=5/9，E=5/12，E=R×C 成立；A 侧召回 12/12，B 侧 9/12。负样本误接受均 0/12，正确对应占所有接受 pair 的 precision=5/8；这不等于单图实例 precision。guidance 两种模式的本次匹配结论一致。','',
        'A3 seed123/789 相对 A0 只增加 1 个 opening 定位命中，door 命中均 69/260。平均候选数 A0/A3 为 2.574/全景，原阈值分位数为 1/2/3/3/6（min/Q1/median/Q3/max）。这属于很弱的局部信号，不满足稳定优于 A1、A2 且下游有效的判据。','',
        '## 6. 校准集相同 FP 上限','',
        '每组都在训练侧 calibration 搜索阈值，以 A0 在 0.47 的 FP 数作为统一上限，在预算内最大化 recall；预算为 {}。这是相同上限，不是各组恰好相同 FP，也不保证监控 FP 一样。校准样本只有 30 张全景，阈值有明显离散性。'.format(res['A0_123']['calibration_FP_budget']),
        '', '|组/种子|阈值|校准 FP|监控 FP|实例 F1|pair R|pair E|','|---|---:|---:|---:|---:|---:|---:|']
    for k,v in res.items():
        s=v['calibrated']['0.3'];p=v['calibrated']['candidate_only']
        lines.append('|{}|{:.2f}|{}|{}|{}|{}|{}|'.format(k,v['calibrated_threshold'],v['calibration_FP'],s['fp'],percent(s['f1']),percent(p['R']),percent(p['E'])))
    lines+=['','**A0 只改变阈值，就从 5/12 提升到 8/12；A3 三个种子也都是 8/12。** 不能把 A3 校准后相对未校准 A0 的提升归因于类型。提高阈值可拆开相邻高概率区间，候选召回未必随阈值单调降低。A1/A2 部分种子为 7/12，差异只涉及 1 个样本，不能据此确立 A3 优势。','',
        '## 7. 类型是否真的代表几何可靠性差异','',
        '|监控类别|实例|平均 relative_delta|平均 h|h<0.5 token 占比（实例均值）|零 delta token 占比（实例均值）|低 relative_delta 实例比例|平均角宽 token|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for typ in ['door','opening']:
        rs=[r for r in diag if r['group']=='monitor' and r['type']==typ];avg=lambda k:float(np.mean([float(r[k]) for r in rs]))
        lines.append('|{}|{}|{:.3f}|{:.3f}|{}|{}|{}|{:.2f}|'.format(typ,len(rs),avg('mean_relative_delta'),avg('mean_h'),percent(avg('negative_prior_fraction')),percent(avg('zero_delta_fraction')),percent(np.mean([float(r['mean_relative_delta'])<.1 for r in rs])),avg('gt_width_tokens')))
    lines+=['','这里“低差”事先定义为区间平均 predicted relative_delta<0.1。opening 的低差比例略高（7/104 vs door 11/260），但均值反而更高；door 被负先验压低的比例更大。不能把 opening 一概视为低深度差类别。fit/calibration 中差异方向也并不一致。door 不代表门板打开，opening 不保证是可见门框或虚拟分界。','',
        '全景最大值归一化相对只用 magnitude_gate 的平均衰减：door {:.3f}、opening {:.3f}（实例均值）。它确实会压低局部较弱峰，但这只是机制和数值诊断，不能证明去掉归一化会提升。'.format(*[float(np.mean([float(r['global_normalization_attenuation']) for r in diag if r['group']=='monitor' and r['type']==t])) for t in ['door','opening']]),
        '', '宽度/距离分层见 stratified_diagnostics.json：宽度是 GT 角宽，仅离线分析；距离是 GT localEndpoints 中点到相机的局部单位距离，不是米，分位阈值只取 fit。relation_scope 分层见 relation_scope_diagnostics.json。同一全景可属于两种 scope，所以该表不能相加为独立样本量。分层样本有限，不能排除角宽/距离/连接范围的混杂。','',
        '|A3 种子|λ_door|λ_opening|选中 epoch|类型 token macro-F1|','|---|---:|---:|---:|---:|']
    for seed in [123,456,789]:
        v=res['A3_'+str(seed)];lam=v['training']['strengths'];lines.append('|{}|{:.3f}|{:.3f}|{}|{}|'.format(seed,*lam,v['training']['selected_epoch'],percent(v['type_macro_f1'])))
    lines+=['','三种子没有学出“opening 应显著弱于 door”的关系，反而 λ_door 更低；seed456 两者几乎相同。类型 macro-F1 是排除冲突后的 token 级指标，不能代替实例类型检测。完整混淆矩阵顺序为 door/opening/background，保存在 results.json。','',
        'A0/A1 没有类别输出，原生分类型 precision/F1/AP 不可定义，报告不拿 GT 给假阳性补类型。原生 A2/A3 类型实例 P/R/F1 已保存在 results.json。为额外提供共同口径，shared_type_readout.json 用同种子 A3 的预测类型只读分配所有组候选，给出分类型 P/R/F1/proposal-AP；该表是共享分类读出的诊断，不是 A0/A1 自身的语义能力，也未用 GT 类型路由。proposal-AP 仅在固定候选集合内按均值概率排序，不能称标准全阈值检测 AP。','',
        '## 8. 标签实例与二值连通分量的结构限制','',
        '将监控 GT 完整二值并集直接作为“完美 token 预测”再走同一候选提取器，只有 {} 个连通分量，对原 {} 个实例的 IoU≥0.3 一对一召回为 {}。这不是可部署 oracle，也不是优化后的理论上界，只说明二值并集会合并接触、重叠或不同深度的区间。实例召回低不能直接等同于 token 检测很差。'.format(union['monitor']['connected_components'],union['monitor']['instances'],percent(union['monitor']['iou03_recall'])),
        '', '这为下一步提供更具体的方向：审计哪些完整/raw 候选是不可见的投影、哪些在 token 轴重叠；考虑圆周端点/中心/实例候选建模，并保持开口几何身份。只缩放已有 logit 的先验项，很难解开已经合并的实例。此建议是根据当前观察提出的下一实验，不是已验证新方法。','',
        '## 9. 不确定性、转移与可视化','',
        '在本轮全部 12 个监控房屋上做 2,000 次配对房屋 bootstrap，同次抽样用于两组。不是完整 92 屋监控的 bootstrap。固定阈值 pair 差异的经验区间为 [0,0]，原因是这 12 屋逐样本输出相同；不能解释为总体差异被精确估计为零。实例指标区间见 instance_house_bootstrap.json；pair 区间见 house_bootstrap.json。','',
        '|A3 seed123 对照|设置|Δ实例 F1 95% CI（百分点）|Δpair E 95% CI（百分点）|','|---|---|---|---|']
    for base in ['A0_123','A1_123','A2_123']:
        for setting in ['fixed','calibrated']:
            ci=iboot['A3_123_'+setting+'_vs_'+base]['ci95'];pc=boot['A3_123_'+setting+'_candidate_only_vs_'+base]['ci95']
            lines.append('|{}|{}|[{:.2f}, {:.2f}]|[{:.2f}, {:.2f}]|'.format(base,setting,100*ci[0][2],100*ci[1][2],100*pc[0][1],100*pc[1][1]))
    lines+=['','固定阈值 A3 seed123 相对 A0：5 个 success→success、4 个 selection_failure→selection_failure、3 个 miss→miss、12 个 reject→reject。漏检恢复、原成功破坏、负 pair 误检均为 0，不伪造这些类别案例。原阈值到校准阈值的全部转移保存在 transitions.json，需注意其中基准统一是固定阈值 A0。','',
        '已按固定哈希规则抽取 12 张真实全景并绘制 GT 区间、A0/A3 概率与预测几何；覆盖 opening、低差 door、跨接缝。未人工确认哪些 opening 是虚拟分界，不把语义标注当成 RGB 证据。案例列表与缺失类别数量分别在 case_selection.json 和 case_category_counts.json。','',
        '![先验分布](prior_distributions.png)','',
        '## 10. 交付、复现与未完成范围','',
        '可执行脚本：tools/validate_type_conditioned_prior.py、tools/type_prior_probe_evaluation.py、tools/report_type_prior_probe.py；风险测试：tests/test_type_prior_probe.py。配置/版本：protocol.json；小模块权重：A1/A2/A3_种子.pt；冻结预测缓存：predictions/；预测候选 CSV/JSON、pair_predictions.json、GT 诊断 CSV、分层诊断、两类 bootstrap 与 12 张案例均在本目录。','',
        '```bash','PY=/home/feixia/anaconda3/envs/bi_layout/bin/python',
        '# 从仓库根目录运行，完整复现时使用新的输出目录',
        '$PY -u tools/validate_type_conditioned_prior.py --stage prepare --output experiments/type_prior_probe_reproduction',
        '$PY -u tools/validate_type_conditioned_prior.py --stage cache --output experiments/type_prior_probe_reproduction',
        '$PY -u tools/validate_type_conditioned_prior.py --stage experiment --output experiments/type_prior_probe_reproduction',
        '$PY -m unittest tests.test_type_prior_probe -v',
        '$PY -m tools.report_type_prior_probe','```','',
        'prepare 禁止覆盖已有实验；cache 可以跳过已存在缓存恢复。报告脚本和集成测试默认读取本次目录。正式复现需同一数据及 checkpoint，并以 protocol 中哈希核对。','',
        '**明确未完成**：没有运行完整 2,352 对内部监控、正式 val/test、SALVe/CNN 训练、位姿求解/完整拼接评估；没有标准多阈值检测 AP、跨全景物理门去重或可见性人工核验。未增加去类型监督消融：当前没有足够稳定的归因信号，不扩大组合。','',
        '建议先把“类型条件几何可靠性”保留为待检验方向，暂不作为论文确定贡献。当前更值得验证的是实例区间重叠/合并与阈值校准，随后才能判断是否需要超出单标量的几何可靠性模型。','']
    lines+=['## 附表：IoU 敏感性与端点误差（固定 0.47）','','|组|IoU|P|R|F1|命中区间端点 MAE/token|','|---|---:|---:|---:|---:|---:|']
    for name in ['A0_123','A1_123','A2_123','A3_123','A3_456','A3_789']:
        for iou in ['0.3','0.5','0.7']:
            x=res[name]['fixed'][iou]
            lines.append('|{}|{}|{}|{}|{}|{:.2f}|'.format(name,iou,percent(x['precision']),percent(x['recall']),percent(x['f1']),x['endpoint_MAE_tokens']))
    lines+=['','端点 MAE 只对当前 IoU 下命中的实例计算，受匹配集合影响，不是无条件边界误差。类别 precision/F1 来自先定位后核对预测类型的关联，不把它冒充标准类感知全阈值 AP。','','7 项风险测试已通过：预测接口无 GT、圆周重叠 ignore、初始输出复现、房屋隔离、冻结状态、guidance 开关与零 delta 几何。测试输出见 tests.log。','']
    (out/'验证报告.md').write_text('\n'.join(lines))
    gallery=['# 固定规则抽样案例','', '以下是本地真实全景，未经虚拟分界人工判定。','']
    cases=json.loads((out/'case_selection.json').read_text())
    for i,k in enumerate(cases['images']):gallery+=['## {:02d} {}'.format(i+1,k),'','![](cases/{:02d}.png)'.format(i+1),'']
    (out/'案例图册.md').write_text('\n'.join(gallery))
    write(out/'source_hashes.json',{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'tools/validate_type_conditioned_prior.py',ROOT/'tools/type_prior_probe_evaluation.py',ROOT/'tools/report_type_prior_probe.py',ROOT/'tests/test_type_prior_probe.py',ROOT/'portallayout/models/cross_scene_matcher.py',ROOT/'configs/layout/zind_all.yaml']})
    print('Report generated:',out/'验证报告.md');print('GT union:',union)
if __name__=='__main__':main()
