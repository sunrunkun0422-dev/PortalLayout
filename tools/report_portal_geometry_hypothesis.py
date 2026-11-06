"""Render the measured geometry-attribution experiment and audit its limits."""
import json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.validate_portal_geometry_hypothesis import DEFAULT,stats,endpoint_error
from tools.validate_type_conditioned_prior import write,digest,stable
from portallayout.geometry.candidate_local_portal_geometry import panorama_ray,bi_to_zind

LABELS={'B_raw':'原预测端点','B_existing_M2':'已有纯预测墙线拟合 M2','O_angle':'GT 端点角位置＋预测深度','O_direction':'仅 GT 线段方向','O_midpoint':'仅 GT 中点位置','O_width':'仅 GT 宽度','O_radial':'GT 端点径向距离＋预测射线','O_support_line':'GT 支撑线＋预测射线','O_full':'完整 GT 端点'}

def main():
    out=DEFAULT;s=json.loads((out/'summary.json').read_text());rows=json.loads((out/'per_pair.json').read_text());protocol=json.loads((out/'protocol.json').read_text());boot=json.loads((out/'house_bootstrap.json').read_text())
    sel=json.loads((out/'selection.json').read_text());correct=[r for r in rows if 'O_full' in r['conditions']];n=len(correct)
    common=[r for r in correct if r['conditions']['B_existing_M2']['valid']]
    supplemental={'M2_common_valid':{c:stats(common,c) for c in ['B_raw','B_existing_M2']},'pose_only_counts':{},'joint_overall_counts':{},'threshold_sensitivity':{}}
    for c in LABELS:
        supplemental['pose_only_counts'][c]=sum(r['conditions'].get(c,{}).get('pose_success_without_scale',False) for r in correct)
        supplemental['joint_overall_counts'][c]=dict(correct_correspondence_and_pose_success=sum(r['conditions'].get(c,{}).get('success',False) for r in correct),denominator_all_positive_pairs=len(rows))
        supplemental['threshold_sensitivity'][c]={}
        for yaw,t,scale in [(5,.25,.1),(10,.5,.1),(15,1.,.2)]:
            count=sum(v.get('valid',False) and v['yaw_deg']<=yaw and v['translation_m']<=t and v['relative_scale_error']<=scale for v in [r['conditions'].get(c,{}) for r in correct])
            supplemental['threshold_sensitivity'][c][str((yaw,t,scale))]=count
    # Pose-only success bootstrap shows the effect is not only a strict scale cutoff.
    rng=np.random.RandomState(123);sensitivity_boot={}
    for c in ['O_direction','O_radial','O_support_line','O_full']:
        delta=np.array([int(r['conditions'].get(c,{}).get('pose_success_without_scale',False))-int(r['conditions']['B_raw']['pose_success_without_scale']) for r in correct])
        samples=delta[rng.randint(n,size=(2000,n))].mean(1)
        sensitivity_boot[c]=np.quantile(samples,[.025,.975]).tolist()
    supplemental['pose_only_paired_bootstrap95']=sensitivity_boot
    supplemental['annotation_consistency_residual_m_quantiles']=np.quantile([r['GT_endpoint_transform_residual_m'] for r in rows],[0,.5,.9,1]).tolist()
    supplemental['GT_full_failure_causes']=dict(Counter('scale_only' if r['conditions']['O_full']['pose_success_without_scale'] else 'pose_and_or_scale' for r in correct if not r['conditions']['O_full']['success']))
    supplemental['type_scope_cross_tab']=dict(Counter('/'.join(r['type_pair'])+'|'+r['scope'] for r in correct))
    write(out/'supplemental.json',supplemental)
    cache_run=json.loads((out/'cache_run.json').read_text());run=json.loads((out/'run.json').read_text())
    pct=lambda x:f'{100*x:.2f}%'
    lines=['# 开口—墙面几何增强命题：验证性实验报告','',
        '日期：2026-09-27。使用当前 ObservablePairs checkpoint 的真实 RGB 预测；没有训练或虚构新模型结果。','',
        '**结论：初步支持“开口端点几何是跨房间对齐的重要改进对象”；不支持“只把墙面方向修正或做局部线拟合就足够”；“学习开口—墙面关联能提升双布局估计”仍然证据不足。**','',
        '在 54 个已选对开口对应的房屋上，保留预测射线、只使用对应 GT 端点的径向距离，旋转/平移/尺度联合达标数从 4/54 提升到 28/54；完整 GT 端点为 48/54。这是含 GT 辅助的离线归因，不是可部署模型收益。更值得优先研究的是开口端点的位置、距离和宽度，而非只强化方向约束。','',
        '## 1. 本轮检验了什么','',
        '上一轮类型先验只修改 opening logit，本轮直接干预送入位姿求解器的几何。命题拆成三层：','',
        '|子命题|本轮判断|理由|','|---|---|---|',
        '|单图开口几何误差会限制下游对齐|初步支持|固定正确对应后，端点径向距离干预明显改善位姿|',
        '|仅靠承载线方向或局部贴墙即可解决|当前不支持其充分性|方向单独干预的联合收益很小；已有 M2 实际更差|',
        '|预测开口—墙面关联的学习模块能提高单图双布局|证据不足，未验证|本轮没有训练该模块，也未比较联合监督与同容量独立细化|','',
        '完整 GT 端点改善本身带有几何求解的必然性，不能单独作为创新证据。更有信息的是分量干预、纯预测对照和不同失败类型的差异。','',
        '## 2. 数据、模型与冻结合同','',
        f'- 当前 HEAD：`{protocol["git_head"]}`。原有工作区、checkpoint、旧实验均保留；新代码和产物独立存放。',
        f'- checkpoint：`{protocol["checkpoint"]}`。',
        f'- checkpoint SHA-256：`{protocol["checkpoint_sha256"]}`。',
        f'- manifest SHA-256：`{protocol["manifest_sha256"]}`。',
        '- 内部监控实际覆盖的 92 个房屋各选 1 个正 pair，共 92 对、184 张全景。按 seed123 哈希选样，relation_scope 交替优先；不按预测结果选样。不是全量 2,352 对监控。',
        '- 24 张缓存经 checkpoint 和 RGB 哈希验证后复用；其余 160 张从 RGB 用冻结模型 CPU FP32 重新预测。没有使用历史 T3 特征或指标。',
        f'- 新缓存耗时 {cache_run["seconds"]:.2f} 秒；归因前向与逐样本评价耗时 {run["seconds"]:.2f} 秒（不含统计图表与额外审计）。4 CPU 线程；未占用 CUDA 显存。',
        '- Bi-Layout、Opening Head、Matcher 全部 eval/no_grad；新缓存过程检查主干所有 buffer 未变。阈值固定 0.47，legacy 圆周连通候选最小宽度 2 token、最多 12 个。未重调阈值。',
        '- 本轮基线是项目当前的 Bi-Layout＋已训练 Opening Head＋Matcher 系统，不是原论文裸 Bi-Layout。内部监控参与过模型选择，预训练可能见过这些房屋；不能当作未见房屋泛化。','',
        '## 3. 坐标、求解器及 GT 使用边界','',
        '所有预测轮廓和端点统一到 camera-height-normalized ZInD xy。GT 端点除以各自 camera_height，评价变换为 inv(D_A)·T_local·D_B，平移米制因子为 a_local_to_meters×height_A。实测本批所有 camera_height=1，但仍实现并测试了不等高度的通用换算。尺度因子仅用于离线单位和评价，不是从 RGB 推出的绝对尺度。','',
        '统一复用 utils.joint_layout.build_joint_layout_from_endpoints：端点相似变换、宽度比标定尺度、预测轮廓重心异侧规则选择方向。所有组都使用同一预测 enclosed 轮廓做方向消歧；GT 位姿、GT 房间多边形不进入求解器。','',
        '先完成候选和 Matcher 前向，再用全部 GT 对应集合进行离线命中检查。只有固定预测最佳 pair 已经正确的样本才接受主表几何干预；GT 关联按双侧圆周 IoU≥0.3、IoU 排序选定，绝不按位姿误差选择。','',
        '## 4. 候选和匹配漏斗','',
        '|阶段|数量|','|---|---:|',
        f'|正样本总数|{s["counts"]["total"]}|',f'|预测候选包含正确对应|{s["counts"]["recalled"]}|',
        f'|Matcher 接受|{s["counts"]["accepted"]}|',f'|Matcher 选对对应，进入主归因|{s["counts"]["oracle_eligible"]}|','',
        '共有 21 对候选漏检，另有 17 对候选命中但选择不正确/未接受。几何替换不修复这些样本，不能把条件子集上的达标率称为端到端全样本达标率。本轮只采正样本，没有评估负样本拒绝率或分类 precision。','',
        '## 5. 对照与干预定义','',
        '|组别|替换内容|GT 是否参与|','|---|---|---|',
        '|B_raw|固定预测区间的两端射线×预测 enclosed 深度|否|',
        '|B_existing_M2|仓库已实现的候选局部预测墙线拟合，再与两端射线求交|否；它是已有对照，不是本轮新方法|',
        '|O_angle|GT 端点角位置，沿该方向对预测深度做圆周插值|是，仅离线|',
        '|O_direction|绕预测中点旋转到 GT 开口线段方向；预测中点和宽度保持不变|是，仅离线|',
        '|O_midpoint|GT 中点；保留预测方向和宽度|是，仅离线|',
        '|O_width|GT 宽度；保留预测中点和方向|是，仅离线|',
        '|O_radial|保留预测端点射线，使用角度关联后的 GT 端点径向距离|是，仅离线|',
        '|O_support_line|预测射线与 GT 端点确定的支撑直线求交|是，仅离线|',
        '|O_full|完整 GT 端点|是，仅离线|','',
        '**分量解释限制：**O_midpoint 同时涉及径向和切向位置，不是纯深度。O_radial 改变后会连带改变线段方向/宽度，并且 GT 端点距离不等于预测射线上真实场景深度；它测试端点距离误差机会，不是独立因果深度分量。O_support_line 同时替换支撑线方向和偏移；预测射线与直线平行或反向时计失败。','',
        'GT 开口线段方向只是支撑方向的代理，不能自动当作真实实体墙法向。本批 54 对的 GT 端点在标注 raw/complete 边上均能找到 0.05 归一化单位内的支撑，但这不证明 opening 在 RGB 中有实体墙或可见门框。','',
        '## 6. 主结果：固定正确匹配的 54 对','',
        '运行前固定联合达标条件：旋转误差≤10°、平移误差≤0.5m、相对尺度误差≤10%。几何失败计不达标；误差中位数只在该组有效输出上计算，不能用有效子集掩盖失败。','',
        '|条件|有效数|旋转中位数/°|平移中位数/m|相对尺度误差中位数|联合达标|仅旋转＋平移达标|','|---|---:|---:|---:|---:|---:|---:|']
    for c,label in LABELS.items():
        v=s['correct_match_subset'][c]
        lines.append(f'|{label}|{v["valid"]}/54|{v["yaw_deg_median"]:.3f}|{v["translation_m_median"]:.3f}|{pct(v["relative_scale_error_median"])}|{v["success"]}/54|{supplemental["pose_only_counts"][c]}/54|')
    lines+=['','完整 GT 端点的 6 个联合失败全部来自尺度误差>10%；去掉尺度条件后为 54/54。不能把 4→28 的联合收益全部解释为位置：只看旋转和平移，径向干预仍从 14/54 提升到 39/54，方向干预为 22/54。方向并非没有价值，只是单独不足。','',
        '对应到全部 92 对，原始“正确对应且联合位姿达标”为 4/92；保持检测与匹配不变、仅在已正确对应中进行径向 oracle 后为 28/92，完整端点 oracle 为 48/92。这仍是 GT 辅助诊断。','',
        '已有 M2 在共同有效的 48 对上，原始端点与 M2 的旋转中位数为 {:.3f}° / {:.3f}°，平移为 {:.3f}m / {:.3f}m，联合达标 4/48 / 0/48；因此退化并非仅由 M2 的 6 个几何失败造成。'.format(supplemental['M2_common_valid']['B_raw']['yaw_deg_median'],supplemental['M2_common_valid']['B_existing_M2']['yaw_deg_median'],supplemental['M2_common_valid']['B_raw']['translation_m_median'],supplemental['M2_common_valid']['B_existing_M2']['translation_m_median']),
        '', '## 7. 配对房屋 bootstrap 与转移','',
        '主表每对来自不同房屋，对全部 54 个进入归因的房屋做 2,000 次配对 bootstrap，seed123。下表是相对 B_raw 的差值；多个探索性干预没有做多重检验校正，区间用于描述本轮信号，不能当作确证性检验。','',
        '|条件|联合达标率差值 95% CI/百分点|平移中位数差值 95% CI/m|原失败→成功|原成功→失败|','|---|---|---|---:|---:|']
    for c in list(LABELS)[1:]:
        ci=boot[c]['ci95'];tr=s['transitions'][c]
        lines.append('|{}|[{:.2f}, {:.2f}]|[{:.3f}, {:.3f}]|{}|{}|'.format(LABELS[c],ci[0][0]*100,ci[1][0]*100,ci[0][2],ci[1][2],tr.get('failure -> success',0),tr.get('success -> failure',0)))
    lines+=['','对于径向距离替换：24 个原失败恢复、0 个原成功破坏；方向替换仅恢复 1 个联合失败。完整 GT 端点也有 1 个原成功变失败，原因仍是 GT 两侧宽度不完全一致导致的尺度估计，不能只展示恢复案例。','',
        '## 8. 类型/连接范围与标注限制','',
        '|子集|数量|原始联合达标|仅方向|仅径向距离|GT 支撑线|完整 GT 端点|','|---|---:|---:|---:|---:|---:|---:|']
    for typ in ['door','opening']:
        v=s['by_type'][typ];lines.append('|{}|{}|{}|{}|{}|{}|{}|'.format(typ,v['B_raw']['n'],*[v[c]['success'] for c in ['B_raw','O_direction','O_radial','O_support_line','O_full']]))
    lines+=['','**本批正确匹配子集中，34 对 door 恰好全部属于 cross_complete_room，20 对 opening 全部属于 within_complete_partial。类型和连接范围完全混杂，不能把上述差异归因于类别，也不能推出“door 更适合某先验”。**','',
        '按 GT 相对变换投影两侧标注端点后的最小配对平均残差，中位数 {:.3f}m、最大 {:.3f}m；这些端点不是严格同一个物理线段的无误差测量。对不同房间侧的门洞，标注位置/宽度差异会给端点求解引入误差底限。'.format(supplemental['annotation_consistency_residual_m_quantiles'][1],supplemental['annotation_consistency_residual_m_quantiles'][-1]),
        '', '## 9. 独立的 GT 对应诊断（不与主表混算）','',
        '为分开检测/匹配与几何，在全部 92 对上额外选取字典序第一组 GT 对应：使用 GT 角位置＋预测深度，联合达标 5/92；完整 GT 端点为 80/92。没有按最低位姿误差选 GT，也没有将该表当作推理结果。这说明即便给出正确对应与端点角位置，预测径向几何仍有明显改进空间。','',
        '## 10. 对研究选题的具体建议','',
        '优先将命题收窄为“面向跨房间对齐的开口端点径向几何与宽度联合细化”。保留开口—墙面关系作为可检验的结构约束，不要先假设纯贴墙或方向正则有效。已有 M2 的退化要求后续学习模块能识别错误支撑、角点和不同结构，而非重复局部直线拟合。','',
        '下一轮才比较四组学习模型：同一 Bi-Layout 初始化、同一训练房屋及标注下，①仅辅助开口监督；②同容量无结构关系的端点/深度细化；③普通局部几何正则；④预测开口—墙面关联驱动细化。必须同时报告全局双布局质量、端点距离/宽度、固定匹配器下的位姿指标；使用未进入本轮诊断的独立评估划分，不能反复在这 92 对上选型后宣称泛化。','',
        '本轮不支持直接写“新模型有效”或“创新点已成立”。它支持继续研究的具体瓶颈，也提供了一个负面结果：已有纯预测局部墙线拟合不足以解决问题。','',
        '## 11. 交付、复现与风险测试','',
        '- protocol.json / selection.json：抽样与实验设置；per_pair.json / pose_predictions.csv：逐样本干预和位姿结果。',
        '- summary.json / supplemental.json / house_bootstrap.json：所有指标、阈值敏感性和区间；per_opening.json：端点误差诊断。',
        '- predictions/：184 张真实全景的冻结预测；cases/ 与 案例图册.md：固定规则抽取 12 例，含恢复、未恢复、原成功被破坏和不同连接范围。',
        '- 五项专项测试通过：不同相机高度坐标换算、分量替换不变量、圆周接缝插值、求解器无 GT 位姿入口、端点反序不改变已知解。合成测试只验证代码，不作为真实实验结果。',
        '', '```bash', 'PY=/home/feixia/anaconda3/envs/bi_layout/bin/python',
        '$PY -u tools/validate_portal_geometry_hypothesis.py --stage prepare --output experiments/portal_geometry_reproduction',
        '$PY -u tools/validate_portal_geometry_hypothesis.py --stage cache --output experiments/portal_geometry_reproduction',
        '$PY -u tools/validate_portal_geometry_hypothesis.py --stage run --output experiments/portal_geometry_reproduction',
        '$PY -m unittest tests.test_portal_geometry_hypothesis -v',
        '$PY -m tools.report_portal_geometry_hypothesis', '```','',
        'prepare 要求新目录；cache 支持断点恢复。报告脚本默认读取本次目录。没有修改生产模型、求解器、正式 val/test 或已有实验。没有训练新模块、运行 SALVe 或验证完整拼接。','']
    # Sensitivity to threshold, all measured on the same subset.
    lines+=['## 附表：联合阈值敏感性','','|条件|5°/0.25m/10%|10°/0.5m/10%|15°/1m/20%|','|---|---:|---:|---:|']
    for c in LABELS:lines.append('|{}|{}|'.format(LABELS[c],'|'.join(str(x)+'/54' for x in supplemental['threshold_sensitivity'][c].values())))
    (out/'验证报告.md').write_text('\n'.join(lines))
    render(out,rows,sel)
    write(out/'source_hashes.json',{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'tools/validate_portal_geometry_hypothesis.py',ROOT/'tools/report_portal_geometry_hypothesis.py',ROOT/'tests/test_portal_geometry_hypothesis.py',ROOT/'portallayout/geometry/joint_layout.py',ROOT/'portallayout/geometry/candidate_local_portal_geometry.py',ROOT/'portallayout/models/cross_scene_matcher.py']})
    print('Report:',out/'验证报告.md')

def render(out,rows,sel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    correct=[r for r in rows if 'O_full' in r['conditions']];cases={}
    groups={
        'radial_recovered':[r for r in correct if not r['conditions']['B_raw']['success'] and r['conditions']['O_radial']['success']],
        'radial_not_recovered':[r for r in correct if not r['conditions']['O_radial']['success']],
        'full_gt_scale_failure':[r for r in correct if not r['conditions']['O_full']['success']],
        'M2_broke_success':[r for r in correct if r['conditions']['B_raw']['success'] and not r['conditions']['B_existing_M2'].get('success')],
        'angle_broke_success':[r for r in correct if r['conditions']['B_raw']['success'] and not r['conditions']['O_angle']['success']],
        'opening':[r for r in correct if r['type_pair'][0]=='opening']}
    for name,rs in groups.items():
        for r in sorted(rs,key=lambda r:stable(123,r['id']))[:2]:cases.setdefault(r['id'],dict(row=r,categories=[]))['categories'].append(name)
    for r in sorted(correct,key=lambda r:stable(123,r['id'])):
        if len(cases)>=12:break
        if r['id'] not in cases:cases[r['id']]=dict(row=r,categories=['hash_fill'])
    directory=out/'cases';directory.mkdir(exist_ok=True);gallery=['# 几何归因固定规则案例','', '所有图中的 oracle 使用 GT，仅用于离线诊断，不是新模型预测。','']
    for i,case in enumerate(cases.values()):
        r=case['row'];fig,axs=plt.subplots(1,2,figsize=(12,5))
        for side,suffix in enumerate(['a','b']):
            k=next(x for x in sel['pairs']['monitor'] if x['id']==r['id'])['image_'+suffix];v=sel['views'][k]
            d=torch.load(out/'predictions'/(v['cache_key']+'.pt'),map_location='cpu')
            contour=bi_to_zind(np.stack([panorama_ray(t)*max(abs(float(d['enclosed'][t])),1e-3) for t in range(256)]))
            ax=axs[side];ax.plot(*np.r_[contour,contour[:1]].T,color='gray',lw=1,label='predicted enclosed');ax.scatter([0],[0],c='black',s=15)
            for name,color in [('B_raw','red'),('B_existing_M2','orange'),('O_radial','royalblue'),('O_full','green')]:
                condition=r['conditions'][name]
                if condition.get('valid'):
                    e=np.asarray(condition['endpoints_'+('A','B')[side]]);ax.plot(*e.T,'o-',color=color,label=name)
            ax.set_aspect('equal');ax.legend(fontsize=7);ax.set_title(suffix.upper()+' '+r['type_pair'][side]);ax.set_xlabel('camera-height-normalized x');ax.set_ylabel('y')
        fig.suptitle(r['house']+' / '+r['scope']+' / '+','.join(case['categories']),fontsize=9);fig.tight_layout();fig.savefig(directory/f'{i+1:02d}.png',dpi=130);plt.close(fig)
        gallery += [f'## {i+1:02d} 房屋 {r["house"]}：'+', '.join(case['categories']),'',f'pair: `{r["id"]}`','',f'![](cases/{i+1:02d}.png)','']
    write(out/'case_selection.json',dict(rule='hash seed123; two per recovery/failure category then hash fill; categories overlap',counts={k:len(v) for k,v in groups.items()},cases=[dict(id=k,categories=v['categories']) for k,v in cases.items()]))
    (out/'案例图册.md').write_text('\n'.join(gallery))
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    names=list(LABELS);table=json.loads((out/'summary.json').read_text())['correct_match_subset']
    axs[0].bar(range(len(names)),[table[c]['success']/54 for c in names]);axs[0].set_ylabel('joint pose success on fixed correct matches');axs[0].set_xticks(range(len(names)));axs[0].set_xticklabels(names,rotation=65,ha='right',fontsize=7)
    for c,color in [('B_raw','gray'),('O_direction','orange'),('O_radial','royalblue'),('O_full','green')]:
        vals=np.sort([r['conditions'][c]['translation_m'] for r in correct]);axs[1].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=c,color=color)
    axs[1].set_xlabel('translation error (m)');axs[1].set_ylabel('empirical CDF');axs[1].legend();fig.tight_layout();fig.savefig(out/'geometry_attribution.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
