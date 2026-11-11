#!/usr/bin/env python3
"""Recompute an evidence dossier from frozen artifacts; never train or infer.

Run with the bi_layout Python environment. Writes only to --output; source
experiments remain untouched. Statistical intervals are exploratory, not tests
of an independently held-out dataset.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / 'experiments/research1_pose_geometry_20261026_0245'
ROUND = ROOT / 'experiments/research1_round2_20261024'
METRICS = ('overall_iou', 'localization_coverage', 'translation_error_cm',
           'rotation_error_degrees', 'wall_f1', 'junction_f1', 'wall_distance_cm')
LABELS = ('整层 IoU', '正确定位覆盖率', '平移中位误差 cm', '旋转中位误差 °',
          'Wall F1', 'Junction F1', '墙距 cm')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def unique(items, description):
    if len(items) != len(set(items)):
        raise ValueError('duplicate ' + description)


def close(a, b):
    return a is b if a is None or b is None else bool(np.isclose(a, b, atol=1e-10, rtol=1e-9))


def bootstrap_delta(a, b):
    """Paired house means; missing houses are excluded explicitly, never zeroed."""
    houses = sorted(h for h in set(a) & set(b) if a[h] is not None and b[h] is not None)
    if not houses:
        return dict(delta=None, ci95=None, houses=[], excluded_houses=sorted(set(a) | set(b)))
    values = np.array([b[h] - a[h] for h in houses])
    rng = np.random.RandomState(123)
    samples = values[rng.randint(0, len(values), (2000, len(values)))].mean(1)
    return dict(delta=float(values.mean()), ci95=np.percentile(samples, [2.5, 97.5]).tolist(),
                houses=houses, excluded_houses=sorted((set(a) | set(b)) - set(houses)))


def aggregate(rows, metric):
    if metric in ('translation_error_cm', 'rotation_error_degrees'):
        field = 'translation_cm' if metric == 'translation_error_cm' else 'rotation_degrees'
        values = [e[field] for r in rows for e in r['pose_errors'] if e.get(field) is not None]
        return float(np.median(values)) if values else None
    values = [r['metrics'][metric] for r in rows if r['metrics'][metric] is not None]
    return float(np.mean(values)) if values else None


def floor_bootstrap(a, b, metric):
    """Cluster resampling preserves floors and pooled observation medians."""
    groups = []
    for rows in (a, b):
        by_house = defaultdict(list)
        for row in rows:
            by_house[row['house_id']].append(row)
        groups.append(by_house)
    if set(groups[0]) != set(groups[1]):
        raise ValueError('floor house sets differ')
    houses = sorted(groups[0]); rng = np.random.RandomState(123); deltas = []
    for sample in rng.randint(0, len(houses), (2000, len(houses))):
        x, y = [aggregate([r for i in sample for r in g[houses[i]]], metric) for g in groups]
        if x is not None and y is not None:
            deltas.append(y - x)
    x, y = aggregate(a, metric), aggregate(b, metric)
    return dict(delta=None if x is None or y is None else y-x,
                ci95=np.percentile(deltas, [2.5, 97.5]).tolist() if deltas else None,
                valid_replicates=len(deltas), houses=houses)


class Dossier:
    def __init__(self, output):
        self.output = output.resolve(); self.sources = {}; self.checks = []
        self.timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
        self.output.mkdir(parents=True, exist_ok=True)

    def source(self, path):
        path = Path(path).resolve()
        self.sources[str(path.relative_to(ROOT))] = digest(path)
        return path

    def read(self, path):
        return json.loads(self.source(path).read_text())

    def check(self, name, result):
        self.checks.append(dict(name=name, passed=bool(result)))
        if not result:
            raise ValueError('evidence check failed: ' + name)

    def write(self, name, data):
        p = self.output/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

    def md(self, name, body):
        p = self.output/name
        history = []
        if p.exists():
            history = [s for s in p.read_text().splitlines() if s.startswith('修改记录：')]
        history.append(f'修改记录：{self.timestamp}（Asia/Shanghai）从冻结原始数据复算证据及执行条件。')
        p.write_text('\n'.join(history)+'\n\n'+body+'\n')

    def csv(self, name, rows):
        with (self.output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, list(rows[0])); writer.writeheader(); writer.writerows(rows)

    def link(self, path):
        return html.escape(os.path.relpath(path, self.output), quote=True)


def geometry(d):
    selection = d.read(GEO/'geometry_selection.json'); rows = []
    for path in sorted((GEO/'geometry_audit').glob('*.json')):
        part = d.read(path); saved = part.pop('fingerprint')
        d.check('geometry fingerprint '+path.stem, fingerprint(part) == saved)
        rows.extend(part['rows'])
    unique([(r['collection'], r['pair_id']) for r in rows], 'geometry pair/collection')
    output = []; rates = {}
    with d.source(GEO/'geometry_comparison.csv').open() as f:
        original = {(r['variant'], r['role'], r['collection'], r['scope']): r for r in csv.DictReader(f)}
    for variant, entry in selection['variants'].items():
        for role in ('fit', 'selection'):
            for collection in ('r3', 'weak'):
                for scope in ('overall', 'cross_room', 'same_room', 'unknown_room'):
                    selected = [r for r in rows if r['role'] == role and r['collection'] == collection
                                and (scope == 'overall' or r['relationship'] == scope)]
                    valid = [r for r in selected if r['gt_valid']]
                    houses = defaultdict(list)
                    for row in valid:
                        houses[row['house_id']].append(int(row['variants'][variant]['success']))
                    hr = {h: float(np.mean(v)) for h, v in houses.items()}
                    count = sum(sum(v) for v in houses.values())
                    rec = dict(variant=variant, role=role, collection=collection, scope=scope,
                               pairs=len(valid), correct_pairs=count,
                               success_rate=count/len(valid) if valid else None,
                               house_macro=float(np.mean(list(hr.values()))) if hr else None,
                               correct_houses=sum(any(v) for v in houses.values()),
                               invalid_gt=len(selected)-len(valid))
                    expected = entry['statistics'][role][collection][scope]
                    d.check('/'.join((variant, role, collection, scope)), all(close(rec[k], expected[k]) for k in rec if k in expected))
                    old = original[(variant, role, collection, scope)]
                    d.check('CSV '+ '/'.join((variant, role, collection, scope)),
                            all(close(rec[k], float(old[k]) if old[k] else None)
                                for k in ('pairs', 'correct_pairs', 'success_rate', 'house_macro')))
                    rates[(variant, role, collection, scope)] = hr
                    output.append(rec)
    contrasts = []
    for collection in ('r3', 'weak'):
        for name, a, b in (
            ('总变化', 'dominant_wall_s1p0', 'local_m2_s0p0'),
            ('原几何下仅尺度先验', 'dominant_wall_s1p0', 'dominant_wall_s0p0'),
            ('单位尺度下仅几何拟合', 'dominant_wall_s0p0', 'local_m2_s0p0'),
            ('原尺度下仅几何拟合', 'dominant_wall_s1p0', 'local_m2_s1p0'),
            ('新几何下仅尺度先验', 'local_m2_s1p0', 'local_m2_s0p0')):
            contrasts.append(dict(name=name, collection=collection, baseline=a, variant=b,
                **bootstrap_delta(rates[(a, 'selection', collection, 'cross_room')],
                                  rates[(b, 'selection', collection, 'cross_room')])))
    paired = [dict(pair_id=r['pair_id'], house_id=r['house_id'], collection=r['collection'],
                   **{k: int(v['success']) for k, v in r['variants'].items()})
              for r in rows if r['role']=='selection' and r['relationship']=='cross_room' and r['gt_valid']]
    d.csv('geometry_recomputed.csv', output); d.csv('geometry_paired_selection.csv', paired)
    d.write('geometry_contrasts.json', contrasts)
    return output, contrasts


def floor_evidence(d):
    comparison = d.read(GEO/'comparison.json'); groups = {}; flat = []; paired = []; intervals = {}
    for role, variants in comparison['summaries'].items():
        for variant, summary in variants.items():
            data = d.read(GEO/'comparison'/role/variant/'metrics.json'); rows = data['floors']
            unique([r['floor_uid'] for r in rows], 'floor UID')
            for r in rows:
                unique([e['observation_id'] for e in r['pose_errors']], 'observation error')
            groups[(role, variant)] = rows
            for k in METRICS:
                value = aggregate(rows, k)
                d.check(f'metrics {role}/{variant}/{k}', close(value, summary['metrics'][k]) and close(value, data['summary']['metrics'][k]))
            d.check('failures '+role+variant, sum(r['runtime_failure'] for r in rows)==summary['runtime_failures'])
            for r in rows:
                flat.append(dict(role=role, variant=variant, floor_uid=r['floor_uid'], house_id=r['house_id'],
                                 runtime_failure=r['runtime_failure'], **r['metrics']))
        a = {r['floor_uid']: r for r in groups[(role, 'A_V0_rank')]}
        b = {r['floor_uid']: r for r in groups[(role, 'B_V1_rank')]}
        d.check('paired floor sets '+role, set(a)==set(b))
        for uid in sorted(a):
            rec = dict(role=role, floor_uid=uid, house_id=a[uid]['house_id'])
            for k in METRICS:
                x,y=a[uid]['metrics'][k],b[uid]['metrics'][k]
                rec.update({k+'_before':x,k+'_after':y,k+'_delta':None if x is None or y is None else y-x})
            paired.append(rec)
        intervals[role] = {k: floor_bootstrap(list(a.values()), list(b.values()), k) for k in METRICS}
    for variant, expected in comparison['coverage'].items():
        rows = groups[('selection',variant)]+groups[('calibration',variant)]
        d.check('coverage '+variant, len(rows)==expected['floors'] and
                sum(r['input_observations'] for r in rows)==expected['observations'] and
                sum(r['input_observations']*(r['input_observations']-1)//2 for r in rows)==expected['pairs'] and
                sum(r['runtime_failure'] for r in rows)==expected['failed_floors'])
    d.csv('all_floor_metrics.csv', flat); d.csv('paired_floor_metrics.csv', paired)
    d.write('floor_bootstrap.json', intervals)
    return comparison, groups, paired


def audit(d):
    contract = d.read(GEO/'run_contract.json'); cohort = contract['cohort']
    unique([c['floor_uid'] for c in cohort], 'cohort floor')
    houses = {role: sorted({c['house_id'] for c in cohort if c['role']==role}) for role in ('fit','selection','calibration')}
    for a,b in (('fit','selection'),('fit','calibration'),('selection','calibration')):
        d.check('house split '+a+'/'+b, not set(houses[a]) & set(houses[b]))
    for field in ('raw_cache_hashes','annotation_hashes'):
        for name, expected in contract[field].items():
            d.check(field+' '+name, digest(d.source(name))==expected)
    d.check('R3 checkpoint', digest(d.source(ROUND/'R3/checkpoint.pt'))==contract['pair_checkpoint_sha256'])
    frozen = d.read(GEO/'frozen_selection.json'); models = {}
    for name in ('V0', 'V1'):
        checkpoint = d.source(GEO/name/'train/checkpoint.pt')
        workpoint = d.read(GEO/name/'selection_workpoint.json')
        policy = d.read(GEO/name/'experimental_policy.json')
        training = d.read(GEO/name/'train/training_report.json')
        d.check(name+' frozen weights', digest(checkpoint)==frozen['checkpoint_hashes'][name])
        # Historical workpoint_hashes stores experimental_policy file hashes.
        d.check(name+' frozen workpoint policy', digest(GEO/name/'experimental_policy.json')==frozen['workpoint_hashes'][name])
        d.check(name+' workpoint policy binding', all(policy[k]==workpoint[k] for k in
                ('status','opening_threshold','weights_hash','feature_version','candidate_policy','source_snapshot','pose_geometry')))
        d.check(name+' infeasible workpoint', workpoint['status']=='selection_workpoint_infeasible'
                and not any(row['passed'] for row in workpoint['grid']))
        models[name] = dict(checkpoint=str(checkpoint.relative_to(ROOT)), sha256=digest(checkpoint),
            geometry=frozen['configs'][name], selected_step=training['stages']['pose']['selected_step'],
            steps=training['stages']['pose']['steps'], workpoint_status=workpoint['status'],
            workpoint_count=len(workpoint['grid']), feature_version=workpoint['feature_version'])
    d.write('model_provenance.json',dict(pair_checkpoint_sha256=contract['pair_checkpoint_sha256'],
                                       pair_checkpoint='experiments/research1_round2_20261024/R3/checkpoint.pt',
                                       pair_selected_step=5000, verifiers=models))
    weak = d.read(ROOT/'experiments/research1_main_20261024/weak/opening_weak.json')
    round_contract = d.read(ROUND/'run_contract.json')
    d.check('weak labels', d.sources['experiments/research1_main_20261024/weak/opening_weak.json']==round_contract['weak_supervision_sha256'])
    d.check('cache snapshot binding', weak['source_snapshot']==contract['source_snapshot']==round_contract['source_snapshot'])
    cache_path = ROOT/'outputs/research1_floor_frontend/main60_20261024/cache_contract.json'
    cache = d.read(cache_path)
    d.check('frontend contract fingerprint', fingerprint(cache['contract'])==cache['fingerprint'])
    package = d.read(ROUND/'review/review_manifest.json'); review = d.read(ROUND/'review/review_template.json')
    unique([r['case_id'] for r in review['cases']], 'human review case')
    d.check('review package binding', review['package_fingerprint']==package['fingerprint'] and
            {r['case_id'] for r in review['cases']}=={r['case_id'] for r in package['cases']})
    counts = Counter(r['judgement'] or '待审核' for r in review['cases'])
    completed = len(review['cases'])-counts['待审核']
    manual = dict(total=len(review['cases']), completed=completed, counts=dict(counts),
                  reviewer=review.get('reviewer'), reviewed_at=review.get('reviewed_at'),
                  status='complete' if completed==len(review['cases']) and review.get('reviewer') and review.get('reviewed_at') else 'pending',
                  population='diagnostic sample; not a random precision estimate',
                  checkpoint_sha256=package['pair_checkpoint_sha256'],
                  note='旧审核包绑定历史模型；完成后仍需核查 A0–A3 新输出，不自动当作新模型通过。')
    d.write('review_status.json', manual)
    drift = [name for name, value in contract['code_hashes'].items() if digest(ROOT/name)!=value]
    audit_result = dict(houses=houses, human_review=manual, historical_code_drift=drift,
                        pretraining_exposure='unknown', cache_snapshot=contract['source_snapshot'],
                        cache_verification='contract and weak-label snapshot binding; feature tensor contents not rehashed',
                        bootstrap=dict(unit='house',samples=2000,seed=123,interpretation='exploratory, development data'),
                        note='原始缓存/标注/匹配权重重新校验；历史源码漂移单列，不能把现代码当成旧运行源码。')
    d.write('data_audit.json', audit_result)
    return audit_result


def edge_evidence(d):
    summary=d.read(GEO/'edge_error_summary.json'); backend=d.read(GEO/'backend_diagnostics.json')
    # Recompute all-edge labels from individual frozen edge diagnostics.
    for role, variants in summary.items():
        for variant, entry in variants.items():
            counts=Counter()
            for floor in backend[variant]:
                if floor['role']!=role: continue
                unique([e['pair_id'] for e in floor['edges']], 'accepted edge')
                for edge in floor['edges']:
                    counts['correct' if edge['target']==1 else 'incorrect' if edge['target']==0 else 'ignored']+=1
            d.check('edge counts '+role+variant, all(counts[k]==entry['counts']['overall'].get(k,0) for k in ('correct','incorrect','ignored')))
    d.write('edge_diagnostics.json', summary)
    return summary


def figures(d, geo, comparison, edges):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':12,'svg.fonttype':'path'})
    def save(fig,name):
        for ext in ('svg','png'):
            fig.savefig(d.output/(name+'.'+ext),dpi=150,facecolor='white')
        plt.close(fig)
    a=comparison['summaries']['calibration']['A_V0_rank']['metrics'];b=comparison['summaries']['calibration']['B_V1_rank']['metrics']
    fig,axes=plt.subplots(2,4,figsize=(16,9));fig.suptitle('已有结果：面积改善，但定位与结构指标未同步改善',fontsize=22)
    for ax,k,label in zip(axes.flat,METRICS,LABELS):
        scale=100 if k in ('overall_iou','localization_coverage','wall_f1','junction_f1') else 1
        values=[a[k]*scale,b[k]*scale];ax.bar(['V0 排序','V1 排序'],values,color=['#738399','#17877a'])
        ax.set_title(label+('（%）' if scale==100 else ''))
        ax.set_ylim(0,max(values)*1.25 or 1)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom')
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.8,'开发验证 10 层\n全部四组各 20 层 / 585 图\n严格过滤：两组均无可行工作点\n排序组仅作诊断\n不是 SALVe 对比',va='top',linespacing=1.7)
    fig.tight_layout(rect=(0,.04,1,.94));save(fig,'01_existing_results')
    fig,axes=plt.subplots(1,2,figsize=(16,9));fig.suptitle('几何 × 尺度因素对照：必须在相同先验下比较几何',fontsize=22)
    for ax,col,title in zip(axes,('r3','weak'),('固定 R3 跨房检索 711 对','固定已知弱对应 203 对')):
        records={r['variant']:r for r in geo if r['role']=='selection' and r['collection']==col and r['scope']=='cross_room'}
        matrix=np.array([[records[f'{m}_s{s}']['success_rate']*100 for s in ('1p0','0p5','0p0')] for m in ('dominant_wall','raw_depth','local_m2')])
        ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(3),['宽度比尺度','半收缩','单位尺度先验']);ax.set_yticks(range(3),['原支撑墙','原始深度','local_m2']);ax.set_title(title)
        for i,m in enumerate(('dominant_wall','raw_depth','local_m2')):
            for j,s in enumerate(('1p0','0p5','0p0')):
                r=records[f'{m}_s{s}'];ax.text(j,i,f"{r['correct_pairs']}/{r['pairs']}\n{matrix[i,j]:.2f}%",ha='center',va='center',color='white' if matrix[i,j]>60 else '#152535')
    fig.text(.5,.06,'R3：30 → 80（只改尺度），80 → 81（同先验下改几何）。这是候选上限，非最终接边率。',ha='center',fontsize=15)
    fig.tight_layout(rect=(0,.12,1,.92));save(fig,'02_geometry_scale')
    fig,axes=plt.subplots(1,2,figsize=(16,9));fig.suptitle('对应到整层：候选改善尚未转化为可靠恢复',fontsize=22)
    for ax,role,title in zip(axes,('selection','calibration'),('选型集跨房接边','开发验证集跨房接边')):
        names=['A_V0_rank','B_V1_rank'];bottom=np.zeros(2)
        for field,label,color in [('correct','正确','#17877a'),('incorrect','错误','#d66257'),('ignored','忽略','#aeb7c2')]:
            vals=[edges[role][n]['counts']['cross_room'].get(field,0) for n in names]
            ax.bar(['V0 排序','V1 排序'],vals,bottom=bottom,label=label,color=color)
            for i,v in enumerate(vals):
                if v:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center')
            bottom+=vals
        ax.set_title(title);ax.set_ylabel('已接受跨房边数（诊断排序模式）');ax.legend()
    fig.text(.5,.07,'对应正确 → 候选可解 → 选解正确 → 可靠接边 → 整层恢复\n各阶段分母不同，不绘制虚假的同分母漏斗；严格过滤两组均为 0 边。',ha='center',fontsize=15)
    fig.tight_layout(rect=(0,.15,1,.92));save(fig,'03_error_propagation')


def execution_spec(d, audit_result):
    common=dict(seed=123,max_steps=6000,validation_interval=500,early_stopping=False,
        selection_rule=['cross_room_house_macro_ap descending','overall_ap descending','balanced_loss ascending','step ascending'],
        objective='known_cells',pair_batch_size=2,
        optimizer=dict(name='AdamW',lr=1e-4,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,scheduler=None),
        sampler='alternate positive/proven-unmatched pair; uniform house then uniform pair; random.Random(seed)',
        frontend='frozen main60_20261024',candidate_ids='frozen',base_pooling=True,
        structure=True,geometry=True,intra_attention=True,dustbin=True,
        initialization='identical initial shared weights and sampling order; train each variant from start',
        followup_seeds=[456,789],followup_gate='primary contrast improves at feasible frozen workpoint')
    spec=dict(schema_version='method-evidence.ablation-spec.v1',status='not_executable_spec',
        training_started=False,common=common,
        groups={f'A{i}':dict(context=bool(i%2),pair_conditioned_retrieval=bool(i//2)) for i in range(4)},
        independent_switch_semantics=dict(context_off='zero context input to fusion gate and mask context branch; retain structure, geometry and intra attention',
            retrieval_off='skip query/memory retrieval; use zero retrieval score channels, retain descriptor and geometry scoring and Sinkhorn'),
        workpoint=dict(overall_precision_min=.95,cross_room_precision_min=.95,negative_pair_far_max=.01,
                       correct_edges_min=20,correct_houses_min=5,selection_only=True,unknown_is_negative=False,
                       pose_independent_evidence_required=True,no_feasible_workpoint='accept no edges'),
        experiment_b=dict(geometry='local_m2_s0p0',backend='forest',room_fusion=False,pgo='separate controlled row',
            verifier='one shared verifier trained on fit-only candidate union, frozen before selection comparison',
            compatibility='existing verifier binds R3 checkpoint; never bypass identity checks to reuse V1 for A0-A3',
            denominator='all fixed floor image pairs, with detection misses and rejected pairs retained'),
        blockers=['human_review_pending' if audit_result['human_review']['status']!='complete' else 'new_variant_review_pending',
                  'independent_context_and_retrieval_switches_not_implemented',
                  'checkpoint_cache_contract_and_shared_verifier_integration_required'],
        acceptance_tests=['context off: perturb context branch cannot change gate or descriptors',
            'retrieval off: perturb memory cannot change matcher scores with fixed descriptors',
            'full/full reproduces historical forward behavior; empty candidates and transpose symmetry preserved',
            'checkpoint resume preserves variant; cross-variant cache reuse rejected',
            'training gradients reach enabled branches; disabled modules are not represented as trained',
            'fit/selection isolation; frozen thresholds only on development validation; no GT in inference'])
    d.write('ablation_execution_spec.json',spec)
    for name,config in spec['groups'].items():
        d.write('ablation_configs/'+name+'.json',dict(group=name,**config,common=common,
            status='specification_only_not_a_current_cli_config',training_started=False))
    return spec


def report(d,geo,contrasts,comparison,paired,audit_result,spec):
    round_comparison=d.read(ROUND/'comparison.json'); ap={}
    for name,item in round_comparison['variants'].items():
        report=d.read(ROUND/name/'training_report.json')
        values=item['selection_metrics']['cross_room_ap_by_house']
        value=float(np.mean([v for v in values.values() if v is not None]))
        d.check('AP '+name,close(value,item['selection_metrics']['cross_room_house_macro_ap']) and close(value,report['selection_metrics']['cross_room_house_macro_ap']))
        ap[name]=dict(value=value,steps=item['steps'],selected_step=item['selected_step'],objective=item['objective'],houses=values)
    d.write('matching_evidence.json',dict(variants=ap,R1_R3=bootstrap_delta(ap['R1']['houses'],ap['R3']['houses']),
        limitation='unequal budget, selection-set checkpoints; not context ablation',recomputation='house AP mean; underlying pair labels/AP ranking not recomputed'))
    rows=[r for r in geo if r['role']=='selection' and r['scope']=='cross_room']
    g={(r['variant'],r['collection']):r for r in rows}
    a,b=[comparison['summaries']['calibration'][n]['metrics'] for n in ('A_V0_rank','B_V1_rank')]
    claims=[dict(id='E1',status='已有初步支持',claim='训练目标调整的开发信号',before=ap['R1']['value'],after=ap['R3']['value'],
                 denominator='9 houses with known positives; 1289 AP N/A',source='matching_evidence.json',limit='预算不同；不能证明上下文模块有效'),
            dict(id='E2',status='已有初步支持',claim='几何与尺度组合提高候选可解上限',before=g[('dominant_wall_s1p0','r3')]['correct_pairs'],after=g[('local_m2_s0p0','r3')]['correct_pairs'],denominator=711,source='geometry_paired_selection.csv',limit='选型集；并非最终接边率'),
            dict(id='E3',status='已有初步支持',claim='已知弱对应下几何可解上限提高',before=g[('dominant_wall_s1p0','weak')]['correct_pairs'],after=g[('local_m2_s0p0','weak')]['correct_pairs'],denominator=203,source='geometry_paired_selection.csv',limit='依赖弱对应；并非纯预测结果'),
            dict(id='E4',status='结果混合',claim='整层面积改善',before=a['overall_iou'],after=b['overall_iou'],denominator=10,source='all_floor_metrics.csv',limit='覆盖率及结构 F1 下降；几何和验证器共同变化'),
            dict(id='E5',status='结果混合',claim='正确定位覆盖率下降',before=a['localization_coverage'],after=b['localization_coverage'],denominator='10 floor macro',source='all_floor_metrics.csv',limit='不支持可靠整层恢复优势'),
            dict(id='E6',status='尚无证据',claim='视觉上下文/条件检索的独立收益',before=None,after=None,denominator=None,source='ablation_execution_spec.json',limit='四组独立消融尚未执行'),
            dict(id='E7',status='尚无证据',claim='完整方法优于 SALVe / 研究二有效',before=None,after=None,denominator=None,source='ablation_execution_spec.json',limit='已有 SALVe smoke 目录不是同协议系统对照；研究二另行实施')]
    d.write('evidence_claims.json',claims)
    text=['# 优势证据清单','', '| ID | 状态 | 主张 | 前 → 后 | 分母 | 限制 |','|---|---|---|---|---|---|']
    for c in claims:text.append(f"| {c['id']} | {c['status']} | {c['claim']} | {c['before']} → {c['after']} | {c['denominator']} | {c['limit']}；[{c['source']}]({c['source']}) |")
    text+=['','## 统计与来源','所有证据源文件的 SHA256 见 source_manifest.json；模型、划分、候选及标注身份见 data_audit.json 和原 run_contract。',
        '几何从逐图像对成功标记复算，全部九组与旧 CSV/JSON 核对；整层从逐层指标复算，平移/旋转用逐观测 pooled median。未重新执行模型或从多边形重算结构指标。',
        '房屋配对 bootstrap 2000 次、seed 123；区间位于 geometry_contrasts.json 与 floor_bootstrap.json。选型集与开发验证上的区间仅为探索性描述，不证明独立泛化。',
        'R1–R4 复算到房屋 AP 的均值层级，未从候选排序重算 AP。weak 标签和人工判断分开；不确定判断不视为已确认正负例。',
        '几何选择在 selection 完成；本次按开发验证 IoU 差值选展示案例属于事后诊断，不用于进一步调参。']
    d.md('优势证据清单.md','\n'.join(text))
    text=['# 几何与尺度贡献分解','','固定 selection 跨房图像对，比较顺序改变时边际收益也会改变，不能解释为唯一贡献比例。',
        '| 对照 | 集合 | 房屋宏平均变化 pp | 95% 探索区间 pp |','|---|---|---:|---|']
    for c in contrasts:text.append(f"| {c['name']} | {c['collection']} | {100*c['delta']:.3f} | [{100*c['ci95'][0]:.3f}, {100*c['ci95'][1]:.3f}] |")
    text+=['','原支撑墙：30→80；单位尺度先验固定后，原支撑墙→local_m2：80→81。不能把组合提升全部归因于几何拟合，也不能由 +1 对断言几何普遍无效。',
        '弱对应组同一先验下为 176→182；两组分母和任务条件不同，不相互替代。完整九组结果见 geometry_recomputed.csv。']
    d.md('因素归因.md','\n'.join(text))
    review=audit_result['human_review']
    d.md('核心消融执行表.md',f'''# 四组核心消融执行表

当前状态：规范已落盘，训练未启动。人工审核 {review['completed']}/{review['total']}；当前模型接口缺少独立开关。配置文件明确标记 specification_only，不能传给现有 CLI 冒充可执行配置。

| 组别 | 单图邻域上下文 | 候选对条件化检索 |
|---|---|---|
| A0 | 关 | 关 |
| A1 | 开 | 关 |
| A2 | 关 | 开 |
| A3 | 开 | 开 |

A2 的关闭只指单图描述子的上下文分支，跨图检索仍可看到二维视觉记忆；不能称其没有上下文视觉信息。主对照 A3−A2、A3−A1，另报告 2×2 交互效应。

四组保留基础池化、结构、几何、图内注意力及 dustbin，采用 known_cells、batch=2、seed=123、6000 步、每 500 步验证，不早停以保证相同预算。共享参数使用相同初始化与采样序列；每组从头训练，不使用 R3 训练后权重作为不同起点。优化器/采样沿用 R3 并在启动时冻结完整参数。选权重按跨房宏 AP、整体 AP、负 balanced loss、较早步数依次打破并列。

## 必须先补的接口

- 单图上下文独立开关：关闭时同时屏蔽融合分支与 gate 的上下文输入，保持其他分支不变。只屏蔽分支会通过 gate 泄露上下文信息。
- 检索独立开关：关闭时跳过 query/二维记忆检索，检索打分通道置零，保留描述子/几何打分。不得用历史 legacy matcher 替代该组。
- 将开关写入 checkpoint 配置、恢复契约与特征缓存身份；拒绝跨组缓存复用。现有累进 ablation 不能实现这一设计。

## 工作点与整层传播

总体/跨房精确率 ≥95%，明确负例 FAR ≤1%，至少 20 条正确边/5 个房屋，并通过位姿独立证据条件。只在 selection 选阈值，开发验证固定；无可行点保留不接边。开口对应质量和位姿接边质量分别过门槛，不用位姿候选条件召回替代全部图像对正确接边率。

实验 B 固定 local_m2+单位尺度先验、森林后端、关闭房间融合。V1 验证器绑定旧 R3，不能绕过契约直接跨组使用；需要在 fit 四组候选并集上训练一个共享验证器，冻结后评价四组。PGO 独立行。额外种子 456/789 仅在可靠工作点与主差异均有收益后启动。

完整机器可读规范与验收测试见 ablation_execution_spec.json。下一阶段需要实现和测试这些开关；本证据包未修改预测接口或模型。
''')
    d.md('开题结论.md',f'''# 开题可用结论

已经获得局部可行性证据：跨房候选可解数量从 30/711 增至 81/711，已知弱对应条件下从 47/203 增至 182/203。尺度先验是主要因素之一：仅修改该先验已达到 80/711。因此目前应表述为“几何与尺度策略改善了位姿候选上限”。

整层开发验证 IoU 从 {a['overall_iou']*100:.2f}% 提高到 {b['overall_iou']*100:.2f}%，但正确定位覆盖率从 {a['localization_coverage']*100:.2f}% 变为 {b['localization_coverage']*100:.2f}%，结构指标下降，严格工作点也未通过。现阶段不能说完整系统可靠恢复能力提升或超过 SALVe。

开口视觉上下文与候选对检索是下一步要验证的核心假设。四组消融将保持基础池化、结构、几何及后端不变，检验它们是否提高高精确率条件下的跨房召回，以及收益是否进入正确接边和整层定位。局部布局优化仍属于后续研究二。

人工审核目前 {review['completed']}/{review['total']}。这是计划中先于核心实验的前置条件，需真实审核者填写旧包中的 review_template.json 并留下姓名与时间；不能自动填写为通过。已有案例包是定向诊断样本，不能直接估计总体精确率。
''')
    d.md('README.md','''# 方法优势证据包

入口：[可视化报告](index.html)。本目录是首轮现成数据取证交付，未运行新训练/推理，也没有修改模型或历史实验。

复算命令（在仓库根目录）：

```bash
/home/feixia/anaconda3/envs/bi_layout/bin/python tools/build_method_advantage_evidence.py
python -m pytest -q tests/test_method_advantage_evidence.py
```

脚本只写入 --output 指定目录。统计文件确定性生成，Markdown 追加修改记录；生成前标记检查状态为 running，失败不能沿用旧 passed。图表为本地 matplotlib 生成的 PNG/SVG。页面引用仓库中的原始案例，迁移目录时应保留相对路径。

- evidence_claims.json：主张、值、分母及适用边界；source_manifest.json：实际读取来源与 SHA256。
- geometry_paired_selection.csv：711 个 R3 对和 203 个弱对应对；geometry_recomputed.csv：九组完整分层结果。
- geometry_contrasts.json / floor_bootstrap.json：2000 次房屋配对区间，开发数据上的探索性统计。
- all_floor_metrics.csv：四组、两种划分、全部 80 行；paired_floor_metrics.csv：20 层排序对照。
- model_provenance.json：冻结匹配器、V0/V1 权重、几何配置、所选步数、工作点状态。
- review_status.json / data_audit.json：人工审核进度、划分、数据身份与未核实事项。
- ablation_configs/：四组配置规范，不是现有训练 CLI 可直接执行的参数；实现缺口及验收条件见核心消融执行表。

下一步需要真实审核者完成原 96 例审核包。计划约定人工审核先于核心实验；不会自动把待审核改成通过。四组独立开关及跨组共享验证器仍需后续实现，未将规范落盘等同于实验完成。
''')
    candidates=sorted([r for r in paired if r['role']=='calibration'],key=lambda r:(r['overall_iou_delta'],r['floor_uid']))
    selected=[('退化最大',candidates[0]),('中位差值（下中位）',candidates[(len(candidates)-1)//2]),('改善最大',candidates[-1])]
    d.write('case_selection.json',dict(rule='calibration IoU delta ascending; UID tie break; min/lower median/max; posthoc diagnostic, no tuning',cases=[dict(label=l,**r) for l,r in selected]))
    sections=['<h1>逐层案例：改善、典型与退化共同展示</h1><p>按开发验证 IoU 差值排序；事后诊断，不作为独立成功率证据。</p>']
    for label,r in selected:
        sections.append(f"<h2>{label} · 房屋 {r['house_id']}</h2><p>IoU 差 {r['overall_iou_delta']*100:+.2f} pp；正确定位覆盖率差 {r['localization_coverage_delta']*100:+.2f} pp</p><div class='pair'>")
        for variant in ('A_V0_rank','B_V1_rank'):
            p=GEO/'predictions'/variant/r['floor_uid']/'gt_comparison.png'
            d.source(p);sections.append(f"<figure><img src='{d.link(p)}'><figcaption>{variant}：统一协议下预测与 GT</figcaption></figure>")
        sections.append('</div>')
    sections.append('<h2>全部 20 层</h2><table><tr><th>划分/房屋</th><th>IoU 差 pp</th><th>覆盖率差 pp</th><th>原始结果</th></tr>')
    for r in paired:
        links=[]
        for variant in ('A_V0_rank','B_V1_rank','C_V0_filter','D_V1_filter'):
            p=GEO/'predictions'/variant/r['floor_uid']/'floor_result.json'
            d.check('case link '+str(p),p.exists());links.append(f"<a href='{d.link(p)}'>{variant}</a>")
        cover=r['localization_coverage_delta']
        sections.append(f"<tr><td>{r['role']} / {r['house_id']}</td><td>{r['overall_iou_delta']*100:+.2f}</td><td>{'N/A' if cover is None else f'{cover*100:+.2f}'}</td><td>{' · '.join(links)}</td></tr>")
    sections.append('</table><p><a href="all_floor_metrics.csv">全部七指标 CSV（四组）</a></p>')
    page(d,'cases.html','\n'.join(sections))
    body=['<h1>方法优势取证 · 开题材料</h1><p>已有局部证据，尚无完整系统优于 SALVe 的结论。所有图由冻结数据复算；没有运行训练或推理。</p>',
          f"<p class='notice'>人工审核 {review['completed']}/{review['total']}；四组消融接口待实现。后续实验未启动。</p>",
          '<nav><a href="优势证据清单.md">证据清单</a> · <a href="因素归因.md">因素归因与区间</a> · <a href="cases.html">全部案例</a> · <a href="核心消融执行表.md">消融执行表</a> · <a href="开题结论.md">开题结论</a> · <a href="validation.json">复算检查</a></nav>']
    for name in ('01_existing_results','02_geometry_scale','03_error_propagation'):
        body.append(f'<figure><img src="{name}.png"><figcaption><a href="{name}.svg">SVG 矢量图</a> · <a href="{name}.png">PNG</a></figcaption></figure>')
    body.append(f"<p><a href='{d.link(ROUND/'review/index.html')}'>96 例人工审核入口</a> · <a href='{d.link(ROUND/'review/review_template.json')}'>待填写的审核文件</a></p>")
    page(d,'index.html','\n'.join(body))


def page(d,name,body):
    (d.output/name).write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>方法优势证据</title><style>
body{font:17px/1.7 system-ui,sans-serif;max-width:1280px;margin:40px auto;padding:0 24px;color:#183146;background:#f6f8fb}h1,h2{line-height:1.3}a{color:#087b79}figure{margin:24px 0;background:white;padding:12px;border-radius:12px}img{width:100%;height:auto}figcaption{font-size:14px}.pair{display:flex;gap:16px}.pair figure{width:50%}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #d4dce5;text-align:left}.notice{padding:16px;background:#fff0d7;border-radius:8px}@media(max-width:700px){.pair{display:block}.pair figure{width:auto}}nav{padding:16px;background:white}</style><body>'''+body+'</body></html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/method_advantage_evidence_20261028')
    args=parser.parse_args()
    if args.output.resolve() in (ROOT,GEO,ROUND) or GEO in args.output.resolve().parents or ROUND in args.output.resolve().parents:
        raise ValueError('output must not overwrite source experiments')
    d=Dossier(args.output)
    d.write('validation.json',dict(status='running',started_at=d.timestamp))
    print('Audit contracts and manual review',flush=True);audit_result=audit(d)
    print('Recompute nine geometry variants',flush=True);geo,contrasts=geometry(d)
    print('Recompute floor metrics and paired house intervals',flush=True);comparison,groups,paired=floor_evidence(d)
    edges=edge_evidence(d);spec=execution_spec(d,audit_result)
    print('Render report and scientific figures',flush=True)
    figures(d,geo,comparison,edges);report(d,geo,contrasts,comparison,paired,audit_result,spec)
    d.source(Path(__file__))
    d.write('source_manifest.json',dict(created_at=d.timestamp,sha256=d.sources,
        source_experiments=[str(GEO.relative_to(ROOT)),str(ROUND.relative_to(ROOT))]))
    d.write('validation.json',dict(status='passed',checks=d.checks,count=len(d.checks),
        limitations=['No new training/inference','No raw polygon metric recomputation','No pair-rank AP recomputation',
                     'Frontend tensors not fully rehashed','Cross-room edge categories sourced from frozen summary; all-edge totals independently recomputed']))
    print(json.dumps(dict(output=str(d.output),checks=len(d.checks),review=audit_result['human_review']['status'],training_started=False),ensure_ascii=False))


if __name__=='__main__':main()
