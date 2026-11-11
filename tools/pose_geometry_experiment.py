"""Geometry-ablation stages for the existing persistent pose controller."""
import csv
import fcntl
import json
import time
from pathlib import Path
import numpy as np

from tools.run_pose_verifier_experiment import ROOT, read, now
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic as write


def intervals(report):
    """Wilson intervals describe count uncertainty, not a new acceptance gate."""
    def wilson(yes,total):
        if not total:return None
        z=1.959963984540054;p=yes/total;den=1+z*z/total
        mid=(p+z*z/(2*total))/den
        delta=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
        return [max(0.,mid-delta),min(1.,mid+delta)]
    for scope in ('overall','cross_room','same_room'):
        if scope not in report:continue
        s=report[scope]
        s['precision_ci95']=wilson(s['correct_edges'],s['correct_edges']+s['incorrect_edges'])
        s['negative_pair_far_ci95']=wilson(s['false_accepted_negative_pairs'],s['negative_pairs'])
    for row in report.get('grid',[]):intervals(row)
    return report


def diagnostic_figures(output,parts,cache,root):
    import matplotlib.pyplot as plt
    from portallayout.datasets.zind_floor_dataset import read_jsonl
    from portallayout.evaluation.pose_candidate_audit import normalized_endpoints
    from portallayout.geometry.candidate_local_portal_geometry import panorama_ray
    candidates=sorted([r for p in parts for r in p['oracles']],key=lambda r:(not r['association_or_annotation_inconsistent'],fingerprint(r['candidate_ids'])))
    selected=[];counts={}
    for row in candidates:
        if counts.get(row['house_id'],0)>=4:continue
        selected.append(row);counts[row['house_id']]=counts.get(row['house_id'],0)+1
        if len(selected)==24:break
    folder=output/'geometry_cases';folder.mkdir(exist_ok=True);html=['<!doctype html><meta charset="utf-8"><h1>Offline geometry diagnostics</h1>']
    for index,row in enumerate(selected):
        uid=row['floor_uid'];overlays=read(output/'geometry_overlays'/(uid+'.json'))['overlays']
        annotations={r['observation_id']:r for r in read_jsonl(Path(root)/'labels'/uid/'observations_gt.jsonl')}
        fig,axes=plt.subplots(1,2,figsize=(11,5))
        for side,(cid,iid) in enumerate(zip(row['candidate_ids'],row['interface_ids'])):
            oid=next(o for o,r in overlays['dominant_wall'].items() if any(x['candidate_id']==cid for x in r['prediction']['openings']))
            record,arrays=cache.load(oid);depth=np.maximum(np.abs(arrays['depth_enclosed']),.001)
            contour=np.stack([panorama_ray(i,len(depth)) for i in range(len(depth))])*depth[:,None]
            ax=axes[side];ax.plot(contour[:,0],contour[:,1],color='gray',alpha=.5,label='raw contour');ax.scatter([0],[0],marker='^',color='black')
            for method in overlays:
                opening=next(x for x in overlays[method][oid]['prediction']['openings'] if x['candidate_id']==cid)
                if opening['geometry_valid']:
                    e=np.asarray(opening['endpoints_local']);ax.plot(e[:,0],e[:,1],'o-',label=method)
            ann=annotations[oid];item=next(x for x in ann['interfaces_by_layout']['layout_raw'] if x['interface_observation_id']==iid)
            e=normalized_endpoints(ann,item);ax.plot(e[:,0],e[:,1],'k--',linewidth=2,label='GT offline')
            ax.axis('equal');ax.legend(fontsize=7);ax.set_title(cid[-12:])
        fig.suptitle(row['house_id']+' / '+row['relationship']+' / GT-full success='+str(row['conditions']['O_full']['success']))
        fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(folder/(str(index)+'.'+ext),dpi=150)
        plt.close(fig);html.append('<h2>'+str(index)+' / '+row['house_id']+'</h2><img style="width:100%" src="'+str(index)+'.png">')
    write(folder/'cases.json',dict(requested=24,actual=len(selected),cases=selected,selection='GT-full failures first; stable ID hash; max four per house'))
    (folder/'index.html').write_text('\n'.join(html))


def geometry_report(output, parts, selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from portallayout.evaluation.pose_geometry import BASELINE
    table = selection['variants']
    rows = []
    for key, value in table.items():
        for role, collections in value['statistics'].items():
            for collection, scopes in collections.items():
                for scope, s in scopes.items():
                    rows.append(dict(variant=key, role=role, collection=collection, scope=scope,
                                     **{k:v for k,v in s.items() if k != 'house_rates'}))
    with (output/'geometry_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    keys=list(table); values=[table[k]['statistics']['selection']['r3']['cross_room']['house_macro'] for k in keys]
    fig,ax=plt.subplots(figsize=(12,5));ax.bar(keys,[v or 0. for v in values]);ax.tick_params(axis='x',rotation=35)
    ax.set_ylabel('Selection cross-room correct-candidate ratio (house macro)');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(output/('geometry_comparison.'+ext),dpi=160)
    plt.close(fig)
    oracles=[r for p in parts for r in p['oracles']]
    oracle_summary={}
    for role in ('fit','selection'):
        oracle_summary[role]={}
        for scope in ('same_room','cross_room','unknown_room'):
            records=[r for r in oracles if r['role']==role and r['relationship']==scope and r['gt_valid']]
            oracle_summary[role][scope]={name:dict(correspondences=len(records),correct=sum(r['conditions'][name]['success'] for r in records))
                for name in ('O_angle','O_support_line','O_direction','O_midpoint','O_width','O_full')}
    write(output/'oracle_summary.json',oracle_summary)
    write(output/'annotation_inconsistencies.json',[r for r in oracles if r['association_or_annotation_inconsistent']])
    lines=['# 固定 R3 的开口几何对照','', '状态：'+selection['status'],
        '前端、描述子、候选身份及 R3 第5000步匹配结果冻结。GT干预仅为离线上限。',
        '', '| 几何/尺度 | selection跨房候选房屋宏平均 | 正确对/固定对数 | 正确房屋 | 弱对应成功率 | 通过收益门槛 |',
        '|---|---:|---:|---:|---:|---|']
    for key,v in table.items():
        r=v['statistics']['selection']['r3']['cross_room'];w=v['statistics']['selection']['weak']['cross_room']
        lines.append(f"| {key} | {r['house_macro']} | {r['correct_pairs']}/{r['pairs']} | {r['correct_houses']} | {w['success_rate']} | {v['passed']} |")
    lines += ['', '基线：'+BASELINE, '获选配置：'+str(selection['selected']),
              '几何失败保留在固定分母；GT无效数量另列。逐房屋结果与配对bootstrap区间见 geometry_selection.json。',
              '无合格几何方案时按预设规则停止训练，不生成新的验证器成绩或20层学习对照。']
    (output/'geometry_report.md').write_text('\n'.join(lines)+'\n')


def train_and_evaluate(output, device, cache, checkpoint, dataset_root, cohort, raw_cache, selected, save):
    from portallayout.datasets.pose_verifier_supervision import build_pose_supervision
    from portallayout.engine.floor_training import train_auxiliaries, AuxiliaryFeatureDataset, predict_auxiliary
    from portallayout.engine.floor_layout_pipeline import FloorModelBundle, estimate_floor, export_floor, failed_floor
    from portallayout.geometry.pose_geometry import FEATURE_VERSION
    from portallayout.engine.pose_verifier_policy import POLICY_VERSION
    from portallayout.utils.pose_inference_guard import inference_guard
    from portallayout.evaluation.pose_verifier import quality, select_workpoint, assess_workpoint
    from portallayout.evaluation.floor_metrics import load_floor_ground_truth,evaluate_floor,write_metrics
    from portallayout.datasets.floor_auxiliary_supervision import pose_target
    from portallayout.visualization.floor_layout import render_evaluated_floor,render_preliminary_materials
    groups={'V0':dict(pose_geometry='dominant_wall',scale_shrinkage=1.),'V1':selected}
    models={};selection_reports={};datasets={}
    for name,config in groups.items():
        folder=output/name; folder.mkdir(exist_ok=True)
        save(stage='pose_feature_export',group=name)
        path=folder/'pose_supervision'/'supervision.json'
        if not path.exists():
            build_pose_supervision(cache,dataset_root,checkpoint,path.parent,device,progress=save,
                                   pose_geometry=config,prediction_cache=raw_cache)
        dataset=AuxiliaryFeatureDataset(path); datasets[name]=dataset
        if len(dataset.data['pair_inventory'])!=23942:raise ValueError('incomplete pair inventory')
        for stage,steps,mode in (('smoke',200,'smoke'),('train',5000,'full')):
            target=folder/stage;save(stage=stage,group=name)
            if not (target/'training_report.json').exists():
                resume=target/'last.pt' if (target/'last.pt').exists() else None
                report=train_auxiliaries(path,checkpoint,target,steps=steps,batch_size=32,seed=123,device=device,
                    mode=mode,tasks=('pose',),selection_metric='cross_room_ap',resume=resume,validation_interval=500,patience=5)
            else:report=read(target/'training_report.json')
            if report['training_status'] not in ('trained','smoke_trained'):
                raise ValueError('insufficient supervision: '+name+' '+stage)
        bundle=FloorModelBundle(dict(pair_checkpoint=str(checkpoint),auxiliary_checkpoint=str(folder/'train/checkpoint.pt')),device)
        models[name]=bundle
        import matplotlib.pyplot as plt
        history=read(folder/'train/pose_history.json');fig,axes=plt.subplots(1,2,figsize=(11,4))
        loss=[h['loss'] if 'loss' in h else h['portallayout.losses'] for h in history]
        axes[0].plot([h['step'] for h in history],[np.mean(loss[max(0,i-49):i+1]) for i in range(len(loss))]);axes[0].set_title('BCE (50-step mean)')
        val=[h for h in history if 'quality' in h]
        axes[1].plot([h['step'] for h in val],[h['quality']['cross_room_house_macro_ap'] for h in val]);axes[1].set_title('Cross-room house macro AP')
        for ext in ('png','svg'):fig.savefig(folder/('training_curves.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
        rows=[r for r in dataset.data['records'] if r['role']=='selection']
        scored=[dict(r,probability=float(p)) for r,p in zip(rows,predict_auxiliary(bundle.pose_model,dataset,rows,device))]
        workpoint=intervals(select_workpoint(scored));workpoint['quality']=quality(scored)
        audit=read(output/'geometry_selection.json');fixed=audit['variants'][audit['baseline']]['statistics']['selection']['r3']
        workpoint['fixed_retrieval_denominators']={scope:fixed[scope]['pairs'] for scope in ('overall','cross_room')}
        if workpoint['status']=='selection_workpoint_found':
            workpoint['fixed_retrieval_correct_rates']={scope:workpoint[scope]['correct_edges']/fixed[scope]['pairs'] if fixed[scope]['pairs'] else None for scope in ('overall','cross_room')}
        workpoint.update(weights_hash=bundle.weights_hash,feature_version=FEATURE_VERSION,candidate_policy=POLICY_VERSION,
                         source_snapshot=cache.snapshot,pose_geometry=config)
        write(folder/'selection_workpoint.json',workpoint);selection_reports[name]=workpoint
        policy={k:workpoint[k] for k in ('status','opening_threshold','weights_hash','feature_version','candidate_policy','source_snapshot','pose_geometry')}
        if workpoint['status']=='selection_workpoint_found':policy.update(threshold=workpoint['threshold'],margin=workpoint['margin'])
        policy['calibrated']=False;policy['fingerprint']=fingerprint(policy);write(folder/'experimental_policy.json',policy)
    # Freeze both checkpoints and working points before opening development scores.
    feasible=[n for n,w in selection_reports.items() if w['status']=='selection_workpoint_found']
    chosen=max(feasible,key=lambda n:(selection_reports[n]['cross_room']['correct_edges'],selection_reports[n]['overall']['correct_edges'],
        selection_reports[n]['overall']['precision'],n=='V0')) if feasible else None
    write(output/'frozen_selection.json',dict(selected_group=chosen,configs=groups,
        checkpoint_hashes={n:content_hash(output/n/'train/checkpoint.pt') for n in groups},
        workpoint_hashes={n:content_hash(output/n/'experimental_policy.json') for n in groups}))
    for name,bundle in models.items():
        dataset=datasets[name];rows=[r for r in dataset.data['records'] if r['role']=='calibration']
        scored=[dict(r,probability=float(p)) for r,p in zip(rows,predict_auxiliary(bundle.pose_model,dataset,rows,device))]
        w=selection_reports[name]
        check=intervals(assess_workpoint(scored,w['threshold'],w['margin'])) if w['status']=='selection_workpoint_found' else dict(passed=False,reason='no_selection_workpoint')
        write(output/name/'development_check.json',dict(quality=quality(scored),fixed_workpoint=check,independent_test=False))
    variants={'A_V0_rank':('V0','verifier_rank','none'),'B_V1_rank':('V1','verifier_rank','none'),
              'C_V0_filter':('V0','verifier_filter','none'),'D_V1_filter':('V1','verifier_filter','none')}
    if chosen:variants['E_selected_filter_pgo']=(chosen,'verifier_filter','robust')
    floors={f['floor_uid']:f for f in cache.floors}; summaries={};coverage={};backend={}
    for role in ('selection','calibration'):
        summaries[role]={}
        for variant,(name,policy,opt) in variants.items():
            metrics=[];bundle=models[name]
            for entry in sorted((c for c in cohort if c['role']==role),key=lambda c:c['house_id']):
                uid=entry['floor_uid'];target=output/'predictions'/variant/uid
                save(stage='floor_comparison',variant=variant,role=role,floor_uid=uid)
                if (target/'floor_result.json').exists():
                    pred=read(target/'floor_result.json');pred['pair_results']=read(target/'pair_results.json')['pairs']
                else:
                    config=dict(groups[name],inference_mode='exploratory',opening_threshold=.7,pose_policy=policy,pose_opt=opt,
                        room_fusion=False,room_association='none',pose_workpoint=str(output/name/'experimental_policy.json'),
                        pair_prediction_cache=str(raw_cache),pose_prediction_cache=str(output/name/'pose_scores'))
                    with inference_guard():
                        try:pred=estimate_floor(floors[uid],bundle,config,cache)
                        except (ValueError,RuntimeError,OSError) as error:
                            pred=failed_floor(floors[uid],error);pred.update(inference_mode='exploratory',validation_status='uncalibrated_preliminary')
                    export_floor(pred,target)
                gt=load_floor_ground_truth(dataset_root,uid);m=evaluate_floor(pred,gt);metrics.append(m)
                render_evaluated_floor(pred,gt,m,target)
                edges=[]
                for pair in pred['pair_results']:
                    if pair.get('calibrated') or pair['pose_status']=='verified':raise ValueError('experimental edge promoted')
                    if pair.get('T_B_to_A') is None:continue
                    a,b=gt['observations'][pair['observation_a']],gt['observations'][pair['observation_b']]
                    truth=np.linalg.inv(a['pose'])@np.asarray(b['pose']) if a['pose_valid'] and b['pose_valid'] else None
                    label,error=pose_target(pair['T_B_to_A'],truth)
                    possible=any(h.get('match_confidence',0.)>=.7 and pose_target(h['T_B_to_A'],truth)[0]==1 for h in pair['pose_hypotheses'])
                    category='correct_pose' if label==1 else 'ignored_pose' if label<0 else 'wrong_hypothesis_selected' if possible else 'no_correct_pose_candidate'
                    edges.append(dict(pair_id=pair['pair_id'],target=label,error=error,error_category=category))
                backend.setdefault(variant,[]).append(dict(floor_uid=uid,role=role,edges=edges,
                    components=[len(c['observation_ids']) for c in pred['components']],
                    optimization_statuses=[c['optimization_status'] for c in pred['components']],status=pred['status']))
                c=coverage.setdefault(variant,dict(floors=0,observations=0,pairs=0,failed_floors=0))
                c['floors']+=1;c['observations']+=len(pred['observation_order']);c['pairs']+=len(pred['pair_results']);c['failed_floors']+=pred['status']=='runtime_failure'
            summaries[role][variant]=write_metrics(metrics,output/'comparison'/role/variant)
    if any((v['floors'],v['observations'],v['pairs'])!=(20,585,9400) for v in coverage.values()):raise ValueError('floor coverage mismatch')
    write(output/'comparison.json',dict(summaries=summaries,coverage=coverage));write(output/'backend_diagnostics.json',backend)
    render_preliminary_materials(summaries,output/'presentation',pipeline_labels=['Frozen RGB frontend','Frozen R3 matching','Geometry overlay + Sim2','Pose verifier','Forest / accepted-edge PGO','Independent seven metrics'])
    import shutil
    cases=read(ROOT/'experiments/research1_preliminary_20261025/presentation/cases.json')
    html=['<!doctype html><meta charset="utf-8"><h1>几何对照与验证器实验</h1><p>开发实验；全部连接未经严格校准。</p>']
    for case_name,case in cases.items():
        html.append('<h2>'+case_name+' / '+case['house_id']+'</h2>')
        for variant in variants:
            for ext in ('png','svg'):
                src=output/'predictions'/variant/case['floor_uid']/('gt_comparison.'+ext)
                dst=output/'presentation'/(case_name+'_'+variant+'.'+ext);shutil.copy2(src,dst)
            html.append('<h3>'+variant+'</h3><img style="width:100%" src="'+case_name+'_'+variant+'.png">')
    (output/'presentation/index.html').write_text('\n'.join(html))
    return dict(coverage=coverage,selected_group=chosen,working_points={n:w['status'] for n,w in selection_reports.items()})


def run(output,device):
    import torch
    from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
    from portallayout.engine.floor_layout_pipeline import FloorModelBundle, exploratory_pair_predictions
    from portallayout.geometry.pose_geometry import opening_overlay,METHODS
    from portallayout.evaluation.pose_geometry import audit_floor,select_geometry
    from portallayout.evaluation.pose_candidate_audit import synthetic_coordinate_audit
    from portallayout.utils.pose_inference_guard import inference_guard
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    lock=open(output/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',started_at=now());start_time=time.perf_counter()
    def save(**kw):state.update(kw,updated_at=now());write(output/'controller_state.json',state)
    try:
        torch.set_num_threads(2)
        checkpoint=ROOT/'experiments/research1_round2_20261024/R3/checkpoint.pt'
        root=ROOT/'data/ZInD-FloorSet-v1/train_20261021'
        raw_cache=ROOT/'experiments/research1_pose_verifier_20261025/pose_supervision/prediction_pairs'
        cache=FrozenOpeningCache(ROOT/'outputs/research1_floor_frontend/main60_20261024')
        cohort=read(ROOT/'experiments/research1_main_20261024/cohort/cohort.json')['cohort']
        files=['portallayout/geometry/pose_geometry.py','portallayout/evaluation/pose_geometry.py','tools/pose_geometry_experiment.py',
               'tools/run_pose_verifier_experiment.py','portallayout/engine/pose_verifier_policy.py','portallayout/geometry/floor_auxiliary_features.py',
               'portallayout/models/floor_auxiliary.py','portallayout/datasets/pose_verifier_supervision.py','portallayout/engine/floor_training.py','portallayout/engine/floor_layout_pipeline.py']
        contract=dict(pair_checkpoint_sha256=content_hash(checkpoint),source_snapshot=cache.snapshot,cohort=cohort,
            raw_cache_hashes={str(p):content_hash(p) for p in sorted(raw_cache.glob('*.json'))},
            annotation_hashes={str(root/'labels'/c['floor_uid']/n):content_hash(root/'labels'/c['floor_uid']/n)
                for c in cohort for n in ('floor_gt.json','observations_gt.jsonl','portal_links.jsonl')},
            code_hashes={p:content_hash(ROOT/p) for p in files},seed=123,opening_threshold=.7,
            stages=['coordinate_audit','geometry_audit','gain_gate','V0_V1_training','working_points','floor_comparison'])
        if (output/'run_contract.json').exists() and read(output/'run_contract.json')!=contract:raise ValueError('experiment contract changed; use a new run')
        write(output/'run_contract.json',contract);write(output/'coordinate_audit.json',synthetic_coordinate_audit())
        bundle=FloorModelBundle(dict(pair_checkpoint=str(checkpoint)),device)
        if bundle.pair_state['selected_step']!=5000:raise ValueError('requires R3 step 5000')
        roles={c['floor_uid']:c['role'] for c in cohort};parts=[];pair_count=0
        for floor in cache.floors:
            uid=floor['floor_uid'];role=roles[uid]
            save(stage='geometry_audit',floor_uid=uid,completed_floors=len(parts))
            if not (raw_cache/(fingerprint(uid)+'.json')).exists():raise ValueError('frozen R3 cache missing floor')
            with inference_guard():obs,raw,_=exploratory_pair_predictions(floor,cache,bundle,dict(pair_prediction_cache=str(raw_cache)))
            pair_count+=len(raw)
            if role not in ('fit','selection'):continue
            target=output/'geometry_audit'/(uid+'.json')
            if target.exists():
                part=read(target);claimed=part.pop('fingerprint')
                if claimed!=fingerprint(part):raise ValueError('audit fingerprint mismatch')
            else:
                overlays={m:{} for m in METHODS}
                with inference_guard():
                    for o in obs:
                        if o['observation_id'] not in cache.records:
                            for method in METHODS: overlays[method][o['observation_id']]=o
                            continue
                        rec,arrays=cache.load(o['observation_id'])
                        for method in METHODS:overlays[method][o['observation_id']]=opening_overlay(rec,arrays,method)
                    pure=dict(floor_uid=uid,source_snapshot=cache.snapshot,overlays=overlays)
                    pure['fingerprint']=fingerprint(pure);write(output/'geometry_overlays'/(uid+'.json'),pure)
                part=audit_floor(cache,floor,raw,root,role,overlays)
                part['fingerprint']=fingerprint(part);write(target,part);part.pop('fingerprint')
            parts.append(part);print('GEOMETRY AUDIT',uid,role,len(part['rows']),flush=True)
        if pair_count!=23942:raise ValueError('expected all 23942 frozen pairs')
        selection=select_geometry(parts);write(output/'geometry_selection.json',selection)
        write(output/'coordinate_audit.json',dict(synthetic=synthetic_coordinate_audit(),real_floors=[p['coordinate_audit'] for p in parts]))
        geometry_report(output,parts,selection)
        diagnostic_figures(output,parts,cache,root)
        save(stage='gain_gate',geometry_status=selection['status'])
        result=dict(training_started=False)
        if selection['status']=='geometry_gain_found':
            result=train_and_evaluate(output,device,cache,checkpoint,root,cohort,raw_cache,selection['selected_config'],save)
            result['training_started']=True
        verified=cache.check_unchanged()
        if content_hash(checkpoint)!=contract['pair_checkpoint_sha256']:raise ValueError('matcher changed')
        if any(content_hash(p)!=h for p,h in contract['raw_cache_hashes'].items()):raise ValueError('frozen matching changed')
        if any(content_hash(p)!=h for p,h in contract['annotation_hashes'].items()):raise ValueError('annotation changed during run')
        if any(content_hash(ROOT/p)!=h for p,h in contract['code_hashes'].items()):raise ValueError('source changed during run')
        write(output/'acceptance.json',dict(result,source_files_verified=verified,pair_inventory=pair_count,geometry_status=selection['status'],
            elapsed_seconds=time.perf_counter()-start_time,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None))
        lines=[(output/'geometry_report.md').read_text(),'\n## 后续阶段\n',json.dumps(result,ensure_ascii=False,indent=2)]
        (output/'final_report.md').write_text('\n'.join(lines))
        save(status='completed',stage='finished',finished_at=now(),**result)
    except BaseException as error:
        save(status='failed',error=repr(error));raise
