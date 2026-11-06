"""Persistent, resumable R3-frozen pose-verifier experiment controller."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import subprocess
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now().astimezone().isoformat()


def run(output,device):
    import torch
    from portallayout.utils.room_feature_cache import content_hash,fingerprint,write_json_atomic as write
    from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
    from portallayout.engine.floor_layout_pipeline import FloorModelBundle,estimate_floor,export_floor,failed_floor
    from portallayout.datasets.pose_verifier_supervision import build_pose_supervision
    from portallayout.engine.floor_training import train_auxiliaries,AuxiliaryFeatureDataset,predict_auxiliary
    from portallayout.evaluation.pose_verifier import quality,select_workpoint,assess_workpoint
    from portallayout.engine.pose_verifier_policy import POLICY_VERSION
    from portallayout.models.floor_auxiliary import POSE_FEATURE_VERSION
    from portallayout.utils.pose_inference_guard import inference_guard
    from portallayout.evaluation.floor_metrics import load_floor_ground_truth,evaluate_floor,write_metrics
    from portallayout.datasets.floor_auxiliary_supervision import pose_target
    from portallayout.visualization.floor_layout import render_evaluated_floor,render_preliminary_materials
    import numpy as np
    output.mkdir(parents=True,exist_ok=True)
    lock=open(output/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',started_at=now(),pid=os.getpid())
    def save(**kw):
        state.update(kw,updated_at=now());write(output/'controller_state.json',state)
    try:
        torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        checkpoint=ROOT/'experiments/research1_round2_20261024/R3/checkpoint.pt'
        root=ROOT/'data/ZInD-FloorSet-v1/train_20261021'
        cache=FrozenOpeningCache(ROOT/'outputs/research1_floor_frontend/main60_20261024')
        cohort=read(ROOT/'experiments/research1_main_20261024/cohort/cohort.json')['cohort']
        assert len(cohort)==60
        files=['portallayout/engine/pose_verifier_policy.py','portallayout/engine/floor_training.py','portallayout/engine/floor_layout_pipeline.py',
               'portallayout/geometry/floor_auxiliary_features.py','portallayout/models/floor_auxiliary.py','portallayout/datasets/pose_verifier_supervision.py',
               'portallayout/evaluation/pose_candidate_audit.py','portallayout/evaluation/pose_verifier.py','tools/run_pose_verifier_experiment.py',
               'portallayout/geometry/exploratory_floor_pose.py','portallayout/evaluation/floor_metrics.py','portallayout/geometry/floor_pose_graph.py']
        contract=dict(pair_checkpoint_sha256=content_hash(checkpoint),source_snapshot=cache.snapshot,feature_version=POSE_FEATURE_VERSION,
            code_hashes={p:content_hash(ROOT/p) for p in files},cohort=cohort,seed=123,opening_threshold=.7,
            stages=['audit','pose_features','smoke200','train5000','selection_workpoint','development_check','floor_comparison'])
        if (output/'run_contract.json').exists() and read(output/'run_contract.json')!=contract:raise ValueError('experiment contract changed; use new run')
        write(output/'run_contract.json',contract)
        pair_state=torch.load(checkpoint,map_location='cpu');assert pair_state['selected_step']==5000
        del pair_state
        labels=output/'pose_supervision'
        save(stage='coordinate_and_candidate_audit')
        if not (labels/'candidate_audit.json').exists():
            build_pose_supervision(cache,root,checkpoint,labels,device,progress=save,audit_only=True)
        save(stage='pose_feature_export')
        if not (labels/'supervision.json').exists():
            build_pose_supervision(cache,root,checkpoint,labels,device,progress=save)
        dataset=AuxiliaryFeatureDataset(labels/'supervision.json')
        assert len(dataset.data['pair_inventory'])==23942
        support={role:{str(t):sum(r['role']==role and r['target']==t for r in dataset.data['records']) for t in (-1,0,1)} for role in ('fit','selection')}
        write(output/'training_support.json',support)
        if any(not support[role]['0'] or not support[role]['1'] for role in support):
            save(status='insufficient_supervision',stage='stopped',reason='fit/selection needs both classes');return
        for stage,steps,mode in [('smoke',200,'smoke'),('train',5000,'full')]:
            save(stage=stage)
            target=output/stage
            if not (target/'training_report.json').exists():
                resume=target/'last.pt' if (target/'last.pt').exists() else None
                training_report=train_auxiliaries(labels/'supervision.json',checkpoint,target,steps=steps,batch_size=32,
                    device=device,mode=mode,tasks=('pose',),selection_metric='cross_room_ap',resume=resume,
                    validation_interval=500,patience=5)
            else:training_report=read(target/'training_report.json')
            if training_report['training_status'] not in ('smoke_trained','trained'):
                save(status='insufficient_supervision',stage='stopped',report=training_report);return
        auxiliary=output/'train/checkpoint.pt';bundle=FloorModelBundle(dict(pair_checkpoint=str(checkpoint),auxiliary_checkpoint=str(auxiliary)),device)
        selection=[r for r in dataset.data['records'] if r['role']=='selection']
        validation=[r for r in dataset.data['records'] if r['role']=='calibration']
        def scored(rows):return [dict(r,probability=float(p)) for r,p in zip(rows,predict_auxiliary(bundle.pose_model,dataset,rows,device))]
        save(stage='selection_workpoint')
        s=scored(selection);workpoint=select_workpoint(s)
        workpoint.update(weights_hash=bundle.weights_hash,candidate_policy=POLICY_VERSION,feature_version=POSE_FEATURE_VERSION,
                         source_snapshot=cache.snapshot,house_protocol=dataset.data['house_protocol'],quality=quality(s))
        write(output/'selection_workpoint.json',workpoint)
        policy={k:workpoint[k] for k in ('status','opening_threshold','weights_hash','candidate_policy','feature_version','source_snapshot')}
        if workpoint['status']=='selection_workpoint_found':policy.update(threshold=workpoint['threshold'],margin=workpoint['margin'])
        policy['calibrated']=False;policy['selection_report_sha256']=content_hash(output/'selection_workpoint.json')
        policy['fingerprint']=fingerprint(policy);write(output/'experimental_policy.json',policy)
        save(stage='fixed_development_check')
        v=scored(validation)
        check=assess_workpoint(v,policy['threshold'],policy['margin']) if policy['status']=='selection_workpoint_found' else dict(passed=False,reason='no_selection_workpoint')
        write(output/'development_check.json',dict(quality=quality(v),fixed_workpoint=check,independent_test=False,calibrated=False))
        floors={f['floor_uid']:f for f in cache.floors};summaries={};references={}
        variants={'geometry_forest':('geometry','none'),'verifier_rank':('verifier_rank','none'),
                  'verifier_filter':('verifier_filter','none'),'verifier_filter_pgo':('verifier_filter','robust')}
        coverage={};backend={}
        for role in ('selection','calibration'):
            summaries[role]={};references[role]={}
            for name,(method,opt) in variants.items():
                metrics=[];refs=[];backend_rows=[]
                for row in sorted((r for r in cohort if r['role']==role),key=lambda r:r['house_id']):
                    uid=row['floor_uid'];target=output/'predictions'/name/uid
                    save(stage='floor_comparison',variant=name,house=row['house_id'],role=role)
                    if (target/'floor_result.json').exists():
                        pred=read(target/'floor_result.json');pred['pair_results']=read(target/'pair_results.json')['pairs']
                    else:
                        cfg=dict(inference_mode='exploratory',opening_threshold=.7,pose_policy=method,pose_opt=opt,
                                 room_fusion=False,room_association='none',pose_workpoint=str(output/'experimental_policy.json'),
                                 pair_prediction_cache=str(labels/'prediction_pairs'),pose_prediction_cache=str(output/'pose_scores'))
                        with inference_guard():
                            try:pred=estimate_floor(floors[uid],bundle,cfg,cache)
                            except (ValueError,RuntimeError,OSError) as error:
                                pred=failed_floor(floors[uid],error);pred.update(inference_mode='exploratory',validation_status='uncalibrated_preliminary')
                        export_floor(pred,target)
                    gt=load_floor_ground_truth(root,uid);m=evaluate_floor(pred,gt);metrics.append(m)
                    render_evaluated_floor(pred,gt,m,target)
                    edges=[]
                    for pair in pred['pair_results']:
                        assert not pair.get('calibrated') and pair['pose_status']!='verified'
                        if pair.get('T_B_to_A') is None:continue
                        a,b=gt['observations'][pair['observation_a']],gt['observations'][pair['observation_b']]
                        truth=np.linalg.inv(a['pose'])@np.asarray(b['pose']) if a['pose_valid'] and b['pose_valid'] else None
                        label,error=pose_target(pair['T_B_to_A'],truth);edges.append(dict(pair_id=pair['pair_id'],target=label,error=error))
                    backend_rows.append(dict(floor_uid=uid,edges=edges,optimization_statuses=[c['optimization_status'] for c in pred['components']],
                        component_sizes=[len(c['observation_ids']) for c in pred['components']],status=pred['status']))
                    refs.append(dict(floor_uid=uid,house_id=row['house_id'],result_file=str(target/'floor_result.json'),metrics=m['metrics']))
                    count=coverage.setdefault(name,dict(floors=0,observations=0,pairs=0,failed_floors=0))
                    count['floors']+=1;count['observations']+=len(pred['observation_order']);count['pairs']+=len(pred['pair_results']);count['failed_floors']+=pred['status']=='runtime_failure'
                summaries[role][name]=write_metrics(metrics,output/'comparison'/role/name)
                references[role][name]=refs;backend.setdefault(role,{})[name]=backend_rows
        for c in coverage.values():assert (c['floors'],c['observations'],c['pairs'])==(20,585,9400)
        write(output/'comparison.json',dict(summaries=summaries,coverage=coverage,references=references))
        write(output/'backend_diagnostics.json',backend)
        render_preliminary_materials(summaries,output/'presentation',pipeline_labels=[
            '整层 RGB 全景\n冻结单图前端','局部布局与开口\nR3 第 5000 步冻结','双向互选＋dustbin\n统一候选与去重',
            '位姿验证器\n排序、拒绝与歧义保留','全局定位\n森林 / robust PGO','七指标独立评价\n关闭房间融合'])
        presentation(output,references)
        verified=cache.check_unchanged()
        assert content_hash(checkpoint)==contract['pair_checkpoint_sha256']
        assert all(content_hash(ROOT/p)==h for p,h in contract['code_hashes'].items())
        write(output/'acceptance.json',dict(status='completed',source_files_verified=verified,coverage=coverage,
              workpoint_status=policy['status'],strict_gate_changed=False,calibrated=False,room_trained=False))
        report(output,summaries,policy,check)
        save(status='completed',stage='finished',workpoint_status=policy['status'],finished_at=now())
    except BaseException as error:
        save(status='failed',error=repr(error));raise


def presentation(output,references):
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=read(output/'train/pose_history.json');fig,axes=plt.subplots(1,2,figsize=(12,4))
    values=[h['loss'] if 'loss' in h else h['portallayout.losses'] for h in history];smooth=[sum(values[max(0,i-49):i+1])/len(values[max(0,i-49):i+1]) for i in range(len(values))]
    axes[0].plot([h['step'] for h in history],smooth);axes[0].set_title('Training BCE (50-step mean)')
    val=[h for h in history if 'quality' in h]
    axes[1].plot([h['step'] for h in val],[h['quality']['cross_room_house_macro_ap'] for h in val]);axes[1].set_title('Selection cross-room house macro AP')
    for ax in axes:ax.set_xlabel('step');ax.grid(alpha=.2)
    for ext in ('svg','png'):fig.savefig(output/'presentation'/('training_curves.'+ext),bbox_inches='tight',dpi=160)
    plt.close(fig)
    old=ROOT/'experiments/research1_preliminary_20261025/presentation/cases.json'
    cases=read(old);parts=[]
    for name,case in cases.items():
        uid=case['floor_uid'];folder=output/'presentation'/name;folder.mkdir(exist_ok=True)
        for variant in ('geometry_forest','verifier_rank','verifier_filter','verifier_filter_pgo'):
            for ext in ('png','svg'):
                shutil.copy2(output/'predictions'/variant/uid/('gt_comparison.'+ext),folder/(variant+'.'+ext))
        parts.append('<h2>'+name+' / '+case['house_id']+'</h2>'+''.join('<h3>'+v+'</h3><img src="'+name+'/'+v+'.png">' for v in ('geometry_forest','verifier_rank','verifier_filter','verifier_filter_pgo')))
    (output/'presentation/index.html').write_text('<!doctype html><meta charset="utf-8"><title>位姿验证器实验</title><style>body{max-width:1250px;margin:30px auto;font-family:sans-serif}img{width:100%}</style><h1>冻结 R3 的位姿验证器实验</h1><p>开发实验，未经严格校准。保留断连；不将连通率当成定位准确率。案例沿用上轮 selection 选定房屋。</p><img src="pipeline.png"><img src="results_table.png"><img src="training_curves.png">'+''.join(parts))


def report(output,summaries,policy,check):
    keys=['overall_iou','localization_coverage','translation_error_cm','rotation_error_degrees','wall_f1','junction_f1','wall_distance_cm']
    lines=['# 冻结 R3 的位姿验证器实验','', '完成时间：'+now(),'',
           '工作点：'+policy['status']+'；固定开发检查通过：'+str(check['passed'])+'。所有连接仍为 tentative，未改变严格模式。',
           '', '| 数据 / 方法 | IoU | 正确定位覆盖 | 平移 cm | 旋转 ° | Wall F1 | Junction F1 | 墙距 cm |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for role,variants in summaries.items():
        for name,s in variants.items():lines.append('| '+role+' / '+name+' | '+' | '.join('N/A' if s['metrics'][k] is None else '{:.4f}'.format(s['metrics'][k]) for k in keys)+' |')
    lines+=['','每组 20 层 / 585 全景 / 9400 对。失败和断连保留在分母。房屋1289缺米制尺度，相关指标N/A。',
            '无可行工作点时，过滤两组不接边；排序组仅为诊断。误差中位数下降可能来自拒绝困难观测，必须同时看正确边数和覆盖率。',
            '材料：presentation/index.html；候选上限：pose_supervision/candidate_audit.json；监督覆盖：training_support.json。',
            '本轮未训练匹配器或房间模块；正确候选不足需要后续针对匹配与几何的独立实验。']
    (output/'final_report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--status',action='store_true');p.add_argument('--unit',default='research1-pose-verifier-20261025.service')
    p.add_argument('--experiment',choices=('pose-verifier','geometry'),default='pose-verifier');args=p.parse_args()
    output=Path(args.output).resolve();os.chdir(ROOT)
    if args.status:
        from portallayout.utils.room_feature_cache import write_json_atomic
        state=read(output/'controller_state.json') if (output/'controller_state.json').exists() else {'status':'not_started'}
        active=subprocess.run(['systemctl','--user','is-active',args.unit],capture_output=True,text=True).stdout.strip()
        log=output/'controller.log'
        write_json_atomic(output/'monitor_status.json',dict(checked_at=now(),service=active,controller=state,
             log_tail=log.read_text(errors='replace').splitlines()[-16:] if log.exists() else []))
    elif args.experiment=='geometry':
        from tools.pose_geometry_experiment import run as run_geometry
        run_geometry(output,args.device)
    else:run(output,args.device)

if __name__=='__main__':main()
