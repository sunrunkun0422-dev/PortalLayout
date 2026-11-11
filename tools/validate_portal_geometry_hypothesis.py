#!/usr/bin/env python3
"""Frozen-prediction geometry attribution, not a learned-method evaluation."""
import argparse, copy, csv, json, math, shutil, subprocess, sys, time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.validate_type_conditioned_prior import digest,stable,write,cache,load_matcher,mask
from portallayout.models.cross_scene_matcher import opening_probabilities_to_intervals,circular_interval_iou
from tools.train_zind_cross_scene_matcher import predicted_candidate_masks,candidate_targets_from_affinity
from portallayout.geometry.candidate_local_portal_geometry import panorama_ray,bi_to_zind,estimate_candidate_local_portal_geometry
from portallayout.geometry.joint_layout import build_joint_layout_from_endpoints

DEFAULT=ROOT/'experiments/portal_geometry_hypothesis_20260927'
OLD=ROOT/'experiments/type_prior_probe_20260923'

def prepare(a):
    if a.output.exists():raise FileExistsError(a.output)
    path=ROOT/'data/ZInD-ObservablePairs-v1/training/best1_balanced_seed123/val.json'
    source=json.loads(path.read_text());byhouse=defaultdict(list)
    for r in source['pairs']:
        if r['supervision']['is_match']:byhouse[r['scene_id']].append(r)
    chosen=[]
    for i,h in enumerate(sorted(byhouse,key=lambda h:stable(123,'probe_houses',h))):
        scope=('cross_complete_room','within_complete_partial')[i%2]
        pool=[r for r in byhouse[h] if r['relation_scope']==scope] or byhouse[h]
        chosen.append(min(pool,key=lambda r:stable(123,'probe_pair',r['id'])))
    views={}
    for r in chosen:
        for s in ['a','b']:
            k=r['image_'+s];views[k]=dict(image=k,house=r['scene_id'],candidates=r['candidates_'+s],group='monitor',data_root=source['dataRoot'],cache_key=stable(k))
    a.output.mkdir(parents=True);(a.output/'predictions').mkdir()
    sha=digest(a.checkpoint);reuse=0
    for k,v in views.items():
        p=OLD/'predictions'/(v['cache_key']+'.pt')
        if p.exists():
            d=torch.load(p,map_location='cpu')
            assert d['checkpoint_sha256']==sha and d['rgb_sha256']==digest(Path(v['data_root'])/k)
            shutil.copyfile(p,a.output/'predictions'/p.name);reuse+=1
    write(a.output/'selection.json',dict(pairs={'monitor':chosen},views=views))
    write(a.output/'protocol.json',dict(date='2026-09-27',checkpoint=str(a.checkpoint),checkpoint_sha256=sha,manifest=str(path),manifest_sha256=digest(path),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),initial_status=subprocess.check_output(['git','status','--short'],text=True),
        seed=123,sampling='one positive pair per actual monitor house; alternating relation scope, hash selection without predictions',houses=len(chosen),pairs=len(chosen),views=len(views),reused_caches=reuse,
        opening_threshold=.47,min_width_tokens=2,max_candidates=12,GT_association_iou=.3,
        pose_success_thresholds={'yaw_deg':10,'translation_m':.5,'relative_scale_error':.1},bootstrap_repeats=2000,
        scope='exploratory internal monitor previously used for model selection, not unseen-house test',
        oracle_interpretation='GT geometry interventions test opportunity, not learnability or deployable gains; no GT poses enter solver',
        coordinate_contract='all endpoints and contours in camera-height-normalized ZInD xy; target T_norm = inv(D_A) T_local D_B; meters per norm unit = a_local_to_meters * height_A'))
    print('prepared',len(chosen),'pairs',len(views),'views',reuse,'reused',flush=True)

def normalized_target(T,ha,hb):
    da=np.diag([ha,ha,1.]);db=np.diag([hb,hb,1.]);return np.linalg.inv(da)@np.asarray(T)@db

def solve(ea,eb,ba,bb):
    j=build_joint_layout_from_endpoints({}, {},ea,eb,boundary_a=ba,boundary_b=bb,calibrate_scale=True)
    return np.asarray(j['alignment']['roomBToWorld']),j['alignment']['mapping'] if 'mapping' in j['alignment'] else j['alignment'].get('endpointMapping')

def metrics(T,G,factor):
    yaw=lambda t:math.atan2(t[1,0],t[0,0]);diff=yaw(T)-yaw(G);yawerr=abs(math.degrees(math.atan2(math.sin(diff),math.cos(diff))))
    scale=lambda t:math.sqrt(abs(np.linalg.det(t[:2,:2])))
    trans=float(np.linalg.norm(T[:2,2]-G[:2,2])*factor);se=abs(scale(T)/scale(G)-1)
    return dict(yaw_deg=yawerr,translation_m=trans,log_scale_error=abs(math.log(scale(T)/scale(G))),relative_scale_error=se,
        success=bool(yawerr<=10 and trans<=.5 and se<=.1),pose_success_without_scale=bool(yawerr<=10 and trans<=.5))

def order_gt(gt,interval):
    rays=bi_to_zind(np.stack([panorama_ray(i) for i in interval]));unit=gt/np.linalg.norm(gt,axis=1,keepdims=True)
    return gt if np.sum(rays*unit)>=np.sum(rays*unit[::-1]) else gt[::-1]

def decompose(e):
    center=e.mean(0);d=e[1]-e[0];w=np.linalg.norm(d)
    if w<1e-8:raise ValueError('zero width')
    return center,d/w,w

def edits(base,gt):
    c,t,w=decompose(base);cg,tg,wg=decompose(gt)
    if np.dot(t,tg)<0:tg=-tg
    segment=lambda center,tangent,width:np.stack([center-.5*width*tangent,center+.5*width*tangent])
    rays=base/np.linalg.norm(base,axis=1,keepdims=True)
    normal=np.array([-tg[1],tg[0]]);offset=float(normal@cg);den=rays@normal
    support=None
    if np.all(np.abs(den)>1e-6):
        radius=offset/den
        if np.all(radius>0):support=rays*radius[:,None]
    return dict(O_direction=segment(c,tg,w),O_midpoint=segment(cg,t,w),O_width=segment(c,t,wg),
        O_radial=rays*np.linalg.norm(gt,axis=1,keepdims=True),O_support_line=support,O_full=gt)

def endpoint_error(e,g):
    return min(float(np.linalg.norm(e-g,axis=1).mean()),float(np.linalg.norm(e-g[::-1],axis=1).mean()))

def pano_lookup(payload):
    result={}
    for floor,rooms in payload['merger'].items():
        for room,partials in rooms.items():
            for partial,panos in partials.items():
                for name,raw in panos.items():result[raw['image_path']]=raw
    return result

def angle_depth_oracle(depth,gt):
    # GT endpoint angles but only predicted radial depths, circular linear interpolation.
    unit=gt/np.linalg.norm(gt,axis=1,keepdims=True)
    u=(np.arctan2(-unit[:,0],unit[:,1])/(2*np.pi)+.5)%1
    at=u*256-.5;lo=np.floor(at).astype(int);frac=at-lo
    radius=depth[lo%256]*(1-frac)+depth[(lo+1)%256]*frac
    return unit*radius[:,None]

def wall_support(gt,raw,height):
    best=1e9
    for name in ['layout_raw','layout_complete']:
        poly=np.asarray(raw[name]['vertices'])/height
        for x,y in zip(poly,np.roll(poly,-1,axis=0)):
            v=y-x;den=float(v@v)
            if den<1e-12:continue
            tau=np.clip((gt-x)@v/den,0,1);error=np.max(np.linalg.norm(gt-(x+tau[:,None]*v),axis=1));best=min(best,float(error))
    return dict(max_endpoint_to_GT_wall_segment_distance=best,GT_support_within_005=best<=.05)

def run(a):
    start=time.time();sel=json.loads((a.output/'selection.json').read_text());protocol=json.loads((a.output/'protocol.json').read_text())
    assert digest(a.checkpoint)==protocol['checkpoint_sha256']
    _,matcher=load_matcher(a.checkpoint);data={k:torch.load(a.output/'predictions'/(v['cache_key']+'.pt'),map_location='cpu') for k,v in sel['views'].items()}
    assert all(d['checkpoint_sha256']==protocol['checkpoint_sha256'] for d in data.values())
    rows=[];opening_rows=[];annotation_hashes={}
    for index,r in enumerate(sel['pairs']['monitor']):
        ks=[r['image_'+s] for s in ['a','b']];ds=[data[k] for k in ks]
        pr=[d['logits'].sigmoid()[None] for d in ds]
        intervals=[opening_probabilities_to_intervals(p[0],.47,2,12) for p in pr]
        cm,valid=zip(*[predicted_candidate_masks(p,.47,2,12) for p in pr])
        with torch.no_grad():
            out=matcher(ds[0]['feature'][None],ds[1]['feature'][None],ds[0]['enclosed'][None],ds[0]['extended'][None],ds[1]['enclosed'][None],ds[1]['extended'][None],candidate_masks_a=cm[0],candidate_masks_b=cm[1],candidate_valid_a=valid[0],candidate_valid_b=valid[1])
        best=out['best_candidate_pair'][0].tolist();accepted=all(i>=0 for i in best)
        affinity=torch.zeros(1,256,256)
        for ia,ib in r['supervision']['correspondence_pairs']:
            affinity[0]+=torch.tensor(mask(r['candidates_a'][ia]['tokenInterval']))[:,None]*torch.tensor(mask(r['candidates_b'][ib]['tokenInterval']))[None,:]
        targets=candidate_targets_from_affinity(cm[0],affinity,cm[1])[0].bool()
        recalled=bool(targets.any());correct=accepted and bool(targets[best[0],best[1]])
        path=Path(sel['views'][ks[0]]['data_root'])/r['scene_id']/'zind_data.json'
        payload=json.loads(path.read_text());annotation_hashes[str(path)]=digest(path);lookup=pano_lookup(payload)
        raw=[lookup[k.split('/',1)[1]] for k in ks];heights=[float(x['camera_height']) for x in raw]
        G=normalized_target(r['supervision']['relative_transform_b_to_a'],*heights)
        factor=float(r['supervision']['a_local_to_meters'])*heights[0]
        contours=[bi_to_zind(np.stack([panorama_ray(t)*max(abs(float(d['enclosed'][t])),1e-3) for t in range(256)])) for d in ds]
        row=dict(id=r['id'],house=r['scene_id'],scope=r['relation_scope'],accepted=accepted,recalled=recalled,matcher_correct=correct,best=best,intervals=intervals,
            camera_heights=heights,metric_factor=factor,gt_pose_normalized=G.tolist(),conditions={},status='correct' if correct else 'selection_failure' if recalled else 'candidate_miss')
        def evaluate_endpoints(ea,eb):
            if ea is None or eb is None:return dict(valid=False,error='invalid endpoint geometry')
            try:
                T,mapping=solve(ea,eb,*contours)
                return dict(valid=True,**metrics(T,G,factor),T=T.tolist(),mapping=mapping,endpoints_A=np.asarray(ea).tolist(),endpoints_B=np.asarray(eb).tolist())
            except (ValueError,np.linalg.LinAlgError) as ex:return dict(valid=False,error=str(ex))
        # Production selected correspondence: no oracle substitution of missed/wrong matches.
        if accepted:
            es=[contours[s][list(intervals[s][best[s]])] for s in range(2)]
            row['conditions']['B_raw']=evaluate_endpoints(*es)
            m2=[]
            for s in range(2):
                result=estimate_candidate_local_portal_geometry(bi_to_zind(contours[s]),intervals[s][best[s]],256,pr[s][0].numpy())
                m2.append(None if result.primary is None or not result.geometry_valid else np.asarray(result.primary.endpoints_local_zind_xy))
            row['conditions']['B_existing_M2']=evaluate_endpoints(*m2)
            if correct:
                # Associate GT solely by angular IoU, never by pose error.
                options=[]
                for ia,ib in r['supervision']['correspondence_pairs']:
                    ious=[circular_interval_iou(intervals[s][best[s]],r['candidates_'+('a','b')[s]][j]['tokenInterval'],256) for s,j in enumerate([ia,ib])]
                    if min(ious)>=.3:options.append((min(ious),sum(ious),-ia,-ib))
                if options:
                    _,_sum,ia,ib=max(options);ids=[-ia,-ib];row['gt_association']=ids
                    gt=[order_gt(np.asarray(r['candidates_'+('a','b')[s]][ids[s]]['localEndpoints'])/heights[s],intervals[s][best[s]]) for s in range(2)]
                    row['type_pair']=[r['candidates_'+('a','b')[s]][ids[s]]['type'] for s in range(2)]
                    row['wall_support']=[wall_support(gt[s],raw[s],heights[s]) for s in range(2)]
                    replacements=[edits(es[s],gt[s]) for s in range(2)]
                    for name in replacements[0]:row['conditions'][name]=evaluate_endpoints(replacements[0][name],replacements[1][name])
                    angular=[angle_depth_oracle(ds[s]['enclosed'].abs().numpy(),gt[s]) for s in range(2)]
                    row['conditions']['O_angle']=evaluate_endpoints(*angular)
                    if all(e is not None for e in m2):
                        mr=[edits(m2[s],gt[s]) for s in range(2)]
                        for name in ['O_direction','O_midpoint','O_width','O_radial','O_full']:
                            row['conditions']['M2_'+name]=evaluate_endpoints(mr[0][name],mr[1][name])
                    for s in range(2):
                        _,t,w=decompose(es[s]);_,tg,wg=decompose(gt[s])
                        opening_rows.append(dict(pair_id=r['id'],house=r['scene_id'],side=s,type=row['type_pair'][s],
                            angular_interval=intervals[s][best[s]],GT_support=row['wall_support'][s]['GT_support_within_005'],
                            endpoint_error_normalized=endpoint_error(es[s],gt[s]),axis_error_deg=float(np.degrees(np.arccos(np.clip(abs(t@tg),0,1)))),
                            relative_width_error=abs(w/wg-1),zero_delta_fraction=float((ds[s]['delta'][torch.tensor(mask(intervals[s][best[s]]))]==0).float().mean())))
                else:row['association_warning']='legacy dense affinity recalled but no individual correspondence meets 0.3 on both sides'
        # Separate GT correspondence/angle diagnostic: useful even if matcher missed, NOT deployable.
        ia,ib=sorted(r['supervision']['correspondence_pairs'])[0]
        gts=[np.asarray(r['candidates_'+('a','b')[s]][j]['localEndpoints'])/heights[s] for s,j in enumerate([ia,ib])]
        gt_angles_pred_depth=[angle_depth_oracle(ds[s]['enclosed'].abs().numpy(),gts[s]) for s in range(2)]
        row['gt_correspondence_diagnostic']=dict(pred_depth=evaluate_endpoints(*gt_angles_pred_depth),full_geometry=evaluate_endpoints(*gts),selection='lexicographically first GT correspondence, not best pose')
        mapped=(np.c_[gts[1],np.ones(2)]@G.T)[:,:2]
        row['GT_endpoint_transform_residual_m']=endpoint_error(mapped,gts[0])*factor
        rows.append(row)
        if (index+1)%10==0:print('evaluated',index+1,'/',len(sel['pairs']['monitor']),flush=True)
    write(a.output/'per_pair.json',rows);write(a.output/'per_opening.json',opening_rows)
    write(a.output/'run.json',dict(seconds=time.time()-start,annotation_hashes=annotation_hashes,checkpoint_sha256=protocol['checkpoint_sha256'],cuda_used=False,training=False))
    summarize(a,rows,opening_rows)

def stats(rows,condition):
    values=[r['conditions'].get(condition,{}) for r in rows];valid=[x for x in values if x.get('valid')]
    result=dict(n=len(rows),valid=len(valid),success=sum(x['success'] for x in valid),success_rate=sum(x['success'] for x in valid)/max(1,len(rows)))
    for key in ['yaw_deg','translation_m','relative_scale_error']:
        result[key+'_median']=float(np.median([x[key] for x in valid])) if valid else None
        result[key+'_mean']=float(np.mean([x[key] for x in valid])) if valid else None
    return result

def summarize(a,rows,opening_rows):
    correct=[r for r in rows if 'O_full' in r['conditions']];conditions=['B_raw','B_existing_M2','O_angle','O_direction','O_midpoint','O_width','O_radial','O_support_line','O_full']
    summary=dict(counts=dict(total=len(rows),houses=len({r['house'] for r in rows}),accepted=sum(r['accepted'] for r in rows),recalled=sum(r['recalled'] for r in rows),correct_match=sum(r['matcher_correct'] for r in rows),oracle_eligible=len(correct)),
        correct_match_subset={c:stats(correct,c) for c in conditions},all_pairs_baselines={c:stats(rows,c) for c in ['B_raw','B_existing_M2']})
    rng=np.random.RandomState(123);bootstrap={}
    for c in conditions[1:]:
        dd=[]
        for _ in range(2000):
            sampled=[correct[i] for i in rng.randint(len(correct),size=len(correct))]
            x=stats(sampled,c);b=stats(sampled,'B_raw')
            dd.append([x['success_rate']-b['success_rate'],x['yaw_deg_median']-b['yaw_deg_median'],x['translation_m_median']-b['translation_m_median']])
        bootstrap[c]=dict(metrics=['delta_success_rate','delta_median_yaw_deg','delta_median_translation_m'],ci95=np.quantile(dd,[.025,.975],axis=0).tolist(),paired_houses=len(correct))
    summary['by_type']={typ:{c:stats([r for r in correct if all(t==typ for t in r['type_pair'])],c) for c in conditions} for typ in ['door','opening']}
    summary['by_scope']={scope:{c:stats([r for r in correct if r['scope']==scope],c) for c in conditions} for scope in ['cross_complete_room','within_complete_partial']}
    summary['M2_oracles']={c:stats([r for r in correct if 'M2_O_full' in r['conditions']],c) for c in ['B_existing_M2','M2_O_direction','M2_O_midpoint','M2_O_width','M2_O_radial','M2_O_full']}
    summary['transitions']={c:dict(Counter(('success' if r['conditions']['B_raw'].get('success') else 'failure')+' -> '+('success' if r['conditions'].get(c,{}).get('success') else 'failure') for r in correct)) for c in conditions}
    for mode in ['pred_depth','full_geometry']:
        temporary=[dict(conditions={mode:r['gt_correspondence_diagnostic'][mode]}) for r in rows]
        summary['GT_correspondence_'+mode]=stats(temporary,mode)
    write(a.output/'summary.json',summary);write(a.output/'house_bootstrap.json',bootstrap)
    with (a.output/'pose_predictions.csv').open('w') as f:
        fields=['pair_id','house','scope','matcher_correct','condition','valid','yaw_deg','translation_m','relative_scale_error','success'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            for c,v in r['conditions'].items():w.writerow(dict(pair_id=r['id'],house=r['house'],scope=r['scope'],matcher_correct=r['matcher_correct'],condition=c,**{k:v.get(k) for k in fields[5:]}))
    print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','cache','run'],required=True);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/End_To_End/zind_observable_v1/best1_balanced_seed123_20260910/best.pt');p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(4);{'prepare':prepare,'cache':cache,'run':run}[a.stage](a)
if __name__=='__main__':main()
