#!/usr/bin/env python3
"""Build a fit/selection review package without rewriting training labels."""
import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from portallayout.datasets.floor_auxiliary_supervision import house_role
from portallayout.datasets.opening_pair_supervision import complete_targets
from portallayout.datasets.zind_floor_dataset import safe_path
from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
from portallayout.utils.room_feature_cache import content_hash,fingerprint,write_json_atomic
from tools.report_opening_matches import checked_json


def select_cases(cases, count=24, house_cap=4):
    selected=[];counts=Counter();seen=set()
    # Equal fit/selection allocation first, then fill remaining capacity.
    for role, quota in [('fit',count//2),('selection',count-count//2),(None,count)]:
        pools=defaultdict(list)
        for row in cases:
            if role is None or row['role']==role:pools[row['house_id']].append(row)
        for rows in pools.values():rows.sort(key=lambda r:(-r.get('priority',r.get('confidence') or 0.),fingerprint([123,r['case_id']])))
        houses=sorted(pools,key=lambda h:fingerprint([123,h]));added=0
        while added<quota and len(selected)<count:
            changed=False
            for house in houses:
                if counts[house]>=house_cap:continue
                row=next((x for x in pools[house] if x['case_id'] not in seen),None)
                if row is None:continue
                selected.append(row);seen.add(row['case_id']);counts[house]+=1;added+=1;changed=True
                if added>=quota or len(selected)>=count:break
            if not changed:break
    return selected


def automatic_audit(weak, cache, annotations, rows):
    issues=[];geometry_count=0
    for path,digest in weak['source_files_sha256'].items():
        if content_hash(path)!=digest:issues.append('annotation_hash_changed:'+path)
    for r in rows:
        a,b=complete_targets(np.asarray(r['cells']).reshape(len(r['candidate_ids_a']),len(r['candidate_ids_b'])))
        if a!=r['target_a'] or b!=r['target_b']:issues.append('assignment_mask_mismatch:'+r['pair_id'])
        cells=np.asarray(r['weak_cells']).reshape(len(r['candidate_ids_a']),len(r['candidate_ids_b']))
        if (cells==1).sum(0).max(initial=0)>1 or (cells==1).sum(1).max(initial=0)>1:
            issues.append('contradictory_positive_identity:'+r['pair_id'])
    for uid,a in annotations.items():
        if uid not in cache.records:continue
        if not a.get('pose_gt_available'):continue
        g=np.asarray(a['G_local_to_floor_gt'],float)
        for item in a['interfaces_by_layout']['layout_raw']:
            local=np.asarray(item['local_endpoints'],float);expected=np.asarray(item['floor_endpoints'],float)
            actual=(np.c_[local,np.ones(len(local))]@g.T)[:,:2];geometry_count+=1
            if not np.isfinite(actual).all() or not np.allclose(actual,expected,atol=1e-5,rtol=1e-5):
                issues.append('coordinate_inconsistency:'+item['interface_observation_id'])
    return dict(status='passed' if not issues else 'failed',issues=issues,checked_pairs=len(rows),
                checked_interface_transforms=geometry_count,source_files_verified=cache.check_unchanged(),
                note='Checks identity/masks/source transforms; physical correspondence correctness still needs human review.')


def summarize_review(package, review):
    if review['package_fingerprint']!=package['fingerprint']:raise ValueError('review package mismatch')
    source={c['case_id']:c for c in package['cases']};seen=set();counts=defaultdict(Counter)
    for row in review['cases']:
        uid=row['case_id'];judgement=row.get('judgement')
        if uid not in source or uid in seen:raise ValueError('unknown or repeated review case')
        seen.add(uid)
        if judgement not in (None,'正确','错误','不确定'):raise ValueError('invalid human judgement')
        if judgement is not None and (not review.get('reviewer') or not review.get('reviewed_at')):
            raise ValueError('completed review requires reviewer and date')
        counts[source[uid]['group']][judgement or '待审核']+=1
    if seen!=set(source):raise ValueError('review omits cases; retain unknown/pending entries')
    return dict(source='human_reviewed_diagnostic_only',counts=dict(counts),
                original_labels_modified=False,training_labels_overridden=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report');p.add_argument('--weak-supervision');p.add_argument('--prediction-cache')
    p.add_argument('--dataset');p.add_argument('--manifests');p.add_argument('--raw-root')
    p.add_argument('--output',required=True);p.add_argument('--review-file');p.add_argument('--package')
    args=p.parse_args(argv);out=Path(args.output)
    if out.exists():p.error('output must be new')
    if args.review_file:
        result=summarize_review(checked_json(args.package),json.loads(Path(args.review_file).read_text()))
        write_json_atomic(out,result);return
    if not all((args.report,args.weak_supervision,args.prediction_cache,args.dataset,args.manifests,args.raw_root)):
        p.error('render requires report, weak supervision, prediction cache, dataset, manifests and raw root')
    report=checked_json(args.report);weak=checked_json(args.weak_supervision);cache=FrozenOpeningCache(args.prediction_cache)
    if report['source_snapshot']!=weak['source_snapshot'] or report['source_snapshot']!=cache.snapshot:
        raise ValueError('review cache/label/report mismatch')
    protocol=report['house_protocol'];rows=[r for r in weak['pairs'] if house_role(r['house_id'],protocol) in ('fit','selection')]
    rowmap={r['pair_id']:r for r in rows};annotations={};floors={}
    for row in rows:floors[row['floor_uid']]=(row['house_id'],house_role(row['house_id'],protocol))
    obs_context={}
    for uid,context in floors.items():
        path=safe_path(Path(args.dataset),'labels/'+uid+'/observations_gt.jsonl')
        for line in path.read_text().splitlines():
            a=json.loads(line);annotations[a['observation_id']]=a;obs_context[a['observation_id']]=context
    out.mkdir(parents=True);audit=automatic_audit(weak,cache,annotations,rows);write_json_atomic(out/'automatic_audit.json',audit)
    if audit['status']!='passed':raise ValueError('systemic label audit failed; inspect automatic_audit.json')
    groups={k:[] for k in ('positive_rejected','high_score_negative','high_score_unknown','ambiguous_or_missed')}
    def add(group,house,role,sides,**values):
        case=dict(group=group,house_id=house,role=role,sides=sides,**values)
        case['case_id']=fingerprint(case);groups[group].append(case)
    with Path(args.report).with_suffix('.pairs.jsonl').open() as stream:
        for line in stream:
            pred=json.loads(line);row=rowmap.get(pred['pair_id'])
            if row is None:continue
            role=house_role(row['house_id'],protocol)
            for d in pred['positive_diagnostics']:
                if not d.get('mutual_with_dustbin'):
                    confidence=None
                    if d['index_a'] in pred.get('indices_a',[]) and d['index_b'] in pred.get('indices_b',[]):
                        i=pred['indices_a'].index(d['index_a']);j=pred['indices_b'].index(d['index_b'])
                        confidence=min(pred['conditional_ab'][i][j],pred['conditional_ba'][j][i])
                    add('positive_rejected',row['house_id'],role,
                        [dict(observation_id=row['observation_'+s],candidate_index=d['index_'+s]) for s in ('a','b')],
                        pair_id=row['pair_id'],weak_target=1,confidence=confidence,diagnostics=d)
            for m in pred['matches']:
                if m['target'] not in (0,-1):continue
                group='high_score_negative' if m['target']==0 else 'high_score_unknown'
                add(group,row['house_id'],role,
                    [dict(observation_id=row['observation_'+s],candidate_index=m['index_'+s]) for s in ('a','b')],
                    pair_id=row['pair_id'],weak_target=m['target'],confidence=m['confidence'])
    for uid,links in weak['associations'].items():
        if uid not in obs_context:continue
        house,role=obs_context[uid]
        for i,link in enumerate(links):
            if link['status']!='associated':
                add('ambiguous_or_missed',house,role,[dict(observation_id=uid,candidate_index=i)],
                    association=link,confidence=None,priority=1-link['iou'])
        recalled={x['interface_id'] for x in links if x['status']=='associated'}
        for item in annotations[uid]['interfaces_by_layout']['layout_raw']:
            if (item['kind'] in ('door','opening') and item['interface_observation_id'] not in recalled
                    and (item.get('projection') or {}).get('length_u',0)>0):
                add('ambiguous_or_missed',house,role,[dict(observation_id=uid,candidate_index=None,
                    interface_id=item['interface_observation_id'])],confidence=None,priority=1.,reason='no_unique_candidate_association')
    cases=[c for group in groups.values() for c in select_cases(group)]
    rgb={}
    for path in Path(args.manifests).glob('*.json'):
        for obs in json.loads(path.read_text())['observations']:rgb[obs['observation_id']]=obs['rgb']['relative_path']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from tools.visual_audit_zind_candidate_failures import _overlay_panorama
    (out/'images').mkdir();cards=[];source_images={}
    for number,case in enumerate(cases,1):
        sides=case['sides'];fig,axes=plt.subplots(len(sides),2,figsize=(14,3*len(sides)),squeeze=False)
        for index,side in enumerate(sides):
            uid=side['observation_id'];record=cache.records[uid];openings=record['prediction']['openings']
            n=record['tensor_spec']['depth_enclosed']['shape'][0];i=side['candidate_index']
            link=weak['associations'][uid][i] if i is not None else {}
            iid=side.get('interface_id') or link.get('interface_id') or link.get('best_interface_id')
            interface=next((x for x in annotations[uid]['interfaces_by_layout']['layout_raw'] if x['interface_observation_id']==iid),None)
            projection=interface.get('projection') if interface else None
            if projection and projection.get('length_u',0)>0:
                start=int(projection['start_u']*n)%n;length=max(1,min(n,int(round(projection['length_u']*n))))
                interval=[start,(start+length-1)%n]
            else:interval=openings[i]['token_interval_inclusive']
            side.update(candidate_id=openings[i]['candidate_id'] if i is not None else None,
                        interface_id=iid,annotation_projection_available=bool(projection),
                        detection_score=openings[i]['score'] if i is not None else None)
            path=safe_path(Path(args.raw_root),rgb[uid]);source_images[str(path)]=content_hash(path)
            image=np.asarray(Image.open(path).convert('RGB').resize((1024,512)))
            room=dict(token_count=n,legacy_candidates=[x['token_interval_inclusive'] for x in openings])
            title='{} C{} | detector={} | {}'.format(uid,i,round(side['detection_score'],3) if i is not None else 'N/A',
                                                   'GT angle' if projection else 'candidate-centered; GT unavailable')
            for col in (0,1):
                _overlay_panorama(axes[index,col],image,room,{'target_interval':interval},
                                  {i} if i is not None and link.get('status')=='associated' else set(),i,title,bool(col))
        fig.suptitle('{} | house={} | match score={}'.format(case['group'],case['house_id'],
                     '{:.4f}'.format(case['confidence']) if case['confidence'] is not None else 'N/A'))
        fig.tight_layout();name='images/{:03d}.png'.format(number);fig.savefig(out/name,dpi=95);plt.close(fig)
        case['image']=name
        cards.append('<article><h2>{}: {}</h2><p>{} · {} · case {}</p><img src="{}" width="100%"></article>'.format(
                     number,case['group'],case['house_id'],case['role'],case['case_id'],name))
    package=dict(schema_version='opening-human-review.v1',cases=cases,
                 requested_per_group=24,actual_per_group=dict(Counter(c['group'] for c in cases)),
                 deficits={g:24-sum(c['group']==g for c in cases) for g in groups},
                 source_snapshot=cache.snapshot,weak_supervision_sha256=content_hash(args.weak_supervision),
                 matching_report_sha256=content_hash(args.report),pair_checkpoint_sha256=report['pair_checkpoint_sha256'],
                 image_sha256=source_images,manual_conclusions_used_for_training=False)
    package['fingerprint']=fingerprint(package);write_json_atomic(out/'review_manifest.json',package)
    write_json_atomic(out/'review_template.json',dict(package_fingerprint=package['fingerprint'],reviewer='',reviewed_at='',
                      cases=[dict(case_id=c['case_id'],judgement=None,notes='') for c in cases]))
    intro='<h1>开口对应人工核查</h1><p>仅 fit/selection。前3组判断显示的候选对是否属于同一物理开口；第4组判断候选是否唯一、完整覆盖标注开口。无法确定填“不确定”。分层样本不能用于估计总体准确率。</p><p>绿带为标注角域；红框为被核查候选；无标注时标题明确说明。填写 review_template.json 中的“正确／错误／不确定”，保留未完成项为 null。原弱标签不会被覆盖。</p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1400px;margin:auto;font-family:sans-serif}article{border-bottom:2px solid #ddd;padding:16px}</style>'+intro+''.join(cards))
    cache.check_unchanged();print(json.dumps({'cases':len(cases),'counts':package['actual_per_group'],'deficits':package['deficits']}))

if __name__=='__main__':main()
