#!/usr/bin/env python3
"""Derive a versioned exploratory training view; never alter ObservablePairs labels.

Internal dev houses are drawn ONLY from official training houses and outside the
human calibration pilot. The trainer's `val` name denotes this internal monitor,
not the unbuilt official ObservablePairs validation set.
"""
import argparse,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from portallayout.datasets.zind_observable_pairs import parse_interfaces,projection,endpoint_error,stable_id
from portallayout.datasets.zind_pair_mining import zind_transform_matrix
from tools.audit_zind_observable_build import digest
from tools.build_zind_observable_pairs import write

def all_candidates(raw,pano_ref,cfg,metric,tokens=256):
    """A single image's complete+raw union; independent of pair GT and scope."""
    candidates=[];source_to_index={};matrix=zind_transform_matrix(raw['floor_plan_transformation'])
    for name in ['layout_complete','layout_raw']:
        items,errors=parse_interfaces(raw[name],matrix,pano_ref+'/'+name)
        errors=[e for e in errors if '/windows' not in e]
        if errors:raise ValueError(('invalid door/opening annotation',pano_ref,errors))
        for it in items:
            if it['kind']=='window':continue
            found=None;xy=np.asarray(it['floor_endpoints']);width=np.linalg.norm(xy[1]-xy[0])
            for j,old in enumerate(candidates):
                distance,_=endpoint_error(xy,old['globalEndpoints'])
                if old['type']==it['kind'] and distance*metric<=cfg['dedup_tolerance_m'] and distance/max(width,1e-6)<=cfg['dedup_relative_tolerance']:
                    found=j;break
            if found is None:
                q=projection(it['local_endpoints'],it['z_bottom_top']);start=int(math.floor(q['start_u']*tokens))%tokens
                end=int(math.floor((q['start_u']+q['length_u'])*tokens))%tokens;found=len(candidates)
                candidates.append({'id':str(found),'type':it['kind'],'tokenInterval':[start,end],
                    'globalEndpoints':it['floor_endpoints'],'localEndpoints':it['local_endpoints'],'projection':q,'sources':[]})
            candidates[found]['sources'].append(it['source']);source_to_index[it['source']]=found
    return candidates,source_to_index

def convert(record,payload,cfg):
    views=[];maps=[]
    for ref in record['provenance']['pano_refs']:
        raw=payload
        for key in ref.split('/'):raw=raw[key]
        candidates,mapping=all_candidates(raw,ref,cfg,record['gt']['metric_scale_floor']);views.append(candidates);maps.append(mapping)
    targets=set()
    for match in record['gt']['all_geometry_matches']:
        indices=[]
        for side,key in enumerate(['opening_a','opening_b']):
            refs=[s['ref'] for s in match[key]['sources'] if s['ref'] in maps[side]]
            if not refs:raise ValueError(('GT candidate missing in selected panorama annotations',record['id'],key))
            indices.append({maps[side][ref] for ref in refs})
        targets.update((ia,ib) for ia in indices[0] for ib in indices[1])
    positive=record['labels']['adjacency']=='positive';assert bool(targets)==positive
    targets=sorted(targets);first=targets[0] if targets else (-1,-1);t=np.array(record['gt']['relative_transform_b_to_a'])
    return {'id':record['id'],'scene_id':record['house'],'floor_id':record['floor'],
        'room_a':record['units'][0],'room_b':record['units'][1],'pano_a':record['panoramas'][0],'pano_b':record['panoramas'][1],
        'image_a':record['inputs']['image_a'],'image_b':record['inputs']['image_b'],'token_count':256,
        'relation_scope':record['relation_scope'],'connection_id':record['connection_id'],
        'candidates_a':views[0],'candidates_b':views[1],
        'supervision':{'is_match':positive,'target_candidate_a':first[0],'target_candidate_b':first[1],
            'correspondence_pairs':[list(x) for x in targets], 'correspondence_source':'immutable_observable_geometry_GT_set',
            'relative_transform_b_to_a':t.tolist(),'relative_yaw_radians':float(math.atan2(t[1,0],t[0,0])),
            'pose_valid':positive,'a_local_to_meters':record['gt']['a_local_to_meters']},
        'observable_labels':record['labels'],'negative_difficulty_proxies':record['quality']['negative_difficulty_proxies'],
        'source_status':'auto_proposed_pending_review','official_source_split':'train'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--dev_houses',type=int,default=100);a=p.parse_args()
    if a.output.exists():p.error('choose a new training-view directory')
    identity=json.loads((a.build/'build_identity.json').read_text());assert identity['split']=='train'
    cfg=json.loads((a.build/'portallayout.config.json').read_text());pilot=json.loads((a.build/'pilot_sampling.json').read_text())
    pilot_houses={s['house'] for s in pilot['samples']};eligible=set(identity['houses'])-pilot_houses
    if not 0<a.dev_houses<len(eligible):p.error('invalid internal dev house count')
    dev=set(sorted(eligible,key=lambda h:stable_id(a.seed,'observable_internal_dev',h))[:a.dev_houses])
    best={};excluded=Counter()
    for line in (a.build/'main_index.jsonl').open():
        r=json.loads(line);adj=r['labels']['adjacency']
        if adj=='unknown':excluded['unknown']+=1;continue
        if adj=='positive' and not r['auto_proposal']['auto_pass']:excluded['positive_proxy_not_pass']+=1;continue
        if adj=='negative' and not r['quality']['negative_certificate']:raise ValueError('uncertified negative')
        # Master order preserves the maximin selection order within a connection.
        if r['connection_id'] in best:excluded['complementary_views_not_in_first_run']+=1;continue
        best[r['connection_id']]=r
    groups=defaultdict(list)
    for r in best.values():groups[('val' if r['house'] in dev else 'train',r['relation_scope'],r['labels']['adjacency'])].append(r)
    chosen=[]
    for split in ['train','val']:
        for scope in ['cross_complete_room','within_complete_partial']:
            positives=groups[split,scope,'positive'];negatives=groups[split,scope,'negative']
            if len(negatives)<len(positives):raise ValueError('insufficient negatives for scope-balanced view')
            chosen+=positives+sorted(negatives,key=lambda r:stable_id(a.seed,'negative',r['id']))[:len(positives)]
    chosen.sort(key=lambda r:(r['house'],r['id']));rows=defaultdict(list);old_house=None;stats=defaultdict(Counter)
    for r in chosen:
        if r['house']!=old_house:
            payload=json.loads((Path(identity['root'])/r['house']/'zind_data.json').read_text());old_house=r['house']
        split='val' if r['house'] in dev else 'train';row=convert(r,payload,cfg);rows[split].append(row)
        stats[split][r['labels']['adjacency']]+=1;stats[split][r['relation_scope']]+=1
        if len(row['supervision']['correspondence_pairs'])>1:stats[split]['multi_gt']+=1
    master_sha=digest(a.build/'main_index.jsonl')
    contract={'portallayout.datasets':'ZInD-ObservablePairs-v1','source_master_sha256':master_sha,'source_build':str(a.build.resolve()),
        'selection':'best passing view per positive connection; best negative view; 1:1 negative sampling within each relation scope',
        'label_status':'auto_proposed_pending_review','evaluation_protocol':'internal_train_house_monitor_NOT_official_val',
        'official_source_split':'train','official_val_test_used':False,'dev_houses_exclude_human_pilot':True,
        'initialization_caveat':'existing pretrained checkpoints may have seen internal dev houses; monitor is not evidence of unseen-house generalization',
        'human_verified':False,'thresholds_calibrated':False,'seed':a.seed,'excluded':dict(excluded)}
    for split in ['train','val']:
        records=sorted(rows[split],key=lambda r:stable_id(a.seed,'training_order',r['id']));houses=sorted({r['scene_id'] for r in records})
        write(a.output/f'{split}.json',{'formatVersion':2,'split':split,'dataRoot':identity['root'],'tokenCount':256,
            'sourceHouseIds':houses,'datasetContract':contract,'pairs':records})
    write(a.output/'summary.json',{**contract,'counts':{k:dict(v) for k,v in stats.items()},
        'internal_dev_houses':sorted(dev),'train_houses':sorted(set(identity['houses'])-dev),
        'manifest_sha256':{s:digest(a.output/f'{s}.json') for s in ['train','val']}})
    print(json.dumps({k:dict(v) for k,v in stats.items()},indent=2))
if __name__=='__main__':main()
