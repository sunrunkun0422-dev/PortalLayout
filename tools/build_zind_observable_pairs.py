#!/usr/bin/env python3
"""Build independent versioned annotations and ID-only views. Train by default."""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter,defaultdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from portallayout.datasets.zind_observable_pairs import load_house,build_floor,stable_id,public_input


def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def gzip_json(p,obj):
    raw=gzip.compress(dumps(obj).encode(),mtime=0);tmp=p.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(p)


def partition(path):
    d=json.loads(path.read_text());sets=[set(d[k]) for k in ['train','val','test']]
    assert all(len(d[k])==len(set(d[k])) for k in ['train','val','test'])
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'house split leakage'
    return d


def pilot_sample(records,size,seed):
    """Stratified two-stage probability sample: houses, then one pair per house.
    Exact pi=(sampled houses / eligible houses)*(1 / pairs in house-stratum).
    Scope is the selected-pair index, not all raw panorama combinations.
    """
    groups=defaultdict(lambda:defaultdict(list))
    for r in records:
        adj=r['labels']['adjacency']
        diff='hard_proxy' if adj=='negative' and r['quality']['negative_difficulty_proxies'] else 'ordinary'
        kind='pass' if r['auto_proposal']['auto_pass'] else 'not_pass'
        key='|'.join([adj,r['relation_scope'],kind if adj=='positive' else diff])
        groups[key][r['house']].append(r)
    quota={k:0 for k in groups}
    for _ in range(size):
        candidates=[k for k in groups if quota[k]<len(groups[k])]
        if not candidates:break
        k=min(candidates,key=lambda x:(quota[x],x));quota[k]+=1
    rows=[];sampling=[]
    for k in sorted(groups):
        houses=groups[k];q=quota[k]
        for hid in sorted(houses,key=lambda h:stable_id(seed,k,h))[:q]:
            rr=sorted(houses[hid],key=lambda r:stable_id(seed,'pair',r['id']))[0]
            pi=q/len(houses)/len(houses[hid])
            rows.append({'id':rr['id'],'stratum':k,'house':hid,'inclusion_probability':pi,'weight':1/pi,
                         'population_house_count':len(houses),'house_stratum_pair_count':len(houses[hid])})
        sampling.append({'stratum':k,'pair_population':sum(map(len,houses.values())),'houses':len(houses),'sampled_houses':q})
    return sorted(rows,key=lambda x:stable_id(seed,'blind_order',x['id'])),sampling


def make_views(records):
    # All views refer to one master index. No copied label records.
    out={k:[] for k in ['all','auto_observable','pending_review','core_verified','hard_or_ambiguous','geometry_positive','geometry_negative','geometry_unknown','cross_complete_room','within_complete_partial','gt_candidate_diagnostic']}
    for r in records:
        pid=r['id'];adj=r['labels']['adjacency'];out['all'].append(pid);out[r['relation_scope']].append(pid);out['geometry_'+adj].append(pid)
        out['pending_review'].append(pid)
        if adj=='positive' and r['auto_proposal']['auto_pass']:out['auto_observable'].append(pid)
        else:out['hard_or_ambiguous'].append(pid)
        if adj=='positive':out['gt_candidate_diagnostic'].append(pid)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT.parent/'zind/data');ap.add_argument('--partition',type=Path,default=ROOT.parent/'zind/zind_partition.json')
    ap.add_argument('--config',type=Path,default=ROOT/'configs/zind_observable_pairs_v1.json')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--split',choices=['train','val','test'],default='train')
    ap.add_argument('--max-houses',type=int,default=0);ap.add_argument('--resume',action='store_true');ap.add_argument('--freeze-record',type=Path)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());parts=partition(args.partition)
    if args.split!='train':
        if not args.freeze_record:ap.error('val/test construction requires a train-only calibration freeze record')
        f=json.loads(args.freeze_record.read_text())
        if f.get('config_sha256')!=sha(args.config) or f.get('calibration_split')!='train' or f.get('status')!='frozen' or not f.get('evidence'):
            ap.error('invalid or mismatched calibration freeze evidence')
        if not set(f.get('calibration_houses',[]))<=set(parts['train']) or not f.get('calibration_houses'):ap.error('invalid calibration house provenance')
    houses=sorted(parts[args.split],key=lambda h:stable_id(cfg['seed'],'build_house',h))
    if args.max_houses:houses=houses[:args.max_houses]
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:ap.error('output exists; use a new directory or --resume with matching configuration')
    out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'shards').mkdir(exist_ok=True)
    identity={'version':cfg['schema_version'],'split':args.split,'houses':houses,'root':str(args.root.resolve()),
              'config_sha256':sha(args.config),'partition_sha256':sha(args.partition),
              'core_source_sha256':sha(ROOT/'portallayout/datasets/zind_observable_pairs.py'),'builder_source_sha256':sha(Path(__file__)),
              'calibration_status':cfg['calibration_status'],'freeze_record':str(args.freeze_record) if args.freeze_record else None}
    if (out/'build_identity.json').exists():assert json.loads((out/'build_identity.json').read_text())==identity,'resume identity changed'
    write(out/'build_identity.json',identity);write(out/'portallayout.config.json',cfg)
    for i,hid in enumerate(houses):
        source=args.root/hid/'zind_data.json';dest=out/'shards'/f'{hid}.json.gz'
        if dest.exists():
            old=json.loads(gzip.decompress(dest.read_bytes()));assert old.get('annotation_sha256')==(sha(source) if source.exists() else None),'annotation changed';continue
        if not source.exists():
            gzip_json(dest,{'house':hid,'annotation_sha256':None,'errors':['missing_annotation'],'counts':{},'connections':[],'records':[],'combinations':[]});continue
        floors,errors,counts=load_house(json.loads(source.read_text()),hid,args.root.resolve(),cfg)
        connections=[];records=[];combinations=[]
        for fid,floor in floors.items():
            for connection,rr,cc in build_floor(hid,fid,floor,args.split,args.root,cfg):
                connections.append(connection);records+=rr;combinations.append({'connection_id':connection['id'],'combinations':cc})
        gzip_json(dest,{'house':hid,'annotation_sha256':sha(source),'errors':errors,'counts':counts,'connections':connections,'records':records,'combinations':combinations})
        print(f'[{args.split}] {i+1}/{len(houses)} {hid}: connections={len(connections)} selected_pairs={len(records)}',flush=True)
    records=[];stats=Counter();scope=defaultdict(Counter);sources={};missing=[];audit_counts=Counter();reselection=Counter()
    with (out/'main_index.jsonl').open('w') as master,(out/'connections.jsonl').open('w') as conn,gzip.GzipFile(filename=str(out/'all_combinations.jsonl.gz'),mode='wb',mtime=0) as combinations:
        for hid in houses:
            d=json.loads(gzip.decompress((out/'shards'/f'{hid}.json.gz').read_bytes()));sources[hid]=d['annotation_sha256'];audit_counts.update(d['counts'])
            if d['errors']:missing.append({'house':hid,'issues':d['errors']})
            for r in d['records']:
                master.write(dumps(r)+'\n')
                records.append({k:r[k] for k in ['id','house','relation_scope','labels','inputs','auto_proposal','quality']})
                stats[r['labels']['adjacency']]+=1;scope[r['relation_scope']][r['labels']['adjacency']]+=1
            byid={r['id']:r for r in d['records']}
            for c in d['connections']:
                conn.write(dumps(c)+'\n');stats['connections']+=1;stats['all_panorama_combinations']+=c['combination_count']
                ref=c['primary_nearest_comparator']
                if c['geometry']['adjacency']=='positive' and ref and c['selected_pair_ids']:
                    primary=byid[c['selected_pair_ids'][0]];reselection['positive_connections_compared']+=1
                    reselection['primary_nearest_proxy_pass']+=int(ref['auto_pass']);reselection['reselected_proxy_pass']+=int(primary['auto_proposal']['auto_pass'])
                    reselection['changed_panorama_pair']+=int(ref['pair_id']!=primary['id'])
            for c in d['combinations']:combinations.write((dumps(c)+'\n').encode())
    views=make_views(records)
    for k,ids in views.items():write(out/'views'/f'{k}.json',{'master':'../main_index.jsonl','master_sha256':sha(out/'main_index.jsonl'),'view':k,'ids':ids,'human_verified':k=='core_verified','gt_assisted':k=='gt_candidate_diagnostic'})
    with (out/'rgb_input_index.jsonl').open('w') as f:
        for r in records:f.write(dumps(public_input(r))+'\n')
    pilot,strata=pilot_sample(records,cfg['pilot_size'],cfg['seed']) if args.split=='train' else ([],[])
    write(out/'pilot_sampling.json',{'split':args.split,'population':'selected pairs in main index, not all camera combinations','design':'stratum=(adjacency,relation scope,proxy pass/difficulty); SRS houses then one SRS pair per house-stratum; stable SHA256 randomization',
                                   'requested':cfg['pilot_size'],'actual':len(pilot),'strata':strata,'samples':pilot,'reviewers_required':2,'human_reviews_completed':0})
    write(out/'source_hashes.json',sources);write(out/'data_issues.json',missing)
    summary={'version':cfg['schema_version'],'split':args.split,'house_count':len(houses),'selected_pair_count':len(records),
             'counts':dict(stats),'relation_scopes':{k:dict(v) for k,v in scope.items()},'input_audit':dict(audit_counts),
             'view_counts':{k:len(v) for k,v in views.items()},'pilot_count':len(pilot),'reselection':dict(reselection),
             'calibration_status':cfg['calibration_status'],'human_verified_count':0,'status':'auto_proposed_pending_review',
             'test_data_accessed':args.split=='test','model_predictions_used_for_selection':False,'model_forward_run':False}
    write(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
