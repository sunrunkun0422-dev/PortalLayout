#!/usr/bin/env python3
"""Plan a deterministic house-isolated cohort; optional frozen frontend caching.

Defaults write RGB manifests only. Selection uses house identity and panorama
count, never GT adjacency or geometry. No neural parameters are trained here.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from portallayout.datasets.floor_auxiliary_supervision import house_protocol
from portallayout.datasets.zind_floor_dataset import ZInDFloorDataset
from portallayout.utils.room_feature_cache import fingerprint,write_json_atomic


def select_cohort(samples, protocol, counts, seed=123, minimum_views=2, maximum_views=80):
    chosen={}
    for role in ('fit','selection','calibration'):
        eligible={}
        for index,row in enumerate(samples):
            if row['house_id'] in protocol[role] and minimum_views<=row['observation_count']<=maximum_views:
                eligible.setdefault(row['house_id'],[]).append((index,row))
        houses=sorted(eligible,key=lambda h:(fingerprint([seed,h]),h))[:counts[role]]
        if len(houses)<counts[role]:raise ValueError('not enough eligible houses for '+role)
        chosen[role]=[min(eligible[h],key=lambda v:(fingerprint([seed,v[1]['floor_uid']]),v[1]['floor_uid'])) for h in houses]
    return chosen


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fit-houses',type=int,default=40);p.add_argument('--selection-houses',type=int,default=10)
    p.add_argument('--calibration-houses',type=int,default=10);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--minimum-views',type=int,default=2);p.add_argument('--maximum-views',type=int,default=80)
    p.add_argument('--encode',action='store_true');p.add_argument('--raw-root');p.add_argument('--cache-output')
    p.add_argument('--frontend-config',default=str(ROOT/'configs/research1_floor_frontend.json'))
    p.add_argument('--device',default='cuda');p.add_argument('--resume-cache',action='store_true')
    args=p.parse_args(argv)
    if args.minimum_views<1 or args.maximum_views<args.minimum_views:p.error('invalid view limits')
    counts={role:getattr(args,role+'_houses') for role in ('fit','selection','calibration')}
    if min(counts.values())<1:p.error('house counts must be positive')
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):p.error('output must be new/empty')
    if args.encode and (not args.raw_root or not args.cache_output):p.error('--encode needs --raw-root and --cache-output')
    dataset=ZInDFloorDataset(args.dataset,official_split='train');protocol=house_protocol(args.dataset,args.seed)
    selected=select_cohort(dataset.samples,protocol,counts,args.seed,args.minimum_views,args.maximum_views)
    manifests=[];rows=[]
    output.mkdir(parents=True,exist_ok=True)
    for role,items in selected.items():
        for index,row in items:
            manifest=dataset[index];manifests.append(manifest)
            path=output/'manifests'/(row['floor_uid']+'.json');write_json_atomic(path,manifest)
            rows.append(dict(role=role,house_id=row['house_id'],floor_uid=row['floor_uid'],
                             observations=row['observation_count'],manifest=str(path)))
    report=dict(status='planned',house_protocol=protocol,seed=args.seed,cohort=rows,
                purpose='expanded_training_development_cohort_not_heldout_test',
                counts=counts,observations=sum(r['observations'] for r in rows))
    write_json_atomic(output/'cohort.json',report)
    if args.encode:
        import torch
        from portallayout.engine.floor_observation_pipeline import ObservationPredictionSettings,frontend_contract,BiLayoutObservationEncoder,predict_floor_observations
        from portallayout.utils.room_feature_cache import ObservationFeatureCache
        torch.set_num_threads(2);torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        cfg=json.loads(Path(args.frontend_config).read_text());settings=ObservationPredictionSettings(**cfg['prediction'])
        paths=[ROOT/cfg[k] for k in ('bi_layout_config','bi_layout_checkpoint','opening_checkpoint')]
        contract=frontend_contract(*paths,settings,execution_device=args.device)
        cache=ObservationFeatureCache(args.cache_output,contract,resume=args.resume_cache)
        encoder=BiLayoutObservationEncoder.from_checkpoints(*paths,settings,args.device)
        result=predict_floor_observations(manifests,args.raw_root,encoder,cache,settings,batch_size=2)
        write_json_atomic(output/'cache_report.json',result)
    print(json.dumps({k:v for k,v in report.items() if k not in ('house_protocol','cohort')},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
