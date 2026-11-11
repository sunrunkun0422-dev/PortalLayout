#!/usr/bin/env python3
"""Reproducible frozen ObservablePairs prior probe; no production model edits."""
import argparse, copy, csv, hashlib, json, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from portallayout.models.cross_scene_matcher import (OpeningGuidedCrossAttentionMatcher, candidate_intervals_to_mask,
    opening_probabilities_to_intervals, resolve_enclosed_extended_depth, opening_detection_loss)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def stable(*parts):return hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def mask(interval):return candidate_intervals_to_mask([interval],256)[0].numpy()
def labels(candidates):
    binary=np.zeros(256,np.float32); covered=np.zeros((3,256),bool)
    for c in candidates:
        m=mask(c['tokenInterval']);binary[m]=1
        covered[{'door':0,'opening':1}.get(c.get('type'),2),m]=True
    y=np.full(256,2,np.int64);y[covered[0]]=0;y[covered[1]]=1
    y[(covered.sum(0)>1)|covered[2]]=-100
    return binary,y

def prepare(a):
    if (a.output/'protocol.json').exists():raise RuntimeError('prepare requires a new directory')
    summaries={};selected={};views={};hashes={}
    for split in ('train','val'):
        path=a.manifest/(split+'.json');hashes[split]=digest(path)
        payload=json.loads(path.read_text());rows=payload['pairs'];unique={}
        for r in rows:
            for side in ('a','b'):
                k=r['image_'+side];c=r['candidates_'+side]
                if k in unique:assert unique[k]['candidates']==c,('inconsistent labels',k)
                else:unique[k]=dict(image=k,house=r['scene_id'],candidates=c)
        stat=dict(pairs=len(rows),houses=len({r['scene_id'] for r in rows}),panoramas=len(unique))
        for typ in ('door','opening','unknown'):
            panos=[v for v in unique.values() if any(c.get('type','unknown')==typ for c in v['candidates'])]
            stat[typ]=dict(houses=len({v['house'] for v in panos}),panoramas=len(panos),
                view_instances=sum(sum(c.get('type','unknown')==typ for c in v['candidates']) for v in panos),
                pairs=sum(any(c.get('type','unknown')==typ for s in ('a','b') for c in r['candidates_'+s]) for r in rows))
        stat['conflict_tokens']=sum(int((labels(v['candidates'])[1]==-100).sum()) for v in unique.values())
        stat['total_tokens']=len(unique)*256;summaries[split]=stat
        houses=sorted({r['scene_id'] for r in rows},key=lambda h:stable(123,'probe_houses',h))
        groups={'fit':houses[:a.fit_houses],'calibration':houses[a.fit_houses:a.fit_houses+a.cal_houses]} if split=='train' else {'monitor':houses[:a.monitor_houses]}
        byhouse=defaultdict(list)
        for r in rows:byhouse[r['scene_id']].append(r)
        for group,chosen_houses in groups.items():
            chosen=[]
            for hi,h in enumerate(chosen_houses):
                for positive in (True,False):
                    cs=[r for r in byhouse[h] if r['supervision']['is_match']==positive]
                    scope=('cross_complete_room','within_complete_partial')[hi%2]
                    pool=[r for r in cs if r['relation_scope']==scope] or cs
                    if pool:chosen.append(min(pool,key=lambda r:stable(123,'probe_pair',r['id'])))
            selected[group]=chosen
            for r in chosen:
                for side in ('a','b'):
                    k=r['image_'+side]
                    if k in views:assert views[k]['group']==group
                    views[k]=dict(unique[k],group=group,data_root=payload['dataRoot'],cache_key=stable(k))
        del rows,payload,unique
    write(a.output/'full_annotation_counts.json',summaries)
    write(a.output/'selection.json',dict(pairs=selected,views=views))
    protocol=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=digest(a.checkpoint),manifest_sha256=hashes,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        initial_git_status=subprocess.check_output(['git','status','--short'],text=True),argv=sys.argv,
        seeds=[123,456,789],epochs=5,batch_size=8,lr=.01,weight_decay=.0001,hidden_dim=16,
        lambda_range=[0,8],lambda0=4.,auxiliary_type_weight=.2,
        background='Outside complete+raw union, annotation-relative background; not proof of visual absence',
        split_caveat='New adapter house isolation only; backbone has seen fit/calibration; monitor previously used in model selection',
        calibration='Epoch minimizes binary opening loss; threshold maximizes recall within A0 calibration FP budget',
        device=a.device,torch_version=torch.__version__,sample_counts={g:dict(pairs=len(rs),houses=len({r['scene_id'] for r in rs}),
            panoramas=sum(v['group']==g for v in views.values())) for g,rs in selected.items()})
    write(a.output/'protocol.json',protocol);print(json.dumps(protocol['sample_counts']),flush=True)

def load_matcher(checkpoint):
    p=torch.load(str(checkpoint),map_location='cpu')
    config=p['initialization']['matcher']['model']
    m=OpeningGuidedCrossAttentionMatcher(**{k:v for k,v in config.items() if k!='has_dustbin'})
    opening=p['initialization']['opening_head']['model']
    assert opening['prior_strength']==4. and opening['prior_relative_scale']==.1
    assert opening['kernel_size']==5
    m.load_state_dict(p['matcher_state_dict'],strict=True);m.eval().requires_grad_(False)
    return p,m

def cache(a):
    from tools.debug_cross_scene_flow import load_bi_layout
    sel=json.loads((a.output/'selection.json').read_text());protocol=json.loads((a.output/'protocol.json').read_text())
    assert digest(a.checkpoint)==protocol['checkpoint_sha256']
    p,matcher=load_matcher(a.checkpoint)
    net,_=load_bi_layout(str(ROOT/'configs/layout/zind_all.yaml'),str(a.checkpoint),torch.device(a.device),False)
    net.load_state_dict(p['bi_layout_state_dict'],strict=True);net.eval().requires_grad_(False);matcher.to(a.device)
    states={n:b.detach().cpu().clone() for n,b in net.named_buffers()}
    directory=a.output/'predictions';directory.mkdir(exist_ok=True);start=time.time();total=0
    for i,(k,v) in enumerate(sorted(sel['views'].items())):
        target=directory/(v['cache_key']+'.pt')
        if target.exists():continue
        path=Path(v['data_root'])/k
        with Image.open(path) as im:arr=np.asarray(im.convert('RGB').resize((1024,512),Image.BICUBIC),np.float32)/255
        x=torch.from_numpy(arr.transpose(2,0,1).copy())[None].to(a.device)
        with torch.no_grad():
            out=net(x,return_features=True);enc,ext=resolve_enclosed_extended_depth(out,'extended_first')
            f=out['layout_feature'];s=matcher.opening_head(f,enc,ext);rel=s['relative_delta']
            h=(rel/rel.amax(1,keepdim=True).clamp_min(1e-6))*(rel/(rel+.1))
            fg=torch.cat([matcher.opening_head.feature_norm(f),torch.stack([s[t] for t in ('delta','relative_delta','log_ratio')],-1)],-1)
            pred=dict(feature=f[0].cpu(),enclosed=enc[0].cpu(),extended=ext[0].cpu(),x=fg[0].cpu(),h=h[0].cpu(),
                logits=s['opening_logits'][0].cpu(),delta=s['delta'][0].cpu(),relative_delta=rel[0].cpu(),
                log_ratio=s['log_ratio'][0].cpu(),expansion=s['expansion_depth'][0].cpu(),rgb_sha256=digest(path),
                checkpoint_sha256=protocol['checkpoint_sha256'])
        assert not any(t.requires_grad for t in pred.values() if torch.is_tensor(t))
        torch.save(pred,target);total+=1
        print('cache {}/{} {:.1f}s {}'.format(i+1,len(sel['views']),time.time()-start,k),flush=True)
    assert all(torch.equal(states[n],b.cpu()) for n,b in net.named_buffers())
    assert all(p.grad is None for p in net.parameters())
    write(a.output/'cache_run.json',dict(seconds=time.time()-start,new_views=total,device=a.device,
        frozen_no_grad=True,buffers_unchanged=True,peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None))

class PriorAdapter(nn.Module):
    def __init__(self,group,dim=515):
        super().__init__();self.group=group
        if group=='A1':self.strength=nn.Parameter(torch.zeros(()))
        elif group in ('A2','A3'):
            self.branch=nn.Sequential(nn.Linear(dim,16),nn.GELU());self.type_head=nn.Linear(16,3)
            if group=='A2':
                self.continuous=nn.Linear(16,1);nn.init.zeros_(self.continuous.weight);nn.init.zeros_(self.continuous.bias)
            else:self.strength=nn.Parameter(torch.zeros(2))
    def forward(self,x,h,old_logits):
        # Prediction-only API: ground truth is never an argument.
        q=None
        if self.group=='A0':strength=torch.full_like(h,4.)
        elif self.group=='A1':strength=(8*self.strength.sigmoid()).expand_as(h)
        else:
            hidden=self.branch(x);q=self.type_head(hidden)
            if self.group=='A2':strength=8*self.continuous(hidden).squeeze(-1).sigmoid()
            else:
                prob=q.softmax(-1);value=8*self.strength.sigmoid()
                strength=prob[...,0]*value[0]+prob[...,1]*value[1]+prob[...,2]*4
        return old_logits+(strength-4)*(h-.5),q,strength

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','cache','experiment'],required=True)
    p.add_argument('--output',type=Path,default=ROOT/'experiments/type_prior_probe_20260923')
    p.add_argument('--manifest',type=Path,default=ROOT/'data/ZInD-ObservablePairs-v1/training/best1_balanced_seed123')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/End_To_End/zind_observable_v1/best1_balanced_seed123_20260910/best.pt')
    p.add_argument('--fit-houses',type=int,default=16);p.add_argument('--cal-houses',type=int,default=8)
    p.add_argument('--monitor-houses',type=int,default=12);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='experiment':
        from tools.type_prior_probe_evaluation import experiment
        experiment(a)
    else:{'prepare':prepare,'cache':cache}[a.stage](a)
if __name__=='__main__':main()
