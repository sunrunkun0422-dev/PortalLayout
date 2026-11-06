#!/usr/bin/env python3
"""Ingest actual reviewer artifacts into a NEW master snapshot; never auto-confirm.

Adjudications are archived. Unresolved/incorrect blind review remains outside core.
Reviewer identity/independence is declared by the submitting humans, not inferred.
"""
import argparse,copy,json,sys,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from portallayout.datasets.zind_observable_pairs import circular_iou
from tools.build_zind_observable_pairs import write,sha,dumps,public_input


def evaluate_correspondence_sets(selected,truth,policy):
    """One-to-one set matching; any annotated true opening may be selected."""
    def compatible(p,t):
        for side in ['A','B']:
            q=p[side];g=t[side]['projection']
            if circular_iou(q,g)<policy['minimum_circular_interval_iou']:return False
            if not g.get('top_v') or not g.get('bottom_v'):return False
            y0,y1=min(g['top_v']),max(g['bottom_v'])
            intersection=max(0,min(q['y1'],y1)-max(q['y0'],y0))
            if intersection/max(min(q['y1']-q['y0'],y1-y0),1e-9)<policy['minimum_vertical_overlap_fraction']:return False
        return True
    edges=[[j for j,t in enumerate(truth) if compatible(p,t)] for p in selected];owner={}
    def assign(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owner or assign(owner[j],seen):owner[j]=i;return True
        return False
    matched=sum(assign(i,set()) for i in range(len(selected)))
    return {'matched':matched,'selected_count':len(selected),'gt_count':len(truth),
            'precision':matched/len(selected) if selected else None,'recall':matched/len(truth) if truth else None,
            'exact_set':bool(truth) and matched==len(truth)==len(selected)}


def validate_answer(answer,stage):
    if not answer.get('id') or not str(answer.get('evidence','')).strip() or not answer.get('timestamp'):raise ValueError('missing reviewer evidence or timestamp')
    timestamp=datetime.fromisoformat(answer['timestamp'].replace('Z','+00:00'))
    if timestamp.tzinfo is None:raise ValueError('review timestamps require timezone')
    if stage==1:
        if answer['decision'] not in ['match','no_match','uncertain']:raise ValueError('invalid blind decision')
        pairs=answer['correspondences']
        if (answer['decision']=='match')!=bool(pairs):raise ValueError('blind decision/regions conflict')
        for pair in pairs:
            for side in ['A','B']:
                x=pair[side]
                if not 0<=x['start_u']<1 or not 0<x['length_u']<=1 or not 0<=x['y0']<x['y1']<=1:raise ValueError('invalid free region')
    else:
        if type(answer.get('annotation_checked')) is not bool:raise ValueError('annotation_checked must be boolean')
        if answer['adjacency'] not in ['positive','negative','unknown']:raise ValueError('invalid adjacency')
        if answer['identifiability'] not in ['clear_evidence','multiple_plausible','insufficient_evidence','unreviewed']:raise ValueError('invalid identifiability')
        for item in answer['visibility_targets']:
            if any(item[s] not in ['clear','partial','not_visible','uncertain'] for s in ['A','B']):raise ValueError('invalid visibility')


def consensus(record,reviews,policy):
    """Pure function; incomplete or conflicting judgments never enter core."""
    result=copy.deepcopy(record);result['auto_labels']=copy.deepcopy(record['labels']);result['reviews']=reviews
    required=policy['minimum_independent_reviewers'];labels=result['labels'];core=False
    if len({r['reviewer_id'] for r in reviews})<required:
        labels['review_state']='review_incomplete' if reviews else 'pending_review';return result,False
    keys=['adjacency','identifiability','annotation_checked','visibility_targets']
    signatures={dumps({k:sorted(r['stage2'][k],key=lambda v:tuple(v['opening_ids'])) if k=='visibility_targets' else r['stage2'][k] for k in keys}) for r in reviews}
    if len(signatures)!=1:
        labels['review_state']='needs_adjudication';return result,False
    agreed=reviews[0]['stage2'];labels['adjacency']=agreed['adjacency'];labels['identifiability']=agreed['identifiability']
    labels['review_state']='verified_consensus'
    labels['visibility']={'source':'independent_human_review','targets':agreed['visibility_targets']}
    if agreed['annotation_checked'] and agreed['identifiability']=='clear_evidence':
        if agreed['adjacency']=='positive' and record['labels']['adjacency']=='positive':
            valid_vis=bool(agreed['visibility_targets']) and all(t[s]=='clear' for t in agreed['visibility_targets'] for s in ['A','B'])
            core=valid_vis and all(r['stage1']['decision']=='match' and r['set_evaluation']['exact_set'] for r in reviews)
        elif agreed['adjacency']=='negative' and record['quality']['negative_certificate'] is not None:
            core=all(r['stage1']['decision']=='no_match' for r in reviews)
    result['core_eligibility']={'eligible':core,'policy':'two independent blind-correct + GT consensus; no automatic admission'}
    return result,core


def apply_adjudication(record,item,reviews,policy):
    """Store an explicit third-party resolution without replacing blind evidence.

    Corrections are usable in diagnostic views, but disputed cases stay outside
    core until a separate independent blind confirmation round is available.
    """
    if len({r['reviewer_id'] for r in reviews})<policy['minimum_independent_reviewers']:
        raise ValueError('adjudication requires two independent review artifacts')
    if not item.get('reviewer_id','').strip() or item['reviewer_id'] in {r['reviewer_id'] for r in reviews}:
        raise ValueError('adjudicator must be distinct from initial reviewers')
    if not str(item.get('rationale','')).strip():raise ValueError('adjudication rationale required')
    resolution=item['resolution'];validate_answer(resolution,2)
    expected=sorted(tuple(v['opening_ids']) for v in record['gt']['correspondences'])
    if sorted(tuple(v['opening_ids']) for v in resolution['visibility_targets'])!=expected:
        raise ValueError('adjudication cannot silently change immutable GT targets')
    result=copy.deepcopy(record);result['adjudication']=copy.deepcopy(item)
    result['labels'].update(adjacency=resolution['adjacency'],identifiability=resolution['identifiability'],
        visibility={'source':'human_adjudication','targets':resolution['visibility_targets']},
        review_state='adjudicated_pending_blind_confirmation')
    result['core_eligibility']={'eligible':False,'policy':'adjudication does not replace independent blind confirmation'}
    return result,False


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--stage1',type=Path,action='append',default=[]);p.add_argument('--stage2',type=Path,action='append',default=[])
    p.add_argument('--adjudication',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=ROOT/'configs/zind_observable_review_v1.json');a=p.parse_args()
    if a.output.exists():p.error('review snapshots are immutable; choose a new directory')
    mapping=json.loads(a.mapping.read_text());dataset_id=sha(a.build/'main_index.jsonl');assert mapping['dataset_id']==dataset_id
    policy=json.loads(a.policy.read_text());first={};second={};archives=[]
    for stage,paths,target in [(1,a.stage1,first),(2,a.stage2,second)]:
        for path in paths:
            d=json.loads(path.read_text());h=sha(path)
            if d.get('synthetic_for_test') or d.get('stage')!=stage or d.get('dataset_id')!=dataset_id or d.get('reviewer_declares_independent_human_review') is not True:raise ValueError('invalid reviewer declaration or data version')
            rid=d.get('reviewer_id','').strip()
            if not rid or rid in target:raise ValueError('missing or duplicate reviewer identity')
            aa={}
            for answer in d['answers']:
                validate_answer(answer,stage)
                if answer['id'] not in mapping['blind_to_master_id'] or answer['id'] in aa:raise ValueError('unknown/duplicate reviewed pair')
                aa[answer['id']]=answer
            target[rid]={'sha256':h,'answers':aa,'stage1_sha256':d.get('stage1_sha256')};archives.append((path,h))
    for rid,s in second.items():
        if rid not in first or s['stage1_sha256']!=first[rid]['sha256']:raise ValueError('stage1 evidence must precede and match stage2 exactly')
        for pid,x in s['answers'].items():
            if pid not in first[rid]['answers']:raise ValueError('stage2 has no blind submission')
            time1=datetime.fromisoformat(first[rid]['answers'][pid]['timestamp'].replace('Z','+00:00'));time2=datetime.fromisoformat(x['timestamp'].replace('Z','+00:00'))
            if time2<time1:raise ValueError('review stages out of order')
    adjudications={}
    if a.adjudication:
        d=json.loads(a.adjudication.read_text());assert d['dataset_id']==dataset_id
        for item in d['adjudications']:
            if item['reviewer_id'] in first or not item.get('rationale') or item['blind_id'] not in mapping['blind_to_master_id']:raise ValueError('invalid independent adjudication')
            adjudications[mapping['blind_to_master_id'][item['blind_id']]]=item
        archives.append((a.adjudication,sha(a.adjudication)))
    a.output.mkdir(parents=True);reverse={v:k for k,v in mapping['blind_to_master_id'].items()};views={k:[] for k in ['all','core_verified','pending_review','adjudicated','human_observability_audit']};counts=Counter();evaluations=[]
    with (a.output/'main_index.jsonl').open('w') as out,(a.output/'rgb_input_index.jsonl').open('w') as safe:
        for l in (a.build/'main_index.jsonl').open():
            r=json.loads(l);bid=reverse.get(r['id']);reviews=[]
            for rid,s in second.items():
                if bid not in s['answers']:continue
                f=first[rid]['answers'][bid];g=s['answers'][bid]
                expected=[v['opening_ids'] for v in r['gt']['correspondences']]
                if sorted(v['opening_ids'] for v in g['visibility_targets'])!=sorted(expected):raise ValueError('review target set differs from immutable GT')
                ev=evaluate_correspondence_sets(f['correspondences'],r['gt']['correspondences'],policy)
                reviews.append({'reviewer_id':rid,'stage1':f,'stage2':g,'stage1_sha256':first[rid]['sha256'],'stage2_sha256':s['sha256'],'set_evaluation':ev})
            rr,core=consensus(r,reviews,policy)
            if r['id'] in adjudications:
                rr,core=apply_adjudication(rr,adjudications[r['id']],reviews,policy);views['adjudicated'].append(r['id'])
            out.write(dumps(rr)+'\n');safe.write(dumps(public_input(rr))+'\n');views['all'].append(r['id'])
            views['core_verified' if core else 'pending_review'].append(r['id']);counts[rr['labels']['review_state']]+=1
            if reviews:views['human_observability_audit'].append(r['id']);evaluations.append({'id':r['id'],'core':core,'reviews':reviews})
    master_sha=sha(a.output/'main_index.jsonl')
    for name,ids in views.items():write(a.output/'views'/f'{name}.json',{'master':'../main_index.jsonl','master_sha256':master_sha,'ids':ids})
    for path,h in archives:(a.output/'raw_reviews').mkdir(exist_ok=True);(a.output/'raw_reviews'/f'{h}.json').write_bytes(path.read_bytes())
    write(a.output/'review_evaluations.json',evaluations);write(a.output/'policy.json',policy)
    write(a.output/'summary.json',{'parent_master_sha256':dataset_id,'counts':dict(counts),'view_counts':{k:len(v) for k,v in views.items()},'raw_review_hashes':[h for _,h in archives],'reviewer_independence':'declared by human reviewers, identities not externally authenticated','test_data_accessed':False})
    print('review snapshot complete',dict(counts))
if __name__=='__main__':main()
