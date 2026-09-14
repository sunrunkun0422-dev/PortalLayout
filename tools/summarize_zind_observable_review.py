#!/usr/bin/env python3
"""Weighted diagnostic summaries; incomplete human review yields no population estimate."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_zind_observable_pairs import write
from tools.audit_zind_observable_build import digest

def summarize(samples,records):
    weights=Counter();respondents=0;unresolved=[]
    for s in samples:
        r=records.get(s['id']);labels=r['labels'] if r else {}
        if labels.get('review_state') not in ['verified_consensus','adjudicated_pending_blind_confirmation']:
            unresolved.append(s['id']);continue
        respondents+=1;adj=labels['adjacency'];ident=labels['identifiability'];w=s['weight']
        targets=labels.get('visibility',{}).get('targets',[])
        visible=bool(targets) and all(t[side]=='clear' for t in targets for side in ['A','B'])
        weights['adjacency:'+adj]+=w;weights['identifiability:'+ident]+=w
        if adj=='positive':weights['positive_bilateral_clear' if visible else 'positive_not_confirmed_bilateral_clear']+=w
    complete=bool(samples) and not unresolved;den=sum(weights['adjacency:'+k] for k in ['positive','negative','unknown'])
    return {'requested_sample_count':len(samples),'resolved_sample_count':respondents,'unresolved_ids':unresolved,
        'population_estimate_available':complete,'reason':None if complete else 'missing independent review or unresolved disagreement; no missing-at-random assumption',
        'weighted_respondent_diagnostics':dict(weights),'horvitz_thompson_totals':dict(weights) if complete else None,
        'hajek_adjacency_proportions':{k:weights['adjacency:'+k]/den for k in ['positive','negative','unknown']} if complete else None,
        'population':'selected panorama-pair index; not raw combinations and not all original ZInD connections',
        'uncertainty':'No confidence interval claimed. House/stratum clustering must be respected in later variance analysis.',
        'interpretation':'human diagnostic evidence, not an information-theoretic upper bound'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--reviewed',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sampling=json.loads((a.build/'pilot_sampling.json').read_text());ids={s['id'] for s in sampling['samples']};records={}
    if a.reviewed:
        provenance=json.loads((a.reviewed/'summary.json').read_text())
        if provenance.get('parent_master_sha256')!=digest(a.build/'main_index.jsonl'):
            raise ValueError('reviewed snapshot does not descend from this sampling population')
        for l in (a.reviewed/'main_index.jsonl').open():
            r=json.loads(l)
            if r['id'] in ids:records[r['id']]=r
    report=summarize(sampling['samples'],records);write(a.output,report)
    print('Resolved',report['resolved_sample_count'],'/',report['requested_sample_count'],'Population estimate available:',report['population_estimate_available'])
if __name__=='__main__':main()
