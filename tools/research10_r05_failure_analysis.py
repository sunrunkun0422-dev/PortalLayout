#!/usr/bin/env python3
"""Recompute prespecified threshold diagnostics after a failed workpoint.

Calibration data has been inspected; these rows cannot serve as an untouched
future calibration set after model or loss revisions.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from portallayout.evaluation.opening_matching import summary
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True);p.add_argument('--pairs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); report_path,pairs_path,out=map(Path,(a.report,a.pairs,a.output))
    if out.exists(): raise ValueError('output must be new')
    report=json.loads(report_path.read_text());claimed=report.get('fingerprint')
    if not claimed or fingerprint({k:v for k,v in report.items() if k!='fingerprint'})!=claimed:
        raise ValueError('report fingerprint mismatch')
    if report.get('pair_results_sha256')!=content_hash(pairs_path): raise ValueError('pair artifact hash mismatch')
    protocol=report['house_protocol'];roles={h:role for role in ('selection','calibration') for h in protocol[role]}
    if set(protocol['selection']) & set(protocol['calibration']): raise ValueError('role overlap')
    rows=[]
    with pairs_path.open() as f:
        for line in f:
            r=json.loads(line); role=roles.get(r['house_id'])
            if role is None: raise ValueError('unexpected non-selection/non-calibration house')
            r['role']=role;rows.append(r)
    tables={}
    for role in ('selection','calibration'):
        relevant=[r for r in rows if r['role']==role and r['relationship']=='cross_room']
        tables[role]={str(t):summary(relevant,t) for t in (0.,.5,.7,.9,.95,.99)}
    data={'schema':'research10.r05-failure-analysis.v1','status':'posthoc_diagnostic_not_a_workpoint',
          'thresholds':[0.,.5,.7,.9,.95,.99],'tables':tables,
          'source_sha256':{str(report_path.resolve()):content_hash(report_path),str(pairs_path.resolve()):content_hash(pairs_path)},
          'limitations':['thresholds summarize observed failure; they are not selected acceptance thresholds',
              'labels are weak and unknowns remain separate',
              'calibration was inspected and cannot be considered untouched for subsequent model revisions',
              'Opening Head training cache exposed the cohort houses']}
    write_json_atomic(out,data)
    print(json.dumps({role:{t:{k:v[k] for k in ('accepted_true','accepted_false','accepted_unknown','precision_known',
          'recall_all_weak_positives','negative_pair_false_acceptance')} for t,v in table.items()} for role,table in tables.items()},indent=2))


if __name__=='__main__':main()
