#!/usr/bin/env python3
"""Validate a real exported human review and report R06 support by split.

Without --review this reports the frozen unreviewed denominator. No auto-labels
are interpreted as human decisions. The optional output path must be new.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def count_review(pack, review=None):
    cases = pack['cases']
    lookup = {c['case_id']: c for c in cases}
    if len(lookup) != len(cases):
        raise ValueError('duplicate pack case')
    decisions = {}
    if review is not None:
        if review.get('schema') != 'r06-human-review.v1' or review.get('pack_id') != pack['pack_id']:
            raise ValueError('review schema or pack identity mismatch')
        if not str(review.get('reviewer', '')).strip() or not review.get('exported_at'):
            raise ValueError('reviewer and export time are required')
        for r in review.get('cases', []):
            uid = r.get('case_id')
            if uid not in lookup or uid in decisions:
                raise ValueError('unknown or duplicate case ID')
            if r.get('shared') not in ('', 'yes', 'no', 'uncertain'):
                raise ValueError('invalid shared decision')
            for key in ('a_to_b', 'b_to_a'):
                d = r.get(key)
                if not isinstance(d, dict) or d.get('status') not in ('', 'available', 'unavailable', 'uncertain'):
                    raise ValueError('invalid direction status')
                if d.get('occlusion') not in ('', 'clear', 'partial', 'blocked', 'uncertain') or d.get('belongs') not in ('', 'yes', 'no', 'uncertain'):
                    raise ValueError('invalid visibility or ownership')
                points = d.get('points')
                if not isinstance(points, list) or any(not isinstance(p, list) or len(p) != 2 or any(not isinstance(x, (int, float)) or not 0 <= x <= 1 for x in p) for p in points):
                    raise ValueError('invalid normalized image points')
                if d['status'] == 'available' and (r['shared'] != 'yes' or d['belongs'] != 'yes' or d['occlusion'] not in ('clear', 'partial') or len(points) < 2):
                    raise ValueError('available direction lacks necessary human labels or points')
            decisions[uid] = r
    report = {'schema': 'r06-review-support.v1', 'pack_id': pack['pack_id'],
              'reviewer': review['reviewer'] if review else None, 'batches': {}}
    for batch in ('initial', 'reserve'):
        group = [c for c in cases if c['batch'] == batch]
        counts, roles = Counter(), {}
        valid_houses = set()
        for c in group:
            d = decisions.get(c['case_id'])
            if not d or not d['shared']:
                counts['shared_pending'] += 1
            else:
                counts['shared_'+d['shared']] += 1
            for direction in ('a_to_b', 'b_to_a'):
                status = d[direction]['status'] if d else ''
                counts['direction_'+(status or 'pending')] += 1
                if status == 'available':
                    valid_houses.add(c['house_id'])
                    roles.setdefault(c['role'], Counter())['available_directions'] += 1
            if d and d['shared'] and d['a_to_b']['status'] and d['b_to_a']['status']:
                counts['both_directions_reviewed'] += 1
        report['batches'][batch] = {'pairs': len(group), 'houses': len({c['house_id'] for c in group}),
                                    'counts': dict(counts), 'available_houses': len(valid_houses),
                                    'available_directions_by_role': {k: dict(v) for k,v in roles.items()}}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--review', type=Path)
    p.add_argument('--output', type=Path)
    a=p.parse_args()
    if a.output and a.output.exists():
        p.error('output exists; use a new path')
    pack=json.loads((a.pack/'cases.json').read_text(encoding='utf-8'))
    review=json.loads(a.review.read_text(encoding='utf-8')) if a.review else None
    report=count_review(pack,review)
    report['qualification'] = 'not_determined_requires_human_review_and_oracle_analysis'
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf-8') as f:f.write(text)
    print(text,end='')


if __name__ == '__main__':
    main()
