#!/usr/bin/env python3
"""Stream-check a train-only automatic build and create evidence statistics.

No learning, predictions, human labels, or val/test annotations are consumed.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_zind_observable_pairs import public_input,make_views,pilot_sample,write

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--partition',type=Path,default=ROOT.parent/'zind/zind_partition.json');a=p.parse_args()
    b=a.build;identity=json.loads((b/'build_identity.json').read_text());cfg=json.loads((b/'portallayout.config.json').read_text());parts=json.loads(a.partition.read_text())
    assert identity['split']=='train' and set(identity['houses'])<=set(parts['train'])
    assert not set(parts['train'])&(set(parts['val'])|set(parts['test']))
    assert digest(a.partition)==identity['partition_sha256']
    for field,file in [('config_sha256',ROOT/'configs/zind_observable_pairs_v1.json'),('core_source_sha256',ROOT/'portallayout/datasets/zind_observable_pairs.py'),('builder_source_sha256',ROOT/'tools/build_zind_observable_pairs.py')]:
        assert digest(file)==identity[field],('build implementation changed',field)
    source_hashes=json.loads((b/'source_hashes.json').read_text())
    assert set(source_hashes)==set(identity['houses'])
    for house,expected in source_hashes.items():
        path=Path(identity['root'])/house/'zind_data.json';assert (digest(path) if path.exists() else None)==expected
    counts=Counter();conflicts=Counter();reasons=Counter();difficulty=Counter();visibility=Counter();kinds=Counter();selected=defaultdict(list)
    rows=[];ids=set();image_hashes={};examples={};connection_counts=Counter();annotation_issues=Counter()
    with (b/'rgb_input_index.jsonl').open() as safe:
        for line in (b/'main_index.jsonl').open():
            r=json.loads(line);pid=r['id'];assert pid not in ids;ids.add(pid)
            assert r['house'] in parts['train'] and r['split']=='train'
            assert json.loads(next(safe))==public_input(r)
            assert r['provenance']['ground_truth_used_for_inference'] is False
            adj=r['labels']['adjacency'];assert r['labels']['identifiability']=='unreviewed' and r['reviews']==[]
            assert r['labels']['review_state']=='pending_review';counts[adj]+=1
            scope=r['relation_scope'];units=r['units']
            assert (units[0].split('/')[0]!=units[1].split('/')[0])==(scope=='cross_complete_room')
            if scope=='within_complete_partial':assert units[0]!=units[1] and all('/' in u for u in units)
            selected[r['connection_id']].append(pid)
            if adj=='negative':assert r['quality']['negative_certificate'] and not r['gt']['all_geometry_matches']
            if adj=='positive':assert r['gt']['correspondences'] and not r['quality']['geometry_conflicts']
            if r['gt']['metric_scale_floor'] is None:
                assert adj=='unknown' and r['gt']['a_local_to_meters'] is None;counts['pairs_missing_metric']+=1
            for conflict in r['quality']['geometry_conflicts']:conflicts[conflict if isinstance(conflict,str) else conflict['reason']]+=1
            for side in ['a','b']:
                q=r['quality']['image_'+side];assert q['decoded'];relative=r['inputs']['image_'+side]
                assert relative.startswith(r['house']+'/')
                if relative in image_hashes:assert image_hashes[relative]==q['sha256']
                image_hashes[relative]=q['sha256']
            difficulty.update(r['quality']['negative_difficulty_proxies'] if adj=='negative' else [])
            if adj=='negative':counts['negative_hard_proxy' if r['quality']['negative_difficulty_proxies'] else 'negative_ordinary']+=1
            if len(r['gt']['correspondences'])>1:counts['multiple_gt_pairs_'+adj]+=1
            for t in r['gt']['correspondences']:
                for side in ['A','B']:
                    v=t[side];projection=v['projection'];visibility[v['proxy_visibility']]+=1;reasons.update(v['reasons'])
                    assert 0<=projection['start_u']<1 and 0<=projection['length_u']<=.5
                    if projection['wraps_seam']:counts['seam_target_sides']+=1
                for key in ['opening_a','opening_b']:assert all(m[key]['kind'] in ['door','opening'] for m in r['gt']['all_geometry_matches'])
            for m in r['gt']['all_geometry_matches']:kinds[m['opening_a']['kind']]+=1
            key=adj+'|'+scope+'|'+str(r['auto_proposal']['auto_pass'])
            if key not in examples:examples[key]=r
            if any(t[s]['projection']['wraps_seam'] for t in r['gt']['correspondences'] for s in ['A','B']):examples.setdefault('seam',r)
            if len(r['gt']['correspondences'])>1:examples.setdefault('multiple_gt',r)
            rows.append({k:r[k] for k in ['id','house','relation_scope','labels','inputs','auto_proposal','quality']})
        assert safe.readline()==''
    for line in (b/'connections.jsonl').open():
        c=json.loads(line);assert selected[c['id']]==c['selected_pair_ids'];assert len(c['selected_pair_ids'])<=cfg['max_pairs_per_connection']
        connection_counts[c['geometry']['adjacency']]+=1
        connection_counts['raw_combinations']+=c['combination_count']
        for issues in c['annotation_issues']:
            for issue in issues:annotation_issues[issue.split(':',1)[-1]]+=1
    combination_count=0;selection_reasons=Counter()
    with gzip.open(b/'all_combinations.jsonl.gz','rt') as f:
        for line in f:
            c=json.loads(line);combination_count+=len(c['combinations']);keep=[]
            for x in c['combinations']:
                selection_reasons[x['selection_reason']]+=1
                if x['selected']:keep.append(x['id'])
            assert set(keep)==set(selected[c['connection_id']])
    assert combination_count==connection_counts['raw_combinations']
    master_sha=digest(b/'main_index.jsonl')
    for name,expected in make_views(rows).items():
        v=json.loads((b/'views'/f'{name}.json').read_text());assert v['master_sha256']==master_sha and v['ids']==expected
    sample,strata=pilot_sample(rows,cfg['pilot_size'],cfg['seed']);saved=json.loads((b/'pilot_sampling.json').read_text())
    assert sample==saved['samples'] and strata==saved['strata']
    for s in sample:assert s['id'] in ids and abs(s['weight']*s['inclusion_probability']-1)<1e-9
    # Check every distinct RGB asset used in the selected index, not only pilot.
    for relative,expected in image_hashes.items():assert digest(Path(identity['root'])/relative)==expected
    report={'status':'passed','master_sha256':master_sha,'selected_pairs':len(ids),'verified_source_annotations':len(source_hashes),
        'verified_selected_image_hashes':len(image_hashes),'counts':dict(counts),'connection_counts':dict(connection_counts),
        'conflict_reason_occurrences_in_selected_pairs':dict(conflicts),'annotation_issue_occurrences_in_connections':dict(annotation_issues),
        'negative_difficulty_proxy_occurrences':dict(difficulty),'opening_kind_occurrences_in_selected_pairs':dict(kinds),
        'proxy_visibility_target_sides':dict(visibility),'proxy_failure_reason_occurrences':dict(reasons),
        'selection_reason_counts':dict(selection_reasons),'pilot_unique_houses':len({s['house'] for s in sample}),
        'pilot_sampling_reproduced':True,'all_views_and_rgb_whitelist_verified':True,'test_annotations_read':False,
        'human_verified':False,'checks_do_not_establish_pixel_observability':True}
    write(a.output/'audit.json',report);write(a.output/'example_records.json',examples)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
