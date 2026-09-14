#!/usr/bin/env python3
"""Verified incremental migration for the specific bare-pano-ID collision fix.

Pinned before/after source hashes prevent this from becoming an unsafe generic
cache bypass. A full clean build with build_zind_observable_pairs.py remains the
reference reproduction command. Only houses without duplicate bare pano IDs in
every floor may reuse old shards; all other houses are rebuilt from raw data.
"""
import argparse,json,shutil,sys,subprocess
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_zind_observable_pairs import sha,write
BEFORE='3d4febe6d8b4a24dd194e627e480d174d9bb3d8157f2d425732fc9caabb9caf0'
AFTER='96ab8557aa2627f6ab1a664853e9aee71e729e34ebbb80f63b1482cbc0b6cd14'
BUILDER='16e192833335647a7a03925e6fccc5bdbd97244b04e5058d3e86d951c1dafd98'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--partition',type=Path,default=ROOT.parent/'zind/zind_partition.json');a=p.parse_args()
    if a.output.exists():p.error('choose a new immutable build directory')
    old=json.loads((a.parent/'build_identity.json').read_text());hashes=json.loads((a.parent/'source_hashes.json').read_text())
    assert old['split']=='train' and old['core_source_sha256']==BEFORE
    assert sha(ROOT/'dataset/zind_observable_pairs.py')==AFTER
    assert old['builder_source_sha256']==sha(ROOT/'tools/build_zind_observable_pairs.py')==BUILDER
    assert old['config_sha256']==sha(ROOT/'configs/zind_observable_pairs_v1.json')
    assert old['partition_sha256']==sha(a.partition)
    parts=json.loads(a.partition.read_text());assert set(old['houses'])==set(parts['train'])
    reused=[];recomputed=[];evidence={}
    for house in old['houses']:
        path=Path(old['root'])/house/'zind_data.json'
        assert (sha(path) if path.exists() else None)==hashes[house]
        duplicates={}
        if path.exists():
            d=json.loads(path.read_text())
            for floor,rooms in d.get('merger',{}).items():
                c=Counter(pid for prs in rooms.values() for panos in prs.values() for pid in panos)
                if any(n>1 for n in c.values()):duplicates[floor]={k:n for k,n in c.items() if n>1}
        if duplicates:recomputed.append(house);evidence[house]=duplicates
        else:reused.append(house)
    (a.output/'shards').mkdir(parents=True)
    shard_hashes={}
    for house in reused:
        source=a.parent/'shards'/f'{house}.json.gz';shard_hashes[house]=sha(source)
        shutil.copy2(source,a.output/'shards'/source.name)
    new={**old,'core_source_sha256':AFTER};write(a.output/'build_identity.json',new)
    write(a.output/'incremental_provenance.json',{'parent':str(a.parent.resolve()),'parent_master_sha256':sha(a.parent/'main_index.jsonl'),
        'before_core_sha256':BEFORE,'after_core_sha256':AFTER,'builder_sha256':BUILDER,
        'reuse_condition':'all raw pano IDs unique within each floor; patched collision branch is inactive and old/new outputs identical',
        'reused_houses':reused,'recomputed_houses':recomputed,'collision_evidence':evidence,'reused_shard_sha256':shard_hashes,
        'reference_reproduction':'clean full train build, no parent required'})
    print('Reusing unchanged houses:',len(reused),'Recomputing:',recomputed,flush=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/build_zind_observable_pairs.py'),'--root',old['root'],
        '--partition',str(a.partition),'--output',str(a.output),'--split','train','--resume'],check=True)
if __name__=='__main__':main()
