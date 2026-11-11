#!/usr/bin/env python3
"""Recompute R01 Opening Head training-cache/house overlap from local artifacts.

The saved checkpoint is trusted local PyTorch data. No test split or RGB bytes
are opened. The output does not infer the unrelated layout model's training set.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT/'experiments/portallayout_design10_20261105/R01/cohort.json'
OPENING = ROOT/'checkpoints/Opening_Head/zind_matching_v1/best.pt'
LAYOUT = ROOT/'checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl'
METADATA = ROOT/'checkpoints/Cross_Scene_Matcher/zind_matching_feature_cache/train/metadata.json'
MANIFEST = ROOT/'data/ZInD_matching/train.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream: return json.load(stream)


def compute():
    cohort, metadata = read(COHORT), read(METADATA)
    checkpoint = torch.load(str(OPENING), map_location='cpu')
    layout = torch.load(str(LAYOUT), map_location='cpu')
    contract, views = checkpoint['train_cache_contract'], metadata['views']
    if any(metadata.get(k) != v for k, v in contract.items()):
        raise ValueError('current training metadata differs from checkpoint contract')
    if sha(MANIFEST) != contract['manifest']['sha256']:
        raise ValueError('current matching train manifest differs from checkpoint contract')
    if (checkpoint['task'] != 'ZInD_matching single-view Opening Head (doors+openings)' or
            checkpoint['training_selection']['max_train_views'] is not None or
            metadata['complete'] is not True or metadata['split'] != 'train' or
            metadata['sample_count'] != len(views) or metadata['completed_count'] != len(views)):
        raise ValueError('Opening Head training cache or selection is incomplete')
    if len({v['relative_image_path'] for v in views}) != len(views):
        raise ValueError('training cache image IDs are not unique')
    for view in views:
        if Path(view['relative_image_path']).parts[0] != view['scene_id']:
            raise ValueError('training cache image/house mismatch')
    per_role, manifest_hashes = {}, {}
    cached = {v['relative_image_path'] for v in views}
    base = Path(cohort['dataset_root']).resolve()
    for role in ('fit','selection','calibration','audit'):
        houses = set(cohort['roles'][role]); role_views = [v for v in views if v['scene_id'] in houses]
        role_rows = [r for r in cohort['cohort'] if r['role'] == role]
        if {r['house_id'] for r in role_rows} != houses or len(role_rows) != len(houses):
            raise ValueError('cohort role/house mismatch')
        observed, by_house = set(), {}
        for row in role_rows:
            path = (base/row['input_manifest']).resolve()
            if base not in path.parents: raise ValueError('manifest outside dataset')
            digest = sha(path)
            if digest != cohort['input_manifest_sha256'][row['floor_uid']]:
                raise ValueError('cohort manifest fingerprint mismatch')
            manifest_hashes[row['floor_uid']] = digest
            m = read(path)
            if m['floor_uid'] != row['floor_uid']: raise ValueError('cohort manifest floor mismatch')
            images = {o['rgb']['relative_path'] for o in m['observations']}
            if any(Path(x).parts[0] != row['house_id'] for x in images):
                raise ValueError('cohort image/house mismatch')
            seen = sorted(images & cached); observed.update(images)
            by_house[row['house_id']] = {'cached_train_views_in_house':sum(v['scene_id'] == row['house_id'] for v in role_views),
                'cohort_images':len(images), 'cohort_images_in_train_cache':len(seen)}
        role_cached = {v['relative_image_path'] for v in role_views}
        exposed = sorted(observed & role_cached)
        per_role[role] = {'house_count':len(houses),
            'houses_with_training_cache_views':len({v['scene_id'] for v in role_views}),
            'cached_training_views_from_houses':len(role_views),
            'cohort_images':len(observed), 'cohort_images_in_training_cache':len(exposed),
            'houses':by_house}
        if role == 'audit': per_role[role]['matching_cohort_image_paths'] = exposed
    batch = int(checkpoint['training_args']['batch_size']); epoch = int(checkpoint['completed_epoch'])
    steps_expected = (epoch+1)*math.ceil(len(views)/batch)
    return {'schema_version':'research10.opening-exposure.v1',
        'source_sha256':{'cohort':sha(COHORT), 'opening_checkpoint':sha(OPENING),
            'layout_checkpoint':sha(LAYOUT), 'train_cache_metadata':sha(METADATA),
            'matching_train_manifest':sha(MANIFEST), 'audit_script':sha(Path(__file__).resolve())},
        'cohort_manifest_sha256':manifest_hashes,
        'opening_training':{'completed_epoch_zero_based':epoch,'global_step':checkpoint['global_step'],
            'batch_size':batch,'train_cache_views':len(views),'max_train_views':None,
            'expected_steps_for_21_epochs':steps_expected,
            'steps_consistent_with_full_epoch_loading':steps_expected == checkpoint['global_step'],
            'cache_contract_matches_checkpoint':True,'train_manifest_matches_checkpoint':True,
            'val_cache_views':checkpoint['val_cache_contract']['sample_count'],
            'selection_threshold':checkpoint['operating_threshold'], 'warm_started':checkpoint['warm_start']['loaded']},
        'overlap':per_role,
        'layout_checkpoint':{'metadata_keys':sorted(layout.keys()),'recorded_epoch':layout.get('epoch'),
            'house_exposure_status':'unknown_no_house_list_or_verified_training_provenance'},
        'limitations':['training cache membership demonstrates Opening Head exposure, not physical-label correctness',
            'layout checkpoint house exposure cannot be inferred from Opening Head training exposure',
            'images absent from this cache are not proven unseen by warm start or other pretraining',
            'official test split is not read or used']}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True)
    args=parser.parse_args(); path=Path(args.output)
    if path.exists(): raise SystemExit('refusing to overwrite existing output')
    result=compute(); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:{x:v[x] for x in ('house_count','houses_with_training_cache_views',
        'cached_training_views_from_houses','cohort_images','cohort_images_in_training_cache')}
        for k,v in result['overlap'].items()},ensure_ascii=False,indent=2))
    print('output_sha256='+sha(path))


if __name__ == '__main__': main()
