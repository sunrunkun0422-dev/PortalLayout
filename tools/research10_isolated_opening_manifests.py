#!/usr/bin/env python3
"""Build fresh Opening Head train/selection manifests from fixed fit houses.

Both outputs come from the official-train manifest. They are for a future
isolated frontend experiment; this does not prove the backbone is unexposed.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from portallayout.datasets.floor_auxiliary_supervision import house_protocol
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def select_records(payload, roles, data_root):
    groups = {role: set(roles[role]) for role in ('fit', 'selection', 'calibration', 'audit')}
    if any(a & b for i, a in enumerate(groups.values()) for b in list(groups.values())[i+1:]):
        raise ValueError('house roles overlap')
    output = {}
    for role, split in (('fit', 'train'), ('selection', 'val')):
        rows = [r for r in payload['pairs'] if str(r['scene_id']) in groups[role]]
        if not rows or {str(r['scene_id']) for r in rows} != groups[role]:
            raise ValueError('some selected houses have no matching pairs: '+role)
        seen, views = set(), set()
        for row in rows:
            if row['id'] in seen: raise ValueError('duplicate pair ID')
            seen.add(row['id'])
            for side in ('a', 'b'):
                name = row['image_'+side]; path = Path(name)
                if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != row['scene_id']:
                    raise ValueError('image path or house identity invalid')
                if not (Path(data_root)/path).is_file(): raise ValueError('image missing: '+name)
                views.add(name)
        positives = sum(r['supervision']['is_match'] is True for r in rows)
        negatives = sum(r['supervision']['is_match'] is False for r in rows)
        if positives+negatives != len(rows): raise ValueError('nonbinary source pair label')
        filtered = dict(payload, split=split, dataRoot=str(Path(data_root).resolve()),
                        sourceHouseIds=sorted(groups[role]), missingHouseIds=[], missingImagePairIds=[],
                        pairs=rows, counts={'housesRequested': len(groups[role]), 'housesConverted': len(groups[role]),
                          'pairs': len(rows), 'positivePairs': positives, 'negativePairs': negatives,
                          'missingHouses': 0, 'missingImagePairs': 0})
        output[split] = (filtered, views)
    if output['train'][1] & output['val'][1]: raise ValueError('image leakage between fit and selection')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--dataset', required=True)
    parser.add_argument('--house-protocol', required=True); parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True); args=parser.parse_args()
    source, protocol_path, out = Path(args.source).resolve(), Path(args.house_protocol).resolve(), Path(args.output).resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('output must be new/empty')
    digest = content_hash(source); payload = json.loads(source.read_text())
    if payload.get('split') != 'train' or payload.get('formatVersion') != 2:
        raise ValueError('source must be the official-train ZInD_matching v2 manifest')
    protocol = house_protocol(args.dataset, protocol_path=protocol_path)
    subsets = select_records(payload, protocol, args.data_root)
    if content_hash(source) != digest: raise ValueError('source manifest changed')
    out.mkdir(parents=True, exist_ok=True)
    for split, (manifest, _) in subsets.items(): write_json_atomic(out/(split+'.json'), manifest)
    report = {'schema': 'research10.isolated-opening-manifests.v1', 'status': 'prepared_not_trained',
              'source_sha256': digest, 'protocol_sha256': content_hash(protocol_path),
              'cohort_sha256': content_hash(protocol_path.parent/'cohort.json'),
              'train': {'house_ids': sorted(protocol['fit']), 'pairs': len(subsets['train'][0]['pairs']),
                        'views': len(subsets['train'][1]), 'manifest_sha256': content_hash(out/'train.json')},
              'selection': {'house_ids': sorted(protocol['selection']), 'pairs': len(subsets['val'][0]['pairs']),
                            'views': len(subsets['val'][1]), 'manifest_sha256': content_hash(out/'val.json')},
              'calibration_or_audit_houses_in_training_or_selection': False,
              'official_val_or_test_used': False,
              'limitations': ['manifest split alone cannot establish backbone pretraining independence',
                              'future Opening Head must be initialized without an exposed warm start',
                              'selection output is a subset of official train, not official validation']}
    report['fingerprint'] = fingerprint(report); write_json_atomic(out/'provenance.json', report)
    print(json.dumps({'status': report['status'], 'train_pairs': report['train']['pairs'],
                      'train_views': report['train']['views'], 'selection_pairs': report['selection']['pairs'],
                      'selection_views': report['selection']['views']}))


if __name__ == '__main__': main()
