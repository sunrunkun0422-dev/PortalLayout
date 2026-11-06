"""Offline room labels on actual predicted pose-graph components.

Inference artifacts are frozen BEFORE GT is read. No GT regrouping or additional
pose edges are permitted. Existing pose samples are copied, original data intact.
"""
import itertools
import json
import shutil
from pathlib import Path
import numpy as np
from portallayout.datasets.floor_auxiliary_supervision import house_role, pose_target
from portallayout.evaluation.floor_metrics import load_floor_ground_truth
from portallayout.geometry.floor_auxiliary_features import auxiliary_features
from portallayout.engine.floor_layout_pipeline import FloorModelBundle, encode_floor
from portallayout.engine.floor_training import AuxiliaryFeatureDataset, _state_hash
from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic


def build_post_pgo_room_supervision(cache, dataset_root, supervision, predictions, bundle_spec, output, device='cpu'):
    source = AuxiliaryFeatureDataset(supervision)
    bundle = FloorModelBundle(bundle_spec, device)
    if source.data['pair_checkpoint_sha256'] != content_hash(bundle.spec['pair_checkpoint']):
        raise ValueError('room export pair weights differ from source supervision')
    output = Path(output).resolve()
    protected = [cache.root, Path(dataset_root).resolve(), source.path.parent, Path(predictions).resolve()]
    if any(output == p or p in output.parents for p in protected) or (output.exists() and any(output.iterdir())):
        raise ValueError('room supervision requires a new independent output directory')
    output.mkdir(parents=True, exist_ok=True); (output/'features').mkdir()
    records = []
    for row in source.data['records']:
        if row['task'] != 'pose': continue
        source.load(row)  # hash and safe-path check
        copied = dict(row); copied['feature_file'] = 'features/'+row['record_id']+'.npz'
        shutil.copyfile(source.path.parent/row['feature_file'], output/copied['feature_file'])
        records.append(copied)
    index = json.loads((Path(predictions)/'run_report.json').read_text())
    floor_lookup = {f['floor_uid']:f for f in cache.floors}
    if len({r['floor_uid'] for r in index['floors']}) != len(index['floors']): raise ValueError('duplicate prediction floor')
    artifacts, annotations, failures = {}, {}, []
    for ref in index['floors']:
        path = Path(ref['result_file']); pred = json.loads(path.read_text()); artifacts[str(path.resolve())] = content_hash(path)
        floor = floor_lookup[pred['floor_uid']]
        if set(pred['observation_order']) != set(floor['observation_order']): raise ValueError('room export floor inputs differ')
        if pred['status'] == 'runtime_failure': failures.append(pred['floor_uid']); continue
        provenance = pred['provenance']
        if provenance['weights_hash'] != bundle.weights_hash or provenance['source_snapshot'] != cache.snapshot:
            raise ValueError('room export prediction/model/cache fingerprint mismatch')
        # A predicted graph is required, not a manually assembled or GT graph.
        if not provenance['calibration_status']['opening'] or not provenance['calibration_status']['pose']:
            failures.append(pred['floor_uid']); continue
        observations, views, _ = encode_floor(cache,floor['observation_order'],bundle)
        lookup = {r['observation_id']:r for r in observations}
        gt = load_floor_ground_truth(dataset_root,pred['floor_uid'])
        for p in gt['source_files']: annotations[p] = content_hash(p)
        role = house_role(gt['house_id'],source.data['house_protocol'])
        if role not in ('fit', 'selection', 'calibration'):
            raise ValueError('audit/heldout house in room supervision')
        for component in pred['components']:
            for aid,bid in itertools.combinations(sorted(component['observation_ids']),2):
                if aid not in views or bid not in views: continue
                transform = np.linalg.inv(component['poses'][aid]) @ np.array(component['poses'][bid])
                a,b = gt['observations'][aid],gt['observations'][bid]
                truth = np.linalg.inv(a['pose']) @ np.array(b['pose']) if a['pose_valid'] and b['pose_valid'] else None
                valid,error = pose_target(transform,truth)
                known = valid == 1 and a['membership_confirmed'] and b['membership_confirmed']
                target = int(a['room_id']==b['room_id']) if known else -1
                feature = auxiliary_features(lookup[aid],lookup[bid],views[aid],views[bid],transform,task='room')
                if not feature['geometry'][:,-1].all(): target = -1
                uid = fingerprint(['room_pgo',pred['floor_uid'],aid,bid,transform.tolist()])
                file = output/'features'/(uid+'.npz')
                np.savez_compressed(file,raster=feature['raster'].astype(np.uint8),visual=feature['visual'].astype(np.float16),geometry=feature['geometry'])
                records.append(dict(record_id=uid,task='room',pair_id=fingerprint([aid,bid]),floor_uid=pred['floor_uid'],
                    observation_a=aid,observation_b=bid,house_id=gt['house_id'],role=role,target=target,
                    T_B_to_A=transform.tolist(),pose_error_offline=error,membership_valid=known,
                    feature_file=str(file.relative_to(output)),feature_sha256=content_hash(file),
                    source='predicted_pgo_confirmed_complete_room',component_id=component['component_id']))
    payload = {k:v for k,v in source.data.items() if k not in ('records','statistics','fingerprint')}
    payload.update(records=records,room_feature_source='predicted_pgo_components',
                   room_pose_model_hash=_state_hash(bundle.pose_model),room_source_bundle_hash=bundle.weights_hash,
                   prediction_sha256=artifacts,room_annotation_sha256=annotations,unusable_floors=failures)
    payload['statistics'] = {role:{str(t):sum(r['task']=='room' and r['role']==role and r['target']==t for r in records)
                                   for t in (-1,0,1)} for role in ('fit','selection','calibration')}
    payload['room_export_status'] = ('labels_exported' if any(r['task']=='room' and r['target']>=0 for r in records)
                                     else 'no_usable_localized_pairs')
    payload['fingerprint'] = fingerprint(payload)
    if any(content_hash(p)!=h for p,h in {**artifacts,**annotations}.items()): raise ValueError('room sources changed during export')
    cache.check_unchanged();write_json_atomic(output/'supervision.json',payload)
    return payload
