"""Resumable pose-only supervision; targets are attached after predicted features."""
import json
from pathlib import Path
import numpy as np
from portallayout.datasets.floor_auxiliary_supervision import house_protocol, house_role, pose_target
from portallayout.datasets.zind_floor_dataset import read_jsonl
from portallayout.evaluation.floor_metrics import load_floor_ground_truth
from portallayout.models.floor_auxiliary import POSE_FEATURE_VERSION
from portallayout.utils.room_feature_cache import fingerprint, content_hash, write_json_atomic
from portallayout.engine.pose_verifier_policy import POLICY_VERSION, prepare_pose_pair


def build_pose_supervision(cache, dataset_root, checkpoint, output, device='cpu', seed=123, progress=None, audit_only=False,
                           pose_geometry=None, prediction_cache=None):
    from portallayout.engine.floor_layout_pipeline import FloorModelBundle, encode_floor, exploratory_pair_predictions
    from portallayout.geometry.exploratory_floor_pose import select_exploratory_poses
    from portallayout.geometry.floor_pose_graph import PoseGraphConfig
    from portallayout.evaluation.pose_candidate_audit import synthetic_coordinate_audit, audit_floor
    from portallayout.utils.pose_inference_guard import inference_guard
    output=Path(output).resolve();root=Path(dataset_root).resolve()
    if output==root or root in output.parents or output==cache.root or cache.root in output.parents:
        raise ValueError('supervision must be separate from source data')
    output.mkdir(parents=True,exist_ok=True)
    protocol=house_protocol(root,seed)
    contract=dict(candidate_policy=POLICY_VERSION,feature_version=POSE_FEATURE_VERSION,
        pair_checkpoint_sha256=content_hash(checkpoint),source_snapshot=cache.snapshot,house_protocol=protocol)
    if pose_geometry is not None:
        from portallayout.geometry.pose_geometry import geometry_config, overlay_floor, regenerate_pair, FEATURE_VERSION as GEOMETRY_FEATURE_VERSION
        pose_geometry = geometry_config(pose_geometry)
        contract.update(pose_geometry=pose_geometry, feature_version=GEOMETRY_FEATURE_VERSION)
    marker=output/'contract.json'
    if marker.exists() and json.loads(marker.read_text())!=contract:raise ValueError('pose export resume contract mismatch')
    write_json_atomic(marker,contract)
    write_json_atomic(output/'synthetic_coordinate_audit.json',synthetic_coordinate_audit())
    bundle=FloorModelBundle({'pair_checkpoint':str(checkpoint)},device)
    records=[];audits=[];all_hashes={};pair_inventory=[]
    for floor in cache.floors:
        uid=floor['floor_uid'];target=output/'floors'/(uid+'.json');base=root/'labels'/uid
        hashes={str(base/n):content_hash(base/n) for n in ('floor_gt.json','observations_gt.jsonl','portal_links.jsonl')}
        all_hashes.update(hashes)
        if progress:progress(stage='pose_feature_export',floor_uid=uid,completed_floors=len(audits),records=len(records))
        if target.exists() and not audit_only:
            part=json.loads(target.read_text());claimed=part.pop('fingerprint')
            if fingerprint(part)!=claimed or part['annotation_hashes']!=hashes:raise ValueError('floor export fingerprint mismatch')
            if any(content_hash(output/r['feature_file'])!=r['feature_sha256'] for r in part['records']):raise ValueError('resumed feature changed')
            records.extend(part['records']);pair_inventory.extend(part['pair_inventory']);audits.append(part['audit']);continue
        gt=load_floor_ground_truth(root,uid);role=house_role(gt['house_id'],protocol)
        if role=='heldout':raise ValueError('heldout floor cannot enter pose supervision')
        annotations={r['observation_id']:r for r in read_jsonl(base/'observations_gt.jsonl')}
        with inference_guard():
            obs,raw,_=exploratory_pair_predictions(floor,cache,bundle,{'pair_prediction_cache':str(prediction_cache or output/'prediction_pairs')})
            views={}
            if not audit_only: _,views,_=encode_floor(cache,floor['observation_order'],bundle)
            selected=select_exploratory_poses(raw,obs,.7,PoseGraphConfig(mode='none')) if pose_geometry is None else []
        audit=(audit_floor(gt,annotations,list(read_jsonl(base/'portal_links.jsonl')),cache.records,obs,raw,selected)
               if role in ('fit','selection') and pose_geometry is None else dict(floor_uid=uid,house_id=gt['house_id'],status='separate_geometry_audit' if pose_geometry else 'reserved_development_validation'))
        audit['role']=role
        if audit_only:
            audits.append(audit)
            write_json_atomic(output/'audits'/(uid+'.json'),audit)
            print('POSE AUDIT',gt['house_id'],role,flush=True)
            continue
        if pose_geometry is not None:
            with inference_guard():
                obs = overlay_floor(obs, cache, pose_geometry['pose_geometry'])
                lookup = {o['observation_id']:o for o in obs}
                raw = [regenerate_pair(p,lookup[p['observation_a']],lookup[p['observation_b']],pose_geometry) for p in raw]
        lookup={o['observation_id']:o for o in obs};floor_records=[];inventory=[]
        for pair in raw:
            aid,bid=pair['observation_a'],pair['observation_b'];ga,gb=gt['observations'][aid],gt['observations'][bid]
            truth=np.linalg.inv(ga['pose'])@np.asarray(gb['pose']) if ga['pose_valid'] and gb['pose_valid'] else None
            relationship=('same_room' if ga['room_id']==gb['room_id'] else 'cross_room') if ga['membership_confirmed'] and gb['membership_confirmed'] else 'unknown_room'
            predicted=[]
            def capture(row,h,f):predicted.append((h,f))
            with inference_guard():
                prepared=prepare_pose_pair(lookup[aid],lookup[bid],views.get(aid),views.get(bid),bundle,callback=capture,raw=pair)
            inventory.append(dict(pair_id=pair['pair_id'],floor_uid=uid,house_id=gt['house_id'],role=role,
                                  relationship=relationship,status=pair['numerical_status'],hypotheses=len(predicted)))
            for h,f in predicted:
                label,error=pose_target(h['T_B_to_A'],truth)
                record_id=fingerprint(['pose',POLICY_VERSION,h['hypothesis_id']]);relative='features/'+uid+'/'+record_id+'.npz'
                path=output/relative;path.parent.mkdir(parents=True,exist_ok=True)
                # Floor commit is atomic. Interrupted files are regenerated before its marker exists.
                np.savez_compressed(path,raster=f['raster'].astype(np.uint8),visual=f['visual'].astype(np.float16),geometry=f['geometry'].astype(np.float32))
                floor_records.append(dict(record_id=record_id,task='pose',pair_id=pair['pair_id'],hypothesis_id=h['hypothesis_id'],
                    floor_uid=uid,observation_a=aid,observation_b=bid,house_id=gt['house_id'],role=role,relationship=relationship,
                    candidate_ids=h['seed_candidate_ids'],sources=h['sources'],T_B_to_A=h['T_B_to_A'],opening_score=h['match_confidence'],
                    target=label,pose_error_offline=error,independent_evidence=f['independent_evidence'],
                    evidence_semantics='nonfit_evidence_available_not_verified',feature_file=relative,feature_sha256=content_hash(path),
                    source='audited_pose_geometry',source_cache_keys=[cache.records[aid]['cache_key'],cache.records[bid]['cache_key']]))
        part=dict(records=floor_records,pair_inventory=inventory,audit=audit,annotation_hashes=hashes)
        part['fingerprint']=fingerprint(part);write_json_atomic(target,part)
        records.extend(floor_records);pair_inventory.extend(inventory);audits.append(audit)
        print('POSE EXPORT',gt['house_id'],role,len(floor_records),flush=True)
    if any(content_hash(p)!=h for p,h in all_hashes.items()):raise ValueError('annotation changed during export')
    if audit_only:
        result=dict(floors=audits,synthetic=synthetic_coordinate_audit(),gt_oracle_offline_only=True)
        write_json_atomic(output/'candidate_audit.json',result);cache.check_unchanged();return result
    payload=dict(schema_version='floor-auxiliary-supervision.v1',**contract,source_contract=cache.fingerprint,
        source_cache_files_sha256=cache.before,records=records,opening_pairs=[],pair_inventory=pair_inventory,
        annotation_sha256=all_hashes,original_visual_labels_modified=False,original_pose_training_mask_modified=False)
    payload['statistics']={role:{str(label):sum(r['role']==role and r['target']==label for r in records) for label in (-1,0,1)} for role in ('fit','selection','calibration')}
    payload['fingerprint']=fingerprint(payload);write_json_atomic(output/'supervision.json',payload)
    write_json_atomic(output/'candidate_audit.json',dict(floors=audits,synthetic=synthetic_coordinate_audit(),gt_oracle_offline_only=True))
    cache.check_unchanged();return payload
