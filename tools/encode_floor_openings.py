#!/usr/bin/env python3
"""Encode fixed predicted openings; save separate descriptors, never edit the front end."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from portallayout.datasets.zind_floor_dataset import safe_path
from portallayout.models.opening_representation import ABLATIONS, OpeningRepresentationConfig, OpeningRepresentationEncoder
from portallayout.utils.opening_representation_adapter import prepare_opening_representation_batch
from portallayout.utils.room_feature_cache import ObservationFeatureCache, content_hash, fingerprint, write_json_atomic


def model_state_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(data.dtype), list(data.shape)]).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def _write_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('nonfinite representation output')
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(stream, **arrays)
        except BaseException:
            temporary.unlink()
            raise
    os.replace(str(temporary), str(path))


def _read_source_floors(root, floor_uids=None):
    result = []
    for path in sorted((root / 'floors').glob('*.json')):
        record = json.loads(path.read_text())
        if floor_uids and record['floor_uid'] not in floor_uids:
            continue
        if record.get('schema_version') != 'floor-observation-predictions.v1':
            raise ValueError('unsupported source floor prediction schema')
        if record['observation_order'] != [x['observation_id'] for x in record['observations']]:
            raise ValueError('source floor observation order mismatch')
        result.append((path, record))
    if not result or (floor_uids and set(floor_uids) - {r['floor_uid'] for _, r in result}):
        raise ValueError('requested source floors were not found')
    return result


def encode_prediction_cache(source_root, output_root, model, *, batch_size=2, resume=False,
                            diagnostics=False, ablation='full', floor_uids=None,
                            training_status='initialized_untrained'):
    """Read only frozen predictions. Missing source observations remain explicit."""
    if batch_size < 1 or ablation not in ABLATIONS:
        raise ValueError('invalid batch size or ablation')
    if training_status not in ('initialized_untrained', 'trained'):
        raise ValueError('checkpoint must state initialized_untrained or trained')
    source, output = Path(source_root).resolve(), Path(output_root).resolve()
    if source == output or source in output.parents:
        raise ValueError('representation output must be outside the frozen source cache')
    marker = json.loads((source / 'cache_contract.json').read_text())
    source_cache = ObservationFeatureCache(source, marker['contract'], resume=True)
    floors = _read_source_floors(source, floor_uids)
    source_files = {source / 'cache_contract.json'} | {path for path, _ in floors}
    entries = {}
    for _, floor in floors:
        if floor['contract_fingerprint'] != source_cache.fingerprint:
            raise ValueError('source floor refers to a different prediction contract')
        for ref in floor['observations']:
            if ref['status'] != 'predicted':
                continue
            path = safe_path(source, ref['metadata_file'])
            record = json.loads(path.read_text())
            checked = source_cache.get(record['observation_id'], record['image_sha256'])
            if checked != ref:
                raise ValueError('source floor/cache reference mismatch')
            tensor = safe_path(source, record['tensor_file'])
            source_files.update((path, tensor))
            key = (record['observation_id'], record['cache_key'])
            entries[key] = (record, path, tensor)
    source_before = {str(p): content_hash(p) for p in source_files}
    code = ('portallayout/models/opening_representation.py', 'portallayout/utils/opening_representation_adapter.py',
            'tools/encode_floor_openings.py')
    contract = {'schema_version': 'opening-representation-contract.v1',
                'model_config': asdict(model.config), 'model_state_sha256': model_state_fingerprint(model),
                'training_status': training_status, 'source_contract_fingerprint': source_cache.fingerprint,
                'code_sha256': {name: content_hash(ROOT / name) for name in code},
                'ablation': ablation, 'save_diagnostics': bool(diagnostics),
                'device': str(next(model.parameters()).device), 'torch_version': torch.__version__,
                'numpy_version': np.__version__, 'uses_ground_truth': False}
    contract_hash = fingerprint(contract)
    contract_record = {'fingerprint': contract_hash, 'contract': contract}
    contract_path = output / 'encoder_contract.json'
    if contract_path.exists():
        if not resume or json.loads(contract_path.read_text()) != contract_record:
            raise ValueError('existing representation output requires --resume with an identical contract')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('refusing nonempty representation output without a contract')
    else:
        write_json_atomic(contract_path, contract_record)
    model_path = output / 'model.pt'
    if model_path.exists():
        saved = torch.load(str(model_path), map_location='cpu')
        if saved['model_state_sha256'] != contract['model_state_sha256']:
            raise ValueError('saved representation model identity mismatch')
    else:
        torch.save({'format_version': 1, 'task': 'single_view_opening_representation',
            'model_config': asdict(model.config), 'state_dict': model.state_dict(),
            'model_state_sha256': contract['model_state_sha256'], 'training_status': training_status}, str(model_path))
    model.eval()
    references, pending = {}, []
    counts = {'observations_encoded': 0, 'observations_reused': 0, 'openings': 0}

    def flush():
        if not pending:
            return
        records, arrays = [], []
        for key, record, _, _ in pending:
            loaded, tensors = source_cache.load(record['observation_id'], record['image_sha256'])
            records.append(loaded)
            arrays.append(tensors)
        prepared = prepare_opening_representation_batch(records, arrays, device=next(model.parameters()).device)
        with torch.inference_mode():
            result = model(*prepared.encoder_inputs(), return_diagnostics=diagnostics, ablation=ablation)
        for i, (key, source_record, metadata_path, tensor_path) in enumerate(pending):
            count = len(source_record['prediction']['openings'])
            ids = list(result.candidate_ids[i][:count])
            expected_ids = [x['candidate_id'] for x in source_record['prediction']['openings']]
            if ids != expected_ids:
                raise ValueError('representation changed candidate identity/order')
            payload = {'descriptors': result.descriptors[i, :count].float().cpu().numpy(),
                       'baseline_descriptors': result.baseline_descriptors[i, :count].float().cpu().numpy(),
                       'valid_mask': result.valid_mask[i, :count].cpu().numpy(),
                       'projected_feature_2d': result.projected_feature_2d[i].float().cpu().numpy()}
            for name, value in (result.diagnostics or {}).items():
                selected = value[i] if name == 'wall_band_valid' else value[i, :count]
                payload['diag_' + name] = selected.detach().cpu().numpy()
            _write_npz(tensor_path, payload)
            record = {'schema_version': 'opening-representation.v1', 'observation_id': key[0],
                'source_cache_key': key[1], 'source_metadata_file': str(entries[key][1].relative_to(source)),
                'source_metadata_sha256': source_before[str(entries[key][1])],
                'contract_fingerprint': contract_hash, 'training_status': training_status,
                'candidate_ids': ids, 'descriptor_shape': list(payload['descriptors'].shape),
                'tensor_file': str(tensor_path.relative_to(output)), 'tensor_sha256': content_hash(tensor_path)}
            write_json_atomic(metadata_path, record)
            references[key] = {'status': 'encoded', 'observation_id': key[0],
                'metadata_file': str(metadata_path.relative_to(output)), 'tensor_file': record['tensor_file']}
            counts['observations_encoded'] += 1
        pending.clear()

    for key, (record, source_metadata, _) in entries.items():
        entry_key = fingerprint([contract_hash, key, source_before[str(source_metadata)]])
        # IDs never become a filesystem component; source snapshots can coexist.
        metadata_path = output / 'observations' / (entry_key + '.json')
        tensor_path = output / 'observations' / (entry_key + '.npz')
        counts['openings'] += len(record['prediction']['openings'])
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            if (previous.get('contract_fingerprint') != contract_hash or
                    previous.get('source_metadata_sha256') != source_before[str(source_metadata)] or
                    previous.get('candidate_ids') != [x['candidate_id'] for x in record['prediction']['openings']] or
                    previous.get('tensor_file') != str(tensor_path.relative_to(output)) or
                    not tensor_path.is_file() or content_hash(tensor_path) != previous['tensor_sha256']):
                raise ValueError('committed representation entry is missing or corrupt')
            references[key] = {'status': 'encoded', 'observation_id': key[0],
                'metadata_file': str(metadata_path.relative_to(output)), 'tensor_file': previous['tensor_file']}
            counts['observations_reused'] += 1
        else:
            pending.append((key, record, metadata_path, tensor_path))
            if len(pending) == batch_size:
                flush()
    flush()
    floor_refs, errors = [], []
    for _, source_floor in floors:
        rows = []
        for ref in source_floor['observations']:
            if ref['status'] == 'predicted':
                rows.append(references[(ref['observation_id'], ref['cache_key'])])
            else:
                row = {'observation_id': ref['observation_id'], 'status': 'source_unavailable',
                       'reason': ref.get('reason', ref['status'])}
                rows.append(row)
                errors.append(row)
        floor_record = {'schema_version': 'floor-opening-representations.v1',
            'sample_id': source_floor['sample_id'], 'floor_uid': source_floor['floor_uid'],
            'observation_order': source_floor['observation_order'], 'observations': rows,
            'contract_fingerprint': contract_hash, 'training_status': training_status}
        path = output / 'floors' / (fingerprint(source_floor['sample_id']) + '.json')
        write_json_atomic(path, floor_record)
        floor_refs.append({'floor_uid': source_floor['floor_uid'], 'sample_id': source_floor['sample_id'],
                           'representations_file': str(path.relative_to(output))})
    for name, digest in source_before.items():
        if content_hash(name) != digest:
            raise ValueError('frozen source prediction changed during encoding: ' + name)
    report = {'status': 'complete' if not errors else 'partial', 'training_status': training_status,
        'contract_fingerprint': contract_hash, 'floor_samples': len(floors), 'unique_observations': len(entries),
        **counts, 'source_errors': errors, 'source_files_unchanged': True,
        'source_files_verified': len(source_before), 'ground_truth_read': False, 'floors': floor_refs}
    write_json_atomic(output / 'run_report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/research1_opening_representation.json'))
    parser.add_argument('--checkpoint')
    parser.add_argument('--floor-uids', nargs='+')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--torch-threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ablation', choices=ABLATIONS, default='full')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--dataset')
    parser.add_argument('--raw-root')
    parser.add_argument('--max-visualizations', type=int, default=20)
    args = parser.parse_args(argv)
    if args.torch_threads < 1 or args.max_visualizations < 1:
        parser.error('thread and visualization counts must be positive')
    if args.visualize and not (args.dataset and args.raw_root):
        parser.error('--visualize requires --dataset and --raw-root for RGB-only previews')
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    config = OpeningRepresentationConfig(**json.loads(Path(args.config).read_text()))
    model = OpeningRepresentationEncoder(config)
    status = 'initialized_untrained'
    checkpoint = args.checkpoint
    if args.resume and checkpoint is None:
        checkpoint = str(Path(args.output) / 'model.pt')
    if checkpoint:
        payload = torch.load(checkpoint, map_location='cpu')
        if (payload.get('task') != 'single_view_opening_representation' or payload.get('format_version') != 1 or
                payload.get('model_config') != asdict(config)):
            raise ValueError('incompatible representation checkpoint')
        model.load_state_dict(payload['state_dict'], strict=True)
        if model_state_fingerprint(model) != payload['model_state_sha256']:
            raise ValueError('checkpoint state content does not match its fingerprint')
        status = payload['training_status']
    model.to(device)
    report = encode_prediction_cache(args.prediction_cache, args.output, model, batch_size=args.batch_size,
        resume=args.resume, diagnostics=args.diagnostics or args.visualize, ablation=args.ablation,
        floor_uids=args.floor_uids, training_status=status)
    if args.visualize:
        from portallayout.visualization.opening_representation import render_cached_representations
        report['review'] = render_cached_representations(args.prediction_cache, args.output,
            args.dataset, args.raw_root, max_observations=args.max_visualizations)
        write_json_atomic(Path(args.output) / 'run_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
