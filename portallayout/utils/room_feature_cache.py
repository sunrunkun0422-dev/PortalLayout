"""Versioned single-observation prediction cache, independent of GT stores."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np


def content_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        except BaseException:
            temporary.unlink()
            raise
    os.replace(str(temporary), str(path))


class ObservationFeatureCache:
    """A JSON commit marker references one atomically written NPZ payload.

    Cache identity covers exact code/model/preprocessing/proposal settings and
    image bytes. No annotation, source membership or GT pose is accepted here.
    """
    ARRAY_FIELDS = {'feature_1d', 'feature_2d', 'depth_enclosed', 'depth_extended',
                    'opening_logits', 'opening_probability', 'ratio'}

    def __init__(self, root, contract, resume=False):
        self.root = Path(root).resolve()
        self.contract = contract
        self.fingerprint = fingerprint(contract)
        marker = self.root / 'cache_contract.json'
        expected = {'schema_version': 'floor-observation-cache.v1',
                    'fingerprint': self.fingerprint, 'contract': contract}
        if marker.exists():
            if not resume:
                raise ValueError('cache already exists; use --resume with the identical contract')
            if json.loads(marker.read_text()) != expected:
                raise ValueError('cache contract mismatch; use a new output directory')
        elif self.root.exists() and any(self.root.iterdir()):
            raise ValueError('refusing a nonempty directory without a cache contract')
        else:
            write_json_atomic(marker, expected)

    def _paths(self, observation_id, image_sha256):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', observation_id):
            raise ValueError('unsafe observation ID')
        if not re.fullmatch(r'[0-9a-f]{64}', image_sha256):
            raise ValueError('invalid image SHA-256')
        key = fingerprint([self.fingerprint, observation_id, image_sha256])
        base = self.root / 'observations' / observation_id / key
        return key, base.with_suffix('.json'), base.with_suffix('.npz')

    def get(self, observation_id, image_sha256):
        key, metadata_path, tensor_path = self._paths(observation_id, image_sha256)
        if not metadata_path.exists():
            return None  # No committed entry; an interrupted NPZ can be replaced.
        record = json.loads(metadata_path.read_text())
        if (record.get('schema_version') != 'floor-observation-prediction.v1' or
                record.get('cache_key') != key or
                record.get('contract_fingerprint') != self.fingerprint or
                record.get('image_sha256') != image_sha256 or
                record.get('observation_id') != observation_id):
            raise ValueError('invalid cache metadata: ' + str(metadata_path))
        if (not tensor_path.is_file() or content_hash(tensor_path) != record['tensor_sha256'] or
                record.get('tensor_file') != tensor_path.relative_to(self.root).as_posix()):
            raise ValueError('cache tensor missing or corrupt: ' + str(tensor_path))
        return {'status': 'predicted', 'observation_id': observation_id,
                'cache_key': key, 'metadata_file': metadata_path.relative_to(self.root).as_posix(),
                'tensor_file': record['tensor_file']}

    def put(self, observation_id, image_sha256, arrays, prediction):
        if set(arrays) != self.ARRAY_FIELDS:
            raise ValueError('cache requires exactly the prediction tensor fields')
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        for name, array in arrays.items():
            if array.dtype.kind != 'f' or not np.isfinite(array).all():
                raise ValueError('nonfinite or nonfloating prediction: ' + name)
        n = arrays['depth_enclosed'].size
        if (arrays['feature_1d'].ndim != 2 or arrays['feature_1d'].shape[0] != n or
                arrays['feature_2d'].ndim != 3 or min(arrays['feature_2d'].shape) < 1 or
                arrays['ratio'].shape != (1,) or n < 3):
            raise ValueError('invalid feature/depth/ratio shapes')
        for name in ('depth_enclosed', 'depth_extended', 'opening_logits', 'opening_probability'):
            if arrays[name].shape != (n,):
                raise ValueError('token shape mismatch: ' + name)
        if not ((arrays['opening_probability'] >= 0) & (arrays['opening_probability'] <= 1)).all():
            raise ValueError('opening probabilities outside [0,1]')
        key, metadata_path, tensor_path = self._paths(observation_id, image_sha256)
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tensor_path.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez(stream, **arrays)
            except BaseException:
                temporary.unlink()
                raise
        os.replace(str(temporary), str(tensor_path))
        write_json_atomic(metadata_path, {
            'schema_version': 'floor-observation-prediction.v1',
            'cache_key': key, 'contract_fingerprint': self.fingerprint,
            'observation_id': observation_id, 'image_sha256': image_sha256,
            'tensor_file': tensor_path.relative_to(self.root).as_posix(),
            'tensor_sha256': content_hash(tensor_path),
            'tensor_spec': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()},
            'prediction': prediction,
        })
        return self.get(observation_id, image_sha256)

    def load(self, observation_id, image_sha256):
        reference = self.get(observation_id, image_sha256)
        if reference is None:
            raise KeyError(observation_id)
        record = json.loads((self.root / reference['metadata_file']).read_text())
        with np.load(self.root / reference['tensor_file'], allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in self.ARRAY_FIELDS}
        return record, arrays
