#!/usr/bin/env python3
"""Score explicitly listed BEV quadruplets without reading or deriving GT labels.

Manifest format (relative image paths are relative to the manifest directory)::

  {"schema": "salve-unlabeled-bev.v1", "candidates": [
    {"candidate_id": "candidate-1", "images": {
      "ceiling_a": "path/a.jpg", "ceiling_b": "path/b.jpg",
      "floor_a": "path/c.jpg", "floor_b": "path/d.jpg"}}
  ]}

This tool neither constructs hypotheses nor renders BEV images. It does not
establish whether input images or candidate selection are GT-independent. Those
upstream stages require their own audit before any end-to-end comparison.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, UnidentifiedImageError


ROLES = ('ceiling_a', 'ceiling_b', 'floor_a', 'floor_b')
OFFICIAL_CHECKPOINT_MD5 = '1200ffbe47d836557d88fef052952337'


def validate_manifest(manifest):
    """Whitelist candidate identity and four image locations; reject labels."""
    if not isinstance(manifest, dict) or set(manifest) != {'schema', 'candidates'}:
        raise ValueError('manifest requires only schema and candidates')
    if manifest['schema'] != 'salve-unlabeled-bev.v1' or not isinstance(manifest['candidates'], list):
        raise ValueError('unsupported manifest schema or candidates')
    seen = set()
    for row in manifest['candidates']:
        if not isinstance(row, dict) or set(row) != {'candidate_id', 'images'}:
            raise ValueError('candidate requires only candidate_id and images')
        candidate_id = row['candidate_id']
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError('candidate_id must be a nonempty string')
        if candidate_id in seen:
            raise ValueError('duplicate candidate_id')
        seen.add(candidate_id)
        if not isinstance(row['images'], dict) or set(row['images']) - set(ROLES):
            raise ValueError('images must only contain the four BEV roles')
        for value in row['images'].values():
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError('image paths must be nonempty strings or null')
    return manifest


def _read_rgb(path):
    # Decode bytes in isolation; the model is never given a path or directory.
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return np.asarray(image.convert('RGB')).copy()


def score_manifest(manifest, scorer, *, base_dir):
    """Return a label-free result for each candidate without dropping failures.

    ``scorer`` is called with a tuple of four RGB arrays, ordered ceiling A,
    ceiling B, floor A, floor B; it returns class-1 probability as a scalar. The
    callback never sees file paths, candidate IDs, or any possible GT labels.
    """
    validate_manifest(manifest)
    base = Path(base_dir)
    rows = []
    for candidate in manifest['candidates']:
        images, missing, invalid = [], [], []
        for role in ROLES:
            value = candidate['images'].get(role)
            if value is None:
                missing.append(role)
                continue
            path = Path(value)
            if not path.is_absolute():
                path = base / path
            try:
                images.append(_read_rgb(path))
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                missing.append(role)
            except (OSError, ValueError, UnidentifiedImageError):
                invalid.append(role)
        row = {'candidate_id': candidate['candidate_id'], 'positive_probability': None}
        if missing or invalid:
            row.update(status='incomplete', reason='image_unavailable',
                       missing_roles=missing, unreadable_roles=invalid)
        else:
            try:
                score = float(scorer(tuple(images)))
                if not math.isfinite(score) or not 0. <= score <= 1.:
                    raise ValueError('invalid probability')
                row.update(status='scored', positive_probability=score, reason=None)
            except Exception:
                # Preserve this candidate in the denominator, with no exception
                # text or paths that could expose directory-derived labels.
                row.update(status='scoring_failed', reason='model_or_probability_error')
        rows.append(row)
    return {'schema': 'salve-unlabeled-bev-scores.v1',
            'candidate_count': len(rows),
            'scored_count': sum(r['status'] == 'scored' for r in rows),
            'incomplete_count': sum(r['status'] == 'incomplete' for r in rows),
            'scoring_failed_count': sum(r['status'] == 'scoring_failed' for r in rows),
            'rows': rows}


def _digest(path, algorithm):
    result = hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


class OfficialSalveVerifier:
    """Official four-image ResNet-152 verifier, loaded lazily from SALVe checkout."""

    def __init__(self, salve_root, device='cpu', threads=2):
        root = Path(salve_root).resolve()
        checkpoint = root / 'checkpoints/salve' / (OFFICIAL_CHECKPOINT_MD5 + '.pth')
        if not checkpoint.is_file():
            raise FileNotFoundError('official SALVe verifier checkpoint missing')
        md5 = _digest(checkpoint, 'md5')
        if md5 != OFFICIAL_CHECKPOINT_MD5:
            raise ValueError('official SALVe verifier checkpoint MD5 mismatch')
        sys.path.insert(0, str(root))
        import torch
        from salve.models.early_fusion import EarlyFusionCEResnet
        from salve.train_utils import get_val_test_transform
        if threads < 1:
            raise ValueError('threads must be positive')
        torch.set_num_threads(threads)
        if device != 'cpu' and not (device.startswith('cuda') and torch.cuda.is_available()):
            raise ValueError('requested device unavailable')
        self.torch, self.device = torch, torch.device(device)
        cfg = SimpleNamespace(modalities=['ceiling_rgb_texture', 'floor_rgb_texture'],
                              resize_h=234, resize_w=234, train_h=224, train_w=224)
        self.model = torch.nn.DataParallel(EarlyFusionCEResnet(152, False, 2, cfg))
        state = torch.load(checkpoint, map_location='cpu')
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.model.to(self.device).eval()
        self.transform = get_val_test_transform(cfg)
        self.metadata = {'architecture': 'official_SALVe_EarlyFusionCEResnet152',
                         'checkpoint_md5': md5, 'checkpoint_sha256': _digest(checkpoint, 'sha256'),
                         'device': device, 'torch_version': torch.__version__,
                         'strict_state_dict_load': True}

    def __call__(self, images):
        tensors = self.transform(*images)
        with self.torch.inference_mode():
            logits = self.model(*[x.unsqueeze(0).to(self.device) for x in tensors], None, None)
            if tuple(logits.shape) != (1, 2) or not self.torch.isfinite(logits).all():
                raise ValueError('invalid verifier logits')
            return float(logits.softmax(1)[0, 1].item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--salve-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new path')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    validate_manifest(manifest)  # Fail before loading 700MB of model weights.
    scorer = OfficialSalveVerifier(args.salve_root, args.device, args.threads)
    result = score_manifest(manifest, scorer, base_dir=args.manifest.resolve().parent)
    result['verifier'] = scorer.metadata
    result['scope'] = 'verifier_scoring_only_upstream_and_full_system_not_audited'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'candidate_count': result['candidate_count'], 'scored_count': result['scored_count'],
                      'incomplete_count': result['incomplete_count'],
                      'scoring_failed_count': result['scoring_failed_count']}))


if __name__ == '__main__':
    main()
