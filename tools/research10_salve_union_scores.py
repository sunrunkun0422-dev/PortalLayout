#!/usr/bin/env python3
"""Bind label-free SALVe BEV scoring to a frozen R09 candidate union.

This is a scoring adapter, not a candidate generator or renderer. A manifest
must be produced from the union by an independently audited upstream pipeline.
Missing renders remain in the denominator as unsupported candidates.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.research10_candidate_comparison import (
    FORBIDDEN_ID, SCORE_VERSION, UNION_VERSION, digest, file_hash, score_accounting,
)
from tools.research10_salve_scoring import OfficialSalveVerifier, score_manifest, validate_manifest


def validate_union(union):
    if not isinstance(union, dict) or union.get('schema_version') != UNION_VERSION:
        raise ValueError('invalid union schema')
    fingerprint = union.get('union_fingerprint')
    # The CLI attaches source hashes after build_union computes its fingerprint.
    payload = {k: v for k, v in union.items() if k not in ('union_fingerprint', 'source_sha256')}
    if fingerprint != digest(payload):
        raise ValueError('union fingerprint mismatch')
    rows = union.get('candidates')
    if not isinstance(rows, list):
        raise ValueError('invalid candidate list')
    ids = [row['candidate_id'] for row in rows]
    if len(set(ids)) != len(ids) or union.get('union_candidate_denominator') != len(ids):
        raise ValueError('union candidate denominator mismatch')
    return set(ids)


def validate_image_paths(manifest, base_dir):
    base = Path(base_dir)
    for row in manifest['candidates']:
        for value in row['images'].values():
            if value is None:
                continue
            path = Path(value)
            resolved = (path if path.is_absolute() else base / path).resolve()
            if any(token in part.lower() for part in resolved.parts for token in FORBIDDEN_ID):
                raise ValueError('label-bearing image path forbidden')


def bind_scores(union, manifest, scorer, *, base_dir):
    known = validate_union(union)
    validate_manifest(manifest)
    provided = {row['candidate_id'] for row in manifest['candidates']}
    if provided - known:
        raise ValueError('manifest contains candidates outside the union')
    validate_image_paths(manifest, base_dir)
    base = Path(base_dir)
    raw = score_manifest(manifest, scorer, base_dir=base)
    indexed = {row['candidate_id']: row for row in raw['rows']}
    scores = []
    for key in sorted(known):
        row = indexed.get(key)
        if row is None:
            scores.append(dict(candidate_id=key, status='unsupported', score=None,
                               reason='render_not_provided'))
        elif row['status'] == 'scored':
            scores.append(dict(candidate_id=key, status='scored',
                               score=row['positive_probability'], reason=None))
        elif row['status'] == 'incomplete':
            scores.append(dict(candidate_id=key, status='unsupported', score=None,
                               reason='render_incomplete'))
        else:
            scores.append(dict(candidate_id=key, status='failed', score=None,
                               reason='verifier_failed'))
    result = dict(schema_version=SCORE_VERSION, method='salve_dav2',
                  union_fingerprint=union['union_fingerprint'], scores=scores)
    # Reuse the exact accounting validator used by the comparison tool.
    return result, score_accounting(union, 'salve_dav2', result), raw


def image_snapshot(manifest, base_dir):
    """Capture each actual image's bytes (including explicit missing paths)."""
    paths = set()
    for row in manifest['candidates']:
        for value in row['images'].values():
            if value is not None:
                path = Path(value)
                paths.add((path if path.is_absolute() else Path(base_dir) / path).resolve())
    return {str(path): file_hash(path) if path.is_file() else None for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--union', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--salve-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a new directory')
    for path in (args.union, args.manifest):
        if any(token in part.lower() for part in path.resolve().parts for token in FORBIDDEN_ID):
            parser.error('label-bearing input path forbidden')
    hashes = {str(p.resolve()): file_hash(p) for p in (args.union, args.manifest)}
    union = json.loads(args.union.read_text(encoding='utf-8'))
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    validate_union(union)
    validate_manifest(manifest)
    validate_image_paths(manifest, args.manifest.resolve().parent)
    images_before = image_snapshot(manifest, args.manifest.resolve().parent)
    project_root = Path(__file__).resolve().parents[1]
    salve_root = args.salve_root.resolve()
    source_paths = [Path(__file__).resolve(), project_root / 'tools/research10_salve_scoring.py',
                    project_root / 'tools/research10_candidate_comparison.py',
                    salve_root / 'salve/models/early_fusion.py', salve_root / 'salve/train_utils.py']
    source_hashes = {str(p): file_hash(p) for p in source_paths}
    scorer = OfficialSalveVerifier(args.salve_root, args.device, args.threads)
    result, accounting, raw = bind_scores(union, manifest, scorer, base_dir=args.manifest.resolve().parent)
    if any(file_hash(path) != expected for path, expected in hashes.items()):
        raise ValueError('input changed during scoring')
    if image_snapshot(manifest, args.manifest.resolve().parent) != images_before:
        raise ValueError('rendered image changed during scoring')
    if any(file_hash(path) != expected for path, expected in source_hashes.items()):
        raise ValueError('source changed during scoring')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, obj in (('scores.json', result), ('accounting.json', accounting),
                      ('verifier_details.json', raw),
                      ('provenance.json', {'input_sha256': hashes, 'verifier': scorer.metadata,
                       'image_sha256_or_missing': images_before,
                       'source_sha256': source_hashes,
                       'scope': 'verifier_only_upstream_rendering_and_coordinates_not_verified'})):
        (args.output / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(accounting['counts']))


if __name__ == '__main__':
    main()
