"""Strict joint representation/matcher checkpoints, with explicit training state."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from portallayout.models.opening_representation import OpeningRepresentationEncoder, OpeningRepresentationConfig
from portallayout.models.opening_pair_matcher import OpeningPairMatcher, OpeningPairConfig


def weights_hash(encoder, matcher):
    h = hashlib.sha256()
    for prefix, model in (('encoder', encoder), ('matcher', matcher)):
        for name, value in sorted(model.state_dict().items()):
            a = value.detach().cpu().contiguous().numpy()
            h.update(json.dumps([prefix, name, list(a.shape), str(a.dtype)]).encode()); h.update(a.tobytes())
    return h.hexdigest()


def checkpoint_payload(encoder, matcher, **metadata):
    return dict(schema_version='opening-pair-checkpoint.v1',
                encoder_config=asdict(encoder.config), matcher_config=asdict(matcher.config),
                encoder_state=encoder.state_dict(), matcher_state=matcher.state_dict(),
                matcher_type=getattr(matcher, 'matcher_type', 'new'),
                weights_hash=weights_hash(encoder, matcher), **metadata)


def save_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as f:
        temp = Path(f.name)
    try:
        torch.save(payload, str(temp)); os.replace(str(temp), str(path))
    finally:
        if temp.exists(): temp.unlink()


def load_checkpoint(path, device='cpu'):
    payload = torch.load(str(path), map_location='cpu')
    if payload.get('schema_version') != 'opening-pair-checkpoint.v1':
        raise ValueError('invalid pair checkpoint schema')
    if payload.get('training_status') not in ('initialized_untrained', 'smoke_trained', 'trained'):
        raise ValueError('checkpoint lacks explicit training status')
    encoder = OpeningRepresentationEncoder(OpeningRepresentationConfig(**payload['encoder_config'])).to(device)
    kind = payload.get('matcher_type', 'new')
    if kind == 'legacy':
        from portallayout.models.legacy_floor_matcher import LegacyFloorMatcher, LegacyFloorConfig
        matcher = LegacyFloorMatcher(LegacyFloorConfig(**payload['matcher_config'])).to(device)
    elif kind == 'new':
        matcher = OpeningPairMatcher(OpeningPairConfig(**payload['matcher_config'])).to(device)
    else:
        raise ValueError('unknown matcher architecture')
    encoder.load_state_dict(payload['encoder_state'], strict=True)
    matcher.load_state_dict(payload['matcher_state'], strict=True)
    if weights_hash(encoder, matcher) != payload['weights_hash']:
        raise ValueError('checkpoint weight hash mismatch')
    return encoder, matcher, payload
