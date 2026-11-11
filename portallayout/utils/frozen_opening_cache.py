"""Read-only, checked prediction cache access shared by pair training/inference."""
import json
from collections import OrderedDict
from pathlib import Path

from portallayout.datasets.zind_floor_dataset import safe_path
from portallayout.utils.room_feature_cache import ObservationFeatureCache, content_hash, fingerprint
from portallayout.models.opening_pair_matcher import OpeningPairView


class FrozenOpeningCache:
    def __init__(self, root):
        self.root = Path(root).resolve()
        marker = self.root/'cache_contract.json'
        contract = json.loads(marker.read_text())
        self.cache = ObservationFeatureCache(self.root, contract['contract'], resume=True)
        self.fingerprint = self.cache.fingerprint
        self.floors, self.records, self.paths = [], {}, {marker}
        for path in sorted((self.root/'floors').glob('*.json')):
            floor = json.loads(path.read_text())
            if floor['contract_fingerprint'] != self.fingerprint:
                raise ValueError('source floor contract mismatch')
            if floor['observation_order'] != [r['observation_id'] for r in floor['observations']]:
                raise ValueError('floor observation order mismatch')
            self.floors.append(floor); self.paths.add(path)
            for ref in floor['observations']:
                if ref['status'] != 'predicted':
                    continue
                meta = safe_path(self.root, ref['metadata_file'])
                record = json.loads(meta.read_text())
                if self.cache.get(record['observation_id'], record['image_sha256']) != ref:
                    raise ValueError('source cache reference mismatch')
                uid = record['observation_id']
                if uid in self.records and self.records[uid] != record:
                    raise ValueError('conflicting source observation identity')
                self.records[uid] = record
                self.paths.update((meta, safe_path(self.root, record['tensor_file'])))
        if not self.floors:
            raise ValueError('no cached floors')
        self.before = {str(p): content_hash(p) for p in self.paths}
        self.snapshot = fingerprint(self.before)
        self._arrays = OrderedDict()

    def load(self, uid):
        record = self.records[uid]
        if uid not in self._arrays:
            _, self._arrays[uid] = self.cache.load(uid, record['image_sha256'])
        self._arrays.move_to_end(uid)
        while len(self._arrays) > 32:
            self._arrays.popitem(last=False)
        return record, self._arrays[uid]

    def check_unchanged(self):
        if any(content_hash(p) != digest for p, digest in self.before.items()):
            raise ValueError('frozen source cache changed during processing')
        return len(self.before)


def pair_views(encoded, prepared):
    """Preserve graph connections during training; callers choose no_grad."""
    result = []
    diagnostics = encoded.diagnostics or {}
    for i in range(len(prepared.observation_ids)):
        result.append(OpeningPairView(encoded.descriptors[i], encoded.projected_feature_2d[i],
            prepared.candidates.intervals[i], prepared.candidates.valid[i],
            prepared.candidates.candidate_ids[i], prepared.feature_1d.shape[1],
            diagnostics['geometry_attributes'][i] if 'geometry_attributes' in diagnostics else None,
            diagnostics['geometry_attribute_valid'][i] if 'geometry_attribute_valid' in diagnostics else None,
            prepared.feature_1d[i], prepared.opening_probability[i]))
    return result
