import builtins
import copy
import hashlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from portallayout.models.opening_representation import OpeningRepresentationEncoder, OpeningRepresentationConfig
from portallayout.models.opening_pair_matcher import OpeningPairMatcher, OpeningPairConfig
from portallayout.utils.room_feature_cache import ObservationFeatureCache, write_json_atomic, fingerprint
from portallayout.utils.frozen_opening_cache import FrozenOpeningCache
from portallayout.utils.opening_pair_checkpoint import checkpoint_payload, save_checkpoint, load_checkpoint, weights_hash
from portallayout.engine.opening_pair_pipeline import match_floors
from portallayout.engine.opening_pair_training import train_pairs, read_supervision, calibrate_pairs
from tests import test_opening_representation as representation_fixture


class PairPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.source = self.root/'source'
        old = ObservationFeatureCache(self.source, {'fixture': 'pair_pipeline'})
        record, arrays = representation_fixture.RepresentationAdapterTest().source()
        arrays['opening_logits'] = np.ones(32, np.float32)
        refs = []
        for i in range(3):
            value = copy.deepcopy(record['prediction'])
            value['openings'][0]['candidate_id'] = 'c'+str(i)
            refs.append(old.put('v'+str(i), hashlib.sha256(str(i).encode()).hexdigest(), arrays, value))
        floor = {'schema_version': 'floor-observation-predictions.v1', 'sample_id': 's', 'floor_uid': 'f',
                 'contract_fingerprint': old.fingerprint, 'observations': refs,
                 'observation_order': [r['observation_id'] for r in refs]}
        write_json_atomic(self.source/'floors/f.json', floor)
        self.cache = FrozenOpeningCache(self.source)
        self.enc_config = OpeningRepresentationConfig(feature_1d_dim=8, feature_2d_dim=12,
            descriptor_dim=16, context_hidden_dim=8, dropout=.1)
        self.match_config = OpeningPairConfig(dim=16, heads=4, chunk_size=2)
        rows = []
        for i, j in itertools.combinations(range(3), 2):
            rows.append({'pair_id': str(i)+str(j), 'observation_a': 'v'+str(i), 'observation_b': 'v'+str(j),
                'candidate_ids_a': ['c'+str(i)], 'candidate_ids_b': ['c'+str(j)],
                'house_id': 'fit_house', 'official_split': 'train', 'training_subset': 'fit',
                'target_a': [0 if j == 1 else -1], 'target_b': [0 if j == 1 else -1],
                'review_target_a': [-2], 'review_target_b': [-2]})
        labels = {'schema_version': 'opening-pair-supervision.v1', 'source_snapshot': self.cache.snapshot,
                  'pairs': rows, 'statistics': {'fixture': True}}
        labels['fingerprint'] = fingerprint(labels); self.labels = self.root/'labels.json'
        write_json_atomic(self.labels, labels)

    def tearDown(self): self.temp.cleanup()

    def models(self):
        return OpeningRepresentationEncoder(self.enc_config), OpeningPairMatcher(self.match_config)

    def checkpoint(self, status='smoke_trained'):
        encoder, matcher = self.models(); path = self.root/'model.pt'
        save_checkpoint(path, checkpoint_payload(encoder, matcher, training_status=status, calibration=None,
            training_contract={'source_contract': self.cache.fingerprint, 'fit_houses': ['fit_house']}))
        return path

    def test_inference_never_reads_labels_and_keeps_disconnected_nodes(self):
        checkpoint = self.checkpoint(); original_open, original_io = builtins.open, io.open
        def guarded(original):
            def call(path, *args, **kwargs):
                if isinstance(path, (str, Path)) and ('labels' in str(path) or 'zind_data' in str(path)):
                    raise AssertionError('inference read ground truth')
                return original(path, *args, **kwargs)
            return call
        with patch('builtins.open', side_effect=guarded(original_open)), patch('io.open', side_effect=guarded(original_io)):
            result = match_floors(self.cache, checkpoint, self.root/'inference', max_visualizations=0)
        self.assertEqual(result['unordered_pairs'], 3)
        self.assertEqual(result['observations_encoded'], 3)
        self.assertEqual(result['verified_edges'], 0)
        self.assertEqual(result['floors'][0]['component_sizes'], [1, 1, 1])
        self.assertFalse(result['ground_truth_read'])

    def test_checkpoint_strict_restore_and_hash_validation(self):
        path = self.checkpoint()
        a, b, state = load_checkpoint(path)
        self.assertEqual(weights_hash(a, b), state['weights_hash'])
        state['weights_hash'] = 'wrong'; save_checkpoint(self.root/'corrupt.pt', state)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): load_checkpoint(self.root/'corrupt.pt')

    def test_source_mutation_is_detected(self):
        path = next((self.source/'observations').rglob('*.json'))
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'changed'): self.cache.check_unchanged()

    def test_unknown_only_supervision_cannot_train(self):
        labels = json.loads(self.labels.read_text()); labels.pop('fingerprint')
        for row in labels['pairs']: row['target_a'], row['target_b'] = [-2], [-2]
        labels['fingerprint'] = fingerprint(labels); write_json_atomic(self.labels, labels)
        with self.assertRaisesRegex(ValueError, 'no eligible'): train_pairs(self.cache, self.labels, self.root/'train', steps=1)

    def test_stale_supervision_rejected(self):
        labels = json.loads(self.labels.read_text()); labels['source_snapshot'] = 'wrong'
        write_json_atomic(self.labels, labels)
        with self.assertRaisesRegex(ValueError, 'mismatch'): read_supervision(self.labels, self.cache)

    def test_smoke_model_cannot_be_calibrated(self):
        with self.assertRaisesRegex(ValueError, 'smoke'):
            calibrate_pairs(self.cache, self.labels, self.checkpoint(), self.root/'calibrated.pt')

    def test_training_resume_matches_uninterrupted_and_updates_both_modules(self):
        enc_factory = lambda: OpeningRepresentationEncoder(self.enc_config)
        match_factory = lambda: OpeningPairMatcher(self.match_config)
        with patch('portallayout.engine.opening_pair_training.OpeningRepresentationEncoder', side_effect=enc_factory), \
             patch('portallayout.engine.opening_pair_training.OpeningPairMatcher', side_effect=match_factory):
            first = train_pairs(self.cache, self.labels, self.root/'split', steps=2)
            resumed = train_pairs(self.cache, self.labels, self.root/'split', steps=4,
                                   resume=self.root/'split/checkpoint.pt')
            full = train_pairs(self.cache, self.labels, self.root/'full', steps=4)
        self.assertTrue(first['parameters_changed'])
        self.assertEqual(resumed['weights_hash'], full['weights_hash'])
        self.assertGreater(first['gradient_norms_first_step']['encoder.projection_2d'], 0)
        self.assertGreater(first['gradient_norms_first_step']['matcher.query'], 0)


if __name__ == '__main__': unittest.main()
