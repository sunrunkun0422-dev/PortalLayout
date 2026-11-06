"""Behavioral checks for the RGB-only front end and versioned observation cache."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from portallayout.engine.floor_observation_pipeline import (
    ObservationPredictionSettings, _opening_records, predict_floor_observations,
    read_rgb_observation,
)
from portallayout.utils.room_feature_cache import ObservationFeatureCache
from portallayout.engine.single_view_prediction import prediction_to_local_geometry


def arrays(value=1.):
    n = 256
    return {'feature_1d': np.full((n, 4), value, dtype=np.float16),
            'feature_2d': np.full((8, 2, 4), value, dtype=np.float16),
            'depth_enclosed': np.full(n, 2., dtype=np.float32),
            'depth_extended': np.full(n, 2., dtype=np.float32),
            'opening_logits': np.full(n, -10., dtype=np.float32),
            'opening_probability': np.zeros(n, dtype=np.float32),
            'ratio': np.ones(1, dtype=np.float32)}


class CountingEncoder:
    def __init__(self):
        self.ids = []
        self.batch_sizes = []

    def __call__(self, ids, images):
        self.ids.extend(ids)
        self.batch_sizes.append(len(ids))
        assert images.dtype == torch.float32 and images.ndim == 4
        assert images.min() >= 0 and images.max() <= 1
        return [(arrays(float(images[i].mean())), {'openings': [], 'no_opening_detected': True})
                for i in range(len(ids))]


class FloorObservationPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.settings = ObservationPredictionSettings()

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, n=3, sample='sample_one'):
        observations = []
        for i in range(n):
            relative = 'panos/rgb_{}.png'.format(i)
            path = self.raw / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                Image.new('RGB', (32, 16), (10 + i, 40, 80)).save(path)
            observations.append({'observation_id': 'view_{}'.format(i),
                                 'rgb': {'root_key': 'zind_raw', 'relative_path': relative},
                                 'image_size_wh': [32, 16]})
        return {'schema_version': 'zind-floorset.input.v1', 'sample_id': sample,
                'floor_uid': 'floor_one', 'task': 'unposed_floor_layout_from_panoramas',
                'input_mode': 'panorama_only', 'selection_policy': 'all_panoramas',
                'projection': 'equirectangular', 'pose_prior': 'none',
                'calibration_mode': 'scale_unknown', 'observations': observations,
                'observation_order': [o['observation_id'] for o in observations]}

    def cache(self, name='cache', resume=False, contract=None):
        return ObservationFeatureCache(self.root / name, contract or {'model': 'fake.v1'}, resume)

    def test_all_views_and_incomplete_final_batch_are_preserved(self):
        for n in (0, 1, 3, 5):
            m = self.manifest(n)
            cache, encoder = self.cache('cache' + str(n)), CountingEncoder()
            r = predict_floor_observations([m], self.raw, encoder, cache, self.settings, batch_size=2)
            self.assertEqual(r['status'], 'complete')
            self.assertEqual(r['observations_inferred'], n)
            record = json.loads((cache.root / r['floors'][0]['predictions_file']).read_text())
            self.assertEqual(record['observation_order'], m['observation_order'])
            self.assertEqual(len(record['observations']), n)
            self.assertEqual(encoder.ids, m['observation_order'])

    def test_repeated_observations_across_samples_are_encoded_once(self):
        first = self.manifest(3)
        second = self.manifest(1, sample='subset')
        encoder = CountingEncoder()
        r = predict_floor_observations([first, second], self.raw, encoder, self.cache(), self.settings)
        self.assertEqual(len(encoder.ids), 3)
        self.assertEqual(len(r['floors']), 2)
        self.assertNotEqual(r['floors'][0]['predictions_file'], r['floors'][1]['predictions_file'])

    def test_resume_never_calls_encoder_and_roundtrips_float_tensors(self):
        m, encoder, cache = self.manifest(3), CountingEncoder(), self.cache()
        predict_floor_observations([m], self.raw, encoder, cache, self.settings)
        with patch.object(CountingEncoder, '__call__', side_effect=AssertionError('unexpected inference')):
            r = predict_floor_observations([m], self.raw, CountingEncoder(), self.cache(resume=True), self.settings)
        self.assertEqual(r['observations_inferred'], 0)
        self.assertEqual(r['observations_reused'], 3)
        sha, _ = read_rgb_observation(self.raw, m['observations'][0], self.settings)
        record, tensors = cache.load('view_0', sha)
        self.assertEqual(tensors['feature_2d'].shape, (8, 2, 4))
        self.assertEqual(tensors['feature_1d'].dtype, np.float16)
        self.assertTrue(record['prediction']['no_opening_detected'])

    def test_ground_truth_and_source_annotation_are_never_opened(self):
        m = self.manifest(2)
        (self.raw / 'zind_data.json').write_text('unreadable GT is not a valid JSON file')
        (self.root / 'labels').mkdir()
        (self.root / 'labels/floor_gt.json').write_text('invalid GT')
        real_open = Path.open
        def guarded(path, *args, **kwargs):
            if 'labels' in path.parts or path.name == 'zind_data.json':
                raise AssertionError('GT accessed')
            return real_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            r = predict_floor_observations([m], self.raw, CountingEncoder(), self.cache(), self.settings)
        self.assertEqual(r['status'], 'complete')
        self.assertFalse(r['ground_truth_read'])

    def test_image_change_invalidates_only_that_observation(self):
        m, cache = self.manifest(3), self.cache()
        predict_floor_observations([m], self.raw, CountingEncoder(), cache, self.settings)
        Image.new('RGB', (32, 16), (255, 0, 0)).save(self.raw / 'panos/rgb_1.png')
        encoder = CountingEncoder()
        r = predict_floor_observations([m], self.raw, encoder, cache, self.settings)
        self.assertEqual(encoder.ids, ['view_1'])
        self.assertEqual(r['observations_reused'], 2)

    def test_incompatible_contract_and_corrupt_payload_are_rejected(self):
        m, cache = self.manifest(1), self.cache()
        predict_floor_observations([m], self.raw, CountingEncoder(), cache, self.settings)
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            self.cache(resume=True, contract={'model': 'different'})
        tensor = next(cache.root.glob('observations/*/*.npz'))
        tensor.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            predict_floor_observations([m], self.raw, CountingEncoder(), cache, self.settings)

    def test_missing_image_is_reported_without_dropping_the_observation(self):
        m, cache = self.manifest(3), self.cache()
        (self.raw / 'panos/rgb_1.png').unlink()
        r = predict_floor_observations([m], self.raw, CountingEncoder(), cache, self.settings)
        self.assertEqual(r['status'], 'partial')
        self.assertEqual(len(r['input_errors']), 1)
        record = json.loads((cache.root / r['floors'][0]['predictions_file']).read_text())
        self.assertEqual(len(record['observations']), 3)
        self.assertEqual(record['observations'][1]['status'], 'input_error')

    def test_nan_outputs_unsafe_ids_and_extra_gt_arrays_are_rejected(self):
        cache = self.cache()
        data = arrays()
        sha = hashlib.sha256(b'image').hexdigest()
        data['depth_enclosed'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            cache.put('view_0', sha, data, {})
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            cache.get('../escape', sha)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            cache.put('view_0', sha, dict(arrays(), gt_pose=np.zeros((3, 3))), {})

    def test_no_openings_and_seam_crossing_preserve_true_empty_and_endpoints(self):
        geometry = {'polygon_local': [[-2, -1], [2, -1], [2, 1], [-2, 1]], 'valid': True}
        probability = np.zeros(256, dtype=np.float32)
        self.assertEqual(_opening_records('v', probability, geometry, self.settings, .5), [])
        probability[[254, 255, 0, 1, 2]] = .9
        records = _opening_records('v', probability, geometry, self.settings, .5)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['token_interval_inclusive'], [254, 2])
        self.assertTrue(records[0]['crosses_seam'])
        self.assertEqual(records[0]['geometry_status'], 'mapped_dominant_wall')
        points = np.asarray(geometry['polygon_local'])
        i = records[0]['wall_index']
        for point in np.asarray(records[0]['endpoints_local']):
            self.assertAlmostEqual(float(np.cross(points[(i+1)%4]-points[i], point-points[i])), 0)

    def test_canonical_geometry_has_no_legacy_camera_height_factor(self):
        depth = np.full(256, 2., dtype=float)
        # Force the explicit fallback so its radial frame and scale are known.
        with patch('portallayout.engine.single_view_prediction.frame_preserving_manhattan', side_effect=ValueError('fixture')):
            g = prediction_to_local_geometry(depth)
        self.assertTrue(g['postprocess']['fallback_used'])
        np.testing.assert_allclose(np.linalg.norm(g['polygon_local'], axis=1), 2., atol=1e-6)
        with self.assertRaises(ValueError):
            prediction_to_local_geometry([1, np.inf, 2])

    def test_conflicting_identity_and_invalid_settings_fail(self):
        m = self.manifest(1)
        other = copy.deepcopy(m)
        other['observations'][0]['image_size_wh'] = [64, 32]
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            predict_floor_observations([m, other], self.raw, CountingEncoder(), self.cache(), self.settings)
        for kwargs in ({'image_width': 2048}, {'maximum_openings': 0},
                       {'feature_pyramid_level': 2.5}, {'opening_threshold': float('nan')}):
            with self.assertRaises(ValueError):
                ObservationPredictionSettings(**kwargs)


if __name__ == '__main__':
    unittest.main()
