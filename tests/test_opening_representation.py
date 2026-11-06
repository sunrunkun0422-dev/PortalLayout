"""Geometry/context encoder invariants, fixed predictions and cache adaptation."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from portallayout.models.opening_representation import (
    OpeningRepresentationConfig, OpeningRepresentationEncoder, RepresentationCandidates,
    PredictedOpeningGeometry, circular_grid_sample,
)
from portallayout.utils.opening_representation_adapter import (
    prepare_opening_representation_batch, predicted_wall_attributes,
)
from tools.encode_floor_openings import encode_prediction_cache
from portallayout.utils.room_feature_cache import ObservationFeatureCache, content_hash, write_json_atomic


class OpeningRepresentationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = OpeningRepresentationConfig(feature_1d_dim=8, feature_2d_dim=12,
            descriptor_dim=16, context_hidden_dim=8, dropout=0.)
        self.model = OpeningRepresentationEncoder(self.config).eval()

    def fixture(self, b=2, k=3):
        f1, f2 = torch.randn(b, 32, 8), torch.randn(b, 12, 8, 16)
        p = torch.rand(b, 32)
        intervals = torch.tensor([[[30, 2], [8, 12], [18, 22]]]).expand(b, -1, -1)[:, :k].clone()
        valid = torch.ones(b, k, dtype=torch.bool)
        ids = tuple(tuple('opening_{}'.format(i) for i in range(k)) for _ in range(b))
        attributes = torch.tensor([1., 4., .25, -.2, .4, .85]).expand(b, k, -1).clone()
        geometry = PredictedOpeningGeometry(1+torch.rand(b, 32), 3+torch.rand(b, 32),
            torch.full((b, 1), 1.4), attributes, torch.ones(b, k, 6, dtype=torch.bool))
        return f1, f2, p, RepresentationCandidates(intervals, valid, ids), geometry

    def test_shapes_identity_and_diagnostic_normalization(self):
        args = self.fixture()
        out = self.model(*args, return_diagnostics=True)
        self.assertEqual(out.descriptors.shape, (2, 3, 16))
        self.assertEqual(out.projected_feature_2d.shape, (2, 16, 8, 16))
        self.assertEqual(out.candidate_ids, args[3].candidate_ids)
        self.assertEqual(out.diagnostics['sampling_uv'].shape, (2, 3, 20, 2))
        for name, axis in [('fusion_weights', -1), ('structure_attention', -1)]:
            torch.testing.assert_close(out.diagnostics[name].sum(axis), torch.ones(2, 3))
        torch.testing.assert_close(out.diagnostics['context_attention'].sum((-2, -1)), torch.ones(2, 3))
        self.assertEqual(out.diagnostics['geometry_attributes'].shape, (2, 3, 12))
        self.assertTrue(all(torch.isfinite(value).all() for value in out.diagnostics.values()))

    def test_empty_single_and_all_padding_candidates(self):
        for k in (0, 1, 3):
            args = list(self.fixture(k=k))
            if k == 3:
                args[3] = replace(args[3], valid=torch.zeros(2, k, dtype=torch.bool))
            out = self.model(*args, return_diagnostics=True)
            self.assertEqual(out.descriptors.shape, (2, k, 16))
            self.assertTrue(torch.isfinite(out.descriptors).all())
            if k in (0, 3):
                self.assertEqual(out.descriptors.count_nonzero(), 0)
        # One image can be entirely padding while another remains valid.
        args = list(self.fixture())
        mask = args[3].valid.clone()
        mask[0] = False
        args[3] = replace(args[3], valid=mask)
        out = self.model(*args)
        self.assertEqual(out.descriptors[0].count_nonzero(), 0)
        self.assertGreater(out.descriptors[1].abs().sum(), 0)

    def test_candidate_permutation_equivariance(self):
        args = list(self.fixture())
        first = self.model(*args).descriptors
        order = torch.tensor([2, 0, 1])
        c, g = args[3:]
        args[3] = RepresentationCandidates(c.intervals[:, order], c.valid[:, order],
            tuple(tuple(row[i] for i in order.tolist()) for row in c.candidate_ids))
        args[4] = replace(g, wall_attributes=g.wall_attributes[:, order],
                          wall_attribute_valid=g.wall_attribute_valid[:, order])
        second = self.model(*args).descriptors
        torch.testing.assert_close(second, first[:, order], atol=2e-6, rtol=2e-5)

    def test_padding_cannot_change_real_candidate_descriptors(self):
        args = list(self.fixture(b=1))
        c, g = args[3:]
        args[3] = replace(c, valid=torch.tensor([[True, True, False]]))
        padded = self.model(*args).descriptors[:, :2]
        args[3] = RepresentationCandidates(c.intervals[:, :2], torch.ones(1, 2, dtype=torch.bool),
                                           (c.candidate_ids[0][:2],))
        args[4] = replace(g, wall_attributes=g.wall_attributes[:, :2],
                         wall_attribute_valid=g.wall_attribute_valid[:, :2])
        trimmed = self.model(*args).descriptors
        torch.testing.assert_close(trimmed, padded, atol=2e-6, rtol=2e-5)

    def test_images_in_a_batch_do_not_exchange_candidate_information(self):
        args = list(self.fixture())
        first = self.model(*args).descriptors[0].detach().clone()
        args[0][1] *= -11
        args[1][1] *= 7
        args[2][1] = 1 - args[2][1]
        args[3] = replace(args[3], intervals=args[3].intervals.clone())
        args[3].intervals[1] = (args[3].intervals[1] + 9) % 32
        second = self.model(*args).descriptors[0]
        torch.testing.assert_close(first, second)

    def test_sampling_is_continuous_and_periodic_at_the_seam(self):
        feature = torch.tensor([[[[0., 10., 20., 30.]]]], requires_grad=True)
        uv = torch.tensor([[[0., .5], [1., .5], [.125, .5], [.875, .5], [1.125, .5],
                            [1e-6, .5], [-1e-6, .5]]])
        result = circular_grid_sample(feature, uv)[0, :, 0]
        torch.testing.assert_close(result[:5], torch.tensor([15., 15., 0., 30., 0.]))
        self.assertLess(abs(float(result[5]-result[6])), .001)
        result.sum().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())

    def test_circular_roll_equivariance_uses_relative_not_absolute_yaw(self):
        args = list(self.fixture(b=1))
        first = self.model(*args, return_diagnostics=True)
        args[0], args[1], args[2] = args[0].roll(4, 1), args[1].roll(2, -1), args[2].roll(4, 1)
        args[3] = replace(args[3], intervals=(args[3].intervals + 4) % 32)
        args[4] = replace(args[4], depth_enclosed=args[4].depth_enclosed.roll(4, 1),
                         depth_extended=args[4].depth_extended.roll(4, 1))
        second = self.model(*args, return_diagnostics=True)
        torch.testing.assert_close(first.descriptors, second.descriptors, atol=1e-5, rtol=2e-5)
        torch.testing.assert_close(first.diagnostics['soft_cover'].roll(2, -1),
                                   second.diagnostics['soft_cover'], atol=2e-6, rtol=2e-5)

    def test_missing_geometry_uses_visual_context_without_nans(self):
        args = list(self.fixture())
        g = args[4]
        args[4] = replace(g, depth_enclosed=g.depth_enclosed.new_full(g.depth_enclosed.shape, torch.nan),
            depth_extended=g.depth_extended.new_full(g.depth_extended.shape, torch.nan),
            height_ratio=g.height_ratio.new_full(g.height_ratio.shape, torch.nan),
            wall_attributes=g.wall_attributes.new_full(g.wall_attributes.shape, torch.nan))
        out = self.model(*args, return_diagnostics=True)
        self.assertTrue(torch.isfinite(out.descriptors).all())
        self.assertEqual(out.diagnostics['sampling_valid'].count_nonzero(), 0)
        self.assertEqual(out.diagnostics['fusion_weights'][..., 0].count_nonzero(), 0)
        # Angular width and detection probabilities remain known.
        expected = torch.zeros(12, dtype=torch.bool)
        expected[[0, 9, 10]] = True
        torch.testing.assert_close(out.diagnostics['geometry_attribute_valid'][0, 0], expected)

    def test_gradients_reach_enhancement_but_fixed_predictions_are_unchanged(self):
        args = list(self.fixture())
        args[0].requires_grad_()
        args[1].requires_grad_()
        args[2].requires_grad_()
        args[4].depth_enclosed.requires_grad_()
        snapshot = [x.detach().clone() for x in args[:3]]
        intervals = args[3].intervals.clone()
        attrs = args[4].wall_attributes.clone()
        out = self.model(*args)
        loss = out.descriptors[..., 0].sum() + out.descriptors.square().mean()
        loss.backward()
        for branch in ('projection_1d', 'projection_2d', 'structure_query', 'context_key', 'geometry_mlp', 'gate', 'intra_attention'):
            parameters = getattr(self.model, branch).parameters()
            gradients = [p.grad for p in parameters]
            self.assertTrue(any(x is not None and x.abs().sum() > 0 for x in gradients), branch)
            self.assertTrue(all(x is None or torch.isfinite(x).all() for x in gradients), branch)
        self.assertIsNone(args[2].grad)
        self.assertIsNone(args[4].depth_enclosed.grad)
        self.assertGreater(args[0].grad.abs().sum(), 0)
        self.assertGreater(args[1].grad.abs().sum(), 0)
        for a, before in zip(args[:3], snapshot):
            torch.testing.assert_close(a, before)
        torch.testing.assert_close(args[3].intervals, intervals)
        torch.testing.assert_close(args[4].wall_attributes, attrs)

    def test_scale_normalized_geometry_attributes_and_validity(self):
        args = list(self.fixture())
        original = self.model(*args, return_diagnostics=True).diagnostics['geometry_attributes']
        g = args[4]
        scaled = g.wall_attributes.clone()
        scaled[..., :2] *= 5
        args[4] = replace(g, wall_attributes=scaled, depth_enclosed=g.depth_enclosed*5,
                         depth_extended=g.depth_extended*5)
        new = self.model(*args, return_diagnostics=True).diagnostics['geometry_attributes']
        torch.testing.assert_close(original, new)

    def test_full_circle_zero_probabilities_and_ablation_switches(self):
        args = list(self.fixture(b=1, k=1))
        args[2] = torch.zeros_like(args[2])
        args[3] = replace(args[3], intervals=torch.tensor([[[0, 31]]]))
        for name, enabled in [('baseline', 0), ('structure', 1), ('structure_context', 2),
                              ('structure_context_geometry', 3), ('full', 3)]:
            out = self.model(*args, return_diagnostics=True, ablation=name)
            self.assertTrue(torch.isfinite(out.descriptors).all())
            self.assertEqual(int(out.diagnostics['branch_valid'].sum()), enabled)
            if name == 'baseline':
                torch.testing.assert_close(out.descriptors, out.baseline_descriptors)
        self.assertAlmostEqual(float(out.diagnostics['candidate_widths'][0, 0]), 1)
        torch.testing.assert_close(out.diagnostics['soft_cover'][..., 0],
                                   out.diagnostics['soft_cover'][..., -1], atol=.05, rtol=0)

    def test_invalid_core_inputs_are_rejected(self):
        args = list(self.fixture())
        bad = args[2].clone()
        bad[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, 'probability'):
            self.model(args[0], args[1], bad, *args[3:])
        with self.assertRaisesRegex(ValueError, 'interval'):
            self.model(*args[:3], replace(args[3], intervals=args[3].intervals + 32), args[4])
        with self.assertRaisesRegex(ValueError, 'IDs'):
            self.model(*args[:3], replace(args[3], candidate_ids=(('x', 'x', 'x'),)*2), args[4])


class RepresentationAdapterTest(unittest.TestCase):
    def source(self):
        p = {'coordinate_spec': {'frame': 'bi_panorama_local_xz', 'units': 'camera_height_normalized'},
             'geometry': {'enclosed': {'valid': True, 'polygon_local': [[-2,-1],[2,-1],[2,1],[-2,1]]}},
             'openings': [{'candidate_id': 'stable_opening', 'token_interval_inclusive': [30,2],
                'geometry_status': 'mapped_dominant_wall', 'wall_index': 0,
                'endpoints_local': [[-1,-1],[1,-1]], 'wall_ratios': [.25,.75],
                'geometry_quality': {'dominantWallFraction': .8}}]}
        arrays = {'feature_1d': np.ones((32,8),np.float16), 'feature_2d': np.ones((12,8,16),np.float16),
            'opening_probability': np.full(32,.8,np.float32), 'depth_enclosed': np.full(32,2.,np.float32),
            'depth_extended': np.full(32,3.,np.float32), 'ratio': np.array([1.4],np.float32)}
        return {'observation_id': 'view_one', 'prediction': p}, arrays

    def test_adapter_retains_identity_and_handles_variable_candidate_counts(self):
        record, arrays = self.source()
        other = copy.deepcopy(record)
        other['observation_id'] = 'view_two'
        other['prediction']['openings'] = []
        snapshot = copy.deepcopy(record)
        batch = prepare_opening_representation_batch([record,other],[arrays,arrays])
        self.assertEqual(batch.candidates.candidate_ids, (('stable_opening',),(None,)))
        self.assertEqual(batch.candidates.valid.tolist(), [[True],[False]])
        self.assertEqual(batch.feature_1d.dtype, torch.float32)
        self.assertEqual(record, snapshot)
        np.testing.assert_allclose(batch.predicted_geometry.wall_attributes[0,0].numpy(), [2,4,.5,0,0,.8])
        self.assertTrue(batch.predicted_geometry.wall_attribute_valid[0,0].all())

    def test_wall_attributes_are_winding_invariant_and_missingness_is_explicit(self):
        record, _ = self.source()
        p = record['prediction']
        a, mask = predicted_wall_attributes(p,p['openings'][0])
        other = copy.deepcopy(p)
        other['geometry']['enclosed']['polygon_local'].reverse()
        other['openings'][0]['wall_index'] = 2
        b, other_mask = predicted_wall_attributes(other,other['openings'][0])
        np.testing.assert_allclose(a,b)
        np.testing.assert_array_equal(mask,other_mask)
        missing = copy.deepcopy(p['openings'][0])
        missing.update(geometry_status='unavailable',geometry_quality={})
        values, available = predicted_wall_attributes(p,missing)
        self.assertFalse(available.any())
        self.assertFalse(values.any())

    def test_adapter_rejects_foreign_coordinate_frames(self):
        record, arrays = self.source()
        record['prediction']['coordinate_spec']['units'] = 'meters'
        with self.assertRaisesRegex(ValueError, 'canonical'):
            prepare_opening_representation_batch([record],[arrays])


class RepresentationCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.output = self.root / 'representations'
        self.cache = ObservationFeatureCache(self.source, {'model': 'fixture'})
        record, arrays = RepresentationAdapterTest().source()
        arrays['opening_logits'] = np.ones(32, np.float32)
        refs = []
        for i in range(2):
            prediction = copy.deepcopy(record['prediction'])
            if i == 1:
                prediction['openings'] = []
            refs.append(self.cache.put('view_' + str(i), hashlib.sha256(str(i).encode()).hexdigest(), arrays, prediction))
        refs.append({'observation_id':'view_missing','status':'input_error','reason':'RGB missing'})
        self.floor = {'schema_version':'floor-observation-predictions.v1',
            'sample_id':'sample_fixture','floor_uid':'floor_fixture',
            'contract_fingerprint':self.cache.fingerprint,
            'observation_order':[x['observation_id'] for x in refs], 'observations':refs}
        write_json_atomic(self.source/'floors/fixture.json',self.floor)
        torch.manual_seed(5)
        self.model = OpeningRepresentationEncoder(OpeningRepresentationConfig(feature_1d_dim=8,
            feature_2d_dim=12, descriptor_dim=16, context_hidden_dim=8, dropout=0.)).eval()

    def tearDown(self):
        self.temp.cleanup()

    def test_cache_roundtrip_and_resume_preserve_source_candidates_and_errors(self):
        before = {str(p):content_hash(p) for p in self.source.rglob('*') if p.is_file()}
        first = encode_prediction_cache(self.source,self.output,self.model,diagnostics=True)
        self.assertEqual(first['status'],'partial')
        self.assertEqual(first['observations_encoded'],2)
        self.assertEqual(first['openings'],1)
        floor = json.loads((self.output/first['floors'][0]['representations_file']).read_text())
        self.assertEqual(floor['observation_order'],self.floor['observation_order'])
        self.assertEqual(floor['observations'][2]['status'],'source_unavailable')
        for i,ref in enumerate(floor['observations'][:2]):
            record = json.loads((self.output/ref['metadata_file']).read_text())
            self.assertEqual(record['training_status'],'initialized_untrained')
            self.assertEqual(record['candidate_ids'],['stable_opening'] if i==0 else [])
            with np.load(self.output/ref['tensor_file'],allow_pickle=False) as arrays:
                self.assertEqual(arrays['descriptors'].shape,(1-i,16))
                self.assertEqual(arrays['projected_feature_2d'].shape,(16,8,16))
        with patch.object(self.model,'forward',side_effect=AssertionError('cache re-encoded')):
            second = encode_prediction_cache(self.source,self.output,self.model,diagnostics=True,resume=True)
        self.assertEqual(second['observations_encoded'],0)
        self.assertEqual(second['observations_reused'],2)
        self.assertEqual(before,{str(p):content_hash(p) for p in self.source.rglob('*') if p.is_file()})

    def test_corruption_and_contract_changes_are_rejected(self):
        encode_prediction_cache(self.source,self.output,self.model,diagnostics=True)
        with self.assertRaisesRegex(ValueError,'identical contract'):
            encode_prediction_cache(self.source,self.output,self.model,resume=True,diagnostics=False)
        tensor = next((self.output/'observations').glob('*.npz'))
        tensor.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'corrupt'):
            encode_prediction_cache(self.source,self.output,self.model,resume=True,diagnostics=True)

    def test_shared_observation_is_encoded_once_across_floor_sample_views(self):
        another = copy.deepcopy(self.floor)
        another['sample_id'] = 'sample_other_view'
        write_json_atomic(self.source/'floors/other.json',another)
        with patch.object(self.model,'forward',wraps=self.model.forward) as forward:
            report = encode_prediction_cache(self.source,self.output,self.model,batch_size=2)
        self.assertEqual(report['floor_samples'],2)
        self.assertEqual(report['unique_observations'],2)
        self.assertEqual(report['observations_encoded'],2)
        self.assertEqual(forward.call_count,1)
        self.assertNotEqual(report['floors'][0]['representations_file'],report['floors'][1]['representations_file'])

    def test_changed_source_prediction_invalidates_only_its_representation(self):
        encode_prediction_cache(self.source,self.output,self.model)
        path = self.source/self.floor['observations'][0]['metadata_file']
        record = json.loads(path.read_text())
        record['prediction']['openings'][0]['token_interval_inclusive']=[29,3]
        write_json_atomic(path,record)
        report = encode_prediction_cache(self.source,self.output,self.model,resume=True)
        self.assertEqual(report['observations_encoded'],1)
        self.assertEqual(report['observations_reused'],1)

    def test_no_dataset_or_gt_files_are_needed(self):
        original = Path.open
        def guarded(path,*args,**kwargs):
            if 'labels' in path.parts or 'index' in path.parts or path.name=='zind_data.json':
                raise AssertionError('GT accessed')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):
            result = encode_prediction_cache(self.source,self.output,self.model,diagnostics=True)
        self.assertFalse(result['ground_truth_read'])


if __name__ == '__main__':
    unittest.main()
