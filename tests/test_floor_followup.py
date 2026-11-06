"""Whole-floor acceptance regressions, independent of legacy pair smoke tests."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from portallayout.geometry.floor_assembly import complete_link_rooms
from portallayout.geometry.floor_pose_graph import optimize_floor_graph, PoseGraphConfig
from portallayout.geometry.floor_geometry import matrix
from portallayout.models.floor_auxiliary import PoseHypothesisVerifier, SameSpaceAssociator


class FloorFollowupTests(unittest.TestCase):
    def test_auxiliary_roles_are_checked_against_house_split(self):
        from portallayout.models.floor_auxiliary import FEATURE_VERSION
        from portallayout.engine.floor_training import AuxiliaryFeatureDataset
        from portallayout.utils.room_feature_cache import fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'supervision.json'
            value = dict(feature_version=FEATURE_VERSION,
                         house_protocol={'fit': ['house_a'], 'selection': [], 'calibration': ['house_b']},
                         records=[dict(record_id='r', house_id='house_a', role='calibration')])
            value['fingerprint'] = fingerprint(value); path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'role disagrees'):
                AuxiliaryFeatureDataset(path)

    def test_failed_floor_keeps_inputs_pairs_and_metric_denominators(self):
        from tests.test_floor_backend import fixture
        from portallayout.engine.floor_layout_pipeline import failed_floor
        from portallayout.evaluation.floor_metrics import evaluate_floor
        prediction, gt, _, _ = fixture(3)
        failure = failed_floor(prediction, RuntimeError('forced numerical failure'))
        self.assertEqual(len(failure['observations']), 3)
        self.assertEqual(len(failure['pair_results']), 3)
        self.assertEqual(failure['components'], [])
        measured = evaluate_floor(failure, gt)
        self.assertTrue(measured['runtime_failure'])
        self.assertEqual(measured['metrics']['overall_iou'], 0.)
        self.assertEqual(measured['metrics']['localization_coverage'], 0.)
        self.assertEqual(measured['metrics']['wall_f1'], 0.)
        self.assertIsNone(measured['metrics']['translation_error_cm'])

    def test_calibration_does_not_skip_narrow_feasible_threshold(self):
        from portallayout.engine.floor_calibration import binary_workpoint
        rows = [dict(probability=float(p), target=0, house_id=str(i%5))
                for i,p in enumerate(np.linspace(0., .5, 1001))]
        rows += [dict(probability=.50001+i*.00001, target=1, house_id=str(i%5)) for i in range(19)]
        rows.append(dict(probability=.9,target=0,house_id='0'))
        result = binary_workpoint(rows)
        self.assertEqual(result['status'],'calibrated')
        self.assertAlmostEqual(result['precision'],.95)
        self.assertEqual(result['recall'],1.)

    def test_room_union_preserves_disconnected_regions_and_unobserved_holes(self):
        from tests.test_floor_backend import fixture
        from portallayout.geometry.floor_assembly import assemble_floor
        from portallayout.geometry.floor_geometry import polygon
        _, _, obs, pairs = fixture(2)
        relation = [dict(a='a', b='b', probability=.999, valid=True)]
        result = assemble_floor(obs,pairs,dict(room_relationships=relation, pose_opt='none'))
        self.assertEqual(polygon(result['rooms'][0]['geometry']).geom_type, 'MultiPolygon')
        _, _, obs, pairs = fixture(4)
        strips = [[[0,0],[4,0],[4,1],[0,1]], [[0,3],[4,3],[4,4],[0,4]],
                  [[0,1],[1,1],[1,3],[0,3]], [[3,1],[4,1],[4,3],[3,3]]]
        for row, points in zip(obs,strips): row['prediction']['geometry']['enclosed']['polygon_local'] = points
        for pair in pairs: pair['T_B_to_A'] = np.eye(3).tolist()
        relations = [dict(a=a,b=b,probability=.999,valid=True) for a,b in itertools.combinations('abcd',2)]
        result = assemble_floor(obs,pairs,dict(room_relationships=relations,pose_opt='none'))
        shape = polygon(result['rooms'][0]['geometry'])
        self.assertEqual(len(result['rooms']),1)
        self.assertEqual(len(shape.interiors),1)
        self.assertAlmostEqual(shape.area,12.)

    def test_pose_feature_evidence_is_stable_across_opening_gate(self):
        from portallayout.engine.floor_layout_pipeline import predict_pair
        matches = [dict(index_a=i, index_b=i, confidence=p) for i,p in enumerate((.9,.3))]
        assigned = dict(status='assigned', indices_a=torch.tensor([0,1]), indices_b=torch.tensor([0,1]),
                        conditional_ab=torch.tensor([[.9,0,.1],[0,.3,.2]]),
                        conditional_ba=torch.tensor([[.9,0,.1],[0,.3,.2]]),
                        iterations=10, marginal_error=0., converged=True)
        view = SimpleNamespace(candidate_ids=['c0','c1'])
        def proposals(a,b, selected, calibrated):
            return [dict(seed_candidate_ids=[m['candidate_id_a'],m['candidate_id_b']],
                         hypotheses=[dict(T_B_to_A=matrix([m['index_a'],0,0,0]).tolist(),
                                          endpoint_mapping='direct')]) for m in selected]
        evidence_counts = []
        def features(*args, **kwargs):
            evidence_counts.append(len(kwargs['matches']))
            return {'independent_evidence': True}
        bundle = SimpleNamespace(matcher=lambda a,b: assigned, auxiliary_status='initialized_untrained',
                                 calibration={'opening':{'threshold':.8}}, calibrated=lambda stage: stage=='opening')
        a,b = [dict(observation_id=uid,prediction={}) for uid in ('a','b')]
        with patch('portallayout.engine.floor_layout_pipeline.mutual_matches', side_effect=lambda *args:[dict(m) for m in matches]), \
             patch('portallayout.engine.floor_layout_pipeline.pose_proposals', side_effect=proposals), \
             patch('portallayout.engine.floor_layout_pipeline.auxiliary_features', side_effect=features):
            accepted = predict_pair(a,b,view,view,bundle)
            bundle.calibrated = lambda stage: False
            diagnostic = predict_pair(a,b,view,view,bundle)
        self.assertEqual(len(accepted['matches']), 1)
        self.assertEqual(len(accepted['pose_hypotheses']), 1)
        self.assertEqual(len(diagnostic['pose_hypotheses']), 2)
        self.assertEqual(evidence_counts, [2,2,2])

    def test_auxiliary_training_resume_matches_uninterrupted(self):
        from portallayout.models.floor_auxiliary import FEATURE_VERSION
        from portallayout.engine.floor_training import train_auxiliaries
        from portallayout.utils.room_feature_cache import content_hash, fingerprint, write_json_atomic
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pair = root/'pair.pt'; pair.write_bytes(b'identity fixture')
            rows = []; rng = np.random.RandomState(123)
            for task, dim in [('pose', 1024), ('room', 512)]:
                for role in ('fit', 'selection'):
                    for target in (0, 1):
                        uid = '{}_{}_{}'.format(task, role, target); path = root/(uid+'.npz')
                        np.savez(path, raster=rng.randn(2,7,16,16).astype('float32'),
                                 visual=rng.randn(2,dim).astype('float32'), geometry=rng.randn(2,32).astype('float32'))
                        rows.append(dict(record_id=uid, task=task, role=role, target=target, house_id=role,
                                         feature_file=path.name, feature_sha256=content_hash(path)))
            data = dict(feature_version=FEATURE_VERSION, pair_checkpoint_sha256=content_hash(pair),
                        house_protocol={'fit':['fit'], 'selection':['selection'], 'calibration':[]}, records=rows)
            data['fingerprint'] = fingerprint(data); labels = root/'supervision.json'; write_json_atomic(labels, data)
            kwargs = dict(batch_size=2, mode='smoke', validation_interval=2)
            train_auxiliaries(labels, pair, root/'split', steps=2, **kwargs)
            train_auxiliaries(labels, pair, root/'split', steps=4, resume=root/'split/last.pt', **kwargs)
            train_auxiliaries(labels, pair, root/'full', steps=4, **kwargs)
            resumed = torch.load(root/'split/checkpoint.pt'); full = torch.load(root/'full/checkpoint.pt')
            for task in ('pose', 'room'):
                for key, value in full[task+'_state'].items():
                    torch.testing.assert_close(value, resumed[task+'_state'][key], rtol=0, atol=0)
            # A previously selected best model may differ from the last optimizer
            # step. Persisting a second head must retain BOTH states for the first.
            from portallayout.engine.floor_training import predict_auxiliary as real_predict
            def selection_schedule():
                calls = {'pose': 0, 'room': 0}
                def predict(model, dataset, values, device='cpu', batch_size=32):
                    if values and values[0]['role'] == 'selection':
                        task = values[0]['task']; calls[task] += 1
                        good = .9 if calls[task] == 1 else .6
                        return [good if row['target'] else 1-good for row in values]
                    return real_predict(model, dataset, values, device, batch_size)
                return predict
            with patch('portallayout.engine.floor_training.predict_auxiliary', side_effect=selection_schedule()):
                train_auxiliaries(labels, pair, root/'best_split', steps=4, **kwargs)
                train_auxiliaries(labels, pair, root/'best_split', steps=6,
                                  resume=root/'best_split/last.pt', **kwargs)
            with patch('portallayout.engine.floor_training.predict_auxiliary', side_effect=selection_schedule()):
                train_auxiliaries(labels, pair, root/'best_full', steps=6, **kwargs)
            resumed_last = torch.load(root/'best_split/last.pt')
            full_last = torch.load(root/'best_full/last.pt')
            for task in ('pose', 'room'):
                for key, value in full_last['stages'][task]['last_model_state'].items():
                    torch.testing.assert_close(value, resumed_last['stages'][task]['last_model_state'][key], rtol=0, atol=0)

    def test_floor_pair_training_resume_matches_uninterrupted(self):
        from tests.test_opening_pair_pipeline import PairPipelineTest
        from portallayout.engine.floor_training import train_floor_pairs
        from portallayout.utils.opening_pair_checkpoint import load_checkpoint
        fixture = PairPipelineTest(); fixture.setUp()
        self.addCleanup(fixture.tearDown)
        torch.set_num_threads(2)
        initialization = fixture.checkpoint()
        protocol = {'fit': ['fit_house'], 'selection': [], 'calibration': []}
        kwargs = dict(batch_size=2, mode='smoke', validation_interval=2, initialization=initialization)
        train_floor_pairs(fixture.cache, fixture.labels, protocol, fixture.root/'split', steps=2, **kwargs)
        train_floor_pairs(fixture.cache, fixture.labels, protocol, fixture.root/'split', steps=4,
                          resume=fixture.root/'split/last.pt', **kwargs)
        train_floor_pairs(fixture.cache, fixture.labels, protocol, fixture.root/'full', steps=4, **kwargs)
        resumed = load_checkpoint(fixture.root/'split/checkpoint.pt')[2]
        full = load_checkpoint(fixture.root/'full/checkpoint.pt')[2]
        self.assertEqual(resumed['weights_hash'], full['weights_hash'])
        self.assertEqual(resumed['training_status'], 'smoke_trained')

    def test_room_unknown_or_conflict_veto_is_order_independent(self):
        positive = dict(a='a', b='b', probability=.999, valid=True)
        for other in (dict(a='b', b='a', probability=.1, valid=True),
                      dict(a='b', b='a', probability=None, valid=False),
                      dict(a='b', b='a', probability=float('nan'), valid=True)):
            for rows in itertools.permutations([positive, other]):
                self.assertEqual(complete_link_rooms(['b', 'a'], rows, .98), [['a'], ['b']])

    def test_auxiliary_prepare_accepts_empty_cli_output(self):
        from portallayout.datasets.floor_auxiliary_supervision import build_auxiliary_supervision
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root/'output'; output.mkdir()
            cache = SimpleNamespace(root=root/'cache', floors=[], snapshot='s', fingerprint='f', before={},
                                    check_unchanged=lambda: 0)
            with patch('portallayout.datasets.floor_auxiliary_supervision.house_protocol', return_value={}), \
                 patch('portallayout.datasets.opening_pair_supervision.build_weak_supervision', return_value={'pairs': []}), \
                 patch('portallayout.engine.floor_layout_pipeline.FloorModelBundle'), \
                 patch('portallayout.datasets.floor_auxiliary_supervision.content_hash', return_value='hash'):
                build_auxiliary_supervision(cache, root/'portallayout.datasets', root/'weights', output)
            self.assertTrue((output/'supervision.json').is_file())
            with self.assertRaisesRegex(ValueError, 'new auxiliary'):
                build_auxiliary_supervision(cache, root/'portallayout.datasets', root/'weights', output)

    def test_auxiliary_symmetry_gradient_and_restore(self):
        torch.set_num_threads(2)
        for cls, dim in ((PoseHypothesisVerifier, 1024), (SameSpaceAssociator, 512)):
            torch.manual_seed(123); model = cls()
            values = [torch.randn(2, 2, 7, 32, 32), torch.randn(2, 2, dim), torch.randn(2, 2, 32)]
            original = model(*values).detach()
            torch.testing.assert_close(original, model(*(v.flip(1) for v in values)).detach())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(*values), torch.tensor([0., 1.]))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            self.assertFalse(torch.equal(original, model(*values)))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'state.pt'; torch.save(model.state_dict(), path)
                restored = cls(); restored.load_state_dict(torch.load(path), strict=True)
                torch.testing.assert_close(model(*values), restored(*values), rtol=0, atol=0)

    def test_optimizer_failure_falls_back_to_forest(self):
        edge = dict(edge_id='ab', a='a', b='b', T_B_to_A=matrix([1, 0, .1, .2]).tolist(), weight=.9)
        with patch('portallayout.geometry.floor_pose_graph.least_squares', side_effect=ValueError('forced failure')):
            result = optimize_floor_graph(['b', 'a'], [edge])
        self.assertEqual(result['components'][0]['optimization_status'], 'optimizer_failed')
        np.testing.assert_allclose(result['poses']['b']['T_local_to_component'], edge['T_B_to_A'])
        self.assertEqual(result['rejected_edges'], [])

    def test_pruned_loop_rebuild_shares_three_round_budget(self):
        positions = {'a':0., 'b':2., 'c':1., 'd':3.}
        edges = [dict(edge_id=a+b, a=a, b=b,
                      T_B_to_A=matrix([positions[b]-positions[a],0,0,0]).tolist(), weight=.95)
                 for a,b in [('a','c'),('c','b'),('a','d'),('d','b')]]
        edges.append(dict(edge_id='bad_ab', a='a', b='b', T_B_to_A=matrix([10,0,0,0]).tolist(), weight=.1))
        result = optimize_floor_graph(list(positions), edges)
        self.assertIn('bad_ab', result['rejected_edges'])
        rounds = sum(p['rounds'] for p in result['pruning_passes'])
        rounds += max(len(c['optimization_history']) for c in result['components'])
        self.assertLessEqual(rounds, 3)
        for uid, x in positions.items():
            np.testing.assert_allclose(result['poses'][uid]['T_local_to_component'], matrix([x,0,0,0]), atol=1e-6)

    def test_bad_loop_does_not_remove_bridge(self):
        edges = [dict(edge_id=a+b, a=a, b=b, T_B_to_A=matrix([x, 0, 0, 0]).tolist(), weight=w)
                 for a,b,x,w in [('a','b',1,.95),('b','c',1,.95),('a','c',8,.2),('c','d',1,.8)]]
        result = optimize_floor_graph(['d','c','b','a'], edges)
        self.assertNotIn('cd', result['rejected_edges'])
        self.assertEqual(len(result['components']), 1)
        diagnostics = {e['edge_id']: e for e in result['components'][0]['edges']}
        self.assertTrue(diagnostics['cd']['bridge'])
        reordered = optimize_floor_graph(['a','b','c','d'], list(reversed(edges)))
        for uid in result['poses']:
            np.testing.assert_allclose(result['poses'][uid]['T_local_to_component'],
                                       reordered['poses'][uid]['T_local_to_component'], atol=1e-6)


if __name__ == '__main__': unittest.main()
