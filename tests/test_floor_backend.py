import copy
import math
import unittest
import numpy as np
from portallayout.geometry.floor_geometry import polygon, geojson, matrix, similarity, partition_graph, sim2_log
from portallayout.geometry.floor_pose_graph import optimize_floor_graph, PoseGraphConfig
from portallayout.geometry.floor_assembly import assemble_floor, complete_link_rooms
from portallayout.evaluation.floor_metrics import (evaluate_floor, maximum_cardinality_matching, FloorMetricProtocol,
    aggregate_metrics, junction_metrics)


def observation(uid):
    return {'observation_id': uid, 'prediction': {'geometry': {'enclosed': {'valid': True,
        'polygon_local': [[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]}}, 'openings': []}}


def fixture(n=3):
    ids = [chr(97+i) for i in range(n)]; observations = [observation(uid) for uid in ids]
    truth = {uid: matrix([2*i, 0, .1*i, .1*i]) for i, uid in enumerate(ids)}
    pairs = []
    for a, b in zip(ids, ids[1:]):
        pairs.append({'pair_id': a+b, 'observation_a': a, 'observation_b': b,
            'match_status': 'accepted', 'pose_status': 'accepted', 'calibrated': True,
            'T_B_to_A': (np.linalg.inv(truth[a]) @ truth[b]).tolist(), 'pose_quality': .9,
            'confidence_source': 'synthetic_test', 'matches': [], 'reason_codes': []})
    pred = assemble_floor(observations, pairs, {'floor_uid': 'f', 'pose_opt': 'none'})
    from portallayout.geometry.floor_geometry import transform_shape
    gt = {'floor_uid': 'f', 'house_id': 'h', 'metric_scale': 1., 'geometry_valid': True,
          'observations': {uid: {'pose': truth[uid].tolist(), 'pose_valid': True} for uid in ids},
          'rooms': {uid: geojson(transform_shape(polygon(observations[i]['prediction']['geometry']['enclosed']['polygon_local']), truth[uid]))
                    for i, uid in enumerate(ids)}}
    return pred, gt, observations, pairs


class FloorMetricsTests(unittest.TestCase):
    def test_perfect(self):
        result = evaluate_floor(*fixture()[:2])
        for k in ('overall_iou', 'localization_coverage', 'wall_f1', 'junction_f1'):
            self.assertAlmostEqual(result['metrics'][k], 1.)
        for k in ('translation_error_cm', 'rotation_error_degrees', 'wall_distance_cm'):
            self.assertAlmostEqual(result['metrics'][k], 0., places=6)

    def test_one_meter_is_not_aligned_away(self):
        pred, gt, obs, pairs = fixture(2)
        pairs[0]['T_B_to_A'][0][2] += 1
        pred = assemble_floor(obs, pairs, {'floor_uid': 'f', 'pose_opt': 'none'})
        m = evaluate_floor(pred, gt)['metrics']
        self.assertAlmostEqual(m['translation_error_cm'], 100.)
        self.assertEqual(m['localization_coverage'], 0.)

    def test_disconnected_targets_stay_in_denominator(self):
        pred, gt, obs, pairs = fixture()
        pred = assemble_floor(obs, pairs[:1], {'floor_uid': 'f', 'pose_opt': 'none'})
        report = evaluate_floor(pred, gt)
        self.assertEqual(report['metrics']['localization_coverage'], .5)
        self.assertLess(report['metrics']['overall_iou'], 1)
        self.assertGreater(report['wall']['fn'], 0)

    def test_isolates_are_not_perfect_poses(self):
        _, gt, obs, _ = fixture()
        pred = assemble_floor(obs, [], {'floor_uid': 'f'})
        m = evaluate_floor(pred, gt)['metrics']
        self.assertEqual(m['localization_coverage'], 0.)
        self.assertIsNone(m['translation_error_cm'])

    def test_single_observation(self):
        m = evaluate_floor(*fixture(1)[:2])['metrics']
        self.assertEqual(m['overall_iou'], 1.)
        self.assertIsNone(m['localization_coverage'])

    def test_missing_metric_scale(self):
        pred, gt, _, _ = fixture(); gt['metric_scale'] = None
        m = evaluate_floor(pred, gt)['metrics']
        self.assertEqual(m['overall_iou'], 1.)
        for k in ('translation_error_cm', 'localization_coverage', 'wall_f1', 'junction_f1', 'wall_distance_cm'):
            self.assertIsNone(m[k])

    def test_display_and_order_invariant(self):
        pred, gt, _, _ = fixture(); base = evaluate_floor(pred, gt)['metrics']
        pred['observations'].reverse(); pred['observation_order'].reverse()
        for c in pred['components']: c['display_transform'] = matrix([999, 999, 1., 0]).tolist()
        self.assertEqual(evaluate_floor(pred, gt)['metrics'], base)

    def test_wrap_yaw(self):
        pred, gt, obs, pairs = fixture(2)
        gt['observations']['b']['pose'] = matrix([2, 0, math.radians(-179), 0]).tolist()
        pairs[0]['T_B_to_A'] = matrix([2, 0, math.radians(179), 0]).tolist()
        pred = assemble_floor(obs, pairs, {'floor_uid': 'f', 'pose_opt': 'none'})
        self.assertAlmostEqual(evaluate_floor(pred, gt)['metrics']['rotation_error_degrees'], 2.)

    def test_matching_maximizes_cardinality_before_distance(self):
        matches = maximum_cardinality_matching([[1, 1], [1, 0]], [[0., 10.], [10., 0.]])
        self.assertEqual(set(matches), {(0, 1), (1, 0)})

    def test_one_prediction_cannot_match_two_junctions(self):
        a = {'junctions': [{'point': [0, 0]}]}
        b = {'junctions': [{'point': [.01, 0]}, {'point': [-.01, 0]}]}
        self.assertEqual(junction_metrics(a, b, .1)['tp'], 1)

    def test_metric_units_conversion(self):
        pred, gt, _, _ = fixture(); first = evaluate_floor(pred, gt)['metrics']
        from portallayout.geometry.floor_geometry import transform_shape
        scale = np.diag([100., 100., 1.]); gt['metric_scale'] = .01
        for row in gt['observations'].values(): row['pose'] = (scale @ np.array(row['pose'])).tolist()
        gt['rooms'] = {k: geojson(transform_shape(polygon(v), scale)) for k, v in gt['rooms'].items()}
        second = evaluate_floor(pred, gt)['metrics']
        for k, v in first.items(): self.assertAlmostEqual(second[k], v, places=7)

    def test_empty_geometry_penalties(self):
        pred, gt, _, _ = fixture()
        for room in pred['rooms']: room['geometry'] = None
        m = evaluate_floor(pred, gt)['metrics']
        self.assertEqual(m['overall_iou'], 0.)
        self.assertEqual(m['wall_f1'], 0.)
        self.assertEqual(m['junction_f1'], 0.)
        self.assertEqual(m['wall_distance_cm'], 200.)

    def test_aggregation_has_pose_counts(self):
        row = evaluate_floor(*fixture()[:2])
        result = aggregate_metrics([row])
        self.assertEqual(result['evaluable_floors']['translation_error_cm'], 1)
        self.assertAlmostEqual(result['metrics']['translation_error_cm'], 0.)


class FloorGeometryTests(unittest.TestCase):
    def test_internal_partition_survives(self):
        a = polygon([[0,0],[1,0],[1,1],[0,1]])
        b = polygon([[1,0],[2,0],[2,1],[1,1]])
        graph = partition_graph({'a':a,'b':b}, .01)
        self.assertEqual(len(graph['walls']), 7)
        self.assertEqual(sum(len(w['room_ids']) == 2 for w in graph['walls']), 1)

    def test_invalid_polygon_not_repaired(self):
        self.assertIsNone(polygon([[0,0],[1,1],[0,1],[1,0]]))

    def test_reflections_rejected(self):
        with self.assertRaises(ValueError): similarity(np.diag([-1,1,1]))

    def test_lie_translation_at_identity(self):
        np.testing.assert_allclose(sim2_log(matrix([2,3,0,0])), [2,3,0,0])

    def test_complete_link_blocks_unknown_chain(self):
        relationships = [dict(a='a',b='b',probability=.99,valid=True), dict(a='b',b='c',probability=.99,valid=True)]
        self.assertEqual(complete_link_rooms(['c','a','b'], relationships, .98), [['a','b'],['c']])

    def test_same_space_union(self):
        _, _, obs, pairs = fixture(2)
        pairs[0]['T_B_to_A'] = np.eye(3).tolist()
        prediction = assemble_floor(obs,pairs,{'floor_uid':'f','room_relationships':[dict(a='a',b='b',probability=.99,valid=True)]})
        self.assertEqual(len(prediction['rooms']), 1)
        self.assertAlmostEqual(polygon(prediction['rooms'][0]['geometry']).area,1.)


class FloorPoseGraphTests(unittest.TestCase):
    def test_chain_scale_and_inverse(self):
        expected = {'a': np.eye(3), 'b': matrix([2,1,.4,.2]), 'c': matrix([-1,2,-.3,-.1])}
        edges = [dict(edge_id=a+b,a=a,b=b,T_B_to_A=(np.linalg.inv(expected[a])@expected[b]).tolist(),weight=.9)
                 for a,b in [('b','a'),('b','c')]]
        graph = optimize_floor_graph(['c','b','a'],edges)
        for uid in expected: np.testing.assert_allclose(graph['poses'][uid]['T_local_to_component'],expected[uid],atol=1e-7)
        self.assertTrue(all(e['bridge'] for e in graph['components'][0]['edges']))

    def test_isolate_retained(self):
        result = optimize_floor_graph(['a','b','c'],[dict(edge_id='ab',a='a',b='b',T_B_to_A=matrix([1,0,0,0]).tolist(),weight=.8)])
        self.assertEqual([len(c['observation_ids']) for c in result['components']],[2,1])

    def test_bad_transform_not_identity_fallback(self):
        result = optimize_floor_graph(['a','b'],[dict(edge_id='ab',a='a',b='b',T_B_to_A=np.zeros((3,3)).tolist(),weight=.8)])
        self.assertEqual(len(result['invalid_edges']),1)
        self.assertEqual(len(result['components']),2)

    def test_noisy_loop_optimizes_all_edges(self):
        edges = [dict(edge_id='ab',a='a',b='b',T_B_to_A=matrix([1,0,0,0]).tolist(),weight=.9),
                 dict(edge_id='bc',a='b',b='c',T_B_to_A=matrix([1,0,0,0]).tolist(),weight=.9),
                 dict(edge_id='ac',a='a',b='c',T_B_to_A=matrix([2.1,0,0,0]).tolist(),weight=.9)]
        result = optimize_floor_graph(['a','b','c'],edges)
        c = result['components'][0]
        self.assertEqual(c['optimization_status'],'optimized')
        self.assertEqual(len(c['edges']),3)
        self.assertLess(c['optimization_history'][0]['final_cost'],c['optimization_history'][0]['initial_cost'])
        self.assertFalse(any(e['bridge'] for e in c['edges']))


if __name__ == '__main__': unittest.main()
