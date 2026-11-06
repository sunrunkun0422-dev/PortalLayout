"""Tentative assembly cannot silently qualify as calibrated reconstruction."""
import builtins
import copy
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from portallayout.geometry.exploratory_floor_pose import (geometry_cost, _heldout_evidence,
    select_exploratory_poses, geometric_room_relationships)
from portallayout.geometry.floor_pose_graph import PoseGraphConfig, optimize_floor_graph
from portallayout.geometry.floor_assembly import assemble_floor, pair_edges
from tests.test_floor_backend import fixture, observation


def opening(uid, wall, points):
    return dict(candidate_id=uid, wall_index=wall, endpoints_local=points, geometry_status='mapped_dominant_wall')


class ExploratoryFloorTests(unittest.TestCase):
    def test_geometry_cost_symmetric_and_identity_retained(self):
        a, b = observation('a'), observation('b')
        s = opening('c', 0, [[-.2,-.5],[.2,-.5]])
        identity = geometry_cost(a,b,np.eye(3),s,s)
        self.assertAlmostEqual(identity['cost'],0.)
        from portallayout.geometry.floor_geometry import matrix
        t=matrix([1.,.2,.5,.1])
        forward=geometry_cost(a,b,t,s,s);reverse=geometry_cost(b,a,np.linalg.inv(t),s,s)
        self.assertAlmostEqual(forward['cost'],reverse['cost'])
        with patch('portallayout.geometry.exploratory_floor_pose._nonseed_segments',return_value=[]):
            missing=geometry_cost(a,b,np.eye(3),s,s)
        self.assertIsNone(missing['same_space_cost'])

    def test_support_excludes_seed_same_wall_and_duplicate_geometry(self):
        seed=opening('seed',0,[[0,0],[1,0]])
        samewall=opening('other',0,[[2,0],[3,0]])
        held=opening('held',1,[[0,1],[1,1]])
        dup=dict(held,candidate_id='duplicate')
        rows=[(seed,seed,.9),(samewall,samewall,.9),(held,held,.8),(dup,dup,.8)]
        evidence=_heldout_evidence(rows,rows[0],np.eye(3))
        self.assertEqual(len(evidence),1);self.assertTrue(evidence[0]['supports'])

    def test_tentative_policy_and_precomputed_graph_cannot_bypass_strict(self):
        _,_,obs,pairs=fixture(2)
        for p in pairs:p.update(pose_status='tentative',match_status='tentative',calibrated=False)
        with self.assertRaisesRegex(ValueError,'tentative'):assemble_floor(obs,pairs)
        result=assemble_floor(obs,pairs,dict(inference_mode='exploratory',pose_opt='none'))
        self.assertEqual(len(result['components']),1)
        self.assertEqual(result['validation_status'],'uncalibrated_preliminary')
        self.assertFalse(result['pair_results'][0]['calibrated'])
        with self.assertRaisesRegex(ValueError,'precomputed'):
            assemble_floor(obs,[],dict(_precomputed_graph=result['pose_graph']))

    def test_room_rule_merges_same_space_but_not_adjacent(self):
        _,_,obs,pairs=fixture(2)
        for t,expected in [(np.eye(3),True),(np.array([[1,0,1],[0,1,0],[0,0,1]]),False)]:
            pairs[0]['T_B_to_A']=t.tolist()
            graph=optimize_floor_graph(['a','b'],pair_edges(pairs),config=PoseGraphConfig(mode='none'))
            relations=geometric_room_relationships(obs,graph)
            self.assertEqual(relations[0]['rule_passed'],expected)
            self.assertNotIn('probability',relations[0])

    def test_unique_independent_path_can_choose_alternative(self):
        from portallayout.geometry.floor_geometry import matrix
        obs=[observation(x) for x in 'abc'];rows=[]
        for a,b,confidence in [('a','b',.9),('b','c',.9),('a','c',.5)]:
            hs=[]
            for k,dx in enumerate(([10.,2.] if a+b=='ac' else [1.])):
                hs.append(dict(hypothesis_id=a+b+str(k),usable=True,match_confidence=confidence,
                    T_B_to_A=matrix([dx,0,0,0]).tolist(),geometry=dict(cost=float(k)),independent_evidence=[]))
            rows.append(dict(pair_id=a+b,observation_a=a,observation_b=b,numerical_status='assigned',
                matches=[dict(confidence=confidence)],pose_hypotheses=hs,pose_status='unverified',
                calibrated=False,reason_codes=[],confidence_source='test'))
        selected=select_exploratory_poses(rows,obs,.3,PoseGraphConfig(mode='none'))
        last=next(r for r in selected if r['pair_id']=='ac')
        self.assertEqual(last['selected_hypothesis_id'],'ac1')
        self.assertNotIn('ac',last['independent_path_edge_ids'])
        self.assertFalse(last['calibrated']);self.assertEqual(last['pose_status'],'tentative')
        reverse=select_exploratory_poses(list(reversed(rows)),list(reversed(obs)),.3,PoseGraphConfig(mode='none'))
        self.assertEqual({r['pair_id']:r['selected_hypothesis_id'] for r in selected},
                         {r['pair_id']:r['selected_hypothesis_id'] for r in reverse})

    def test_empty_predictions_keep_isolated_observations(self):
        obs=[observation('a'),observation('b')]
        result=assemble_floor(obs,[],dict(inference_mode='exploratory'))
        self.assertEqual([len(c['observation_ids']) for c in result['components']],[1,1])

    def test_sparse_pose_solver_matches_consistent_graph(self):
        _,_,obs,pairs=fixture(3)
        edges=pair_edges(pairs)
        dense=optimize_floor_graph(['a','b','c'],edges,config=PoseGraphConfig())
        sparse=optimize_floor_graph(['a','b','c'],edges,config=PoseGraphConfig(sparse_jacobian=True))
        for uid in 'abc':
            np.testing.assert_allclose(dense['poses'][uid]['T_local_to_component'],
                                       sparse['poses'][uid]['T_local_to_component'],atol=1e-6)

    def test_vector_residual_matches_scalar_near_identity_and_wrap(self):
        from portallayout.geometry.floor_geometry import matrix,sim2_log
        from portallayout.geometry.floor_pose_graph import _sim2_logs
        mats=np.stack([matrix([1.,-2.,a,s]) for a,s in [(0,0),(1e-9,1e-9),(3.14,.3),(-3.14,-.3),(.8,1.)]])
        np.testing.assert_allclose(_sim2_logs(mats),np.stack([sim2_log(m) for m in mats]),atol=1e-9)

    def test_prediction_cache_reuse_binding_and_no_ground_truth_reads(self):
        from tests.test_opening_pair_pipeline import PairPipelineTest
        from portallayout.engine.floor_layout_pipeline import FloorModelBundle, estimate_floor
        from portallayout.utils.room_feature_cache import fingerprint
        fixture=PairPipelineTest();fixture.setUp()
        try:
            bundle=FloorModelBundle({'pair_checkpoint':str(fixture.checkpoint())})
            cfg=dict(inference_mode='exploratory',pose_opt='none',pair_prediction_cache=str(fixture.root/'pairs'))
            def guarded(original):
                def call(path,*args,**kwargs):
                    if isinstance(path,(str,Path)) and ('labels' in str(path) or 'supervision' in str(path)):
                        raise AssertionError('GT read during inference')
                    return original(path,*args,**kwargs)
                return call
            with patch('builtins.open',side_effect=guarded(builtins.open)),patch('io.open',side_effect=guarded(io.open)):
                first=estimate_floor(fixture.cache.floors[0],bundle,cfg,fixture.cache)
                with patch.object(bundle.matcher,'forward',side_effect=AssertionError('cached matcher rerun')):
                    second=estimate_floor(fixture.cache.floors[0],bundle,cfg,fixture.cache)
                    strict=estimate_floor(fixture.cache.floors[0],bundle,dict(cfg,inference_mode='strict'),fixture.cache)
            self.assertFalse(first['provenance']['pure_prediction_cache_reused'])
            self.assertTrue(second['provenance']['pure_prediction_cache_reused'])
            self.assertEqual(len(second['pair_results']),3)
            self.assertEqual(len(strict['components']),3)
            self.assertGreater(fixture.cache.check_unchanged(),0)
            p=next((fixture.root/'pairs').glob('*.json'));data=json.loads(p.read_text());data.pop('fingerprint')
            data['binding']['weights_hash']='wrong';data['fingerprint']=fingerprint(data);p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'binding'):
                estimate_floor(fixture.cache.floors[0],bundle,cfg,fixture.cache)
        finally:fixture.tearDown()
