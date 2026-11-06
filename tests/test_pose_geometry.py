import copy
import unittest
from types import SimpleNamespace
import numpy as np
import torch

from portallayout.geometry.pose_geometry import opening_overlay,pose_hypotheses,regenerate_pair,FEATURE_VERSION
from portallayout.engine.pose_verifier_policy import prepare_pose_pair,canonical_candidates
from portallayout.geometry.floor_auxiliary_features import _nonseed_segments
from portallayout.evaluation.pose_geometry import CONFIGS,BASELINE,select_geometry
from tests.test_floor_backend import observation
from portallayout.models.opening_pair_matcher import OpeningPairView
from portallayout.utils.room_feature_cache import fingerprint


def record():
    obs=observation('a')
    obs['prediction']['openings']=[dict(candidate_id='c',token_interval_inclusive=[255,0],
        geometry_status='mapped_dominant_wall',endpoints_local=[[-.2,-.5],[.2,-.5]],wall_index=0)]
    return obs


class PoseGeometryTests(unittest.TestCase):
    def test_cpu_checkpoint_does_not_initialize_cuda(self):
        import random
        from unittest.mock import patch
        from portallayout.engine.floor_training import _rng_state, _restore_rng
        with patch('torch.cuda.is_initialized',return_value=False), \
             patch('torch.cuda.get_rng_state_all',side_effect=AssertionError('must not initialize CUDA')), \
             patch('torch.cuda.set_rng_state_all',side_effect=AssertionError('must not initialize CUDA')):
            rng=random.Random(123);state=_rng_state(rng)
            self.assertEqual(state['cuda'],[])
            _restore_rng(rng,state)
            # Older CPU checkpoints may have saved unrelated CUDA states.
            state['cuda']=[torch.zeros(1,dtype=torch.uint8)];_restore_rng(rng,state)

    def test_seam_rays_use_depth_not_feature_count_and_input_immutable(self):
        r=record();before=copy.deepcopy(r)
        arrays=dict(depth_enclosed=np.ones(256)*2,opening_probability=np.ones(256),feature_1d=np.zeros((256,512)))
        out=opening_overlay(r,arrays,'raw_depth');e=np.array(out['prediction']['openings'][0]['endpoints_local'])
        np.testing.assert_allclose(np.linalg.norm(e,axis=1),2)
        self.assertLess(np.linalg.norm(e[0]-e[1]),.05)
        self.assertEqual(r,before);self.assertEqual(out['prediction']['geometry'],r['prediction']['geometry'])

    def test_shrinkage_inverse_and_raw_scale_gate(self):
        a=np.array([[0.,1.],[1.,1.]])
        b=np.array([[1.,2.],[3.,2.]])
        for alpha in (1.,.5,0.):
            forward=pose_hypotheses(a,b,alpha);reverse=pose_hypotheses(b,a,alpha)
            self.assertEqual(len(forward),2)
            for h in forward:
                self.assertTrue(any(np.allclose(np.linalg.inv(h['T_B_to_A']),r['T_B_to_A']) for r in reverse))
                self.assertAlmostEqual(h['scale_B_to_A'],.5**alpha)
            flipped=pose_hypotheses(a,b[::-1],alpha)
            self.assertTrue(all(any(np.allclose(h['T_B_to_A'],r['T_B_to_A']) for r in flipped) for h in forward))
        self.assertEqual(pose_hypotheses(a,b*10,0.),[])

    def test_frozen_scores_features_and_dedup(self):
        a=record();a['prediction']['openings'][0].update(geometry_status='predicted_geometry_overlay',geometry_valid=True,
            support_wall_ids=[0],support_provenance_valid=True)
        b=copy.deepcopy(a);b['observation_id']='b'
        raw=dict(pair_id='p',matches=[dict(candidate_id_a='c',candidate_id_b='c',index_a=0,index_b=0,confidence=.9)],
            scores=[[2.]],conditional_ab=[[.9,.1]],conditional_ba=[[.9,.1]],pose_hypotheses=[])
        before=copy.deepcopy(raw)
        row=regenerate_pair(raw,a,b,{});self.assertEqual(raw,before)
        for k in ('scores','matches','conditional_ab','conditional_ba'):self.assertEqual(row[k],raw[k])
        duplicate=copy.deepcopy(row['pose_hypotheses'][0]);duplicate['hypothesis_id']='duplicate'
        row['pose_hypotheses'].append(duplicate);self.assertEqual(len(canonical_candidates(row)),2)
        view=OpeningPairView(torch.randn(1,256),torch.randn(256,2,4),torch.tensor([[0,1]]),torch.tensor([True]),('c',),8)
        bundle=SimpleNamespace(auxiliary_trained=lambda _:False)
        outputs=[]
        for _ in range(2):
            features=[];prepare_pose_pair(a,b,view,view,bundle,lambda r,h,f:features.append(f),raw=row);outputs.append(features)
        for x,y in zip(*outputs):
            self.assertEqual(x['feature_version'],FEATURE_VERSION)
            for key in ('raster','visual','geometry'):np.testing.assert_array_equal(x[key],y[key])

    def test_all_fit_support_walls_excluded(self):
        r=record();seed=r['prediction']['openings'][0]
        seed.update(support_wall_ids=[0,1,2,3],support_provenance_valid=True)
        self.assertEqual(_nonseed_segments(r,seed),[])
        seed.update(support_wall_ids=[],support_provenance_valid=False)
        self.assertEqual(_nonseed_segments(r,seed),[])

    def test_failed_geometry_stays_in_denominator_and_no_gain_stops(self):
        from portallayout.geometry.pose_geometry import variant_id
        variants={variant_id(c):dict(success=False,hypotheses=0) for c in CONFIGS}
        rows=[dict(role='selection',house_id='h',relationship='cross_room',collection=c,gt_valid=True,variants=variants)
              for c in ('weak','r3')]
        report=select_geometry([dict(rows=rows)])
        self.assertEqual(report['status'],'no_geometry_gain')
        self.assertEqual(report['variants'][BASELINE]['statistics']['selection']['r3']['cross_room']['pairs'],1)
        self.assertEqual(report['variants'][BASELINE]['statistics']['selection']['r3']['cross_room']['success_rate'],0)

    def test_successful_geometry_selection_returns_the_actual_config(self):
        from portallayout.geometry.pose_geometry import variant_id
        winner = dict(pose_geometry='local_m2', scale_shrinkage=0.)
        variants = {variant_id(c): dict(success=(c == winner), hypotheses=1) for c in CONFIGS}
        rows = [dict(role='selection', house_id='h', relationship='cross_room', collection=c,
                     gt_valid=True, variants=variants) for c in ('weak', 'r3')]
        report = select_geometry([dict(rows=rows)])
        self.assertEqual(report['status'], 'geometry_gain_found')
        self.assertEqual(report['selected_config'], winner)

if __name__=='__main__':unittest.main()
