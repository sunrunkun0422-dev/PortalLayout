import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from portallayout.models.opening_pair_matcher import OpeningPairView
from portallayout.models.floor_auxiliary import POSE_FEATURE_VERSION
from portallayout.engine.pose_verifier_policy import canonical_candidates,prepare_pose_pair,select_verifier_pairs,accepted_best
from portallayout.evaluation.pose_verifier import ap,quality,select_workpoint,assess_workpoint
from portallayout.evaluation.pose_candidate_audit import synthetic_coordinate_audit
from tests.test_floor_backend import observation
from tests.test_exploratory_floor import opening


def hypothesis(uid='h',score=.9,seed='c',dx=0):
    t=np.eye(3);t[0,2]=dx
    return dict(hypothesis_id=uid,T_B_to_A=t.tolist(),usable=True,match_confidence=score,
                seed_candidate_ids=[seed,seed],verifier_probability=.99,nonfit_evidence_available=True)


def metric_row(uid,target=1,probability=.99,house='h0',scope='cross_room',**kw):
    return dict(pair_id=uid,hypothesis_id=uid,house_id=house,target=target,probability=probability,
                relationship=scope,opening_score=.9,independent_evidence=True,**kw)


class PoseVerifierPolicyTests(unittest.TestCase):
    def test_coordinate_contract_known_transforms(self):
        self.assertEqual(len(synthetic_coordinate_audit()['cases']),24)

    def test_canonical_before_features_and_permutation(self):
        hs=[hypothesis('weak',.7,'z'),hypothesis('best',.9,'a'),hypothesis('second',.8,'b',1)]
        row={'pose_hypotheses':hs}
        result=canonical_candidates(row)
        self.assertEqual([h['hypothesis_id'] for h in result],['best','second'])
        self.assertEqual(len(result[0]['sources']),2)
        self.assertEqual(result,canonical_candidates({'pose_hypotheses':list(reversed(hs))}))
        hs[1]['T_B_to_A'][0][0]=float('nan')
        self.assertTrue(all(np.isfinite(h['T_B_to_A']).all() for h in canonical_candidates(row)))

    def test_feature_export_and_inference_share_canonical_inputs(self):
        torch.manual_seed(3)
        a,b=observation('a'),observation('b')
        seed=opening('c',0,[[-.2,-.5],[.2,-.5]])
        a['prediction']['openings']=[seed];b['prediction']['openings']=[seed]
        view=OpeningPairView(torch.randn(1,256),torch.randn(256,2,4),torch.tensor([[0,1]]),torch.tensor([True]),('c',),8)
        row=dict(pair_id='p',pose_hypotheses=[hypothesis('h'),hypothesis('duplicate',.8)],
                 matches=[dict(candidate_id_a='c',candidate_id_b='c',index_a=0,index_b=0,confidence=.9)],
                 conditional_ab=[[.9,.1]],conditional_ba=[[.9,.1]])
        features=[];bundle=SimpleNamespace(auxiliary_trained=lambda _:False)
        p=prepare_pose_pair(a,b,view,view,bundle,lambda r,h,f:features.append(f),raw=row)
        self.assertEqual(len(features),1)
        again=[];q=prepare_pose_pair(a,b,view,view,bundle,lambda r,h,f:again.append(f),raw=row)
        self.assertEqual(p,q)
        for k in ('raster','visual','geometry'):np.testing.assert_array_equal(features[0][k],again[0][k])
        self.assertEqual(features[0]['feature_version'],POSE_FEATURE_VERSION)
        self.assertIsNone(p['pose_hypotheses'][0]['evidence_supports_pose'])

    def test_filter_never_reconnects_and_empty(self):
        row=dict(pair_id='p',pose_hypotheses=[hypothesis()],pose_status='tentative',T_B_to_A=np.eye(3).tolist(),reason_codes=[])
        rejected=select_verifier_pairs([row],'verifier_filter',dict(status='selection_workpoint_infeasible',opening_threshold=.7))[0]
        self.assertIsNone(rejected['T_B_to_A']);self.assertEqual(rejected['pose_status'],'unverified')
        self.assertFalse(rejected['calibrated'])
        ranked=select_verifier_pairs([row],'verifier_rank')[0]
        self.assertEqual(ranked['pose_status'],'tentative');self.assertFalse(ranked['calibrated'])
        self.assertEqual(select_verifier_pairs([],'verifier_filter'),[])
        row['pose_hypotheses'][0]['nonfit_evidence_available']=False
        self.assertIsNone(accepted_best(row['pose_hypotheses'],.5,0))

    def test_tied_ap_unknown_and_house_macro(self):
        rows=[metric_row('a',1,.5),metric_row('b',0,.5),metric_row('c',-1,1.)]
        self.assertEqual(ap(rows),.5)
        rows += [metric_row('d',1,.9,house='h1')]
        self.assertEqual(quality(rows)['cross_room_house_macro_ap'],.75)
        self.assertIsNone(ap([metric_row('x',0)]))

    def test_workpoint_recall_keeps_evidence_rejects_unknown_not_negative(self):
        rows=[metric_row(str(i),house='h'+str(i%5)) for i in range(20)]
        rows += [metric_row('neg'+str(i),0,.1) for i in range(100)]
        self.assertEqual(select_workpoint(rows)['status'],'selection_workpoint_found')
        rows[0]['independent_evidence']=False
        r=assess_workpoint(rows,.9,.1)
        self.assertEqual(r['overall']['positive_pairs'],20)
        self.assertEqual(r['overall']['recall'],19/20)
        self.assertFalse(r['passed'])
        self.assertEqual(select_workpoint([metric_row('only')]+rows[20:])['status'],'selection_workpoint_infeasible')
        rows.append(metric_row('unknown',-1,.99))
        r=assess_workpoint(rows,.9,.1)
        self.assertEqual(r['overall']['negative_pairs'],100);self.assertEqual(r['overall']['accepted_unknown'],1)

    def test_pose_only_ap_training_resume(self):
        from portallayout.engine.floor_training import train_auxiliaries
        from portallayout.utils.room_feature_cache import content_hash,fingerprint,write_json_atomic
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pair=root/'pair.pt';pair.write_bytes(b'fixture');rows=[];rng=np.random.RandomState(123)
            for role in ('fit','selection'):
                for target in (0,1):
                    uid=role+str(target);p=root/(uid+'.npz')
                    np.savez(p,raster=rng.randn(2,7,16,16).astype('float32'),visual=rng.randn(2,1024).astype('float32'),geometry=rng.randn(2,32).astype('float32'))
                    rows.append(dict(record_id=uid,task='pose',role=role,target=target,house_id=role,
                        feature_file=p.name,feature_sha256=content_hash(p),pair_id=uid,hypothesis_id=uid,relationship='cross_room'))
            data=dict(feature_version=POSE_FEATURE_VERSION,pair_checkpoint_sha256=content_hash(pair),
                      house_protocol=dict(fit=['fit'],selection=['selection'],calibration=[]),records=rows)
            data['fingerprint']=fingerprint(data);labels=root/'supervision.json';write_json_atomic(labels,data)
            kw=dict(batch_size=2,mode='full',validation_interval=2,tasks=('pose',),selection_metric='cross_room_ap')
            r=train_auxiliaries(labels,pair,root/'split',steps=2,**kw)
            self.assertEqual(r['training_status'],'trained');self.assertEqual(r['stages']['room']['status'],'untrained')
            train_auxiliaries(labels,pair,root/'split',steps=4,resume=root/'split/last.pt',**kw)
            train_auxiliaries(labels,pair,root/'full',steps=4,**kw)
            a=torch.load(root/'split/checkpoint.pt');b=torch.load(root/'full/checkpoint.pt')
            for key in a['pose_state']:torch.testing.assert_close(a['pose_state'][key],b['pose_state'][key],rtol=0,atol=0)
            for key in a['room_state']:torch.testing.assert_close(a['room_state'][key],b['room_state'][key],rtol=0,atol=0)
            self.assertTrue((root/'full/best_bce.pt').exists())

    def test_pipeline_does_not_override_filter_with_geometry_or_paths(self):
        from unittest.mock import patch
        from tests.test_opening_pair_pipeline import PairPipelineTest
        from tests.test_floor_backend import fixture
        from portallayout.engine.floor_layout_pipeline import FloorModelBundle,estimate_floor
        from portallayout.utils.room_feature_cache import fingerprint
        source=PairPipelineTest();source.setUp()
        try:
            bundle=FloorModelBundle({'pair_checkpoint':str(source.checkpoint())})
            _,_,obs,pairs=fixture(3)
            for o in obs:o['input_status']='predicted'
            for p in pairs:p['pose_hypotheses']=[hypothesis()]
            workpoint=dict(status='selection_workpoint_infeasible',opening_threshold=.7,weights_hash=bundle.weights_hash,
                           candidate_policy='pose-verifier-candidates.v1',source_snapshot=source.cache.snapshot)
            workpoint['fingerprint']=fingerprint(workpoint)
            cfg=dict(inference_mode='exploratory',pose_policy='verifier_filter',pose_workpoint=workpoint,
                     pose_opt='robust',room_fusion=False)
            with patch('portallayout.engine.pose_verifier_policy.score_floor',return_value=(obs,pairs,dict(encoding_seconds=0.,pair_seconds=0.,reused=True))), \
                 patch('portallayout.geometry.exploratory_floor_pose.select_exploratory_poses',side_effect=AssertionError('geometry must not reconnect')), \
                 patch('portallayout.engine.floor_layout_pipeline.resolve_with_independent_paths',side_effect=AssertionError('path must not reconnect')):
                pred=estimate_floor(dict(floor_uid='f',observation_order=['a','b','c']),bundle,cfg,source.cache)
            self.assertEqual([len(c['observation_ids']) for c in pred['components']],[1,1,1])
            self.assertTrue(all(p['T_B_to_A'] is None for p in pred['pair_results']))
        finally:source.tearDown()
