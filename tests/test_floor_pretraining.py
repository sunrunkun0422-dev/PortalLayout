"""Non-training acceptance of baseline interfaces and experiment readiness."""
import tempfile
import numpy as np
import unittest
from pathlib import Path
import torch
from portallayout.models.opening_pair_matcher import OpeningPairView, assignment_loss
from portallayout.models.legacy_floor_matcher import LegacyFloorMatcher, LegacyFloorConfig
from portallayout.models.opening_representation import OpeningRepresentationEncoder
from portallayout.utils.opening_pair_checkpoint import checkpoint_payload,save_checkpoint,load_checkpoint


class LegacyFloorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(123)
        self.model=LegacyFloorMatcher(LegacyFloorConfig(feature_dim=8,heads=2,hidden_dim=8)).eval()
    def view(self, count):
        return OpeningPairView(torch.randn(count,256),torch.randn(256,2,4),
            torch.tensor([[7,1],[3,5]][:count]).reshape(count,2),torch.ones(count,dtype=torch.bool),
            tuple('c'+str(i) for i in range(count)),8,feature_1d=torch.randn(8,8),opening_probability=torch.rand(8))
    def test_symmetry_padding_empty_and_cached_signals(self):
        a,b=self.view(2),self.view(1)
        before=a.opening_probability.clone()
        x=self.model(a,b);y=self.model(b,a)
        torch.testing.assert_close(x['scores'],y['scores'].T)
        torch.testing.assert_close(x['conditional_ab'],y['conditional_ba'])
        torch.testing.assert_close(a.opening_probability,before,rtol=0,atol=0)
        self.assertEqual(x['assignment_method'],'legacy_bidirectional_softmax')
        self.assertEqual(x['iterations'],0)
        a.valid[:]=False
        self.assertEqual(self.model(a,b)['status'],'no_candidates')
        self.assertEqual(self.model(self.view(0),b)['status'],'no_candidates')
    def test_permutation_and_gradient(self):
        a,b=self.view(2),self.view(2); original=self.model(a,b)
        swapped=self.view(2)
        for key in ('descriptors','intervals','valid'): setattr(swapped,key,getattr(a,key).flip(0))
        swapped.feature_1d=a.feature_1d;swapped.opening_probability=a.opening_probability
        swapped.candidate_ids=a.candidate_ids[::-1]
        torch.testing.assert_close(original['scores'].flip(0),self.model(swapped,b)['scores'])
        loss=assignment_loss(original,[0,-1],[0,-1]);loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.model.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.core.opening_head.parameters()))
    def test_checkpoint_architecture_dispatch(self):
        encoder=OpeningRepresentationEncoder()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.pt'
            save_checkpoint(path,checkpoint_payload(encoder,self.model,training_status='initialized_untrained'))
            _,matcher,state=load_checkpoint(path)
            self.assertIsInstance(matcher,LegacyFloorMatcher)
            self.assertEqual(state['matcher_type'],'legacy')
            for k,v in self.model.state_dict().items():torch.testing.assert_close(v,matcher.state_dict()[k],rtol=0,atol=0)

    def test_legacy_training_uses_frozen_encoder(self):
        from tests.test_opening_pair_pipeline import PairPipelineTest
        from portallayout.engine.floor_training import train_floor_pairs
        fixture=PairPipelineTest();fixture.setUp();self.addCleanup(fixture.tearDown)
        initialization=fixture.checkpoint()
        before=load_checkpoint(initialization)[0].state_dict()
        report=train_floor_pairs(fixture.cache,fixture.labels,
            {'fit':['fit_house'],'selection':[],'calibration':[]},fixture.root/'legacy',
            steps=2,batch_size=2,mode='smoke',initialization=initialization,matcher_type='legacy')
        encoder,matcher,_=load_checkpoint(fixture.root/'legacy/checkpoint.pt')
        self.assertIsInstance(matcher,LegacyFloorMatcher)
        self.assertTrue(report['encoder_frozen'])
        for key,value in before.items():torch.testing.assert_close(value,encoder.state_dict()[key],rtol=0,atol=0)


class StageContractTests(unittest.TestCase):
    def test_post_pgo_export_preserves_unknown_and_uses_predicted_transform(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from tests.test_floor_backend import fixture
        from portallayout.datasets.floor_room_supervision import build_post_pgo_room_supervision
        from portallayout.models.floor_auxiliary import FEATURE_VERSION
        from portallayout.utils.room_feature_cache import content_hash,fingerprint,write_json_atomic
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pair=root/'pair.pt';pair.write_bytes(b'fixture')
            source=root/'source';source.mkdir()
            data=dict(feature_version=FEATURE_VERSION,pair_checkpoint_sha256=content_hash(pair),
                      house_protocol={'fit':['h'],'selection':[],'calibration':[]},records=[],opening_pairs=[])
            data['fingerprint']=fingerprint(data);write_json_atomic(source/'labels.json',data)
            pred,gt,obs,_=fixture(2);gt['source_files']=[]
            for uid,row in gt['observations'].items():row.update(membership_confirmed=True,room_id=uid)
            pred['provenance']=dict(weights_hash='weights',source_snapshot='snapshot',calibration_status={'opening':True,'pose':True})
            predictions=root/'predictions';predictions.mkdir()
            write_json_atomic(predictions/'f.json',pred)
            write_json_atomic(predictions/'run_report.json',{'floors':[{'floor_uid':'f','result_file':str(predictions/'f.json')}]})
            cache=SimpleNamespace(root=root/'cache',snapshot='snapshot',floors=[{'floor_uid':'f','observation_order':['a','b']}],check_unchanged=lambda:0)
            bundle=SimpleNamespace(spec={'pair_checkpoint':str(pair)},weights_hash='weights',pose_model=torch.nn.Linear(1,1))
            feature=dict(raster=np.zeros((2,7,16,16)),visual=np.zeros((2,512)),geometry=np.ones((2,32)))
            with patch('portallayout.datasets.floor_room_supervision.FloorModelBundle',return_value=bundle), \
                 patch('portallayout.datasets.floor_room_supervision.encode_floor',return_value=(obs,{'a':None,'b':None},0)), \
                 patch('portallayout.datasets.floor_room_supervision.load_floor_ground_truth',return_value=gt), \
                 patch('portallayout.datasets.floor_room_supervision.auxiliary_features',return_value=feature):
                result=build_post_pgo_room_supervision(cache,root/'portallayout.datasets',source/'labels.json',predictions,{},root/'known')
                self.assertEqual(result['records'][0]['target'],0)
                self.assertEqual(result['room_export_status'],'labels_exported')
                np.testing.assert_allclose(result['records'][0]['T_B_to_A'],pred['components'][0]['poses']['b'])
                gt['observations']['b']['membership_confirmed']=False
                unknown=build_post_pgo_room_supervision(cache,root/'portallayout.datasets',source/'labels.json',predictions,{},root/'unknown')
                self.assertEqual(unknown['records'][0]['target'],-1)
                self.assertEqual(unknown['room_export_status'],'no_usable_localized_pairs')

    def test_pose_bootstrap_then_frozen_room_training(self):
        from portallayout.models.floor_auxiliary import FEATURE_VERSION
        from portallayout.engine.floor_training import train_auxiliaries,_state_hash
        from portallayout.engine.floor_layout_pipeline import FloorModelBundle
        from portallayout.utils.room_feature_cache import content_hash,fingerprint,write_json_atomic
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pair=root/'pair.pt';pair.write_bytes(b'fixture')
            rows=[];rng=np.random.RandomState(123)
            for task,dim in [('pose',1024),('room',512)]:
                for role in ('fit','selection'):
                    for target in (0,1):
                        uid=f'{task}_{role}_{target}';path=root/(uid+'.npz')
                        np.savez(path,raster=rng.randn(2,7,16,16).astype('float32'),
                                 visual=rng.randn(2,dim).astype('float32'),geometry=rng.randn(2,32).astype('float32'))
                        rows.append(dict(record_id=uid,task=task,role=role,target=target,house_id=role,
                                         feature_file=path.name,feature_sha256=content_hash(path)))
            data=dict(feature_version=FEATURE_VERSION,pair_checkpoint_sha256=content_hash(pair),
                      house_protocol={'fit':['fit'],'selection':['selection'],'calibration':[]},records=rows)
            data['fingerprint']=fingerprint(data);labels=root/'labels.json';write_json_atomic(labels,data)
            train_auxiliaries(labels,pair,root/'pose',steps=2,batch_size=2,mode='full',validation_interval=2,tasks=('pose',))
            initial=torch.load(root/'pose/checkpoint.pt')
            bundle=FloorModelBundle.__new__(FloorModelBundle);bundle.auxiliary_state=initial
            self.assertTrue(bundle.auxiliary_trained('pose'));self.assertFalse(bundle.auxiliary_trained('room'))
            from portallayout.models.floor_auxiliary import PoseHypothesisVerifier
            pose=PoseHypothesisVerifier();pose.load_state_dict(initial['pose_state'])
            data.pop('fingerprint');data.update(room_feature_source='predicted_pgo_components',room_pose_model_hash=_state_hash(pose))
            data['fingerprint']=fingerprint(data);write_json_atomic(labels,data)
            train_auxiliaries(labels,pair,root/'room',steps=2,batch_size=2,mode='full',validation_interval=2,
                              tasks=('room',),initialization=root/'pose/checkpoint.pt')
            final=torch.load(root/'room/checkpoint.pt')
            self.assertEqual(final['training_status'],'trained')
            for key,value in initial['pose_state'].items():torch.testing.assert_close(value,final['pose_state'][key],rtol=0,atol=0)

    def test_cohort_is_stable_and_house_isolated(self):
        from tools.prepare_floor_training_cohort import select_cohort
        protocol={'fit':['a','b'],'selection':['c'],'calibration':['d']}
        rows=[dict(house_id=h,floor_uid=h+str(i),observation_count=10) for h in 'abcd' for i in (1,2)]
        counts={'fit':2,'selection':1,'calibration':1}
        a=select_cohort(rows,protocol,counts);b=select_cohort(list(reversed(rows)),protocol,counts)
        for role in counts:self.assertEqual([r['floor_uid'] for _,r in a[role]],[r['floor_uid'] for _,r in b[role]])

if __name__=='__main__':unittest.main()
