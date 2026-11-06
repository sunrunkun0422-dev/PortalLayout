import inspect
import json
import unittest
from pathlib import Path
import numpy as np
import torch
from tools.validate_type_conditioned_prior import PriorAdapter, labels, mask, load_matcher
from tools.type_prior_probe_evaluation import pair_eval, predict
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/type_prior_probe_20260923'

class TypePriorProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        # This historical integration test needs local ZInD prediction caches
        # and model weights, which are intentionally not shipped with Git.
        required = [OUT/'selection.json', OUT/'cache_run.json', OUT/'protocol.json', OUT/'predictions']
        if any(not p.exists() for p in required):
            raise unittest.SkipTest('historical type-prior cache and weights are not available locally')
        cls.sel=json.loads((OUT/'selection.json').read_text())
        if any(not (OUT/'predictions'/(v['cache_key']+'.pt')).is_file() for v in cls.sel['views'].values()):
            raise unittest.SkipTest('historical type-prior prediction cache is incomplete')
        protocol=json.loads((OUT/'protocol.json').read_text())
        if not Path(protocol['checkpoint']).is_file():
            raise unittest.SkipTest('historical type-prior checkpoint is not available locally')
        cls.data={k:torch.load(OUT/'predictions'/(v['cache_key']+'.pt'),map_location='cpu') for k,v in cls.sel['views'].items()}
    def test_prediction_api_and_cache_exclude_labels(self):
        self.assertEqual(list(inspect.signature(PriorAdapter.forward).parameters),['self','x','h','old_logits'])
        for d in self.data.values():
            self.assertFalse(set(d)&{'type','binary','candidates','gt','supervision'})
            self.assertTrue(all(not v.requires_grad for v in d.values() if torch.is_tensor(v)))
    def test_circular_overlap_and_unknown_are_ignored(self):
        y,t=labels([dict(type='door',tokenInterval=[254,2]),dict(type='opening',tokenInterval=[1,4]),dict(type='unknown',tokenInterval=[8,8])])
        self.assertEqual(int(y.sum()),8)
        self.assertEqual(t[[254,255,0]].tolist(),[0,0,0])
        self.assertEqual(t[[1,2,8]].tolist(),[-100,-100,-100])
        self.assertEqual(t[[3,4,9]].tolist(),[1,1,2])
    def test_initialization_reproduces_real_baseline(self):
        for group in ['A0','A1','A2','A3']:
            m=PriorAdapter(group)
            for d in list(self.data.values())[:8]:
                z,_,w=m(d['x'],d['h'],d['logits'])
                self.assertTrue(torch.allclose(z,d['logits'],atol=1e-6,rtol=0))
                self.assertTrue(torch.allclose(w,torch.full_like(w,4),atol=1e-6,rtol=0))
    def test_house_and_panorama_isolation(self):
        groups={g:{v['house'] for v in self.sel['views'].values() if v['group']==g} for g in ['fit','calibration','monitor']}
        self.assertFalse(groups['fit']&groups['calibration']);self.assertFalse(groups['fit']&groups['monitor']);self.assertFalse(groups['monitor']&groups['calibration'])
    def test_frozen_buffers_and_gradient_contract(self):
        r=json.loads((OUT/'cache_run.json').read_text())
        self.assertTrue(r['frozen_no_grad']);self.assertTrue(r['buffers_unchanged'])
        protocol=json.loads((OUT/'protocol.json').read_text());_,matcher=load_matcher(protocol['checkpoint'])
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in matcher.parameters()))
    def test_guidance_switch_uses_only_predictions(self):
        pair=self.sel['pairs']['monitor'][:1];keys=[pair[0]['image_'+s] for s in ['a','b']]
        pr={k:dict(p=np.full(256,.75,np.float32)) for k in keys}
        class Recorder:
            def __init__(self):self.kwargs=None
            def __call__(self,*args,**kwargs):
                self.kwargs=kwargs
                return dict(best_candidate_pair=torch.tensor([[-1,-1]]))
        old=Recorder();new=Recorder()
        pair_eval(pr,pair,self.data,old,.47,'candidate_only')
        pair_eval(pr,pair,self.data,new,.47,'normal_guidance')
        for i,s in enumerate(['a','b']):
            self.assertTrue(torch.equal(old.kwargs['candidate_masks_'+s],new.kwargs['candidate_masks_'+s]))
            self.assertTrue(torch.equal(old.kwargs['opening_guidance_'+s],self.data[keys[i]]['logits'].sigmoid()[None]))
            self.assertTrue(torch.equal(new.kwargs['opening_guidance_'+s],torch.tensor(pr[keys[i]]['p'])[None]))
    def test_expansion_zero_delta_stays_zero(self):
        pred=predict(PriorAdapter('A3'),self.sel['views'],self.data)
        for k,d in self.data.items():
            zero=d['delta'].numpy()==0
            self.assertTrue((pred[k]['expansion'][zero]==0).all())
            self.assertTrue(np.allclose(pred[k]['expansion'],d['expansion'].numpy(),atol=1e-6))
if __name__=='__main__':unittest.main()
