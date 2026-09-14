import copy,json,unittest
from pathlib import Path
from dataset.zind_matching_feature_dataset import ZInDMatchingFeatureDataset
from tools.prepare_zind_observable_training import all_candidates

CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/zind_observable_pairs_v1.json').read_text())

class ObservableTrainingTests(unittest.TestCase):
    def test_explicit_multi_gt_set_is_not_remined_with_legacy_threshold(self):
        dataset=object.__new__(ZInDMatchingFeatureDataset)
        record={'id':'fixture','candidates_a':[{},{}],'candidates_b':[{},{}],
            'supervision':{'is_match':True,'target_candidate_a':1,'target_candidate_b':0,
                           'correspondence_pairs':[[0,1],[1,0]]}}
        self.assertEqual(dataset._matching_candidate_pairs(record),((0,1),(1,0)))
        record['supervision']['is_match']=False
        with self.assertRaises(ValueError):dataset._matching_candidate_pairs(record)
        record['supervision']['correspondence_pairs']=[]
        self.assertEqual(dataset._matching_candidate_pairs(record),())

    def test_union_has_real_and_virtual_openings_and_is_image_specific(self):
        raw={'floor_plan_transformation':{'translation':[0,0],'rotation':0,'scale':1},
            'layout_complete':{'doors':[[0,1],[1,1],[-1,1]],'openings':[],'windows':[]},
            'layout_raw':{'doors':[[0,1],[1,1],[-1,1]],'openings':[[-.1,-1],[.1,-1],[-1,1]],'windows':[[0,1]]}}
        candidates,mapping=all_candidates(raw,'pano/ref',CFG,1)
        self.assertEqual(len(candidates),2);self.assertEqual({c['type'] for c in candidates},{'door','opening'})
        self.assertEqual(mapping['pano/ref/layout_raw/doors/0'],mapping['pano/ref/layout_complete/doors/0'])
        seam=next(c for c in candidates if c['type']=='opening');self.assertGreater(seam['tokenInterval'][0],seam['tokenInterval'][1])
        self.assertEqual((candidates,mapping),all_candidates(copy.deepcopy(raw),'pano/ref',CFG,1))

    def test_invalid_door_annotation_is_not_silently_dropped(self):
        raw={'floor_plan_transformation':{'translation':[0,0],'rotation':0,'scale':1},
            'layout_complete':{'doors':[[0,1]],'openings':[],'windows':[]},'layout_raw':{}}
        with self.assertRaises(ValueError):all_candidates(raw,'pano/ref',CFG,1)

if __name__=='__main__':unittest.main()
