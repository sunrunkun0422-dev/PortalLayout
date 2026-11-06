"""Synthetic unit fixtures only: never exported as dataset human evidence."""
import copy,json,unittest,subprocess,sys,tempfile,hashlib
from pathlib import Path
from tools.import_zind_observable_reviews import evaluate_correspondence_sets,validate_answer,consensus,apply_adjudication
from tools.summarize_zind_observable_review import summarize
ROOT=Path(__file__).resolve().parents[1]
POLICY=json.loads((ROOT/'configs/zind_observable_review_v1.json').read_text())

def fixture():
    p={'start_u':.97,'length_u':.08,'top_v':[.2,.2],'bottom_v':[.8,.8]}
    t={'opening_ids':['a','b'],'A':{'projection':p},'B':{'projection':p}}
    region={'start_u':.97,'length_u':.08,'y0':.2,'y1':.8}
    f={'id':'blind','decision':'match','correspondences':[{'A':region,'B':region}],
       'evidence':'synthetic unit fixture','timestamp':'2026-09-10T00:00:00Z'}
    g={'id':'blind','adjacency':'positive','identifiability':'clear_evidence','annotation_checked':True,
       'visibility_targets':[{'opening_ids':['a','b'],'A':'clear','B':'clear'}],
       'evidence':'synthetic unit fixture','timestamp':'2026-09-10T01:00:00Z'}
    r={'id':'master','labels':{'adjacency':'positive','identifiability':'unreviewed','review_state':'pending_review'},
       'quality':{'negative_certificate':None},'gt':{'correspondences':[t]}}
    reviews=[{'reviewer_id':rid,'stage1':copy.deepcopy(f),'stage2':copy.deepcopy(g),
              'set_evaluation':evaluate_correspondence_sets(f['correspondences'],[t],POLICY)} for rid in ['unit_A','unit_B']]
    return r,reviews

class ReviewTests(unittest.TestCase):
    def test_cli_empty_review_snapshot_and_rejection_of_synthetic_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);build=base/'build';build.mkdir();r,_=fixture()
            r['inputs']={'image_a':'a.jpg','image_b':'b.jpg','input_resolution':[1024,512]}
            raw=(json.dumps(r)+'\n').encode();(build/'main_index.jsonl').write_bytes(raw);dataset_id=hashlib.sha256(raw).hexdigest()
            mapping=base/'mapping.json';mapping.write_text(json.dumps({'dataset_id':dataset_id,'blind_to_master_id':{'blind':'master'}}))
            cmd=[sys.executable,'-B',str(ROOT/'tools/import_zind_observable_reviews.py'),'--build',str(build),'--mapping',str(mapping)]
            out=base/'empty';p=subprocess.run(cmd+['--output',str(out)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads((out/'views/core_verified.json').read_text())['ids'],[])
            self.assertEqual(json.loads((out/'main_index.jsonl').read_text())['labels']['review_state'],'pending_review')
            f=base/'synthetic.json';f.write_text(json.dumps({'stage':1,'dataset_id':dataset_id,'synthetic_for_test':True,'reviewer_declares_independent_human_review':True,'reviewer_id':'synthetic','answers':[]}))
            rejected=base/'rejected';p=subprocess.run(cmd+['--stage1',str(f),'--output',str(rejected)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertFalse(rejected.exists())
    def test_weights_and_nonresponse_do_not_fabricate_population_estimate(self):
        samples=[{'id':'a','weight':10},{'id':'b','weight':90}]
        records={'a':{'labels':{'review_state':'verified_consensus','adjacency':'positive','identifiability':'clear_evidence','visibility':{'targets':[{'A':'clear','B':'clear'}]}}}}
        self.assertFalse(summarize(samples,records)['population_estimate_available'])
        records['b']={'labels':{'review_state':'verified_consensus','adjacency':'negative','identifiability':'clear_evidence','visibility':{'targets':[]}}}
        report=summarize(samples,records);self.assertTrue(report['population_estimate_available'])
        self.assertEqual(report['hajek_adjacency_proportions']['positive'],.1)
    def test_no_review_or_one_reviewer_never_core(self):
        r,reviews=fixture();rr,core=consensus(r,[],POLICY);self.assertFalse(core);self.assertEqual(rr['labels']['review_state'],'pending_review')
        self.assertFalse(consensus(r,reviews[:1],POLICY)[1])
        self.assertFalse(consensus(r,[reviews[0],reviews[0]],POLICY)[1])
    def test_independent_correct_blind_sets_and_clear_gt_required(self):
        r,reviews=fixture();self.assertTrue(consensus(r,reviews,POLICY)[1])
        reviews[1]['set_evaluation']['exact_set']=False;self.assertFalse(consensus(r,reviews,POLICY)[1])
        for review in reviews:review['stage2']['visibility_targets'][0]['A']='partial'
        self.assertFalse(consensus(r,reviews,POLICY)[1])
        self.assertEqual(r['labels']['identifiability'],'unreviewed')
    def test_disagreement_and_explicit_adjudication_preserve_blind_evidence(self):
        r,reviews=fixture();reviews[1]['stage2']['identifiability']='multiple_plausible'
        rr,core=consensus(r,reviews,POLICY);self.assertFalse(core);self.assertEqual(rr['labels']['review_state'],'needs_adjudication')
        item={'reviewer_id':'unit_C','rationale':'synthetic fixture','resolution':copy.deepcopy(reviews[1]['stage2'])}
        fixed,core=apply_adjudication(rr,item,reviews,POLICY);self.assertFalse(core)
        self.assertEqual(fixed['labels']['identifiability'],'multiple_plausible');self.assertEqual(fixed['reviews'],reviews)
        item['reviewer_id']='unit_A'
        with self.assertRaises(ValueError):apply_adjudication(rr,item,reviews,POLICY)
    def test_another_true_opening_is_correct_and_duplicates_do_not_cover_set(self):
        r,reviews=fixture();t=copy.deepcopy(r['gt']['correspondences'][0]);t['opening_ids']=['c','d']
        for s in ['A','B']:t[s]['projection']['start_u']=.4
        selected=copy.deepcopy(reviews[0]['stage1']['correspondences']);q=copy.deepcopy(selected[0])
        for s in ['A','B']:q[s]['start_u']=.4
        truth=r['gt']['correspondences']+[t]
        ev=evaluate_correspondence_sets([q],truth,POLICY);self.assertEqual(ev['precision'],1);self.assertEqual(ev['recall'],.5)
        self.assertTrue(evaluate_correspondence_sets([q]+selected,truth,POLICY)['exact_set'])
        self.assertFalse(evaluate_correspondence_sets(selected*2,truth,POLICY)['exact_set'])
    def test_wrong_vertical_region_and_malformed_confirmation(self):
        r,reviews=fixture();f=reviews[0]['stage1'];f['correspondences'][0]['A']['y0']=.85;f['correspondences'][0]['A']['y1']=.95
        self.assertEqual(evaluate_correspondence_sets(f['correspondences'],r['gt']['correspondences'],POLICY)['matched'],0)
        g=reviews[0]['stage2'];g['annotation_checked']='false'
        with self.assertRaises(ValueError):validate_answer(g,2)
    def test_nonadjacent_requires_annotation_certificate_and_two_blind_no_matches(self):
        r,reviews=fixture();r['labels']['adjacency']='negative';r['gt']['correspondences']=[]
        for review in reviews:
            review['stage1'].update(decision='no_match',correspondences=[])
            review['stage2'].update(adjacency='negative',visibility_targets=[])
        self.assertFalse(consensus(r,reviews,POLICY)[1])
        r['quality']['negative_certificate']={'clearance_m':1};self.assertTrue(consensus(r,reviews,POLICY)[1])
    def test_unfrozen_val_and_test_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            partition=Path(tmp)/'partition.json'
            partition.write_text(json.dumps({'train':['0001'],'val':['0002'],'test':['0003']}))
            for split in ['val','test']:
                out=Path(tmp)/split
                p=subprocess.run([sys.executable,'-B',str(ROOT/'tools/build_zind_observable_pairs.py'),'--partition',str(partition),'--split',split,'--output',str(out)],capture_output=True,text=True)
                self.assertNotEqual(p.returncode,0);self.assertIn('train-only calibration freeze record',p.stderr);self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
