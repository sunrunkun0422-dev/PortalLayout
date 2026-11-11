"""Partial labels, task-based selection, and nontrivial calibration gates."""
import copy
import json
import unittest
from unittest.mock import patch
import torch
from portallayout.engine.opening_pair_training import pair_objective, sampling_pools
from portallayout.evaluation.opening_matching import average_precision, quality_metrics, score_record
from portallayout.engine.floor_calibration import strict_opening_workpoint


def output(scores, dustbin=.8):
    s = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
    b = torch.tensor(dustbin, requires_grad=True)
    m, n = s.shape
    logits = torch.cat((torch.cat((s, b.expand(m, 1)), 1), b.expand(1, n+1)), 0)
    lab, lba = logits[:-1].log_softmax(-1), logits.T[:-1].log_softmax(-1)
    return dict(status='assigned', scores=s, dustbin_score=b, indices_a=torch.arange(m), indices_b=torch.arange(n),
                candidate_ids_a=['a'+str(i) for i in range(m)], candidate_ids_b=['b'+str(j) for j in range(n)],
                log_conditional_ab=lab, log_conditional_ba=lba, conditional_ab=lab.exp(), conditional_ba=lba.exp(),
                iterations=0, marginal_error=0.)


def label(cells):
    m, n = len(cells), len(cells[0])
    return dict(weak_cells=cells, target_a=[-2]*m, target_b=[-2]*n,
                candidate_ids_a=['a'+str(i) for i in range(m)], candidate_ids_b=['b'+str(j) for j in range(n)])


class PartialObjectiveTests(unittest.TestCase):
    def test_partial_positive_gets_gradient_unknown_does_not(self):
        for objective in ('positive_margin', 'known_cells'):
            o = output([[.1, .3]]); r = label([[1, -1]])
            loss, parts = pair_objective(o, r, objective)
            self.assertIsNone(parts['assignment']); loss.backward()
            self.assertLess(o['scores'].grad[0, 0], 0)
            self.assertEqual(o['scores'].grad[0, 1], 0)
            self.assertGreater(o['dustbin_score'].grad, 0)

    def test_negative_and_positive_classes_are_balanced(self):
        o = output([[.1, .1, .1, .4]]); r = label([[1, 0, 0, -1]])
        loss, _ = pair_objective(o, r, 'known_cells'); loss.backward()
        self.assertLess(o['scores'].grad[0, 0], 0)
        self.assertGreater(o['scores'].grad[0, 1], 0)
        self.assertEqual(o['scores'].grad[0, 3], 0)
        a, _ = pair_objective(output([[.1, .1]]), label([[1, 0]]), 'known_cells')
        torch.testing.assert_close(a, loss)

    def test_unknown_negative_margin_and_unconverged_skip(self):
        for objective in ('known_cells', 'positive_margin'):
            self.assertIsNone(pair_objective(output([[.1]]), label([[-1]]), objective)[0])
            o = output([[.1]]); o['status'] = 'numerical_unconverged'
            self.assertIsNone(pair_objective(o, label([[1]]), objective)[0])
        self.assertIsNone(pair_objective(output([[.1]]), label([[0]]), 'positive_margin')[0])
        self.assertIsNotNone(pair_objective(output([[.1]]), label([[0]]), 'known_cells')[0])

    def test_padding_original_indices_permutation_and_swap(self):
        r = label([[-1,-1,-1],[-1,1,0],[-1,-1,-1]])
        o = output([[.1, .2]]); o['indices_a']=torch.tensor([1]);o['indices_b']=torch.tensor([1,2])
        for objective in ('positive_margin','known_cells'):
            loss,_=pair_objective(o,r,objective)
            perm=output([[.2,.1]]);perm['indices_a']=torch.tensor([1]);perm['indices_b']=torch.tensor([2,1])
            torch.testing.assert_close(loss,pair_objective(perm,r,objective)[0])
            swapped=output([[.1],[.2]]);swapped['indices_a']=torch.tensor([1,2]);swapped['indices_b']=torch.tensor([1])
            rr=label([list(x) for x in zip(*r['weak_cells'])])
            torch.testing.assert_close(loss,pair_objective(swapped,rr,objective)[0])

    def test_pool_keeps_partial_positive_and_reclassifies_mixed_pair(self):
        pos=label([[1,-1]]);pos['pair_id']='p'
        mixed=label([[1,-1],[0,0]]);mixed['target_a']=[-2,-1];mixed['pair_id']='mixed'
        neg=label([[0,-1]]);neg['target_b']=[-1,-2];neg['pair_id']='n'
        unknown=label([[-1,-1]]);unknown['pair_id']='u'
        p,n=sampling_pools([pos,mixed,neg,unknown],'known_cells')
        self.assertEqual([r['pair_id'] for r in p],['p','mixed']);self.assertEqual([r['pair_id'] for r in n],['n'])


class OpeningQualityTests(unittest.TestCase):
    def row(self, positives, matches, house='h', scope='cross_room'):
        return dict(positive_cells=positives,matches=[dict(target=t,confidence=p) for t,p in matches],
                    house_id=house,relationship=scope,known_negative_pair=positives==0,
                    numerical_status='assigned',role='calibration')

    def test_ap_ties_missing_positives_unknown_and_house_macro(self):
        rows=[self.row(2,[(1,.8),(0,.8),(-1,.99)])]
        self.assertEqual(average_precision(rows),.25)
        rows[0]['matches'].reverse();self.assertEqual(average_precision(rows),.25)
        rows+=[self.row(1,[(1,.9)],'other'),self.row(0,[(0,.9)],'empty')]
        quality=quality_metrics(rows)
        self.assertEqual(quality['cross_room_house_macro_ap'],.625)
        self.assertIn('empty',quality['zero_positive_house_false_acceptance'])

    def test_perfect_calibration_and_weak_support_rejection(self):
        rows=[self.row(1,[(1,.9)],str(i%5)) for i in range(25)]
        rows += [self.row(0,[],str(i%5)) for i in range(100)]
        report=strict_opening_workpoint(rows)
        self.assertEqual(report['status'],'calibrated');self.assertEqual(report['cross_room']['recall_all_weak_positives'],1.)
        self.assertEqual(strict_opening_workpoint(rows[:1]+rows[25:])['status'],'calibration_infeasible')
        low=copy.deepcopy(rows)
        for r in low[:25]:r['positive_cells']=100
        self.assertEqual(strict_opening_workpoint(low)['status'],'calibration_infeasible')
        same=copy.deepcopy(rows)+[self.row(0,[(0,.99)],'h','same_room') for _ in range(10)]
        self.assertEqual(strict_opening_workpoint(same)['status'],'calibration_infeasible')

    def test_empty_and_failed_keep_positive_denominator(self):
        r=dict(label([[1]]),pair_id='p',floor_uid='f',house_id='h',observation_a='a',observation_b='b')
        record=score_record({'status':'no_candidates'},r,'cross_room')
        self.assertEqual(average_precision([record]),0.)
        self.assertEqual(record['positive_cells'],1)

    def test_objective_resume_exact_and_contract_mismatch(self):
        from tests.test_opening_pair_pipeline import PairPipelineTest
        from portallayout.engine.floor_training import train_floor_pairs
        from portallayout.utils.opening_pair_checkpoint import load_checkpoint
        from portallayout.utils.room_feature_cache import fingerprint,write_json_atomic
        fixture=PairPipelineTest();fixture.setUp();self.addCleanup(fixture.tearDown)
        torch.set_num_threads(2)
        data=json.loads(fixture.labels.read_text());data.pop('fingerprint')
        for i,r in enumerate(data['pairs']):
            r['weak_cells']=[[1 if r['target_a'][0]>=0 else 0]];r['floor_uid']='f'
            if i==2:r['house_id']='dev_house'
        data['pairs'][2]['weak_cells']=[[1]];data['pairs'][2]['target_a']=[0];data['pairs'][2]['target_b']=[0]
        data['fingerprint']=fingerprint(data);write_json_atomic(fixture.labels,data)
        initial=fixture.checkpoint();protocol=dict(fit=['fit_house'],selection=['dev_house'],calibration=[])
        kwargs=dict(mode='smoke',validation_interval=1,initialization=initial,objective='known_cells',
                    selection_metric='cross_room_ap',dataset_root=fixture.root)
        scopes={r['pair_id']:'cross_room' for r in data['pairs']}
        with patch('portallayout.evaluation.opening_matching.room_scopes',return_value=(scopes,{})):
            train_floor_pairs(fixture.cache,fixture.labels,protocol,fixture.root/'split',steps=2,**kwargs)
            train_floor_pairs(fixture.cache,fixture.labels,protocol,fixture.root/'split',steps=4,
                              resume=fixture.root/'split/last.pt',**kwargs)
            train_floor_pairs(fixture.cache,fixture.labels,protocol,fixture.root/'full',steps=4,**kwargs)
            a=load_checkpoint(fixture.root/'split/last.pt')[2];b=load_checkpoint(fixture.root/'full/last.pt')[2]
            self.assertEqual(a['weights_hash'],b['weights_hash'])
            changed=dict(kwargs,objective='positive_margin')
            with self.assertRaisesRegex(ValueError,'contract differs'):
                train_floor_pairs(fixture.cache,fixture.labels,protocol,fixture.root/'split',steps=5,
                                  resume=fixture.root/'split/last.pt',**changed)


class HumanReviewTests(unittest.TestCase):
    def test_house_cap_role_isolation_and_determinism(self):
        from tools.review_floor_opening_matches import select_cases
        rows=[dict(case_id=str(i),house_id=str(i%8),role='fit' if i%8<4 else 'selection',confidence=i/100.)
              for i in range(80)]
        first=select_cases(rows);second=select_cases(rows[::-1])
        self.assertEqual(first,second);self.assertEqual(len(first),24)
        from collections import Counter
        self.assertTrue(max(Counter(r['house_id'] for r in first).values())<=4)
        self.assertEqual(Counter(r['role'] for r in first),{'fit':12,'selection':12})

    def test_human_review_requires_identity_and_auditor(self):
        from tools.review_floor_opening_matches import summarize_review
        package=dict(fingerprint='p',cases=[dict(case_id='c',group='g')])
        review=dict(package_fingerprint='p',reviewer='',reviewed_at='',cases=[dict(case_id='c',judgement=None)])
        self.assertFalse(summarize_review(package,review)['training_labels_overridden'])
        review['cases'][0]['judgement']='正确'
        with self.assertRaisesRegex(ValueError,'reviewer'):summarize_review(package,review)
        review.update(reviewer='auditor',reviewed_at='2026-10-24')
        self.assertEqual(summarize_review(package,review)['counts']['g']['正确'],1)
        review['package_fingerprint']='wrong'
        with self.assertRaisesRegex(ValueError,'mismatch'):summarize_review(package,review)


if __name__=='__main__':unittest.main()
