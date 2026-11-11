import copy
from dataclasses import replace
import unittest

import torch

from portallayout.models.opening_pair_matcher import (OpeningPairMatcher, OpeningPairConfig, OpeningPairView,
    log_partial_transport, assignment_loss, mutual_matches)


class PairMatcherTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = OpeningPairMatcher(OpeningPairConfig(dim=16, heads=4, chunk_size=2)).eval()

    def view(self, k=3):
        return OpeningPairView(torch.randn(k, 16), torch.randn(16, 4, 8),
            torch.tensor([[30, 2], [8, 12], [18, 22]])[:k], torch.ones(k, dtype=torch.bool),
            tuple('candidate_'+str(i) for i in range(k)), 32, torch.randn(k, 12), torch.ones(k, 12, dtype=torch.bool))

    def test_swap_is_transpose_for_scores_and_transport(self):
        a, b = self.view(), self.view(2)
        x, y = self.model(a, b), self.model(b, a)
        torch.testing.assert_close(x['scores'], y['scores'].T)
        torch.testing.assert_close(x['log_transport'], y['log_transport'].T)
        self.assertTrue(x['converged'])

    def test_candidate_permutation(self):
        a, b = self.view(), self.view(2); order = torch.tensor([2, 0, 1])
        perm = replace(a, descriptors=a.descriptors[order], intervals=a.intervals[order],
            candidate_ids=tuple(a.candidate_ids[i] for i in order), geometry=a.geometry[order],
            geometry_valid=a.geometry_valid[order])
        original, changed = self.model(a, b), self.model(perm, b)
        torch.testing.assert_close(original['scores'][order], changed['scores'])
        torch.testing.assert_close(original['conditional_ab'][order], changed['conditional_ab'])

    def test_target_intervention_changes_query_and_retrieval(self):
        a, b = self.view(), self.view(2)
        x = self.model(a, b, diagnostics=True)['diagnostics']
        changed = replace(b, descriptors=b.descriptors.clone()); changed.descriptors[0] += 2
        y = self.model(a, changed, diagnostics=True)['diagnostics']
        self.assertGreater(float((x['queries_ab'][:, 0]-y['queries_ab'][:, 0]).abs().max()), 1e-4)
        self.assertGreater(float((x['attention_ab'][:, 0]-y['attention_ab'][:, 0]).abs().max()), 1e-6)
        torch.testing.assert_close(x['attention_ab'][:, 1], y['attention_ab'][:, 1])

    def test_target_interval_changes_attention(self):
        a, b = self.view(), self.view(2)
        changed = replace(b, intervals=b.intervals.clone()); changed.intervals[0] = torch.tensor([12, 16])
        x = self.model(a, b, True)['diagnostics']['attention_ab']
        y = self.model(a, changed, True)['diagnostics']['attention_ab']
        self.assertGreater(float((x[:, 0]-y[:, 0]).abs().max()), 1e-4)

    def test_circle_roll_invariance(self):
        a, b = self.view(), self.view(2)
        rolled = replace(b, memory=torch.roll(b.memory, 1, -1), intervals=(b.intervals+4) % 32)
        torch.testing.assert_close(self.model(a, b)['scores'], self.model(a, rolled)['scores'], atol=2e-6, rtol=2e-5)

    def test_empty_and_all_padding(self):
        a, b = self.view(), self.view(0)
        output = self.model(a, b)
        self.assertEqual(output['status'], 'no_candidates')
        self.assertEqual(output['conditional_ab'].tolist(), [[1.], [1.], [1.]])
        self.assertEqual(mutual_matches(output, 0), [])
        self.assertIsNone(assignment_loss(output, [-1]*3, []))
        masked = replace(a, valid=torch.zeros(3, dtype=torch.bool))
        self.assertEqual(self.model(masked, a)['status'], 'no_candidates')

    def test_padding_is_excluded_and_identity_retained(self):
        a, b = self.view(), self.view(2)
        a.valid[1] = False; a.descriptors[1] = torch.nan
        result = self.model(a, b)
        self.assertEqual(result['indices_a'].tolist(), [0, 2])
        self.assertEqual(result['scores'].shape, (2, 2))
        self.assertEqual(result['candidate_ids_a'], a.candidate_ids)

    def test_missing_geometry_and_backprop(self):
        a, b = self.view(), self.view(2)
        a.geometry[:] = torch.nan; a.geometry_valid[:] = False
        result = self.model(a, b)
        loss = assignment_loss(result, [0, 1, -1], [0, 1]); loss.backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(float(self.model.query[0].weight.grad.abs().sum()), 0)

    def test_unknown_loss_and_nonconverged_skip(self):
        result = self.model(self.view(), self.view(2))
        self.assertIsNone(assignment_loss(result, [-2]*3, [-2]*2))
        result['status'] = 'numerical_unconverged'
        self.assertIsNone(assignment_loss(result, [0, 1, -1], [0, 1]))
        self.assertEqual(mutual_matches(result, 0), [])

    def test_extreme_all_unmatched_adapts_and_backpropagates(self):
        scores = torch.full((3, 7), -20., requires_grad=True)
        result = log_partial_transport(scores, torch.tensor(0., requires_grad=True))
        self.assertTrue(result['converged'])
        self.assertGreater(result['iterations'], 50)
        self.assertLessEqual(result['marginal_error'], 1e-3)
        unmatched = (result['log_transport'][:-1, -1]-result['mu'][:-1].log()).exp()
        self.assertTrue((unmatched > .99).all())
        loss = -result['log_transport'][:-1, -1].sum(); loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_sinkhorn_iteration_limit_is_explicit(self):
        result = log_partial_transport(torch.full((3, 7), -20.), torch.tensor(0.), checks=(1,))
        self.assertFalse(result['converged']); self.assertEqual(result['iterations'], 1)

    def test_chunk_size_does_not_change_results(self):
        a, b = self.view(), self.view(2)
        other = OpeningPairMatcher(replace(self.model.config, chunk_size=20))
        other.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(self.model(a, b)['scores'], other(a, b)['scores'])

    def test_duplicate_ids_rejected(self):
        a = replace(self.view(2), candidate_ids=('same', 'same'))
        with self.assertRaisesRegex(ValueError, 'IDs'): self.model(a, self.view())


if __name__ == '__main__': unittest.main()
