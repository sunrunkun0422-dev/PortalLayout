import argparse
import unittest

import torch
import torch.nn as nn

from tools.train_zind_end_to_end import (
    EndToEndOpeningMatcher,
    _optimization_loss,
    _optimizer,
    configure_training_stage,
    matcher_without_opening_parameters,
)
from tools.preflight_zind_t4_matcher_only import (
    _conditional_margin_gate,
    _protocol_v2,
)


class _TinyMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.opening_head = nn.Linear(3, 3)
        self.query_projection = nn.Linear(3, 3)
        self.dropout = nn.Dropout(0.5)
        self.dustbin_score = nn.Parameter(torch.tensor(0.0))


def _model():
    return EndToEndOpeningMatcher(
        nn.Sequential(nn.Linear(3, 3), nn.Dropout(0.2)),
        _TinyMatcher(),
        activation_checkpointing=False,
    )


class MatcherOnlyStageTest(unittest.TestCase):
    def test_stage_freezes_upstream_and_disables_diagnostic_dropout(self):
        model = _model()
        configure_training_stage(
            model,
            "matcher_only_finetune",
            deterministic_matcher_dropout=True,
        )
        self.assertTrue(model.training)
        self.assertFalse(model.bi_layout.training)
        self.assertFalse(model.matcher.opening_head.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.bi_layout.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.matcher.opening_head.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in matcher_without_opening_parameters(model)
            )
        )
        self.assertFalse(model.matcher.dropout.training)

    def test_optimizer_excludes_bi_layout_and_opening_head(self):
        model = _model()
        configure_training_stage(model, "matcher_only_finetune")
        args = argparse.Namespace(
            training_stage="matcher_only_finetune",
            matcher_lr=3e-5,
            bi_layout_lr=2e-6,
            opening_lr=1e-5,
            weight_decay=1e-4,
        )
        optimizer = _optimizer(model, args)
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["name"], "matcher")
        self.assertTrue(
            all(
                id(parameter) not in optimizer_ids
                for parameter in model.bi_layout.parameters()
            )
        )
        self.assertTrue(
            all(
                id(parameter) not in optimizer_ids
                for parameter in model.matcher.opening_head.parameters()
            )
        )
        self.assertEqual(
            optimizer_ids,
            {
                id(parameter)
                for parameter in matcher_without_opening_parameters(model)
            },
        )

    def test_backward_uses_matching_optim_loss_not_reported_total(self):
        matcher_value = torch.tensor(2.0, requires_grad=True)
        diagnostic_value = torch.tensor(7.0, requires_grad=True)
        args = argparse.Namespace(training_stage="matcher_only_finetune")
        selected = _optimization_loss(
            {
                "matching_optim_loss": matcher_value,
                "reported_end_to_end_loss": diagnostic_value,
            },
            args,
        )
        selected.backward()
        self.assertEqual(matcher_value.grad.item(), 1.0)
        self.assertIsNone(diagnostic_value.grad)

    def test_v2_conditional_margin_gate_handles_zero_initial_loss(self):
        initial = {
            "loss_positive_dustbin_margin": 0.0,
            "true_to_bidirectional_dustbin_margin": 2.0,
        }
        final = {
            "loss_positive_dustbin_margin": 0.0,
            "true_to_bidirectional_dustbin_margin": 2.5,
        }
        result = _conditional_margin_gate(initial, final)
        self.assertTrue(result["pass"])
        self.assertEqual(
            result["branch"],
            "zero_margin_loss_stays_zero_and_true_margin_nondecreases",
        )

    def test_v2_protocol_freezes_original_optimizer_and_step_budget(self):
        args = argparse.Namespace(
            seed=123,
            overfit_lr=3e-4,
            grad_clip=1.0,
            single_batch_steps=200,
        )
        protocol = _protocol_v2(args)
        immutable = protocol["immutable_training_protocol"]
        self.assertEqual(immutable["seed"], 123)
        self.assertEqual(immutable["optimizer"], "AdamW")
        self.assertEqual(immutable["lr"], 3e-4)
        self.assertEqual(immutable["weight_decay"], 0.0)
        self.assertEqual(immutable["grad_clip"], 1.0)
        self.assertEqual(immutable["steps"], 200)
        self.assertEqual(len(protocol["protocol_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
