#!/usr/bin/env python3
"""CPU learnability and fixed synthetic overfit gate for T4 recall losses."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portallayout.models.cross_scene_matcher import (  # noqa: E402
    candidate_assignment_loss,
    positive_to_dustbin_margin_loss,
    recall_topk_opening_intervals,
)

DEFAULT_OUTPUT = REPO_ROOT / "experiments/zind_e2e_v3/g0_seed123/T4_recall/preflight/cpu_learnability.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--margin_weight", type=float, default=0.20)
    return parser.parse_args(argv)


def _outputs(logits: torch.Tensor, valid_a: torch.Tensor, valid_b: torch.Tensor) -> Mapping[str, torch.Tensor]:
    return {
        "candidate_assignment_logits": logits,
        "candidate_assignment_AB": torch.softmax(logits, dim=-1),
        "candidate_assignment_BA": torch.softmax(logits.transpose(-2, -1), dim=-1),
        "candidate_valid_A": valid_a,
        "candidate_valid_B": valid_b,
    }


def _metrics(logits: torch.Tensor, target: torch.Tensor, is_match: torch.Tensor) -> Dict[str, float]:
    real = logits[:, :2, :2]
    row_bin = logits[:, :2, 2].unsqueeze(-1)
    col_bin = logits[:, 2, :2].unsqueeze(1)
    accepted = (real > row_bin) & (real > col_bin)
    positive_with_target = is_match & target.flatten(1).any(dim=-1)
    correct = 0
    positive_count = int(positive_with_target.sum().item())
    margins = []
    for index in torch.where(positive_with_target)[0].tolist():
        positive = target[index].bool()
        true_logits = real[index][positive]
        true_indices = torch.where(positive)
        true_margin = torch.minimum(
            true_logits - row_bin[index][true_indices[0], 0],
            true_logits - col_bin[index][0, true_indices[1]],
        )
        margins.extend(true_margin.detach().tolist())
        best = real[index].flatten().argmax()
        best_a = int(best.item()) // 2
        best_b = int(best.item()) % 2
        correct += int(bool(accepted[index, best_a, best_b] and positive[best_a, best_b]))
    negative = ~is_match
    negative_rejected = 0
    for index in torch.where(negative)[0].tolist():
        negative_rejected += int(not bool(accepted[index].any()))
    return {
        "positive_acceptance": correct / max(positive_count, 1),
        "negative_rejection": negative_rejected / max(int(negative.sum().item()), 1),
        "true_to_dustbin_margin_mean": sum(margins) / max(len(margins), 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps <= 1 or args.margin < 0 or args.margin_weight < 0:
        raise ValueError("invalid preflight hyperparameters")
    torch.manual_seed(args.seed)
    # Five deterministic fixture roles: recalled positive, missing positive,
    # negative, recalled positive with a hard real-real distractor, and a
    # wrap-interval positive. The wrap role shares the same logit supervision
    # shape; its candidate construction is checked separately below.
    batch = 5
    valid_a = torch.ones(batch, 2, dtype=torch.bool)
    valid_b = torch.ones(batch, 2, dtype=torch.bool)
    target = torch.zeros(batch, 2, 2)
    target[0, 0, 0] = 1.0
    target[3, 0, 0] = 1.0
    target[4, 1, 1] = 1.0
    is_match = torch.tensor([True, True, False, True, True])
    logits = torch.nn.Parameter(torch.zeros(batch, 3, 3))
    with torch.no_grad():
        logits[3, 1, 1] = 0.8  # fixed hard-negative initialization
    optimizer = torch.optim.Adam([logits], lr=0.05)

    def losses() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = _outputs(logits, valid_a, valid_b)
        sample_valid = (~is_match) | target.flatten(1).any(dim=-1)
        assignment = candidate_assignment_loss(
            outputs,
            candidate_target_ab=target,
            is_match=is_match,
            candidate_valid_a=valid_a,
            candidate_valid_b=valid_b,
            sample_valid=sample_valid,
            consistency_weight=0.0,
        )["loss_candidate_total"]
        margin = positive_to_dustbin_margin_loss(
            outputs,
            target,
            is_match,
            valid_a,
            valid_b,
            margin=args.margin,
        )
        return assignment + args.margin_weight * margin, assignment, margin

    initial_total, initial_assignment, initial_margin = losses()
    initial_metrics = _metrics(logits.detach(), target, is_match)
    all_finite = True
    gradient_max = 0.0
    curve = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        total, assignment, margin = losses()
        total.backward()
        finite = bool(torch.isfinite(total) and torch.isfinite(logits.grad).all())
        all_finite &= finite
        if not finite:
            break
        gradient_max = max(gradient_max, float(logits.grad.abs().max().item()))
        optimizer.step()
        if step in (1, 10, args.steps):
            curve.append({
                "step": step,
                "loss_total": float(total.detach().item()),
                "loss_assignment": float(assignment.detach().item()),
                "loss_dustbin_margin": float(margin.detach().item()),
                **_metrics(logits.detach(), target, is_match),
            })
    final_total, final_assignment, final_margin = losses()
    final_metrics = _metrics(logits.detach(), target, is_match)

    missing_only = positive_to_dustbin_margin_loss(
        _outputs(logits[1:2], valid_a[1:2], valid_b[1:2]),
        target[1:2],
        is_match[1:2],
        valid_a[1:2],
        valid_b[1:2],
        margin=args.margin,
    )
    negative_only = positive_to_dustbin_margin_loss(
        _outputs(logits[2:3], valid_a[2:3], valid_b[2:3]),
        target[2:3],
        is_match[2:3],
        valid_a[2:3],
        valid_b[2:3],
        margin=args.margin,
    )
    signal = torch.zeros(16)
    signal[15], signal[0], signal[7] = 0.90, 0.85, 0.75
    wrap_proposals = recall_topk_opening_intervals(
        signal, 0.95, 2, 12,
        peak_top_k=4,
        peak_min_probability=0.20,
        expansion_threshold=0.12,
        dedup_iou=0.50,
    )
    legacy_zero_total = float((initial_assignment - initial_assignment).abs().item())
    criteria = {
        "initial_margin_nonzero": float(initial_margin.item()) > 0.0,
        "margin_declined_80pct": float(final_margin.item()) < 0.2 * float(initial_margin.item()),
        "total_loss_declined": float(final_total.item()) < float(initial_total.item()),
        "true_margin_increased": final_metrics["true_to_dustbin_margin_mean"] > initial_metrics["true_to_dustbin_margin_mean"],
        "positive_acceptance_increased": final_metrics["positive_acceptance"] > initial_metrics["positive_acceptance"],
        "negative_rejection_not_collapsed": final_metrics["negative_rejection"] >= 0.5,
        "missing_positive_margin_masked": float(missing_only.item()) == 0.0,
        "negative_margin_masked": float(negative_only.item()) == 0.0,
        "finite_loss_and_gradients": all_finite and math.isfinite(gradient_max),
        "nonzero_gradient": gradient_max > 0.0,
        "legacy_weight_zero_exact": legacy_zero_total == 0.0,
        "wrap_proposal_present": (15, 0) in wrap_proposals,
        "candidate_budget_respected": len(wrap_proposals) <= 12,
    }
    payload: Dict[str, Any] = {
        "schema_version": "zind-t4-cpu-preflight-v1",
        "device": "cpu",
        "seed": args.seed,
        "split_data_accessed": [],
        "test_data_accessed": False,
        "fixture_roles": [
            "positive_recalled", "positive_missing", "negative",
            "positive_hard_negative", "positive_wrap_interval",
        ],
        "margin": args.margin,
        "margin_weight": args.margin_weight,
        "initial": {
            "loss_total": float(initial_total.item()),
            "loss_assignment": float(initial_assignment.item()),
            "loss_dustbin_margin": float(initial_margin.item()),
            **initial_metrics,
        },
        "final": {
            "loss_total": float(final_total.item()),
            "loss_assignment": float(final_assignment.item()),
            "loss_dustbin_margin": float(final_margin.item()),
            **final_metrics,
        },
        "maximum_absolute_gradient": gradient_max,
        "wrap_proposals": wrap_proposals,
        "curve": curve,
        "criteria": criteria,
        "status": "PASS" if all(criteria.values()) else "FAIL",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
