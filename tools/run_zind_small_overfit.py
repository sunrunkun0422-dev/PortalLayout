#!/usr/bin/env python3
"""Run fixed-subset T0-T3 ZInD end-to-end overfit preflight experiments."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portallayout.datasets.zind_bipair_adapter import adapt_zind_bipair_batch
from portallayout.datasets.zind_bipair_dataset import collate_zind_bipair
from portallayout.models.cross_scene_matcher import candidate_assignment_loss
from tools.analyze_zind_candidate_recall import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
    _images,
    _loader,
    _predicted_probabilities,
    _seed_everything,
    _slice_output,
    _training_args,
    _write_csv,
    _write_json,
    load_historical_model,
)
from tools.train_zind_cross_scene_matcher import (
    candidate_targets_from_affinity,
    predicted_candidate_masks,
)
from tools.train_zind_end_to_end import (
    JointZInDMatchingDataset,
    _candidate_supervision_validity,
    _forward_loss,
    _gradient_norm,
    _optimizer,
)


EXPERIMENTS = (
    ("T0", "legacy_dustbin", 0.0),
    ("T1", "mask_positive", 0.0),
    ("T2", "legacy_dustbin", 0.1),
    ("T3", "mask_positive", 0.1),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--subset_size", type=int, default=24)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--selection_scan_limit", type=int, default=256)
    return parser.parse_args(argv)


def training_dataset(checkpoint: Mapping[str, Any]) -> JointZInDMatchingDataset:
    data = checkpoint["data_contract"]
    training = checkpoint["training_contract"]
    return JointZInDMatchingDataset(
        data["manifests"]["train"],
        data["feature_caches"]["train"],
        data_root=None,
        max_pairs=None,
        expected_split="train",
        image_shape=(training["image_height"], training["image_width"]),
        rotation_augmentation=False,
    )


def select_subset(
    model,
    checkpoint: Mapping[str, Any],
    dataset: JointZInDMatchingDataset,
    device: torch.device,
    threshold: float,
    min_width: int,
    max_candidates: int,
    subset_size: int,
    scan_limit: int,
) -> List[Dict[str, Any]]:
    if subset_size % 3 != 0:
        raise ValueError("subset_size must be divisible by three")
    per_category = subset_size // 3
    wanted_pool = max(per_category * 2, per_category)
    pools: Dict[str, List[Dict[str, Any]]] = {
        "positive_with_target": [],
        "positive_without_target": [],
        "negative": [],
    }
    selection_count = min(scan_limit, len(dataset))
    selection_dataset = torch.utils.data.Subset(dataset, list(range(selection_count)))
    loader = torch.utils.data.DataLoader(
        selection_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_zind_bipair,
    )
    offset = 0
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            pair = adapt_zind_bipair_batch(raw_batch).to(device)
            image_a, image_b = _images(raw_batch, device)
            live = model.live_layout(image_a, image_b)
            probability_a, probability_b = _predicted_probabilities(model, live)
            masks_a, valid_a = predicted_candidate_masks(
                probability_a, threshold, min_width, max_candidates
            )
            masks_b, valid_b = predicted_candidate_masks(
                probability_b, threshold, min_width, max_candidates
            )
            target = candidate_targets_from_affinity(
                masks_a, pair.affinity_ab, masks_b
            )
            has_target = target.flatten(1).any(-1)
            for local in range(int(pair.batch_size)):
                if bool(pair.is_match[local]):
                    category = (
                        "positive_with_target"
                        if bool(has_target[local])
                        else "positive_without_target"
                    )
                else:
                    category = "negative"
                pools[category].append(
                    {
                        "dataset_index": offset + local,
                        "pair_id": raw_batch["pair_id"][local],
                        "category": category,
                        "candidate_count_a": int(valid_a[local].sum().item()),
                        "candidate_count_b": int(valid_b[local].sum().item()),
                        "candidate_count_total": int(
                            valid_a[local].sum().item() + valid_b[local].sum().item()
                        ),
                        "has_target_at_selection": bool(has_target[local]),
                    }
                )
            offset += int(pair.batch_size)
            if all(len(pool) >= wanted_pool for pool in pools.values()):
                break
    if any(len(pool) < per_category for pool in pools.values()):
        raise RuntimeError(
            "could not build balanced subset: {}".format(
                {key: len(value) for key, value in pools.items()}
            )
        )

    selected_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for category, pool in pools.items():
        ordered = sorted(pool, key=lambda item: (item["candidate_count_total"], item["dataset_index"]))
        low_count = per_category // 2
        high_count = per_category - low_count
        selected = ordered[:low_count] + ordered[-high_count:]
        unique = {item["dataset_index"]: item for item in selected}
        if len(unique) < per_category:
            for item in ordered:
                unique.setdefault(item["dataset_index"], item)
                if len(unique) == per_category:
                    break
        selected_by_category[category] = list(unique.values())[:per_category]

    interleaved = []
    category_order = (
        "positive_with_target",
        "positive_without_target",
        "negative",
    )
    for index in range(per_category):
        for category in category_order:
            interleaved.append(selected_by_category[category][index])
    for order, item in enumerate(interleaved):
        item["fixed_order"] = order
    return interleaved


def _batch_indices(step: int, item_count: int, batch_size: int) -> List[int]:
    start = ((step - 1) * batch_size) % item_count
    return [(start + offset) % item_count for offset in range(batch_size)]


def _ranking_stats(
    outputs: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
    is_match: torch.Tensor,
) -> Dict[str, float]:
    positive_scores = []
    hardest_negative_scores = []
    ranks = []
    reciprocal_ranks = []
    top1 = []
    score = outputs["candidate_pair_score"]
    for index in range(int(score.shape[0])):
        if not bool(is_match[index]):
            continue
        valid_pair = valid_a[index].unsqueeze(-1) & valid_b[index].unsqueeze(0)
        positive = (target[index] > 0) & valid_pair
        if not bool(positive.any()):
            continue
        negative = valid_pair & ~positive
        positive_scores.append(float(score[index][positive].mean().item()))
        if bool(negative.any()):
            hardest_negative_scores.append(float(score[index][negative].max().item()))
        flat_valid = torch.where(valid_pair.flatten())[0]
        ranked = flat_valid[
            torch.argsort(score[index].flatten()[flat_valid], descending=True)
        ]
        positions = torch.where(positive.flatten()[ranked])[0]
        rank = int(positions[0].item()) + 1
        ranks.append(float(rank))
        reciprocal_ranks.append(1.0 / rank)
        top1.append(float(rank == 1))
    positive_mean = float(np.mean(positive_scores)) if positive_scores else float("nan")
    negative_mean = (
        float(np.mean(hardest_negative_scores))
        if hardest_negative_scores
        else float("nan")
    )
    return {
        "correct_candidate_score": positive_mean,
        "hardest_negative_score": negative_mean,
        "score_margin": positive_mean - negative_mean,
        "correct_candidate_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "top1": float(np.mean(top1)) if top1 else float("nan"),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
        "target_sample_count": float(len(ranks)),
    }


def _mean_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate_window(
    experiment: str,
    step: int,
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    keys = (
        "loss_total",
        "loss_candidate_assignment",
        "loss_candidate_rank",
        "correct_candidate_score",
        "hardest_negative_score",
        "score_margin",
        "correct_candidate_rank",
        "top1",
        "mrr",
        "masked_positive_count",
        "backbone_gradient_norm",
        "opening_head_gradient_norm",
        "matcher_gradient_norm",
        "dustbin_gradient_abs",
        "step_seconds",
        "peak_memory_gib",
    )
    row: Dict[str, Any] = {
        "experiment": experiment,
        "step": step,
        "window_start": records[0]["step"],
        "window_end": records[-1]["step"],
        "all_losses_finite": all(record["losses_finite"] for record in records),
        "all_gradients_finite": all(record["gradients_finite"] for record in records),
        "masked_positive_assignment_max_abs": max(
            float(record["masked_positive_assignment_max_abs"]) for record in records
        ),
        "negative_dustbin_nonzero_steps": sum(
            int(record["negative_present"] and record["dustbin_gradient_abs"] > 0)
            for record in records
        ),
    }
    for key in keys:
        row[key] = _mean_finite([record[key] for record in records])
    return row


def run_experiment(
    name: str,
    policy: str,
    rank_weight: float,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    cached_samples: Sequence[Mapping[str, Any]],
    subset: Sequence[Mapping[str, Any]],
    device: torch.device,
    threshold: float,
    batch_size: int,
    steps: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _seed_everything(seed)
    model, loaded_checkpoint = load_historical_model(checkpoint_path, device)
    args = _training_args(loaded_checkpoint)
    args.candidate_missing_target_policy = policy
    args.candidate_rank_loss_weight = rank_weight
    args.opening_guidance_source = "predicted"
    args.candidate_source = "predicted"
    model.train()
    optimizer = _optimizer(model, args)
    step_records: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    masked_assignment_max = 0.0
    negative_dustbin_nonzero = 0
    all_finite = True
    all_module_gradients_seen = {
        "backbone": False,
        "opening_head": False,
        "matcher": False,
    }

    for step in range(1, steps + 1):
        positions = _batch_indices(step, len(cached_samples), batch_size)
        raw_batch = collate_zind_bipair([cached_samples[position] for position in positions])
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        outputs, pair, target, valid_a, valid_b, losses = _forward_loss(
            model,
            raw_batch,
            device,
            opening_guidance_source="predicted",
            candidate_source="predicted",
            opening_threshold=threshold,
            args=args,
        )
        losses_finite = all(
            bool(torch.isfinite(value))
            for value in losses.values()
            if torch.is_tensor(value)
        )
        if not losses_finite:
            raise RuntimeError("{} produced non-finite loss at step {}".format(name, step))
        ranking = _ranking_stats(outputs, target, valid_a, valid_b, pair.is_match)
        sample_valid, _ = _candidate_supervision_validity(
            target, pair.is_match, valid_a, valid_b, policy
        )
        masked_positive = pair.is_match.bool() & ~sample_valid
        per_sample_masked_max = 0.0
        for index in torch.where(masked_positive)[0].tolist():
            isolated = _slice_output(outputs, index)
            isolated_loss = candidate_assignment_loss(
                isolated,
                candidate_target_ab=target[index : index + 1],
                is_match=pair.is_match[index : index + 1],
                candidate_valid_a=valid_a[index : index + 1],
                candidate_valid_b=valid_b[index : index + 1],
                sample_valid=torch.zeros(1, dtype=torch.bool, device=device),
                consistency_weight=0.0,
            )["loss_candidate_assignment"]
            per_sample_masked_max = max(per_sample_masked_max, abs(float(isolated_loss.item())))
        losses["loss_total"].backward()
        gradients = {
            "backbone": _gradient_norm(model.bi_layout),
            "opening_head": _gradient_norm(model.matcher.opening_head),
            "matcher": _gradient_norm(model.matcher),
        }
        gradients_finite = all(math.isfinite(value) for value in gradients.values())
        if not gradients_finite:
            nonfinite_parameters = [
                parameter_name
                for parameter_name, parameter in model.named_parameters()
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            ]
            raise RuntimeError(
                "{} produced non-finite gradient at step {}: {}".format(
                    name, step, ", ".join(nonfinite_parameters[:20])
                )
            )
        for key, value in gradients.items():
            all_module_gradients_seen[key] |= value > 0
        dustbin_gradient = (
            abs(float(model.matcher.dustbin_score.grad.item()))
            if model.matcher.dustbin_score.grad is not None
            else 0.0
        )
        negative_present = bool((~pair.is_match.bool()).any())
        negative_dustbin_nonzero += int(negative_present and dustbin_gradient > 0)
        masked_assignment_max = max(masked_assignment_max, per_sample_masked_max)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        step_seconds = time.perf_counter() - start
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            if device.type == "cuda"
            else 0.0
        )
        record = {
            "step": step,
            "loss_total": float(losses["loss_total"].detach().item()),
            "loss_candidate_assignment": float(
                losses["loss_candidate_assignment"].detach().item()
            ),
            "loss_candidate_rank": float(losses["loss_candidate_rank"].detach().item()),
            **ranking,
            "masked_positive_count": float(masked_positive.sum().item()),
            "masked_positive_assignment_max_abs": per_sample_masked_max,
            "backbone_gradient_norm": gradients["backbone"],
            "opening_head_gradient_norm": gradients["opening_head"],
            "matcher_gradient_norm": gradients["matcher"],
            "dustbin_gradient_abs": dustbin_gradient,
            "negative_present": negative_present,
            "losses_finite": losses_finite,
            "gradients_finite": gradients_finite,
            "step_seconds": step_seconds,
            "peak_memory_gib": peak_memory,
        }
        step_records.append(record)
        all_finite &= losses_finite and gradients_finite
        if step == 1:
            curve_rows.append(_aggregate_window(name, step, step_records[-1:]))
        if step % 10 == 0:
            curve_rows.append(_aggregate_window(name, step, step_records[-10:]))
            latest = curve_rows[-1]
            print(
                "{} step={}/{} loss={:.4f} assignment={:.4f} rank={:.4f} "
                "top1={:.4f} mrr={:.4f} time={:.2f}s".format(
                    name,
                    step,
                    steps,
                    latest["loss_total"],
                    latest["loss_candidate_assignment"],
                    latest["loss_candidate_rank"],
                    latest["top1"],
                    latest["mrr"],
                    latest["step_seconds"],
                ),
                flush=True,
            )

    # The 100-step gate compares the first and last 10-step windows.  A short
    # smoke run has only one 10-step window, so compare it with step 1 instead.
    first_window = (
        next(row for row in curve_rows if row["step"] == 10)
        if steps > 10
        else curve_rows[0]
    )
    last_window = curve_rows[-1]
    rank_decline = last_window["loss_candidate_rank"] < first_window["loss_candidate_rank"]
    directions = {
        "correct_rank_improved": (
            last_window["correct_candidate_rank"] < first_window["correct_candidate_rank"]
        ),
        "mrr_improved": last_window["mrr"] > first_window["mrr"],
        "score_margin_improved": (
            last_window["score_margin"] > first_window["score_margin"]
        ),
    }
    summary = {
        "experiment": name,
        "missing_target_policy": policy,
        "candidate_rank_loss_weight": rank_weight,
        "steps": steps,
        "all_losses_and_gradients_finite": all_finite,
        "all_expected_modules_received_nonzero_gradient": all(all_module_gradients_seen.values()),
        "module_gradient_seen": all_module_gradients_seen,
        "masked_positive_assignment_max_abs": masked_assignment_max,
        "negative_dustbin_nonzero_steps": negative_dustbin_nonzero,
        "ranking_loss_first_10": first_window["loss_candidate_rank"],
        "ranking_loss_last_10": last_window["loss_candidate_rank"],
        "ranking_loss_declined": rank_decline,
        "ranking_direction_checks": directions,
        "at_least_one_ranking_direction_improved": any(directions.values()),
        "first_window": first_window,
        "last_window": last_window,
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return curve_rows, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.steps <= 0 or args.steps > 100:
        raise ValueError("steps must be in [1, 100]")
    if args.subset_size < 24 or args.subset_size > 32:
        raise ValueError("subset_size must be in [24, 32]")
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    g1_path = output_dir / "g1_selection.json"
    if not g1_path.is_file():
        raise FileNotFoundError("run candidate analysis first: {}".format(g1_path))
    g1 = json.loads(g1_path.read_text(encoding="utf-8"))["g1"]
    threshold = float(g1["threshold"])
    min_width = int(g1["min_width"])
    max_candidates = int(g1["max_candidates"])
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("small overfit requires the configured CUDA GPU")
    _seed_everything(args.seed)
    selection_model, checkpoint = load_historical_model(checkpoint_path, device)
    dataset = training_dataset(checkpoint)
    subset = select_subset(
        selection_model,
        checkpoint,
        dataset,
        device,
        threshold,
        min_width,
        max_candidates,
        args.subset_size,
        args.selection_scan_limit,
    )
    del selection_model
    gc.collect()
    torch.cuda.empty_cache()
    cached_samples = [dataset[item["dataset_index"]] for item in subset]
    subset_payload = {
        "split": "train",
        "validation_or_test_used_for_training": False,
        "seed": args.seed,
        "fixed_order": True,
        "subset_size": len(subset),
        "g1": {
            "threshold": threshold,
            "min_width": min_width,
            "max_candidates": max_candidates,
        },
        "category_counts": {
            category: sum(item["category"] == category for item in subset)
            for category in (
                "positive_with_target",
                "positive_without_target",
                "negative",
            )
        },
        "samples": subset,
    }
    _write_json(output_dir / "overfit_subset.json", subset_payload)
    commands_path = output_dir / "commands.txt"
    with commands_path.open("a", encoding="utf-8") as file:
        file.write(" ".join(sys.argv) + "\n")

    all_rows: List[Dict[str, Any]] = []
    summaries = []
    for name, policy, rank_weight in EXPERIMENTS:
        rows, summary = run_experiment(
            name,
            policy,
            rank_weight,
            checkpoint_path,
            checkpoint,
            cached_samples,
            subset,
            device,
            threshold,
            args.batch_size,
            args.steps,
            args.seed,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        _write_csv(output_dir / "overfit_t0_t3.csv", all_rows)
        _write_json(output_dir / "overfit_summary.json", {"experiments": summaries})

    by_name = {summary["experiment"]: summary for summary in summaries}
    gate_pass = (
        all(summary["all_losses_and_gradients_finite"] for summary in summaries)
        and all(summary["all_expected_modules_received_nonzero_gradient"] for summary in summaries)
        and by_name["T1"]["masked_positive_assignment_max_abs"] == 0.0
        and by_name["T3"]["masked_positive_assignment_max_abs"] == 0.0
        and by_name["T0"]["negative_dustbin_nonzero_steps"] > 0
        and by_name["T2"]["negative_dustbin_nonzero_steps"] > 0
        and by_name["T2"]["ranking_loss_declined"]
        and by_name["T3"]["ranking_loss_declined"]
        and by_name["T2"]["at_least_one_ranking_direction_improved"]
        and by_name["T3"]["at_least_one_ranking_direction_improved"]
    )
    _write_json(
        output_dir / "overfit_gate.json",
        {
            "status": "PASS" if gate_pass else "FAIL",
            "criteria": {
                "finite": all(summary["all_losses_and_gradients_finite"] for summary in summaries),
                "all_module_gradients": all(
                    summary["all_expected_modules_received_nonzero_gradient"]
                    for summary in summaries
                ),
                "t1_masked_assignment_zero": by_name["T1"]["masked_positive_assignment_max_abs"] == 0,
                "t3_masked_assignment_zero": by_name["T3"]["masked_positive_assignment_max_abs"] == 0,
                "t0_negative_dustbin_gradient": by_name["T0"]["negative_dustbin_nonzero_steps"] > 0,
                "t2_negative_dustbin_gradient": by_name["T2"]["negative_dustbin_nonzero_steps"] > 0,
                "t2_ranking_loss_declined": by_name["T2"]["ranking_loss_declined"],
                "t3_ranking_loss_declined": by_name["T3"]["ranking_loss_declined"],
                "t2_ranking_direction_improved": by_name["T2"]["at_least_one_ranking_direction_improved"],
                "t3_ranking_direction_improved": by_name["T3"]["at_least_one_ranking_direction_improved"],
            },
            "note": "small-subset overfit demonstrates learnability only, not validation superiority",
        },
    )
    print("overfit gate {}".format("PASS" if gate_pass else "FAIL"), flush=True)
    return 0 if gate_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
