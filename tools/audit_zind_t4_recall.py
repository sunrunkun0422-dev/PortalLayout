#!/usr/bin/env python3
"""CPU-only recall audit over frozen T3 validation predictions.

This tool is deliberately val-only.  It replays candidate proposals from the
stored Opening Head probabilities and never loads a model or a test manifest.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cross_scene_matcher import (  # noqa: E402
    circular_interval_iou,
    opening_probabilities_to_intervals,
    recall_topk_opening_intervals,
)


DEFAULT_FROZEN_ROOT = REPO_ROOT / "src/output/zind_geometry_val_full/t3_frozen_m2"
DEFAULT_VAL_MANIFEST = REPO_ROOT / "src/dataset/ZInD_matching/val.json"
DEFAULT_OUTPUT = REPO_ROOT / "experiments/zind_e2e_v3/g0_seed123/T4_recall/preflight/recall_audit"


PROPOSAL_GRID = (
    {"id": "P0_t047_legacy_k12", "family": "P0", "mode": "legacy", "threshold": 0.47, "max_candidates": 12},
    {"id": "P1_t042_legacy_k12", "family": "P1", "mode": "legacy", "threshold": 0.42, "max_candidates": 12},
    {"id": "P1_t037_legacy_k12", "family": "P1", "mode": "legacy", "threshold": 0.37, "max_candidates": 12},
    {"id": "P2_t047_peak2_k12", "family": "P2", "mode": "recall_topk", "threshold": 0.47, "peak_top_k": 2, "max_candidates": 12},
    {"id": "P2_t047_peak4_k12", "family": "P2", "mode": "recall_topk", "threshold": 0.47, "peak_top_k": 4, "max_candidates": 12},
    {"id": "P3_t042_peak2_k12", "family": "P3", "mode": "recall_topk", "threshold": 0.42, "peak_top_k": 2, "max_candidates": 12},
    {"id": "P3_t042_peak4_k12", "family": "P3", "mode": "recall_topk", "threshold": 0.42, "peak_top_k": 4, "max_candidates": 12},
    {"id": "P3_t037_peak4_k12", "family": "P3", "mode": "recall_topk", "threshold": 0.37, "peak_top_k": 4, "max_candidates": 12},
    {"id": "P3_t042_peak4_k16", "family": "P3", "mode": "recall_topk", "threshold": 0.42, "peak_top_k": 4, "max_candidates": 16},
)
FIXED_PROPOSAL_DEFAULTS = {
    "min_width_tokens": 2,
    "peak_min_probability": 0.20,
    "expansion_threshold": 0.12,
    "dedup_iou": 0.50,
    "target_iou": 0.30,
}
THRESHOLD_SWEEP = (-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_root", default=str(DEFAULT_FROZEN_ROOT))
    parser.add_argument("--val_manifest", default=str(DEFAULT_VAL_MANIFEST))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def _quantiles(values: Iterable[float]) -> Dict[str, float | int | None]:
    data = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not data:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "p90": None, "p95": None, "min": None, "max": None}

    def percentile(q: float) -> float:
        location = (len(data) - 1) * q
        lower = int(math.floor(location))
        upper = int(math.ceil(location))
        if lower == upper:
            return data[lower]
        return data[lower] * (upper - location) + data[upper] * (location - lower)

    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "min": data[0],
        "max": data[-1],
    }


def _interval(value: Any) -> Tuple[int, int]:
    if isinstance(value, Mapping):
        return int(value["tokenStart"]), int(value["tokenEnd"])
    return int(value[0]), int(value[1])


def _proposals(probability: Sequence[float], config: Mapping[str, Any]) -> list:
    signal = torch.as_tensor(probability, dtype=torch.float32)
    common = {
        "threshold": float(config["threshold"]),
        "min_width_tokens": FIXED_PROPOSAL_DEFAULTS["min_width_tokens"],
        "max_intervals": int(config["max_candidates"]),
    }
    if config["mode"] == "legacy":
        return opening_probabilities_to_intervals(signal, **common)
    return recall_topk_opening_intervals(
        signal,
        **common,
        peak_top_k=int(config["peak_top_k"]),
        peak_min_probability=FIXED_PROPOSAL_DEFAULTS["peak_min_probability"],
        expansion_threshold=FIXED_PROPOSAL_DEFAULTS["expansion_threshold"],
        dedup_iou=FIXED_PROPOSAL_DEFAULTS["dedup_iou"],
    )


def _side_recalled(proposals: Sequence[Tuple[int, int]], target: Tuple[int, int], token_count: int) -> bool:
    return any(
        circular_interval_iou(proposal, target, token_count) >= FIXED_PROPOSAL_DEFAULTS["target_iou"]
        for proposal in proposals
    )


def _token_set(interval: Tuple[int, int], token_count: int) -> set:
    start, end = interval
    start %= token_count
    end %= token_count
    if start <= end:
        return set(range(start, end + 1))
    return set(range(start, token_count)) | set(range(0, end + 1))


def _endpoint_error(candidate_a: Mapping[str, Any], candidate_b: Mapping[str, Any]) -> float:
    a = candidate_a["globalEndpoints"]
    b = candidate_b["globalEndpoints"]

    def distance(p, q):
        return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))

    direct = 0.5 * (distance(a[0], b[0]) + distance(a[1], b[1]))
    reverse = 0.5 * (distance(a[0], b[1]) + distance(a[1], b[0]))
    return min(direct, reverse)


def _projected_target_recalled(
    proposed_a: Sequence[Tuple[int, int]],
    proposed_b: Sequence[Tuple[int, int]],
    record: Mapping[str, Any],
) -> bool:
    """Mirror candidate_targets_from_affinity without materializing [N,N]."""
    token_count = int(record["token_count"])
    matches = []
    for candidate_a in record["candidates_a"]:
        mask_a = _token_set(_interval(candidate_a["tokenInterval"]), token_count)
        for candidate_b in record["candidates_b"]:
            if _endpoint_error(candidate_a, candidate_b) <= 0.06:
                mask_b = _token_set(_interval(candidate_b["tokenInterval"]), token_count)
                matches.append((mask_a, mask_b))
    for interval_a in proposed_a:
        proposal_a = _token_set(interval_a, token_count)
        for interval_b in proposed_b:
            proposal_b = _token_set(interval_b, token_count)
            connected_a = set().union(*(
                mask_a for mask_a, mask_b in matches if proposal_b & mask_b
            )) if matches else set()
            connected_b = set().union(*(
                mask_b for mask_a, mask_b in matches if proposal_a & mask_a
            )) if matches else set()
            if not (proposal_a & connected_a) or not (proposal_b & connected_b):
                continue
            union_a = proposal_a | connected_a
            union_b = proposal_b | connected_b
            iou_a = len(proposal_a & connected_a) / len(union_a)
            iou_b = len(proposal_b & connected_b) / len(union_b)
            if iou_a >= FIXED_PROPOSAL_DEFAULTS["target_iou"] and iou_b >= FIXED_PROPOSAL_DEFAULTS["target_iou"]:
                return True
    return False


def _load_inputs(frozen_root: Path, val_manifest: Path):
    if "test" in val_manifest.name.lower() or "test" in str(frozen_root).lower():
        raise ValueError("val-only audit refuses any path containing 'test'")
    manifest = json.loads(val_manifest.read_text())
    if manifest.get("split") != "val":
        raise ValueError("manifest split must be val")
    records = manifest["pairs"]
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("val manifest contains duplicate pair ids")

    per_pair = {}
    with (frozen_root / "per_pair.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("split") != "val" or record.get("test_data_accessed"):
                raise ValueError("frozen pair violates val-only contract")
            per_pair[record["pair_id"]] = record
    artifacts = {}
    for path in sorted((frozen_root / "frozen_predictions").glob("*.json.gz")):
        with gzip.open(path, "rt") as handle:
            artifact = json.load(handle)
        if artifact.get("test_data_accessed") or artifact.get("gt_available_to_prediction"):
            raise ValueError("frozen prediction violates inference contract")
        artifacts[artifact["pair_id"]] = artifact
    expected = set(by_id)
    if set(per_pair) != expected or set(artifacts) != expected:
        raise ValueError("manifest, pair report, and frozen artifact ids differ")
    return records, by_id, per_pair, artifacts


def _audit_proposals(records, per_pair, artifacts):
    rows = []
    p0_exact_interval_pairs = 0
    p0_recall_parity = 0
    for config in PROPOSAL_GRID:
        recalled = 0
        positive = 0
        wrap_positive = 0
        wrap_recalled = 0
        counts = []
        products = []
        negative_counts = []
        negative_products = []
        type_totals = Counter()
        type_recalled = Counter()
        for record in records:
            pair_id = record["id"]
            artifact = artifacts[pair_id]
            proposed_a = _proposals(artifact["opening_probability_A"], config)
            proposed_b = _proposals(artifact["opening_probability_B"], config)
            if config["family"] == "P0":
                frozen_a = [_interval(value) for value in artifact["predicted_intervals_A"]]
                frozen_b = [_interval(value) for value in artifact["predicted_intervals_B"]]
                p0_exact_interval_pairs += int(proposed_a == frozen_a and proposed_b == frozen_b)
            count_a, count_b = len(proposed_a), len(proposed_b)
            counts.extend((count_a, count_b))
            products.append(count_a * count_b)
            is_match = bool(record["supervision"]["is_match"])
            if not is_match:
                negative_counts.extend((count_a, count_b))
                negative_products.append(count_a * count_b)
                continue
            positive += 1
            target_a_index = int(record["supervision"]["target_candidate_a"])
            target_b_index = int(record["supervision"]["target_candidate_b"])
            target_a = _interval(record["candidates_a"][target_a_index]["tokenInterval"])
            target_b = _interval(record["candidates_b"][target_b_index]["tokenInterval"])
            token_count = int(record["token_count"])
            hit = _projected_target_recalled(proposed_a, proposed_b, record)
            recalled += int(hit)
            is_wrap = target_a[0] > target_a[1] or target_b[0] > target_b[1]
            wrap_positive += int(is_wrap)
            wrap_recalled += int(is_wrap and hit)
            interface = "{}-{}".format(
                record["candidates_a"][target_a_index]["type"],
                record["candidates_b"][target_b_index]["type"],
            )
            type_totals[interface] += 1
            type_recalled[interface] += int(hit)
            if config["family"] == "P0":
                p0_recall_parity += int(hit == bool(per_pair[pair_id]["evaluation"]["gt_candidate_recalled"]))
        rows.append({
            **config,
            **FIXED_PROPOSAL_DEFAULTS,
            "positive_count": positive,
            "positive_recalled": recalled,
            "positive_recall": recalled / positive,
            "recall_delta_count_vs_p0": 0,
            "candidate_count": _quantiles(counts),
            "candidate_product": _quantiles(products),
            "negative_candidate_count": _quantiles(negative_counts),
            "negative_candidate_product": _quantiles(negative_products),
            "wrap": {"positive": wrap_positive, "recalled": wrap_recalled, "recall": wrap_recalled / wrap_positive if wrap_positive else None},
            "interface_type": {
                key: {"positive": type_totals[key], "recalled": type_recalled[key], "recall": type_recalled[key] / type_totals[key]}
                for key in sorted(type_totals)
            },
        })
    baseline = rows[0]["positive_recalled"]
    for row in rows:
        row["recall_delta_count_vs_p0"] = row["positive_recalled"] - baseline
    parity = {
        "p0_exact_interval_pair_count": p0_exact_interval_pairs,
        "p0_recall_parity_positive_count": p0_recall_parity,
        "pair_count": len(records),
        "positive_count": rows[0]["positive_count"],
        "exact": (
            p0_exact_interval_pairs == len(records)
            and p0_recall_parity == rows[0]["positive_count"]
        ),
    }
    return rows, parity


def _safe_score(matrix, index_a: int, index_b: int, count_a: int, count_b: int):
    if 0 <= index_a < count_a and 0 <= index_b < count_b:
        return float(matrix[index_a][index_b])
    return None


def _audit_scores(records, per_pair, artifacts):
    recalled_true_scores = []
    rejected_true_scores = []
    accepted_true_scores = []
    positive_margins = []
    rejected_margins = []
    accepted_margins = []
    negative_margins = []
    wrong_gaps = []
    wrong_products = []
    correct_products = []
    wrong_types = Counter()
    sweep_inputs = []
    for record in records:
        pair_id = record["id"]
        report = per_pair[pair_id]
        artifact = artifacts[pair_id]
        evaluation = report["evaluation"]
        is_match = bool(record["supervision"]["is_match"])
        match_score = float(artifact["pair_match_probability"])
        no_match_score = float(artifact["pair_no_match_probability"])
        margin = match_score - no_match_score
        selected = tuple(int(value) for value in artifact["best_candidate_pair"])
        accepted = selected[0] >= 0 and selected[1] >= 0
        if not is_match:
            negative_margins.append(margin)
            sweep_inputs.append((False, margin, False))
            continue
        if not evaluation["gt_candidate_recalled"]:
            sweep_inputs.append((True, margin, False))
            continue
        positive_margins.append(margin)
        target_pairs = [tuple(int(value) for value in pair) for pair in evaluation["target_predicted_candidate_pairs"]]
        matrix = artifact["candidate_pair_scores"]
        count_a = int(artifact["candidate_count_A"])
        count_b = int(artifact["candidate_count_B"])
        true_values = [
            _safe_score(matrix, ia, ib, count_a, count_b) for ia, ib in target_pairs
        ]
        true_values = [value for value in true_values if value is not None]
        best_true = max(true_values) if true_values else None
        real_top_correct = False
        if count_a > 0 and count_b > 0:
            flat = [
                (float(matrix[ia][ib]), ia, ib)
                for ia in range(count_a)
                for ib in range(count_b)
            ]
            _, top_a, top_b = max(flat, key=lambda item: item[0])
            real_top_correct = (top_a, top_b) in target_pairs
        sweep_inputs.append((True, margin, real_top_correct))
        if best_true is not None:
            recalled_true_scores.append(best_true)
            (accepted_true_scores if accepted else rejected_true_scores).append(best_true)
        (accepted_margins if accepted else rejected_margins).append(margin)
        if accepted and not evaluation["selected_candidate_correct"] and best_true is not None:
            selected_score = _safe_score(matrix, selected[0], selected[1], count_a, count_b)
            if selected_score is not None:
                wrong_gaps.append(selected_score - best_true)
            wrong_products.append(count_a * count_b)
            intervals_a = [_interval(value) for value in artifact["predicted_intervals_A"]]
            intervals_b = [_interval(value) for value in artifact["predicted_intervals_B"]]
            inferred = []
            for side, selected_index, intervals, candidates in (
                ("A", selected[0], intervals_a, record["candidates_a"]),
                ("B", selected[1], intervals_b, record["candidates_b"]),
            ):
                if selected_index >= len(intervals):
                    inferred.append("unknown")
                    continue
                proposal = intervals[selected_index]
                best = max(
                    candidates,
                    key=lambda item: circular_interval_iou(proposal, _interval(item["tokenInterval"]), int(record["token_count"])),
                )
                inferred.append(str(best["type"]))
            wrong_types["{}-{}".format(*inferred)] += 1
        elif evaluation["selected_candidate_correct"]:
            correct_products.append(count_a * count_b)

    sweep = []
    positive_total = sum(1 for is_match, _, _ in sweep_inputs if is_match)
    negative_total = len(sweep_inputs) - positive_total
    for threshold in THRESHOLD_SWEEP:
        tp = fp = tn = fn = s3_proxy = 0
        for is_match, margin, candidate_correct in sweep_inputs:
            prediction = margin >= threshold
            tp += int(is_match and prediction)
            fn += int(is_match and not prediction)
            fp += int(not is_match and prediction)
            tn += int(not is_match and not prediction)
            s3_proxy += int(is_match and prediction and candidate_correct)
        positive_recall = tp / positive_total
        negative_rejection = tn / negative_total
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * positive_recall / (precision + positive_recall) if precision + positive_recall else 0.0
        harmonic = 2 * positive_recall * negative_rejection / (positive_recall + negative_rejection) if positive_recall + negative_rejection else 0.0
        sweep.append({
            "threshold": threshold,
            "positive_recall": positive_recall,
            "negative_rejection": negative_rejection,
            "precision": precision,
            "f1": f1,
            "balanced": 0.5 * (positive_recall + negative_rejection),
            "harmonic": harmonic,
            "s3_reach_proxy": s3_proxy / positive_total,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return {
        "frozen_field_availability": {
            "true_candidate_pair_score": True,
            "pair_match_probability": True,
            "pair_no_match_probability": True,
            "candidate_assignment_logits": False,
            "row_dustbin_score": False,
            "column_dustbin_score": False,
            "limitation": "Exact row/column true-to-dustbin logits were not serialized; they are not reconstructed or guessed.",
        },
        "recalled_true_candidate_score": _quantiles(recalled_true_scores),
        "accepted_true_candidate_score": _quantiles(accepted_true_scores),
        "rejected_true_candidate_score": _quantiles(rejected_true_scores),
        "positive_pair_margin": _quantiles(positive_margins),
        "accepted_positive_pair_margin": _quantiles(accepted_margins),
        "rejected_positive_pair_margin": _quantiles(rejected_margins),
        "negative_pair_margin": _quantiles(negative_margins),
        "threshold_sweep": sweep,
        "wrong_candidate": {
            "selected_minus_best_true_score_gap": _quantiles(wrong_gaps),
            "candidate_product": _quantiles(wrong_products),
            "correct_candidate_product": _quantiles(correct_products),
            "inferred_selected_interface_type": dict(sorted(wrong_types.items())),
            "ranking_loss_includes_dustbin": False,
            "ranking_loss_scope": "real-real candidate_pair_score only",
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frozen_root = Path(args.frozen_root).resolve()
    val_manifest = Path(args.val_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, _, per_pair, artifacts = _load_inputs(frozen_root, val_manifest)
    proposal_rows, parity = _audit_proposals(records, per_pair, artifacts)
    score_audit = _audit_scores(records, per_pair, artifacts)
    selected = max(
        proposal_rows[1:],
        key=lambda row: (
            row["positive_recall"],
            -row["candidate_product"]["p95"],
            -row["candidate_count"]["mean"],
        ),
    )
    proposal_config = {
        key: selected[key]
        for key in (
            "id", "family", "mode", "threshold", "max_candidates",
            "peak_top_k", "min_width_tokens", "peak_min_probability",
            "expansion_threshold", "dedup_iou", "target_iou",
        ) if key in selected
    }
    proposal_hash = hashlib.sha256(
        json.dumps(proposal_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    positive = sum(bool(record["supervision"]["is_match"]) for record in records)
    recalled = sum(bool(per_pair[record["id"]]["evaluation"]["gt_candidate_recalled"]) for record in records)
    selected_correct = sum(bool(per_pair[record["id"]]["evaluation"]["selected_candidate_correct"]) for record in records)
    rejected = sum(
        bool(record["supervision"]["is_match"])
        and bool(per_pair[record["id"]]["evaluation"]["gt_candidate_recalled"])
        and int(artifacts[record["id"]]["best_candidate_pair"][0]) < 0
        for record in records
    )
    wrong = recalled - rejected - selected_correct
    result = {
        "schema_version": "zind-t4-recall-audit-v1",
        "cpu_only": True,
        "split": "val",
        "test_data_accessed": False,
        "inputs": {"frozen_root": str(frozen_root), "val_manifest": str(val_manifest), "pair_count": len(records)},
        "decomposition": {
            "positive": positive,
            "candidate_not_recalled": positive - recalled,
            "candidate_recalled": recalled,
            "matcher_no_match_after_recall": rejected,
            "matcher_wrong_after_recall": wrong,
            "candidate_selected_correct": selected_correct,
            "candidate_recall_ceiling": recalled / positive,
            "s3_rate": selected_correct / positive,
            "candidate_correct_conditional_recalled": selected_correct / recalled,
            "dustbin_rejection_conditional_recalled": rejected / recalled,
            "wrong_selection_conditional_recalled": wrong / recalled,
        },
        "proposal_grid_predeclared": list(PROPOSAL_GRID),
        "proposal_fixed_defaults": FIXED_PROPOSAL_DEFAULTS,
        "proposal_results": proposal_rows,
        "legacy_parity": parity,
        "selected_pareto_config": proposal_config,
        "selected_pareto_config_sha256": proposal_hash,
        "score_audit": score_audit,
    }
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    flat_proposals = [{
        "id": row["id"], "family": row["family"], "mode": row["mode"],
        "threshold": row["threshold"], "peak_top_k": row.get("peak_top_k", 0),
        "max_candidates": row["max_candidates"], "positive_recalled": row["positive_recalled"],
        "positive_recall": row["positive_recall"], "delta": row["recall_delta_count_vs_p0"],
        "candidate_mean": row["candidate_count"]["mean"], "candidate_p95": row["candidate_count"]["p95"],
        "product_mean": row["candidate_product"]["mean"], "product_p95": row["candidate_product"]["p95"],
        "negative_product_mean": row["negative_candidate_product"]["mean"],
        "negative_product_p95": row["negative_candidate_product"]["p95"],
    } for row in proposal_rows]
    _write_csv(output_dir / "proposal_grid.csv", flat_proposals, tuple(flat_proposals[0]))
    sweep = score_audit["threshold_sweep"]
    _write_csv(output_dir / "pair_margin_sweep.csv", sweep, tuple(sweep[0]))
    print(json.dumps({
        "output": str(output_dir / "audit.json"),
        "decomposition": result["decomposition"],
        "legacy_parity": parity,
        "selected": proposal_config,
        "selected_hash": proposal_hash,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
