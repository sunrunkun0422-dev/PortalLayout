"""Trace-only diagnostics for the ZInD circular opening proposal pipeline.

The helpers in this module mirror the production generators but never alter
their return values or defaults.  Ground truth is accepted only by the
separate diagnostic helpers at the bottom of the file.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from models.cross_scene_matcher import (
    circular_interval_iou,
    opening_probabilities_to_intervals,
    recall_topk_opening_intervals,
)


Interval = Tuple[int, int]


def interval_tokens(interval: Sequence[int], token_count: int) -> List[int]:
    """Return the inclusive circular token sequence for one interval."""
    start, end = (int(interval[0]) % token_count, int(interval[1]) % token_count)
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, token_count)) + list(range(0, end + 1))


def interval_center(interval: Sequence[int], token_count: int) -> float:
    tokens = interval_tokens(interval, token_count)
    return float((tokens[0] + 0.5 * (len(tokens) - 1)) % token_count)


def circular_center_error(
    first: Sequence[int], second: Sequence[int], token_count: int
) -> float:
    delta = abs(interval_center(first, token_count) - interval_center(second, token_count))
    return float(min(delta, token_count - delta))


def endpoint_error_tokens(
    first: Sequence[int], second: Sequence[int], token_count: int
) -> float:
    def distance(a: int, b: int) -> int:
        delta = abs((int(a) % token_count) - (int(b) % token_count))
        return min(delta, token_count - delta)

    direct = distance(first[0], second[0]) + distance(first[1], second[1])
    reverse = distance(first[0], second[1]) + distance(first[1], second[0])
    return float(0.5 * min(direct, reverse))


def _components(mask: torch.Tensor) -> List[Interval]:
    token_count = int(mask.numel())
    if not bool(mask.any()):
        return []
    if bool(mask.all()):
        return [(0, token_count - 1)]
    starts = torch.where(mask & ~torch.roll(mask, shifts=1))[0].tolist()
    result: List[Interval] = []
    for start in starts:
        end = int(start)
        while bool(mask[(end + 1) % token_count]):
            end = (end + 1) % token_count
        result.append((int(start), int(end)))
    return result


def _expanded_peak(
    values: torch.Tensor,
    peak: int,
    expansion_threshold: float,
    min_width_tokens: int,
) -> Interval:
    token_count = int(values.numel())
    tokens = {int(peak)}
    cursor = (int(peak) - 1) % token_count
    while cursor not in tokens and float(values[cursor].item()) >= expansion_threshold:
        tokens.add(cursor)
        cursor = (cursor - 1) % token_count
    cursor = (int(peak) + 1) % token_count
    while cursor not in tokens and float(values[cursor].item()) >= expansion_threshold:
        tokens.add(cursor)
        cursor = (cursor + 1) % token_count
    left = (int(peak) - 1) % token_count
    right = (int(peak) + 1) % token_count
    while len(tokens) < int(min_width_tokens) and len(tokens) < token_count:
        while left in tokens:
            left = (left - 1) % token_count
        while right in tokens:
            right = (right + 1) % token_count
        chosen = left if float(values[left].item()) >= float(values[right].item()) else right
        tokens.add(chosen)
    if len(tokens) == token_count:
        return (0, token_count - 1)
    starts = [index for index in tokens if (index - 1) % token_count not in tokens]
    if len(starts) != 1:
        raise ValueError("expanded peak is not one circular interval")
    start = int(starts[0])
    end = start
    while (end + 1) % token_count in tokens:
        end = (end + 1) % token_count
    return start, int(end)


def _interval_record(
    interval: Interval, values: torch.Tensor, *, source: str, rank: int
) -> Dict[str, Any]:
    token_count = int(values.numel())
    tokens = interval_tokens(interval, token_count)
    selected = values[torch.as_tensor(tokens, dtype=torch.long, device=values.device)]
    return {
        "interval": [int(interval[0]), int(interval[1])],
        "wrap": bool(interval[0] > interval[1]),
        "width_tokens": len(tokens),
        "confidence_mean": float(selected.mean().item()),
        "confidence_max": float(selected.max().item()),
        "source": source,
        "rank": int(rank),
    }


def trace_opening_proposals(
    probability: Sequence[float] | torch.Tensor,
    *,
    threshold: float = 0.47,
    min_width_tokens: int = 2,
    max_candidates: int = 12,
    peak_top_k: int = 4,
    peak_min_probability: float = 0.20,
    expansion_threshold: float = 0.12,
    dedup_iou: float = 0.50,
) -> Dict[str, Any]:
    """Return every deterministic proposal stage plus exact production parity."""
    values = torch.as_tensor(probability, dtype=torch.float32).detach().reshape(-1)
    if values.numel() <= 0 or not torch.isfinite(values).all():
        raise ValueError("opening probability must be finite and non-empty")
    token_count = int(values.numel())

    threshold_mask = values >= float(threshold)
    components = _components(threshold_mask)
    component_rows = []
    ranked_components = []
    for interval in components:
        tokens = interval_tokens(interval, token_count)
        score = float(values[torch.as_tensor(tokens, device=values.device)].mean().item())
        ranked_components.append((score, len(tokens), interval[0], interval[1]))
    ranked_components.sort(key=lambda item: (-item[0], -item[1], item[2]))
    for rank, (score, width, start, end) in enumerate(ranked_components, 1):
        row = _interval_record((start, end), values, source="threshold_component", rank=rank)
        row.update(
            passes_min_width=bool(width >= min_width_tokens),
            removed_by_min_width=bool(width < min_width_tokens),
            removed_by_cap=bool(width >= min_width_tokens and rank > max_candidates),
        )
        component_rows.append(row)

    legacy = opening_probabilities_to_intervals(
        values,
        threshold=threshold,
        min_width_tokens=min_width_tokens,
        max_intervals=max_candidates,
    )

    previous = torch.roll(values, shifts=1)
    following = torch.roll(values, shifts=-1)
    peak_mask = (
        (values >= previous)
        & (values >= following)
        & (values >= float(peak_min_probability))
    )
    ranked_peaks = sorted(
        torch.where(peak_mask)[0].tolist(),
        key=lambda index: (-float(values[index].item()), int(index)),
    )
    all_peak_rows = []
    for rank, peak in enumerate(ranked_peaks, 1):
        interval = _expanded_peak(
            values, int(peak), float(expansion_threshold), min_width_tokens
        )
        row = _interval_record(interval, values, source="local_peak", rank=rank)
        row.update(
            peak_index=int(peak),
            peak_probability=float(values[peak].item()),
            accepted=False,
            removed_by_dedup=False,
            dedup_kept_interval=None,
            removed_by_peak_top_k=False,
            removed_by_cap=False,
        )
        all_peak_rows.append(row)

    proposals = list(legacy)
    accepted_fallbacks = 0
    for row in all_peak_rows:
        interval = tuple(row["interval"])
        if accepted_fallbacks >= int(peak_top_k):
            row["removed_by_peak_top_k"] = True
            continue
        if len(proposals) >= int(max_candidates):
            row["removed_by_cap"] = True
            continue
        duplicate = next(
            (
                existing
                for existing in proposals
                if circular_interval_iou(interval, existing, token_count)
                >= float(dedup_iou)
            ),
            None,
        )
        if duplicate is not None:
            row["removed_by_dedup"] = True
            row["dedup_kept_interval"] = [int(duplicate[0]), int(duplicate[1])]
            continue
        proposals.append(interval)
        row["accepted"] = True
        row["final_index"] = len(proposals) - 1
        accepted_fallbacks += 1

    recall = recall_topk_opening_intervals(
        values,
        threshold=threshold,
        min_width_tokens=min_width_tokens,
        max_intervals=max_candidates,
        peak_top_k=peak_top_k,
        peak_min_probability=peak_min_probability,
        expansion_threshold=expansion_threshold,
        dedup_iou=dedup_iou,
    )
    if list(proposals[:max_candidates]) != list(recall):
        raise RuntimeError("trace diverged from production Recall-TopK output")

    # O4 deliberately bypasses threshold, dedup, peak-top-k, and candidate cap.
    # It still uses the production local-peak and expansion definitions.
    raw = [tuple(row["interval"]) for row in all_peak_rows]
    return {
        "schema_version": "zind-candidate-proposal-trace-v1",
        "token_count": token_count,
        "config": {
            "threshold": float(threshold),
            "min_width_tokens": int(min_width_tokens),
            "max_candidates": int(max_candidates),
            "peak_top_k": int(peak_top_k),
            "peak_min_probability": float(peak_min_probability),
            "expansion_threshold": float(expansion_threshold),
            "dedup_iou": float(dedup_iou),
        },
        "threshold_components": component_rows,
        "local_peaks": all_peak_rows,
        "raw_hypotheses": [list(value) for value in raw],
        "legacy_candidates": [list(value) for value in legacy],
        "recall_topk_candidates": [list(value) for value in recall],
        "production_parity": True,
        "gt_available_to_production": False,
    }


def target_interval_diagnostics(
    probability: Sequence[float] | torch.Tensor,
    trace: Mapping[str, Any],
    target: Sequence[int],
    *,
    target_iou: float = 0.30,
) -> Dict[str, Any]:
    """Attach GT-only diagnostics without feeding GT to the generator."""
    values = torch.as_tensor(probability, dtype=torch.float32).reshape(-1)
    token_count = int(values.numel())
    target_tuple = (int(target[0]), int(target[1]))
    tokens = interval_tokens(target_tuple, token_count)
    selected = values[torch.as_tensor(tokens, dtype=torch.long)]

    def rows(name: str) -> List[Mapping[str, Any]]:
        source = trace.get(name, [])
        return source if isinstance(source, list) else []

    def intervals(name: str) -> List[Interval]:
        output = []
        for value in rows(name):
            item = value.get("interval") if isinstance(value, Mapping) else value
            output.append((int(item[0]), int(item[1])))
        return output

    stage_intervals = {
        "raw": intervals("local_peaks"),
        "threshold": intervals("threshold_components"),
        "legacy_final": intervals("legacy_candidates"),
        "recall_final": intervals("recall_topk_candidates"),
    }
    stage = {}
    for name, candidates in stage_intervals.items():
        scored = [
            {
                "interval": [candidate[0], candidate[1]],
                "iou": circular_interval_iou(candidate, target_tuple, token_count),
                "center_error_tokens": circular_center_error(
                    candidate, target_tuple, token_count
                ),
                "endpoint_error_tokens": endpoint_error_tokens(
                    candidate, target_tuple, token_count
                ),
            }
            for candidate in candidates
        ]
        best = max(scored, key=lambda value: value["iou"], default=None)
        stage[name] = {
            "candidate_count": len(candidates),
            "recalled": bool(best is not None and best["iou"] >= target_iou),
            "best": best,
        }
    peak_center = interval_center(target_tuple, token_count)
    peak_rows = rows("local_peaks")
    nearest_peak = min(
        peak_rows,
        key=lambda row: min(
            abs(float(row["peak_index"]) - peak_center),
            token_count - abs(float(row["peak_index"]) - peak_center),
        ),
        default=None,
    )
    return {
        "target_interval": [target_tuple[0], target_tuple[1]],
        "target_wrap": bool(target_tuple[0] > target_tuple[1]),
        "target_width_tokens": len(tokens),
        "target_probability_max": float(selected.max().item()),
        "target_probability_mean": float(selected.mean().item()),
        "target_center": peak_center,
        "nearest_peak": nearest_peak,
        "stages": stage,
    }


def assign_side_reason(
    diagnostic: Mapping[str, Any], trace: Mapping[str, Any], *, target_iou: float = 0.30
) -> str:
    """Assign the first deterministic side-level loss point."""
    stages = diagnostic["stages"]
    if stages["legacy_final"]["recalled"]:
        return "present"

    target = tuple(diagnostic["target_interval"])
    token_count = int(trace["token_count"])
    target_set = set(interval_tokens(target, token_count))
    threshold_rows = trace.get("threshold_components", [])
    overlapping = []
    for row in threshold_rows:
        candidate = tuple(row["interval"])
        candidate_set = set(interval_tokens(candidate, token_count))
        if target_set & candidate_set:
            overlapping.append((row, candidate_set))
    if len(overlapping) > 1:
        return "interval_split"
    if overlapping:
        row, candidate_set = overlapping[0]
        if len(candidate_set) > len(target_set) and len(target_set & candidate_set) / len(target_set) >= 0.5:
            return "interval_merge"
    if not stages["raw"]["recalled"]:
        return "no_opening_signal"
    if not overlapping:
        return "below_threshold"
    row, candidate_set = overlapping[0]
    if diagnostic.get("target_wrap") and not bool(row.get("wrap")):
        return "wrap_failure"
    if row.get("removed_by_min_width"):
        return "interval_expansion_failure"
    if row.get("removed_by_cap"):
        return "candidate_cap_removed"
    for peak in trace.get("local_peaks", []):
        interval = tuple(peak["interval"])
        if circular_interval_iou(interval, target, token_count) < target_iou:
            continue
        if peak.get("removed_by_dedup"):
            return "dedup_removed"
        if peak.get("removed_by_cap") or peak.get("removed_by_peak_top_k"):
            return "candidate_cap_removed"
    return "interval_expansion_failure"


def combine_side_reason(reason_a: str, reason_b: str) -> str:
    """Map two side reasons to the request's mutually-exclusive pair label."""
    if reason_a == "no_opening_signal" or reason_b == "no_opening_signal":
        if reason_a == reason_b == "no_opening_signal":
            return "no_opening_signal_both"
        return "no_opening_signal_A" if reason_a == "no_opening_signal" else "no_opening_signal_B"
    if reason_a == "below_threshold" or reason_b == "below_threshold":
        if reason_a == reason_b == "below_threshold":
            return "below_threshold_both"
        return "below_threshold_A" if reason_a == "below_threshold" else "below_threshold_B"
    priority = (
        "interval_split",
        "interval_merge",
        "interval_expansion_failure",
        "wrap_failure",
        "dedup_removed",
        "candidate_cap_removed",
    )
    for reason in priority:
        if reason in (reason_a, reason_b):
            return reason
    return "side_candidates_present_pair_target_missing"


def finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(child) for child in value)
    return True
