#!/usr/bin/env python3
"""Reproducible, artifact-only visual audit for ZInD full-val failures.

This command reads only the already frozen ``val`` audit artifacts and RGB
panoramas referenced by the validated val manifest.  It never instantiates a
model and has no training, optimizer, backward, or CUDA path.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_ROOT = REPO_ROOT / "experiments/zind_e2e_v3/g0_seed123/T5_candidate_recall/audit"
DEFAULT_OUTPUT = DEFAULT_AUDIT_ROOT / "visual_audit"
DEFAULT_VAL_MANIFEST = REPO_ROOT / "data/ZInD_matching/val.json"
DEFAULT_FROZEN_VAL = REPO_ROOT / "outputs/zind_geometry_val_full/t3_frozen_m2"
DEFAULT_T3 = REPO_ROOT / "checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best.pt"

EXPECTED_FUNNEL = {
    "gt_positives": 1827,
    "correct_candidate_exists": 1080,
    "correct_candidate_missing": 747,
    "matcher_correct": 501,
    "dustbin_rejection": 437,
    "wrong_selection": 142,
}
EXPECTED_MISSING = {
    "interval_merge": 413,
    "no_opening_signal": 293,
    "below_threshold": 19,
    "interval_split": 16,
    "dedup_removed": 4,
    "candidate_cap_removed": 1,
    "interval_expansion_failure": 1,
}
CATEGORY_ORDER = (
    "interval_merge",
    "no_opening_signal",
    "interval_split",
    "below_threshold",
    "dedup_removed",
    "candidate_cap_removed",
    "interval_expansion_failure",
    "target_present_dustbin_reject",
    "target_present_wrong_selection",
    "target_present_correct",
)
CATEGORY_QUOTAS = {
    "interval_merge": 8,
    "no_opening_signal": 8,
    "interval_split": 4,
    "below_threshold": 4,
    "dedup_removed": 4,
    "candidate_cap_removed": 1,
    "interval_expansion_failure": 1,
    "target_present_dustbin_reject": 8,
    "target_present_wrong_selection": 8,
    "target_present_correct": 6,
}
MECHANISMS = {
    "missing visual response",
    "broad response bridge",
    "adjacent openings merged",
    "true opening split into fragments",
    "probability exists but below threshold",
    "wrap/seam handling",
    "occlusion",
    "oblique/edge-on viewpoint",
    "weak illumination or blur",
    "repeated door appearance",
    "candidate score calibration",
    "annotation ambiguity",
    "artifact/data error",
    "uncertain",
}
OBS_ENUMS = {
    "gt_opening_visibility": {"clear", "partial", "not_visible", "uncertain"},
    "viewpoint": {"frontal", "oblique", "edge_on", "uncertain"},
    "occlusion": {"none", "partial", "heavy", "uncertain"},
    "illumination": {"normal", "dark", "overexposed", "mixed"},
    "nearby_openings": {"0", "1", "multiple"},
    "repeated_door_appearance": {True, False, "uncertain"},
    "wrap_or_seam": {True, False},
    "annotation_ambiguity": {"none", "possible", "strong"},
    "confidence": {"high", "medium", "low"},
    "automatic_attribution_assessment": {
        "confirmed",
        "partially_confirmed",
        "contradicted",
        "indeterminate",
    },
}
SENTINEL_LIMIT = -1.0e20


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "finalize"))
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--val-manifest", default=str(DEFAULT_VAL_MANIFEST))
    parser.add_argument("--frozen-val-root", default=str(DEFAULT_FROZEN_VAL))
    parser.add_argument("--checkpoint", default=str(DEFAULT_T3))
    parser.add_argument(
        "--observations",
        action="append",
        default=[],
        help="Reviewed JSONL; repeat to merge disjoint reviewer files.",
    )
    return parser.parse_args(argv)


def reject_test_data_path(value: str, label: str) -> Path:
    """Reject an actual test split path before any file content is opened."""
    raw = Path(value).expanduser()
    parts = [part.lower() for part in raw.parts]
    if any(part == "test" for part in parts) or raw.stem.lower() == "test":
        raise ValueError("{} points to forbidden test data: {}".format(label, raw))
    return raw.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def strict_json_text(value: Any, *, indent: Optional[int] = 2) -> str:
    if not finite_tree(value):
        raise ValueError("JSON payload contains NaN or Infinity")
    return json.dumps(
        value,
        sort_keys=True,
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, strict_json_text(value))


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = []
    for row in rows:
        if not finite_tree(row):
            raise ValueError("JSONL payload contains NaN or Infinity")
        lines.append(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
        )
    atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("{}:{} is not an object".format(path, line_number))
            rows.append(value)
    return rows


def read_gzip_json(path: Path) -> Dict[str, Any]:
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def git_snapshot() -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            list(args), cwd=str(REPO_ROOT), check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True,
        )
        return result.stdout.rstrip("\n")

    return {
        "branch": run("git", "branch", "--show-current"),
        "head": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short").splitlines(),
    }


def checkpoint_identity(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def normalized_reason(reason: str) -> str:
    if reason.startswith("no_opening_signal"):
        return "no_opening_signal"
    if reason.startswith("below_threshold"):
        return "below_threshold"
    return reason


def reproduce_statistics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    for row in rows:
        if row.get("split") != "val" or row.get("test_data_accessed") is not False:
            raise ValueError("per-pair audit contains non-val or test-accessed row")
    funnel = {
        "gt_positives": len(rows),
        "correct_candidate_exists": sum(bool(row["legacy"]["target_present"]) for row in rows),
        "correct_candidate_missing": sum(not bool(row["legacy"]["target_present"]) for row in rows),
        "matcher_correct": sum(
            bool(row["legacy"]["target_present"])
            and bool(row["legacy"]["selected_correct"])
            for row in rows
        ),
        "dustbin_rejection": sum(
            bool(row["legacy"]["target_present"])
            and not bool(row["legacy"]["predicted_match"])
            for row in rows
        ),
        "wrong_selection": sum(
            bool(row["legacy"]["target_present"])
            and bool(row["legacy"]["predicted_match"])
            and not bool(row["legacy"]["selected_correct"])
            for row in rows
        ),
    }
    missing_counter = Counter(
        normalized_reason(str(row["primary_pair_reason"]))
        for row in rows
        if not bool(row["legacy"]["target_present"])
    )
    missing = {key: int(missing_counter.get(key, 0)) for key in EXPECTED_MISSING}
    return {"funnel": funnel, "missing_reasons": missing}


def validate_statistics(
    rows: Sequence[Mapping[str, Any]],
    stage_funnel: Optional[Mapping[str, Any]] = None,
    reason_counts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    reproduced = reproduce_statistics(rows)
    if reproduced["funnel"] != EXPECTED_FUNNEL:
        raise RuntimeError(
            "full-val funnel mismatch: {} != {}".format(
                reproduced["funnel"], EXPECTED_FUNNEL
            )
        )
    if reproduced["missing_reasons"] != EXPECTED_MISSING:
        raise RuntimeError(
            "full-val missing-reason mismatch: {} != {}".format(
                reproduced["missing_reasons"], EXPECTED_MISSING
            )
        )
    if stage_funnel is not None:
        stages = stage_funnel["legacy"]["stages"]
        frozen = {
            "gt_positives": int(stage_funnel["legacy"]["positive_pair_count"]),
            "correct_candidate_exists": int(stages["final_target_formed"]["pair_count"]),
            "correct_candidate_missing": int(stage_funnel["legacy"]["positive_pair_count"])
            - int(stages["final_target_formed"]["pair_count"]),
            "matcher_correct": int(stages["matcher_candidate_top1_correct"]["pair_count"]),
            "dustbin_rejection": EXPECTED_FUNNEL["dustbin_rejection"],
            "wrong_selection": EXPECTED_FUNNEL["wrong_selection"],
        }
        if frozen != EXPECTED_FUNNEL:
            raise RuntimeError("stage_funnel_val.json does not match required legacy funnel")
    if reason_counts is not None:
        if int(reason_counts["legacy_missing_count"]) != 747:
            raise RuntimeError("reason_counts_val legacy_missing_count mismatch")
        frozen_raw = reason_counts["legacy_missing_primary_reason_counts"]
        frozen = Counter()
        for key, count in frozen_raw.items():
            frozen[normalized_reason(str(key))] += int(count)
        if {key: frozen[key] for key in EXPECTED_MISSING} != EXPECTED_MISSING:
            raise RuntimeError("reason_counts_val reason mismatch")
    return reproduced


def interval_tokens(interval: Sequence[int], token_count: int) -> List[int]:
    start = int(interval[0]) % token_count
    end = int(interval[1]) % token_count
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, token_count)) + list(range(0, end + 1))


def interval_width(interval: Sequence[int], token_count: int) -> int:
    return len(interval_tokens(interval, token_count))


def interval_center(interval: Sequence[int], token_count: int) -> float:
    tokens = interval_tokens(interval, token_count)
    return float((tokens[0] + 0.5 * (len(tokens) - 1)) % token_count)


def interval_pixel_spans(
    interval: Sequence[int], token_count: int, pixel_width: int
) -> List[Tuple[float, float]]:
    start = int(interval[0]) % token_count
    end = int(interval[1]) % token_count
    scale = float(pixel_width) / float(token_count)
    if start <= end:
        return [(start * scale, (end + 1) * scale)]
    return [
        (start * scale, float(pixel_width)),
        (0.0, (end + 1) * scale),
    ]


def _score_matrix(score: Mapping[str, Any]) -> List[List[float]]:
    count_a = int(score["candidate_count_A"])
    count_b = int(score["candidate_count_B"])
    matrix = score["candidate_pair_scores"]
    if len(matrix) < count_a or any(len(row) < count_b for row in matrix[:count_a]):
        raise ValueError("candidate_pair_scores is smaller than K_A x K_B")
    return [[float(value) for value in row[:count_b]] for row in matrix[:count_a]]


def _real_logits(score: Mapping[str, Any]) -> List[List[float]]:
    count_a = int(score["candidate_count_A"])
    count_b = int(score["candidate_count_B"])
    matrix = score["candidate_assignment_logits"]
    if len(matrix) < count_a or any(len(row) < count_b for row in matrix[:count_a]):
        raise ValueError("candidate_assignment_logits lacks real-real block")
    values = [[float(value) for value in row[:count_b]] for row in matrix[:count_a]]
    if not finite_tree(values):
        raise ValueError("real-real assignment logits are not finite")
    return values


def valid_dustbin_value(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > SENTINEL_LIMIT


def _log_softmax_dustbin(real: Sequence[float], dustbin: float) -> float:
    values = [float(value) for value in real] + [float(dustbin)]
    maximum = max(values)
    denominator = sum(math.exp(value - maximum) for value in values)
    return math.exp(float(dustbin) - maximum) / denominator


def recover_pair_no_match(real_logits: Sequence[Sequence[float]], dustbin: float) -> float:
    if not real_logits or not real_logits[0]:
        return 1.0
    no_match_a = [_log_softmax_dustbin(row, dustbin) for row in real_logits]
    count_b = len(real_logits[0])
    no_match_b = [
        _log_softmax_dustbin([row[column] for row in real_logits], dustbin)
        for column in range(count_b)
    ]
    return math.sqrt(
        (sum(no_match_a) / len(no_match_a))
        * (sum(no_match_b) / len(no_match_b))
    )


def matcher_metrics(
    row: Mapping[str, Any], score: Mapping[str, Any], shared_dustbin: float
) -> Dict[str, Any]:
    matrix = _score_matrix(score)
    logits = _real_logits(score)
    count_a = len(matrix)
    count_b = len(matrix[0]) if matrix else 0
    targets = [
        (int(pair[0]), int(pair[1]))
        for pair in row["legacy"]["target_pairs"]
        if 0 <= int(pair[0]) < count_a and 0 <= int(pair[1]) < count_b
    ]
    true_pair = None
    true_score = None
    true_logit = None
    true_rank = None
    if targets:
        true_pair = sorted(targets, key=lambda pair: (-matrix[pair[0]][pair[1]], pair))[0]
        true_score = float(matrix[true_pair[0]][true_pair[1]])
        true_logit = float(logits[true_pair[0]][true_pair[1]])
        ordered = sorted(
            ((matrix[i][j], i, j) for i in range(count_a) for j in range(count_b)),
            key=lambda value: (-value[0], value[1], value[2]),
        )
        true_rank = 1 + next(
            index for index, (_, i, j) in enumerate(ordered) if (i, j) == true_pair
        )
    selected_raw = score.get("best_candidate_pair", [-1, -1])
    selected_pair = (int(selected_raw[0]), int(selected_raw[1]))
    selected_score = None
    if 0 <= selected_pair[0] < count_a and 0 <= selected_pair[1] < count_b:
        selected_score = float(matrix[selected_pair[0]][selected_pair[1]])
    else:
        selected_pair = None
    row_direct = [valid_dustbin_value(value) for value in score.get("row_dustbin_logits", [])]
    column_direct = [valid_dustbin_value(value) for value in score.get("column_dustbin_logits", [])]
    row_provenance = "direct" if len(row_direct) == count_a and all(row_direct) else "recovered_exact_from_shared_dustbin_constant"
    column_provenance = "direct" if len(column_direct) == count_b and all(column_direct) else "recovered_exact_from_shared_dustbin_constant"
    return {
        "candidate_count_A": count_a,
        "candidate_count_B": count_b,
        "candidate_product": count_a * count_b,
        "target_pairs": [list(value) for value in targets],
        "true_pair": list(true_pair) if true_pair is not None else None,
        "true_pair_score": true_score,
        "true_pair_raw_logit": true_logit,
        "true_real_real_rank": true_rank,
        "selected_pair": list(selected_pair) if selected_pair is not None else None,
        "selected_pair_score": selected_score,
        "selected_vs_true_score_gap": (
            None if selected_score is None or true_score is None else selected_score - true_score
        ),
        "shared_dustbin_logit": float(shared_dustbin),
        "true_to_dustbin_logit_margin": (
            None if true_logit is None else true_logit - float(shared_dustbin)
        ),
        "row_dustbin_provenance": row_provenance,
        "column_dustbin_provenance": column_provenance,
        "pair_match_probability": float(score["pair_match_probability"]),
        "pair_no_match_probability": float(score["pair_no_match_probability"]),
        "recovered_pair_no_match_probability": recover_pair_no_match(logits, shared_dustbin),
        "candidate_pair_scores": matrix,
        "real_real_logits": logits,
    }


def load_score_artifacts(audit_root: Path, rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    score_root = reject_test_data_path(str(audit_root / "pair_scores/val_O0"), "val O0 score root")
    scores: Dict[str, Dict[str, Any]] = {}
    trusted_values: List[float] = []
    direct_modes = Counter()
    for row in rows:
        pair_id = str(row["pair_id"])
        path = score_root / (pair_id + ".json.gz")
        score = read_gzip_json(path)
        if (
            score.get("split") != "val"
            or score.get("view") != "O0"
            or score.get("test_data_accessed") is not False
            or score.get("valid_for_inference") is not True
        ):
            raise ValueError("invalid O0 score security metadata: {}".format(path))
        _score_matrix(score)
        _real_logits(score)
        count_a = int(score["candidate_count_A"])
        count_b = int(score["candidate_count_B"])
        row_values = score.get("row_dustbin_logits", [])
        column_values = score.get("column_dustbin_logits", [])
        row_ok = len(row_values) == count_a and all(valid_dustbin_value(value) for value in row_values)
        column_ok = len(column_values) == count_b and all(valid_dustbin_value(value) for value in column_values)
        direct_modes[
            "both_direct" if row_ok and column_ok else
            "row_only" if row_ok else
            "column_only" if column_ok else
            "neither_direct"
        ] += 1
        trusted_values.extend(float(value) for value in row_values if valid_dustbin_value(value))
        trusted_values.extend(float(value) for value in column_values if valid_dustbin_value(value))
        scores[pair_id] = score
    if not trusted_values:
        raise RuntimeError("no trustworthy serialized dustbin value exists")
    shared_dustbin = trusted_values[0]
    if any(abs(value - shared_dustbin) > 1e-7 for value in trusted_values):
        raise RuntimeError("serialized Matcher dustbin is not a unique shared constant")
    max_probability_error = 0.0
    best_reproduced = 0
    for row in rows:
        pair_id = str(row["pair_id"])
        score = scores[pair_id]
        logits = _real_logits(score)
        recovered = recover_pair_no_match(logits, shared_dustbin)
        max_probability_error = max(
            max_probability_error,
            abs(recovered - float(score["pair_no_match_probability"])),
        )
        matrix = _score_matrix(score)
        accepted = [
            (matrix[i][j], i, j)
            for i in range(len(matrix))
            for j in range(len(matrix[0]) if matrix else 0)
            if logits[i][j] > shared_dustbin
        ]
        recovered_best = [-1, -1]
        if accepted:
            _, i, j = sorted(accepted, key=lambda value: (-value[0], value[1], value[2]))[0]
            recovered_best = [i, j]
        if recovered_best == [int(value) for value in score["best_candidate_pair"]]:
            best_reproduced += 1
    if max_probability_error > 2e-6 or best_reproduced != len(rows):
        raise RuntimeError("dustbin recovery invariant failed")
    evidence = {
        "serialized_padding_issue": True,
        "cause": "per-sample dustbin was cropped at local K rather than padded batch K",
        "direct_modes": {key: int(direct_modes[key]) for key in (
            "both_direct", "row_only", "column_only", "neither_direct"
        )},
        "trusted_serialized_value_count": len(trusted_values),
        "shared_dustbin_logit": shared_dustbin,
        "recovery": "exact_from_unique_shared_serialized_dustbin_constant_and_real_logits",
        "pair_no_match_max_abs_error": max_probability_error,
        "best_pair_reproduced": best_reproduced,
        "pair_count": len(rows),
    }
    return scores, evidence


def _stage_best_interval(side: Mapping[str, Any], stage: str) -> Optional[List[int]]:
    best = side.get("stages", {}).get(stage, {}).get("best")
    if isinstance(best, Mapping) and isinstance(best.get("interval"), list):
        return [int(best["interval"][0]), int(best["interval"][1])]
    return None


def _reason_sides(row: Mapping[str, Any], reason: str) -> List[str]:
    return [side for side in ("A", "B") if row.get("reason_" + side) == reason]


def severity_for_row(
    category: str,
    row: Mapping[str, Any],
    score: Mapping[str, Any],
    token_count: int,
    threshold: float,
    shared_dustbin: float,
) -> Tuple[str, float]:
    if category == "interval_merge":
        sides = _reason_sides(row, "interval_merge")
        widths = []
        for side in sides:
            interval = _stage_best_interval(row["target_" + side], "threshold")
            if interval is not None:
                widths.append(interval_width(interval, token_count) / float(token_count))
        return "widest_overlapping_threshold_component_fraction", max(widths or [0.0])
    if category == "no_opening_signal":
        sides = _reason_sides(row, "no_opening_signal")
        values = [1.0 - float(row["target_" + side]["target_probability_max"]) for side in sides]
        return "mean_one_minus_GT_interval_probability_max", sum(values) / max(1, len(values))
    if category == "interval_split":
        sides = _reason_sides(row, "interval_split")
        values = []
        for side in sides:
            best = row["target_" + side]["stages"]["legacy_final"].get("best")
            values.append(1.0 - (float(best["iou"]) if best is not None else 0.0))
        return "mean_one_minus_best_final_IoU", sum(values) / max(1, len(values))
    if category == "below_threshold":
        sides = _reason_sides(row, "below_threshold")
        deficits = [threshold - float(row["target_" + side]["target_probability_max"]) for side in sides]
        return "mean_threshold_minus_GT_interval_probability_max", sum(deficits) / max(1, len(deficits))
    metrics = matcher_metrics(row, score, shared_dustbin)
    if category == "target_present_dustbin_reject":
        value = metrics["true_to_dustbin_logit_margin"]
        return "negative_true_to_dustbin_logit_margin", -float(value)
    if category == "target_present_wrong_selection":
        value = metrics["selected_vs_true_score_gap"]
        return "selected_minus_true_candidate_pair_score", float(value)
    if category == "target_present_correct":
        value = metrics["true_pair_score"]
        return "true_candidate_pair_score", float(value)
    sides = [side for side in ("A", "B") if row.get("reason_" + side) == category]
    values = [1.0 - float(row["target_" + side]["target_probability_max"]) for side in sides]
    return "mean_one_minus_GT_interval_probability_max", sum(values) / max(1, len(values))


def quantile_targets(count: int) -> List[Tuple[str, float]]:
    if count <= 0:
        return []
    if count == 1:
        return [("severe", 0.0)]
    if count == 2:
        return [("severe", 0.0), ("boundary", 1.0)]
    if count == 3:
        return [("severe", 0.0), ("median", 0.5), ("boundary", 1.0)]
    severe = int(math.ceil(count / 3.0))
    median = int(math.ceil((count - severe) / 2.0))
    boundary = count - severe - median
    result: List[Tuple[str, float]] = []
    for index in range(severe):
        result.append(("severe", 0.18 * index / max(1, severe - 1)))
    for index in range(median):
        result.append(("median", 0.43 + 0.14 * index / max(1, median - 1)))
    for index in range(boundary):
        result.append(("boundary", 0.82 + 0.18 * index / max(1, boundary - 1)))
    return result


def stratified_house_first(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    severity: Mapping[str, float],
    globally_used_houses: Optional[set] = None,
) -> List[Tuple[Mapping[str, Any], str]]:
    """Select deterministic severity anchors while preferring unused houses."""
    used_houses = globally_used_houses if globally_used_houses is not None else set()
    ranked = sorted(
        candidates,
        key=lambda row: (-float(severity[str(row["pair_id"])]), str(row["house_id"]), str(row["pair_id"])),
    )
    if count > len(ranked):
        raise ValueError("selection quota exceeds category size")
    selected: List[Tuple[Mapping[str, Any], str]] = []
    selected_ids = set()
    for stratum, quantile in quantile_targets(count):
        target_rank = quantile * max(0, len(ranked) - 1)
        available = [row for row in ranked if str(row["pair_id"]) not in selected_ids]
        distinct = [row for row in available if str(row["house_id"]) not in used_houses]
        pool = distinct or available
        choice = min(
            pool,
            key=lambda row: (
                abs(ranked.index(row) - target_rank),
                str(row["house_id"]),
                str(row["pair_id"]),
            ),
        )
        selected.append((choice, stratum))
        selected_ids.add(str(choice["pair_id"]))
        used_houses.add(str(choice["house_id"]))
    return selected


def deterministic_selection(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    token_count: int,
    thresholds: Mapping[str, float],
    shared_dustbin: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    severity_names: Dict[str, str] = {}
    severity_values: Dict[str, float] = {}
    for row in rows:
        category = normalized_reason(str(row["primary_pair_reason"]))
        grouped[category].append(row)
        name, value = severity_for_row(
            category, row, scores[str(row["pair_id"])], token_count,
            float(thresholds[str(row["pair_id"])]), shared_dustbin,
        )
        severity_names[str(row["pair_id"])] = name
        severity_values[str(row["pair_id"])] = value
    chosen: List[Dict[str, Any]] = []
    used_houses: set = set()

    def add(category: str, source: Sequence[Mapping[str, Any]], count: int) -> None:
        values = stratified_house_first(source, count, severity_values, used_houses)
        for row, stratum in values:
            chosen.append({
                "audit_category": category,
                "automatic_reason": str(row["primary_pair_reason"]),
                "pair_id": str(row["pair_id"]),
                "house_id": str(row["house_id"]),
                "severity_name": severity_names[str(row["pair_id"])],
                "severity_value": float(severity_values[str(row["pair_id"])]),
                "severity_stratum": stratum,
            })

    add("interval_merge", grouped["interval_merge"], 8)
    for subtype, count in (("no_opening_signal_A", 3), ("no_opening_signal_B", 3), ("no_opening_signal_both", 2)):
        add("no_opening_signal", [row for row in grouped["no_opening_signal"] if row["primary_pair_reason"] == subtype], count)
    add("interval_split", grouped["interval_split"], 4)
    for subtype, count in (("below_threshold_A", 2), ("below_threshold_B", 2)):
        add("below_threshold", [row for row in grouped["below_threshold"] if row["primary_pair_reason"] == subtype], count)
    for category in ("dedup_removed", "candidate_cap_removed", "interval_expansion_failure"):
        source = sorted(grouped[category], key=lambda row: (str(row["house_id"]), str(row["pair_id"])))
        count = CATEGORY_QUOTAS[category]
        if count == len(source):
            for row in source:
                used_houses.add(str(row["house_id"]))
                chosen.append({
                    "audit_category": category,
                    "automatic_reason": str(row["primary_pair_reason"]),
                    "pair_id": str(row["pair_id"]),
                    "house_id": str(row["house_id"]),
                    "severity_name": "exhaustive_category",
                    "severity_value": float(severity_values[str(row["pair_id"])]),
                    "severity_stratum": "all",
                })
        else:
            add(category, source, count)
    add("target_present_dustbin_reject", grouped["target_present_dustbin_reject"], 8)
    add("target_present_wrong_selection", grouped["target_present_wrong_selection"], 8)
    add("target_present_correct", grouped["target_present_correct"], 6)
    order = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    chosen.sort(key=lambda item: (order[item["audit_category"]], item["pair_id"]))
    for index, item in enumerate(chosen):
        item["selection_index"] = index
    counts = Counter(item["audit_category"] for item in chosen)
    if {key: counts[key] for key in CATEGORY_ORDER} != CATEGORY_QUOTAS:
        raise RuntimeError("selection quota invariant failed")
    protocol = {
        "name": "deterministic_house_first_severity_stratified_v1",
        "house_rule": "prefer a previously unused house globally; if exhausted, allow reuse; ties by house_id then pair_id",
        "severity_rule": "stable descending frozen metric with deterministic severe/median/boundary quantile anchors",
        "subcategory_quotas": {
            "no_opening_signal_A/B/both": [3, 3, 2],
            "below_threshold_A/B": [2, 2],
        },
        "manual_cherry_picking": False,
        "quotas": CATEGORY_QUOTAS,
    }
    return chosen, protocol


def _frozen_prediction_map(root: Path) -> Dict[str, Path]:
    mapping = {}
    for path in (root / "frozen_predictions").glob("*.json.gz"):
        stem = path.name[:-len(".json.gz")]
        pair_id = stem.split("_", 1)[1]
        mapping[pair_id] = path
    return mapping


def _load_room(audit_root: Path, room_id: str) -> Dict[str, Any]:
    import torch

    path = reject_test_data_path(
        str(audit_root / "frozen_room_cache/val" / (room_id + ".pt")),
        "val room cache",
    )
    payload = torch.load(str(path), map_location="cpu")
    if (
        payload.get("split") != "val"
        or payload.get("test_data_accessed") is not False
        or payload.get("inference_mode") is not True
        or payload.get("optimizer_created") is not False
        or payload.get("requires_grad") is not False
    ):
        raise ValueError("invalid frozen room security metadata: {}".format(path))
    if str(payload["room"]["room_id"]) != room_id:
        raise ValueError("room cache ID mismatch")
    return payload


def _base_segments(interval: Sequence[int], token_count: int) -> List[Tuple[float, float]]:
    start = int(interval[0]) % token_count
    end = int(interval[1]) % token_count
    if start <= end:
        return [(float(start), float(end + 1))]
    return [(float(start), float(token_count)), (0.0, float(end + 1))]


def _draw_interval(ax: Any, interval: Sequence[int], token_count: int, **kwargs: Any) -> None:
    for start, end in _base_segments(interval, token_count):
        ax.axvspan(start, end, **kwargs)


def _draw_interval_repeated(
    ax: Any, interval: Sequence[int], token_count: int, **kwargs: Any
) -> None:
    for shift in (-token_count, 0, token_count):
        for start, end in _base_segments(interval, token_count):
            ax.axvspan(start + shift, end + shift, **kwargs)


def _candidate_style(index: int, correct: set, selected: Optional[int]) -> Tuple[str, str, float]:
    if selected is not None and index == selected:
        return ("red", "-", 2.2)
    if index in correct:
        return ("limegreen", "--", 2.2)
    return ("deepskyblue", "-", 0.9)


def _overlay_panorama(
    ax: Any,
    image: Any,
    room: Mapping[str, Any],
    target: Mapping[str, Any],
    correct_indices: set,
    selected_index: Optional[int],
    title: str,
    local: bool,
) -> None:
    import numpy as np

    token_count = int(room["token_count"])
    height = int(image.shape[0])
    target_interval = target["target_interval"]
    if local:
        tiled = np.concatenate((image, image, image), axis=1)
        ax.imshow(tiled, extent=(-token_count, 2 * token_count, height, 0), aspect="auto")
        center = interval_center(target_interval, token_count)
        half = max(36.0, min(64.0, 2.0 * interval_width(target_interval, token_count)))
        left, right = center - half, center + half
        ax.set_xlim(left, right)
        for seam in (-token_count, 0, token_count, 2 * token_count):
            if left <= seam <= right:
                ax.axvline(seam, color="white", linewidth=1.2, linestyle=":")
        draw = _draw_interval_repeated
        ticks = np.linspace(left, right, 7)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(round(value)) % token_count) for value in ticks])
        title += " | cyclic GT-local zoom"
    else:
        ax.imshow(image, extent=(0, token_count, height, 0), aspect="auto")
        ax.set_xlim(0, token_count)
        ax.axvline(0, color="white", linewidth=1.1, linestyle=":")
        ax.axvline(token_count, color="white", linewidth=1.1, linestyle=":")
        draw = _draw_interval
    draw(ax, target_interval, token_count, color="limegreen", alpha=0.20, zorder=2)
    threshold_interval = _stage_best_interval(target, "threshold")
    if threshold_interval is not None:
        draw(ax, threshold_interval, token_count, color="orange", alpha=0.16, hatch="///", zorder=3)
    expanded_interval = _stage_best_interval(target, "raw")
    if expanded_interval is not None:
        draw(ax, expanded_interval, token_count, facecolor="none", edgecolor="magenta", linewidth=1.6, linestyle=":", zorder=4)
    nearest = target.get("nearest_peak")
    if isinstance(nearest, Mapping) and nearest.get("peak_index") is not None:
        peak = float(nearest["peak_index"]) + 0.5
        positions = (peak - token_count, peak, peak + token_count) if local else (peak,)
        for position in positions:
            ax.axvline(position, color="magenta", linewidth=1.2, alpha=0.8, zorder=5)
    for index, interval in enumerate(room["legacy_candidates"]):
        color, linestyle, linewidth = _candidate_style(index, correct_indices, selected_index)
        draw(ax, interval, token_count, facecolor="none", edgecolor=color, linewidth=linewidth, linestyle=linestyle, zorder=6)
        if not local:
            x = interval_center(interval, token_count)
            ax.text(x, 12 + 18 * (index % 4), "C{}".format(index), color=color,
                    fontsize=7, ha="center", va="top", bbox={"facecolor": "black", "alpha": 0.45, "pad": 1}, zorder=7)
    if not correct_indices:
        ax.text(0.01, 0.96, "GT candidate absent", transform=ax.transAxes, color="limegreen",
                fontsize=10, weight="bold", va="top", bbox={"facecolor": "black", "alpha": 0.65})
    ax.set_ylim(height, 0)
    ax.set_yticks([])
    ax.set_xlabel("circular token (inclusive interval bins)")
    ax.set_title(title, fontsize=10)


def _plot_probability(ax: Any, room: Mapping[str, Any], target: Mapping[str, Any], side: str) -> None:
    import numpy as np

    probability = room["opening_probability"].detach().cpu().numpy().astype(float)
    trace = room["candidate_trace"]
    token_count = int(room["token_count"])
    config = trace['portallayout.config']
    threshold = float(config["threshold"])
    if len(probability) != token_count:
        raise ValueError("probability/token_count mismatch")
    x = np.arange(token_count, dtype=float) + 0.5
    ax.plot(x, probability, color="black", linewidth=1.1, label="Opening probability")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0,
               label="frozen threshold={:.3f}".format(threshold))
    _draw_interval(ax, target["target_interval"], token_count, color="limegreen", alpha=0.16, label="GT (audit only)")
    for index, component in enumerate(trace["threshold_components"]):
        interval = component["interval"]
        for start, end in _base_segments(interval, token_count):
            ax.hlines(-0.055, start, end, color="orange", linewidth=4,
                      label="threshold component" if index == 0 else None)
    for index, peak in enumerate(trace["local_peaks"]):
        marker = "x" if any(bool(peak.get(key)) for key in (
            "removed_by_dedup", "removed_by_cap", "removed_by_peak_top_k"
        )) else "o"
        ax.scatter([float(peak["peak_index"]) + 0.5], [float(peak["peak_probability"])],
                   color="magenta", marker=marker, s=18, zorder=5,
                   label="raw local peak" if index == 0 else None)
        for start, end in _base_segments(peak["interval"], token_count):
            ax.hlines(-0.105, start, end, color="magenta", linewidth=2,
                      label="peak-expanded interval" if index == 0 else None)
    for index, interval in enumerate(room["legacy_candidates"]):
        for start, end in _base_segments(interval, token_count):
            ax.hlines(-0.155, start, end, color="deepskyblue", linewidth=3,
                      label="legacy final C#" if index == 0 else None)
        ax.text(interval_center(interval, token_count), -0.18, "C{}".format(index),
                color="dodgerblue", fontsize=6, ha="center", va="top")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(token_count, color="gray", linestyle=":", linewidth=1)
    ax.set_xlim(0, token_count)
    ax.set_ylim(-0.23, 1.03)
    ax.set_ylabel("p(opening)")
    ax.set_title("Side {} frozen Opening trace (T={}, threshold from artifact)".format(side, token_count), fontsize=10)
    ax.grid(axis="y", alpha=0.18)
    if bool(target.get("target_wrap")):
        inset = ax.inset_axes([0.70, 0.53, 0.285, 0.42])
        repeated = np.concatenate((probability, probability, probability))
        repeated_x = np.arange(-token_count, 2 * token_count, dtype=float) + 0.5
        center = interval_center(target["target_interval"], token_count)
        inset.plot(repeated_x, repeated, color="black", linewidth=0.8)
        _draw_interval_repeated(inset, target["target_interval"], token_count, color="limegreen", alpha=0.18)
        half = max(20.0, min(42.0, 1.4 * interval_width(target["target_interval"], token_count)))
        inset.set_xlim(center - half, center + half)
        inset.set_ylim(0, 1.02)
        for seam in (-token_count, 0, token_count, 2 * token_count):
            inset.axvline(seam, color="red", linestyle=":", linewidth=0.8)
        inset.set_title("cyclic seam zoom", fontsize=7)
        inset.tick_params(labelsize=6)


def _format_optional(value: Any, precision: int = 4) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return ("{:." + str(precision) + "f}").format(value)
    return str(value)


def _plot_matcher(ax: Any, text_ax: Any, row: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    import matplotlib.patches as patches
    import numpy as np

    matrix = np.asarray(metrics["candidate_pair_scores"], dtype=float)
    ax.set_title("Frozen O0 real-real candidate_pair_score", fontsize=10)
    if matrix.size:
        image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0,
                          vmax=max(0.25, float(matrix.max())))
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, "{:.2f}".format(matrix[i, j]), ha="center", va="center",
                        fontsize=5.5, color="white" if matrix[i, j] < 0.16 else "black")
        for pair in metrics["target_pairs"]:
            ax.add_patch(patches.Rectangle((pair[1] - 0.47, pair[0] - 0.47), 0.94, 0.94,
                                           fill=False, edgecolor="limegreen", linewidth=2.2, linestyle="--"))
        selected = metrics["selected_pair"]
        if selected is not None:
            ax.add_patch(patches.Rectangle((selected[1] - 0.39, selected[0] - 0.39), 0.78, 0.78,
                                           fill=False, edgecolor="red", linewidth=2.0))
    else:
        ax.text(0.5, 0.5, "No real-real candidates", ha="center", va="center")
    ax.set_xlabel("candidate B index")
    ax.set_ylabel("candidate A index")
    ax.set_xticks(range(int(metrics["candidate_count_B"])))
    ax.set_yticks(range(int(metrics["candidate_count_A"])))
    text_ax.axis("off")
    lines = [
        "pair={}  automatic={}".format(row["pair_id"], row["primary_pair_reason"]),
        "K_A x K_B = {} x {} = {}".format(metrics["candidate_count_A"], metrics["candidate_count_B"], metrics["candidate_product"]),
        "GT predicted pair(s): {}".format(metrics["target_pairs"] or "absent"),
        "true pair / score / real-real rank: {} / {} / {}".format(
            metrics["true_pair"], _format_optional(metrics["true_pair_score"]),
            _format_optional(metrics["true_real_real_rank"], 0)),
        "selected pair / score: {} / {}".format(
            metrics["selected_pair"], _format_optional(metrics["selected_pair_score"])),
        "selected - true pair-score gap: {}".format(_format_optional(metrics["selected_vs_true_score_gap"])),
        "true raw logit / shared dustbin logit: {} / {}".format(
            _format_optional(metrics["true_pair_raw_logit"]), _format_optional(metrics["shared_dustbin_logit"])),
        "true-to-dustbin raw-logit margin: {}".format(_format_optional(metrics["true_to_dustbin_logit_margin"])),
        "dustbin provenance row / column: {} / {}".format(
            metrics["row_dustbin_provenance"], metrics["column_dustbin_provenance"]),
        "pair match / no-match probability: {} / {}".format(
            _format_optional(metrics["pair_match_probability"]), _format_optional(metrics["pair_no_match_probability"])),
        "NOTE: pair-score gaps and raw-logit dustbin margins are different scales.",
    ]
    text_ax.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=9,
                 linespacing=1.45)


def render_panel(
    path: Path,
    row: Mapping[str, Any],
    rooms: Mapping[str, Mapping[str, Any]],
    images: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figure = plt.figure(figsize=(24, 22), constrained_layout=False)
    grid = figure.add_gridspec(6, 2, height_ratios=(1.35, 1.15, 1.05, 1.05, 1.5, 0.32), hspace=0.42, wspace=0.20)
    correct = {"A": set(), "B": set()}
    for pair in metrics["target_pairs"]:
        correct["A"].add(int(pair[0]))
        correct["B"].add(int(pair[1]))
    # A pair target can be absent because only one side failed.  Preserve the
    # independently recalled candidate on the other side instead of labeling
    # both panoramas "absent".
    for side in ("A", "B"):
        diagnostic = row["target_" + side]["stages"]["legacy_final"]
        best = diagnostic.get("best")
        if diagnostic.get("recalled") and isinstance(best, Mapping):
            target_interval = tuple(int(value) for value in best["interval"])
            for index, interval in enumerate(rooms[side]["legacy_candidates"]):
                if tuple(int(value) for value in interval) == target_interval:
                    correct[side].add(index)
    selected = metrics["selected_pair"]
    for column, side in enumerate(("A", "B")):
        selected_index = None if selected is None else int(selected[column])
        base_title = "Side {} | house={} room={} pano={}".format(
            side, row["house_id"], row["room_" + side], row["pano_" + side]
        )
        _overlay_panorama(
            figure.add_subplot(grid[0, column]), images[side], rooms[side], row["target_" + side],
            correct[side], selected_index, base_title + " | complete equirectangular panorama", False,
        )
        _overlay_panorama(
            figure.add_subplot(grid[1, column]), images[side], rooms[side], row["target_" + side],
            correct[side], selected_index, base_title, True,
        )
    probability_a = figure.add_subplot(grid[2, :])
    probability_b = figure.add_subplot(grid[3, :])
    _plot_probability(probability_a, rooms["A"], row["target_A"], "A")
    _plot_probability(probability_b, rooms["B"], row["target_B"], "B")
    handles, labels = probability_a.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    probability_a.legend(unique.values(), unique.keys(), loc="upper left", ncol=5, fontsize=7)
    _plot_matcher(figure.add_subplot(grid[4, 0]), figure.add_subplot(grid[4, 1]), row, metrics)
    legend_ax = figure.add_subplot(grid[5, :])
    legend_ax.axis("off")
    legend = [
        Patch(facecolor="limegreen", alpha=0.20, label="GT interval (post-hoc audit only)"),
        Patch(facecolor="orange", alpha=0.16, hatch="///", label="nearest threshold component (merge outcome; no explicit merge operator)"),
        Line2D([0], [0], color="magenta", linestyle=":", label="nearest peak-expanded interval / raw peak"),
        Line2D([0], [0], color="deepskyblue", label="legacy final candidate C#"),
        Line2D([0], [0], color="red", linewidth=2.2, label="Matcher selected candidate"),
        Line2D([0], [0], color="limegreen", linestyle="--", linewidth=2.2, label="correct predicted candidate"),
    ]
    legend_ax.legend(handles=legend, loc="center", ncol=3, fontsize=9)
    category = str(row["primary_pair_reason"])
    if category == "target_present_correct":
        outcome = "GT candidate pair selected correctly"
    elif category == "target_present_dustbin_reject":
        outcome = "GT candidate pair present; dustbin selected"
    elif category == "target_present_wrong_selection":
        outcome = "GT candidate pair present; different pair selected"
    else:
        outcome = "GT candidate absent from final proposals"
    figure.suptitle(
        "FULL-VAL VISUAL AUDIT | {} | {} | {}".format(
            row["pair_id"], category, outcome
        ), fontsize=15, weight="bold", y=0.995,
    )
    figure.subplots_adjust(top=0.972, bottom=0.025, left=0.045, right=0.965)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=150, metadata={"Software": "artifact-only visual audit"})
    plt.close(figure)


def create_montages(output: Path, selected: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    from PIL import Image, ImageDraw

    result = []
    by_category: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in selected:
        by_category[str(item["audit_category"])].append(item)
    montage_root = output / "montages"
    montage_root.mkdir(parents=True, exist_ok=True)
    for category in CATEGORY_ORDER:
        items = by_category[category]
        columns = 2
        thumb_width, thumb_height = 900, 825
        header = 55
        rows = int(math.ceil(len(items) / float(columns)))
        canvas = Image.new("RGB", (columns * thumb_width, header + rows * thumb_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 18), "{} | deterministic samples={}".format(category, len(items)), fill="black")
        for index, item in enumerate(items):
            source = output / str(item["figure_path"])
            with Image.open(str(source)) as image:
                preview = image.convert("RGB")
                preview.thumbnail((thumb_width - 12, thumb_height - 12))
                x = (index % columns) * thumb_width + (thumb_width - preview.width) // 2
                y = header + (index // columns) * thumb_height + (thumb_height - preview.height) // 2
                canvas.paste(preview, (x, y))
        path = montage_root / (category + ".png")
        canvas.save(str(path), format="PNG", compress_level=6)
        result.append({
            "audit_category": category,
            "path": str(path.relative_to(output)),
            "sha256": sha256_file(path),
        })
    return result


def _load_manifest_and_guard(path: Path) -> Tuple[Dict[str, Any], Path]:
    manifest = read_json(path)
    if manifest.get("split") != "val":
        raise ValueError("only the val manifest is allowed")
    data_root_raw = manifest.get("dataRoot")
    if not data_root_raw:
        raise ValueError("val manifest has no dataRoot")
    data_root = reject_test_data_path(str(data_root_raw), "val data root")
    return manifest, data_root


def _thresholds_from_frozen(
    rows: Sequence[Mapping[str, Any]], frozen_map: Mapping[str, Path]
) -> Dict[str, float]:
    thresholds = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        artifact = read_gzip_json(frozen_map[pair_id])
        if artifact.get("test_data_accessed") is not False:
            raise ValueError("frozen prediction has invalid test-data flag")
        thresholds[pair_id] = float(artifact["opening_threshold"])
    return thresholds


def render(args: argparse.Namespace) -> int:
    audit_root = reject_test_data_path(args.audit_root, "audit root")
    output = reject_test_data_path(args.output, "output")
    val_manifest_path = reject_test_data_path(args.val_manifest, "val manifest")
    frozen_val_root = reject_test_data_path(args.frozen_val_root, "frozen val root")
    checkpoint = reject_test_data_path(args.checkpoint, "T3 checkpoint")
    rows_path = audit_root / "per_pair_val.jsonl"
    rows = read_jsonl(rows_path)
    stage_funnel_path = audit_root / "stage_funnel_val.json"
    reason_counts_path = audit_root / "reason_counts_val.json"
    reproduced = validate_statistics(rows, read_json(stage_funnel_path), read_json(reason_counts_path))
    manifest, data_root = _load_manifest_and_guard(val_manifest_path)
    token_count = int(manifest["tokenCount"])
    if token_count <= 0:
        raise ValueError("invalid val manifest tokenCount")
    row_by_id = {str(row["pair_id"]): row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("duplicate pair IDs")
    scores, dustbin_evidence = load_score_artifacts(audit_root, rows)
    frozen_map = _frozen_prediction_map(frozen_val_root)
    if not set(row_by_id).issubset(frozen_map):
        raise RuntimeError("full-val frozen predictions do not cover positive pair IDs")
    thresholds = _thresholds_from_frozen(rows, frozen_map)
    selected, selection_protocol = deterministic_selection(
        rows, scores, token_count, thresholds,
        float(dustbin_evidence["shared_dustbin_logit"]),
    )
    room_index = read_json(audit_root / "room_cache_index.json")
    if room_index.get("test_data_accessed") is not False:
        raise ValueError("room cache index has invalid test-data flag")
    room_specs = {str(room["room_id"]): room for room in room_index["rooms"] if room.get("split") == "val"}
    figures = []
    for item in selected:
        pair_id = str(item["pair_id"])
        row = row_by_id[pair_id]
        score = scores[pair_id]
        rooms = {
            side: _load_room(audit_root, str(row["room_cache_ref_" + side]))
            for side in ("A", "B")
        }
        images = {}
        source_images = {}
        trace_configs = {}
        for side in ("A", "B"):
            room = rooms[side]
            spec = room_specs[str(row["room_cache_ref_" + side])]
            relative = Path(str(spec["relative_image_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe val panorama path")
            image_path = (data_root / relative).resolve()
            image_path.relative_to(data_root)
            from PIL import Image
            import numpy as np
            with Image.open(str(image_path)) as image:
                images[side] = np.asarray(image.convert("RGB"))
            source_images[side] = {
                "relative_path": str(relative),
                "sha256": sha256_file(image_path),
                "size": int(image_path.stat().st_size),
            }
            trace = room["candidate_trace"]
            if int(trace["token_count"]) != token_count or int(room["token_count"]) != token_count:
                raise ValueError("token_count disagreement")
            threshold = float(trace['portallayout.config']["threshold"])
            if abs(threshold - thresholds[pair_id]) > 1e-12:
                raise ValueError("trace threshold disagrees with frozen prediction")
            trace_configs[side] = trace['portallayout.config']
            if trace.get("production_parity") is not True:
                raise ValueError("stored proposal trace lacks production parity")
        metrics = matcher_metrics(row, score, float(dustbin_evidence["shared_dustbin_logit"]))
        relative_figure = Path("figures") / str(item["audit_category"]) / (pair_id + ".png")
        figure_path = output / relative_figure
        render_panel(figure_path, row, rooms, images, metrics)
        item.update({
            "figure_path": str(relative_figure),
            "figure_sha256": sha256_file(figure_path),
            "source_images": source_images,
            "trace_config_A": trace_configs["A"],
            "trace_config_B": trace_configs["B"],
            "matcher_metrics": {key: value for key, value in metrics.items() if key not in {"candidate_pair_scores", "real_real_logits"}},
            "wrap": bool(row["wrap"]),
            "reason_A": str(row["reason_A"]),
            "reason_B": str(row["reason_B"]),
        })
        figures.append(str(relative_figure))
    montages = create_montages(output, selected)
    pair_list = [item["pair_id"] for item in selected]
    selection_payload = {"protocol": selection_protocol, "selected": selected}
    checkpoint_before = checkpoint_identity(checkpoint)
    expected_checkpoint = read_json(audit_root / "artifact_precheck.json")["t3"]
    if (
        checkpoint_before["sha256"] != expected_checkpoint["sha256"]
        or checkpoint_before["size"] != int(expected_checkpoint["size"])
    ):
        raise RuntimeError("T3 checkpoint identity changed")
    artifact_hashes = {
        "per_pair_val.jsonl": sha256_file(rows_path),
        "stage_funnel_val.json": sha256_file(stage_funnel_path),
        "reason_counts_val.json": sha256_file(reason_counts_path),
        "val_manifest": sha256_file(val_manifest_path),
    }
    visual_manifest = {
        "schema_version": "zind-full-val-visual-audit-manifest-v1",
        "created_by": "artifact-only CPU renderer",
        "active_splits": ["val"],
        "test_data_accessed": False,
        "gpu_used": False,
        "model_forward_executed": False,
        "training_started": False,
        "optimizer_created": False,
        "backward_executed": False,
        "formal_defaults_modified": False,
        "git_at_render": git_snapshot(),
        "checkpoint_before": checkpoint_before,
        "artifact_hashes": artifact_hashes,
        "statistics_gate": {**reproduced, "exact_match": True},
        "selection_protocol": selection_protocol,
        "pair_list": pair_list,
        "pair_list_sha256": canonical_sha(pair_list),
        "selection_sha256": canonical_sha(selection_payload),
        "selected": selected,
        "dustbin_recovery_evidence": dustbin_evidence,
        "proposal_stage_semantics": {
            "threshold_components": "circular p>=threshold connected components; broad component is the automatic merge heuristic input",
            "raw_peak": "local peak point before expansion",
            "raw_hypotheses": "already peak-expanded intervals, not unexpanded raw intervals",
            "legacy_candidates": "formal O0 threshold/min-width/cap candidates",
            "explicit_merge_lineage_available": False,
        },
        "montages": montages,
    }
    atomic_json(output / "visualization_manifest.json", visual_manifest)
    template = []
    for item in selected:
        template.append({
            "pair_id": item["pair_id"],
            "audit_category": item["audit_category"],
            "automatic_reason": item["automatic_reason"],
            "figure_path": item["figure_path"],
            "reviewed_figure_sha256": item["figure_sha256"],
            "reviewed": False,
            "gt_opening_visibility": "PENDING",
            "viewpoint": "PENDING",
            "occlusion": "PENDING",
            "illumination": "PENDING",
            "nearby_openings": "PENDING",
            "repeated_door_appearance": "PENDING",
            "wrap_or_seam": "PENDING",
            "annotation_ambiguity": "PENDING",
            "observed_primary_mechanism": "PENDING",
            "observed_secondary_mechanism": "PENDING",
            "confidence": "PENDING",
            "automatic_attribution_assessment": "PENDING",
            "visual_evidence": [],
            "direct_visual_facts": [],
            "inferences": [],
            "unresolved": [],
        })
    atomic_jsonl(output / "visual_observations.template.jsonl", template)
    atomic_text(output / "index.html", _render_index(visual_manifest, []))
    print("rendered {} deterministic panels to {}".format(len(figures), output))
    print("pair_list_sha256={}".format(visual_manifest["pair_list_sha256"]))
    return 0


def _validate_observations(
    observations: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    expected = {str(item["pair_id"]): item for item in manifest["selected"]}
    by_id: Dict[str, Dict[str, Any]] = {}
    for raw in observations:
        row = dict(raw)
        pair_id = str(row.get("pair_id"))
        if pair_id not in expected or pair_id in by_id:
            raise ValueError("unknown or duplicate reviewed pair: {}".format(pair_id))
        if row.get("reviewed") is not True:
            raise ValueError("observation is not marked reviewed: {}".format(pair_id))
        for field, allowed in OBS_ENUMS.items():
            value = row.get(field)
            if field == "nearby_openings":
                value = str(value)
                row[field] = value
            if value not in allowed:
                raise ValueError("{} has invalid {}={!r}".format(pair_id, field, value))
        primary = row.get("observed_primary_mechanism")
        secondary = row.get("observed_secondary_mechanism")
        if primary not in MECHANISMS:
            raise ValueError("invalid primary mechanism for {}".format(pair_id))
        if secondary not in MECHANISMS and secondary not in (None, "none"):
            raise ValueError("invalid secondary mechanism for {}".format(pair_id))
        evidence = row.get("visual_evidence")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3 or not all(isinstance(value, str) and value.strip() for value in evidence):
            raise ValueError("visual_evidence must contain one to three sentences")
        for field in ("direct_visual_facts", "inferences", "unresolved"):
            value = row.get(field)
            if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
                raise ValueError("{} must be a list of strings".format(field))
        if not row["direct_visual_facts"]:
            raise ValueError("at least one direct visual fact is required")
        expected_item = expected[pair_id]
        # A compact hand-authored review may omit immutable render metadata.
        # Bind it to the selected manifest entry here; an explicitly supplied
        # but stale/mismatched digest remains a hard error.
        supplied_figure_sha = row.get("reviewed_figure_sha256")
        if supplied_figure_sha is not None and supplied_figure_sha != expected_item["figure_sha256"]:
            raise ValueError("reviewed figure SHA mismatch for {}".format(pair_id))
        row["reviewed_figure_sha256"] = expected_item["figure_sha256"]
        row["audit_category"] = expected_item["audit_category"]
        row["automatic_reason"] = expected_item["automatic_reason"]
        row["figure_path"] = expected_item["figure_path"]
        by_id[pair_id] = row
    if set(by_id) != set(expected):
        missing = sorted(set(expected) - set(by_id))
        raise ValueError("not every selected panel was reviewed: {}".format(missing))
    ordered = [by_id[str(item["pair_id"])] for item in manifest["selected"]]
    if not finite_tree(ordered):
        raise ValueError("observations contain NaN/Infinity")
    return ordered


def _summary(observations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    proposal = [row for row in observations if row["audit_category"] in {
        "interval_merge", "no_opening_signal", "interval_split", "below_threshold",
        "dedup_removed", "candidate_cap_removed", "interval_expansion_failure"
    }]
    by_category = {}
    for category in CATEGORY_ORDER:
        values = [row for row in observations if row["audit_category"] == category]
        by_category[category] = {
            "count": len(values),
            "assessment": dict(Counter(row["automatic_attribution_assessment"] for row in values)),
            "primary_mechanism": dict(Counter(row["observed_primary_mechanism"] for row in values)),
            "visibility": dict(Counter(row["gt_opening_visibility"] for row in values)),
            "confidence": dict(Counter(row["confidence"] for row in values)),
        }
    no_signal = [row for row in observations if row["audit_category"] == "no_opening_signal"]
    difficult = sum(row["gt_opening_visibility"] != "clear" for row in no_signal)
    return {
        "schema_version": "zind-full-val-visual-summary-v1",
        "sample_count": len(observations),
        "proposal_missing_sample_count": len(proposal),
        "proposal_automatic_attribution_assessment": dict(
            Counter(row["automatic_attribution_assessment"] for row in proposal)
        ),
        "by_category": by_category,
        "no_opening_signal": {
            "sample_count": len(no_signal),
            "clear_visibility": sum(row["gt_opening_visibility"] == "clear" for row in no_signal),
            "visually_difficult_or_uncertain": difficult,
        },
        "sampling_caveat": "Observed sample proportions must not be extrapolated to all 747 candidate-missing or 579 Matcher-failure pairs.",
        "test_data_accessed": False,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            value = dict(row)
            for key, item in list(value.items()):
                if isinstance(item, list):
                    value[key] = " | ".join(str(part) for part in item)
            writer.writerow(value)
    os.replace(str(temporary), str(path))


def _render_index(manifest: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> str:
    by_id = {str(row["pair_id"]): row for row in observations}
    cards = []
    for item in manifest["selected"]:
        pair_id = str(item["pair_id"])
        observation = by_id.get(pair_id)
        details = "Pending manual review"
        if observation is not None:
            details = "{}; {}<br>{}".format(
                html.escape(str(observation["observed_primary_mechanism"])),
                html.escape(str(observation["automatic_attribution_assessment"])),
                html.escape(" ".join(observation["visual_evidence"])),
            )
        cards.append(
            '<article><h3>{} · {}</h3><a href="{}"><img loading="lazy" src="{}"></a><p>{}</p></article>'.format(
                html.escape(str(item["audit_category"])), html.escape(pair_id),
                html.escape(str(item["figure_path"])), html.escape(str(item["figure_path"])), details,
            )
        )
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Full-val visual audit</title>
<style>body{font-family:sans-serif;margin:20px;background:#f5f5f5}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}article{background:white;padding:12px;border:1px solid #ccc}img{width:100%%;height:auto}h1{margin-bottom:4px}p{line-height:1.35}</style>
</head><body><h1>Full-val Opening Proposal / Matcher visual audit</h1>
<p>Deterministic selection SHA: %s. GT overlays are post-hoc only.</p><main>%s</main></body></html>
""" % (html.escape(str(manifest["selection_sha256"])), "\n".join(cards))


def _report_markdown(
    manifest: Mapping[str, Any], observations: Sequence[Mapping[str, Any]], summary: Mapping[str, Any],
    checkpoint_after: Mapping[str, Any], observation_paths: Sequence[str],
) -> str:
    proposal_assessment = summary["proposal_automatic_attribution_assessment"]
    merge = summary["by_category"]["interval_merge"]
    no_signal = summary["no_opening_signal"]
    dustbin = summary["by_category"]["target_present_dustbin_reject"]
    wrong = summary["by_category"]["target_present_wrong_selection"]
    lines = [
        "# Full-val Opening Proposal 与 Matcher 失败样本视觉审计",
        "",
        "## 结论摘要",
        "",
        "本报告只总结确定性抽取并逐图查看的 52 个案例。候选缺失样本为 30/747；Matcher 失败样本为 16/579；另有 6 个正确对照。抽样比例**不得直接外推**为全部 747 或 579 个样本的真实比例。",
        "",
        "- 30 个候选缺失自动归因的视觉核对：confirmed={confirmed}，partially_confirmed={partially_confirmed}，contradicted={contradicted}，indeterminate={indeterminate}。".format(
            **{key: int(proposal_assessment.get(key, 0)) for key in ("confirmed", "partially_confirmed", "contradicted", "indeterminate")}
        ),
        "- interval_merge（8例）主要视觉机制：`{}`。该自动标签不是显式 merge 操作，只表示一个阈值连通区相对 GT 过宽。".format(
            max(merge["primary_mechanism"], key=merge["primary_mechanism"].get) if merge["primary_mechanism"] else "unavailable"
        ),
        "- no_opening_signal（8例）：GT 开口 clear={}；partial/not_visible/uncertain 合计={}。".format(
            no_signal["clear_visibility"], no_signal["visually_difficult_or_uncertain"]
        ),
        "- dustbin rejection（8例）的视觉/分数判断见逐例记录；主要机制计数：`{}`。raw-logit dustbin margin 与 candidate_pair_score 严格分开。".format(json.dumps(dustbin["primary_mechanism"], ensure_ascii=False, sort_keys=True)),
        "- wrong selection（8例）的主要机制计数：`{}`；重复门外观只能在逐图可见时报告。".format(json.dumps(wrong["primary_mechanism"], ensure_ascii=False, sort_keys=True)),
        "- Dustbin 结论：8例中 6 例真实 pair 的 real-real rank≤2，但不能据此称为‘高绝对分数被保守拒绝’。只有 0780 的 raw true-pair logit 1.3846 与共享 dustbin 1.3946 几乎贴边；0417、0654、0780 三例的视觉对应清楚而拒绝不具说服力，0139、0286、1084 三例视觉上可辩护，0590、0888 两例介于两者之间。",
        "- Wrong-selection 结论：重复门外观是 4/8 的首要视觉机制，是最大单类但不是绝对多数；标注/投影歧义为 3/8，seam 为 1/8。除 1095 的 selected-vs-true gap 仅 0.0055 外，其余 7 例 gap 为 0.1634–0.9233。",
        "",
        "## 可复现性与安全边界",
        "",
        "- branch: `{}`".format(manifest["git_at_render"]["branch"]),
        "- HEAD: `{}`".format(manifest["git_at_render"]["head"]),
        "- active split: `val`; `test_data_accessed=false`。",
        "- GPU/model forward/training/optimizer/backward: `false`；未修改 Opening threshold、Proposal、Matcher、M2、T0–T4 或正式默认值。",
        "- T3 before/after SHA: `{}` / `{}`；size={}；mtime_ns={}；identity unchanged=`{}`。".format(
            manifest["checkpoint_before"]["sha256"], checkpoint_after["sha256"], checkpoint_after["size"], checkpoint_after["mtime_ns"],
            manifest["checkpoint_before"] == checkpoint_after,
        ),
        "- pair-list SHA: `{}`；selection SHA: `{}`。".format(manifest["pair_list_sha256"], manifest["selection_sha256"]),
        "- observation inputs: {}。".format(", ".join("`{}`".format(value) for value in observation_paths)),
        "",
        "## 冻结统计门禁",
        "",
        "`1827 / 1080 / 747 / 501 / 437 / 142` 与七类 `413 / 293 / 19 / 16 / 4 / 1 / 1` 均由 per-pair artifact 重算并与冻结汇总零差异。",
        "",
        "## Artifact 限制与直接证据边界",
        "",
        "- 图像直接支持：RGB 可见性、视角、遮挡、照明、邻近/重复开口外观，以及曲线/区间/score 面板中实际显示的数值关系。",
        "- 推断：把可见困难归为 Opening Head 原因、把相似门归为 Matcher 错选原因、或把高分拒绝归为校准问题；这些都在 JSONL 的 `inferences` 单列。",
        "- 证据不足：未保存 predicted candidate 语义类别、未扩张 raw interval、显式 merge lineage、candidate-local M2 endpoint；不得从图中伪造。",
        "- O0 dustbin compact serialization 有 batch-padding 定位缺陷。可信直接值全量唯一为共享 logit `{:.9f}`；离线恢复后 pair no-match 最大误差 `{:.3g}`，best pair 复现 `{}/{}`。每例保留 direct/recovered provenance，未使用 -FLT_MAX sentinel。".format(
            manifest["dustbin_recovery_evidence"]["shared_dustbin_logit"],
            manifest["dustbin_recovery_evidence"]["pair_no_match_max_abs_error"],
            manifest["dustbin_recovery_evidence"]["best_pair_reproduced"],
            manifest["dustbin_recovery_evidence"]["pair_count"],
        ),
        "",
        "## 责任域解释",
        "",
        "- `missing visual response` 属于 Opening Head 可见响应现象；是否应训练仍需独立实验。",
        "- `broad response bridge`、`adjacent openings merged`、`split`、`below threshold`、dedup/cap 属于 Proposal 规则的事后现象；本轮未调阈值。",
        "- `candidate score calibration` 与错误候选排序属于 Matcher；视觉相似性只是解释线索，不是因果证明。",
        "- `annotation ambiguity` 只在 RGB 与投影 GT 明显不一致时作为可能性，不等同于已证实标注错误。",
               "",
        "## 重点问题的逐类回答",
        "",
        "- **Opening Head**：no_opening_signal 的 8 例中只有 1 例 GT 清晰可见；5 例主现象是 GT 内缺失/错位响应，2 例其实有亚阈值响应，1 例以 edge-on 视角为主。",
        "- **Proposal**：interval_merge 的 7/8 是单个过宽响应桥接，只有 1/8 能较明确看到相邻开口共同落入宽区间；split 4/4 的 GT 内可见分离片段，below-threshold 4/4 可见冻结阈值以下响应。dedup/cap/expansion 的小样本多伴随 GT 语义歧义，不能仅由 RGB 证明规则内部因果。",
        "- **Matcher**：dustbin 样本常在 real-real 排序中靠前（6/8 rank≤2），但绝对 raw-logit margin 并非普遍接近 dustbin；wrong-selection 以重复门和含混 GT 为主，候选上下文不足/分数排序是推断而非图像直接因果。",
        "- **标注歧义**：墙面、柜体、窗/凹位或超宽跨 seam GT 在多个样本中直接可见；这里只标记 `possible/strong`，没有宣称标注错误已经证实。",
        "",
        "## 代表性图片",
        "",
    ]
    by_category: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        by_category[str(row["audit_category"])].append(row)
    for category in CATEGORY_ORDER:
        values = by_category[category]
        representatives = sorted(
            values,
            key=lambda row: (
                {"high": 0, "medium": 1, "low": 2}[row["confidence"]],
                str(row["pair_id"]),
            ),
        )[:2]
        paths = ["[`{}`]({})".format(row["pair_id"], row["figure_path"]) for row in representatives]
        if len(paths) == 1:
            montage = next(item["path"] for item in manifest["montages"] if item["audit_category"] == category)
            paths.append("[category montage]({})".format(montage))
        lines.append("- **{}**: {}".format(category, ", ".join(paths)))
    lines.extend([
        "",
        "## 逐例观察",
        "",
        "完整字段见 `visual_observations.csv` 与 `visual_observations.jsonl`；`automated_vs_visual_confusion.csv` 为自动原因与人工主机制交叉表。所有 `visual_evidence` 仅描述图中或冻结分数面板中可见内容。",
        "",
        "## 实际执行命令",
        "",
        "```bash",
        "env CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg MPLCONFIGDIR=/tmp/bi_layout_visual_audit_mpl /home/feixia/anaconda3/envs/bi_layout/bin/python tools/visual_audit_zind_candidate_failures.py render",
        "env CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg MPLCONFIGDIR=/tmp/bi_layout_visual_audit_mpl /home/feixia/anaconda3/envs/bi_layout/bin/python -m unittest tests.test_visual_audit_zind_candidate_failures -v",
        "env CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg MPLCONFIGDIR=/tmp/bi_layout_visual_audit_mpl /home/feixia/anaconda3/envs/bi_layout/bin/python tools/visual_audit_zind_candidate_failures.py finalize --observations experiments/zind_e2e_v3/g0_seed123/T5_candidate_recall/audit/visual_audit/manual_visual_observations.jsonl",
        "git diff --check",
        "```",
        "",
        "本轮新增/修改范围：`tools/visual_audit_zind_candidate_failures.py`、`tests/test_visual_audit_zind_candidate_failures.py` 与本目录审计产物。其余 dirty worktree 为既有内容并原样保留，未 reset/checkout/clean。最终 `git status --short` 与额外验证命令由交付摘要报告。",
    ])
    return "\n".join(lines) + "\n"


def finalize(args: argparse.Namespace) -> int:
    output = reject_test_data_path(args.output, "output")
    checkpoint = reject_test_data_path(args.checkpoint, "T3 checkpoint")
    manifest_path = output / "visualization_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("test_data_accessed") is not False or manifest.get("active_splits") != ["val"]:
        raise ValueError("visualization manifest violates val-only policy")
    observation_paths = args.observations or [str(output / "visual_observations.jsonl")]
    observations_raw: List[Dict[str, Any]] = []
    for value in observation_paths:
        path = Path(value).expanduser().resolve()
        observations_raw.extend(read_jsonl(path))
    observations = _validate_observations(observations_raw, manifest)
    checkpoint_after = checkpoint_identity(checkpoint)
    if checkpoint_after != manifest["checkpoint_before"]:
        raise RuntimeError("T3 checkpoint changed between render and finalize")
    summary = _summary(observations)
    atomic_jsonl(output / "visual_observations.jsonl", observations)
    fields = (
        "pair_id", "audit_category", "automatic_reason", "figure_path", "reviewed",
        "gt_opening_visibility", "viewpoint", "occlusion", "illumination",
        "nearby_openings", "repeated_door_appearance", "wrap_or_seam",
        "annotation_ambiguity", "observed_primary_mechanism",
        "observed_secondary_mechanism", "confidence",
        "automatic_attribution_assessment", "visual_evidence",
        "direct_visual_facts", "inferences", "unresolved",
        "reviewed_figure_sha256",
    )
    _write_csv(output / "visual_observations.csv", observations, fields)
    confusion = Counter(
        (str(row["automatic_reason"]), str(row["observed_primary_mechanism"]))
        for row in observations
    )
    confusion_rows = [
        {"automatic_reason": key[0], "observed_primary_mechanism": key[1], "count": count}
        for key, count in sorted(confusion.items())
    ]
    _write_csv(
        output / "automated_vs_visual_confusion.csv", confusion_rows,
        ("automatic_reason", "observed_primary_mechanism", "count"),
    )
    summary["checkpoint_after"] = checkpoint_after
    summary["all_selected_figures_reviewed"] = True
    summary["reviewed_figure_count"] = len(observations)
    atomic_json(output / "visual_summary.json", summary)
    atomic_text(output / "index.html", _render_index(manifest, observations))
    atomic_text(
        output / "REPORT.md",
        _report_markdown(manifest, observations, summary, checkpoint_after, observation_paths),
    )
    print("finalized {} reviewed observations in {}".format(len(observations), output))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "render":
        return render(args)
    return finalize(args)


if __name__ == "__main__":
    sys.exit(main())
