#!/usr/bin/env python3
"""Resume the SALVe verifier preflight with exact frozen negative-room data.

This entry point is intentionally isolated from production evaluators.  GPU
work is restricted to frozen T3 room inference and frozen Matcher scoring;
the lightweight Phase-B verifier is trained only after those artifacts pass
their parity and integrity gates.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import os
import random
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_zind_candidate_failures import (  # noqa: E402
    AuditLogger,
    _cache_views,
    _data_root,
    _freeze_rooms,
    _load_frozen_model,
    _load_manifest,
    _load_room,
    _medium_room_reference,
    _medium_shard_lookup,
    _room_payloads,
    _score_batch,
    _state_hash,
    finite_tree,
)
from tools.preflight_zind_bev_verifier import (  # noqa: E402
    AUDIT_ROOT,
    BASE_FEATURES,
    DEFAULT_OUTPUT,
    EXPECTED_T3_SHA256,
    INPUT_FEATURES,
    SEED,
    T3_CHECKPOINT,
    atomic_json,
    atomic_text,
    candidate_feature_row,
    canonical_sha,
    feature_indices,
    feature_names_are_predicted_only,
    house_split,
    house_round_robin_negatives,
    load_context,
    load_pair_score,
    manifest_target,
    normalization,
    normalized_vector,
    read_jsonl,
    reject_test_path,
    sha256_file,
    stable_topn,
    write_jsonl,
    PredictedGeometryStore,
)
from tools.train_zind_end_to_end import (  # noqa: E402
    DEFAULT_BI_LAYOUT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_MATCHER,
    DEFAULT_OPENING,
)


DEFAULT_MEDIUM = REPO_ROOT / "experiments/zind_e2e_v3/g0_seed123/T4_recall/medium_diagnostic"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "smoke", "cache", "score", "features", "train", "report", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit_root", default=str(AUDIT_ROOT))
    parser.add_argument("--train_manifest", default=str(REPO_ROOT / "data/ZInD_matching/train.json"))
    parser.add_argument("--val_manifest", default=str(REPO_ROOT / "data/ZInD_matching/val.json"))
    parser.add_argument("--train_cache_dir", default=str(DEFAULT_CACHE_ROOT / "train"))
    parser.add_argument("--val_cache_dir", default=str(DEFAULT_CACHE_ROOT / "val"))
    parser.add_argument("--t3_checkpoint", default=str(T3_CHECKPOINT))
    parser.add_argument("--bi_layout_config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bi_layout_checkpoint", default=str(DEFAULT_BI_LAYOUT))
    parser.add_argument("--opening_checkpoint", default=str(DEFAULT_OPENING))
    parser.add_argument("--matcher_checkpoint", default=str(DEFAULT_MATCHER))
    parser.add_argument("--medium_root", default=str(DEFAULT_MEDIUM))
    parser.add_argument("--data_root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--oracle_batch_size", type=int, default=4)
    parser.add_argument("--progress_every", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    return parser.parse_args(argv)


def load_base(args: argparse.Namespace, output: Path) -> Dict[str, Any]:
    if args.seed != SEED or args.bootstrap_replicates != 2000:
        raise ValueError("Phase-B protocol is fixed to seed=123/bootstrap=2000")
    if not (output / ".phase_A_passed").is_file():
        raise RuntimeError("Phase A must pass before Phase B")
    context_args = argparse.Namespace(
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        audit_root=args.audit_root,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
    )
    context = load_context(context_args, output)
    if sha256_file(T3_CHECKPOINT) != EXPECTED_T3_SHA256:
        raise ValueError("T3 checkpoint SHA changed")
    return context


def _selected_negatives(context: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    train = house_round_robin_negatives(context["train"]["pairs"], 4096)
    val = sorted(
        (record for record in context["val"]["pairs"] if not bool(record["supervision"]["is_match"])),
        key=lambda record: str(record["id"]),
    )
    if len(train) != 4096 or len(val) != 1803:
        raise ValueError(f"negative cohort mismatch train={len(train)} val={len(val)}")
    return {"train": train, "val": val}


def _medium_references(medium_root: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    output: Dict[Tuple[str, str], Dict[str, str]] = {}
    for split in ("train", "val"):
        path = reject_test_path(medium_root / f"{split}_medium.json", f"{split} medium")
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in sorted(payload.get("pairs", []), key=lambda value: str(value["id"])):
            for side in ("A", "B"):
                relative = str(record[f"image_{side.lower()}"])
                output.setdefault((split, relative), {"pair_id": str(record["id"]), "side": side})
    return output


def prepare_negative_index(
    args: argparse.Namespace, context: Mapping[str, Any], output: Path, log: AuditLogger,
) -> Dict[str, Any]:
    selected = _selected_negatives(context)
    view_indices = {
        split: _cache_views(reject_test_path(getattr(args, f"{split}_cache_dir"), f"{split} cache"), split)[0]
        for split in ("train", "val")
    }
    existing = {
        (str(spec["split"]), str(spec["relative_image_path"])): str(spec["room_id"])
        for spec in context["room_index"]["rooms"]
    }
    medium = _medium_references(reject_test_path(args.medium_root, "medium root"))
    specs: Dict[str, Dict[str, Any]] = {}
    pair_refs: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "val"):
        for record in selected[split]:
            refs: Dict[str, Any] = {"split": split}
            for side in ("A", "B"):
                suffix = side.lower()
                relative = str(record[f"image_{suffix}"])
                key = (split, relative)
                if key in existing:
                    refs[side] = {"room_id": existing[key], "source": "candidate_audit"}
                    continue
                if relative not in view_indices[split]:
                    raise ValueError(f"feature cache misses {split}/{relative}")
                identity = {
                    "split": split,
                    "house_id": str(record["scene_id"]),
                    "floor_id": str(record["floor_id"]),
                    "complete_room_id": str(record[f"room_{suffix}"]),
                    "pano_id": str(record[f"pano_{suffix}"]),
                    "relative_image_path": relative,
                }
                room_id = canonical_sha(identity)[:24]
                spec = {
                    **identity,
                    "room_id": room_id,
                    "cache_index": int(view_indices[split][relative]),
                    "reference_pair_id": str(record["id"]),
                    "reference_side": side,
                    "medium_reference": medium.get(key),
                }
                previous = specs.setdefault(room_id, spec)
                immutable = (
                    "split", "house_id", "floor_id", "complete_room_id", "pano_id",
                    "relative_image_path", "cache_index",
                )
                if any(previous[name] != spec[name] for name in immutable):
                    raise ValueError(f"negative room cache collision: {room_id}")
                if (str(record["id"]), side) < (str(previous["reference_pair_id"]), str(previous["reference_side"])):
                    previous["reference_pair_id"] = str(record["id"])
                    previous["reference_side"] = side
                refs[side] = {"room_id": room_id, "source": "phase_b_negative"}
            pair_refs[str(record["id"])] = refs

    ordered = sorted(
        specs.values(),
        key=lambda value: (
            value["split"] != "val", value["house_id"], value["floor_id"],
            value["complete_room_id"], value["pano_id"], value["cache_index"],
        ),
    )
    counts = {
        split: sum(spec["split"] == split for spec in ordered) for split in ("train", "val")
    }
    if counts != {"train": 2496, "val": 141}:
        raise ValueError(f"negative missing-room parity failed: {counts}")
    payload = {
        "schema_version": "zind-salve-phase-b-negative-room-index-v1",
        "rooms": ordered,
        "pair_refs": pair_refs,
        "splits": {
            split: {
                "pair_count": len(selected[split]),
                "missing_unique_room_count": counts[split],
                "medium_reference_room_count": sum(
                    spec["split"] == split and spec.get("medium_reference") is not None for spec in ordered
                ),
            }
            for split in ("train", "val")
        },
        "index_sha256": None,
        "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    payload["index_sha256"] = canonical_sha({**payload, "index_sha256": None})
    atomic_json(output / "negative_room_cache_index.json", payload)
    atomic_text(output / ".phase_B_negative_prepare_complete", sha256_file(output / "negative_room_cache_index.json") + "\n")
    log(f"negative prepare complete rooms={len(ordered)} medium_overlap={sum(spec.get('medium_reference') is not None for spec in ordered)}")
    return payload


def _freeze_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        data_root=args.data_root,
        t3_checkpoint=args.t3_checkpoint,
        bi_layout_config=args.bi_layout_config,
        bi_layout_checkpoint=args.bi_layout_checkpoint,
        opening_checkpoint=args.opening_checkpoint,
        matcher_checkpoint=args.matcher_checkpoint,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )


def smoke_negative_cache(
    args: argparse.Namespace, context: Mapping[str, Any], output: Path, log: AuditLogger,
) -> Dict[str, Any]:
    index = json.loads((output / "negative_room_cache_index.json").read_text(encoding="utf-8"))
    rooms = index["rooms"]
    smoke = sorted(
        rooms,
        key=lambda spec: (
            spec.get("medium_reference") is None, spec["split"] != "val",
            spec["house_id"], spec["floor_id"], spec["room_id"],
        ),
    )[:64]
    if len(smoke) != 64:
        raise ValueError("Phase-B smoke requires 64 missing rooms")
    freeze = _freeze_rooms(_freeze_args(args), output, smoke, phase_name="phaseB_negative_smoke", log=log)
    cache_root = output / "frozen_room_cache"

    # Compare against already frozen medium artifacts wherever available.
    medium_root = reject_test_path(args.medium_root, "medium root")
    lookups = {split: _medium_shard_lookup(medium_root, split) for split in ("train", "val")}
    loaded: Dict[str, Any] = {}
    maxima: Dict[str, float] = defaultdict(float)
    overlap = 0
    for spec in smoke:
        reference = _medium_room_reference(medium_root, lookups[spec["split"]], spec, loaded)
        if reference is None:
            continue
        room = _load_room(cache_root, str(spec["room_id"]), str(spec["split"]))
        overlap += 1
        for field in ("layout_feature", "depth", "new_depth", "enclosed_depth", "extended_depth", "opening_probability"):
            maxima[field] = max(maxima[field], float((room[field] - reference[field]).abs().max().item()))

    # A same-microbatch duplicate forward verifies exact repeatability even if
    # the medium overlap is small.
    duplicate_specs = smoke[:4]
    device = torch.device(args.device)
    manifests = {
        split: _load_manifest(reject_test_path(getattr(args, f"{split}_manifest"), f"{split} manifest"), split)
        for split in ("train", "val")
    }
    roots = {
        split: _data_root(manifests[split], reject_test_path(getattr(args, f"{split}_manifest"), f"{split} manifest"), args.data_root)
        for split in ("train", "val")
    }
    model, _, _ = _load_frozen_model(_freeze_args(args), device)
    duplicate, _ = _room_payloads(
        model, duplicate_specs, roots, device, EXPECTED_T3_SHA256,
        freeze["config_sha256"],
    )
    repeat_max = 0.0
    for spec, value in zip(duplicate_specs, duplicate):
        room = _load_room(cache_root, str(spec["room_id"]), str(spec["split"]))
        for field in ("layout_feature", "depth", "new_depth", "enclosed_depth", "extended_depth", "ratio", "opening_probability"):
            repeat_max = max(repeat_max, float((room[field] - value[field]).abs().max().item()))
    parameter_hash = _state_hash(model)
    del model
    torch.cuda.empty_cache()

    mean_size = float(np.mean([(cache_root / spec["split"] / f"{spec['room_id']}.pt").stat().st_size for spec in smoke]))
    projected = int(mean_size * len(rooms))
    available = os.statvfs(output).f_bavail * os.statvfs(output).f_frsize
    gates = {
        "room_count_64": len(smoke) == 64,
        "all_finite": all(finite_tree(_load_room(cache_root, str(spec["room_id"]), str(spec["split"]))) for spec in smoke),
        "medium_overlap_present": overlap > 0,
        "medium_reference_within_1e_2": all(value <= 1e-2 for value in maxima.values()),
        "repeat_forward_within_2e_5": repeat_max <= 2e-5,
        "checkpoint_unchanged": freeze["checkpoint_before"] == freeze["checkpoint_after"],
        "parameters_unchanged": freeze["parameter_hash_before"] == freeze["parameter_hash_after"] == parameter_hash,
        "no_grad": freeze["all_requires_grad_false"],
        "no_optimizer": freeze["optimizer_created"] is False,
        "projected_disk_below_25pct_available": projected <= 0.25 * available,
    }
    result = {
        "schema_version": "zind-salve-phase-b-negative-smoke-v1",
        "smoke_room_ids": [spec["room_id"] for spec in smoke],
        "freeze": freeze,
        "medium_reference_count": overlap,
        "medium_max_abs_differences": dict(maxima),
        "repeat_forward_max_abs_difference": repeat_max,
        "mean_room_file_bytes": mean_size,
        "projected_bytes": projected,
        "available_bytes": available,
        "gates": gates,
        "passed": all(gates.values()),
        "training_started": False,
        "optimizer_created": False,
        "test_data_accessed": False,
    }
    atomic_json(output / "phase_B_negative_smoke.json", result)
    if not result["passed"]:
        raise RuntimeError("Phase-B negative smoke failed: " + ", ".join(name for name, value in gates.items() if not value))
    atomic_text(output / ".phase_B_negative_smoke_passed", sha256_file(output / "phase_B_negative_smoke.json") + "\n")
    log(f"negative smoke PASS medium={overlap} repeat_max={repeat_max:.3g}")
    return result


def cache_negative_rooms(
    args: argparse.Namespace, context: Mapping[str, Any], output: Path, log: AuditLogger,
) -> Dict[str, Any]:
    if not (output / ".phase_B_negative_smoke_passed").is_file():
        raise RuntimeError("negative smoke must pass before full caching")
    index = json.loads((output / "negative_room_cache_index.json").read_text(encoding="utf-8"))
    result = _freeze_rooms(_freeze_args(args), output, index["rooms"], phase_name="phaseB_negative_cache", log=log)
    cache_root = output / "frozen_room_cache"
    complete = sum(
        (cache_root / spec["split"] / f"{spec['room_id']}.pt").is_file()
        and (cache_root / spec["split"] / f"{spec['room_id']}.completed").is_file()
        for spec in index["rooms"]
    )
    if complete != len(index["rooms"]):
        raise RuntimeError(f"negative cache incomplete: {complete}/{len(index['rooms'])}")
    result.update({"processed_room_count": complete, "complete": True})
    atomic_json(output / "phase_B_negative_cache_summary.json", result)
    atomic_text(output / ".phase_B_negative_cache_complete", sha256_file(output / "phase_B_negative_cache_summary.json") + "\n")
    log(f"negative room cache complete {complete}/{len(index['rooms'])}")
    return result


def _composite_room(
    context: Mapping[str, Any], output: Path, split: str, reference: Mapping[str, Any],
) -> Dict[str, Any]:
    root = context["audit"] / "frozen_room_cache" if reference["source"] == "candidate_audit" else output / "frozen_room_cache"
    return _load_room(root, str(reference["room_id"]), split)


def _atomic_gzip_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def score_negative_pairs(
    args: argparse.Namespace, context: Mapping[str, Any], output: Path, log: AuditLogger,
) -> Dict[str, Any]:
    if not (output / ".phase_B_negative_cache_complete").is_file():
        raise RuntimeError("negative room cache must complete before Matcher scoring")
    index = json.loads((output / "negative_room_cache_index.json").read_text(encoding="utf-8"))
    selected = _selected_negatives(context)
    device = torch.device(args.device)
    model, _, report = _load_frozen_model(_freeze_args(args), device)
    hash_before = _state_hash(model)
    summaries = {}
    for split in ("train", "val"):
        score_root = output / "negative_pair_scores" / split
        pending = [record for record in selected[split] if not (score_root / f"{record['id']}.json.gz").is_file()]
        initial = len(selected[split]) - len(pending)
        start = time.perf_counter()
        for offset in range(0, len(pending), args.oracle_batch_size):
            batch = pending[offset:offset + args.oracle_batch_size]
            rooms_a = [_composite_room(context, output, split, index["pair_refs"][str(record["id"])]["A"]) for record in batch]
            rooms_b = [_composite_room(context, output, split, index["pair_refs"][str(record["id"])]["B"]) for record in batch]
            intervals_a = [room["legacy_candidates"] for room in rooms_a]
            intervals_b = [room["legacy_candidates"] for room in rooms_b]
            values = _score_batch(model, rooms_a, rooms_b, intervals_a, intervals_b, device)
            for record, value, left, right in zip(batch, values, intervals_a, intervals_b):
                payload = {
                    "schema_version": "zind-salve-phase-b-negative-score-v1",
                    "pair_id": str(record["id"]), "split": split,
                    "view": "O0", "intervals_A": left, "intervals_B": right,
                    **value, "model_eval": True, "requires_grad": False,
                    "optimizer_created": False, "test_data_accessed": False,
                }
                _atomic_gzip_json(score_root / f"{record['id']}.json.gz", payload)
            done = initial + offset + len(batch)
            if done % 128 < len(batch) or done == len(selected[split]):
                log(f"negative Matcher {split} {done}/{len(selected[split])}")
        summaries[split] = {
            "pair_count": len(selected[split]), "initial_complete": initial,
            "new_complete": len(pending), "runtime_seconds": time.perf_counter() - start,
        }
    hash_after = _state_hash(model)
    if hash_before != hash_after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen Matcher changed during negative scoring")
    summary = {
        "schema_version": "zind-salve-phase-b-negative-score-summary-v1",
        "splits": summaries, "model_report": report,
        "parameter_hash_before": hash_before, "parameter_hash_after": hash_after,
        "model_eval": not model.training, "training_started": False,
        "optimizer_created": False, "test_data_accessed": False,
    }
    del model
    torch.cuda.empty_cache()
    atomic_json(output / "phase_B_negative_score_summary.json", summary)
    atomic_text(output / ".phase_B_negative_score_complete", sha256_file(output / "phase_B_negative_score_summary.json") + "\n")
    return summary


def _load_negative_score(output: Path, split: str, pair_id: str) -> Dict[str, Any]:
    path = output / "negative_pair_scores" / split / f"{pair_id}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("split") != split or value.get("test_data_accessed") is not False:
        raise ValueError(f"invalid negative pair score: {path}")
    return value


def _selected_positives(context: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    train_by_id = {str(record["id"]): record for record in context["train"]["pairs"]}
    train = [train_by_id[pair_id] for pair_id in context["train_sample"]["pair_ids"]]
    val = sorted(
        (record for record in context["val"]["pairs"] if bool(record["supervision"]["is_match"])),
        key=lambda record: str(record["id"]),
    )
    if len(train) != 4096 or len(val) != 1827:
        raise ValueError(f"positive cohort mismatch train={len(train)} val={len(val)}")
    return {"train": train, "val": val}


def _phase_a_feature_lookup(output: Path, split: str) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    path = output / f"candidate_features_{split}.jsonl"
    if not path.is_file():
        return {}
    result = {}
    for row in read_jsonl(path):
        key = (str(row["pair_id"]), int(row["candidate_A"]), int(row["candidate_B"]))
        if key in result:
            raise ValueError(f"duplicate Phase-A feature row: {key}")
        result[key] = row
    return result


def generate_phase_b_features(
    args: argparse.Namespace, context: Mapping[str, Any], output: Path, log: AuditLogger,
) -> Dict[str, Any]:
    if not (output / ".phase_B_negative_score_complete").is_file():
        raise RuntimeError("negative Matcher score must complete before feature generation")
    marker = output / ".phase_B_features_complete"
    summary_path = output / "phase_B_feature_summary.json"
    if args.resume and marker.is_file() and summary_path.is_file():
        log("Phase-B features already complete; resume skips regeneration")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    n = int(json.loads((output / "shortlist_audit.json").read_text(encoding="utf-8"))["selection_protocol"]["selected_N"])
    if n != 3:
        raise ValueError(f"frozen shortlist N changed: {n}")
    positives = _selected_positives(context)
    negatives = _selected_negatives(context)
    negative_index = json.loads((output / "negative_room_cache_index.json").read_text(encoding="utf-8"))
    augmented = dict(context)
    augmented["room_index"] = dict(context["room_index"])
    augmented_refs = dict(context["room_index"]["pair_refs"])
    augmented_refs.update({
        pair_id: {"split": ref["split"], "A": ref["A"]["room_id"], "B": ref["B"]["room_id"]}
        for pair_id, ref in negative_index["pair_refs"].items()
    })
    augmented["room_index"]["pair_refs"] = augmented_refs
    dustbin_payload = torch.load(T3_CHECKPOINT, map_location="cpu")
    dustbin = float(dustbin_payload["matcher_state_dict"]["dustbin_score"].item())
    del dustbin_payload

    summaries: Dict[str, Any] = {}
    for split in ("train", "val"):
        store = PredictedGeometryStore(augmented, output / "frozen_room_cache")
        phase_a = _phase_a_feature_lookup(output, split)
        positive_rows = {
            row["pair_id"]: row
            for row in read_jsonl(context["audit"] / ("per_pair_train_4096.jsonl" if split == "train" else "per_pair_val.jsonl"))
        }
        partial_rows_path = output / f"phase_B_candidate_features_{split}.positive.partial.jsonl"
        partial_meta_path = output / f"phase_B_pair_meta_{split}.positive.partial.jsonl"
        if args.resume and partial_rows_path.is_file() and partial_meta_path.is_file():
            rows = read_jsonl(partial_rows_path); pair_meta = read_jsonl(partial_meta_path)
            if len(pair_meta) != len(positives[split]):
                raise ValueError(f"invalid {split} positive partial checkpoint")
            log(f"Phase-B features {split} resumes after {len(pair_meta)} positive pairs")
        else:
            rows = []; pair_meta = []
        reused = sum(
            (str(row["pair_id"]), int(row["candidate_A"]), int(row["candidate_B"])) in phase_a
            for row in rows
        )
        generated = len(rows) - reused
        zero_candidate_pairs = sum(int(meta["candidate_count"]) == 0 for meta in pair_meta)
        geometry_valid = sum(row["candidate_geometry_valid"] for row in rows)
        hard = sum(row["m2_A"]["hard_failure"] or row["m2_B"]["hard_failure"] for row in rows)
        soft = sum(
            (row["m2_A"]["soft_failure"] or row["m2_B"]["soft_failure"])
            and not (row["m2_A"]["hard_failure"] or row["m2_B"]["hard_failure"])
            for row in rows
        )
        combined: List[Tuple[Mapping[str, Any], bool]] = [
            *((record, True) for record in positives[split]),
            *((record, False) for record in negatives[split]),
        ]
        start_pair = len(pair_meta)
        for offset, (record, adjacent) in enumerate(combined[start_pair:], start_pair + 1):
            pair_id = str(record["id"])
            score = load_pair_score(context["audit"], split, pair_id) if adjacent else _load_negative_score(output, split, pair_id)
            product = int(score["candidate_count_A"]) * int(score["candidate_count_B"])
            ranked = stable_topn(score["candidate_pair_scores"], n) if product else []
            target, mapping = manifest_target(record, score) if adjacent else (None, {"mapped": False, "reason": "negative_pair"})
            pair_row = positive_rows[pair_id] if adjacent else {
                "primary_pair_reason": "negative_pair",
                "legacy": {"target_pairs": [], "target_present": False, "selected_correct": False},
            }
            current: List[Dict[str, Any]] = []
            for candidate in ranked:
                key = (pair_id, int(candidate.candidate_a), int(candidate.candidate_b))
                if key in phase_a:
                    row = copy.deepcopy(phase_a[key]); reused += 1
                else:
                    row = candidate_feature_row(
                        augmented, store, split, record, pair_row, score,
                        candidate, ranked, target, dustbin,
                        pair_is_adjacent=adjacent,
                    )
                    generated += 1
                if bool(row["pair_is_adjacent"]) != adjacent:
                    raise ValueError(f"candidate adjacency mismatch: {pair_id}")
                current.append(row); rows.append(row)
                geometry_valid += int(row["candidate_geometry_valid"])
                side_hard = bool(row["m2_A"]["hard_failure"] or row["m2_B"]["hard_failure"])
                side_soft = bool(row["m2_A"]["soft_failure"] or row["m2_B"]["soft_failure"])
                hard += int(side_hard); soft += int(side_soft and not side_hard)
            if not current:
                zero_candidate_pairs += 1
            strict_positions = [index for index, row in enumerate(current) if row["candidate_is_true"]]
            if len(strict_positions) > 1:
                raise ValueError(f"multiple strict targets in shortlist: {pair_id}")
            physical_positions = [index for index, row in enumerate(current) if row["candidate_is_any_physical_true"]]
            best = [int(value) for value in score["best_candidate_pair"]]
            baseline_match = best[0] >= 0 and best[1] >= 0
            pair_meta.append({
                "schema_version": "zind-salve-phase-b-pair-meta-v1",
                "split": split, "pair_id": pair_id,
                "house_id": str(record["scene_id"]), "floor_id": str(record["floor_id"]),
                "pair_is_adjacent": adjacent,
                "candidate_count": len(current),
                "strict_target_position": strict_positions[0] if strict_positions else None,
                "physical_target_positions": physical_positions,
                "shortlist_has_target": bool(strict_positions),
                "legacy_target_present": bool(pair_row["legacy"].get("target_present", False)),
                "target_mapping": mapping,
                "baseline_predicted_match": baseline_match,
                "baseline_best_candidate_pair": best,
                "baseline_physical_correct": bool(pair_row["legacy"].get("selected_correct", False)),
                "baseline_primary_reason": pair_row["primary_pair_reason"],
                "input_uses_gt": False, "test_data_accessed": False,
            })
            if offset % 256 == 0 or offset == len(combined):
                log(f"Phase-B features {split} pairs {offset}/{len(combined)} rows={len(rows)} reused={reused}")
            if offset == len(positives[split]):
                write_jsonl(partial_rows_path, rows)
                write_jsonl(partial_meta_path, pair_meta)
                log(f"Phase-B features {split} positive partial checkpoint saved")
        feature_path = output / f"phase_B_candidate_features_{split}.jsonl"
        meta_path = output / f"phase_B_pair_meta_{split}.jsonl"
        write_jsonl(feature_path, rows); write_jsonl(meta_path, pair_meta)
        summaries[split] = {
            "pair_count": len(pair_meta),
            "positive_pair_count": len(positives[split]), "negative_pair_count": len(negatives[split]),
            "candidate_row_count": len(rows), "phase_A_reused_row_count": reused,
            "newly_generated_row_count": generated, "zero_candidate_pair_count": zero_candidate_pairs,
            "strict_target_shortlist_count": sum(meta["shortlist_has_target"] for meta in pair_meta if meta["pair_is_adjacent"]),
            "legacy_physical_target_present_count": sum(meta["legacy_target_present"] for meta in pair_meta if meta["pair_is_adjacent"]),
            "candidate_geometry_valid_count": geometry_valid,
            "candidate_geometry_hard_failure_count": hard,
            "candidate_geometry_soft_failure_count": soft,
            "feature_sha256": sha256_file(feature_path), "pair_meta_sha256": sha256_file(meta_path),
            "all_finite": all(all(math.isfinite(value) for value in row["feature_vector"]) for row in rows),
        }
    summary = {
        "schema_version": "zind-salve-phase-b-feature-summary-v1",
        "shortlist_N": n, "splits": summaries,
        "input_uses_gt": False, "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    atomic_json(summary_path, summary)
    atomic_text(marker, sha256_file(summary_path) + "\n")
    return summary


def _phase_b_groups(output: Path, split: str) -> List[Dict[str, Any]]:
    rows_by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(output / f"phase_B_candidate_features_{split}.jsonl"):
        rows_by_pair[str(row["pair_id"])].append(row)
    groups = []
    for meta in read_jsonl(output / f"phase_B_pair_meta_{split}.jsonl"):
        rows = sorted(rows_by_pair[str(meta["pair_id"])], key=lambda value: int(value["candidate_rank"]))
        if len(rows) != int(meta["candidate_count"]):
            raise ValueError(f"Phase-B row count mismatch: {meta['pair_id']}")
        groups.append({**meta, "rows": rows})
    return groups


def _phase_b_house_split(groups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    houses = sorted({str(group["house_id"]) for group in groups})
    shuffled = list(houses); np.random.RandomState(SEED).shuffle(shuffled)
    dev_count = max(1, int(math.ceil(0.20 * len(shuffled))))
    dev = set(shuffled[:dev_count]); subtrain = set(shuffled[dev_count:])
    return {
        "schema_version": "zind-salve-phase-b-house-split-v1", "seed": SEED,
        "algorithm": "sorted train houses; RandomState(123); first ceil(20%) dev",
        "subtrain_houses": sorted(subtrain), "train_dev_houses": sorted(dev),
        "house_overlap": sorted(subtrain & dev), "val_used": False,
        "test_data_accessed": False,
    }


class DustbinLinearRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__(); self.candidate = nn.Linear(input_dim, 1); self.dustbin_score = nn.Parameter(torch.zeros(()))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        candidate = self.candidate(value).squeeze(-1).masked_fill(~mask, -1e9)
        dustbin = self.dustbin_score.expand(value.shape[0], 1)
        return torch.cat((candidate, dustbin), dim=1)


class DustbinMLPRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.dustbin_score = nn.Parameter(torch.zeros(()))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        candidate = self.candidate(value).squeeze(-1).masked_fill(~mask, -1e9)
        dustbin = self.dustbin_score.expand(value.shape[0], 1)
        return torch.cat((candidate, dustbin), dim=1)


def _phase_b_tensors(
    groups: Sequence[Mapping[str, Any]], norm: Mapping[str, Any], indices: Sequence[int], maximum: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = np.zeros((len(groups), maximum, len(indices)), dtype=np.float32)
    masks = np.zeros((len(groups), maximum), dtype=bool)
    targets = np.full(len(groups), maximum, dtype=np.int64)
    adjacent = np.zeros(len(groups), dtype=bool)
    for group_index, group in enumerate(groups):
        if len(group["rows"]) > maximum:
            raise ValueError("Phase-B candidate rows exceed frozen shortlist")
        for row_index, row in enumerate(group["rows"]):
            values[group_index, row_index] = normalized_vector(row, norm)[list(indices)]
            masks[group_index, row_index] = True
        if group["strict_target_position"] is not None:
            targets[group_index] = int(group["strict_target_position"])
        adjacent[group_index] = bool(group["pair_is_adjacent"])
    return (
        torch.from_numpy(values), torch.from_numpy(masks),
        torch.from_numpy(targets), torch.from_numpy(adjacent),
    )


def _dev_score(
    model: nn.Module, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Dict[str, float]:
    x, mask, target, adjacent = data
    model.eval()
    with torch.no_grad():
        score = model(x, mask); predicted = score.argmax(-1)
    dustbin = mask.shape[1]
    positive_count = int(adjacent.sum()); negative_count = int((~adjacent).sum())
    strict_s3 = int(((predicted == target) & adjacent & (target != dustbin)).sum())
    negative_rejected = int(((predicted == dustbin) & (~adjacent)).sum())
    s3_rate = strict_s3 / max(1, positive_count)
    rejection = negative_rejected / max(1, negative_count)
    harmonic = 2 * s3_rate * rejection / (s3_rate + rejection) if s3_rate + rejection else 0.0
    return {
        "strict_s3": strict_s3, "positive_count": positive_count,
        "negative_rejected": negative_rejected, "negative_count": negative_count,
        "s3_rate": s3_rate, "negative_rejection": rejection,
        "harmonic": harmonic, "pair_accuracy": float((predicted == target).float().mean()),
    }


def _train_phase_b_ranker(
    kind: str, subtrain: Sequence[Mapping[str, Any]], dev: Sequence[Mapping[str, Any]],
    norm: Mapping[str, Any], indices: Sequence[int], seed: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model: nn.Module = DustbinLinearRanker(len(indices)) if kind == "linear" else DustbinMLPRanker(len(indices))
    lr = 1e-2 if kind == "linear" else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_data = _phase_b_tensors(subtrain, norm, indices)
    dev_data = _phase_b_tensors(dev, norm, indices)
    x, mask, target, _ = train_data
    generator = torch.Generator().manual_seed(seed)
    best = None; stale = 0; history = []
    for epoch in range(200):
        model.train(); order = torch.randperm(len(target), generator=generator); losses = []
        for offset in range(0, len(order), 64):
            batch = order[offset:offset + 64]
            score = model(x[batch], mask[batch]); loss = F.cross_entropy(score, target[batch])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        dev_metric = _dev_score(model, dev_data)
        mean_loss = float(np.mean(losses))
        key = (dev_metric["harmonic"], dev_metric["pair_accuracy"], -mean_loss, -epoch)
        history.append({"epoch": epoch, "train_loss": mean_loss, **dev_metric})
        if best is None or key > best[0]:
            best = (key, copy.deepcopy(model.state_dict()), epoch); stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    model.load_state_dict(best[1]); model.eval()
    return model, {
        "kind": kind, "optimizer": "Adam", "lr": lr, "weight_decay": 1e-4,
        "batch_size": 64, "max_epochs": 200, "patience": 20,
        "selection_metric": "train-dev harmonic(strict S3, negative rejection)",
        "best_epoch": best[2], "history": history,
        "dev": _dev_score(model, dev_data), "val_used": False,
        "test_data_accessed": False,
    }


def _model_predictions(
    model: nn.Module, groups: Sequence[Mapping[str, Any]], norm: Mapping[str, Any], indices: Sequence[int],
) -> Tuple[List[int], List[List[float]]]:
    x, mask, _, _ = _phase_b_tensors(groups, norm, indices)
    model.eval()
    with torch.no_grad():
        score = model(x, mask); predicted = score.argmax(-1)
    return predicted.tolist(), score.tolist()


def _positive_house_bootstrap(
    groups: Sequence[Mapping[str, Any]], success: Mapping[str, bool], baseline: Mapping[str, bool],
) -> Dict[str, Any]:
    positives = [group for group in groups if group["pair_is_adjacent"]]
    by_house: Dict[str, List[str]] = defaultdict(list)
    for group in positives:
        by_house[str(group["house_id"])].append(str(group["pair_id"]))
    houses = sorted(by_house); rng = np.random.RandomState(SEED); draws = []
    for _ in range(2000):
        sample = rng.choice(houses, size=len(houses), replace=True)
        ids = [pair_id for house in sample for pair_id in by_house[str(house)]]
        draws.append(float(np.mean([int(success[pair_id]) - int(baseline[pair_id]) for pair_id in ids])))
    observed = float(np.mean([int(success[group["pair_id"]]) - int(baseline[group["pair_id"]]) for group in positives]))
    return {
        "observed": observed, "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)), "replicates": 2000,
        "house_count": len(houses),
    }


def _phase_b_metrics(
    name: str, groups: Sequence[Mapping[str, Any]], predictions: Optional[Sequence[int]],
    scores: Optional[Sequence[Sequence[float]]], baseline_success: Mapping[str, bool],
) -> Dict[str, Any]:
    positive_count = sum(group["pair_is_adjacent"] for group in groups)
    negative_count = len(groups) - positive_count; dustbin_index = 3
    outcomes: Dict[str, Dict[str, bool]] = {}
    ranking_correct = accepted_positive = false_positive = 0
    strict_s3 = physical_s3 = original_retained = wrong_recovered = dust_recovered = 0
    improved_houses = set(); degraded = 0
    for index, group in enumerate(groups):
        pair_id = str(group["pair_id"]); adjacent = bool(group["pair_is_adjacent"])
        if predictions is None:
            predicted_match = bool(group["baseline_predicted_match"])
            physical_correct = bool(group["baseline_physical_correct"])
            strict_correct = physical_correct and bool(group["shortlist_has_target"])
            selected_position = None
        else:
            selected_position = int(predictions[index])
            predicted_match = selected_position != dustbin_index
            selected = group["rows"][selected_position] if predicted_match and selected_position < len(group["rows"]) else None
            strict_correct = bool(selected is not None and selected["candidate_is_true"])
            physical_correct = bool(selected is not None and selected["candidate_is_any_physical_true"])
        if adjacent and group["rows"]:
            if scores is None:
                rank_position = 0
            else:
                candidate_scores = scores[index][:len(group["rows"])]
                rank_position = min(range(len(candidate_scores)), key=lambda value: (-float(candidate_scores[value]), value))
            ranking_correct += int(group["rows"][rank_position]["candidate_is_any_physical_true"])
        if adjacent:
            accepted_positive += int(predicted_match); strict_s3 += int(strict_correct); physical_s3 += int(physical_correct)
            original_retained += int(baseline_success[pair_id] and physical_correct)
            degraded += int(baseline_success[pair_id] and not physical_correct)
            if physical_correct and not baseline_success[pair_id]: improved_houses.add(str(group["house_id"]))
            if physical_correct and group["baseline_primary_reason"] == "target_present_wrong_selection": wrong_recovered += 1
            if physical_correct and group["baseline_primary_reason"] == "target_present_dustbin_reject": dust_recovered += 1
        else:
            false_positive += int(predicted_match)
        outcomes[pair_id] = {"predicted_match": predicted_match, "physical_s3": physical_correct, "strict_s3": strict_correct}
    true_negative = negative_count - false_positive
    precision = accepted_positive / max(1, accepted_positive + false_positive)
    recall = accepted_positive / max(1, positive_count)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rejection = true_negative / max(1, negative_count)
    e2e = physical_s3 / max(1, positive_count)
    harmonic = 2 * e2e * rejection / (e2e + rejection) if e2e + rejection else 0.0
    baseline_count = sum(baseline_success.values())
    bootstrap = _positive_house_bootstrap(groups, {pair_id:value["physical_s3"] for pair_id,value in outcomes.items() if pair_id in baseline_success}, baseline_success)
    return {
        "name": name, "positive_count": positive_count, "negative_count": negative_count,
        "candidate_recall_ceiling": sum(group["legacy_target_present"] for group in groups if group["pair_is_adjacent"]) / max(1, positive_count),
        "candidate_recall_count": sum(group["legacy_target_present"] for group in groups if group["pair_is_adjacent"]),
        "positive_match_recall": recall,
        "candidate_conditional_top1": physical_s3 / max(1, sum(group["legacy_target_present"] for group in groups if group["pair_is_adjacent"])),
        "candidate_rerank_top1_ignoring_dustbin": ranking_correct / max(1, sum(group["legacy_target_present"] for group in groups if group["pair_is_adjacent"])),
        "strict_end_to_end_s3_count": strict_s3,
        "end_to_end_s3_count": physical_s3, "end_to_end_s3": e2e,
        "negative_rejection_count": true_negative, "negative_rejection": rejection,
        "precision": precision, "recall": recall, "f1": f1,
        "balanced_accuracy": 0.5 * (recall + rejection),
        "balanced_flow_score": 0.5 * (e2e + rejection), "harmonic": harmonic,
        "false_positive_count": false_positive,
        "dustbin_reject_recovered": dust_recovered, "wrong_selection_recovered": wrong_recovered,
        "original_correct_retained": original_retained,
        "original_correct_denominator": baseline_count,
        "original_correct_retention_rate": original_retained / max(1, baseline_count),
        "original_correct_degraded": degraded, "improved_house_count": len(improved_houses),
        "house_bootstrap": bootstrap, "outcomes": outcomes,
    }


def train_and_evaluate_phase_b(
    args: argparse.Namespace, context: Mapping[str, Any], output: Path, log: AuditLogger,
) -> Dict[str, Any]:
    if not (output / ".phase_B_features_complete").is_file():
        raise RuntimeError("Phase-B features must complete before training")
    train_groups = _phase_b_groups(output, "train"); val_groups = _phase_b_groups(output, "val")
    split = _phase_b_house_split(train_groups); atomic_json(output / "phase_B_subtrain_dev_split.json", split)
    sub_houses = set(split["subtrain_houses"]); dev_houses = set(split["train_dev_houses"])
    subtrain = [group for group in train_groups if group["house_id"] in sub_houses]
    dev = [group for group in train_groups if group["house_id"] in dev_houses]
    norm = normalization(subtrain); atomic_json(output / "phase_B_normalization.json", norm)
    indices = feature_indices(("t3", "m2", "relative", "bev"))
    models = {}; protocols = {}; artifacts = {}
    for name, kind, seed in (("E1_logistic", "linear", SEED), ("E2_mlp", "mlp", SEED + 1)):
        model, protocol = _train_phase_b_ranker(kind, subtrain, dev, norm, indices, seed)
        path = output / f"phase_B_{name}.pt"; temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        torch.save({"state_dict": model.state_dict(), "protocol": protocol, "indices": indices}, temporary); os.replace(temporary, path)
        models[name] = model; protocols[name] = protocol
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
        log(f"trained Phase-B {name} dev_harmonic={protocol['dev']['harmonic']:.4f}")
    frozen = {
        "schema_version": "zind-salve-phase-b-frozen-before-val-v1",
        'portallayout.models': artifacts, "protocols": protocols,
        "normalization_sha256": sha256_file(output / "phase_B_normalization.json"),
        "split_sha256": sha256_file(output / "phase_B_subtrain_dev_split.json"),
        "val_used": False, "test_data_accessed": False,
    }
    atomic_json(output / "phase_B_frozen_before_val.json", frozen)

    baseline_success = {
        str(group["pair_id"]): bool(group["baseline_physical_correct"])
        for group in val_groups if group["pair_is_adjacent"]
    }
    results = {"E0_T3_legacy": _phase_b_metrics("E0_T3_legacy", val_groups, None, None, baseline_success)}
    for name, model in models.items():
        predicted, scores = _model_predictions(model, val_groups, norm, indices)
        results[name] = _phase_b_metrics(name, val_groups, predicted, scores, baseline_success)
    baseline = results["E0_T3_legacy"]
    leakage = {
        "feature_names_predicted_only": feature_names_are_predicted_only(BASE_FEATURES),
        "rows_input_uses_gt_false": all(not row["input_uses_gt"] for group in train_groups + val_groups for row in group["rows"]),
        "train_val_house_disjoint": not bool({g["house_id"] for g in train_groups} & {g["house_id"] for g in val_groups}),
        "test_not_accessed": True,
    }
    for name in ("E1_logistic", "E2_mlp"):
        value = results[name]
        gates = {
            "S3_strictly_increases": value["end_to_end_s3_count"] > baseline["end_to_end_s3_count"],
            "harmonic_strictly_increases": value["harmonic"] > baseline["harmonic"],
            "negative_rejection_drop_at_most_3pp": value["negative_rejection"] >= baseline["negative_rejection"] - 0.03,
            "original_501_retention_at_least_95pct": value["original_correct_retention_rate"] >= 0.95,
            "improvement_multiple_houses": value["improved_house_count"] >= 2,
            "paired_S3_bootstrap_ci_low_nonnegative": value["house_bootstrap"]["ci95_low"] >= 0.0,
            "no_GT_feature_leakage": all(leakage.values()),
        }
        value["gates"] = gates; value["passed"] = all(gates.values())
    passed = any(results[name]["passed"] for name in ("E1_logistic", "E2_mlp"))
    summary = {
        "schema_version": "zind-salve-phase-b-v2",
        "status": "passed" if passed else "failed_hard_gate",
        'portallayout.models': results, "model_artifacts": artifacts,
        "training_protocols": protocols, "leakage_gates": leakage,
        "passed": passed, "cnn_training_started": False,
        "active_splits": ["train", "val"], "test_data_accessed": False,
    }
    atomic_json(output / "phase_B_summary.json", summary)
    atomic_json(output / "phase_B_house_bootstrap.json", {name:value["house_bootstrap"] for name,value in results.items()})
    atomic_text(output / (".phase_B_passed" if passed else ".phase_B_failed"), sha256_file(output / "phase_B_summary.json") + "\n")
    log(f"Phase B {'PASS' if passed else 'FAIL'}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output = reject_test_path(args.output_dir, "output")
    output.mkdir(parents=True, exist_ok=True)
    log = AuditLogger(output / "phase_B.log")
    context = load_base(args, output)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    atomic_text(output / "phase_B_command.txt", command + "\n")
    atomic_text(output / "phase_B_resume_command.txt", command + ("" if args.resume else " --resume") + "\n")
    phases = [args.phase] if args.phase != "all" else ["prepare", "smoke", "cache", "score", "features", "train", "report"]
    handlers = {
        "prepare": prepare_negative_index,
        "smoke": smoke_negative_cache,
        "cache": cache_negative_rooms,
        "score": score_negative_pairs,
        "features": generate_phase_b_features,
        "train": train_and_evaluate_phase_b,
    }
    for phase in phases:
        log(f"start phase={phase}")
        if phase not in handlers:
            raise NotImplementedError(f"Phase-B phase is not implemented yet: {phase}")
        handlers[phase](args, context, output, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
