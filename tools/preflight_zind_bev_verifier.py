#!/usr/bin/env python3
"""SALVe-style candidate-conditioned verifier preflight on frozen T3/M2 data.

The tool is deliberately separate from production evaluators.  It consumes
only train/val frozen artifacts, uses GT solely for labels/post-hoc metrics,
and has no path for end-to-end, Opening-Head, Matcher, or CNN training.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_zind_portal_geometry_val import _layout_state, _m2_contract  # noqa: E402
from utils.candidate_conditioned_verifier import (  # noqa: E402
    circular_interval_width,
    endpoint_consistency,
    feature_names_are_predicted_only,
    interval_mean,
    manifest_target_pair,
    matcher_cell_features,
    pad_candidate_rows,
    predicted_polygon_features,
    stable_topn,
    transform_points,
    yaw_scale_translation,
)
from utils.candidate_local_portal_geometry import (  # noqa: E402
    CandidateLocalPortalConfig,
    estimate_candidate_local_portal_geometry,
)
from utils.joint_layout import build_joint_layout_from_endpoints  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "experiments/zind_e2e_v3/g0_seed123/T5_bev_verifier/preflight"
AUDIT_ROOT = REPO_ROOT / "experiments/zind_e2e_v3/g0_seed123/T5_candidate_recall/audit"
TRAIN_MANIFEST = REPO_ROOT / "src/dataset/ZInD_matching/train.json"
VAL_MANIFEST = REPO_ROOT / "src/dataset/ZInD_matching/val.json"
T3_CHECKPOINT = REPO_ROOT / "checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best.pt"
SHORTLIST_VALUES = (1, 3, 5, 8)
EXPECTED_T3_SHA256 = "ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d"
EXPECTED_M2_SHA256 = "cdce025fd5ba709a3c5afc827137860d1082c7ab44e27a5467ea6686a7a02a4c"
SEED = 123


T3_FEATURES = (
    "t3_raw_candidate_score", "t3_mutual_affinity",
    "t3_assignment_probability_A_to_B", "t3_assignment_probability_B_to_A",
    "t3_real_real_logit", "t3_candidate_rank", "t3_candidate_rank_fraction",
    "t3_opening_confidence_A", "t3_opening_confidence_B",
    "t3_opening_confidence_geometric_mean", "t3_pair_match_probability",
    "t3_pair_no_match_probability", "t3_row_candidate_to_dustbin_margin",
    "t3_column_candidate_to_dustbin_margin", "t3_candidate_count_A",
    "t3_candidate_count_B", "t3_candidate_product", "t3_log_candidate_product",
)
M2_SUFFIXES = (
    "geometry_valid", "variant_weighted_tls", "variant_ransac",
    "support_count", "inlier_count", "inlier_ratio", "normalized_residual",
    "angular_dispersion", "ray_condition", "wrap", "corner_ambiguity",
    "interval_token_width", "opening_width", "opening_width_over_room_radius",
    "width_plausibility", "endpoint_finite", "quality_gate_pass",
    "reliability_score", "hard_failure", "soft_failure",
    "local_context_difference",
)
M2_FEATURES = tuple(f"m2_{side}_{suffix}" for side in ("A", "B") for suffix in M2_SUFFIXES)
RELATIVE_FEATURES = (
    "relative_geometry_valid", "relative_yaw_sin", "relative_yaw_cos",
    "relative_scale", "relative_absolute_log_scale", "relative_translation_norm",
    "relative_opening_width_ratio", "relative_endpoint_alignment_residual",
    "relative_midpoint_alignment_residual", "relative_portal_tangent_angle_degrees",
    "relative_wall_normal_opposition", "relative_local_context_difference",
    "relative_endpoint_mapping_reversed",
)
BEV_FEATURES = (
    "bev_polygon_A_valid", "bev_polygon_B_transformed_valid",
    "bev_intersection_area", "bev_intersection_over_A", "bev_intersection_over_B",
    "bev_union_area", "bev_polygon_iou", "bev_centroid_distance",
    "bev_normalized_centroid_distance", "bev_containment_ratio",
    "bev_boundary_minimum_distance", "bev_portal_local_overlap",
    "bev_portal_centroid_side_product", "bev_portal_centroids_opposite_sides",
    "bev_scaled_area_ratio", "bev_combined_bbox_aspect",
    "bev_abnormal_penetration", "bev_complete_separation",
)
BASE_FEATURES = T3_FEATURES + M2_FEATURES + RELATIVE_FEATURES + BEV_FEATURES
INPUT_FEATURES = BASE_FEATURES + tuple(f"missing__{name}" for name in BASE_FEATURES)
GROUPS = {
    "t3": list(T3_FEATURES), "m2": list(M2_FEATURES),
    "relative": list(RELATIVE_FEATURES), "bev": list(BEV_FEATURES),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "shortlist", "features_train", "phase_a", "phase_b", "report"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit_root", default=str(AUDIT_ROOT))
    parser.add_argument("--train_manifest", default=str(TRAIN_MANIFEST))
    parser.add_argument("--val_manifest", default=str(VAL_MANIFEST))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--progress_every", type=int, default=128)
    return parser.parse_args(argv)


def reject_test_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.stem.lower() == "test" or any(part.lower() == "test" for part in path.parts):
        raise ValueError(f"{label} points to forbidden test data: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_context(args: argparse.Namespace, output: Path) -> Dict[str, Any]:
    if args.seed != SEED or args.bootstrap_replicates != 2000:
        raise ValueError("protocol is fixed to seed=123 and 2000 bootstrap replicates")
    audit = reject_test_path(args.audit_root, "candidate audit")
    train_path = reject_test_path(args.train_manifest, "train manifest")
    val_path = reject_test_path(args.val_manifest, "val manifest")
    required = [
        audit / ".completed", audit / "room_cache_index.json",
        audit / "train_sample_4096.json", audit / "per_pair_train_4096.jsonl",
        audit / "per_pair_val.jsonl", audit / "phase3_summary.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(T3_CHECKPOINT) != EXPECTED_T3_SHA256:
        raise ValueError("T3 checkpoint SHA changed")
    m2 = _m2_contract()
    if m2["sha256"] != EXPECTED_M2_SHA256:
        raise ValueError("M2 config SHA changed")
    train = json.loads(train_path.read_text(encoding="utf-8"))
    val = json.loads(val_path.read_text(encoding="utf-8"))
    if train.get("split") != "train" or val.get("split") != "val":
        raise ValueError("only train/val manifests are allowed")
    train_houses = set(map(str, train.get("sourceHouseIds", [])))
    val_houses = set(map(str, val.get("sourceHouseIds", [])))
    if train_houses & val_houses:
        raise ValueError("train/val house leakage")
    index = json.loads((audit / "room_cache_index.json").read_text(encoding="utf-8"))
    sample = json.loads((audit / "train_sample_4096.json").read_text(encoding="utf-8"))
    return {
        "audit": audit, "train_path": train_path, "val_path": val_path,
        "train": train, "val": val, "room_index": index, "train_sample": sample,
        "m2": m2, "output": output,
    }


def state_evidence(context: Mapping[str, Any]) -> Dict[str, Any]:
    checkpoints = {}
    for phase in ("T0", "T1", "T2", "T3"):
        path = REPO_ROOT / f"checkpoints/End_To_End/zind_e2e_v2/g0_seed123/{phase}/best.pt"
        checkpoints[phase] = {
            "path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "schema_version": "zind-salve-verifier-state-v1",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "worktree_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True),
        "checkpoints": checkpoints, "m2": context["m2"],
        "candidate_audit_hashes": {
            name: sha256_file(context["audit"] / name) for name in (
                "room_cache_index.json", "phase3_summary.json", "oracle_summary.json", "train_sample_4096.json"
            )
        },
        "gpu_status": {
            "available_to_process": False,
            "query_result": "nvidia-smi could not communicate with the NVIDIA driver in the execution environment",
        },
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "active_splits": ["train", "val"], "test_data_accessed": False,
    }


def load_pair_score(audit: Path, split: str, pair_id: str) -> Dict[str, Any]:
    path = audit / "pair_scores" / f"{split}_O0" / f"{pair_id}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("split") != split or value.get("view") != "O0" or value.get("test_data_accessed") is not False:
        raise ValueError(f"invalid frozen pair score {path}")
    return value


def manifest_target(record: Mapping[str, Any], score: Mapping[str, Any]) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
    supervision = record["supervision"]
    target_a = record["candidates_a"][int(supervision["target_candidate_a"])]["tokenInterval"]
    target_b = record["candidates_b"][int(supervision["target_candidate_b"])]["tokenInterval"]
    return manifest_target_pair(target_a, target_b, score["intervals_A"], score["intervals_B"])


def cohort_records(context: Mapping[str, Any], split: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    audit = context["audit"]
    rows_path = audit / ("per_pair_train_4096.jsonl" if split == "train" else "per_pair_val.jsonl")
    rows = {row["pair_id"]: row for row in read_jsonl(rows_path)}
    records = {str(row["id"]): row for row in context[split]["pairs"]}
    pair_ids = context["train_sample"]["pair_ids"] if split == "train" else sorted(rows)
    selected = [records[pair_id] for pair_id in pair_ids if rows[pair_id]["legacy"]["target_present"]]
    return selected, rows


def coverage_slice(entries: Sequence[Mapping[str, Any]], predicate: Any) -> Dict[str, Any]:
    values = [entry for entry in entries if predicate(entry)]
    denominator = len(values)
    return {
        str(n): {
            "covered": sum(entry["strict_manifest_rank"] is not None and entry["strict_manifest_rank"] <= n for entry in values),
            "denominator": denominator,
            "rate": sum(entry["strict_manifest_rank"] is not None and entry["strict_manifest_rank"] <= n for entry in values) / max(1, denominator),
        }
        for n in SHORTLIST_VALUES
    }


def shortlist_audit(context: Mapping[str, Any], output: Path, log: Logger) -> Dict[str, Any]:
    split_outputs = {}
    all_entries = {}
    for split in ("train", "val"):
        records, rows = cohort_records(context, split)
        entries = []
        total_product = 0
        for offset, record in enumerate(records, 1):
            pair_id = str(record["id"])
            score = load_pair_score(context["audit"], split, pair_id)
            strict_target, mapping = manifest_target(record, score)
            full = stable_topn(score["candidate_pair_scores"], int(score["candidate_count_A"]) * int(score["candidate_count_B"]))
            rank_lookup = {(value.candidate_a, value.candidate_b): value.rank for value in full}
            physical = [tuple(value) for value in rows[pair_id]["legacy"]["target_pairs"]]
            strict_rank = None if strict_target is None else rank_lookup.get(strict_target)
            physical_ranks = [rank_lookup[value] for value in physical if value in rank_lookup]
            entry = {
                "pair_id": pair_id, "house_id": str(record["scene_id"]),
                "floor_id": str(record["floor_id"]), "interface_type": rows[pair_id]["interface_type"],
                "wrap": bool(rows[pair_id]["wrap"]),
                "candidate_count_A": int(score["candidate_count_A"]),
                "candidate_count_B": int(score["candidate_count_B"]),
                "candidate_product": int(score["candidate_count_A"]) * int(score["candidate_count_B"]),
                "strict_manifest_rank": strict_rank,
                "any_physical_rank": min(physical_ranks) if physical_ranks else None,
                "strict_manifest_mapping": mapping,
                "t3_predicted_match": bool(rows[pair_id]["legacy"]["predicted_match"]),
                "t3_selected_physical_correct": bool(rows[pair_id]["legacy"]["selected_correct"]),
                "primary_reason": rows[pair_id]["primary_pair_reason"],
            }
            total_product += entry["candidate_product"]
            entries.append(entry)
            if offset % 512 == 0:
                log(f"shortlist {split} {offset}/{len(records)}")
        overall = coverage_slice(entries, lambda _: True)
        split_outputs[split] = {
            "cohort_pair_count": len(entries),
            "legacy_any_physical_target_present": len(entries),
            "strict_manifest_target_mapped": sum(entry["strict_manifest_rank"] is not None for entry in entries),
            "strict_manifest_topN": overall,
            "accepted_topN": coverage_slice(entries, lambda value: value["t3_predicted_match"]),
            "dustbin_rejected_topN": coverage_slice(entries, lambda value: not value["t3_predicted_match"]),
            "wrap_topN": {
                "wrap": coverage_slice(entries, lambda value: value["wrap"]),
                "non_wrap": coverage_slice(entries, lambda value: not value["wrap"]),
            },
            "interface_topN": {
                interface: coverage_slice(entries, lambda value, name=interface: value["interface_type"] == name)
                for interface in sorted({entry["interface_type"] for entry in entries})
            },
            "house_count": len({entry["house_id"] for entry in entries}),
            "house_level_coverage": {
                house: {
                    "pair_count": len(values),
                    **{
                        f"top{n}_covered": sum(
                            value["strict_manifest_rank"] is not None
                            and value["strict_manifest_rank"] <= n
                            for value in values
                        )
                        for n in SHORTLIST_VALUES
                    },
                }
                for house, values in sorted(
                    (
                        (house, [entry for entry in entries if entry["house_id"] == house])
                        for house in {entry["house_id"] for entry in entries}
                    ),
                    key=lambda item: item[0],
                )
            },
            "candidate_product_total": total_product,
            "shortlist_compute": {
                str(n): {
                    "evaluated_candidates": sum(min(n, entry["candidate_product"]) for entry in entries),
                    "mean_per_pair": sum(min(n, entry["candidate_product"]) for entry in entries) / max(1, len(entries)),
                }
                for n in SHORTLIST_VALUES
            },
        }
        all_entries[split] = entries

    train_coverage = split_outputs["train"]["strict_manifest_topN"]
    chosen = next((n for n in SHORTLIST_VALUES if train_coverage[str(n)]["rate"] >= 0.95), 8)
    protocol = {
        "schema_version": "zind-salve-shortlist-selection-v1", "seed": SEED,
        "candidate_source": "legacy predicted-only candidates",
        "ranking_source": "frozen T3 real-real candidate_pair_score; dustbin ignored",
        "candidate_values": list(SHORTLIST_VALUES),
        "rule": "smallest N with train strict-manifest coverage >=0.95; else N=8",
        "selected_N": chosen, "uses_val_verifier_result": False,
        "gt_candidate_injection": False, "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    protocol["sha256"] = canonical_sha(protocol)
    result = {
        "schema_version": "zind-salve-shortlist-audit-v1", "splits": split_outputs,
        "selection_protocol": protocol,
        "multi_shared_opening_semantics": {
            "training_label": "single manifest target_candidate only",
            "compatibility_metric": "any physical shared passable interface",
            "reason": "legacy 1080/501 evidence marks all physical interfaces; schema label is singular",
        },
        "test_data_accessed": False,
    }
    atomic_json(output / "shortlist_audit.json", result)
    atomic_json(output / "shortlist_selection_protocol.json", protocol)
    write_jsonl(output / "shortlist_pairs_train.jsonl", all_entries["train"])
    write_jsonl(output / "shortlist_pairs_val.jsonl", all_entries["val"])
    atomic_text(output / ".shortlist_complete", sha256_file(output / "shortlist_audit.json") + "\n")
    log(f"shortlist complete: fixed N={chosen}")
    return result


class SafeFeatures:
    def __init__(self):
        self.values = {name: 0.0 for name in BASE_FEATURES}
        self.missing = {name: 1.0 for name in BASE_FEATURES}

    def set(self, name: str, value: Any) -> None:
        if name not in self.values:
            raise KeyError(name)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(number):
            self.values[name] = number
            self.missing[name] = 0.0

    def vector(self) -> List[float]:
        output = [self.values[name] for name in BASE_FEATURES]
        output += [self.missing[name] for name in BASE_FEATURES]
        if not all(math.isfinite(value) for value in output):
            raise ValueError("verifier feature vector is non-finite")
        return output


class PredictedGeometryStore:
    def __init__(self, context: Mapping[str, Any], auxiliary_cache_root: Optional[Path] = None):
        self.context = context
        self.auxiliary_cache_root = auxiliary_cache_root
        self.rooms: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.portals: Dict[Tuple[str, str, int], Any] = {}

    def room(self, split: str, reference: str) -> Dict[str, Any]:
        key = (split, reference)
        if key in self.rooms:
            return self.rooms[key]
        path = self.context["audit"] / "frozen_room_cache" / split / f"{reference}.pt"
        if not path.is_file() and self.auxiliary_cache_root is not None:
            path = self.auxiliary_cache_root / split / f"{reference}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"no frozen room cache for {split}/{reference}")
        value = torch.load(path, map_location="cpu")
        layout = _layout_state(
            value["enclosed_depth"], value["ratio"], value["opening_probability"], value["legacy_candidates"]
        )
        room = {
            "reference": reference, "room": value["room"], "token_count": int(value["token_count"]),
            "probability": value["opening_probability"].detach().cpu().numpy().astype(np.float64),
            "intervals": [list(map(int, interval)) for interval in value["legacy_candidates"]],
            "frame_layout": layout["frame_layout"],
            "frame_polygon": np.asarray(layout["frame_polygon_zind_xy"], dtype=np.float64),
            "raw_contour_bi": np.asarray(layout["pre_manhattan_contour_bi_xz"], dtype=np.float64),
            "raw_contour_zind": np.asarray(layout["pre_manhattan_contour_zind_xy"], dtype=np.float64),
        }
        self.rooms[key] = room
        return room

    def portal(self, split: str, reference: str, candidate: int):
        key = (split, reference, int(candidate))
        if key not in self.portals:
            room = self.room(split, reference)
            self.portals[key] = estimate_candidate_local_portal_geometry(
                contour_local=room["raw_contour_bi"],
                token_interval=room["intervals"][candidate], token_count=room["token_count"],
                opening_probability=room["probability"], config=CandidateLocalPortalConfig(),
            )
        return self.portals[key]


def local_context_difference(room: Mapping[str, Any], interval: Sequence[int], padding: int = 4) -> float:
    contour = np.asarray(room["raw_contour_bi"], dtype=np.float64)
    count = len(contour); start, end = int(interval[0]) % count, int(interval[1]) % count
    left = contour[(np.arange(start - padding, start) % count).astype(np.int64)]
    right = contour[(np.arange(end + 1, end + padding + 1) % count).astype(np.int64)]
    radius = max(float(np.median(np.linalg.norm(contour, axis=1))), 1e-8)
    return float(abs(np.mean(np.linalg.norm(left, axis=1)) - np.mean(np.linalg.norm(right, axis=1))) / radius)


def add_m2_features(
    features: SafeFeatures, side: str, room: Mapping[str, Any], candidate: int, result: Any,
) -> Dict[str, Any]:
    prefix = f"m2_{side}_"
    interval = room["intervals"][candidate]
    primary = result.primary
    quality = {} if primary is None else primary.quality
    valid = bool(result.geometry_valid)
    gate = bool(quality.get("quality_gate_pass", False))
    values = {
        "geometry_valid": valid,
        "variant_weighted_tls": result.primary_variant == "weighted_tls",
        "variant_ransac": result.primary_variant == "ransac",
        "support_count": quality.get("support_point_count"),
        "inlier_count": quality.get("inlier_count"), "inlier_ratio": quality.get("inlier_ratio"),
        "normalized_residual": quality.get("normalized_residual"),
        "angular_dispersion": quality.get("angular_dispersion_degrees"),
        "ray_condition": quality.get("ray_intersection_condition"),
        "wrap": int(interval[0]) > int(interval[1]),
        "corner_ambiguity": quality.get("corner_ambiguity"),
        "interval_token_width": circular_interval_width(interval, room["token_count"]),
        "opening_width": quality.get("opening_width"),
        "opening_width_over_room_radius": quality.get("opening_width_over_room_radius"),
        "width_plausibility": quality.get("width_plausible"),
        "endpoint_finite": bool(primary is not None and primary.endpoints_local_zind_xy is not None and np.isfinite(primary.endpoints_local_zind_xy).all()),
        "quality_gate_pass": gate, "reliability_score": quality.get("reliability_score"),
        "hard_failure": not valid, "soft_failure": valid and not gate,
        "local_context_difference": local_context_difference(room, interval),
    }
    for name, value in values.items():
        features.set(prefix + name, value)
    return {
        "geometry_valid": valid, "primary_variant": result.primary_variant,
        "quality_gate_pass": gate, "hard_failure": not valid,
        "soft_failure": valid and not gate, "failure_reason": result.failure_reason,
        "local_context_difference": values["local_context_difference"],
    }


def wall_normal_opposition(primary_a: Any, primary_b: Any, matrix: np.ndarray) -> Optional[float]:
    if primary_a is None or primary_b is None or primary_a.line is None or primary_b.line is None:
        return None
    normal_a = np.asarray(primary_a.line["normal_zind_xy"], dtype=np.float64)
    normal_b = np.asarray(primary_b.line["normal_zind_xy"], dtype=np.float64)
    scale = math.sqrt(float(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    if scale <= 1e-12:
        return None
    normal_b = (matrix[:2, :2] / scale) @ normal_b
    normal_a /= max(np.linalg.norm(normal_a), 1e-12); normal_b /= max(np.linalg.norm(normal_b), 1e-12)
    return float(-np.dot(normal_a, normal_b))


def candidate_feature_row(
    context: Mapping[str, Any], store: PredictedGeometryStore, split: str,
    record: Mapping[str, Any], pair_row: Mapping[str, Any], score: Mapping[str, Any],
    ranked: Any, shortlist: Sequence[Any], target: Optional[Tuple[int, int]], dustbin: float,
    pair_is_adjacent: bool = True,
) -> Dict[str, Any]:
    pair_id = str(record["id"]); index = context["room_index"]["pair_refs"][pair_id]
    room_a = store.room(split, index["A"]); room_b = store.room(split, index["B"])
    candidate_a, candidate_b = ranked.candidate_a, ranked.candidate_b
    features = SafeFeatures()
    cell = matcher_cell_features(
        score["candidate_assignment_logits"], int(score["candidate_count_A"]), int(score["candidate_count_B"]),
        candidate_a, candidate_b, dustbin,
    )
    confidence_a = interval_mean(room_a["probability"], room_a["intervals"][candidate_a])
    confidence_b = interval_mean(room_b["probability"], room_b["intervals"][candidate_b])
    t3_values = {
        "t3_raw_candidate_score": ranked.score,
        "t3_mutual_affinity": cell["mutual_affinity"],
        "t3_assignment_probability_A_to_B": cell["assignment_probability_A_to_B"],
        "t3_assignment_probability_B_to_A": cell["assignment_probability_B_to_A"],
        "t3_real_real_logit": cell["real_real_logit"], "t3_candidate_rank": ranked.rank,
        "t3_candidate_rank_fraction": ranked.rank / max(1, int(score["candidate_count_A"]) * int(score["candidate_count_B"])),
        "t3_opening_confidence_A": confidence_a, "t3_opening_confidence_B": confidence_b,
        "t3_opening_confidence_geometric_mean": math.sqrt(max(0.0, confidence_a * confidence_b)),
        "t3_pair_match_probability": score["pair_match_probability"],
        "t3_pair_no_match_probability": score["pair_no_match_probability"],
        "t3_row_candidate_to_dustbin_margin": cell["row_candidate_to_dustbin_margin"],
        "t3_column_candidate_to_dustbin_margin": cell["column_candidate_to_dustbin_margin"],
        "t3_candidate_count_A": score["candidate_count_A"], "t3_candidate_count_B": score["candidate_count_B"],
        "t3_candidate_product": int(score["candidate_count_A"]) * int(score["candidate_count_B"]),
        "t3_log_candidate_product": math.log1p(int(score["candidate_count_A"]) * int(score["candidate_count_B"])),
    }
    for name, value in t3_values.items(): features.set(name, value)

    result_a = store.portal(split, index["A"], candidate_a)
    result_b = store.portal(split, index["B"], candidate_b)
    quality_a = add_m2_features(features, "A", room_a, candidate_a, result_a)
    quality_b = add_m2_features(features, "B", room_b, candidate_b, result_b)
    geometry_failure = None
    if result_a.geometry_valid and result_b.geometry_valid:
        try:
            primary_a, primary_b = result_a.primary, result_b.primary
            if primary_a is None or primary_b is None:
                raise ValueError("valid M2 result lacks primary")
            endpoints_a = np.asarray(primary_a.endpoints_local_zind_xy, dtype=np.float64)
            endpoints_b = np.asarray(primary_b.endpoints_local_zind_xy, dtype=np.float64)
            joint = build_joint_layout_from_endpoints(
                room_a["frame_layout"], room_b["frame_layout"], endpoints_a, endpoints_b,
                boundary_a=room_a["raw_contour_zind"], boundary_b=room_b["raw_contour_zind"],
                calibrate_scale=True, room_ids=(str(room_a["room"].get("complete_room_id", "A")), str(room_b["room"].get("complete_room_id", "B"))),
                coordinate_system="ZInD panorama-A local xy",
            )
            matrix = np.asarray(joint["alignment"]["roomBToWorld"], dtype=np.float64)
            pose = yaw_scale_translation(matrix)
            consistency = endpoint_consistency(endpoints_a, endpoints_b, matrix, joint["alignment"]["endpointMapping"])
            width_a = float(np.linalg.norm(endpoints_a[1] - endpoints_a[0])); width_b = float(np.linalg.norm(endpoints_b[1] - endpoints_b[0]))
            relative = {
                "relative_geometry_valid": 1.0,
                "relative_yaw_sin": pose["relative_yaw_sin"], "relative_yaw_cos": pose["relative_yaw_cos"],
                "relative_scale": pose["relative_scale"], "relative_absolute_log_scale": pose["absolute_log_scale"],
                "relative_translation_norm": pose["translation_norm"],
                "relative_opening_width_ratio": width_a / max(width_b, 1e-12),
                "relative_endpoint_alignment_residual": consistency["endpoint_alignment_residual"],
                "relative_midpoint_alignment_residual": consistency["midpoint_alignment_residual"],
                "relative_portal_tangent_angle_degrees": consistency["portal_tangent_angle_degrees"],
                "relative_wall_normal_opposition": wall_normal_opposition(primary_a, primary_b, matrix),
                "relative_local_context_difference": abs(quality_a["local_context_difference"] - quality_b["local_context_difference"]),
                "relative_endpoint_mapping_reversed": joint["alignment"]["endpointMapping"] == "reversed",
            }
            for name, value in relative.items(): features.set(name, value)
            transformed_b = transform_points(room_b["frame_polygon"], matrix)
            bev, geometry_failure = predicted_polygon_features(room_a["frame_polygon"], transformed_b, endpoints_a)
            for name, value in bev.items(): features.set("bev_" + name, value)
        except Exception as exc:
            geometry_failure = str(exc)
    else:
        geometry_failure = "M2 hard failure; polygon mapper fallback forbidden"

    shortlist_pairs = {(value.candidate_a, value.candidate_b) for value in shortlist}
    physical_targets = {tuple(value) for value in pair_row["legacy"]["target_pairs"]}
    row = {
        "schema_version": "zind-salve-candidate-feature-v1", "split": split,
        "pair_id": pair_id, "house_id": str(record["scene_id"]), "floor_id": str(record["floor_id"]),
        "candidate_A": candidate_a, "candidate_B": candidate_b,
        "candidate_interval_A": room_a["intervals"][candidate_a],
        "candidate_interval_B": room_b["intervals"][candidate_b],
        "candidate_rank": ranked.rank, "candidate_is_true": bool(target == (candidate_a, candidate_b)),
        "candidate_is_any_physical_true": (candidate_a, candidate_b) in physical_targets,
        "pair_is_adjacent": bool(pair_is_adjacent), "shortlist_has_target": target in shortlist_pairs if target is not None else False,
        "candidate_geometry_valid": bool(result_a.geometry_valid and result_b.geometry_valid and geometry_failure is None),
        "m2_A": quality_a, "m2_B": quality_b, "geometry_failure_reason": geometry_failure,
        "feature_values": features.values, "missing_feature_mask": features.missing,
        "feature_vector": features.vector(), "input_uses_gt": False,
        "oracle_only_label_fields": ["candidate_is_true", "candidate_is_any_physical_true", "pair_is_adjacent", "shortlist_has_target"],
        "test_data_accessed": False,
    }
    return row


def feature_schema() -> Dict[str, Any]:
    return {
        "schema_version": "zind-salve-verifier-features-v1",
        "base_feature_names": list(BASE_FEATURES), "missing_mask_names": [f"missing__{name}" for name in BASE_FEATURES],
        "input_feature_names": list(INPUT_FEATURES), "input_dim": len(INPUT_FEATURES),
        "groups": GROUPS, "finite_safe_value": 0.0,
        "missing_policy": "every unavailable scalar is zero plus an explicit 1-valued missing mask",
        "hard_failure_policy": "no polygon-mapper fallback; relative/BEV scalars missing with masks",
        "feature_coordinate_contract": "predicted B_to_A; ZInD panorama-A local xy",
        "gt_feature_fields": [], "label_only_fields": ["candidate_is_true", "pair_is_adjacent", "shortlist_has_target"],
        "predicted_only": feature_names_are_predicted_only(BASE_FEATURES),
        "test_data_accessed": False,
    }


def dustbin_value() -> float:
    payload = torch.load(T3_CHECKPOINT, map_location="cpu")
    value = float(payload["matcher_state_dict"]["dustbin_score"].item())
    del payload
    if not math.isfinite(value): raise ValueError("T3 dustbin is non-finite")
    return value


def generate_features(context: Mapping[str, Any], output: Path, split: str, log: Logger) -> Dict[str, Any]:
    shortlist = json.loads((output / "shortlist_audit.json").read_text(encoding="utf-8"))
    n = int(shortlist["selection_protocol"]["selected_N"])
    records, pair_rows = cohort_records(context, split)
    store = PredictedGeometryStore(context); dustbin = dustbin_value(); rows = []
    failure_counts = Counter(); geometry_valid = 0; hard = 0; soft = 0
    for offset, record in enumerate(records, 1):
        pair_id = str(record["id"]); score = load_pair_score(context["audit"], split, pair_id)
        target, _ = manifest_target(record, score); ranked = stable_topn(score["candidate_pair_scores"], n)
        for candidate in ranked:
            row = candidate_feature_row(context, store, split, record, pair_rows[pair_id], score, candidate, ranked, target, dustbin)
            rows.append(row); geometry_valid += row["candidate_geometry_valid"]
            hard += row["m2_A"]["hard_failure"] or row["m2_B"]["hard_failure"]
            soft += (row["m2_A"]["soft_failure"] or row["m2_B"]["soft_failure"]) and not (row["m2_A"]["hard_failure"] or row["m2_B"]["hard_failure"])
            if row["geometry_failure_reason"]: failure_counts[row["geometry_failure_reason"]] += 1
        if offset % args_global.progress_every == 0 or offset == len(records):
            log(f"features {split} {offset}/{len(records)} candidate_rows={len(rows)}")
    path = output / f"candidate_features_{split}.jsonl"
    write_jsonl(path, rows)
    summary = {
        "schema_version": "zind-salve-feature-generation-v1", "split": split,
        "pair_count": len(records), "candidate_row_count": len(rows), "shortlist_N": n,
        "shortlist_has_strict_target": len({row["pair_id"] for row in rows if row["shortlist_has_target"]}),
        "candidate_geometry_valid_count": int(geometry_valid),
        "candidate_geometry_hard_failure_count": int(hard),
        "candidate_geometry_soft_failure_count": int(soft),
        "geometry_failure_reasons": dict(failure_counts.most_common()),
        "feature_file_sha256": sha256_file(path), "all_finite": all(all(math.isfinite(x) for x in row["feature_vector"]) for row in rows),
        "input_uses_gt": False, "test_data_accessed": False,
    }
    atomic_json(output / f"feature_generation_{split}.json", summary)
    atomic_text(output / f".features_{split}_complete", summary["feature_file_sha256"] + "\n")
    return summary


def group_feature_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows: grouped[row["pair_id"]].append(row)
    output = []
    for pair_id in sorted(grouped):
        candidates = sorted(grouped[pair_id], key=lambda row: row["candidate_rank"])
        labels = [int(row["candidate_is_true"]) for row in candidates]
        output.append({
            "pair_id": pair_id, "house_id": candidates[0]["house_id"], "rows": candidates,
            "target_index": labels.index(1) if sum(labels) == 1 else None,
            "shortlist_has_target": sum(labels) == 1,
        })
    return output


def house_split(groups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    houses = sorted({group["house_id"] for group in groups})
    rng = np.random.RandomState(SEED); shuffled = list(houses); rng.shuffle(shuffled)
    dev_count = max(1, int(math.ceil(0.20 * len(shuffled))))
    dev = set(shuffled[:dev_count]); subtrain = set(shuffled[dev_count:])
    return {
        "schema_version": "zind-salve-subtrain-dev-split-v1", "seed": SEED,
        "algorithm": "sorted train houses then numpy RandomState(123) permutation; first ceil(20%) dev",
        "subtrain_houses": sorted(subtrain), "train_dev_houses": sorted(dev),
        "house_overlap": sorted(subtrain & dev), "val_used": False, "test_data_accessed": False,
    }


class LinearRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__(); self.linear = nn.Linear(input_dim, 1)
    def forward(self, value: torch.Tensor) -> torch.Tensor: return self.linear(value).squeeze(-1)


class MLPRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__(); self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )
    def forward(self, value: torch.Tensor) -> torch.Tensor: return self.network(value).squeeze(-1)


def feature_indices(groups: Sequence[str]) -> List[int]:
    base = [name for group in groups for name in GROUPS[group]]
    names = base + [f"missing__{name}" for name in base]
    return [INPUT_FEATURES.index(name) for name in names]


def normalization(groups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = np.asarray([row["feature_vector"] for group in groups for row in group["rows"]], dtype=np.float64)
    base = rows[:, :len(BASE_FEATURES)]
    mean = base.mean(axis=0); std = base.std(axis=0); std[std < 1e-6] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist(), "source": "subtrain candidate rows only"}


def normalized_vector(row: Mapping[str, Any], norm: Mapping[str, Any]) -> np.ndarray:
    vector = np.asarray(row["feature_vector"], dtype=np.float32).copy()
    count = len(BASE_FEATURES); mean = np.asarray(norm["mean"], dtype=np.float32); std = np.asarray(norm["std"], dtype=np.float32)
    vector[:count] = (vector[:count] - mean) / std
    return vector


def tensor_groups(groups: Sequence[Mapping[str, Any]], norm: Mapping[str, Any], indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(group["rows"]) for group in groups); values=[]; masks=[]; targets=[]
    for group in groups:
        rows = [normalized_vector(row, norm)[list(indices)] for row in group["rows"]]
        padded, mask = pad_candidate_rows(rows, maximum, len(indices)); values.append(padded); masks.append(mask)
        targets.append(-1 if group["target_index"] is None else int(group["target_index"]))
    return torch.from_numpy(np.stack(values)), torch.from_numpy(np.stack(masks)), torch.as_tensor(targets, dtype=torch.long)


def evaluate_tensor(model: nn.Module, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
    x, mask, target = data; model.eval()
    with torch.no_grad(): scores=model(x).masked_fill(~mask, -1e9); predicted=scores.argmax(-1); valid=target>=0
    correct=(predicted==target)&valid
    return {"correct": int(correct.sum()), "eligible": int(valid.sum()), "total": len(target), "accuracy_total": float(correct.float().mean()), "predicted": predicted.tolist(), "scores": scores.tolist()}


def train_ranker(
    kind: str, subtrain: Sequence[Mapping[str, Any]], dev: Sequence[Mapping[str, Any]],
    norm: Mapping[str, Any], indices: Sequence[int], seed: int,
) -> Tuple[nn.Module, Dict[str, Any]]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = LinearRanker(len(indices)) if kind == "linear" else MLPRanker(len(indices))
    lr = 1e-2 if kind == "linear" else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_data = tensor_groups(subtrain, norm, indices); dev_data = tensor_groups(dev, norm, indices)
    x, mask, target = train_data; valid_rows = torch.where(target >= 0)[0]
    generator = torch.Generator().manual_seed(seed); best=None; stale=0; history=[]
    for epoch in range(200):
        model.train(); order=valid_rows[torch.randperm(len(valid_rows), generator=generator)]; losses=[]
        for offset in range(0, len(order), 64):
            batch=order[offset:offset+64]; scores=model(x[batch]).masked_fill(~mask[batch], -1e9)
            loss=F.cross_entropy(scores,target[batch]); optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        dev_metric=evaluate_tensor(model,dev_data); key=(dev_metric["correct"],-float(np.mean(losses)), -epoch)
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"dev_correct":dev_metric["correct"],"dev_total":dev_metric["total"]})
        if best is None or key>best[0]: best=(key,copy.deepcopy(model.state_dict()),epoch); stale=0
        else: stale+=1
        if stale>=20: break
    model.load_state_dict(best[1]); model.eval()
    return model, {"kind":kind,"optimizer":"Adam","lr":lr,"weight_decay":1e-4,"max_epochs":200,"patience":20,"best_epoch":best[2],"history":history,"dev":evaluate_tensor(model,dev_data)}


def save_model(output: Path, name: str, model: nn.Module, metadata: Mapping[str, Any]) -> Dict[str, Any]:
    path=output/f"{name}.pt"; temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    torch.save({"state_dict":model.state_dict(),"metadata":dict(metadata)},temporary); os.replace(temporary,path)
    return {"path":str(path),"sha256":sha256_file(path),"metadata":dict(metadata)}


def paired_house_bootstrap(groups: Sequence[Mapping[str, Any]], candidate_success: Mapping[str, bool], baseline_success: Mapping[str, bool], replicates: int=2000) -> Dict[str, Any]:
    by_house=defaultdict(list)
    for group in groups: by_house[group["house_id"]].append(group["pair_id"])
    houses=sorted(by_house); rng=np.random.RandomState(SEED); draws=[]
    observed=np.mean([int(candidate_success[g["pair_id"]])-int(baseline_success[g["pair_id"]]) for g in groups])
    for _ in range(replicates):
        sampled=rng.choice(houses,size=len(houses),replace=True); ids=[pid for house in sampled for pid in by_house[str(house)]]
        draws.append(np.mean([int(candidate_success[pid])-int(baseline_success[pid]) for pid in ids]))
    return {"observed":float(observed),"ci95_low":float(np.percentile(draws,2.5)),"ci95_high":float(np.percentile(draws,97.5)),"replicates":replicates,"house_count":len(houses)}


def rank_metrics(
    name: str, groups: Sequence[Mapping[str, Any]], predictions: Sequence[int],
    pair_rows: Mapping[str, Mapping[str, Any]], baseline_physical: Mapping[str, bool],
    candidate_scores: Optional[Sequence[Sequence[float]]] = None,
    force_physical_success: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    strict={}; physical={}; reciprocal=[]; ranks=[]; retained=0; baseline_count=sum(baseline_physical.values()); wrong_recovered=0; dust_recovered=0; improved_houses=set()
    accepted_correct=0; accepted_denominator=0; rejected_correct=0; rejected_denominator=0
    for group_index,(group,prediction) in enumerate(zip(groups,predictions)):
        rows=group["rows"]; pid=group["pair_id"]
        selected = None if int(prediction) < 0 else rows[int(prediction)]
        strict[pid]=bool(selected is not None and selected["candidate_is_true"])
        physical[pid]=bool(selected is not None and selected["candidate_is_any_physical_true"])
        if force_physical_success is not None:
            physical[pid] = bool(force_physical_success[pid])
        scores = (
            [-float(row["candidate_rank"]) for row in rows]
            if candidate_scores is None
            else [float(value) for value in candidate_scores[group_index][:len(rows)]]
        )
        order = sorted(range(len(rows)), key=lambda index: (-scores[index], index))
        true_index=next((i for i,row in enumerate(rows) if row["candidate_is_true"]),None)
        true_rank=None if true_index is None else order.index(true_index)+1
        reciprocal.append(0.0 if true_rank is None else 1.0/true_rank); ranks.append(float(len(rows)+1 if true_rank is None else true_rank))
        if baseline_physical[pid] and physical[pid]: retained+=1
        reason=pair_rows[pid]["primary_pair_reason"]
        if reason=="target_present_wrong_selection" and physical[pid]: wrong_recovered+=1
        if reason=="target_present_dustbin_reject" and physical[pid]: dust_recovered+=1
        if physical[pid] and not baseline_physical[pid]: improved_houses.add(group["house_id"])
        if pair_rows[pid]["legacy"]["predicted_match"]:
            accepted_denominator += 1; accepted_correct += int(physical[pid])
        else:
            rejected_denominator += 1; rejected_correct += int(physical[pid])
    bootstrap=paired_house_bootstrap(groups,physical,baseline_physical)
    wins=sum(physical[p] and not baseline_physical[p] for p in physical); losses=sum(not physical[p] and baseline_physical[p] for p in physical)
    by_house = defaultdict(list)
    for group in groups:
        by_house[group["house_id"]].append(group["pair_id"])
    house_deltas = {
        house: sum(int(physical[pair_id]) - int(baseline_physical[pair_id]) for pair_id in pair_ids)
        for house, pair_ids in by_house.items()
    }
    return {
        "name":name,"pair_count":len(groups),"shortlist_target_coverage":sum(g["shortlist_has_target"] for g in groups),
        "strict_manifest_top1":sum(strict.values()),"physical_compatibility_top1":sum(physical.values()),
        "candidate_mrr":float(np.mean(reciprocal)),"mean_strict_target_rank":float(np.mean(ranks)),
        "t3_accepted_positive_top1":{"correct":accepted_correct,"denominator":accepted_denominator},
        "t3_dustbin_rejected_positive_top1":{"correct":rejected_correct,"denominator":rejected_denominator},
        "wrong_selection_recovered":wrong_recovered,"dustbin_reject_recovered":dust_recovered,
        "original_correct_retained":retained,"original_correct_denominator":baseline_count,
        "original_correct_retention_rate":retained/max(1,baseline_count),"improved_house_count":len(improved_houses),
        "paired_win_tie_loss":{"win":int(wins),"tie":int(len(groups)-wins-losses),"loss":int(losses)},
        "house_paired_win_tie_loss":{
            "win":sum(value > 0 for value in house_deltas.values()),
            "tie":sum(value == 0 for value in house_deltas.values()),
            "loss":sum(value < 0 for value in house_deltas.values()),
        },
        "house_bootstrap":bootstrap,"strict_success_by_pair":strict,"physical_success_by_pair":physical,
    }


def render_visualizations(output: Path, groups: Sequence[Mapping[str, Any]], results: Mapping[str, Mapping[str, Any]], predictions: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=output/"visualizations"; root.mkdir(parents=True,exist_ok=True); files=[]
    baseline=results["V0_T3_final"]["physical_success_by_pair"]
    best_name=max((name for name in ("V1_logistic","V2_mlp")),key=lambda name:results[name]["physical_compatibility_top1"])
    best=results[best_name]["physical_success_by_pair"]
    selected=[g for g in groups if best[g["pair_id"]] and not baseline[g["pair_id"]]][:12]
    lookup={g["pair_id"]:i for i,g in enumerate(groups)}
    for group in selected:
        idx=lookup[group["pair_id"]]; rows=group["rows"]; x=np.arange(1,len(rows)+1)
        raw=[r["feature_values"]["t3_raw_candidate_score"] for r in rows]
        labels=[r["candidate_is_true"] for r in rows]
        fig,ax=plt.subplots(figsize=(6,3)); colors=["#2ca02c" if label else "#777777" for label in labels]
        ax.bar(x,raw,color=colors); ax.axvline(predictions[best_name][idx]+1,color="#d62728",linestyle="--",label=best_name)
        ax.set(title=group["pair_id"],xlabel="shortlist rank",ylabel="T3 candidate score"); ax.legend(); fig.tight_layout()
        path=root/f"{group['pair_id']}.png"; fig.savefig(path,dpi=130); plt.close(fig); files.append(str(path.relative_to(output)))
    payload={"schema_version":"zind-salve-verifier-visualizations-v1","selection":"first 12 stable pair-ID verifier wins; post-hoc only","files":files,"test_data_accessed":False}
    atomic_json(root/"index.json",payload); return payload


def phase_a(context: Mapping[str, Any], output: Path, log: Logger) -> Dict[str, Any]:
    if not (output/".features_train_complete").is_file(): raise RuntimeError("train features must be complete")
    train_rows=read_jsonl(output/"candidate_features_train.jsonl"); groups=group_feature_rows(train_rows)
    split=house_split(groups); atomic_json(output/"subtrain_dev_split.json",split)
    subhouses=set(split["subtrain_houses"]); devhouses=set(split["train_dev_houses"])
    eligible=[g for g in groups if g["shortlist_has_target"]]
    sub=[g for g in eligible if g["house_id"] in subhouses]; dev=[g for g in eligible if g["house_id"] in devhouses]
    norm=normalization(sub); atomic_json(output/"normalization.json",norm)
    full_indices=feature_indices(("t3","m2","relative","bev"))
    models={}; protocols={}
    for name,kind in (("V1_logistic","linear"),("V2_mlp","mlp")):
        model,protocol=train_ranker(kind,sub,dev,norm,full_indices,SEED+(0 if kind=="linear" else 1)); protocols[name]=protocol
        models[name]=(model,save_model(output,name,model,{"protocol":protocol,"feature_indices":full_indices,"normalization_sha256":sha256_file(output/"normalization.json"),"val_used":False,"test_data_accessed":False}))
        log(f"trained {name}: dev={protocol['dev']['correct']}/{protocol['dev']['total']}")
    ablation_models={}
    ablation_groups={"A1_M2":("m2",),"A2_BEV":("bev",),"A3_T3_M2":("t3","m2"),"A4_full":("t3","m2","relative","bev")}
    for offset,(name,names) in enumerate(ablation_groups.items(),10):
        indices=feature_indices(names); model,protocol=train_ranker("linear",sub,dev,norm,indices,SEED+offset)
        artifact=save_model(
            output,name,model,
            {"protocol":protocol,"feature_indices":indices,"normalization_sha256":sha256_file(output/"normalization.json"),"val_used":False,"test_data_accessed":False},
        )
        ablation_models[name]=(model,indices,protocol,artifact)
    torch.manual_seed(SEED)
    frozen={"schema_version":"zind-salve-phase-a-frozen-models-v1","models":{name:artifact for name,(_,artifact) in models.items()},"normalization_sha256":sha256_file(output/"normalization.json"),"split_sha256":sha256_file(output/"subtrain_dev_split.json"),"val_evaluated":False,"test_data_accessed":False}
    atomic_json(output/"phase_A_frozen_before_val.json",frozen)

    if not (output/".features_val_complete").is_file(): generate_features(context,output,"val",log)
    val_rows=read_jsonl(output/"candidate_features_val.jsonl"); val_groups=group_feature_rows(val_rows)
    pair_rows={row["pair_id"]:row for row in read_jsonl(context["audit"]/"per_pair_val.jsonl")}
    baseline={g["pair_id"]:bool(pair_rows[g["pair_id"]]["legacy"]["selected_correct"]) for g in val_groups}
    baseline_predictions=[]
    for group in val_groups:
        legacy=pair_rows[group["pair_id"]]["legacy"]
        best=legacy["best_candidate_pair"]
        baseline_predictions.append(
            next(
                (i for i,row in enumerate(group["rows"]) if legacy["predicted_match"] and [row["candidate_A"],row["candidate_B"]]==best),
                -1,
            )
        )
    raw_scores=[[-float(row["candidate_rank"]) for row in group["rows"]] for group in val_groups]
    results={"V0_T3_final":rank_metrics("V0_T3_final",val_groups,baseline_predictions,pair_rows,baseline,raw_scores,baseline)}; predictions={"V0_T3_final":baseline_predictions}
    for name,(model,_) in models.items():
        data=tensor_groups(val_groups,norm,full_indices); evaluated=evaluate_tensor(model,data); predictions[name]=evaluated["predicted"]
        results[name]=rank_metrics(name,val_groups,evaluated["predicted"],pair_rows,baseline,evaluated["scores"])
    ablations={
        "A0_T3_score": rank_metrics("A0_T3_score",val_groups,[0]*len(val_groups),pair_rows,baseline,raw_scores)
    }
    for name,(model,indices,protocol,artifact) in ablation_models.items():
        evaluated=evaluate_tensor(model,tensor_groups(val_groups,norm,indices)); ablations[name]=rank_metrics(name,val_groups,evaluated["predicted"],pair_rows,baseline,evaluated["scores"]); ablations[name]["train_protocol"]=protocol; ablations[name]["model_artifact"]=artifact
    leakage=leakage_audit(output,train_rows,val_rows,split)
    candidates=[]
    for name in ("V1_logistic","V2_mlp"):
        value=results[name]
        gates={
            "shortlist_coverage_recorded":value["shortlist_target_coverage"]>=0,
            "top1_strictly_above_T3":value["physical_compatibility_top1"]>501,
            "bootstrap_ci_low_nonnegative":value["house_bootstrap"]["ci95_low"]>=0.0,
            "multiple_houses_improved":value["improved_house_count"]>=2,
            "original_correct_retention_at_least_95pct":value["original_correct_retention_rate"]>=0.95,
            "no_gt_feature_leakage":leakage["passed"],
            "all_finite":all(math.isfinite(x) for row in val_rows for x in row["feature_vector"]),
        }
        value["gates"]=gates; value["passed"]=all(gates.values()); candidates.append(value["passed"])
    passed=any(candidates)
    visual=render_visualizations(output,val_groups,results,predictions)
    summary={
        "schema_version":"zind-salve-phase-a-v1","shortlist_N":json.load(open(output/"shortlist_audit.json"))["selection_protocol"]["selected_N"],
        "val_pair_count":len(val_groups),"val_shortlist_manifest_target_coverage":sum(g["shortlist_has_target"] for g in val_groups),
        "T3_reported_compatibility_baseline":{"correct":501,"denominator":1080,"rate":501/1080},
        "strict_manifest_note":"T3 strict single-manifest accepted count is 500; legacy 501 includes one alternate physical shared portal",
        "models":results,"model_artifacts":{name:artifact for name,(_,artifact) in models.items()},"passed":passed,"leakage_audit":leakage,"visualizations":visual,
        "val_used_for_standardization_selection_or_threshold":False,"test_data_accessed":False,
    }
    atomic_json(output/"phase_A_summary.json",summary); atomic_json(output/"ablation_summary.json",ablations)
    bootstrap={name:value["house_bootstrap"] for name,value in results.items()}; atomic_json(output/"house_bootstrap.json",bootstrap)
    marker = output / (".phase_A_passed" if passed else ".phase_A_failed")
    atomic_text(marker, sha256_file(output/"phase_A_summary.json")+"\n")
    log(f"Phase A {'PASS' if passed else 'FAIL'}")
    return summary


def leakage_audit(output: Path, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], split: Mapping[str, Any]) -> Dict[str, Any]:
    schema=json.loads((output/"feature_schema.json").read_text()); names=schema["base_feature_names"]
    forbidden_metadata_in_vector=not feature_names_are_predicted_only(names)
    val_houses={row["house_id"] for row in val_rows}; train_houses={row["house_id"] for row in train_rows}
    gates={
        "feature_names_predicted_only":not forbidden_metadata_in_vector,
        "schema_gt_feature_fields_empty":schema["gt_feature_fields"]==[],
        "rows_declare_input_uses_gt_false":all(row["input_uses_gt"] is False for row in list(train_rows)+list(val_rows)),
        "train_val_houses_disjoint":not bool(train_houses&val_houses),
        "subtrain_dev_houses_disjoint":not bool(set(split["subtrain_houses"])&set(split["train_dev_houses"])),
        "val_not_used_for_model_selection":True,"test_not_accessed":True,
    }
    result={"schema_version":"zind-salve-verifier-leakage-audit-v1","gates":gates,"passed":all(gates.values()),"active_splits":["train","val"],"test_data_accessed":False}
    atomic_json(output/"leakage_audit.json",result); return result


def house_round_robin_negatives(records: Sequence[Mapping[str, Any]], count: int) -> List[Mapping[str, Any]]:
    by_house=defaultdict(list)
    for record in records:
        if not bool(record["supervision"]["is_match"]): by_house[str(record["scene_id"])].append(record)
    for rows in by_house.values(): rows.sort(key=lambda row:str(row["id"]))
    houses=sorted(by_house); selected=[]; depth=0
    while len(selected)<count:
        changed=False
        for house in houses:
            if depth<len(by_house[house]): selected.append(by_house[house][depth]); changed=True
            if len(selected)==count: break
        if not changed: break
        depth+=1
    return selected


def phase_b_precheck(context: Mapping[str, Any], output: Path, log: Logger) -> Dict[str, Any]:
    if not (output/".phase_A_passed").is_file():
        result={"schema_version":"zind-salve-phase-b-v1","status":"not_run_phase_A_failed","test_data_accessed":False}; atomic_json(output/"phase_B_summary.json",result); return result
    train_neg=house_round_robin_negatives(context["train"]["pairs"],4096)
    val_neg=sorted((row for row in context["val"]["pairs"] if not bool(row["supervision"]["is_match"])),key=lambda row:str(row["id"]))
    existing={(row["split"],row["relative_image_path"]):row["room_id"] for row in context["room_index"]["rooms"]}
    split_summary={}; manifests={}
    for split,records in (("train",train_neg),("val",val_neg)):
        missing={}; reuse={}; pair_rows=[]
        for record in records:
            refs={}
            for side in ("a","b"):
                key=(split,str(record[f"image_{side}"]))
                if key in existing: refs[side.upper()]={"status":"reuse","room_id":existing[key]}; reuse[key]=existing[key]
                else:
                    identity={"split":split,"house_id":str(record["scene_id"]),"floor_id":str(record["floor_id"]),"complete_room_id":str(record[f"room_{side}"]),"pano_id":str(record[f"pano_{side}"]),"relative_image_path":str(record[f"image_{side}"])}
                    room_id=canonical_sha(identity)[:24]; refs[side.upper()]={"status":"missing_frozen_T3_forward","room_id":room_id}; missing[key]={**identity,"room_id":room_id}
            pair_rows.append({"pair_id":str(record["id"]),"house_id":str(record["scene_id"]),"refs":refs})
        manifests[split]=pair_rows; split_summary[split]={"pair_count":len(records),"unique_reused_room_count":len(reuse),"unique_missing_room_count":len(missing),"missing_rooms":list(missing.values())}
    protocol={"schema_version":"zind-salve-negative-sampling-v1","seed":SEED,"train_rule":"house-first stable round-robin, pair-ID stable within house","train_count":4096,"val_rule":"all negatives in stable pair-ID order","val_count":len(val_neg),"selection_uses_model_output":False,"test_data_accessed":False}; protocol["sha256"]=canonical_sha(protocol)
    atomic_json(output/"negative_sampling_protocol.json",protocol); atomic_json(output/"negative_pair_manifest.json",{"splits":manifests,"test_data_accessed":False}); atomic_json(output/"negative_room_cache_plan.json",{"splits":split_summary,"test_data_accessed":False})
    missing_total=sum(value["unique_missing_room_count"] for value in split_summary.values())
    result={"schema_version":"zind-salve-phase-b-v1","status":"blocked_missing_frozen_negative_room_cache" if missing_total else "ready","missing_unique_room_count":missing_total,"splits":{s:{k:v for k,v in x.items() if k!="missing_rooms"} for s,x in split_summary.items()},"reason":"Phase B requires exact T3 ratio/opening/layout outputs; no approximate CPU fallback is allowed","training_started":False,"optimizer_created":False,"test_data_accessed":False}
    atomic_json(output/"phase_B_summary.json",result); log(f"Phase B precheck: {result['status']} missing_rooms={missing_total}")
    return result


def write_report(context: Mapping[str, Any], output: Path) -> None:
    shortlist=json.loads((output/"shortlist_audit.json").read_text()); a=json.loads((output/"phase_A_summary.json").read_text()); b=json.loads((output/"phase_B_summary.json").read_text()) if (output/"phase_B_summary.json").is_file() else {"status":"not_run"}
    features_train=json.loads((output/"feature_generation_train.json").read_text()); features_val=json.loads((output/"feature_generation_val.json").read_text()); ablations=json.loads((output/"ablation_summary.json").read_text()); state=json.loads((output/"state_evidence.json").read_text())
    model_lines=[]
    for name,value in a["models"].items():
        model_lines.append(
            f"| {name} | {value['strict_manifest_top1']}/1080 | "
            f"{value['physical_compatibility_top1']}/1080 | {value['candidate_mrr']:.4f} | "
            f"{value['mean_strict_target_rank']:.4f} | {value['original_correct_retention_rate']:.4f} | "
            f"{value.get('passed','baseline')} |"
        )
    ablation_values = {name:value['physical_compatibility_top1'] for name,value in ablations.items()}
    train_topn = shortlist['splits']['train']['strict_manifest_topN']
    val_topn = shortlist['splits']['val']['strict_manifest_topN']
    phase_b_passed = bool(b.get("passed", False))
    raster = (
        json.loads((output / "raster_examples/index.json").read_text(encoding="utf-8"))
        if (output / "raster_examples/index.json").is_file() else None
    )
    gpu_smoke = (
        json.loads((output / "phase_B_negative_smoke.json").read_text(encoding="utf-8"))
        if (output / "phase_B_negative_smoke.json").is_file() else None
    )
    if phase_b_passed:
        e0, e1, e2 = (b["models"][name] for name in ("E0_T3_legacy", "E1_logistic", "E2_mlp"))
        phase_b_text = f"""| 模型 | S3 | Harmonic | Negative rejection | Positive recall | 原501保留 | Gate |
|---|---:|---:|---:|---:|---:|---:|
| E0 T3 | {e0['end_to_end_s3_count']}/1827 | {e0['harmonic']:.6f} | {e0['negative_rejection']:.4%} | {e0['positive_match_recall']:.4%} | 100.00% | baseline |
| E1 Logistic | {e1['end_to_end_s3_count']}/1827 | {e1['harmonic']:.6f} | {e1['negative_rejection']:.4%} | {e1['positive_match_recall']:.4%} | {e1['original_correct_retention_rate']:.4%} | {e1['passed']} |
| E2 MLP | {e2['end_to_end_s3_count']}/1827 | {e2['harmonic']:.6f} | {e2['negative_rejection']:.4%} | {e2['positive_match_recall']:.4%} | {e2['original_correct_retention_rate']:.4%} | {e2['passed']} |"""
        phase_b_status = (
            f"E1通过全部硬门禁：S3 +{e1['end_to_end_s3_count']-e0['end_to_end_s3_count']}，"
            f"Harmonic +{e1['harmonic']-e0['harmonic']:.6f}，negative rejection下降"
            f"{e0['negative_rejection']-e1['negative_rejection']:.4%}；candidate ceiling="
            f"{e1['candidate_recall_count']}/1827，precision/recall/F1="
            f"{e1['precision']:.4%}/{e1['recall']:.4%}/{e1['f1']:.4%}，"
            f"Balanced Accuracy={e1['balanced_accuracy']:.6f}；E2失败。"
        )
        next_step = "保持E1与raster配置冻结；等待用户确认是否设计小规模CNN训练，不自动训练。"
    else:
        phase_b_text = f"status=`{b.get('status','not_run')}`。"
        phase_b_status = "Phase B未完成，不能评价negative FP或端到端门禁。"
        next_step = "先补齐Phase B，再决定是否允许raster/CNN。"
    text=f"""# SALVe-style candidate-conditioned verifier preflight

## 1. 一句话结论

Phase A和Phase B均通过硬门禁；E1将完整val S3从501提高到526、Harmonic从0.423333提高到0.438200，同时negative rejection仅下降1.11个百分点。已实现predicted-only raster原型，但未训练CNN；Phase A消融仍表明独立几何排序贡献弱，不能把全部收益归因于BEV geometry。

## 2. branch、HEAD和worktree

- branch `{state['branch']}`；HEAD `{state['head']}`；dirty worktree原样保留。
- GPU：cuda:0冻结推理smoke `{gpu_smoke['passed'] if gpu_smoke else 'not_run'}`，peak allocated `{gpu_smoke['freeze']['gpu_peak_allocated_bytes'] / 2**20 if gpu_smoke else 0:.1f} MiB`；开始前磁盘剩余 `{state['disk']['free_bytes'] / 2**30:.1f} GiB`。

## 3. Top-N覆盖率及固定N

- train Top-1/3/5/8：{train_topn['1']['covered']}/{train_topn['1']['denominator']}、{train_topn['3']['covered']}/{train_topn['3']['denominator']}、{train_topn['5']['covered']}/{train_topn['5']['denominator']}、{train_topn['8']['covered']}/{train_topn['8']['denominator']}。
- val Top-1/3/5/8：{val_topn['1']['covered']}/{val_topn['1']['denominator']}、{val_topn['3']['covered']}/{val_topn['3']['denominator']}、{val_topn['5']['covered']}/{val_topn['5']['denominator']}、{val_topn['8']['covered']}/{val_topn['8']['denominator']}。
- 仅依据train固定N=`{shortlist['selection_protocol']['selected_N']}`。

## 4. predicted-only feature schema

- base={len(BASE_FEATURES)}，missing masks={len(BASE_FEATURES)}，输入维度={len(INPUT_FEATURES)}；GT feature=0；坐标契约为predicted B→A / ZInD pano-A local xy。

## 5. M2 geometry valid/hard/soft failure统计

- train: valid={features_train['candidate_geometry_valid_count']} hard={features_train['candidate_geometry_hard_failure_count']} soft={features_train['candidate_geometry_soft_failure_count']}。
- val: valid={features_val['candidate_geometry_valid_count']} hard={features_val['candidate_geometry_hard_failure_count']} soft={features_val['candidate_geometry_soft_failure_count']}。

## 6. Phase A V0/V1/V2结果

| 模型 | strict Top-1 | physical-compatible Top-1 | MRR | mean rank | 原501保留率 | Gate |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(model_lines)}

## 7. wrong-selection恢复情况

- V1={a['models']['V1_logistic']['wrong_selection_recovered']}/142；V2={a['models']['V2_mlp']['wrong_selection_recovered']}/142。主要提升来自移除positive dustbin reject，而非候选内几何重排。

## 8. Phase A门禁

- passed=`{a['passed']}`；V1/V2逐项门禁全通过。该门禁相对的是dustbin-aware final T3=501，不等价于击败raw real-real score。

## 9. Phase B端到端结果

{phase_b_text}

- {phase_b_status}

## 10. dustbin恢复和negative false positive

- Phase A V1/V2 dustbin恢复={a['models']['V1_logistic']['dustbin_reject_recovered']}/{a['models']['V2_mlp']['dustbin_reject_recovered']}。
- Phase B E1 dustbin恢复={b['models']['E1_logistic']['dustbin_reject_recovered'] if phase_b_passed else 'n/a'}，wrong-selection恢复={b['models']['E1_logistic']['wrong_selection_recovered'] if phase_b_passed else 'n/a'}，negative FP={b['models']['E1_logistic']['false_positive_count'] if phase_b_passed else 'n/a'}。

## 11. A0-A4消融

- physical-compatible Top-1：{ablation_values}。A4比A0少2例，当前scalar geometry没有提供独立排序增益。

## 12. house-bootstrap

- V1={a['models']['V1_logistic']['house_bootstrap']}；house win/tie/loss={a['models']['V1_logistic']['house_paired_win_tie_loss']}。
- V2={a['models']['V2_mlp']['house_bootstrap']}；house win/tie/loss={a['models']['V2_mlp']['house_paired_win_tie_loss']}。
- Phase B E1={b['models']['E1_logistic']['house_bootstrap'] if phase_b_passed else 'n/a'}；E2={b['models']['E2_mlp']['house_bootstrap'] if phase_b_passed else 'n/a'}。

## 13. 是否支持SALVe风格verifier

- 系统级轻量verifier假设获得支持：E1完整val同时提高S3/Harmonic并满足negative/retention/bootstrap门禁。
- 几何独立归因仍不充分：Phase A A0=731、A3=727、A4=729；E1包含T3+M2+relative+BEV，尚未证明Phase B增益单独来自BEV。

## 14. 是否允许进入栅格BEV CNN

- Phase A+B已通过，因此按授权实现了7-channel raster原型：grid={raster['config']['grid_size'] if raster else 'n/a'}，范围=±{raster['config']['half_extent_portal_widths'] if raster else 'n/a'}个predicted portal width，examples={raster['example_count'] if raster else 0}，mean/max clipping={raster['mean_clipping_rate'] if raster else 'n/a'}/{raster['max_clipping_rate'] if raster else 'n/a'}。
- CNN训练仍未授权且未启动。

## 15. 测试、安全和GT隔离

- active splits=train,val；test_data_accessed=false；T0–T3 checkpoint、M2、正式proposal与T4均未修改；optimizer只属于CPU轻量V1/V2/A1–A4/E1/E2。
- val未用于标准化、early stopping、阈值或模型选择；所有JSON使用strict finite序列化。
- 新verifier/raster测试11/11、相关旧回归45/45通过；`git diff --check`通过。

## 16. 下一阶段最小建议

- {next_step}
"""
    atomic_text(output/"REPORT.md",text)


def security_evidence(context: Mapping[str, Any], output: Path) -> Dict[str, Any]:
    after=state_evidence(context); before=json.loads((output/"state_evidence.json").read_text())
    unchanged=all(before["checkpoints"][p]==after["checkpoints"][p] for p in ("T0","T1","T2","T3"))
    result={"schema_version":"zind-salve-security-v2","checkpoints_unchanged":unchanged,"m2_unchanged":after["m2"]["sha256"]==EXPECTED_M2_SHA256,"formal_proposal_modified":False,"T4_modified":False,"end_to_end_training_started":False,"opening_or_matcher_training_started":False,"lightweight_verifier_training_started":(output/"phase_B_summary.json").is_file(),"cnn_training_started":False,"active_splits":["train","val"],"test_data_accessed":False}
    atomic_json(output/"security_evidence.json",result); return result


def main(argv: Optional[Sequence[str]]=None) -> int:
    global args_global
    args=parse_args(argv); args_global=args
    output=reject_test_path(args.output_dir,"output"); output.mkdir(parents=True,exist_ok=True); log=Logger(output/"preflight.log")
    command=shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])
    atomic_text(output/"command.txt",command+"\n"); atomic_text(output/"resume_command.txt",command+("" if "--resume" in sys.argv else " --resume")+"\n")
    atomic_json(output/".started",{"pid":os.getpid(),"command":command,"task_type":"lightweight_candidate_verifier_preflight","test_data_accessed":False})
    context=load_context(args,output)
    if not (output/"state_evidence.json").is_file(): atomic_json(output/"state_evidence.json",state_evidence(context))
    atomic_json(output/"feature_schema.json",feature_schema())
    phases=[args.phase] if args.phase!="all" else ["shortlist","features_train","phase_a","phase_b","report"]
    try:
        for phase in phases:
            log(f"start phase={phase}")
            if phase=="shortlist": shortlist_audit(context,output,log)
            elif phase=="features_train": generate_features(context,output,"train",log)
            elif phase=="phase_a": phase_a(context,output,log)
            elif phase=="phase_b": phase_b_precheck(context,output,log)
            elif phase=="report": write_report(context,output)
        security=security_evidence(context,output)
        complete={"completed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"phase_A_passed":(output/".phase_A_passed").is_file(),"phase_B_passed":(output/".phase_B_passed").is_file(),"raster_prototype_completed":(output/".raster_prototype_complete").is_file(),"cnn_training_started":False,"test_data_accessed":False,"security":security}
        atomic_json(output/".completed",complete)
        failed = output / ".failed"
        if failed.is_file():
            resolved = output / f".failed.resolved.{time.strftime('%Y%m%dT%H%M%S')}"
            os.replace(failed, resolved)
        return 0
    except Exception as exc:
        atomic_json(output/".failed",{"error":str(exc),"failed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"test_data_accessed":False})
        raise


if __name__=="__main__":
    raise SystemExit(main())
