#!/usr/bin/env python3
"""Run a fixed train-only ZInD room group through conditional pair fusion.

This is an isolated debugging/evaluation entry point.  It never changes the
pairwise datasets, feature caches, trainers, or checkpoints.  Pair decisions
are made from predicted openings and the matcher's learned dustbin only; ZInD
labels are attached afterward for diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import sys
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DISCLAIMER = "debug/qualitative only, not a scientific evaluation"
DEFAULT_GROUP = (
    REPO_ROOT
    / "data/ZInD_matching_groups/debug_train_0665_floor_03.json"
)
DEFAULT_MANIFEST = REPO_ROOT / "data/ZInD_matching/train.json"
DEFAULT_CACHE = (
    REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_feature_cache/train"
)
DEFAULT_CONFIG = REPO_ROOT / "configs/layout/zind_all.yaml"
DEFAULT_BI_LAYOUT = (
    REPO_ROOT / "checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl"
)
DEFAULT_OPENING = REPO_ROOT / "checkpoints/Opening_Head/zind_matching_v1/best.pt"
DEFAULT_MATCHER = (
    REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_v1/best.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/zind_room_group_debug"
_PASSABLE_TYPES = frozenset(("door", "opening"))
_BI_TO_ZIND = np.diag([-1.0, 1.0, 1.0])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", "--group_manifest", dest="group", default=str(DEFAULT_GROUP))
    parser.add_argument("--pair_manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--data_root")
    parser.add_argument("--feature_cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--config", "--bi_layout_config", dest='config', default=str(DEFAULT_CONFIG))
    parser.add_argument("--bi_layout_checkpoint", default=str(DEFAULT_BI_LAYOUT))
    parser.add_argument("--opening_checkpoint", default=str(DEFAULT_OPENING))
    parser.add_argument("--matcher_checkpoint", default=str(DEFAULT_MATCHER))
    parser.add_argument(
        "--end_to_end_checkpoint",
        help="optional T0-T3-compatible full checkpoint loaded last",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metadata_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--endpoint_tolerance", type=float, default=0.06)
    parser.add_argument("--min_opening_width_tokens", type=int, default=2)
    parser.add_argument("--max_openings_per_view", type=int, default=12)
    parser.add_argument("--opening_probability_threshold", type=float)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument(
        "--layout_frame_mode",
        choices=("legacy_fixed_axis", "panorama_preserving"),
        default="legacy_fixed_axis",
        help=(
            "single-room Manhattan output frame; the default reproduces the "
            "historical fixed-axis geometry"
        ),
    )
    parser.add_argument(
        "--portal_geometry_mode",
        choices=("polygon_mapper", "candidate_local_m2"),
        default="polygon_mapper",
        help=(
            "geometry used only after the matcher freezes a positive candidate "
            "pair; the default preserves the historical polygon mapper"
        ),
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolved_file(value: str, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    _reject_test_reference(path, name)
    return path


def _reject_test_reference(path: Path, name: str) -> None:
    if any(part.lower() == "test" for part in path.parts) or path.stem.lower() == "test":
        raise ValueError(f"{name} points to forbidden test data: {path}")


def _pair_key(room_a: str, room_b: str) -> str:
    if room_a == room_b:
        raise ValueError("a room pair must contain two distinct rooms")
    return "|".join(sorted((str(room_a), str(room_b))))


def _edge_set(values: Any, name: str, rooms: Sequence[str]) -> set:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    output = set()
    for value in values:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"each {name} item must be a two-room list")
        if value[0] not in rooms or value[1] not in rooms:
            raise ValueError(f"{name} contains a room outside room_ids")
        key = _pair_key(value[0], value[1])
        if key in output:
            raise ValueError(f"{name} contains duplicate edge {key}")
        output.add(key)
    return output


def validate_group_schema(group: Mapping[str, Any]) -> Dict[str, Any]:
    required = (
        "schema_version",
        "split",
        "group_id",
        "house_id",
        "floor_id",
        "room_ids",
        "room_to_pano",
        "pair_ids",
        "positive_edges",
        "negative_edges",
        "adjacency_matrix",
        "instance_layout_refs",
        "scale_meters_per_coordinate",
        "scale",
    )
    missing = [key for key in required if key not in group]
    if missing:
        raise ValueError("group is missing fields: " + ", ".join(missing))
    if group.get("schema_version") != "zind-room-group-v1":
        raise ValueError("unsupported group schema_version")
    if group.get("split") != "train":
        raise ValueError("room-group evaluation is train-only")
    if group.get("purpose") != DISCLAIMER:
        raise ValueError(f"group purpose must be exactly: {DISCLAIMER}")

    rooms = group.get("room_ids")
    if not isinstance(rooms, list) or len(rooms) != 3:
        raise ValueError("the minimal room group must contain exactly three rooms")
    if len(set(rooms)) != 3 or not all(isinstance(room, str) and room for room in rooms):
        raise ValueError("room_ids must contain three distinct non-empty strings")
    all_pairs = {_pair_key(a, b) for a, b in combinations(rooms, 2)}

    pair_ids = group.get("pair_ids")
    if not isinstance(pair_ids, Mapping) or set(pair_ids) != all_pairs:
        raise ValueError("pair_ids must cover the three unordered room pairs exactly")
    if len(set(pair_ids.values())) != 3:
        raise ValueError("pair_ids must reference three distinct pair records")

    pair_order = group.get("pair_order")
    if not isinstance(pair_order, list) or len(pair_order) != 3:
        raise ValueError("pair_order must enumerate exactly three room pairs")
    ordered_keys = [_pair_key(str(edge[0]), str(edge[1])) for edge in pair_order]
    if set(ordered_keys) != all_pairs or len(set(ordered_keys)) != 3:
        raise ValueError("pair_order must enumerate every unordered pair once")

    positives = _edge_set(group["positive_edges"], "positive_edges", rooms)
    negatives = _edge_set(group["negative_edges"], "negative_edges", rooms)
    if positives & negatives or positives | negatives != all_pairs:
        raise ValueError("positive_edges and negative_edges must partition all pairs")
    if len(positives) < 2:
        raise ValueError("the debug group must contain at least two positive edges")

    adjacency = group.get("adjacency_matrix")
    if not isinstance(adjacency, list) or len(adjacency) != 3:
        raise ValueError("adjacency_matrix must be 3x3")
    for row in adjacency:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("adjacency_matrix must be 3x3")
    for i, room_a in enumerate(rooms):
        for j, room_b in enumerate(rooms):
            expected = 0 if i == j else int(_pair_key(room_a, room_b) in positives)
            if adjacency[i][j] != expected or adjacency[i][j] != adjacency[j][i]:
                raise ValueError("adjacency_matrix disagrees with positive_edges")

    visited = {0}
    frontier = [0]
    while frontier:
        index = frontier.pop()
        for other, connected in enumerate(adjacency[index]):
            if connected and other not in visited:
                visited.add(other)
                frontier.append(other)
    if visited != {0, 1, 2} or group.get("connected") is not True:
        raise ValueError("the three-room positive graph must be connected")

    room_to_pano = group.get("room_to_pano")
    refs = group.get("instance_layout_refs")
    if not isinstance(room_to_pano, Mapping) or set(room_to_pano) != set(rooms):
        raise ValueError("room_to_pano must cover room_ids exactly")
    if not isinstance(refs, Mapping) or set(refs) != set(rooms):
        raise ValueError("instance_layout_refs must cover room_ids exactly")
    panos = []
    for room in rooms:
        selection = room_to_pano[room]
        if not isinstance(selection, Mapping):
            raise ValueError(f"room_to_pano[{room}] must be an object")
        if not selection.get("partial_room_id") or not selection.get("pano_id"):
            raise ValueError(f"room_to_pano[{room}] is incomplete")
        panos.append(str(selection["pano_id"]))
    if len(set(panos)) != 3:
        raise ValueError("each room must use one distinct panorama")

    scale = group.get("scale")
    if not isinstance(scale, Mapping):
        raise ValueError("scale must be an object")
    floor_scale = float(scale.get("scale_meters_per_coordinate", float("nan")))
    if not math.isfinite(floor_scale) or floor_scale <= 0:
        raise ValueError("scale_meters_per_coordinate must be positive and finite")
    if not math.isclose(
        float(group["scale_meters_per_coordinate"]),
        floor_scale,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("top-level and referenced scale_meters_per_coordinate differ")
    return {
        "rooms": list(rooms),
        "all_pair_keys": all_pairs,
        "positive_keys": positives,
        "negative_keys": negatives,
        "pair_order": [list(edge) for edge in pair_order],
        "floor_scale": floor_scale,
    }


def _json_pointer(root: Mapping[str, Any], pointer: str) -> Any:
    value: Any = root
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("json_pointer must start with '/'")
    for raw in pointer[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"JSON pointer does not exist: {pointer}")
        value = value[key]
    return value


def _physical_candidate_pairs(
    record: Mapping[str, Any], tolerance: float
) -> List[Tuple[int, int, float]]:
    matches = []
    candidates_a = record.get("candidates_a", [])
    candidates_b = record.get("candidates_b", [])
    for index_a, candidate_a in enumerate(candidates_a):
        if candidate_a.get("type") not in _PASSABLE_TYPES:
            continue
        first = np.asarray(candidate_a.get("globalEndpoints"), dtype=np.float64)
        for index_b, candidate_b in enumerate(candidates_b):
            if candidate_b.get("type") not in _PASSABLE_TYPES:
                continue
            second = np.asarray(candidate_b.get("globalEndpoints"), dtype=np.float64)
            if first.shape != (2, 2) or second.shape != (2, 2):
                raise ValueError(f"pair {record.get('id')} has malformed endpoints")
            direct = np.linalg.norm(first[0] - second[0]) + np.linalg.norm(
                first[1] - second[1]
            )
            reversed_order = np.linalg.norm(first[0] - second[1]) + np.linalg.norm(
                first[1] - second[0]
            )
            # Equivalent to dataset.zind_pair_mining.segment_endpoint_error,
            # kept dependency-free so --metadata_only does not import torch.
            error = float(min(direct, reversed_order) / 2.0)
            if error <= tolerance:
                matches.append((index_a, index_b, float(error)))
    return matches


def _compact_layout(layout: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(layout.get(key, []))
        for key in ("vertices", "doors", "openings", "windows")
    }


def _validate_polygon_vertices(value: Any, name: str) -> None:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"{name} must contain an [N, 2] polygon")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    x, y = points[:, 0], points[:, 1]
    area = abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))) / 2.0
    if area <= 1e-8:
        raise ValueError(f"{name} polygon has zero area")


def _normalize_image_path(data_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = data_root / path
    path = path.resolve()
    _reject_test_reference(path, "panorama image")
    return path


def load_and_validate_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    if args.endpoint_tolerance <= 0 or not math.isfinite(args.endpoint_tolerance):
        raise ValueError("endpoint_tolerance must be positive and finite")
    if min(args.torch_threads, args.min_opening_width_tokens, args.max_openings_per_view, args.top_k) <= 0:
        raise ValueError("thread, opening, and top-k settings must be positive")
    if args.opening_probability_threshold is not None and not (
        0.0 <= args.opening_probability_threshold <= 1.0
    ):
        raise ValueError("opening_probability_threshold must be in [0, 1]")

    group_path = _resolved_file(args.group, "group config")
    manifest_path = _resolved_file(args.pair_manifest, "pair manifest")
    group = dict(_load_json(group_path))
    schema = validate_group_schema(group)
    manifest = _load_json(manifest_path)
    if manifest.get("split") != "train":
        raise ValueError("pair manifest must declare split=train")
    if group["house_id"] not in set(manifest.get("sourceHouseIds", [])):
        raise ValueError("group house is not declared by the train manifest")
    token_count = int(manifest.get("tokenCount", 0))
    if token_count <= 0:
        raise ValueError("pair manifest tokenCount must be positive")

    wanted_ids = set(group["pair_ids"].values())
    records: Dict[str, Mapping[str, Any]] = {}
    for record in manifest.get("pairs", []):
        pair_id = record.get("id") if isinstance(record, Mapping) else None
        if pair_id in wanted_ids:
            if pair_id in records:
                raise ValueError(f"duplicate selected pair id: {pair_id}")
            records[str(pair_id)] = record
    missing = sorted(wanted_ids - set(records))
    if missing:
        raise ValueError("selected pair ids are missing from train manifest: " + ", ".join(missing))

    manifest_root = Path(str(manifest.get("dataRoot", manifest_path.parent))).expanduser()
    if not manifest_root.is_absolute():
        manifest_root = manifest_path.parent / manifest_root
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else manifest_root.resolve()
    _reject_test_reference(data_root, "data_root")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    for key, pair_id in group["pair_ids"].items():
        record = records[str(pair_id)]
        record_key = _pair_key(str(record.get("room_a")), str(record.get("room_b")))
        if record_key != key:
            raise ValueError(f"pair {pair_id} rooms disagree with group key {key}")
        if record.get("scene_id") != group["house_id"] or record.get("floor_id") != group["floor_id"]:
            raise ValueError(f"pair {pair_id} is outside the selected house/floor")
        if int(record.get("token_count", token_count)) != token_count:
            raise ValueError(f"pair {pair_id} token count is inconsistent")
        expected_positive = key in schema["positive_keys"]
        supervision = record.get("supervision")
        if not isinstance(supervision, Mapping) or supervision.get("is_match") is not expected_positive:
            raise ValueError(f"pair {pair_id} label disagrees with group edges")
        for side in ("a", "b"):
            selection = group["room_to_pano"][record[f"room_{side}"]]
            if selection["partial_room_id"] != record[f"partial_room_{side}"]:
                raise ValueError(f"pair {pair_id} partial-room mapping changed")
            if selection["pano_id"] != record[f"pano_{side}"]:
                raise ValueError(f"pair {pair_id} panorama mapping changed")
            image = _normalize_image_path(data_root, record[f"image_{side}"])
            if not image.is_file():
                raise FileNotFoundError(image)
            for candidate in record.get(f"candidates_{side}", []):
                if candidate.get("type") not in _PASSABLE_TYPES:
                    raise ValueError(f"pair {pair_id} contains a non-passable candidate")
        physical = _physical_candidate_pairs(record, args.endpoint_tolerance)
        if expected_positive:
            target = (
                int(supervision.get("target_candidate_a", -1)),
                int(supervision.get("target_candidate_b", -1)),
            )
            if not physical or target not in {(a, b) for a, b, _ in physical}:
                raise ValueError(f"positive pair {pair_id} has no valid shared interface")
            if supervision.get("pose_valid") is not True:
                raise ValueError(f"positive pair {pair_id} has invalid pose")
        else:
            if physical:
                raise ValueError(f"negative pair {pair_id} actually shares a passable interface")
            if supervision.get("pose_valid") is not False:
                raise ValueError(f"negative pair {pair_id} must expose pose_valid=false")

    annotations: Dict[str, Dict[str, Any]] = {}
    annotation_files: Dict[Path, Mapping[str, Any]] = {}
    for room in schema["rooms"]:
        reference = group["instance_layout_refs"][room]
        if not isinstance(reference, Mapping):
            raise ValueError(f"layout reference for {room} must be an object")
        annotation_path = _normalize_image_path(data_root, reference.get("annotation_path"))
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        payload = annotation_files.setdefault(annotation_path, _load_json(annotation_path))
        panorama = _json_pointer(payload, str(reference.get("json_pointer")))
        selection = group["room_to_pano"][room]
        expected_image = f"panos/{group['floor_id']}_{selection['partial_room_id']}_{selection['pano_id']}.jpg"
        if panorama.get("image_path") != expected_image:
            raise ValueError(f"raw annotation image mapping changed for {room}")
        if panorama.get("is_primary") is not True or panorama.get("is_inside") is not True:
            raise ValueError(f"selected panorama for {room} must be primary and inside")
        for layout_name in ("layout_raw", "layout_visible", "layout_complete"):
            layout = panorama.get(layout_name)
            if not isinstance(layout, Mapping) or len(layout.get("vertices", [])) < 3:
                raise ValueError(f"{room} is missing valid {layout_name}")
            _validate_polygon_vertices(layout["vertices"], f"{room}.{layout_name}")
        transformation = panorama.get("floor_plan_transformation")
        if not isinstance(transformation, Mapping):
            raise ValueError(f"{room} is missing floor_plan_transformation")
        if not math.isfinite(float(panorama.get("camera_height", float("nan")))):
            raise ValueError(f"{room} is missing finite camera_height")
        image_path = _normalize_image_path(annotation_path.parent, panorama["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        annotations[room] = {
            "annotation_path": annotation_path,
            "image_path": image_path,
            "panorama": panorama,
            "reference": reference,
        }

    raw_scale = annotation_files[next(iter(annotation_files))].get(
        "scale_meters_per_coordinate", {}
    ).get(group["floor_id"])
    if not math.isclose(float(raw_scale), schema["floor_scale"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("group floor scale disagrees with raw ZInD annotation")

    cache_dir = Path(args.feature_cache).expanduser().resolve()
    _reject_test_reference(cache_dir, "feature_cache")
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    cache = _load_json(metadata_path)
    if cache.get("split") != "train" or cache.get("complete") is not True:
        raise ValueError("feature cache must be a complete train cache")
    if int(cache.get("token_count", 0)) != token_count:
        raise ValueError("feature cache and pair manifest token counts differ")
    cache_manifest = cache.get("manifest")
    if not isinstance(cache_manifest, Mapping):
        raise ValueError("feature cache is missing manifest identity")
    cached_manifest_path = Path(str(cache_manifest.get("path", ""))).expanduser().resolve()
    _reject_test_reference(cached_manifest_path, "feature-cache manifest")
    if cached_manifest_path != manifest_path:
        raise ValueError("feature cache was not built from the selected train manifest")

    view_indices: Dict[str, int] = {}
    selected_paths = {str(value["image_path"]): room for room, value in annotations.items()}
    for index, view in enumerate(cache.get("views", [])):
        if not isinstance(view, Mapping):
            continue
        candidates = []
        for field in ("image_path", "relative_image_path"):
            if view.get(field):
                candidates.append(str(_normalize_image_path(data_root, view[field])))
        for candidate in candidates:
            if candidate in selected_paths:
                room = selected_paths[candidate]
                if room in view_indices and view_indices[room] != index:
                    raise ValueError(f"feature cache has duplicate selected view for {room}")
                view_indices[room] = index
    if set(view_indices) != set(schema["rooms"]):
        raise ValueError("not every selected room has one train feature-cache entry")

    return {
        "group_path": group_path,
        "manifest_path": manifest_path,
        "cache_dir": cache_dir,
        "data_root": data_root,
        "group": group,
        "schema": schema,
        "records": records,
        "annotations": annotations,
        "cache_metadata": cache,
        "cache_indices": view_indices,
        "token_count": token_count,
        "image_shape": tuple(int(value) for value in cache.get("image_shape", (512, 1024))),
    }


def metadata_report(audit: Mapping[str, Any]) -> Dict[str, Any]:
    group = audit["group"]
    pairs = []
    for edge in audit["schema"]["pair_order"]:
        key = _pair_key(*edge)
        record = audit["records"][group["pair_ids"][key]]
        supervision = record["supervision"]
        target_a = int(supervision.get("target_candidate_a", -1))
        target_b = int(supervision.get("target_candidate_b", -1))
        interface_types = None
        if supervision["is_match"]:
            interface_types = [
                record["candidates_a"][target_a]["type"],
                record["candidates_b"][target_b]["type"],
            ]
        pairs.append({
            "pair_id": record["id"],
            "group_pair": list(edge),
            "manifest_orientation": [record["room_a"], record["room_b"]],
            "is_match": bool(supervision["is_match"]),
            "shared_interface_types": interface_types,
            "pose_valid": bool(supervision["pose_valid"]),
        })
    return {
        "status": "PASS",
        "purpose": DISCLAIMER,
        "mode": "metadata-only",
        "split": "train",
        "group_id": group["group_id"],
        "house_id": group["house_id"],
        "floor_id": group["floor_id"],
        "room_ids": list(group["room_ids"]),
        "room_to_pano": copy.deepcopy(group["room_to_pano"]),
        "cache_indices": dict(audit["cache_indices"]),
        "pairs": pairs,
        "adjacency_matrix": copy.deepcopy(group["adjacency_matrix"]),
        "test_data_accessed": False,
        "model_loaded": False,
        "gpu_used": False,
    }


def _checkpoint_paths(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    values = [
        ("Bi-Layout checkpoint", Path(args.bi_layout_checkpoint)),
        ("Opening Head checkpoint", Path(args.opening_checkpoint)),
        ("matcher checkpoint", Path(args.matcher_checkpoint)),
    ]
    if args.end_to_end_checkpoint:
        values.append(("end-to-end checkpoint", Path(args.end_to_end_checkpoint)))
    return [(name, value.expanduser().resolve()) for name, value in values]


def dry_run_report(args: argparse.Namespace, audit: Mapping[str, Any]) -> Dict[str, Any]:
    report = metadata_report(audit)
    report["mode"] = "dry-run"
    report["checkpoints"] = []
    _resolved_file(args.config, "Bi-Layout config")
    for name, path in _checkpoint_paths(args):
        _reject_test_reference(path, name)
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        report["checkpoints"].append({"name": name, "path": str(path), "exists": True})
    report["planned_model_forwards"] = {
        "bi_layout_batch_size": 3,
        "forward_count_per_room": 1,
        "pair_count": 3,
        "opening_guidance": "predicted",
        "candidate_source": "predicted",
        "match_decision": "learned bidirectional dustbin",
        "layout_frame_mode": args.layout_frame_mode,
        "portal_geometry_mode": args.portal_geometry_mode,
    }
    return report


def route_predicted_decision(
    best_candidate_pair: Sequence[int],
    on_match: Callable[[int, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    """Route one frozen matcher decision without accepting any GT argument."""
    best = [int(value) for value in best_candidate_pair]
    if len(best) != 2:
        raise ValueError("best_candidate_pair must contain two indices")
    if best[0] < 0 or best[1] < 0:
        return {
            "is_match": False,
            "shared_candidate_pairs": [],
            "shared_opening_details": [],
            "relative_pose": None,
            "pose_valid": False,
            "joint_bev": None,
            "joint_bev_frame_preserving_polygon": None,
            "joint_bev_pre_manhattan_polygon": None,
            "joint_bev_generated": False,
            "predicted_relative_pose": None,
            "portal_geometry_source_A": None,
            "portal_geometry_source_B": None,
            "portal_geometry_quality_A": None,
            "portal_geometry_quality_B": None,
            "portal_geometry_valid_A": False,
            "portal_geometry_valid_B": False,
            "portal_geometry_failure_reason_A": None,
            "portal_geometry_failure_reason_B": None,
            "portal_geometry_A": None,
            "portal_geometry_B": None,
            "failure_stage": None,
            "failure_reason": None,
        }
    payload = dict(on_match(best[0], best[1]))
    return {
        "is_match": True,
        "shared_candidate_pairs": payload["shared_candidate_pairs"],
        "shared_opening_details": payload.get("shared_opening_details", []),
        "relative_pose": payload.get("relative_pose"),
        "predicted_relative_pose": payload.get(
            "predicted_relative_pose", payload.get("relative_pose")
        ),
        "pose_valid": bool(payload.get("pose_valid", False)),
        "joint_bev": payload.get("joint_bev"),
        "joint_bev_frame_preserving_polygon": payload.get(
            "joint_bev_frame_preserving_polygon"
        ),
        "joint_bev_pre_manhattan_polygon": payload.get(
            "joint_bev_pre_manhattan_polygon"
        ),
        "joint_bev_generated": bool(payload.get("joint_bev_generated", False)),
        "portal_geometry_source_A": payload.get("portal_geometry_source_A"),
        "portal_geometry_source_B": payload.get("portal_geometry_source_B"),
        "portal_geometry_quality_A": payload.get("portal_geometry_quality_A"),
        "portal_geometry_quality_B": payload.get("portal_geometry_quality_B"),
        "portal_geometry_valid_A": bool(payload.get("portal_geometry_valid_A", False)),
        "portal_geometry_valid_B": bool(payload.get("portal_geometry_valid_B", False)),
        "portal_geometry_failure_reason_A": payload.get(
            "portal_geometry_failure_reason_A"
        ),
        "portal_geometry_failure_reason_B": payload.get(
            "portal_geometry_failure_reason_B"
        ),
        "portal_geometry_A": payload.get("portal_geometry_A"),
        "portal_geometry_B": payload.get("portal_geometry_B"),
        "failure_stage": payload.get("failure_stage"),
        "failure_reason": payload.get("failure_reason"),
    }


def strict_ground_truth_payload(
    record: Mapping[str, Any],
    annotations: Mapping[str, Mapping[str, Any]],
    floor_scale: float,
    endpoint_tolerance: float,
) -> Dict[str, Any]:
    """Build evaluation labels, deliberately sanitizing all negative poses."""
    supervision = record["supervision"]
    is_match = bool(supervision["is_match"])
    common = {
        "is_match": is_match,
        "shared_candidate_pairs": [],
        "shared_interface_details": [],
        "relative_pose_gt": None,
        "pose_valid": False,
        "joint_bev_target": None,
    }
    if not is_match:
        return common

    transform = np.asarray(supervision["relative_transform_b_to_a"], dtype=np.float64)
    if transform.shape != (3, 3):
        raise ValueError("positive GT transform must be 3x3")
    room_a, room_b = record["room_a"], record["room_b"]
    pano_a = annotations[room_a]["panorama"]
    pano_b = annotations[room_b]["panorama"]
    polygon_a = np.asarray(pano_a["layout_complete"]["vertices"], dtype=np.float64)
    polygon_b = np.asarray(pano_b["layout_complete"]["vertices"], dtype=np.float64)
    from portallayout.datasets.zind_pair_mining import transform_points

    polygon_b_in_a = transform_points(polygon_b, transform)
    physical = _physical_candidate_pairs(record, endpoint_tolerance)
    for index_a, index_b, error in physical:
        candidate_a = record["candidates_a"][index_a]
        candidate_b = record["candidates_b"][index_b]
        common["shared_candidate_pairs"].append([index_a, index_b])
        common["shared_interface_details"].append({
            "candidate_a": index_a,
            "candidate_b": index_b,
            "type_a": candidate_a["type"],
            "type_b": candidate_b["type"],
            "local_endpoints_a": candidate_a["localEndpoints"],
            "local_endpoints_b": candidate_b["localEndpoints"],
            "global_endpoints_a": candidate_a["globalEndpoints"],
            "global_endpoints_b": candidate_b["globalEndpoints"],
            "endpoint_error": error,
        })
    scale = float(math.hypot(transform[0, 0], transform[1, 0]))
    yaw = float(math.atan2(transform[1, 0], transform[0, 0]))
    translation = transform[:2, 2]
    metric_factor = float(pano_a["floor_plan_transformation"]["scale"]) * floor_scale
    common.update(
        relative_pose_gt={
            "T_B_to_A": transform.tolist(),
            "relative_yaw_radians": yaw,
            "relative_yaw_degrees": math.degrees(yaw),
            "translation": translation.tolist(),
            "translation_meters": (translation * metric_factor).tolist(),
            "relative_scale": scale,
            "coordinate_system": "ZInD panorama-A local xy",
        },
        pose_valid=True,
        joint_bev_target={
            "coordinate_system": "ZInD panorama-A local xy",
            "anchor_room": room_a,
            "instances": [
                {"room_id": room_a, "polygon": polygon_a.tolist()},
                {"room_id": room_b, "polygon": polygon_b_in_a.tolist()},
            ],
        },
    )
    return common


def attach_post_inference_evaluation(
    prediction: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach GT after prediction; the copied prediction stays byte-for-byte equal."""
    return {
        "prediction": copy.deepcopy(dict(prediction)),
        "ground_truth": copy.deepcopy(dict(ground_truth)),
        'portallayout.evaluation': copy.deepcopy(dict(metrics or {})),
        "ground_truth_usage": "post-inference evaluation only",
    }


def _pose(matrix: np.ndarray) -> Dict[str, Any]:
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    yaw = float(math.atan2(matrix[1, 0], matrix[0, 0]))
    return {
        "T_B_to_A": matrix.tolist(),
        "relative_yaw_radians": yaw,
        "relative_yaw_degrees": math.degrees(yaw),
        "translation": matrix[:2, 2].tolist(),
        "relative_scale": scale,
        "coordinate_system": "ZInD panorama-A local xy",
    }


def _bi_matrix_to_zind(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("relative transform must be 3x3")
    return _BI_TO_ZIND @ value @ _BI_TO_ZIND


def _bi_points_to_zind(points: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("BEV points must have shape [N, 2]")
    output = value.copy()
    output[:, 0] *= -1.0
    return output


def _standardize_joint(joint: Mapping[str, Any], room_a: str, room_b: str) -> Dict[str, Any]:
    output = copy.deepcopy(dict(joint))
    output["coordinateSystem"] = "ZInD panorama-A local xy"
    for room, room_id in zip(output["rooms"], (room_a, room_b)):
        room["id"] = room_id
        room["boundary"] = _bi_points_to_zind(room["boundary"]).tolist()
        room["cameraCenter"] = _bi_points_to_zind([room["cameraCenter"]])[0].tolist()
        for line in room.get("boundaryLines", []):
            line["start"] = _bi_points_to_zind([line["start"]])[0].tolist()
            line["end"] = _bi_points_to_zind([line["end"]])[0].tolist()
    shared = output.get("sharedOpening")
    if shared:
        shared["worldEndpoints"] = _bi_points_to_zind(shared["worldEndpoints"]).tolist()
        for side in ("roomA", "roomB"):
            shared[side]["localEndpoints"] = _bi_points_to_zind(
                shared[side]["localEndpoints"]
            ).tolist()
            shared[side]["worldEndpoints"] = _bi_points_to_zind(
                shared[side]["worldEndpoints"]
            ).tolist()
        output["sharedDoor"] = copy.deepcopy(shared)
    matrix = _bi_matrix_to_zind(output["alignment"]["roomBToWorld"])
    output["alignment"]["roomBToWorld"] = matrix.tolist()
    output["alignment"]["transformDirection"] = "B_to_A"
    return output


def _tensor_json(value: torch.Tensor) -> Any:
    return value.detach().cpu().tolist()


def _apply_transform_xy(
    points: Sequence[Sequence[float]], matrix: Sequence[Sequence[float]]
) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("display boundary must be finite [N, 2]")
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("display transform must be finite 3x3")
    homogeneous = np.concatenate(
        [value, np.ones((len(value), 1), dtype=np.float64)], axis=1
    )
    return (transform @ homogeneous.T).T[:, :2]


def _boundary_lines(points: np.ndarray) -> List[Dict[str, Any]]:
    return [
        {
            "wallIndex": index,
            "start": points[index].tolist(),
            "end": points[(index + 1) % len(points)].tolist(),
        }
        for index in range(len(points))
    ]


def _joint_with_display_boundaries(
    template: Mapping[str, Any],
    boundary_a: Sequence[Sequence[float]],
    boundary_b: Sequence[Sequence[float]],
    representation: str,
) -> Dict[str, Any]:
    """Change only displayed room polygons while preserving the frozen M2 pose."""
    output = copy.deepcopy(dict(template))
    matrix = np.asarray(output["alignment"]["roomBToWorld"], dtype=np.float64)
    points_a = np.asarray(boundary_a, dtype=np.float64)
    points_b_world = _apply_transform_xy(boundary_b, matrix)
    for room, points, camera in (
        (output["rooms"][0], points_a, np.zeros(2, dtype=np.float64)),
        (output["rooms"][1], points_b_world, matrix[:2, 2]),
    ):
        room["boundary"] = points.tolist()
        room["boundaryLines"] = _boundary_lines(points)
        room["cameraCenter"] = np.asarray(camera, dtype=np.float64).tolist()
    output["polygonRepresentation"] = representation
    output["poseSource"] = "candidate_local_m2"
    return output


def _candidate_local_pair_fusion(
    state_a: Mapping[str, Any],
    state_b: Mapping[str, Any],
    index_a: int,
    index_b: int,
    shared_details: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fuse a matcher-frozen pair using only predicted M2 geometry inputs."""
    from portallayout.geometry.candidate_local_portal_geometry import (
        CandidateLocalPortalConfig,
        estimate_candidate_local_portal_geometry,
    )
    from portallayout.geometry.joint_layout import build_joint_layout_from_endpoints

    def estimate(state: Mapping[str, Any], candidate_index: int):
        intervals = state["opening_prediction"]["intervals"]
        if not 0 <= candidate_index < len(intervals):
            raise ValueError(f"candidate index {candidate_index} is outside predicted intervals")
        candidate = state["_opening_candidates"][candidate_index]
        interval = [int(intervals[candidate_index][0]), int(intervals[candidate_index][1])]
        if interval != [int(candidate.token_start), int(candidate.token_end)]:
            raise RuntimeError("matcher candidate and predicted interval order diverged")
        contour = state["single_room_layout"][
            "layout_polygon_pre_manhattan_local_bi_xz"
        ]
        if contour is None:
            raise ValueError("layout_polygon_pre_manhattan_local is unavailable")
        probability = state["opening_prediction"]["probability"]
        return estimate_candidate_local_portal_geometry(
            contour_local=contour,
            token_interval=interval,
            token_count=len(probability),
            opening_probability=probability,
            config=CandidateLocalPortalConfig(),
        )

    result_a = estimate(state_a, index_a)
    result_b = estimate(state_b, index_b)
    portal_a = result_a.to_json()
    portal_b = result_b.to_json()
    portal_a.update(candidate_index=int(index_a), entered_joint_layout=False)
    portal_b.update(candidate_index=int(index_b), entered_joint_layout=False)
    common = {
        "shared_candidate_pairs": [[int(index_a), int(index_b)]],
        "shared_opening_details": copy.deepcopy(list(shared_details)),
        "portal_geometry_source_A": "candidate_local_m2",
        "portal_geometry_source_B": "candidate_local_m2",
        "portal_geometry_quality_A": portal_a["predicted_only_quality"],
        "portal_geometry_quality_B": portal_b["predicted_only_quality"],
        "portal_geometry_valid_A": bool(result_a.geometry_valid),
        "portal_geometry_valid_B": bool(result_b.geometry_valid),
        "portal_geometry_failure_reason_A": result_a.failure_reason,
        "portal_geometry_failure_reason_B": result_b.failure_reason,
        "portal_geometry_A": portal_a,
        "portal_geometry_B": portal_b,
    }
    if not result_a.geometry_valid or not result_b.geometry_valid:
        failed_sides = [
            side
            for side, result in (("A", result_a), ("B", result_b))
            if not result.geometry_valid
        ]
        common.update({
            "relative_pose": None,
            "predicted_relative_pose": None,
            "pose_valid": False,
            "joint_bev": None,
            "joint_bev_frame_preserving_polygon": None,
            "joint_bev_pre_manhattan_polygon": None,
            "joint_bev_generated": False,
            "failure_stage": "candidate_local_portal_geometry",
            "failure_reason": "M2 hard geometry failure on side " + "/".join(failed_sides),
        })
        return common, {"native_joint": None, "raw_joint": None}

    primary_a, primary_b = result_a.primary, result_b.primary
    if primary_a is None or primary_b is None:
        raise RuntimeError("valid M2 result is missing its primary hypothesis")
    endpoints_a = np.asarray(primary_a.endpoints_local_zind_xy, dtype=np.float64)
    endpoints_b = np.asarray(primary_b.endpoints_local_zind_xy, dtype=np.float64)
    raw_a = _bi_points_to_zind(
        state_a["single_room_layout"]["layout_polygon_pre_manhattan_local_bi_xz"]
    )
    raw_b = _bi_points_to_zind(
        state_b["single_room_layout"]["layout_polygon_pre_manhattan_local_bi_xz"]
    )
    frame_a = np.asarray(
        state_a["single_room_layout"]["predicted_polygon_zind_local_xy"],
        dtype=np.float64,
    )
    frame_b = np.asarray(
        state_b["single_room_layout"]["predicted_polygon_zind_local_xy"],
        dtype=np.float64,
    )
    try:
        raw_joint = build_joint_layout_from_endpoints(
            state_a["single_room_layout"]["predicted_layout_bi_xz"],
            state_b["single_room_layout"]["predicted_layout_bi_xz"],
            endpoints_a,
            endpoints_b,
            boundary_a=raw_a,
            boundary_b=raw_b,
            calibrate_scale=True,
            room_ids=(str(state_a["room_id"]), str(state_b["room_id"])),
            coordinate_system="ZInD panorama-A local xy",
            opening_metadata_a={
                "method": "candidate_local_m2",
                "candidate_index": int(index_a),
                "token_interval": list(result_a.token_interval),
            },
            opening_metadata_b={
                "method": "candidate_local_m2",
                "candidate_index": int(index_b),
                "token_interval": list(result_b.token_interval),
            },
        )
        raw_joint["polygonRepresentation"] = "pre_manhattan_panorama_local"
        raw_joint["poseSource"] = "candidate_local_m2"
        frame_joint = _joint_with_display_boundaries(
            raw_joint,
            frame_a,
            frame_b,
            "frame_preserving_manhattan_panorama_local",
        )
        pose = _pose(np.asarray(raw_joint["alignment"]["roomBToWorld"]))
    except Exception as exc:
        common.update({
            "relative_pose": None,
            "predicted_relative_pose": None,
            "pose_valid": False,
            "joint_bev": None,
            "joint_bev_frame_preserving_polygon": None,
            "joint_bev_pre_manhattan_polygon": None,
            "joint_bev_generated": False,
            "failure_stage": "candidate_local_joint_layout",
            "failure_reason": str(exc),
        })
        return common, {"native_joint": None, "raw_joint": None}

    portal_a["entered_joint_layout"] = True
    portal_b["entered_joint_layout"] = True
    for detail in common["shared_opening_details"]:
        detail["portal_geometry_mode"] = "candidate_local_m2"
        detail["portal_endpoints_A"] = endpoints_a.tolist()
        detail["portal_endpoints_B"] = endpoints_b.tolist()
    common.update({
        "relative_pose": pose,
        "predicted_relative_pose": pose,
        "pose_valid": True,
        "joint_bev": frame_joint,
        "joint_bev_frame_preserving_polygon": frame_joint,
        "joint_bev_pre_manhattan_polygon": raw_joint,
        "joint_bev_generated": True,
        "failure_stage": None,
        "failure_reason": None,
    })
    return common, {"native_joint": frame_joint, "raw_joint": raw_joint}


def _matcher_evidence(outputs: Mapping[str, torch.Tensor], score: float) -> Dict[str, Any]:
    return {"candidate_pair_score": np.asarray([[score]], dtype=np.float64)}


def _load_models(args: argparse.Namespace, audit: Mapping[str, Any], device: torch.device):
    import torch

    from tools.debug_cross_scene_flow import load_bi_layout
    from portallayout.utils.matcher_checkpoint import load_cross_scene_matcher_checkpoint
    from portallayout.utils.opening_checkpoint import (
        load_opening_head_checkpoint,
        resolve_opening_probability_threshold,
    )

    config_path = str(_resolved_file(args.config, "Bi-Layout config"))
    bi_path = str(_resolved_file(args.bi_layout_checkpoint, "Bi-Layout checkpoint"))
    matcher_path = str(_resolved_file(args.matcher_checkpoint, "matcher checkpoint"))
    opening_path = str(_resolved_file(args.opening_checkpoint, "Opening Head checkpoint"))
    bi_layout, bi_report = load_bi_layout(config_path, bi_path, device, True)
    matcher, matcher_report = load_cross_scene_matcher_checkpoint(
        matcher_path,
        expected_feature_dim=int(bi_layout.patch_dim),
        expected_token_count=int(bi_layout.patch_num),
        expected_branch_order="extended_first",
        bi_layout_config_path=config_path,
        bi_layout_checkpoint_path=bi_path,
        device=device,
    )
    opening_report = load_opening_head_checkpoint(
        matcher.opening_head,
        opening_path,
        expected_feature_dim=int(bi_layout.patch_dim),
        expected_branch_order="extended_first",
        expected_token_count=int(bi_layout.patch_num),
        bi_layout_config_path=config_path,
        bi_layout_checkpoint_path=bi_path,
    )
    opening_report["loadOrder"] = "matcher_checkpoint_then_opening_override"
    e2e_report = {"loaded": False, "path": None}
    e2e_threshold = None
    if args.end_to_end_checkpoint:
        e2e_path = _resolved_file(args.end_to_end_checkpoint, "end-to-end checkpoint")
        payload = torch.load(str(e2e_path), map_location="cpu")
        if not isinstance(payload, Mapping) or payload.get("task") != "ZInD_matching full end-to-end joint fine-tuning":
            raise ValueError("end-to-end checkpoint has the wrong task contract")
        _reject_test_checkpoint_metadata(payload.get("data_contract"))
        bi_layout.load_state_dict(payload["bi_layout_state_dict"], strict=True)
        matcher.load_state_dict(payload["matcher_state_dict"], strict=True)
        e2e_threshold = payload.get("opening_probability_threshold")
        e2e_report = {
            "loaded": True,
            "path": str(e2e_path),
            "completedEpoch": payload.get("completed_epoch"),
            "task": payload.get("task"),
        }
        del payload
    threshold = resolve_opening_probability_threshold(
        args.opening_probability_threshold,
        opening_report,
        matcher_report,
    )
    if args.opening_probability_threshold is None and e2e_threshold is not None:
        value = float(e2e_threshold)
        if not 0 <= value <= 1:
            raise ValueError("end-to-end checkpoint opening threshold is invalid")
        threshold = {"value": value, "source": "end_to_end_checkpoint"}
    bi_layout.eval().requires_grad_(False)
    matcher.eval().requires_grad_(False)
    return bi_layout, matcher, threshold, {
        "bi_layout": bi_report,
        "matcher": matcher_report,
        "opening_head": opening_report,
        "end_to_end": e2e_report,
        "feature_cache_used_for_inference": False,
        "feature_cache_role": "train-only identity/coverage validation",
    }


def _reject_test_checkpoint_metadata(data_contract: Any) -> None:
    if data_contract is None:
        return

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if str(key).lower() == "test" and child not in (None, False, [], {}):
                    raise ValueError(f"checkpoint metadata points to test: {child_path}")
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            candidate = Path(value).expanduser()
            if any(part.lower() == "test" for part in candidate.parts):
                raise ValueError(f"checkpoint metadata points to test: {path}")

    walk(data_contract, "data_contract")


def _room_inference_inputs(audit: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return the image-only room inputs visible to model inference."""
    return {
        room: {
            "room_id": room,
            "partial_room_id": audit["group"]["room_to_pano"][room]["partial_room_id"],
            "pano_id": audit["group"]["room_to_pano"][room]["pano_id"],
            "image_path": str(audit["annotations"][room]["image_path"]),
            "cache_index": int(audit["cache_indices"][room]),
        }
        for room in audit["schema"]["rooms"]
    }


def _pair_inference_input(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip a manifest record down to identifiers; supervision is excluded."""
    return {
        "pair_id": str(record["id"]),
        "room_a": str(record["room_a"]),
        "room_b": str(record["room_b"]),
        "pano_a": str(record["pano_a"]),
        "pano_b": str(record["pano_b"]),
    }


def _load_image_batch(
    room_inputs: Mapping[str, Mapping[str, Any]],
    room_order: Sequence[str],
    image_shape: Sequence[int],
) -> torch.Tensor:
    import torch
    from PIL import Image

    height, width = (int(image_shape[0]), int(image_shape[1]))
    arrays = []
    resampling = getattr(Image, "Resampling", Image)
    for room in room_order:
        with Image.open(room_inputs[room]["image_path"]) as image:
            image = image.convert("RGB").resize((width, height), resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
        if array.shape != (height, width, 3) or not np.isfinite(array).all():
            raise ValueError(f"invalid panorama image for {room}")
        arrays.append(torch.from_numpy(array.transpose(2, 0, 1).copy()))
    return torch.stack(arrays, dim=0)


def _run_room_forward(
    args: argparse.Namespace,
    room_inputs: Mapping[str, Mapping[str, Any]],
    room_order: Sequence[str],
    image_shape: Sequence[int],
    bi_layout: torch.nn.Module,
    matcher: torch.nn.Module,
    device: torch.device,
    opening_threshold: float,
) -> Dict[str, Dict[str, Any]]:
    import torch

    from portallayout.models.cross_scene_matcher import (
        opening_probabilities_to_intervals,
        resolve_enclosed_extended_depth,
    )
    from tools.debug_cross_scene_flow import prediction_to_layout
    from tools.train_zind_cross_scene_matcher import predicted_candidate_masks
    from portallayout.engine.cross_scene_estimator import opening_candidates_from_intervals
    from portallayout.geometry.joint_layout import layout_xz

    images = _load_image_batch(room_inputs, room_order, image_shape).to(
        device=device, dtype=torch.float32
    )
    with torch.no_grad():
        outputs = bi_layout(images, return_features=True)
        enclosed, extended = resolve_enclosed_extended_depth(outputs, "extended_first")
        opening_signal = matcher.opening_head(
            outputs["layout_feature"], enclosed, extended
        )
        masks, valid = predicted_candidate_masks(
            opening_signal["opening_probability"],
            opening_threshold,
            args.min_opening_width_tokens,
            args.max_openings_per_view,
        )

    rooms: Dict[str, Dict[str, Any]] = {}
    for index, room in enumerate(room_order):
        enclosed_layout, enclosed_report = prediction_to_layout(
            enclosed[index],
            outputs["ratio"][index],
            layout_frame_mode=args.layout_frame_mode,
        )
        extended_layout, extended_report = prediction_to_layout(
            extended[index],
            outputs["ratio"][index],
            layout_frame_mode=args.layout_frame_mode,
        )
        probability = opening_signal["opening_probability"][index]
        intervals = opening_probabilities_to_intervals(
            probability,
            threshold=opening_threshold,
            min_width_tokens=args.min_opening_width_tokens,
            max_intervals=args.max_openings_per_view,
        )
        candidates = opening_candidates_from_intervals(
            enclosed_layout, intervals, probability
        )
        valid_count = int(valid[index].sum().item())
        if valid_count != len(intervals) or valid_count != len(candidates):
            raise RuntimeError("predicted opening candidate order diverged")
        room_input = room_inputs[room]
        predicted_polygon = _bi_points_to_zind(layout_xz(enclosed_layout))
        rooms[room] = {
            "room_id": room,
            "partial_room_id": room_input["partial_room_id"],
            "pano_id": room_input["pano_id"],
            "image_path": room_input["image_path"],
            "cache_index": room_input["cache_index"],
            "model_forward_count": 1,
            "single_room_layout": {
                "predicted_layout_bi_xz": enclosed_layout,
                "predicted_polygon_zind_local_xy": predicted_polygon.tolist(),
                "layout_frame_mode": enclosed_report["layoutFrameMode"],
                "layout_frame": enclosed_report["layoutFrame"],
                "not_valid_for_cross_scene_pose": enclosed_report[
                    "notValidForCrossScenePose"
                ],
                "manhattan_heading": {
                    "radians": enclosed_report.get("manhattanHeadingRadians"),
                    "degrees": enclosed_report.get("manhattanHeadingDegrees"),
                    "source": enclosed_report.get("headingSource"),
                    "confidence": enclosed_report.get("confidence"),
                },
                "layout_polygon_pre_manhattan_local_bi_xz": enclosed_report.get(
                    "layoutPolygonPreManhattanLocalBiXZ"
                ),
                "layout_polygon_manhattan_canonical_bi_xz": enclosed_report.get(
                    "layoutPolygonManhattanCanonicalBiXZ"
                ),
                "layout_polygon_panorama_local_bi_xz": enclosed_report.get(
                    "layoutPolygonPanoramaLocalBiXZ"
                ),
                "postprocess": enclosed_report,
                "extended_layout_bi_xz": extended_layout,
                "extended_postprocess": extended_report,
            },
            "opening_prediction": {
                "probability": _tensor_json(probability),
                "threshold": opening_threshold,
                "candidate_count": len(candidates),
                "intervals": [list(value) for value in intervals],
                "candidates": [candidate.to_json() for candidate in candidates],
            },
            "_feature": outputs["layout_feature"][index : index + 1],
            "_enclosed": enclosed[index : index + 1],
            "_extended": extended[index : index + 1],
            "_candidate_masks": masks[index : index + 1],
            "_candidate_valid": valid[index : index + 1],
            "_opening_candidates": candidates,
        }
    return rooms


def _inference_room_public(room: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in room.items() if not key.startswith("_")}


def _room_result_with_post_inference_gt(
    room: Mapping[str, Any], annotation: Mapping[str, Any], floor_scale: float
) -> Dict[str, Any]:
    """Attach independent raw/visible/complete supervision after forward."""
    output = _inference_room_public(room)
    panorama = annotation["panorama"]
    output["ground_truth"] = {
        "usage": "post-inference evaluation only",
        "layouts": {
            "raw": _compact_layout(panorama["layout_raw"]),
            "visible": _compact_layout(panorama["layout_visible"]),
            "complete": _compact_layout(panorama["layout_complete"]),
        },
        "instance_polygon_source": "layout_complete",
        "floor_plan_transformation": copy.deepcopy(panorama["floor_plan_transformation"]),
        "camera_height": float(panorama["camera_height"]),
        "ceiling_height": float(panorama["ceiling_height"]),
        "scale_meters_per_coordinate": float(floor_scale),
        "annotation_ref": {
            "path": str(annotation["annotation_path"]),
            "json_pointer": annotation["reference"]["json_pointer"],
        },
    }
    return output


def _run_pair_prediction(
    args: argparse.Namespace,
    matcher: torch.nn.Module,
    pair_input: Mapping[str, Any],
    rooms: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import torch

    from portallayout.engine.cross_scene_pipeline import CrossScenePipeline, CrossScenePipelineConfig

    room_a, room_b = str(pair_input["room_a"]), str(pair_input["room_b"])
    state_a, state_b = rooms[room_a], rooms[room_b]
    with torch.no_grad():
        outputs = matcher(
            state_a["_feature"],
            state_b["_feature"],
            state_a["_enclosed"],
            state_a["_extended"],
            state_b["_enclosed"],
            state_b["_extended"],
            candidate_masks_a=state_a["_candidate_masks"],
            candidate_masks_b=state_b["_candidate_masks"],
            candidate_valid_a=state_a["_candidate_valid"],
            candidate_valid_b=state_b["_candidate_valid"],
            opening_guidance_a=None,
            opening_guidance_b=None,
            opening_guidance_mode="predicted",
        )
    best = [int(value) for value in outputs["best_candidate_pair"][0].tolist()]
    artifacts: Dict[str, Any] = {
        "matcher_outputs": outputs,
        "native_joint": None,
        "raw_joint": None,
        "candidate_masks_a": state_a["_candidate_masks"],
        "candidate_masks_b": state_b["_candidate_masks"],
        "candidate_valid_a": state_a["_candidate_valid"],
        "candidate_valid_b": state_b["_candidate_valid"],
    }

    def on_match(index_a: int, index_b: int) -> Mapping[str, Any]:
        selected_a = state_a["_opening_candidates"][index_a]
        selected_b = state_b["_opening_candidates"][index_b]
        score = float(outputs["candidate_pair_score"][0, index_a, index_b].item())
        shared_details = [{
            "candidate_a": index_a,
            "candidate_b": index_b,
            "opening_a": selected_a.to_json(),
            "opening_b": selected_b.to_json(),
            "candidate_pair_score": score,
        }]
        if args.portal_geometry_mode == "candidate_local_m2":
            try:
                payload, geometry_artifacts = _candidate_local_pair_fusion(
                    state_a,
                    state_b,
                    index_a,
                    index_b,
                    shared_details,
                )
                artifacts.update(geometry_artifacts)
                return payload
            except Exception as exc:
                return {
                    "shared_candidate_pairs": [[index_a, index_b]],
                    "shared_opening_details": shared_details,
                    "relative_pose": None,
                    "predicted_relative_pose": None,
                    "pose_valid": False,
                    "joint_bev": None,
                    "joint_bev_frame_preserving_polygon": None,
                    "joint_bev_pre_manhattan_polygon": None,
                    "joint_bev_generated": False,
                    "portal_geometry_source_A": "candidate_local_m2",
                    "portal_geometry_source_B": "candidate_local_m2",
                    "portal_geometry_quality_A": None,
                    "portal_geometry_quality_B": None,
                    "portal_geometry_valid_A": False,
                    "portal_geometry_valid_B": False,
                    "portal_geometry_failure_reason_A": str(exc),
                    "portal_geometry_failure_reason_B": str(exc),
                    "portal_geometry_A": None,
                    "portal_geometry_B": None,
                    "failure_stage": "candidate_local_portal_geometry",
                    "failure_reason": str(exc),
                }
        try:
            pipeline = CrossScenePipeline(
                CrossScenePipelineConfig(
                    top_k=args.top_k,
                    use_passability=True,
                    feature_weight=1.0,
                    strict_polygon_validation=True,
                )
            )
            result = pipeline.run(
                state_a["single_room_layout"]["predicted_layout_bi_xz"],
                state_b["single_room_layout"]["predicted_layout_bi_xz"],
                match_evidence=_matcher_evidence(outputs, score),
                openings_a=[replace(selected_a, candidate_index=0)],
                openings_b=[replace(selected_b, candidate_index=0)],
            )
            artifacts["native_joint"] = result.best_joint_layout
            standardized = _standardize_joint(result.best_joint_layout, room_a, room_b)
            pose = _pose(np.asarray(standardized["alignment"]["roomBToWorld"]))
            return {
                "shared_candidate_pairs": [[index_a, index_b]],
                "shared_opening_details": shared_details,
                "relative_pose": pose,
                "predicted_relative_pose": pose,
                "pose_valid": True,
                "joint_bev": standardized,
                "joint_bev_frame_preserving_polygon": standardized,
                "joint_bev_pre_manhattan_polygon": None,
                "joint_bev_generated": True,
                "portal_geometry_source_A": "polygon_mapper",
                "portal_geometry_source_B": "polygon_mapper",
                "portal_geometry_quality_A": None,
                "portal_geometry_quality_B": None,
                "portal_geometry_valid_A": True,
                "portal_geometry_valid_B": True,
                "portal_geometry_failure_reason_A": None,
                "portal_geometry_failure_reason_B": None,
                "portal_geometry_A": None,
                "portal_geometry_B": None,
                "failure_stage": None,
                "failure_reason": None,
            }
        except Exception as exc:
            return {
                "shared_candidate_pairs": [[index_a, index_b]],
                "shared_opening_details": shared_details,
                "relative_pose": None,
                "predicted_relative_pose": None,
                "pose_valid": False,
                "joint_bev": None,
                "joint_bev_frame_preserving_polygon": None,
                "joint_bev_pre_manhattan_polygon": None,
                "joint_bev_generated": False,
                "portal_geometry_source_A": "polygon_mapper",
                "portal_geometry_source_B": "polygon_mapper",
                "portal_geometry_quality_A": None,
                "portal_geometry_quality_B": None,
                "portal_geometry_valid_A": False,
                "portal_geometry_valid_B": False,
                "portal_geometry_failure_reason_A": str(exc),
                "portal_geometry_failure_reason_B": str(exc),
                "portal_geometry_A": None,
                "portal_geometry_B": None,
                "failure_stage": "predicted_geometry_fusion",
                "failure_reason": str(exc),
            }

    routed = route_predicted_decision(best, on_match)
    valid_a = state_a["_candidate_valid"][0]
    valid_b = state_b["_candidate_valid"][0]
    public = {
        "pair_id": pair_input["pair_id"],
        "room_a": room_a,
        "room_b": room_b,
        "pano_a": pair_input["pano_a"],
        "pano_b": pair_input["pano_b"],
        "decision_source": "predicted openings + matcher learned dustbin",
        "opening_guidance_mode": "predicted",
        "candidate_source": "predicted",
        "portal_geometry_mode": args.portal_geometry_mode,
        "candidate_count_a": int(valid_a.sum().item()),
        "candidate_count_b": int(valid_b.sum().item()),
        "best_candidate_pair": best,
        "pair_match_probability": float(outputs["pair_match_probability"][0].item()),
        "pair_no_match_probability": float(outputs["pair_no_match_probability"][0].item()),
        "candidate_pair_scores": _tensor_json(outputs["candidate_pair_score"][0]),
    }
    public.update(routed)
    return public, artifacts


def _candidate_rank_after_inference(
    record: Mapping[str, Any], artifacts: Mapping[str, Any], tolerance: float
) -> Dict[str, Any]:
    if not record["supervision"]["is_match"]:
        return {"gt_candidate_rank": None, "gt_candidate_recalled": None}
    from portallayout.models.cross_scene_matcher import candidate_intervals_to_mask
    from tools.train_zind_cross_scene_matcher import candidate_targets_from_affinity
    import torch

    token_count = int(record["token_count"])
    affinity = torch.zeros((1, token_count, token_count), dtype=torch.float32)
    for index_a, index_b, _ in _physical_candidate_pairs(record, tolerance):
        mask_a = candidate_intervals_to_mask(
            [record["candidates_a"][index_a]["tokenInterval"]], token_count
        )[0]
        mask_b = candidate_intervals_to_mask(
            [record["candidates_b"][index_b]["tokenInterval"]], token_count
        )[0]
        affinity[0] = torch.maximum(
            affinity[0], torch.outer(mask_a.float(), mask_b.float())
        )
    masks_a = artifacts["candidate_masks_a"].detach().cpu()
    masks_b = artifacts["candidate_masks_b"].detach().cpu()
    targets = candidate_targets_from_affinity(masks_a, affinity, masks_b)[0].bool()
    valid = (
        artifacts["candidate_valid_a"].detach().cpu()[0].unsqueeze(-1)
        & artifacts["candidate_valid_b"].detach().cpu()[0].unsqueeze(0)
    )
    scores = artifacts["matcher_outputs"]["candidate_pair_score"][0].detach().cpu()
    target_indices = torch.nonzero(targets & valid, as_tuple=False)
    if target_indices.numel() == 0:
        return {"gt_candidate_rank": None, "gt_candidate_recalled": False}
    valid_scores = scores[valid]
    ranks = []
    for index in target_indices:
        score = scores[int(index[0]), int(index[1])]
        ranks.append(1 + int((valid_scores > score).sum().item()))
    return {"gt_candidate_rank": min(ranks), "gt_candidate_recalled": True}


def _wrapped_angle_error(first: float, second: float) -> float:
    return float(abs((first - second + math.pi) % (2.0 * math.pi) - math.pi))


def _portal_endpoint_metrics(
    predicted: Sequence[Sequence[float]], target: Sequence[Sequence[float]]
) -> Dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    if pred.shape != (2, 2) or gt.shape != (2, 2):
        raise ValueError("portal endpoint metrics require two [x, y] endpoints")
    direct = float(math.sqrt(np.mean(np.square(pred - gt))))
    reverse = float(math.sqrt(np.mean(np.square(pred - gt[::-1]))))
    order = "same" if direct <= reverse else "reversed"
    aligned = gt if order == "same" else gt[::-1]
    return {
        "endpoint_order": order,
        "endpoint_rmse": min(direct, reverse),
        "midpoint_error": float(np.linalg.norm(pred.mean(axis=0) - gt.mean(axis=0))),
        "width_error": abs(
            float(np.linalg.norm(pred[1] - pred[0]))
            - float(np.linalg.norm(gt[1] - gt[0]))
        ),
        "aligned_target_endpoints": aligned.tolist(),
    }


def _pair_metrics(
    prediction: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    record: Mapping[str, Any],
    annotations: Mapping[str, Mapping[str, Any]],
    floor_scale: float,
    tolerance: float,
) -> Dict[str, Any]:
    from portallayout.engine.cross_scene_estimator import polygon_overlap_ratio

    gt_match = bool(ground_truth["is_match"])
    pred_match = bool(prediction["is_match"])
    metrics: Dict[str, Any] = {
        "classification_correct": pred_match == gt_match,
        "false_positive": pred_match and not gt_match,
        "false_negative": gt_match and not pred_match,
    }
    metrics.update(_candidate_rank_after_inference(record, artifacts, tolerance))
    metrics.update({
        "yaw_error_degrees": None,
        "translation_error": None,
        "translation_error_meters": None,
        "absolute_log_scale_error": None,
        "joint_bev_instance_overlap_A": None,
        "joint_bev_instance_overlap_B": None,
        "joint_bev_frame_preserving_instance_overlap_A": None,
        "joint_bev_frame_preserving_instance_overlap_B": None,
        "joint_bev_pre_manhattan_instance_overlap_A": None,
        "joint_bev_pre_manhattan_instance_overlap_B": None,
        "portal_endpoint_A": None,
        "portal_endpoint_B": None,
    })
    if not (gt_match and prediction.get("pose_valid")):
        return metrics
    predicted_pose = prediction["relative_pose"]
    target_pose = ground_truth["relative_pose_gt"]
    metrics["yaw_error_degrees"] = math.degrees(
        _wrapped_angle_error(
            float(predicted_pose["relative_yaw_radians"]),
            float(target_pose["relative_yaw_radians"]),
        )
    )
    translation_error = float(np.linalg.norm(
        np.asarray(predicted_pose["translation"], dtype=np.float64)
        - np.asarray(target_pose["translation"], dtype=np.float64)
    ))
    metrics["translation_error"] = translation_error
    pano_a = annotations[record["room_a"]]["panorama"]
    metric_factor = float(pano_a["floor_plan_transformation"]["scale"]) * floor_scale
    metrics["translation_error_meters"] = translation_error * metric_factor
    metrics["absolute_log_scale_error"] = abs(math.log(
        max(float(predicted_pose["relative_scale"]), 1e-8)
        / max(float(target_pose["relative_scale"]), 1e-8)
    ))
    target_instances = ground_truth["joint_bev_target"]["instances"]
    representations = {
        "frame_preserving": prediction.get("joint_bev_frame_preserving_polygon"),
        "pre_manhattan": prediction.get("joint_bev_pre_manhattan_polygon"),
    }
    for name, joint in representations.items():
        if joint is None:
            continue
        predicted_instances = joint["rooms"]
        for side, index in (("A", 0), ("B", 1)):
            metrics[f"joint_bev_{name}_instance_overlap_{side}"] = polygon_overlap_ratio(
                np.asarray(predicted_instances[index]["boundary"]),
                np.asarray(target_instances[index]["polygon"]),
            )
    metrics["joint_bev_instance_overlap_A"] = metrics[
        "joint_bev_frame_preserving_instance_overlap_A"
    ]
    metrics["joint_bev_instance_overlap_B"] = metrics[
        "joint_bev_frame_preserving_instance_overlap_B"
    ]
    gt_interfaces = ground_truth.get("shared_interface_details", [])
    if gt_interfaces:
        target = gt_interfaces[0]
        for side in ("A", "B"):
            portal = prediction.get(f"portal_geometry_{side}")
            if portal is None or portal.get("endpoints_local") is None:
                continue
            metrics[f"portal_endpoint_{side}"] = _portal_endpoint_metrics(
                portal["endpoints_local"], target[f"local_endpoints_{side.lower()}"]
            )
    return metrics


def _render_polygons_svg(
    path: Path, polygons: Sequence[Tuple[str, Sequence[Sequence[float]], str]]
) -> None:
    arrays = [(name, np.asarray(points, dtype=np.float64), color) for name, points, color in polygons]
    arrays = [item for item in arrays if item[1].ndim == 2 and len(item[1]) >= 3]
    if not arrays:
        return
    all_points = np.concatenate([points for _, points, _ in arrays], axis=0)
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = 700.0 / float(span.max())
    center = (minimum + maximum) / 2.0

    def project(points: np.ndarray) -> np.ndarray:
        pixels = (points - center) * scale
        pixels[:, 1] *= -1
        pixels += 400
        return pixels

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800" viewBox="0 0 800 800">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for name, points, color in arrays:
        pixel = project(points)
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in pixel)
        lines.append(f'<polygon points="{encoded}" fill="none" stroke="{color}" stroke-width="4"/>')
        at = pixel.mean(axis=0)
        lines.append(
            f'<text x="{at[0]:.2f}" y="{at[1]:.2f}" fill="{color}" font-size="20">'
            f'{html.escape(name)}</text>'
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_portal_geometry_svg(
    path: Path,
    raw_contour_bi_xz: Sequence[Sequence[float]],
    portal: Mapping[str, Any],
    gt_endpoints_zind_xy: Optional[Sequence[Sequence[float]]] = None,
) -> None:
    raw = _bi_points_to_zind(raw_contour_bi_xz)
    endpoints = np.asarray(portal["endpoints_local"], dtype=np.float64)
    tangent = np.asarray(portal["wall_tangent"], dtype=np.float64)
    radius = max(float(np.max(np.linalg.norm(raw, axis=1))), 1e-6)
    midpoint = endpoints.mean(axis=0)
    wall = np.stack([midpoint - 0.8 * radius * tangent, midpoint + 0.8 * radius * tangent])
    values = [raw, endpoints, wall, np.zeros((1, 2), dtype=np.float64)]
    gt = None
    if gt_endpoints_zind_xy is not None:
        gt = np.asarray(gt_endpoints_zind_xy, dtype=np.float64)
        values.append(gt)
    all_points = np.concatenate(values, axis=0)
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = 680.0 / float(span.max())
    center = (minimum + maximum) / 2.0

    def project(points: np.ndarray) -> np.ndarray:
        pixel = (points - center) * scale
        pixel[:, 1] *= -1
        pixel += 400
        return pixel

    raw_px, endpoint_px, wall_px = project(raw), project(endpoints), project(wall)
    origin = project(np.zeros((1, 2), dtype=np.float64))[0]
    raw_encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in raw_px)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800" viewBox="0 0 800 800">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<polygon points="{raw_encoded}" fill="none" stroke="#777" stroke-width="2"/>',
    ]
    for endpoint in endpoint_px:
        lines.append(
            f'<line x1="{origin[0]:.2f}" y1="{origin[1]:.2f}" '
            f'x2="{endpoint[0]:.2f}" y2="{endpoint[1]:.2f}" '
            'stroke="#68a" stroke-width="2" stroke-dasharray="8,5"/>'
        )
    lines.extend([
        f'<line x1="{wall_px[0,0]:.2f}" y1="{wall_px[0,1]:.2f}" '
        f'x2="{wall_px[1,0]:.2f}" y2="{wall_px[1,1]:.2f}" '
        'stroke="#1774c8" stroke-width="4"/>',
        f'<line x1="{endpoint_px[0,0]:.2f}" y1="{endpoint_px[0,1]:.2f}" '
        f'x2="{endpoint_px[1,0]:.2f}" y2="{endpoint_px[1,1]:.2f}" '
        'stroke="#d62728" stroke-width="9"/>',
    ])
    if gt is not None:
        gt_px = project(gt)
        lines.append(
            f'<line x1="{gt_px[0,0]:.2f}" y1="{gt_px[0,1]:.2f}" '
            f'x2="{gt_px[1,0]:.2f}" y2="{gt_px[1,1]:.2f}" '
            'stroke="#289128" stroke-width="6" stroke-dasharray="10,5"/>'
        )
    quality = portal.get("predicted_only_quality", {})
    label = (
        f"{portal.get('primary_variant')} residual={quality.get('normalized_residual')} "
        f"inliers={quality.get('inlier_ratio')} gate={quality.get('quality_gate_pass')}"
    )
    lines.append(
        f'<text x="20" y="30" fill="#222" font-size="16">{html.escape(label)}</text>'
    )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_outputs(
    output_dir: Path,
    audit: Mapping[str, Any],
    rooms: Mapping[str, Mapping[str, Any]],
    public_rooms: Mapping[str, Mapping[str, Any]],
    pair_results: Sequence[Mapping[str, Any]],
    pair_artifacts: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    from portallayout.engine.cross_scene_pipeline import atomic_write_json
    from portallayout.geometry.joint_layout import render_joint_boundary, render_joint_boundary_svg

    room_links = []
    for room_id, state in rooms.items():
        public = public_rooms[room_id]
        room_json = output_dir / "rooms" / f"{room_id}.json"
        atomic_write_json(str(room_json), public)
        predicted = public["single_room_layout"]["predicted_polygon_zind_local_xy"]
        target = public["ground_truth"]["layouts"]["complete"]["vertices"]
        room_svg = output_dir / "rooms" / f"{room_id}.svg"
        _render_polygons_svg(
            room_svg,
            (("predicted", predicted, "#1e6ecd"), ("GT complete", target, "#289128")),
        )
        room_links.append((room_id, room_json.relative_to(output_dir), room_svg.relative_to(output_dir)))

    pair_links = []
    for result in pair_results:
        pair_id = result["prediction"]["pair_id"]
        pair_json = output_dir / "pairs" / f"{pair_id}.json"
        atomic_write_json(str(pair_json), result)
        room_a = result["prediction"]["room_a"]
        room_b = result["prediction"]["room_b"]
        independent_svg = output_dir / "pairs" / f"{pair_id}_independent.svg"
        _render_polygons_svg(independent_svg, (
            (room_a, rooms[room_a]["single_room_layout"]["predicted_polygon_zind_local_xy"], "#1e6ecd"),
            (room_b, rooms[room_b]["single_room_layout"]["predicted_polygon_zind_local_xy"], "#8b5cf6"),
        ))
        joint_links = []
        if result["prediction"]["is_match"] and result["prediction"]["joint_bev_generated"]:
            prediction = result["prediction"]
            if prediction["portal_geometry_mode"] == "candidate_local_m2":
                representations = (
                    (
                        "frame_preserving",
                        prediction["joint_bev_frame_preserving_polygon"],
                    ),
                    ("pre_manhattan", prediction["joint_bev_pre_manhattan_polygon"]),
                )
            else:
                representations = (("predicted", pair_artifacts[pair_id]["native_joint"]),)
            for label, joint in representations:
                if joint is None:
                    continue
                svg_path = output_dir / "pairs" / f"{pair_id}_{label}_joint.svg"
                png_path = output_dir / "pairs" / f"{pair_id}_{label}_joint.png"
                render_joint_boundary_svg(dict(joint), str(svg_path), show_wall_indices=False)
                render_joint_boundary(dict(joint), str(png_path), show_wall_indices=False)
                joint_links.extend(
                    [svg_path.relative_to(output_dir), png_path.relative_to(output_dir)]
                )
            gt_interfaces = result["ground_truth"].get("shared_interface_details", [])
            gt_interface = gt_interfaces[0] if gt_interfaces else None
            for side, room_id in (("A", room_a), ("B", room_b)):
                portal = prediction.get(f"portal_geometry_{side}")
                if portal is None or not portal.get("geometry_valid"):
                    continue
                portal_path = output_dir / "pairs" / f"{pair_id}_portal_{side}.svg"
                gt_endpoints = (
                    None
                    if gt_interface is None
                    else gt_interface[f"local_endpoints_{side.lower()}"]
                )
                _render_portal_geometry_svg(
                    portal_path,
                    rooms[room_id]["single_room_layout"][
                        "layout_polygon_pre_manhattan_local_bi_xz"
                    ],
                    portal,
                    gt_endpoints,
                )
                joint_links.append(portal_path.relative_to(output_dir))
        if result["ground_truth"]["joint_bev_target"] is not None:
            target_svg = output_dir / "pairs" / f"{pair_id}_gt_target.svg"
            instances = result["ground_truth"]["joint_bev_target"]["instances"]
            _render_polygons_svg(target_svg, (
                (instances[0]["room_id"], instances[0]["polygon"], "#289128"),
                (instances[1]["room_id"], instances[1]["polygon"], "#e18a00"),
            ))
        pair_links.append((pair_id, pair_json.relative_to(output_dir), independent_svg.relative_to(output_dir), joint_links))

    rows = []
    for pair_id, json_path, independent, joint in pair_links:
        result = next(value for value in pair_results if value["prediction"]["pair_id"] == pair_id)
        joint_html = " ".join(
            f'<a href="{html.escape(str(path))}">{html.escape(path.name)}</a>' for path in joint
        ) or "none (predicted no-match or geometry failure)"
        rows.append(
            "<tr>"
            f"<td>{html.escape(pair_id)}</td>"
            f"<td>{result['ground_truth']['is_match']}</td>"
            f"<td>{result['prediction']['is_match']}</td>"
            f'<td><a href="{html.escape(str(json_path))}">json</a></td>'
            f'<td><a href="{html.escape(str(independent))}">independent</a></td>'
            f"<td>{joint_html}</td></tr>"
        )
    room_html = "".join(
        f'<li>{html.escape(room)}: <a href="{json_path}">json</a> '
        f'<a href="{svg_path}">layout</a></li>'
        for room, json_path, svg_path in room_links
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ZInD three-room debug</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.5rem}}</style></head>
<body><h1>ZInD three-room conditional fusion</h1><p>{html.escape(DISCLAIMER)}</p>
<h2>Independent rooms</h2><ul>{room_html}</ul>
<h2>Pairs</h2><table><tr><th>pair</th><th>GT</th><th>prediction</th><th>record</th><th>independent</th><th>predicted joint BEV</th></tr>
{''.join(rows)}</table></body></html>"""
    overview = output_dir / "overview.html"
    overview.write_text(document, encoding="utf-8")
    return {"overview_html": str(overview)}


def _summary(
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    pair_results: Sequence[Mapping[str, Any]],
    runtime_seconds: float,
    model_report: Mapping[str, Any],
) -> Dict[str, Any]:
    rooms = audit["schema"]["rooms"]
    predicted = np.zeros((3, 3), dtype=np.uint8)
    index = {room: value for value, room in enumerate(rooms)}
    for result in pair_results:
        prediction = result["prediction"]
        if prediction["is_match"]:
            a, b = index[prediction["room_a"]], index[prediction["room_b"]]
            predicted[a, b] = predicted[b, a] = 1
    tp = sum(bool(item["prediction"]["is_match"] and item["ground_truth"]["is_match"]) for item in pair_results)
    tn = sum(bool(not item["prediction"]["is_match"] and not item["ground_truth"]["is_match"]) for item in pair_results)
    fp = sum(bool(item["prediction"]["is_match"] and not item["ground_truth"]["is_match"]) for item in pair_results)
    fn = sum(bool(not item["prediction"]["is_match"] and item["ground_truth"]["is_match"]) for item in pair_results)
    return {
        "schema_version": "zind-room-group-evaluation-v1",
        "purpose": DISCLAIMER,
        "split": "train",
        "group_id": audit["group"]["group_id"],
        "house_id": audit["group"]["house_id"],
        "floor_id": audit["group"]["floor_id"],
        "room_ids": list(rooms),
        "room_count": 3,
        "pair_count": 3,
        "single_room_forward_count": {room: 1 for room in rooms},
        "gt_adjacency_matrix": copy.deepcopy(audit["group"]["adjacency_matrix"]),
        "predicted_adjacency_matrix": predicted.tolist(),
        "classification": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn},
        "predicted_joint_bev_count": sum(bool(item["prediction"]["joint_bev_generated"]) for item in pair_results),
        "portal_geometry_mode": args.portal_geometry_mode,
        "inference_contract": {
            "candidate_source": "predicted",
            "opening_guidance_mode": "predicted",
            "decision": "learned bidirectional dustbin",
            "gt_available_to_inference": False,
            "negative_fusion_called": False,
            "coordinate_convention": "ZInD panorama-A local xy; no GT post-alignment",
            "layout_frame_mode": args.layout_frame_mode,
            "portal_geometry_mode": args.portal_geometry_mode,
            "portal_geometry_runs_after_frozen_matcher_decision": True,
            "portal_geometry_gt_inputs": False,
            "candidate_local_hard_failure_fallback": "none",
            "layout_frame_for_cross_scene_pose": (
                "panorama_local"
                if args.layout_frame_mode == "panorama_preserving"
                else "manhattan_canonical_legacy"
            ),
            "heading_source": (
                "predicted pre-Manhattan geometry only"
                if args.layout_frame_mode == "panorama_preserving"
                else "none (legacy fixed axis)"
            ),
        },
        "test_data_accessed": False,
        "seed": int(args.seed),
        "runtime_seconds": runtime_seconds,
        "model": copy.deepcopy(dict(model_report)),
    }


def run(args: argparse.Namespace, audit: Mapping[str, Any]) -> Dict[str, Any]:
    import torch

    from portallayout.engine.cross_scene_pipeline import atomic_write_json

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(args.output_dir).expanduser().resolve()
    _reject_test_reference(output_dir, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    bi_layout, matcher, threshold, model_report = _load_models(args, audit, device)
    room_inputs = _room_inference_inputs(audit)
    rooms = _run_room_forward(
        args,
        room_inputs,
        audit["schema"]["rooms"],
        audit["image_shape"],
        bi_layout,
        matcher,
        device,
        float(threshold["value"]),
    )
    public_rooms = {
        room: _room_result_with_post_inference_gt(
            state, audit["annotations"][room], audit["schema"]["floor_scale"]
        )
        for room, state in rooms.items()
    }
    pair_results = []
    pair_artifacts = {}
    for edge in audit["schema"]["pair_order"]:
        key = _pair_key(*edge)
        pair_id = audit["group"]["pair_ids"][key]
        record = audit["records"][pair_id]
        prediction, artifacts = _run_pair_prediction(
            args, matcher, _pair_inference_input(record), rooms
        )
        prediction["group_pair"] = list(edge)
        ground_truth = strict_ground_truth_payload(
            record,
            audit["annotations"],
            audit["schema"]["floor_scale"],
            args.endpoint_tolerance,
        )
        metrics = _pair_metrics(
            prediction,
            ground_truth,
            artifacts,
            record,
            audit["annotations"],
            audit["schema"]["floor_scale"],
            args.endpoint_tolerance,
        )
        pair_results.append(
            attach_post_inference_evaluation(prediction, ground_truth, metrics)
        )
        pair_artifacts[pair_id] = artifacts
    runtime = time.perf_counter() - start
    summary = _summary(args, audit, pair_results, runtime, model_report)
    summary["opening_probability_threshold"] = dict(threshold)
    portal_geometry = {
        "schema_version": "zind-candidate-local-portal-geometry-v1",
        "split": "train",
        "group_id": audit["group"]["group_id"],
        "portal_geometry_mode": args.portal_geometry_mode,
        "gt_available_to_portal_geometry": False,
        "gt_usage": "post-inference visualization and metrics only",
        "pairs": {},
    }
    for result in pair_results:
        prediction = result["prediction"]
        gt_interfaces = result["ground_truth"].get("shared_interface_details", [])
        portal_geometry["pairs"][prediction["pair_id"]] = {
            "room_a": prediction["room_a"],
            "room_b": prediction["room_b"],
            "is_match": prediction["is_match"],
            "best_candidate_pair": prediction["best_candidate_pair"],
            "m2_called": bool(
                prediction["is_match"]
                and args.portal_geometry_mode == "candidate_local_m2"
            ),
            "portal_geometry_A": prediction.get("portal_geometry_A"),
            "portal_geometry_B": prediction.get("portal_geometry_B"),
            "entered_joint_layout": bool(prediction.get("joint_bev_generated")),
            "pose_valid": bool(prediction.get("pose_valid")),
            "posthoc_gt_portal": copy.deepcopy(gt_interfaces[0]) if gt_interfaces else None,
            "posthoc_endpoint_metrics_A": result['portallayout.evaluation'].get("portal_endpoint_A"),
            "posthoc_endpoint_metrics_B": result['portallayout.evaluation'].get("portal_endpoint_B"),
        }
    visualization = _render_outputs(
        output_dir, audit, rooms, public_rooms, pair_results, pair_artifacts
    )
    summary.update(visualization)
    summary["portal_geometry_json"] = str(output_dir / "portal_geometry.json")
    atomic_write_json(str(output_dir / "group_summary.json"), summary)
    atomic_write_json(str(output_dir / "rooms.json"), public_rooms)
    atomic_write_json(str(output_dir / "pairs.json"), {"pairs": pair_results})
    atomic_write_json(str(output_dir / "portal_geometry.json"), portal_geometry)
    atomic_write_json(
        str(output_dir / "adjacency_gt.json"),
        {
            "room_ids": list(audit["schema"]["rooms"]),
            "adjacency_matrix": summary["gt_adjacency_matrix"],
        },
    )
    atomic_write_json(
        str(output_dir / "adjacency_pred.json"),
        {
            "room_ids": list(audit["schema"]["rooms"]),
            "adjacency_matrix": summary["predicted_adjacency_matrix"],
        },
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    audit = load_and_validate_inputs(args)
    if args.metadata_only:
        print(json.dumps(metadata_report(audit), indent=2, ensure_ascii=False))
        return 0
    if args.dry_run:
        print(json.dumps(dry_run_report(args, audit), indent=2, ensure_ascii=False))
        return 0
    summary = run(args, audit)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
