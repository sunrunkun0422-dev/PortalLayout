#!/usr/bin/env python3
"""Val-only geometry preflight on one deterministic ZInD matching smoke set.

Each selected pair receives exactly one live Bi-Layout/Opening-Head/Matcher
prediction.  The frozen matcher result then feeds three analytical geometry
branches: legacy polygon mapper, frame-preserving polygon mapper, and the
production candidate-local M2 estimator.  Ground truth is attached only after
the frozen prediction and all three geometry branches have been serialized.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PURPOSE = "validation smoke test / engineering preflight; not a scientific final result"
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "src/dataset/ZInD_matching"
DEFAULT_VAL_MANIFEST = DEFAULT_MANIFEST_ROOT / "val.json"
DEFAULT_TRAIN_MANIFEST = DEFAULT_MANIFEST_ROOT / "train.json"
DEFAULT_VAL_CACHE = (
    REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_feature_cache/val"
)
DEFAULT_CONFIG = REPO_ROOT / "src/config/zind_all.yaml"
DEFAULT_BI_LAYOUT = (
    REPO_ROOT / "checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl"
)
DEFAULT_OPENING = REPO_ROOT / "checkpoints/Opening_Head/zind_matching_v1/best.pt"
DEFAULT_MATCHER = (
    REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_v1/best.pt"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "src/output/zind_geometry_val_smoke/t3_seed123_64"
GEOMETRY_MODES = (
    "legacy_polygon",
    "frame_polygon",
    "candidate_local_m2",
)
SOFT_GATE_THRESHOLDS = {
    "support_count_min": 4,
    "inlier_ratio_min": 0.60,
    "normalized_residual_max": 0.05,
    "angular_dispersion_degrees_max": 25.0,
    "ray_condition_min": 0.05,
}
POSE_THRESHOLDS = (
    (10.0, 0.5),
    (15.0, 1.0),
    (30.0, 2.0),
)
UNIT_SCALE_TOLERANCE = 0.05
PAIRED_TIE_EPSILON = 1e-9


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--val_manifest", default=str(DEFAULT_VAL_MANIFEST))
    parser.add_argument("--train_manifest", default=str(DEFAULT_TRAIN_MANIFEST))
    parser.add_argument("--val_cache", default=str(DEFAULT_VAL_CACHE))
    parser.add_argument("--data_root")
    parser.add_argument("--sample_manifest")
    parser.add_argument(
        "--geometry_modes",
        default=",".join(GEOMETRY_MODES),
        help="must contain legacy_polygon,frame_polygon,candidate_local_m2 exactly once",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--bi_layout_config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bi_layout_checkpoint", default=str(DEFAULT_BI_LAYOUT))
    parser.add_argument("--opening_checkpoint", default=str(DEFAULT_OPENING))
    parser.add_argument("--matcher_checkpoint", default=str(DEFAULT_MATCHER))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--positive_count", type=int, default=32)
    parser.add_argument("--negative_count", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--min_opening_width_tokens", type=int, default=2)
    parser.add_argument("--max_openings_per_view", type=int, default=12)
    parser.add_argument("--opening_probability_threshold", type=float)
    parser.add_argument("--endpoint_tolerance", type=float, default=0.06)
    parser.add_argument("--prepare_sample", action="store_true")
    parser.add_argument("--metadata_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if args.sample_manifest is None:
        args.sample_manifest = str(output / "sample_manifest.json")
    args.geometry_modes = tuple(
        value.strip() for value in args.geometry_modes.split(",") if value.strip()
    )
    return args


def reject_test_path(path: Path, label: str) -> None:
    value = path.expanduser().resolve()
    if value.stem.lower() == "test" or any(part.lower() == "test" for part in value.parts):
        raise ValueError(f"{label} points to forbidden test data: {value}")


def _resolved_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    reject_test_path(path, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _resolved_directory(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    reject_test_path(path, label)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _jsonable(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            _jsonable(payload),
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file.write("\n")
    os.replace(str(temporary), str(path))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_args(args: argparse.Namespace) -> None:
    if args.split != "val":
        raise ValueError("this evaluator is val-only")
    if tuple(args.geometry_modes) != GEOMETRY_MODES:
        raise ValueError(
            "geometry_modes must be exactly " + ",".join(GEOMETRY_MODES)
        )
    if args.seed != 123 or args.positive_count != 32 or args.negative_count != 32:
        raise ValueError("this smoke contract is fixed to seed=123 and 32+32 pairs")
    if min(
        args.positive_count,
        args.negative_count,
        args.torch_threads,
        args.min_opening_width_tokens,
        args.max_openings_per_view,
    ) <= 0:
        raise ValueError("count/thread/opening settings must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if not math.isfinite(args.endpoint_tolerance) or args.endpoint_tolerance <= 0:
        raise ValueError("endpoint_tolerance must be positive and finite")
    if args.opening_probability_threshold is not None and not (
        0.0 <= args.opening_probability_threshold <= 1.0
    ):
        raise ValueError("opening_probability_threshold must be in [0,1]")
    for value, label in (
        (args.val_manifest, "val manifest"),
        (args.train_manifest, "train manifest"),
        (args.val_cache, "val cache"),
        (args.sample_manifest, "sample manifest"),
        (args.checkpoint, "checkpoint"),
        (args.output_dir, "output directory"),
    ):
        reject_test_path(Path(value), label)


def _load_split_manifest(path: Path, expected_split: str) -> Dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, Mapping) or value.get("split") != expected_split:
        raise ValueError(f"manifest must declare split={expected_split}: {path}")
    records = value.get("pairs")
    if not isinstance(records, list) or not records:
        raise ValueError(f"manifest contains no pairs: {path}")
    ids = [str(record.get("id", "")) for record in records]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError(f"manifest pair IDs are missing or duplicated: {path}")
    return dict(value)


def _house_first_select(
    records: Sequence[Mapping[str, Any]],
    label: str,
    count: int,
    seed: int,
) -> List[Mapping[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda value: str(value["id"])):
        grouped[str(record["scene_id"])].append(record)
    houses = sorted(
        grouped,
        key=lambda house: (
            stable_hash(f"{seed}:{label}:house:{house}"),
            house,
        ),
    )
    selected: List[Mapping[str, Any]] = []
    depth = 0
    while len(selected) < count:
        added = False
        for house in houses:
            if depth < len(grouped[house]):
                selected.append(grouped[house][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != count:
        raise ValueError(f"val manifest does not contain {count} {label} pairs")
    return selected


def build_sample_manifest(
    val_manifest: Mapping[str, Any],
    val_manifest_path: Path,
    seed: int = 123,
    positive_count: int = 32,
    negative_count: int = 32,
) -> Dict[str, Any]:
    records = val_manifest["pairs"]
    positives = [value for value in records if bool(value["supervision"]["is_match"])]
    negatives = [value for value in records if not bool(value["supervision"]["is_match"])]
    selected = _house_first_select(positives, "positive", positive_count, seed)
    selected += _house_first_select(negatives, "negative", negative_count, seed)
    selected.sort(
        key=lambda value: (
            stable_hash(f"{seed}:order:{value['id']}"),
            str(value["id"]),
        )
    )
    selected_positive = [value for value in selected if value["supervision"]["is_match"]]
    interface_types = Counter()
    for record in selected_positive:
        supervision = record["supervision"]
        first = record["candidates_a"][int(supervision["target_candidate_a"])]["type"]
        second = record["candidates_b"][int(supervision["target_candidate_b"])]["type"]
        interface_types[f"{first}-{second}"] += 1
    houses = Counter(str(value["scene_id"]) for value in selected)
    return {
        "schema_version": "zind-val-geometry-smoke-v1",
        "split": "val",
        "purpose": PURPOSE,
        "source_manifest": {
            "path": str(val_manifest_path),
            "sha256": sha256_file(val_manifest_path),
        },
        "selection": {
            "seed": int(seed),
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "algorithm": (
                "GT-label stratification; stable pair-ID sort within each house; "
                "seeded-hash house order; house-first round-robin; seeded-hash final order"
            ),
            "model_outputs_used": False,
            "geometry_or_quality_used": False,
        },
        "pair_ids": [str(value["id"]) for value in selected],
        "counts": {
            "pair_count": len(selected),
            "positive": len(selected_positive),
            "negative": len(selected) - len(selected_positive),
            "house_count": len(houses),
            "positive_house_count": len(
                {str(value["scene_id"]) for value in selected_positive}
            ),
            "negative_house_count": len(
                {
                    str(value["scene_id"])
                    for value in selected
                    if not value["supervision"]["is_match"]
                }
            ),
        },
        "house_pair_counts": dict(sorted(houses.items())),
        "positive_interface_types": dict(sorted(interface_types.items())),
        "test_data_accessed": False,
    }


def validate_sample_manifest(
    sample: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if dict(sample) != dict(expected):
        raise ValueError(
            "sample manifest does not match deterministic seed-123 selection"
        )
    if sample.get("split") != "val" or sample.get("purpose") != PURPOSE:
        raise ValueError("sample manifest has the wrong val smoke contract")
    ids = sample.get("pair_ids")
    if not isinstance(ids, list) or len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError("sample manifest must contain 64 unique pair IDs")


def checkpoint_contract(checkpoint_path: Path, val_manifest_path: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("task") != (
        "ZInD_matching full end-to-end joint fine-tuning"
    ):
        raise ValueError("checkpoint has the wrong end-to-end task")
    contract = payload.get("data_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint has no data_contract")
    if "val" not in contract.get("active_splits", []):
        raise ValueError("checkpoint data contract does not allow val")
    if contract.get("test_evaluation_enabled") is not False:
        raise ValueError("checkpoint unexpectedly enables test evaluation")
    manifests = contract.get("manifests")
    if not isinstance(manifests, Mapping) or "test" in manifests:
        raise ValueError("checkpoint data contract exposes a test manifest")
    saved_val = Path(str(manifests.get("val", ""))).expanduser().resolve()
    if saved_val != val_manifest_path:
        raise ValueError("checkpoint val manifest differs from requested val manifest")
    output = {
        "task": payload.get("task"),
        "format_version": payload.get("format_version"),
        "completed_epoch": payload.get("completed_epoch"),
        "opening_probability_threshold": payload.get(
            "opening_probability_threshold"
        ),
        "data_contract": copy.deepcopy(dict(contract)),
        "allows_val": True,
        "test_evaluation_enabled": False,
    }
    del payload
    return output


def load_context(args: argparse.Namespace, require_sample: bool = True) -> Dict[str, Any]:
    validate_args(args)
    val_path = _resolved_file(args.val_manifest, "val manifest")
    train_path = _resolved_file(args.train_manifest, "train manifest")
    checkpoint_path = _resolved_file(args.checkpoint, "T3 checkpoint")
    cache_dir = _resolved_directory(args.val_cache, "val feature cache")
    output_dir = Path(args.output_dir).expanduser().resolve()
    reject_test_path(output_dir, "output directory")
    val = _load_split_manifest(val_path, "val")
    train = _load_split_manifest(train_path, "train")
    train_houses = {str(value) for value in train.get("sourceHouseIds", [])}
    val_houses = {str(value) for value in val.get("sourceHouseIds", [])}
    overlap = sorted(train_houses & val_houses)
    if overlap:
        raise ValueError("house leakage between train and val: " + ", ".join(overlap[:3]))
    cache_metadata_path = cache_dir / "metadata.json"
    if not cache_metadata_path.is_file():
        raise FileNotFoundError(cache_metadata_path)
    cache_metadata = load_json(cache_metadata_path)
    if cache_metadata.get("split") != "val" or cache_metadata.get("complete") is not True:
        raise ValueError("val cache is incomplete or declares the wrong split")
    cached_manifest = Path(
        str(cache_metadata.get("manifest", {}).get("path", ""))
    ).expanduser().resolve()
    if cached_manifest != val_path:
        raise ValueError("val cache was not built from the requested val manifest")
    expected_sample = build_sample_manifest(
        val,
        val_path,
        seed=args.seed,
        positive_count=args.positive_count,
        negative_count=args.negative_count,
    )
    sample_path = Path(args.sample_manifest).expanduser().resolve()
    reject_test_path(sample_path, "sample manifest")
    sample = None
    if sample_path.is_file():
        sample = load_json(sample_path)
        validate_sample_manifest(sample, expected_sample)
    elif require_sample:
        raise FileNotFoundError(
            f"sample manifest does not exist; run --prepare_sample first: {sample_path}"
        )
    contract = checkpoint_contract(checkpoint_path, val_path)
    records_by_id = {str(value["id"]): value for value in val["pairs"]}
    checkpoint_sha256 = sha256_file(checkpoint_path) if require_sample else None
    return {
        "args": args,
        "val_manifest_path": val_path,
        "train_manifest_path": train_path,
        "checkpoint_path": checkpoint_path,
        "cache_dir": cache_dir,
        "cache_metadata": cache_metadata,
        "output_dir": output_dir,
        "sample_path": sample_path,
        "sample": sample,
        "expected_sample": expected_sample,
        "val_manifest": val,
        "records_by_id": records_by_id,
        "checkpoint_contract": contract,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_sha256": sha256_file(sample_path) if sample is not None else None,
        "val_manifest_sha256": expected_sample["source_manifest"]["sha256"],
        "house_split": {
            "train_house_count": len(train_houses),
            "val_house_count": len(val_houses),
            "train_val_overlap_count": 0,
        },
        "test_data_accessed": False,
    }


def metadata_report(context: Mapping[str, Any]) -> Dict[str, Any]:
    sample = context["sample"]
    return {
        "status": "PASS",
        "mode": "metadata-only",
        "split": "val",
        "purpose": PURPOSE,
        "sample_manifest": str(context["sample_path"]),
        "sample_counts": copy.deepcopy(sample["counts"]),
        "positive_interface_types": copy.deepcopy(
            sample["positive_interface_types"]
        ),
        "house_split": copy.deepcopy(context["house_split"]),
        "checkpoint": {
            "path": str(context["checkpoint_path"]),
            "sha256": context["checkpoint_sha256"],
            **copy.deepcopy(context["checkpoint_contract"]),
        },
        "val_cache": {
            "path": str(context["cache_dir"]),
            "split": context["cache_metadata"].get("split"),
            "complete": context["cache_metadata"].get("complete"),
        },
        "geometry_modes": list(GEOMETRY_MODES),
        "model_loaded": False,
        "gpu_used": False,
        "test_data_accessed": False,
    }


def _m2_contract() -> Dict[str, Any]:
    from utils.candidate_local_portal_geometry import CandidateLocalPortalConfig

    config = CandidateLocalPortalConfig().to_json()
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return {
        "config": config,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "source": "utils.candidate_local_portal_geometry.CandidateLocalPortalConfig",
        "tuned_on_val": False,
    }


def dry_run_report(context: Mapping[str, Any]) -> Dict[str, Any]:
    import torch
    from dataset.zind_bipair_dataset import collate_zind_bipair
    from tools.train_zind_end_to_end import JointZInDMatchingDataset
    from torch.utils.data import DataLoader, Subset

    args = context["args"]
    dataset = JointZInDMatchingDataset(
        str(context["val_manifest_path"]),
        str(context["cache_dir"]),
        data_root=args.data_root,
        max_pairs=None,
        expected_split="val",
        image_shape=tuple(context["cache_metadata"].get("image_shape", (512, 1024))),
        rotation_augmentation=False,
    )
    indices_by_id = {
        str(record["id"]): index for index, record in enumerate(dataset.base.records)
    }
    indices = [indices_by_id[value] for value in context["sample"]["pair_ids"]]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_zind_bipair,
        drop_last=False,
    )
    observed_ids: List[str] = []
    image_shapes = set()
    all_images_finite = True
    for batch in loader:
        if len(batch["pair_id"]) != 1:
            raise ValueError("dry-run DataLoader must preserve batch_size=1")
        observed_ids.append(str(batch["pair_id"][0]))
        for key in ("image_A", "image_B"):
            image_shapes.add(tuple(int(value) for value in batch[key].shape))
            all_images_finite = all_images_finite and bool(
                torch.isfinite(batch[key]).all().item()
            )
    expected_ids = list(context["sample"]["pair_ids"])
    if observed_ids != expected_ids:
        raise ValueError("dry-run DataLoader changed deterministic sample order")
    if not all_images_finite:
        raise ValueError("dry-run found NaN/Inf in a selected panorama tensor")
    return {
        **metadata_report(context),
        "mode": "dry-run",
        "formal_dataset": "JointZInDMatchingDataset",
        "formal_dataloader": "torch.utils.data.DataLoader + collate_zind_bipair",
        "dataset_split": dataset.split,
        "dataset_pair_count": len(dataset),
        "selected_pair_count": len(indices),
        "loader_batch_count": len(loader),
        "validated_pair_count": len(observed_ids),
        "batch_pair_order_matches_sample": True,
        "image_tensor_shapes": [list(value) for value in sorted(image_shapes)],
        "all_images_finite": True,
        "model_forward_count": 0,
        "rotation_augmentation": False,
        "live_forward_planned": True,
        "one_backbone_call_per_unfinished_pair": True,
        "geometry_fanout_after_frozen_prediction": list(GEOMETRY_MODES),
        "m2": _m2_contract(),
    }


class AnnotationResolver:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._houses: Dict[str, Mapping[str, Any]] = {}

    def pair_ground_truth(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        from dataset.zind_pair_mining import transform_points

        house = str(record["scene_id"])
        annotation_path = (self.data_root / house / "zind_data.json").resolve()
        reject_test_path(annotation_path, "val annotation")
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        payload = self._houses.get(house)
        if payload is None:
            payload = load_json(annotation_path)
            self._houses[house] = payload
        floor = str(record["floor_id"])

        def panorama(side: str) -> Mapping[str, Any]:
            room = str(record[f"room_{side}"])
            partial = str(record[f"partial_room_{side}"])
            pano = str(record[f"pano_{side}"])
            value = payload["merger"][floor][room][partial][pano]
            expected = str(record[f"image_{side}"])
            if value.get("image_path") != expected.split(f"{house}/", 1)[-1]:
                raise ValueError(f"annotation image mapping changed for {record['id']}:{side}")
            return value

        pano_a, pano_b = panorama("a"), panorama("b")
        polygon_a = np.asarray(pano_a["layout_complete"]["vertices"], dtype=np.float64)
        polygon_b = np.asarray(pano_b["layout_complete"]["vertices"], dtype=np.float64)
        floor_scale_value = payload["scale_meters_per_coordinate"].get(floor)
        floor_scale = (
            None if floor_scale_value is None else float(floor_scale_value)
        )
        if floor_scale is not None and (
            not math.isfinite(floor_scale) or floor_scale <= 0.0
        ):
            raise ValueError(f"invalid floor metric scale for {record['id']}")
        metric_factor_a = (
            None
            if floor_scale is None
            else float(pano_a["floor_plan_transformation"]["scale"])
            * floor_scale
        )
        supervision = record["supervision"]
        is_match = bool(supervision["is_match"])
        output: Dict[str, Any] = {
            "is_match": is_match,
            "relative_pose": None,
            "joint_bev_target": None,
            "target_candidate_pair": [
                int(supervision.get("target_candidate_a", -1)),
                int(supervision.get("target_candidate_b", -1)),
            ],
            "interface_types": None,
            "floor_scale_meters_per_coordinate": floor_scale,
            "metric_factor_A": metric_factor_a,
            "metric_translation_available": metric_factor_a is not None,
            "polygon_complete_A": polygon_a.tolist(),
            "polygon_complete_B": polygon_b.tolist(),
            "gt_scale_is_approximately_unit": None,
        }
        if not is_match:
            return output
        transform = np.asarray(
            supervision["relative_transform_b_to_a"], dtype=np.float64
        )
        scale = float(math.hypot(transform[0, 0], transform[1, 0]))
        yaw = float(math.atan2(transform[1, 0], transform[0, 0]))
        target_a, target_b = output["target_candidate_pair"]
        candidate_a = record["candidates_a"][target_a]
        candidate_b = record["candidates_b"][target_b]
        output.update({
            "relative_pose": {
                "T_B_to_A": transform.tolist(),
                "relative_yaw_radians": yaw,
                "relative_yaw_degrees": math.degrees(yaw),
                "translation": transform[:2, 2].tolist(),
                "relative_scale": scale,
            },
            "joint_bev_target": {
                "instances": [
                    {"room_id": record["room_a"], "polygon": polygon_a.tolist()},
                    {
                        "room_id": record["room_b"],
                        "polygon": transform_points(polygon_b, transform).tolist(),
                    },
                ]
            },
            "interface_types": [candidate_a["type"], candidate_b["type"]],
            "target_portal_endpoints_A": copy.deepcopy(
                candidate_a["localEndpoints"]
            ),
            "target_portal_endpoints_B": copy.deepcopy(
                candidate_b["localEndpoints"]
            ),
            "target_interval_wrap_A": bool(
                candidate_a["tokenInterval"][0] > candidate_a["tokenInterval"][1]
            ),
            "target_interval_wrap_B": bool(
                candidate_b["tokenInterval"][0] > candidate_b["tokenInterval"][1]
            ),
            "gt_scale_is_approximately_unit": abs(scale - 1.0)
            <= UNIT_SCALE_TOLERANCE,
        })
        return output


def _data_root(context: Mapping[str, Any]) -> Path:
    args = context["args"]
    value = args.data_root or context["val_manifest"].get("dataRoot")
    if not value:
        raise ValueError("val manifest has no dataRoot and --data_root was omitted")
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = context["val_manifest_path"].parent / root
    root = root.resolve()
    reject_test_path(root, "val data_root")
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _load_models(args: argparse.Namespace, device: Any):
    from types import SimpleNamespace
    from tools.evaluate_zind_room_group import _load_models as load_room_group_models

    model_args = SimpleNamespace(
        config=args.bi_layout_config,
        bi_layout_checkpoint=args.bi_layout_checkpoint,
        opening_checkpoint=args.opening_checkpoint,
        matcher_checkpoint=args.matcher_checkpoint,
        end_to_end_checkpoint=args.checkpoint,
        opening_probability_threshold=args.opening_probability_threshold,
    )
    return load_room_group_models(model_args, {}, device)


def _layout_state(
    depth: Any,
    ratio: Any,
    probability: Any,
    intervals: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    from tools.debug_cross_scene_flow import prediction_to_layout
    from utils.cross_scene_estimator import opening_candidates_from_intervals
    from utils.joint_layout import layout_xz
    from tools.evaluate_zind_room_group import _bi_points_to_zind

    legacy_layout, legacy_report = prediction_to_layout(
        depth, ratio, layout_frame_mode="legacy_fixed_axis"
    )
    frame_layout, frame_report = prediction_to_layout(
        depth, ratio, layout_frame_mode="panorama_preserving"
    )
    legacy_candidates = opening_candidates_from_intervals(
        legacy_layout, intervals, probability
    )
    frame_candidates = opening_candidates_from_intervals(
        frame_layout, intervals, probability
    )
    if len(legacy_candidates) != len(intervals) or len(frame_candidates) != len(
        intervals
    ):
        raise RuntimeError("polygon mapper changed predicted candidate order")
    raw = frame_report.get("layoutPolygonPreManhattanLocalBiXZ")
    if raw is None:
        raise ValueError("panorama-preserving postprocess omitted pre-Manhattan contour")
    return {
        "legacy_layout": legacy_layout,
        "legacy_report": legacy_report,
        "frame_layout": frame_layout,
        "frame_report": frame_report,
        "legacy_candidates": [value.to_json() for value in legacy_candidates],
        "frame_candidates": [value.to_json() for value in frame_candidates],
        "legacy_polygon_zind_xy": _bi_points_to_zind(layout_xz(legacy_layout)).tolist(),
        "frame_polygon_zind_xy": _bi_points_to_zind(layout_xz(frame_layout)).tolist(),
        "pre_manhattan_contour_bi_xz": copy.deepcopy(raw),
        "pre_manhattan_contour_zind_xy": _bi_points_to_zind(raw).tolist(),
    }


def infer_frozen_prediction(
    args: argparse.Namespace,
    image_a: Any,
    image_b: Any,
    identifiers: Mapping[str, Any],
    bi_layout: Any,
    matcher: Any,
    device: Any,
    opening_threshold: float,
) -> Dict[str, Any]:
    """One live forward and one matcher decision; accepts no GT argument."""
    import torch
    from models.cross_scene_matcher import (
        opening_probabilities_to_intervals,
        resolve_enclosed_extended_depth,
    )
    from tools.train_zind_cross_scene_matcher import predicted_candidate_masks

    images = torch.cat((image_a, image_b), dim=0).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    with torch.no_grad():
        outputs = bi_layout(images, return_features=True)
        enclosed, extended = resolve_enclosed_extended_depth(outputs, "extended_first")
        opening = matcher.opening_head(outputs["layout_feature"], enclosed, extended)
        probability = opening["opening_probability"]
        masks, valid = predicted_candidate_masks(
            probability,
            opening_threshold,
            args.min_opening_width_tokens,
            args.max_openings_per_view,
        )
        matcher_output = matcher(
            outputs["layout_feature"][0:1],
            outputs["layout_feature"][1:2],
            enclosed[0:1],
            extended[0:1],
            enclosed[1:2],
            extended[1:2],
            candidate_masks_a=masks[0:1],
            candidate_masks_b=masks[1:2],
            candidate_valid_a=valid[0:1],
            candidate_valid_b=valid[1:2],
            opening_guidance_a=None,
            opening_guidance_b=None,
            opening_guidance_mode="predicted",
        )
    intervals = [
        opening_probabilities_to_intervals(
            probability[side],
            threshold=opening_threshold,
            min_width_tokens=args.min_opening_width_tokens,
            max_intervals=args.max_openings_per_view,
        )
        for side in range(2)
    ]
    if int(valid[0].sum().item()) != len(intervals[0]) or int(
        valid[1].sum().item()
    ) != len(intervals[1]):
        raise RuntimeError("candidate mask order diverged from predicted intervals")
    best = [int(value) for value in matcher_output["best_candidate_pair"][0].tolist()]
    frozen: Dict[str, Any] = {
        "schema_version": "zind-val-frozen-prediction-v1",
        "pair_id": str(identifiers["pair_id"]),
        "house_id": str(identifiers["house_id"]),
        "floor_id": str(identifiers["floor_id"]),
        "room_a": str(identifiers["room_a"]),
        "room_b": str(identifiers["room_b"]),
        "pano_a": str(identifiers["pano_a"]),
        "pano_b": str(identifiers["pano_b"]),
        "model_forward_count": 1,
        "opening_threshold": float(opening_threshold),
        "opening_probability_A": probability[0].detach().cpu().tolist(),
        "opening_probability_B": probability[1].detach().cpu().tolist(),
        "predicted_intervals_A": [list(value) for value in intervals[0]],
        "predicted_intervals_B": [list(value) for value in intervals[1]],
        "candidate_count_A": int(valid[0].sum().item()),
        "candidate_count_B": int(valid[1].sum().item()),
        "best_candidate_pair": best,
        "is_match": bool(best[0] >= 0 and best[1] >= 0),
        "candidate_pair_scores": matcher_output["candidate_pair_score"][0]
        .detach()
        .cpu()
        .tolist(),
        "pair_match_probability": float(
            matcher_output["pair_match_probability"][0].item()
        ),
        "pair_no_match_probability": float(
            matcher_output["pair_no_match_probability"][0].item()
        ),
        "layout_A": None,
        "layout_B": None,
        "layout_failure_A": None,
        "layout_failure_B": None,
        "gt_available_to_prediction": False,
        "test_data_accessed": False,
    }
    for side, index in (("A", 0), ("B", 1)):
        try:
            frozen[f"layout_{side}"] = _layout_state(
                enclosed[index],
                outputs["ratio"][index],
                probability[index],
                intervals[index],
            )
        except Exception as exc:
            frozen[f"layout_failure_{side}"] = str(exc)
    return frozen


def _opening_candidate_from_json(value: Mapping[str, Any]):
    from utils.cross_scene_estimator import OpeningCandidate

    return OpeningCandidate(
        wall_index=int(value["wallIndex"]),
        start_ratio=float(value["startRatio"]),
        end_ratio=float(value["endRatio"]),
        confidence=float(value["confidence"]),
        token_start=int(value["tokenStart"]),
        token_end=int(value["tokenEnd"]),
        source=str(value["source"]),
        candidate_index=int(value["candidateIndex"]),
        metrics=dict(value.get("metrics", {})),
    )


def _empty_geometry(mode: str, reason: Optional[str] = None) -> Dict[str, Any]:
    return {
        "mode": mode,
        "attempted": False,
        "pose_valid": False,
        "hard_failure": False,
        "failure_reason": reason,
        "relative_pose": None,
        "joint_bev_frame_preserving_polygon": None,
        "joint_bev_pre_manhattan_polygon": None,
        "portal_geometry_A": None,
        "portal_geometry_B": None,
    }


def _polygon_geometry(
    frozen: Mapping[str, Any], mode: str
) -> Dict[str, Any]:
    from tools.evaluate_zind_room_group import (
        _joint_with_display_boundaries,
        _matcher_evidence,
        _pose,
        _standardize_joint,
    )
    from utils.cross_scene_pipeline import CrossScenePipeline, CrossScenePipelineConfig

    if not frozen["is_match"]:
        return _empty_geometry(mode)
    index_a, index_b = frozen["best_candidate_pair"]
    layout_a_state, layout_b_state = frozen["layout_A"], frozen["layout_B"]
    if layout_a_state is None or layout_b_state is None:
        reason = "layout postprocess failure: " + "; ".join(
            value
            for value in (
                frozen.get("layout_failure_A"),
                frozen.get("layout_failure_B"),
            )
            if value
        )
        output = _empty_geometry(mode, reason)
        output.update(attempted=True, hard_failure=True)
        return output
    key = "legacy" if mode == "legacy_polygon" else "frame"
    layout_a = layout_a_state[f"{key}_layout"]
    layout_b = layout_b_state[f"{key}_layout"]
    candidates_a = layout_a_state[f"{key}_candidates"]
    candidates_b = layout_b_state[f"{key}_candidates"]
    try:
        selected_a = _opening_candidate_from_json(candidates_a[index_a])
        selected_b = _opening_candidate_from_json(candidates_b[index_b])
        score = float(frozen["candidate_pair_scores"][index_a][index_b])
        pipeline = CrossScenePipeline(
            CrossScenePipelineConfig(
                top_k=1,
                use_passability=True,
                feature_weight=1.0,
                strict_polygon_validation=True,
            )
        )
        result = pipeline.run(
            layout_a,
            layout_b,
            match_evidence=_matcher_evidence({}, score),
            openings_a=[replace(selected_a, candidate_index=0)],
            openings_b=[replace(selected_b, candidate_index=0)],
        )
        standardized = _standardize_joint(
            result.best_joint_layout, frozen["room_a"], frozen["room_b"]
        )
        frame_joint = _joint_with_display_boundaries(
            standardized,
            layout_a_state["frame_polygon_zind_xy"],
            layout_b_state["frame_polygon_zind_xy"],
            "frame_preserving_manhattan_panorama_local",
        )
        raw_joint = _joint_with_display_boundaries(
            standardized,
            layout_a_state["pre_manhattan_contour_zind_xy"],
            layout_b_state["pre_manhattan_contour_zind_xy"],
            "pre_manhattan_panorama_local",
        )
        pose = _pose(np.asarray(standardized["alignment"]["roomBToWorld"]))
        return {
            "mode": mode,
            "attempted": True,
            "pose_valid": True,
            "hard_failure": False,
            "failure_reason": None,
            "relative_pose": pose,
            "joint_bev_frame_preserving_polygon": frame_joint,
            "joint_bev_pre_manhattan_polygon": raw_joint,
            "portal_geometry_A": None,
            "portal_geometry_B": None,
        }
    except Exception as exc:
        output = _empty_geometry(mode, str(exc))
        output.update(attempted=True, hard_failure=True)
        return output


def _candidate_local_geometry(frozen: Mapping[str, Any]) -> Dict[str, Any]:
    from tools.evaluate_zind_room_group import _candidate_local_pair_fusion
    from utils.cross_scene_estimator import opening_candidates_from_intervals

    mode = "candidate_local_m2"
    if not frozen["is_match"]:
        return _empty_geometry(mode)
    if frozen["layout_A"] is None or frozen["layout_B"] is None:
        output = _empty_geometry(mode, "layout postprocess failure")
        output.update(attempted=True, hard_failure=True)
        return output

    def state(side: str) -> Dict[str, Any]:
        layout = frozen[f"layout_{side}"]
        probability = frozen[f"opening_probability_{side}"]
        intervals = frozen[f"predicted_intervals_{side}"]
        candidates = opening_candidates_from_intervals(
            layout["frame_layout"], intervals, probability
        )
        return {
            "room_id": frozen[f"room_{side.lower()}"],
            "single_room_layout": {
                "predicted_layout_bi_xz": layout["frame_layout"],
                "predicted_polygon_zind_local_xy": layout[
                    "frame_polygon_zind_xy"
                ],
                "layout_polygon_pre_manhattan_local_bi_xz": layout[
                    "pre_manhattan_contour_bi_xz"
                ],
            },
            "opening_prediction": {
                "probability": probability,
                "intervals": intervals,
            },
            "_opening_candidates": candidates,
        }

    index_a, index_b = frozen["best_candidate_pair"]
    score = float(frozen["candidate_pair_scores"][index_a][index_b])
    detail = [{
        "candidate_a": index_a,
        "candidate_b": index_b,
        "candidate_pair_score": score,
    }]
    try:
        payload, _ = _candidate_local_pair_fusion(
            state("A"), state("B"), index_a, index_b, detail
        )
        return {
            "mode": mode,
            "attempted": True,
            "pose_valid": bool(payload["pose_valid"]),
            "hard_failure": not bool(payload["pose_valid"]),
            "failure_reason": payload.get("failure_reason"),
            "relative_pose": payload.get("relative_pose"),
            "joint_bev_frame_preserving_polygon": payload.get(
                "joint_bev_frame_preserving_polygon"
            ),
            "joint_bev_pre_manhattan_polygon": payload.get(
                "joint_bev_pre_manhattan_polygon"
            ),
            "portal_geometry_A": payload.get("portal_geometry_A"),
            "portal_geometry_B": payload.get("portal_geometry_B"),
        }
    except Exception as exc:
        output = _empty_geometry(mode, str(exc))
        output.update(attempted=True, hard_failure=True)
        return output


def geometry_fanout(frozen: Mapping[str, Any]) -> Dict[str, Any]:
    """Run all modes from one frozen prediction; accepts no GT argument."""
    modes = {
        "legacy_polygon": _polygon_geometry(frozen, "legacy_polygon"),
        "frame_polygon": _polygon_geometry(frozen, "frame_polygon"),
        "candidate_local_m2": _candidate_local_geometry(frozen),
    }
    signatures = {
        mode: {
            "best_candidate_pair": list(frozen["best_candidate_pair"]),
            "candidate_pair_scores": copy.deepcopy(frozen["candidate_pair_scores"]),
            "pair_match_probability": frozen["pair_match_probability"],
            "pair_no_match_probability": frozen["pair_no_match_probability"],
        }
        for mode in GEOMETRY_MODES
    }
    arrays = [
        np.asarray(value["candidate_pair_scores"], dtype=np.float64)
        for value in signatures.values()
    ]
    maximum = max(
        float(np.max(np.abs(arrays[0] - value))) if arrays[0].size else 0.0
        for value in arrays[1:]
    )
    return {
        "modes": modes,
        "matcher_signature_by_mode": signatures,
        "matcher_score_max_abs_diff": maximum,
        "best_candidate_identical": len(
            {tuple(value["best_candidate_pair"]) for value in signatures.values()}
        )
        == 1,
        "gt_available_to_geometry": False,
    }


def _candidate_evaluation(frozen: Mapping[str, Any], batch: Mapping[str, Any]) -> Dict[str, Any]:
    import torch
    from models.cross_scene_matcher import candidate_intervals_to_mask
    from tools.train_zind_cross_scene_matcher import candidate_targets_from_affinity

    count = int(batch["affinity_gt"].shape[-1])

    def masks(side: str) -> torch.Tensor:
        intervals = frozen[f"predicted_intervals_{side}"]
        value = candidate_intervals_to_mask(intervals, count)
        return value.unsqueeze(0)

    masks_a, masks_b = masks("A"), masks("B")
    affinity = batch["affinity_gt"].to(dtype=torch.float32)
    targets = candidate_targets_from_affinity(masks_a, affinity, masks_b)[0].bool()
    recalled = bool(targets.any().item())
    best_a, best_b = frozen["best_candidate_pair"]
    selected_correct = bool(
        best_a >= 0
        and best_b >= 0
        and best_a < targets.shape[0]
        and best_b < targets.shape[1]
        and targets[best_a, best_b].item()
    )
    rank = None
    if recalled and targets.numel():
        scores = np.asarray(frozen["candidate_pair_scores"], dtype=np.float64)
        target_indices = torch.nonzero(targets, as_tuple=False).tolist()
        ranks = []
        for first, second in target_indices:
            score = scores[first, second]
            ranks.append(1 + int(np.count_nonzero(scores > score)))
        rank = min(ranks)
    return {
        "gt_candidate_recalled": recalled,
        "selected_candidate_correct": selected_correct,
        "gt_candidate_rank": rank,
        "target_predicted_candidate_pairs": torch.nonzero(
            targets, as_tuple=False
        ).tolist(),
    }


def _wrapped_angle_error(first: float, second: float) -> float:
    return float(abs((first - second + math.pi) % (2.0 * math.pi) - math.pi))


def _safe_overlap(predicted: Any, target: Any) -> Optional[float]:
    from utils.cross_scene_estimator import polygon_overlap_ratio

    try:
        return float(
            polygon_overlap_ratio(
                np.asarray(predicted, dtype=np.float64),
                np.asarray(target, dtype=np.float64),
            )
        )
    except Exception:
        return None


def _polygon_valid(points: Any) -> bool:
    from utils.cross_scene_estimator import polygon_validity

    try:
        return bool(polygon_validity(np.asarray(points, dtype=np.float64))["valid"])
    except Exception:
        return False


def evaluate_geometry_mode(
    geometry: Mapping[str, Any], ground_truth: Mapping[str, Any]
) -> Dict[str, Any]:
    metrics = {
        "yaw_error_degrees": None,
        "translation_error": None,
        "translation_error_meters": None,
        "absolute_log_scale_error": None,
        "relative_scale_error": None,
        "predicted_scale": None,
        "gt_scale": None,
        "bev": {
            "frame_preserving": None,
            "pre_manhattan": None,
        },
    }
    if not ground_truth["is_match"] or not geometry["pose_valid"]:
        return metrics
    pose = geometry["relative_pose"]
    target = ground_truth["relative_pose"]
    predicted_scale = float(pose["relative_scale"])
    gt_scale = float(target["relative_scale"])
    translation_error = float(
        np.linalg.norm(
            np.asarray(pose["translation"], dtype=np.float64)
            - np.asarray(target["translation"], dtype=np.float64)
        )
    )
    metrics.update({
        "yaw_error_degrees": math.degrees(
            _wrapped_angle_error(
                float(pose["relative_yaw_radians"]),
                float(target["relative_yaw_radians"]),
            )
        ),
        "translation_error": translation_error,
        "translation_error_meters": (
            None
            if ground_truth["metric_factor_A"] is None
            else translation_error * float(ground_truth["metric_factor_A"])
        ),
        "absolute_log_scale_error": abs(
            math.log(max(predicted_scale, 1e-12) / max(gt_scale, 1e-12))
        ),
        "relative_scale_error": abs(predicted_scale / max(gt_scale, 1e-12) - 1.0),
        "predicted_scale": predicted_scale,
        "gt_scale": gt_scale,
    })
    targets = ground_truth["joint_bev_target"]["instances"]
    for name, key in (
        ("frame_preserving", "joint_bev_frame_preserving_polygon"),
        ("pre_manhattan", "joint_bev_pre_manhattan_polygon"),
    ):
        joint = geometry.get(key)
        if joint is None:
            continue
        rooms = joint["rooms"]
        overlap_a = _safe_overlap(rooms[0]["boundary"], targets[0]["polygon"])
        overlap_b = _safe_overlap(rooms[1]["boundary"], targets[1]["polygon"])
        metrics["bev"][name] = {
            "instance_A_overlap": overlap_a,
            "instance_B_overlap": overlap_b,
            "instance_mean_overlap": (
                None
                if overlap_a is None or overlap_b is None
                else (overlap_a + overlap_b) / 2.0
            ),
            "polygon_A_valid": _polygon_valid(rooms[0]["boundary"]),
            "polygon_B_valid": _polygon_valid(rooms[1]["boundary"]),
        }
    return metrics


def attach_posthoc_evaluation(
    frozen: Mapping[str, Any],
    fanout: Mapping[str, Any],
    batch: Mapping[str, Any],
    record: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach manifest/annotation GT after predictions and geometry are frozen."""
    candidate = _candidate_evaluation(frozen, batch)
    modes = copy.deepcopy(fanout["modes"])
    for mode in GEOMETRY_MODES:
        modes[mode]["metrics"] = evaluate_geometry_mode(modes[mode], ground_truth)
    m2 = modes["candidate_local_m2"]
    if ground_truth["is_match"] and m2["pose_valid"]:
        from tools.evaluate_zind_room_group import _portal_endpoint_metrics

        for side in ("A", "B"):
            portal = m2.get(f"portal_geometry_{side}")
            target = ground_truth.get(f"target_portal_endpoints_{side}")
            if portal is not None and portal.get("endpoints_local") is not None and target:
                m2.setdefault("portal_endpoint_metrics", {})[side] = (
                    _portal_endpoint_metrics(portal["endpoints_local"], target)
                )
    return {
        "schema_version": "zind-val-geometry-pair-v1",
        "pair_id": frozen["pair_id"],
        "house_id": frozen["house_id"],
        "floor_id": frozen["floor_id"],
        "room_a": frozen["room_a"],
        "room_b": frozen["room_b"],
        "pano_a": frozen["pano_a"],
        "pano_b": frozen["pano_b"],
        "split": "val",
        "purpose": PURPOSE,
        "prediction": {
            "model_forward_count": frozen["model_forward_count"],
            "is_match": frozen["is_match"],
            "best_candidate_pair": copy.deepcopy(frozen["best_candidate_pair"]),
            "candidate_count_A": frozen["candidate_count_A"],
            "candidate_count_B": frozen["candidate_count_B"],
            "candidate_pair_scores": copy.deepcopy(
                frozen["candidate_pair_scores"]
            ),
            "pair_match_probability": frozen["pair_match_probability"],
            "pair_no_match_probability": frozen["pair_no_match_probability"],
            "predicted_intervals_A": copy.deepcopy(
                frozen["predicted_intervals_A"]
            ),
            "predicted_intervals_B": copy.deepcopy(
                frozen["predicted_intervals_B"]
            ),
            "geometry_modes": modes,
            "matcher_score_max_abs_diff": fanout[
                "matcher_score_max_abs_diff"
            ],
            "best_candidate_identical": fanout["best_candidate_identical"],
            "gt_available_to_prediction": False,
            "gt_available_to_geometry": False,
        },
        "ground_truth": copy.deepcopy(dict(ground_truth)),
        "evaluation": candidate,
        "ground_truth_usage": "post-frozen-prediction evaluation only",
        "test_data_accessed": False,
    }


def _pair_file(output_dir: Path, index: int, pair_id: str) -> Path:
    safe = pair_id.replace("/", "_")
    return output_dir / "pairs" / f"{index:03d}_{safe}.json"


def _frozen_file(output_dir: Path, index: int, pair_id: str) -> Path:
    safe = pair_id.replace("/", "_")
    return output_dir / "frozen_predictions" / f"{index:03d}_{safe}.json"


def load_completed_records(
    output_dir: Path, pair_ids: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for index, pair_id in enumerate(pair_ids):
        path = _pair_file(output_dir, index, pair_id)
        if not path.is_file():
            continue
        value = load_json(path)
        if value.get("pair_id") != pair_id or value.get("split") != "val":
            raise ValueError(f"resume pair artifact is corrupt: {path}")
        output[pair_id] = value
    return output


def rewrite_per_pair_jsonl(
    output_dir: Path,
    pair_ids: Sequence[str],
    completed: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        json.dumps(_jsonable(completed[pair_id]), sort_keys=True, allow_nan=False)
        for pair_id in pair_ids
        if pair_id in completed
    ]
    atomic_write_text(
        output_dir / "per_pair.jsonl", "\n".join(lines) + ("\n" if lines else "")
    )


def _stats(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _threshold_rates(
    values: Sequence[Optional[float]], thresholds: Sequence[float], denominator: int
) -> Dict[str, Any]:
    return {
        f"lt_{threshold:g}": {
            "count": sum(
                value is not None and math.isfinite(float(value)) and value < threshold
                for value in values
            ),
            "denominator": denominator,
            "rate": (
                sum(
                    value is not None
                    and math.isfinite(float(value))
                    and value < threshold
                    for value in values
                )
                / denominator
                if denominator
                else None
            ),
        }
        for threshold in thresholds
    }


def _mode_subset_summary(
    records: Sequence[Mapping[str, Any]], mode: str
) -> Dict[str, Any]:
    count = len(records)
    geometries = [value["prediction"]["geometry_modes"][mode] for value in records]
    metrics = [value["metrics"] for value in geometries]
    yaw = [value["yaw_error_degrees"] for value in metrics]
    translation = [value["translation_error_meters"] for value in metrics]
    log_scale = [value["absolute_log_scale_error"] for value in metrics]
    relative_scale = [value["relative_scale_error"] for value in metrics]
    predicted_scale = [value["predicted_scale"] for value in metrics]
    valid_count = sum(bool(value["pose_valid"]) for value in geometries)
    hard_count = sum(bool(value["hard_failure"]) for value in geometries)
    joint = {}
    for yaw_limit, translation_limit in POSE_THRESHOLDS:
        key = f"yaw_lt_{yaw_limit:g}_translation_lt_{translation_limit:g}m"
        success = sum(
            first is not None
            and second is not None
            and first < yaw_limit
            and second < translation_limit
            for first, second in zip(yaw, translation)
        )
        success_scale = sum(
            first is not None
            and second is not None
            and third is not None
            and first < yaw_limit
            and second < translation_limit
            and third < 0.20
            for first, second, third in zip(yaw, translation, relative_scale)
        )
        joint[key] = {"count": success, "denominator": count, "rate": success / count if count else None}
        joint[key + "_scale_rel_lt_20pct"] = {
            "count": success_scale,
            "denominator": count,
            "rate": success_scale / count if count else None,
        }
    scale_layers = {}
    for name, predicate in (
        ("gt_scale_approximately_unit", lambda value: value is True),
        ("gt_scale_non_unit", lambda value: value is False),
    ):
        selected = [
            metric
            for record, metric in zip(records, metrics)
            if predicate(record["ground_truth"]["gt_scale_is_approximately_unit"])
        ]
        scale_layers[name] = {
            "pair_count": len(selected),
            "absolute_log_scale_error": _stats(
                value["absolute_log_scale_error"] for value in selected
            ),
            "relative_scale_error": _stats(
                value["relative_scale_error"] for value in selected
            ),
        }
    bev = {}
    for representation in ("frame_preserving", "pre_manhattan"):
        values = [metric["bev"][representation] for metric in metrics]
        valid_values = [value for value in values if value is not None]
        mean_overlaps = [value["instance_mean_overlap"] for value in valid_values]
        pose_good_shape_bad = sum(
            metric["yaw_error_degrees"] is not None
            and metric["translation_error_meters"] is not None
            and metric["yaw_error_degrees"] < 15.0
            and metric["translation_error_meters"] < 1.0
            and value is not None
            and value["instance_mean_overlap"] is not None
            and value["instance_mean_overlap"] < 0.5
            for metric, value in zip(metrics, values)
        )
        bev[representation] = {
            "valid_pose_bev_count": len(valid_values),
            "instance_A_overlap": _stats(
                value["instance_A_overlap"] for value in valid_values
            ),
            "instance_B_overlap": _stats(
                value["instance_B_overlap"] for value in valid_values
            ),
            "instance_mean_overlap": _stats(mean_overlaps),
            "invalid_polygon_count": sum(
                not value["polygon_A_valid"] or not value["polygon_B_valid"]
                for value in valid_values
            ),
            "invalid_polygon_rate": (
                sum(
                    not value["polygon_A_valid"] or not value["polygon_B_valid"]
                    for value in valid_values
                )
                / len(valid_values)
                if valid_values
                else None
            ),
            "pose_good_but_mean_overlap_lt_0.5_count": pose_good_shape_bad,
            "pose_good_but_mean_overlap_lt_0.5_rate_over_valid_pose": (
                pose_good_shape_bad / valid_count if valid_count else None
            ),
        }
    return {
        "pair_count": count,
        "coverage": {
            "predicted_match_count": sum(
                bool(value["prediction"]["is_match"]) for value in records
            ),
            "geometry_attempted_count": sum(bool(value["attempted"]) for value in geometries),
            "pose_valid_count": valid_count,
            "pose_valid_rate": valid_count / count if count else None,
            "hard_failure_count": hard_count,
            "hard_failure_rate": hard_count / count if count else None,
            "metric_translation_scale_available_count": sum(
                bool(value["ground_truth"].get("metric_translation_available"))
                for value in records
            ),
            "metric_translation_error_available_count": sum(
                value is not None for value in translation
            ),
        },
        "yaw_error_degrees": {
            "distribution_over_valid_pose": _stats(yaw),
            "coverage_aware_threshold_rates": _threshold_rates(
                yaw, (5.0, 10.0, 15.0, 30.0), count
            ),
        },
        "translation_error_meters": {
            "distribution_over_valid_pose": _stats(translation),
            "coverage_aware_threshold_rates": _threshold_rates(
                translation, (0.25, 0.5, 1.0, 2.0), count
            ),
        },
        "scale": {
            "absolute_log_scale_error": _stats(log_scale),
            "relative_scale_error": _stats(relative_scale),
            "predicted_width_ratio_scale": _stats(predicted_scale),
            "coverage_aware_relative_error_rates": _threshold_rates(
                relative_scale, (0.05, 0.10, 0.20), count
            ),
            "gt_scale_layers": scale_layers,
        },
        "joint_pose_success": joint,
        "bev": bev,
    }


def _matching_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    positives = [value for value in records if value["ground_truth"]["is_match"]]
    negatives = [value for value in records if not value["ground_truth"]["is_match"]]
    tp = sum(value["prediction"]["is_match"] for value in positives)
    fn = len(positives) - tp
    fp = sum(value["prediction"]["is_match"] for value in negatives)
    tn = len(negatives) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(positives) if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_rejection = tn / len(negatives) if negatives else 0.0
    candidate_recall = sum(
        value["evaluation"]["gt_candidate_recalled"] for value in positives
    ) / len(positives)
    end_to_end_top1 = sum(
        value["evaluation"]["selected_candidate_correct"] for value in positives
    ) / len(positives)
    conditional = [value for value in positives if value["prediction"]["is_match"]]
    conditional_top1 = (
        sum(value["evaluation"]["selected_candidate_correct"] for value in conditional)
        / len(conditional)
        if conditional
        else 0.0
    )
    harmonic = (
        2 * end_to_end_top1 * negative_rejection
        / (end_to_end_top1 + negative_rejection)
        if end_to_end_top1 + negative_rejection
        else 0.0
    )
    return {
        "counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "positive_match_recall": recall,
        "negative_rejection": negative_rejection,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + negative_rejection),
        "candidate_recall": candidate_recall,
        "conditional_candidate_top1": conditional_top1,
        "end_to_end_candidate_top1": end_to_end_top1,
        "harmonic": harmonic,
        "matcher_score_max_abs_diff_across_geometry_modes": max(
            float(value["prediction"]["matcher_score_max_abs_diff"])
            for value in records
        ),
        "best_candidate_identical_across_geometry_modes": all(
            value["prediction"]["best_candidate_identical"] for value in records
        ),
    }


def _soft_fail_reasons(portal: Mapping[str, Any]) -> List[str]:
    quality = portal.get("predicted_only_quality") or {}
    reasons = []
    if int(quality.get("support_point_count", 0)) < 4:
        reasons.append("support_count")
    if float(quality.get("inlier_ratio", 0.0)) < 0.60:
        reasons.append("inlier_ratio")
    residual = quality.get("normalized_residual")
    if residual is None or float(residual) > 0.05:
        reasons.append("normalized_residual")
    angular = quality.get("angular_dispersion_degrees")
    if angular is None or float(angular) > 25.0:
        reasons.append("angular_dispersion")
    if float(quality.get("ray_intersection_condition", 0.0)) < 0.05:
        reasons.append("ray_condition")
    if not bool(quality.get("width_plausible", False)):
        reasons.append("width_plausibility")
    if quality.get("left_right_context_agreement") is False:
        reasons.append("context_disagreement")
    if bool(quality.get("corner_ambiguity", False)):
        reasons.append("corner_ambiguity")
    return reasons


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(first: Sequence[float], second: Sequence[float]) -> Dict[str, Any]:
    if len(first) < 3 or len(first) != len(second):
        return {"count": len(first), "pearson": None, "spearman": None}
    first_value, second_value = np.asarray(first), np.asarray(second)
    if np.std(first_value) <= 1e-12 or np.std(second_value) <= 1e-12:
        pearson = None
    else:
        pearson = float(np.corrcoef(first_value, second_value)[0, 1])
    ranks_first, ranks_second = _rankdata(first), _rankdata(second)
    spearman = (
        None
        if np.std(ranks_first) <= 1e-12 or np.std(ranks_second) <= 1e-12
        else float(np.corrcoef(ranks_first, ranks_second)[0, 1])
    )
    return {"count": len(first), "pearson": pearson, "spearman": spearman}


def quality_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    positives = [value for value in records if value["ground_truth"]["is_match"]]
    predicted_matches = [
        value for value in records if value["prediction"]["is_match"]
    ]
    all_m2_values = [
        value["prediction"]["geometry_modes"]["candidate_local_m2"]
        for value in predicted_matches
    ]
    all_hard = Counter(
        str(value.get("failure_reason") or "unknown")
        for value in all_m2_values
        if value["hard_failure"]
    )
    all_soft_pairs = []
    all_soft_reasons = Counter()
    for record, geometry in zip(predicted_matches, all_m2_values):
        reasons = []
        for side in ("A", "B"):
            portal = geometry.get(f"portal_geometry_{side}")
            if portal is None:
                continue
            side_reasons = _soft_fail_reasons(portal)
            reasons.extend(f"{side}:{value}" for value in side_reasons)
            all_soft_reasons.update(side_reasons)
        if reasons:
            all_soft_pairs.append((record, geometry, reasons))
    m2_values = [
        value["prediction"]["geometry_modes"]["candidate_local_m2"]
        for value in positives
    ]
    hard = Counter(
        str(value.get("failure_reason") or "unknown")
        for value in m2_values
        if value["hard_failure"]
    )
    portals = []
    soft_pairs = []
    for record, geometry in zip(positives, m2_values):
        pair_reasons = []
        for side in ("A", "B"):
            portal = geometry.get(f"portal_geometry_{side}")
            if portal is None:
                continue
            reasons = _soft_fail_reasons(portal)
            portals.append((record, geometry, side, portal, reasons))
            pair_reasons.extend(f"{side}:{value}" for value in reasons)
        if pair_reasons:
            soft_pairs.append((record, geometry, pair_reasons))
    reason_counts = Counter(
        reason for _, _, _, _, reasons in portals for reason in reasons
    )
    fields = {
        "support_count": "support_point_count",
        "inlier_ratio": "inlier_ratio",
        "normalized_residual": "normalized_residual",
        "angular_dispersion": "angular_dispersion_degrees",
        "ray_condition": "ray_intersection_condition",
        "opening_width": "opening_width",
        "reliability_score": "reliability_score",
    }
    distributions = {
        output: _stats(
            portal["predicted_only_quality"].get(source)
            for _, _, _, portal, _ in portals
        )
        for output, source in fields.items()
    }
    variants = Counter(
        str(portal.get("primary_variant")) for _, _, _, portal, _ in portals
    )
    wrap_sides = sum(
        bool(portal["token_interval"][0] > portal["token_interval"][1])
        for _, _, _, portal, _ in portals
    )
    correlations = {}
    for quality_name, source in fields.items():
        for error_name in (
            "yaw_error_degrees",
            "translation_error_meters",
            "absolute_log_scale_error",
        ):
            first, second = [], []
            for record, geometry, side, portal, _ in portals:
                value = portal["predicted_only_quality"].get(source)
                error = geometry["metrics"].get(error_name)
                if value is not None and error is not None:
                    first.append(float(value))
                    second.append(float(error))
            correlations[f"{quality_name}_vs_{error_name}"] = _correlation(
                first, second
            )
    pose_correct = lambda geometry: (
        geometry["metrics"]["yaw_error_degrees"] is not None
        and geometry["metrics"]["translation_error_meters"] is not None
        and geometry["metrics"]["yaw_error_degrees"] < 15.0
        and geometry["metrics"]["translation_error_meters"] < 1.0
    )
    return {
        "predicted_match_denominator": len(predicted_matches),
        "all_predicted_match_geometry_valid_count": sum(
            value["pose_valid"] for value in all_m2_values
        ),
        "all_predicted_match_geometry_valid_rate": (
            sum(value["pose_valid"] for value in all_m2_values)
            / len(predicted_matches)
            if predicted_matches
            else None
        ),
        "all_predicted_match_hard_failure_count": sum(
            value["hard_failure"] for value in all_m2_values
        ),
        "all_predicted_match_hard_failure_rate": (
            sum(value["hard_failure"] for value in all_m2_values)
            / len(predicted_matches)
            if predicted_matches
            else None
        ),
        "all_predicted_match_hard_failure_reasons": dict(all_hard),
        "all_predicted_match_soft_fail_pair_count": len(all_soft_pairs),
        "all_predicted_match_soft_fail_pair_rate": (
            len(all_soft_pairs) / len(predicted_matches)
            if predicted_matches
            else None
        ),
        "all_predicted_match_soft_fail_side_reason_counts": dict(
            all_soft_reasons
        ),
        "gt_negative_predicted_match_count": sum(
            not value["ground_truth"]["is_match"] for value in predicted_matches
        ),
        "gt_negative_pose_generated_count": sum(
            not record["ground_truth"]["is_match"] and geometry["pose_valid"]
            for record, geometry in zip(predicted_matches, all_m2_values)
        ),
        "positive_denominator": len(positives),
        "m2_geometry_valid_count": sum(value["pose_valid"] for value in m2_values),
        "m2_geometry_valid_rate_over_all_positive": (
            sum(value["pose_valid"] for value in m2_values) / len(positives)
            if positives
            else None
        ),
        "hard_failure_count": sum(value["hard_failure"] for value in m2_values),
        "hard_failure_reasons": dict(hard),
        "soft_fail_pair_count": len(soft_pairs),
        "soft_fail_pair_rate_over_all_positive": (
            len(soft_pairs) / len(positives) if positives else None
        ),
        "soft_fail_side_reason_counts": dict(reason_counts),
        "soft_fail_but_pose_correct_count": sum(
            pose_correct(geometry) for _, geometry, _ in soft_pairs
        ),
        "soft_fail_and_pose_incorrect_count": sum(
            not pose_correct(geometry) for _, geometry, _ in soft_pairs
        ),
        "portal_side_count": len(portals),
        "variant_counts": dict(variants),
        "wrap_interval_side_count": wrap_sides,
        "wrap_interval_side_rate": wrap_sides / len(portals) if portals else None,
        "corner_ambiguity_count": sum(
            bool(portal["predicted_only_quality"].get("corner_ambiguity"))
            for _, _, _, portal, _ in portals
        ),
        "width_implausible_count": sum(
            not bool(portal["predicted_only_quality"].get("width_plausible"))
            for _, _, _, portal, _ in portals
        ),
        "predicted_only_distributions": distributions,
        "posthoc_quality_error_correlations": correlations,
        "gate_thresholds_frozen": copy.deepcopy(SOFT_GATE_THRESHOLDS),
        "gate_calibrated_on_val": False,
    }


def paired_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for record in records:
        if not (
            record["ground_truth"]["is_match"]
            and record["evaluation"]["selected_candidate_correct"]
        ):
            continue
        g1 = record["prediction"]["geometry_modes"]["frame_polygon"]
        g2 = record["prediction"]["geometry_modes"]["candidate_local_m2"]
        first, second = g1["metrics"], g2["metrics"]

        def delta(key: str) -> Optional[float]:
            if first[key] is None or second[key] is None:
                return None
            return float(second[key] - first[key])

        def bev_delta(representation: str) -> Optional[float]:
            a = first["bev"][representation]
            b = second["bev"][representation]
            if a is None or b is None:
                return None
            if a["instance_mean_overlap"] is None or b["instance_mean_overlap"] is None:
                return None
            return float(b["instance_mean_overlap"] - a["instance_mean_overlap"])

        output.append({
            "pair_id": record["pair_id"],
            "house_id": record["house_id"],
            "g1_pose_valid": g1["pose_valid"],
            "m2_pose_valid": g2["pose_valid"],
            "g1_yaw_error_degrees": first["yaw_error_degrees"],
            "m2_yaw_error_degrees": second["yaw_error_degrees"],
            "delta_yaw_degrees": delta("yaw_error_degrees"),
            "g1_translation_error_meters": first["translation_error_meters"],
            "m2_translation_error_meters": second["translation_error_meters"],
            "delta_translation_meters": delta("translation_error_meters"),
            "g1_absolute_log_scale_error": first["absolute_log_scale_error"],
            "m2_absolute_log_scale_error": second["absolute_log_scale_error"],
            "delta_absolute_log_scale": delta("absolute_log_scale_error"),
            "delta_bev_frame_mean_overlap": bev_delta("frame_preserving"),
            "delta_bev_raw_mean_overlap": bev_delta("pre_manhattan"),
            "both_fail_medium_pose_threshold": not (
                first["yaw_error_degrees"] is not None
                and first["translation_error_meters"] is not None
                and (
                    first["yaw_error_degrees"] < 15.0
                    and first["translation_error_meters"] < 1.0
                )
            )
            and not (
                second["yaw_error_degrees"] is not None
                and second["translation_error_meters"] is not None
                and (
                    second["yaw_error_degrees"] < 15.0
                    and second["translation_error_meters"] < 1.0
                )
            ),
        })
    return output


def _win_tie_loss(rows: Sequence[Mapping[str, Any]], key: str, lower_is_better: bool) -> Dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if lower_is_better:
        wins = sum(value < -PAIRED_TIE_EPSILON for value in values)
        losses = sum(value > PAIRED_TIE_EPSILON for value in values)
    else:
        wins = sum(value > PAIRED_TIE_EPSILON for value in values)
        losses = sum(value < -PAIRED_TIE_EPSILON for value in values)
    ties = len(values) - wins - losses
    return {
        "comparable_count": len(values),
        "m2_win": wins,
        "tie": ties,
        "m2_loss": losses,
        "median_delta": float(np.median(values)) if values else None,
    }


def paired_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    obvious_improvement = [
        row["pair_id"]
        for row in rows
        if (row.get("delta_yaw_degrees") is not None and row["delta_yaw_degrees"] <= -10.0)
        or (
            row.get("delta_translation_meters") is not None
            and row["delta_translation_meters"] <= -0.5
        )
    ]
    obvious_worsening = [
        row["pair_id"]
        for row in rows
        if (row.get("delta_yaw_degrees") is not None and row["delta_yaw_degrees"] >= 10.0)
        or (
            row.get("delta_translation_meters") is not None
            and row["delta_translation_meters"] >= 0.5
        )
    ]
    return {
        "S3_pair_count": len(rows),
        "yaw": _win_tie_loss(rows, "delta_yaw_degrees", True),
        "translation": _win_tie_loss(rows, "delta_translation_meters", True),
        "absolute_log_scale": _win_tie_loss(
            rows, "delta_absolute_log_scale", True
        ),
        "bev_frame_mean_overlap": _win_tie_loss(
            rows, "delta_bev_frame_mean_overlap", False
        ),
        "bev_raw_mean_overlap": _win_tie_loss(
            rows, "delta_bev_raw_mean_overlap", False
        ),
        "obvious_improvement_definition": "delta_yaw<=-10deg or delta_translation<=-0.5m",
        "obvious_worsening_definition": "delta_yaw>=10deg or delta_translation>=0.5m",
        "obvious_improvement_pair_ids": obvious_improvement,
        "obvious_worsening_pair_ids": obvious_worsening,
        "both_fail_medium_pose_pair_ids": [
            row["pair_id"] for row in rows if row["both_fail_medium_pose_threshold"]
        ],
        "confidence_interval_reported": False,
        "house_level_bootstrap_ready": True,
        "house_level_bootstrap_executed": False,
    }


def build_summary(
    records: Sequence[Mapping[str, Any]], run_metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    positives = [value for value in records if value["ground_truth"]["is_match"]]
    predicted_match = [
        value
        for value in positives
        if value["prediction"]["is_match"]
    ]
    correct_candidate = [
        value
        for value in positives
        if value["evaluation"]["selected_candidate_correct"]
    ]
    m2_valid = [
        value
        for value in positives
        if value["prediction"]["geometry_modes"]["candidate_local_m2"][
            "pose_valid"
        ]
    ]
    sets = {
        "S0_all_pairs": records,
        "S1_all_gt_positive": positives,
        "S2_gt_positive_predicted_match": predicted_match,
        "S3_gt_positive_selected_candidate_correct": correct_candidate,
        "S4_gt_positive_m2_geometry_valid": m2_valid,
    }
    geometry = {
        mode: {
            name: _mode_subset_summary(values, mode)
            for name, values in sets.items()
            if name != "S0_all_pairs"
        }
        for mode in GEOMETRY_MODES
    }
    rows = paired_rows(records)
    return {
        "schema_version": "zind-val-geometry-smoke-summary-v1",
        "split": "val",
        "purpose": PURPOSE,
        "pair_count": len(records),
        "completed_pair_count": len(records),
        "set_counts": {name: len(values) for name, values in sets.items()},
        "set_definitions": {
            "S0_all_pairs": "all 64 sampled pairs",
            "S1_all_gt_positive": "all sampled GT-positive pairs; missing pose remains failure",
            "S2_gt_positive_predicted_match": "GT positive and matcher predicted a real candidate pair",
            "S3_gt_positive_selected_candidate_correct": "GT positive and selected predicted candidate overlaps the GT portal at IoU>=0.30",
            "S4_gt_positive_m2_geometry_valid": "GT positive with a valid M2 portal pose; coverage over S1 is always reported",
        },
        "matching": _matching_summary(records),
        "geometry": geometry,
        "quality": quality_summary(records),
        "paired_comparison": paired_summary(rows),
        "run_metadata": copy.deepcopy(dict(run_metadata)),
        "test_data_accessed": False,
        "scientific_final_result": False,
    }


def write_paired_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["pair_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _render_visualizations(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    from tools.evaluate_zind_room_group import _render_polygons_svg

    by_id = {value["pair_id"]: value for value in records}
    selected: List[Tuple[str, str]] = []
    selected.extend(("fixed_order", value["pair_id"]) for value in records[:4])
    comparable = [
        value
        for value in paired
        if value.get("delta_yaw_degrees") is not None
        and value.get("delta_translation_meters") is not None
    ]
    comparable.sort(
        key=lambda value: value["delta_yaw_degrees"] / 10.0
        + value["delta_translation_meters"] / 0.5
    )
    selected.extend(("m2_improved", value["pair_id"]) for value in comparable[:3])
    selected.extend(("m2_worsened", value["pair_id"]) for value in comparable[-3:])
    for record in records:
        m2 = record["prediction"]["geometry_modes"]["candidate_local_m2"]
        if m2["hard_failure"]:
            selected.append(("hard_failure", record["pair_id"]))
    for record in records:
        m2 = record["prediction"]["geometry_modes"]["candidate_local_m2"]
        portals = [m2.get("portal_geometry_A"), m2.get("portal_geometry_B")]
        soft = any(
            portal is not None
            and not bool(portal["predicted_only_quality"].get("quality_gate_pass"))
            for portal in portals
        )
        metrics = m2["metrics"]
        correct = (
            metrics["yaw_error_degrees"] is not None
            and metrics["translation_error_meters"] is not None
            and metrics["yaw_error_degrees"] < 15.0
            and metrics["translation_error_meters"] < 1.0
        )
        if soft:
            selected.append(("soft_fail_pose_correct" if correct else "soft_fail_pose_wrong", record["pair_id"]))
    deduplicated = []
    seen = set()
    category_counts = Counter()
    for category, pair_id in selected:
        if pair_id in seen or category_counts[category] >= 3:
            continue
        seen.add(pair_id)
        category_counts[category] += 1
        deduplicated.append((category, pair_id))
    artifacts = []
    for index, (category, pair_id) in enumerate(deduplicated):
        record = by_id[pair_id]
        gt = record["ground_truth"]["joint_bev_target"]
        if gt is None:
            continue
        polygons = [
            ("GT A", gt["instances"][0]["polygon"], "#289128"),
            ("GT B", gt["instances"][1]["polygon"], "#7a9b22"),
        ]
        for mode, color in (("frame_polygon", "#e18a00"), ("candidate_local_m2", "#1774c8")):
            joint = record["prediction"]["geometry_modes"][mode].get(
                "joint_bev_frame_preserving_polygon"
            )
            if joint is not None:
                polygons.append((mode + " B", joint["rooms"][1]["boundary"], color))
        path = output_dir / "visualizations" / f"{index:02d}_{category}_{pair_id}.svg"
        _render_polygons_svg(path, polygons)
        artifacts.append({
            "category": category,
            "pair_id": pair_id,
            "path": str(path),
            "selection_is_posthoc_diagnostic": category != "fixed_order",
        })
    report = {
        "selection_policy": (
            "first four fixed pair IDs plus bounded posthoc largest improvements, "
            "largest worsenings, hard failures, and soft-fail outcome strata"
        ),
        "not_random_qualitative_examples": True,
        "artifacts": artifacts,
    }
    atomic_write_json(output_dir / "visualizations" / "selection.json", report)
    return report


def _write_failures(output_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    (output_dir / "failures").mkdir(parents=True, exist_ok=True)
    for record in records:
        for mode in GEOMETRY_MODES:
            geometry = record["prediction"]["geometry_modes"][mode]
            if geometry["hard_failure"]:
                atomic_write_json(
                    output_dir
                    / "failures"
                    / f"{record['pair_id']}_{mode}.json",
                    {
                        "pair_id": record["pair_id"],
                        "house_id": record["house_id"],
                        "mode": mode,
                        "failure_reason": geometry["failure_reason"],
                        "best_candidate_pair": record["prediction"][
                            "best_candidate_pair"
                        ],
                        "gt_usage": "failure recorded before posthoc metric selection",
                    },
                )


def _report_markdown(summary: Mapping[str, Any]) -> str:
    matching = summary["matching"]
    sets = summary["set_counts"]
    quality = summary["quality"]
    paired = summary["paired_comparison"]
    lines = [
        "# ZInD val geometry smoke preflight",
        "",
        PURPOSE,
        "",
        "## Completion and safety",
        "",
        f"- completed: {summary['completed_pair_count']}/{summary['pair_count']}",
        "- split: val",
        "- test data accessed: false",
        "- one frozen model/matcher prediction fans out to G0/G1/G2",
        "",
        "## Matching",
        "",
        f"- TP/TN/FP/FN: {matching['counts']['TP']}/{matching['counts']['TN']}/{matching['counts']['FP']}/{matching['counts']['FN']}",
        f"- positive recall: {matching['positive_match_recall']:.4f}",
        f"- negative rejection: {matching['negative_rejection']:.4f}",
        f"- end-to-end candidate top1: {matching['end_to_end_candidate_top1']:.4f}",
        f"- matcher maximum branch difference: {matching['matcher_score_max_abs_diff_across_geometry_modes']}",
        "",
        "## Evaluation strata",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sets.items())
    lines.extend([
        "",
        "## Geometry on S3 (correct selected candidate)",
        "",
        "| mode | pose coverage | yaw median | translation median (m) | log-scale median |",
        "|---|---:|---:|---:|---:|",
    ])
    for mode in GEOMETRY_MODES:
        value = summary["geometry"][mode][
            "S3_gt_positive_selected_candidate_correct"
        ]
        lines.append(
            "| {} | {}/{} | {} | {} | {} |".format(
                mode,
                value["coverage"]["pose_valid_count"],
                value["pair_count"],
                value["yaw_error_degrees"]["distribution_over_valid_pose"]["median"],
                value["translation_error_meters"]["distribution_over_valid_pose"]["median"],
                value["scale"]["absolute_log_scale_error"]["median"],
            )
        )
    lines.extend([
        "",
        "## M2 quality",
        "",
        f"- valid over all positives: {quality['m2_geometry_valid_count']}/{quality['positive_denominator']}",
        f"- hard failures: {quality['hard_failure_count']}",
        f"- soft-fail pairs: {quality['soft_fail_pair_count']}",
        f"- variants: {quality['variant_counts']}",
        "",
        "## Paired S3 comparison (M2 - G1)",
        "",
        f"- yaw W/T/L: {paired['yaw']['m2_win']}/{paired['yaw']['tie']}/{paired['yaw']['m2_loss']}",
        f"- translation W/T/L: {paired['translation']['m2_win']}/{paired['translation']['tie']}/{paired['translation']['m2_loss']}",
        f"- log-scale W/T/L: {paired['absolute_log_scale']['m2_win']}/{paired['absolute_log_scale']['tie']}/{paired['absolute_log_scale']['m2_loss']}",
        "",
        "No confidence interval or significance claim is made for this 64-pair smoke set.",
    ])
    return "\n".join(lines) + "\n"


def _run_metadata(
    context: Mapping[str, Any],
    completed: int,
    start_time: float,
    frozen_prediction_count: int,
    new_live_forward_count: int,
) -> Dict[str, Any]:
    args = context["args"]
    return {
        "schema_version": "zind-val-geometry-smoke-run-v1",
        "split": "val",
        "purpose": PURPOSE,
        "seed": args.seed,
        "pair_count": 64,
        "completed_pair_count": completed,
        "geometry_modes": list(GEOMETRY_MODES),
        "checkpoint": {
            "path": str(context["checkpoint_path"]),
            "sha256": context["checkpoint_sha256"],
            **copy.deepcopy(context["checkpoint_contract"]),
        },
        "m2": _m2_contract(),
        "sample_manifest": {
            "path": str(context["sample_path"]),
            "sha256": context["sample_sha256"],
        },
        "val_manifest": {
            "path": str(context["val_manifest_path"]),
            "sha256": context["val_manifest_sha256"],
        },
        "val_cache": str(context["cache_dir"]),
        "house_split": copy.deepcopy(context["house_split"]),
        "one_live_backbone_call_per_new_pair": True,
        "requested_device": args.device,
        "gpu_used": True,
        "frozen_prediction_count": frozen_prediction_count,
        "new_live_model_forward_count_this_invocation": new_live_forward_count,
        "geometry_fanout_after_frozen_prediction": True,
        "resume_supported_by_atomic_frozen_and_pair_artifacts": True,
        "soft_gate_used_for_rejection": False,
        "gt_available_to_prediction": False,
        "gt_available_to_geometry": False,
        "test_data_accessed": False,
        "runtime_seconds_so_far": time.perf_counter() - start_time,
    }


def run(context: Mapping[str, Any]) -> Dict[str, Any]:
    import torch
    from dataset.zind_bipair_dataset import collate_zind_bipair
    from tools.train_zind_end_to_end import JointZInDMatchingDataset
    from torch.utils.data import DataLoader, Subset

    args = context["args"]
    output_dir = context["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_ids = context["sample"]["pair_ids"]
    completed = load_completed_records(output_dir, pair_ids)
    if completed and not args.resume:
        raise FileExistsError(
            "completed pair artifacts already exist; pass --resume instead of rerunning"
        )
    metadata_path = output_dir / "run_metadata.json"
    previous_metadata = load_json(metadata_path) if metadata_path.is_file() else None
    prior_gpu_process_invocations = (
        int(previous_metadata.get("gpu_process_invocation_count", 1))
        if isinstance(previous_metadata, Mapping)
        else 0
    )
    gpu_process_invocation_count = prior_gpu_process_invocations + 1
    initial_completed_pair_count = len(completed)
    start = time.perf_counter()
    frozen_prediction_count = sum(
        _frozen_file(output_dir, index, pair_id).is_file()
        for index, pair_id in enumerate(pair_ids)
    )
    initial_frozen_prediction_count = frozen_prediction_count
    new_live_forward_count = 0
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the authorized smoke inference requires an available CUDA device")

    dataset = JointZInDMatchingDataset(
        str(context["val_manifest_path"]),
        str(context["cache_dir"]),
        data_root=args.data_root,
        max_pairs=None,
        expected_split="val",
        image_shape=tuple(context["cache_metadata"].get("image_shape", (512, 1024))),
        rotation_augmentation=False,
    )
    index_by_id = {
        str(record["id"]): index for index, record in enumerate(dataset.base.records)
    }
    selected_indices = [index_by_id[value] for value in pair_ids]
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_zind_bipair,
        drop_last=False,
    )
    bi_layout, matcher, threshold, model_report = _load_models(args, device)
    resolver = AnnotationResolver(_data_root(context))

    def current_run_metadata(completed_count: int) -> Dict[str, Any]:
        metadata = _run_metadata(
            context,
            completed_count,
            start,
            frozen_prediction_count,
            new_live_forward_count,
        )
        metadata.update({
            "model": model_report,
            "opening_probability_threshold": dict(threshold),
            "gpu_process_invocation_count": gpu_process_invocation_count,
            "resumed_after_interruption": initial_frozen_prediction_count > 0,
            "invocation_start_state": {
                "completed_pair_count": initial_completed_pair_count,
                "frozen_prediction_count": initial_frozen_prediction_count,
            },
            "aggregate_model_forward_count_from_frozen_artifacts": (
                frozen_prediction_count
            ),
        })
        return metadata

    run_metadata = current_run_metadata(len(completed))
    atomic_write_json(metadata_path, run_metadata)

    for index, batch in enumerate(loader):
        pair_id = str(batch["pair_id"][0])
        expected_id = pair_ids[index]
        if pair_id != expected_id:
            raise RuntimeError(
                f"DataLoader order changed: expected {expected_id}, received {pair_id}"
            )
        if pair_id in completed:
            print(f"[{index + 1}/64] resume skip {pair_id}", flush=True)
            continue
        record = context["records_by_id"][pair_id]
        frozen_path = _frozen_file(output_dir, index, pair_id)
        if frozen_path.is_file():
            frozen = load_json(frozen_path)
            if frozen.get("pair_id") != pair_id:
                raise ValueError(f"corrupt frozen prediction: {frozen_path}")
        else:
            identifiers = {
                "pair_id": pair_id,
                "house_id": record["scene_id"],
                "floor_id": record["floor_id"],
                "room_a": record["room_a"],
                "room_b": record["room_b"],
                "pano_a": record["pano_a"],
                "pano_b": record["pano_b"],
            }
            frozen = infer_frozen_prediction(
                args,
                batch["image_A"],
                batch["image_B"],
                identifiers,
                bi_layout,
                matcher,
                device,
                float(threshold["value"]),
            )
            atomic_write_json(frozen_path, frozen)
            frozen_prediction_count += 1
            new_live_forward_count += 1
        fanout = geometry_fanout(frozen)
        if fanout["matcher_score_max_abs_diff"] != 0.0 or not fanout[
            "best_candidate_identical"
        ]:
            raise RuntimeError("geometry branches changed frozen matcher outputs")
        ground_truth = resolver.pair_ground_truth(record)
        result = attach_posthoc_evaluation(
            frozen, fanout, batch, record, ground_truth
        )
        atomic_write_json(_pair_file(output_dir, index, pair_id), result)
        completed[pair_id] = result
        rewrite_per_pair_jsonl(output_dir, pair_ids, completed)
        run_metadata = current_run_metadata(len(completed))
        atomic_write_json(metadata_path, run_metadata)
        print(
            f"[{index + 1}/64] complete {pair_id} "
            f"match={frozen['is_match']} best={frozen['best_candidate_pair']}",
            flush=True,
        )

    ordered = [completed[value] for value in pair_ids if value in completed]
    if len(ordered) != 64:
        raise RuntimeError(f"smoke run is incomplete: {len(ordered)}/64")
    if not all(_finite_tree(value) for value in ordered):
        raise ValueError("completed pair output contains NaN or Inf")
    run_metadata = current_run_metadata(64)
    run_metadata["runtime_seconds"] = time.perf_counter() - start
    summary = build_summary(ordered, run_metadata)
    rows = paired_rows(ordered)
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "quality_summary.json", summary["quality"])
    write_paired_csv(output_dir / "paired_comparison.csv", rows)
    atomic_write_text(output_dir / "report.md", _report_markdown(summary))
    _write_failures(output_dir, ordered)
    visualizations = _render_visualizations(output_dir, ordered, rows)
    run_metadata["visualizations"] = visualizations
    atomic_write_json(metadata_path, run_metadata)
    return summary


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(child) for child in value)
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.prepare_sample:
        context = load_context(args, require_sample=False)
        sample_path = context["sample_path"]
        expected = context["expected_sample"]
        if sample_path.exists():
            validate_sample_manifest(load_json(sample_path), expected)
        else:
            atomic_write_json(sample_path, expected)
        print(json.dumps(expected, indent=2, ensure_ascii=False))
        return 0
    context = load_context(args, require_sample=True)
    if args.metadata_only:
        print(json.dumps(metadata_report(context), indent=2, ensure_ascii=False))
        return 0
    if args.dry_run:
        print(json.dumps(dry_run_report(context), indent=2, ensure_ascii=False))
        return 0
    summary = run(context)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
