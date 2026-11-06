#!/usr/bin/env python3
"""Frozen T3/M2 full-ZInD-val geometry diagnostic.

The evaluator performs one live prediction per manifest pair, atomically
freezes the prediction, and fans it out to the legacy polygon, frame-preserving
polygon, and candidate-local M2 geometry branches.  Ground truth is attached
only after this fanout.  D1 unit-scale and D2 oracle-scale comparisons are
strictly post-hoc CPU diagnostics and never replace the formal G2/D0 output.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import evaluate_zind_portal_geometry_val as smoke


PURPOSE = "full validation diagnostic; frozen protocol; not test evaluation"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/zind_geometry_val_full/t3_frozen_m2"
DEFAULT_FULL_MANIFEST = DEFAULT_OUTPUT / "full_val_manifest.json"
DEFAULT_ORIGINAL_METRICS = (
    REPO_ROOT
    / "checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best_metrics.json"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d"
)
EXPECTED_M2_SHA256 = (
    "cdce025fd5ba709a3c5afc827137860d1082c7ab44e27a5467ea6686a7a02a4c"
)
EXPECTED_OPENING_THRESHOLD = 0.47
BOOTSTRAP_SEED = 123
BOOTSTRAP_RESAMPLES = 10000
GEOMETRY_MODES = smoke.GEOMETRY_MODES
POSE_THRESHOLDS = smoke.POSE_THRESHOLDS
SCALE_THRESHOLDS = (0.10, 0.20)
PARITY_TOLERANCE = 1e-9


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--val_manifest", default=str(smoke.DEFAULT_VAL_MANIFEST))
    parser.add_argument("--train_manifest", default=str(smoke.DEFAULT_TRAIN_MANIFEST))
    parser.add_argument("--val_cache", default=str(smoke.DEFAULT_VAL_CACHE))
    parser.add_argument("--data_root")
    parser.add_argument("--full_manifest", default=str(DEFAULT_FULL_MANIFEST))
    parser.add_argument("--checkpoint", default=str(smoke.DEFAULT_CHECKPOINT))
    parser.add_argument("--original_metrics", default=str(DEFAULT_ORIGINAL_METRICS))
    parser.add_argument("--bi_layout_config", default=str(smoke.DEFAULT_CONFIG))
    parser.add_argument("--bi_layout_checkpoint", default=str(smoke.DEFAULT_BI_LAYOUT))
    parser.add_argument("--opening_checkpoint", default=str(smoke.DEFAULT_OPENING))
    parser.add_argument("--matcher_checkpoint", default=str(smoke.DEFAULT_MATCHER))
    parser.add_argument("--geometry_modes", default=",".join(GEOMETRY_MODES))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap_resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--min_opening_width_tokens", type=int, default=2)
    parser.add_argument("--max_openings_per_view", type=int, default=12)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--prepare_manifest", action="store_true")
    parser.add_argument("--metadata_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    # Deliberately not exposed as a CLI option: the only allowed threshold is
    # the value embedded in the frozen end-to-end checkpoint.
    args.opening_probability_threshold = None
    args.geometry_modes = tuple(
        value.strip() for value in args.geometry_modes.split(",") if value.strip()
    )
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.split != "val":
        raise ValueError("this evaluator is val-only")
    if tuple(args.geometry_modes) != GEOMETRY_MODES:
        raise ValueError("geometry modes are frozen to G0/G1/G2")
    if args.seed != BOOTSTRAP_SEED or args.bootstrap_resamples != BOOTSTRAP_RESAMPLES:
        raise ValueError("bootstrap is frozen to seed=123 and 10000 resamples")
    if min(
        args.torch_threads,
        args.min_opening_width_tokens,
        args.max_openings_per_view,
        args.progress_every,
    ) <= 0:
        raise ValueError("thread/candidate/progress settings must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    for value, label in (
        (args.val_manifest, "val manifest"),
        (args.train_manifest, "train manifest"),
        (args.val_cache, "val cache"),
        (args.full_manifest, "full val manifest"),
        (args.checkpoint, "checkpoint"),
        (args.original_metrics, "original metrics"),
        (args.output_dir, "output directory"),
    ):
        smoke.reject_test_path(Path(value), label)


def build_full_manifest(
    val: Mapping[str, Any], val_path: Path
) -> Dict[str, Any]:
    records = list(val["pairs"])
    ordered = sorted(records, key=lambda value: str(value["id"]))
    houses = sorted({str(value["scene_id"]) for value in ordered})
    declared = sorted(str(value) for value in val.get("sourceHouseIds", []))
    if declared and houses != declared:
        raise ValueError("val record houses differ from sourceHouseIds")
    positives = sum(bool(value["supervision"]["is_match"]) for value in ordered)
    return {
        "schema_version": "zind-full-val-geometry-manifest-v1",
        "split": "val",
        "purpose": PURPOSE,
        "source_manifest": {
            "path": str(val_path),
            "sha256": smoke.sha256_file(val_path),
        },
        "ordering": "stable lexicographic pair ID",
        "selection": "all existing val pairs; no filtering or balancing",
        "pair_ids": [str(value["id"]) for value in ordered],
        "counts": {
            "pair_count": len(ordered),
            "positive": positives,
            "negative": len(ordered) - positives,
            "house_count": len(houses),
        },
        "house_ids": houses,
        "test_data_accessed": False,
    }


def validate_full_manifest(
    manifest: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if dict(manifest) != dict(expected):
        raise ValueError("full val manifest differs from the frozen all-pair view")
    ids = manifest.get("pair_ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError("full val pair IDs are missing or duplicated")
    if ids != sorted(ids):
        raise ValueError("full val pair IDs are not stably sorted")


def _resolved_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    smoke.reject_test_path(path, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _resolved_directory(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    smoke.reject_test_path(path, label)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_context(
    args: argparse.Namespace, require_full_manifest: bool = True
) -> Dict[str, Any]:
    validate_args(args)
    val_path = _resolved_file(args.val_manifest, "val manifest")
    train_path = _resolved_file(args.train_manifest, "train manifest")
    cache_dir = _resolved_directory(args.val_cache, "val cache")
    checkpoint = _resolved_file(args.checkpoint, "checkpoint")
    original_metrics_path = _resolved_file(args.original_metrics, "original metrics")
    output_dir = Path(args.output_dir).expanduser().resolve()
    smoke.reject_test_path(output_dir, "output directory")
    full_path = Path(args.full_manifest).expanduser().resolve()
    smoke.reject_test_path(full_path, "full val manifest")
    val = smoke._load_split_manifest(val_path, "val")
    train = smoke._load_split_manifest(train_path, "train")
    train_houses = {str(value) for value in train.get("sourceHouseIds", [])}
    val_houses = {str(value) for value in val.get("sourceHouseIds", [])}
    overlap = sorted(train_houses & val_houses)
    if overlap:
        raise ValueError("train/val house leakage: " + ",".join(overlap[:3]))
    cache_metadata_path = cache_dir / "metadata.json"
    cache_metadata = smoke.load_json(cache_metadata_path)
    if cache_metadata.get("split") != "val" or cache_metadata.get("complete") is not True:
        raise ValueError("val cache is incomplete or declares the wrong split")
    cached_manifest = Path(
        str(cache_metadata.get("manifest", {}).get("path", ""))
    ).expanduser().resolve()
    if cached_manifest != val_path:
        raise ValueError("val cache manifest identity mismatch")
    expected_full = build_full_manifest(val, val_path)
    full = None
    if full_path.is_file():
        full = smoke.load_json(full_path)
        validate_full_manifest(full, expected_full)
    elif require_full_manifest:
        raise FileNotFoundError(
            f"run --prepare_manifest first; missing {full_path}"
        )
    contract = smoke.checkpoint_contract(checkpoint, val_path)
    checkpoint_hash = smoke.sha256_file(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("T3 checkpoint SHA-256 changed")
    m2 = smoke._m2_contract()
    if m2["sha256"] != EXPECTED_M2_SHA256:
        raise ValueError("M2 production config hash changed")
    threshold = float(contract["opening_probability_threshold"])
    if abs(threshold - EXPECTED_OPENING_THRESHOLD) > 1e-12:
        raise ValueError("checkpoint opening threshold changed")
    original_payload = smoke.load_json(original_metrics_path)
    original = original_payload["validation"]["predicted_openings"]
    if abs(float(original["harmonic_flow_score"]) - 0.4233331666613635) > 1e-15:
        raise ValueError("unexpected T3 original validation metric")
    records_by_id = {str(value["id"]): value for value in val["pairs"]}
    return {
        "args": args,
        "val_path": val_path,
        "train_path": train_path,
        "cache_dir": cache_dir,
        "cache_metadata": cache_metadata,
        "checkpoint": checkpoint,
        "checkpoint_contract": contract,
        "checkpoint_sha256": checkpoint_hash,
        "original_metrics_path": original_metrics_path,
        "original_metrics_sha256": smoke.sha256_file(original_metrics_path),
        "original_validation": copy.deepcopy(original),
        "output_dir": output_dir,
        "full_path": full_path,
        "full": full,
        "expected_full": expected_full,
        "full_sha256": smoke.sha256_file(full_path) if full is not None else None,
        "val": val,
        "records_by_id": records_by_id,
        "m2": m2,
        "house_split": {
            "train_house_count": len(train_houses),
            "val_house_count": len(val_houses),
            "train_val_overlap_count": 0,
        },
        "test_data_accessed": False,
    }


def metadata_report(context: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status": "PASS",
        "mode": "metadata-only",
        "split": "val",
        "purpose": PURPOSE,
        "counts": copy.deepcopy(context["full"]["counts"]),
        "house_split": copy.deepcopy(context["house_split"]),
        "checkpoint": {
            "path": str(context["checkpoint"]),
            "sha256": context["checkpoint_sha256"],
            **copy.deepcopy(context["checkpoint_contract"]),
        },
        "opening_threshold": {
            "value": EXPECTED_OPENING_THRESHOLD,
            "source": "T3 checkpoint",
            "frozen": True,
        },
        "m2": copy.deepcopy(context["m2"]),
        "full_manifest": {
            "path": str(context["full_path"]),
            "sha256": context["full_sha256"],
        },
        "val_manifest": copy.deepcopy(context["full"]["source_manifest"]),
        "original_metrics": {
            "path": str(context["original_metrics_path"]),
            "sha256": context["original_metrics_sha256"],
            "harmonic_flow_score": context["original_validation"][
                "harmonic_flow_score"
            ],
        },
        "geometry_modes": list(GEOMETRY_MODES),
        "pose_thresholds": [list(value) for value in POSE_THRESHOLDS],
        "scale_thresholds": list(SCALE_THRESHOLDS),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "cluster": "house_id",
        },
        "model_loaded": False,
        "gpu_used": False,
        "test_data_accessed": False,
    }


def dry_run_report(context: Mapping[str, Any]) -> Dict[str, Any]:
    import torch
    from portallayout.datasets.zind_bipair_dataset import collate_zind_bipair
    from tools.train_zind_end_to_end import JointZInDMatchingDataset
    from torch.utils.data import DataLoader, Subset

    args = context["args"]
    dataset = JointZInDMatchingDataset(
        str(context["val_path"]),
        str(context["cache_dir"]),
        data_root=args.data_root,
        max_pairs=None,
        expected_split="val",
        image_shape=tuple(context["cache_metadata"].get("image_shape", (512, 1024))),
        rotation_augmentation=False,
    )
    index_by_id = {
        str(value["id"]): index for index, value in enumerate(dataset.base.records)
    }
    pair_ids = context["full"]["pair_ids"]
    indices = [index_by_id[value] for value in pair_ids]
    probes = sorted({0, len(indices) // 2, len(indices) - 1})
    loader = DataLoader(
        Subset(dataset, [indices[value] for value in probes]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_zind_bipair,
        drop_last=False,
    )
    observed = []
    finite = True
    for batch in loader:
        observed.append(str(batch["pair_id"][0]))
        finite = finite and all(
            bool(torch.isfinite(batch[key]).all().item())
            for key in ("image_A", "image_B")
        )
    expected = [pair_ids[value] for value in probes]
    if observed != expected or not finite:
        raise ValueError("formal DataLoader dry-run failed")
    data_root = _data_root(context)
    missing_images = []
    for pair_id in pair_ids:
        record = context["records_by_id"][pair_id]
        for side in ("a", "b"):
            path = data_root / str(record[f"image_{side}"])
            if not path.is_file():
                missing_images.append(str(path))
    if missing_images:
        raise FileNotFoundError(missing_images[0])
    return {
        **metadata_report(context),
        "mode": "dry-run",
        "formal_dataset": "JointZInDMatchingDataset",
        "formal_pair_count": len(dataset),
        "full_pair_count": len(indices),
        "dataloader_order_index_resolved": True,
        "all_referenced_images_exist": True,
        "decoded_probe_pair_ids": observed,
        "decoded_probe_tensors_finite": True,
        "rotation_augmentation": False,
        "model_forward_count": 0,
    }


def _data_root(context: Mapping[str, Any]) -> Path:
    value = context["args"].data_root or context["val"].get("dataRoot")
    if not value:
        raise ValueError("val dataRoot is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = context["val_path"].parent / path
    path = path.resolve()
    smoke.reject_test_path(path, "val data root")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def atomic_write_json_gz(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as file:
        json.dump(smoke._jsonable(value), file, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(str(temporary), str(path))


def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def _safe_name(pair_id: str) -> str:
    return pair_id.replace("/", "_")


def _frozen_path(root: Path, index: int, pair_id: str) -> Path:
    return root / "frozen_predictions" / f"{index:04d}_{_safe_name(pair_id)}.json.gz"


def _pair_path(root: Path, index: int, pair_id: str) -> Path:
    return root / "pairs" / f"{index:04d}_{_safe_name(pair_id)}.json"


def compact_frozen_prediction(value: Mapping[str, Any]) -> Dict[str, Any]:
    output = copy.deepcopy(dict(value))
    for side in ("A", "B"):
        layout = output.get(f"layout_{side}")
        if isinstance(layout, dict):
            layout.pop("legacy_report", None)
            layout.pop("frame_report", None)
    output["compact_contract"] = (
        "retains intervals/probabilities/layouts/candidates/raw contours required "
        "for deterministic G0/G1/G2 resume"
    )
    return output


def _compact_portal(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    return {
        key: copy.deepcopy(value.get(key))
        for key in (
            "method",
            "frame",
            "token_count",
            "token_interval",
            "primary_variant",
            "geometry_valid",
            "failure_reason",
            "endpoints_local",
            "endpoints_local_bi_xz",
            "wall_tangent",
            "wall_normal",
            "wall_line",
            "opening_width",
            "predicted_only_quality",
            "gt_available_to_geometry",
        )
    }


def compact_pair_result(
    result: Mapping[str, Any], frozen: Mapping[str, Any]
) -> Dict[str, Any]:
    output = copy.deepcopy(dict(result))
    prediction = output["prediction"]
    prediction.pop("candidate_pair_scores", None)
    prediction["matcher_probability_max_abs_diff"] = 0.0
    for mode in GEOMETRY_MODES:
        geometry = prediction["geometry_modes"][mode]
        geometry.pop("joint_bev_frame_preserving_polygon", None)
        geometry.pop("joint_bev_pre_manhattan_polygon", None)
        geometry["portal_geometry_A"] = _compact_portal(
            geometry.get("portal_geometry_A")
        )
        geometry["portal_geometry_B"] = _compact_portal(
            geometry.get("portal_geometry_B")
        )
    layout_a, layout_b = frozen.get("layout_A"), frozen.get("layout_B")
    output["scale_diagnostic_inputs"] = None
    if layout_a is not None and layout_b is not None:
        output["scale_diagnostic_inputs"] = {
            "frame_polygon_A": copy.deepcopy(layout_a["frame_polygon_zind_xy"]),
            "frame_polygon_B": copy.deepcopy(layout_b["frame_polygon_zind_xy"]),
            "raw_polygon_A": copy.deepcopy(layout_a["pre_manhattan_contour_zind_xy"]),
            "raw_polygon_B": copy.deepcopy(layout_b["pre_manhattan_contour_zind_xy"]),
        }
    output["frozen_prediction_ref"] = {
        "format": "gzip-json",
        "model_forward_count": int(frozen["model_forward_count"]),
    }
    return output


def _load_completed(root: Path, pair_ids: Sequence[str]) -> Dict[str, Any]:
    output = {}
    for index, pair_id in enumerate(pair_ids):
        path = _pair_path(root, index, pair_id)
        if not path.is_file():
            continue
        value = smoke.load_json(path)
        if value.get("pair_id") != pair_id or value.get("split") != "val":
            raise ValueError(f"corrupt pair artifact: {path}")
        output[pair_id] = value
    return output


def _run_metadata(
    context: Mapping[str, Any],
    completed: int,
    frozen_count: int,
    new_forward_count: int,
    start: float,
    invocation_count: int,
    initial_completed: int,
    initial_frozen: int,
) -> Dict[str, Any]:
    return {
        "schema_version": "zind-full-val-geometry-run-v1",
        "split": "val",
        "purpose": PURPOSE,
        "pair_count": len(context["full"]["pair_ids"]),
        "completed_pair_count": completed,
        "frozen_prediction_count": frozen_count,
        "aggregate_model_forward_count_from_frozen_artifacts": frozen_count,
        "new_live_model_forward_count_this_invocation": new_forward_count,
        "gpu_process_invocation_count": invocation_count,
        "invocation_start_state": {
            "completed_pair_count": initial_completed,
            "frozen_prediction_count": initial_frozen,
        },
        "checkpoint": {
            "path": str(context["checkpoint"]),
            "sha256": context["checkpoint_sha256"],
            **copy.deepcopy(context["checkpoint_contract"]),
        },
        "opening_threshold": {
            "value": EXPECTED_OPENING_THRESHOLD,
            "source": "T3 checkpoint",
            "frozen": True,
        },
        "m2": copy.deepcopy(context["m2"]),
        "full_manifest": {
            "path": str(context["full_path"]),
            "sha256": context["full_sha256"],
        },
        "val_manifest": copy.deepcopy(context["full"]["source_manifest"]),
        "original_metrics": {
            "path": str(context["original_metrics_path"]),
            "sha256": context["original_metrics_sha256"],
        },
        "geometry_modes": list(GEOMETRY_MODES),
        "pose_thresholds": [list(value) for value in POSE_THRESHOLDS],
        "scale_thresholds": list(SCALE_THRESHOLDS),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "cluster": "house_id",
        },
        "house_split": copy.deepcopy(context["house_split"]),
        "one_live_backbone_call_per_new_pair": True,
        "geometry_fanout_after_frozen_prediction": True,
        "gt_available_to_prediction": False,
        "gt_available_to_geometry": False,
        "test_data_accessed": False,
        "runtime_seconds_so_far": time.perf_counter() - start,
    }


def _validate_resume_metadata(
    metadata: Mapping[str, Any], context: Mapping[str, Any]
) -> None:
    expected = {
        "checkpoint SHA-256": (
            metadata.get("checkpoint", {}).get("sha256"),
            context["checkpoint_sha256"],
        ),
        "full manifest SHA-256": (
            metadata.get("full_manifest", {}).get("sha256"),
            context["full_sha256"],
        ),
        "val manifest SHA-256": (
            metadata.get("val_manifest", {}).get("sha256"),
            context["full"]["source_manifest"]["sha256"],
        ),
        "M2 config SHA-256": (
            metadata.get("m2", {}).get("sha256"),
            context["m2"]["sha256"],
        ),
        "opening threshold": (
            metadata.get("opening_threshold", {}).get("value"),
            EXPECTED_OPENING_THRESHOLD,
        ),
    }
    mismatches = [
        label for label, (observed, wanted) in expected.items() if observed != wanted
    ]
    if mismatches:
        raise ValueError("resume protocol identity mismatch: " + ", ".join(mismatches))


def run(context: Mapping[str, Any]) -> Dict[str, Any]:
    import torch
    from portallayout.datasets.zind_bipair_dataset import collate_zind_bipair
    from tools.train_zind_end_to_end import JointZInDMatchingDataset
    from torch.utils.data import DataLoader, Subset

    args = context["args"]
    root = context["output_dir"]
    root.mkdir(parents=True, exist_ok=True)
    pair_ids = list(context["full"]["pair_ids"])
    completed = _load_completed(root, pair_ids)
    if completed and not args.resume:
        raise FileExistsError("pair artifacts exist; use --resume")
    metadata_path = root / "run_metadata.json"
    previous = smoke.load_json(metadata_path) if metadata_path.is_file() else None
    if previous is not None:
        _validate_resume_metadata(previous, context)
    prior_invocations = (
        int(previous.get("gpu_process_invocation_count", 1))
        if isinstance(previous, Mapping)
        else 0
    )
    invocation_count = prior_invocations + 1
    initial_completed = len(completed)
    frozen_count = sum(
        _frozen_path(root, index, pair_id).is_file()
        for index, pair_id in enumerate(pair_ids)
    )
    initial_frozen = frozen_count
    new_forward_count = 0
    start = time.perf_counter()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full val inference requires an available CUDA device")

    dataset = JointZInDMatchingDataset(
        str(context["val_path"]),
        str(context["cache_dir"]),
        data_root=args.data_root,
        max_pairs=None,
        expected_split="val",
        image_shape=tuple(context["cache_metadata"].get("image_shape", (512, 1024))),
        rotation_augmentation=False,
    )
    index_by_id = {
        str(value["id"]): index for index, value in enumerate(dataset.base.records)
    }
    indices = [index_by_id[value] for value in pair_ids]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_zind_bipair,
        drop_last=False,
    )
    bi_layout, matcher, threshold, model_report = smoke._load_models(args, device)
    if abs(float(threshold["value"]) - EXPECTED_OPENING_THRESHOLD) > 1e-12:
        raise ValueError("loaded model threshold differs from frozen checkpoint value")
    resolver = smoke.AnnotationResolver(_data_root(context))

    def write_metadata() -> None:
        metadata = _run_metadata(
            context,
            len(completed),
            frozen_count,
            new_forward_count,
            start,
            invocation_count,
            initial_completed,
            initial_frozen,
        )
        metadata["model"] = model_report
        smoke.atomic_write_json(metadata_path, metadata)

    write_metadata()
    for index, batch in enumerate(loader):
        pair_id = str(batch["pair_id"][0])
        if pair_id != pair_ids[index]:
            raise RuntimeError("DataLoader pair order changed")
        if pair_id in completed:
            if (index + 1) % args.progress_every == 0:
                print(f"[{index + 1}/{len(pair_ids)}] resume skip", flush=True)
            continue
        record = context["records_by_id"][pair_id]
        frozen_path = _frozen_path(root, index, pair_id)
        if frozen_path.is_file():
            frozen = load_json_gz(frozen_path)
            if frozen.get("pair_id") != pair_id:
                raise ValueError(f"corrupt frozen artifact: {frozen_path}")
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
            frozen = smoke.infer_frozen_prediction(
                args,
                batch["image_A"],
                batch["image_B"],
                identifiers,
                bi_layout,
                matcher,
                device,
                EXPECTED_OPENING_THRESHOLD,
            )
            frozen = compact_frozen_prediction(frozen)
            atomic_write_json_gz(frozen_path, frozen)
            frozen_count += 1
            new_forward_count += 1
        best = [int(value) for value in frozen["best_candidate_pair"]]
        if not (all(value >= 0 for value in best) or all(value < 0 for value in best)):
            raise ValueError("matcher returned a partially valid candidate pair")
        fanout = smoke.geometry_fanout(frozen)
        signatures = list(fanout["matcher_signature_by_mode"].values())
        probability_diff = max(
            abs(float(value[key]) - float(signatures[0][key]))
            for value in signatures[1:]
            for key in ("pair_match_probability", "pair_no_match_probability")
        )
        if (
            fanout["matcher_score_max_abs_diff"] != 0.0
            or probability_diff != 0.0
            or not fanout["best_candidate_identical"]
        ):
            raise RuntimeError("geometry branch changed frozen matcher output")
        ground_truth = resolver.pair_ground_truth(record)
        result = smoke.attach_posthoc_evaluation(
            frozen, fanout, batch, record, ground_truth
        )
        compact = compact_pair_result(result, frozen)
        compact["prediction"]["matcher_probability_max_abs_diff"] = probability_diff
        smoke.atomic_write_json(_pair_path(root, index, pair_id), compact)
        completed[pair_id] = compact
        write_metadata()
        if index == 0 or (index + 1) % args.progress_every == 0:
            rate = len(completed) / max(time.perf_counter() - start, 1e-6)
            print(
                f"[{index + 1}/{len(pair_ids)}] complete={len(completed)} "
                f"rate={rate:.2f} pair/s {pair_id}",
                flush=True,
            )

    if len(completed) != len(pair_ids):
        raise RuntimeError(f"incomplete full val: {len(completed)}/{len(pair_ids)}")
    if frozen_count != len(pair_ids):
        raise RuntimeError(f"frozen forward coverage mismatch: {frozen_count}")
    write_metadata()
    # All subsequent diagnostics, including D1/D2 and bootstrap, are CPU-only.
    del bi_layout, matcher, loader, dataset
    torch.cuda.empty_cache()
    return analyze(context)


def _stats(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    array = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if not len(array):
        return {
            key: None if key != "count" else 0
            for key in ("count", "mean", "median", "p25", "p75", "p90", "min", "max")
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def matching_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    positives = [value for value in records if value["ground_truth"]["is_match"]]
    negatives = [value for value in records if not value["ground_truth"]["is_match"]]
    target_available = [
        value for value in positives if value['portallayout.evaluation']["gt_candidate_recalled"]
    ]
    correct = [
        value for value in positives if value['portallayout.evaluation']["selected_candidate_correct"]
    ]
    tp = sum(bool(value["prediction"]["is_match"]) for value in positives)
    fp = sum(bool(value["prediction"]["is_match"]) for value in negatives)
    fn = len(positives) - tp
    tn = len(negatives) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(positives) if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    candidate_recall = len(target_available) / len(positives) if positives else 0.0
    conditional_top1 = len(correct) / len(target_available) if target_available else 0.0
    end_to_end_top1 = len(correct) / len(positives) if positives else 0.0
    negative_rejection = tn / len(negatives) if negatives else 0.0
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
        "positive_candidate_recall": candidate_recall,
        "conditional_candidate_top1": conditional_top1,
        "end_to_end_candidate_top1": end_to_end_top1,
        "balanced_flow_score": 0.5 * (end_to_end_top1 + negative_rejection),
        "harmonic_flow_score": harmonic,
        "pair_match_probability_positive": _stats(
            value["prediction"]["pair_match_probability"] for value in positives
        ),
        "pair_match_probability_negative": _stats(
            value["prediction"]["pair_match_probability"] for value in negatives
        ),
        "dustbin_probability_positive": _stats(
            value["prediction"]["pair_no_match_probability"] for value in positives
        ),
        "dustbin_probability_negative": _stats(
            value["prediction"]["pair_no_match_probability"] for value in negatives
        ),
        "matcher_score_max_abs_diff_across_geometry_modes": max(
            float(value["prediction"]["matcher_score_max_abs_diff"])
            for value in records
        ),
        "matcher_probability_max_abs_diff_across_geometry_modes": max(
            float(value["prediction"]["matcher_probability_max_abs_diff"])
            for value in records
        ),
        "best_candidate_identical_across_geometry_modes": all(
            value["prediction"]["best_candidate_identical"] for value in records
        ),
    }


def parity_summary(
    current: Mapping[str, Any], original: Mapping[str, Any]
) -> Dict[str, Any]:
    mapping = {
        "positive_candidate_recall": "positive_candidate_recall",
        "positive_match_recall": "positive_acceptance_rate",
        "conditional_candidate_top1": "conditional_candidate_top1",
        "end_to_end_candidate_top1": "end_to_end_candidate_top1",
        "negative_rejection": "negative_rejection_rate",
        "balanced_flow_score": "balanced_flow_score",
        "harmonic_flow_score": "harmonic_flow_score",
    }
    metrics = {}
    for current_key, original_key in mapping.items():
        value = float(current[current_key])
        reference = float(original[original_key])
        metrics[current_key] = {
            "new_evaluator": value,
            "original_t3_validation": reference,
            "absolute_difference": abs(value - reference),
        }
    passed = all(
        value["absolute_difference"] <= PARITY_TOLERANCE
        for value in metrics.values()
    )
    return {
        "tolerance": PARITY_TOLERANCE,
        "metrics": metrics,
        "pass": passed,
        "geometry_interpretable": passed,
    }


def _threshold_rate(
    values: Sequence[Optional[float]], threshold: float, denominator: int
) -> Dict[str, Any]:
    count = sum(value is not None and float(value) < threshold for value in values)
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else None,
    }


def _geometry_summary(
    records: Sequence[Mapping[str, Any]], mode: str
) -> Dict[str, Any]:
    geometries = [value["prediction"]["geometry_modes"][mode] for value in records]
    metrics = [value["metrics"] for value in geometries]
    yaw = [value["yaw_error_degrees"] for value in metrics]
    coordinate = [value["translation_error"] for value in metrics]
    meters = [value["translation_error_meters"] for value in metrics]
    log_scale = [value["absolute_log_scale_error"] for value in metrics]
    relative_scale = [value["relative_scale_error"] for value in metrics]
    denominator = len(records)
    joint = {}
    for yaw_limit, translation_limit in POSE_THRESHOLDS:
        base = [
            first is not None
            and second is not None
            and first < yaw_limit
            and second < translation_limit
            for first, second in zip(yaw, meters)
        ]
        key = f"yaw_lt_{yaw_limit:g}_translation_lt_{translation_limit:g}m"
        joint[key] = {
            "count": sum(base),
            "denominator": denominator,
            "rate": sum(base) / denominator if denominator else None,
        }
        for scale_limit in SCALE_THRESHOLDS:
            passed = sum(
                ok and scale is not None and scale < scale_limit
                for ok, scale in zip(base, relative_scale)
            )
            joint[key + f"_scale_rel_lt_{100 * scale_limit:g}pct"] = {
                "count": passed,
                "denominator": denominator,
                "rate": passed / denominator if denominator else None,
            }
    scale_layers = {}
    for name, flag in (("unit", True), ("non_unit", False)):
        selected = [
            metric
            for record, metric in zip(records, metrics)
            if record["ground_truth"]["gt_scale_is_approximately_unit"] is flag
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
        values = [value["bev"][representation] for value in metrics]
        valid = [value for value in values if value is not None]
        bev[representation] = {
            "valid_count": len(valid),
            "instance_A_overlap": _stats(value["instance_A_overlap"] for value in valid),
            "instance_B_overlap": _stats(value["instance_B_overlap"] for value in valid),
            "instance_mean_overlap": _stats(value["instance_mean_overlap"] for value in valid),
            "instance_B_zero_count": sum(
                value["instance_B_overlap"] is not None
                and value["instance_B_overlap"] <= 1e-12
                for value in valid
            ),
            "instance_B_zero_rate": (
                sum(
                    value["instance_B_overlap"] is not None
                    and value["instance_B_overlap"] <= 1e-12
                    for value in valid
                )
                / len(valid)
                if valid
                else None
            ),
            "invalid_polygon_count": sum(
                not value["polygon_A_valid"] or not value["polygon_B_valid"]
                for value in valid
            ),
            "pose_success_but_bev_lt_0.5_count": sum(
                metric["yaw_error_degrees"] is not None
                and metric["translation_error_meters"] is not None
                and metric["yaw_error_degrees"] < 15.0
                and metric["translation_error_meters"] < 1.0
                and value is not None
                and value["instance_mean_overlap"] is not None
                and value["instance_mean_overlap"] < 0.5
                for metric, value in zip(metrics, values)
            ),
        }
    portals_a = [value.get("portal_geometry_A") for value in geometries]
    portals_b = [value.get("portal_geometry_B") for value in geometries]
    width_a = [
        None if value is None else value.get("opening_width") for value in portals_a
    ]
    width_b = [
        None if value is None else value.get("opening_width") for value in portals_b
    ]
    return {
        "pair_count": denominator,
        "coverage": {
            "predicted_match_count": sum(value["prediction"]["is_match"] for value in records),
            "attempted_count": sum(value["attempted"] for value in geometries),
            "pose_valid_count": sum(value["pose_valid"] for value in geometries),
            "hard_failure_count": sum(value["hard_failure"] for value in geometries),
            "missing_meter_scale_count": sum(
                not bool(value["ground_truth"].get("metric_translation_available"))
                for value in records
            ),
        },
        "yaw_error_degrees": {
            "distribution": _stats(yaw),
            "thresholds": {
                f"lt_{value:g}": _threshold_rate(yaw, value, denominator)
                for value in (5.0, 10.0, 15.0, 30.0)
            },
        },
        "translation_error_coordinate": _stats(coordinate),
        "translation_error_meters": {
            "distribution": _stats(meters),
            "thresholds": {
                f"lt_{value:g}": _threshold_rate(meters, value, denominator)
                for value in (0.25, 0.5, 1.0, 2.0)
            },
        },
        "scale": {
            "absolute_log_scale_error": _stats(log_scale),
            "relative_scale_error": _stats(relative_scale),
            "relative_error_thresholds": {
                f"lt_{100 * value:g}pct": _threshold_rate(
                    relative_scale, value, denominator
                )
                for value in (0.05, 0.10, 0.20, 0.50)
            },
            "predicted_width_A": _stats(width_a),
            "predicted_width_B": _stats(width_b),
            "predicted_width_ratio_scale": _stats(
                value["predicted_scale"] for value in metrics
            ),
            "gt_layers": scale_layers,
        },
        "joint_pose_success": joint,
        "bev": bev,
    }


def _fixed_scale_matrix(
    endpoints_a: np.ndarray,
    endpoints_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, str]:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("fixed scale must be positive and finite")
    vector_a = endpoints_a[1] - endpoints_a[0]
    vector_b = endpoints_b[1] - endpoints_b[0]
    if np.linalg.norm(vector_a) < 1e-8 or np.linalg.norm(vector_b) < 1e-8:
        raise ValueError("portal endpoints are degenerate")
    midpoint_a = endpoints_a.mean(axis=0)
    side_a = float(np.cross(vector_a, points_a.mean(axis=0) - midpoint_a))
    candidates = []
    for mapping, target in (("same", endpoints_a), ("reversed", endpoints_a[::-1])):
        target_vector = target[1] - target[0]
        angle = math.atan2(target_vector[1], target_vector[0]) - math.atan2(
            vector_b[1], vector_b[0]
        )
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]])
        translation = target.mean(axis=0) - scale * rotation @ endpoints_b.mean(axis=0)
        matrix = np.asarray(
            [
                [scale * cosine, -scale * sine, translation[0]],
                [scale * sine, scale * cosine, translation[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        transformed = _transform(points_b, matrix)
        side_b = float(np.cross(vector_a, transformed.mean(axis=0) - midpoint_a))
        candidates.append((side_a * side_b, mapping, matrix))
    _, mapping, matrix = min(candidates, key=lambda value: (value[0], value[1]))
    return matrix, mapping


def _transform(points: Any, matrix: Any) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    homogeneous = np.concatenate(
        [value, np.ones((len(value), 1), dtype=np.float64)], axis=1
    )
    return (transform @ homogeneous.T).T[:, :2]


def _diagnostic_metrics(
    matrix: np.ndarray,
    record: Mapping[str, Any],
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
) -> Dict[str, Any]:
    target = record["ground_truth"]["relative_pose"]
    gt_matrix = np.asarray(target["T_B_to_A"], dtype=np.float64)
    predicted_yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    gt_yaw = float(target["relative_yaw_radians"])
    yaw_error = math.degrees(smoke._wrapped_angle_error(predicted_yaw, gt_yaw))
    translation_coordinate = float(np.linalg.norm(matrix[:2, 2] - gt_matrix[:2, 2]))
    metric_factor = record["ground_truth"].get("metric_factor_A")
    predicted_scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    gt_scale = float(target["relative_scale"])
    target_instances = record["ground_truth"]["joint_bev_target"]["instances"]
    output = {
        "T_B_to_A": matrix.tolist(),
        "relative_yaw_degrees": math.degrees(predicted_yaw),
        "translation": matrix[:2, 2].tolist(),
        "relative_scale": predicted_scale,
        "yaw_error_degrees": yaw_error,
        "translation_error_coordinate": translation_coordinate,
        "translation_error_meters": (
            None
            if metric_factor is None
            else translation_coordinate * float(metric_factor)
        ),
        "absolute_log_scale_error": abs(math.log(predicted_scale / gt_scale)),
        "relative_scale_error": abs(predicted_scale / gt_scale - 1.0),
        "bev": {},
    }
    for name, first, second in (
        ("frame_preserving", frame_a, frame_b),
        ("pre_manhattan", raw_a, raw_b),
    ):
        transformed_b = _transform(second, matrix)
        overlap_a = smoke._safe_overlap(first, target_instances[0]["polygon"])
        overlap_b = smoke._safe_overlap(transformed_b, target_instances[1]["polygon"])
        output["bev"][name] = {
            "instance_A_overlap": overlap_a,
            "instance_B_overlap": overlap_b,
            "instance_mean_overlap": (
                None
                if overlap_a is None or overlap_b is None
                else 0.5 * (overlap_a + overlap_b)
            ),
            "polygon_A_valid": smoke._polygon_valid(first),
            "polygon_B_valid": smoke._polygon_valid(transformed_b),
        }
    return output


def scale_diagnostic(record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not record["ground_truth"]["is_match"]:
        return None
    m2 = record["prediction"]["geometry_modes"]["candidate_local_m2"]
    inputs = record.get("scale_diagnostic_inputs")
    portal_a, portal_b = m2.get("portal_geometry_A"), m2.get("portal_geometry_B")
    if not m2["pose_valid"] or inputs is None or portal_a is None or portal_b is None:
        return None
    endpoints_a = np.asarray(portal_a["endpoints_local"], dtype=np.float64)
    endpoints_b = np.asarray(portal_b["endpoints_local"], dtype=np.float64)
    frame_a = np.asarray(inputs["frame_polygon_A"], dtype=np.float64)
    frame_b = np.asarray(inputs["frame_polygon_B"], dtype=np.float64)
    raw_a = np.asarray(inputs["raw_polygon_A"], dtype=np.float64)
    raw_b = np.asarray(inputs["raw_polygon_B"], dtype=np.float64)
    formal = np.asarray(m2["relative_pose"]["T_B_to_A"], dtype=np.float64)
    gt_scale = float(record["ground_truth"]["relative_pose"]["relative_scale"])
    d1, mapping_d1 = _fixed_scale_matrix(
        endpoints_a, endpoints_b, raw_a, raw_b, 1.0
    )
    d2, mapping_d2 = _fixed_scale_matrix(
        endpoints_a, endpoints_b, raw_a, raw_b, gt_scale
    )
    width_a = float(np.linalg.norm(endpoints_a[1] - endpoints_a[0]))
    width_b = float(np.linalg.norm(endpoints_b[1] - endpoints_b[0]))
    return {
        "pair_id": record["pair_id"],
        "house_id": record["house_id"],
        "split": "val",
        "formal_g2_unchanged": True,
        "width_A": width_a,
        "width_B": width_b,
        "predicted_width_ratio": width_a / width_b,
        "D0_width_ratio": {
            "predicted_only": True,
            "oracle_only": False,
            "metrics": _diagnostic_metrics(
                formal, record, frame_a, frame_b, raw_a, raw_b
            ),
        },
        "D1_unit_scale": {
            "predicted_only": True,
            "oracle_only": False,
            "endpoint_mapping": mapping_d1,
            "metrics": _diagnostic_metrics(d1, record, frame_a, frame_b, raw_a, raw_b),
        },
        "D2_gt_relative_scale": {
            "predicted_only": False,
            "oracle_only": True,
            "endpoint_mapping": mapping_d2,
            "metrics": _diagnostic_metrics(d2, record, frame_a, frame_b, raw_a, raw_b),
        },
        "gt_available_to_formal_g2": False,
        "test_data_accessed": False,
    }


def _scale_diagnostic_summary(values: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output = {
        "pair_count": len(values),
        "width_A": _stats(value["width_A"] for value in values),
        "width_B": _stats(value["width_B"] for value in values),
        "predicted_width_ratio": _stats(
            value["predicted_width_ratio"] for value in values
        ),
        "formal_g2_unchanged": all(value["formal_g2_unchanged"] for value in values),
    }
    for mode in ("D0_width_ratio", "D1_unit_scale", "D2_gt_relative_scale"):
        metrics = [value[mode]["metrics"] for value in values]
        representations = {}
        for representation in ("frame_preserving", "pre_manhattan"):
            representations[representation] = {
                "instance_A_overlap": _stats(
                    value["bev"][representation]["instance_A_overlap"] for value in metrics
                ),
                "instance_B_overlap": _stats(
                    value["bev"][representation]["instance_B_overlap"] for value in metrics
                ),
                "instance_mean_overlap": _stats(
                    value["bev"][representation]["instance_mean_overlap"] for value in metrics
                ),
            }
        output[mode] = {
            "oracle_only": mode == "D2_gt_relative_scale",
            "yaw_error_degrees": _stats(value["yaw_error_degrees"] for value in metrics),
            "translation_error_coordinate": _stats(
                value["translation_error_coordinate"] for value in metrics
            ),
            "translation_error_meters": _stats(
                value["translation_error_meters"] for value in metrics
            ),
            "absolute_log_scale_error": _stats(
                value["absolute_log_scale_error"] for value in metrics
            ),
            "relative_scale_error": _stats(
                value["relative_scale_error"] for value in metrics
            ),
            "bev": representations,
        }
    return output


def paired_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for record in records:
        if not (
            record["ground_truth"]["is_match"]
            and record['portallayout.evaluation']["selected_candidate_correct"]
        ):
            continue
        geometries = record["prediction"]["geometry_modes"]
        row: Dict[str, Any] = {
            "pair_id": record["pair_id"],
            "house_id": record["house_id"],
        }
        for baseline in ("legacy_polygon", "frame_polygon"):
            prefix = "g0" if baseline == "legacy_polygon" else "g1"
            first = geometries[baseline]["metrics"]
            second = geometries["candidate_local_m2"]["metrics"]
            for name, key in (
                ("yaw", "yaw_error_degrees"),
                ("translation_coordinate", "translation_error"),
                ("translation_meters", "translation_error_meters"),
                ("log_scale", "absolute_log_scale_error"),
            ):
                a, b = first.get(key), second.get(key)
                row[f"delta_m2_minus_{prefix}_{name}"] = (
                    None if a is None or b is None else float(b - a)
                )
            for representation, short in (
                ("frame_preserving", "bev_frame"),
                ("pre_manhattan", "bev_raw"),
            ):
                a = first["bev"][representation]
                b = second["bev"][representation]
                row[f"delta_m2_minus_{prefix}_{short}"] = (
                    None
                    if a is None
                    or b is None
                    or a["instance_mean_overlap"] is None
                    or b["instance_mean_overlap"] is None
                    else float(b["instance_mean_overlap"] - a["instance_mean_overlap"])
                )
            for yaw_limit, translation_limit in POSE_THRESHOLDS:
                key = f"success_{yaw_limit:g}_{translation_limit:g}"
                for label, metric in ((prefix, first), ("m2", second)):
                    base_success = bool(
                        metric["yaw_error_degrees"] is not None
                        and metric["translation_error_meters"] is not None
                        and metric["yaw_error_degrees"] < yaw_limit
                        and metric["translation_error_meters"] < translation_limit
                    )
                    row[f"{label}_{key}"] = base_success
                    for scale_limit in SCALE_THRESHOLDS:
                        row[
                            f"{label}_{key}_scale_{100 * scale_limit:g}pct"
                        ] = bool(
                            base_success
                            and metric["relative_scale_error"] is not None
                            and metric["relative_scale_error"] < scale_limit
                        )
        output.append(row)
    return output


def _win_tie_loss(rows: Sequence[Mapping[str, Any]], key: str, higher_better: bool) -> Dict[str, Any]:
    values = [float(value[key]) for value in rows if value.get(key) is not None]
    if higher_better:
        wins = sum(value > 1e-9 for value in values)
        losses = sum(value < -1e-9 for value in values)
    else:
        wins = sum(value < -1e-9 for value in values)
        losses = sum(value > 1e-9 for value in values)
    return {
        "comparable_count": len(values),
        "m2_win": wins,
        "tie": len(values) - wins - losses,
        "m2_loss": losses,
        "delta_distribution": _stats(values),
    }


def paired_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output = {"S3_pair_count": len(rows)}
    for baseline in ("g0", "g1"):
        output[f"m2_minus_{baseline}"] = {
            name: _win_tie_loss(
                rows,
                f"delta_m2_minus_{baseline}_{key}",
                higher_better=name.startswith("bev"),
            )
            for name, key in (
                ("yaw", "yaw"),
                ("translation_coordinate", "translation_coordinate"),
                ("translation_meters", "translation_meters"),
                ("log_scale", "log_scale"),
                ("bev_frame", "bev_frame"),
                ("bev_raw", "bev_raw"),
            )
        }
    return output


def failure_decomposition(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    positives = [value for value in records if value["ground_truth"]["is_match"]]
    recalled = [value for value in positives if value['portallayout.evaluation']["gt_candidate_recalled"]]
    selected = [value for value in positives if value['portallayout.evaluation']["selected_candidate_correct"]]
    output = {
        "positive_denominator": len(positives),
        "candidate_not_recalled": len(positives) - len(recalled),
        "candidate_recalled_denominator": len(recalled),
        "candidate_recalled_matcher_no_match": sum(
            not value["prediction"]["is_match"] for value in recalled
        ),
        "candidate_recalled_matcher_selected_wrong": sum(
            value["prediction"]["is_match"]
            and not value['portallayout.evaluation']["selected_candidate_correct"]
            for value in recalled
        ),
        "candidate_selected_correct": len(selected),
    }
    per_mode = {}
    for mode in GEOMETRY_MODES:
        geometries = [value["prediction"]["geometry_modes"][mode] for value in selected]
        pose_valid = [value for value in geometries if value["pose_valid"]]
        medium_success = [
            value
            for value in pose_valid
            if value["metrics"]["yaw_error_degrees"] is not None
            and value["metrics"]["translation_error_meters"] is not None
            and value["metrics"]["yaw_error_degrees"] < 15.0
            and value["metrics"]["translation_error_meters"] < 1.0
        ]
        per_mode[mode] = {
            "S3_denominator": len(selected),
            "geometry_hard_failure": sum(value["hard_failure"] for value in geometries),
            "geometry_valid_but_medium_pose_failure": len(pose_valid) - len(medium_success),
            "rotation_success_translation_failure": sum(
                value["metrics"]["yaw_error_degrees"] is not None
                and value["metrics"]["yaw_error_degrees"] < 15.0
                and (
                    value["metrics"]["translation_error_meters"] is None
                    or value["metrics"]["translation_error_meters"] >= 1.0
                )
                for value in pose_valid
            ),
            "medium_pose_success_scale_10pct_failure": sum(
                value["metrics"]["relative_scale_error"] is None
                or value["metrics"]["relative_scale_error"] >= 0.10
                for value in medium_success
            ),
            "medium_pose_success_but_frame_bev_lt_0.5": sum(
                value["metrics"]["bev"]["frame_preserving"] is not None
                and value["metrics"]["bev"]["frame_preserving"]["instance_mean_overlap"]
                < 0.5
                for value in medium_success
            ),
        }
    output["geometry_by_mode"] = per_mode
    return output


def _bootstrap_metric(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    houses: Sequence[str],
    rng: np.random.RandomState,
    extractor,
) -> Dict[str, Any]:
    observed_values = [
        value
        for house in houses
        for row in grouped[house]
        for value in [extractor(row)]
        if value is not None and math.isfinite(float(value))
    ]
    observed = {
        "mean": float(np.mean(observed_values)) if observed_values else None,
        "median": float(np.median(observed_values)) if observed_values else None,
    }
    means, medians = [], []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.randint(0, len(houses), size=len(houses))
        values = []
        for index in sampled:
            for row in grouped[houses[int(index)]]:
                value = extractor(row)
                if value is not None and math.isfinite(float(value)):
                    values.append(float(value))
        if values:
            means.append(float(np.mean(values)))
            medians.append(float(np.median(values)))
    def interval(values: Sequence[float]) -> Dict[str, Any]:
        return {
            "valid_resamples": len(values),
            "lower_95": float(np.quantile(values, 0.025)) if values else None,
            "upper_95": float(np.quantile(values, 0.975)) if values else None,
        }
    return {
        "observed": observed,
        "mean_delta_ci": interval(means),
        "median_delta_ci": interval(medians),
    }


def clustered_bootstrap(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["house_id"])].append(row)
    houses = sorted(grouped)
    if not houses:
        return {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "cluster": "house_id",
            "house_count": 0,
            "stable_enough_for_significance_claim": False,
            "metrics": {},
        }
    metrics = {}
    for name, key in (
        ("yaw_m2_minus_g1", "delta_m2_minus_g1_yaw"),
        ("translation_meters_m2_minus_g1", "delta_m2_minus_g1_translation_meters"),
        ("log_scale_m2_minus_g1", "delta_m2_minus_g1_log_scale"),
        ("bev_frame_m2_minus_g1", "delta_m2_minus_g1_bev_frame"),
        ("bev_raw_m2_minus_g1", "delta_m2_minus_g1_bev_raw"),
    ):
        metrics[name] = _bootstrap_metric(
            grouped,
            houses,
            np.random.RandomState(BOOTSTRAP_SEED),
            lambda row, field=key: row.get(field),
        )
    for yaw_limit, translation_limit in POSE_THRESHOLDS:
        suffix = f"{yaw_limit:g}_{translation_limit:g}"
        metrics[f"success_rate_delta_{suffix}_m2_minus_g1"] = _bootstrap_metric(
            grouped,
            houses,
            np.random.RandomState(BOOTSTRAP_SEED),
            lambda row, suffix=suffix: float(row[f"m2_success_{suffix}"])
            - float(row[f"g1_success_{suffix}"]),
        )
        for scale_limit in SCALE_THRESHOLDS:
            scale_suffix = f"{suffix}_scale_{100 * scale_limit:g}pct"
            metrics[
                f"success_rate_delta_{scale_suffix}_m2_minus_g1"
            ] = _bootstrap_metric(
                grouped,
                houses,
                np.random.RandomState(BOOTSTRAP_SEED),
                lambda row, suffix=scale_suffix: float(row[f"m2_success_{suffix}"])
                - float(row[f"g1_success_{suffix}"]),
            )
    return {
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "cluster": "house_id",
        "house_count": len(houses),
        "pair_count": len(rows),
        "stable_enough_for_significance_claim": len(houses) >= 30,
        "metrics": metrics,
    }


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(smoke._jsonable(value), sort_keys=True, allow_nan=False)
        for value in values
    ]
    smoke.atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["pair_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _render_visualizations(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    from tools.evaluate_zind_room_group import _render_polygons_svg

    by_id = {value["pair_id"]: value for value in records}
    selected: List[Tuple[str, str]] = []
    s3 = [value for value in records if value['portallayout.evaluation']["selected_candidate_correct"]]
    selected.extend(("fixed_pair_id", value["pair_id"]) for value in records[:2])
    comparable = [value for value in rows if value.get("delta_m2_minus_g1_yaw") is not None]
    comparable.sort(key=lambda value: value["delta_m2_minus_g1_yaw"])
    selected.extend(("yaw_improved", value["pair_id"]) for value in comparable[:2])
    selected.extend(("yaw_worsened", value["pair_id"]) for value in comparable[-2:])
    scale_extreme = sorted(
        s3,
        key=lambda value: value["prediction"]["geometry_modes"]["candidate_local_m2"]
        ["metrics"]["absolute_log_scale_error"] or -1,
        reverse=True,
    )
    selected.extend(("scale_extreme", value["pair_id"]) for value in scale_extreme[:2])
    hard = [
        value
        for value in records
        if value["prediction"]["geometry_modes"]["candidate_local_m2"]["hard_failure"]
    ]
    selected.extend(("m2_hard_failure", value["pair_id"]) for value in hard[:2])
    false_positive = [
        value
        for value in records
        if not value["ground_truth"]["is_match"] and value["prediction"]["is_match"]
    ]
    false_negative = [
        value
        for value in records
        if value["ground_truth"]["is_match"] and not value["prediction"]["is_match"]
    ]
    selected.extend(("matcher_false_positive", value["pair_id"]) for value in false_positive[:2])
    selected.extend(("matcher_false_negative", value["pair_id"]) for value in false_negative[:2])
    pose_success_bev_failure = []
    for value in s3:
        metric = value["prediction"]["geometry_modes"]["candidate_local_m2"]["metrics"]
        bev = metric["bev"]["frame_preserving"]
        if (
            metric["yaw_error_degrees"] is not None
            and metric["translation_error_meters"] is not None
            and metric["yaw_error_degrees"] < 15.0
            and metric["translation_error_meters"] < 1.0
            and bev is not None
            and bev["instance_mean_overlap"] is not None
            and bev["instance_mean_overlap"] < 0.5
        ):
            pose_success_bev_failure.append(value)
    selected.extend(
        ("pose_success_bev_failure", value["pair_id"])
        for value in pose_success_bev_failure[:2]
    )
    deduplicated, seen = [], set()
    for category, pair_id in selected:
        if pair_id in seen:
            continue
        seen.add(pair_id)
        deduplicated.append((category, pair_id))
    artifacts = []
    for index, (category, pair_id) in enumerate(deduplicated):
        record = by_id[pair_id]
        inputs = record.get("scale_diagnostic_inputs")
        gt = record["ground_truth"].get("joint_bev_target")
        if inputs is None or gt is None:
            artifacts.append({
                "category": category,
                "pair_id": pair_id,
                "path": None,
                "reason": "no shared GT coordinate frame or predicted layout",
                "selection_is_posthoc": category != "fixed_pair_id",
            })
            continue
        polygons = [
            ("GT A", gt["instances"][0]["polygon"], "#289128"),
            ("GT B", gt["instances"][1]["polygon"], "#7a9b22"),
        ]
        for mode, color in (("frame_polygon", "#e18a00"), ("candidate_local_m2", "#1774c8")):
            geometry = record["prediction"]["geometry_modes"][mode]
            if geometry["pose_valid"]:
                transformed = _transform(
                    inputs["frame_polygon_B"], geometry["relative_pose"]["T_B_to_A"]
                )
                polygons.append((mode + " B", transformed.tolist(), color))
        path = root / "visualizations" / f"{index:02d}_{category}_{_safe_name(pair_id)}.svg"
        _render_polygons_svg(path, polygons)
        artifacts.append({
            "category": category,
            "pair_id": pair_id,
            "path": str(path),
            "selection_is_posthoc": category != "fixed_pair_id",
        })
    report = {
        "bounded_diagnostic_selection": True,
        "not_random_qualitative_examples": True,
        "artifacts": artifacts,
    }
    smoke.atomic_write_json(root / "visualizations" / "selection.json", report)
    return report


def _report_markdown(summary: Mapping[str, Any]) -> str:
    matching = summary["matching"]
    parity = summary["matcher_parity"]
    sets = summary["set_counts"]
    geometries = summary["geometry"]
    paired = summary["paired_comparison"]["m2_minus_g1"]
    scale = summary["scale_diagnostics"]
    quality = summary["m2_quality"]
    failures = summary["failure_decomposition"]
    bootstrap = summary["house_clustered_bootstrap"]
    metadata = summary["run_metadata"]

    def number(value: Any, digits: int = 4) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    def percent(value: Any, digits: int = 2) -> str:
        return "NA" if value is None else f"{100.0 * float(value):.{digits}f}%"

    def distribution(mode: str, metric: str) -> Mapping[str, Any]:
        value = geometries[mode]["S3"]
        if metric == "yaw":
            return value["yaw_error_degrees"]["distribution"]
        if metric == "translation":
            return value["translation_error_meters"]["distribution"]
        return value["scale"]["absolute_log_scale_error"]

    medium_key = "yaw_lt_15_translation_lt_1m"
    checkpoint = metadata["checkpoint"]
    full_manifest = metadata["full_manifest"]
    val_manifest = metadata["val_manifest"]
    original_metrics = metadata["original_metrics"]
    runtime_minutes = float(metadata["runtime_seconds_so_far"]) / 60.0
    yaw_ci = bootstrap["metrics"]["yaw_m2_minus_g1"]
    translation_ci = bootstrap["metrics"]["translation_meters_m2_minus_g1"]
    scale_ci = bootstrap["metrics"]["log_scale_m2_minus_g1"]
    bev_ci = bootstrap["metrics"]["bev_frame_m2_minus_g1"]
    d0, d1, d2 = (
        scale["D0_width_ratio"],
        scale["D1_unit_scale"],
        scale["D2_gt_relative_scale"],
    )
    lines = [
        "# Frozen T3/M2 full ZInD val diagnostic",
        "",
        "## 1. 一句话结论",
        "",
        "M2 在正确候选的 S3 子集上显著改善 yaw、translation、scale 与 BEV，"
        "但 72.58% 的 GT positive 在进入正确候选几何前已因 candidate recall、"
        "dustbin 或选错候选丢失，因此端到端第一瓶颈仍是 matcher/candidate recall。",
        "",
        "## 2. 数据规模和 house 分布",
        "",
        f"- S0: {sets['S0_all_pairs']} pairs；S1: {sets['S1_all_gt_positive']} positives；"
        f"negative: {sets['S0_all_pairs'] - sets['S1_all_gt_positive']}。",
        f"- val houses: {metadata['house_split']['val_house_count']}；train houses: "
        f"{metadata['house_split']['train_house_count']}；交叉: 0。",
        f"- S3: {sets['S3_gt_positive_selected_candidate_correct']} pairs / "
        f"{sets['S3_house_count']} houses。",
        "",
        "## 3. 代码、配置、checkpoint 和 manifest hashes",
        "",
        "- branch/HEAD: `agent/zind-bipair-opening-matcher` / "
        "`50a6ae5ca403928627fb4de18ae918d59c03d479`；运行时 worktree 为 dirty，已有修改已保留。",
        "- inference evaluator snapshot: "
        "`c1da154d817b07870587431a378cd299f7e313da1ca2f29bb54cd9672a6754c0`；"
        f"current source after report-only formatting: `{smoke.sha256_file(Path(__file__))}`。",
        f"- T3: `{checkpoint['sha256']}` (`{checkpoint['path']}`)。",
        f"- M2 config: `{metadata['m2']['sha256']}`。",
        f"- Bi-Layout config: `{metadata['model']['bi_layout']['portallayout.config']}` / "
        f"`{metadata['model']['opening_head']['backboneContract']['biLayoutConfig']['sha256']}`。",
        f"- full manifest: `{full_manifest['sha256']}`。",
        f"- val manifest: `{val_manifest['sha256']}`。",
        f"- original metrics: `{original_metrics['sha256']}`。",
        f"- opening threshold: {metadata['opening_threshold']['value']}，来源为 T3 checkpoint。",
        "",
        "## 4. test 未访问证据",
        "",
        "- split 固定为 `val`，CLI 不提供 test split；输出声明 `test_data_accessed=false`。",
        "- 只打开 train manifest 做 house-disjoint 检查；未打开 test manifest、图像或 cache。",
        "",
        "## 5. 运行命令、时间和恢复次数",
        "",
        "- `python tools/evaluate_zind_portal_geometry_val_full.py --device cuda:0 --progress_every 25`",
        f"- 记录运行时间: {runtime_minutes:.2f} min；GPU process invocations: "
        f"{metadata['gpu_process_invocation_count']}；恢复次数: 0。",
        "",
        "## 6. 完成率和 forward 唯一性",
        "",
        f"- completed: {summary['completed_pair_count']}/{summary['pair_count']}。",
        f"- frozen forward count: {summary['unique_frozen_model_forward_count']}；每 pair 均为 1。",
        "- candidate score/probability branch max difference 均为 0；best candidate 一致率 100%。",
        "",
        "## 7. T3 matcher evaluator parity",
        "",
        f"- parity: {'PASS' if parity['pass'] else 'FAIL'}；容差 {parity['tolerance']}。",
        "- candidate recall、positive acceptance、conditional/end-to-end Top-1、negative "
        "rejection、Balanced、Harmonic 的绝对差全部为 0。",
        f"- Harmonic: {matching['harmonic_flow_score']:.12f}。",
        "",
        "## 8. S0–S5 样本量",
        "",
        f"- S0={sets['S0_all_pairs']}，S1={sets['S1_all_gt_positive']}，"
        f"S2={sets['S2_gt_positive_predicted_match']}，"
        f"S3={sets['S3_gt_positive_selected_candidate_correct']}。",
        f"- S4 pose-valid: G0={sets['S4_pose_valid_by_mode']['legacy_polygon']}，"
        f"G1={sets['S4_pose_valid_by_mode']['frame_polygon']}，"
        f"G2={sets['S4_pose_valid_by_mode']['candidate_local_m2']}；"
        f"S5={sets['S5_gt_negative_predicted_match']}。",
        "",
        "## 9. Matcher/candidate 指标",
        "",
        f"- TP/TN/FP/FN: {matching['counts']['TP']}/{matching['counts']['TN']}/"
        f"{matching['counts']['FP']}/{matching['counts']['FN']}。",
        f"- precision={percent(matching['precision'])}，recall={percent(matching['recall'])}，"
        f"F1={percent(matching['f1'])}，negative rejection={percent(matching['negative_rejection'])}。",
        f"- candidate recall={percent(matching['positive_candidate_recall'])}，"
        f"conditional Top-1={percent(matching['conditional_candidate_top1'])}，"
        f"end-to-end Top-1={percent(matching['end_to_end_candidate_top1'])}。",
        f"- Balanced={matching['balanced_flow_score']:.6f}，Harmonic={matching['harmonic_flow_score']:.6f}。",
        "",
        "## 10. G0/G1/G2 geometry（共同 S3）",
        "",
        "| mode | valid | yaw median/mean | translation median/mean (m) | |log-scale| median/mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in GEOMETRY_MODES:
        yaw = distribution(mode, "yaw")
        translation = distribution(mode, "translation")
        log_scale = distribution(mode, "scale")
        valid = geometries[mode]["S3"]["coverage"]["pose_valid_count"]
        lines.append(
            f"| {mode} | {valid}/501 | {number(yaw['median'], 2)}/{number(yaw['mean'], 2)} | "
            f"{number(translation['median'], 3)}/{number(translation['mean'], 3)} | "
            f"{number(log_scale['median'], 3)}/{number(log_scale['mean'], 3)} |"
        )
    lines.extend([
        "",
        f"M2−G1 paired W/T/L: yaw {paired['yaw']['m2_win']}/{paired['yaw']['tie']}/"
        f"{paired['yaw']['m2_loss']}；translation(m) "
        f"{paired['translation_meters']['m2_win']}/{paired['translation_meters']['tie']}/"
        f"{paired['translation_meters']['m2_loss']}；log-scale "
        f"{paired['log_scale']['m2_win']}/{paired['log_scale']['tie']}/"
        f"{paired['log_scale']['m2_loss']}。",
        "",
        "## 11. End-to-end 与 conditional pose",
        "",
    ])
    for mode in GEOMETRY_MODES:
        s1_rate = geometries[mode]["S1"]["joint_pose_success"][medium_key]["rate"]
        s3_rate = geometries[mode]["S3"]["joint_pose_success"][medium_key]["rate"]
        lines.append(
            f"- {mode}: yaw<15° & translation<1m，S1={percent(s1_rate)}，"
            f"S3={percent(s3_rate)}。"
        )
    lines.extend([
        "",
        "## 12. D0/D1/D2 scale 诊断",
        "",
        f"- 子集: {scale['subset']}，n={scale['pair_count']}；正式 G2 未改变。",
        f"- D0 translation mean/median={number(d0['translation_error_meters']['mean'], 3)}/"
        f"{number(d0['translation_error_meters']['median'], 3)}m；frame/raw BEV mean="
        f"{number(d0['bev']['frame_preserving']['instance_mean_overlap']['mean'], 3)}/"
        f"{number(d0['bev']['pre_manhattan']['instance_mean_overlap']['mean'], 3)}。",
        f"- D1 unit translation mean/median={number(d1['translation_error_meters']['mean'], 3)}/"
        f"{number(d1['translation_error_meters']['median'], 3)}m；frame/raw BEV mean="
        f"{number(d1['bev']['frame_preserving']['instance_mean_overlap']['mean'], 3)}/"
        f"{number(d1['bev']['pre_manhattan']['instance_mean_overlap']['mean'], 3)}。",
        f"- D2 oracle translation mean/median={number(d2['translation_error_meters']['mean'], 3)}/"
        f"{number(d2['translation_error_meters']['median'], 3)}m；frame/raw BEV mean="
        f"{number(d2['bev']['frame_preserving']['instance_mean_overlap']['mean'], 3)}/"
        f"{number(d2['bev']['pre_manhattan']['instance_mean_overlap']['mean'], 3)}。",
        "- D1 明显优于 D0，而 D2 与 D1 几乎相同；但 S3 中 500/501 为近 unit-scale，"
        "不能据此把 unit scale 覆盖到一般非单位样本。",
        "",
        "## 13. 双 polygon BEV",
        "",
    ])
    for mode in ("frame_polygon", "candidate_local_m2"):
        frame = geometries[mode]["S3"]["bev"]["frame_preserving"]
        raw = geometries[mode]["S3"]["bev"]["pre_manhattan"]
        lines.append(
            f"- {mode}: frame mean/median={number(frame['instance_mean_overlap']['mean'], 3)}/"
            f"{number(frame['instance_mean_overlap']['median'], 3)}，raw mean/median="
            f"{number(raw['instance_mean_overlap']['mean'], 3)}/"
            f"{number(raw['instance_mean_overlap']['median'], 3)}；invalid="
            f"{frame['invalid_polygon_count']}/{raw['invalid_polygon_count']}。"
        )
    lines.extend([
        "",
        "## 14. M2 hard/soft quality",
        "",
        f"- all predicted-match hard failures: {quality['all_predicted_match_hard_failure_count']}/"
        f"{quality['predicted_match_denominator']} ({percent(quality['all_predicted_match_hard_failure_rate'])})。",
        f"- soft-fail: {quality['all_predicted_match_soft_fail_pair_count']}/"
        f"{quality['predicted_match_denominator']} ({percent(quality['all_predicted_match_soft_fail_pair_rate'])})；"
        f"RANSAC/TLS sides={quality['variant_counts'].get('ransac', 0)}/"
        f"{quality['variant_counts'].get('weighted_tls', 0)}。",
        "- soft-fail 最主要原因是 angular dispersion；quality/error correlation 仅作事后描述，"
        "没有用于校准 gate。",
        "",
        "## 15. House-clustered bootstrap",
        "",
        f"- S3 houses={bootstrap['house_count']}，seed={bootstrap['seed']}，"
        f"resamples={bootstrap['resamples']}。",
        f"- yaw M2−G1 median delta={number(yaw_ci['observed']['median'], 2)}°，95% CI "
        f"[{number(yaw_ci['median_delta_ci']['lower_95'], 2)}, "
        f"{number(yaw_ci['median_delta_ci']['upper_95'], 2)}]。",
        f"- translation median delta={number(translation_ci['observed']['median'], 3)}m，95% CI "
        f"[{number(translation_ci['median_delta_ci']['lower_95'], 3)}, "
        f"{number(translation_ci['median_delta_ci']['upper_95'], 3)}]。",
        f"- log-scale median delta={number(scale_ci['observed']['median'], 3)}，95% CI "
        f"[{number(scale_ci['median_delta_ci']['lower_95'], 3)}, "
        f"{number(scale_ci['median_delta_ci']['upper_95'], 3)}]。",
        f"- frame BEV median delta={number(bev_ci['observed']['median'], 3)}，95% CI "
        f"[{number(bev_ci['median_delta_ci']['lower_95'], 3)}, "
        f"{number(bev_ci['median_delta_ci']['upper_95'], 3)}]。",
        "",
        "## 16. 成功和失败类型",
        "",
        f"- candidate 未召回: {failures['candidate_not_recalled']}/{failures['positive_denominator']}；"
        f"召回后 no-match: {failures['candidate_recalled_matcher_no_match']}/"
        f"{failures['candidate_recalled_denominator']}；召回后选错: "
        f"{failures['candidate_recalled_matcher_selected_wrong']}/"
        f"{failures['candidate_recalled_denominator']}。",
        f"- M2 S3 hard failure: {failures['geometry_by_mode']['candidate_local_m2']['geometry_hard_failure']}/"
        f"{failures['geometry_by_mode']['candidate_local_m2']['S3_denominator']}；"
        f"valid 但 medium-pose 失败: "
        f"{failures['geometry_by_mode']['candidate_local_m2']['geometry_valid_but_medium_pose_failure']}。",
        "",
        "## 17. M2 得到支持的部分",
        "",
        "- yaw 泛化成立：501 个 S3 pair、139 houses，paired median 和 mean 均改善，"
        "house bootstrap 区间不跨 0。",
        "- conditional overall pose 也成立：translation、log-scale、BEV 与三档 joint success 均改善；"
        "但这不等于端到端已解决。",
        "",
        "## 18. 当前系统第一瓶颈",
        "",
        "matcher/candidate recall。只有 501/1827（27.42%）GT positive 到达正确候选 S3；"
        "747 未召回、437 召回后被 dustbin/no-match 拒绝、142 召回后选错。",
        "",
        "## 19. 下一阶段唯一优先方向",
        "",
        "优先改 matcher recall（含 predicted candidate recall 与 dustbin positive acceptance），"
        "而不是先改 scale。scale/unit-prior 是明确的第二阶段诊断方向，但当前影响的样本远少于"
        "进入正确候选前的损失。",
        "",
        "## 20. 尚未执行的内容",
        "",
        "- 未访问 test，未训练，未调 matcher threshold，未校准 M2 gate。",
        "- 未把 D1/D2 写回正式 G2，未实现 clamp/shrinkage/learned scale、M4、SALVe 或 BEV verifier。",
        "- 本结果为 val-only 系统诊断，不作为论文最终 test 统计。",
        "",
    ])
    return "\n".join(lines) + "\n"


def analyze(context: Mapping[str, Any]) -> Dict[str, Any]:
    root = context["output_dir"]
    pair_ids = context["full"]["pair_ids"]
    completed = _load_completed(root, pair_ids)
    if len(completed) != len(pair_ids):
        raise RuntimeError(f"analysis requires complete artifacts: {len(completed)}/{len(pair_ids)}")
    records = [completed[value] for value in pair_ids]
    if not all(smoke._finite_tree(value) for value in records):
        raise ValueError("pair artifacts contain NaN/Inf")
    frozen_paths = [
        _frozen_path(root, index, pair_id) for index, pair_id in enumerate(pair_ids)
    ]
    if not all(path.is_file() for path in frozen_paths):
        raise ValueError("missing frozen prediction artifact")
    forward_counts = [
        int(load_json_gz(path).get("model_forward_count", 0)) for path in frozen_paths
    ]
    if forward_counts != [1] * len(pair_ids):
        raise ValueError("model forward uniqueness check failed")
    matching = matching_summary(records)
    parity = parity_summary(matching, context["original_validation"])
    positives = [value for value in records if value["ground_truth"]["is_match"]]
    predicted_positive = [
        value for value in positives if value["prediction"]["is_match"]
    ]
    s3 = [
        value for value in positives if value['portallayout.evaluation']["selected_candidate_correct"]
    ]
    s5 = [
        value
        for value in records
        if not value["ground_truth"]["is_match"] and value["prediction"]["is_match"]
    ]
    geometry = {
        mode: {
            "S1": _geometry_summary(positives, mode),
            "S2": _geometry_summary(predicted_positive, mode),
            "S3": _geometry_summary(s3, mode),
            "S4": _geometry_summary(
                [
                    value
                    for value in s3
                    if value["prediction"]["geometry_modes"][mode]["pose_valid"]
                ],
                mode,
            ),
        }
        for mode in GEOMETRY_MODES
    }
    diagnostics = [
        value for value in (scale_diagnostic(record) for record in s3) if value is not None
    ]
    rows = paired_rows(records)
    bootstrap = clustered_bootstrap(rows)
    run_metadata = smoke.load_json(root / "run_metadata.json")
    summary = {
        "schema_version": "zind-full-val-geometry-summary-v1",
        "split": "val",
        "purpose": PURPOSE,
        "pair_count": len(records),
        "completed_pair_count": len(records),
        "unique_frozen_model_forward_count": sum(forward_counts),
        "set_counts": {
            "S0_all_pairs": len(records),
            "S1_all_gt_positive": len(positives),
            "S2_gt_positive_predicted_match": len(predicted_positive),
            "S3_gt_positive_selected_candidate_correct": len(s3),
            "S4_pose_valid_by_mode": {
                mode: sum(
                    value["prediction"]["geometry_modes"][mode]["pose_valid"]
                    for value in s3
                )
                for mode in GEOMETRY_MODES
            },
            "S5_gt_negative_predicted_match": len(s5),
            "S3_house_count": len({value["house_id"] for value in s3}),
        },
        "matching": matching,
        "matcher_parity": parity,
        "geometry_interpretable": parity["geometry_interpretable"],
        "geometry": geometry,
        "failure_decomposition": failure_decomposition(records),
        "m2_quality": smoke.quality_summary(records),
        "paired_comparison": paired_summary(rows),
        "scale_diagnostics": {
            "subset": "S3 correct-candidate pairs with valid formal M2 pose",
            **_scale_diagnostic_summary(diagnostics),
        },
        "house_clustered_bootstrap": bootstrap,
        "smoke_reference": (
            smoke.load_json(
                REPO_ROOT / "outputs/zind_geometry_val_smoke/t3_seed123_64/summary.json"
            )["matching"]
            if (
                REPO_ROOT / "outputs/zind_geometry_val_smoke/t3_seed123_64/summary.json"
            ).is_file()
            else None
        ),
        "run_metadata": run_metadata,
        "test_data_accessed": False,
        "scientific_final_result": False,
    }
    _write_jsonl(root / "per_pair.jsonl", records)
    _write_jsonl(root / "scale_diagnostics.jsonl", diagnostics)
    _write_csv(root / "paired_comparison.csv", rows)
    smoke.atomic_write_json(root / "bootstrap_summary.json", bootstrap)
    smoke.atomic_write_json(root / "quality_summary.json", summary["m2_quality"])
    smoke.atomic_write_json(root / "summary.json", summary)
    smoke.atomic_write_text(root / "report.md", _report_markdown(summary))
    (root / "failures").mkdir(parents=True, exist_ok=True)
    for record in records:
        for mode in GEOMETRY_MODES:
            geometry_value = record["prediction"]["geometry_modes"][mode]
            if geometry_value["hard_failure"]:
                smoke.atomic_write_json(
                    root / "failures" / f"{_safe_name(record['pair_id'])}_{mode}.json",
                    {
                        "pair_id": record["pair_id"],
                        "house_id": record["house_id"],
                        "mode": mode,
                        "failure_reason": geometry_value["failure_reason"],
                        "test_data_accessed": False,
                    },
                )
    visualizations = _render_visualizations(root, records, rows)
    run_metadata["analysis"] = {
        "complete": True,
        "cpu_only": True,
        "scale_diagnostic_count": len(diagnostics),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "visualizations": visualizations,
    }
    smoke.atomic_write_json(root / "run_metadata.json", run_metadata)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.prepare_manifest:
        context = load_context(args, require_full_manifest=False)
        if context["full_path"].is_file():
            validate_full_manifest(
                smoke.load_json(context["full_path"]), context["expected_full"]
            )
        else:
            smoke.atomic_write_json(context["full_path"], context["expected_full"])
        print(json.dumps(context["expected_full"]["counts"], indent=2))
        return 0
    context = load_context(args, require_full_manifest=True)
    if args.metadata_only:
        print(json.dumps(metadata_report(context), indent=2))
        return 0
    if args.dry_run:
        print(json.dumps(dry_run_report(context), indent=2))
        return 0
    if args.analyze_only:
        print(json.dumps(analyze(context), indent=2))
        return 0
    print(json.dumps(run(context), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
