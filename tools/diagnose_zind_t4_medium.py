#!/usr/bin/env python3
"""Deterministic T4 matcher-only medium-scale generalization diagnostic.

The tool is deliberately train/val only.  It creates house-first reference
manifests, freezes the T3 upstream once, audits proposal parity, and runs the
four Recall-TopK matcher-only injection/margin factors on shared cache shards.
It never launches the formal run; launch policy is recorded separately so a
caller can apply an explicitly supplied gate after inspecting the artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.zind_bipair_adapter import adapt_zind_bipair_batch  # noqa: E402
from dataset.zind_bipair_dataset import collate_zind_bipair  # noqa: E402
from tools.preflight_zind_t4_matcher_only import (  # noqa: E402
    _finite_mapping,
    _load_t3_model,
    _state_hash,
    _tensor_hash,
)
from tools.train_zind_cross_scene_matcher import (  # noqa: E402
    MetricAccumulator,
    _atomic_json,
    _atomic_torch_save,
    candidate_targets_from_affinity,
    predicted_candidate_masks,
    resolve_device,
    set_seed,
)
from tools.train_zind_end_to_end import (  # noqa: E402
    DEFAULT_BI_LAYOUT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_MATCHER,
    DEFAULT_OPENING,
    EndToEndOpeningMatcher,
    JointZInDMatchingDataset,
    _candidate_supervision_validity,
    _inject_gt_candidates_for_training,
    _matching_loss_bundle,
    _sha256_file,
    configure_training_stage,
    matcher_without_opening_parameters,
)


DEFAULT_T3 = REPO_ROOT / "checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best.pt"
DEFAULT_SMALL_T4 = (
    REPO_ROOT
    / "checkpoints/End_To_End/zind_e2e_v3/g0_seed123/"
    "T4_recall_matcher_only_preflight_v2_6ep/best.pt"
)
DEFAULT_SMALL_VAL = (
    REPO_ROOT
    / "experiments/zind_e2e_v3/g0_seed123/T4_recall/preflight/subsets/val_preflight.json"
)
DEFAULT_SMALL_GATE = (
    REPO_ROOT
    / "experiments/zind_e2e_v3/g0_seed123/T4_recall/matcher_only_preflight/"
    "gpu_small_preflight_gate_v2.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/zind_e2e_v3/g0_seed123/T4_recall/medium_diagnostic"
)
PROTOCOL_VERSION = "zind-t4-matcher-only-medium-diagnostic-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--t3_checkpoint", default=str(DEFAULT_T3))
    parser.add_argument("--small_t4_checkpoint", default=str(DEFAULT_SMALL_T4))
    parser.add_argument("--small_val_manifest", default=str(DEFAULT_SMALL_VAL))
    parser.add_argument("--small_gate_artifact", default=str(DEFAULT_SMALL_GATE))
    parser.add_argument("--train_manifest", default=str(REPO_ROOT / "src/dataset/ZInD_matching/train.json"))
    parser.add_argument("--val_manifest", default=str(REPO_ROOT / "src/dataset/ZInD_matching/val.json"))
    parser.add_argument("--train_cache_dir", default=str(DEFAULT_CACHE_ROOT / "train"))
    parser.add_argument("--val_cache_dir", default=str(DEFAULT_CACHE_ROOT / "val"))
    parser.add_argument("--data_root")
    parser.add_argument("--bi_layout_config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bi_layout_checkpoint", default=str(DEFAULT_BI_LAYOUT))
    parser.add_argument("--opening_checkpoint", default=str(DEFAULT_OPENING))
    parser.add_argument("--matcher_checkpoint", default=str(DEFAULT_MATCHER))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_positive", type=int, default=512)
    parser.add_argument("--train_negative", type=int, default=512)
    parser.add_argument("--val_positive", type=int, default=256)
    parser.add_argument("--val_negative", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--matcher_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--minimum_lr_factor", type=float, default=0.1)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--reuse_cache", action="store_true")
    parser.add_argument(
        "--resume_partial_cache",
        action="store_true",
        help="reuse verified shards from an interrupted cache build; requires no schema",
    )
    parser.add_argument("--skip_small_audit", action="store_true")
    parser.add_argument("--skip_factors", action="store_true")
    parser.add_argument(
        "--sanitize_val_cache",
        action="store_true",
        help="remove a legacy unused injected view from val shards and exit",
    )
    return parser.parse_args(argv)


def _reject_test_paths(paths: Iterable[str | Path]) -> None:
    for path in paths:
        parts = [part.lower() for part in Path(path).expanduser().parts]
        if any(part == "test" or part.startswith("test.") for part in parts):
            raise ValueError("test path is forbidden: {}".format(path))


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stable_key(seed: int, split: str, value: str) -> str:
    return hashlib.sha256("{}:{}:{}".format(seed, split, value).encode()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _manifest_subset(payload: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result = dict(payload)
    result["pairs"] = list(records)
    houses = sorted({str(record["scene_id"]) for record in records})
    result["sourceHouseIds"] = houses
    positive = sum(bool(record["supervision"]["is_match"]) for record in records)
    result["counts"] = {
        "pairCount": len(records),
        "positiveCount": positive,
        "negativeCount": len(records) - positive,
        "houseCount": len(houses),
    }
    result["mediumDiagnosticReferenceOnly"] = True
    result["testDataAccessed"] = False
    return result


def _house_first_class(
    records: Sequence[Mapping[str, Any]],
    *,
    positive: bool,
    count: int,
    seed: int,
    split: str,
    avoid_houses: set[str] | None = None,
) -> List[Mapping[str, Any]]:
    pools: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        if bool(record["supervision"]["is_match"]) != bool(positive):
            continue
        pools.setdefault(str(record["scene_id"]), []).append(record)
    for house, values in pools.items():
        values.sort(key=lambda row: (_stable_key(seed, split, str(row["id"])), str(row["id"])))
    avoid_houses = avoid_houses or set()
    houses = sorted(
        pools,
        key=lambda house: (
            house in avoid_houses,
            _stable_key(seed, split, house),
            house,
        ),
    )
    selected: List[Mapping[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        progress = False
        for house in houses:
            values = pools[house]
            if round_index < len(values):
                selected.append(values[round_index])
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            break
        round_index += 1
    if len(selected) != count:
        raise RuntimeError(
            "insufficient {} {} records: {} < {}".format(
                split, "positive" if positive else "negative", len(selected), count
            )
        )
    return selected


def house_first_select(
    payload: Mapping[str, Any], positive_count: int, negative_count: int, seed: int
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    split = str(payload.get("split"))
    if split not in ("train", "val"):
        raise ValueError("only train/val manifests are accepted")
    records = payload.get("pairs")
    if not isinstance(records, list):
        raise ValueError("manifest has no pairs")
    positives = _house_first_class(
        records, positive=True, count=positive_count, seed=seed, split=split
    )
    positive_houses = {str(row["scene_id"]) for row in positives}
    negatives = _house_first_class(
        records,
        positive=False,
        count=negative_count,
        seed=seed,
        split=split,
        avoid_houses=positive_houses,
    )
    selected = positives + negatives
    selected.sort(key=lambda row: (_stable_key(seed, split, str(row["id"])), str(row["id"])))
    houses: Dict[str, Dict[str, int]] = {}
    for record in selected:
        house = str(record["scene_id"])
        row = houses.setdefault(house, {"positive": 0, "negative": 0, "total": 0})
        label = "positive" if record["supervision"]["is_match"] else "negative"
        row[label] += 1
        row["total"] += 1
    metadata = {
        "split": split,
        "selector": (
            "house-first deterministic round-robin; houses and in-house pairs are "
            "ordered by SHA256(seed, split, id); negative selection prefers houses "
            "not already used by positives"
        ),
        "selection_uses_model_performance": False,
        "seed": int(seed),
        "pair_count": len(selected),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "house_count": len(houses),
        "house_distribution": houses,
        "pair_order": [str(record["id"]) for record in selected],
        "pair_order_sha256": _canonical_sha([str(record["id"]) for record in selected]),
        "test_data_accessed": False,
    }
    return selected, metadata


def prepare_manifests(args: argparse.Namespace, output: Path) -> Dict[str, Any]:
    paths = [args.train_manifest, args.val_manifest]
    _reject_test_paths(paths)
    train_payload = json.loads(Path(args.train_manifest).read_text(encoding="utf-8"))
    val_payload = json.loads(Path(args.val_manifest).read_text(encoding="utf-8"))
    train_records, train_meta = house_first_select(
        train_payload, args.train_positive, args.train_negative, args.seed
    )
    val_records, val_meta = house_first_select(
        val_payload, args.val_positive, args.val_negative, args.seed
    )
    train_houses = {str(row["scene_id"]) for row in train_records}
    val_houses = {str(row["scene_id"]) for row in val_records}
    overlap = sorted(train_houses & val_houses)
    if overlap:
        raise RuntimeError("train/val house overlap: {}".format(overlap[:3]))
    train_path = output / "train_medium.json"
    val_path = output / "val_medium.json"
    _atomic_text(train_path, json.dumps(_manifest_subset(train_payload, train_records), indent=2) + "\n")
    _atomic_text(val_path, json.dumps(_manifest_subset(val_payload, val_records), indent=2) + "\n")
    report = {
        "schema_version": "zind-t4-medium-house-first-v1",
        "seed": int(args.seed),
        "train": train_meta,
        "val": val_meta,
        "train_val_house_overlap": overlap,
        "manifests": {
            "train": {"path": str(train_path), "sha256": _sha256_file(train_path)},
            "val": {"path": str(val_path), "sha256": _sha256_file(val_path)},
        },
        "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    report["sample_manifest_sha256"] = _canonical_sha(report)
    _atomic_json(output / "sample_manifest.json", report)
    return report


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        t3_checkpoint=args.t3_checkpoint,
        bi_layout_config=args.bi_layout_config,
        bi_layout_checkpoint=args.bi_layout_checkpoint,
        opening_checkpoint=args.opening_checkpoint,
        matcher_checkpoint=args.matcher_checkpoint,
    )


def _load_model(
    args: argparse.Namespace, device: torch.device, checkpoint: str | None = None
) -> Tuple[EndToEndOpeningMatcher, float, Dict[str, Any]]:
    model, threshold, report = _load_t3_model(_model_args(args), device)
    if checkpoint is None:
        return model, threshold, report
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(str(path), map_location="cpu")
    model.bi_layout.load_state_dict(payload["bi_layout_state_dict"], strict=True)
    model.matcher.load_state_dict(payload["matcher_state_dict"], strict=True)
    report = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "completed_epoch": int(payload.get("completed_epoch", -1)),
        "initial_end_to_end_checkpoint": dict(payload.get("initialization", {})).get(
            "initial_end_to_end_checkpoint"
        ),
    }
    return model, float(payload.get("opening_probability_threshold", threshold)), report


def _loss_args(margin_enabled: bool) -> argparse.Namespace:
    return argparse.Namespace(
        candidate_loss_weight=1.0,
        candidate_missing_target_policy="mask_positive",
        candidate_rank_loss_weight=0.1,
        token_affinity_weight=1.0,
        shared_response_weight=0.0,
        portal_shift_loss_weight=0.2,
        consistency_weight=0.1,
        positive_dustbin_margin_loss_weight=0.2 if margin_enabled else 0.0,
        positive_dustbin_margin=0.2,
        metric_top_k=3,
    )


def _view(
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
    affinity: torch.Tensor,
    *,
    injected_count: int = 0,
) -> Dict[str, Any]:
    target = candidate_targets_from_affinity(masks_a, affinity, masks_b)
    return {
        "candidate_masks_A": masks_a.detach().cpu(),
        "candidate_masks_B": masks_b.detach().cpu(),
        "candidate_valid_A": valid_a.detach().cpu(),
        "candidate_valid_B": valid_b.detach().cpu(),
        "candidate_target_AB": target.detach().cpu(),
        "intervals_A": _mask_intervals(masks_a, valid_a).detach().cpu(),
        "intervals_B": _mask_intervals(masks_b, valid_b).detach().cpu(),
        "gt_candidate_injection_count": int(injected_count),
    }


def _mask_intervals(masks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    batch, count, token_count = masks.shape
    intervals = torch.full((batch, count, 2), -1, dtype=torch.long, device=masks.device)
    for batch_index in range(batch):
        for candidate_index in torch.where(valid[batch_index])[0].tolist():
            mask = masks[batch_index, candidate_index].bool()
            if not bool(mask.any()):
                continue
            if bool(mask.all()):
                intervals[batch_index, candidate_index] = torch.tensor(
                    [0, token_count - 1], device=masks.device
                )
                continue
            starts = torch.where(mask & ~torch.roll(mask, shifts=1))[0]
            if starts.numel() != 1:
                raise ValueError("candidate mask is not one circular interval")
            start = int(starts.item())
            end = start
            while bool(mask[(end + 1) % token_count]):
                end = (end + 1) % token_count
            intervals[batch_index, candidate_index] = torch.tensor(
                [start, end], device=masks.device
            )
    return intervals


@torch.no_grad()
def _freeze_batch(
    model: EndToEndOpeningMatcher,
    raw: MutableMapping[str, Any],
    device: torch.device,
    threshold: float,
    *,
    include_injected_view: bool,
) -> Dict[str, Any]:
    model.eval()
    pair_cpu = adapt_zind_bipair_batch(raw)
    pair = pair_cpu.to(device)
    image_a = raw["image_A"].to(device=device, dtype=torch.float32)
    image_b = raw["image_B"].to(device=device, dtype=torch.float32)
    live = {key: value.detach() for key, value in model.live_layout(image_a, image_b).items()}
    signal_a = model.matcher.opening_head(
        live["layout_feature_A"], live["enclosed_depth_A"], live["extended_depth_A"]
    )
    signal_b = model.matcher.opening_head(
        live["layout_feature_B"], live["enclosed_depth_B"], live["extended_depth_B"]
    )
    proposal_kwargs = dict(
        threshold=threshold,
        min_width_tokens=2,
        max_openings=12,
        peak_top_k=4,
        peak_min_probability=0.20,
        expansion_threshold=0.12,
        dedup_iou=0.50,
    )
    views: Dict[str, Dict[str, Any]] = {}
    for name, mode in (("legacy_predicted", "legacy"), ("recall_predicted", "recall_topk")):
        masks_a, valid_a = predicted_candidate_masks(
            signal_a["opening_probability"], proposal_mode=mode, **proposal_kwargs
        )
        masks_b, valid_b = predicted_candidate_masks(
            signal_b["opening_probability"], proposal_mode=mode, **proposal_kwargs
        )
        views[name] = _view(masks_a, masks_b, valid_a, valid_b, pair.affinity_ab)
    if include_injected_view:
        recall = views["recall_predicted"]
        masks_a = recall["candidate_masks_A"].to(device)
        masks_b = recall["candidate_masks_B"].to(device)
        valid_a = recall["candidate_valid_A"].to(device)
        valid_b = recall["candidate_valid_B"].to(device)
        manifest_target = raw["candidate_target_AB"].to(device=device, dtype=torch.float32)
        manifest_a = raw["manifest_candidate_masks_A"].to(device=device, dtype=torch.bool)
        manifest_b = raw["manifest_candidate_masks_B"].to(device=device, dtype=torch.bool)
        masks_a, valid_a, injected_a = _inject_gt_candidates_for_training(
            masks_a, valid_a, manifest_a, manifest_target.any(dim=-1), 12
        )
        masks_b, valid_b, injected_b = _inject_gt_candidates_for_training(
            masks_b, valid_b, manifest_b, manifest_target.any(dim=-2), 12
        )
        views["recall_injected"] = _view(
            masks_a,
            masks_b,
            valid_a,
            valid_b,
            pair.affinity_ab,
            injected_count=injected_a + injected_b,
        )
    pair_cache = replace(
        pair_cpu,
        image_a=None,
        image_b=None,
        metadata={
            "pair_id": list(raw["pair_id"]),
            "merged_pair_count": raw["merged_pair_count"].detach().cpu(),
        },
    )
    return {
        "pair_ids": list(raw["pair_id"]),
        "live": {key: value.detach().cpu() for key, value in live.items()},
        "pair": pair_cache,
        "opening_probability_A": signal_a["opening_probability"].detach().cpu(),
        "opening_probability_B": signal_b["opening_probability"].detach().cpu(),
        "views": views,
    }


def _dataset(
    manifest: Path, cache: str, split: str, args: argparse.Namespace
) -> JointZInDMatchingDataset:
    return JointZInDMatchingDataset(
        str(manifest),
        cache,
        data_root=args.data_root,
        max_pairs=None,
        expected_split=split,
        image_shape=(512, 1024),
        rotation_augmentation=False,
    )


def _entry_signature(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "pair_ids": list(entry["pair_ids"]),
        "live": {key: _tensor_hash(value) for key, value in sorted(entry["live"].items())},
        "opening_probability_A": _tensor_hash(entry["opening_probability_A"]),
        "opening_probability_B": _tensor_hash(entry["opening_probability_B"]),
        "is_match": _tensor_hash(entry["pair"].is_match),
        "affinity_ab": _tensor_hash(entry["pair"].affinity_ab),
        "views": {
            name: {
                key: _tensor_hash(value) if torch.is_tensor(value) else value
                for key, value in sorted(view.items())
            }
            for name, view in sorted(entry["views"].items())
        },
    }


def build_or_load_cache(
    args: argparse.Namespace,
    output: Path,
    model: EndToEndOpeningMatcher,
    threshold: float,
    sample_report: Mapping[str, Any],
) -> Tuple[Dict[str, List[Mapping[str, Any]]], Dict[str, Any]]:
    cache_root = output / "upstream_cache"
    schema_path = cache_root / "schema.json"
    if args.reuse_cache:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if schema.get("sample_manifest_sha256") != sample_report["sample_manifest_sha256"]:
            raise ValueError("upstream cache sample manifest changed")
        entries: Dict[str, List[Mapping[str, Any]]] = {"train": [], "val": []}
        for split in entries:
            for shard in schema["splits"][split]["shards"]:
                path = cache_root / shard["relative_path"]
                if _sha256_file(path) != shard["sha256"]:
                    raise ValueError("cache shard hash mismatch: {}".format(path))
                entries[split].append(torch.load(str(path), map_location="cpu")["entry"])
        return entries, schema

    if schema_path.exists():
        raise FileExistsError("cache already exists; pass --reuse_cache")
    if not args.resume_partial_cache and any(cache_root.glob("*/shard_*.pt")):
        raise FileExistsError(
            "partial cache shards exist; pass --resume_partial_cache to verify and continue"
        )
    entries = {"train": [], "val": []}
    split_schema: Dict[str, Any] = {}
    for split, manifest, cache in (
        ("train", output / "train_medium.json", args.train_cache_dir),
        ("val", output / "val_medium.json", args.val_cache_dir),
    ):
        dataset = _dataset(manifest, cache, split, args)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_zind_bipair,
            drop_last=False,
        )
        shards = []
        for index, raw in enumerate(loader):
            relative = Path(split) / "shard_{:04d}.pt".format(index)
            path = cache_root / relative
            if path.exists():
                payload = torch.load(str(path), map_location="cpu")
                if payload.get("test_data_accessed") is not False:
                    raise ValueError("partial shard test-data flag is invalid")
                entry = payload.get("entry")
                if not isinstance(entry, Mapping):
                    raise ValueError("partial cache shard has no entry: {}".format(path))
                if list(entry.get("pair_ids", ())) != list(raw["pair_id"]):
                    raise ValueError("partial cache shard pair order mismatch: {}".format(path))
            else:
                entry = _freeze_batch(
                    model,
                    raw,
                    resolve_device(args.device),
                    threshold,
                    include_injected_view=(split == "train"),
                )
                _atomic_torch_save(path, {"entry": entry, "test_data_accessed": False})
            shards.append(
                {
                    "relative_path": str(relative),
                    "sha256": _sha256_file(path),
                    "size": path.stat().st_size,
                    "pair_ids": list(entry["pair_ids"]),
                    "tensor_signature": _entry_signature(entry),
                }
            )
            entries[split].append(entry)
            if (index + 1) % 32 == 0 or index + 1 == len(loader):
                print("freeze {} {}/{} batches".format(split, index + 1, len(loader)), flush=True)
        split_schema[split] = {
            "pair_count": len(dataset),
            "batch_count": len(shards),
            "shards": shards,
        }
    proposal = {
        "opening_threshold": float(threshold),
        "min_width_tokens": 2,
        "max_candidates": 12,
        "recall_peak_top_k": 4,
        "recall_peak_min_probability": 0.20,
        "recall_expansion_threshold": 0.12,
        "recall_dedup_iou": 0.50,
    }
    schema = {
        "schema_version": "zind-t4-medium-upstream-cache-v2",
        "protocol_version": PROTOCOL_VERSION,
        "sample_manifest_sha256": sample_report["sample_manifest_sha256"],
        "t3": {
            "path": str(Path(args.t3_checkpoint).resolve()),
            "sha256": _sha256_file(Path(args.t3_checkpoint)),
        },
        "upstream_hashes": {
            "bi_layout": _state_hash(model.bi_layout),
            "opening_head": _state_hash(model.matcher.opening_head),
        },
        "proposal": proposal,
        "config_sha256": _canonical_sha(proposal),
        "stored_fields": [
            "pair_id/split",
            "live layout feature/depth",
            "opening probability",
            "legacy intervals/candidates/masks",
            "Recall-TopK intervals/candidates/masks",
            "predicted-only target mapping",
            "train-only injected candidate view",
        ],
        "splits": split_schema,
        "active_splits": ["train", "val"],
        "validation_candidate_inference_uses_gt": False,
        "validation_cached_views": ["legacy_predicted", "recall_predicted"],
        "test_data_accessed": False,
    }
    schema["schema_sha256"] = _canonical_sha(schema)
    _atomic_json(schema_path, schema)
    return entries, schema


def sanitize_validation_cache(output: Path) -> Dict[str, Any]:
    """Remove the never-consumed GT-injected view from validation shards."""
    cache_root = output / "upstream_cache"
    schema_path = cache_root / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    original_schema_file_sha = _sha256_file(schema_path)
    train_hashes_before = [row["sha256"] for row in schema["splits"]["train"]["shards"]]
    changed = 0
    predicted_signatures_before = []
    predicted_signatures_after = []
    for shard in schema["splits"]["val"]["shards"]:
        path = cache_root / shard["relative_path"]
        payload = torch.load(str(path), map_location="cpu")
        entry = payload["entry"]
        predicted_signatures_before.append(
            {
                name: {
                    key: _tensor_hash(value) if torch.is_tensor(value) else value
                    for key, value in sorted(entry["views"][name].items())
                }
                for name in ("legacy_predicted", "recall_predicted")
            }
        )
        if "recall_injected" in entry["views"]:
            del entry["views"]["recall_injected"]
            _atomic_torch_save(
                path, {"entry": entry, "test_data_accessed": False}
            )
            changed += 1
        predicted_signatures_after.append(
            {
                name: {
                    key: _tensor_hash(value) if torch.is_tensor(value) else value
                    for key, value in sorted(entry["views"][name].items())
                }
                for name in ("legacy_predicted", "recall_predicted")
            }
        )
        shard.update(
            sha256=_sha256_file(path),
            size=path.stat().st_size,
            tensor_signature=_entry_signature(entry),
        )
    schema["schema_version"] = "zind-t4-medium-upstream-cache-v2"
    schema["validation_cached_views"] = [
        "legacy_predicted",
        "recall_predicted",
    ]
    schema["validation_candidate_inference_uses_gt"] = False
    schema["test_data_accessed"] = False
    schema.pop("schema_sha256", None)
    schema["schema_sha256"] = _canonical_sha(schema)
    _atomic_json(schema_path, schema)
    train_hashes_after = [row["sha256"] for row in schema["splits"]["train"]["shards"]]
    report = {
        "schema_version": "zind-t4-medium-val-cache-sanitization-v1",
        "reason": (
            "strict val predicted-only cache: remove an unused diagnostic "
            "recall_injected view; no model inference or factor result changes"
        ),
        "changed_val_shards": changed,
        "val_shard_count": len(schema["splits"]["val"]["shards"]),
        "predicted_view_signatures_unchanged": (
            predicted_signatures_before == predicted_signatures_after
        ),
        "train_shard_hashes_unchanged": train_hashes_before == train_hashes_after,
        "original_schema_file_sha256": original_schema_file_sha,
        "sanitized_schema_file_sha256": _sha256_file(schema_path),
        "sanitized_schema_content_sha256": schema["schema_sha256"],
        "validation_cached_views": schema["validation_cached_views"],
        "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    _atomic_json(output / "val_cache_sanitization.json", report)
    return report


def cache_distribution(entries: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for split, split_entries in entries.items():
        views = sorted(set.intersection(*[set(entry["views"]) for entry in split_entries]))
        split_report: Dict[str, Any] = {}
        predicted_presence: List[bool] = []
        injected_presence: List[bool] = []
        for view_name in views:
            positive_count = target_count = pair_count = injection_count = 0
            counts: List[int] = []
            products: List[int] = []
            presence: List[bool] = []
            for entry in split_entries:
                view = entry["views"][view_name]
                positive = entry["pair"].is_match.bool()
                has_target = view["candidate_target_AB"].flatten(1).gt(0).any(dim=-1)
                pair_count += len(entry["pair_ids"])
                positive_count += int(positive.sum().item())
                target_count += int((positive & has_target).sum().item())
                injection_count += int(view["gt_candidate_injection_count"])
                for index in range(len(entry["pair_ids"])):
                    count_a = int(view["candidate_valid_A"][index].sum().item())
                    count_b = int(view["candidate_valid_B"][index].sum().item())
                    counts.extend((count_a, count_b))
                    products.append(count_a * count_b)
                    if bool(positive[index]):
                        presence.append(bool(has_target[index]))
            if view_name == "recall_predicted":
                predicted_presence = presence
            if view_name == "recall_injected":
                injected_presence = presence
            split_report[view_name] = {
                "pair_count": pair_count,
                "positive_count": positive_count,
                "positive_target_count": target_count,
                "positive_candidate_recall": target_count / max(positive_count, 1),
                "candidate_count_mean": statistics.fmean(counts),
                "candidate_count_median": statistics.median(counts),
                "candidate_count_p95": _percentile(counts, 95),
                "candidate_product_mean": statistics.fmean(products),
                "candidate_product_median": statistics.median(products),
                "candidate_product_p95": _percentile(products, 95),
                "injected_candidate_count": injection_count,
            }
        if predicted_presence and injected_presence:
            split_report["injection_distribution_shift"] = {
                "positive_targets_rescued": sum(
                    injected and not predicted
                    for predicted, injected in zip(
                        predicted_presence, injected_presence
                    )
                ),
                "predicted_target_presence_sha256": _canonical_sha(
                    predicted_presence
                ),
                "injected_target_presence_sha256": _canonical_sha(
                    injected_presence
                ),
            }
        report[split] = split_report
    return {
        "schema_version": "zind-t4-medium-cache-distribution-v1",
        "splits": report,
        "validation_injected_view_present": "recall_injected" in report["val"],
        "test_data_accessed": False,
    }


def _move(value: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in value.items()}


def _forward_view(
    model: EndToEndOpeningMatcher,
    entry: Mapping[str, Any],
    view_name: str,
    device: torch.device,
    loss_args: argparse.Namespace,
) -> Tuple[Mapping[str, torch.Tensor], Any, Mapping[str, torch.Tensor], Mapping[str, Any]]:
    live = _move(entry["live"], device)
    pair = entry["pair"].to(device)
    view = entry["views"][view_name]
    masks_a = view["candidate_masks_A"].to(device)
    masks_b = view["candidate_masks_B"].to(device)
    valid_a = view["candidate_valid_A"].to(device)
    valid_b = view["candidate_valid_B"].to(device)
    target = view["candidate_target_AB"].to(device)
    outputs = model.forward_matcher(
        live,
        candidate_masks_a=masks_a,
        candidate_masks_b=masks_b,
        candidate_valid_a=valid_a,
        candidate_valid_b=valid_b,
        opening_guidance_a=None,
        opening_guidance_b=None,
        opening_guidance_mode="predicted",
        opening_guidance_mix_weight=0.5,
    )
    losses = _matching_loss_bundle(outputs, pair, target, valid_a, valid_b, loss_args)
    tensors = {
        "masks_a": masks_a,
        "masks_b": masks_b,
        "valid_a": valid_a,
        "valid_b": valid_b,
        "target": target,
    }
    return outputs, pair, tensors, losses


def _percentile(values: Sequence[int | float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _score_row(
    pair_id: str,
    record: Mapping[str, Any],
    outputs: Mapping[str, torch.Tensor],
    tensors: Mapping[str, torch.Tensor],
    opening_a: torch.Tensor,
    opening_b: torch.Tensor,
    intervals_a: torch.Tensor,
    intervals_b: torch.Tensor,
    index: int,
) -> Dict[str, Any]:
    valid_a = tensors["valid_a"][index].bool()
    valid_b = tensors["valid_b"][index].bool()
    target = tensors["target"][index].gt(0) & valid_a.unsqueeze(-1) & valid_b.unsqueeze(0)
    positive = bool(record["supervision"]["is_match"])
    best = [int(value) for value in outputs["best_candidate_pair"][index].tolist()]
    accepted = best[0] >= 0 and best[1] >= 0
    correct = bool(accepted and target[best[0], best[1]]) if positive else False
    count_a = int(valid_a.sum().item())
    count_b = int(valid_b.sum().item())
    score = outputs["candidate_pair_score"][index]
    valid_pairs = valid_a.unsqueeze(-1) & valid_b.unsqueeze(0)
    true_score = None
    row_margin = None
    column_margin = None
    if bool(target.any()):
        true_score = float(score[target].max().item())
        logits = outputs["candidate_assignment_logits"][index]
        targets = torch.where(target)
        real = logits[targets]
        # Dustbins live at the padded matrix's final row/column, not at each
        # sample's number of valid prefix candidates.
        dustbin_row = int(logits.shape[0] - 1)
        dustbin_column = int(logits.shape[1] - 1)
        row_dustbin = logits[targets[0], dustbin_column]
        column_dustbin = logits[dustbin_row, targets[1]]
        row_margin = float((real - row_dustbin).mean().item())
        column_margin = float((real - column_dustbin).mean().item())
    wrong_mask = valid_pairs & ~target
    wrong_score = float(score[wrong_mask].max().item()) if bool(wrong_mask.any()) else None
    if positive:
        if not bool(target.any()):
            failure = "candidate_absent"
        elif not accepted:
            failure = "dustbin_reject"
        elif not correct:
            failure = "wrong_selection"
        else:
            failure = "success"
    else:
        failure = "correct_no_match" if not accepted else "false_match"
    padded_assignment = outputs["candidate_assignment_logits"][index]
    assignment_rows = torch.cat(
        (
            torch.arange(count_a, device=padded_assignment.device),
            torch.tensor([padded_assignment.shape[0] - 1], device=padded_assignment.device),
        )
    )
    assignment_columns = torch.cat(
        (
            torch.arange(count_b, device=padded_assignment.device),
            torch.tensor([padded_assignment.shape[1] - 1], device=padded_assignment.device),
        )
    )
    assignment = padded_assignment[assignment_rows][:, assignment_columns]
    return {
        "pair_id": pair_id,
        "house_id": str(record["scene_id"]),
        "gt_is_match": positive,
        "gt_candidate_present": bool(target.any()),
        "candidate_count_A": count_a,
        "candidate_count_B": count_b,
        "candidate_product": count_a * count_b,
        "candidate_intervals_A": intervals_a[index, :count_a].tolist(),
        "candidate_intervals_B": intervals_b[index, :count_b].tolist(),
        "best_candidate": best,
        "accepted_match": accepted,
        "candidate_correct": correct,
        "s3": bool(correct),
        "true_candidate_score": true_score,
        "strongest_wrong_candidate_score": wrong_score,
        "row_true_to_dustbin_margin": row_margin,
        "column_true_to_dustbin_margin": column_margin,
        "failure_type": failure,
        "opening_probability_A_sha256": _tensor_hash(opening_a[index]),
        "opening_probability_B_sha256": _tensor_hash(opening_b[index]),
        "candidate_mask_A_sha256": _tensor_hash(tensors["masks_a"][index]),
        "candidate_mask_B_sha256": _tensor_hash(tensors["masks_b"][index]),
        "candidate_feature_A_sha256": _tensor_hash(outputs["E_A_open"][index, :count_a]),
        "candidate_feature_B_sha256": _tensor_hash(outputs["E_B_open"][index, :count_b]),
        "score_matrix_sha256": _tensor_hash(score[:count_a, :count_b]),
        "assignment_with_dustbins_sha256": _tensor_hash(assignment),
    }


@torch.no_grad()
def evaluate_entries(
    model: EndToEndOpeningMatcher,
    entries: Sequence[Mapping[str, Any]],
    view_name: str,
    device: torch.device,
    records: Mapping[str, Mapping[str, Any]],
    *,
    margin_enabled: bool,
    pair_rows: bool = True,
) -> Dict[str, Any]:
    model.eval()
    loss_args = _loss_args(margin_enabled)
    accumulator = MetricAccumulator(3)
    rows: List[Dict[str, Any]] = []
    counts: List[int] = []
    products: List[int] = []
    for entry in entries:
        outputs, pair, tensors, losses = _forward_view(
            model, entry, view_name, device, loss_args
        )
        accumulator.update(
            outputs,
            pair,
            tensors["target"],
            tensors["valid_a"],
            tensors["valid_b"],
            losses,
        )
        intervals_a = entry["views"][view_name]["intervals_A"]
        intervals_b = entry["views"][view_name]["intervals_B"]
        for index, pair_id in enumerate(entry["pair_ids"]):
            count_a = int(tensors["valid_a"][index].sum().item())
            count_b = int(tensors["valid_b"][index].sum().item())
            counts.extend((count_a, count_b))
            products.append(count_a * count_b)
            if pair_rows:
                rows.append(
                    _score_row(
                        pair_id,
                        records[pair_id],
                        outputs,
                        tensors,
                        entry["opening_probability_A"],
                        entry["opening_probability_B"],
                        intervals_a,
                        intervals_b,
                        index,
                    )
                )
    metrics = accumulator.result()
    positive_rows = [row for row in rows if row["gt_is_match"]]
    negative_rows = [row for row in rows if not row["gt_is_match"]]
    true_positive = sum(row["accepted_match"] for row in positive_rows)
    false_positive = sum(row["accepted_match"] for row in negative_rows)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(len(positive_rows), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics.update(
        classification_precision=precision,
        classification_recall=recall,
        classification_f1=f1,
        candidate_count={
            "mean": statistics.fmean(counts) if counts else 0.0,
            "median": statistics.median(counts) if counts else 0.0,
            "p95": _percentile(counts, 95),
        },
        candidate_product={
            "mean": statistics.fmean(products) if products else 0.0,
            "median": statistics.median(products) if products else 0.0,
            "p95": _percentile(products, 95),
        },
        recalled_positive_acceptance=(
            sum(row["accepted_match"] and row["gt_candidate_present"] for row in positive_rows)
            / max(sum(row["gt_candidate_present"] for row in positive_rows), 1)
        ),
        test_data_accessed=False,
    )
    return {"metrics": metrics, "pairs": rows}


def _records(path: Path) -> Dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") not in ("train", "val"):
        raise ValueError("only train/val records are accepted")
    return {str(record["id"]): record for record in payload["pairs"]}


def _compare_rows(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    left_map = {row["pair_id"]: row for row in left}
    right_map = {row["pair_id"]: row for row in right}
    fields = (
        "opening_probability_A_sha256",
        "opening_probability_B_sha256",
        "candidate_intervals_A",
        "candidate_intervals_B",
        "candidate_mask_A_sha256",
        "candidate_mask_B_sha256",
        "candidate_feature_A_sha256",
        "candidate_feature_B_sha256",
        "score_matrix_sha256",
        "assignment_with_dustbins_sha256",
        "best_candidate",
        "accepted_match",
    )
    mismatches = []
    for pair_id in sorted(left_map):
        for field in fields:
            if left_map[pair_id][field] != right_map[pair_id][field]:
                mismatches.append({"pair_id": pair_id, "field": field})
    return {
        "fields": list(fields),
        "pair_count": len(left_map),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "elementwise_exact": not mismatches,
    }


def _proposal_comparison(p0: Mapping[str, Any], p1: Mapping[str, Any]) -> Dict[str, Any]:
    left = {row["pair_id"]: row for row in p0["pairs"]}
    right = {row["pair_id"]: row for row in p1["pairs"]}
    wins = ties = losses = 0
    new_recall = new_recall_s3 = interference = 0
    pair_changes = []
    for pair_id in sorted(left):
        a, b = left[pair_id], right[pair_id]
        correct_a = a["s3"] if a["gt_is_match"] else not a["accepted_match"]
        correct_b = b["s3"] if b["gt_is_match"] else not b["accepted_match"]
        wins += int(correct_b and not correct_a)
        losses += int(correct_a and not correct_b)
        ties += int(correct_a == correct_b)
        if a["gt_is_match"] and not a["gt_candidate_present"] and b["gt_candidate_present"]:
            new_recall += 1
            new_recall_s3 += int(b["s3"])
        if a["gt_is_match"] and a["s3"] and not b["s3"]:
            interference += 1
        if correct_a != correct_b or a["gt_candidate_present"] != b["gt_candidate_present"]:
            pair_changes.append(
                {
                    "pair_id": pair_id,
                    "house_id": a["house_id"],
                    "gt_is_match": a["gt_is_match"],
                    "legacy": a,
                    "recall_topk": b,
                }
            )
    return {
        "pair_level_win_tie_loss": {"win": wins, "tie": ties, "loss": losses},
        "newly_recalled_positive": new_recall,
        "newly_recalled_and_s3": new_recall_s3,
        "new_recall_not_converted_to_s3": new_recall - new_recall_s3,
        "legacy_s3_lost_after_added_candidates": interference,
        "changed_pairs": pair_changes,
        "test_data_accessed": False,
    }


def small_audit(
    args: argparse.Namespace, output: Path, device: torch.device
) -> Dict[str, Any]:
    manifest = Path(args.small_val_manifest).resolve()
    _reject_test_paths([manifest])
    dataset = _dataset(manifest, args.val_cache_dir, "val", args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_zind_bipair,
    )
    t3, threshold, t3_report = _load_model(args, device)
    entries = [
        _freeze_batch(
            t3, raw, device, threshold, include_injected_view=False
        )
        for raw in loader
    ]
    records = _records(manifest)
    p0 = evaluate_entries(t3, entries, "legacy_predicted", device, records, margin_enabled=True)
    p1 = evaluate_entries(t3, entries, "recall_predicted", device, records, margin_enabled=True)
    p1_repeat = evaluate_entries(t3, entries, "recall_predicted", device, records, margin_enabled=True)
    parity = _compare_rows(p1["pairs"], p1_repeat["pairs"])
    t4, _, t4_report = _load_model(args, device, args.small_t4_checkpoint)
    t4_p1 = evaluate_entries(t4, entries, "recall_predicted", device, records, margin_enabled=True)
    old_gate = json.loads(Path(args.small_gate_artifact).read_text(encoding="utf-8"))
    old = old_gate["t3_subset_baseline"]

    def distance(metrics: Mapping[str, Any]) -> float:
        mapping = {
            "positive_match_recall": "positive_acceptance_rate",
            "end_to_end_candidate_top1": "end_to_end_candidate_top1",
            "negative_rejection_rate": "negative_rejection_rate",
            "harmonic_flow_score": "harmonic_flow_score",
        }
        return sum(abs(float(old[key]) - float(metrics[value])) for key, value in mapping.items())

    distance_p0 = distance(p0["metrics"])
    distance_p1 = distance(p1["metrics"])
    baseline_proposal = "legacy" if distance_p0 < distance_p1 else "recall_topk"
    transitions: Dict[str, List[Dict[str, Any]]] = {
        "t3_match_to_t4_no_match": [],
        "t3_no_match_to_t4_match": [],
        "t3_correct_to_t4_wrong_candidate": [],
        "t3_wrong_to_t4_correct_candidate": [],
        "t3_s3_to_t4_non_s3": [],
        "t3_non_s3_to_t4_s3": [],
    }
    t3_rows = {row["pair_id"]: row for row in p1["pairs"]}
    t4_rows = {row["pair_id"]: row for row in t4_p1["pairs"]}
    for pair_id in sorted(t3_rows):
        a, b = t3_rows[pair_id], t4_rows[pair_id]
        merged = {
            "pair_id": pair_id,
            "house_id": a["house_id"],
            "gt_is_match": a["gt_is_match"],
            "legacy_candidate_count_A": next(row for row in p0["pairs"] if row["pair_id"] == pair_id)["candidate_count_A"],
            "legacy_candidate_count_B": next(row for row in p0["pairs"] if row["pair_id"] == pair_id)["candidate_count_B"],
            "recall_candidate_count_A": a["candidate_count_A"],
            "recall_candidate_count_B": a["candidate_count_B"],
            "t3": a,
            "t4": b,
            "first_observed_degradation_checkpoint_epoch": (
                t4_report["completed_epoch"]
                if a["gt_is_match"]
                and ((a["accepted_match"] and not b["accepted_match"]) or (a["s3"] and not b["s3"]))
                else None
            ),
            "exact_first_epoch_available": False,
            "exact_first_epoch_limitation": (
                "only best(epoch 1) and last(epoch 5) states were retained; aggregate "
                "metrics cannot reconstruct pair-level intermediate states"
            ),
        }
        if a["accepted_match"] and not b["accepted_match"]:
            transitions["t3_match_to_t4_no_match"].append(merged)
        if not a["accepted_match"] and b["accepted_match"]:
            transitions["t3_no_match_to_t4_match"].append(merged)
        if a["candidate_correct"] and b["accepted_match"] and not b["candidate_correct"]:
            transitions["t3_correct_to_t4_wrong_candidate"].append(merged)
        if a["accepted_match"] and not a["candidate_correct"] and b["candidate_correct"]:
            transitions["t3_wrong_to_t4_correct_candidate"].append(merged)
        if a["s3"] and not b["s3"]:
            transitions["t3_s3_to_t4_non_s3"].append(merged)
        if not a["s3"] and b["s3"]:
            transitions["t3_non_s3_to_t4_s3"].append(merged)
    initialization = t4_report.get("initial_end_to_end_checkpoint") or {}
    initialization_hash_ok = initialization.get("sha256") == t3_report["sha256"]
    upstream_exact = (
        _state_hash(t3.bi_layout) == _state_hash(t4.bi_layout)
        and _state_hash(t3.matcher.opening_head) == _state_hash(t4.matcher.opening_head)
    )
    report = {
        "schema_version": "zind-t4-small-epoch0-parity-v1",
        "protocol_version": PROTOCOL_VERSION,
        "pair_count": len(records),
        "t3": t3_report,
        "t4": t4_report,
        "reported_t3_subset_baseline": old,
        "reported_baseline_proposal": baseline_proposal,
        "reported_baseline_distance": {"legacy": distance_p0, "recall_topk": distance_p1},
        "P0_t3_legacy": p0,
        "P1_t3_recall_topk": p1,
        "P1_t4_best": t4_p1,
        "recall_topk_impact": _proposal_comparison(p0, p1),
        "epoch0_elementwise_parity": parity,
        "t4_initialization_t3_hash_exact": initialization_hash_ok,
        "t3_t4_upstream_state_exact": upstream_exact,
        "parity_pass": bool(parity["elementwise_exact"] and initialization_hash_ok and upstream_exact),
        "pair_transitions": transitions,
        "active_splits": ["val"],
        "test_data_accessed": False,
    }
    _atomic_json(output / "small_32_epoch0_parity_and_transitions.json", report)
    return report


def _optimizer(model: EndToEndOpeningMatcher, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [{"name": "matcher", "params": list(matcher_without_opening_parameters(model)), "lr": args.matcher_lr}],
        weight_decay=args.weight_decay,
    )


def _scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    def factor(epoch: int) -> float:
        progress = min(max(epoch, 0), args.epochs) / max(args.epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.minimum_lr_factor + (1.0 - args.minimum_lr_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _gradient_stats(module: torch.nn.Module) -> Dict[str, Any]:
    tensors = finite = nonzero = 0
    norm_sq = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        tensors += 1
        finite += int(bool(torch.isfinite(parameter.grad).all()))
        norm = float(parameter.grad.detach().float().norm().item())
        nonzero += int(norm > 0)
        norm_sq += norm * norm
    return {
        "gradient_tensors": tensors,
        "finite_gradient_tensors": finite,
        "nonzero_gradient_tensors": nonzero,
        "gradient_norm": math.sqrt(norm_sq),
    }


def _epoch_order(count: int, seed: int, epoch: int) -> List[int]:
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(count, generator=generator).tolist()


def run_factor(
    args: argparse.Namespace,
    factor_id: str,
    injection: bool,
    margin: bool,
    entries: Mapping[str, Sequence[Mapping[str, Any]]],
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    device: torch.device,
    output: Path,
) -> Dict[str, Any]:
    set_seed(args.seed)
    model, _, initialization = _load_model(args, device)
    configure_training_stage(model, "matcher_only_finetune")
    optimizer = _optimizer(model, args)
    scheduler = _scheduler(optimizer, args)
    loss_args = _loss_args(margin)
    upstream_before = {
        "bi_layout": _state_hash(model.bi_layout),
        "opening_head": _state_hash(model.matcher.opening_head),
    }
    initial_matcher_hash = _state_hash(model.matcher)
    baseline = evaluate_entries(
        model,
        entries["val"],
        "recall_predicted",
        device,
        records["val"],
        margin_enabled=margin,
        pair_rows=True,
    )["metrics"]
    epochs = []
    best: Dict[str, Any] | None = None
    best_state = None
    gradient_audit = None
    order_hashes = []
    for epoch in range(args.epochs):
        configure_training_stage(model, "matcher_only_finetune")
        order = _epoch_order(len(entries["train"]), args.seed, epoch)
        order_hashes.append(_canonical_sha(order))
        train_accumulator = MetricAccumulator(3)
        injection_count = 0
        start = time.perf_counter()
        for step, entry_index in enumerate(order):
            entry = entries["train"][entry_index]
            view_name = "recall_injected" if injection else "recall_predicted"
            optimizer.zero_grad(set_to_none=True)
            outputs, pair, tensors, losses = _forward_view(
                model, entry, view_name, device, loss_args
            )
            loss = losses["loss_total"]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite factor loss")
            loss.backward()
            if step == 0 and epoch == 0:
                gradient_audit = {
                    "bi_layout": _gradient_stats(model.bi_layout),
                    "opening_head": _gradient_stats(model.matcher.opening_head),
                    "matcher": _gradient_stats(model.matcher),
                }
            torch.nn.utils.clip_grad_norm_(
                list(matcher_without_opening_parameters(model)), args.grad_clip
            )
            optimizer.step()
            train_accumulator.update(
                outputs,
                pair,
                tensors["target"],
                tensors["valid_a"],
                tensors["valid_b"],
                losses,
            )
            injection_count += int(entry["views"][view_name]["gt_candidate_injection_count"])
        model.eval()
        validation = evaluate_entries(
            model,
            entries["val"],
            "recall_predicted",
            device,
            records["val"],
            margin_enabled=margin,
            pair_rows=True,
        )["metrics"]
        scheduler.step()
        epoch_report = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {**train_accumulator.result(), "gt_candidate_injection_count": injection_count},
            "validation_predicted_only": {**validation, "gt_candidate_injection_count": 0},
            "runtime_seconds": time.perf_counter() - start,
            "pair_order_sha256": order_hashes[-1],
            "finite": _finite_mapping({"train": train_accumulator.result(), "val": validation}),
        }
        epochs.append(epoch_report)
        key = (
            float(validation["harmonic_flow_score"]),
            float(validation["balanced_flow_score"]),
            float(validation["end_to_end_candidate_top1"]),
        )
        if best is None or key > tuple(best["selection_key"]):
            best = {
                "epoch": epoch,
                "selection_key": list(key),
                "metrics": validation,
            }
            best_state = copy.deepcopy(model.matcher.state_dict())
        print(
            "{} epoch {}/{} train_loss={:.4f} val_accept={:.4f} val_s3={:.4f} neg={:.4f} H={:.4f}".format(
                factor_id,
                epoch + 1,
                args.epochs,
                train_accumulator.result()["loss_total"],
                validation["positive_acceptance_rate"],
                validation["end_to_end_candidate_top1"],
                validation["negative_rejection_rate"],
                validation["harmonic_flow_score"],
            ),
            flush=True,
        )
    assert best is not None and best_state is not None
    model.matcher.load_state_dict(best_state, strict=True)
    best_eval = evaluate_entries(
        model,
        entries["val"],
        "recall_predicted",
        device,
        records["val"],
        margin_enabled=margin,
        pair_rows=True,
    )
    upstream_after = {
        "bi_layout": _state_hash(model.bi_layout),
        "opening_head": _state_hash(model.matcher.opening_head),
    }
    report = {
        "schema_version": "zind-t4-medium-factor-v1",
        "protocol_version": PROTOCOL_VERSION,
        "factor_id": factor_id,
        "train_gt_injection": injection,
        "dustbin_margin": margin,
        "initialization": initialization,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "matcher_lr": args.matcher_lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "scheduler": "cosine to minimum factor {}".format(args.minimum_lr_factor),
            "seed": args.seed,
            "candidate_missing_target_policy": "mask_positive",
            "matcher_train_state": True,
            "fixed_epoch_pair_order_sha256": order_hashes,
        },
        "initial_validation_predicted_only": baseline,
        "epochs": epochs,
        "best": {**best, "evaluation": best_eval},
        "gradient_audit": gradient_audit,
        "initial_matcher_hash": initial_matcher_hash,
        "final_matcher_hash": _state_hash(model.matcher),
        "matcher_updated": initial_matcher_hash != _state_hash(model.matcher),
        "upstream_hash_before": upstream_before,
        "upstream_hash_after": upstream_after,
        "upstream_hash_unchanged": upstream_before == upstream_after,
        "validation_gt_candidate_injection_count": 0,
        "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    factor_path = output / "factors" / (factor_id + ".json")
    _atomic_json(factor_path, report)
    return report


def _protocol(args: argparse.Namespace) -> Dict[str, Any]:
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "sample": {
            "train": {"positive": args.train_positive, "negative": args.train_negative},
            "val": {"positive": args.val_positive, "negative": args.val_negative},
            "selector": "deterministic house-first, model-performance independent",
        },
        "proposal": {
            "mode": "recall_topk",
            "threshold": 0.47,
            "peak_top_k": 4,
            "peak_min_probability": 0.20,
            "expansion_threshold": 0.12,
            "dedup_iou": 0.50,
            "max_candidates": 12,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "matcher_lr": args.matcher_lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "minimum_lr_factor": args.minimum_lr_factor,
            "seed": args.seed,
            "candidate_missing_target_policy": "mask_positive",
            "ranking_loss_weight": 0.1,
        },
        "factors": {
            "I0_M0": {"train_gt_injection": False, "dustbin_margin": False},
            "I1_M0": {"train_gt_injection": True, "dustbin_margin": False},
            "I0_M1": {"train_gt_injection": False, "dustbin_margin": True},
            "I1_M1": {"train_gt_injection": True, "dustbin_margin": True},
        },
        "launch_policy": {
            "status": "INHERITED_MINIMUM_NON_REGRESSION",
            "source": "matcher_only_preflight/gpu_small_preflight_gate_v2.json",
            "evaluated_factor": "I1_M1 (the formal injection+margin configuration)",
            "criteria": [
                "all finite; matcher-only gradients; frozen upstream unchanged",
                "validation predicted-only with zero GT injection",
                "positive acceptance or S3 improves",
                "negative rejection drops by at most 5 percentage points",
                "Harmonic is not below identical-proposal T3",
            ],
            "note": (
                "the received message ends at the 2x2 table header; this reuses "
                "the already-established GPU gate and does not loosen it"
            ),
        },
        "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    protocol["protocol_sha256"] = _canonical_sha(protocol)
    return protocol


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size != 4:
        raise ValueError("the frozen comparison requires batch_size=4")
    _reject_test_paths(
        [
            args.train_manifest,
            args.val_manifest,
            args.train_cache_dir,
            args.val_cache_dir,
            args.small_val_manifest,
        ]
    )
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(args)
    _atomic_json(output / "protocol.json", protocol)
    sample_report = prepare_manifests(args, output)
    if args.sanitize_val_cache:
        sanitize_validation_cache(output)
        return 0
    if args.prepare_only:
        return 0
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("medium diagnostic requires CUDA")
    small = None
    if not args.skip_small_audit:
        small = small_audit(args, output, device)
        if not small["parity_pass"]:
            _atomic_json(
                output / "decision.json",
                {
                    "decision": "STOP_EPOCH0_PARITY",
                    "formal_training_started": False,
                    "test_data_accessed": False,
                },
            )
            return 2
    t3, threshold, _ = _load_model(args, device)
    entries, cache_schema = build_or_load_cache(
        args, output, t3, threshold, sample_report
    )
    distribution = cache_distribution(entries)
    if distribution["validation_injected_view_present"]:
        raise RuntimeError(
            "validation cache contains a forbidden injected candidate view"
        )
    _atomic_json(output / "cache_distribution.json", distribution)
    records = {
        "train": _records(output / "train_medium.json"),
        "val": _records(output / "val_medium.json"),
    }
    p0 = evaluate_entries(
        t3,
        entries["val"],
        "legacy_predicted",
        device,
        records["val"],
        margin_enabled=True,
    )
    p1 = evaluate_entries(
        t3,
        entries["val"],
        "recall_predicted",
        device,
        records["val"],
        margin_enabled=True,
    )
    proposal = {
        "P0_t3_legacy": p0,
        "P1_t3_recall_topk": p1,
        "comparison": _proposal_comparison(p0, p1),
        "test_data_accessed": False,
    }
    _atomic_json(output / "medium_proposal_isolation.json", proposal)
    if args.skip_factors:
        return 0
    factor_reports = {}
    for factor_id, config in protocol["factors"].items():
        factor_reports[factor_id] = run_factor(
            args,
            factor_id,
            config["train_gt_injection"],
            config["dustbin_margin"],
            entries,
            records,
            device,
            output,
        )
    formal_factor = factor_reports["I1_M1"]
    formal_metrics = formal_factor["best"]["metrics"]
    baseline_metrics = p1["metrics"]
    inherited_gates = {
        "all_finite": all(epoch["finite"] for epoch in formal_factor["epochs"]),
        "matcher_only_gradients": (
            formal_factor["gradient_audit"]["bi_layout"]["gradient_tensors"] == 0
            and formal_factor["gradient_audit"]["opening_head"]["gradient_tensors"] == 0
            and formal_factor["gradient_audit"]["matcher"]["nonzero_gradient_tensors"] > 0
        ),
        "upstream_unchanged": formal_factor["upstream_hash_unchanged"],
        "validation_predicted_only": formal_factor["validation_gt_candidate_injection_count"] == 0,
        "positive_acceptance_or_s3_improves": (
            formal_metrics["positive_acceptance_rate"]
            > baseline_metrics["positive_acceptance_rate"]
            or formal_metrics["end_to_end_candidate_top1"]
            > baseline_metrics["end_to_end_candidate_top1"]
        ),
        "negative_rejection_drop_at_most_5pp": (
            formal_metrics["negative_rejection_rate"]
            >= baseline_metrics["negative_rejection_rate"] - 0.05
        ),
        "harmonic_not_below_same_proposal_t3": (
            formal_metrics["harmonic_flow_score"]
            >= baseline_metrics["harmonic_flow_score"]
        ),
    }
    medium_gate_pass = all(inherited_gates.values())
    summary = {
        "schema_version": "zind-t4-medium-diagnostic-summary-v1",
        "protocol": protocol,
        "sample_manifest": sample_report,
        "upstream_cache": {
            "schema_sha256": cache_schema["schema_sha256"],
            "config_sha256": cache_schema["config_sha256"],
            "splits": {
                split: {
                    "pair_count": cache_schema["splits"][split]["pair_count"],
                    "batch_count": cache_schema["splits"][split]["batch_count"],
                }
                for split in ("train", "val")
            },
        },
        "small_audit": {
            "parity_pass": None if small is None else small["parity_pass"],
            "reported_baseline_proposal": None if small is None else small["reported_baseline_proposal"],
        },
        "proposal_isolation": {
            "P0": p0["metrics"],
            "P1": p1["metrics"],
            "comparison": proposal["comparison"],
        },
        "factors": {
            factor_id: {
                "train_gt_injection": report["train_gt_injection"],
                "dustbin_margin": report["dustbin_margin"],
                "best_epoch": report["best"]["epoch"],
                "best_metrics": report["best"]["metrics"],
                "upstream_hash_unchanged": report["upstream_hash_unchanged"],
                "validation_gt_candidate_injection_count": 0,
            }
            for factor_id, report in factor_reports.items()
        },
        "medium_gate": {"gates": inherited_gates, "pass": medium_gate_pass},
        "decision": "PASS_MEDIUM_GATE" if medium_gate_pass else "STOP_MEDIUM_GATE",
        "formal_training_started": False,
        "formal_training_path": str(
            REPO_ROOT
            / "checkpoints/End_To_End/zind_e2e_v3/g0_seed123/T4_recall_matcher_ft"
        ),
        "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        output / "decision.json",
        {
            "decision": summary["decision"],
            "formal_training_started": False,
            "reason": (
                "diagnostic gate passed; formal launch requires caller orchestration"
                if medium_gate_pass
                else "one or more inherited medium non-regression gates failed"
            ),
            "test_data_accessed": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
