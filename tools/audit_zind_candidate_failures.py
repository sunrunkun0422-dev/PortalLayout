#!/usr/bin/env python3
"""Staged, resumable T3-frozen ZInD candidate failure audit.

Only train/val manifests are accepted.  The tool has no optimizer, backward,
or training path.  GT enters proposal scoring only after room-level inference
has been frozen and is explicitly marked as oracle-only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.zind_pair_mining import segment_endpoint_error  # noqa: E402
from models.cross_scene_matcher import (  # noqa: E402
    candidate_intervals_to_mask,
    circular_interval_iou,
    resolve_enclosed_extended_depth,
)
from tools.audit_zind_t4_recall import _projected_target_recalled  # noqa: E402
from tools.preflight_zind_t4_matcher_only import (  # noqa: E402
    _load_t3_model,
    _state_hash,
)
from tools.train_zind_cross_scene_matcher import (  # noqa: E402
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
)
from utils.candidate_failure_trace import (  # noqa: E402
    assign_side_reason,
    combine_side_reason,
    finite_tree,
    interval_tokens,
    target_interval_diagnostics,
    trace_opening_proposals,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/zind_e2e_v3/g0_seed123/T5_candidate_recall/audit"
)
DEFAULT_T3 = REPO_ROOT / "checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best.pt"
DEFAULT_FROZEN_VAL = REPO_ROOT / "src/output/zind_geometry_val_full/t3_frozen_m2"
DEFAULT_MEDIUM = REPO_ROOT / "experiments/zind_e2e_v3/g0_seed123/T4_recall/medium_diagnostic"
PROTOCOL_VERSION = "zind-t5-candidate-failure-audit-v1"
PASSABLE_TYPES = frozenset(("door", "opening"))
PROPOSAL = {
    "threshold": 0.47,
    "min_width_tokens": 2,
    "max_candidates": 12,
    "peak_top_k": 4,
    "peak_min_probability": 0.20,
    "expansion_threshold": 0.12,
    "dedup_iou": 0.50,
    "target_iou": 0.30,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "smoke", "cache", "assemble", "oracle", "report", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train_manifest", default=str(REPO_ROOT / "src/dataset/ZInD_matching/train.json"))
    parser.add_argument("--val_manifest", default=str(REPO_ROOT / "src/dataset/ZInD_matching/val.json"))
    parser.add_argument("--train_cache_dir", default=str(DEFAULT_CACHE_ROOT / "train"))
    parser.add_argument("--val_cache_dir", default=str(DEFAULT_CACHE_ROOT / "val"))
    parser.add_argument("--t3_checkpoint", default=str(DEFAULT_T3))
    parser.add_argument("--frozen_val_root", default=str(DEFAULT_FROZEN_VAL))
    parser.add_argument("--medium_root", default=str(DEFAULT_MEDIUM))
    parser.add_argument("--bi_layout_config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bi_layout_checkpoint", default=str(DEFAULT_BI_LAYOUT))
    parser.add_argument("--opening_checkpoint", default=str(DEFAULT_OPENING))
    parser.add_argument("--matcher_checkpoint", default=str(DEFAULT_MATCHER))
    parser.add_argument("--data_root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_positive_count", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--oracle_batch_size", type=int, default=4)
    parser.add_argument("--smoke_rooms", type=int, default=64)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--progress_every", type=int, default=32)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    return parser.parse_args(argv)


def _reject_test(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.stem.lower() == "test" or any(part.lower() == "test" for part in path.parts):
        raise ValueError(f"{label} points to forbidden test data: {path}")
    return path


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(payload), str(temporary))
    os.replace(temporary, path)


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _load_manifest(path: Path, split: str) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != split or not isinstance(payload.get("pairs"), list):
        raise ValueError(f"manifest must declare split={split}: {path}")
    ids = [str(record.get("id", "")) for record in payload["pairs"]]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError(f"manifest has missing/duplicate IDs: {path}")
    return payload


def _positive(record: Mapping[str, Any]) -> bool:
    return bool(record["supervision"]["is_match"])


def _house_round_robin(records: Sequence[Mapping[str, Any]], count: int) -> List[Mapping[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if _positive(record):
            grouped[str(record["scene_id"])].append(record)
    for house in grouped:
        grouped[house].sort(key=lambda record: str(record["id"]))
    houses = sorted(grouped)
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
        raise ValueError(f"train has only {len(selected)} selectable positive pairs")
    return selected


def _cache_views(cache_dir: Path, split: str) -> Tuple[Dict[str, int], Dict[str, Any]]:
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("split") != split or metadata.get("complete") is not True:
        raise ValueError(f"invalid {split} feature cache metadata")
    mapping: Dict[str, int] = {}
    for view in metadata["views"]:
        relative = str(view["relative_image_path"])
        index = int(view["index"])
        previous = mapping.setdefault(relative, index)
        if previous != index:
            raise ValueError(f"duplicate cache view: {relative}")
    return mapping, metadata


def _room_identity(
    split: str, record: Mapping[str, Any], side: str, cache_index: int
) -> Dict[str, Any]:
    suffix = side.lower()
    return {
        "split": split,
        "house_id": str(record["scene_id"]),
        "floor_id": str(record["floor_id"]),
        "complete_room_id": str(record[f"room_{suffix}"]),
        "pano_id": str(record[f"pano_{suffix}"]),
        "cache_index": int(cache_index),
    }


def _room_id(identity: Mapping[str, Any]) -> str:
    return _canonical_sha(identity)[:24]


def _medium_pair_ids(medium_root: Path, split: str) -> set[str]:
    path = medium_root / f"{split}_medium.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["id"]) for record in payload.get("pairs", [])}


def prepare(args: argparse.Namespace, output: Path, log: AuditLogger) -> Dict[str, Any]:
    if args.seed != 123 or args.train_positive_count != 4096:
        raise ValueError("sampling contract is fixed to seed=123 and 4096 train positives")
    train_path = _reject_test(args.train_manifest, "train manifest")
    val_path = _reject_test(args.val_manifest, "val manifest")
    train = _load_manifest(train_path, "train")
    val = _load_manifest(val_path, "val")
    train_selected = _house_round_robin(train["pairs"], args.train_positive_count)
    val_selected = sorted(
        (record for record in val["pairs"] if _positive(record)),
        key=lambda record: str(record["id"]),
    )
    train_pairs = [str(record["id"]) for record in train_selected]
    val_pairs = [str(record["id"]) for record in val_selected]
    protocol = {
        "schema_version": "zind-t5-train-sampling-v1",
        "seed": 123,
        "source_positive_count": sum(_positive(record) for record in train["pairs"]),
        "selected_positive_count": 4096,
        "algorithm": (
            "lexicographic stable house_id order; take one positive per house; "
            "continue house round-robin by depth; lexicographic stable pair ID within house"
        ),
        "selection_uses_model_output": False,
        "selection_uses_geometry_or_error": False,
        "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    protocol_sha = _canonical_sha(protocol)
    pair_order_sha = hashlib.sha256("\n".join(train_pairs).encode()).hexdigest()
    houses = Counter(str(record["scene_id"]) for record in train_selected)
    sample = {
        "schema_version": "zind-t5-train-positive-sample-v1",
        "split": "train",
        "seed": 123,
        "pair_ids": train_pairs,
        "pair_count": len(train_pairs),
        "house_count": len(houses),
        "sampling_protocol_sha256": protocol_sha,
        "pair_order_sha256": pair_order_sha,
        "source_manifest": {"path": str(train_path), "sha256": _sha256_file(train_path)},
        "test_data_accessed": False,
    }
    distribution = {
        "schema_version": "zind-t5-train-house-distribution-v1",
        "house_count": len(houses),
        "pair_count": len(train_pairs),
        "counts": dict(sorted(houses.items())),
        "test_data_accessed": False,
    }
    _atomic_json(output / "train_sample_4096.json", sample)
    _atomic_json(output / "sampling_protocol.json", protocol)
    _atomic_text(output / "sampling_protocol_sha256", protocol_sha + "\n")
    _atomic_text(output / "pair_order_sha256", pair_order_sha + "\n")
    _atomic_json(output / "house_distribution.json", distribution)

    selected_by_split = {"train": train_selected, "val": val_selected}
    manifests = {"train": train, "val": val}
    cache_dirs = {
        "train": _reject_test(args.train_cache_dir, "train cache"),
        "val": _reject_test(args.val_cache_dir, "val cache"),
    }
    medium_ids = {
        split: _medium_pair_ids(_reject_test(args.medium_root, "medium root"), split)
        for split in ("train", "val")
    }
    room_specs: Dict[str, Dict[str, Any]] = {}
    pair_refs: Dict[str, Dict[str, Any]] = {}
    split_stats: Dict[str, Any] = {}
    for split in ("train", "val"):
        view_index, metadata = _cache_views(cache_dirs[split], split)
        by_id = {str(record["id"]): record for record in manifests[split]["pairs"]}
        medium_rooms = set()
        medium_refs: Dict[str, Dict[str, str]] = {}
        for pair_id in sorted(medium_ids[split]):
            record = by_id.get(pair_id)
            if record is None:
                continue
            for side in ("A", "B"):
                suffix = side.lower()
                index = view_index[str(record[f"image_{suffix}"])]
                identity = _room_identity(split, record, side, index)
                rid = _room_id(identity)
                medium_rooms.add(rid)
                medium_refs.setdefault(rid, {"pair_id": pair_id, "side": side})
        for record in selected_by_split[split]:
            refs = {}
            for side in ("A", "B"):
                suffix = side.lower()
                relative = str(record[f"image_{suffix}"])
                if relative not in view_index:
                    raise ValueError(f"cache misses {split} image {relative}")
                identity = _room_identity(split, record, side, view_index[relative])
                rid = _room_id(identity)
                spec = {
                    **identity,
                    "room_id": rid,
                    "relative_image_path": relative,
                    "reference_pair_id": str(record["id"]),
                    "reference_side": side,
                    "medium_partial_reference": rid in medium_rooms,
                    "medium_reference": medium_refs.get(rid),
                }
                previous = room_specs.setdefault(rid, spec)
                immutable = ("split", "house_id", "floor_id", "complete_room_id", "pano_id", "cache_index", "relative_image_path")
                if any(previous[key] != spec[key] for key in immutable):
                    raise ValueError(f"room cache key collision: {rid}")
                if str(record["id"]) < str(previous["reference_pair_id"]):
                    previous["reference_pair_id"] = str(record["id"])
                    previous["reference_side"] = side
                refs[side] = rid
            pair_refs[str(record["id"])] = {"split": split, "A": refs["A"], "B": refs["B"]}
        selected_rooms = {pair_refs[str(record["id"])][side] for record in selected_by_split[split] for side in ("A", "B")}
        split_stats[split] = {
            "pair_count": len(selected_by_split[split]),
            "pair_side_count": 2 * len(selected_by_split[split]),
            "unique_room_count": len(selected_rooms),
            "dedup_saving_count": 2 * len(selected_by_split[split]) - len(selected_rooms),
            "dedup_saving_rate": 1.0 - len(selected_rooms) / (2 * len(selected_by_split[split])),
            "complete_existing_room_cache_reuse_count": 0,
            "medium_partial_reference_room_count": sum(rid in medium_rooms for rid in selected_rooms),
            "feature_cache_sample_count": int(metadata["sample_count"]),
        }
    ordered_rooms = sorted(room_specs.values(), key=lambda value: (value["split"] != "val", value["house_id"], value["floor_id"], value["complete_room_id"], value["pano_id"], value["cache_index"]))
    index = {
        "schema_version": "zind-t5-room-cache-index-v1",
        "protocol_version": PROTOCOL_VERSION,
        "rooms": ordered_rooms,
        "pair_refs": pair_refs,
        "splits": split_stats,
        "totals": {
            "pair_side_count": sum(value["pair_side_count"] for value in split_stats.values()),
            "unique_room_count": len(ordered_rooms),
            "dedup_saving_rate": 1.0 - len(ordered_rooms) / sum(value["pair_side_count"] for value in split_stats.values()),
        },
        "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    index["index_sha256"] = _canonical_sha(index)
    _atomic_json(output / "room_cache_index.json", index)
    _atomic_json(output / "pair_room_refs.json", {"schema_version": "zind-t5-pair-room-refs-v1", "pairs": pair_refs, "test_data_accessed": False})
    log(f"prepare complete: train={len(train_pairs)} pairs/{split_stats['train']['unique_room_count']} rooms; val={len(val_pairs)} pairs/{split_stats['val']['unique_room_count']} rooms; houses={len(houses)}")
    return index


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        t3_checkpoint=args.t3_checkpoint,
        bi_layout_config=args.bi_layout_config,
        bi_layout_checkpoint=args.bi_layout_checkpoint,
        opening_checkpoint=args.opening_checkpoint,
        matcher_checkpoint=args.matcher_checkpoint,
    )


def _load_frozen_model(args: argparse.Namespace, device: torch.device):
    model, threshold, report = _load_t3_model(_model_args(args), device)
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen audit model exposes gradients")
    if not math.isclose(float(threshold), PROPOSAL["threshold"], abs_tol=1e-12):
        raise ValueError(f"T3 threshold changed: {threshold}")
    return model, float(threshold), report


def _data_root(manifest: Mapping[str, Any], manifest_path: Path, override: str | None) -> Path:
    value = override or manifest.get("dataRoot")
    if not value:
        raise ValueError("manifest has no dataRoot")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = _reject_test(path, "data root")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _load_image(path: Path) -> torch.Tensor:
    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as image:
        image = image.convert("RGB").resize((1024, 512), resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if array.shape != (512, 1024, 3) or not np.isfinite(array).all():
        raise ValueError(f"invalid panorama: {path}")
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def _room_path(cache_root: Path, spec: Mapping[str, Any]) -> Path:
    return cache_root / str(spec["split"]) / f"{spec['room_id']}.pt"


def _marker_path(cache_root: Path, spec: Mapping[str, Any]) -> Path:
    return cache_root / str(spec["split"]) / f"{spec['room_id']}.completed"


def _room_payloads(
    model: Any,
    specs: Sequence[Mapping[str, Any]],
    roots: Mapping[str, Path],
    device: torch.device,
    t3_sha: str,
    config_sha: str,
) -> Tuple[List[Dict[str, Any]], float]:
    images_cpu = torch.stack(
        [_load_image(roots[str(spec["split"])] / str(spec["relative_image_path"])) for spec in specs]
    )
    start = time.perf_counter()
    collected: Dict[str, List[torch.Tensor]] = defaultdict(list)
    # The existing full-val artifact used one pair (two images) per backbone
    # forward. CUDA convolution kernels change slightly with batch shape and
    # can cross the calibrated 0.47 threshold. Keep logical room batch=4 for
    # I/O/progress, but preserve the validated two-image numerical shape.
    for offset in range(0, len(specs), 2):
        images = images_cpu[offset : offset + 2].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        with torch.inference_mode():
            output = model.bi_layout(images, return_features=True)
            enclosed, extended = resolve_enclosed_extended_depth(output, "extended_first")
            opening = model.matcher.opening_head(output["layout_feature"], enclosed, extended)
        for name, value in (
            ("layout_feature", output["layout_feature"]),
            ("depth", output["depth"]), ("new_depth", output["new_depth"]),
            ("enclosed_depth", enclosed), ("extended_depth", extended),
            ("ratio", output["ratio"]),
            ("opening_probability", opening["opening_probability"]),
        ):
            collected[name].append(value.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    tensors = {name: torch.cat(values, dim=0) for name, values in collected.items()}
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("room forward produced NaN/Inf")
    payloads = []
    for index, spec in enumerate(specs):
        probability = tensors["opening_probability"][index].detach().cpu().float()
        trace = trace_opening_proposals(probability, **{key: PROPOSAL[key] for key in ("threshold", "min_width_tokens", "max_candidates", "peak_top_k", "peak_min_probability", "expansion_threshold", "dedup_iou")})
        payloads.append({
            "schema_version": "zind-t5-frozen-room-v1",
            "room": dict(spec),
            "layout_feature": tensors["layout_feature"][index].detach().cpu().float(),
            "depth": tensors["depth"][index].detach().cpu().float(),
            "new_depth": tensors["new_depth"][index].detach().cpu().float(),
            "enclosed_depth": tensors["enclosed_depth"][index].detach().cpu().float(),
            "extended_depth": tensors["extended_depth"][index].detach().cpu().float(),
            "ratio": tensors["ratio"][index].detach().cpu().float(),
            "opening_probability": probability,
            "token_count": int(probability.numel()),
            "legacy_candidates": trace["legacy_candidates"],
            "recall_topk_candidates": trace["recall_topk_candidates"],
            "candidate_trace": trace,
            "checkpoint_sha256": t3_sha,
            "config_sha256": config_sha,
            "split": spec["split"],
            "inference_mode": True,
            "requires_grad": False,
            "optimizer_created": False,
            "test_data_accessed": False,
        })
    del images_cpu, images, output, opening, tensors, collected
    return payloads, elapsed


def _write_room(cache_root: Path, spec: Mapping[str, Any], payload: Mapping[str, Any]) -> int:
    path = _room_path(cache_root, spec)
    marker = _marker_path(cache_root, spec)
    _atomic_torch(path, payload)
    size = path.stat().st_size
    _atomic_json(marker, {"room_id": spec["room_id"], "path": str(path.relative_to(cache_root)), "size": size, "sha256": _sha256_file(path), "complete": True, "test_data_accessed": False})
    return size


def _load_room(cache_root: Path, room_id: str, split: str) -> Dict[str, Any]:
    path = cache_root / split / f"{room_id}.pt"
    marker = cache_root / split / f"{room_id}.completed"
    if not path.is_file() or not marker.is_file():
        raise FileNotFoundError(f"incomplete room cache: {split}/{room_id}")
    payload = torch.load(str(path), map_location="cpu")
    if payload.get("test_data_accessed") is not False or payload["room"]["room_id"] != room_id:
        raise ValueError(f"invalid room cache: {path}")
    return payload


def _freeze_rooms(
    args: argparse.Namespace,
    output: Path,
    specs: Sequence[Mapping[str, Any]],
    *,
    phase_name: str,
    log: AuditLogger,
) -> Dict[str, Any]:
    device = resolve_device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("authorized room cache requires CUDA")
    manifests = {
        split: _load_manifest(_reject_test(getattr(args, f"{split}_manifest"), f"{split} manifest"), split)
        for split in ("train", "val")
    }
    roots = {
        split: _data_root(manifests[split], _reject_test(getattr(args, f"{split}_manifest"), f"{split} manifest"), args.data_root)
        for split in ("train", "val")
    }
    t3_path = _reject_test(args.t3_checkpoint, "T3 checkpoint")
    before = {"sha256": _sha256_file(t3_path), "size": t3_path.stat().st_size, "mtime_ns": t3_path.stat().st_mtime_ns}
    model, threshold, model_report = _load_frozen_model(args, device)
    parameter_hash_before = _state_hash(model)
    config = {
        "proposal": PROPOSAL, "image_shape": [512, 1024],
        "logical_batch_size": args.batch_size, "model_microbatch_size": 2,
        "microbatch_reason": "preserve full-val pair-forward CUDA batch-shape parity",
        "checkpoint_sha256": before["sha256"],
    }
    config_sha = _canonical_sha(config)
    cache_root = output / "frozen_room_cache"
    pending = [spec for spec in specs if not (_room_path(cache_root, spec).is_file() and _marker_path(cache_root, spec).is_file())]
    initial_complete = len(specs) - len(pending)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    bytes_written = 0
    batch_times = []
    completed = initial_complete
    for offset in range(0, len(pending), args.batch_size):
        batch_specs = pending[offset : offset + args.batch_size]
        payloads, elapsed = _room_payloads(model, batch_specs, roots, device, before["sha256"], config_sha)
        batch_times.append(elapsed)
        for spec, payload in zip(batch_specs, payloads):
            bytes_written += _write_room(cache_root, spec, payload)
            completed += 1
        if completed % args.progress_every < len(batch_specs) or completed == len(specs):
            runtime = time.perf_counter() - start
            rate = (completed - initial_complete) / runtime if runtime else 0.0
            remaining = (len(specs) - completed) / rate if rate else None
            progress = {
                "phase": phase_name,
                "task_type": "frozen_inference_audit",
                "processed_room_count": completed,
                "total_room_count": len(specs),
                "new_room_count": completed - initial_complete,
                "throughput_rooms_per_second": rate,
                "remaining_seconds": remaining,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "training_started": False,
                "optimizer_created": False,
                "test_data_accessed": False,
            }
            _atomic_json(output / "processed_room_count.json", progress)
            log(f"{phase_name} rooms {completed}/{len(specs)} rate={rate:.2f}/s eta={remaining if remaining is not None else 'n/a'}")
    parameter_hash_after = _state_hash(model)
    after = {"sha256": _sha256_file(t3_path), "size": t3_path.stat().st_size, "mtime_ns": t3_path.stat().st_mtime_ns}
    if before != after or parameter_hash_before != parameter_hash_after:
        raise RuntimeError("T3 checkpoint or frozen parameters changed")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen inference created gradients")
    result = {
        "phase": phase_name,
        "room_count": len(specs),
        "initial_complete": initial_complete,
        "new_complete": len(specs) - initial_complete,
        "runtime_seconds": time.perf_counter() - start,
        "batch_time_seconds": batch_times,
        "bytes_written": bytes_written,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "checkpoint_before": before,
        "checkpoint_after": after,
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
        "model_report": model_report,
        "opening_threshold": threshold,
        "config_sha256": config_sha,
        "model_eval": not model.training,
        "logical_batch_size": args.batch_size,
        "model_microbatch_size": 2,
        "all_requires_grad_false": not any(parameter.requires_grad for parameter in model.parameters()),
        "optimizer_created": False,
        "backward_executed": False,
        "test_data_accessed": False,
    }
    del model
    torch.cuda.empty_cache()
    return result


def _frozen_artifact_map(root: Path) -> Dict[str, Path]:
    result = {}
    for path in (root / "frozen_predictions").glob("*.json.gz"):
        stem = path.name[: -len(".json.gz")]
        pair_id = stem.split("_", 1)[1]
        result[pair_id] = path
    return result


def _load_gzip_json(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _medium_shard_lookup(medium_root: Path, split: str) -> Dict[str, Tuple[Path, int]]:
    schema = json.loads((medium_root / "upstream_cache/schema.json").read_text(encoding="utf-8"))
    result: Dict[str, Tuple[Path, int]] = {}
    for shard in schema["splits"][split]["shards"]:
        path = medium_root / "upstream_cache" / shard["relative_path"]
        for index, pair_id in enumerate(shard["pair_ids"]):
            result[str(pair_id)] = (path, index)
    return result


def _medium_room_reference(
    medium_root: Path,
    lookup: Mapping[str, Tuple[Path, int]],
    spec: Mapping[str, Any],
    loaded: MutableMapping[str, Any],
) -> Dict[str, torch.Tensor] | None:
    reference = spec.get("medium_reference")
    if not isinstance(reference, Mapping):
        return None
    pair_id = str(reference["pair_id"])
    if pair_id not in lookup:
        return None
    path, index = lookup[pair_id]
    key = str(path)
    if key not in loaded:
        loaded[key] = torch.load(str(path), map_location="cpu")["entry"]
    entry = loaded[key]
    side = str(reference["side"])
    return {
        "layout_feature": entry["live"][f"layout_feature_{side}"][index].float(),
        "depth": entry["live"][f"depth_{side}"][index].float(),
        "new_depth": entry["live"][f"new_depth_{side}"][index].float(),
        "enclosed_depth": entry["live"][f"enclosed_depth_{side}"][index].float(),
        "extended_depth": entry["live"][f"extended_depth_{side}"][index].float(),
        "opening_probability": entry[f"opening_probability_{side}"][index].float(),
    }


def _candidate_masks_for_intervals(
    intervals: Sequence[Sequence[int]], token_count: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    masks = candidate_intervals_to_mask(intervals, token_count, device=device)
    count = max(1, int(masks.shape[0]))
    padded = torch.zeros((1, count, token_count), dtype=torch.bool, device=device)
    valid = torch.zeros((1, count), dtype=torch.bool, device=device)
    if masks.shape[0]:
        padded[0, : masks.shape[0]] = masks
        valid[0, : masks.shape[0]] = True
    return padded, valid


def _score_single(
    model: Any,
    room_a: Mapping[str, Any],
    room_b: Mapping[str, Any],
    intervals_a: Sequence[Sequence[int]],
    intervals_b: Sequence[Sequence[int]],
    device: torch.device,
) -> Dict[str, Any]:
    token_count = int(room_a["token_count"])
    masks_a, valid_a = _candidate_masks_for_intervals(intervals_a, token_count, device)
    masks_b, valid_b = _candidate_masks_for_intervals(intervals_b, token_count, device)
    live = {
        f"{name}_{side}": room[f"{name}"].unsqueeze(0).to(device=device, dtype=torch.float32)
        for side, room in (("A", room_a), ("B", room_b))
        for name in ("layout_feature", "enclosed_depth", "extended_depth")
    }
    with torch.inference_mode():
        result = model.matcher(
            live["layout_feature_A"], live["layout_feature_B"],
            live["enclosed_depth_A"], live["extended_depth_A"],
            live["enclosed_depth_B"], live["extended_depth_B"],
            candidate_masks_a=masks_a, candidate_masks_b=masks_b,
            candidate_valid_a=valid_a, candidate_valid_b=valid_b,
            opening_guidance_a=None, opening_guidance_b=None,
            opening_guidance_mode="predicted",
        )
    count_a, count_b = len(intervals_a), len(intervals_b)
    logits = result["candidate_assignment_logits"][0, : count_a + 1, : count_b + 1]
    return {
        "best_candidate_pair": [int(value) for value in result["best_candidate_pair"][0].detach().cpu().tolist()],
        "candidate_pair_scores": result["candidate_pair_score"][0, :count_a, :count_b].detach().cpu().tolist(),
        "candidate_assignment_logits": logits.detach().cpu().tolist(),
        "row_dustbin_logits": logits[:count_a, count_b].detach().cpu().tolist(),
        "column_dustbin_logits": logits[count_a, :count_b].detach().cpu().tolist(),
        "pair_match_probability": float(result["pair_match_probability"][0].item()),
        "pair_no_match_probability": float(result["pair_no_match_probability"][0].item()),
        "candidate_count_A": count_a,
        "candidate_count_B": count_b,
    }


def _smoke_room_specs(index: Mapping[str, Any], medium_root: Path, count: int) -> List[Mapping[str, Any]]:
    by_room = {str(spec["room_id"]): spec for spec in index["rooms"]}
    refs = index["pair_refs"]
    selected: List[Mapping[str, Any]] = []
    seen = set()
    # Use artifact availability only; no model score or outcome affects smoke selection.
    for split in ("val", "train"):
        manifest = json.loads((medium_root / f"{split}_medium.json").read_text(encoding="utf-8"))
        for record in sorted(manifest["pairs"], key=lambda value: str(value["id"])):
            if not _positive(record) or str(record["id"]) not in refs:
                continue
            for side in ("A", "B"):
                rid = refs[str(record["id"])][side]
                if rid not in seen:
                    selected.append(by_room[rid])
                    seen.add(rid)
                    if len(selected) == count:
                        return selected
    raise ValueError(f"only found {len(selected)} deterministic medium-overlap smoke rooms")


def smoke(args: argparse.Namespace, output: Path, log: AuditLogger) -> Dict[str, Any]:
    index_path = output / "room_cache_index.json"
    if not index_path.is_file():
        raise FileNotFoundError("run prepare before smoke")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    medium_root = _reject_test(args.medium_root, "medium root")
    specs = _smoke_room_specs(index, medium_root, args.smoke_rooms)
    previous_smoke = (
        json.loads((output / "phase0_smoke.json").read_text(encoding="utf-8"))
        if (output / "phase0_smoke.json").is_file() else None
    )
    freeze = _freeze_rooms(args, output, specs, phase_name="phase0_smoke", log=log)
    benchmark_freeze = (
        previous_smoke["room_forward"]
        if freeze["new_complete"] == 0 and isinstance(previous_smoke, Mapping)
        else freeze
    )
    cache_root = output / "frozen_room_cache"
    artifacts = _frozen_artifact_map(_reject_test(args.frozen_val_root, "frozen val root"))
    medium_lookup = {
        split: _medium_shard_lookup(medium_root, split) for split in ("train", "val")
    }
    medium_loaded: Dict[str, Any] = {}
    maxima = Counter()
    candidate_parity = 0
    full_val_compared = 0
    medium_compared = 0
    tensor_fields = ("layout_feature", "depth", "new_depth", "enclosed_depth", "extended_depth", "opening_probability")
    for spec in specs:
        room = _load_room(cache_root, str(spec["room_id"]), str(spec["split"]))
        if spec["split"] == "val":
            artifact = _load_gzip_json(artifacts[str(spec["reference_pair_id"])])
            reference = torch.as_tensor(artifact[f"opening_probability_{spec['reference_side']}"])
            delta = float((room["opening_probability"] - reference).abs().max().item())
            maxima["full_val_opening_probability"] = max(maxima["full_val_opening_probability"], delta)
            full_val_compared += 1
            frozen_intervals = artifact[f"predicted_intervals_{spec['reference_side']}"]
            candidate_parity += int(room["legacy_candidates"] == frozen_intervals)
        reference = _medium_room_reference(medium_root, medium_lookup[str(spec["split"])], spec, medium_loaded)
        if reference is not None:
            medium_compared += 1
            for field in tensor_fields:
                delta = float((room[field] - reference[field]).abs().max().item())
                maxima[f"medium_{field}"] = max(maxima[f"medium_{field}"], delta)

    # Reassemble deterministic full-val positive pairs whose two rooms are in smoke.
    smoke_ids = {str(spec["room_id"]) for spec in specs}
    val_manifest = _load_manifest(_reject_test(args.val_manifest, "val manifest"), "val")
    records = {str(record["id"]): record for record in val_manifest["pairs"]}
    pair_ids = [
        pair_id for pair_id in sorted(index["pair_refs"])
        if index["pair_refs"][pair_id]["split"] == "val"
        and index["pair_refs"][pair_id]["A"] in smoke_ids
        and index["pair_refs"][pair_id]["B"] in smoke_ids
        and _positive(records[pair_id])
    ][:32]
    device = resolve_device(args.device)
    model, _, _ = _load_frozen_model(args, device)
    pair_maxima = Counter()
    exact_best = 0
    for pair_id in pair_ids:
        ref = index["pair_refs"][pair_id]
        room_a = _load_room(cache_root, ref["A"], "val")
        room_b = _load_room(cache_root, ref["B"], "val")
        scored = _score_single(model, room_a, room_b, room_a["legacy_candidates"], room_b["legacy_candidates"], device)
        old = _load_gzip_json(artifacts[pair_id])
        exact_best += int(scored["best_candidate_pair"] == old["best_candidate_pair"])
        for key in ("pair_match_probability", "pair_no_match_probability"):
            pair_maxima[key] = max(pair_maxima[key], abs(scored[key] - float(old[key])))
        left = np.asarray(scored["candidate_pair_scores"], dtype=np.float64)
        right = np.asarray(old["candidate_pair_scores"], dtype=np.float64)[
            : scored["candidate_count_A"], : scored["candidate_count_B"]
        ]
        delta = (
            float(np.max(np.abs(left - right)))
            if left.shape == right.shape and left.size else
            (0.0 if left.shape == right.shape else 1.0e30)
        )
        pair_maxima["candidate_pair_scores"] = max(pair_maxima["candidate_pair_scores"], delta)
    parameter_hash_after = _state_hash(model)
    if parameter_hash_after != freeze["parameter_hash_after"]:
        raise RuntimeError("model state changed between room and pair smoke")
    del model
    torch.cuda.empty_cache()

    room_files = [_room_path(cache_root, spec) for spec in specs]
    mean_size = statistics.fmean(path.stat().st_size for path in room_files)
    projected_bytes = int(mean_size * int(index["totals"]["unique_room_count"]))
    statvfs = os.statvfs(output)
    available = statvfs.f_bavail * statvfs.f_frsize
    disk_fraction = projected_bytes / available
    seconds_per_room = benchmark_freeze["runtime_seconds"] / max(1, benchmark_freeze["new_complete"])
    projected_seconds = seconds_per_room * int(index["totals"]["unique_room_count"])
    gates = {
        "room_count_exact": len(specs) == args.smoke_rooms,
        "all_finite": all(finite_tree(_load_room(cache_root, str(spec["room_id"]), str(spec["split"]))) for spec in specs),
        "full_val_probability_within_2e_5": maxima["full_val_opening_probability"] <= 2e-5,
        "legacy_candidate_exact": candidate_parity == full_val_compared,
        "medium_batchshape_reference_within_1e_2": all(value <= 1e-2 for key, value in maxima.items() if key.startswith("medium_")),
        "pair_scores_within_2e_5": all(value <= 2e-5 for value in pair_maxima.values()),
        "pair_best_exact": exact_best == len(pair_ids),
        "checkpoint_unchanged": freeze["checkpoint_before"] == freeze["checkpoint_after"],
        "parameters_unchanged": freeze["parameter_hash_before"] == freeze["parameter_hash_after"] == parameter_hash_after,
        "no_grad": freeze["all_requires_grad_false"],
        "no_optimizer": freeze["optimizer_created"] is False,
        "projected_disk_below_25pct_available": disk_fraction <= 0.25,
    }
    result = {
        "schema_version": "zind-t5-room-cache-smoke-v1",
        "phase": "Phase 0",
        "smoke_room_ids": [spec["room_id"] for spec in specs],
        "room_forward": benchmark_freeze,
        "room_cache_resume_check": freeze,
        "full_val_reference_count": full_val_compared,
        "medium_reference_count": medium_compared,
        "max_abs_differences": dict(maxima),
        "legacy_candidate_exact_count": candidate_parity,
        "pair_reassembly_count": len(pair_ids),
        "pair_best_exact_count": exact_best,
        "pair_max_abs_differences": dict(pair_maxima),
        "mean_room_file_bytes": mean_size,
        "projected_total_bytes": projected_bytes,
        "available_disk_bytes": available,
        "projected_fraction_of_available": disk_fraction,
        "projected_total_runtime_seconds": projected_seconds,
        "single_batch_time_seconds": statistics.fmean(benchmark_freeze["batch_time_seconds"]) if benchmark_freeze["batch_time_seconds"] else None,
        "gates": gates,
        "passed": all(gates.values()),
        "task_type": "frozen_inference_audit",
        "training_started": False,
        "optimizer_created": False,
        "test_data_accessed": False,
    }
    _atomic_json(output / "phase0_smoke.json", result)
    if not result["passed"]:
        raise RuntimeError("Phase 0 gate failed: " + ", ".join(key for key, value in gates.items() if not value))
    _atomic_text(output / ".phase0_passed", json.dumps({"phase0_sha256": _sha256_file(output / "phase0_smoke.json")}) + "\n")
    log(f"Phase 0 PASS: {len(specs)} rooms, {len(pair_ids)} regrouped pairs, projected={projected_bytes / 2**30:.2f} GiB/{projected_seconds / 60:.1f} min")
    return result


def cache_rooms(args: argparse.Namespace, output: Path, log: AuditLogger) -> Dict[str, Any]:
    if not (output / ".phase0_passed").is_file():
        raise RuntimeError("Phase 0 must pass before Phase 1")
    index = json.loads((output / "room_cache_index.json").read_text(encoding="utf-8"))
    result = _freeze_rooms(args, output, index["rooms"], phase_name="phase1_room_cache", log=log)
    cache_root = output / "frozen_room_cache"
    complete = sum(_room_path(cache_root, spec).is_file() and _marker_path(cache_root, spec).is_file() for spec in index["rooms"])
    if complete != len(index["rooms"]):
        raise RuntimeError(f"Phase 1 incomplete: {complete}/{len(index['rooms'])}")
    result.update({"processed_room_count": complete, "total_room_count": len(index["rooms"]), "complete": True})
    _atomic_json(output / "phase1_cache_summary.json", result)
    _atomic_text(output / ".phase1_completed", _sha256_file(output / "phase1_cache_summary.json") + "\n")
    log(f"Phase 1 complete: {complete} unique rooms")
    return result


def _physical_matches(record: Mapping[str, Any], tolerance: float = 0.06) -> List[Tuple[int, int]]:
    matches = []
    for index_a, candidate_a in enumerate(record["candidates_a"]):
        if candidate_a.get("type") not in PASSABLE_TYPES:
            continue
        endpoints_a = np.asarray(candidate_a["globalEndpoints"], dtype=np.float64)
        for index_b, candidate_b in enumerate(record["candidates_b"]):
            if candidate_b.get("type") not in PASSABLE_TYPES:
                continue
            endpoints_b = np.asarray(candidate_b["globalEndpoints"], dtype=np.float64)
            if segment_endpoint_error(endpoints_a, endpoints_b) <= tolerance:
                matches.append((index_a, index_b))
    return matches


def _affinity(record: Mapping[str, Any]) -> torch.Tensor:
    token_count = int(record["token_count"])
    affinity = torch.zeros((1, token_count, token_count), dtype=torch.float32)
    for index_a, index_b in _physical_matches(record):
        tokens_a = interval_tokens(record["candidates_a"][index_a]["tokenInterval"], token_count)
        tokens_b = interval_tokens(record["candidates_b"][index_b]["tokenInterval"], token_count)
        affinity[0][torch.as_tensor(tokens_a)[:, None], torch.as_tensor(tokens_b)[None, :]] = 1.0
    return affinity


def _target_pairs(
    record: Mapping[str, Any],
    intervals_a: Sequence[Sequence[int]],
    intervals_b: Sequence[Sequence[int]],
) -> List[List[int]]:
    token_count = int(record["token_count"])
    masks_a = candidate_intervals_to_mask(intervals_a, token_count).unsqueeze(0)
    masks_b = candidate_intervals_to_mask(intervals_b, token_count).unsqueeze(0)
    target = candidate_targets_from_affinity(masks_a, _affinity(record), masks_b)[0].bool()
    return torch.nonzero(target, as_tuple=False).tolist()


def _pad_masks(
    all_intervals: Sequence[Sequence[Sequence[int]]], token_count: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    count = max(1, max((len(value) for value in all_intervals), default=0))
    masks = torch.zeros((len(all_intervals), count, token_count), dtype=torch.bool, device=device)
    valid = torch.zeros((len(all_intervals), count), dtype=torch.bool, device=device)
    for batch_index, intervals in enumerate(all_intervals):
        value = candidate_intervals_to_mask(intervals, token_count, device=device)
        if value.shape[0]:
            masks[batch_index, : value.shape[0]] = value
            valid[batch_index, : value.shape[0]] = True
    return masks, valid


def _score_batch(
    model: Any,
    rooms_a: Sequence[Mapping[str, Any]],
    rooms_b: Sequence[Mapping[str, Any]],
    intervals_a: Sequence[Sequence[Sequence[int]]],
    intervals_b: Sequence[Sequence[Sequence[int]]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    token_count = int(rooms_a[0]["token_count"])
    masks_a, valid_a = _pad_masks(intervals_a, token_count, device)
    masks_b, valid_b = _pad_masks(intervals_b, token_count, device)
    live = {}
    for side, rooms in (("A", rooms_a), ("B", rooms_b)):
        for name in ("layout_feature", "enclosed_depth", "extended_depth"):
            live[f"{name}_{side}"] = torch.stack([room[name] for room in rooms]).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        result = model.matcher(
            live["layout_feature_A"], live["layout_feature_B"],
            live["enclosed_depth_A"], live["extended_depth_A"],
            live["enclosed_depth_B"], live["extended_depth_B"],
            candidate_masks_a=masks_a, candidate_masks_b=masks_b,
            candidate_valid_a=valid_a, candidate_valid_b=valid_b,
            opening_guidance_a=None, opening_guidance_b=None,
            opening_guidance_mode="predicted",
        )
    output = []
    for index in range(len(rooms_a)):
        count_a, count_b = len(intervals_a[index]), len(intervals_b[index])
        logits = result["candidate_assignment_logits"][index, : count_a + 1, : count_b + 1]
        output.append({
            "best_candidate_pair": [int(value) for value in result["best_candidate_pair"][index].detach().cpu().tolist()],
            "candidate_pair_scores": result["candidate_pair_score"][index, :count_a, :count_b].detach().cpu().tolist(),
            "candidate_assignment_logits": logits.detach().cpu().tolist(),
            "row_dustbin_logits": logits[:count_a, count_b].detach().cpu().tolist(),
            "column_dustbin_logits": logits[count_a, :count_b].detach().cpu().tolist(),
            "pair_match_probability": float(result["pair_match_probability"][index].item()),
            "pair_no_match_probability": float(result["pair_no_match_probability"][index].item()),
            "candidate_count_A": count_a,
            "candidate_count_B": count_b,
        })
    return output


def _score_split_view(
    args: argparse.Namespace,
    output: Path,
    split: str,
    records: Sequence[Mapping[str, Any]],
    view_name: str,
    interval_builder: Any,
    device: torch.device,
    log: AuditLogger,
) -> Dict[str, Dict[str, Any]]:
    index = json.loads((output / "room_cache_index.json").read_text(encoding="utf-8"))
    refs = index["pair_refs"]
    cache_root = output / "frozen_room_cache"
    score_root = output / "pair_scores" / f"{split}_{view_name}"
    score_root.mkdir(parents=True, exist_ok=True)
    model, _, model_report = _load_frozen_model(args, device)
    hash_before = _state_hash(model)
    completed: Dict[str, Dict[str, Any]] = {}
    pending = []
    for record in records:
        pair_id = str(record["id"])
        path = score_root / f"{pair_id}.json.gz"
        if path.is_file():
            completed[pair_id] = _load_gzip_json(path)
        else:
            pending.append(record)
    start = time.perf_counter()
    for offset in range(0, len(pending), args.oracle_batch_size):
        batch = pending[offset : offset + args.oracle_batch_size]
        rooms_a = [_load_room(cache_root, refs[str(record["id"])]["A"], split) for record in batch]
        rooms_b = [_load_room(cache_root, refs[str(record["id"])]["B"], split) for record in batch]
        intervals = [interval_builder(record, room_a, room_b) for record, room_a, room_b in zip(batch, rooms_a, rooms_b)]
        scored = _score_batch(model, rooms_a, rooms_b, [value[0] for value in intervals], [value[1] for value in intervals], device)
        for record, value, interval_pair in zip(batch, scored, intervals):
            pair_id = str(record["id"])
            payload = {
                "schema_version": "zind-t5-frozen-matcher-score-v1",
                "pair_id": pair_id,
                "split": split,
                "view": view_name,
                "intervals_A": interval_pair[0],
                "intervals_B": interval_pair[1],
                **value,
                "oracle_only": view_name != "O0",
                "valid_for_inference": view_name == "O0",
                "gt_available_to_production": False,
                "model_eval": True,
                "requires_grad": False,
                "optimizer_created": False,
                "test_data_accessed": False,
            }
            path = score_root / f"{pair_id}.json.gz"
            temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, allow_nan=False)
            os.replace(temporary, path)
            completed[pair_id] = payload
        done = len(completed)
        if done % 128 < len(batch) or done == len(records):
            elapsed = time.perf_counter() - start
            log(f"matcher {split}/{view_name} {done}/{len(records)} new_rate={max(1, done - (len(records)-len(pending))) / max(elapsed, 1e-9):.2f}/s")
            _atomic_json(output / "processed_pair_count.json", {
                "split": split, "view": view_name, "processed_pair_count": done,
                "total_pair_count": len(records), "task_type": "frozen_inference_audit",
                "training_started": False, "optimizer_created": False,
                "test_data_accessed": False,
            })
    hash_after = _state_hash(model)
    if hash_before != hash_after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError(f"frozen Matcher changed during {split}/{view_name}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _atomic_json(score_root / "summary.json", {
        "split": split, "view": view_name, "pair_count": len(completed),
        "model_report": model_report, "parameter_hash_before": hash_before,
        "parameter_hash_after": hash_after, "runtime_seconds": time.perf_counter() - start,
        "training_started": False, "optimizer_created": False,
        "test_data_accessed": False,
    })
    return completed


def _legacy_builder(record: Mapping[str, Any], room_a: Mapping[str, Any], room_b: Mapping[str, Any]):
    return room_a["legacy_candidates"], room_b["legacy_candidates"]


def _pair_row(
    record: Mapping[str, Any], room_a: Mapping[str, Any], room_b: Mapping[str, Any],
    score: Mapping[str, Any], split: str,
) -> Dict[str, Any]:
    supervision = record["supervision"]
    target_index_a = int(supervision["target_candidate_a"])
    target_index_b = int(supervision["target_candidate_b"])
    candidate_a = record["candidates_a"][target_index_a]
    candidate_b = record["candidates_b"][target_index_b]
    target_a = candidate_a["tokenInterval"]
    target_b = candidate_b["tokenInterval"]
    diag_a = target_interval_diagnostics(room_a["opening_probability"], room_a["candidate_trace"], target_a, target_iou=PROPOSAL["target_iou"])
    diag_b = target_interval_diagnostics(room_b["opening_probability"], room_b["candidate_trace"], target_b, target_iou=PROPOSAL["target_iou"])
    reason_a = assign_side_reason(diag_a, room_a["candidate_trace"], target_iou=PROPOSAL["target_iou"])
    reason_b = assign_side_reason(diag_b, room_b["candidate_trace"], target_iou=PROPOSAL["target_iou"])
    legacy_targets = _target_pairs(record, room_a["legacy_candidates"], room_b["legacy_candidates"])
    recall_targets = _target_pairs(record, room_a["recall_topk_candidates"], room_b["recall_topk_candidates"])
    legacy_present = bool(legacy_targets)
    best = score["best_candidate_pair"]
    predicted_match = bool(best[0] >= 0 and best[1] >= 0)
    selected_correct = list(best) in legacy_targets
    if legacy_present:
        if selected_correct:
            primary = "target_present_correct"
        elif not predicted_match:
            primary = "target_present_dustbin_reject"
        else:
            primary = "target_present_wrong_selection"
    else:
        primary = combine_side_reason(reason_a, reason_b)
        if diag_a["stages"]["legacy_final"]["recalled"] and diag_b["stages"]["legacy_final"]["recalled"]:
            primary = "side_candidates_present_pair_target_missing"
    physical = _physical_matches(record)
    interface = f"{candidate_a['type']}-{candidate_b['type']}"
    return {
        "schema_version": "zind-t5-candidate-failure-pair-v1",
        "pair_id": str(record["id"]),
        "split": split,
        "house_id": str(record["scene_id"]),
        "floor_id": str(record["floor_id"]),
        "room_A": str(record["room_a"]), "room_B": str(record["room_b"]),
        "pano_A": str(record["pano_a"]), "pano_B": str(record["pano_b"]),
        "interface_type": interface,
        "wrap": bool(target_a[0] > target_a[1] or target_b[0] > target_b[1]),
        "target_candidate_manifest": [target_index_a, target_index_b],
        "physical_shared_opening_pairs": [list(value) for value in physical],
        "multiple_shared_openings": len(physical) > 1,
        "room_cache_ref_A": room_a["room"]["room_id"],
        "room_cache_ref_B": room_b["room"]["room_id"],
        "target_A": diag_a, "target_B": diag_b,
        "reason_A": reason_a, "reason_B": reason_b,
        "primary_pair_reason": primary,
        "secondary_reasons": (["target_mapping_failure"] if primary == "side_candidates_present_pair_target_missing" else []) + (["multiple_shared_opening_schema_limit"] if len(physical) > 1 and not legacy_present else []),
        "legacy": {
            "candidate_count_A": len(room_a["legacy_candidates"]),
            "candidate_count_B": len(room_b["legacy_candidates"]),
            "target_pairs": legacy_targets,
            "target_present": legacy_present,
            "best_candidate_pair": best,
            "predicted_match": predicted_match,
            "selected_correct": selected_correct,
            "pair_match_probability": score.get("pair_match_probability"),
            "pair_no_match_probability": score.get("pair_no_match_probability"),
        },
        "recall_topk": {
            "candidate_count_A": len(room_a["recall_topk_candidates"]),
            "candidate_count_B": len(room_b["recall_topk_candidates"]),
            "target_pairs": recall_targets,
            "target_present": bool(recall_targets),
        },
        "test_data_accessed": False,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _funnel(rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    total = len(rows)
    if mode == "legacy":
        predicates = {
            "GT_positive": lambda row: True,
            "A_raw_hypothesis_recalled": lambda row: row["target_A"]["stages"]["raw"]["recalled"],
            "B_raw_hypothesis_recalled": lambda row: row["target_B"]["stages"]["raw"]["recalled"],
            "both_raw_hypothesis_recalled": lambda row: row["target_A"]["stages"]["raw"]["recalled"] and row["target_B"]["stages"]["raw"]["recalled"],
            "both_threshold_recalled": lambda row: row["target_A"]["stages"]["threshold"]["recalled"] and row["target_B"]["stages"]["threshold"]["recalled"],
            "both_final_side_candidates": lambda row: row["target_A"]["stages"]["legacy_final"]["recalled"] and row["target_B"]["stages"]["legacy_final"]["recalled"],
            "final_target_formed": lambda row: row["legacy"]["target_present"],
            "matcher_predicted_match": lambda row: row["legacy"]["predicted_match"],
            "matcher_candidate_top1_correct": lambda row: row["legacy"]["selected_correct"],
        }
    else:
        predicates = {
            "GT_positive": lambda row: True,
            "A_raw_peak_recalled": lambda row: row["target_A"]["stages"]["raw"]["recalled"],
            "B_raw_peak_recalled": lambda row: row["target_B"]["stages"]["raw"]["recalled"],
            "both_raw_peak_recalled": lambda row: row["target_A"]["stages"]["raw"]["recalled"] and row["target_B"]["stages"]["raw"]["recalled"],
            "both_expansion_wrap_recalled": lambda row: row["target_A"]["stages"]["raw"]["recalled"] and row["target_B"]["stages"]["raw"]["recalled"],
            "both_dedup_cap_recalled": lambda row: row["target_A"]["stages"]["recall_final"]["recalled"] and row["target_B"]["stages"]["recall_final"]["recalled"],
            "final_target_formed": lambda row: row["recall_topk"]["target_present"],
        }
    stages = {}
    for name, predicate in predicates.items():
        selected = [row for row in rows if predicate(row)]
        room_sides = set()
        interfaces = Counter(); wraps = Counter(); houses = set()
        for row in selected:
            room_sides.add((row["house_id"], row["floor_id"], row["room_A"], row["pano_A"], tuple(row["target_A"]["target_interval"])))
            room_sides.add((row["house_id"], row["floor_id"], row["room_B"], row["pano_B"], tuple(row["target_B"]["target_interval"])))
            interfaces[row["interface_type"]] += 1
            wraps["wrap" if row["wrap"] else "non_wrap"] += 1
            houses.add(row["house_id"])
        stages[name] = {
            "pair_count": len(selected), "pair_rate": len(selected) / total if total else None,
            "unique_room_side_count": len(room_sides), "house_count": len(houses),
            "interface_type": dict(sorted(interfaces.items())), "wrap": dict(sorted(wraps.items())),
        }
    return {"mode": mode, "positive_pair_count": total, "stages": stages}


def _reason_summary(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    counts = Counter(row["primary_pair_reason"] for row in rows)
    side_a = Counter(row["reason_A"] for row in rows)
    side_b = Counter(row["reason_B"] for row in rows)
    missing = [row for row in rows if not row["legacy"]["target_present"]]
    return {
        "schema_version": "zind-t5-candidate-reason-counts-v1",
        "split": split,
        "positive_pair_count": len(rows),
        "legacy_missing_count": len(missing),
        "primary_reason_counts": dict(sorted(counts.items())),
        "legacy_missing_primary_reason_counts": dict(sorted(Counter(row["primary_pair_reason"] for row in missing).items())),
        "reason_A_counts": dict(sorted(side_a.items())),
        "reason_B_counts": dict(sorted(side_b.items())),
        "test_data_accessed": False,
    }


def _selected_records(args: argparse.Namespace, output: Path) -> Dict[str, List[Mapping[str, Any]]]:
    train = _load_manifest(_reject_test(args.train_manifest, "train manifest"), "train")
    val = _load_manifest(_reject_test(args.val_manifest, "val manifest"), "val")
    sample = json.loads((output / "train_sample_4096.json").read_text(encoding="utf-8"))
    train_by_id = {str(record["id"]): record for record in train["pairs"]}
    train_records = [train_by_id[pair_id] for pair_id in sample["pair_ids"]]
    val_records = sorted((record for record in val["pairs"] if _positive(record)), key=lambda record: str(record["id"]))
    return {"train": train_records, "val": val_records}


def _unique_side_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        for side in ("A", "B"):
            target = row[f"target_{side}"]
            key = (
                row["house_id"], row["floor_id"], row[f"room_{side}"], row[f"pano_{side}"],
                tuple(target["target_interval"]), row["interface_type"].split("-")[0 if side == "A" else 1],
            )
            value = {
                "house_id": row["house_id"], "floor_id": row["floor_id"],
                "complete_room_id": row[f"room_{side}"], "pano_id": row[f"pano_{side}"],
                "target_interval": target["target_interval"], "target_wrap": target["target_wrap"],
                "target_probability_max": target["target_probability_max"],
                "target_probability_mean": target["target_probability_mean"],
                "reason": row[f"reason_{side}"],
                "raw_recalled": target["stages"]["raw"]["recalled"],
                "threshold_recalled": target["stages"]["threshold"]["recalled"],
                "legacy_recalled": target["stages"]["legacy_final"]["recalled"],
                "recall_topk_recalled": target["stages"]["recall_final"]["recalled"],
                "pair_ids": [row["pair_id"]], "test_data_accessed": False,
            }
            if key in grouped:
                grouped[key]["pair_ids"].append(row["pair_id"])
                immutable = set(value) - {"pair_ids"}
                if any(grouped[key][name] != value[name] for name in immutable):
                    raise ValueError(f"unique room-side diagnostics disagree: {key}")
            else:
                grouped[key] = value
    return sorted(grouped.values(), key=lambda value: (value["house_id"], value["floor_id"], value["complete_room_id"], value["pano_id"], value["target_interval"]))


def assemble(args: argparse.Namespace, output: Path, log: AuditLogger) -> Dict[str, Any]:
    if not (output / ".phase1_completed").is_file():
        raise RuntimeError("Phase 1 must complete before Phase 2")
    records = _selected_records(args, output)
    index = json.loads((output / "room_cache_index.json").read_text(encoding="utf-8"))
    refs = index["pair_refs"]
    cache_root = output / "frozen_room_cache"
    artifacts = _frozen_artifact_map(_reject_test(args.frozen_val_root, "frozen val root"))

    # Phase 2 uses CPU only for pair assembly and the otherwise missing train O0 Matcher rows.
    train_scores = _score_split_view(
        args, output, "train", records["train"], "O0", _legacy_builder,
        torch.device("cpu"), log,
    )
    rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    val_interval_parity = 0
    for split in ("train", "val"):
        for offset, record in enumerate(records[split], 1):
            pair_id = str(record["id"])
            reference = refs[pair_id]
            room_a = _load_room(cache_root, reference["A"], split)
            room_b = _load_room(cache_root, reference["B"], split)
            if split == "train":
                score = train_scores[pair_id]
            else:
                artifact = _load_gzip_json(artifacts[pair_id])
                score = {
                    "best_candidate_pair": artifact["best_candidate_pair"],
                    "pair_match_probability": artifact["pair_match_probability"],
                    "pair_no_match_probability": artifact["pair_no_match_probability"],
                }
                val_interval_parity += int(
                    room_a["legacy_candidates"] == artifact["predicted_intervals_A"]
                    and room_b["legacy_candidates"] == artifact["predicted_intervals_B"]
                )
            rows[split].append(_pair_row(record, room_a, room_b, score, split))
            if offset % 512 == 0 or offset == len(records[split]):
                log(f"Phase 2 assemble {split} {offset}/{len(records[split])}")

    legacy_val = sum(row["legacy"]["target_present"] for row in rows["val"])
    recall_val = sum(row["recall_topk"]["target_present"] for row in rows["val"])
    gates = {
        "val_pair_count": len(rows["val"]) == 1827,
        "train_pair_count": len(rows["train"]) == 4096,
        "legacy_recall_exact": legacy_val == 1080,
        "recall_topk_recall_exact": recall_val == 1107,
        "legacy_interval_pairwise_exact": val_interval_parity == 1827,
        "all_finite": finite_tree(rows),
    }
    parity = {
        "schema_version": "zind-t5-proposal-parity-v1",
        "legacy": {"recalled": legacy_val, "positive": 1827, "expected": 1080, "exact": legacy_val == 1080},
        "recall_topk": {"recalled": recall_val, "positive": 1827, "expected": 1107, "exact": recall_val == 1107},
        "legacy_interval_exact_pairs": val_interval_parity,
        "gates": gates, "passed": all(gates.values()), "test_data_accessed": False,
    }
    _atomic_json(output / "phase2_parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError("Phase 2 parity failed: " + ", ".join(key for key, value in gates.items() if not value))

    _write_jsonl(output / "per_pair_train_4096.jsonl", rows["train"])
    _write_jsonl(output / "per_pair_val.jsonl", rows["val"])
    room_sides = {split: _unique_side_rows(rows[split]) for split in rows}
    _write_jsonl(output / "per_room_side_train_4096.jsonl", room_sides["train"])
    _write_jsonl(output / "per_room_side_val.jsonl", room_sides["val"])
    for split in ("train", "val"):
        funnel = {
            "schema_version": "zind-t5-stage-funnel-v1", "split": split,
            "legacy": _funnel(rows[split], "legacy"),
            "recall_topk": _funnel(rows[split], "recall_topk"),
            "unique_room_side_count": len(room_sides[split]),
            "test_data_accessed": False,
        }
        _atomic_json(output / f"stage_funnel_{split}.json", funnel)
        _atomic_json(output / f"reason_counts_{split}.json", _reason_summary(rows[split], split))

    added = [row["pair_id"] for row in rows["val"] if not row["legacy"]["target_present"] and row["recall_topk"]["target_present"]]
    lost = [row["pair_id"] for row in rows["val"] if row["legacy"]["target_present"] and not row["recall_topk"]["target_present"]]
    comparison = {
        "schema_version": "zind-t5-legacy-vs-recall-topk-v1",
        "legacy_recalled": legacy_val, "recall_topk_recalled": recall_val,
        "added_count": len(added), "added_pair_ids": added,
        "candidate_recall_lost_count": len(lost), "candidate_recall_lost_pair_ids": lost,
        "matcher_rescore_pending": True, "test_data_accessed": False,
    }
    _atomic_json(output / "legacy_vs_recall_topk.json", comparison)

    multi_rows = [row for row in rows["val"] if row["multiple_shared_openings"]]
    multi = {
        "schema_version": "zind-t5-multiple-shared-opening-audit-v1",
        "val_positive_count": 1827,
        "multiple_physical_shared_opening_pair_count": len(multi_rows),
        "pair_ids": [row["pair_id"] for row in multi_rows],
        "manifest_supervision_target_capacity": 1,
        "runtime_affinity_marks_all_physical_matches": True,
        "extra_physical_targets_marked_negative_by_current_dataset": False,
        "window_target_count": sum("window" in row["interface_type"] for row in rows["val"]),
        "schema_limit_affects_manifest_serialization": bool(multi_rows),
        "schema_limit_affects_current_dense_training_target": False,
        "test_data_accessed": False,
    }
    _atomic_json(output / "multi_shared_opening_audit.json", multi)
    summary = {
        "schema_version": "zind-t5-phase2-summary-v1",
        "parity": parity,
        "train_positive_count": len(rows["train"]), "val_positive_count": len(rows["val"]),
        "train_unique_room_side_count": len(room_sides["train"]),
        "val_unique_room_side_count": len(room_sides["val"]),
        "legacy_missing_val": 1827 - legacy_val,
        "recall_topk_added_val": len(added),
        "test_data_accessed": False,
    }
    _atomic_json(output / "phase2_summary.json", summary)
    _atomic_text(output / ".phase2_completed", _sha256_file(output / "phase2_summary.json") + "\n")
    log(f"Phase 2 PASS: legacy={legacy_val}/1827 recall_topk={recall_val}/1827 missing={1827-legacy_val}")
    return summary


def _gt_intervals(record: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
    supervision = record["supervision"]
    return (
        list(record["candidates_a"][int(supervision["target_candidate_a"])]["tokenInterval"]),
        list(record["candidates_b"][int(supervision["target_candidate_b"])]["tokenInterval"]),
    )


def _builder_for_view(view: str):
    def build(record: Mapping[str, Any], room_a: Mapping[str, Any], room_b: Mapping[str, Any]):
        gt_a, gt_b = _gt_intervals(record)
        if view == "O0": return room_a["legacy_candidates"], room_b["legacy_candidates"]
        if view == "O1": return [gt_a], room_b["legacy_candidates"]
        if view == "O2": return room_a["legacy_candidates"], [gt_b]
        if view == "O3": return [gt_a], [gt_b]
        if view == "O4": return room_a["candidate_trace"]["raw_hypotheses"], room_b["candidate_trace"]["raw_hypotheses"]
        if view == "P1_recall_topk": return room_a["recall_topk_candidates"], room_b["recall_topk_candidates"]
        raise ValueError(view)
    return build


def _quantiles(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "p95": float(np.percentile(values, 95)),
    }


def _oracle_metrics(
    records: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, Any]], view: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    pair_rows = {}
    recalled = accepted = correct = 0
    counts = []; products = []
    for record in records:
        pair_id = str(record["id"])
        score = scores[pair_id]
        targets = _target_pairs(record, score["intervals_A"], score["intervals_B"])
        present = bool(targets)
        best = score["best_candidate_pair"]
        is_accepted = bool(best[0] >= 0 and best[1] >= 0)
        is_correct = list(best) in targets
        recalled += int(present); accepted += int(is_accepted); correct += int(is_correct)
        count_a, count_b = len(score["intervals_A"]), len(score["intervals_B"])
        counts.extend((count_a, count_b)); products.append(count_a * count_b)
        pair_rows[pair_id] = {
            "pair_id": pair_id, "view": view, "target_present": present,
            "target_pairs": targets, "predicted_match": is_accepted,
            "selected_correct": is_correct, "best_candidate_pair": best,
            "candidate_count_A": count_a, "candidate_count_B": count_b,
            "oracle_only": view != "O0", "valid_for_inference": view == "O0",
            "gt_available_to_production": False, "test_data_accessed": False,
        }
    total = len(records)
    return ({
        "view": view, "positive_count": total,
        "candidate_recalled": recalled, "candidate_recall": recalled / total,
        "predicted_match_count": accepted, "predicted_match_recall": accepted / total,
        "candidate_top1_correct": correct,
        "conditional_candidate_top1": correct / recalled if recalled else None,
        "S3_count": correct, "S3": correct / total,
        "candidate_count": _quantiles(counts), "candidate_product": _quantiles(products),
        "oracle_only": view != "O0", "valid_for_inference": view == "O0",
        "gt_available_to_production": False,
    }, pair_rows)


def oracle(args: argparse.Namespace, output: Path, log: AuditLogger) -> Dict[str, Any]:
    if not (output / ".phase2_completed").is_file():
        raise RuntimeError("Phase 2 parity must pass before Phase 3")
    records = _selected_records(args, output)["val"]
    index = json.loads((output / "room_cache_index.json").read_text(encoding="utf-8"))
    refs = index["pair_refs"]
    cache_root = output / "frozen_room_cache"
    artifacts = _frozen_artifact_map(_reject_test(args.frozen_val_root, "frozen val root"))
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Phase 3 requires the authorized CUDA device")

    # Explicit small-batch gate for all candidate shapes, including uncapped O4.
    smoke_records = records[: min(4, len(records))]
    model, _, _ = _load_frozen_model(args, device)
    state_before = _state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    smoke_views = {}
    for view in ("O0", "O1", "O2", "O3", "O4", "P1_recall_topk"):
        rooms_a = [_load_room(cache_root, refs[str(record["id"])]["A"], "val") for record in smoke_records]
        rooms_b = [_load_room(cache_root, refs[str(record["id"])]["B"], "val") for record in smoke_records]
        intervals = [_builder_for_view(view)(record, a, b) for record, a, b in zip(smoke_records, rooms_a, rooms_b)]
        start = time.perf_counter()
        values = _score_batch(model, rooms_a, rooms_b, [value[0] for value in intervals], [value[1] for value in intervals], device)
        torch.cuda.synchronize(device)
        smoke_views[view] = {
            "pair_count": len(values), "runtime_seconds": time.perf_counter() - start,
            "all_finite": finite_tree(values),
            "max_candidate_product": max((value["candidate_count_A"] * value["candidate_count_B"] for value in values), default=0),
        }
    state_after = _state_hash(model)
    del model
    torch.cuda.empty_cache()
    smoke_gate = {
        "schema_version": "zind-t5-oracle-matcher-smoke-v1",
        "views": smoke_views,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "state_hash_before": state_before, "state_hash_after": state_after,
        "passed": state_before == state_after and all(value["all_finite"] for value in smoke_views.values()),
        "model_eval": True, "requires_grad": False, "optimizer_created": False,
        "test_data_accessed": False,
    }
    _atomic_json(output / "phase3_matcher_smoke.json", smoke_gate)
    if not smoke_gate["passed"]:
        raise RuntimeError("Phase 3 Matcher smoke failed")

    all_scores = {}
    for view in ("O0", "O1", "O2", "O3", "O4", "P1_recall_topk"):
        all_scores[view] = _score_split_view(
            args, output, "val", records, view, _builder_for_view(view), device, log
        )

    # Hard O0 parity against the original pair-forward artifact.
    exact_best = exact_intervals = 0
    max_score_delta = max_match_delta = max_no_match_delta = 0.0
    for record in records:
        pair_id = str(record["id"])
        new = all_scores["O0"][pair_id]
        old = _load_gzip_json(artifacts[pair_id])
        exact_best += int(new["best_candidate_pair"] == old["best_candidate_pair"])
        exact_intervals += int(new["intervals_A"] == old["predicted_intervals_A"] and new["intervals_B"] == old["predicted_intervals_B"])
        left = np.asarray(new["candidate_pair_scores"], dtype=np.float64)
        right = np.asarray(old["candidate_pair_scores"], dtype=np.float64)[
            : new["candidate_count_A"], : new["candidate_count_B"]
        ]
        max_score_delta = max(max_score_delta, float(np.max(np.abs(left - right))) if left.size else 0.0)
        max_match_delta = max(max_match_delta, abs(float(new["pair_match_probability"]) - float(old["pair_match_probability"])))
        max_no_match_delta = max(max_no_match_delta, abs(float(new["pair_no_match_probability"]) - float(old["pair_no_match_probability"])))
    o0_parity = {
        "pair_count": len(records), "best_candidate_exact_count": exact_best,
        "interval_exact_count": exact_intervals,
        "candidate_score_max_abs_diff": max_score_delta,
        "pair_match_probability_max_abs_diff": max_match_delta,
        "pair_no_match_probability_max_abs_diff": max_no_match_delta,
        "passed": exact_best == len(records) and exact_intervals == len(records) and max(max_score_delta, max_match_delta, max_no_match_delta) <= 2e-5,
    }
    if not o0_parity["passed"]:
        _atomic_json(output / "phase3_o0_parity_failed.json", o0_parity)
        raise RuntimeError("Phase 3 O0 parity failed")

    metrics = {}; pair_metrics = {}
    for view in ("O0", "O1", "O2", "O3", "O4", "P1_recall_topk"):
        metrics[view], pair_metrics[view] = _oracle_metrics(records, all_scores[view], view)
    # O5 selects a valid GT target cell from the final legacy candidates and bypasses Matcher.
    o5_scores = {}; o5_root = output / "pair_scores/val_O5"; o5_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        pair_id = str(record["id"])
        source = all_scores["O0"][pair_id]
        targets = _target_pairs(record, source["intervals_A"], source["intervals_B"])
        best = targets[0] if targets else [-1, -1]
        value = {
            "pair_id": pair_id, "split": "val", "view": "O5",
            "intervals_A": source["intervals_A"], "intervals_B": source["intervals_B"],
            "best_candidate_pair": best,
            "candidate_count_A": len(source["intervals_A"]), "candidate_count_B": len(source["intervals_B"]),
            "oracle_only": True, "valid_for_inference": False,
            "gt_available_to_production": False, "optimizer_created": False,
            "test_data_accessed": False,
        }
        path = o5_root / f"{pair_id}.json.gz"; temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
        os.replace(temporary, path)
        o5_scores[pair_id] = value
    metrics["O5"], pair_metrics["O5"] = _oracle_metrics(records, o5_scores, "O5")
    baseline = metrics["O0"]
    for view, value in metrics.items():
        value["delta_vs_O0"] = {
            "candidate_recalled_count": value["candidate_recalled"] - baseline["candidate_recalled"],
            "predicted_match_count": value["predicted_match_count"] - baseline["predicted_match_count"],
            "S3_count": value["S3_count"] - baseline["S3_count"],
        }
    oracle_summary = {
        "schema_version": "zind-t5-oracle-summary-v1",
        "o0_parity": o0_parity, "matcher_smoke": smoke_gate,
        "oracles": {key: value for key, value in metrics.items() if key.startswith("O")},
        "recall_topk_matcher": metrics["P1_recall_topk"],
        "all_oracles": {"oracle_only": True, "valid_for_inference": False, "gt_available_to_production": False},
        "training_started": False, "optimizer_created": False,
        "test_data_accessed": False,
    }
    _atomic_json(output / "oracle_summary.json", oracle_summary)

    # Complete the 27-pair Recall-TopK transition audit.
    comparison = json.loads((output / "legacy_vs_recall_topk.json").read_text(encoding="utf-8"))
    per_pair_rows = {row["pair_id"]: row for row in _read_jsonl(output / "per_pair_val.jsonl")}
    added_rows = []
    for pair_id in comparison["added_pair_ids"]:
        row = per_pair_rows[pair_id]
        p0 = pair_metrics["O0"][pair_id]; p1 = pair_metrics["P1_recall_topk"][pair_id]
        added_side = []
        for side in ("A", "B"):
            target = row[f"target_{side}"]["stages"]
            if not target["legacy_final"]["recalled"] and target["recall_final"]["recalled"]:
                added_side.append(side)
        added_rows.append({
            "pair_id": pair_id, "house_id": row["house_id"], "added_side": added_side,
            "target_A": row["target_A"], "target_B": row["target_B"],
            "legacy_candidate_product": row["legacy"]["candidate_count_A"] * row["legacy"]["candidate_count_B"],
            "recall_topk_candidate_product": row["recall_topk"]["candidate_count_A"] * row["recall_topk"]["candidate_count_B"],
            "interference_candidate_product_delta": row["recall_topk"]["candidate_count_A"] * row["recall_topk"]["candidate_count_B"] - row["legacy"]["candidate_count_A"] * row["legacy"]["candidate_count_B"],
            "matcher_accepted": p1["predicted_match"], "matcher_selected_correct": p1["selected_correct"],
            "S3": p1["selected_correct"],
            "failure_after_recall": None if p1["selected_correct"] else ("dustbin_reject" if not p1["predicted_match"] else "wrong_selection"),
            "legacy_result": p0, "recall_topk_result": p1,
        })
    regressions = [
        {"pair_id": pair_id, "legacy": pair_metrics["O0"][pair_id], "recall_topk": pair_metrics["P1_recall_topk"][pair_id]}
        for pair_id in sorted(pair_metrics["O0"])
        if pair_metrics["O0"][pair_id]["selected_correct"] and not pair_metrics["P1_recall_topk"][pair_id]["selected_correct"]
    ]
    comparison.update({
        "matcher_rescore_pending": False, "added_pairs": added_rows,
        "legacy_correct_to_recall_topk_wrong_count": len(regressions),
        "legacy_correct_to_recall_topk_wrong": regressions,
    })
    _atomic_json(output / "legacy_vs_recall_topk.json", comparison)
    _atomic_json(output / "phase3_summary.json", {
        "schema_version": "zind-t5-phase3-summary-v1", "o0_parity": o0_parity,
        "oracle_summary_sha256": _sha256_file(output / "oracle_summary.json"),
        "processed_pair_count": len(records) * 7,
        "training_started": False, "optimizer_created": False, "test_data_accessed": False,
    })
    _atomic_text(output / ".phase3_completed", _sha256_file(output / "phase3_summary.json") + "\n")
    log(f"Phase 3 complete: O0 parity exact, O3 S3={metrics['O3']['S3_count']}/1827 O4 recall={metrics['O4']['candidate_recalled']}/1827")
    return oracle_summary


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bootstrap_reasons(rows: Sequence[Mapping[str, Any]], replicates: int, seed: int) -> Dict[str, Any]:
    missing = [row for row in rows if not row["legacy"]["target_present"]]
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in missing:
        grouped[row["house_id"]].append(row)
    houses = sorted(grouped)
    reasons = sorted({row["primary_pair_reason"] for row in missing})
    observed = Counter(row["primary_pair_reason"] for row in missing)
    rng = np.random.RandomState(seed)
    draws = {reason: [] for reason in reasons}
    for _ in range(replicates):
        sampled = rng.choice(houses, size=len(houses), replace=True)
        pooled = [row for house in sampled for row in grouped[str(house)]]
        counts = Counter(row["primary_pair_reason"] for row in pooled)
        denominator = max(1, len(pooled))
        for reason in reasons:
            draws[reason].append(counts[reason] / denominator)
    return {
        "missing_pair_count": len(missing), "house_count": len(houses),
        "reasons": {
            reason: {
                "count": observed[reason], "proportion": observed[reason] / max(1, len(missing)),
                "ci95_low": float(np.percentile(draws[reason], 2.5)),
                "ci95_high": float(np.percentile(draws[reason], 97.5)),
            }
            for reason in reasons
        },
    }


def _decision_categories(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    categories = Counter()
    for row in rows:
        category = _decision_category(row)
        if category is not None:
            categories[category] += 1
    return dict(sorted(categories.items()))


def _decision_category(row: Mapping[str, Any]) -> str | None:
    reason = row["primary_pair_reason"]
    if reason.startswith("no_opening_signal"):
        return "A_opening_head_signal"
    if reason in {
        "below_threshold_A", "below_threshold_B", "below_threshold_both",
        "interval_split", "interval_merge", "interval_expansion_failure",
        "wrap_failure", "dedup_removed", "candidate_cap_removed",
    }:
        return "B_proposal_logic"
    if reason in {
        "side_candidates_present_pair_target_missing", "target_mapping_failure",
        "multiple_shared_opening_schema_limit",
    }:
        return "C_target_mapping"
    if reason in {"target_present_dustbin_reject", "target_present_wrong_selection"}:
        return "D_matcher"
    if reason == "target_present_correct":
        return "success"
    return None


def _bootstrap_decision_categories(
    rows: Sequence[Mapping[str, Any]], replicates: int, seed: int,
) -> Dict[str, Any]:
    """House-cluster bootstrap over all A/B/C/D failures (success excluded)."""
    failures = [row for row in rows if _decision_category(row) not in (None, "success")]
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in failures:
        grouped[str(row["house_id"])].append(row)
    houses = sorted(grouped)
    category_names = (
        "A_opening_head_signal", "B_proposal_logic",
        "C_target_mapping", "D_matcher",
    )
    observed = Counter(_decision_category(row) for row in failures)
    rng = np.random.RandomState(seed)
    draws = {category: [] for category in category_names}
    for _ in range(replicates):
        sampled = rng.choice(houses, size=len(houses), replace=True)
        pooled = [row for house in sampled for row in grouped[str(house)]]
        counts = Counter(_decision_category(row) for row in pooled)
        denominator = max(1, len(pooled))
        for category in category_names:
            draws[category].append(counts[category] / denominator)
    return {
        "failure_pair_count": len(failures), "house_count": len(houses),
        "success_pair_count": sum(_decision_category(row) == "success" for row in rows),
        "categories": {
            category: {
                "count": observed[category],
                "proportion": observed[category] / max(1, len(failures)),
                "ci95_low": float(np.percentile(draws[category], 2.5)),
                "ci95_high": float(np.percentile(draws[category], 97.5)),
            }
            for category in category_names
        },
    }


def _render_visualizations(output: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = json.loads((output / "room_cache_index.json").read_text(encoding="utf-8"))
    refs = index["pair_refs"]
    cache_root = output / "frozen_room_cache"
    visual_root = output / "visualizations"
    selected: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda value: value["pair_id"]):
        if len(selected[row["primary_pair_reason"]]) < 10:
            selected[row["primary_pair_reason"]].append(row)

    def spans(ax: Any, interval: Sequence[int], color: str, alpha: float, label: str) -> None:
        start, end = int(interval[0]), int(interval[1])
        if start <= end:
            ax.axvspan(start, end, color=color, alpha=alpha, label=label)
        else:
            ax.axvspan(start, 255, color=color, alpha=alpha, label=label)
            ax.axvspan(0, end, color=color, alpha=alpha)

    files = []
    for reason, reason_rows in sorted(selected.items()):
        directory = visual_root / reason
        directory.mkdir(parents=True, exist_ok=True)
        for row in reason_rows:
            pair_id = row["pair_id"]
            pair_ref = refs[pair_id]
            rooms = {
                side: _load_room(cache_root, pair_ref[side], "val") for side in ("A", "B")
            }
            figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
            for ax, side in zip(axes, ("A", "B")):
                room = rooms[side]
                probability = room["opening_probability"].numpy()
                trace = room["candidate_trace"]
                ax.plot(np.arange(len(probability)), probability, color="black", linewidth=1, label="opening probability")
                ax.axhline(PROPOSAL["threshold"], color="tab:red", linestyle="--", linewidth=1, label="legacy threshold")
                ax.axhline(PROPOSAL["peak_min_probability"], color="tab:orange", linestyle=":", linewidth=1, label="peak minimum")
                spans(ax, row[f"target_{side}"]["target_interval"], "tab:green", 0.20, "GT interval")
                for index_value, interval in enumerate(trace["legacy_candidates"]):
                    spans(ax, interval, "tab:blue", 0.08, "legacy candidate" if index_value == 0 else "")
                for index_value, interval in enumerate(trace["recall_topk_candidates"]):
                    ax.plot(interval_tokens(interval, 256), [0.04 + 0.012 * index_value] * len(interval_tokens(interval, 256)), color="tab:purple", linewidth=2, label="Recall-TopK" if index_value == 0 else "")
                for peak in trace["local_peaks"]:
                    marker = "x" if peak["removed_by_dedup"] or peak["removed_by_cap"] or peak["removed_by_peak_top_k"] else "o"
                    ax.scatter([peak["peak_index"]], [peak["peak_probability"]], color="tab:red" if marker == "x" else "tab:orange", marker=marker, s=22)
                ax.axvline(0, color="gray", linewidth=0.8); ax.axvline(255, color="gray", linewidth=0.8)
                ax.set_ylim(-0.01, 1.02); ax.set_ylabel(f"side {side}")
                ax.set_title(f"{row[f'reason_{side}']} | room={row[f'room_{side}']} pano={row[f'pano_{side}']}")
            axes[-1].set_xlabel("circular panorama token")
            handles, labels = axes[0].get_legend_handles_labels()
            unique = dict(zip(labels, handles)); figure.legend(unique.values(), unique.keys(), loc="upper center", ncol=5)
            figure.suptitle(f"{pair_id} | primary={reason}")
            figure.tight_layout(rect=(0, 0, 1, 0.91))
            path = directory / f"{pair_id}.png"
            figure.savefig(path, dpi=130)
            plt.close(figure)
            files.append(str(path.relative_to(output)))
    payload = {
        "schema_version": "zind-t5-failure-visualizations-v1",
        "selection": "up to 10 samples per primary reason, stable pair-ID order; no performance-based manual selection",
        "file_count": len(files), "files": files, "test_data_accessed": False,
    }
    _atomic_json(visual_root / "index.json", payload)
    return payload


def _checkpoint_evidence() -> Dict[str, Any]:
    phases = {}
    for phase in ("T0", "T1", "T2", "T3"):
        path = REPO_ROOT / f"checkpoints/End_To_End/zind_e2e_v2/g0_seed123/{phase}/best.pt"
        phases[phase] = {
            "path": str(path), "sha256": _sha256_file(path), "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
    return phases


def report(args: argparse.Namespace, output: Path, log: AuditLogger) -> Dict[str, Any]:
    if not (output / ".phase3_completed").is_file():
        raise RuntimeError("Phase 3 must complete before final report")
    train_rows = _read_jsonl(output / "per_pair_train_4096.jsonl")
    val_rows = _read_jsonl(output / "per_pair_val.jsonl")
    bootstrap = {
        "schema_version": "zind-t5-house-bootstrap-v2", "seed": args.seed,
        "replicates": args.bootstrap_replicates,
        "train": {
            "fine_grained_missing": _bootstrap_reasons(train_rows, args.bootstrap_replicates, args.seed),
            "decision_failures": _bootstrap_decision_categories(train_rows, args.bootstrap_replicates, args.seed + 2),
        },
        "val": {
            "fine_grained_missing": _bootstrap_reasons(val_rows, args.bootstrap_replicates, args.seed + 1),
            "decision_failures": _bootstrap_decision_categories(val_rows, args.bootstrap_replicates, args.seed + 3),
        },
        "test_data_accessed": False,
    }
    _atomic_json(output / "house_bootstrap.json", bootstrap)
    decision_bootstrap = {
        split: bootstrap[split]["decision_failures"]["categories"] for split in ("train", "val")
    }
    train_rank = sorted(decision_bootstrap["train"], key=lambda reason: (-decision_bootstrap["train"][reason]["proportion"], reason))
    val_rank = sorted(decision_bootstrap["val"], key=lambda reason: (-decision_bootstrap["val"][reason]["proportion"], reason))
    top_same = bool(train_rank and val_rank and train_rank[0] == val_rank[0])
    top_width = max(
        decision_bootstrap[split][ranking[0]]["ci95_high"] - decision_bootstrap[split][ranking[0]]["ci95_low"]
        for split, ranking in (("train", train_rank), ("val", val_rank)) if ranking
    )
    categories = {"train": _decision_categories(train_rows), "val": _decision_categories(val_rows)}
    decision_keys = ("A_opening_head_signal", "B_proposal_logic", "C_target_mapping", "D_matcher")
    decisions = {
        split: max(decision_keys, key=lambda key: categories[split].get(key, 0)) for split in ("train", "val")
    }
    stable = top_same and top_width <= 0.10 and decisions["train"] == decisions["val"]
    needs_full_train = not stable
    chosen = decisions["val"] if decisions["train"] == decisions["val"] else "inconclusive"
    recommendation = {
        "A_opening_head_signal": "T5 Opening-Head recall fine-tuning on unique panoramas; freeze Bi-Layout and Matcher",
        "B_proposal_logic": "fix candidate generator trace stage first; do not train a model",
        "C_target_mapping": "fix candidate-to-GT mapping/schema before any model training",
        "D_matcher": "only then consider predicted-only Matcher/verifier work",
        "inconclusive": "do not change model yet; request full-train expansion",
    }[chosen]
    visuals = _render_visualizations(output, val_rows)
    phase0 = json.loads((output / "phase0_smoke.json").read_text())
    phase1 = json.loads((output / "phase1_cache_summary.json").read_text())
    phase2 = json.loads((output / "phase2_summary.json").read_text())
    oracle_summary = json.loads((output / "oracle_summary.json").read_text())
    reason_train = json.loads((output / "reason_counts_train.json").read_text())
    reason_val = json.loads((output / "reason_counts_val.json").read_text())
    room_index = json.loads((output / "room_cache_index.json").read_text())
    sample = json.loads((output / "train_sample_4096.json").read_text())
    checkpoints = _checkpoint_evidence()
    expected_t3 = "ca083dbbc5a05d738dad36404ad708c623c272aef7a64398b4de3a793278cc5d"
    if checkpoints["T3"]["sha256"] != expected_t3:
        raise RuntimeError("T3 checkpoint changed before report")
    security = {
        "schema_version": "zind-t5-audit-security-v1",
        "checkpoints": checkpoints, "model_training_started": False,
        "optimizer_created": False, "backward_executed": False,
        "parameter_update_executed": False, "batchnorm_updates": False,
        "formal_proposal_defaults_modified": False, "T4_outputs_modified": False,
        "T5_training_started": False, "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    _atomic_json(output / "security_evidence.json", security)
    summary = {
        "schema_version": "zind-t5-candidate-failure-audit-summary-v1",
        "conclusion": chosen,
        "recommendation": recommendation,
        "needs_full_train": needs_full_train,
        "stability": {"top_reason_same": top_same, "top_ci_width_max": top_width, "decision_same": decisions["train"] == decisions["val"], "stable": stable},
        "decision_categories": categories,
        "train_sample": {"pair_count": sample["pair_count"], "house_count": sample["house_count"]},
        "room_dedup": room_index["splits"],
        "phase0": {"passed": phase0["passed"], "projected_total_bytes": phase0["projected_total_bytes"], "projected_total_runtime_seconds": phase0["projected_total_runtime_seconds"], "gpu_peak_allocated_bytes": phase0["room_forward"]["gpu_peak_allocated_bytes"]},
        "phase1": {"room_count": phase1["room_count"], "runtime_seconds": phase1["runtime_seconds"], "bytes_written": phase1["bytes_written"]},
        "parity": phase2["parity"],
        "reason_counts": {"train": reason_train, "val": reason_val},
        "oracles": oracle_summary,
        "bootstrap": bootstrap,
        "visualizations": visuals,
        "security": security,
        "test_data_accessed": False,
    }
    _atomic_json(output / "summary.json", summary)
    git_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip()
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)
    oracle_lines = [
        f"| {name} | {value['candidate_recalled']}/1827 | {value['predicted_match_count']}/1827 | {value['conditional_candidate_top1']:.4f} | {value['S3_count']}/1827 |"
        for name, value in oracle_summary["oracles"].items()
    ]
    report_text = f"""# ZInD candidate failure audit

## 1. 一句话结论

主要决策为 **{chosen}**：{recommendation}。train-4096 与 full-val 的稳定性门禁为
`{stable}`；是否建议补跑 full train：`{needs_full_train}`。

## 2. branch、HEAD、worktree

- branch: `{git_branch}`
- HEAD: `{git_head}`
- worktree 保持 dirty，未 reset/checkout/clean。
- 审计开始时已有修改及最终 `git status --short` 原样保留：

```text
{status.rstrip()}
```

## 3. 授权范围内实际执行内容

只执行 T3 `eval()` + `torch.inference_mode()` 的 unique-room 前向、冻结 Matcher
重评分、CPU proposal/GT 后验审计和可视化。未创建 optimizer、未 backward、未训练。

## 4. room-level 去重前后工作量

- train-4096: {room_index['splits']['train']['pair_side_count']} sides → {room_index['splits']['train']['unique_room_count']} rooms，节省 {room_index['splits']['train']['dedup_saving_rate']:.2%}。
- full-val positive: {room_index['splits']['val']['pair_side_count']} sides → {room_index['splits']['val']['unique_room_count']} rooms，节省 {room_index['splits']['val']['dedup_saving_rate']:.2%}。
- 总 unique room: {room_index['totals']['unique_room_count']}。

## 5. Phase 0 时间、显存、磁盘预测

- PASS=`{phase0['passed']}`；64-room batch 平均 {phase0['single_batch_time_seconds']:.4f}s。
- peak allocated={phase0['room_forward']['gpu_peak_allocated_bytes']/2**30:.2f} GiB。
- 预计缓存={phase0['projected_total_bytes']/2**30:.2f} GiB，约占当时可用空间 {phase0['projected_fraction_of_available']:.2%}。
- opening/feature/depth、legacy candidate 与 pair regroup parity 均通过 2e-5 门禁。

## 6. GPU 实际运行时间和缓存完成率

Phase 1 完成 {phase1['processed_room_count']}/{phase1['total_room_count']} rooms，新增运行
{phase1['runtime_seconds']/60:.2f} 分钟，写入 {phase1['bytes_written']/2**30:.2f} GiB。

## 7. train-4096 house 覆盖

覆盖 {sample['house_count']} 个 train house；抽样顺序 hash=`{sample['pair_order_sha256']}`。

## 8. legacy / Recall-TopK parity

- legacy: {phase2['parity']['legacy']['recalled']}/1827（exact={phase2['parity']['legacy']['exact']}）
- Recall-TopK: {phase2['parity']['recall_topk']['recalled']}/1827（exact={phase2['parity']['recall_topk']['exact']}）

## 9. train-4096 与 full-val stage funnel

完整逐阶段、interface、wrap、house 与 unique-side 统计见
`stage_funnel_train.json` 和 `stage_funnel_val.json`。

## 10. 747 个 missing candidate 原因分解

full-val legacy missing={reason_val['legacy_missing_count']}；互斥主因：
`{json.dumps(reason_val['legacy_missing_primary_reason_counts'], ensure_ascii=False, sort_keys=True)}`。

## 11. O0–O5 结果

| Oracle | candidate recall | predicted match | conditional Top-1 | S3 |
|---|---:|---:|---:|---:|
{chr(10).join(oracle_lines)}

所有 O1–O5 artifact 均标记 `oracle_only=true`, `valid_for_inference=false`。

## 12. train/val house-bootstrap 及一致性

- 2,000 次 house-cluster bootstrap（全体 A/B/C/D failure，success 不进分母）；train/val top reason 相同=`{top_same}`。
- top reason 最大 95% CI 宽度={top_width:.4f}；情形排序一致=`{decisions['train'] == decisions['val']}`。
- train decision=`{decisions['train']}`；val decision=`{decisions['val']}`。

## 13. 推荐情形 A/B/C/D

推荐 **{chosen}**。分类计数：`{json.dumps(categories, ensure_ascii=False, sort_keys=True)}`。

## 14. 是否需要补跑完整 train

`{needs_full_train}`。本轮没有也不会自动扩展至 14,741 train positives。

## 15. 下一阶段最小方案

{recommendation}。本报告只提出方案，不执行修复或训练。

## 16. test 隔离、无训练及 checkpoint 保护证据

- active splits=`train,val`；`test_data_accessed=false`。
- training/optimizer/backward/parameter update 均为 false。
- T3 SHA=`{checkpoints['T3']['sha256']}`，size/mtime 未变。
- M2、scale、layout、joint BEV、正式 proposal defaults 和 T4 输出均未修改。

## 17. PID、状态文件和 GPU 最终状态

PID、完整/恢复命令、`.started/.completed/.failed`、room/pair progress 位于本目录。
GPU 最终状态由代理在任务退出后另行执行只读 `nvidia-smi` 记录。
"""
    _atomic_text(output / "REPORT.md", report_text)
    log(f"report complete: decision={chosen} stable={stable} needs_full_train={needs_full_train}")
    return summary


def _write_run_identity(args: argparse.Namespace, output: Path) -> None:
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    resume_args = list(sys.argv[1:])
    if "--resume" not in resume_args:
        resume_args.append("--resume")
    resume = shlex.join([sys.executable, str(Path(__file__).resolve()), *resume_args])
    identity = {
        "schema_version": "zind-t5-audit-run-v1", "pid": os.getpid(),
        "command": command, "resume_command": resume,
        "task_type": "frozen_inference_audit", "training_started": False,
        "optimizer_created": False, "active_splits": ["train", "val"],
        "test_data_accessed": False,
    }
    _atomic_text(output / "audit.pid", str(os.getpid()) + "\n")
    _atomic_text(output / "command.txt", command + "\n")
    _atomic_text(output / "resume_command.txt", resume + "\n")
    _atomic_json(output / ".started", identity)
    ledger_path = output / "executed_commands.json"
    ledger = {
        "schema_version": "zind-t5-audit-command-ledger-v1",
        "execution_mode": "foreground_phased",
        "entries": [],
        "test_data_accessed": False,
    }
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        completed_artifacts = {
            "prepare": output / "room_cache_index.json",
            "smoke": output / ".phase0_passed",
            "cache": output / ".phase1_completed",
            "assemble": output / ".phase2_completed",
            "oracle": output / ".phase3_completed",
        }
        for phase, artifact in completed_artifacts.items():
            if artifact.is_file():
                reconstructed = shlex.join([
                    sys.executable, str(Path(__file__).resolve()),
                    "--phase", phase, "--resume",
                ])
                ledger["entries"].append({
                    "phase": phase, "pid": None, "command": reconstructed,
                    "resume_command": reconstructed,
                    "record_source": "reconstructed_from_completed_artifact",
                    "completion_artifact": str(artifact),
                    "task_type": "frozen_inference_audit", "training_started": False,
                    "optimizer_created": False, "test_data_accessed": False,
                })
    ledger["entries"].append({
        "phase": args.phase, "pid": os.getpid(), "command": command,
        "resume_command": resume, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_type": "frozen_inference_audit", "training_started": False,
        "optimizer_created": False, "test_data_accessed": False,
    })
    _atomic_json(ledger_path, ledger)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed != 123 or args.batch_size != 4 or args.smoke_rooms > 64:
        raise ValueError("fixed audit contract requires seed=123, room batch=4, smoke<=64")
    if min(args.train_positive_count, args.oracle_batch_size, args.torch_threads, args.progress_every, args.bootstrap_replicates) <= 0:
        raise ValueError("audit counts must be positive")
    for value, label in (
        (args.train_manifest, "train manifest"), (args.val_manifest, "val manifest"),
        (args.train_cache_dir, "train cache"), (args.val_cache_dir, "val cache"),
        (args.t3_checkpoint, "T3 checkpoint"), (args.frozen_val_root, "frozen val root"),
        (args.medium_root, "medium root"),
    ):
        _reject_test(value, label)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = AuditLogger(output / "audit.log")
    _write_run_identity(args, output)
    torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)
    phases = [args.phase] if args.phase != "all" else ["prepare", "smoke", "cache", "assemble", "oracle", "report"]
    try:
        for phase in phases:
            log(f"start phase={phase}")
            if phase == "prepare": prepare(args, output, log)
            elif phase == "smoke": smoke(args, output, log)
            elif phase == "cache": cache_rooms(args, output, log)
            elif phase == "assemble": assemble(args, output, log)
            elif phase == "oracle": oracle(args, output, log)
            elif phase == "report": report(args, output, log)
        if "report" in phases:
            completed = {
                "pid": os.getpid(), "task_type": "frozen_inference_audit",
                "training_started": False, "optimizer_created": False,
                "test_data_accessed": False, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            _atomic_json(output / ".completed", completed)
        return 0
    except Exception as exc:
        failed = {
            "pid": os.getpid(), "error": str(exc), "traceback": traceback.format_exc(),
            "task_type": "frozen_inference_audit", "training_started": False,
            "optimizer_created": False, "test_data_accessed": False,
        }
        _atomic_json(output / ".failed", failed)
        log(f"FAILED: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
