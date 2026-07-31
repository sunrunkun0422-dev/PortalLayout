#!/usr/bin/env python3
"""Build frozen Bi-Layout feature caches for ZInD_matching panoramas.

The cache is deliberately single-view keyed: every unique panorama in one
matching split is evaluated once, even when it appears in many pair records.
Existing compatible Opening-Head caches are copied before inference so the
large frozen backbone is never run twice for the same image.

Only model predictions are stored.  Candidate/opening annotations from the
matching manifest are not read as depth inputs and no ground-truth depth array
is part of this cache contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cross_scene_matcher import resolve_enclosed_extended_depth
from tools.debug_cross_scene_flow import load_bi_layout


DEFAULT_MANIFEST_ROOT = REPO_ROOT / "src/dataset/ZInD_matching"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_feature_cache"
)
DEFAULT_CONFIG = REPO_ROOT / "src/config/zind_all.yaml"
DEFAULT_BACKBONE_CHECKPOINT = (
    REPO_ROOT / "checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl"
)
DEFAULT_REUSE_CACHES = (
    REPO_ROOT / "checkpoints/Opening_Head/zind_bipair_v1/cache/train",
    REPO_ROOT / "checkpoints/Opening_Head/zind_bipair_v1/cache/val",
)


@dataclass(frozen=True)
class MatchingView:
    index: int
    image_path: str
    relative_image_path: str
    pair_id: str
    side: str
    scene_id: str
    floor_id: str
    opening_target: Tuple[int, ...]


@dataclass
class ReuseCache:
    path: Path
    metadata: Mapping[str, Any]
    features: np.ndarray
    enclosed_depth: np.ndarray
    extended_depth: np.ndarray
    path_to_index: Mapping[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_root", default=str(DEFAULT_MANIFEST_ROOT))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument("--data_root")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--reuse_cache",
        action="append",
        default=None,
        help=(
            "compatible single-view cache to reuse; repeat for multiple paths. "
            "Defaults to the trained Opening Head train/val caches"
        ),
    )
    parser.add_argument("--bi_layout_config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--bi_layout_checkpoint", default=str(DEFAULT_BACKBONE_CHECKPOINT)
    )
    parser.add_argument("--branch_order", choices=("extended_first", "enclosed_first"), default="extended_first")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--progress_every", type=int, default=250)
    parser.add_argument(
        "--max_images_per_split",
        type=int,
        help="stable per-split limit for smoke tests",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--audit_only",
        action="store_true",
        help="report unique-image/reuse coverage without loading Bi-Layout",
    )
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.batch_size,
        args.torch_threads,
        args.image_height,
        args.image_width,
        args.progress_every,
    )
    if min(positive) <= 0:
        raise ValueError("batch/thread/image/progress arguments must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.max_images_per_split is not None and args.max_images_per_split <= 0:
        raise ValueError("--max_images_per_split must be positive")


def _resolve_device(requested: str) -> torch.device:
    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    return device


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, hash_contents: bool = True) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat()
    identity: Dict[str, Any] = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_contents:
        identity["sha256"] = _sha256(path)
    return identity


def _canonical_path(value: str, base_dir: Optional[Path] = None) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        if base_dir is None:
            raise ValueError("relative image path requires a data root: {}".format(value))
        path = base_dir / path
    return str(path.resolve())


def _manifest_data_root(
    payload: Mapping[str, Any], manifest_path: Path, explicit_root: Optional[str]
) -> Path:
    value = explicit_root
    if value is None:
        value = payload.get("dataRoot", payload.get("data_root"))
    if not value:
        raise ValueError("manifest has no dataRoot and --data_root was not provided")
    root = Path(str(value)).expanduser()
    if not root.is_absolute():
        root = manifest_path.parent / root
    return root.resolve()


def collect_unique_views(
    manifest_path: Path,
    data_root: Optional[str] = None,
    max_images: Optional[int] = None,
    validate_paths: bool = True,
) -> Tuple[List[MatchingView], int, Path]:
    """Return stable first-occurrence unique panoramas from a pair manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping):
        raise ValueError("matching manifest root must be an object")
    records = payload.get("pairs")
    if not isinstance(records, list) or not records:
        raise ValueError("matching manifest must contain a non-empty pairs list")
    root = _manifest_data_root(payload, manifest_path, data_root)
    token_count = int(payload.get("tokenCount", payload.get("token_count", 256)))
    if token_count <= 0:
        raise ValueError("manifest tokenCount must be positive")

    views: List[MatchingView] = []
    seen = set()
    candidate_signatures: Dict[str, Tuple[Tuple[str, int, int], ...]] = {}
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("pair record {} must be an object".format(record_index))
        declared_tokens = int(
            record.get("token_count", record.get("tokenCount", token_count))
        )
        if declared_tokens != token_count:
            raise ValueError("all pair records must use manifest tokenCount")
        pair_id = str(record.get("id", record.get("pair_id", "pair_{}".format(record_index))))
        for side, key in (("A", "image_a"), ("B", "image_b")):
            raw_path = record.get(key, record.get("image{}".format(side)))
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("pair {} is missing {}".format(pair_id, key))
            absolute_path = _canonical_path(raw_path, root)
            candidates_key = "candidates_{}".format(side.lower())
            raw_candidates = record.get(
                candidates_key, record.get("candidates{}".format(side))
            )
            if not isinstance(raw_candidates, list):
                raise ValueError(
                    "pair {} {} must be a candidate list".format(
                        pair_id, candidates_key
                    )
                )
            normalized_candidates: List[Tuple[str, int, int]] = []
            opening_target = np.zeros(token_count, dtype=np.uint8)
            for candidate_index, candidate in enumerate(raw_candidates):
                if not isinstance(candidate, Mapping):
                    raise ValueError(
                        "pair {} {} candidate {} must be an object".format(
                            pair_id, side, candidate_index
                        )
                    )
                label_type = str(candidate.get("type", "")).strip().lower()
                if label_type not in ("door", "opening"):
                    continue
                interval = candidate.get(
                    "tokenInterval", candidate.get("token_interval")
                )
                if (
                    not isinstance(interval, Sequence)
                    or isinstance(interval, (str, bytes))
                    or len(interval) != 2
                ):
                    raise ValueError(
                        "pair {} {} candidate {} has no two-token interval".format(
                            pair_id, side, candidate_index
                        )
                    )
                start = int(interval[0]) % token_count
                end = int(interval[1]) % token_count
                normalized_candidates.append((label_type, start, end))
                if start <= end:
                    opening_target[start : end + 1] = 1
                else:
                    opening_target[start:] = 1
                    opening_target[: end + 1] = 1
            signature = tuple(sorted(normalized_candidates))
            previous_signature = candidate_signatures.get(absolute_path)
            if previous_signature is not None and signature != previous_signature:
                raise ValueError(
                    "inconsistent door/opening candidates for panorama {}: "
                    "pair {} side {} differs from an earlier occurrence".format(
                        absolute_path, pair_id, side
                    )
                )
            candidate_signatures[absolute_path] = signature
            if absolute_path in seen:
                continue
            seen.add(absolute_path)
            try:
                relative_path = str(Path(absolute_path).relative_to(root))
            except ValueError:
                relative_path = raw_path
            views.append(
                MatchingView(
                    index=len(views),
                    image_path=absolute_path,
                    relative_image_path=relative_path,
                    pair_id=pair_id,
                    side=side,
                    scene_id=str(record.get("scene_id", record.get("house_id", ""))),
                    floor_id=str(record.get("floor_id", "")),
                    opening_target=tuple(int(value) for value in opening_target),
                )
            )
            if max_images is not None and len(views) >= max_images:
                break
        if max_images is not None and len(views) >= max_images:
            break
    if not views:
        raise ValueError("matching manifest contains no panorama paths")
    if validate_paths:
        missing = [view.image_path for view in views if not Path(view.image_path).is_file()]
        if missing:
            raise FileNotFoundError(
                "matching manifest references missing panoramas: {}".format(
                    ", ".join(missing[:3])
                )
            )
    return views, token_count, root


def _serialize_views(views: Sequence[MatchingView]) -> List[Dict[str, Any]]:
    return [
        {
            "index": int(view.index),
            "image_path": view.image_path,
            "relative_image_path": view.relative_image_path,
            "pair_id": view.pair_id,
            "side": view.side,
            "scene_id": view.scene_id,
            "floor_id": view.floor_id,
        }
        for view in views
    ]


def _contract(
    split: str,
    views: Sequence[MatchingView],
    token_count: int,
    feature_dim: int,
    manifest_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    image_shape: Tuple[int, int],
    branch_order: str,
    max_images: Optional[int],
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "task": "ZInD_matching frozen Bi-Layout complete-room feature cache",
        "split": split,
        "manifest": _file_identity(manifest_path, hash_contents=True),
        "bi_layout_config": _file_identity(config_path, hash_contents=True),
        # Avoid hashing the 1.2 GiB checkpoint on every resume.  Resolved path,
        # exact size, and nanosecond mtime form the same strict identity used by
        # the Opening Head cache.
        "bi_layout_checkpoint": _file_identity(checkpoint_path, hash_contents=False),
        "image_shape": list(image_shape),
        "branch_order": branch_order,
        "sample_count": len(views),
        "token_count": int(token_count),
        "feature_dim": int(feature_dim),
        "feature_dtype": "float16",
        "depth_dtype": "float32",
        "target_dtype": "uint8",
        "opening_label_types": ["door", "opening"],
        "feature_source": "frozen Bi-Layout layout_feature prediction",
        "depth_source": "frozen Bi-Layout depth/new_depth predictions",
        "uses_ground_truth_depth": False,
        "selection": {"max_images_per_split": max_images},
    }


def _same_contract(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    runtime_keys = {
        "complete",
        "completed_count",
        "reused_count",
        "inferred_count",
        "runtime_seconds",
        "views",
    }
    left = {key: value for key, value in first.items() if key not in runtime_keys}
    right = {key: value for key, value in second.items() if key not in runtime_keys}
    return left == right


_ARRAY_SPECS = {
    "features": lambda count, tokens, channels: (np.float16, (count, tokens, channels)),
    "enclosed_depth": lambda count, tokens, channels: (np.float32, (count, tokens)),
    "extended_depth": lambda count, tokens, channels: (np.float32, (count, tokens)),
    "targets": lambda count, tokens, channels: (np.uint8, (count, tokens)),
    "completed": lambda count, tokens, channels: (np.uint8, (count,)),
    "origins": lambda count, tokens, channels: (np.uint8, (count,)),
}


def _create_array_atomically(path: Path, dtype: np.dtype, shape: Tuple[int, ...]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    array = np.lib.format.open_memmap(
        str(temporary), mode="w+", dtype=dtype, shape=shape
    )
    array[:] = 0
    array.flush()
    del array
    os.replace(str(temporary), str(path))


def _initialize_output(
    cache_dir: Path,
    contract: Mapping[str, Any],
    views: Sequence[MatchingView],
    rebuild: bool,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    existing: Optional[Mapping[str, Any]] = None
    if metadata_path.is_file() and not rebuild:
        with metadata_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if not isinstance(existing, Mapping) or not _same_contract(existing, contract):
            raise ValueError(
                "cache contract changed at {}; pass --rebuild".format(cache_dir)
            )

    count = int(contract["sample_count"])
    tokens = int(contract["token_count"])
    channels = int(contract["feature_dim"])
    create = existing is None or rebuild
    arrays: Dict[str, np.ndarray] = {}
    for name, spec in _ARRAY_SPECS.items():
        dtype, shape = spec(count, tokens, channels)
        path = cache_dir / "{}.npy".format(name)
        if create:
            _create_array_atomically(path, dtype, shape)
        if not path.is_file():
            raise FileNotFoundError("cache array is missing: {}".format(path))
        array = np.load(str(path), mmap_mode="r+", allow_pickle=False)
        if array.dtype != np.dtype(dtype) or tuple(array.shape) != tuple(shape):
            raise ValueError(
                "cache array {} has dtype/shape {}/{}, expected {}/{}".format(
                    path, array.dtype, tuple(array.shape), np.dtype(dtype), tuple(shape)
                )
            )
        arrays[name] = array

    if create:
        metadata: Dict[str, Any] = {
            **dict(contract),
            "complete": False,
            "completed_count": 0,
            "reused_count": 0,
            "inferred_count": 0,
            "runtime_seconds": 0.0,
            "views": _serialize_views(views),
        }
        _atomic_write_json(metadata_path, metadata)
    else:
        metadata = dict(existing or {})
        expected_views = _serialize_views(views)
        if metadata.get("views") != expected_views:
            raise ValueError("cache view order changed at {}".format(cache_dir))
    expected_targets = np.asarray(
        [view.opening_target for view in views], dtype=np.uint8
    )
    if tuple(expected_targets.shape) != (count, tokens):
        raise ValueError("matching opening targets have the wrong shape")
    if create:
        arrays["targets"][:] = expected_targets
        arrays["targets"].flush()
    elif not np.array_equal(np.asarray(arrays["targets"]), expected_targets):
        raise ValueError("cached opening targets disagree with the matching manifest")
    return metadata, arrays


def _identity_matches(current: Mapping[str, Any], cached: Mapping[str, Any]) -> bool:
    if current.get("sha256") is not None and cached.get("sha256") is not None:
        return (
            current.get("sha256") == cached.get("sha256")
            and current.get("size") == cached.get("size")
        )
    fields = ("path", "size", "mtime_ns")
    return all(current.get(field) == cached.get(field) for field in fields)


def load_reuse_cache(
    cache_dir: Path,
    config_identity: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    image_shape: Tuple[int, int],
    branch_order: str,
    token_count: int,
    feature_dim: int,
) -> ReuseCache:
    cache_dir = cache_dir.expanduser().resolve()
    metadata_path = cache_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, Mapping) or metadata.get("complete") is not True:
        raise ValueError("reuse cache is not complete: {}".format(cache_dir))
    checks = {
        "image_shape": list(image_shape),
        "branch_order": branch_order,
        "token_count": int(token_count),
        "feature_dim": int(feature_dim),
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(
                "reuse cache {} contract mismatch for {}: {} != {}".format(
                    cache_dir, key, metadata.get(key), expected
                )
            )
    if not _identity_matches(config_identity, metadata.get("bi_layout_config", {})):
        raise ValueError("reuse cache Bi-Layout config identity mismatch: {}".format(cache_dir))
    if not _identity_matches(
        checkpoint_identity, metadata.get("bi_layout_checkpoint", {})
    ):
        raise ValueError(
            "reuse cache Bi-Layout checkpoint identity mismatch: {}".format(cache_dir)
        )

    count = int(metadata.get("sample_count", 0))
    specs = {
        "features": (np.float16, (count, token_count, feature_dim)),
        "enclosed_depth": (np.float32, (count, token_count)),
        "extended_depth": (np.float32, (count, token_count)),
    }
    arrays = {}
    for name, (dtype, shape) in specs.items():
        path = cache_dir / "{}.npy".format(name)
        value = np.load(str(path), mmap_mode="r", allow_pickle=False)
        if value.dtype != np.dtype(dtype) or tuple(value.shape) != tuple(shape):
            raise ValueError("reuse cache array has wrong dtype/shape: {}".format(path))
        arrays[name] = value
    raw_views = metadata.get("views")
    if not isinstance(raw_views, list) or len(raw_views) != count:
        raise ValueError("reuse cache views do not match sample_count: {}".format(cache_dir))
    path_to_index: Dict[str, int] = {}
    for index, view in enumerate(raw_views):
        if not isinstance(view, Mapping):
            raise ValueError("reuse cache view must be an object")
        image_path = view.get("image_path")
        if not image_path:
            image_path = view.get("relative_image_path")
        if not image_path:
            raise ValueError("reuse cache view has no image path")
        base_dir = None
        if not Path(str(image_path)).expanduser().is_absolute():
            # Opening Head views always include image_path.  This fallback is
            # retained for compatible external caches that declare data_root.
            declared_root = metadata.get("data_root")
            if not declared_root:
                raise ValueError("relative reuse-cache path requires metadata data_root")
            base_dir = Path(str(declared_root)).expanduser().resolve()
        canonical = _canonical_path(str(image_path), base_dir)
        path_to_index.setdefault(canonical, index)
    return ReuseCache(
        path=cache_dir,
        metadata=metadata,
        features=arrays["features"],
        enclosed_depth=arrays["enclosed_depth"],
        extended_depth=arrays["extended_depth"],
        path_to_index=path_to_index,
    )


def _build_reuse_lookup(
    caches: Sequence[ReuseCache],
) -> Dict[str, Tuple[ReuseCache, int]]:
    lookup: Dict[str, Tuple[ReuseCache, int]] = {}
    for cache in caches:
        for path, index in cache.path_to_index.items():
            lookup.setdefault(path, (cache, index))
    return lookup


def _validate_prediction_batch(
    features: np.ndarray,
    enclosed: np.ndarray,
    extended: np.ndarray,
    expected_tokens: int,
    expected_channels: int,
) -> None:
    if tuple(features.shape[1:]) != (expected_tokens, expected_channels):
        raise ValueError("Bi-Layout layout_feature has unexpected shape {}".format(features.shape))
    if tuple(enclosed.shape[1:]) != (expected_tokens,):
        raise ValueError("Bi-Layout enclosed depth has unexpected shape {}".format(enclosed.shape))
    if tuple(extended.shape[1:]) != (expected_tokens,):
        raise ValueError("Bi-Layout extended depth has unexpected shape {}".format(extended.shape))
    if not (
        np.isfinite(features).all()
        and np.isfinite(enclosed).all()
        and np.isfinite(extended).all()
    ):
        raise ValueError("Bi-Layout prediction cache contains non-finite values")


def _flush_values(arrays: Mapping[str, np.ndarray], names: Iterable[str]) -> None:
    for name in names:
        arrays[name].flush()


def copy_reusable_predictions(
    views: Sequence[MatchingView],
    arrays: Mapping[str, np.ndarray],
    lookup: Mapping[str, Tuple[ReuseCache, int]],
    chunk_size: int = 256,
) -> int:
    """Copy compatible existing predictions; mark complete only after flush."""

    pending_by_cache: Dict[str, List[Tuple[int, ReuseCache, int]]] = {}
    completed = np.asarray(arrays["completed"])
    for view in views:
        if completed[view.index] != 0:
            continue
        source = lookup.get(view.image_path)
        if source is None:
            continue
        cache, source_index = source
        pending_by_cache.setdefault(str(cache.path), []).append(
            (view.index, cache, source_index)
        )

    copied = 0
    for entries in pending_by_cache.values():
        for offset in range(0, len(entries), chunk_size):
            chunk = entries[offset : offset + chunk_size]
            cache = chunk[0][1]
            destination_indices = np.asarray([item[0] for item in chunk], dtype=np.int64)
            source_indices = np.asarray([item[2] for item in chunk], dtype=np.int64)
            features = np.asarray(cache.features[source_indices], dtype=np.float16)
            enclosed = np.asarray(cache.enclosed_depth[source_indices], dtype=np.float32)
            extended = np.asarray(cache.extended_depth[source_indices], dtype=np.float32)
            _validate_prediction_batch(
                features,
                enclosed,
                extended,
                arrays["features"].shape[1],
                arrays["features"].shape[2],
            )
            arrays["features"][destination_indices] = features
            arrays["enclosed_depth"][destination_indices] = enclosed
            arrays["extended_depth"][destination_indices] = extended
            _flush_values(
                arrays, ("features", "enclosed_depth", "extended_depth")
            )
            arrays["origins"][destination_indices] = 1
            arrays["completed"][destination_indices] = 1
            _flush_values(arrays, ("origins", "completed"))
            copied += len(chunk)
    return copied


class PendingImageDataset(Dataset):
    def __init__(self, views: Sequence[MatchingView], image_shape: Tuple[int, int]):
        self.views = list(views)
        self.image_shape = tuple(image_shape)

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        view = self.views[index]
        height, width = self.image_shape
        resampling = getattr(Image, "Resampling", Image)
        with Image.open(view.image_path) as image:
            image = image.convert("RGB").resize((width, height), resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
        expected_shape = (height, width, 3)
        if tuple(array.shape) != expected_shape or not np.isfinite(array).all():
            raise ValueError("invalid panorama image: {}".format(view.image_path))
        return {
            "index": int(view.index),
            "image": torch.from_numpy(array.transpose(2, 0, 1).copy()),
        }


def infer_pending_predictions(
    views: Sequence[MatchingView],
    arrays: Mapping[str, np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    image_shape: Tuple[int, int],
    branch_order: str,
    batch_size: int,
    workers: int,
    progress_every: int,
    cache_dir: Path,
    metadata: Dict[str, Any],
) -> int:
    pending = [view for view in views if arrays["completed"][view.index] == 0]
    if not pending:
        return 0
    loader = DataLoader(
        PendingImageDataset(pending, image_shape),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model.eval().requires_grad_(False)
    inferred = 0
    last_reported = 0
    stage_start = time.perf_counter()
    for batch in loader:
        indices = batch["index"].cpu().numpy().astype(np.int64)
        images = batch["image"].to(
            device, dtype=torch.float32, non_blocking=device.type == "cuda"
        )
        with torch.no_grad():
            output = model(images, return_features=True)
            enclosed, extended = resolve_enclosed_extended_depth(
                output, branch_order=branch_order
            )
        features_np = output["layout_feature"].detach().cpu().numpy().astype(np.float16)
        enclosed_np = enclosed.detach().cpu().numpy().astype(np.float32)
        extended_np = extended.detach().cpu().numpy().astype(np.float32)
        _validate_prediction_batch(
            features_np,
            enclosed_np,
            extended_np,
            arrays["features"].shape[1],
            arrays["features"].shape[2],
        )
        arrays["features"][indices] = features_np
        arrays["enclosed_depth"][indices] = enclosed_np
        arrays["extended_depth"][indices] = extended_np
        _flush_values(arrays, ("features", "enclosed_depth", "extended_depth"))
        arrays["origins"][indices] = 2
        arrays["completed"][indices] = 1
        _flush_values(arrays, ("origins", "completed"))
        inferred += len(indices)
        if inferred - last_reported >= progress_every or inferred == len(pending):
            elapsed = time.perf_counter() - stage_start
            metadata.update(
                complete=False,
                completed_count=int(np.asarray(arrays["completed"]).sum()),
                reused_count=int(np.count_nonzero(np.asarray(arrays["origins"]) == 1)),
                inferred_count=int(np.count_nonzero(np.asarray(arrays["origins"]) == 2)),
                runtime_seconds=float(metadata.get("runtime_seconds", 0.0)) + elapsed,
            )
            _atomic_write_json(cache_dir / "metadata.json", metadata)
            print(
                "cache {} inference [{}/{}], total complete {}/{}".format(
                    cache_dir.name,
                    inferred,
                    len(pending),
                    metadata["completed_count"],
                    len(views),
                ),
                flush=True,
            )
            stage_start = time.perf_counter()
            last_reported = inferred
    return inferred


def finalize_cache(
    cache_dir: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    for value in arrays.values():
        value.flush()
    sample_count = int(metadata["sample_count"])
    completed_count = int(np.asarray(arrays["completed"]).sum())
    if completed_count != sample_count:
        raise RuntimeError(
            "cache incomplete at {}: {}/{}".format(
                cache_dir, completed_count, sample_count
            )
        )
    metadata.update(
        complete=True,
        completed_count=completed_count,
        reused_count=int(np.count_nonzero(np.asarray(arrays["origins"]) == 1)),
        inferred_count=int(np.count_nonzero(np.asarray(arrays["origins"]) == 2)),
    )
    _atomic_write_json(cache_dir / "metadata.json", metadata)
    return metadata


def _coverage(
    views: Sequence[MatchingView],
    lookup: Mapping[str, Tuple[ReuseCache, int]],
) -> int:
    return sum(view.image_path in lookup for view in views)


def _estimate_seconds_per_image(caches: Sequence[ReuseCache]) -> float:
    samples = 0
    seconds = 0.0
    for cache in caches:
        count = int(cache.metadata.get("inferred_count", cache.metadata.get("completed_count", 0)))
        runtime = float(cache.metadata.get("runtime_seconds", 0.0))
        if count > 0 and runtime > 0:
            samples += count
            seconds += runtime
    return seconds / samples if samples else 0.05


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        args.splits = ("test",)
        args.max_images_per_split = args.max_images_per_split or 2
        args.batch_size = min(args.batch_size, 2)
        args.workers = 0
        args.progress_every = 1
        if Path(args.output_dir).expanduser().resolve() == DEFAULT_OUTPUT_DIR.resolve():
            args.output_dir = "/tmp/zind_matching_feature_cache_smoke"
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    manifest_root = Path(args.manifest_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.bi_layout_config).expanduser().resolve()
    checkpoint_path = Path(args.bi_layout_checkpoint).expanduser().resolve()
    for path in (config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config_identity = _file_identity(config_path, hash_contents=True)
    checkpoint_identity = _file_identity(checkpoint_path, hash_contents=False)

    split_views: Dict[str, List[MatchingView]] = {}
    split_tokens: Dict[str, int] = {}
    split_roots: Dict[str, Path] = {}
    for split in args.splits:
        manifest_path = manifest_root / "{}.json".format(split)
        views, token_count, data_root = collect_unique_views(
            manifest_path,
            data_root=args.data_root,
            max_images=args.max_images_per_split,
        )
        split_views[split] = views
        split_tokens[split] = token_count
        split_roots[split] = data_root
    if len(set(split_tokens.values())) != 1:
        raise ValueError("all requested splits must use one token count")
    token_count = next(iter(split_tokens.values()))

    # The current Bi-Layout checkpoint contract fixes this at 512.  Audit mode
    # obtains it from a compatible existing cache and avoids loading 1.2 GiB.
    feature_dim_hint = None
    raw_reuse_paths = args.reuse_cache or [str(path) for path in DEFAULT_REUSE_CACHES]
    for raw_path in raw_reuse_paths:
        metadata_path = Path(raw_path).expanduser().resolve() / "metadata.json"
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as file:
                candidate_metadata = json.load(file)
            value = candidate_metadata.get("feature_dim")
            if isinstance(value, int) and value > 0:
                feature_dim_hint = value
                break
    if feature_dim_hint is None and args.audit_only:
        raise ValueError("audit requires at least one compatible cache to infer feature_dim")

    device = _resolve_device(args.device)
    model = None
    backbone_report: Mapping[str, Any] = {}
    if not args.audit_only:
        model, backbone_report = load_bi_layout(
            str(config_path), str(checkpoint_path), device, load_checkpoint=True
        )
        if not backbone_report.get("loaded"):
            raise RuntimeError("Bi-Layout checkpoint was not loaded")
        if backbone_report.get("missingKeys") or backbone_report.get("unexpectedKeys"):
            raise RuntimeError("Bi-Layout checkpoint is incompatible")
        feature_dim = int(model.patch_dim)
        if feature_dim_hint is not None and feature_dim_hint != feature_dim:
            raise ValueError("reuse cache feature_dim disagrees with Bi-Layout model")
    else:
        feature_dim = int(feature_dim_hint)

    reuse_caches: List[ReuseCache] = []
    for raw_path in raw_reuse_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            print("warning: reuse cache does not exist: {}".format(path), flush=True)
            continue
        reuse_caches.append(
            load_reuse_cache(
                path,
                config_identity,
                checkpoint_identity,
                (args.image_height, args.image_width),
                args.branch_order,
                token_count,
                feature_dim,
            )
        )
    reuse_lookup = _build_reuse_lookup(reuse_caches)
    seconds_per_image = _estimate_seconds_per_image(reuse_caches)

    audit_splits = []
    total_unique = 0
    total_reusable = 0
    for split in args.splits:
        unique_count = len(split_views[split])
        reusable_count = _coverage(split_views[split], reuse_lookup)
        missing_count = unique_count - reusable_count
        total_unique += unique_count
        total_reusable += reusable_count
        audit_splits.append(
            {
                "split": split,
                "unique_images": unique_count,
                "reusable_images": reusable_count,
                "inference_images": missing_count,
                "reuse_coverage": reusable_count / unique_count,
                "estimated_inference_seconds": missing_count * seconds_per_image,
            }
        )
    audit = {
        "task": "ZInD_matching feature cache audit",
        "splits": audit_splits,
        "total_unique_images": total_unique,
        "total_reusable_images": total_reusable,
        "total_inference_images": total_unique - total_reusable,
        "estimated_seconds_per_image": seconds_per_image,
        "estimated_inference_seconds": (total_unique - total_reusable)
        * seconds_per_image,
        "reuse_caches": [str(cache.path) for cache in reuse_caches],
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if args.audit_only:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in args.splits:
        split_start = time.perf_counter()
        views = split_views[split]
        cache_dir = output_root / split
        contract = _contract(
            split,
            views,
            split_tokens[split],
            feature_dim,
            manifest_root / "{}.json".format(split),
            config_path,
            checkpoint_path,
            (args.image_height, args.image_width),
            args.branch_order,
            args.max_images_per_split,
        )
        metadata, arrays = _initialize_output(
            cache_dir, contract, views, rebuild=args.rebuild
        )
        if metadata.get("complete") is True and int(arrays["completed"].sum()) == len(views):
            print("cache already complete: {} ({} images)".format(cache_dir, len(views)))
        else:
            copied = copy_reusable_predictions(views, arrays, reuse_lookup)
            if copied:
                print(
                    "cache {} reused {} compatible predictions".format(split, copied),
                    flush=True,
                )
            metadata.update(
                completed_count=int(np.asarray(arrays["completed"]).sum()),
                reused_count=int(np.count_nonzero(np.asarray(arrays["origins"]) == 1)),
                inferred_count=int(np.count_nonzero(np.asarray(arrays["origins"]) == 2)),
            )
            _atomic_write_json(cache_dir / "metadata.json", metadata)
            infer_pending_predictions(
                views,
                arrays,
                model,
                device,
                (args.image_height, args.image_width),
                args.branch_order,
                args.batch_size,
                args.workers,
                args.progress_every,
                cache_dir,
                metadata,
            )
            metadata = finalize_cache(cache_dir, arrays, metadata)
        summaries.append(
            {
                "split": split,
                "cache_dir": str(cache_dir),
                "sample_count": int(metadata["sample_count"]),
                "reused_count": int(metadata.get("reused_count", 0)),
                "inferred_count": int(metadata.get("inferred_count", 0)),
                "complete": bool(metadata.get("complete")),
                "wall_seconds": time.perf_counter() - split_start,
            }
        )
        print(
            "cache complete: {} reuse={} inference={}".format(
                cache_dir,
                metadata.get("reused_count", 0),
                metadata.get("inferred_count", 0),
            ),
            flush=True,
        )

    summary = {
        "format_version": 1,
        "task": "ZInD_matching frozen Bi-Layout feature cache build",
        "output_dir": str(output_root),
        "device": str(device),
        "audit": audit,
        "bi_layout_config": config_identity,
        "bi_layout_checkpoint": checkpoint_identity,
        "backbone_loaded": bool(backbone_report.get("loaded")),
        "uses_ground_truth_depth": False,
        "splits": summaries,
    }
    _atomic_write_json(output_root / "summary.json", summary)
    print("feature cache build complete: {}".format(output_root), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
