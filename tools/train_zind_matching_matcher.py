#!/usr/bin/env python3
"""Train the cross-scene matcher on complete-room ZInD_matching pairs.

This entry point keeps the original panorama-pair candidate identities, uses
the doors+openings Opening Head checkpoint, and reports teacher-forced and
predicted-candidate metrics separately.  A BiPair matcher checkpoint is a
weight initializer only; resume remains restricted to this exact data and
training contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portallayout.datasets.zind_bipair_dataset import collate_zind_bipair
from portallayout.datasets.zind_matching_feature_dataset import (
    ZInDMatchingFeatureDataset,
    validate_disjoint_house_splits,
)
from portallayout.models.cross_scene_matcher import OpeningGuidedCrossAttentionMatcher
from tools.train_zind_cross_scene_matcher import (
    _append_jsonl,
    _atomic_json,
    _atomic_torch_save,
    evaluate,
    resolve_device,
    set_seed,
    train_one_epoch,
)
from tools.train_zind_opening_head import _restore_rng_state, _rng_state
from portallayout.utils.matcher_checkpoint import load_cross_scene_matcher_checkpoint
from portallayout.utils.opening_checkpoint import load_opening_head_checkpoint


DEFAULT_MANIFEST_ROOT = REPO_ROOT / "data/ZInD_matching"
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_feature_cache"
)
DEFAULT_OPENING_CHECKPOINT = REPO_ROOT / "checkpoints/Opening_Head/zind_matching_v1/best.pt"
DEFAULT_INITIALIZER = REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_bipair_v1/best.pt"
DEFAULT_CONFIG = REPO_ROOT / "configs/layout/zind_all.yaml"
DEFAULT_BACKBONE = REPO_ROOT / "checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_v1"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_root", default=str(DEFAULT_MANIFEST_ROOT))
    parser.add_argument("--train_manifest")
    parser.add_argument("--val_manifest")
    parser.add_argument("--test_manifest")
    parser.add_argument("--data_root")
    parser.add_argument("--cache_root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--train_cache_dir")
    parser.add_argument("--val_cache_dir")
    parser.add_argument("--test_cache_dir")
    parser.add_argument("--opening_checkpoint", default=str(DEFAULT_OPENING_CHECKPOINT))
    parser.add_argument("--init_matcher_checkpoint", default=str(DEFAULT_INITIALIZER))
    parser.add_argument("--no_init_matcher", action="store_true")
    parser.add_argument("--bi_layout_config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bi_layout_checkpoint", default=str(DEFAULT_BACKBONE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--candidate_temperature", type=float, default=0.2)
    parser.add_argument("--shift_temperature", type=float, default=0.05)
    parser.add_argument("--dustbin_score", type=float, default=0.0)
    parser.add_argument("--gt_guidance_epochs", type=int, default=2)
    parser.add_argument("--mix_guidance_epochs", type=int, default=2)
    parser.add_argument("--mix_guidance_weight", type=float, default=0.5)
    parser.add_argument("--candidate_loss_weight", type=float, default=1.0)
    parser.add_argument("--token_affinity_weight", type=float, default=1.0)
    parser.add_argument("--shared_response_weight", type=float, default=0.0)
    parser.add_argument("--portal_shift_loss_weight", type=float, default=0.2)
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--opening_threshold", type=float)
    parser.add_argument("--min_opening_width_tokens", type=int, default=2)
    parser.add_argument("--max_openings_per_view", type=int, default=12)
    parser.add_argument("--metric_top_k", type=int, default=3)
    parser.add_argument("--max_train_pairs", type=int)
    parser.add_argument("--max_val_pairs", type=int)
    parser.add_argument("--max_test_pairs", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.epochs,
        args.batch_size,
        args.lr,
        args.heads,
        args.hidden_dim,
        args.candidate_temperature,
        args.shift_temperature,
        args.min_opening_width_tokens,
        args.max_openings_per_view,
        args.metric_top_k,
        args.torch_threads,
        args.progress_every,
    )
    if min(positive) <= 0:
        raise ValueError("positive training/model arguments must be > 0")
    nonnegative = (
        args.workers,
        args.weight_decay,
        args.grad_clip,
        args.gt_guidance_epochs,
        args.mix_guidance_epochs,
        args.candidate_loss_weight,
        args.token_affinity_weight,
        args.shared_response_weight,
        args.portal_shift_loss_weight,
        args.consistency_weight,
    )
    if min(nonnegative) < 0:
        raise ValueError("workers, schedules, and loss weights must be non-negative")
    for name in ("dropout", "mix_guidance_weight"):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError("{} must be in [0, 1]".format(name))
    if args.opening_threshold is not None and not 0.0 <= args.opening_threshold <= 1.0:
        raise ValueError("opening_threshold must be in [0, 1]")
    for value in (args.max_train_pairs, args.max_val_pairs, args.max_test_pairs):
        if value is not None and value <= 0:
            raise ValueError("max pair limits must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, hash_contents: bool = True) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat()
    output: Dict[str, Any] = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_contents:
        output["sha256"] = _sha256(path)
    return output


def _identity_fingerprint(value: Any, name: str) -> Tuple[Any, ...]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a file identity mapping".format(name))
    size = value.get("size")
    if type(size) is not int or size < 0:
        raise ValueError("{}.size must be a non-negative integer".format(name))
    sha256 = value.get("sha256")
    if sha256 is not None:
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError("{}.sha256 must be a SHA-256 string".format(name))
        return (size, "sha256", sha256)
    mtime_ns = value.get("mtime_ns", value.get("mtimeNs"))
    if type(mtime_ns) is not int or mtime_ns < 0:
        raise ValueError("{}.mtime_ns must be a non-negative integer".format(name))
    return (size, "mtime_ns", mtime_ns)


def validate_cache_contract(
    dataset: ZInDMatchingFeatureDataset,
    manifest_path: Path,
    opening_train_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = dataset.metadata
    if metadata.get("complete") is not True:
        raise ValueError("{} cache is incomplete".format(dataset.split))
    if metadata.get("branch_order") != "extended_first":
        raise ValueError("{} cache must use extended_first".format(dataset.split))
    if metadata.get("uses_ground_truth_depth") is not False:
        raise ValueError("{} cache must declare predicted, not GT, depth".format(dataset.split))
    if set(metadata.get("opening_label_types", ())) != {"door", "opening"}:
        raise ValueError("{} cache must use door+opening targets".format(dataset.split))
    saved_manifest = _identity_fingerprint(
        metadata.get("manifest"), "{} cache manifest".format(dataset.split)
    )
    current_manifest = _identity_fingerprint(
        _file_identity(manifest_path, hash_contents=True),
        "{} current manifest".format(dataset.split),
    )
    if saved_manifest != current_manifest:
        raise ValueError("{} cache manifest identity changed".format(dataset.split))
    for key in ("bi_layout_config", "bi_layout_checkpoint"):
        actual = _identity_fingerprint(
            metadata.get(key), "{} cache {}".format(dataset.split, key)
        )
        expected = _identity_fingerprint(
            opening_train_contract.get(key), "Opening checkpoint {}".format(key)
        )
        if actual != expected:
            raise ValueError(
                "{} cache {} differs from Opening checkpoint".format(dataset.split, key)
            )
    return {
        "validated": True,
        "split": dataset.split,
        "pairCount": len(dataset),
        "viewCount": dataset.sample_count,
        "tokenCount": dataset.token_count,
        "featureDim": dataset.feature_dim,
        "openingLabelTypes": ["door", "opening"],
        "usesGroundTruthDepth": False,
        "manifestIdentity": list(saved_manifest),
    }


def _make_loader(
    manifest: Path,
    cache_dir: Path,
    max_pairs: Optional[int],
    args: argparse.Namespace,
    shuffle: bool,
    device: torch.device,
    split: str,
) -> Tuple[ZInDMatchingFeatureDataset, DataLoader]:
    dataset = ZInDMatchingFeatureDataset(
        str(manifest),
        str(cache_dir),
        data_root=args.data_root,
        max_pairs=max_pairs,
        expected_split=split,
    )
    loader = _loader_for_dataset(
        dataset,
        args,
        shuffle=shuffle,
        device=device,
        seed=args.seed,
    )
    return dataset, loader


def _loader_for_dataset(
    dataset: ZInDMatchingFeatureDataset,
    args: argparse.Namespace,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_zind_bipair,
        generator=generator if shuffle else None,
    )
    return loader


def _model_contract(matcher: OpeningGuidedCrossAttentionMatcher, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "feature_dim": int(matcher.feature_dim),
        "heads": int(matcher.heads),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "opening_bias_strength": float(matcher.opening_bias_strength),
        "candidate_temperature": float(args.candidate_temperature),
        "shift_temperature": float(args.shift_temperature),
        "has_dustbin": True,
    }


def _assert_model_contract(saved: Any, expected: Mapping[str, Any], scope: str) -> None:
    if not isinstance(saved, Mapping):
        raise ValueError("{} is missing model contract".format(scope))
    for key, expected_value in expected.items():
        actual = saved.get(key)
        if isinstance(expected_value, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(
                "{} model.{} mismatch: {!r} != {!r}".format(
                    scope, key, actual, expected_value
                )
            )


def _data_contract(
    datasets: Mapping[str, ZInDMatchingFeatureDataset],
    cache_validation: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        'portallayout.datasets': "ZInD_matching formatVersion=2 complete-room pairs",
        "manifests": {name: str(dataset.manifest_path) for name, dataset in datasets.items()},
        "feature_caches": {name: str(dataset.cache_dir) for name, dataset in datasets.items()},
        "pair_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "token_count": datasets["train"].token_count,
        "feature_dim": datasets["train"].feature_dim,
        "opening_label_types": ["door", "opening"],
        "gt_candidate_supervision": "manifest identity-preserving multi-hot candidate_target_AB",
        "predicted_candidate_supervision": "dense portal affinity projected with IoU>=0.30",
        "opening_guidance": "union of all doors+openings; never shared target only",
        "portal_shift": "circular shared portal center B-minus-A; multi-interface pairs excluded",
        "camera_pose_yaw_is_not_portal_shift": True,
        "coordinate_frame": "zind_pano_local_xy",
        "transform_direction": "b_to_a",
        "feature_cache_validation": dict(cache_validation),
    }


def _training_contract(args: argparse.Namespace) -> Dict[str, Any]:
    keys = (
        "batch_size", "lr", "weight_decay", "grad_clip", "gt_guidance_epochs",
        "mix_guidance_epochs", "mix_guidance_weight", "candidate_loss_weight",
        "token_affinity_weight", "shared_response_weight",
        "portal_shift_loss_weight", "consistency_weight", "min_opening_width_tokens",
        "max_openings_per_view", "metric_top_k", "opening_threshold",
        "dustbin_score", "max_train_pairs", "max_val_pairs", "max_test_pairs", "seed",
    )
    return {
        **{key: getattr(args, key) for key in keys},
        "candidate_source_schedule": "GT during gt guidance; predicted during mix/predicted guidance",
    }


def _validate_resume(
    checkpoint: Mapping[str, Any],
    model_contract: Mapping[str, Any],
    data_contract: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    opening_identity: Mapping[str, Any],
) -> None:
    if checkpoint.get("task") != "ZInD_matching complete-room cross-scene opening matcher":
        raise ValueError("resume checkpoint task is not ZInD_matching")
    _assert_model_contract(checkpoint.get("model"), model_contract, "resume")
    if checkpoint.get("data_contract") != data_contract:
        raise ValueError("resume data contract changed")
    if checkpoint.get("training_contract") != training_contract:
        raise ValueError("resume training contract changed")
    saved_identity = checkpoint.get("opening_checkpoint_identity")
    if _identity_fingerprint(saved_identity, "resume Opening checkpoint") != _identity_fingerprint(
        opening_identity, "current Opening checkpoint"
    ):
        raise ValueError("resume Opening checkpoint changed")


def _checkpoint_payload(
    matcher: OpeningGuidedCrossAttentionMatcher,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    validation_gt: Mapping[str, Any],
    validation_predicted: Mapping[str, Any],
    opening_report: Mapping[str, Any],
    opening_identity: Mapping[str, Any],
    initializer: Optional[Mapping[str, Any]],
    data_contract: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    model_contract: Mapping[str, Any],
    opening_threshold: float,
    cumulative_runtime_seconds: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "task": "ZInD_matching complete-room cross-scene opening matcher",
        "completed_epoch": int(epoch),
        "next_epoch": int(epoch + 1),
        "matcher_state_dict": matcher.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "opening_head_dependency": dict(opening_report),
        "opening_checkpoint_identity": dict(opening_identity),
        "opening_probability_threshold": float(opening_threshold),
        "opening_label_types": ["door", "opening"],
        "initializer": None if initializer is None else dict(initializer),
        "model": dict(model_contract),
        "data_contract": dict(data_contract),
        "training_contract": dict(training_contract),
        "training_args": vars(args),
        "cumulative_runtime_seconds": float(cumulative_runtime_seconds),
        "train": dict(train_metrics),
        "validation": {
            "teacher_forced_manifest_candidates": dict(validation_gt),
            "predicted_openings": dict(validation_predicted),
        },
        "best": dict(best),
        "rng_state": _rng_state(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        args.epochs = 1
        args.max_train_pairs = args.max_train_pairs or 16
        args.max_val_pairs = args.max_val_pairs or 8
        args.max_test_pairs = args.max_test_pairs or 8
        args.batch_size = min(args.batch_size, 2)
        args.workers = 0
        args.progress_every = min(args.progress_every, 2)
    args.sync_candidate_schedule = True
    _validate_args(args)
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    print("device: {}".format(device), flush=True)

    manifest_root = Path(args.manifest_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    manifests = {
        "train": Path(args.train_manifest).expanduser().resolve() if args.train_manifest else manifest_root / "train.json",
        "val": Path(args.val_manifest).expanduser().resolve() if args.val_manifest else manifest_root / "val.json",
        "test": Path(args.test_manifest).expanduser().resolve() if args.test_manifest else manifest_root / "test.json",
    }
    caches = {
        "train": Path(args.train_cache_dir).expanduser().resolve() if args.train_cache_dir else cache_root / "train",
        "val": Path(args.val_cache_dir).expanduser().resolve() if args.val_cache_dir else cache_root / "val",
        "test": Path(args.test_cache_dir).expanduser().resolve() if args.test_cache_dir else cache_root / "test",
    }
    opening_checkpoint = Path(args.opening_checkpoint).expanduser().resolve()
    config_path = Path(args.bi_layout_config).expanduser().resolve()
    backbone_path = Path(args.bi_layout_checkpoint).expanduser().resolve()
    required = list(manifests.values()) + [
        cache / "metadata.json" for cache in caches.values()
    ] + [opening_checkpoint, config_path, backbone_path]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    limits = {
        "train": args.max_train_pairs,
        "val": args.max_val_pairs,
        "test": args.max_test_pairs,
    }
    datasets: Dict[str, ZInDMatchingFeatureDataset] = {}
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        datasets[split], loaders[split] = _make_loader(
            manifests[split], caches[split], limits[split], args,
            shuffle=split == "train", device=device, split=split,
        )
    house_counts = validate_disjoint_house_splits(datasets)
    dimensions = {(dataset.token_count, dataset.feature_dim) for dataset in datasets.values()}
    if len(dimensions) != 1:
        raise ValueError("train/val/test cache dimensions differ")
    token_count, feature_dim = next(iter(dimensions))
    if feature_dim % args.heads != 0:
        raise ValueError("feature_dim must be divisible by --heads")
    print(
        "pairs train/val/test={}/{}/{}; houses={}; feature=[{}, {}]".format(
            len(datasets["train"]), len(datasets["val"]), len(datasets["test"]),
            house_counts, token_count, feature_dim,
        ),
        flush=True,
    )

    opening_payload = torch.load(str(opening_checkpoint), map_location="cpu")
    if not isinstance(opening_payload, Mapping):
        raise ValueError("Opening checkpoint root must be a mapping")
    opening_train_contract = opening_payload.get("train_cache_contract")
    if not isinstance(opening_train_contract, Mapping):
        raise ValueError("Opening checkpoint is missing train cache contract")
    label_types = opening_payload.get(
        "opening_label_types", opening_train_contract.get("opening_label_types")
    )
    if set(label_types or ()) != {"door", "opening"}:
        raise ValueError("Opening checkpoint was not trained on doors+openings")
    cache_validation = {
        split: validate_cache_contract(datasets[split], manifests[split], opening_train_contract)
        for split in ("train", "val", "test")
    }

    matcher = OpeningGuidedCrossAttentionMatcher(
        feature_dim=feature_dim,
        heads=args.heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        candidate_temperature=args.candidate_temperature,
        shift_temperature=args.shift_temperature,
        dustbin_score=args.dustbin_score,
    ).to(device)
    expected_model = _model_contract(matcher, args)
    initializer_report: Optional[Mapping[str, Any]] = None
    initializer_path = Path(args.init_matcher_checkpoint).expanduser().resolve()
    if not args.resume and not args.no_init_matcher:
        initialized, report = load_cross_scene_matcher_checkpoint(
            str(initializer_path),
            expected_feature_dim=feature_dim,
            expected_token_count=token_count,
            expected_branch_order="extended_first",
            bi_layout_config_path=str(config_path),
            bi_layout_checkpoint_path=str(backbone_path),
            device=device,
        )
        _assert_model_contract(report.get("model"), expected_model, "initializer")
        matcher = initialized
        initializer_report = {
            **dict(report),
            "identity": _file_identity(initializer_path, hash_contents=True),
            "mode": "weights_only_optimizer_and_schedule_reset",
        }
        print("warm-started matcher weights: {}".format(initializer_path), flush=True)

    opening_report = load_opening_head_checkpoint(
        matcher.opening_head,
        str(opening_checkpoint),
        expected_feature_dim=feature_dim,
        expected_branch_order="extended_first",
        expected_token_count=token_count,
        bi_layout_config_path=str(config_path),
        bi_layout_checkpoint_path=str(backbone_path),
    )
    opening_threshold = (
        float(args.opening_threshold)
        if args.opening_threshold is not None
        else float(opening_report["operatingThreshold"])
    )
    matcher.opening_head.requires_grad_(False).eval()
    opening_identity = _file_identity(opening_checkpoint, hash_contents=True)
    model_contract = _model_contract(matcher, args)
    data_contract = _data_contract(datasets, cache_validation)
    training_contract = _training_contract(args)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in matcher.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    cumulative_runtime = 0.0
    best: Dict[str, Any] = {
        "epoch": -1,
        "harmonic_flow_score": -1.0,
        "balanced_flow_score": -1.0,
        "end_to_end_candidate_top1": -1.0,
        "negative_rejection_rate": -1.0,
    }
    if args.resume:
        resume_path = output_dir / "last.pt" if args.resume == "auto" else Path(args.resume).expanduser().resolve()
        checkpoint = torch.load(str(resume_path), map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("resume checkpoint root must be a mapping")
        _validate_resume(
            checkpoint, model_contract, data_contract, training_contract, opening_identity
        )
        matcher.load_state_dict(checkpoint["matcher_state_dict"], strict=True)
        opening_report = load_opening_head_checkpoint(
            matcher.opening_head,
            str(opening_checkpoint),
            expected_feature_dim=feature_dim,
            expected_branch_order="extended_first",
            expected_token_count=token_count,
            bi_layout_config_path=str(config_path),
            bi_layout_checkpoint_path=str(backbone_path),
        )
        matcher.opening_head.requires_grad_(False).eval()
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["next_epoch"])
        best = dict(checkpoint.get("best", best))
        cumulative_runtime = float(checkpoint.get("cumulative_runtime_seconds", 0.0))
        initializer_report = checkpoint.get("initializer")
        _restore_rng_state(checkpoint.get("rng_state"))
        print("resumed {} at epoch {}".format(resume_path, start_epoch), flush=True)
    elif args.overwrite:
        for name in (
            "last.pt",
            "best.pt",
            "metrics.jsonl",
            "latest_metrics.json",
            "best_metrics.json",
            "test_metrics.json",
        ):
            path = output_dir / name
            if path.exists():
                path.unlink()
    elif any(
        (output_dir / name).exists()
        for name in ("last.pt", "best.pt", "metrics.jsonl")
    ):
        raise FileExistsError(
            "training artifacts exist in {}; use --resume or --overwrite".format(
                output_dir
            )
        )

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        print("epoch {}/{}".format(epoch + 1, args.epochs), flush=True)
        # Epoch-derived shuffling keeps sample order reproducible across an
        # interrupted/resumed run without serializing a DataLoader generator.
        train_loader = _loader_for_dataset(
            datasets["train"],
            args,
            shuffle=True,
            device=device,
            seed=args.seed + epoch,
        )
        train_metrics = train_one_epoch(
            matcher, train_loader, optimizer, device, epoch, opening_threshold, args
        )
        validation_gt = evaluate(
            matcher, loaders["val"], device, "gt", opening_threshold, args
        )
        validation_predicted = evaluate(
            matcher, loaders["val"], device, "predicted", opening_threshold, args
        )
        epoch_runtime = time.perf_counter() - epoch_start
        cumulative_runtime += epoch_runtime
        current_score = (
            float(validation_predicted["harmonic_flow_score"]),
            float(validation_predicted["balanced_flow_score"]),
            float(validation_predicted["end_to_end_candidate_top1"]),
            float(validation_predicted["negative_rejection_rate"]),
        )
        best_score = (
            float(best["harmonic_flow_score"]),
            float(best["balanced_flow_score"]),
            float(best["end_to_end_candidate_top1"]),
            float(best["negative_rejection_rate"]),
        )
        improved = current_score > best_score
        if improved:
            best = {
                "epoch": int(epoch),
                "harmonic_flow_score": current_score[0],
                "balanced_flow_score": current_score[1],
                "end_to_end_candidate_top1": current_score[2],
                "negative_rejection_rate": current_score[3],
            }
        report = {
            "epoch": int(epoch),
            "epoch_runtime_seconds": epoch_runtime,
            "cumulative_runtime_seconds": cumulative_runtime,
            "train": train_metrics,
            "validation": {
                "teacher_forced_manifest_candidates": validation_gt,
                "predicted_openings": validation_predicted,
            },
            "opening_threshold": opening_threshold,
            "improved": improved,
            "best": best,
        }
        payload = _checkpoint_payload(
            matcher, optimizer, epoch, best, train_metrics, validation_gt,
            validation_predicted, opening_report, opening_identity,
            initializer_report, data_contract, training_contract, model_contract,
            opening_threshold, cumulative_runtime, args,
        )
        _atomic_torch_save(output_dir / "last.pt", payload)
        if improved:
            _atomic_torch_save(output_dir / "best.pt", payload)
            _atomic_json(output_dir / "best_metrics.json", report)
        _append_jsonl(output_dir / "metrics.jsonl", report)
        _atomic_json(output_dir / "latest_metrics.json", report)
        print(
            "  val GT top1={:.4f}; predicted recall={:.4f} top1={:.4f} "
            "negative_reject={:.4f} harmonic={:.4f} balanced={:.4f}; epoch={:.1f}s{}".format(
                validation_gt["end_to_end_candidate_top1"],
                validation_predicted["positive_candidate_recall"],
                validation_predicted["end_to_end_candidate_top1"],
                validation_predicted["negative_rejection_rate"],
                validation_predicted["harmonic_flow_score"],
                validation_predicted["balanced_flow_score"],
                epoch_runtime,
                " BEST" if improved else "",
            ),
            flush=True,
        )

    if start_epoch >= args.epochs:
        print("checkpoint already reached requested epochs={}".format(args.epochs), flush=True)
    if not args.skip_test:
        best_path = output_dir / "best.pt"
        checkpoint = torch.load(str(best_path), map_location="cpu")
        matcher.load_state_dict(checkpoint["matcher_state_dict"], strict=True)
        load_opening_head_checkpoint(
            matcher.opening_head,
            str(opening_checkpoint),
            expected_feature_dim=feature_dim,
            expected_branch_order="extended_first",
            expected_token_count=token_count,
            bi_layout_config_path=str(config_path),
            bi_layout_checkpoint_path=str(backbone_path),
        )
        matcher.opening_head.requires_grad_(False).eval()
        test_start = time.perf_counter()
        test_gt = evaluate(
            matcher, loaders["test"], device, "gt", opening_threshold, args
        )
        test_predicted = evaluate(
            matcher, loaders["test"], device, "predicted", opening_threshold, args
        )
        test_report = {
            "checkpoint": str(best_path),
            "best_epoch": int(checkpoint["completed_epoch"]),
            "opening_threshold": opening_threshold,
            "runtime_seconds": time.perf_counter() - test_start,
            "teacher_forced_manifest_candidates": test_gt,
            "predicted_openings": test_predicted,
        }
        _atomic_json(output_dir / "test_metrics.json", test_report)
        print(
            "test predicted recall={:.4f} top1={:.4f} negative_reject={:.4f} harmonic={:.4f} balanced={:.4f}".format(
                test_predicted["positive_candidate_recall"],
                test_predicted["end_to_end_candidate_top1"],
                test_predicted["negative_rejection_rate"],
                test_predicted["harmonic_flow_score"],
                test_predicted["balanced_flow_score"],
            ),
            flush=True,
        )
    print("training complete: {}".format(output_dir / "best.pt"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
