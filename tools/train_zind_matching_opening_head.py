#!/usr/bin/env python3
"""Domain-adapt OpeningSignalHead on cached ZInD_matching doors/openings.

This entry point never runs Bi-Layout.  It consumes the complete train/val
feature caches produced by ``cache_zind_matching_features.py``.  A warm start
loads only Opening Head weights and begins a fresh optimizer/epoch schedule;
``--resume`` instead restores the full optimizer/RNG state after requiring the
same cache and training-selection contracts.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portallayout.datasets.zind_opening_dataset import OpeningFeatureCacheDataset
from portallayout.models.cross_scene_matcher import OpeningSignalHead
from tools.train_zind_opening_head import (
    _append_jsonl,
    _atomic_torch_save,
    _atomic_write_json,
    _jsonable_args,
    _restore_rng_state,
    _rng_state,
    resolve_device,
    set_seed,
    train_one_epoch,
    validate_opening_head,
)
from portallayout.utils.opening_checkpoint import load_opening_head_checkpoint


DEFAULT_CACHE_ROOT = (
    REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_feature_cache"
)
DEFAULT_WARM_START = REPO_ROOT / "checkpoints/Opening_Head/zind_bipair_v1/best.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints/Opening_Head/zind_matching_v1"
OPENING_LABEL_TYPES = ("door", "opening")
_CACHE_CONTRACT_KEYS = (
    "format_version",
    "task",
    "split",
    "manifest",
    "bi_layout_config",
    "bi_layout_checkpoint",
    "image_shape",
    "branch_order",
    "sample_count",
    "token_count",
    "feature_dim",
    "feature_dtype",
    "depth_dtype",
    "target_dtype",
    "opening_label_types",
    "feature_source",
    "depth_source",
    "uses_ground_truth_depth",
    "selection",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--train_cache")
    parser.add_argument("--val_cache")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--warm_start", default=str(DEFAULT_WARM_START))
    parser.add_argument(
        "--no_warm_start",
        action="store_true",
        help="initialize a new head instead of loading the BiPair Opening Head",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="resume full state from a checkpoint, defaulting to output_dir/last.pt",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=2.5)
    parser.add_argument("--tversky_weight", type=float, default=0.5)
    parser.add_argument("--tversky_alpha", type=float, default=0.3)
    parser.add_argument("--tversky_beta", type=float, default=0.7)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--prior_strength", type=float, default=4.0)
    parser.add_argument("--prior_relative_scale", type=float, default=0.1)
    parser.add_argument("--roll_probability", type=float, default=1.0)
    parser.add_argument("--flip_probability", type=float, default=0.5)
    parser.add_argument("--precision_target", type=float, default=0.80)
    parser.add_argument("--scan_min", type=float, default=0.0)
    parser.add_argument("--scan_max", type=float, default=1.0)
    parser.add_argument("--scan_steps", type=int, default=201)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--max_train_views", type=int)
    parser.add_argument("--max_val_views", type=int)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.epochs,
        args.batch_size,
        args.lr,
        args.pos_weight,
        args.tversky_alpha,
        args.tversky_beta,
        args.hidden_dim,
        args.kernel_size,
        args.prior_relative_scale,
        args.scan_steps,
        args.torch_threads,
    )
    if min(positive) <= 0:
        raise ValueError("positive training/model arguments must be > 0")
    if args.workers < 0 or args.weight_decay < 0 or args.tversky_weight < 0:
        raise ValueError("workers, weight_decay, and tversky_weight must be non-negative")
    if args.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    for name in ("roll_probability", "flip_probability", "precision_target"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError("{} must be in [0, 1]".format(name))
    if not math.isfinite(args.prior_strength):
        raise ValueError("prior_strength must be finite")
    if args.scan_min >= args.scan_max:
        raise ValueError("scan_min must be lower than scan_max")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")
    for value in (args.max_train_views, args.max_val_views):
        if value is not None and value <= 0:
            raise ValueError("max view limits must be positive")


def _cache_contract(metadata: Mapping[str, Any], expected_split: str) -> Dict[str, Any]:
    if metadata.get("complete") is not True:
        raise ValueError("{} matching feature cache is incomplete".format(expected_split))
    if metadata.get("split") != expected_split:
        raise ValueError(
            "cache split {!r} does not match {!r}".format(
                metadata.get("split"), expected_split
            )
        )
    task = str(metadata.get("task", ""))
    if "ZInD_matching" not in task or "feature cache" not in task:
        raise ValueError("cache task is not a ZInD_matching feature cache")
    if tuple(metadata.get("opening_label_types", ())) != OPENING_LABEL_TYPES:
        raise ValueError("matching cache opening_label_types must be doors+openings")
    if metadata.get("branch_order") != "extended_first":
        raise ValueError("matching Opening Head requires branch_order=extended_first")
    if metadata.get("uses_ground_truth_depth") is not False:
        raise ValueError("matching cache must contain predicted, not ground-truth, depths")
    missing = [key for key in _CACHE_CONTRACT_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            "matching cache metadata is missing contract fields: {}".format(
                ", ".join(missing)
            )
        )
    return {key: metadata[key] for key in _CACHE_CONTRACT_KEYS}


def _validate_cache_pair(
    train_metadata: Mapping[str, Any], val_metadata: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_contract = _cache_contract(train_metadata, "train")
    val_contract = _cache_contract(val_metadata, "val")
    for key in (
        "bi_layout_config",
        "bi_layout_checkpoint",
        "image_shape",
        "branch_order",
        "token_count",
        "feature_dim",
        "feature_dtype",
        "depth_dtype",
        "target_dtype",
        "opening_label_types",
        "feature_source",
        "depth_source",
        "uses_ground_truth_depth",
    ):
        if train_contract[key] != val_contract[key]:
            raise ValueError("train/val matching cache contract differs for {}".format(key))
    return train_contract, val_contract


def _model_contract(
    feature_dim: int, args: argparse.Namespace
) -> Dict[str, Any]:
    return {
        "feature_dim": int(feature_dim),
        "hidden_dim": int(args.hidden_dim),
        "kernel_size": int(args.kernel_size),
        "prior_strength": float(args.prior_strength),
        "prior_relative_scale": float(args.prior_relative_scale),
        "branch_order": "extended_first",
    }


def _selection_contract(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "max_train_views": args.max_train_views,
        "max_val_views": args.max_val_views,
    }


def _training_contract(args: argparse.Namespace) -> Dict[str, Any]:
    keys = (
        "batch_size",
        "lr",
        "weight_decay",
        "pos_weight",
        "tversky_weight",
        "tversky_alpha",
        "tversky_beta",
        "roll_probability",
        "flip_probability",
        "precision_target",
        "scan_min",
        "scan_max",
        "scan_steps",
        "grad_clip",
        "seed",
    )
    return {key: getattr(args, key) for key in keys}


def _checkpoint_payload(
    head: OpeningSignalHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best: Mapping[str, Any],
    validation: Mapping[str, Any],
    args: argparse.Namespace,
    train_contract: Mapping[str, Any],
    val_contract: Mapping[str, Any],
    warm_start: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "task": "ZInD_matching single-view Opening Head (doors+openings)",
        "training_stage": "cross-complete-room opening domain adaptation",
        "opening_label_types": list(OPENING_LABEL_TYPES),
        "completed_epoch": int(epoch),
        "next_epoch": int(epoch + 1),
        "global_step": int(global_step),
        "opening_head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "operating_threshold": float(validation["selectedThreshold"]),
        "threshold_policy": validation["policy"],
        "threshold_fallback": bool(validation["fallback"]),
        "validation": dict(validation),
        "best": dict(best),
        "model": _model_contract(int(train_contract["feature_dim"]), args),
        "training_args": _jsonable_args(args),
        "training_contract": _training_contract(args),
        "training_selection": _selection_contract(args),
        "warm_start": dict(warm_start),
        "train_cache_contract": dict(train_contract),
        "val_cache_contract": dict(val_contract),
        "rng_state": _rng_state(),
    }


def _validate_resume_payload(
    checkpoint: Mapping[str, Any],
    train_contract: Mapping[str, Any],
    val_contract: Mapping[str, Any],
    feature_dim: int,
    args: argparse.Namespace,
) -> None:
    if checkpoint.get("task") != "ZInD_matching single-view Opening Head (doors+openings)":
        raise ValueError("resume checkpoint task is not ZInD_matching doors+openings")
    if tuple(checkpoint.get("opening_label_types", ())) != OPENING_LABEL_TYPES:
        raise ValueError("resume checkpoint opening_label_types changed")
    expected = (
        ("train_cache_contract", dict(train_contract)),
        ("val_cache_contract", dict(val_contract)),
        ("model", _model_contract(feature_dim, args)),
        ("training_contract", _training_contract(args)),
        ("training_selection", _selection_contract(args)),
    )
    for key, current in expected:
        if checkpoint.get(key) != current:
            raise ValueError("resume checkpoint {} contract changed".format(key))


def _limited_dataset(dataset: OpeningFeatureCacheDataset, limit: Optional[int]):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def train_from_cache(
    train_cache_dir: Path,
    val_cache_dir: Path,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    train_dataset = OpeningFeatureCacheDataset(
        str(train_cache_dir),
        augment=True,
        roll_probability=args.roll_probability,
        flip_probability=args.flip_probability,
    )
    val_dataset = OpeningFeatureCacheDataset(str(val_cache_dir), augment=False)
    train_contract, val_contract = _validate_cache_pair(
        train_dataset.metadata, val_dataset.metadata
    )
    if train_dataset.feature_dim != val_dataset.feature_dim:
        raise ValueError("train/val cache feature dimensions differ")

    train_data = _limited_dataset(train_dataset, args.max_train_views)
    val_data = _limited_dataset(val_dataset, args.max_val_views)
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    head = OpeningSignalHead(
        feature_dim=train_dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        prior_strength=args.prior_strength,
        prior_relative_scale=args.prior_relative_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best: Dict[str, Any] = {
        "epoch": -1,
        "component_recall_iou_0_3": -1.0,
        "average_precision": -1.0,
        "token_f1": -1.0,
        "operating_threshold": None,
    }
    warm_start_report: Dict[str, Any] = {
        "loaded": False,
        "optimizer_reset": True,
        "epoch_reset": True,
    }
    if args.resume:
        resume_path = (
            output_dir / "last.pt"
            if args.resume == "auto"
            else Path(args.resume).expanduser().resolve()
        )
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        checkpoint = torch.load(str(resume_path), map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("resume checkpoint root must be a mapping")
        _validate_resume_payload(
            checkpoint,
            train_contract,
            val_contract,
            train_dataset.feature_dim,
            args,
        )
        head.load_state_dict(checkpoint["opening_head_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint.get("global_step", 0))
        best = dict(checkpoint.get("best", best))
        warm_start_report = dict(checkpoint.get("warm_start", warm_start_report))
        _restore_rng_state(checkpoint.get("rng_state"))
        print("resumed: {}, next_epoch={}".format(resume_path, start_epoch))
    else:
        last_path = output_dir / "last.pt"
        if last_path.exists() and not args.overwrite:
            raise FileExistsError(
                "{} exists; pass --resume or --overwrite".format(last_path)
            )
        if args.overwrite:
            for name in ("metrics.jsonl", "latest_metrics.json"):
                path = output_dir / name
                if path.exists():
                    path.unlink()
        if not args.no_warm_start and args.warm_start:
            warm_path = Path(args.warm_start).expanduser().resolve()
            config_path = Path(train_contract["bi_layout_config"]["path"])
            backbone_path = Path(train_contract["bi_layout_checkpoint"]["path"])
            report = load_opening_head_checkpoint(
                head,
                str(warm_path),
                expected_feature_dim=train_dataset.feature_dim,
                expected_branch_order="extended_first",
                expected_token_count=train_dataset.token_count,
                bi_layout_config_path=str(config_path),
                bi_layout_checkpoint_path=str(backbone_path),
            )
            warm_start_report.update(report)
            warm_start_report.update(optimizer_reset=True, epoch_reset=True)
            print("warm-started Opening Head weights: {}".format(warm_path))

        initial_validation = validate_opening_head(head, val_loader, device, args)
        initial_selected = initial_validation['portallayout.evaluation'][
            "metricsAtSelectedThreshold"
        ]
        initial_component = initial_validation['portallayout.evaluation'][
            "componentMetricsAtSelectedThreshold"
        ]["overlap"]["iouAtLeast0.30"]
        initial_report = {
            "stage": "before ZInD_matching doors+openings adaptation",
            "warm_start": dict(warm_start_report),
            "validation": initial_validation,
        }
        _atomic_write_json(output_dir / "warm_start_baseline.json", initial_report)
        print(
            "initial val_P={:.4f} val_R={:.4f} val_F1={:.4f} "
            "interval_R@0.3={:.4f} threshold={:.3f}".format(
                initial_selected["precision"],
                initial_selected["recall"],
                initial_selected["f1"],
                initial_component["recall"],
                initial_validation["selectedThreshold"],
            )
        )

    last_report: Dict[str, Any] = {}
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        train_metrics = train_one_epoch(head, train_loader, optimizer, device, args)
        global_step += len(train_loader)
        validation = validate_opening_head(head, val_loader, device, args)
        selected = validation['portallayout.evaluation']["metricsAtSelectedThreshold"]
        component = validation['portallayout.evaluation']["componentMetricsAtSelectedThreshold"]
        component_recall = float(component["overlap"]["iouAtLeast0.30"]["recall"])
        score = (
            component_recall,
            float(validation['portallayout.evaluation']["averagePrecision"]),
            float(selected["f1"]),
        )
        best_score = (
            float(best["component_recall_iou_0_3"]),
            float(best["average_precision"]),
            float(best["token_f1"]),
        )
        improved = score > best_score
        if improved:
            best = {
                "epoch": int(epoch),
                "component_recall_iou_0_3": score[0],
                "average_precision": score[1],
                "token_f1": score[2],
                "operating_threshold": float(validation["selectedThreshold"]),
            }
        report = {
            "epoch": int(epoch),
            "runtime_seconds": time.perf_counter() - epoch_start,
            "train": train_metrics,
            "validation": validation,
            "improved": improved,
            "best": best,
            "training_view_count": len(train_data),
            "validation_view_count": len(val_data),
        }
        payload = _checkpoint_payload(
            head,
            optimizer,
            epoch,
            global_step,
            best,
            validation,
            args,
            train_contract,
            val_contract,
            warm_start_report,
        )
        _atomic_torch_save(output_dir / "last.pt", payload)
        if improved:
            _atomic_torch_save(output_dir / "best.pt", payload)
            _atomic_write_json(output_dir / "best_metrics.json", report)
        _append_jsonl(output_dir / "metrics.jsonl", report)
        _atomic_write_json(output_dir / "latest_metrics.json", report)
        print(
            "epoch {}/{} loss={:.4f} val_P={:.4f} val_R={:.4f} "
            "val_F1={:.4f} interval_R@0.3={:.4f} threshold={:.3f}{}".format(
                epoch + 1,
                args.epochs,
                train_metrics["loss_total"],
                selected["precision"],
                selected["recall"],
                selected["f1"],
                component_recall,
                validation["selectedThreshold"],
                " BEST" if improved else "",
            )
        )
        last_report = report
    if start_epoch >= args.epochs:
        print("checkpoint already reached requested epochs={}".format(args.epochs))
        last_report = {"already_complete": True, "best": best}
    return last_report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        args.epochs = 1
        args.max_train_views = args.max_train_views or 16
        args.max_val_views = args.max_val_views or 8
        args.batch_size = min(args.batch_size, 4)
        args.workers = 0
    validate_args(args)
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    cache_root = Path(args.cache_root).expanduser().resolve()
    train_cache = (
        Path(args.train_cache).expanduser().resolve()
        if args.train_cache
        else cache_root / "train"
    )
    val_cache = (
        Path(args.val_cache).expanduser().resolve()
        if args.val_cache
        else cache_root / "val"
    )
    for path in (train_cache, val_cache):
        if not path.is_dir():
            raise FileNotFoundError(path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    print(
        "device={} cached views train={} val={}".format(
            device,
            train_cache,
            val_cache,
        )
    )
    train_from_cache(train_cache, val_cache, output_dir, device, args)
    print("training complete: {}".format(output_dir / "best.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
