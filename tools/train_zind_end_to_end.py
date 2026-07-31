#!/usr/bin/env python3
"""Jointly fine-tune Bi-Layout, Opening Head, and Cross-Scene Matcher.

The complete-room ZInD_matching manifests provide the end-to-end supervision.
Every training step runs raw panoramas through the live Bi-Layout backbone,
then through the live Opening Head and Matcher.  Matching and opening losses
therefore backpropagate into the shared panorama model.  The frozen feature
cache is used only as a low-weight teacher regularizer; it is never substituted
for the live forward pass.

M6-M9 remain analytical/discrete geometry stages and are evaluated after the
learned M1/M2/M4/M5 chain.  They are not differentiable training modules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.zind_bipair_adapter import adapt_zind_bipair_batch
from dataset.zind_bipair_dataset import collate_zind_bipair
from dataset.zind_matching_feature_dataset import (
    ZInDMatchingFeatureDataset,
    validate_disjoint_house_splits,
)
from models.cross_scene_matcher import (
    OpeningGuidedCrossAttentionMatcher,
    opening_detection_loss,
    resolve_enclosed_extended_depth,
)
from tools.debug_cross_scene_flow import load_bi_layout
from tools.train_zind_cross_scene_matcher import (
    MetricAccumulator,
    _append_jsonl,
    _atomic_json,
    _atomic_torch_save,
    _losses as matcher_losses,
    candidate_targets_from_affinity,
    guidance_mode_for_epoch,
    predicted_candidate_masks,
    resolve_device,
    set_seed,
)
from utils.matcher_checkpoint import load_cross_scene_matcher_checkpoint
from utils.opening_checkpoint import (
    load_opening_head_checkpoint,
    resolve_opening_probability_threshold,
)


DEFAULT_MANIFEST_ROOT = REPO_ROOT / "src/dataset/ZInD_matching"
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_feature_cache"
)
DEFAULT_CONFIG = REPO_ROOT / "src/config/zind_all.yaml"
DEFAULT_BI_LAYOUT = (
    REPO_ROOT / "checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl"
)
DEFAULT_OPENING = REPO_ROOT / "checkpoints/Opening_Head/zind_matching_v1/best.pt"
DEFAULT_MATCHER = REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_v1/best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "checkpoints/End_To_End/zind_matching_v1"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_root", default=str(DEFAULT_MANIFEST_ROOT))
    parser.add_argument("--cache_root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--train_manifest")
    parser.add_argument("--val_manifest")
    parser.add_argument("--test_manifest")
    parser.add_argument("--train_cache_dir")
    parser.add_argument("--val_cache_dir")
    parser.add_argument("--test_cache_dir")
    parser.add_argument("--data_root")
    parser.add_argument("--bi_layout_config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bi_layout_checkpoint", default=str(DEFAULT_BI_LAYOUT))
    parser.add_argument("--opening_checkpoint", default=str(DEFAULT_OPENING))
    parser.add_argument("--matcher_checkpoint", default=str(DEFAULT_MATCHER))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bi_layout_lr", type=float, default=2e-6)
    parser.add_argument("--opening_lr", type=float, default=1e-5)
    parser.add_argument("--matcher_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--minimum_lr_factor", type=float, default=0.1)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="experimental autocast; stable full-project default is GPU FP32",
    )
    parser.add_argument("--no_amp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--amp_dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
        help="bfloat16 is the stable default for the custom Bi-Layout transformer",
    )
    parser.add_argument("--no_activation_checkpoint", action="store_true")
    parser.add_argument("--no_rotation_aug", action="store_true")

    parser.add_argument("--gt_guidance_epochs", type=int, default=1)
    parser.add_argument("--mix_guidance_epochs", type=int, default=1)
    parser.add_argument("--mix_guidance_weight", type=float, default=0.5)
    parser.add_argument("--opening_loss_weight", type=float, default=1.0)
    parser.add_argument("--feature_anchor_weight", type=float, default=0.02)
    parser.add_argument("--depth_anchor_weight", type=float, default=0.02)
    parser.add_argument("--candidate_loss_weight", type=float, default=1.0)
    parser.add_argument("--token_affinity_weight", type=float, default=1.0)
    parser.add_argument("--shared_response_weight", type=float, default=0.0)
    parser.add_argument("--portal_shift_loss_weight", type=float, default=0.2)
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--opening_pos_weight", type=float, default=2.5)
    parser.add_argument("--opening_tversky_weight", type=float, default=0.5)
    parser.add_argument("--opening_threshold", type=float)
    parser.add_argument("--min_opening_width_tokens", type=int, default=2)
    parser.add_argument("--max_openings_per_view", type=int, default=12)
    parser.add_argument("--metric_top_k", type=int, default=3)

    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--max_train_pairs", type=int)
    parser.add_argument("--max_val_pairs", type=int)
    parser.add_argument("--max_test_pairs", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="run one mixed-precision forward/backward and report gradient flow",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="one tiny epoch; intended for code validation, not model training",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.epochs,
        args.batch_size,
        args.accumulation_steps,
        args.bi_layout_lr,
        args.opening_lr,
        args.matcher_lr,
        args.opening_pos_weight,
        args.min_opening_width_tokens,
        args.max_openings_per_view,
        args.metric_top_k,
        args.image_height,
        args.image_width,
        args.torch_threads,
        args.progress_every,
    )
    if min(positive) <= 0:
        raise ValueError("positive training arguments must be > 0")
    nonnegative = (
        args.early_stop_patience,
        args.workers,
        args.weight_decay,
        args.grad_clip,
        args.gt_guidance_epochs,
        args.mix_guidance_epochs,
        args.opening_loss_weight,
        args.feature_anchor_weight,
        args.depth_anchor_weight,
        args.candidate_loss_weight,
        args.token_affinity_weight,
        args.shared_response_weight,
        args.portal_shift_loss_weight,
        args.consistency_weight,
        args.opening_tversky_weight,
    )
    if min(nonnegative) < 0:
        raise ValueError("non-negative training arguments must be >= 0")
    for name in ("mix_guidance_weight", "minimum_lr_factor"):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError("{} must be in [0, 1]".format(name))
    if args.opening_threshold is not None:
        if not 0.0 <= float(args.opening_threshold) <= 1.0:
            raise ValueError("opening_threshold must be in [0, 1]")
    for value in (
        args.max_train_pairs,
        args.max_val_pairs,
        args.max_test_pairs,
    ):
        if value is not None and value <= 0:
            raise ValueError("max pair limits must be positive")
    if args.image_width % 256 != 0:
        raise ValueError("image_width must be divisible by the 256 layout tokens")


class JointZInDMatchingDataset(Dataset):
    """Add raw panoramas and circular rotation augmentation to cached labels."""

    _SIDE_TOKEN_KEYS = {
        "A": (
            "opening_mask_all_A",
            "portal_mask_A",
            "manifest_candidate_masks_A",
            "feature_A",
            "depth_enclosed_A",
            "depth_extended_A",
        ),
        "B": (
            "opening_mask_all_B",
            "portal_mask_B",
            "manifest_candidate_masks_B",
            "feature_B",
            "depth_enclosed_B",
            "depth_extended_B",
        ),
    }

    def __init__(
        self,
        manifest_path: str,
        cache_dir: str,
        *,
        data_root: Optional[str],
        max_pairs: Optional[int],
        expected_split: str,
        image_shape: Tuple[int, int],
        rotation_augmentation: bool,
    ) -> None:
        self.base = ZInDMatchingFeatureDataset(
            manifest_path,
            cache_dir,
            data_root=data_root,
            max_pairs=max_pairs,
            expected_split=expected_split,
        )
        self.image_shape = tuple(int(value) for value in image_shape)
        self.rotation_augmentation = bool(rotation_augmentation)
        self.split = self.base.split
        self.token_count = self.base.token_count
        self.feature_dim = self.base.feature_dim
        self.house_ids = self.base.house_ids
        self.manifest_path = self.base.manifest_path
        self.cache_dir = self.base.cache_dir
        self.metadata = self.base.metadata

    def __len__(self) -> int:
        return len(self.base)

    def _load_image(self, path: str) -> torch.Tensor:
        height, width = self.image_shape
        resampling = getattr(Image, "Resampling", Image)
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (width, height), resampling.BICUBIC
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        expected = (height, width, 3)
        if tuple(array.shape) != expected or not np.isfinite(array).all():
            raise ValueError("invalid panorama image: {}".format(path))
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    @staticmethod
    def _roll_side(
        sample: Dict[str, Any],
        side: str,
        token_shift: int,
        pixel_shift: int,
    ) -> None:
        sample["image_{}".format(side)] = torch.roll(
            sample["image_{}".format(side)], shifts=int(pixel_shift), dims=-1
        )
        for key in JointZInDMatchingDataset._SIDE_TOKEN_KEYS[side]:
            sample[key] = torch.roll(
                sample[key], shifts=int(token_shift), dims=-1
            )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.base[index]
        sample["image_A"] = self._load_image(sample["image_path_A"])
        sample["image_B"] = self._load_image(sample["image_path_B"])
        if self.rotation_augmentation:
            shift_a = random.randrange(self.token_count)
            shift_b = random.randrange(self.token_count)
            pixels_per_token = self.image_shape[1] // self.token_count
            self._roll_side(
                sample, "A", shift_a, shift_a * pixels_per_token
            )
            self._roll_side(
                sample, "B", shift_b, shift_b * pixels_per_token
            )
            sample["affinity_gt"] = torch.roll(
                sample["affinity_gt"],
                shifts=(shift_a, shift_b),
                dims=(-2, -1),
            )
        return sample


class EndToEndOpeningMatcher(nn.Module):
    """Memory-aware live Bi-Layout -> Opening Head -> Matcher composition."""

    def __init__(
        self,
        bi_layout: nn.Module,
        matcher: OpeningGuidedCrossAttentionMatcher,
        *,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.bi_layout = bi_layout
        self.matcher = matcher
        self.activation_checkpointing = bool(activation_checkpointing)

    def _backbone_tensors(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.bi_layout(images, return_features=True)
        return (
            output["layout_feature"],
            output["depth"],
            output["new_depth"],
        )

    def live_layout(
        self, image_a: torch.Tensor, image_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size = int(image_a.shape[0])
        images = torch.cat((image_a, image_b), dim=0)
        if self.training and self.activation_checkpointing:
            # Re-entrant checkpointing in the project's PyTorch 1.12 requires
            # one grad-requiring input even when all desired gradients are for
            # module parameters.
            images = images.detach().requires_grad_(True)
            feature, depth, new_depth = activation_checkpoint(
                self._backbone_tensors, images
            )
        else:
            feature, depth, new_depth = self._backbone_tensors(images)
        output = {
            "layout_feature_A": feature[:batch_size],
            "layout_feature_B": feature[batch_size:],
            "depth_A": depth[:batch_size],
            "depth_B": depth[batch_size:],
            "new_depth_A": new_depth[:batch_size],
            "new_depth_B": new_depth[batch_size:],
        }
        enclosed_a, extended_a = resolve_enclosed_extended_depth(
            {"depth": output["depth_A"], "new_depth": output["new_depth_A"]},
            "extended_first",
        )
        enclosed_b, extended_b = resolve_enclosed_extended_depth(
            {"depth": output["depth_B"], "new_depth": output["new_depth_B"]},
            "extended_first",
        )
        output.update(
            enclosed_depth_A=enclosed_a,
            extended_depth_A=extended_a,
            enclosed_depth_B=enclosed_b,
            extended_depth_B=extended_b,
        )
        return output

    def forward_matcher(
        self,
        live: Mapping[str, torch.Tensor],
        *,
        candidate_masks_a: torch.Tensor,
        candidate_masks_b: torch.Tensor,
        candidate_valid_a: torch.Tensor,
        candidate_valid_b: torch.Tensor,
        opening_guidance_a: torch.Tensor,
        opening_guidance_b: torch.Tensor,
        opening_guidance_mode: str,
        opening_guidance_mix_weight: float,
    ) -> Mapping[str, torch.Tensor]:
        return self.matcher(
            live["layout_feature_A"],
            live["layout_feature_B"],
            live["enclosed_depth_A"],
            live["extended_depth_A"],
            live["enclosed_depth_B"],
            live["extended_depth_B"],
            candidate_masks_a=candidate_masks_a,
            candidate_masks_b=candidate_masks_b,
            candidate_valid_a=candidate_valid_a,
            candidate_valid_b=candidate_valid_b,
            opening_guidance_a=opening_guidance_a,
            opening_guidance_b=opening_guidance_b,
            opening_guidance_mode=opening_guidance_mode,
            opening_guidance_mix_weight=opening_guidance_mix_weight,
        )


def _make_loader(
    dataset: JointZInDMatchingDataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_zind_bipair,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def _candidate_inputs(
    model: EndToEndOpeningMatcher,
    live: Mapping[str, torch.Tensor],
    raw_batch: Mapping[str, Any],
    pair: Any,
    candidate_source: str,
    opening_threshold: float,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = live["layout_feature_A"].device
    if candidate_source == "gt":
        masks_a = raw_batch["manifest_candidate_masks_A"].to(
            device=device, dtype=torch.bool, non_blocking=True
        )
        masks_b = raw_batch["manifest_candidate_masks_B"].to(
            device=device, dtype=torch.bool, non_blocking=True
        )
        valid_a = raw_batch["manifest_candidate_valid_A"].to(
            device=device, dtype=torch.bool, non_blocking=True
        )
        valid_b = raw_batch["manifest_candidate_valid_B"].to(
            device=device, dtype=torch.bool, non_blocking=True
        )
        target = raw_batch["candidate_target_AB"].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        return masks_a, masks_b, valid_a, valid_b, target
    if candidate_source != "predicted":
        raise ValueError("candidate_source must be gt or predicted")
    with torch.no_grad():
        signal_a = model.matcher.opening_head(
            live["layout_feature_A"].detach(),
            live["enclosed_depth_A"].detach(),
            live["extended_depth_A"].detach(),
        )
        signal_b = model.matcher.opening_head(
            live["layout_feature_B"].detach(),
            live["enclosed_depth_B"].detach(),
            live["extended_depth_B"].detach(),
        )
        masks_a, valid_a = predicted_candidate_masks(
            signal_a["opening_probability"],
            opening_threshold,
            args.min_opening_width_tokens,
            args.max_openings_per_view,
        )
        masks_b, valid_b = predicted_candidate_masks(
            signal_b["opening_probability"],
            opening_threshold,
            args.min_opening_width_tokens,
            args.max_openings_per_view,
        )
        target = candidate_targets_from_affinity(
            masks_a, pair.affinity_ab, masks_b
        )
    return masks_a, masks_b, valid_a, valid_b, target


def _teacher_anchor_losses(
    live: Mapping[str, torch.Tensor],
    raw_batch: Mapping[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_losses = []
    depth_losses = []
    for side in ("A", "B"):
        teacher_feature = raw_batch["feature_{}".format(side)].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        live_feature = live["layout_feature_{}".format(side)].float()
        feature_losses.append(
            (1.0 - F.cosine_similarity(live_feature, teacher_feature, dim=-1)).mean()
        )
        for semantic in ("enclosed", "extended"):
            teacher_depth = raw_batch[
                "depth_{}_{}".format(semantic, side)
            ].to(device=device, dtype=torch.float32, non_blocking=True)
            live_depth = live[
                "{}_depth_{}".format(semantic, side)
            ].float()
            depth_losses.append(F.smooth_l1_loss(live_depth, teacher_depth))
    return torch.stack(feature_losses).mean(), torch.stack(depth_losses).mean()


def _forward_loss(
    model: EndToEndOpeningMatcher,
    raw_batch: Mapping[str, Any],
    device: torch.device,
    *,
    guidance_mode: str,
    candidate_source: str,
    opening_threshold: float,
    args: argparse.Namespace,
) -> Tuple[
    Mapping[str, torch.Tensor],
    Any,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, torch.Tensor],
]:
    pair = adapt_zind_bipair_batch(raw_batch).to(device)
    image_a = raw_batch["image_A"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    image_b = raw_batch["image_B"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    live = model.live_layout(image_a, image_b)
    masks_a, masks_b, valid_a, valid_b, target = _candidate_inputs(
        model,
        live,
        raw_batch,
        pair,
        candidate_source,
        opening_threshold,
        args,
    )
    outputs = model.forward_matcher(
        live,
        candidate_masks_a=masks_a,
        candidate_masks_b=masks_b,
        candidate_valid_a=valid_a,
        candidate_valid_b=valid_b,
        opening_guidance_a=pair.opening_all_a.float(),
        opening_guidance_b=pair.opening_all_b.float(),
        opening_guidance_mode=guidance_mode,
        opening_guidance_mix_weight=args.mix_guidance_weight,
    )
    matching = matcher_losses(outputs, pair, target, valid_a, valid_b, args)
    opening_a = opening_detection_loss(
        outputs["opening_logits_A"].float(),
        pair.opening_all_a.float(),
        pos_weight=args.opening_pos_weight,
        tversky_weight=args.opening_tversky_weight,
    )
    opening_b = opening_detection_loss(
        outputs["opening_logits_B"].float(),
        pair.opening_all_b.float(),
        pos_weight=args.opening_pos_weight,
        tversky_weight=args.opening_tversky_weight,
    )
    opening_total = 0.5 * (
        opening_a["loss_total"] + opening_b["loss_total"]
    )
    feature_anchor, depth_anchor = _teacher_anchor_losses(
        live, raw_batch, device
    )
    total = (
        matching["loss_total"]
        + args.opening_loss_weight * opening_total
        + args.feature_anchor_weight * feature_anchor
        + args.depth_anchor_weight * depth_anchor
    )
    losses = {
        **matching,
        "loss_matching": matching["loss_total"],
        "loss_opening": opening_total,
        "loss_opening_bce": 0.5 * (
            opening_a["loss_bce"] + opening_b["loss_bce"]
        ),
        "loss_opening_tversky": 0.5 * (
            opening_a["loss_tversky"] + opening_b["loss_tversky"]
        ),
        "loss_feature_anchor": feature_anchor,
        "loss_depth_anchor": depth_anchor,
        "loss_total": total,
    }
    return outputs, pair, target, valid_a, valid_b, losses


def _schedule(epoch: int, args: argparse.Namespace) -> Tuple[str, str]:
    mode = guidance_mode_for_epoch(
        epoch, args.gt_guidance_epochs, args.mix_guidance_epochs
    )
    candidate_source = "gt" if mode == "gt" else "predicted"
    return mode, candidate_source


def _opening_counts(
    outputs: Mapping[str, torch.Tensor],
    pair: Any,
    threshold: float,
) -> Tuple[int, int, int]:
    true_positive = false_positive = false_negative = 0
    for probability, target in (
        (outputs["P_A_open"], pair.opening_all_a),
        (outputs["P_B_open"], pair.opening_all_b),
    ):
        prediction = probability.detach() >= float(threshold)
        target = target.bool()
        true_positive += int((prediction & target).sum().item())
        false_positive += int((prediction & ~target).sum().item())
        false_negative += int((~prediction & target).sum().item())
    return true_positive, false_positive, false_negative


def _autocast_dtype(args: argparse.Namespace) -> torch.dtype:
    return (
        torch.bfloat16
        if args.amp_dtype == "bfloat16"
        else torch.float16
    )


def _amp_enabled(args: argparse.Namespace, device: torch.device) -> bool:
    return device.type == "cuda" and args.amp and not args.no_amp


def train_one_epoch(
    model: EndToEndOpeningMatcher,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    opening_threshold: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.train()
    mode, candidate_source = _schedule(epoch, args)
    metrics = MetricAccumulator(args.metric_top_k)
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    opening_tp = opening_fp = opening_fn = 0
    use_amp = _amp_enabled(args, device)
    for step, raw_batch in enumerate(loader, start=1):
        with autocast(enabled=use_amp, dtype=_autocast_dtype(args)):
            outputs, pair, target, valid_a, valid_b, losses = _forward_loss(
                model,
                raw_batch,
                device,
                guidance_mode=mode,
                candidate_source=candidate_source,
                opening_threshold=opening_threshold,
                args=args,
            )
            scaled_loss = losses["loss_total"] / args.accumulation_steps
        scaler.scale(scaled_loss).backward()
        should_step = (
            step % args.accumulation_steps == 0 or step == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        metrics.update(outputs, pair, target, valid_a, valid_b, losses)
        counts = _opening_counts(outputs, pair, opening_threshold)
        opening_tp += counts[0]
        opening_fp += counts[1]
        opening_fn += counts[2]
        if step % args.progress_every == 0 or step == len(loader):
            elapsed = time.perf_counter() - start
            average_loss = metrics.loss_sums.get("loss_total", 0.0)
            average_loss /= max(metrics.samples, 1)
            rate = metrics.samples / max(elapsed, 1e-6)
            print(
                "  train {}/{} pairs={} loss={:.4f} rate={:.2f} pair/s "
                "elapsed={:.1f}s".format(
                    step,
                    len(loader),
                    metrics.samples,
                    average_loss,
                    rate,
                    elapsed,
                ),
                flush=True,
            )
    result: Dict[str, Any] = metrics.result()
    precision = opening_tp / max(opening_tp + opening_fp, 1)
    recall = opening_tp / max(opening_tp + opening_fn, 1)
    result.update(
        opening_token_precision=precision,
        opening_token_recall=recall,
        opening_token_f1=(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        ),
        guidance_mode=mode,
        candidate_source=candidate_source,
        runtime_seconds=time.perf_counter() - start,
    )
    return result


@torch.no_grad()
def evaluate(
    model: EndToEndOpeningMatcher,
    loader: DataLoader,
    device: torch.device,
    *,
    candidate_source: str,
    opening_threshold: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.eval()
    metrics = MetricAccumulator(args.metric_top_k)
    opening_tp = opening_fp = opening_fn = 0
    start = time.perf_counter()
    guidance_mode = "gt" if candidate_source == "gt" else "predicted"
    use_amp = _amp_enabled(args, device)
    for raw_batch in loader:
        with autocast(enabled=use_amp, dtype=_autocast_dtype(args)):
            outputs, pair, target, valid_a, valid_b, losses = _forward_loss(
                model,
                raw_batch,
                device,
                guidance_mode=guidance_mode,
                candidate_source=candidate_source,
                opening_threshold=opening_threshold,
                args=args,
            )
        metrics.update(outputs, pair, target, valid_a, valid_b, losses)
        counts = _opening_counts(outputs, pair, opening_threshold)
        opening_tp += counts[0]
        opening_fp += counts[1]
        opening_fn += counts[2]
    result: Dict[str, Any] = metrics.result()
    precision = opening_tp / max(opening_tp + opening_fp, 1)
    recall = opening_tp / max(opening_tp + opening_fn, 1)
    result.update(
        opening_token_precision=precision,
        opening_token_recall=recall,
        opening_token_f1=(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        ),
        guidance_mode=guidance_mode,
        candidate_source=candidate_source,
        runtime_seconds=time.perf_counter() - start,
    )
    return result


def _gradient_norm(module: nn.Module) -> float:
    square_sum = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            value = float(parameter.grad.detach().float().norm().item())
            square_sum += value * value
    return math.sqrt(square_sum)


def run_preflight(
    model: EndToEndOpeningMatcher,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    opening_threshold: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    raw_batch = next(iter(loader))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    use_amp = _amp_enabled(args, device)
    with autocast(enabled=use_amp, dtype=_autocast_dtype(args)):
        outputs, pair, target, valid_a, valid_b, losses = _forward_loss(
            model,
            raw_batch,
            device,
            guidance_mode="predicted",
            candidate_source="predicted",
            opening_threshold=opening_threshold,
            args=args,
        )
    scaler.scale(losses["loss_total"]).backward()
    scaler.unscale_(optimizer)
    matcher_without_opening = nn.ModuleList(
        [
            module
            for name, module in model.matcher.named_children()
            if name != "opening_head"
        ]
    )
    gradients = {
        "bi_layout_feature_extractor": _gradient_norm(
            model.bi_layout.feature_extractor
        ),
        "bi_layout_transformer": _gradient_norm(model.bi_layout.transformer),
        "bi_layout_depth_heads": math.sqrt(
            _gradient_norm(model.bi_layout.linear_depth_output) ** 2
            + _gradient_norm(model.bi_layout.linear_depth_output_2) ** 2
        ),
        "opening_head": _gradient_norm(model.matcher.opening_head),
        "matcher": _gradient_norm(matcher_without_opening),
    }
    missing = [name for name, value in gradients.items() if not value > 0.0]
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    result = {
        "loss_total": float(losses["loss_total"].detach().item()),
        "gradient_norms": gradients,
        "peak_memory_bytes": peak_memory,
        "peak_memory_gib": peak_memory / (1024.0 ** 3),
        "runtime_seconds": time.perf_counter() - start,
        "amp": use_amp,
        "activation_checkpointing": model.activation_checkpointing,
        "batch_size": int(pair.batch_size),
        "candidate_source": "predicted",
    }
    if args.preflight_only:
        # The first AdamW step materializes its moment tensors.  Include that
        # allocation in standalone preflight memory reporting.  The process
        # exits immediately afterwards, so this diagnostic update can never
        # leak into a real training run.
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            result["peak_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            result["peak_memory_gib"] = result["peak_memory_bytes"] / (
                1024.0 ** 3
            )
    else:
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if missing:
        raise RuntimeError(
            "end-to-end gradient did not reach: {}".format(", ".join(missing))
        )
    return result


def _optimizer(
    model: EndToEndOpeningMatcher, args: argparse.Namespace
) -> torch.optim.Optimizer:
    opening_parameters = list(model.matcher.opening_head.parameters())
    opening_ids = {id(parameter) for parameter in opening_parameters}
    matcher_parameters = [
        parameter
        for parameter in model.matcher.parameters()
        if id(parameter) not in opening_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "name": "bi_layout",
                "params": list(model.bi_layout.parameters()),
                "lr": args.bi_layout_lr,
            },
            {
                "name": "opening_head",
                "params": opening_parameters,
                "lr": args.opening_lr,
            },
            {
                "name": "matcher",
                "params": matcher_parameters,
                "lr": args.matcher_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    minimum_factor: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(epoch: int) -> float:
        progress = min(max(epoch, 0), epochs) / max(epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(minimum_factor) + (1.0 - float(minimum_factor)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, Mapping):
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if torch.is_tensor(state.get("torch")):
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _model_contract(
    model: EndToEndOpeningMatcher, args: argparse.Namespace
) -> Dict[str, Any]:
    return {
        "composition": "live Bi-Layout -> OpeningSignalHead -> CrossAttentionMatcher",
        "feature_dim": int(model.bi_layout.patch_dim),
        "token_count": int(model.bi_layout.patch_num),
        "depth_branch_order": "extended_first",
        "bi_layout_output_number": int(model.bi_layout.output_number),
        "matcher_heads": int(model.matcher.heads),
        "activation_checkpointing": bool(model.activation_checkpointing),
        "amp": bool(args.amp and not args.no_amp),
        "amp_dtype": args.amp_dtype,
    }


def _training_contract(args: argparse.Namespace) -> Dict[str, Any]:
    keys = (
        "epochs",
        "batch_size",
        "accumulation_steps",
        "bi_layout_lr",
        "opening_lr",
        "matcher_lr",
        "weight_decay",
        "grad_clip",
        "minimum_lr_factor",
        "gt_guidance_epochs",
        "mix_guidance_epochs",
        "mix_guidance_weight",
        "opening_loss_weight",
        "feature_anchor_weight",
        "depth_anchor_weight",
        "candidate_loss_weight",
        "token_affinity_weight",
        "shared_response_weight",
        "portal_shift_loss_weight",
        "consistency_weight",
        "opening_pos_weight",
        "opening_tversky_weight",
        "min_opening_width_tokens",
        "max_openings_per_view",
        "metric_top_k",
        "image_height",
        "image_width",
        "max_train_pairs",
        "max_val_pairs",
        "max_test_pairs",
        "seed",
    )
    return {
        **{key: getattr(args, key) for key in keys},
        "rotation_augmentation": not args.no_rotation_aug,
        "candidate_schedule": "GT in GT-guidance epoch, predicted thereafter",
        "teacher_cache_role": "regularization only; never live feature replacement",
    }


def _checkpoint_payload(
    model: EndToEndOpeningMatcher,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    *,
    epoch: int,
    best: Mapping[str, Any],
    stale_epochs: int,
    train_metrics: Mapping[str, Any],
    validation_gt: Mapping[str, Any],
    validation_predicted: Mapping[str, Any],
    opening_threshold: float,
    model_contract: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    data_contract: Mapping[str, Any],
    initialization: Mapping[str, Any],
    runtime_seconds: float,
    include_optimizer: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "format_version": 1,
        "task": "ZInD_matching full end-to-end joint fine-tuning",
        "learned_chain": ["Bi-Layout", "OpeningSignalHead", "CrossSceneMatcher"],
        "analytical_evaluation_chain": [
            "PairGeometryRefiner",
            "RelativePose",
            "GeometryConsistencySelector",
            "JointLayoutFusion",
        ],
        "completed_epoch": int(epoch),
        "next_epoch": int(epoch + 1),
        "bi_layout_state_dict": model.bi_layout.state_dict(),
        "matcher_state_dict": model.matcher.state_dict(),
        "opening_probability_threshold": float(opening_threshold),
        "model_contract": dict(model_contract),
        "training_contract": dict(training_contract),
        "data_contract": dict(data_contract),
        "initialization": dict(initialization),
        "best": dict(best),
        "stale_epochs": int(stale_epochs),
        "cumulative_runtime_seconds": float(runtime_seconds),
        "train": dict(train_metrics),
        "validation": {
            "teacher_forced_manifest_candidates": dict(validation_gt),
            "predicted_openings": dict(validation_predicted),
        },
        "rng_state": _rng_state(),
    }
    if include_optimizer:
        payload.update(
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            scaler_state_dict=scaler.state_dict(),
        )
    return payload


def _best_key(metrics: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(metrics["harmonic_flow_score"]),
        float(metrics["balanced_flow_score"]),
        float(metrics["end_to_end_candidate_top1"]),
        float(metrics["opening_token_f1"]),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        args.epochs = 1
        args.early_stop_patience = 0
        args.max_train_pairs = args.max_train_pairs or 4
        args.max_val_pairs = args.max_val_pairs or 2
        args.max_test_pairs = args.max_test_pairs or 2
        args.workers = 0
        args.progress_every = 1
        args.accumulation_steps = 1
    validate_args(args)
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("full end-to-end training requires a CUDA GPU")
    print("device: {}".format(device), flush=True)

    manifest_root = Path(args.manifest_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    manifests = {
        split: (
            Path(getattr(args, "{}_manifest".format(split))).expanduser().resolve()
            if getattr(args, "{}_manifest".format(split))
            else manifest_root / "{}.json".format(split)
        )
        for split in ("train", "val", "test")
    }
    caches = {
        split: (
            Path(getattr(args, "{}_cache_dir".format(split))).expanduser().resolve()
            if getattr(args, "{}_cache_dir".format(split))
            else cache_root / split
        )
        for split in ("train", "val", "test")
    }
    config_path = Path(args.bi_layout_config).expanduser().resolve()
    bi_layout_path = Path(args.bi_layout_checkpoint).expanduser().resolve()
    opening_path = Path(args.opening_checkpoint).expanduser().resolve()
    matcher_path = Path(args.matcher_checkpoint).expanduser().resolve()
    required = (
        list(manifests.values())
        + [cache / "metadata.json" for cache in caches.values()]
        + [config_path, bi_layout_path, opening_path, matcher_path]
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    limits = {
        "train": args.max_train_pairs,
        "val": args.max_val_pairs,
        "test": args.max_test_pairs,
    }
    datasets: Dict[str, JointZInDMatchingDataset] = {}
    for split in ("train", "val", "test"):
        datasets[split] = JointZInDMatchingDataset(
            str(manifests[split]),
            str(caches[split]),
            data_root=args.data_root,
            max_pairs=limits[split],
            expected_split=split,
            image_shape=(args.image_height, args.image_width),
            rotation_augmentation=(
                split == "train" and not args.no_rotation_aug
            ),
        )
    house_counts = validate_disjoint_house_splits(
        {split: dataset.base for split, dataset in datasets.items()}
    )
    dimensions = {
        (dataset.token_count, dataset.feature_dim)
        for dataset in datasets.values()
    }
    if len(dimensions) != 1:
        raise ValueError("train/val/test dimensions differ")
    token_count, feature_dim = next(iter(dimensions))
    print(
        "pairs train/val/test={}/{}/{} houses={} tensor=[{}, {}]".format(
            len(datasets["train"]),
            len(datasets["val"]),
            len(datasets["test"]),
            house_counts,
            token_count,
            feature_dim,
        ),
        flush=True,
    )

    bi_layout, bi_layout_report = load_bi_layout(
        str(config_path), str(bi_layout_path), device, True
    )
    if bi_layout.output_number != 2:
        raise ValueError("end-to-end matching requires Bi-Layout output_number=2")
    if int(bi_layout.patch_dim) != feature_dim:
        raise ValueError("Bi-Layout patch_dim does not match matching cache")
    matcher, matcher_report = load_cross_scene_matcher_checkpoint(
        str(matcher_path),
        expected_feature_dim=feature_dim,
        expected_token_count=token_count,
        expected_branch_order="extended_first",
        bi_layout_config_path=str(config_path),
        bi_layout_checkpoint_path=str(bi_layout_path),
        device=device,
    )
    opening_report = load_opening_head_checkpoint(
        matcher.opening_head,
        str(opening_path),
        expected_feature_dim=feature_dim,
        expected_branch_order="extended_first",
        expected_token_count=token_count,
        bi_layout_config_path=str(config_path),
        bi_layout_checkpoint_path=str(bi_layout_path),
    )
    threshold_report = resolve_opening_probability_threshold(
        args.opening_threshold, opening_report, matcher_report
    )
    opening_threshold = float(threshold_report["value"])
    model = EndToEndOpeningMatcher(
        bi_layout,
        matcher,
        activation_checkpointing=not args.no_activation_checkpoint,
    ).to(device)
    model.requires_grad_(True)
    optimizer = _optimizer(model, args)
    scheduler = _scheduler(
        optimizer, args.epochs, args.minimum_lr_factor
    )
    scaler = GradScaler(
        enabled=args.amp and not args.no_amp and args.amp_dtype == "float16"
    )
    parameter_counts = {
        "bi_layout": sum(parameter.numel() for parameter in model.bi_layout.parameters()),
        "opening_head": sum(
            parameter.numel() for parameter in model.matcher.opening_head.parameters()
        ),
        "matcher_total": sum(
            parameter.numel() for parameter in model.matcher.parameters()
        ),
    }
    print(
        "trainable parameters: Bi-Layout={:,}, Opening={:,}, Matcher={:,}; "
        "threshold={:.4f} ({})".format(
            parameter_counts["bi_layout"],
            parameter_counts["opening_head"],
            parameter_counts["matcher_total"],
            opening_threshold,
            threshold_report["source"],
        ),
        flush=True,
    )

    loaders = {
        split: _make_loader(
            datasets[split],
            args,
            shuffle=False,
            seed=args.seed,
            device=device,
        )
        for split in ("val", "test")
    }
    preflight_loader = _make_loader(
        datasets["train"],
        args,
        shuffle=False,
        seed=args.seed,
        device=device,
    )
    preflight = run_preflight(
        model,
        preflight_loader,
        optimizer,
        scaler,
        device,
        opening_threshold,
        args,
    )
    print(
        "preflight PASS loss={:.4f} peak={:.2f} GiB time={:.1f}s gradients={}".format(
            preflight["loss_total"],
            preflight["peak_memory_gib"],
            preflight["runtime_seconds"],
            {
                key: round(value, 6)
                for key, value in preflight["gradient_norms"].items()
            },
        ),
        flush=True,
    )
    if args.preflight_only:
        return 0

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_contract = _model_contract(model, args)
    training_contract = _training_contract(args)
    data_contract = {
        "dataset": "ZInD_matching formatVersion=2 complete-room pairs",
        "manifests": {
            split: str(manifests[split]) for split in ("train", "val", "test")
        },
        "feature_caches": {
            split: str(caches[split]) for split in ("train", "val", "test")
        },
        "pair_counts": {
            split: len(datasets[split]) for split in ("train", "val", "test")
        },
        "house_counts": dict(house_counts),
        "token_count": int(token_count),
        "feature_dim": int(feature_dim),
        "raw_image_live_forward": True,
        "uses_ground_truth_depth_as_model_input": False,
    }
    initialization = {
        "bi_layout": dict(bi_layout_report),
        "opening_head": dict(opening_report),
        "matcher": dict(matcher_report),
        "opening_threshold": dict(threshold_report),
        "parameter_counts": parameter_counts,
        "preflight": preflight,
    }

    start_epoch = 0
    cumulative_runtime = 0.0
    stale_epochs = 0
    best: Dict[str, Any] = {
        "epoch": -1,
        "harmonic_flow_score": -1.0,
        "balanced_flow_score": -1.0,
        "end_to_end_candidate_top1": -1.0,
        "opening_token_f1": -1.0,
    }
    if args.resume:
        resume_path = (
            output_dir / "last.pt"
            if args.resume == "auto"
            else Path(args.resume).expanduser().resolve()
        )
        checkpoint = torch.load(str(resume_path), map_location="cpu")
        if checkpoint.get("task") != "ZInD_matching full end-to-end joint fine-tuning":
            raise ValueError("resume checkpoint is not an end-to-end checkpoint")
        if checkpoint.get("model_contract") != model_contract:
            raise ValueError("resume model contract changed")
        if checkpoint.get("training_contract") != training_contract:
            raise ValueError("resume training contract changed")
        if checkpoint.get("data_contract") != data_contract:
            raise ValueError("resume data contract changed")
        model.bi_layout.load_state_dict(
            checkpoint["bi_layout_state_dict"], strict=True
        )
        model.matcher.load_state_dict(
            checkpoint["matcher_state_dict"], strict=True
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["next_epoch"])
        cumulative_runtime = float(
            checkpoint.get("cumulative_runtime_seconds", 0.0)
        )
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        best = dict(checkpoint.get("best", best))
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
        print(
            "epoch {}/{} lr={}".format(
                epoch + 1,
                args.epochs,
                {
                    group.get("name", str(index)): group["lr"]
                    for index, group in enumerate(optimizer.param_groups)
                },
            ),
            flush=True,
        )
        train_loader = _make_loader(
            datasets["train"],
            args,
            shuffle=True,
            seed=args.seed + epoch,
            device=device,
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            opening_threshold,
            args,
        )
        validation_gt = evaluate(
            model,
            loaders["val"],
            device,
            candidate_source="gt",
            opening_threshold=opening_threshold,
            args=args,
        )
        validation_predicted = evaluate(
            model,
            loaders["val"],
            device,
            candidate_source="predicted",
            opening_threshold=opening_threshold,
            args=args,
        )
        scheduler.step()
        epoch_runtime = time.perf_counter() - epoch_start
        cumulative_runtime += epoch_runtime
        improved = _best_key(validation_predicted) > (
            float(best["harmonic_flow_score"]),
            float(best["balanced_flow_score"]),
            float(best["end_to_end_candidate_top1"]),
            float(best["opening_token_f1"]),
        )
        if improved:
            stale_epochs = 0
            best = {
                "epoch": int(epoch),
                "harmonic_flow_score": float(
                    validation_predicted["harmonic_flow_score"]
                ),
                "balanced_flow_score": float(
                    validation_predicted["balanced_flow_score"]
                ),
                "end_to_end_candidate_top1": float(
                    validation_predicted["end_to_end_candidate_top1"]
                ),
                "opening_token_f1": float(
                    validation_predicted["opening_token_f1"]
                ),
            }
        else:
            stale_epochs += 1
        report = {
            "epoch": int(epoch),
            "epoch_runtime_seconds": epoch_runtime,
            "cumulative_runtime_seconds": cumulative_runtime,
            "learning_rates": {
                group.get("name", str(index)): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "train": train_metrics,
            "validation": {
                "teacher_forced_manifest_candidates": validation_gt,
                "predicted_openings": validation_predicted,
            },
            "opening_threshold": opening_threshold,
            "improved": improved,
            "stale_epochs": stale_epochs,
            "best": best,
        }
        last_payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            best=best,
            stale_epochs=stale_epochs,
            train_metrics=train_metrics,
            validation_gt=validation_gt,
            validation_predicted=validation_predicted,
            opening_threshold=opening_threshold,
            model_contract=model_contract,
            training_contract=training_contract,
            data_contract=data_contract,
            initialization=initialization,
            runtime_seconds=cumulative_runtime,
            include_optimizer=True,
        )
        _atomic_torch_save(output_dir / "last.pt", last_payload)
        if improved:
            best_payload = _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                best=best,
                stale_epochs=stale_epochs,
                train_metrics=train_metrics,
                validation_gt=validation_gt,
                validation_predicted=validation_predicted,
                opening_threshold=opening_threshold,
                model_contract=model_contract,
                training_contract=training_contract,
                data_contract=data_contract,
                initialization=initialization,
                runtime_seconds=cumulative_runtime,
                include_optimizer=False,
            )
            _atomic_torch_save(output_dir / "best.pt", best_payload)
            _atomic_json(output_dir / "best_metrics.json", report)
        _append_jsonl(output_dir / "metrics.jsonl", report)
        _atomic_json(output_dir / "latest_metrics.json", report)
        print(
            "  val predicted opening_f1={:.4f} recall={:.4f} top1={:.4f} "
            "negative_reject={:.4f} harmonic={:.4f} balanced={:.4f} "
            "epoch={:.1f}s{}".format(
                validation_predicted["opening_token_f1"],
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
        del last_payload
        if (
            args.early_stop_patience > 0
            and stale_epochs >= args.early_stop_patience
        ):
            print(
                "early stopping after {} non-improving epochs".format(
                    stale_epochs
                ),
                flush=True,
            )
            break

    if not args.skip_test:
        best_path = output_dir / "best.pt"
        checkpoint = torch.load(str(best_path), map_location="cpu")
        model.bi_layout.load_state_dict(
            checkpoint["bi_layout_state_dict"], strict=True
        )
        model.matcher.load_state_dict(
            checkpoint["matcher_state_dict"], strict=True
        )
        test_gt = evaluate(
            model,
            loaders["test"],
            device,
            candidate_source="gt",
            opening_threshold=opening_threshold,
            args=args,
        )
        test_predicted = evaluate(
            model,
            loaders["test"],
            device,
            candidate_source="predicted",
            opening_threshold=opening_threshold,
            args=args,
        )
        test_report = {
            "checkpoint": str(best_path),
            "best_epoch": int(checkpoint["completed_epoch"]),
            "opening_threshold": opening_threshold,
            "teacher_forced_manifest_candidates": test_gt,
            "predicted_openings": test_predicted,
        }
        _atomic_json(output_dir / "test_metrics.json", test_report)
        print(
            "test predicted opening_f1={:.4f} recall={:.4f} top1={:.4f} "
            "negative_reject={:.4f} harmonic={:.4f} balanced={:.4f}".format(
                test_predicted["opening_token_f1"],
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
