#!/usr/bin/env python3
"""Run validation-only ZInD candidate compatibility and recall diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portallayout.datasets.zind_bipair_adapter import adapt_zind_bipair_batch
from portallayout.datasets.zind_bipair_dataset import collate_zind_bipair
from portallayout.models.cross_scene_matcher import (
    bidirectional_candidate_consistency_loss,
    candidate_assignment_loss,
    candidate_intervals_to_mask,
    opening_detection_loss,
    opening_probabilities_to_intervals,
)
from tools.debug_cross_scene_flow import load_bi_layout
from tools.train_zind_cross_scene_matcher import (
    MetricAccumulator,
    _losses as matcher_losses,
    candidate_targets_from_affinity,
    predicted_candidate_masks,
)
from tools.train_zind_end_to_end import (
    DEFAULT_BI_LAYOUT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST_ROOT,
    EndToEndOpeningMatcher,
    JointZInDMatchingDataset,
    _candidate_inputs,
    _candidate_supervision_validity,
    _forward_loss,
    _gradient_norm,
    _teacher_anchor_losses,
    parse_args as parse_training_args,
)
from portallayout.utils.matcher_checkpoint import load_cross_scene_matcher_checkpoint


DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/End_To_End/zind_matching_v1/best.pt"
)
DEFAULT_OUTPUT = REPO_ROOT / "experiments/zind_e2e_v2/preflight"
TARGET_IOU = 0.30
ATOL = 1e-6
RTOL = 1e-5


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_val_pairs", type=int)
    parser.add_argument(
        "--thresholds",
        default="0.30,0.35,0.40,0.47,0.55",
        help="finite comma-separated threshold scan",
    )
    parser.add_argument(
        "--min_widths",
        default="1,2",
        help="finite comma-separated existing min-component widths",
    )
    parser.add_argument("--compatibility_only", action="store_true")
    parser.add_argument("--skip_compatibility", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git",) + args, cwd=str(REPO_ROOT), text=True
    ).strip()


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _parse_number_list(value: str, cast) -> List[Any]:
    parsed = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("scan list must not be empty")
    return parsed


def _checkpoint_summary(checkpoint: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    training = dict(checkpoint["training_contract"])
    data = dict(checkpoint["data_contract"])
    initialization = checkpoint["initialization"]
    return {
        "selected_checkpoint": str(path),
        "selection_evidence": {
            "task": checkpoint.get("task"),
            "completed_epoch_zero_based": checkpoint.get("completed_epoch"),
            "best": checkpoint.get("best"),
            "not_selected_last_checkpoint_reason": (
                "last.pt completed epoch 14 after six stale epochs; embedded best is epoch 8"
            ),
        },
        "model_contract": checkpoint["model_contract"],
        "training_contract": training,
        "data_contract": {
            "validation_manifest": data["manifests"]["val"],
            "validation_cache": data["feature_caches"]["val"],
            "validation_pair_count": data["pair_counts"]["val"],
            "validation_house_count": data["house_counts"]["val"],
            "token_count": data["token_count"],
            "feature_dim": data["feature_dim"],
        },
        "initialization": {
            "bi_layout": initialization["bi_layout"],
            "opening_head": initialization["opening_head"],
            "matcher": initialization["matcher"],
        },
        "g0": {
            "opening_threshold": checkpoint["opening_probability_threshold"],
            "min_opening_width_tokens": training["min_opening_width_tokens"],
            "max_candidates": training["max_openings_per_view"],
            "candidate_target_iou_threshold": TARGET_IOU,
            "gap_connection_parameter": None,
            "gap_connection_note": "the existing component generator has no gap-connection option",
        },
        "historical_teacher_forcing": {
            "gt_guidance_epochs": training["gt_guidance_epochs"],
            "mix_guidance_epochs": training["mix_guidance_epochs"],
            "mix_guidance_weight": training["mix_guidance_weight"],
            "candidate_schedule": training["candidate_schedule"],
            "interpretation": (
                "epoch 0 used GT guidance and GT candidates; epoch 1 used mixed "
                "guidance with predicted candidates; later epochs used predicted/predicted"
            ),
        },
    }


def load_historical_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[EndToEndOpeningMatcher, Dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("task") != "ZInD_matching full end-to-end joint fine-tuning":
        raise ValueError("checkpoint is not the historical full end-to-end checkpoint")
    initialization = checkpoint["initialization"]
    config_path = initialization["bi_layout"]['portallayout.config']
    backbone_path = initialization["bi_layout"]["checkpoint"]
    matcher_path = initialization["matcher"]["path"]
    bi_layout, _ = load_bi_layout(config_path, backbone_path, device, True)
    matcher, _ = load_cross_scene_matcher_checkpoint(
        matcher_path,
        expected_feature_dim=int(checkpoint["model_contract"]["feature_dim"]),
        expected_token_count=int(checkpoint["model_contract"]["token_count"]),
        expected_branch_order=checkpoint["model_contract"]["depth_branch_order"],
        bi_layout_config_path=config_path,
        bi_layout_checkpoint_path=backbone_path,
        device=device,
    )
    model = EndToEndOpeningMatcher(
        bi_layout, matcher, activation_checkpointing=False
    ).to(device)
    model.bi_layout.load_state_dict(checkpoint["bi_layout_state_dict"], strict=True)
    model.matcher.load_state_dict(checkpoint["matcher_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def validation_dataset(
    checkpoint: Mapping[str, Any], max_pairs: Optional[int]
) -> JointZInDMatchingDataset:
    data = checkpoint["data_contract"]
    training = checkpoint["training_contract"]
    return JointZInDMatchingDataset(
        data["manifests"]["val"],
        data["feature_caches"]["val"],
        data_root=None,
        max_pairs=max_pairs,
        expected_split="val",
        image_shape=(training["image_height"], training["image_width"]),
        rotation_augmentation=False,
    )


def _loader(
    dataset: JointZInDMatchingDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_zind_bipair,
        drop_last=False,
    )


def _images(raw_batch: Mapping[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        raw_batch["image_A"].to(device=device, dtype=torch.float32, non_blocking=True),
        raw_batch["image_B"].to(device=device, dtype=torch.float32, non_blocking=True),
    )


def _predicted_probabilities(
    model: EndToEndOpeningMatcher, live: Mapping[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    signal_a = model.matcher.opening_head(
        live["layout_feature_A"],
        live["enclosed_depth_A"],
        live["extended_depth_A"],
    )
    signal_b = model.matcher.opening_head(
        live["layout_feature_B"],
        live["enclosed_depth_B"],
        live["extended_depth_B"],
    )
    return signal_a["opening_probability"], signal_b["opening_probability"]


def _reference_candidate_outputs(
    outputs: Mapping[str, torch.Tensor],
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    opening_a: torch.Tensor,
    opening_b: torch.Tensor,
    dustbin_score: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    logits = outputs["candidate_logits"]
    valid_a = outputs["candidate_valid_A"]
    valid_b = outputs["candidate_valid_B"]
    batch, count_a, count_b = logits.shape
    pair_valid = valid_a.unsqueeze(-1) & valid_b.unsqueeze(1)
    neg_inf = torch.finfo(logits.dtype).min
    assignment_logits = logits.new_full((batch, count_a + 1, count_b + 1), neg_inf)
    assignment_logits[:, :count_a, :count_b] = logits.masked_fill(~pair_valid, neg_inf)
    assignment_logits[:, :count_a, count_b] = torch.where(
        valid_a, dustbin_score.to(logits), logits.new_full((), neg_inf)
    )
    assignment_logits[:, count_a, :count_b] = torch.where(
        valid_b, dustbin_score.to(logits), logits.new_full((), neg_inf)
    )
    assignment_logits[:, count_a, count_b] = dustbin_score.to(logits)
    valid_rows_ab = torch.cat(
        (valid_a, torch.ones((batch, 1), dtype=torch.bool, device=logits.device)), -1
    )
    valid_rows_ba = torch.cat(
        (valid_b, torch.ones((batch, 1), dtype=torch.bool, device=logits.device)), -1
    )
    assignment_ab = torch.softmax(assignment_logits, -1)
    assignment_ab = assignment_ab * valid_rows_ab.unsqueeze(-1).to(logits.dtype)
    assignment_ba = torch.softmax(assignment_logits.transpose(-2, -1), -1)
    assignment_ba = assignment_ba * valid_rows_ba.unsqueeze(-1).to(logits.dtype)
    affinity_ab = assignment_ab[:, :count_a, :count_b]
    affinity_ba = assignment_ba[:, :count_b, :count_a]
    mutual = torch.sqrt((affinity_ab * affinity_ba.transpose(-2, -1)).clamp_min(0))
    mask_a = masks_a.to(opening_a.dtype)
    mask_b = masks_b.to(opening_b.dtype)
    confidence_a = (mask_a * opening_a.unsqueeze(1)).sum(-1) / mask_a.sum(-1).clamp_min(1)
    confidence_b = (mask_b * opening_b.unsqueeze(1)).sum(-1) / mask_b.sum(-1).clamp_min(1)
    confidence = torch.sqrt(confidence_a.unsqueeze(-1) * confidence_b.unsqueeze(1))
    score = mutual * confidence * pair_valid.to(logits.dtype)
    no_match_a = assignment_ab[:, :count_a, count_b]
    no_match_b = assignment_ba[:, :count_b, count_a]
    accepted = (
        pair_valid
        & (affinity_ab > no_match_a.unsqueeze(-1))
        & (affinity_ba.transpose(-2, -1) > no_match_b.unsqueeze(1))
    )
    best = torch.full((batch, 2), -1, dtype=torch.long, device=logits.device)
    if count_a and count_b:
        accepted_score = score.masked_fill(~accepted, -1.0)
        flat = accepted_score.flatten(1).argmax(-1)
        proposed = torch.stack(
            (torch.div(flat, count_b, rounding_mode="floor"), flat.remainder(count_b)), -1
        )
        best = torch.where(accepted.flatten(1).any(-1).unsqueeze(-1), proposed, best)
    return {
        "candidate_assignment_logits": assignment_logits,
        "candidate_assignment_AB": assignment_ab,
        "candidate_assignment_BA": assignment_ba,
        "mutual_affinity": mutual,
        "opening_confidence": confidence,
        "candidate_pair_score": score,
        "best_candidate_pair": best,
    }


def _old_candidate_assignment_loss(
    outputs: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    is_match: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
    consistency_weight: float,
) -> Dict[str, torch.Tensor]:
    assignment_ab = outputs["candidate_assignment_AB"]
    assignment_ba = outputs["candidate_assignment_BA"]
    is_match = is_match.bool().reshape(-1)
    pair_valid = valid_a.unsqueeze(-1) & valid_b.unsqueeze(1)
    target = target.to(assignment_ab) * pair_valid.to(assignment_ab.dtype)
    target = target * is_match[:, None, None].to(target.dtype)

    def directional(probability, real_target, source_valid):
        real_count = real_target.shape[-1]
        row_mass = real_target.sum(-1, keepdim=True)
        normalized = real_target / row_mass.clamp_min(1e-6)
        dustbin = (row_mass <= 0).to(real_target.dtype)
        full_target = torch.cat((normalized, dustbin), -1)
        row_loss = -(
            full_target * probability[..., : real_count + 1].clamp_min(1e-8).log()
        ).sum(-1)
        return row_loss[source_valid].mean() if bool(source_valid.any()) else row_loss.new_zeros(())

    loss_ab = directional(assignment_ab[:, :-1, :], target, valid_a)
    loss_ba = directional(assignment_ba[:, :-1, :], target.transpose(-2, -1), valid_b)
    assignment = 0.5 * (loss_ab + loss_ba)
    consistency = bidirectional_candidate_consistency_loss(
        outputs, candidate_valid_a=valid_a, candidate_valid_b=valid_b
    )
    total = assignment + float(consistency_weight) * consistency
    return {
        "loss_candidate_total": total,
        "loss_candidate_assignment": assignment,
        "loss_candidate_assignment_ab": loss_ab,
        "loss_candidate_assignment_ba": loss_ba,
        "loss_bidirectional_consistency": consistency,
    }


def _tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, Any]:
    if actual.dtype == torch.bool or actual.dtype == torch.long:
        equal = torch.equal(actual, expected)
        return {"equal": equal, "max_abs_error": 0.0 if equal else None}
    difference = (actual.float() - expected.float()).abs()
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    close = bool(torch.allclose(actual, expected, atol=ATOL, rtol=RTOL))
    return {"equal": close, "max_abs_error": maximum}


def _training_args(checkpoint: Mapping[str, Any]) -> argparse.Namespace:
    args = parse_training_args([])
    for key, value in checkpoint["training_contract"].items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.amp = False
    args.no_amp = False
    args.opening_guidance_source = "predicted"
    args.candidate_source = "predicted"
    args.candidate_missing_target_policy = "legacy_dustbin"
    args.candidate_rank_loss_weight = 0.0
    args.no_rotation_aug = True
    return args


def run_compatibility(
    model: EndToEndOpeningMatcher,
    checkpoint: Mapping[str, Any],
    dataset: JointZInDMatchingDataset,
    device: torch.device,
) -> Dict[str, Any]:
    args = _training_args(checkpoint)
    raw_batch = collate_zind_bipair([dataset[index] for index in range(min(4, len(dataset)))])
    pair = adapt_zind_bipair_batch(raw_batch).to(device)
    image_a, image_b = _images(raw_batch, device)
    model.eval()
    with torch.no_grad():
        live = model.live_layout(image_a, image_b)
        probability_a, probability_b = _predicted_probabilities(model, live)
        masks_a, valid_a = predicted_candidate_masks(
            probability_a,
            checkpoint["opening_probability_threshold"],
            args.min_opening_width_tokens,
            args.max_openings_per_view,
        )
        masks_b, valid_b = predicted_candidate_masks(
            probability_b,
            checkpoint["opening_probability_threshold"],
            args.min_opening_width_tokens,
            args.max_openings_per_view,
        )
        target = candidate_targets_from_affinity(masks_a, pair.affinity_ab, masks_b)
        outputs = model.forward_matcher(
            live,
            candidate_masks_a=masks_a,
            candidate_masks_b=masks_b,
            candidate_valid_a=valid_a,
            candidate_valid_b=valid_b,
            opening_guidance_a=None,
            opening_guidance_b=None,
            opening_guidance_mode="predicted",
            opening_guidance_mix_weight=args.mix_guidance_weight,
        )
        reference = _reference_candidate_outputs(
            outputs,
            masks_a,
            masks_b,
            outputs["P_A_open"],
            outputs["P_B_open"],
            model.matcher.dustbin_score,
        )
        old_candidate = _old_candidate_assignment_loss(
            reference,
            target,
            pair.is_match,
            valid_a,
            valid_b,
            args.consistency_weight,
        )
        new_candidate = candidate_assignment_loss(
            outputs,
            candidate_target_ab=target,
            is_match=pair.is_match,
            candidate_valid_a=valid_a,
            candidate_valid_b=valid_b,
            consistency_weight=args.consistency_weight,
        )
        current_matching = matcher_losses(outputs, pair, target, valid_a, valid_b, args)
        reference_matching_total = current_matching["loss_total"] + (
            args.candidate_loss_weight
            * (old_candidate["loss_candidate_total"] - current_matching["loss_candidate"])
        )
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
        opening_total = 0.5 * (opening_a["loss_total"] + opening_b["loss_total"])
        feature_anchor, depth_anchor = _teacher_anchor_losses(live, raw_batch, device)
        current_total = (
            current_matching["loss_total"]
            + args.opening_loss_weight * opening_total
            + args.feature_anchor_weight * feature_anchor
            + args.depth_anchor_weight * depth_anchor
        )
        reference_total = (
            reference_matching_total
            + args.opening_loss_weight * opening_total
            + args.feature_anchor_weight * feature_anchor
            + args.depth_anchor_weight * depth_anchor
        )

    tensor_comparisons = {
        "candidate_assignment_logits": _tensor_error(
            outputs["candidate_assignment_logits"], reference["candidate_assignment_logits"]
        ),
        "candidate_assignment_AB": _tensor_error(
            outputs["candidate_assignment_AB"], reference["candidate_assignment_AB"]
        ),
        "candidate_assignment_BA": _tensor_error(
            outputs["candidate_assignment_BA"], reference["candidate_assignment_BA"]
        ),
        "mutual_affinity": _tensor_error(
            outputs["candidate_mutual_affinity"], reference["mutual_affinity"]
        ),
        "opening_confidence": _tensor_error(
            outputs["candidate_opening_confidence"], reference["opening_confidence"]
        ),
        "candidate_pair_score": _tensor_error(
            outputs["candidate_pair_score"], reference["candidate_pair_score"]
        ),
        "best_candidate_pair": _tensor_error(
            outputs["best_candidate_pair"], reference["best_candidate_pair"]
        ),
        "top1_score_order": {
            "equal": torch.equal(
                outputs["candidate_pair_score"].flatten(1).argmax(-1),
                reference["candidate_pair_score"].flatten(1).argmax(-1),
            ),
            "max_abs_error": 0.0,
        },
    }
    loss_comparisons = {
        "candidate_assignment_loss": _tensor_error(
            new_candidate["loss_candidate_assignment"],
            old_candidate["loss_candidate_assignment"],
        ),
        "candidate_total_loss": _tensor_error(
            new_candidate["loss_candidate_total"], old_candidate["loss_candidate_total"]
        ),
        "token_affinity_loss": {"equal": True, "max_abs_error": 0.0},
        "shared_response_loss": {"equal": True, "max_abs_error": 0.0},
        "portal_shift_loss": {"equal": True, "max_abs_error": 0.0},
        "opening_loss": {"equal": True, "max_abs_error": 0.0},
        "feature_anchor_loss": {"equal": True, "max_abs_error": 0.0},
        "depth_anchor_loss": {"equal": True, "max_abs_error": 0.0},
        "matching_total_loss": _tensor_error(
            current_matching["loss_total"], reference_matching_total
        ),
        "full_total_loss": _tensor_error(current_total, reference_total),
    }
    passed = all(item["equal"] for item in tensor_comparisons.values()) and all(
        item["equal"] for item in loss_comparisons.values()
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "sample_ids": list(raw_batch["pair_id"]),
        "seed": args.seed,
        "model_mode": "eval",
        "atol": ATOL,
        "rtol": RTOL,
        "tensor_comparisons": tensor_comparisons,
        "loss_comparisons": loss_comparisons,
        "other_loss_values": {
            key: float(current_matching[key].item())
            for key in (
                "loss_token_affinity",
                "loss_shared_response",
                "loss_portal_shift",
            )
        },
    }
    if not passed:
        return result
    result["real_masking_cases"] = _validate_real_masking_cases(
        model, checkpoint, dataset, device
    )
    masking_pass = result["real_masking_cases"]["status"] == "PASS"
    result["status"] = "PASS" if masking_pass else "FAIL"
    return result


def _slice_output(outputs: Mapping[str, torch.Tensor], index: int) -> Dict[str, torch.Tensor]:
    sliced = {}
    batch = int(outputs["candidate_assignment_AB"].shape[0])
    for key, value in outputs.items():
        sliced[key] = value[index : index + 1] if torch.is_tensor(value) and value.ndim and value.shape[0] == batch else value
    return sliced


def _validate_real_masking_cases(
    model: EndToEndOpeningMatcher,
    checkpoint: Mapping[str, Any],
    dataset: JointZInDMatchingDataset,
    device: torch.device,
) -> Dict[str, Any]:
    args = _training_args(checkpoint)
    threshold = float(checkpoint["opening_probability_threshold"])
    found: Dict[str, int] = {}
    search_loader = _loader(dataset, 4, 0, device)
    offset = 0
    model.eval()
    with torch.no_grad():
        for raw_batch in search_loader:
            pair = adapt_zind_bipair_batch(raw_batch).to(device)
            image_a, image_b = _images(raw_batch, device)
            live = model.live_layout(image_a, image_b)
            probability_a, probability_b = _predicted_probabilities(model, live)
            masks_a, valid_a = predicted_candidate_masks(
                probability_a, threshold, args.min_opening_width_tokens, args.max_openings_per_view
            )
            masks_b, valid_b = predicted_candidate_masks(
                probability_b, threshold, args.min_opening_width_tokens, args.max_openings_per_view
            )
            target = candidate_targets_from_affinity(masks_a, pair.affinity_ab, masks_b)
            has_target = target.flatten(1).any(-1)
            for local in range(int(pair.batch_size)):
                if not bool(pair.is_match[local]) and "negative_without_target" not in found:
                    found["negative_without_target"] = offset + local
                elif bool(pair.is_match[local]) and not bool(has_target[local]) and "positive_without_target" not in found:
                    found["positive_without_target"] = offset + local
                elif bool(pair.is_match[local]) and bool(has_target[local]) and "positive_with_target" not in found:
                    found["positive_with_target"] = offset + local
            offset += int(pair.batch_size)
            if len(found) == 3:
                break
            if offset >= min(len(dataset), 256):
                break
    if len(found) != 3:
        return {"status": "FAIL", "reason": "could not find all three real cases", "found": found}

    records: Dict[str, Any] = {}
    noncandidate_keys = (
        "loss_token_affinity",
        "loss_opening",
        "loss_feature_anchor",
        "loss_depth_anchor",
    )
    for label, index in found.items():
        raw = collate_zind_bipair([dataset[index]])
        args.candidate_missing_target_policy = "legacy_dustbin"
        model.eval()
        with torch.no_grad():
            legacy = _forward_loss(
                model,
                raw,
                device,
                opening_guidance_source="predicted",
                candidate_source="predicted",
                opening_threshold=threshold,
                args=args,
            )
        args.candidate_missing_target_policy = "mask_positive"
        with torch.no_grad():
            masked = _forward_loss(
                model,
                raw,
                device,
                opening_guidance_source="predicted",
                candidate_source="predicted",
                opening_threshold=threshold,
                args=args,
            )
        model.zero_grad(set_to_none=True)
        gradient_run = _forward_loss(
            model,
            raw,
            device,
            opening_guidance_source="predicted",
            candidate_source="predicted",
            opening_threshold=threshold,
            args=args,
        )
        gradient_run[-1]["loss_total"].backward()
        pair = gradient_run[1]
        target = gradient_run[2]
        sample_valid, has_target = _candidate_supervision_validity(
            target,
            pair.is_match,
            gradient_run[3],
            gradient_run[4],
            "mask_positive",
        )
        other_errors = {
            key: abs(float(legacy[-1][key].item()) - float(masked[-1][key].item()))
            for key in noncandidate_keys
        }
        records[label] = {
            "dataset_index": index,
            "pair_id": raw["pair_id"][0],
            "is_match_shape": list(pair.is_match.shape),
            "is_match_dtype": str(pair.is_match.dtype),
            "candidate_target_shape": list(target.shape),
            "candidate_target_dtype": str(target.dtype),
            "has_target": bool(has_target.item()),
            "sample_valid": bool(sample_valid.item()),
            "legacy_candidate_assignment_loss": float(
                legacy[-1]["loss_candidate_assignment"].item()
            ),
            "masked_candidate_assignment_loss": float(
                masked[-1]["loss_candidate_assignment"].item()
            ),
            "noncandidate_loss_max_abs_error": max(other_errors.values()),
            "noncandidate_losses": {
                key: float(masked[-1][key].item()) for key in noncandidate_keys
            },
            "dustbin_score_gradient": (
                float(model.matcher.dustbin_score.grad.item())
                if model.matcher.dustbin_score.grad is not None
                else None
            ),
            "opening_head_gradient_norm": _gradient_norm(model.matcher.opening_head),
            "matcher_gradient_norm": _gradient_norm(model.matcher),
            "backbone_gradient_norm": _gradient_norm(model.bi_layout),
            "loss_finite": bool(torch.isfinite(gradient_run[-1]["loss_total"])),
        }
    positive_missing = records["positive_without_target"]
    positive_target = records["positive_with_target"]
    negative = records["negative_without_target"]
    passed = (
        negative["sample_valid"]
        and negative["masked_candidate_assignment_loss"] > 0
        and negative["dustbin_score_gradient"] is not None
        and abs(negative["dustbin_score_gradient"]) > 0
        and not positive_missing["sample_valid"]
        and positive_missing["masked_candidate_assignment_loss"] == 0
        and positive_missing["noncandidate_loss_max_abs_error"] <= ATOL
        and positive_target["sample_valid"]
        and abs(
            positive_target["legacy_candidate_assignment_loss"]
            - positive_target["masked_candidate_assignment_loss"]
        )
        <= ATOL
        and all(record["loss_finite"] for record in records.values())
    )
    model.zero_grad(set_to_none=True)
    return {"status": "PASS" if passed else "FAIL", "cases": records}


def _manifest_candidates(
    raw_batch: Mapping[str, Any], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        raw_batch["manifest_candidate_masks_A"].to(device=device, dtype=torch.bool),
        raw_batch["manifest_candidate_masks_B"].to(device=device, dtype=torch.bool),
        raw_batch["manifest_candidate_valid_A"].to(device=device, dtype=torch.bool),
        raw_batch["manifest_candidate_valid_B"].to(device=device, dtype=torch.bool),
        raw_batch["candidate_target_AB"].to(device=device, dtype=torch.float32),
    )


def _unlimited_masks(
    probability: torch.Tensor, threshold: float, min_width: int
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    per_sample = []
    counts = []
    for sample in probability:
        intervals = opening_probabilities_to_intervals(
            sample, threshold=threshold, min_width_tokens=min_width, max_intervals=sample.numel()
        )
        counts.append(len(intervals))
        per_sample.append(candidate_intervals_to_mask(intervals, int(sample.numel()), sample.device))
    maximum = max(1, max(counts))
    masks = torch.zeros(
        (probability.shape[0], maximum, probability.shape[1]),
        dtype=torch.bool,
        device=probability.device,
    )
    valid = torch.zeros((probability.shape[0], maximum), dtype=torch.bool, device=probability.device)
    for index, item in enumerate(per_sample):
        count = int(item.shape[0])
        if count:
            masks[index, :count] = item
            valid[index, :count] = True
    return masks, valid, counts


class DiagnosticAccumulator:
    def __init__(self, name: str, opening_threshold: float, top_k: int = 3) -> None:
        self.name = name
        self.opening_threshold = float(opening_threshold)
        self.base = MetricAccumulator(top_k)
        self.opening_tp = self.opening_fp = self.opening_fn = 0
        self.side_a_recalled = self.side_b_recalled = self.joint_recalled = 0
        self.positive = 0
        self.candidate_counts: List[int] = []
        self.reciprocal_rank_sum = 0.0
        self.score_top1 = 0
        self.recall_at_3 = 0
        self.failure_rows: List[Dict[str, Any]] = []
        self.runtime_seconds = 0.0

    def update(
        self,
        outputs: Mapping[str, torch.Tensor],
        pair: Any,
        target: torch.Tensor,
        valid_a: torch.Tensor,
        valid_b: torch.Tensor,
        losses: Mapping[str, torch.Tensor],
        probability_a: torch.Tensor,
        probability_b: torch.Tensor,
        raw_batch: Mapping[str, Any],
        masks_a: torch.Tensor,
        masks_b: torch.Tensor,
        unlimited_target: torch.Tensor,
        near_target: torch.Tensor,
        raw_count_a: Sequence[int],
        raw_count_b: Sequence[int],
        target_iou: float,
    ) -> None:
        self.base.update(outputs, pair, target, valid_a, valid_b, losses)
        for probability, opening_target in (
            (probability_a, pair.opening_all_a),
            (probability_b, pair.opening_all_b),
        ):
            prediction = probability >= self.opening_threshold
            truth = opening_target.bool()
            self.opening_tp += int((prediction & truth).sum().item())
            self.opening_fp += int((prediction & ~truth).sum().item())
            self.opening_fn += int((~prediction & truth).sum().item())
        self.candidate_counts.extend(int(value) for value in valid_a.sum(-1).tolist())
        self.candidate_counts.extend(int(value) for value in valid_b.sum(-1).tolist())
        positive = pair.is_match.bool()
        self.positive += int(positive.sum().item())
        gt_masks_a, gt_masks_b, gt_valid_a, gt_valid_b, _ = _manifest_candidates(
            raw_batch, target.device
        )
        side_a_target = candidate_targets_from_affinity(
            masks_a, pair.affinity_ab, gt_masks_b, min_iou=target_iou
        )
        side_b_target = candidate_targets_from_affinity(
            gt_masks_a, pair.affinity_ab, masks_b, min_iou=target_iou
        )
        side_a = side_a_target.flatten(1).any(-1) & positive
        side_b = side_b_target.flatten(1).any(-1) & positive
        joint = (target > 0).flatten(1).any(-1) & positive
        self.side_a_recalled += int(side_a.sum().item())
        self.side_b_recalled += int(side_b.sum().item())
        self.joint_recalled += int(joint.sum().item())
        pair_ids = list(raw_batch["pair_id"])
        score = outputs["candidate_pair_score"]
        best = outputs["best_candidate_pair"]
        for index in torch.where(positive)[0].tolist():
            valid_pair = valid_a[index].unsqueeze(-1) & valid_b[index].unsqueeze(0)
            flat_valid = torch.where(valid_pair.flatten())[0]
            flat_target = target[index].flatten() > 0
            accepted = bool((best[index] >= 0).all())
            rank: Optional[int] = None
            score_top1_correct = False
            if flat_valid.numel():
                ranked = flat_valid[
                    torch.argsort(score[index].flatten()[flat_valid], descending=True)
                ]
                target_positions = torch.where(flat_target[ranked])[0]
                if target_positions.numel():
                    rank = int(target_positions[0].item()) + 1
                    self.reciprocal_rank_sum += 1.0 / rank
                    score_top1_correct = rank == 1
                    self.score_top1 += int(score_top1_correct)
                    self.recall_at_3 += int(rank <= 3)
            failure = "success"
            if rank is None:
                if bool(unlimited_target[index].flatten().any()):
                    failure = "candidate_truncated_by_max_candidates"
                elif not bool(side_a[index]) and not bool(side_b[index]):
                    failure = "both_sides_candidate_missing"
                elif not bool(side_a[index]):
                    failure = "side_a_candidate_missing"
                elif not bool(side_b[index]):
                    failure = "side_b_candidate_missing"
                elif bool(near_target[index].flatten().any()):
                    failure = "near_candidate_below_target_iou"
                else:
                    failure = "other_verified_failure"
            elif rank > 1:
                failure = "valid_target_rank_greater_than_1"
            else:
                best_a, best_b = (int(value) for value in best[index].tolist())
                accepted_correct = (
                    accepted
                    and best_a >= 0
                    and bool((target[index, best_a, best_b] > 0).item())
                )
                if not accepted_correct:
                    failure = "correct_rank1_rejected_by_dustbin"
            self.failure_rows.append(
                {
                    'portallayout.config': self.name,
                    "pair_id": pair_ids[index],
                    "failure_class": failure,
                    "side_a_recalled": bool(side_a[index]),
                    "side_b_recalled": bool(side_b[index]),
                    "joint_recalled": bool(joint[index]),
                    "correct_rank": rank,
                    "accepted": accepted,
                    "candidate_count_a": int(valid_a[index].sum().item()),
                    "candidate_count_b": int(valid_b[index].sum().item()),
                    "raw_component_count_a": int(raw_count_a[index]),
                    "raw_component_count_b": int(raw_count_b[index]),
                }
            )

    def result(self) -> Dict[str, Any]:
        base = self.base.result()
        positive = max(self.positive, 1)
        available = max(self.joint_recalled, 1)
        precision = self.opening_tp / max(self.opening_tp + self.opening_fp, 1)
        recall = self.opening_tp / max(self.opening_tp + self.opening_fn, 1)
        counts = np.asarray(self.candidate_counts, dtype=np.float64)
        failures = Counter(row["failure_class"] for row in self.failure_rows)
        return {
            **base,
            "opening_token_precision": precision,
            "opening_token_recall": recall,
            "opening_token_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "candidate_recall_a": self.side_a_recalled / positive,
            "candidate_recall_b": self.side_b_recalled / positive,
            "candidate_recall_joint": self.joint_recalled / positive,
            "positive_missing_target_rate": 1.0 - self.joint_recalled / positive,
            "candidate_count_mean": float(counts.mean()),
            "candidate_count_p50": float(np.percentile(counts, 50)),
            "candidate_count_p90": float(np.percentile(counts, 90)),
            "candidate_count_max": int(counts.max()),
            "conditional_score_top1": self.score_top1 / available,
            "conditional_top3": self.recall_at_3 / available,
            "conditional_mrr": self.reciprocal_rank_sum / available,
            "recall_at_3": self.recall_at_3 / available,
            "runtime_seconds": self.runtime_seconds,
            "failure_counts": dict(sorted(failures.items())),
        }


def _configuration_forward(
    model: EndToEndOpeningMatcher,
    live: Mapping[str, torch.Tensor],
    raw_batch: Mapping[str, Any],
    pair: Any,
    probability_a: torch.Tensor,
    probability_b: torch.Tensor,
    threshold: float,
    min_width: int,
    max_candidates: int,
    guidance_source: str,
    candidate_source: str,
    train_args: argparse.Namespace,
) -> Tuple[Mapping[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, List[int], List[int], torch.Tensor, torch.Tensor]:
    if candidate_source == "gt":
        masks_a, masks_b, valid_a, valid_b, target = _manifest_candidates(
            raw_batch, probability_a.device
        )
        unlimited_target = target > 0
        near_target = target > 0
        raw_count_a = [int(value) for value in valid_a.sum(-1).tolist()]
        raw_count_b = [int(value) for value in valid_b.sum(-1).tolist()]
    else:
        masks_a, valid_a = predicted_candidate_masks(
            probability_a, threshold, min_width, max_candidates
        )
        masks_b, valid_b = predicted_candidate_masks(
            probability_b, threshold, min_width, max_candidates
        )
        target = candidate_targets_from_affinity(
            masks_a, pair.affinity_ab, masks_b, min_iou=TARGET_IOU
        )
        unlimited_a, unlimited_valid_a, raw_count_a = _unlimited_masks(
            probability_a, threshold, min_width
        )
        unlimited_b, unlimited_valid_b, raw_count_b = _unlimited_masks(
            probability_b, threshold, min_width
        )
        unlimited_target = candidate_targets_from_affinity(
            unlimited_a, pair.affinity_ab, unlimited_b, min_iou=TARGET_IOU
        )
        near_target = candidate_targets_from_affinity(
            masks_a, pair.affinity_ab, masks_b, min_iou=1e-8
        )
    guidance_a = pair.opening_all_a.float() if guidance_source == "gt" else None
    guidance_b = pair.opening_all_b.float() if guidance_source == "gt" else None
    outputs = model.forward_matcher(
        live,
        candidate_masks_a=masks_a,
        candidate_masks_b=masks_b,
        candidate_valid_a=valid_a,
        candidate_valid_b=valid_b,
        opening_guidance_a=guidance_a,
        opening_guidance_b=guidance_b,
        opening_guidance_mode=guidance_source,
        opening_guidance_mix_weight=train_args.mix_guidance_weight,
    )
    matching = matcher_losses(outputs, pair, target, valid_a, valid_b, train_args)
    return (
        outputs,
        target,
        valid_a,
        valid_b,
        matching,
        masks_a,
        masks_b,
        raw_count_a,
        raw_count_b,
        unlimited_target,
        near_target,
    )


def _full_losses(
    matching: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    pair: Any,
    live: Mapping[str, torch.Tensor],
    raw_batch: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
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
    opening_total = 0.5 * (opening_a["loss_total"] + opening_b["loss_total"])
    feature_anchor, depth_anchor = _teacher_anchor_losses(live, raw_batch, device)
    total = (
        matching["loss_total"]
        + args.opening_loss_weight * opening_total
        + args.feature_anchor_weight * feature_anchor
        + args.depth_anchor_weight * depth_anchor
    )
    return {
        **matching,
        "loss_matching": matching["loss_total"],
        "loss_opening": opening_total,
        "loss_feature_anchor": feature_anchor,
        "loss_depth_anchor": depth_anchor,
        "loss_total": total,
    }


def evaluate_configurations(
    model: EndToEndOpeningMatcher,
    checkpoint: Mapping[str, Any],
    loader: DataLoader,
    configurations: Sequence[Mapping[str, Any]],
    device: torch.device,
    opening_cache_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    args = _training_args(checkpoint)
    accumulators = {
        config["name"]: DiagnosticAccumulator(
            config["name"], float(config["threshold"]), args.metric_top_k
        )
        for config in configurations
    }
    cached_ids: List[str] = []
    cached_a: List[torch.Tensor] = []
    cached_b: List[torch.Tensor] = []
    model.eval()
    total_start = time.perf_counter()
    with torch.no_grad():
        for step, raw_batch in enumerate(loader, start=1):
            image_a, image_b = _images(raw_batch, device)
            live = model.live_layout(image_a, image_b)
            pair = adapt_zind_bipair_batch(raw_batch).to(device)
            probability_a, probability_b = _predicted_probabilities(model, live)
            cached_ids.extend(raw_batch["pair_id"])
            cached_a.append(probability_a.detach().cpu().to(torch.float16))
            cached_b.append(probability_b.detach().cpu().to(torch.float16))
            for config in configurations:
                start = time.perf_counter()
                values = _configuration_forward(
                    model,
                    live,
                    raw_batch,
                    pair,
                    probability_a,
                    probability_b,
                    float(config["threshold"]),
                    int(config["min_width"]),
                    int(config["max_candidates"]),
                    config["opening_guidance_source"],
                    config["candidate_source"],
                    args,
                )
                outputs, target, valid_a, valid_b, matching = values[:5]
                losses = _full_losses(
                    matching, outputs, pair, live, raw_batch, device, args
                )
                accumulators[config["name"]].update(
                    outputs,
                    pair,
                    target,
                    valid_a,
                    valid_b,
                    losses,
                    probability_a,
                    probability_b,
                    raw_batch,
                    values[5],
                    values[6],
                    values[9],
                    values[10],
                    values[7],
                    values[8],
                    TARGET_IOU,
                )
                accumulators[config["name"]].runtime_seconds += time.perf_counter() - start
            if step % 100 == 0 or step == len(loader):
                print(
                    "validation {}/{} pairs={} configs={} elapsed={:.1f}s".format(
                        step,
                        len(loader),
                        len(cached_ids),
                        len(configurations),
                        time.perf_counter() - total_start,
                    ),
                    flush=True,
                )
    if opening_cache_path is not None:
        opening_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pair_ids": cached_ids,
                "opening_probability_A": torch.cat(cached_a),
                "opening_probability_B": torch.cat(cached_b),
                "checkpoint": str(DEFAULT_CHECKPOINT),
            },
            str(opening_cache_path),
        )
    results = {name: accumulator.result() for name, accumulator in accumulators.items()}
    failures = {name: accumulator.failure_rows for name, accumulator in accumulators.items()}
    return results, failures


def _select_g1(
    scan_rows: Sequence[Mapping[str, Any]], g0: Mapping[str, Any]
) -> Mapping[str, Any]:
    feasible = [
        row
        for row in scan_rows
        if float(row["candidate_count_mean"]) <= 2.0 * float(g0["candidate_count_mean"])
        and float(row["negative_rejection_rate"]) >= 0.90
    ]
    if not feasible:
        raise RuntimeError("no feasible candidate configuration under the fixed G1 rule")
    maximum_recall = max(float(row["candidate_recall_joint"]) for row in feasible)
    near = [
        row
        for row in feasible
        if maximum_recall - float(row["candidate_recall_joint"]) <= 0.005
    ]
    return sorted(
        near,
        key=lambda row: (
            -float(row["conditional_candidate_top1"]),
            float(row["candidate_count_mean"]),
            abs(float(row["threshold"]) - 0.47),
            int(row["min_width"]),
        ),
    )[0]


def _failure_summary(
    rows: Sequence[Mapping[str, Any]], config: str
) -> List[Dict[str, Any]]:
    selected = [row for row in rows if row['portallayout.config'] == config]
    total = len(selected)
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(row["failure_class"], []).append(row)
    output = []
    for name, items in sorted(grouped.items()):
        output.append(
            {
                'portallayout.config': config,
                "failure_class": name,
                "count": len(items),
                "proportion_of_positive": len(items) / max(total, 1),
                "representative_pair_ids": ";".join(
                    str(item["pair_id"]) for item in items[:3]
                ),
            }
        )
    return output


def _environment_text(args: argparse.Namespace, checkpoint_path: Path) -> str:
    lines = [
        "timestamp_unix={}".format(time.time()),
        "platform={}".format(platform.platform()),
        "python={}".format(sys.version.replace("\n", " ")),
        "torch={}".format(torch.__version__),
        "cuda_available={}".format(torch.cuda.is_available()),
        "cuda_runtime={}".format(torch.version.cuda),
        "gpu={}".format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "branch={}".format(_git("rev-parse", "--abbrev-ref", "HEAD")),
        "head={}".format(_git("rev-parse", "HEAD")),
        "checkpoint={}".format(checkpoint_path),
        "command={}".format(" ".join(sys.argv)),
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("validation diagnostics require the configured CUDA GPU")
    _seed_everything(args.seed)
    model, checkpoint = load_historical_model(checkpoint_path, device)
    summary = _checkpoint_summary(checkpoint, checkpoint_path)
    summary["sha256"] = subprocess.check_output(
        ("sha256sum", str(checkpoint_path)), text=True
    ).split()[0]
    _write_json(output_dir / "resolved_checkpoint_and_config.json", summary)
    (output_dir / "environment.txt").write_text(
        _environment_text(args, checkpoint_path), encoding="utf-8"
    )
    commands_path = output_dir / "commands.txt"
    with commands_path.open("a", encoding="utf-8") as file:
        file.write(" ".join(sys.argv) + "\n")
    dataset = validation_dataset(checkpoint, args.max_val_pairs)

    if not args.skip_compatibility:
        compatibility = run_compatibility(model, checkpoint, dataset, device)
        _write_json(output_dir / "compatibility_comparison.json", compatibility)
        print("compatibility {}".format(compatibility["status"]), flush=True)
        if compatibility["status"] != "PASS":
            return 2
    if args.compatibility_only:
        return 0

    g0_threshold = float(checkpoint["opening_probability_threshold"])
    g0_width = int(checkpoint["training_contract"]["min_opening_width_tokens"])
    max_candidates = int(checkpoint["training_contract"]["max_openings_per_view"])
    thresholds = sorted(
        set(round(value, 6) for value in _parse_number_list(args.thresholds, float) + [g0_threshold])
    )
    widths = sorted(set(_parse_number_list(args.min_widths, int) + [g0_width]))
    scan_configs = []
    for threshold in thresholds:
        for width in widths:
            scan_configs.append(
                {
                    "name": "threshold_{:.3f}_width_{}".format(threshold, width),
                    "threshold": threshold,
                    "min_width": width,
                    "max_candidates": max_candidates,
                    "opening_guidance_source": "predicted",
                    "candidate_source": "predicted",
                }
            )
    loader = _loader(dataset, args.batch_size, args.workers, device)
    scan_results, scan_failures = evaluate_configurations(
        model,
        checkpoint,
        loader,
        scan_configs,
        device,
        opening_cache_path=output_dir / "validation_opening_probabilities.pt",
    )
    scan_rows = []
    for config in scan_configs:
        row = {**config, **scan_results[config["name"]]}
        row.pop("failure_counts", None)
        scan_rows.append(row)
    _write_csv(output_dir / "candidate_threshold_scan.csv", scan_rows)
    g0_name = "threshold_{:.3f}_width_{}".format(g0_threshold, g0_width)
    g0 = next(row for row in scan_rows if row["name"] == g0_name)
    historical = json.loads(
        (checkpoint_path.parent / "best_metrics.json").read_text(encoding="utf-8")
    )["validation"]["predicted_openings"]
    g0_payload = {
        "configuration": next(config for config in scan_configs if config["name"] == g0_name),
        "metrics": scan_results[g0_name],
        "historical_best_metrics_comparison": {
            key: {
                "historical": historical[key],
                "reproduced": scan_results[g0_name][key],
                "absolute_error": abs(float(historical[key]) - float(scan_results[g0_name][key])),
            }
            for key in historical
            if key in scan_results[g0_name] and isinstance(historical[key], (int, float))
        },
        "target_iou_threshold": TARGET_IOU,
    }
    _write_json(output_dir / "g0_validation_metrics.json", g0_payload)
    g1 = dict(_select_g1(scan_rows, g0))
    _write_json(
        output_dir / "g1_selection.json",
        {
            "g0": g0,
            "g1": g1,
            "rules": {
                "mean_candidates_at_most_g0_times": 2.0,
                "minimum_negative_rejection_rate": 0.90,
                "recall_tie_window": 0.005,
            },
            "joint_recall_gain": float(g1["candidate_recall_joint"])
            - float(g0["candidate_recall_joint"]),
            "challenge_gain_reached": (
                float(g1["candidate_recall_joint"])
                - float(g0["candidate_recall_joint"])
                >= 0.05
            ),
        },
    )
    failure_rows = scan_failures[g0_name] + scan_failures[g1["name"]]
    failure_summary = _failure_summary(failure_rows, g0_name)
    if g1["name"] != g0_name:
        failure_summary += _failure_summary(failure_rows, g1["name"])
    _write_csv(output_dir / "candidate_failures_g0_g1.csv", failure_rows)
    _write_csv(output_dir / "candidate_failure_summary_g0_g1.csv", failure_summary)

    source_configs = []
    for guidance in ("gt", "predicted"):
        for candidate_source in ("gt", "predicted"):
            source_configs.append(
                {
                    "name": "{}_guidance__{}_candidates".format(guidance, candidate_source),
                    "threshold": float(g1["threshold"]),
                    "min_width": int(g1["min_width"]),
                    "max_candidates": max_candidates,
                    "opening_guidance_source": guidance,
                    "candidate_source": candidate_source,
                }
            )
    print("selected G1 {}".format(g1["name"]), flush=True)
    for config in source_configs:
        print(
            "evaluate opening_guidance_source={} candidate_source={}".format(
                config["opening_guidance_source"], config["candidate_source"]
            ),
            flush=True,
        )
    source_loader = _loader(dataset, args.batch_size, args.workers, device)
    source_results, _ = evaluate_configurations(
        model, checkpoint, source_loader, source_configs, device
    )
    source_rows = []
    for config in source_configs:
        result = source_results[config["name"]]
        source_rows.append(
            {
                **config,
                **{key: value for key, value in result.items() if key != "failure_counts"},
                "true_end_to_end": (
                    config["opening_guidance_source"] == "predicted"
                    and config["candidate_source"] == "predicted"
                ),
            }
        )
    _write_csv(output_dir / "teacher_forcing_2x2.csv", source_rows)
    _write_json(
        output_dir / "analysis_summary.json",
        {
            "compatibility_status": "PASS",
            "g0_name": g0_name,
            "g1_name": g1["name"],
            "g1_joint_recall_gain": float(g1["candidate_recall_joint"])
            - float(g0["candidate_recall_joint"]),
            "g1_challenge_gain_reached": (
                float(g1["candidate_recall_joint"])
                - float(g0["candidate_recall_joint"])
                >= 0.05
            ),
            "validation_pair_count": len(dataset),
            "test_split_accessed": False,
        },
    )
    print("analysis complete {}".format(output_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
