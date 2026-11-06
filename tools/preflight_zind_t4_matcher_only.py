#!/usr/bin/env python3
"""Deterministic T3-initialized matcher-only T4 overfit gates.

This tool only reads the fixed train preflight manifest.  It freezes and
caches every upstream tensor before optimizing the matcher.  Validation is
intentionally left to the normal end-to-end trainer after gates A and B pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portallayout.datasets.zind_bipair_adapter import adapt_zind_bipair_batch
from portallayout.datasets.zind_bipair_dataset import collate_zind_bipair
from portallayout.datasets.zind_bipair_adapter import opening_mask_to_candidate_masks
from portallayout.models.cross_scene_contracts import PairBatch
from portallayout.models.cross_scene_matcher import (
    candidate_assignment_loss,
    opening_detection_loss,
)
from tools.debug_cross_scene_flow import load_bi_layout
from tools.train_zind_cross_scene_matcher import (
    MetricAccumulator,
    _atomic_json,
    _atomic_torch_save,
    resolve_device,
    set_seed,
)
from tools.train_zind_end_to_end import (
    DEFAULT_BI_LAYOUT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_MATCHER,
    DEFAULT_OPENING,
    EndToEndOpeningMatcher,
    JointZInDMatchingDataset,
    _candidate_inputs,
    _candidate_supervision_validity,
    _matching_loss_bundle,
    _sha256_file,
    _teacher_anchor_losses,
    configure_training_stage,
    matcher_without_opening_parameters,
)
from portallayout.utils.matcher_checkpoint import load_cross_scene_matcher_checkpoint
from portallayout.utils.opening_checkpoint import load_opening_head_checkpoint


DEFAULT_T3 = (
    REPO_ROOT / "checkpoints/End_To_End/zind_e2e_v2/g0_seed123/T3/best.pt"
)
DEFAULT_TRAIN_MANIFEST = (
    REPO_ROOT
    / "experiments/zind_e2e_v3/g0_seed123/T4_recall/preflight/subsets/train_preflight.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/zind_e2e_v3/g0_seed123/T4_recall/matcher_only_preflight"
)
PROTOCOL_VERSION = "zind-t4-matcher-only-gates-v2"


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t3_checkpoint", default=str(DEFAULT_T3))
    parser.add_argument("--train_manifest", default=str(DEFAULT_TRAIN_MANIFEST))
    parser.add_argument(
        "--train_cache_dir", default=str(DEFAULT_CACHE_ROOT / "train")
    )
    parser.add_argument("--data_root")
    parser.add_argument("--bi_layout_config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--bi_layout_checkpoint", default=str(DEFAULT_BI_LAYOUT))
    parser.add_argument("--opening_checkpoint", default=str(DEFAULT_OPENING))
    parser.add_argument("--matcher_checkpoint", default=str(DEFAULT_MATCHER))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--single_batch_steps", type=int, default=200)
    parser.add_argument("--full_subset_steps", type=int, default=360)
    parser.add_argument("--overfit_lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument(
        "--reuse_frozen_cache",
        action="store_true",
        help="load and verify the existing frozen input artifact without rewriting it",
    )
    return parser.parse_args(argv)


def _protocol_v2(args: argparse.Namespace) -> Dict[str, Any]:
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "revision_scope": "gate predicates only; optimizer and inputs unchanged",
        "immutable_training_protocol": {
            "batch_composition": "same frozen 2-positive plus 2-negative Gate A batch",
            "seed": int(args.seed),
            "optimizer": "AdamW",
            "lr": float(args.overfit_lr),
            "weight_decay": 0.0,
            "grad_clip": float(args.grad_clip),
            "steps": int(args.single_batch_steps),
            "frozen_input_cache": True,
            "augmentation": False,
            "shuffle": False,
            "dynamic_candidates": False,
        },
        "gate_A": {
            "all_finite": True,
            "matcher_updated": True,
            "upstream_hash_unchanged": True,
            "matching_optim_loss_relative_drop_min": 0.20,
            "candidate_assignment_relative_drop_min": 0.50,
            "ranking_relative_drop_min": 0.50,
            "negative_no_match_relative_drop_min": 0.50,
            "margin_epsilon": 1e-8,
            "margin_rule": (
                "decline when initial>epsilon; otherwise final<=epsilon and "
                "true-to-dustbin margin nondecreasing"
            ),
            "acceptance_rejection_top1_nondecreasing": True,
            "frozen_artifact_identity_required": True,
        },
        "gate_B": {
            "matching_optim_loss_relative_drop_min": 0.20,
            "positive_acceptance_absolute_gain_min": 0.15,
            "negative_rejection_absolute_drop_max": 0.05,
            "target_present_top1_nondecreasing": True,
            "assignment_ranking_negative_no_match_decline": True,
            "margin_rule": "same conditional rule as Gate A",
            "all_finite": True,
            "upstream_hash_unchanged": True,
            "fixed_seed_short_rerun_exact": True,
        },
        "test_data_accessed": False,
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return protocol


def _relative_drop(initial: float, final: float) -> float:
    if initial <= 0.0:
        return 0.0 if final >= initial else float("inf")
    return (float(initial) - float(final)) / float(initial)


def _conditional_margin_gate(
    initial: Mapping[str, Any], final: Mapping[str, Any], epsilon: float = 1e-8
) -> Dict[str, Any]:
    initial_loss = float(initial["loss_positive_dustbin_margin"])
    final_loss = float(final["loss_positive_dustbin_margin"])
    initial_margin = float(initial["true_to_bidirectional_dustbin_margin"])
    final_margin = float(final["true_to_bidirectional_dustbin_margin"])
    if initial_loss > float(epsilon):
        passed = final_loss < initial_loss
        branch = "active_margin_loss_must_decline"
    else:
        passed = final_loss <= float(epsilon) and final_margin >= initial_margin
        branch = "zero_margin_loss_stays_zero_and_true_margin_nondecreases"
    return {
        "pass": bool(passed),
        "branch": branch,
        "epsilon": float(epsilon),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_true_margin": initial_margin,
        "final_true_margin": final_margin,
    }


def _loss_args() -> argparse.Namespace:
    return argparse.Namespace(
        training_stage="matcher_only_finetune",
        candidate_loss_weight=1.0,
        candidate_missing_target_policy="mask_positive",
        candidate_rank_loss_weight=0.1,
        token_affinity_weight=1.0,
        shared_response_weight=0.0,
        portal_shift_loss_weight=0.2,
        consistency_weight=0.1,
        positive_dustbin_margin_loss_weight=0.2,
        positive_dustbin_margin=0.2,
        opening_loss_weight=1.0,
        feature_anchor_weight=0.02,
        depth_anchor_weight=0.02,
        opening_pos_weight=2.5,
        opening_tversky_weight=0.5,
        mix_guidance_weight=0.5,
        min_opening_width_tokens=2,
        max_openings_per_view=12,
        candidate_proposal_mode="recall_topk",
        proposal_peak_top_k=4,
        proposal_peak_min_probability=0.20,
        proposal_expansion_threshold=0.12,
        proposal_dedup_iou=0.50,
        train_gt_candidate_injection=True,
        metric_top_k=3,
    )


def _tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(_tensor_hash(value).encode())
    return digest.hexdigest()


def _parameter_snapshot(parameters: Iterable[nn.Parameter]) -> List[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _parameter_delta(
    parameters: Iterable[nn.Parameter], before: Sequence[torch.Tensor]
) -> float:
    square = 0.0
    for parameter, initial in zip(parameters, before):
        delta = parameter.detach().cpu().float() - initial.float()
        square += float(delta.square().sum().item())
    return math.sqrt(square)


def _load_t3_model(
    args: argparse.Namespace, device: torch.device
) -> Tuple[EndToEndOpeningMatcher, float, Dict[str, Any]]:
    t3_path = Path(args.t3_checkpoint).expanduser().resolve()
    checkpoint = torch.load(str(t3_path), map_location="cpu")
    if checkpoint.get("task") != "ZInD_matching full end-to-end joint fine-tuning":
        raise ValueError("T3 checkpoint task contract is invalid")
    if checkpoint.get("data_contract", {}).get("test_evaluation_enabled") is not False:
        raise ValueError("T3 checkpoint must have test evaluation disabled")
    token_count = int(checkpoint["model_contract"]["token_count"])
    feature_dim = int(checkpoint["model_contract"]["feature_dim"])
    config = Path(args.bi_layout_config).expanduser().resolve()
    bi_checkpoint = Path(args.bi_layout_checkpoint).expanduser().resolve()
    opening_checkpoint = Path(args.opening_checkpoint).expanduser().resolve()
    matcher_checkpoint = Path(args.matcher_checkpoint).expanduser().resolve()
    bi_layout, _ = load_bi_layout(
        str(config), str(bi_checkpoint), device, True
    )
    matcher, _ = load_cross_scene_matcher_checkpoint(
        str(matcher_checkpoint),
        expected_feature_dim=feature_dim,
        expected_token_count=token_count,
        expected_branch_order="extended_first",
        bi_layout_config_path=str(config),
        bi_layout_checkpoint_path=str(bi_checkpoint),
        device=device,
    )
    load_opening_head_checkpoint(
        matcher.opening_head,
        str(opening_checkpoint),
        expected_feature_dim=feature_dim,
        expected_branch_order="extended_first",
        expected_token_count=token_count,
        bi_layout_config_path=str(config),
        bi_layout_checkpoint_path=str(bi_checkpoint),
    )
    model = EndToEndOpeningMatcher(
        bi_layout, matcher, activation_checkpointing=False
    ).to(device)
    model.bi_layout.load_state_dict(checkpoint["bi_layout_state_dict"], strict=True)
    model.matcher.load_state_dict(checkpoint["matcher_state_dict"], strict=True)
    threshold = float(checkpoint["opening_probability_threshold"])
    report = {
        "path": str(t3_path),
        "sha256": _sha256_file(t3_path),
        "size": t3_path.stat().st_size,
        "mtime_ns": t3_path.stat().st_mtime_ns,
        "completed_epoch": int(checkpoint.get("completed_epoch", -1)),
    }
    del checkpoint
    configure_training_stage(
        model, "matcher_only_finetune", deterministic_matcher_dropout=True
    )
    return model, threshold, report


def _fixed_order(length: int) -> List[int]:
    if length != 24:
        raise ValueError("the deterministic gate requires the exact 24-pair subset")
    # The manifest stores 16 positives followed by 8 negatives.  Put two of
    # each into the first batch, then retain a deterministic complete order.
    first = [0, 1, 16, 17]
    return first + [index for index in range(length) if index not in set(first)]


def _raw_signature(raw: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in sorted(raw):
        value = raw[key]
        if torch.is_tensor(value):
            result[key] = {
                "shape": list(value.shape),
                "sha256": _tensor_hash(value),
            }
        elif isinstance(value, list):
            result[key] = list(value)
    return result


@torch.no_grad()
def _build_cache(
    model: EndToEndOpeningMatcher,
    dataset: JointZInDMatchingDataset,
    order: Sequence[int],
    device: torch.device,
    opening_threshold: float,
    loss_args: argparse.Namespace,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    configure_training_stage(
        model, "matcher_only_finetune", deterministic_matcher_dropout=True
    )
    loader = DataLoader(
        Subset(dataset, list(order)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_zind_bipair,
        drop_last=False,
    )
    entries: List[Dict[str, Any]] = []
    raw_signatures: List[Dict[str, Any]] = []
    for raw in loader:
        raw_signatures.append(_raw_signature(raw))
        pair_cpu = adapt_zind_bipair_batch(raw)
        pair = pair_cpu.to(device)
        image_a = raw["image_A"].to(device=device, dtype=torch.float32)
        image_b = raw["image_B"].to(device=device, dtype=torch.float32)
        live = {
            key: value.detach()
            for key, value in model.live_layout(image_a, image_b).items()
        }
        masks_a, masks_b, valid_a, valid_b, target = _candidate_inputs(
            model,
            live,
            raw,
            pair,
            "predicted",
            opening_threshold,
            loss_args,
        )
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
        minimal_metadata = {
            "pair_id": list(raw["pair_id"]),
            "merged_pair_count": raw["merged_pair_count"].detach().cpu().clone(),
        }
        pair_cache = replace(
            pair_cpu,
            image_a=None,
            image_b=None,
            metadata=minimal_metadata,
        )
        teacher = {
            key: raw[key].detach().cpu().clone()
            for key in (
                "feature_A",
                "feature_B",
                "depth_enclosed_A",
                "depth_enclosed_B",
                "depth_extended_A",
                "depth_extended_B",
            )
        }
        entries.append(
            {
                "pair_ids": list(raw["pair_id"]),
                "live": {
                    key: value.detach().cpu().clone()
                    for key, value in live.items()
                },
                "teacher": teacher,
                "pair": pair_cache,
                "candidate_masks_A": masks_a.detach().cpu().clone(),
                "candidate_masks_B": masks_b.detach().cpu().clone(),
                "candidate_valid_A": valid_a.detach().cpu().clone(),
                "candidate_valid_B": valid_b.detach().cpu().clone(),
                "candidate_target_AB": target.detach().cpu().clone(),
                "opening_probability_A": signal_a["opening_probability"].detach().cpu().clone(),
                "opening_probability_B": signal_b["opening_probability"].detach().cpu().clone(),
                "gt_candidate_injection_count": int(
                    raw.get("_t4_gt_candidate_injection_count", 0)
                ),
            }
        )
    return entries, raw_signatures


def _entry_signature(entry: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"pair_ids": list(entry["pair_ids"])}
    for group in ("live", "teacher"):
        result[group] = {
            key: {"shape": list(value.shape), "sha256": _tensor_hash(value)}
            for key, value in sorted(entry[group].items())
        }
    for key in (
        "candidate_masks_A",
        "candidate_masks_B",
        "candidate_valid_A",
        "candidate_valid_B",
        "candidate_target_AB",
        "opening_probability_A",
        "opening_probability_B",
    ):
        value = entry[key]
        result[key] = {"shape": list(value.shape), "sha256": _tensor_hash(value)}
    result["is_match"] = _tensor_hash(entry["pair"].is_match)
    result["affinity_ab"] = _tensor_hash(entry["pair"].affinity_ab)
    result["injection_count"] = int(entry["gt_candidate_injection_count"])
    return result


def _cache_metadata(
    entries: Sequence[Mapping[str, Any]],
    signatures: Sequence[Mapping[str, Any]],
    raw_signatures: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    t3_report: Mapping[str, Any],
) -> Dict[str, Any]:
    proposal = {
        "mode": "recall_topk",
        "threshold": 0.47,
        "peak_top_k": 4,
        "peak_min_probability": 0.20,
        "expansion_threshold": 0.12,
        "dedup_iou": 0.50,
        "max_candidates": 12,
        "train_gt_candidate_injection": True,
    }
    config_hash = hashlib.sha256(
        json.dumps(
            {"t3_sha256": t3_report["sha256"], "proposal": proposal},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    positives = sum(int(entry["pair"].is_match.sum().item()) for entry in entries)
    target_positive = sum(
        int(
            (
                entry["pair"].is_match
                & entry["candidate_target_AB"].flatten(1).gt(0).any(dim=-1)
            ).sum().item()
        )
        for entry in entries
    )
    return {
        "schema_version": "zind-t4-frozen-matcher-input-v1",
        "config_sha256": config_hash,
        "t3": dict(t3_report),
        "source_manifest": str(Path(args.train_manifest).expanduser().resolve()),
        "source_split": "train",
        "train_only": True,
        "validation_candidate_inference_uses_gt": False,
        "test_data_accessed": False,
        "proposal": proposal,
        "pair_order": [pair for entry in entries for pair in entry["pair_ids"]],
        "batch_order": [list(entry["pair_ids"]) for entry in entries],
        "pair_count": sum(len(entry["pair_ids"]) for entry in entries),
        "positive_count": positives,
        "positive_target_count": target_positive,
        "positive_target_rate": target_positive / max(positives, 1),
        "tensor_signatures": list(signatures),
        "raw_input_signatures": list(raw_signatures),
    }


def _load_and_verify_frozen_cache(
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    cache_path = output_dir / "frozen_matcher_inputs.pt"
    schema_path = output_dir / "frozen_matcher_inputs.schema.json"
    old_gate_path = output_dir / "gate_A_single_batch.json"
    for path in (cache_path, schema_path, old_gate_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    identity_before = {
        "cache_sha256": _sha256_file(cache_path),
        "cache_size": cache_path.stat().st_size,
        "cache_mtime_ns": cache_path.stat().st_mtime_ns,
        "schema_sha256": _sha256_file(schema_path),
        "schema_size": schema_path.stat().st_size,
        "schema_mtime_ns": schema_path.stat().st_mtime_ns,
    }
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    payload = torch.load(str(cache_path), map_location="cpu")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("frozen matcher cache has no entries")
    signatures = [_entry_signature(entry) for entry in entries]
    if signatures != schema.get("tensor_signatures"):
        raise ValueError("frozen cache tensor signatures differ from saved schema")
    pair_order = [pair for entry in entries for pair in entry["pair_ids"]]
    if pair_order != schema.get("pair_order"):
        raise ValueError("frozen cache pair order differs from saved schema")
    old_gate = json.loads(old_gate_path.read_text(encoding="utf-8"))
    old_protocol = old_gate.get("protocol", {})
    immutable_expected = {
        "seed": 123,
        "optimizer": "AdamW",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "max_steps": 200,
        "batch_pair_ids": list(entries[0]["pair_ids"]),
    }
    immutable_observed = {
        key: old_protocol.get(key) for key in immutable_expected
    }
    if immutable_observed != immutable_expected:
        raise ValueError(
            "existing Gate A protocol does not match immutable v2 requirements"
        )
    identity_after = {
        "cache_sha256": _sha256_file(cache_path),
        "cache_size": cache_path.stat().st_size,
        "cache_mtime_ns": cache_path.stat().st_mtime_ns,
        "schema_sha256": _sha256_file(schema_path),
        "schema_size": schema_path.stat().st_size,
        "schema_mtime_ns": schema_path.stat().st_mtime_ns,
    }
    verification = {
        "identity_before": identity_before,
        "identity_after_load": identity_after,
        "identity_unchanged": identity_before == identity_after,
        "tensor_signatures_exact": True,
        "pair_order_exact": True,
        "gate_A_batch_exact": list(entries[0]["pair_ids"])
        == old_protocol.get("batch_pair_ids"),
        "old_protocol_immutable_fields": immutable_observed,
        "test_data_accessed": False,
    }
    return entries, schema, verification


def _move_mapping(
    values: Mapping[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in values.items()}


def _true_dustbin_margin(
    outputs: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    positive: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
) -> Tuple[float, int]:
    logits = outputs["candidate_assignment_logits"].detach()
    count_a = valid_a.shape[1]
    count_b = valid_b.shape[1]
    true_pair = (
        target.gt(0)
        & valid_a.unsqueeze(-1)
        & valid_b.unsqueeze(1)
        & positive[:, None, None]
    )
    count = int(true_pair.sum().item())
    if count == 0:
        return 0.0, 0
    real = logits[:, :count_a, :count_b]
    margin_a = real - logits[:, :count_a, count_b].unsqueeze(-1)
    margin_b = real - logits[:, count_a, :count_b].unsqueeze(1)
    return float((0.5 * (margin_a + margin_b))[true_pair].mean().item()), count


def _forward_cached(
    model: EndToEndOpeningMatcher,
    entry: Mapping[str, Any],
    device: torch.device,
    loss_args: argparse.Namespace,
) -> Tuple[Mapping[str, torch.Tensor], Any, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    live = _move_mapping(entry["live"], device)
    teacher = _move_mapping(entry["teacher"], device)
    pair = entry["pair"].to(device)
    masks_a = entry["candidate_masks_A"].to(device)
    masks_b = entry["candidate_masks_B"].to(device)
    valid_a = entry["candidate_valid_A"].to(device)
    valid_b = entry["candidate_valid_B"].to(device)
    target = entry["candidate_target_AB"].to(device)
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
    matching = _matching_loss_bundle(
        outputs, pair, target, valid_a, valid_b, loss_args
    )
    opening_a = opening_detection_loss(
        outputs["opening_logits_A"].float(),
        pair.opening_all_a.float(),
        pos_weight=loss_args.opening_pos_weight,
        tversky_weight=loss_args.opening_tversky_weight,
    )
    opening_b = opening_detection_loss(
        outputs["opening_logits_B"].float(),
        pair.opening_all_b.float(),
        pos_weight=loss_args.opening_pos_weight,
        tversky_weight=loss_args.opening_tversky_weight,
    )
    opening = 0.5 * (opening_a["loss_total"] + opening_b["loss_total"])
    feature_anchor, depth_anchor = _teacher_anchor_losses(live, teacher, device)
    reported = (
        matching["loss_total"]
        + loss_args.opening_loss_weight * opening
        + loss_args.feature_anchor_weight * feature_anchor
        + loss_args.depth_anchor_weight * depth_anchor
    )
    positive = pair.is_match.bool()
    sample_valid, has_target = _candidate_supervision_validity(
        target, positive, valid_a, valid_b, "mask_positive"
    )
    positive_selected = positive & has_target
    negative_selected = ~positive
    positive_assignment = candidate_assignment_loss(
        outputs,
        candidate_target_ab=target,
        is_match=positive,
        candidate_valid_a=valid_a,
        candidate_valid_b=valid_b,
        sample_valid=positive_selected,
        consistency_weight=loss_args.consistency_weight,
    )
    negative_assignment = candidate_assignment_loss(
        outputs,
        candidate_target_ab=target,
        is_match=positive,
        candidate_valid_a=valid_a,
        candidate_valid_b=valid_b,
        sample_valid=negative_selected,
        consistency_weight=loss_args.consistency_weight,
    )
    margin_mean, margin_count = _true_dustbin_margin(
        outputs, target, positive, valid_a, valid_b
    )
    losses = {
        **matching,
        "matching_optim_loss": matching["loss_total"],
        "reported_end_to_end_loss": reported,
        "loss_opening": opening,
        "loss_feature_anchor": feature_anchor,
        "loss_depth_anchor": depth_anchor,
        "loss_positive_assignment": positive_assignment[
            "loss_candidate_assignment"
        ],
        "loss_no_match_negative": negative_assignment[
            "loss_candidate_assignment"
        ],
    }
    counts = {
        "samples": int(pair.batch_size),
        "positive_samples": int(positive.sum().item()),
        "negative_samples": int((~positive).sum().item()),
        "candidate_loss_valid_samples": int(sample_valid.sum().item()),
        "positive_target_samples": int(positive_selected.sum().item()),
        "positive_assignment_rows_A": int(valid_a[positive_selected].sum().item()),
        "positive_assignment_rows_B": int(valid_b[positive_selected].sum().item()),
        "negative_assignment_rows_A": int(valid_a[negative_selected].sum().item()),
        "negative_assignment_rows_B": int(valid_b[negative_selected].sum().item()),
        "ranking_samples": int(positive_selected.sum().item()),
        "margin_samples": int(positive_selected.sum().item()),
        "margin_target_cells": margin_count,
        "true_to_bidirectional_dustbin_margin": margin_mean,
        "token_affinity_rows_A": int(pair.affinity_ab.sum(-1).gt(0).sum().item()),
        "token_affinity_rows_B": int(pair.affinity_ab.sum(-2).gt(0).sum().item()),
        "opening_tokens": int(2 * pair.batch_size * pair.token_count),
    }
    return outputs, pair, target, valid_a, valid_b, losses, counts


@torch.no_grad()
def _evaluate_cache(
    model: EndToEndOpeningMatcher,
    entries: Sequence[Mapping[str, Any]],
    device: torch.device,
    loss_args: argparse.Namespace,
) -> Dict[str, Any]:
    configure_training_stage(
        model, "matcher_only_finetune", deterministic_matcher_dropout=True
    )
    accumulator = MetricAccumulator(loss_args.metric_top_k)
    margin_sum = 0.0
    margin_count = 0
    counts_total: Dict[str, int] = {}
    for entry in entries:
        outputs, pair, target, valid_a, valid_b, losses, counts = _forward_cached(
            model, entry, device, loss_args
        )
        accumulator.update(outputs, pair, target, valid_a, valid_b, losses)
        count = int(counts["margin_target_cells"])
        margin_sum += counts["true_to_bidirectional_dustbin_margin"] * count
        margin_count += count
        for key, value in counts.items():
            if isinstance(value, int):
                counts_total[key] = counts_total.get(key, 0) + value
    result: Dict[str, Any] = accumulator.result()
    result["true_to_bidirectional_dustbin_margin"] = (
        margin_sum / max(margin_count, 1)
    )
    result["effective_counts"] = counts_total
    return result


def _finite_mapping(values: Mapping[str, Any]) -> bool:
    for value in values.values():
        if isinstance(value, Mapping):
            if not _finite_mapping(value):
                return False
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Mapping) and not _finite_mapping(item):
                    return False
                if isinstance(item, (int, float)) and not math.isfinite(float(item)):
                    return False
        elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def _optimizer(model: EndToEndOpeningMatcher, lr: float) -> torch.optim.Optimizer:
    parameters = list(matcher_without_opening_parameters(model))
    return torch.optim.AdamW(
        [{"name": "matcher", "params": parameters, "lr": float(lr)}],
        weight_decay=0.0,
    )


def _gradient_parameter_stats(module: nn.Module) -> Dict[str, Any]:
    finite = nonzero = tensors = elements = 0
    norm_square = 0.0
    for parameter in module.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        tensors += 1
        elements += parameter.numel()
        is_finite = bool(torch.isfinite(parameter.grad).all())
        finite += int(is_finite)
        value = float(parameter.grad.detach().float().norm().item())
        nonzero += int(value > 0.0)
        norm_square += value * value
    return {
        "gradient_tensors": tensors,
        "gradient_elements": elements,
        "finite_gradient_tensors": finite,
        "nonzero_gradient_tensors": nonzero,
        "gradient_norm": math.sqrt(norm_square),
    }


def _parameter_audit(
    model: EndToEndOpeningMatcher, optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    candidate_modules = nn.ModuleList(
        [model.matcher.query_projection, model.matcher.key_projection]
    )
    matcher_encoder = nn.ModuleList(
        [
            module
            for name, module in model.matcher.named_children()
            if name not in ("opening_head", "query_projection", "key_projection")
        ]
    )
    groups = {
        "bi_layout_backbone": model.bi_layout,
        "opening_head": model.matcher.opening_head,
        "matcher_encoder_exclusive": matcher_encoder,
        "candidate_scoring_shared_query_key": candidate_modules,
    }
    report: Dict[str, Any] = {}
    for name, module in groups.items():
        parameters = list(module.parameters())
        report[name] = {
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "requires_grad_count": sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
            "optimizer_parameter_count": sum(
                parameter.numel()
                for parameter in parameters
                if id(parameter) in optimizer_ids
            ),
            "train": bool(module.training),
        }
    dustbin = model.matcher.dustbin_score
    report["dustbin_parameter"] = {
        "parameter_count": dustbin.numel(),
        "requires_grad_count": dustbin.numel() if dustbin.requires_grad else 0,
        "optimizer_parameter_count": dustbin.numel() if id(dustbin) in optimizer_ids else 0,
        "value": float(dustbin.detach().item()),
        "train": bool(model.matcher.training),
    }
    report["candidate_pair_score_head"] = {
        "parameter_count": 0,
        "note": "parameter-free score; query/key projections are shared with token attention",
    }
    return report


def _curve_point(step: int, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "matching_optim_loss",
        "loss_candidate_assignment",
        "loss_candidate_rank",
        "loss_positive_dustbin_margin",
        "loss_no_match_negative",
        "positive_acceptance_rate",
        "negative_rejection_rate",
        "conditional_candidate_top1",
        "true_to_bidirectional_dustbin_margin",
    )
    return {"step": int(step), **{key: metrics[key] for key in keys}}


def _train_step(
    model: EndToEndOpeningMatcher,
    entry: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_args: argparse.Namespace,
    grad_clip: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    configure_training_stage(
        model, "matcher_only_finetune", deterministic_matcher_dropout=True
    )
    optimizer.zero_grad(set_to_none=True)
    _, _, _, _, _, losses, counts = _forward_cached(
        model, entry, device, loss_args
    )
    losses["matching_optim_loss"].backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            list(matcher_without_opening_parameters(model)), grad_clip
        )
    optimizer.step()
    return losses, counts


def run_single_batch_gate(
    model: EndToEndOpeningMatcher,
    entry: Mapping[str, Any],
    device: torch.device,
    loss_args: argparse.Namespace,
    args: argparse.Namespace,
    frozen_artifact_exact: bool,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    optimizer = _optimizer(model, args.overfit_lr)
    matcher_parameters = list(matcher_without_opening_parameters(model))
    matcher_before = _parameter_snapshot(matcher_parameters)
    frozen_before = {
        "bi_layout": _state_hash(model.bi_layout),
        "opening_head": _state_hash(model.matcher.opening_head),
    }
    initial = _evaluate_cache(model, [entry], device, loss_args)
    curve = [_curve_point(0, initial)]
    gradient_audit = None
    for step in range(1, args.single_batch_steps + 1):
        _train_step(
            model, entry, optimizer, device, loss_args, args.grad_clip
        )
        if step == 1:
            gradient_audit = {
                "bi_layout": _gradient_parameter_stats(model.bi_layout),
                "opening_head": _gradient_parameter_stats(model.matcher.opening_head),
                "matcher": _gradient_parameter_stats(model.matcher),
            }
        if step % 10 == 0 or step == args.single_batch_steps:
            curve.append(
                _curve_point(
                    step, _evaluate_cache(model, [entry], device, loss_args)
                )
            )
    final = _evaluate_cache(model, [entry], device, loss_args)
    frozen_after = {
        "bi_layout": _state_hash(model.bi_layout),
        "opening_head": _state_hash(model.matcher.opening_head),
    }
    matcher_update = _parameter_delta(matcher_parameters, matcher_before)
    margin_gate = _conditional_margin_gate(initial, final)
    finite_gradients = bool(
        gradient_audit
        and gradient_audit["matcher"]["finite_gradient_tensors"]
        == gradient_audit["matcher"]["gradient_tensors"]
    )
    gates = {
        "finite_losses_gradients_parameter_update": (
            _finite_mapping(initial)
            and _finite_mapping(final)
            and _finite_mapping({"curve": curve})
            and finite_gradients
            and math.isfinite(matcher_update)
        ),
        "matcher_updated": matcher_update > 0.0,
        "upstream_hash_unchanged": frozen_before == frozen_after,
        "matching_loss_down_20pct": _relative_drop(
            initial["matching_optim_loss"], final["matching_optim_loss"]
        ) >= 0.20,
        "candidate_assignment_down_50pct": _relative_drop(
            initial["loss_candidate_assignment"],
            final["loss_candidate_assignment"],
        ) >= 0.50,
        "ranking_down_50pct": _relative_drop(
            initial["loss_candidate_rank"], final["loss_candidate_rank"]
        ) >= 0.50,
        "negative_no_match_down_50pct": _relative_drop(
            initial["loss_no_match_negative"], final["loss_no_match_negative"]
        ) >= 0.50,
        "conditional_margin": margin_gate["pass"],
        "positive_acceptance_not_lower": final["positive_acceptance_rate"]
        >= initial["positive_acceptance_rate"],
        "negative_rejection_not_lower": final["negative_rejection_rate"]
        >= initial["negative_rejection_rate"],
        "candidate_top1_not_lower": final["conditional_candidate_top1"]
        >= initial["conditional_candidate_top1"],
        "frozen_artifact_exact": bool(frozen_artifact_exact),
    }
    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol": {
            "seed": args.seed,
            "optimizer": "AdamW",
            "lr": args.overfit_lr,
            "weight_decay": 0.0,
            "grad_clip": args.grad_clip,
            "max_steps": args.single_batch_steps,
            "batch_pair_ids": list(entry["pair_ids"]),
            "matcher_train": True,
            "matcher_dropout_eval_for_determinism": True,
            "upstream_eval": True,
        },
        "initial": initial,
        "final": final,
        "curve_every_10_steps": curve,
        "matcher_parameter_update_norm": matcher_update,
        "frozen_hash_before": frozen_before,
        "frozen_hash_after": frozen_after,
        "gradient_audit_after_first_backward": gradient_audit,
        "parameter_audit": _parameter_audit(model, optimizer),
        "relative_drops": {
            "matching_optim_loss": _relative_drop(
                initial["matching_optim_loss"], final["matching_optim_loss"]
            ),
            "candidate_assignment": _relative_drop(
                initial["loss_candidate_assignment"],
                final["loss_candidate_assignment"],
            ),
            "ranking": _relative_drop(
                initial["loss_candidate_rank"], final["loss_candidate_rank"]
            ),
            "negative_no_match": _relative_drop(
                initial["loss_no_match_negative"],
                final["loss_no_match_negative"],
            ),
        },
        "conditional_margin_gate": margin_gate,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _short_reproducibility(
    model: EndToEndOpeningMatcher,
    initial_matcher_state: Mapping[str, torch.Tensor],
    entries: Sequence[Mapping[str, Any]],
    device: torch.device,
    loss_args: argparse.Namespace,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    sequences = []
    for _ in range(2):
        model.matcher.load_state_dict(initial_matcher_state, strict=True)
        configure_training_stage(
            model, "matcher_only_finetune", deterministic_matcher_dropout=True
        )
        set_seed(args.seed)
        optimizer = _optimizer(model, args.overfit_lr)
        losses = []
        pair_order = []
        target_hashes = []
        for entry in entries[:3]:
            with torch.no_grad():
                metrics = _evaluate_cache(model, [entry], device, loss_args)
            losses.append(float(metrics["matching_optim_loss"]))
            pair_order.extend(entry["pair_ids"])
            target_hashes.append(_tensor_hash(entry["candidate_target_AB"]))
            _train_step(
                model, entry, optimizer, device, loss_args, args.grad_clip
            )
        sequences.append(
            {
                "pair_order": pair_order,
                "target_hashes": target_hashes,
                "pre_step_losses": losses,
            }
        )
    exact = sequences[0] == sequences[1]
    return {"runs": sequences, "exact": exact}


def run_full_subset_gate(
    model: EndToEndOpeningMatcher,
    entries: Sequence[Mapping[str, Any]],
    device: torch.device,
    loss_args: argparse.Namespace,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    initial_matcher_state = copy.deepcopy(model.matcher.state_dict())
    upstream_before = {
        "bi_layout": _state_hash(model.bi_layout),
        "opening_head": _state_hash(model.matcher.opening_head),
    }
    optimizer = _optimizer(model, args.overfit_lr)
    initial = _evaluate_cache(model, entries, device, loss_args)
    curve = [_curve_point(0, initial)]
    for step in range(1, args.full_subset_steps + 1):
        entry = entries[(step - 1) % len(entries)]
        _train_step(
            model, entry, optimizer, device, loss_args, args.grad_clip
        )
        if step % 30 == 0 or step == args.full_subset_steps:
            curve.append(
                _curve_point(
                    step, _evaluate_cache(model, entries, device, loss_args)
                )
            )
    final = _evaluate_cache(model, entries, device, loss_args)
    upstream_after_training = {
        "bi_layout": _state_hash(model.bi_layout),
        "opening_head": _state_hash(model.matcher.opening_head),
    }
    reproducibility = _short_reproducibility(
        model,
        initial_matcher_state,
        entries,
        device,
        loss_args,
        args,
    )
    upstream_after_repro = {
        "bi_layout": _state_hash(model.bi_layout),
        "opening_head": _state_hash(model.matcher.opening_head),
    }
    margin_gate = _conditional_margin_gate(initial, final)
    gates = {
        "finite": _finite_mapping(initial) and _finite_mapping(final),
        "matching_loss_down_20pct": final["matching_optim_loss"] <= 0.8 * initial["matching_optim_loss"],
        "positive_acceptance_up_15pp": final["positive_acceptance_rate"] >= initial["positive_acceptance_rate"] + 0.15,
        "negative_rejection_drop_at_most_5pp": final["negative_rejection_rate"] >= initial["negative_rejection_rate"] - 0.05,
        "target_present_top1_not_lower": final["conditional_candidate_top1"] >= initial["conditional_candidate_top1"],
        "candidate_assignment_declined": final["loss_candidate_assignment"]
        < initial["loss_candidate_assignment"],
        "ranking_declined": final["loss_candidate_rank"]
        < initial["loss_candidate_rank"],
        "negative_no_match_declined": final["loss_no_match_negative"]
        < initial["loss_no_match_negative"],
        "conditional_margin": margin_gate["pass"],
        "upstream_hash_unchanged": upstream_before
        == upstream_after_training
        == upstream_after_repro,
        "short_rerun_exact": reproducibility["exact"],
    }
    return {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol": {
            "seed": args.seed,
            "optimizer": "AdamW",
            "lr": args.overfit_lr,
            "weight_decay": 0.0,
            "grad_clip": args.grad_clip,
            "max_steps": args.full_subset_steps,
            "evaluation_every_steps": 30,
            "pair_order": [pair for entry in entries for pair in entry["pair_ids"]],
            "augmentation": False,
            "shuffle": False,
            "workers": 0,
            "matcher_dropout_eval_for_determinism": True,
        },
        "initial": initial,
        "final": final,
        "curve": curve,
        "reproducibility": reproducibility,
        "conditional_margin_gate": margin_gate,
        "relative_drops": {
            "matching_optim_loss": _relative_drop(
                initial["matching_optim_loss"], final["matching_optim_loss"]
            ),
            "candidate_assignment": _relative_drop(
                initial["loss_candidate_assignment"],
                final["loss_candidate_assignment"],
            ),
            "ranking": _relative_drop(
                initial["loss_candidate_rank"], final["loss_candidate_rank"]
            ),
            "negative_no_match": _relative_drop(
                initial["loss_no_match_negative"],
                final["loss_no_match_negative"],
            ),
        },
        "upstream_hash_before": upstream_before,
        "upstream_hash_after_training": upstream_after_training,
        "upstream_hash_after_repro": upstream_after_repro,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _sample_entry(entry: Mapping[str, Any], index: int) -> Dict[str, Any]:
    pair = entry["pair"]
    sample_slice = slice(index, index + 1)
    opening_a = pair.opening_all_a[sample_slice].clone()
    opening_b = pair.opening_all_b[sample_slice].clone()
    metadata = pair.metadata or {}
    pair_ids = list(entry["pair_ids"])
    sample_pair = PairBatch(
        image_a=None,
        image_b=None,
        depth_enclosed_a=pair.depth_enclosed_a[sample_slice].clone(),
        depth_enclosed_b=pair.depth_enclosed_b[sample_slice].clone(),
        depth_extended_a=pair.depth_extended_a[sample_slice].clone(),
        depth_extended_b=pair.depth_extended_b[sample_slice].clone(),
        opening_all_a=opening_a,
        opening_all_b=opening_b,
        candidates_a=opening_mask_to_candidate_masks(opening_a),
        candidates_b=opening_mask_to_candidate_masks(opening_b),
        shared_portal_a=pair.shared_portal_a[sample_slice].clone(),
        shared_portal_b=pair.shared_portal_b[sample_slice].clone(),
        affinity_ab=pair.affinity_ab[sample_slice].clone(),
        is_match=pair.is_match[sample_slice].clone(),
        transform_b_to_a=pair.transform_b_to_a[sample_slice].clone(),
        relative_yaw=pair.relative_yaw[sample_slice].clone(),
        pose_valid=pair.pose_valid[sample_slice].clone(),
        frame=pair.frame,
        scale_meters_per_coordinate=(
            pair.scale_meters_per_coordinate[sample_slice].clone()
            if pair.scale_meters_per_coordinate is not None
            else None
        ),
        metadata={
            "pair_id": [pair_ids[index]],
            "merged_pair_count": metadata["merged_pair_count"][sample_slice].clone(),
        },
    )
    return {
        "pair_ids": [pair_ids[index]],
        "live": {
            key: value[sample_slice].clone() for key, value in entry["live"].items()
        },
        "teacher": {
            key: value[sample_slice].clone()
            for key, value in entry["teacher"].items()
        },
        "pair": sample_pair,
        **{
            key: entry[key][sample_slice].clone()
            for key in (
                "candidate_masks_A",
                "candidate_masks_B",
                "candidate_valid_A",
                "candidate_valid_B",
                "candidate_target_AB",
                "opening_probability_A",
                "opening_probability_B",
            )
        },
        "gt_candidate_injection_count": 0,
    }


def _combine_samples(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(samples) != 4:
        raise ValueError("margin stress batch requires exactly four samples")
    pairs = [sample["pair"] for sample in samples]
    opening_a = torch.cat([pair.opening_all_a for pair in pairs], dim=0)
    opening_b = torch.cat([pair.opening_all_b for pair in pairs], dim=0)

    def cat_field(name: str) -> torch.Tensor:
        return torch.cat([getattr(pair, name) for pair in pairs], dim=0)

    scales = [pair.scale_meters_per_coordinate for pair in pairs]
    combined_pair = PairBatch(
        image_a=None,
        image_b=None,
        depth_enclosed_a=cat_field("depth_enclosed_a"),
        depth_enclosed_b=cat_field("depth_enclosed_b"),
        depth_extended_a=cat_field("depth_extended_a"),
        depth_extended_b=cat_field("depth_extended_b"),
        opening_all_a=opening_a,
        opening_all_b=opening_b,
        candidates_a=opening_mask_to_candidate_masks(opening_a),
        candidates_b=opening_mask_to_candidate_masks(opening_b),
        shared_portal_a=cat_field("shared_portal_a"),
        shared_portal_b=cat_field("shared_portal_b"),
        affinity_ab=cat_field("affinity_ab"),
        is_match=cat_field("is_match"),
        transform_b_to_a=cat_field("transform_b_to_a"),
        relative_yaw=cat_field("relative_yaw"),
        pose_valid=cat_field("pose_valid"),
        frame=pairs[0].frame,
        scale_meters_per_coordinate=(
            torch.cat(scales, dim=0) if all(value is not None for value in scales) else None
        ),
        metadata={
            "pair_id": [sample["pair_ids"][0] for sample in samples],
            "merged_pair_count": torch.cat(
                [pair.metadata["merged_pair_count"] for pair in pairs], dim=0
            ),
        },
    )
    return {
        "pair_ids": [sample["pair_ids"][0] for sample in samples],
        "live": {
            key: torch.cat([sample["live"][key] for sample in samples], dim=0)
            for key in samples[0]["live"]
        },
        "teacher": {
            key: torch.cat(
                [sample["teacher"][key] for sample in samples], dim=0
            )
            for key in samples[0]["teacher"]
        },
        "pair": combined_pair,
        **{
            key: torch.cat([sample[key] for sample in samples], dim=0)
            for key in (
                "candidate_masks_A",
                "candidate_masks_B",
                "candidate_valid_A",
                "candidate_valid_B",
                "candidate_target_AB",
                "opening_probability_A",
                "opening_probability_B",
            )
        },
        "gt_candidate_injection_count": 0,
    }


def run_margin_stress_audit(
    model: EndToEndOpeningMatcher,
    entries: Sequence[Mapping[str, Any]],
    device: torch.device,
    loss_args: argparse.Namespace,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    samples = [
        _sample_entry(entry, index)
        for entry in entries
        for index in range(len(entry["pair_ids"]))
    ]
    samples.sort(key=lambda sample: sample["pair_ids"][0])
    positive_active = []
    negatives = []
    selection_rows = []
    for sample in samples:
        metrics = _evaluate_cache(model, [sample], device, loss_args)
        row = {
            "pair_id": sample["pair_ids"][0],
            "is_positive": bool(sample["pair"].is_match.item()),
            "initial_margin_loss": float(
                metrics["loss_positive_dustbin_margin"]
            ),
        }
        selection_rows.append(row)
        if row["is_positive"] and row["initial_margin_loss"] > 1e-8:
            positive_active.append(sample)
        elif not row["is_positive"]:
            negatives.append(sample)
    if len(positive_active) < 2 or len(negatives) < 2:
        return {
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": protocol["protocol_sha256"],
            "selector": (
                "stable pair_id sort; first two train positives with initial "
                "margin loss>1e-8 plus first two train negatives"
            ),
            "selection_rows": selection_rows,
            "run": False,
            "reason": "insufficient deterministic stress samples",
            "test_data_accessed": False,
        }
    selected = sorted(
        positive_active[:2] + negatives[:2], key=lambda sample: sample["pair_ids"][0]
    )
    stress_entry = _combine_samples(selected)
    result = run_single_batch_gate(
        model,
        stress_entry,
        device,
        loss_args,
        args,
        frozen_artifact_exact=True,
        protocol=protocol,
    )
    result.update(
        run=True,
        role="supplemental train-only margin diagnostic; never substitutes Gate A",
        selector=(
            "stable pair_id sort; first two train positives with initial "
            "margin loss>1e-8 plus first two train negatives"
        ),
        selected_pair_ids=list(stress_entry["pair_ids"]),
        selected_positive_count=int(stress_entry["pair"].is_match.sum().item()),
        selected_negative_count=int((~stress_entry["pair"].is_match).sum().item()),
        test_data_accessed=False,
    )
    return result


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    if args.batch_size != 4:
        raise ValueError("the frozen protocol requires batch_size=4")
    if not 1 <= args.single_batch_steps <= 200:
        raise ValueError("single_batch_steps must be in [1, 200]")
    if args.full_subset_steps <= 0:
        raise ValueError("full_subset_steps must be positive")
    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("matcher-only gate requires CUDA")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.reuse_frozen_cache:
        raise ValueError(
            "v2 protocol requires --reuse_frozen_cache; rebuilding inputs is forbidden"
        )
    protocol = _protocol_v2(args)
    _atomic_json(output_dir / "protocol_v2.json", protocol)
    loss_args = _loss_args()
    entries, cache_metadata, cache_verification = _load_and_verify_frozen_cache(
        output_dir
    )
    if cache_metadata.get("t3", {}).get("sha256") != _sha256_file(
        Path(args.t3_checkpoint).expanduser().resolve()
    ):
        raise ValueError("frozen input cache was not built from the requested T3")
    model, _, t3_report = _load_t3_model(args, device)
    upstream_initial = {
        "bi_layout": _state_hash(model.bi_layout),
        "opening_head": _state_hash(model.matcher.opening_head),
    }
    determinism = dict(cache_metadata.get("determinism", {}))
    determinism["v2_cache_verification"] = cache_verification
    torch.cuda.empty_cache()
    gate_a = run_single_batch_gate(
        model,
        entries[0],
        device,
        loss_args,
        args,
        frozen_artifact_exact=all(
            cache_verification[key]
            for key in (
                "identity_unchanged",
                "tensor_signatures_exact",
                "pair_order_exact",
                "gate_A_batch_exact",
            )
        ),
        protocol=protocol,
    )
    _atomic_json(output_dir / "gate_A_single_batch_v2.json", gate_a)
    if not gate_a["pass"]:
        final = {
            "schema_version": PROTOCOL_VERSION,
            "protocol": protocol,
            "decision": "STOP_GATE_A",
            "gate_A": gate_a,
            "gate_B": None,
            "margin_stress_audit": None,
            "determinism": determinism,
            "cache_verification": cache_verification,
            "t3": t3_report,
            "upstream_initial_hash": upstream_initial,
            "full_training_started": False,
            "do_not_start_full_training": True,
            "test_data_accessed": False,
        }
        _atomic_json(output_dir / "launch_gate_report_v2.json", final)
        return 2

    model, _, _ = _load_t3_model(args, device)
    gate_b = run_full_subset_gate(
        model, entries, device, loss_args, args, protocol
    )
    _atomic_json(output_dir / "gate_B_fixed_24_v2.json", gate_b)
    upstream_final = {
        "bi_layout": _state_hash(model.bi_layout),
        "opening_head": _state_hash(model.matcher.opening_head),
    }
    margin_stress = None
    if gate_b["pass"]:
        stress_model, _, _ = _load_t3_model(args, device)
        margin_stress = run_margin_stress_audit(
            stress_model, entries, device, loss_args, args, protocol
        )
        _atomic_json(output_dir / "margin_stress_audit_v2.json", margin_stress)
        del stress_model
        torch.cuda.empty_cache()
    cache_path = output_dir / "frozen_matcher_inputs.pt"
    schema_path = output_dir / "frozen_matcher_inputs.schema.json"
    cache_identity_final = {
        "cache_sha256": _sha256_file(cache_path),
        "cache_size": cache_path.stat().st_size,
        "cache_mtime_ns": cache_path.stat().st_mtime_ns,
        "schema_sha256": _sha256_file(schema_path),
        "schema_size": schema_path.stat().st_size,
        "schema_mtime_ns": schema_path.stat().st_mtime_ns,
    }
    cache_unchanged_after_run = (
        cache_identity_final == cache_verification["identity_before"]
    )
    final = {
        "schema_version": PROTOCOL_VERSION,
        "protocol": protocol,
        "decision": "PASS_AB" if gate_b["pass"] else "STOP_GATE_B",
        "gate_A": gate_a,
        "gate_B": gate_b,
        "margin_stress_audit": margin_stress,
        "determinism": determinism,
        "cache_verification": cache_verification,
        "cache_identity_final": cache_identity_final,
        "cache_unchanged_after_run": cache_unchanged_after_run,
        "t3": t3_report,
        "upstream_initial_hash": upstream_initial,
        "upstream_final_hash": upstream_final,
        "upstream_hash_unchanged": upstream_initial == upstream_final,
        "full_training_started": False,
        "do_not_start_full_training": not gate_b["pass"],
        "next_gate": "GPU_SMALL_PREFLIGHT" if gate_b["pass"] else None,
        "test_data_accessed": False,
        "ability_boundary": {
            "legacy_recalled_positive": 1080,
            "recall_topk_recalled_positive": 1107,
            "increment": 27,
            "recalled_but_dustbin_rejected": 437,
            "recalled_but_wrong_candidate": 142,
            "proposal_missing_positive": 747,
        },
    }
    _atomic_json(output_dir / "launch_gate_report_v2.json", final)
    return 0 if gate_b["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
