#!/usr/bin/env python3
"""Prepare deterministic train/val reference-only manifests for T4 preflight."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.zind_matching_feature_dataset import ZInDMatchingFeatureDataset  # noqa: E402
from tools.train_zind_cross_scene_matcher import (  # noqa: E402
    candidate_targets_from_affinity,
    predicted_candidate_masks,
)
from utils.matcher_checkpoint import load_cross_scene_matcher_checkpoint  # noqa: E402
from utils.opening_checkpoint import load_opening_head_checkpoint  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "experiments/zind_e2e_v3/g0_seed123/T4_recall/preflight/subsets"
DEFAULT_FROZEN = REPO_ROOT / "src/output/zind_geometry_val_full/t3_frozen_m2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", default=str(REPO_ROOT / "src/dataset/ZInD_matching/train.json"))
    parser.add_argument("--val_manifest", default=str(REPO_ROOT / "src/dataset/ZInD_matching/val.json"))
    parser.add_argument("--train_cache", default=str(REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_feature_cache/train"))
    parser.add_argument("--matcher_checkpoint", default=str(REPO_ROOT / "checkpoints/Cross_Scene_Matcher/zind_matching_v1/best.pt"))
    parser.add_argument("--opening_checkpoint", default=str(REPO_ROOT / "checkpoints/Opening_Head/zind_matching_v1/best.pt"))
    parser.add_argument("--bi_layout_config", default=str(REPO_ROOT / "src/config/zind_all.yaml"))
    parser.add_argument("--bi_layout_checkpoint", default=str(REPO_ROOT / "checkpoints/Bi_Layout_Net/zind_all/zind_all_best_model.pkl"))
    parser.add_argument("--frozen_val_root", default=str(DEFAULT_FROZEN))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--scan_limit", type=int, default=2048)
    parser.add_argument("--train_per_category", type=int, default=8)
    parser.add_argument("--val_positive", type=int, default=16)
    parser.add_argument("--val_negative", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args(argv)


def _manifest_subset(payload: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result = dict(payload)
    result["pairs"] = list(records)
    result["sourceHouseIds"] = sorted({str(record["scene_id"]) for record in records})
    result["counts"] = {
        "pairCount": len(records),
        "positiveCount": sum(bool(record["supervision"]["is_match"]) for record in records),
        "negativeCount": sum(not bool(record["supervision"]["is_match"]) for record in records),
        "houseCount": len(result["sourceHouseIds"]),
    }
    result["preflightReferenceOnly"] = True
    result["testDataAccessed"] = False
    return result


def _wrap_target(record: Mapping[str, Any]) -> bool:
    if not record["supervision"]["is_match"]:
        return False
    ia = int(record["supervision"]["target_candidate_a"])
    ib = int(record["supervision"]["target_candidate_b"])
    a = record["candidates_a"][ia]["tokenInterval"]
    b = record["candidates_b"][ib]["tokenInterval"]
    return int(a[0]) > int(a[1]) or int(b[0]) > int(b[1])


def _select_train(args, train_payload):
    dataset = ZInDMatchingFeatureDataset(
        args.train_manifest,
        args.train_cache,
        max_pairs=min(args.scan_limit, len(train_payload["pairs"])),
        validate_paths=False,
        expected_split="train",
    )
    device = torch.device("cpu")
    matcher, matcher_report = load_cross_scene_matcher_checkpoint(
        args.matcher_checkpoint,
        expected_feature_dim=dataset.feature_dim,
        expected_token_count=dataset.token_count,
        expected_branch_order="extended_first",
        bi_layout_config_path=args.bi_layout_config,
        bi_layout_checkpoint_path=args.bi_layout_checkpoint,
        device=device,
    )
    opening_report = load_opening_head_checkpoint(
        matcher.opening_head,
        args.opening_checkpoint,
        expected_feature_dim=dataset.feature_dim,
        expected_branch_order="extended_first",
        expected_token_count=dataset.token_count,
        bi_layout_config_path=args.bi_layout_config,
        bi_layout_checkpoint_path=args.bi_layout_checkpoint,
    )
    matcher.eval()
    pools: Dict[str, list] = {
        "positive_recalled": [],
        "positive_missing": [],
        "negative": [],
    }
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            record = dataset.records[index]
            probabilities = []
            for side in ("A", "B"):
                signal = matcher.opening_head(
                    sample["feature_{}".format(side)].unsqueeze(0),
                    sample["depth_enclosed_{}".format(side)].unsqueeze(0),
                    sample["depth_extended_{}".format(side)].unsqueeze(0),
                )
                probabilities.append(signal["opening_probability"])
            masks_a, valid_a = predicted_candidate_masks(
                probabilities[0], 0.47, 2, 12,
                proposal_mode="recall_topk",
                peak_top_k=4,
                peak_min_probability=0.20,
                expansion_threshold=0.12,
                dedup_iou=0.50,
            )
            masks_b, valid_b = predicted_candidate_masks(
                probabilities[1], 0.47, 2, 12,
                proposal_mode="recall_topk",
                peak_top_k=4,
                peak_min_probability=0.20,
                expansion_threshold=0.12,
                dedup_iou=0.50,
            )
            affinity = sample["affinity_gt"].float().unsqueeze(0)
            target = candidate_targets_from_affinity(masks_a, affinity, masks_b)
            is_match = bool(sample["is_positive"])
            if is_match:
                category = "positive_recalled" if bool(target.any()) else "positive_missing"
            else:
                category = "negative"
            pools[category].append({
                "dataset_index": index,
                "pair_id": record["id"],
                "category": category,
                "candidate_count_A": int(valid_a.sum().item()),
                "candidate_count_B": int(valid_b.sum().item()),
                "candidate_product": int(valid_a.sum().item() * valid_b.sum().item()),
                "wrap_target": _wrap_target(record),
            })
    quota = int(args.train_per_category)
    if any(len(values) < quota for values in pools.values()):
        raise RuntimeError("insufficient deterministic train categories: {}".format({key: len(value) for key, value in pools.items()}))
    selected = []
    for category in ("positive_recalled", "positive_missing"):
        ordered = sorted(pools[category], key=lambda row: (not row["wrap_target"], -row["candidate_product"], row["pair_id"]))
        selected.extend(ordered[:quota])
    negatives = sorted(pools["negative"], key=lambda row: (-row["candidate_product"], row["pair_id"]))
    selected.extend(negatives[:quota])
    selected.sort(key=lambda row: (row["category"], row["pair_id"]))
    records = [train_payload["pairs"][row["dataset_index"]] for row in selected]
    metadata = {
        "split": "train",
        "selection_model": {"matcher": matcher_report, "opening": opening_report},
        "scan_count": len(dataset),
        "category_pool_counts": {key: len(value) for key, value in pools.items()},
        "category_counts": {key: sum(row["category"] == key for row in selected) for key in pools},
        "hard_negative_definition": "negative pairs sorted by predicted K_A*K_B descending",
        "wrap_target_count": sum(row["wrap_target"] for row in selected),
        "samples": selected,
    }
    return records, metadata


def _load_frozen(frozen_root: Path):
    reports = {}
    with (frozen_root / "per_pair.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("split") != "val" or item.get("test_data_accessed"):
                raise ValueError("frozen report is not val-only")
            reports[item["pair_id"]] = item
    artifacts = {}
    for path in (frozen_root / "frozen_predictions").glob("*.json.gz"):
        with gzip.open(path, "rt") as handle:
            item = json.load(handle)
        artifacts[item["pair_id"]] = item
    return reports, artifacts


def _baseline(records, reports, artifacts):
    positive = negative = recalled = accepted_positive = correct = rejected_negative = 0
    counts = []
    products = []
    for record in records:
        report = reports[record["id"]]
        artifact = artifacts[record["id"]]
        is_match = bool(record["supervision"]["is_match"])
        accepted = all(int(value) >= 0 for value in artifact["best_candidate_pair"])
        count_a = int(artifact["candidate_count_A"])
        count_b = int(artifact["candidate_count_B"])
        counts.extend((count_a, count_b))
        products.append(count_a * count_b)
        if is_match:
            positive += 1
            recalled += int(report["evaluation"]["gt_candidate_recalled"])
            accepted_positive += int(accepted)
            correct += int(report["evaluation"]["selected_candidate_correct"])
        else:
            negative += 1
            rejected_negative += int(not accepted)
    candidate_recall = recalled / positive
    conditional = correct / max(recalled, 1)
    end_to_end = correct / positive
    neg_rejection = rejected_negative / negative
    harmonic = 2 * end_to_end * neg_rejection / (end_to_end + neg_rejection) if end_to_end + neg_rejection else 0.0
    return {
        "source": "frozen full-val T3 artifacts filtered by exact pair id",
        "pair_count": len(records),
        "positive_count": positive,
        "negative_count": negative,
        "positive_candidate_recall": candidate_recall,
        "positive_match_recall": accepted_positive / positive,
        "negative_rejection_rate": neg_rejection,
        "conditional_candidate_top1": conditional,
        "end_to_end_candidate_top1": end_to_end,
        "harmonic_flow_score": harmonic,
        "candidate_count_mean": float(np.mean(counts)),
        "candidate_count_p95": float(np.percentile(counts, 95)),
        "candidate_product_mean": float(np.mean(products)),
        "candidate_product_p95": float(np.percentile(products, 95)),
    }


def _select_val(args, val_payload, frozen_root):
    reports, artifacts = _load_frozen(frozen_root)
    positives = [record for record in val_payload["pairs"] if record["supervision"]["is_match"]]
    negatives = [record for record in val_payload["pairs"] if not record["supervision"]["is_match"]]
    positive_order = sorted(
        positives,
        key=lambda record: (
            not _wrap_target(record),
            record["id"],
        ),
    )
    negative_order = sorted(
        negatives,
        key=lambda record: (
            -int(artifacts[record["id"]]["candidate_count_A"]) * int(artifacts[record["id"]]["candidate_count_B"]),
            record["id"],
        ),
    )
    selected = positive_order[: args.val_positive] + negative_order[: args.val_negative]
    selected.sort(key=lambda record: record["id"])
    return selected, {
        "split": "val",
        "selection_uses_model_performance": False,
        "positive_selection": "manifest-GT wrap targets first, then stable pair id; no model score or correctness is used",
        "negative_selection": "candidate burden descending, then stable pair id",
        "frozen_t3_baseline": _baseline(selected, reports, artifacts),
        "pair_ids": [record["id"] for record in selected],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.train_manifest, args.val_manifest, args.train_cache, args.frozen_val_root):
        if "test" in str(path).lower():
            raise ValueError("preflight subset preparation refuses test paths")
    train_payload = json.loads(Path(args.train_manifest).read_text())
    val_payload = json.loads(Path(args.val_manifest).read_text())
    if train_payload.get("split") != "train" or val_payload.get("split") != "val":
        raise ValueError("expected train and val manifests")
    train_records, train_meta = _select_train(args, train_payload)
    val_records, val_meta = _select_val(args, val_payload, Path(args.frozen_val_root))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_preflight.json").write_text(json.dumps(_manifest_subset(train_payload, train_records), indent=2) + "\n")
    (output / "val_preflight.json").write_text(json.dumps(_manifest_subset(val_payload, val_records), indent=2) + "\n")
    metadata = {
        "schema_version": "zind-t4-preflight-subsets-v1",
        "seed": args.seed,
        "test_data_accessed": False,
        "train": train_meta,
        "val": val_meta,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_dir": str(output),
        "train_category_counts": train_meta["category_counts"],
        "train_wrap_target_count": train_meta["wrap_target_count"],
        "val_t3_baseline": val_meta["frozen_t3_baseline"],
        "test_data_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
