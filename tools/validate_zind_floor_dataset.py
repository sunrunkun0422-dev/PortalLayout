#!/usr/bin/env python3
"""Validate floor schemas, all RGB memberships, GT references and lazy pairs."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from jsonschema import Draft7Validator
from portallayout.datasets.zind_floor_builder import sha256, write_json
from portallayout.datasets.zind_floor_dataset import read_json, read_jsonl, safe_path, validate_public_manifest, iter_floor_pairs
from portallayout.datasets.zind_floor_schemas import SCHEMAS


def validate_dataset(root, raw_root=None, check_images=False, all_pairs=False, progress=False):
    root = Path(root).resolve()
    errors, warnings = [], []
    def require(condition, message):
        if not condition:
            errors.append(message)
    meta = read_json(root / "dataset_meta.json")
    require(meta.get("status") == "complete", "build is not complete")
    partition = read_json(root / "splits/official_house_split.json")
    house_split = {}
    for split, houses in partition.items():
        for house in houses:
            require(house not in house_split, "official house split leakage: " + house)
            house_split[house] = split
    derived = read_json(root / "splits/derived_house_split.json")
    require(not set(derived["fit"]) & set(derived["dev"]), "fit/dev house leakage")
    require(set(derived["fit"]) | set(derived["dev"]) == set(partition["train"]), "derived partition mismatch")
    inventory = list(read_jsonl(root / "index/observations.jsonl"))
    by_floor = defaultdict(list)
    for observation in inventory:
        by_floor[(observation["house_id"], observation["floor_id"])].append(observation)
    inv = {x["observation_id"]: x for x in inventory}
    require(len(inv) == len(inventory), "duplicate observation IDs")
    for x in inventory:
        require(x["official_split"] == house_split[x["house_id"]], "observation official split mismatch")
    floors = list(read_jsonl(root / "index/floors.jsonl"))
    validators = {k: Draft7Validator(v) for k, v in SCHEMAS.items()}
    pair_samples = 0
    source_checks = 0
    seen = set()
    for floor_number, record in enumerate(floors, 1):
        fid = record["floor_uid"]
        require(fid not in seen, "duplicate floor UID: " + fid)
        seen.add(fid)
        manifest = read_json(safe_path(root, record["input_manifest"]))
        validate_public_manifest(manifest)
        base = safe_path(root, record["label_dir"])
        gt = read_json(base / "floor_gt.json")
        validators["floor_gt"].validate(gt)
        views = list(read_jsonl(base / "observations_gt.jsonl"))
        byid = {x["observation_id"]: x for x in views}
        require(len(byid) == len(views), fid + ": duplicate labels")
        floor_inventory = by_floor[(record["house_id"], record["floor_id"])]
        expected = sorted(x["observation_id"] for x in floor_inventory if x["image"]["decoded"])
        require(expected == manifest["observation_order"], fid + ": RGB inputs dropped or duplicated")
        require(set(byid) == {x["observation_id"] for x in floor_inventory},
                fid + ": annotation inventory mismatch")
        require(len(expected) == record["selected_observation_count"], fid + ": count mismatch")
        require(gt["metric_valid"] == (isinstance(gt["metric_scale"], (float, int)) and
                not isinstance(gt["metric_scale"], bool) and gt["metric_scale"] > 0), fid + ": invalid metric mask")
        if not gt["metric_valid"]:
            require(gt["metric_scale"] is None, fid + ": missing metric scale was imputed")
        room_ids = {r["room_uid"] for r in gt["complete_rooms"]}
        spaces = {r["room_uid"] for r in gt["partial_spaces"]}
        interfaces = set()
        for view in views:
            validators["observation_gt"].validate(view)
            uid = view["observation_id"]
            require(view["complete_room_uid"] in room_ids and view["partial_space_uid"] in spaces,
                    uid + ": unknown room reference")
            matrix = view["G_local_to_floor_gt"]
            if view["pose_gt_available"]:
                a = np.asarray(matrix)
                require(a.shape == (3, 3) and np.isfinite(a).all() and abs(np.linalg.det(a)) > 1e-12,
                        uid + ": invalid transform")
            else:
                require(matrix is None, uid + ": invalid pose must be null")
            for family in view["interfaces_by_layout"].values():
                for it in family:
                    interfaces.add(it["interface_observation_id"])
                    require(len(it["local_endpoints"]) == 2 and len(it["z_bottom_top"]) == 2,
                            uid + ": malformed triplet")
        for link in read_jsonl(base / "portal_links.jsonl"):
            require(link["observations"] == sorted(link["observations"]) and set(link["observations"]).issubset(byid),
                    fid + ": invalid portal observations")
            require(link["source_interface_a"] in interfaces and link["source_interface_b"] in interfaces,
                    fid + ": invalid portal source")
            require(not link["human_verified"], fid + ": unexpected human verification")
        policy = read_json(base / "pairs_policy.json")
        total = len(views) * (len(views) - 1) // 2
        require(policy["total_annotation_pairs"] == total, fid + ": wrong pair cardinality")
        pair_iter = iter_floor_pairs(base)
        count = 0
        for pair in pair_iter if all_pairs else itertools.islice(pair_iter, 5):
            validators["pair_gt"].validate(pair)
            count += 1
            require(pair["visual_portal_label"] == "unknown" and not pair["supervision_masks"]["visual_assignment"],
                    fid + ": unjustified visual label")
            if pair["pose_gt_available"]:
                a, b = byid[pair["observation_a"]], byid[pair["observation_b"]]
                require(np.allclose(np.asarray(a["G_local_to_floor_gt"]) @ np.asarray(pair["T_B_to_A_gt"]),
                                    b["G_local_to_floor_gt"], atol=1e-8), fid + ": wrong transform direction")
        require(count == (total if all_pairs else min(total, 5)), fid + ": lazy pair enumeration mismatch")
        pair_samples += count
        if policy["storage"] == "materialized":
            require(sum(1 for _ in read_jsonl(base / "pairs_gt.jsonl")) == total, fid + ": materialized pairs mismatch")
        else:
            require((base / "pairs_gt.jsonl").stat().st_size == 0, fid + ": unexpected lazy pair records")
        if progress and (floor_number % 100 == 0 or floor_number == len(floors)):
            print("validated floors {}/{}; pairs={}; errors={}".format(
                floor_number, len(floors), pair_samples, len(errors)), file=sys.stderr, flush=True)
    for name in ("all_panoramas", "primary_only", "one_per_space", "sampled_N_views"):
        for sample in read_jsonl(root / ("samples/" + name + ".jsonl")):
            manifest = validate_public_manifest(read_json(safe_path(root, sample["input_manifest"])))
            require(len(manifest["observations"]) == sample["observation_count"], "derived sample count mismatch")
            require(set(manifest["observation_order"]).issubset(inv), "unknown derived observation")
        if progress:
            print("validated sample view: " + name, file=sys.stderr, flush=True)
    if check_images:
        if raw_root is None:
            raise ValueError("--raw-root is required for image hash checks")
        for view in inventory:
            if not view["image"]["decoded"]:
                continue
            path = safe_path(raw_root, view["rgb_relative_path"])
            require(path.is_file(), "missing RGB: " + str(path))
            if path.is_file():
                require(sha256(path) == view["image"]["sha256"], "RGB changed: " + str(path))
                source_checks += 1
                if progress and source_checks % 5000 == 0:
                    print("verified image hashes: {}; errors={}".format(
                        source_checks, len(errors)), file=sys.stderr, flush=True)
    require(len(floors) == meta["statistics"]["floors"], "floor statistics mismatch")
    require(len(inventory) == meta["statistics"]["annotated_observations"], "observation statistics mismatch")
    return {"status": "passed" if not errors else "failed", "floor_count": len(floors),
            "observation_count": len(inventory), "pair_records_checked": pair_samples,
            "all_pair_records_checked": bool(all_pairs), "image_hashes_checked": source_checks,
            "errors": errors, "warnings": warnings, "model_inference_run": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset')
    p.add_argument("--raw-root")
    p.add_argument("--check-images", action="store_true")
    p.add_argument("--all-pairs", action="store_true")
    p.add_argument("--report")
    a = p.parse_args(argv)
    result = validate_dataset(a.dataset, a.raw_root, a.check_images, a.all_pairs, progress=True)
    if a.report:
        write_json(a.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
