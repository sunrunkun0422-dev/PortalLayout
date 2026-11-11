"""Floor RGB inputs, explicitly separate supervision, and lazy all-pair labels.

The RGB loader reads samples and input manifests only, never the label/index
directories. It returns variable-length lists; batching panoramas is a model
pipeline responsibility. No ground-truth room grouping is exposed by default.
"""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from portallayout.datasets.zind_floor_schemas import INPUT_SCHEMA
from portallayout.datasets.zind_observable_pairs import stable_id


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def safe_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("expected a relative path")
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def validate_public_manifest(manifest):
    from jsonschema import Draft7Validator
    Draft7Validator(INPUT_SCHEMA).validate(manifest)
    ids = [x["observation_id"] for x in manifest["observations"]]
    if ids != manifest["observation_order"] or len(set(ids)) != len(ids):
        raise ValueError("observation_order must enumerate unique observations exactly")
    for x in manifest["observations"]:
        p = Path(x["rgb"]["relative_path"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("unsafe RGB path")
    return manifest


class ZInDFloorDataset:
    """RGB-only floor view. Pass raw_root explicitly for portable image access."""

    def __init__(self, root, raw_root=None, sample_view="all_panoramas",
                 load_images=False, image_shape=(512, 1024), official_split=None, training_subset=None):
        self.root = Path(root).resolve()
        self.raw_root = Path(raw_root).resolve() if raw_root is not None else None
        self.samples = list(read_jsonl(safe_path(self.root, "samples/" + sample_view + ".jsonl")))
        if official_split is not None:
            self.samples = [s for s in self.samples if s["official_split"] == official_split]
        if training_subset is not None:
            self.samples = [s for s in self.samples if s["training_subset"] == training_subset]
        self.load_images = bool(load_images)
        self.image_shape = image_shape
        if self.load_images and self.raw_root is None:
            raise ValueError("raw_root is required when load_images=True")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        manifest = validate_public_manifest(read_json(safe_path(
            self.root, self.samples[index]["input_manifest"])))
        result = copy.deepcopy(manifest)
        if self.load_images:
            images = []
            for observation in manifest["observations"]:
                path = safe_path(self.raw_root, observation["rgb"]["relative_path"])
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if self.image_shape is not None:
                        h, w = self.image_shape
                        image = image.resize((w, h), Image.BICUBIC)
                    images.append(np.asarray(image).copy())
            result["images"] = images
        return result

    def supervision(self, index):
        """Explicit opt-in offline accessor; __getitem__ never calls this."""
        floor_uid = read_json(safe_path(self.root, self.samples[index]["input_manifest"]))["floor_uid"]
        base = safe_path(self.root, "labels/" + floor_uid)
        return {"offline_only": True, "floor": read_json(base / "floor_gt.json"),
                "observations": list(read_jsonl(base / "observations_gt.jsonl"))}

    def iter_pairs(self, index):
        floor_uid = read_json(safe_path(self.root, self.samples[index]["input_manifest"]))["floor_uid"]
        base = safe_path(self.root, "labels/" + floor_uid)
        manifest = read_json(safe_path(self.root, self.samples[index]["input_manifest"]))
        yield from iter_floor_pairs(base, manifest["observation_order"])


class ZInDFloorSupervisedDataset(ZInDFloorDataset):
    def __getitem__(self, index):
        # Deliberately avoid calling self[index] from supervision.
        inputs = super().__getitem__(index)
        base = safe_path(self.root, "labels/" + inputs["floor_uid"])
        return {"inputs": inputs, "targets": {"offline_only": True,
                "floor": read_json(base / "floor_gt.json"),
                "observations": list(read_jsonl(base / "observations_gt.jsonl"))}}


def iter_floor_pairs(label_dir, observation_ids=None):
    """Enumerate each unordered pair once, including missing/negative/unknown.

    The canonical label store contains observations plus sparse interface links;
    pair records are generated without model predictions or image access.
    """
    base = Path(label_dir)
    floor = read_json(base / "floor_gt.json")
    observations = {x["observation_id"]: x for x in read_jsonl(base / "observations_gt.jsonl")}
    ids = sorted(observations if observation_ids is None else observation_ids)
    if len(set(ids)) != len(ids) or not set(ids).issubset(observations):
        raise ValueError("invalid pair observation selection")
    links = {}
    for x in read_jsonl(base / "portal_links.jsonl"):
        links.setdefault(tuple(x["observations"]), []).append(x)
    adjacency = {tuple(sorted(x["room_uids"])): x for x in floor["room_adjacency"]}
    inverse = {uid: np.linalg.inv(np.asarray(x["G_local_to_floor_gt"], dtype=float))
               for uid, x in observations.items() if x["pose_gt_available"]}
    for aid, bid in itertools.combinations(ids, 2):
        a, b = observations[aid], observations[bid]
        same = a["complete_room_uid"] == b["complete_room_uid"]
        scope = ("same_partial" if a["partial_space_uid"] == b["partial_space_uid"] else
                 "same_complete_different_partial" if same else "different_complete")
        candidates = links.get((aid, bid), [])
        usable = [x for x in candidates if not x["ambiguous"]]
        relation = adjacency.get(tuple(sorted((a["complete_room_uid"], b["complete_room_uid"]))), {})
        negative = not same and relation.get("label") == "negative" and not candidates
        physical = "shared" if usable else "no_shared" if negative else "unknown"
        valid = aid in inverse and bid in inverse
        matrix = (inverse[aid] @ np.asarray(b["G_local_to_floor_gt"])) if valid else None
        yield {
            "schema_version": "zind-floorset.pair-gt.v1", "pair_id": stable_id("pair-v1", aid, bid),
            "observation_a": aid, "observation_b": bid, "relation_scope": scope,
            "same_room_gt": same, "same_room_partition": "complete_room",
            "physical_portal_label": physical, "visual_portal_label": "unknown",
            "portal_correspondences": usable, "ambiguous_portal_links": len(candidates) - len(usable),
            "covisibility": None, "pose_gt_available": valid,
            "pose_supervision_valid": False, "T_B_to_A_gt": None if matrix is None else matrix.tolist(),
            "coordinate_spec": {"frame": "zind_pano_local_xy", "units": "zind_local_normalized",
                                "transform_direction": "B_to_A", "equation": "p_A=T_B_to_A*p_B"},
            "supervision_masks": {"room_membership": a["membership_status"] == b["membership_status"] == "confirmed",
                                  "physical_portal": physical != "unknown", "visual_assignment": False,
                                  "pose_evaluation": valid, "pose_training": False},
            "evidence": relation.get("negative_certificate") if negative else None,
            "label_source": "annotation_geometry_proxy", "review_status": "pending_review",
            "human_verified": False,
        }
