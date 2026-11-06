"""Read-only ZInD -> RGB-only floor manifests and offline annotation labels.

No image inference or GT-assisted model inputs. Conservative geometry proxies
are explicitly distinguished from reviewed visual labels. House transactions
support resume without appending duplicate global index records.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import random
import time
from contextlib import ExitStack
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from portallayout.datasets.zind_floor_dataset import read_json, read_jsonl, safe_path, validate_public_manifest, iter_floor_pairs
from portallayout.datasets.zind_floor_schemas import SCHEMAS
from portallayout.datasets.zind_observable_pairs import (
    stable_id, finite_positive, polygon, parse_interfaces, projection,
    endpoint_error, make_unit, geometry_relation,
)
from portallayout.datasets.zind_pair_mining import zind_transform_matrix

FAMILIES = ("layout_raw", "layout_complete", "layout_visible")
FRAME = {"frame": "zind_floor_native_xy", "units": "zind_floor_coordinates",
         "matrix_equation": "p_floor=s*R*p_local+t", "annotation_origin": "merger"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(value):
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(x) for x in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(clean_json(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(str(tmp), str(path))


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    os.replace(str(tmp), str(path))


def inspect_image(item):
    root, relative = item
    result = {"decoded": False, "sha256": None, "size_wh": None, "reason": None}
    try:
        path = safe_path(root, relative)
        result["sha256"] = sha256(path)
        with Image.open(path) as image:
            image.load()  # Full decode; header-only verify is insufficient.
            result["size_wh"] = list(image.size)
            result["mode"] = image.mode
        result["decoded"] = True
        result["aspect_ratio_valid"] = result["size_wh"][0] == 2 * result["size_wh"][1]
    except (OSError, ValueError, TypeError, Image.DecompressionBombError) as error:
        result["reason"] = type(error).__name__ + ": " + str(error)
    return result


def native_matrix(raw):
    try:
        t = raw["floor_plan_transformation"]
        if not finite_positive(t["scale"]):
            return None
        matrix = zind_transform_matrix(t)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) <= 1e-15:
            return None
        return matrix
    except (KeyError, ValueError, TypeError, OverflowError):
        return None


def source_observations(payload, house, selected_floors=None):
    for fid, rooms in sorted(payload.get("merger", {}).items()):
        if selected_floors and fid not in selected_floors:
            continue
        if not isinstance(rooms, dict):
            continue
        for cr, partials in sorted(rooms.items()):
            if not isinstance(partials, dict):
                continue
            for pr, views in sorted(partials.items()):
                if not isinstance(views, dict):
                    continue
                for pid, raw in sorted(views.items()):
                    if not isinstance(raw, dict):
                        raw = {}
                    ref = "/merger/{}/{}/{}/{}".format(fid, cr, pr, pid)
                    relative = str(Path(house) / str(raw.get("image_path", "__missing_image__")))
                    uid = "view_" + stable_id("zind-floorset-v1", house, ref, relative)
                    yield {"observation_id": uid, "house_id": house, "floor_id": fid,
                           "complete_id": cr, "partial_id": pr, "pano_id": pid,
                           "source_ref": house + "/zind_data.json#" + ref,
                           "rgb_relative_path": relative, "raw": raw}


def observation_gt(source, image, metric):
    raw = source["raw"]
    uid = source["observation_id"]
    matrix = native_matrix(raw)
    layouts, interfaces, validity, issues = {}, {}, {}, []
    for family in FAMILIES:
        layout = raw.get(family)
        layouts[family] = copy.deepcopy(layout) if isinstance(layout, dict) else None
        valid = polygon(layout.get("vertices")) is not None if isinstance(layout, dict) else False
        validity[family] = valid
        if not valid:
            issues.append(family + ":missing_or_invalid_polygon")
        parsed, errors = parse_interfaces(layout, matrix if matrix is not None else np.eye(3),
                                          source["source_ref"] + "/" + family) if isinstance(layout, dict) else ([], [family + ":missing"])
        issues.extend(errors)
        for item in parsed:
            item["interface_observation_id"] = "interface_" + stable_id(uid, item["source"])
            item["observation_id"] = uid
            item["projection"] = projection(item["local_endpoints"], item["z_bottom_top"])
            item["height_floor_z"] = ((np.asarray(item["z_bottom_top"]) + raw["camera_height"]) *
                                       float(raw["floor_plan_transformation"]["scale"])).tolist() if matrix is not None and finite_positive(raw.get("camera_height")) else None
            if matrix is None:
                item["floor_endpoints"] = None
        interfaces[family] = parsed
        validity[family + "_interface_inventory"] = isinstance(layout, dict) and not errors
    if matrix is None:
        issues.append("invalid_or_missing_gt_pose")
    cr = "room_" + stable_id(source["house_id"], source["floor_id"], source["complete_id"])
    pr = "space_" + stable_id(cr, source["partial_id"])
    height = raw.get("camera_height")
    return clean_json({
        "schema_version": "zind-floorset.observation-gt.v1", "observation_id": uid,
        "source_ref": source["source_ref"], "complete_room_uid": cr, "partial_space_uid": pr,
        "membership_status": "confirmed" if raw.get("is_inside") is True else "unconfirmed",
        "is_primary": bool(raw.get("is_primary")), "is_inside": raw.get("is_inside"),
        "is_ceiling_flat": raw.get("is_ceiling_flat"), "room_label": raw.get("label"),
        "layouts": layouts, "layout_validity": validity, "interfaces_by_layout": interfaces,
        "pose_gt_available": matrix is not None,
        "G_local_to_floor_gt": None if matrix is None else matrix.tolist(),
        "camera_height_annotation": height, "ceiling_height_annotation": raw.get("ceiling_height"),
        "camera_height_m": metric * float(raw["floor_plan_transformation"]["scale"]) * height
        if metric is not None and matrix is not None and finite_positive(height) else None,
        "coordinate_spec": {"frame": "zind_pano_local_xy", "units": "zind_local_normalized",
                            "camera_height_normalized_to_one_by_source": True},
        "metric_valid": metric is not None and matrix is not None,
        "supervision_masks": {"depth_enclosed": validity["layout_raw"],
                              "depth_extended": validity["layout_visible"],
                              "opening_raw": validity["layout_raw_interface_inventory"],
                              "pose": matrix is not None, "rgb_available": image["decoded"]},
        "annotation_origin": "source_annotation", "review_status": "unreviewed", "issues": issues,
    })


def compatible_interfaces(a, b, metric, cfg):
    if a["kind"] != b["kind"] or metric is None:
        return None
    xa, xb = np.asarray(a["floor_endpoints"]), np.asarray(b["floor_endpoints"])
    distance, reverse = endpoint_error(xa, xb)
    va, vb = xa[1] - xa[0], xb[1] - xb[0]
    wa, wb = np.linalg.norm(va), np.linalg.norm(vb)
    relative_width = abs(wa - wb) / max(wa, wb, 1e-12)
    angle = math.degrees(math.acos(float(np.clip(abs(np.dot(va, vb)) / max(wa * wb, 1e-12), -1, 1))))
    ha, hb = a.get("height_floor_z"), b.get("height_floor_z")
    height_overlap = None if ha is None or hb is None else max(0, min(ha[1], hb[1]) - max(ha[0], hb[0])) / max(1e-12, min(ha[1] - ha[0], hb[1] - hb[0]))
    if (distance * metric > cfg["endpoint_tolerance_m"] or relative_width > cfg["width_relative_tolerance"]
            or angle > cfg["direction_tolerance_deg"] or height_overlap is None
            or height_overlap < cfg["minimum_height_overlap_ratio"]):
        return None
    return {"endpoint_error_m": float(distance * metric), "reversed_b": reverse,
            "width_relative_error": float(relative_width), "direction_error_deg": angle,
            "height_overlap_ratio": float(height_overlap)}


def portal_links(observations, metric, cfg):
    """Sparse raw-interface matches; no door-open or visual-positive inference."""
    if metric is None:
        return [], []
    byid = {x["observation_id"]: x for x in observations}
    items = sorted((x for o in observations for x in o["interfaces_by_layout"]["layout_raw"]
                    if x["kind"] != "window" and x["floor_endpoints"] is not None),
                   key=lambda x: (x["observation_id"], x["interface_observation_id"]))
    cell = cfg["endpoint_tolerance_m"]
    buckets, edges = defaultdict(list), []
    parent = list(range(len(items)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, item in enumerate(items):
        center = np.asarray(item["floor_endpoints"]).mean(axis=0) * metric
        key = tuple(np.floor(center / cell).astype(int))
        for dx, dy in itertools.product((-1, 0, 1), repeat=2):
            for j in buckets[(item["kind"], key[0] + dx, key[1] + dy)]:
                other = items[j]
                if item["observation_id"] == other["observation_id"]:
                    continue
                evidence = compatible_interfaces(other, item, metric, cfg)
                if evidence is not None:
                    edges.append((j, i, evidence))
                    parent[find(i)] = find(j)
        buckets[(item["kind"], key[0], key[1])].append(i)
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    entities, group_info = [], {}
    for group in groups.values():
        ambiguous = len({items[i]["observation_id"] for i in group}) != len(group)
        # A chain of near matches must not join incompatible endpoint extremes.
        ambiguous = ambiguous or any(compatible_interfaces(items[i], items[j], metric, cfg) is None
                                     for i, j in itertools.combinations(group, 2))
        entity = "portal_" + stable_id(*sorted(items[i]["interface_observation_id"] for i in group))
        for i in group:
            group_info[i] = (entity, ambiguous)
        entities.append({"interface_uid": entity, "kind": items[group[0]]["kind"],
                         "member_interface_ids": [items[i]["interface_observation_id"] for i in group],
                         "annotation_origin": "geometry_proxy", "ambiguous": ambiguous,
                         "human_verified": False})
    pair_counts = Counter()
    for i, j, _ in edges:
        aid, bid = items[i]["observation_id"], items[j]["observation_id"]
        pair_counts[(aid, bid, items[i]["interface_observation_id"])] += 1
        pair_counts[(aid, bid, items[j]["interface_observation_id"])] += 1
    links = []
    for i, j, evidence in edges:
        a, b = items[i], items[j]
        aid, bid = a["observation_id"], b["observation_id"]
        xa, xb = np.asarray(a["floor_endpoints"]), np.asarray(b["floor_endpoints"])
        segment = (xa + (xb[::-1] if evidence["reversed_b"] else xb)) / 2
        tangent = segment[1] - segment[0]
        origin = segment.mean(axis=0)
        def side(uid):
            t = np.asarray(byid[uid]["G_local_to_floor_gt"])[:2, 2] - origin
            d = (tangent[0] * t[1] - tangent[1] * t[0]) / max(np.linalg.norm(tangent), 1e-12) * metric
            return 0 if abs(d) <= cfg["side_tolerance_m"] else int(np.sign(d))
        sa, sb = side(aid), side(bid)
        side_relation = "unknown" if not sa or not sb else "same_side" if sa == sb else "opposite_side"
        entity, ambiguous = group_info[i]
        ambiguous = ambiguous or max(pair_counts[(aid, bid, a["interface_observation_id"])],
                                     pair_counts[(aid, bid, b["interface_observation_id"])]) > 1
        links.append({"observations": [aid, bid],
                      "source_interface_a": a["interface_observation_id"],
                      "source_interface_b": b["interface_observation_id"],
                      "physical_interface_uid": entity, "type": a["kind"], "side_relation": side_relation,
                      "ambiguous": ambiguous, "quality": evidence,
                      "label_source": "raw_annotation_geometry_proxy", "review_status": "pending_review",
                      "human_verified": False})
    return links, entities


def make_rooms(sources, gt, metric, cfg):
    groups = {"complete": defaultdict(list), "partial": defaultdict(list)}
    byid = {x["observation_id"]: x for x in gt}
    for source in sources:
        record = byid[source["observation_id"]]
        groups["complete"][record["complete_room_uid"]].append(source)
        groups["partial"][record["partial_space_uid"]].append(source)
    result, units = {}, {}
    for level, grouped in groups.items():
        rooms = []
        family = "layout_complete" if level == "complete" else "layout_raw"
        for uid, members in sorted(grouped.items()):
            panos = [{"id": x["observation_id"], "raw": x["raw"], "source": x["source_ref"],
                      "primary": bool(x["raw"].get("is_primary")), "matrix": native_matrix(x["raw"])}
                     for x in members if native_matrix(x["raw"]) is not None]
            unit = make_unit(uid, uid if level == "complete" else byid[members[0]["observation_id"]]["complete_room_uid"],
                             None if level == "complete" else uid, panos, family, metric, cfg)
            if len(panos) != len(members):
                unit["inventory_complete"] = False
                unit["issues"].append("members_with_missing_gt_pose")
            canonical = next((p for p in sorted(panos, key=lambda p: (not p["primary"], p["id"]))
                              if polygon(p["raw"].get(family, {}).get("vertices")) is not None), None)
            poly = unit["polygon"]
            coherent = poly is not None and not any("polygon_disagreement" in x for x in unit["issues"])
            rooms.append({"room_uid": uid, "partition": level + "_room" if level == "complete" else "partial_space",
                          "observation_ids": sorted(x["observation_id"] for x in members),
                          "canonical_observation_id": None if canonical is None else canonical["id"],
                          "canonical_selection": "first_valid_primary_then_stable_id",
                          "polygon_floor_native": None if poly is None else list(poly.exterior.coords)[:-1],
                          "layout_family": family, "geometry_available": poly is not None,
                          "geometry_consistent": coherent, "annotation_inventory_consistent": unit["inventory_complete"],
                          "issues": unit["issues"], "source_refs": [x["source_ref"] for x in members]})
            units[uid] = unit
        result[level] = rooms
    adjacency = []
    for a, b in itertools.combinations(result["complete"], 2):
        relation = geometry_relation(units[a["room_uid"]], units[b["room_uid"]], metric, cfg)
        adjacency.append({"room_uids": [a["room_uid"], b["room_uid"]], "label": relation["adjacency"],
                          "negative_certificate": relation["negative_certificate"],
                          "matched_interface_count": len(relation["matches"]),
                          "conflicts": relation["conflicts"], "annotation_origin": "geometry_proxy",
                          "human_verified": False})
    return result["complete"], result["partial"], adjacency


def structural_graph(rooms, metric, namespace):
    """Derived partition-boundary graph, not manually verified wall surfaces."""
    vertices, walls = {}, {}
    step = 0.05 / metric if metric is not None else 1e-6
    collapsed = 0
    for room in rooms:
        points = room["polygon_floor_native"]
        if points is None:
            continue
        keys = []
        for point in points:
            key = tuple(int(x) for x in np.rint(np.asarray(point) / step))
            uid = "junction_" + stable_id(namespace, *key)
            vertices.setdefault(uid, {"junction_uid": uid, "point_floor_native": [x * step for x in key],
                                      "incident_wall_uids": [], "annotation_origin": "derived"})
            keys.append(uid)
        for i, a in enumerate(keys):
            b = keys[(i + 1) % len(keys)]
            if a == b:
                collapsed += 1
                continue
            edge = sorted((a, b))
            uid = "wall_" + stable_id(namespace, *edge)
            if uid not in walls:
                walls[uid] = {"wall_uid": uid, "junction_uids": edge, "room_uids": [],
                              "annotation_origin": "derived_partition_boundary"}
                for vertex in edge:
                    vertices[vertex]["incident_wall_uids"].append(uid)
            walls[uid]["room_uids"].append(room["room_uid"])
    return list(walls.values()), list(vertices.values()), {
        "annotation_origin": "derived", "graph_type": "snapped_partition_boundaries_not_physical_surfaces",
        "snap_step_m": 0.05 if metric is not None else None, "snap_step_native": step,
        "collapsed_segments": collapsed, "planarized_intersections": False,
        "manual_global_wall_identity": False}


def public_manifest(sample_id, floor_uid, sources, images, policy="all_panoramas"):
    observations = [{"observation_id": x["observation_id"],
                     "rgb": {"root_key": "zind_raw", "relative_path": x["rgb_relative_path"]},
                     "image_size_wh": images[x["observation_id"]]["size_wh"]}
                    for x in sorted(sources, key=lambda x: x["observation_id"])
                    if images[x["observation_id"]]["decoded"]]
    return validate_public_manifest({
        "schema_version": "zind-floorset.input.v1", "sample_id": sample_id, "floor_uid": floor_uid,
        "task": "unposed_floor_layout_from_panoramas", "input_mode": "panorama_only",
        "selection_policy": policy, "projection": "equirectangular", "pose_prior": "none",
        "calibration_mode": "scale_unknown", "observations": observations,
        "observation_order": [x["observation_id"] for x in observations]})


def build_house(root, house, output, cfg, worker_pool, selected_floors=None, materialize=False):
    source_path = safe_path(root, house + "/zind_data.json")
    payload = read_json(source_path)
    sources = list(source_observations(payload, house, selected_floors))
    images = dict(zip((x["observation_id"] for x in sources), worker_pool.map(
        inspect_image, ((root, x["rgb_relative_path"]) for x in sources))))
    grouped = defaultdict(list)
    for fid, rooms in payload.get("merger", {}).items():
        if isinstance(rooms, dict) and (not selected_floors or fid in selected_floors):
            grouped[fid] = []
    for source in sources:
        grouped[source["floor_id"]].append(source)
    result = {"house_id": house, "source_sha256": sha256(source_path), "floors": [],
              "observations": [], "source_map": [], "issues": [], "exclusions": [], "samples": defaultdict(list),
              "files": [], "image_fingerprints": {}}
    def save(relative, value, lines=False):
        (write_jsonl if lines else write_json)(output / relative, value)
        result["files"].append(relative)
    for fid, members in sorted(grouped.items()):
        floor_uid = "floor_" + stable_id(house, fid)
        sample_id = "sample_" + stable_id("all_panoramas", floor_uid)
        raw_metric = payload.get("scale_meters_per_coordinate", {}).get(fid)
        metric = float(raw_metric) if finite_positive(raw_metric) else None
        gt = sorted((observation_gt(x, images[x["observation_id"]], metric) for x in members),
                    key=lambda x: x["observation_id"])
        complete, partial, adjacency = make_rooms(members, gt, metric, cfg)
        walls, junctions, graph_info = structural_graph(complete, metric, floor_uid)
        links, entities = portal_links(gt, metric, cfg)
        manifest = public_manifest(sample_id, floor_uid, members, images)
        label_base = "labels/" + floor_uid + "/"
        floor_gt = {"schema_version": "zind-floorset.floor-gt.v1", "floor_uid": floor_uid,
                    "house_id": house, "floor_id": fid, "source_family": "merger", "coordinate_spec": FRAME,
                    "metric_scale": metric, "metric_valid": metric is not None,
                    "room_partition_default": "complete_room", "complete_rooms": complete,
                    "partial_spaces": partial,
                    "observations_membership": [{"observation_id": x["observation_id"],
                        "complete_room_uid": x["complete_room_uid"], "partial_space_uid": x["partial_space_uid"],
                        "membership_status": x["membership_status"]} for x in gt],
                    "structural_walls": walls, "junctions": junctions, "structural_graph_spec": graph_info,
                    "interfaces": entities, "room_adjacency": adjacency,
                    "target_scope": "all_annotated_rooms_on_floor", "label_status": "auto_proposed_pending_review",
                    "evaluation_masks": {"metric": metric is not None,
                        "complete_room_geometry_available": bool(complete) and all(x["geometry_available"] for x in complete),
                        "complete_room_geometry_consistent": bool(complete) and all(x["geometry_consistent"] for x in complete),
                        "human_verified_visual_correspondence": False, "manual_wall_instances": False},
                    "issues": ([] if metric is not None else ["missing_metric_scale"]),
                    "source_refs": [house + "/zind_data.json#/merger/" + fid]}
        save(label_base + "floor_gt.json", floor_gt)
        save(label_base + "observations_gt.jsonl", gt, True)
        save(label_base + "portal_links.jsonl", links, True)
        expected_pairs = len(gt) * (len(gt) - 1) // 2
        save(label_base + "pairs_policy.json", {"storage": "materialized" if materialize else "lazy_all_unordered",
             "enumeration": "sorted observation_id, itertools.combinations(ids,2)",
             "total_annotation_pairs": expected_pairs,
             "total_input_pairs": len(manifest["observations"]) * (len(manifest["observations"]) - 1) // 2,
             "canonical_sources": ["observations_gt.jsonl", "floor_gt.json", "portal_links.jsonl"],
             "visual_labels": "unknown_without_review", "prediction_status": "not_run"})
        save(label_base + "pairs_gt.jsonl", iter_floor_pairs(output / label_base) if materialize else [], True)
        input_ref = "inputs/" + sample_id + "/manifest.json"
        save(input_ref, manifest)
        sample = {"sample_id": sample_id, "floor_uid": floor_uid, "house_id": house,
                  "input_manifest": input_ref, "observation_count": len(manifest["observations"])}
        result["samples"]["all_panoramas"].append(sample)
        selections = [("primary_only", [x for x in members if x["raw"].get("is_primary")])]
        # One-per-space is deliberately GT-assisted and not the main task.
        per_space = defaultdict(list)
        for x in members:
            if images[x["observation_id"]]["decoded"]:
                per_space[(x["complete_id"], x["partial_id"])].append(x)
        selections.append(("one_per_space", [min(xs, key=lambda x: (not x["raw"].get("is_primary"), x["observation_id"]))
                                               for xs in per_space.values()]))
        usable = [x for x in members if images[x["observation_id"]]["decoded"]]
        for n in cfg["sample_view_counts"]:
            if len(usable) >= n:
                rng = random.Random(int(stable_id(cfg["seed"], floor_uid, n), 16))
                selections.append(("sampled_N_views_" + str(n), rng.sample(sorted(usable, key=lambda x: x["observation_id"]), n)))
        for policy, subset in selections:
            sid = "sample_" + stable_id(policy, floor_uid, cfg["seed"])
            mode = "sampled_N_views" if policy.startswith("sampled_N_views_") else policy
            derived = public_manifest(sid, floor_uid, subset, images, mode)
            ref = "inputs/" + sid + "/manifest.json"
            save(ref, derived)
            record = {**sample, "sample_id": sid, "input_manifest": ref,
                      "observation_count": len(derived["observations"]), "selection_policy": policy,
                      "seed": cfg["seed"], "gt_assisted_selection": policy == "one_per_space"}
            result["samples"][mode].append(record)
        floor_record = {**sample, "floor_id": fid, "label_dir": label_base.rstrip("/"),
                        "annotated_observation_count": len(gt), "selected_observation_count": len(manifest["observations"]),
                        "unreadable_count": len(gt) - len(manifest["observations"]),
                        "missing_prediction_count": len(manifest["observations"]),
                        "pair_count": expected_pairs, "metric_valid": metric is not None,
                        "complete_room_count": len(complete), "partial_space_count": len(partial),
                        "complete_geometry_consistent": floor_gt["evaluation_masks"]["complete_room_geometry_consistent"],
                        "portal_link_count": len(links)}
        result["floors"].append(floor_record)
    referenced = set()
    for source in sources:
        uid = source["observation_id"]
        image = images[uid]
        referenced.add(source["rgb_relative_path"])
        inventory = {k: v for k, v in source.items() if k != "raw"}
        inventory.update({"image": image, "is_primary": bool(source["raw"].get("is_primary")),
                          "prediction_status": "not_run"})
        result["observations"].append(inventory)
        result["source_map"].append({k: v for k, v in source.items() if k not in ("raw",)})
        if not image["decoded"]:
            result["exclusions"].append({"observation_id": uid, "reason": "rgb_unreadable", "details": image["reason"]})
        elif not image["aspect_ratio_valid"]:
            result["issues"].append({"observation_id": uid, "reason": "non_2_to_1_aspect_retained"})
        try:
            stat = safe_path(root, source["rgb_relative_path"]).stat()
            result["image_fingerprints"][source["rgb_relative_path"]] = [stat.st_size, stat.st_mtime_ns]
        except (OSError, ValueError):
            result["image_fingerprints"][source["rgb_relative_path"]] = None
    bare = Counter((s["floor_id"], s["pano_id"]) for s in sources)
    for (fid, pid), count in bare.items():
        if count > 1:
            result["issues"].append({"house_id": house, "floor_id": fid, "bare_pano_id": pid,
                                      "reason": "duplicate_bare_pano_id_preserved", "count": count})
    if not selected_floors:
        for path in sorted((root / house / "panos").glob("*")):
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                relative = path.relative_to(root).as_posix()
                if relative not in referenced:
                    result["issues"].append({"house_id": house, "relative_path": relative,
                                              "reason": "orphan_image_not_assigned_to_floor"})
    for floor in result["floors"]:
        for record in read_jsonl(output / floor["label_dir"] / "observations_gt.jsonl"):
            for issue in record["issues"]:
                result["issues"].append({"observation_id": record["observation_id"], "reason": issue})
    return clean_json(result)


def chosen_houses(partition, splits, houses=None, max_houses=None):
    if not splits or not set(splits).issubset({"train", "val", "test"}):
        raise ValueError("invalid official splits")
    lookup = {}
    for split, members in partition.items():
        for house in members:
            if house in lookup:
                raise ValueError("house occurs in multiple official splits: " + house)
            lookup[house] = split
    selected = sorted(h for h, s in lookup.items() if s in splits)
    if houses:
        if not set(houses).issubset(selected):
            raise ValueError("requested house is outside explicitly selected splits")
        selected = sorted(set(houses))
    return (selected if max_houses is None else selected[:max_houses]), lookup


def _build_house_worker(root, house, output, cfg, workers, floors, materialize):
    with ThreadPoolExecutor(max_workers=workers) as image_pool:
        return build_house(root, house, output, cfg, image_pool, floors, materialize)


def build_dataset(root, partition_path, output, config, splits=("train",), houses=None,
                  floors=None, workers=4, resume=False, materialize=False, max_houses=None, house_workers=1):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError("raw root is not a directory")
    if workers < 1 or house_workers < 1:
        raise ValueError("worker counts must be positive")
    if output == root or root in output.parents:
        raise ValueError("output must be outside the read-only raw data root")
    cfg = copy.deepcopy(config)
    partition = read_json(partition_path)
    selected, lookup = chosen_houses(partition, splits, houses, max_houses)
    if not selected:
        raise ValueError("selection is empty")
    contract = {'config': cfg, "splits": list(splits), "houses": selected, "floors": sorted(floors or []),
                "partition_sha256": sha256(partition_path), "materialize_pairs": bool(materialize),
                "raw_root": str(root), "builder_sha256": sha256(__file__),
                "dataset_reader_sha256": sha256(Path(__file__).with_name("zind_floor_dataset.py")),
                "schema_sha256": sha256(Path(__file__).with_name("zind_floor_schemas.py")),
                "reused_geometry_sha256": sha256(Path(__file__).with_name("zind_observable_pairs.py")),
                "reused_coordinates_sha256": sha256(Path(__file__).with_name("zind_pair_mining.py"))}
    marker = output / "audit/build_contract.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not marker.is_file() or read_json(marker) != contract:
            raise ValueError("nonempty output requires --resume and an identical source/config/code contract")
    output.mkdir(parents=True, exist_ok=True)
    write_json(marker, contract)
    write_json(output / "dataset_meta.json", {'dataset': "ZInD-FloorSet-v1", "status": "building",
               "predictions": "not_run", "selected_splits": list(splits)})
    for name, schema in SCHEMAS.items():
        write_json(output / ("schemas/" + name + ".schema.json"), schema)
    write_json(output / "splits/official_house_split.json", partition)
    order = sorted(partition["train"], key=lambda h: stable_id(cfg["seed"], h))
    ndev = int(math.ceil(len(order) * cfg["dev_fraction"]))
    dev = set(order[:ndev])
    write_json(output / "splits/derived_house_split.json", {"seed": cfg["seed"], "source": "official_train_only",
               "fit": sorted(set(order) - dev), "dev": sorted(dev), "official_val_test_reassigned": False})
    started = time.monotonic()
    states = []
    with ExitStack() as stack:
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
        house_pool = stack.enter_context(ProcessPoolExecutor(max_workers=house_workers)) if house_workers > 1 else None
        pending = {}
        window = 2 * house_workers
        def schedule(index):
            if house_pool is None or index >= len(selected):
                return
            h = selected[index]
            if (resume and (output / ("audit/houses/" + h + ".json")).is_file()) or not safe_path(root, h + "/zind_data.json").is_file():
                return
            pending[h] = house_pool.submit(_build_house_worker, root, h, output, cfg, workers, floors, materialize)
        for index in range(min(window, len(selected))):
            schedule(index)
        for number, house in enumerate(selected, 1):
            source = safe_path(root, house + "/zind_data.json")
            state_path = output / ("audit/houses/" + house + ".json")
            if resume and state_path.is_file():
                state = read_json(state_path)
                if not source.is_file() or sha256(source) != state["source_sha256"]:
                    raise ValueError("resume source annotation changed: " + house)
                for relative, fingerprint in state["image_fingerprints"].items():
                    try:
                        stat = safe_path(root, relative).stat()
                        current = [stat.st_size, stat.st_mtime_ns]
                    except (OSError, ValueError):
                        current = None
                    if current != fingerprint:
                        raise ValueError("resume source image changed: " + relative)
                for relative, digest in state["output_sha256"].items():
                    if not (output / relative).is_file() or sha256(output / relative) != digest:
                        raise ValueError("resume committed output changed: " + relative)
            elif source.is_file():
                state = pending.pop(house).result() if house_pool is not None else build_house(root, house, output, cfg, pool, floors, materialize)
                state["official_split"] = lookup[house]
                state["training_subset"] = "dev" if house in dev else "fit" if lookup[house] == "train" else lookup[house]
                state["output_sha256"] = {p: sha256(output / p) for p in state["files"]}
                write_json(state_path, state)
            else:
                state = {"house_id": house, "missing_source": True, "floors": [], "observations": [],
                         "source_map": [], "samples": {}, "exclusions": [{"house_id": house, "reason": "missing_house_annotation"}],
                         "issues": [], "official_split": lookup[house]}
            states.append(state)
            schedule(number - 1 + window)
            if number == 1 or number % 25 == 0 or number == len(selected):
                print("houses {}/{} floors={} observations={} elapsed={:.1f}s".format(
                    number, len(selected), sum(len(x["floors"]) for x in states),
                    sum(len(x["observations"]) for x in states), time.monotonic() - started), flush=True)
    # Rebuild rather than append aggregate indexes on resume.
    for name, key in [("floors", "floors"), ("observations", "observations"), ("source_map", "source_map")]:
        write_jsonl(output / ("index/" + name + ".jsonl"),
                    ({**r, "official_split": state["official_split"],
                      "training_subset": state.get("training_subset")} for state in states for r in state[key]))
    for view in ("all_panoramas", "primary_only", "one_per_space", "sampled_N_views"):
        write_jsonl(output / ("samples/" + view + ".jsonl"),
                    ({**r, "official_split": state["official_split"],
                      "training_subset": state.get("training_subset")} for state in states for r in state["samples"].get(view, [])))
    for name, key in [("label_issues", "issues"), ("exclusions", "exclusions")]:
        write_jsonl(output / ("audit/" + name + ".jsonl"), (r for state in states for r in state[key]))
    obs = [r for state in states for r in state["observations"]]
    fs = [r for state in states for r in state["floors"]]
    hashes = Counter(x["image"]["sha256"] for x in obs if x["image"]["sha256"])
    stats = {"selected_houses": len(selected), "available_houses": sum(not x.get("missing_source", False) for x in states),
             "floors": len(fs), "annotated_observations": len(obs),
             "input_observations": sum(x["image"]["decoded"] for x in obs),
             "unreadable_observations": sum(not x["image"]["decoded"] for x in obs),
             "primary_observations": sum(x["is_primary"] for x in obs),
             "primary_input_observations": sum(x["observation_count"] for st in states for x in st["samples"].get("primary_only", [])),
             "annotation_pairs": sum(x["pair_count"] for x in fs),
             "input_pairs": sum(x["selected_observation_count"] * (x["selected_observation_count"] - 1) // 2 for x in fs),
             "metric_floors": sum(x["metric_valid"] for x in fs),
             "complete_rooms": sum(x["complete_room_count"] for x in fs),
             "partial_spaces": sum(x["partial_space_count"] for x in fs),
             "inconsistent_complete_geometry_floors": sum(not x["complete_geometry_consistent"] for x in fs),
             "portal_links": sum(x["portal_link_count"] for x in fs),
             "duplicate_image_hash_groups": sum(c > 1 for c in hashes.values()),
             "duplicate_image_extra_observations": sum(c - 1 for c in hashes.values()),
             "pair_storage": "materialized" if materialize else "lazy_all_unordered",
             "visual_correspondences_human_verified": 0, "visual_pair_labels": "unknown_without_review",
             "predictions": "not_run", "val_test_source_accessed": bool(set(splits) & {"val", "test"}),
             "runtime_seconds_this_invocation": time.monotonic() - started}
    write_json(output / "audit/statistics.json", stats)
    write_json(output / "audit/source_hashes.json", {"partition": contract["partition_sha256"],
               "houses": {s["house_id"]: s.get("source_sha256") for s in states},
               "image_hash_records": "index/observations.jsonl", "contract": contract})
    write_json(output / "dataset_meta.json", {'dataset': "ZInD-FloorSet-v1", "schema_version": "1.0",
               "status": "complete", "statistics": stats, "selected_splits": list(splits),
               "root_binding_hint": {"zind_raw": str(root)}, "source_family": "merger",
               "prediction_status": "not_run", "pretraining_exposure": "unknown_existing_checkpoints_may_have_seen_houses",
               "label_status": "auto_proposed_pending_review", "pair_reader": 'portallayout.datasets.zind_floor_dataset.iter_floor_pairs'})
    (output / "README.md").write_text(
        "# ZInD-FloorSet-v1\n\nRGB-only all-panorama floor samples; GT is offline.\n\n"
        + "Statistics: audit/statistics.json. Exact build identity: audit/build_contract.json.\n\n"
        + "Images are referenced, never copied. Bind zind_raw explicitly on a new machine.\n"
        + "Use ZInDFloorDataset(root, raw_root, load_images=True); uint8 RGB lists are resized to 512x1024 by default.\n"
        + "Use training_subset='fit' or 'dev' to select house-disjoint training/development inputs.\n"
        + "supervision(index) and iter_pairs(index) explicitly load GT; forward only receives RGB inputs.\n\n"
        + "Default pair labels are lazy: pairs_gt.jsonl is intentionally empty; pairs_policy.json declares counts and canonical sources.\n"
        + "iter_pairs enumerates every selected unordered pair; --materialize-pairs writes the full JSONL instead.\n\n"
        + "Visual labels and covisibility are unknown, not negative. Geometry links are unreviewed proxies.\n"
        + "Pose GT availability is independent of matchability; pose supervision is disabled pending a training protocol.\n"
        + "Complete/partial geometry disagreements are retained and masked. Derived boundary graph is not manually labeled physical wall instances.\n"
        + "No predictions/checkpoints/training or final test evaluation were run by this builder.\n", encoding="utf-8")
    return stats
