"""Behavioral tests for variable-N floor organization and RGB/GT separation."""
import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from portallayout.datasets.zind_floor_builder import build_dataset, write_json, portal_links, observation_gt, inspect_image
from portallayout.datasets.zind_floor_dataset import (
    ZInDFloorDataset, ZInDFloorSupervisedDataset, iter_floor_pairs,
    read_json, read_jsonl, safe_path, validate_public_manifest,
)
from tools.validate_zind_floor_dataset import validate_dataset


CONFIG = read_json(Path(__file__).parents[1] / "configs/zind_floorset_v1.json")


class FloorDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.raw = self.base / "raw"
        self.raw.mkdir()
        self.partition = self.base / "partition.json"
        write_json(self.partition, {"train": ["0001"], "val": ["0002"], "test": ["0003"]})

    def tearDown(self):
        self.temp.cleanup()

    def source(self, n=3, same_room=False, metric=2.0):
        rooms = {}
        for i in range(n):
            cr = "complete_0" if same_room else "complete_" + str(i)
            # Duplicate bare IDs across complete/partial paths are intentional.
            pr = "partial_" + str(i)
            path = "panos/view_" + str(i) + ".jpg"
            image = self.raw / "0001" / path
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 16), (i * 20 % 255, 40, 80)).save(image)
            layout = {"vertices": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
                      "doors": [], "openings": [], "windows": []}
            raw = {"image_path": path, "camera_height": 1.0, "ceiling_height": 2.5,
                   "floor_plan_transformation": {"translation": [0, 0] if same_room else [3 * i, 0],
                                                 "rotation": 0, "scale": 1.0},
                   "is_primary": i == 0, "is_inside": True, "is_ceiling_flat": True,
                   "layout_raw": copy.deepcopy(layout), "layout_complete": copy.deepcopy(layout),
                   "layout_visible": copy.deepcopy(layout)}
            rooms.setdefault(cr, {}).setdefault(pr, {})["pano_1"] = raw
        payload = {"merger": {"floor_01": rooms}, "scale_meters_per_coordinate": {"floor_01": metric}, "redraw": {}}
        write_json(self.raw / "0001/zind_data.json", payload)
        return payload

    def build(self, name="output", **kwargs):
        out = self.base / name
        build_dataset(self.raw, self.partition, out, CONFIG, workers=2, **kwargs)
        return out

    def first_labels(self, out):
        record = next(read_jsonl(out / "index/floors.jsonl"))
        return out / record["label_dir"]

    def test_variable_n_and_all_pairs_without_connectivity_filter(self):
        for n in (1, 2, 3, 5):
            with self.subTest(n=n):
                self.source(n)
                out = self.build("n" + str(n))
                ds = ZInDFloorDataset(out)
                self.assertEqual(len(ds[0]["observations"]), n)
                pairs = list(ds.iter_pairs(0))
                self.assertEqual(len(pairs), n * (n - 1) // 2)
                self.assertTrue(all(p["visual_portal_label"] == "unknown" for p in pairs))
                self.assertEqual(validate_dataset(out, self.raw, True, True)["status"], "passed")

    def test_same_room_views_not_duplicated_rooms(self):
        self.source(3, same_room=True)
        out = self.build()
        gt = ZInDFloorDataset(out).supervision(0)
        self.assertEqual(len(gt["floor"]["complete_rooms"]), 1)
        self.assertEqual(len(gt["floor"]["partial_spaces"]), 3)
        self.assertTrue(all(p["same_room_gt"] for p in ZInDFloorDataset(out).iter_pairs(0)))

    def test_rgb_input_survives_missing_pose_layout_and_scale(self):
        payload = self.source(2, metric=None)
        raw = payload["merger"]["floor_01"]["complete_1"]["partial_1"]["pano_1"]
        raw.pop("floor_plan_transformation")
        raw.pop("layout_raw")
        write_json(self.raw / "0001/zind_data.json", payload)
        out = self.build()
        ds = ZInDFloorDataset(out)
        self.assertEqual(len(ds[0]["observations"]), 2)
        gt = ds.supervision(0)
        self.assertIsNone(gt["floor"]["metric_scale"])
        self.assertEqual(sum(v["pose_gt_available"] for v in gt["observations"]), 1)
        pair = next(ds.iter_pairs(0))
        self.assertFalse(pair["pose_gt_available"])
        self.assertIsNone(pair["T_B_to_A_gt"])

    def test_missing_rgb_keeps_annotation_and_reason(self):
        self.source(3)
        (self.raw / "0001/panos/view_1.jpg").unlink()
        out = self.build()
        self.assertEqual(len(ZInDFloorDataset(out)[0]["observations"]), 2)
        self.assertEqual(len(ZInDFloorDataset(out).supervision(0)["observations"]), 3)
        self.assertEqual(len(list(read_jsonl(out / "audit/exclusions.jsonl"))), 1)

    def test_duplicate_bare_ids_preserved_and_input_order_stable(self):
        payload = self.source(3)
        out = self.build("first")
        ids = ZInDFloorDataset(out)[0]["observation_order"]
        self.assertEqual(len(set(ids)), 3)
        payload["merger"]["floor_01"] = dict(reversed(list(payload["merger"]["floor_01"].items())))
        write_json(self.raw / "0001/zind_data.json", payload)
        other = self.build("second")
        self.assertEqual(ids, ZInDFloorDataset(other)[0]["observation_order"])
        self.assertEqual([p["pair_id"] for p in ZInDFloorDataset(out).iter_pairs(0)],
                         [p["pair_id"] for p in ZInDFloorDataset(other).iter_pairs(0)])

    def test_input_does_not_read_gt_and_labels_are_explicit(self):
        self.source(2)
        out = self.build()
        ds = ZInDFloorDataset(out)
        original = ds[0]
        (out / "labels").rename(out / "hidden_labels")
        self.assertEqual(ds[0], original)
        (out / "hidden_labels").rename(out / "labels")
        gtpath = self.first_labels(out) / "floor_gt.json"
        gt = read_json(gtpath)
        gt["complete_rooms"] = []
        write_json(gtpath, gt)
        self.assertEqual(ds[0], original)
        self.assertEqual(ZInDFloorSupervisedDataset(out)[0]["inputs"], original)
        for key in ("room_id", "gt", "T_B_to_A_gt"):
            bad = copy.deepcopy(original)
            bad[key] = "leaked"
            with self.assertRaises(Exception):
                validate_public_manifest(bad)

    def test_image_loader_returns_rgb_without_gt(self):
        self.source(1)
        out = self.build()
        record = ZInDFloorDataset(out, self.raw, load_images=True, image_shape=(8, 16))[0]
        self.assertEqual(record["images"][0].shape, (8, 16, 3))
        self.assertEqual(record["images"][0].dtype, np.uint8)

    def test_official_nontrain_sources_not_accessed_by_default(self):
        self.source(1)
        # A malformed val/test annotation would fail immediately if opened.
        for house in ("0002", "0003"):
            path = self.raw / house / "zind_data.json"
            path.parent.mkdir()
            path.write_text("NOT JSON")
        out = self.build()
        self.assertFalse(read_json(out / "audit/statistics.json")["val_test_source_accessed"])
        with self.assertRaises(ValueError):
            self.build("bad_selection", houses=["0003"])

    def test_resume_is_idempotent_and_protects_source_and_output(self):
        self.source(2)
        out = self.build()
        before = (out / "index/observations.jsonl").read_bytes()
        with self.assertRaises(ValueError):
            self.build()
        self.build(resume=True)
        self.assertEqual((out / "index/observations.jsonl").read_bytes(), before)
        manifest = next(read_jsonl(out / "index/floors.jsonl"))["input_manifest"]
        (out / manifest).write_text("{}")
        with self.assertRaises(ValueError):
            self.build(resume=True)

    def test_materialized_equals_lazy(self):
        self.source(3)
        out = self.build(materialize=True)
        base = self.first_labels(out)
        self.assertEqual(list(read_jsonl(base / "pairs_gt.jsonl")), list(iter_floor_pairs(base)))

    def test_transform_direction_scale_and_height_are_separate(self):
        payload = self.source(2)
        r = payload["merger"]["floor_01"]["complete_1"]["partial_1"]["pano_1"]
        r["camera_height"] = 1.5
        r["floor_plan_transformation"].update({"rotation": 90, "scale": 2})
        write_json(self.raw / "0001/zind_data.json", payload)
        out = self.build()
        ds = ZInDFloorDataset(out)
        gt = {v["observation_id"]: v for v in ds.supervision(0)["observations"]}
        p = next(ds.iter_pairs(0))
        ga, gb = (np.asarray(gt[p[k]]["G_local_to_floor_gt"]) for k in ("observation_a", "observation_b"))
        t = np.asarray(p["T_B_to_A_gt"])
        np.testing.assert_allclose(ga @ t, gb, atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(t) @ t, np.eye(3), atol=1e-12)
        self.assertIn(6.0, [v["camera_height_m"] for v in gt.values()])

    def test_local_interfaces_triplets_seam_same_side_and_opposite_side(self):
        items = []
        for i, camera in enumerate(([-1, 0], [1, 0], [2, 0])):
            uid = "view_" + str(i)
            raw = {"camera_height": 1, "floor_plan_transformation": {"rotation": 0, "scale": 1, "translation": camera}}
            endpoints = (np.array([[0, -0.4], [0, 0.4]]) - camera).tolist()
            raw.update({family: {"vertices": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
                                 "doors": endpoints + [[-1, 1]], "windows": [], "openings": []}
                        for family in ("layout_raw", "layout_complete", "layout_visible")})
            source = {"observation_id": uid, "source_ref": "house#/" + uid, "house_id": "house",
                      "floor_id": "floor", "complete_id": str(i), "partial_id": str(i), "raw": raw}
            items.append(observation_gt(source, {"decoded": True}, 1))
        links, entities = portal_links(items, 1, CONFIG)
        self.assertEqual(len(links), 3)
        sides = {tuple(x["observations"]): x["side_relation"] for x in links}
        self.assertEqual(sides[("view_0", "view_1")], "opposite_side")
        self.assertEqual(sides[("view_1", "view_2")], "same_side")
        self.assertEqual(len(items[0]["interfaces_by_layout"]["layout_raw"]), 1)
        from portallayout.datasets.zind_observable_pairs import projection
        interval = projection([[-0.01, -1], [0.01, -1]])
        self.assertTrue(interval["wraps_seam"])
        self.assertLess(interval["length_u"], 0.01)

    def test_path_escape_and_missing_image_status(self):
        for value in ("../outside.jpg", "/tmp/outside.jpg"):
            with self.assertRaises(ValueError):
                safe_path(self.raw, value)
        record = inspect_image((self.raw, "../outside.jpg"))
        self.assertFalse(record["decoded"])

    def test_orphans_and_inconsistent_geometry_are_reported(self):
        payload = self.source(2, same_room=True)
        Image.new("RGB", (32, 16)).save(self.raw / "0001/panos/orphan.jpg")
        r = payload["merger"]["floor_01"]["complete_0"]["partial_1"]["pano_1"]
        r["layout_complete"]["vertices"] = [[3, 3], [4, 3], [4, 4], [3, 4]]
        write_json(self.raw / "0001/zind_data.json", payload)
        out = self.build()
        gt = ZInDFloorDataset(out).supervision(0)["floor"]
        self.assertFalse(gt["complete_rooms"][0]["geometry_consistent"])
        self.assertEqual(len(ZInDFloorDataset(out)[0]["observations"]), 2)
        self.assertTrue(any(x["reason"] == "orphan_image_not_assigned_to_floor"
                            for x in read_jsonl(out / "audit/label_issues.jsonl")))

    def test_structural_ids_are_namespaced_by_floor(self):
        from portallayout.datasets.zind_floor_builder import structural_graph
        rooms = [{"room_uid": "room_a", "polygon_floor_native": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
        w1, v1, _ = structural_graph(rooms, 1, "floor_a")
        w2, v2, _ = structural_graph(rooms, 1, "floor_b")
        self.assertFalse({x["wall_uid"] for x in w1} & {x["wall_uid"] for x in w2})
        self.assertFalse({x["junction_uid"] for x in v1} & {x["junction_uid"] for x in v2})

    def test_parallel_house_output_and_subset_filter_match(self):
        self.source(2)
        serial = self.build("serial")
        parallel = self.build("parallel", house_workers=2)
        self.assertEqual(ZInDFloorDataset(serial)[0], ZInDFloorDataset(parallel)[0])
        self.assertEqual(list(ZInDFloorDataset(serial).iter_pairs(0)), list(ZInDFloorDataset(parallel).iter_pairs(0)))
        samples = ZInDFloorDataset(parallel).samples
        subset = samples[0]["training_subset"]
        self.assertEqual(len(ZInDFloorDataset(parallel, training_subset=subset)), 1)
        self.assertEqual(len(ZInDFloorDataset(parallel, official_split="test")), 0)


if __name__ == "__main__":
    unittest.main()
