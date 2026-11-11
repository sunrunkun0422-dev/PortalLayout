import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from portallayout.models.cross_scene_matcher import OpeningSignalHead
from tools.cache_zind_matching_features import _file_identity
from tools.train_zind_matching_matcher import main


class TrainZInDMatchingMatcherTest(unittest.TestCase):
    def _write_split(self, root, split, house, config, backbone):
        data = root / "data"
        data.mkdir(exist_ok=True)
        image_names = ["{}_{}.jpg".format(split, index) for index in range(4)]
        for name in image_names:
            (data / name).write_bytes(b"path validation only")
        candidate_a = {
            "type": "door",
            "index": 0,
            "tokenInterval": [1, 2],
            "globalEndpoints": [[0.0, 0.0], [1.0, 0.0]],
        }
        candidate_b = {
            "type": "door",
            "index": 0,
            "tokenInterval": [4, 5],
            "globalEndpoints": [[1.0, 0.0], [0.0, 0.0]],
        }
        records = [
            {
                "id": "{}_positive".format(split),
                "scene_id": house,
                "floor_id": "floor_01",
                "image_a": image_names[0],
                "image_b": image_names[1],
                "token_count": 8,
                "candidates_a": [candidate_a],
                "candidates_b": [candidate_b],
                "supervision": {
                    "is_match": True,
                    "target_candidate_a": 0,
                    "target_candidate_b": 0,
                    "relative_transform_b_to_a": np.eye(3).tolist(),
                    "relative_yaw_radians": 0.0,
                    "pose_valid": True,
                },
            },
            {
                "id": "{}_negative".format(split),
                "scene_id": house,
                "floor_id": "floor_01",
                "image_a": image_names[2],
                "image_b": image_names[3],
                "token_count": 8,
                "candidates_a": [candidate_a],
                "candidates_b": [candidate_b],
                "supervision": {
                    "is_match": False,
                    "target_candidate_a": -1,
                    "target_candidate_b": -1,
                    "relative_transform_b_to_a": np.eye(3).tolist(),
                    "relative_yaw_radians": 0.0,
                    "pose_valid": False,
                },
            },
        ]
        manifest = root / "manifests" / "{}.json".format(split)
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "formatVersion": 2,
                    "split": split,
                    "dataRoot": str(data),
                    "tokenCount": 8,
                    "sourceHouseIds": [house],
                    "pairs": records,
                }
            ),
            encoding="utf-8",
        )
        cache = root / "cache" / split
        cache.mkdir(parents=True)
        views = [
            {
                "index": index,
                "image_path": str(data / name),
                "relative_image_path": name,
            }
            for index, name in enumerate(image_names)
        ]
        metadata = {
            "format_version": 1,
            "task": "ZInD_matching frozen Bi-Layout complete-room feature cache",
            "split": split,
            "manifest": _file_identity(manifest, hash_contents=True),
            "bi_layout_config": _file_identity(config, hash_contents=True),
            "bi_layout_checkpoint": _file_identity(backbone, hash_contents=False),
            "image_shape": [8, 16],
            "branch_order": "extended_first",
            "sample_count": 4,
            "token_count": 8,
            "feature_dim": 4,
            "feature_dtype": "float16",
            "depth_dtype": "float32",
            "target_dtype": "uint8",
            "opening_label_types": ["door", "opening"],
            "feature_source": "frozen Bi-Layout layout_feature prediction",
            "depth_source": "frozen Bi-Layout depth/new_depth predictions",
            "uses_ground_truth_depth": False,
            "selection": {"max_images_per_split": None},
            "complete": True,
            "views": views,
        }
        (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        rng = np.random.RandomState(int(house))
        np.save(cache / "features.npy", rng.randn(4, 8, 4).astype(np.float16))
        np.save(cache / "enclosed_depth.npy", np.ones((4, 8), dtype=np.float32))
        np.save(cache / "extended_depth.npy", np.ones((4, 8), dtype=np.float32) * 2)
        return manifest, cache, metadata

    def test_complete_smoke_train_and_test_contract(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'portallayout.config.yaml'
            config.write_text("model: fake\n", encoding="utf-8")
            backbone = root / "backbone.pt"
            backbone.write_bytes(b"frozen")
            metadata = {}
            for split, house in (("train", "0001"), ("val", "0002"), ("test", "0003")):
                _, _, metadata[split] = self._write_split(
                    root, split, house, config, backbone
                )
            head = OpeningSignalHead(
                feature_dim=4,
                hidden_dim=4,
                kernel_size=5,
                prior_strength=4.0,
                prior_relative_scale=0.1,
            )
            opening = root / "opening.pt"
            torch.save(
                {
                    "format_version": 1,
                    "task": "ZInD_matching single-view Opening Head (doors+openings)",
                    "completed_epoch": 0,
                    "opening_label_types": ["door", "opening"],
                    "opening_head_state_dict": head.state_dict(),
                    "operating_threshold": 0.5,
                    "threshold_policy": "test",
                    "threshold_fallback": False,
                    "model": {
                        "feature_dim": 4,
                        "hidden_dim": 4,
                        "kernel_size": 5,
                        "prior_strength": 4.0,
                        "prior_relative_scale": 0.1,
                        "branch_order": "extended_first",
                    },
                    "train_cache_contract": metadata["train"],
                    "val_cache_contract": metadata["val"],
                },
                str(opening),
            )
            output = root / "output"
            result = main(
                [
                    "--manifest_root", str(root / "manifests"),
                    "--cache_root", str(root / "cache"),
                    "--opening_checkpoint", str(opening),
                    "--bi_layout_config", str(config),
                    "--bi_layout_checkpoint", str(backbone),
                    "--output_dir", str(output),
                    "--no_init_matcher",
                    "--device", "cpu",
                    "--workers", "0",
                    "--torch_threads", "1",
                    "--heads", "1",
                    "--hidden_dim", "4",
                    "--dropout", "0",
                    "--batch_size", "2",
                    "--progress_every", "1",
                    "--gt_guidance_epochs", "1",
                    "--mix_guidance_epochs", "0",
                    "--min_opening_width_tokens", "1",
                    "--max_openings_per_view", "4",
                    "--smoke_test",
                ]
            )
            checkpoint = torch.load(str(output / "best.pt"), map_location="cpu")
            test_metrics = json.loads((output / "test_metrics.json").read_text())

        self.assertEqual(result, 0)
        self.assertEqual(
            checkpoint["task"],
            "ZInD_matching complete-room cross-scene opening matcher",
        )
        self.assertEqual(checkpoint["opening_label_types"], ["door", "opening"])
        self.assertTrue(checkpoint["data_contract"]["camera_pose_yaw_is_not_portal_shift"])
        self.assertEqual(test_metrics["best_epoch"], 0)
        self.assertIn("predicted_openings", test_metrics)


if __name__ == "__main__":
    unittest.main()
