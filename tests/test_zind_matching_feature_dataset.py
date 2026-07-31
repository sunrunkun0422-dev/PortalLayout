import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from dataset.zind_bipair_dataset import collate_zind_bipair
from dataset.zind_matching_feature_dataset import ZInDMatchingFeatureDataset
from models.cross_scene_matcher import OpeningGuidedCrossAttentionMatcher
from tools.train_zind_cross_scene_matcher import _forward_batch


def _candidate(interval, endpoints, kind="door"):
    return {
        "type": kind,
        "index": 0,
        "tokenInterval": list(interval),
        "globalEndpoints": endpoints,
    }


class ZInDMatchingFeatureDatasetTest(unittest.TestCase):
    def test_preserves_explicit_candidates_and_multi_interface_targets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            manifest = root / "train.json"
            candidates_a = [
                _candidate((0, 2), [[0.0, 0.0], [1.0, 0.0]]),
                _candidate((2, 4), [[0.0, 2.0], [1.0, 2.0]]),
            ]
            candidates_b = [
                _candidate((4, 6), [[1.0, 0.0], [0.0, 0.0]]),
                _candidate((6, 0), [[1.0, 2.0], [0.0, 2.0]]),
            ]
            record = {
                "id": "positive",
                "scene_id": "0000",
                "floor_id": "floor_01",
                "image_a": "a.jpg",
                "image_b": "b.jpg",
                "token_count": 8,
                "candidates_a": candidates_a,
                "candidates_b": candidates_b,
                "supervision": {
                    "is_match": True,
                    "target_candidate_a": 0,
                    "target_candidate_b": 0,
                    "relative_transform_b_to_a": np.eye(3).tolist(),
                    "relative_yaw_radians": 0.0,
                    "pose_valid": True,
                },
            }
            manifest.write_text(
                json.dumps(
                    {
                        "split": "train",
                        "dataRoot": str(root),
                        "tokenCount": 8,
                        "sourceHouseIds": ["0000"],
                        "pairs": [record],
                    }
                ),
                encoding="utf-8",
            )
            views = [
                {"index": 0, "image_path": str(root / "a.jpg"), "relative_image_path": "a.jpg"},
                {"index": 1, "image_path": str(root / "b.jpg"), "relative_image_path": "b.jpg"},
            ]
            (cache / "metadata.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "split": "train",
                        "sample_count": 2,
                        "token_count": 8,
                        "feature_dim": 4,
                        "branch_order": "extended_first",
                        "views": views,
                    }
                ),
                encoding="utf-8",
            )
            np.save(cache / "features.npy", np.zeros((2, 8, 4), dtype=np.float16))
            np.save(cache / "enclosed_depth.npy", np.ones((2, 8), dtype=np.float32))
            np.save(cache / "extended_depth.npy", np.ones((2, 8), dtype=np.float32) * 2)

            dataset = ZInDMatchingFeatureDataset(
                str(manifest), str(cache), validate_paths=False
            )
            sample = dataset[0]
            batch = collate_zind_bipair([sample])
            matcher = OpeningGuidedCrossAttentionMatcher(
                feature_dim=4, heads=1, hidden_dim=4, dropout=0.0
            )
            outputs, _, target, valid_a, valid_b = _forward_batch(
                matcher,
                batch,
                torch.device("cpu"),
                guidance_mode="gt",
                mix_weight=0.5,
                candidate_source="gt",
                opening_threshold=0.5,
                min_width_tokens=1,
                max_openings=4,
            )

        self.assertEqual(sample["merged_pair_count"].item(), 2)
        self.assertFalse(sample["portal_token_shift_valid"].item())
        self.assertEqual(sample["candidate_target_AB"].tolist(), [[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(sample["affinity_gt"][0, 4])
        self.assertTrue(sample["affinity_gt"][3, 7])
        self.assertFalse(sample["affinity_gt"][0, 7])
        self.assertEqual(target[0].tolist(), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(valid_a.tolist(), [[True, True]])
        self.assertEqual(valid_b.tolist(), [[True, True]])
        self.assertEqual(tuple(outputs["candidate_pair_score"].shape), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
