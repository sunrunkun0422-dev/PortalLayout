import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from models.cross_scene_matcher import OpeningSignalHead
from tools.cache_zind_matching_features import _file_identity
from tools.train_zind_matching_opening_head import main
from utils.opening_checkpoint import load_opening_head_checkpoint


class TrainZInDMatchingOpeningHeadTest(unittest.TestCase):
    def _write_cache(
        self,
        root,
        split,
        sample_count,
        config,
        backbone,
        feature_dim=4,
        token_count=8,
    ):
        cache = root / "cache" / split
        cache.mkdir(parents=True)
        manifest = root / "{}_manifest.json".format(split)
        manifest.write_text(json.dumps({"split": split}), encoding="utf-8")
        metadata = {
            "format_version": 1,
            "task": "ZInD_matching frozen Bi-Layout complete-room feature cache",
            "split": split,
            "manifest": _file_identity(manifest, hash_contents=True),
            "bi_layout_config": _file_identity(config, hash_contents=True),
            "bi_layout_checkpoint": _file_identity(backbone, hash_contents=False),
            "image_shape": [8, 16],
            "branch_order": "extended_first",
            "sample_count": sample_count,
            "token_count": token_count,
            "feature_dim": feature_dim,
            "feature_dtype": "float16",
            "depth_dtype": "float32",
            "target_dtype": "uint8",
            "opening_label_types": ["door", "opening"],
            "feature_source": "frozen Bi-Layout layout_feature prediction",
            "depth_source": "frozen Bi-Layout depth/new_depth predictions",
            "uses_ground_truth_depth": False,
            "selection": {"max_images_per_split": None},
            "complete": True,
            "views": [
                {"index": index, "image_path": "{}_{}.jpg".format(split, index)}
                for index in range(sample_count)
            ],
        }
        (cache / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        generator = np.random.RandomState(10 if split == "train" else 20)
        np.save(
            cache / "features.npy",
            generator.randn(sample_count, token_count, feature_dim).astype(np.float16),
        )
        np.save(
            cache / "enclosed_depth.npy",
            np.ones((sample_count, token_count), dtype=np.float32),
        )
        extended = np.ones((sample_count, token_count), dtype=np.float32)
        targets = np.zeros((sample_count, token_count), dtype=np.uint8)
        for index in range(sample_count):
            start = index % (token_count - 2)
            targets[index, start : start + 2] = 1
            extended[index, start : start + 2] = 1.8
        np.save(cache / "extended_depth.npy", extended)
        np.save(cache / "targets.npy", targets)
        return cache, metadata

    def _write_warm_start(self, path, config, backbone):
        head = OpeningSignalHead(
            feature_dim=4,
            hidden_dim=4,
            kernel_size=3,
            prior_strength=4.0,
            prior_relative_scale=0.1,
        )
        torch.save(
            {
                "format_version": 1,
                "task": "source BiPair Opening Head",
                "completed_epoch": 3,
                "opening_head_state_dict": head.state_dict(),
                "operating_threshold": 0.4,
                "threshold_policy": "test",
                "threshold_fallback": False,
                "model": {
                    "feature_dim": 4,
                    "hidden_dim": 4,
                    "kernel_size": 3,
                    "prior_strength": 4.0,
                    "prior_relative_scale": 0.1,
                    "branch_order": "extended_first",
                },
                "train_cache_contract": {
                    "feature_dim": 4,
                    "token_count": 8,
                    "bi_layout_config": _file_identity(config, hash_contents=True),
                    "bi_layout_checkpoint": _file_identity(
                        backbone, hash_contents=False
                    ),
                },
            },
            str(path),
        )

    def test_smoke_warm_start_resume_and_loader_contract(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text("model: fake\n", encoding="utf-8")
            backbone = root / "backbone.pt"
            backbone.write_bytes(b"frozen backbone identity")
            train_cache, _ = self._write_cache(
                root, "train", 4, config, backbone
            )
            val_cache, _ = self._write_cache(root, "val", 3, config, backbone)
            warm = root / "warm.pt"
            self._write_warm_start(warm, config, backbone)
            output = root / "output"
            common = [
                "--train_cache",
                str(train_cache),
                "--val_cache",
                str(val_cache),
                "--output_dir",
                str(output),
                "--warm_start",
                str(warm),
                "--device",
                "cpu",
                "--workers",
                "0",
                "--batch_size",
                "2",
                "--hidden_dim",
                "4",
                "--kernel_size",
                "3",
                "--scan_steps",
                "5",
                "--roll_probability",
                "0",
                "--flip_probability",
                "0",
            ]

            self.assertEqual(main(common + ["--smoke_test"]), 0)
            first = torch.load(str(output / "last.pt"), map_location="cpu")
            self.assertEqual(first["completed_epoch"], 0)
            self.assertEqual(
                first["task"],
                "ZInD_matching single-view Opening Head (doors+openings)",
            )
            self.assertEqual(first["opening_label_types"], ["door", "opening"])
            self.assertTrue(first["warm_start"]["loaded"])
            self.assertTrue(first["warm_start"]["optimizer_reset"])
            self.assertTrue(first["warm_start"]["epoch_reset"])

            loaded_head = OpeningSignalHead(
                feature_dim=4,
                hidden_dim=4,
                kernel_size=3,
                prior_strength=4.0,
                prior_relative_scale=0.1,
            )
            report = load_opening_head_checkpoint(
                loaded_head,
                str(output / "best.pt"),
                expected_feature_dim=4,
                expected_branch_order="extended_first",
                expected_token_count=8,
                bi_layout_config_path=str(config),
                bi_layout_checkpoint_path=str(backbone),
            )
            self.assertTrue(report["loaded"])
            self.assertIn("doors+openings", report["task"])

            self.assertEqual(
                main(
                    common
                    + [
                        "--resume",
                        "--epochs",
                        "2",
                        "--max_train_views",
                        "16",
                        "--max_val_views",
                        "8",
                    ]
                ),
                0,
            )
            resumed = torch.load(str(output / "last.pt"), map_location="cpu")
            self.assertEqual(resumed["completed_epoch"], 1)
            self.assertEqual(resumed["global_step"], 4)

            metadata_path = val_cache / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["selection"] = {"max_images_per_split": 99}
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "val_cache_contract contract changed"):
                main(
                    common
                    + [
                        "--resume",
                        "--epochs",
                        "3",
                        "--max_train_views",
                        "16",
                        "--max_val_views",
                        "8",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
