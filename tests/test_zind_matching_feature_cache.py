import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.cache_zind_matching_features import (
    MatchingView,
    _contract,
    _file_identity,
    _initialize_output,
    collect_unique_views,
    copy_reusable_predictions,
    finalize_cache,
    infer_pending_predictions,
    load_reuse_cache,
)


class _FakeBiLayout(torch.nn.Module):
    patch_dim = 4

    def forward(self, images, return_features=False):
        batch_size = images.shape[0]
        base = images.mean(dim=(1, 2, 3), keepdim=False)
        feature = base[:, None, None].expand(batch_size, 8, 4).contiguous()
        extended = torch.full((batch_size, 8), 3.0, dtype=images.dtype)
        enclosed = torch.full((batch_size, 8), 2.0, dtype=images.dtype)
        return {
            "layout_feature": feature,
            "depth": extended,
            "new_depth": enclosed,
        }


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 16, 3), value, dtype=np.uint8)).save(str(path))


class ZInDMatchingFeatureCacheTest(unittest.TestCase):
    def test_collects_each_manifest_image_once_in_stable_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            for name, value in (("a.jpg", 10), ("b.jpg", 20), ("c.jpg", 30)):
                _write_image(data_root / name, value)
            manifest = root / "train.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataRoot": str(data_root),
                        "tokenCount": 8,
                        "pairs": [
                            {
                                "id": "p0",
                                "image_a": "a.jpg",
                                "image_b": "b.jpg",
                                "scene_id": "h0",
                                "floor_id": "f0",
                                "candidates_a": [
                                    {"type": "door", "tokenInterval": [7, 1]}
                                ],
                                "candidates_b": [
                                    {"type": "opening", "tokenInterval": [3, 4]}
                                ],
                            },
                            {
                                "id": "p1",
                                "image_a": "a.jpg",
                                "image_b": "c.jpg",
                                "scene_id": "h0",
                                "floor_id": "f0",
                                "candidates_a": [
                                    {"type": "door", "tokenInterval": [7, 1]}
                                ],
                                "candidates_b": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            views, token_count, resolved_root = collect_unique_views(manifest)

            self.assertEqual(token_count, 8)
            self.assertEqual(resolved_root, data_root.resolve())
            self.assertEqual([view.relative_image_path for view in views], ["a.jpg", "b.jpg", "c.jpg"])
            self.assertEqual([view.index for view in views], [0, 1, 2])
            self.assertEqual(views[0].opening_target, (1, 1, 0, 0, 0, 0, 0, 1))

    def test_rejects_inconsistent_candidates_for_the_same_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            _write_image(data_root / "a.jpg", 10)
            _write_image(data_root / "b.jpg", 20)
            _write_image(data_root / "c.jpg", 30)
            manifest = root / "train.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataRoot": str(data_root),
                        "tokenCount": 8,
                        "pairs": [
                            {
                                "id": "p0",
                                "image_a": "a.jpg",
                                "image_b": "b.jpg",
                                "candidates_a": [{"type": "door", "tokenInterval": [1, 2]}],
                                "candidates_b": [],
                            },
                            {
                                "id": "p1",
                                "image_a": "a.jpg",
                                "image_b": "c.jpg",
                                "candidates_a": [{"type": "door", "tokenInterval": [2, 3]}],
                                "candidates_b": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "inconsistent door/opening"):
                collect_unique_views(manifest)

    def test_inference_cache_contains_only_predicted_features_and_depths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.jpg"
            _write_image(image_path, 64)
            manifest = root / "test.json"
            manifest.write_text("{}", encoding="utf-8")
            config = root / 'portallayout.config.yaml'
            config.write_text("model: fake\n", encoding="utf-8")
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"fake checkpoint")
            view = MatchingView(
                index=0,
                image_path=str(image_path.resolve()),
                relative_image_path="image.jpg",
                pair_id="p0",
                side="A",
                scene_id="h0",
                floor_id="f0",
                opening_target=(1, 1, 0, 0, 0, 0, 0, 0),
            )
            cache_dir = root / "cache"
            contract = _contract(
                "test",
                [view],
                8,
                4,
                manifest,
                config,
                checkpoint,
                (8, 16),
                "extended_first",
                None,
            )
            metadata, arrays = _initialize_output(
                cache_dir, contract, [view], rebuild=False
            )

            inferred = infer_pending_predictions(
                [view],
                arrays,
                _FakeBiLayout(),
                torch.device("cpu"),
                (8, 16),
                "extended_first",
                batch_size=1,
                workers=0,
                progress_every=1,
                cache_dir=cache_dir,
                metadata=metadata,
            )
            metadata = finalize_cache(cache_dir, arrays, metadata)

            self.assertEqual(inferred, 1)
            self.assertTrue(metadata["complete"])
            self.assertFalse(metadata["uses_ground_truth_depth"])
            self.assertEqual(metadata["inferred_count"], 1)
            self.assertTrue(np.allclose(np.load(cache_dir / "enclosed_depth.npy"), 2.0))
            self.assertTrue(np.allclose(np.load(cache_dir / "extended_depth.npy"), 3.0))
            self.assertEqual(
                np.load(cache_dir / "targets.npy").tolist(),
                [[1, 1, 0, 0, 0, 0, 0, 0]],
            )
            self.assertEqual(metadata["target_dtype"], "uint8")
            self.assertEqual(metadata["opening_label_types"], ["door", "opening"])

    def test_reuses_only_a_strictly_compatible_prediction_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.jpg"
            _write_image(image_path, 128)
            manifest = root / "train.json"
            manifest.write_text("{}", encoding="utf-8")
            config = root / 'portallayout.config.yaml'
            config.write_text("model: fake\n", encoding="utf-8")
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"fake checkpoint")
            config_identity = _file_identity(config, hash_contents=True)
            checkpoint_identity = _file_identity(checkpoint, hash_contents=False)

            source = root / "source"
            source.mkdir()
            np.save(source / "features.npy", np.full((1, 8, 4), 7, dtype=np.float16))
            np.save(source / "enclosed_depth.npy", np.full((1, 8), 2, dtype=np.float32))
            np.save(source / "extended_depth.npy", np.full((1, 8), 3, dtype=np.float32))
            (source / "metadata.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "sample_count": 1,
                        "token_count": 8,
                        "feature_dim": 4,
                        "image_shape": [8, 16],
                        "branch_order": "extended_first",
                        "bi_layout_config": config_identity,
                        "bi_layout_checkpoint": checkpoint_identity,
                        "views": [{"index": 0, "image_path": str(image_path.resolve())}],
                    }
                ),
                encoding="utf-8",
            )
            reuse = load_reuse_cache(
                source,
                config_identity,
                checkpoint_identity,
                (8, 16),
                "extended_first",
                8,
                4,
            )
            view = MatchingView(
                0,
                str(image_path.resolve()),
                "image.jpg",
                "p0",
                "A",
                "h0",
                "f0",
                (0, 0, 1, 1, 0, 0, 0, 0),
            )
            target = root / "target"
            contract = _contract(
                "train",
                [view],
                8,
                4,
                manifest,
                config,
                checkpoint,
                (8, 16),
                "extended_first",
                None,
            )
            metadata, arrays = _initialize_output(target, contract, [view], False)

            copied = copy_reusable_predictions(
                [view], arrays, {str(image_path.resolve()): (reuse, 0)}
            )
            metadata = finalize_cache(target, arrays, metadata)

            self.assertEqual(copied, 1)
            self.assertEqual(metadata["reused_count"], 1)
            self.assertEqual(metadata["inferred_count"], 0)
            self.assertTrue(np.all(np.load(target / "features.npy") == 7))


if __name__ == "__main__":
    unittest.main()
