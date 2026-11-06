"""Frozen-feature view of the cross-complete-room ``ZInD_matching`` data.

The compact matching manifests store circular opening intervals and image
references.  This module joins those records to one frozen Bi-Layout cache and
expands the supervision into the legacy dense fields consumed by
``adapt_zind_bipair_batch`` and the current matcher trainer.

Camera pose yaw and portal token shift are intentionally separate labels.  The
former belongs to ``T_B_to_A``; the latter is the circular B-token minus
A-token displacement used by the matcher's cyclic-shift branch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from portallayout.datasets.zind_pair_mining import segment_endpoint_error


_CACHE_FILES = {
    "features": "features.npy",
    "enclosed_depth": "enclosed_depth.npy",
    "extended_depth": "extended_depth.npy",
}
_VALID_SPLITS = frozenset(("train", "val", "test"))


def _resolve_root(value: Optional[str], manifest_dir: Path) -> Path:
    if value is None:
        return manifest_dir.resolve()
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _interval_mask(interval: Sequence[int], token_count: int) -> torch.Tensor:
    if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)):
        raise ValueError("candidate token interval must be a two-value sequence")
    if len(interval) != 2:
        raise ValueError("candidate token interval must contain exactly two values")
    start, end = int(interval[0]), int(interval[1])
    if not 0 <= start < token_count or not 0 <= end < token_count:
        raise ValueError(
            "candidate token interval [{}, {}] is outside [0, {})".format(
                start, end, token_count
            )
        )
    mask = torch.zeros(token_count, dtype=torch.bool)
    if start <= end:
        mask[start : end + 1] = True
    else:
        mask[start:] = True
        mask[: end + 1] = True
    return mask


def _portal_shift_radians(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """Circular portal-center displacement, defined as B token minus A token."""
    if mask_a.ndim != 1 or mask_b.shape != mask_a.shape:
        raise ValueError("portal masks must have the same one-dimensional shape")
    if not bool(mask_a.any()) or not bool(mask_b.any()):
        return 0.0
    token_count = int(mask_a.numel())
    angles = torch.arange(token_count, dtype=torch.float64) * (
        2.0 * math.pi / token_count
    )

    def center(mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=torch.float64)
        return torch.atan2(
            (weights * torch.sin(angles)).sum(),
            (weights * torch.cos(angles)).sum(),
        )

    difference = center(mask_b) - center(mask_a)
    return float(torch.atan2(torch.sin(difference), torch.cos(difference)).item())


class ZInDMatchingFeatureDataset(Dataset):
    """Join a ``ZInD_matching`` split with frozen single-panorama features.

    The returned mapping is directly stackable with
    :func:`dataset.zind_bipair_dataset.collate_zind_bipair` and consumable by
    :func:`dataset.zind_bipair_adapter.adapt_zind_bipair_batch`.

    ``manifest_candidate_masks_*`` preserves the manifest's candidate order,
    including touching or overlapping intervals.  ``opening_mask_all_*`` is
    their union and remains the only opening guidance label.  On negative
    pairs the shared portal masks and dense affinity are all zero, providing
    explicit dustbin/no-match supervision.
    """

    def __init__(
        self,
        manifest_path: str,
        feature_cache_dir: str,
        data_root: Optional[str] = None,
        max_pairs: Optional[int] = None,
        validate_paths: bool = True,
        expected_split: Optional[str] = None,
        endpoint_match_tolerance: float = 0.06,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                "ZInD matching manifest does not exist: {}".format(self.manifest_path)
            )
        if max_pairs is not None and (type(max_pairs) is not int or max_pairs <= 0):
            raise ValueError("max_pairs must be a positive integer when provided")
        self.max_pairs = max_pairs
        self.endpoint_match_tolerance = float(endpoint_match_tolerance)
        if not math.isfinite(self.endpoint_match_tolerance) or self.endpoint_match_tolerance <= 0:
            raise ValueError("endpoint_match_tolerance must be finite and positive")

        with self.manifest_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, Mapping):
            raise ValueError("ZInD matching manifest must be a JSON object")
        records = payload.get("pairs")
        if not isinstance(records, list) or not records:
            raise ValueError("ZInD matching manifest must contain non-empty pairs")
        if not all(isinstance(record, Mapping) for record in records):
            raise ValueError("every ZInD matching pair must be an object")
        self.records = records
        self.dataset_contract = payload.get("datasetContract")

        self.split = str(payload.get("split") or self.manifest_path.stem)
        if self.split not in _VALID_SPLITS:
            raise ValueError("ZInD matching split must be train, val, or test")
        if expected_split is not None and self.split != str(expected_split):
            raise ValueError(
                "manifest split {!r} does not match expected split {!r}".format(
                    self.split, expected_split
                )
            )
        self.data_root = _resolve_root(
            data_root if data_root is not None else payload.get("dataRoot"),
            self.manifest_path.parent,
        )
        self.token_count = int(payload.get("tokenCount", 0))
        if self.token_count <= 0:
            raise ValueError("ZInD matching manifest tokenCount must be positive")

        record_house_ids = {str(record.get("scene_id", "")) for record in records}
        if "" in record_house_ids:
            raise ValueError("every ZInD matching pair must contain scene_id")
        declared_houses = payload.get("sourceHouseIds")
        if declared_houses is not None:
            if not isinstance(declared_houses, list):
                raise ValueError("sourceHouseIds must be a list when provided")
            declared_house_ids = {str(value) for value in declared_houses}
            undeclared = record_house_ids - declared_house_ids
            if undeclared:
                raise ValueError(
                    "pair records contain houses outside sourceHouseIds: {}".format(
                        ", ".join(sorted(undeclared)[:3])
                    )
                )
            self.house_ids = frozenset(declared_house_ids)
        else:
            self.house_ids = frozenset(record_house_ids)

        selected_records = records[: len(self)]
        self.max_candidates = max(
            1,
            max(
                max(len(record.get("candidates_a", ())), len(record.get("candidates_b", ())))
                for record in selected_records
            ),
        )
        self._validate_records(selected_records, validate_paths=validate_paths)

        self.cache_dir = Path(feature_cache_dir).expanduser().resolve()
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "frozen feature cache metadata does not exist: {}".format(metadata_path)
            )
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, Mapping) or metadata.get("complete") is not True:
            raise ValueError("frozen feature cache is incomplete: {}".format(self.cache_dir))
        self.metadata = dict(metadata)
        cache_split = self.metadata.get("split")
        if cache_split is not None and str(cache_split) != self.split:
            raise ValueError(
                "feature cache split {!r} does not match manifest split {!r}".format(
                    cache_split, self.split
                )
            )
        self.sample_count = int(self.metadata.get("sample_count", 0))
        cache_token_count = int(self.metadata.get("token_count", 0))
        self.feature_dim = int(self.metadata.get("feature_dim", 0))
        if min(self.sample_count, cache_token_count, self.feature_dim) <= 0:
            raise ValueError("feature cache metadata has invalid tensor dimensions")
        if cache_token_count != self.token_count:
            raise ValueError(
                "feature cache token_count {} does not match manifest {}".format(
                    cache_token_count, self.token_count
                )
            )
        if self.metadata.get("branch_order") not in (None, "extended_first"):
            raise ValueError("matcher requires the extended_first feature-cache contract")

        views = self.metadata.get("views")
        if not isinstance(views, list) or len(views) != self.sample_count:
            raise ValueError("feature cache metadata views do not match sample_count")
        self._view_index: Dict[str, int] = {}
        for index, view in enumerate(views):
            if not isinstance(view, Mapping):
                raise ValueError("feature cache view metadata must be objects")
            path_values = [
                view.get(key)
                for key in ("image_path", "relative_image_path")
                if view.get(key)
            ]
            if not path_values:
                raise ValueError("feature cache view is missing an image path")
            for value in path_values:
                normalized = self._normalize_path(value)
                previous = self._view_index.setdefault(normalized, index)
                if previous != index:
                    raise ValueError(
                        "feature cache contains duplicate image path: {}".format(value)
                    )

        self._array_paths = {
            name: self.cache_dir / filename for name, filename in _CACHE_FILES.items()
        }
        missing_arrays = [
            str(path) for path in self._array_paths.values() if not path.is_file()
        ]
        if missing_arrays:
            raise FileNotFoundError(
                "frozen feature cache is missing arrays: {}".format(
                    ", ".join(missing_arrays)
                )
            )
        self._features = None
        self._enclosed_depth = None
        self._extended_depth = None
        self._open_arrays_and_validate()
        self._validate_pair_coverage(selected_records)

    def _normalize_path(self, value: Any) -> str:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.data_root / path
        return str(path.resolve())

    def _validate_records(
        self, records: Sequence[Mapping[str, Any]], validate_paths: bool
    ) -> None:
        missing_paths = []
        seen_ids = set()
        for record in records:
            pair_id = str(record.get("id", ""))
            if not pair_id:
                raise ValueError("every ZInD matching pair must contain id")
            if pair_id in seen_ids:
                raise ValueError("duplicate ZInD matching pair id: {}".format(pair_id))
            seen_ids.add(pair_id)
            if int(record.get("token_count", self.token_count)) != self.token_count:
                raise ValueError("pair {} has a mismatched token_count".format(pair_id))
            for side in ("a", "b"):
                image_key = "image_{}".format(side)
                if not record.get(image_key):
                    raise ValueError("pair {} is missing {}".format(pair_id, image_key))
                candidates = record.get("candidates_{}".format(side))
                if not isinstance(candidates, list):
                    raise ValueError("pair {} candidates_{} must be a list".format(pair_id, side))
                for candidate in candidates:
                    if not isinstance(candidate, Mapping):
                        raise ValueError("pair {} candidates must be objects".format(pair_id))
                    _interval_mask(candidate.get("tokenInterval"), self.token_count)
                if validate_paths:
                    image_path = Path(self._normalize_path(record[image_key]))
                    if not image_path.is_file():
                        missing_paths.append(str(image_path))
            supervision = record.get("supervision")
            if not isinstance(supervision, Mapping):
                raise ValueError("pair {} supervision must be an object".format(pair_id))
            is_match = bool(supervision.get("is_match", False))
            target_a = int(supervision.get("target_candidate_a", -1))
            target_b = int(supervision.get("target_candidate_b", -1))
            target_valid = (
                0 <= target_a < len(record["candidates_a"])
                and 0 <= target_b < len(record["candidates_b"])
            )
            if is_match != target_valid:
                raise ValueError(
                    "pair {} positive label and target candidate indices disagree".format(
                        pair_id
                    )
                )
            transform = np.asarray(
                supervision.get("relative_transform_b_to_a", np.eye(3)),
                dtype=np.float32,
            )
            if transform.shape != (3, 3) or not np.isfinite(transform).all():
                raise ValueError("pair {} transform must be finite [3, 3]".format(pair_id))
            if len(missing_paths) >= 3:
                break
        if missing_paths:
            raise FileNotFoundError(
                "ZInD matching manifest references missing images: {}".format(
                    ", ".join(missing_paths[:3])
                )
            )

    def _open_arrays_and_validate(self) -> None:
        self._features = np.load(
            str(self._array_paths["features"]), mmap_mode="r", allow_pickle=False
        )
        self._enclosed_depth = np.load(
            str(self._array_paths["enclosed_depth"]), mmap_mode="r", allow_pickle=False
        )
        self._extended_depth = np.load(
            str(self._array_paths["extended_depth"]), mmap_mode="r", allow_pickle=False
        )
        expected_features = (self.sample_count, self.token_count, self.feature_dim)
        expected_depth = (self.sample_count, self.token_count)
        if tuple(self._features.shape) != expected_features:
            raise ValueError(
                "cached features have shape {}, expected {}".format(
                    tuple(self._features.shape), expected_features
                )
            )
        for name, value in (
            ("enclosed_depth", self._enclosed_depth),
            ("extended_depth", self._extended_depth),
        ):
            if tuple(value.shape) != expected_depth:
                raise ValueError(
                    "cached {} has shape {}, expected {}".format(
                        name, tuple(value.shape), expected_depth
                    )
                )

    def _validate_pair_coverage(self, records: Sequence[Mapping[str, Any]]) -> None:
        missing = []
        for record in records:
            for side in ("a", "b"):
                path = self._normalize_path(record["image_{}".format(side)])
                if path not in self._view_index:
                    missing.append(path)
                    if len(missing) >= 3:
                        break
            if len(missing) >= 3:
                break
        if missing:
            raise ValueError(
                "feature cache does not cover ZInD matching views: {}".format(
                    ", ".join(missing)
                )
            )

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_features"] = None
        state["_enclosed_depth"] = None
        state["_extended_depth"] = None
        return state

    def _ensure_arrays_open(self) -> None:
        if self._features is None:
            self._open_arrays_and_validate()

    def __len__(self) -> int:
        length = len(self.records)
        return length if self.max_pairs is None else min(length, self.max_pairs)

    @staticmethod
    def _tensor(row: np.ndarray, name: str, pair_id: str) -> torch.Tensor:
        value = np.array(row, dtype=np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError(
                "cached {} for pair {} contains non-finite values".format(name, pair_id)
            )
        return torch.from_numpy(value)

    def _candidate_masks(
        self, record: Mapping[str, Any], side: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = torch.zeros(
            (self.max_candidates, self.token_count), dtype=torch.bool
        )
        candidates = record["candidates_{}".format(side)]
        for index, candidate in enumerate(candidates):
            masks[index] = _interval_mask(candidate["tokenInterval"], self.token_count)
        valid = torch.zeros(self.max_candidates, dtype=torch.bool)
        valid[: len(candidates)] = True
        return masks, valid

    def _matching_candidate_pairs(
        self, record: Mapping[str, Any]
    ) -> Tuple[Tuple[int, int], ...]:
        """Return every physically shared interface, preserving candidate IDs."""
        supervision = record["supervision"]
        if "correspondence_pairs" in supervision:
            # Versioned datasets can provide a verified offline GT set. Never
            # re-mine it with this legacy loader's unrelated coordinate threshold.
            pairs = tuple(tuple(pair) for pair in supervision["correspondence_pairs"])
            if len(set(pairs)) != len(pairs) or any(
                len(pair) != 2 or any(type(i) is not int for i in pair)
                or not (0 <= pair[0] < len(record["candidates_a"]) and
                        0 <= pair[1] < len(record["candidates_b"])) for pair in pairs
            ):
                raise ValueError("invalid explicit correspondence set")
            if bool(pairs) != bool(supervision["is_match"]):
                raise ValueError("explicit correspondence set contradicts adjacency")
            if pairs and (int(supervision["target_candidate_a"]), int(supervision["target_candidate_b"])) not in pairs:
                raise ValueError("primary target is missing from explicit GT set")
            return pairs
        if not bool(supervision["is_match"]):
            return ()
        matches = []
        for index_a, candidate_a in enumerate(record["candidates_a"]):
            endpoints_a = np.asarray(candidate_a.get("globalEndpoints"), dtype=np.float64)
            if endpoints_a.shape != (2, 2) or not np.isfinite(endpoints_a).all():
                raise ValueError(
                    "pair {} candidate A {} has invalid globalEndpoints".format(
                        record["id"], index_a
                    )
                )
            for index_b, candidate_b in enumerate(record["candidates_b"]):
                endpoints_b = np.asarray(candidate_b.get("globalEndpoints"), dtype=np.float64)
                if endpoints_b.shape != (2, 2) or not np.isfinite(endpoints_b).all():
                    raise ValueError(
                        "pair {} candidate B {} has invalid globalEndpoints".format(
                            record["id"], index_b
                        )
                    )
                if (
                    segment_endpoint_error(endpoints_a, endpoints_b)
                    <= self.endpoint_match_tolerance
                ):
                    matches.append((index_a, index_b))
        manifest_target = (
            int(supervision["target_candidate_a"]),
            int(supervision["target_candidate_b"]),
        )
        if manifest_target not in matches:
            raise ValueError(
                "pair {} manifest target exceeds endpoint tolerance {}".format(
                    record["id"], self.endpoint_match_tolerance
                )
            )
        return tuple(matches)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        self._ensure_arrays_open()
        record = self.records[index]
        pair_id = str(record["id"])
        supervision = record["supervision"]
        is_match = bool(supervision["is_match"])
        candidate_masks_a, candidate_valid_a = self._candidate_masks(record, "a")
        candidate_masks_b, candidate_valid_b = self._candidate_masks(record, "b")
        opening_a = candidate_masks_a[candidate_valid_a].any(dim=0)
        opening_b = candidate_masks_b[candidate_valid_b].any(dim=0)
        portal_a = torch.zeros(self.token_count, dtype=torch.bool)
        portal_b = torch.zeros(self.token_count, dtype=torch.bool)
        target_a = int(supervision.get("target_candidate_a", -1))
        target_b = int(supervision.get("target_candidate_b", -1))
        target_pairs = self._matching_candidate_pairs(record)
        candidate_target = torch.zeros(
            (self.max_candidates, self.max_candidates), dtype=torch.float32
        )
        affinity = torch.zeros(
            (self.token_count, self.token_count), dtype=torch.uint8
        )
        for matched_a, matched_b in target_pairs:
            mask_a = candidate_masks_a[matched_a]
            mask_b = candidate_masks_b[matched_b]
            portal_a |= mask_a
            portal_b |= mask_b
            candidate_target[matched_a, matched_b] = 1.0
            affinity |= torch.outer(mask_a, mask_b).to(dtype=torch.uint8)
        portal_shift_valid = len(target_pairs) == 1
        portal_shift = (
            _portal_shift_radians(portal_a, portal_b)
            if portal_shift_valid
            else 0.0
        )

        image_path_a = self._normalize_path(record["image_a"])
        image_path_b = self._normalize_path(record["image_b"])
        output: Dict[str, Any] = {
            "pair_id": pair_id,
            "pair_type": str(record.get("relation_scope", "cross_complete_room")) + ("_positive" if is_match else "_negative"),
            "house_id": str(record["scene_id"]),
            "floor_id": str(record.get("floor_id", "")),
            "complete_room_id": "{}<->{}".format(
                record.get("room_a", ""), record.get("room_b", "")
            ),
            "image_path_A": image_path_a,
            "image_path_B": image_path_b,
            "partial_room_id_A": str(record.get("partial_room_a", "")),
            "partial_room_id_B": str(record.get("partial_room_b", "")),
            "pano_id_A": str(record.get("pano_a", "")),
            "pano_id_B": str(record.get("pano_b", "")),
            "opening_mask_all_A": opening_a.to(dtype=torch.uint8),
            "opening_mask_all_B": opening_b.to(dtype=torch.uint8),
            "portal_mask_A": portal_a.to(dtype=torch.uint8),
            "portal_mask_B": portal_b.to(dtype=torch.uint8),
            "affinity_gt": affinity,
            "is_positive": torch.tensor(is_match, dtype=torch.bool),
            "T_B_to_A": torch.tensor(
                supervision.get("relative_transform_b_to_a", np.eye(3)),
                dtype=torch.float32,
            ),
            # Pose yaw is retained for geometry/diagnostics, never used as
            # cyclic portal-shift supervision.
            "relative_yaw_gt": torch.tensor(
                float(supervision.get("relative_yaw_radians", 0.0)),
                dtype=torch.float32,
            ),
            "pose_valid": torch.tensor(
                bool(supervision.get("pose_valid", is_match)), dtype=torch.bool
            ),
            "manifest_candidate_masks_A": candidate_masks_a,
            "manifest_candidate_masks_B": candidate_masks_b,
            "manifest_candidate_valid_A": candidate_valid_a,
            "manifest_candidate_valid_B": candidate_valid_b,
            "target_candidate_pair": torch.tensor(
                [target_a, target_b], dtype=torch.long
            ),
            "candidate_target_AB": candidate_target,
            "merged_pair_count": torch.tensor(
                len(target_pairs), dtype=torch.long
            ),
            "portal_token_shift_radians": torch.tensor(
                portal_shift, dtype=torch.float32
            ),
            "portal_token_shift_valid": torch.tensor(
                portal_shift_valid, dtype=torch.bool
            ),
        }
        for side, image_path in (("A", image_path_a), ("B", image_path_b)):
            view_index = self._view_index[image_path]
            output["feature_{}".format(side)] = self._tensor(
                self._features[view_index], "features", pair_id
            )
            output["depth_enclosed_{}".format(side)] = self._tensor(
                self._enclosed_depth[view_index], "enclosed_depth", pair_id
            )
            output["depth_extended_{}".format(side)] = self._tensor(
                self._extended_depth[view_index], "extended_depth", pair_id
            )
            output["feature_cache_index_{}".format(side)] = torch.tensor(
                view_index, dtype=torch.long
            )
        return output


def validate_disjoint_house_splits(
    datasets: Mapping[str, ZInDMatchingFeatureDataset],
) -> Dict[str, int]:
    """Fail when any house appears in more than one train/val/test dataset."""
    names = tuple(datasets)
    for name in names:
        if name not in _VALID_SPLITS:
            raise ValueError("dataset split mapping keys must be train, val, or test")
        if datasets[name].split != name:
            raise ValueError(
                "dataset under {!r} declares split {!r}".format(
                    name, datasets[name].split
                )
            )
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = datasets[left].house_ids & datasets[right].house_ids
            if overlap:
                raise ValueError(
                    "house leakage between {} and {}: {}".format(
                        left, right, ", ".join(sorted(overlap)[:3])
                    )
                )
    return {name: len(datasets[name].house_ids) for name in names}


__all__ = ["ZInDMatchingFeatureDataset", "validate_disjoint_house_splits"]
