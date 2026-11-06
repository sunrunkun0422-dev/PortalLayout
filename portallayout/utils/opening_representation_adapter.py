"""Convert frozen single-view predictions to representation tensors; no GT I/O."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from portallayout.models.opening_representation import PredictedOpeningGeometry, RepresentationCandidates


def _finite_vector(value, shape):
    try:
        result = np.asarray(value, dtype=float)
        return result if result.shape == shape and np.isfinite(result).all() else None
    except (TypeError, ValueError):
        return None


def _corner_cosine(polygon, vertex):
    a = polygon[(vertex - 1) % len(polygon)] - polygon[vertex]
    b = polygon[(vertex + 1) % len(polygon)] - polygon[vertex]
    divisor = np.linalg.norm(a) * np.linalg.norm(b)
    return None if divisor < 1e-10 else float(np.clip(a @ b / divisor, -1, 1))


def predicted_wall_attributes(prediction, opening):
    """Six fixed geometric measurements with independent validity flags.

    Reversing polygon winding preserves the measurements: corner cosines are
    sorted and the center position is measured to the nearest wall endpoint.
    """
    attributes = np.zeros(6, dtype=np.float32)
    valid = np.zeros(6, dtype=bool)
    quality = opening.get('geometry_quality', {})
    fraction = quality.get('dominantWallFraction')
    if isinstance(fraction, (int, float)) and np.isfinite(fraction) and 0 <= fraction <= 1:
        attributes[5], valid[5] = fraction, True
    if opening.get('geometry_status') != 'mapped_dominant_wall':
        return attributes, valid
    endpoint = _finite_vector(opening.get('endpoints_local'), (2, 2))
    if endpoint is not None:
        width = float(np.linalg.norm(endpoint[1] - endpoint[0]))
        if width > 1e-6:
            attributes[0], valid[0] = width, True
    ratios = _finite_vector(opening.get('wall_ratios'), (2,))
    if ratios is not None and 0 <= ratios[0] < ratios[1] <= 1:
        center = float(ratios.mean())
        attributes[2], valid[2] = min(center, 1-center), True
    geometry = prediction.get('geometry', {}).get('enclosed', {})
    try:
        polygon = np.asarray(geometry.get('polygon_local'), dtype=float)
    except (TypeError, ValueError):
        return attributes, valid
    wall = opening.get('wall_index')
    if (not geometry.get('valid', False) or polygon.ndim != 2 or polygon.shape[1] != 2 or
            len(polygon) < 3 or not np.isfinite(polygon).all() or
            type(wall) is not int or not 0 <= wall < len(polygon)):
        return attributes, valid
    wall_length = float(np.linalg.norm(polygon[(wall+1) % len(polygon)] - polygon[wall]))
    if wall_length > 1e-6:
        attributes[1], valid[1] = wall_length, True
    cosines = [_corner_cosine(polygon, i) for i in (wall, (wall+1) % len(polygon))]
    if all(x is not None for x in cosines):
        attributes[3:5], valid[3:5] = sorted(cosines), True
    return attributes, valid


@dataclass(frozen=True)
class PreparedOpeningBatch:
    feature_1d: torch.Tensor
    feature_2d: torch.Tensor
    opening_probability: torch.Tensor
    candidates: RepresentationCandidates
    predicted_geometry: PredictedOpeningGeometry
    observation_ids: Tuple[str, ...]

    def encoder_inputs(self):
        return (self.feature_1d, self.feature_2d, self.opening_probability,
                self.candidates, self.predicted_geometry)


def prepare_opening_representation_batch(records, array_records, device='cpu'):
    """Accept only prediction-cache records, never dataset supervision records."""
    if not records or len(records) != len(array_records):
        raise ValueError('a nonempty aligned list of prediction records and arrays is required')
    predictions = [record['prediction'] for record in records]
    for prediction in predictions:
        spec = prediction.get('coordinate_spec', {})
        if spec.get('frame') != 'bi_panorama_local_xz' or spec.get('units') != 'camera_height_normalized':
            raise ValueError('representation adapter requires canonical Bi panorama-local normalized geometry')
    b = len(records)
    k = max(len(p['openings']) for p in predictions)
    intervals = np.full((b, k, 2), -1, dtype=np.int64)
    valid = np.zeros((b, k), dtype=bool)
    attributes = np.zeros((b, k, 6), dtype=np.float32)
    attribute_valid = np.zeros((b, k, 6), dtype=bool)
    ids = []
    for i, prediction in enumerate(predictions):
        row = [None] * k
        for j, opening in enumerate(prediction['openings']):
            interval = opening['token_interval_inclusive']
            if not isinstance(interval, list) or len(interval) != 2 or any(type(x) is not int for x in interval):
                raise ValueError('opening interval must contain exactly two integer endpoints')
            intervals[i, j], valid[i, j] = interval, True
            row[j] = opening['candidate_id']
            attributes[i, j], attribute_valid[i, j] = predicted_wall_attributes(prediction, opening)
        ids.append(tuple(row))

    def stack(name):
        try:
            value = np.stack([arrays[name] for arrays in array_records]).copy()
        except (KeyError, ValueError) as exc:
            raise ValueError('incompatible cached tensor: ' + name) from exc
        return torch.as_tensor(value, device=device, dtype=torch.float32)

    candidates = RepresentationCandidates(torch.as_tensor(intervals, device=device),
        torch.as_tensor(valid, device=device), tuple(ids))
    geometry = PredictedOpeningGeometry(stack('depth_enclosed'), stack('depth_extended'),
        stack('ratio'), torch.as_tensor(attributes, device=device), torch.as_tensor(attribute_valid, device=device))
    return PreparedOpeningBatch(stack('feature_1d'), stack('feature_2d'), stack('opening_probability'),
                                candidates, geometry, tuple(r['observation_id'] for r in records))
