"""Predicted-only portal-centered BEV rasterization for verifier prototypes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box


CHANNEL_NAMES = (
    "A_occupancy", "B_transformed_occupancy", "A_boundary",
    "B_transformed_boundary", "overlap", "portal_line", "validity",
)


@dataclass(frozen=True)
class CandidateBEVRasterConfig:
    grid_size: int = 128
    half_extent_portal_widths: float = 6.0
    boundary_width_pixels: int = 1
    portal_width_pixels: int = 2


def normalize_portal_frame(
    points: Sequence[Sequence[float]], portal_endpoints_a: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    value = np.asarray(points, dtype=np.float64)
    portal = np.asarray(portal_endpoints_a, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or portal.shape != (2, 2):
        raise ValueError("points must be [N,2] and portal must be [2,2]")
    if not np.isfinite(value).all() or not np.isfinite(portal).all():
        raise ValueError("raster geometry must be finite")
    tangent = portal[1] - portal[0]
    width = float(np.linalg.norm(tangent))
    if width <= 1e-8:
        raise ValueError("portal width is degenerate")
    angle = math.atan2(float(tangent[1]), float(tangent[0]))
    cosine, sine = math.cos(-angle), math.sin(-angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    midpoint = portal.mean(axis=0)
    normalized = ((value - midpoint) @ rotation.T) / width
    return normalized, {
        "portal_midpoint_A_xy": midpoint.tolist(),
        "portal_width_A": width,
        "rotation_radians_A_to_raster": -angle,
        "scale_normalization": "divide all coordinates by predicted A portal width",
        "portal_center_convention": "portal midpoint maps to (0,0)",
        "portal_direction_convention": "A interval start-to-end tangent maps to +x",
    }


def _pixels(points: np.ndarray, config: CandidateBEVRasterConfig) -> np.ndarray:
    extent = float(config.half_extent_portal_widths)
    scaled = (points + extent) / (2.0 * extent) * (config.grid_size - 1)
    scaled[:, 1] = (config.grid_size - 1) - scaled[:, 1]
    return np.rint(scaled).astype(np.int32)


def _mask(points: np.ndarray, config: CandidateBEVRasterConfig, *, fill: bool, width: int = 1) -> np.ndarray:
    image = Image.new("L", (config.grid_size, config.grid_size), 0)
    draw = ImageDraw.Draw(image)
    pixels = [tuple(map(int, point)) for point in _pixels(points, config)]
    if fill:
        draw.polygon(pixels, fill=255)
    else:
        draw.line(pixels + [pixels[0]], fill=255, width=width, joint="curve")
    return np.asarray(image, dtype=np.float32) / 255.0


def rasterize_candidate_bev(
    polygon_a: Sequence[Sequence[float]],
    polygon_b_in_a: Sequence[Sequence[float]],
    portal_endpoints_a: Sequence[Sequence[float]],
    *,
    geometry_valid: bool = True,
    config: CandidateBEVRasterConfig = CandidateBEVRasterConfig(),
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if config.grid_size < 16 or config.half_extent_portal_widths <= 0:
        raise ValueError("invalid raster configuration")
    empty = torch.zeros((len(CHANNEL_NAMES), config.grid_size, config.grid_size), dtype=torch.float32)
    if not geometry_valid:
        return empty, {
            "valid": False, "failure_reason": "geometry_valid=false",
            "channel_names": list(CHANNEL_NAMES), "config": asdict(config),
            "clipping_rate": None, "predicted_only": True,
        }
    try:
        a, frame = normalize_portal_frame(polygon_a, portal_endpoints_a)
        b, _ = normalize_portal_frame(polygon_b_in_a, portal_endpoints_a)
        portal, _ = normalize_portal_frame(portal_endpoints_a, portal_endpoints_a)
        shape_a, shape_b = Polygon(a), Polygon(b)
        if not shape_a.is_valid or not shape_b.is_valid or min(shape_a.area, shape_b.area) <= 1e-10:
            raise ValueError("predicted polygon is invalid")
        extent = float(config.half_extent_portal_widths)
        window = box(-extent, -extent, extent, extent)
        combined = shape_a.union(shape_b)
        clipped_area = max(0.0, float(combined.area - combined.intersection(window).area))
        clipping_rate = clipped_area / max(float(combined.area), 1e-12)
        occupancy_a = _mask(a, config, fill=True)
        occupancy_b = _mask(b, config, fill=True)
        boundary_a = _mask(a, config, fill=False, width=config.boundary_width_pixels)
        boundary_b = _mask(b, config, fill=False, width=config.boundary_width_pixels)
        portal_image = Image.new("L", (config.grid_size, config.grid_size), 0)
        ImageDraw.Draw(portal_image).line(
            [tuple(map(int, point)) for point in _pixels(portal, config)],
            fill=255, width=config.portal_width_pixels,
        )
        tensor = torch.from_numpy(np.stack((
            occupancy_a, occupancy_b, boundary_a, boundary_b,
            np.minimum(occupancy_a, occupancy_b),
            np.asarray(portal_image, dtype=np.float32) / 255.0,
            np.ones_like(occupancy_a, dtype=np.float32),
        ))).float()
        if tensor.shape != (7, config.grid_size, config.grid_size) or not torch.isfinite(tensor).all():
            raise ValueError("raster tensor is malformed or non-finite")
        return tensor, {
            "valid": True, "failure_reason": None,
            "channel_names": list(CHANNEL_NAMES), "config": asdict(config),
            "coordinate_frame": "predicted portal-centered BEV; B transformed B_to_A before rasterization",
            "A_B_channel_order": "A channels precede transformed-B channels",
            "clipping_rate": clipping_rate, **frame,
            "predicted_only": True,
        }
    except Exception as exc:
        return empty, {
            "valid": False, "failure_reason": str(exc),
            "channel_names": list(CHANNEL_NAMES), "config": asdict(config),
            "clipping_rate": None, "predicted_only": True,
        }


def raster_preview_rgb(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().float().numpy()
    if value.ndim != 3 or value.shape[0] != 7 or not np.isfinite(value).all():
        raise ValueError("preview expects a finite [7,H,W] tensor")
    red = np.clip(value[1] * 0.70 + value[4] * 0.30 + value[5], 0.0, 1.0)
    green = np.clip(value[4] * 0.85 + value[5], 0.0, 1.0)
    blue = np.clip(value[0] * 0.70 + value[4] * 0.30 + value[5], 0.0, 1.0)
    return np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
