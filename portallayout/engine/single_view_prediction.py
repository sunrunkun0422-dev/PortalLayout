"""Shared Bi-Layout loading and layout serialization; no dataset or GT access."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch

from portallayout.config.defaults import merge_from_file
from portallayout.models.bi_layout import Bi_Layout
from portallayout.postprocessing.post_process import frame_preserving_manhattan, post_process
from portallayout.geometry.conversion import depth2xyz
from portallayout.engine.cross_scene_estimator import polygon_validity
from portallayout.utils.writer import xyz2json

REPO_ROOT = Path(__file__).resolve().parents[2]

def load_bi_layout(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    load_checkpoint: bool,
    strict_checkpoint: bool = False,
) -> Tuple[Bi_Layout, Dict[str, Any]]:
    config = merge_from_file(config_path)
    model_args = dict(config.MODEL.ARGS[0])

    # torchvision resolves pretrained ResNet weights from <hub_dir>/checkpoints.
    torch.hub.set_dir(str(REPO_ROOT))
    model = Bi_Layout(**model_args)
    checkpoint_report = {
        'portallayout.config': str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "loaded": False,
        "modelArgs": model_args,
    }
    if load_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("net", checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=strict_checkpoint)
        checkpoint_report.update(
            loaded=True,
            epoch=checkpoint.get("epoch"),
            missingKeys=list(incompatible.missing_keys),
            unexpectedKeys=list(incompatible.unexpected_keys),
        )
        del checkpoint, state_dict
        gc.collect()

    model.to(device).eval()
    return model, checkpoint_report


def _convex_layout(depth: np.ndarray, ratio: float) -> Dict[str, Any]:
    xyz = depth2xyz(np.maximum(np.abs(depth), 1e-3))
    hull = cv2.convexHull(xyz[..., ::2].astype(np.float32))[:, 0, :]
    if len(hull) < 3:
        extent = max(float(np.max(np.abs(xyz[..., ::2]))), 1.0)
        hull = np.asarray(
            [[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]],
            dtype=np.float32,
        )
    hull_xyz = np.insert(hull, 1, 1.0, axis=1)
    return xyz2json(hull_xyz, max(abs(float(ratio)), 1e-3))


LAYOUT_FRAME_MODES = ("legacy_fixed_axis", "panorama_preserving")


def prediction_to_layout(
    depth: torch.Tensor,
    ratio: torch.Tensor,
    layout_frame_mode: str = "legacy_fixed_axis",
) -> Tuple[Dict, Dict]:
    """Serialize one predicted layout with an explicit horizontal frame.

    ``legacy_fixed_axis`` deliberately preserves the historical fixed-axis
    Manhattan behavior.  ``panorama_preserving`` estimates a Manhattan family
    only from the predicted pre-fit depth geometry, fits in that canonical
    frame, and inverse-rotates the fitted polygon before serialization.
    """
    if layout_frame_mode not in LAYOUT_FRAME_MODES:
        raise ValueError(
            "layout_frame_mode must be one of {}".format(
                ", ".join(LAYOUT_FRAME_MODES)
            )
        )
    depth_np = depth.detach().cpu().numpy().reshape(1, -1)
    ratio_value = float(ratio.detach().cpu().reshape(-1)[0])
    safe_depth = np.maximum(np.abs(depth_np), 1e-3)
    pre_local = np.asarray(depth2xyz(safe_depth, plan_y=1)[0][..., ::2])
    report = {
        "postProcessing": "manhattan",
        "fallbackUsed": False,
        "layoutFrameMode": layout_frame_mode,
        "layoutPolygonPreManhattanLocalBiXZ": pre_local.tolist(),
    }
    try:
        if layout_frame_mode == "legacy_fixed_axis":
            processed_xyz = post_process(
                safe_depth, type_name="manhattan"
            )[0]
            canonical = np.asarray(processed_xyz[..., ::2], dtype=np.float64)
            report.update(
                layoutFrame="manhattan_canonical",
                notValidForCrossScenePose=True,
                headingSource="legacy_fixed_axis_no_heading_estimation",
                manhattanHeadingRadians=0.0,
                manhattanHeadingDegrees=0.0,
                layoutPolygonManhattanCanonicalBiXZ=canonical.tolist(),
                layoutPolygonPanoramaLocalBiXZ=None,
            )
        else:
            processed_batch, frame_reports = frame_preserving_manhattan(safe_depth)
            processed_xyz = processed_batch[0]
            report.update(frame_reports[0])
            report.update(
                layoutFrame="panorama_local",
                notValidForCrossScenePose=False,
            )
        layout = xyz2json(processed_xyz, max(abs(ratio_value), 1e-3))
        validity = polygon_validity(layout)
        if not validity["valid"]:
            raise ValueError("Manhattan output failed polygon validation: {}".format(validity))
    except Exception as exc:
        layout = _convex_layout(depth_np[0], ratio_value)
        report.update(
            postProcessing="convex_hull_fallback",
            fallbackUsed=True,
            fallbackReason=str(exc),
            layoutFrame="panorama_local_fallback",
            notValidForCrossScenePose=False,
            headingSource="none_convex_hull_fallback",
            manhattanHeadingRadians=None,
            manhattanHeadingDegrees=None,
            layoutPolygonManhattanCanonicalBiXZ=None,
            layoutPolygonPanoramaLocalBiXZ=None,
        )
    report["validity"] = polygon_validity(layout)
    return layout, report


def prediction_to_local_geometry(depth):
    """Fit a polygon in Bi panorama-local x/z, without writer transforms.

    Coordinates are normalized by camera height; no metric height is assumed.
    The legacy JSON writer's axis conversion and default 1.6 factor are absent.
    """
    values = np.asarray(depth, dtype=np.float64).reshape(-1)
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("depth must contain at least three finite values")
    safe = np.maximum(np.abs(values), 1e-3)
    report = {"depth_transform": "abs_then_clamp_min_0.001",
              "negative_depth_count": int((values < 0).sum()),
              "clamped_depth_count": int((np.abs(values) < 1e-3).sum()),
              "fallback_used": False}
    try:
        xyz, reports = frame_preserving_manhattan(safe[None])
        polygon = xyz[0][:, [0, 2]]
        if not polygon_validity(polygon)["valid"]:
            raise ValueError("invalid fitted polygon")
        report.update(reports[0])
    except (ValueError, RuntimeError, AssertionError, FloatingPointError) as exc:
        radial = depth2xyz(safe[None], plan_y=1)[0][:, [0, 2]]
        polygon = cv2.convexHull(radial.astype(np.float32))[:, 0, :]
        report.update(fallback_used=True, fallback_reason=str(exc))
    validity = polygon_validity(polygon)
    return {"polygon_local": polygon.tolist(), "valid": bool(validity["valid"]),
            "postprocess": report, "validity": validity}

