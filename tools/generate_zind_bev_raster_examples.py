#!/usr/bin/env python3
"""Generate a few predicted-only BEV raster examples after Phase-B passes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.preflight_zind_bev_verifier import (  # noqa: E402
    AUDIT_ROOT,
    DEFAULT_OUTPUT,
    PredictedGeometryStore,
    atomic_json,
    atomic_text,
    feature_indices,
    load_context,
    reject_test_path,
    sha256_file,
    transform_points,
)
from tools.run_zind_bev_verifier_phase_b import (  # noqa: E402
    DustbinLinearRanker,
    _model_predictions,
    _phase_b_groups,
)
from portallayout.utils.candidate_bev_rasterizer import (  # noqa: E402
    CandidateBEVRasterConfig,
    normalize_portal_frame,
    raster_preview_rgb,
    rasterize_candidate_bev,
)
from portallayout.geometry.joint_layout import build_joint_layout_from_endpoints  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit_root", default=str(AUDIT_ROOT))
    parser.add_argument("--train_manifest", default=str(REPO_ROOT / "data/ZInD_matching/train.json"))
    parser.add_argument("--val_manifest", default=str(REPO_ROOT / "data/ZInD_matching/val.json"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    return parser.parse_args(argv)


def _svg(points_a: np.ndarray, points_b: np.ndarray, portal: np.ndarray, config: CandidateBEVRasterConfig) -> str:
    extent = config.half_extent_portal_widths; size = 512

    def pixels(points: np.ndarray) -> str:
        value = (points + extent) / (2 * extent) * size
        value[:, 1] = size - value[:, 1]
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in value)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
<rect width="100%" height="100%" fill="#111"/>
<polygon points="{pixels(points_a)}" fill="#2f6fff" fill-opacity="0.35" stroke="#72a1ff" stroke-width="2"/>
<polygon points="{pixels(points_b)}" fill="#ef4b4b" fill-opacity="0.35" stroke="#ff8b8b" stroke-width="2"/>
<polyline points="{pixels(portal)}" fill="none" stroke="#ffe45c" stroke-width="5"/>
</svg>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.count <= 0 or args.count > 16:
        raise ValueError("example count must be in [1,16]")
    output = reject_test_path(args.output_dir, "output")
    phase_b = json.loads((output / "phase_B_summary.json").read_text(encoding="utf-8"))
    if not phase_b.get("passed") or not (output / ".phase_B_passed").is_file():
        raise RuntimeError("Phase B must pass before raster prototype generation")
    context_args = argparse.Namespace(
        seed=args.seed, bootstrap_replicates=args.bootstrap_replicates,
        audit_root=args.audit_root, train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
    )
    context = load_context(context_args, output)
    groups = _phase_b_groups(output, "val")
    norm = json.loads((output / "phase_B_normalization.json").read_text(encoding="utf-8"))
    indices = feature_indices(("t3", "m2", "relative", "bev"))
    model = DustbinLinearRanker(len(indices))
    artifact = torch.load(output / "phase_B_E1_logistic.pt", map_location="cpu")
    model.load_state_dict(artifact["state_dict"]); model.eval()
    predictions, _ = _model_predictions(model, groups, norm, indices)
    selected = [
        (group, prediction)
        for group, prediction in zip(groups, predictions)
        if group["pair_is_adjacent"]
        and prediction < len(group["rows"])
        and group["rows"][prediction]["candidate_is_any_physical_true"]
        and group["rows"][prediction]["candidate_geometry_valid"]
        and not group["baseline_physical_correct"]
    ]
    selected.sort(key=lambda item: str(item[0]["pair_id"]))
    selected = selected[:args.count]
    if len(selected) != args.count:
        raise RuntimeError(f"only {len(selected)} deterministic E1 wins are rasterizable")
    records = {str(record["id"]): record for record in context["val"]["pairs"]}
    store = PredictedGeometryStore(context)
    config = CandidateBEVRasterConfig()
    root = output / "raster_examples"; root.mkdir(parents=True, exist_ok=True)
    examples = []
    for group, prediction in selected:
        pair_id = str(group["pair_id"]); row = group["rows"][prediction]
        refs = context["room_index"]["pair_refs"][pair_id]
        room_a = store.room("val", refs["A"]); room_b = store.room("val", refs["B"])
        portal_a = store.portal("val", refs["A"], int(row["candidate_A"]))
        portal_b = store.portal("val", refs["B"], int(row["candidate_B"]))
        if not portal_a.geometry_valid or not portal_b.geometry_valid:
            raise RuntimeError(f"selected raster candidate has M2 hard failure: {pair_id}")
        endpoints_a = np.asarray(portal_a.primary.endpoints_local_zind_xy, dtype=np.float64)
        endpoints_b = np.asarray(portal_b.primary.endpoints_local_zind_xy, dtype=np.float64)
        joint = build_joint_layout_from_endpoints(
            room_a["frame_layout"], room_b["frame_layout"], endpoints_a, endpoints_b,
            boundary_a=room_a["raw_contour_zind"], boundary_b=room_b["raw_contour_zind"],
            calibrate_scale=True,
            room_ids=(str(records[pair_id]["room_a"]), str(records[pair_id]["room_b"])),
            coordinate_system="ZInD panorama-A local xy",
        )
        matrix = np.asarray(joint["alignment"]["roomBToWorld"], dtype=np.float64)
        polygon_b = transform_points(room_b["frame_polygon"], matrix)
        tensor, metadata = rasterize_candidate_bev(room_a["frame_polygon"], polygon_b, endpoints_a, config=config)
        if not metadata["valid"]:
            raise RuntimeError(f"selected raster candidate invalid: {pair_id}: {metadata['failure_reason']}")
        tensor_path = root / f"{pair_id}.pt"
        temporary = tensor_path.with_name(tensor_path.name + f".tmp.{os.getpid()}")
        torch.save({"tensor": tensor, "metadata": metadata, "pair_id": pair_id}, temporary); os.replace(temporary, tensor_path)
        png_path = root / f"{pair_id}.png"; Image.fromarray(raster_preview_rgb(tensor)).save(png_path)
        norm_a, _ = normalize_portal_frame(room_a["frame_polygon"], endpoints_a)
        norm_b, _ = normalize_portal_frame(polygon_b, endpoints_a)
        norm_portal, _ = normalize_portal_frame(endpoints_a, endpoints_a)
        svg_path = root / f"{pair_id}.svg"; atomic_text(svg_path, _svg(norm_a, norm_b, norm_portal, config))
        examples.append({
            "pair_id": pair_id, "candidate_A": int(row["candidate_A"]),
            "candidate_B": int(row["candidate_B"]),
            "selection": "stable first E1 physical-S3 win over E0; post-hoc visualization only",
            "tensor": str(tensor_path.relative_to(output)), "tensor_sha256": sha256_file(tensor_path),
            "png": str(png_path.relative_to(output)), "png_sha256": sha256_file(png_path),
            "svg": str(svg_path.relative_to(output)), "svg_sha256": sha256_file(svg_path),
            "metadata": metadata, "input_uses_gt": False,
            "selection_uses_posthoc_GT": True, "test_data_accessed": False,
        })
    summary: Dict[str, Any] = {
        "schema_version": "zind-salve-bev-raster-prototype-v1",
        'portallayout.config': config.__dict__, "example_count": len(examples),
        "examples": examples,
        "mean_clipping_rate": float(np.mean([value["metadata"]["clipping_rate"] for value in examples])),
        "max_clipping_rate": float(max(value["metadata"]["clipping_rate"] for value in examples)),
        "cnn_training_started": False, "input_uses_gt": False,
        "active_splits": ["train", "val"], "test_data_accessed": False,
    }
    atomic_json(root / "index.json", summary)
    atomic_text(output / ".raster_prototype_complete", sha256_file(root / "index.json") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
