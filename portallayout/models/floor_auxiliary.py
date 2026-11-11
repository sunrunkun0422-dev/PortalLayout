"""Independent geometry-supervised pose and same-space classifiers."""
import torch
from torch import nn

FEATURE_VERSION = 'floor-auxiliary.predicted-only.v2'
POSE_FEATURE_VERSION = 'floor-auxiliary.pose-filter.v3'
SUPPORTED_FEATURE_VERSIONS = (FEATURE_VERSION, POSE_FEATURE_VERSION, 'floor-auxiliary.pose-geometry.v4')
GEOMETRY_NAMES = ('iou', 'overlap_min', 'log_area_ratio', 'camera_a_inside_b', 'camera_b_inside_a',
    'boundary_mean', 'boundary_p90', 'nonseed_boundary_mean', 'nonseed_boundary_p90',
    'other_opening_mean', 'other_opening_max', 'match_score', 'dustbin_margin',
    'absolute_log_scale', 'independent_wall_fraction', 'geometry_valid')


class SymmetricBEVClassifier(nn.Module):
    def __init__(self, visual_dim):
        super().__init__()
        self.visual_dim = visual_dim
        self.cnn = nn.Sequential(nn.Conv2d(7, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.score = nn.Sequential(nn.Linear(128+visual_dim+2*len(GEOMETRY_NAMES), 256), nn.ReLU(),
                                   nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, raster, visual, geometry):
        """[B,2,7,H,W], [B,2,D], [B,2,32]; direction axis swaps with A/B."""
        if raster.ndim != 5 or raster.shape[1:3] != (2, 7): raise ValueError('expected bidirectional BEV rasters')
        b = raster.shape[0]
        if visual.shape != (b, 2, self.visual_dim) or geometry.shape != (b, 2, 2*len(GEOMETRY_NAMES)):
            raise ValueError('auxiliary feature contract mismatch')
        if not all(torch.isfinite(x).all() for x in (raster, visual, geometry)):
            raise ValueError('nonfinite auxiliary input')
        values = self.cnn(raster.reshape(b*2, *raster.shape[2:])).flatten(1)
        values = torch.cat((values, self.visual_norm(visual.reshape(b*2, -1)), geometry.reshape(b*2, -1)), dim=1)
        return self.score(values).reshape(b, 2).mean(dim=1)


class PoseHypothesisVerifier(SymmetricBEVClassifier):
    def __init__(self): super().__init__(1024)


class SameSpaceAssociator(SymmetricBEVClassifier):
    def __init__(self): super().__init__(512)
